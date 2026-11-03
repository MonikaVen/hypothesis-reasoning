"""Model backends for a live investigation.

`local` does not use this module. An OpenAI-compatible HTTP backend is used
when a key is configured. The key is read from the environment and is never
written into a run record.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Callable

from beyond_retrieval.errors import InfrastructureError


class MissingCredentials(RuntimeError):
    """Raised when a hosted backend is selected and no API key is set."""


@dataclass(frozen=True)
class Completion:
    text: str
    prompt_tokens: int
    completion_tokens: int


Transport = Callable[[str, dict[str, str], dict[str, object], float], dict[str, object]]


GROQ_BASE_URL = "https://api.groq.com/openai/v1"
# Groq shut down deepseek-r1-distill-llama-70b on 2025-10-02. This id is the
# replacement still listed on a live Groq account.
GROQ_DEFAULT_MODEL = "openai/gpt-oss-120b"


class OpenAICompatibleBackend:
    """Chat completions against any API that speaks the OpenAI request shape."""

    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        timeout_seconds: float = 90,
        transport: Transport | None = None,
        name_prefix: str = "openai",
        json_object: bool = False,
        reasoning_effort: str | None = None,
    ) -> None:
        if not api_key.strip():
            raise MissingCredentials("BR_API_KEY or OPENAI_API_KEY is empty")
        self.model = model
        self.name = f"{name_prefix}:{model}"
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout_seconds
        self._transport = transport or _urllib_post
        self._json_object = json_object
        self._reasoning_effort = reasoning_effort

    @classmethod
    def from_env(cls, model: str | None = None, transport: Transport | None = None) -> OpenAICompatibleBackend:
        key = os.environ.get("BR_API_KEY") or os.environ.get("OPENAI_API_KEY") or ""
        if not key.strip():
            raise MissingCredentials(
                "Set BR_API_KEY or OPENAI_API_KEY to run a hosted model, or pass --backend local."
            )
        chosen = model or os.environ.get("BR_MODEL") or "gpt-4o-mini"
        base = os.environ.get("BR_BASE_URL") or "https://api.openai.com/v1"
        return cls(model=chosen, api_key=key, base_url=base, transport=transport)

    @classmethod
    def for_groq(cls, model: str | None = None, transport: Transport | None = None) -> OpenAICompatibleBackend:
        """Groq's OpenAI-compatible chat API. The key stays in the environment."""
        key = (
            os.environ.get("GROQ_API_KEY")
            or os.environ.get("BR_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
            or ""
        )
        if not key.strip():
            raise MissingCredentials(
                "Set GROQ_API_KEY or BR_API_KEY to run a Groq model, or pass --backend local."
            )
        chosen = model or os.environ.get("BR_MODEL") or GROQ_DEFAULT_MODEL
        base = os.environ.get("BR_BASE_URL") or GROQ_BASE_URL
        return cls(
            model=chosen,
            api_key=key,
            base_url=base,
            timeout_seconds=120,
            transport=transport,
            name_prefix="groq",
            json_object=True,
            reasoning_effort="low",
        )

    def complete(self, messages: list[dict[str, str]], *, max_tokens: int = 800) -> Completion:
        payload: dict[str, object] = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": max_tokens,
            "messages": messages,
        }
        if self._json_object:
            payload["response_format"] = {"type": "json_object"}
        if self._reasoning_effort:
            payload["reasoning_effort"] = self._reasoning_effort
        url = self._base_url + "/chat/completions"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "beyond-retrieval/0.1",
        }
        data = self._post_with_retries(url, headers, payload)
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            raise RuntimeError("model response did not include choices")
        message = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
        content = message.get("content", "") if isinstance(message, dict) else ""
        text = _flatten_content(content)
        if not text.strip() and isinstance(message, dict):
            reasoning = message.get("reasoning") or message.get("reasoning_content") or ""
            text = _flatten_content(reasoning)
        usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
        prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)  # type: ignore[union-attr]
        completion_tokens = int(usage.get("completion_tokens", 0) or 0)  # type: ignore[union-attr]
        return Completion(text=text, prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)

    def _post_with_retries(self, url: str, headers: dict[str, str], payload: dict[str, object]) -> dict[str, object]:
        """Retry provider rate limits. A limit that never clears is an outage, not an agent error."""
        attempts = 6
        for attempt in range(attempts):
            try:
                return self._transport(url, headers, payload, self._timeout)
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")[:300]
                delay = _retry_delay(exc.code, detail, attempt)
                if delay is None or attempt + 1 == attempts:
                    if exc.code in {429, 500, 502, 503, 504}:
                        raise InfrastructureError(f"model HTTP {exc.code} after {attempt + 1} attempts") from exc
                    raise RuntimeError(f"model HTTP {exc.code}: {detail}") from exc
                time.sleep(delay)
            except urllib.error.URLError as exc:
                if attempt + 1 == attempts:
                    raise InfrastructureError(f"model request failed: {exc.reason}") from exc
                time.sleep(min(30.0, 2.0 * (attempt + 1)))
        raise InfrastructureError("model request failed")


def _retry_delay(status: int, detail: str, attempt: int) -> float | None:
    if status not in {429, 500, 502, 503, 504}:
        return None
    match = re.search(r"try again in ([0-9]+(?:\.[0-9]+)?)s", detail)
    if match:
        return min(60.0, float(match.group(1)) + 0.25)
    return min(60.0, 2.0 * (attempt + 1))


def _urllib_post(url: str, headers: dict[str, str], payload: dict[str, object], timeout: float) -> dict[str, object]:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        parsed = json.loads(response.read().decode("utf-8"))
    if not isinstance(parsed, dict):
        raise RuntimeError("model response was not a JSON object")
    return parsed


def _flatten_content(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "\n".join(parts)
    return str(content)
