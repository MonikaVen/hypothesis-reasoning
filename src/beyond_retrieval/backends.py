"""Model backends for a live investigation.

`local` does not use this module. An OpenAI-compatible HTTP backend is used
when a key is configured. The key is read from the environment and is never
written into a run record.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Callable


class MissingCredentials(RuntimeError):
    """Raised when a hosted backend is selected and no API key is set."""


@dataclass(frozen=True)
class Completion:
    text: str
    prompt_tokens: int
    completion_tokens: int


Transport = Callable[[str, dict[str, str], dict[str, object], float], dict[str, object]]


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
    ) -> None:
        if not api_key.strip():
            raise MissingCredentials("BR_API_KEY or OPENAI_API_KEY is empty")
        self.model = model
        self.name = f"openai:{model}"
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout_seconds
        self._transport = transport or _urllib_post

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

    def complete(self, messages: list[dict[str, str]], *, max_tokens: int = 800) -> Completion:
        payload: dict[str, object] = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": max_tokens,
            "messages": messages,
        }
        url = self._base_url + "/chat/completions"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        try:
            data = self._transport(url, headers, payload, self._timeout)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:300]
            raise RuntimeError(f"model HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"model request failed: {exc.reason}") from exc
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            raise RuntimeError("model response did not include choices")
        message = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
        content = message.get("content", "") if isinstance(message, dict) else ""
        text = _flatten_content(content)
        usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
        prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)  # type: ignore[union-attr]
        completion_tokens = int(usage.get("completion_tokens", 0) or 0)  # type: ignore[union-attr]
        return Completion(text=text, prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)


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
