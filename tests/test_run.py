"""The campaign can execute without a hosted model."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from beyond_retrieval.backends import Completion, MissingCredentials, OpenAICompatibleBackend
from beyond_retrieval.campaign import run_campaign
from beyond_retrieval.fixtures import domain_reference, outbound_case
from beyond_retrieval.investigator import LocalInvestigator, ModelInvestigator, proposal_from_json
from beyond_retrieval.models import Capabilities, ConditionId, RunSpec, Variant
from beyond_retrieval.protocol import make_protocol
from beyond_retrieval.runner import ManualClock, execute_run
from beyond_retrieval.scoring import revision_label


def test_claim_prompts_do_not_contain_the_gold_host():
    for prompts in outbound_case().claim_prompts.values():
        blob = " ".join(item.prompt for item in prompts)
        assert "srv-app-03" not in blob
        assert "jdoe" not in blob
        assert "10.0.0.5" not in blob


def test_local_pilot_executes_and_stays_unadjudicated(tmp_path: Path):
    records = run_campaign(experiment="pilot", out=tmp_path, clock=ManualClock())
    assert len(records) == 4
    by_key = {(item.spec.variant, item.spec.condition): item for item in records}
    full_a = by_key[(Variant.V0, ConditionId.A)]
    full_h = by_key[(Variant.V0, ConditionId.H)]
    missing_h = by_key[(Variant.V4, ConditionId.H)]
    assert full_a.snapshot_hash == full_h.snapshot_hash
    assert full_a.mechanical is not None and full_a.mechanical.passed
    assert full_h.mechanical is not None and full_h.mechanical.passed
    assert full_h.falsification_attempt_present is True
    assert full_a.falsification_attempt_present is False
    assert any(step.contract_valid is True for step in full_h.steps)
    assert all(step.contract_valid is None for step in full_a.steps)
    assert missing_h.mechanical is not None and missing_h.mechanical.passed
    assert missing_h.final_answer is not None
    assert missing_h.final_answer.verdict.value == "insufficient_evidence"
    assert all(item.esr is not None and item.esr.status == "pending_adjudication" for item in records)
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["backend"] == "local-baseline"
    assert manifest["note"]


def test_local_delayed_release_resolves_after_the_second_stage(tmp_path: Path):
    case = outbound_case()
    record = execute_run(
        RunSpec(
            experiment="pilot",
            case_id=case.record.case_id,
            variant=Variant.V5,
            condition=ConditionId.H,
            backbone_id="local-baseline",
            repeat_index=0,
        ),
        case,
        LocalInvestigator(Capabilities.for_condition(ConditionId.H)),
        make_protocol(sealed=False),
        domain_reference(),
        clock=ManualClock(),
        workspace=tmp_path,
    )
    assert record.provisional_answer is not None
    assert record.provisional_answer.verdict.value == "insufficient_evidence"
    assert record.final_answer is not None
    assert record.final_answer.verdict.value == "malicious"
    assert record.provisional_mechanical is not None and record.provisional_mechanical.passed
    assert record.mechanical is not None and record.mechanical.passed
    assert (
        revision_label(
            record.provisional_mechanical,
            record.mechanical,
            record.provisional_answer.verdict.value,
            record.final_answer.verdict.value,
        )
        == "mechanical_resolution"
    )


def test_model_investigator_follows_a_stub_backend(tmp_path: Path):
    class Stub:
        name = "stub-model"

        def __init__(self) -> None:
            self.replies = [
                json.dumps({"kind": "query", "tool": "list_sources", "parameters": {}}),
                json.dumps({"kind": "stop"}),
                json.dumps(
                    {
                        "verdict": "insufficient_evidence",
                        "probabilities": {"malicious": 0.2, "benign": 0.2, "insufficient_evidence": 0.6},
                        "findings": [
                            {"finding_id": "F_HOST", "text": "Host not established.", "record_ids": []},
                            {"finding_id": "F_PROC", "text": "Process not established.", "record_ids": []},
                            {"finding_id": "F_AUTH", "text": "No job checked.", "record_ids": []},
                        ],
                        "limitations": "Stub model stopped after the catalog.",
                        "unresolved": ["originating host"],
                    }
                ),
            ]

        def complete(self, messages: list[dict[str, str]], max_tokens: int = 800) -> Completion:
            return Completion(self.replies.pop(0), prompt_tokens=4, completion_tokens=6)

    case = outbound_case()
    record = execute_run(
        RunSpec(
            experiment="pilot",
            case_id=case.record.case_id,
            variant=Variant.V0,
            condition=ConditionId.A,
            backbone_id="stub-model",
            repeat_index=0,
        ),
        case,
        ModelInvestigator(Capabilities.for_condition(ConditionId.A), Stub()),
        make_protocol(sealed=False),
        domain_reference(),
        clock=ManualClock(),
        workspace=tmp_path,
    )
    assert record.status.value == "completed"
    assert record.steps[0].tool == "list_sources"
    assert record.budget.generated_tokens > 0
    assert record.final_answer is not None
    assert record.esr is not None and record.esr.passed is False


def test_invalid_model_json_stops_the_action(tmp_path: Path):
    with pytest.raises(ValueError):
        proposal_from_json("not json", Capabilities.for_condition(ConditionId.A))


def test_openai_backend_posts_chat_completions_without_leaking_the_key():
    captured: dict[str, object] = {}

    def transport(url: str, headers: dict[str, str], payload: dict[str, object], timeout: float) -> dict[str, object]:
        captured["url"] = url
        captured["authorization"] = headers["Authorization"]
        captured["model"] = payload["model"]
        return {
            "choices": [{"message": {"content": "{\"kind\": \"stop\"}"}}],
            "usage": {"prompt_tokens": 2, "completion_tokens": 3},
        }

    backend = OpenAICompatibleBackend(
        model="gpt-4o-mini",
        api_key="sk-test-secret",
        base_url="https://example.test/v1/",
        transport=transport,
    )
    completion = backend.complete([{"role": "user", "content": "hello"}])
    assert captured["url"] == "https://example.test/v1/chat/completions"
    assert captured["authorization"] == "Bearer sk-test-secret"
    assert captured["model"] == "gpt-4o-mini"
    assert completion.completion_tokens == 3
    assert "sk-test-secret" not in completion.text


def test_hosted_backend_requires_a_key(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("BR_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(MissingCredentials):
        OpenAICompatibleBackend.from_env()


def test_groq_backend_targets_groq_and_keeps_the_key_out_of_the_name(monkeypatch: pytest.MonkeyPatch):
    captured: dict[str, object] = {}

    def transport(url: str, headers: dict[str, str], payload: dict[str, object], timeout: float) -> dict[str, object]:
        captured["url"] = url
        captured["json_object"] = payload.get("response_format")
        captured["reasoning_effort"] = payload.get("reasoning_effort")
        captured["authorization"] = headers["Authorization"]
        return {
            "choices": [{"message": {"content": "", "reasoning": "<think>draft</think>{\"kind\": \"stop\"}"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 2},
        }

    monkeypatch.setenv("GROQ_API_KEY", "gsk-test-secret")
    monkeypatch.delenv("BR_MODEL", raising=False)
    monkeypatch.delenv("BR_BASE_URL", raising=False)
    backend = OpenAICompatibleBackend.for_groq(transport=transport)
    completion = backend.complete([{"role": "user", "content": "hello"}])
    assert backend.name == "groq:openai/gpt-oss-120b"
    assert backend.model == "openai/gpt-oss-120b"
    assert captured["url"] == "https://api.groq.com/openai/v1/chat/completions"
    assert captured["authorization"] == "Bearer gsk-test-secret"
    assert captured["json_object"] == {"type": "json_object"}
    assert captured["reasoning_effort"] == "low"
    assert "gsk-test-secret" not in backend.name
    assert "gsk-test-secret" not in completion.text
    assert proposal_from_json(completion.text, Capabilities.for_condition(ConditionId.A)).kind.value == "stop"


def test_groq_backend_retries_a_rate_limit(monkeypatch: pytest.MonkeyPatch):
    import io
    import urllib.error

    calls = {"n": 0}

    def transport(url: str, headers: dict[str, str], payload: dict[str, object], timeout: float) -> dict[str, object]:
        calls["n"] += 1
        if calls["n"] == 1:
            raise urllib.error.HTTPError(
                url,
                429,
                "Too Many Requests",
                hdrs=None,
                fp=io.BytesIO(b'{"error":{"message":"try again in 0.01s"}}'),
            )
        return {
            "choices": [{"message": {"content": "{\"kind\": \"stop\"}"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }

    monkeypatch.setenv("GROQ_API_KEY", "gsk-test-secret")
    backend = OpenAICompatibleBackend.for_groq(model="openai/gpt-oss-20b", transport=transport)
    completion = backend.complete([{"role": "user", "content": "hello"}])
    assert calls["n"] == 2
    assert completion.text == '{"kind": "stop"}'


def test_groq_backend_requires_a_key(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("BR_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(MissingCredentials):
        OpenAICompatibleBackend.for_groq()


def test_script_campaign_validates_only_the_full_method(tmp_path: Path):
    records = run_campaign(experiment="scripts", out=tmp_path, clock=ManualClock())
    by_condition = {item.spec.condition: item for item in records}
    assert by_condition[ConditionId.AS].steps[-1].script_validation == "not_applicable"
    assert by_condition[ConditionId.HS].steps[-1].script_validation == "pass"
    assert any(node.kind.value == "artifact" for node in by_condition[ConditionId.HS].graph_nodes)
    assert by_condition[ConditionId.HS].mechanical is not None
    assert by_condition[ConditionId.HS].mechanical.passed


def test_cli_run_writes_a_manifest(tmp_path: Path):
    from beyond_retrieval.cli import main

    code = main(["run", "--backend", "local", "--experiment", "pilot", "--out", str(tmp_path)])
    assert code == 0
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert len(manifest["records"]) == 4
    assert (tmp_path / "V0-H-r0.json").exists()
