"""Investigators that drive the runner.

`LocalInvestigator` is a deterministic tool user. It reads whatever the
query service returns and does not consult the rubric. It is the offline
way to execute the pipeline.

`ModelInvestigator` asks an OpenAI-compatible backend for the next action
and the final answer. Contract repairs still happen in the runner.
"""

from __future__ import annotations

import json
import re
from typing import Protocol

from beyond_retrieval.models import (
    ActionContract,
    ActionKind,
    AgentRecord,
    Capabilities,
    FinalAnswer,
    FindingAssertion,
    HypothesisPrediction,
    Proposal,
    ScriptContract,
    TaskView,
    ToolResult,
    Verdict,
)
from beyond_retrieval.tools import TOOL_SCHEMA

COUNT_SCRIPT = """
import json, sqlite3, sys
con = sqlite3.connect(sys.argv[1])
rows = list(con.execute("select record_id from events order by record_id"))
json.dump([{"finding_id": "sources", "n": len(rows), "record_id": rows[0][0]}], sys.stdout)
""".strip()


class Completer(Protocol):
    name: str

    def complete(self, messages: list[dict[str, str]], *, max_tokens: int = 800) -> object:
        """Return an object with text, prompt_tokens, and completion_tokens."""


class LocalInvestigator:
    """Search every visible source once, then answer from the retrieved rows."""

    def __init__(self, capabilities: Capabilities) -> None:
        self.capabilities = capabilities
        self.view: TaskView | None = None
        self.records: dict[str, AgentRecord] = {}
        self.pending = "catalog"
        self.queue: list[str] = []
        self.scripted = False
        self.need_rescan = False

    def start(self, view: TaskView) -> None:
        self.view = view

    def notify_release(self) -> None:
        self.need_rescan = True
        self.scripted = False

    def propose(self, observation: ToolResult | None) -> Proposal:
        self._absorb(observation)
        if self.need_rescan:
            self.pending = "catalog"
            self.queue = []
            self.need_rescan = False
        if self.view is None or "list_sources" not in self.view.tools:
            return Proposal(kind=ActionKind.STOP, generated_tokens=4)
        if self.pending == "catalog":
            self.pending = "listed"
            return self._query("list_sources", {})
        if self.pending == "listed":
            self.queue = _sources(observation)
            self.pending = "search"
        if self.queue:
            source = self.queue.pop(0)
            last = not self.queue and not self.capabilities.custom_scripts
            return self._query(
                "search_events",
                {"source": source},
                falsify=last and self.capabilities.contradiction_check,
            )
        if self.capabilities.custom_scripts and not self.scripted:
            self.scripted = True
            return self._script()
        return Proposal(kind=ActionKind.STOP, generated_tokens=4)

    def finalize(self) -> FinalAnswer:
        return answer_from_records(list(self.records.values()), self.view)

    def _absorb(self, observation: ToolResult | None) -> None:
        if observation is None:
            return
        for record in observation.records:
            self.records.setdefault(record.record_id, record)

    def _query(self, tool: str, parameters: dict[str, object], falsify: bool = False) -> Proposal:
        return Proposal(
            kind=ActionKind.QUERY,
            tool=tool,
            parameters=dict(parameters),
            contract=self._contract(tool, parameters) if self.capabilities.action_contracts else None,
            generated_tokens=12,
            falsification_check=falsify,
        )

    def _script(self) -> Proposal:
        contract = ScriptContract(
            inputs=["events"],
            join_keys=["record_id"],
            units={"n": "count"},
            output_schema={"n": "int", "record_id": "str"},
            assumptions=["each visible row is one event"],
        )
        return Proposal(
            kind=ActionKind.SCRIPT,
            script_source=COUNT_SCRIPT,
            script_contract=contract,
            contract=self._contract("script", {}) if self.capabilities.action_contracts else None,
            generated_tokens=24,
        )

    def _contract(self, tool: str, parameters: dict[str, object]) -> ActionContract:
        return ActionContract(
            question=f"What does {tool} add about the host, the process, or the job window?",
            hypotheses=[
                HypothesisPrediction(
                    hypothesis_id="h_linked",
                    explanation="The transfer is tied to an internal host and a process.",
                    predicted_observation="Rows name an internal host or a process that shares the flow addresses.",
                ),
                HypothesisPrediction(
                    hypothesis_id="h_unlinked",
                    explanation="This source does not identify the internal host.",
                    predicted_observation="Rows are empty or do not tie a host and a process to the flow.",
                ),
            ],
            tool=tool,
            parameters=dict(parameters),
            discrimination="A host or process tied to the flow supports the linked explanation. Absence supports the unlinked one.",
            resource_estimate="1 call",
        )


class ModelInvestigator:
    """Ask a hosted model for each action. JSON repairs are one extra completion."""

    def __init__(self, capabilities: Capabilities, backend: Completer) -> None:
        self.capabilities = capabilities
        self.backend = backend
        self.view: TaskView | None = None
        self.messages: list[dict[str, str]] = []
        self.records: dict[str, AgentRecord] = {}
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def start(self, view: TaskView) -> None:
        self.view = view
        if not self.messages:
            self.messages.append({"role": "system", "content": _system_prompt(self.capabilities)})
            self.messages.append({"role": "user", "content": _task_prompt(view)})

    def notify_release(self) -> None:
        self.messages.append(
            {
                "role": "user",
                "content": "New records were released. Continue from the same case. Search again before you stop.",
            }
        )

    def propose(self, observation: ToolResult | None) -> Proposal:
        if observation is not None:
            for record in observation.records:
                self.records.setdefault(record.record_id, record)
            self.messages.append({"role": "user", "content": render_observation(observation)})
        self.messages.append(
            {
                "role": "user",
                "content": "Reply with one JSON object for the next action. kind is query, script, or stop.",
            }
        )
        text, tokens = self._complete()
        try:
            proposal = proposal_from_json(text, self.capabilities)
        except ValueError:
            self.messages.append({"role": "user", "content": "That was not valid action JSON. Return one JSON object only."})
            text, extra = self._complete()
            tokens += extra
            try:
                proposal = proposal_from_json(text, self.capabilities)
            except ValueError:
                return Proposal(kind=ActionKind.STOP, generated_tokens=tokens)
        proposal.generated_tokens = tokens
        return proposal

    def finalize(self) -> FinalAnswer:
        claims = [] if self.view is None else [item.finding_id for item in self.view.claims]
        self.messages.append(
            {
                "role": "user",
                "content": (
                    "Reply with the final answer JSON only. "
                    f"Include a finding for each of these ids: {claims}. "
                    "probabilities must cover malicious, benign, and insufficient_evidence and sum to 1."
                ),
            }
        )
        text, _tokens = self._complete(max_tokens=1200)
        try:
            return answer_from_json(text)
        except ValueError:
            return answer_from_records(list(self.records.values()), self.view)

    def _complete(self, max_tokens: int = 800) -> tuple[str, int]:
        completion = self.backend.complete(self.messages, max_tokens=max_tokens)
        text = str(getattr(completion, "text", ""))
        prompt = int(getattr(completion, "prompt_tokens", 0) or 0)
        produced = int(getattr(completion, "completion_tokens", 0) or 0)
        self.prompt_tokens += prompt
        self.completion_tokens += produced
        self.messages.append({"role": "assistant", "content": text})
        return text, produced or max(1, len(text) // 4)


def answer_from_records(records: list[AgentRecord], view: TaskView | None) -> FinalAnswer:
    flows = [item for item in records if item.event_type == "flow" or item.source == "netflow"]
    flow = flows[0] if flows else None
    host, host_ids = _linked_host(records, flow)
    process = _process_record(records, host, flow)
    covers, jobs = _job_covers(records, host, flow)
    if host and process is not None and not covers:
        verdict = Verdict.MALICIOUS
        probabilities = {"malicious": 0.72, "benign": 0.08, "insufficient_evidence": 0.20}
    elif host and process is not None and covers:
        verdict = Verdict.BENIGN
        probabilities = {"malicious": 0.10, "benign": 0.74, "insufficient_evidence": 0.16}
    else:
        verdict = Verdict.INSUFFICIENT
        probabilities = {"malicious": 0.12, "benign": 0.10, "insufficient_evidence": 0.78}
    prompts = () if view is None else view.claims
    findings = [
        _finding(prompt.finding_id, prompt.prompt, records, flow, host, host_ids, process, covers, jobs)
        for prompt in prompts
    ]
    unresolved = []
    if verdict is Verdict.INSUFFICIENT:
        unresolved = ["originating host", "initiating process"]
    return FinalAnswer(
        probabilities=probabilities,
        verdict=verdict,
        findings=findings,
        limitations="The answer cites only rows returned by the query service.",
        unresolved=unresolved,
    )


def proposal_from_json(text: str, capabilities: Capabilities) -> Proposal:
    payload = _extract_json(text)
    kind = str(payload.get("kind", "")).lower()
    if kind == "stop":
        return Proposal(kind=ActionKind.STOP)
    if kind == "script":
        if not capabilities.custom_scripts:
            raise ValueError("scripts are not enabled")
        source = payload.get("script_source")
        if not isinstance(source, str) or not source.strip():
            raise ValueError("script_source is required")
        return Proposal(
            kind=ActionKind.SCRIPT,
            script_source=source,
            script_contract=_script_contract(payload.get("script_contract")),
            contract=_action_contract(payload.get("contract")) if capabilities.action_contracts else None,
            falsification_check=bool(payload.get("falsification_check", False)),
        )
    if kind != "query":
        raise ValueError("kind must be query, script, or stop")
    tool = payload.get("tool")
    if not isinstance(tool, str) or tool not in capabilities_tools(capabilities):
        raise ValueError("tool is not available")
    parameters = payload.get("parameters") or {}
    if not isinstance(parameters, dict):
        raise ValueError("parameters must be an object")
    return Proposal(
        kind=ActionKind.QUERY,
        tool=tool,
        parameters=parameters,
        contract=_action_contract(payload.get("contract")) if capabilities.action_contracts else None,
        falsification_check=bool(payload.get("falsification_check", False)),
    )


def answer_from_json(text: str) -> FinalAnswer:
    payload = _extract_json(text)
    verdict = Verdict(str(payload.get("verdict")))
    raw_probs = payload.get("probabilities")
    if not isinstance(raw_probs, dict):
        raise ValueError("probabilities must be an object")
    probabilities = {str(key): float(value) for key, value in raw_probs.items()}
    findings = []
    raw_findings = payload.get("findings") or []
    if not isinstance(raw_findings, list):
        raise ValueError("findings must be a list")
    for item in raw_findings:
        if not isinstance(item, dict):
            continue
        record_ids = item.get("record_ids") or []
        if not isinstance(record_ids, list):
            record_ids = []
        findings.append(
            FindingAssertion(
                finding_id=str(item.get("finding_id", "")),
                text=str(item.get("text", "")),
                record_ids=[str(record_id) for record_id in record_ids],
            )
        )
    limitations = str(payload.get("limitations", ""))
    unresolved = payload.get("unresolved") or []
    if not isinstance(unresolved, list):
        unresolved = []
    return FinalAnswer(
        probabilities=probabilities,
        verdict=verdict,
        findings=findings,
        limitations=limitations,
        unresolved=[str(item) for item in unresolved],
    )


def render_observation(observation: ToolResult) -> str:
    rows = [
        {
            "record_id": record.record_id,
            "source": record.source,
            "event_type": record.event_type,
            "timestamp": record.timestamp,
            "entities": record.entities,
            "raw": record.raw[:500],
            "parser_status": record.parser_status,
        }
        for record in observation.records[:20]
    ]
    return json.dumps(
        {
            "status": observation.status.value,
            "summary": observation.summary,
            "error": observation.error,
            "truncated": observation.truncated,
            "cursor": observation.cursor,
            "records": rows,
        },
        sort_keys=True,
    )


def capabilities_tools(capabilities: Capabilities) -> set[str]:
    if not capabilities.adaptive_queries:
        return set()
    return {"list_sources", "describe_schema", "search_events", "get_records", "aggregate_events"}


def _sources(observation: ToolResult | None) -> list[str]:
    if observation is None or not observation.summary:
        return []
    names = []
    for line in observation.summary.splitlines():
        name = line.split("\t", 1)[0].strip()
        if name:
            names.append(name)
    return names


def _linked_host(records: list[AgentRecord], flow: AgentRecord | None) -> tuple[str | None, list[str]]:
    if flow is None:
        return None, []
    src = flow.entities.get("src_ip")
    dst = flow.entities.get("dst_ip")
    for record in records:
        host = record.entities.get("host")
        if host and record.entities.get("external_ip") == src:
            return host, [flow.record_id, record.record_id]
    for record in records:
        host = record.entities.get("host")
        if host and dst and record.entities.get("dst_ip") == dst and record.record_id != flow.record_id:
            return host, [record.record_id]
    return None, [flow.record_id]


def _process_record(
    records: list[AgentRecord], host: str | None, flow: AgentRecord | None
) -> AgentRecord | None:
    for record in records:
        if record.source != "process" and record.event_type != "process_start":
            continue
        if "process" not in record.entities:
            continue
        if host is not None and record.entities.get("host") not in {None, host}:
            continue
        if flow is not None and record.entities.get("dst_ip") not in {None, flow.entities.get("dst_ip")}:
            if record.entities.get("host") != host:
                continue
        return record
    return None


def _job_covers(
    records: list[AgentRecord], host: str | None, flow: AgentRecord | None
) -> tuple[bool, list[AgentRecord]]:
    jobs = [item for item in records if item.event_type == "scheduled_job" or item.source == "jobs"]
    if not jobs or flow is None or host is None:
        return False, jobs
    job = jobs[0]
    same_host = job.entities.get("host") == host
    same_hour = job.timestamp[:13] == flow.timestamp[:13]
    same_process = job.entities.get("process") and job.entities.get("process") == _process_name(records, host)
    return bool(same_host and same_hour and same_process), jobs


def _process_name(records: list[AgentRecord], host: str | None) -> str | None:
    for record in records:
        if record.entities.get("host") == host and record.entities.get("process"):
            if record.event_type != "scheduled_job" and record.source != "jobs":
                return record.entities.get("process")
    return None


def _finding(
    finding_id: str,
    prompt: str,
    records: list[AgentRecord],
    flow: AgentRecord | None,
    host: str | None,
    host_ids: list[str],
    process: AgentRecord | None,
    covers: bool,
    jobs: list[AgentRecord],
) -> FindingAssertion:
    flow_ids = [flow.record_id] if flow is not None else _any_ids(records)
    if finding_id == "F_HOST":
        if host:
            return FindingAssertion(
                finding_id=finding_id,
                text=f"The internal host linked to the transfer is {host}.",
                record_ids=host_ids,
            )
        return FindingAssertion(
            finding_id=finding_id,
            text="The internal host cannot be determined from the accessible records.",
            record_ids=flow_ids,
        )
    if finding_id == "F_PROC":
        if process is not None:
            account = process.entities.get("user", "unknown account")
            name = process.entities.get("process", "unknown process")
            return FindingAssertion(
                finding_id=finding_id,
                text=f"The initiating process is {name}, run by {account}.",
                record_ids=[process.record_id],
            )
        return FindingAssertion(
            finding_id=finding_id,
            text="The initiating process cannot be determined from the accessible records.",
            record_ids=flow_ids,
        )
    if finding_id == "F_AUTH":
        ids = flow_ids + [job.record_id for job in jobs]
        if jobs and covers:
            return FindingAssertion(
                finding_id=finding_id,
                text="A scheduled job on the same host and window matches the transfer process.",
                record_ids=ids,
            )
        if jobs:
            return FindingAssertion(
                finding_id=finding_id,
                text="The retrieved scheduled job does not cover this transfer.",
                record_ids=ids,
            )
        return FindingAssertion(
            finding_id=finding_id,
            text="No scheduled job was retrieved.",
            record_ids=flow_ids,
        )
    if finding_id == "F_FLOW":
        if flow is not None:
            src = flow.entities.get("src_ip", "an address")
            return FindingAssertion(
                finding_id=finding_id,
                text=f"An outbound transfer from {src} is present.",
                record_ids=[flow.record_id],
            )
        return FindingAssertion(finding_id=finding_id, text="No flow row was retrieved.", record_ids=flow_ids)
    if finding_id == "F_GAP":
        if host or process is not None:
            return FindingAssertion(
                finding_id=finding_id,
                text="The accessible records do link a host or a process to the transfer.",
                record_ids=host_ids or flow_ids,
            )
        return FindingAssertion(
            finding_id=finding_id,
            text="The internal host and initiating process cannot be determined from the accessible records.",
            record_ids=flow_ids,
        )
    return FindingAssertion(
        finding_id=finding_id,
        text=f"Retrieved rows were checked for: {prompt}",
        record_ids=flow_ids,
    )


def _any_ids(records: list[AgentRecord]) -> list[str]:
    return [records[0].record_id] if records else []


def _system_prompt(capabilities: Capabilities) -> str:
    script = "You may return kind script with script_source when a count or join is useful." if capabilities.custom_scripts else "Do not write scripts."
    contract = (
        "Every query and script needs a contract. Copy the same tool and parameters onto the contract. "
        "Include two hypotheses with different predicted_observation strings, plus non-empty discrimination "
        "and resource_estimate. contract.tool must equal the action tool."
        if capabilities.action_contracts
        else "Do not include a contract."
    )
    params = {
        name: spec["params"]
        for name, spec in TOOL_SCHEMA.items()
        if isinstance(spec, dict) and isinstance(spec.get("params"), list)
    }
    return (
        "You are investigating a security case through a read-only query service. "
        "Use only tool results. Do not invent record ids. "
        f"{script} {contract} "
        "Call list_sources before searching. search_events has no free-text query. "
        "Match source, event_type, or entity_key plus entity_value. "
        "aggregate_events group_by is only source, event_type, or parser_status. "
        f"Allowed parameters: {json.dumps(params, sort_keys=True)}. "
        "Action JSON fields: kind, tool, parameters, falsification_check, "
        "contract {question, hypotheses[{hypothesis_id, explanation, predicted_observation}], "
        "open_set_alternative, tool, parameters, discrimination, resource_estimate}, "
        "script_source, script_contract {inputs, join_keys, units, output_schema, assumptions}."
    )


def _task_prompt(view: TaskView) -> str:
    claims = [{"finding_id": item.finding_id, "prompt": item.prompt} for item in view.claims]
    return json.dumps(
        {
            "case_id": view.case_id,
            "alert": view.alert,
            "question": view.question,
            "tools": list(view.tools),
            "guidance": view.guidance_text,
            "required_claims": claims,
        },
        sort_keys=True,
    )


_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def _extract_json(text: str) -> dict[str, object]:
    cleaned = _THINK_BLOCK.sub("", text)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start < 0 or end <= start:
        start = text.find("{")
        end = text.rfind("}")
        cleaned = text
    if start < 0 or end <= start:
        raise ValueError("no JSON object")
    payload = json.loads(cleaned[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("JSON value must be an object")
    return payload


def _action_contract(raw: object) -> ActionContract | None:
    if not isinstance(raw, dict):
        return None
    hypotheses = []
    for item in raw.get("hypotheses") or []:
        if not isinstance(item, dict):
            continue
        hypotheses.append(
            HypothesisPrediction(
                hypothesis_id=str(item.get("hypothesis_id", "h")),
                explanation=str(item.get("explanation", "")),
                predicted_observation=str(item.get("predicted_observation", "")),
            )
        )
    parameters = raw.get("parameters") if isinstance(raw.get("parameters"), dict) else {}
    return ActionContract(
        question=str(raw.get("question", "")),
        hypotheses=hypotheses,
        open_set_alternative=bool(raw.get("open_set_alternative", False)),
        tool=str(raw.get("tool", "")),
        parameters=parameters,
        discrimination=str(raw.get("discrimination", "")),
        resource_estimate=str(raw.get("resource_estimate", "")),
    )


def _script_contract(raw: object) -> ScriptContract | None:
    if not isinstance(raw, dict):
        return None
    units = raw.get("units") if isinstance(raw.get("units"), dict) else {}
    schema = raw.get("output_schema") if isinstance(raw.get("output_schema"), dict) else {}
    return ScriptContract(
        inputs=[str(item) for item in raw.get("inputs") or []],
        join_keys=[str(item) for item in raw.get("join_keys") or []],
        units={str(key): str(value) for key, value in units.items()},
        output_schema={str(key): str(value) for key, value in schema.items()},
        assumptions=[str(item) for item in raw.get("assumptions") or []],
    )
