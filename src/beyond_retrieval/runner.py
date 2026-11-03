"""Execute one run spec.

The runner is the only component that holds a mount, a rubric, and a policy
at the same time. It never hands the rubric to the policy. Invalid agent
calls and exhausted budgets stay in the scored set. Infrastructure outages
are missing runs and can be rerun under the same spec.
"""

from __future__ import annotations

import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Protocol

from beyond_retrieval.agent import Budget, EvidenceGraph, allowed_tools, assemble_guidance, validate_contract
from beyond_retrieval.errors import (
    CounterfactualRequiresRegeneration,
    InfrastructureError,
    ProtocolError,
    UnsealedProtocol,
)
from beyond_retrieval.evidence import EvidenceMount, prepare_mount, rubric_for
from beyond_retrieval.hashing import content_hash, sha256_text
from beyond_retrieval.models import (
    QUERY_TOOLS,
    ActionKind,
    BudgetCaps,
    BudgetUsage,
    Capabilities,
    CaseBundle,
    DomainReference,
    EdgeRelation,
    FinalAnswer,
    FrozenLessons,
    ObservationStatus,
    Playbook,
    Proposal,
    ProtocolFreeze,
    QueryLimits,
    RunMode,
    RunRecord,
    RunSpec,
    RunStatus,
    ScriptContract,
    StepTrace,
    TaskView,
    ToolResult,
    Variant,
)
from beyond_retrieval.scoring import esr_result, mechanical_score
from beyond_retrieval.tools import QueryService, ScriptSandbox


class Clock(Protocol):
    def now(self) -> datetime: ...

    def monotonic(self) -> float: ...


class SystemClock:
    def now(self) -> datetime:
        return datetime.now(timezone.utc)

    def monotonic(self) -> float:
        import time

        return time.monotonic()


class ManualClock:
    """Frozen clock for tests. Wall time advances only when a test asks."""

    def __init__(self) -> None:
        self.t = 0.0
        self.start = datetime(2026, 11, 3, 14, 0, tzinfo=timezone.utc)

    def now(self) -> datetime:
        return self.start + timedelta(seconds=self.t)

    def monotonic(self) -> float:
        return self.t


class ScriptValidator(Protocol):
    def validate(
        self,
        source: str,
        contract: ScriptContract | None,
        findings: list[dict[str, object]],
        case_db: Path,
    ) -> bool:
        """Run the script on a known-answer fixture the validator owns.

        `case_db` is the agent mount, used only for an independent recompute.
        The script author's own output is not enough.
        """


class _LoopResult:
    def __init__(self) -> None:
        self.status = RunStatus.COMPLETED
        self.budget = Budget(BudgetCaps())
        self.steps: list[StepTrace] = []
        self.answer: FinalAnswer | None = None
        self.retrieved: set[str] = set()
        self.falsification = False
        self.infeasible: str | None = None


def protocol_hash(protocol: ProtocolFreeze) -> str:
    return content_hash(protocol.model_dump(mode="json"))


def execute_run(
    spec: RunSpec,
    case: CaseBundle,
    policy: object,
    protocol: ProtocolFreeze,
    reference: DomainReference,
    *,
    playbook: Playbook | None = None,
    lessons: FrozenLessons | None = None,
    clock: Clock | None = None,
    confirmatory: bool = False,
    script_validator: ScriptValidator | None = None,
    workspace: Path | None = None,
    raise_infra: bool = False,
) -> RunRecord:
    if confirmatory and not protocol.sealed:
        raise UnsealedProtocol("confirmatory runs require a sealed protocol")
    if spec.variant is Variant.V3:
        raise CounterfactualRequiresRegeneration(
            "V3 is outside the recorded-data runner. Generate a paired recording first."
        )
    if spec.case_id != case.record.case_id:
        raise ProtocolError("run spec case_id does not match the bundle")

    clock = clock or SystemClock()
    capabilities = Capabilities.for_condition(spec.condition, spec.ablation).for_mode(spec.mode)
    guidance = assemble_guidance(capabilities, reference, playbook, lessons, spec.mode)
    started = clock.now()
    graph = EvidenceGraph()

    if spec.mode is RunMode.SYNTHESIS:
        return _synthesis(spec, case, policy, protocol, guidance, clock, started)

    if spec.variant is Variant.V5:
        return _v5(
            spec, case, policy, protocol, guidance, capabilities, clock, started, graph,
            script_validator, workspace, raise_infra,
        )

    if spec.mode is RunMode.NO_EVIDENCE:
        mount = EvidenceMount([], spec.variant, "no-evidence")
    else:
        mount = prepare_mount(case, spec.variant, spec.seed)
    rubric = rubric_for(case, spec.variant if spec.mode is not RunMode.NO_EVIDENCE else Variant.V0)
    loop = _loop(
        policy, case, capabilities, mount, protocol.caps, protocol.query_limits, clock, graph,
        guidance, spec, script_validator, workspace, raise_infra,
    )
    mechanical = _score(loop, rubric, mount.ids())
    return _record(
        spec, protocol, mount.content_hash, loop, clock, started, graph, guidance,
        final=loop.answer, mechanical=mechanical, stages={},
    )


def _synthesis(spec, case, policy, protocol, guidance, clock, started) -> RunRecord:
    view = _view(case, spec, guidance, tools=())
    status = RunStatus.COMPLETED
    answer: FinalAnswer | None = None
    try:
        policy.start(view)
        produced = policy.finalize()
        if not isinstance(produced, FinalAnswer):
            status = RunStatus.AGENT_ERROR
        else:
            answer = produced
    except InfrastructureError:
        status = RunStatus.INFRA_OUTAGE
    except Exception:
        status = RunStatus.AGENT_ERROR
    loop = _LoopResult()
    loop.status = status
    loop.answer = answer
    return _record(
        spec, protocol, content_hash({"synthesis": spec.case_id}), loop, clock, started,
        EvidenceGraph(), guidance, final=answer, mechanical=None, stages={}, scored=False,
    )


def _v5(spec, case, policy, protocol, guidance, capabilities, clock, started, graph, script_validator, workspace, raise_infra) -> RunRecord:
    caps = protocol.caps.split_in_half()
    mount1 = prepare_mount(case, Variant.V5, spec.seed, stage="stage1")
    first = _loop(
        policy, case, capabilities, mount1, caps, protocol.query_limits, clock, graph,
        guidance, spec, script_validator, workspace, raise_infra, stage="stage1",
    )
    provisional = first.answer
    provisional_mechanical = _score(first, rubric_for(case, Variant.V5, stage="stage1"), mount1.ids())
    if first.status in {RunStatus.INFRA_OUTAGE, RunStatus.AGENT_ERROR}:
        return _record(
            spec, protocol, mount1.content_hash, first, clock, started, graph, guidance,
            final=None, provisional=provisional, mechanical=None,
            provisional_mechanical=provisional_mechanical,
            stages={"stage1": mount1.content_hash},
        )
    notify = getattr(policy, "notify_release", None)
    if callable(notify):
        notify()
    mount2 = prepare_mount(case, Variant.V5, spec.seed, stage="stage2")
    second = _loop(
        policy, case, capabilities, mount2, caps, protocol.query_limits, clock, graph,
        guidance, spec, script_validator, workspace, raise_infra, stage="stage2",
    )
    combined = _combine(first, second)
    mechanical = _score(combined, rubric_for(case, Variant.V5, stage="stage2"), mount2.ids())
    return _record(
        spec, protocol, mount2.content_hash, combined, clock, started, graph, guidance,
        final=second.answer, provisional=provisional, mechanical=mechanical,
        provisional_mechanical=provisional_mechanical,
        stages={"stage1": mount1.content_hash, "stage2": mount2.content_hash},
    )


def _loop(
    policy,
    case: CaseBundle,
    capabilities: Capabilities,
    mount: EvidenceMount,
    caps: BudgetCaps,
    limits: QueryLimits,
    clock: Clock,
    graph: EvidenceGraph,
    guidance: str,
    spec: RunSpec,
    script_validator: ScriptValidator | None,
    workspace: Path | None,
    raise_infra: bool,
    stage: str | None = None,
) -> _LoopResult:
    result = _LoopResult()
    result.budget = Budget(caps)
    view = _view(case, spec, guidance, tools=allowed_tools(capabilities), stage=stage)
    try:
        policy.start(view)
    except Exception:
        result.status = RunStatus.AGENT_ERROR
        return result

    observation: ToolResult | None = None
    while True:
        if result.budget.exhausted():
            result.status = RunStatus.BUDGET_EXHAUSTED
            break
        try:
            proposal = policy.propose(observation)
        except InfrastructureError:
            result.status = RunStatus.INFRA_OUTAGE
            break
        except Exception:
            result.status = RunStatus.AGENT_ERROR
            break
        if not isinstance(proposal, Proposal):
            result.status = RunStatus.AGENT_ERROR
            break
        result.budget.charge_tokens(proposal.generated_tokens)
        if proposal.falsification_check or proposal.falsification_infeasible:
            result.falsification = True
        if proposal.falsification_infeasible:
            result.infeasible = proposal.falsification_infeasible
        if result.budget.exhausted():
            result.status = RunStatus.BUDGET_EXHAUSTED
            break
        if proposal.kind is ActionKind.STOP:
            result.status = RunStatus.COMPLETED
            break

        denial = _authorize(proposal, capabilities)
        if denial is not None:
            result.budget.charge_call()
            observation = ToolResult(status=ObservationStatus.FAILED_TOOL, error=denial)
            result.steps.append(_step(len(result.steps), proposal, "denied", denial, contract_valid=None))
            continue
        if capabilities.action_contracts and proposal.kind in {ActionKind.QUERY, ActionKind.SCRIPT}:
            errors = validate_contract(proposal.contract)
            if (
                proposal.kind is ActionKind.QUERY
                and proposal.contract is not None
                and proposal.contract.tool != (proposal.tool or "")
            ):
                errors.append("contract tool does not match the action")
            if errors:
                result.budget.charge_call()
                observation = ToolResult(status=ObservationStatus.FAILED_TOOL, error="; ".join(errors))
                result.steps.append(
                    _step(
                        len(result.steps), proposal, "contract_rejected", "; ".join(errors),
                        contract_valid=False, contract_errors=errors,
                    )
                )
                continue
            _record_hypotheses(graph, proposal)

        try:
            if raise_infra:
                raise InfrastructureError("injected outage")
            tool_result, script_hash = _dispatch(
                proposal, mount, capabilities, limits, script_validator, workspace, caps
            )
        except InfrastructureError:
            result.status = RunStatus.INFRA_OUTAGE
            break

        extra = 1 if proposal.kind is ActionKind.SCRIPT and capabilities.script_validation else 0
        result.budget.charge_call(script_cpu_seconds=tool_result.script_cpu_seconds)
        for _ in range(extra):
            result.budget.charge_call()
        for record in tool_result.records:
            result.retrieved.add(record.record_id)
            graph.add_observation(record.record_id, record.raw, record.content_hash)
        for index, finding in enumerate(tool_result.admitted_findings):
            artifact_id = f"{len(result.steps)}-{index}"
            graph.add_artifact(artifact_id, str(finding), script_validated=True)
            record_id = finding.get("record_id")
            if isinstance(record_id, str):
                located = next(
                    (item for item in mount.agent_records() if item.record_id == record_id),
                    None,
                )
                if located is not None:
                    graph.add_observation(located.record_id, located.raw, located.content_hash)
                graph.add_edge(
                    f"art:{artifact_id}",
                    f"obs:{record_id}",
                    EdgeRelation.DERIVATION,
                    "script",
                    script_validated=True,
                )
        observation = tool_result
        result.steps.append(
            _step(
                len(result.steps),
                proposal,
                tool_result.status.value,
                tool_result.error,
                record_ids=[record.record_id for record in tool_result.records],
                observation_hash=tool_result.content_hash,
                script_validation=tool_result.validation,
                script_hash=script_hash,
                contract_valid=True if proposal.contract else None,
            )
        )

    if result.status in {RunStatus.COMPLETED, RunStatus.BUDGET_EXHAUSTED}:
        try:
            produced = policy.finalize()
        except Exception:
            result.status = RunStatus.AGENT_ERROR
            return result
        if not isinstance(produced, FinalAnswer):
            result.status = RunStatus.AGENT_ERROR
            return result
        result.answer = produced
    return result


def _authorize(proposal: Proposal, capabilities: Capabilities) -> str | None:
    if proposal.kind is ActionKind.QUERY:
        if not capabilities.adaptive_queries:
            return "queries are not available to this condition"
        if proposal.tool not in QUERY_TOOLS:
            return "unknown tool"
        return None
    if proposal.kind is ActionKind.SCRIPT:
        if not capabilities.custom_scripts:
            return "scripts are not available to this condition"
        if not proposal.script_source:
            return "missing script source"
        return None
    return "unsupported action"


def _dispatch(proposal, mount, capabilities, limits, script_validator, workspace, caps: BudgetCaps):
    if proposal.kind is ActionKind.QUERY:
        service = QueryService(mount.agent_records(), limits)
        return service.execute(proposal.tool or "", dict(proposal.parameters)), ""
    database = _materialize(mount, workspace)
    sandbox = ScriptSandbox(timeout_seconds=caps.max_script_cpu_seconds)
    ran = sandbox.run(proposal.script_source or "", database)
    admitted: list[dict[str, object]] = []
    if capabilities.script_validation:
        passed = False
        if script_validator is not None and ran.execution_success:
            passed = bool(
                script_validator.validate(
                    proposal.script_source or "",
                    proposal.script_contract,
                    ran.findings,
                    database,
                )
            )
        validation = "pass" if passed else "fail"
        if passed:
            admitted = list(ran.findings)
    else:
        validation = "not_applicable"
    result = ToolResult(
        status=ObservationStatus.OK if ran.execution_success else ObservationStatus.FAILED_TOOL,
        execution_success=ran.execution_success,
        script_cpu_seconds=ran.cpu_seconds,
        validation=validation,  # type: ignore[arg-type]
        admitted_findings=admitted,
        claimed_findings=list(ran.findings),
        error=ran.error,
        content_hash=content_hash(ran.findings),
    )
    return result, sha256_text(proposal.script_source or "")


def _materialize(mount: EvidenceMount, workspace: Path | None) -> Path:
    root = workspace if workspace is not None else Path(tempfile.mkdtemp(prefix="br-mount-"))
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"mount-{mount.content_hash[:16]}.sqlite"
    mount.materialize(path)
    return path


def _record_hypotheses(graph: EvidenceGraph, proposal: Proposal) -> None:
    contract = proposal.contract
    if contract is None:
        return
    graph.add_question(f"q{len(graph.nodes)}", contract.question)
    for item in contract.hypotheses:
        graph.add_hypothesis(item.hypothesis_id, item.explanation)


def _score(loop: _LoopResult, rubric, visible: set[str]):
    if loop.answer is None or loop.status is RunStatus.INFRA_OUTAGE:
        return None
    if loop.status is RunStatus.AGENT_ERROR:
        return None
    return mechanical_score(loop.answer, rubric, visible, loop.retrieved)


def _combine(first: _LoopResult, second: _LoopResult) -> _LoopResult:
    combined = _LoopResult()
    combined.status = second.status
    combined.answer = second.answer
    combined.steps = first.steps + second.steps
    combined.retrieved = first.retrieved | second.retrieved
    combined.falsification = first.falsification or second.falsification
    combined.infeasible = second.infeasible or first.infeasible
    combined.budget = second.budget
    combined.budget.usage = BudgetUsage(
        tool_calls=first.budget.usage.tool_calls + second.budget.usage.tool_calls,
        generated_tokens=first.budget.usage.generated_tokens + second.budget.usage.generated_tokens,
        wall_seconds=first.budget.usage.wall_seconds + second.budget.usage.wall_seconds,
        script_cpu_seconds=first.budget.usage.script_cpu_seconds + second.budget.usage.script_cpu_seconds,
        token_accounting=first.budget.usage.token_accounting,
    )
    return combined


def _record(
    spec: RunSpec,
    protocol: ProtocolFreeze,
    snapshot_hash: str,
    loop: _LoopResult,
    clock: Clock,
    started: datetime,
    graph: EvidenceGraph,
    guidance: str,
    *,
    final: FinalAnswer | None,
    mechanical,
    stages: dict[str, str],
    provisional: FinalAnswer | None = None,
    provisional_mechanical=None,
    scored: bool = True,
) -> RunRecord:
    return RunRecord(
        spec=spec,
        protocol_hash=protocol_hash(protocol),
        guidance_hash=content_hash(guidance),
        snapshot_hash=snapshot_hash,
        status=loop.status,
        counts_as_missing=loop.status is RunStatus.INFRA_OUTAGE,
        started_at=_iso(started),
        ended_at=_iso(clock.now()),
        budget=loop.budget.usage,
        steps=loop.steps,
        graph_nodes=list(graph.nodes.values()),
        graph_edges=list(graph.edges),
        provisional_answer=provisional,
        final_answer=final,
        falsification_attempt_present=loop.falsification,
        falsification_infeasible=loop.infeasible,
        mechanical=mechanical,
        provisional_mechanical=provisional_mechanical,
        esr=esr_result(loop.status, mechanical, None, scored=scored and loop.status is not RunStatus.INFRA_OUTAGE),
        exclusion_reason=(
            "infrastructure outage; rerun the same spec" if loop.status is RunStatus.INFRA_OUTAGE else None
        ),
        stage_snapshot_hashes=stages,
    )


def _view(
    case: CaseBundle,
    spec: RunSpec,
    guidance: str,
    tools: tuple[str, ...],
    stage: str | None = None,
) -> TaskView:
    return TaskView(
        case_id=case.record.case_id,
        question=case.record.question,
        alert=case.record.alert,
        tools=tools,
        guidance_text=guidance,
        variant=spec.variant,
        mode=spec.mode,
        claims=_claim_prompts(case, spec.variant, stage),
    )


def _claim_prompts(case: CaseBundle, variant: Variant, stage: str | None) -> tuple:
    """Agent-visible questions. Gold rubric sentences stay on the bundle."""
    if variant is Variant.V5:
        key = "V5:stage1" if stage == "stage1" else "V5:stage2"
    elif variant is Variant.V1:
        key = Variant.V0.value
    else:
        key = variant.value
    return tuple(case.claim_prompts.get(key, ()))


def _step(
    index: int,
    proposal: Proposal,
    status: str,
    error: str | None,
    record_ids: list[str] | None = None,
    observation_hash: str = "",
    script_validation: str = "not_run",
    script_hash: str = "",
    contract_valid: bool | None = None,
    contract_errors: list[str] | None = None,
) -> StepTrace:
    return StepTrace(
        index=index,
        kind=proposal.kind,
        tool=proposal.tool,
        parameters=dict(proposal.parameters),
        contract_valid=contract_valid,
        contract_errors=contract_errors or [],
        status=status,
        record_ids=record_ids or [],
        observation_hash=observation_hash,
        generated_tokens=proposal.generated_tokens,
        script_validation=script_validation,
        falsification_check=proposal.falsification_check,
        script_hash=script_hash,
        error=error,
    )


def _iso(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
