"""One synthetic investigation used to exercise the harness.

The episode follows the outbound-transfer sketch in the protocol. It is not
an AIT case, and a run on it is not a study result.
"""

from __future__ import annotations

from beyond_retrieval.models import (
    ActionContract,
    ActionKind,
    Capabilities,
    CaseBundle,
    CaseRecord,
    ClaimPrompt,
    ConditionId,
    DomainReference,
    EntityMapping,
    Event,
    FinalAnswer,
    FindingAssertion,
    HypothesisPrediction,
    MandatoryFinding,
    Partition,
    Playbook,
    PlaybookStep,
    Proposal,
    Proposition,
    Rubric,
    SourceId,
    V4Spec,
    V5Spec,
    Variant,
    Verdict,
)
from beyond_retrieval.scripted import ScriptedPolicy

ENVIRONMENT = "synthetic-lab"


def domain_reference() -> DomainReference:
    return DomainReference(
        propositions=(
            Proposition(
                prop_id="p1",
                text="Address translation can make a shared external address a poor identifier of the originating host.",
            ),
            Proposition(
                prop_id="p2",
                text="A scheduled job authorizes only the process and the time window it names.",
            ),
            Proposition(
                prop_id="p3",
                text="A transfer conclusion needs the internal host, the initiating process, and the authorization context together.",
            ),
        )
    )


def playbook() -> Playbook:
    return Playbook(
        steps=(
            PlaybookStep(kind="fact", text="Check whether translation hides the internal host.", proposition_ids=("p1",)),
            PlaybookStep(kind="procedure", text="Retrieve translation records before attributing the flow.", proposition_ids=()),
            PlaybookStep(kind="fact", text="Compare the flow with the named job window.", proposition_ids=("p2",)),
            PlaybookStep(kind="fact", text="Require host, process, and authorization together.", proposition_ids=("p3",)),
        )
    )


def manifest_records() -> list[CaseRecord]:
    return [outbound_case().record, train_case()]


def train_case() -> CaseRecord:
    return CaseRecord(
        case_id="fox-001",
        partition=Partition.TRAIN,
        source=SourceId.SYNTHETIC,
        environment="fox",
        campaign_id="fox-camp",
        mechanism_group="training-only",
        overlap_group="fox-camp",
        question="Training case. Not visible at test time.",
        alert="Training alert.",
    )


def outbound_case() -> CaseBundle:
    events = (_flow(), _nat(), _process(), _job())
    return CaseBundle(
        record=CaseRecord(
            case_id="synthetic-outbound-01",
            partition=Partition.TEST,
            source=SourceId.SYNTHETIC,
            environment=ENVIRONMENT,
            campaign_id="synthetic-camp-1",
            mechanism_group="outbound-transfer",
            overlap_group="synthetic-camp-1",
            question=(
                "An outbound transfer exceeded the usual volume from a server segment. "
                "Determine whether the observed transfer is authorized activity, a "
                "compromise-related transfer, or unresolved from available evidence. "
                "Identify the originating host and support the conclusion."
            ),
            alert="An outbound transfer exceeded the usual volume from a server segment.",
        ),
        events=events,
        distractors=_distractors(),
        rubrics=_rubrics(),
        v1_mapping=entity_mapping(),
        v4=V4Spec(blocked_finding_id="F_HOST", cut_record_ids=("nat-1", "proc-1")),
        v5=V5Spec(stage1_ids=("flow-1", "job-1"), stage2_ids=("nat-1", "proc-1")),
        eligible_variants=(Variant.V0, Variant.V1, Variant.V2, Variant.V4, Variant.V5),
        provisional_rubric=_gap_rubric(Variant.V5, stage="provisional"),
        leak_markers=("ATTACK_LABEL_EXFIL",),
        claim_prompts=_claim_prompts(),
    )


def _claim_prompts() -> dict[str, tuple[ClaimPrompt, ...]]:
    full = (
        ClaimPrompt(
            finding_id="F_HOST",
            prompt="Identify the internal host that originated the transfer and cite the records that link it to the flow.",
        ),
        ClaimPrompt(
            finding_id="F_PROC",
            prompt="Identify the initiating process and account, and cite the records that show them.",
        ),
        ClaimPrompt(
            finding_id="F_AUTH",
            prompt="State whether a scheduled job authorizes this transfer, and cite the records used.",
        ),
    )
    gap = (
        ClaimPrompt(
            finding_id="F_FLOW",
            prompt="Describe the outbound transfer that is visible and cite its record.",
        ),
        ClaimPrompt(
            finding_id="F_GAP",
            prompt="State whether the internal host and initiating process can be determined from the accessible records.",
        ),
    )
    return {"V0": full, "V2": full, "V4": gap, "V5:stage1": gap, "V5:stage2": full}


def entity_mapping() -> EntityMapping:
    return EntityMapping(
        hosts={"srv-app-03": "host-7f3a", "srv-backup-01": "host-c2b9"},
        users={"jdoe": "user-11", "svc-backup": "user-22"},
        ips={
            "10.0.0.5": "10.9.1.5",
            "10.1.4.20": "10.9.9.7",
            "203.0.113.50": "198.51.100.20",
        },
        preserve_exact=("curl", "backup.sh"),
    )


def investigation_policy(condition: ConditionId, variant: Variant) -> ScriptedPolicy:
    """A scripted walk of the synthetic case for conditions A and H."""
    if condition not in {ConditionId.A, ConditionId.H}:
        raise ValueError("the synthetic walk is defined for A and H")
    capabilities = Capabilities.for_condition(condition)
    contracted = condition is ConditionId.H
    if variant is Variant.V5:
        stages = [
            [
                _query("search_events", {"source": "netflow"}, contracted, falsify=False),
                _query("search_events", {"source": "jobs"}, contracted, falsify=False),
            ],
            [
                _query("search_events", {"source": "nat"}, contracted, falsify=False),
                _query("search_events", {"source": "process"}, contracted, falsify=contracted),
            ],
        ]
        answers = [_gap_answer(), _malicious_answer()]
    else:
        stages = [[
            _query("list_sources", {}, contracted, falsify=False),
            _query("search_events", {"source": "netflow"}, contracted, falsify=False),
            _query("search_events", {"source": "nat"}, contracted, falsify=False),
            _query("search_events", {"source": "process"}, contracted, falsify=False),
            _query("search_events", {"source": "jobs"}, contracted, falsify=contracted),
        ]]
        answers = [_gap_answer() if variant is Variant.V4 and condition is ConditionId.H else _answer_for(condition, variant)]
    return ScriptedPolicy(capabilities, stages, answers)


def _answer_for(condition: ConditionId, variant: Variant) -> FinalAnswer:
    if variant is Variant.V4 and condition is ConditionId.A:
        return FinalAnswer(
            probabilities=_probs(0.66, 0.14, 0.20),
            verdict=Verdict.MALICIOUS,
            findings=[
                FindingAssertion(
                    finding_id="F_HOST",
                    text="The transfer originated at srv-app-03.",
                    record_ids=["proc-1"],
                )
            ],
            limitations="",
        )
    if variant is Variant.V4:
        return _gap_answer()
    return _malicious_answer()


def _malicious_answer() -> FinalAnswer:
    return FinalAnswer(
        probabilities=_probs(0.74, 0.08, 0.18),
        verdict=Verdict.MALICIOUS,
        findings=[
            FindingAssertion(
                finding_id="F_HOST",
                text="The internal host of the transfer is srv-app-03.",
                record_ids=["flow-1", "nat-1"],
            ),
            FindingAssertion(
                finding_id="F_PROC",
                text="The initiating process is curl, run by jdoe.",
                record_ids=["proc-1"],
            ),
            FindingAssertion(
                finding_id="F_AUTH",
                text="No scheduled backup covers the 14:22 transfer.",
                record_ids=["flow-1", "job-1"],
            ),
        ],
        limitations="Destination reputation was not treated as authorization.",
    )


def _gap_answer() -> FinalAnswer:
    return FinalAnswer(
        probabilities=_probs(0.12, 0.10, 0.78),
        verdict=Verdict.INSUFFICIENT,
        findings=[
            FindingAssertion(
                finding_id="F_FLOW",
                text="A large outbound transfer from 10.0.0.5 is present.",
                record_ids=["flow-1"],
            ),
            FindingAssertion(
                finding_id="F_GAP",
                text="The internal host and initiating process are not determined by the accessible records.",
                record_ids=["flow-1"],
            ),
        ],
        limitations="The mount does not link the external address to an internal host.",
    )


def _query(tool: str, parameters: dict[str, object], contracted: bool, falsify: bool) -> Proposal:
    contract = None
    if contracted:
        contract = ActionContract(
            question=f"What does {tool} {parameters or 'the catalog'} discriminate?",
            hypotheses=[
                HypothesisPrediction(
                    hypothesis_id="h_backup",
                    explanation="Authorized backup during its named window.",
                    predicted_observation="A backup job on the translated host overlaps the flow and names the same process.",
                ),
                HypothesisPrediction(
                    hypothesis_id="h_unauth",
                    explanation="Unauthorized transfer outside any backup window.",
                    predicted_observation="No backup job overlaps the flow, and a non-backup process starts immediately before it.",
                ),
            ],
            tool=tool,
            parameters=dict(parameters),
            discrimination="Only one of the two predicted observations can match the returned rows.",
            resource_estimate="1 call",
        )
    return Proposal(
        kind=ActionKind.QUERY,
        tool=tool,
        parameters=dict(parameters),
        contract=contract,
        generated_tokens=8,
        falsification_check=falsify,
    )


def _probs(malicious: float, benign: float, insufficient: float) -> dict[str, float]:
    return {
        Verdict.MALICIOUS.value: malicious,
        Verdict.BENIGN.value: benign,
        Verdict.INSUFFICIENT.value: insufficient,
    }


def _rubrics() -> dict[Variant, Rubric]:
    v0 = Rubric(
        variant=Variant.V0,
        admissible=(Verdict.MALICIOUS,),
        mandatory=(
            MandatoryFinding(
                finding_id="F_HOST",
                description="The internal host of the transfer is srv-app-03.",
                evidence_paths=(("flow-1", "nat-1"), ("proc-1",)),
            ),
            MandatoryFinding(
                finding_id="F_PROC",
                description="The initiating process is curl, run by jdoe.",
                evidence_paths=(("proc-1",),),
            ),
            MandatoryFinding(
                finding_id="F_AUTH",
                description="No scheduled backup on srv-backup-01 covers the 14:22 transfer.",
                evidence_paths=(("flow-1", "job-1"),),
            ),
        ),
    )
    return {
        Variant.V0: v0,
        Variant.V2: v0.model_copy(update={"variant": Variant.V2}),
        Variant.V4: _gap_rubric(Variant.V4, stage="single"),
        Variant.V5: v0.model_copy(update={"variant": Variant.V5, "stage": "final"}),
    }


def _gap_rubric(variant: Variant, stage: str) -> Rubric:
    return Rubric(
        variant=variant,
        admissible=(Verdict.INSUFFICIENT,),
        stage=stage,  # type: ignore[arg-type]
        mandatory=(
            MandatoryFinding(
                finding_id="F_FLOW",
                description="A large outbound transfer from 10.0.0.5 is present.",
                evidence_paths=(("flow-1",),),
            ),
            MandatoryFinding(
                finding_id="F_GAP",
                description="The internal host and initiating process are not determined by the accessible records.",
                evidence_paths=(("flow-1",),),
            ),
        ),
    )


def _flow() -> Event:
    return _event(
        "flow-1",
        "netflow",
        "flow",
        "2024-04-11T14:22:03Z",
        {"src_ip": "10.0.0.5", "dst_ip": "203.0.113.50"},
        "flow src=10.0.0.5 dst=203.0.113.50 bytes=5000000000 proto=tcp",
    )


def _nat() -> Event:
    return _event(
        "nat-1",
        "nat",
        "translation",
        "2024-04-11T14:22:03Z",
        {"external_ip": "10.0.0.5", "internal_ip": "10.1.4.20", "host": "srv-app-03"},
        "nat external=10.0.0.5 internal=10.1.4.20 host=srv-app-03",
    )


def _process() -> Event:
    return _event(
        "proc-1",
        "process",
        "process_start",
        "2024-04-11T14:21:58Z",
        {"host": "srv-app-03", "user": "jdoe", "process": "curl", "dst_ip": "203.0.113.50"},
        "process host=srv-app-03 user=jdoe exe=curl dst=203.0.113.50",
    )


def _job() -> Event:
    return _event(
        "job-1",
        "jobs",
        "scheduled_job",
        "2024-04-11T02:00:00Z",
        {"host": "srv-backup-01", "user": "svc-backup", "process": "backup.sh"},
        "job host=srv-backup-01 user=svc-backup exe=backup.sh window=02:00-02:30",
    )


def _distractors() -> tuple[Event, ...]:
    rows = []
    for index in range(1, 6):
        rows.append(
            _event(
                f"dns-{index}",
                "dns",
                "query",
                f"2024-04-11T09:00:0{index}Z",
                {"host": "srv-mail-02", "query": f"name-{index}.example.test"},
                f"dns host=srv-mail-02 q=name-{index}.example.test",
                unrelated=True,
            )
        )
    rows.append(
        _event(
            "foreign-1",
            "dns",
            "query",
            "2024-04-11T09:30:00Z",
            {"host": "srv-other"},
            "dns host=srv-other q=other.example.test",
            unrelated=True,
            environment="other-lab",
        )
    )
    rows.append(
        _event(
            "related-1",
            "dns",
            "query",
            "2024-04-11T14:22:04Z",
            {"host": "srv-app-03"},
            "dns host=srv-app-03 q=collector.example.test",
            unrelated=False,
        )
    )
    return tuple(rows)


def _event(
    record_id: str,
    source: str,
    event_type: str,
    timestamp: str,
    entities: dict[str, str],
    raw: str,
    unrelated: bool = False,
    environment: str = ENVIRONMENT,
) -> Event:
    return Event(
        record_id=record_id,
        source=source,
        event_type=event_type,
        timestamp=timestamp,
        entities=entities,
        raw=raw,
        origin_hash=f"origin-{record_id}",
        origin_offset=1,
        environment=environment,
        unrelated=unrelated,
        parser_status="ok",
    )
