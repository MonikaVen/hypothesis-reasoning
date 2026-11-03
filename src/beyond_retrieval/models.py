"""Shared contracts.

Evaluator-only objects (rubrics, labels, variant cuts) and agent-facing
objects (task views, tool results) live in this module so every layer
imports the types from one place and none of them invent a parallel schema.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Verdict(StrEnum):
    MALICIOUS = "malicious"
    BENIGN = "benign"
    INSUFFICIENT = "insufficient_evidence"


class Variant(StrEnum):
    V0 = "V0"
    V1 = "V1"
    V2 = "V2"
    V3 = "V3"
    V4 = "V4"
    V5 = "V5"


class ConditionId(StrEnum):
    R = "R"
    A = "A"
    P = "P"
    H = "H"
    AS = "AS"
    HS = "HS"


class Ablation(StrEnum):
    NONE = "none"
    FLAT_MEMORY = "flat_memory"
    NO_PREDICTIONS = "no_predictions"
    NO_CONTRADICTION_CHECK = "no_contradiction_check"
    NO_SCRIPT_VALIDATION = "no_script_validation"
    NO_LESSONS = "no_lessons"
    FIXED_PROCEDURAL_LESSONS = "fixed_procedural_lessons"
    MATCHED_BUDGET_REFLECTION = "matched_budget_reflection"


class RunMode(StrEnum):
    INVESTIGATION = "investigation"
    SYNTHESIS = "synthesis"
    NO_EVIDENCE = "no_evidence"


class Partition(StrEnum):
    TRAIN = "train"
    DEVELOPMENT = "development"
    TEST = "test"
    EXTERNAL = "external"


class SourceId(StrEnum):
    AIT = "ait"
    OTRF = "otrf"
    OPTC = "optc"
    LANL = "lanl"
    BOTSV3 = "botsv3"
    SYNTHETIC = "synthetic"


class NodeKind(StrEnum):
    OBSERVATION = "observation"
    HYPOTHESIS = "hypothesis"
    ARTIFACT = "artifact"
    QUESTION = "question"


class EdgeRelation(StrEnum):
    SUPPORT = "support"
    CONTRADICTION = "contradiction"
    DERIVATION = "derivation"
    DEPENDENCY = "dependency"


class ObservationStatus(StrEnum):
    OK = "ok"
    FAILED_TOOL = "failed_tool"
    EMPTY = "empty"
    TRUNCATED = "truncated"
    VALID_NEGATIVE = "valid_negative"


class ActionKind(StrEnum):
    QUERY = "query"
    SCRIPT = "script"
    STOP = "stop"


class RunStatus(StrEnum):
    COMPLETED = "completed"
    BUDGET_EXHAUSTED = "budget_exhausted"
    AGENT_ERROR = "agent_error"
    INFRA_OUTAGE = "infra_outage"


class GuidanceKind(StrEnum):
    DOMAIN_FACTS = "domain_facts"
    PLAYBOOK = "playbook"
    DOMAIN_FACTS_AND_LESSONS = "domain_facts_and_lessons"
    FIXED_RETRIEVAL = "fixed_retrieval"


RECORDED_VARIANTS: tuple[Variant, ...] = (
    Variant.V0,
    Variant.V1,
    Variant.V2,
    Variant.V4,
    Variant.V5,
)

CORE_CONDITIONS: tuple[ConditionId, ...] = (
    ConditionId.A,
    ConditionId.H,
    ConditionId.AS,
    ConditionId.HS,
)

QUERY_TOOLS: tuple[str, ...] = (
    "list_sources",
    "describe_schema",
    "search_events",
    "get_records",
    "aggregate_events",
)

# Curator fields that must never appear on an agent-facing record.
FORBIDDEN_AGENT_KEYS: frozenset[str] = frozenset(
    {
        "label",
        "attack_label",
        "scenario_title",
        "walkthrough",
        "evaluator_notes",
        "rubric",
        "operator_intent",
        "unrelated",
        "adjudication",
        "mechanism",
        "cut_record_ids",
        "evidence_paths",
    }
)


class BudgetCaps(FrozenModel):
    max_tool_calls: int = 40
    max_generated_tokens: int = 30_000
    max_wall_seconds: float = 20 * 60
    max_script_cpu_seconds: float = 120

    def split_in_half(self) -> BudgetCaps:
        """Each V5 stage gets half of the cumulative cap. Per-call script CPU stays."""
        return BudgetCaps(
            max_tool_calls=self.max_tool_calls // 2,
            max_generated_tokens=self.max_generated_tokens // 2,
            max_wall_seconds=self.max_wall_seconds / 2,
            max_script_cpu_seconds=self.max_script_cpu_seconds,
        )


class QueryLimits(FrozenModel):
    page_size: int = 200
    max_chars: int = 12_000


class InferenceSettings(FrozenModel):
    temperature: float
    top_p: float
    reasoning_effort: str = "unset"
    input_token_ceiling: int = 12_000
    output_token_ceiling: int = 4_000


class Capabilities(FrozenModel):
    condition: ConditionId
    adaptive_queries: bool
    custom_scripts: bool
    script_validation: bool
    evidence_graph: bool
    graph_organization: bool
    action_contracts: bool
    contradiction_check: bool
    experience_lessons: bool
    procedural_lessons: bool
    matched_reflection: bool
    guidance: GuidanceKind

    @classmethod
    def for_condition(
        cls,
        condition: ConditionId,
        ablation: Ablation = Ablation.NONE,
    ) -> Capabilities:
        """Capability matrix from the protocol.

        Ablations can remove a capability the condition already has.
        They cannot grant scripts, lessons, or contracts to a condition
        that does not have them. H never receives experience lessons:
        the H-versus-A contrast is structure, not extra guidance.
        """
        base = {
            ConditionId.R: dict(
                adaptive_queries=False,
                custom_scripts=False,
                script_validation=False,
                evidence_graph=False,
                graph_organization=False,
                action_contracts=False,
                contradiction_check=False,
                experience_lessons=False,
                guidance=GuidanceKind.FIXED_RETRIEVAL,
            ),
            ConditionId.A: dict(
                adaptive_queries=True,
                custom_scripts=False,
                script_validation=False,
                evidence_graph=False,
                graph_organization=False,
                action_contracts=False,
                contradiction_check=False,
                experience_lessons=False,
                guidance=GuidanceKind.DOMAIN_FACTS,
            ),
            ConditionId.P: dict(
                adaptive_queries=True,
                custom_scripts=False,
                script_validation=False,
                evidence_graph=False,
                graph_organization=False,
                action_contracts=False,
                contradiction_check=False,
                experience_lessons=False,
                guidance=GuidanceKind.PLAYBOOK,
            ),
            ConditionId.H: dict(
                adaptive_queries=True,
                custom_scripts=False,
                script_validation=False,
                evidence_graph=True,
                graph_organization=True,
                action_contracts=True,
                contradiction_check=True,
                experience_lessons=False,
                guidance=GuidanceKind.DOMAIN_FACTS,
            ),
            ConditionId.AS: dict(
                adaptive_queries=True,
                custom_scripts=True,
                script_validation=False,
                evidence_graph=False,
                graph_organization=False,
                action_contracts=False,
                contradiction_check=False,
                experience_lessons=False,
                guidance=GuidanceKind.DOMAIN_FACTS,
            ),
            ConditionId.HS: dict(
                adaptive_queries=True,
                custom_scripts=True,
                script_validation=True,
                evidence_graph=True,
                graph_organization=True,
                action_contracts=True,
                contradiction_check=True,
                experience_lessons=True,
                guidance=GuidanceKind.DOMAIN_FACTS_AND_LESSONS,
            ),
        }[condition]
        procedural = False
        matched = False
        if ablation is Ablation.FLAT_MEMORY and base["evidence_graph"]:
            base["graph_organization"] = False
        elif ablation is Ablation.NO_PREDICTIONS and base["action_contracts"]:
            base["action_contracts"] = False
        elif ablation is Ablation.NO_CONTRADICTION_CHECK and base["contradiction_check"]:
            base["contradiction_check"] = False
        elif ablation is Ablation.NO_SCRIPT_VALIDATION and base["custom_scripts"]:
            base["script_validation"] = False
        elif ablation is Ablation.NO_LESSONS and base["experience_lessons"]:
            base["experience_lessons"] = False
            base["guidance"] = GuidanceKind.DOMAIN_FACTS
        elif ablation is Ablation.FIXED_PROCEDURAL_LESSONS and base["experience_lessons"]:
            procedural = True
        elif ablation is Ablation.MATCHED_BUDGET_REFLECTION:
            matched = True
        return cls(
            condition=condition,
            procedural_lessons=procedural,
            matched_reflection=matched,
            **base,
        )

    def for_mode(self, mode: RunMode) -> Capabilities:
        if mode is RunMode.INVESTIGATION:
            return self
        if mode is RunMode.NO_EVIDENCE:
            return self.model_copy(update={"custom_scripts": False, "script_validation": False})
        return self.model_copy(
            update={
                "adaptive_queries": False,
                "custom_scripts": False,
                "script_validation": False,
                "action_contracts": False,
            }
        )


class Proposition(FrozenModel):
    prop_id: str
    text: str


class DomainReference(FrozenModel):
    propositions: tuple[Proposition, ...]

    def by_id(self) -> dict[str, Proposition]:
        return {item.prop_id: item for item in self.propositions}


class PlaybookStep(FrozenModel):
    kind: Literal["fact", "procedure"]
    text: str
    proposition_ids: tuple[str, ...] = ()


class Playbook(FrozenModel):
    steps: tuple[PlaybookStep, ...]


class Lesson(FrozenModel):
    lesson_id: str
    applicability: str
    recommended_test: str
    failure_conditions: str
    provenance_case_ids: tuple[str, ...]


class FrozenLessons(FrozenModel):
    lessons: tuple[Lesson, ...]
    content_hash: str


class HypothesisPrediction(StrictModel):
    hypothesis_id: str
    explanation: str
    predicted_observation: str


class ActionContract(StrictModel):
    question: str
    hypotheses: list[HypothesisPrediction] = Field(default_factory=list)
    open_set_alternative: bool = False
    tool: str
    parameters: dict[str, object] = Field(default_factory=dict)
    discrimination: str
    resource_estimate: str


class ScriptContract(StrictModel):
    inputs: list[str]
    join_keys: list[str]
    units: dict[str, str]
    output_schema: dict[str, str]
    assumptions: list[str]


class Proposal(StrictModel):
    kind: ActionKind
    tool: str | None = None
    parameters: dict[str, object] = Field(default_factory=dict)
    contract: ActionContract | None = None
    script_source: str | None = None
    script_contract: ScriptContract | None = None
    generated_tokens: int = 0
    falsification_check: bool = False
    falsification_infeasible: str | None = None


class AgentRecord(FrozenModel):
    """One row as the agent is allowed to see it."""

    record_id: str
    source: str
    event_type: str
    timestamp: str
    timezone: str
    entities: dict[str, str]
    raw: str
    content_hash: str
    origin_offset: int
    parser_status: Literal["ok", "failed", "missing_field"] = "ok"


class ToolResult(StrictModel):
    status: ObservationStatus
    records: list[AgentRecord] = Field(default_factory=list)
    summary: str = ""
    truncated: bool = False
    cursor: str | None = None
    content_hash: str = ""
    error: str | None = None
    script_cpu_seconds: float = 0.0
    execution_success: bool | None = None
    validation: Literal["pass", "fail", "not_applicable", "not_run"] = "not_run"
    admitted_findings: list[dict[str, object]] = Field(default_factory=list)
    claimed_findings: list[dict[str, object]] = Field(default_factory=list)


class FindingAssertion(StrictModel):
    finding_id: str
    text: str
    record_ids: list[str]


class FinalAnswer(StrictModel):
    probabilities: dict[str, float]
    verdict: Verdict
    findings: list[FindingAssertion] = Field(default_factory=list)
    limitations: str = ""
    unresolved: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _known_categories(self) -> FinalAnswer:
        unknown = set(self.probabilities) - {item.value for item in Verdict}
        if unknown:
            raise ValueError(f"unknown probability categories: {sorted(unknown)}")
        return self


class MandatoryFinding(FrozenModel):
    finding_id: str
    description: str
    evidence_paths: tuple[tuple[str, ...], ...] = ()


class Rubric(FrozenModel):
    variant: Variant
    admissible: tuple[Verdict, ...]
    mandatory: tuple[MandatoryFinding, ...]
    stage: Literal["single", "provisional", "final"] = "single"

    def finding(self, finding_id: str) -> MandatoryFinding:
        for item in self.mandatory:
            if item.finding_id == finding_id:
                return item
        raise KeyError(finding_id)


class Event(FrozenModel):
    record_id: str
    source: str
    event_type: str
    timestamp: str
    timezone: str = "UTC"
    entities: dict[str, str]
    raw: str
    origin_hash: str
    origin_offset: int
    parser_status: Literal["ok", "failed", "missing_field"] = "ok"
    environment: str
    unrelated: bool = False

    def agent_record(self, content_hash: str) -> AgentRecord:
        return AgentRecord(
            record_id=self.record_id,
            source=self.source,
            event_type=self.event_type,
            timestamp=self.timestamp,
            timezone=self.timezone,
            entities=dict(self.entities),
            raw=self.raw,
            content_hash=content_hash,
            origin_offset=self.origin_offset,
            parser_status=self.parser_status,
        )


class EntityMapping(FrozenModel):
    hosts: dict[str, str] = Field(default_factory=dict)
    users: dict[str, str] = Field(default_factory=dict)
    ips: dict[str, str] = Field(default_factory=dict)
    preserve_exact: tuple[str, ...] = ()

    def pairs(self) -> list[tuple[str, str]]:
        items: list[tuple[str, str]] = []
        items.extend(self.hosts.items())
        items.extend(self.users.items())
        items.extend(self.ips.items())
        return items


class V4Spec(FrozenModel):
    blocked_finding_id: str
    cut_record_ids: tuple[str, ...]


class V5Spec(FrozenModel):
    stage1_ids: tuple[str, ...]
    stage2_ids: tuple[str, ...]


class CaseRecord(FrozenModel):
    case_id: str
    partition: Partition
    source: SourceId
    environment: str
    campaign_id: str
    mechanism_group: str
    overlap_group: str
    question: str
    alert: str


class ClaimPrompt(FrozenModel):
    """A required claim the agent must answer. The prompt does not contain the gold text."""

    finding_id: str
    prompt: str


class CaseBundle(FrozenModel):
    """Full curator object. Never pass this to a policy."""

    record: CaseRecord
    events: tuple[Event, ...]
    distractors: tuple[Event, ...] = ()
    rubrics: dict[Variant, Rubric]
    v1_mapping: EntityMapping | None = None
    v4: V4Spec | None = None
    v5: V5Spec | None = None
    eligible_variants: tuple[Variant, ...] = (Variant.V0,)
    provisional_rubric: Rubric | None = None
    leak_markers: tuple[str, ...] = ()
    claim_prompts: dict[str, tuple[ClaimPrompt, ...]] = Field(default_factory=dict)


class TaskView(FrozenModel):
    """Everything a policy may see at the start of a run."""

    case_id: str
    question: str
    alert: str
    tools: tuple[str, ...]
    guidance_text: str
    variant: Variant
    mode: RunMode
    claims: tuple[ClaimPrompt, ...] = ()


class RunSpec(FrozenModel):
    experiment: str
    case_id: str
    variant: Variant
    condition: ConditionId
    backbone_id: str
    repeat_index: int
    ablation: Ablation = Ablation.NONE
    mode: RunMode = RunMode.INVESTIGATION
    seed: int = 0


class GraphNode(FrozenModel):
    node_id: str
    kind: NodeKind
    payload_hash: str
    text: str


class GraphEdge(FrozenModel):
    edge_id: str
    source: str
    target: str
    relation: EdgeRelation
    asserted_by: Literal["agent", "script"]
    script_validated: bool = False


class BudgetUsage(StrictModel):
    tool_calls: int = 0
    generated_tokens: int = 0
    wall_seconds: float = 0.0
    script_cpu_seconds: float = 0.0
    currency_cost: float | None = None
    token_accounting: Literal["approximate", "provider"] = "approximate"


class StepTrace(StrictModel):
    index: int
    kind: ActionKind
    tool: str | None
    parameters: dict[str, object] = Field(default_factory=dict)
    contract_valid: bool | None = None
    contract_errors: list[str] = Field(default_factory=list)
    status: str
    record_ids: list[str] = Field(default_factory=list)
    observation_hash: str = ""
    generated_tokens: int = 0
    script_validation: str = "not_run"
    falsification_check: bool = False
    script_hash: str = ""
    error: str | None = None


class MechanicalScore(StrictModel):
    passed: bool
    admissible_verdict: bool
    missing_findings: list[str]
    unknown_citations: list[str]
    unretrieved_citations: list[str]
    empty_answer: bool
    probability_sum_ok: bool
    needs_human: bool = True


class Adjudication(StrictModel):
    reviewer_ids: list[str]
    support_holds: bool
    material_unsupported: bool
    notes: str = ""
    source: Literal["human", "fixture_double"]


class ESRResult(StrictModel):
    status: Literal["pass", "fail", "pending_adjudication", "missing", "not_scored"]
    passed: bool


class RunRecord(StrictModel):
    spec: RunSpec
    protocol_hash: str
    guidance_hash: str
    snapshot_hash: str
    status: RunStatus
    counts_as_missing: bool
    started_at: str
    ended_at: str
    budget: BudgetUsage
    steps: list[StepTrace]
    graph_nodes: list[GraphNode]
    graph_edges: list[GraphEdge]
    provisional_answer: FinalAnswer | None = None
    final_answer: FinalAnswer | None = None
    falsification_attempt_present: bool = False
    falsification_infeasible: str | None = None
    mechanical: MechanicalScore | None = None
    provisional_mechanical: MechanicalScore | None = None
    adjudication: Adjudication | None = None
    esr: ESRResult | None = None
    exclusion_reason: str | None = None
    stage_snapshot_hashes: dict[str, str] = Field(default_factory=dict)


class Contrast(FrozenModel):
    hypothesis: str
    treatment: str
    control: str
    variant: str
    endpoint: str
    experiment: str
    direction: Literal["higher_is_better"] = "higher_is_better"


class ProtocolFreeze(FrozenModel):
    freeze_id: str
    sealed: bool
    backbone_ids: tuple[str, ...]
    inference: InferenceSettings
    caps: BudgetCaps
    query_limits: QueryLimits
    guidance_hash: str
    lesson_set_hash: str
    analysis_hash: str
    tool_schema_hash: str
    rubric_hashes: dict[str, str]


class CaseEligibility(FrozenModel):
    case_id: str
    environment: str
    campaign_id: str
    variants: tuple[Variant, ...]
