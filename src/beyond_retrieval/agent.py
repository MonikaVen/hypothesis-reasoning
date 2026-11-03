"""Policy boundary, budget, action contracts, and the audit graph.

Policies propose. The runner executes tools, charges the budget, and
appends immutable observation nodes. Nothing in this module opens a
snapshot or reads a rubric.
"""

from __future__ import annotations

from typing import Protocol

from beyond_retrieval.errors import CapabilityError
from beyond_retrieval.hashing import content_hash
from beyond_retrieval.models import (
    QUERY_TOOLS,
    ActionContract,
    BudgetCaps,
    BudgetUsage,
    Capabilities,
    DomainReference,
    EdgeRelation,
    FinalAnswer,
    FrozenLessons,
    GraphEdge,
    GraphNode,
    NodeKind,
    Playbook,
    Proposal,
    RunMode,
    TaskView,
    ToolResult,
)

USELESS_PREDICTIONS = frozenset(
    {
        "useful information",
        "the tool will provide useful information",
        "the tool will return useful information",
    }
)


class Policy(Protocol):
    capabilities: Capabilities

    def start(self, view: TaskView) -> None:
        """Receive the agent-facing task. No rubric is available here."""

    def propose(self, observation: ToolResult | None) -> Proposal:
        """Return the next action. The runner validates and executes it."""

    def finalize(self) -> FinalAnswer:
        """Structured answer. Verbosity is not a score."""

    def notify_release(self) -> None:
        """V5 only: the runner has widened the mount. Default is a no-op."""


class Budget:
    def __init__(self, caps: BudgetCaps) -> None:
        self.caps = caps
        self.usage = BudgetUsage()
        self.stop_reason: str | None = None

    def exhausted(self) -> bool:
        if self.usage.tool_calls >= self.caps.max_tool_calls:
            self.stop_reason = "tool_calls"
            return True
        if self.usage.generated_tokens >= self.caps.max_generated_tokens:
            self.stop_reason = "generated_tokens"
            return True
        if self.usage.wall_seconds >= self.caps.max_wall_seconds:
            self.stop_reason = "wall_seconds"
            return True
        return False

    def charge_tokens(self, tokens: int) -> None:
        if tokens < 0:
            raise CapabilityError("negative token charge")
        self.usage.generated_tokens += tokens

    def charge_call(self, wall_seconds: float = 0.0, script_cpu_seconds: float = 0.0) -> None:
        self.usage.tool_calls += 1
        self.usage.wall_seconds += max(0.0, wall_seconds)
        self.usage.script_cpu_seconds += max(0.0, script_cpu_seconds)


def validate_contract(contract: ActionContract | None) -> list[str]:
    """Structure only. The validator does not judge whether a prediction is true."""
    if contract is None:
        return ["missing action contract"]
    errors: list[str] = []
    if len(contract.question.strip()) < 8:
        errors.append("question is not specific")
    if not contract.tool.strip():
        errors.append("missing tool")
    if not contract.discrimination.strip():
        errors.append("missing discrimination statement")
    if not contract.resource_estimate.strip():
        errors.append("missing resource estimate")
    predictions = [item.predicted_observation.strip() for item in contract.hypotheses]
    lowered = [item.lower() for item in predictions]
    if any(item in USELESS_PREDICTIONS for item in lowered):
        errors.append("predictions must be distinguishable observations")
    if contract.open_set_alternative:
        if len(contract.hypotheses) < 1:
            errors.append("open-set alternative still needs a named hypothesis")
    elif len(contract.hypotheses) < 2:
        errors.append("need two hypotheses or an open-set alternative")
    if len(predictions) >= 2 and len(set(lowered)) < 2:
        errors.append("predictions are not distinguishable")
    return errors


class EvidenceGraph:
    """Audit graph. Edges are assertions. There is no field for 'this edge is true'."""

    def __init__(self) -> None:
        self.nodes: dict[str, GraphNode] = {}
        self.edges: list[GraphEdge] = []

    def add_observation(self, record_id: str, text: str, payload_hash: str) -> None:
        self._add(
            GraphNode(
                node_id=f"obs:{record_id}",
                kind=NodeKind.OBSERVATION,
                payload_hash=payload_hash,
                text=text,
            )
        )

    def add_hypothesis(self, hypothesis_id: str, text: str) -> None:
        self._add(
            GraphNode(
                node_id=f"hyp:{hypothesis_id}",
                kind=NodeKind.HYPOTHESIS,
                payload_hash=content_hash(text),
                text=text,
            )
        )

    def add_artifact(self, artifact_id: str, text: str, script_validated: bool) -> None:
        self._add(
            GraphNode(
                node_id=f"art:{artifact_id}",
                kind=NodeKind.ARTIFACT,
                payload_hash=content_hash({"text": text, "script_validated": script_validated}),
                text=text,
            )
        )

    def add_question(self, question_id: str, text: str) -> None:
        self._add(
            GraphNode(
                node_id=f"q:{question_id}",
                kind=NodeKind.QUESTION,
                payload_hash=content_hash(text),
                text=text,
            )
        )

    def add_edge(
        self,
        source: str,
        target: str,
        relation: EdgeRelation,
        asserted_by: str,
        script_validated: bool = False,
    ) -> None:
        if asserted_by not in {"agent", "script"}:
            raise CapabilityError("edge asserted_by must be agent or script")
        edge = GraphEdge(
            edge_id=f"e{len(self.edges)}",
            source=source,
            target=target,
            relation=relation,
            asserted_by=asserted_by,  # type: ignore[arg-type]
            script_validated=script_validated,
        )
        self.edges.append(edge)

    def render(self, organized: bool) -> str:
        """Typed records, or the same nodes as a flat list. Same texts either way."""
        lines = []
        for node in self.nodes.values():
            if organized:
                lines.append(f"{node.kind.value}\t{node.node_id}\t{node.text}")
            else:
                lines.append(node.text)
        if organized:
            for edge in self.edges:
                lines.append(
                    f"edge\t{edge.source}\t{edge.relation.value}\t{edge.target}\tasserted_by={edge.asserted_by}"
                )
        return "\n".join(lines)

    def _add(self, node: GraphNode) -> None:
        current = self.nodes.get(node.node_id)
        if current is not None and current.payload_hash != node.payload_hash:
            raise CapabilityError(f"observation {node.node_id} is immutable")
        if current is None:
            self.nodes[node.node_id] = node


def assemble_guidance(
    capabilities: Capabilities,
    reference: DomainReference,
    playbook: Playbook | None,
    lessons: FrozenLessons | None,
    mode: RunMode,
) -> str:
    """Build the only guidance string the policy will receive."""
    if capabilities.experience_lessons:
        if lessons is None:
            raise CapabilityError("condition requires a frozen lesson set")
    elif lessons is not None and lessons.lessons:
        raise CapabilityError(
            f"{capabilities.condition.value} cannot receive experience lessons"
        )
    if capabilities.guidance.value == "playbook":
        if playbook is None:
            raise CapabilityError("playbook condition has no playbook")
        lines = ["Playbook:"]
        for step in playbook.steps:
            lines.append(f"{step.kind}: {step.text}")
        return "\n".join(lines)
    lines = ["Domain reference:"]
    for prop in reference.propositions:
        lines.append(f"{prop.prop_id}: {prop.text}")
    if capabilities.experience_lessons and lessons is not None:
        lines.append("Frozen lessons:")
        for lesson in lessons.lessons:
            lines.append(
                f"{lesson.lesson_id}: if {lesson.applicability}; test {lesson.recommended_test}; "
                f"failure {lesson.failure_conditions}"
            )
    if mode is RunMode.SYNTHESIS:
        lines.append("Synthesis mode: no further retrieval or code execution is available.")
    if mode is RunMode.NO_EVIDENCE:
        lines.append("No incident records are mounted.")
    return "\n".join(lines)


def allowed_tools(capabilities: Capabilities) -> tuple[str, ...]:
    if not capabilities.adaptive_queries:
        return ()
    return QUERY_TOOLS
