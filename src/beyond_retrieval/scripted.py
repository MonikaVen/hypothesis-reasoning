"""Deterministic policies for the harness.

These are not the experimental conditions. They walk the same proposal
interface a model policy will implement, so the runner can be tested
without a backbone.
"""

from __future__ import annotations

from beyond_retrieval.models import ActionKind, Capabilities, FinalAnswer, Proposal, TaskView, ToolResult


class ScriptedPolicy:
    def __init__(
        self,
        capabilities: Capabilities,
        stages: list[list[Proposal]],
        answers: list[FinalAnswer],
    ) -> None:
        if not stages or not answers:
            raise ValueError("scripted policy needs at least one stage and one answer")
        self.capabilities = capabilities
        self.stages = stages
        self.answers = answers
        self.stage = 0
        self.cursor = 0
        self.views: list[TaskView] = []

    def start(self, view: TaskView) -> None:
        self.views.append(view)

    def notify_release(self) -> None:
        self.stage += 1
        self.cursor = 0

    def propose(self, observation: ToolResult | None) -> Proposal:
        actions = self.stages[min(self.stage, len(self.stages) - 1)]
        if self.cursor >= len(actions):
            return Proposal(kind=ActionKind.STOP)
        proposal = actions[self.cursor]
        self.cursor += 1
        return proposal

    def finalize(self) -> FinalAnswer:
        return self.answers[min(self.stage, len(self.answers) - 1)]
