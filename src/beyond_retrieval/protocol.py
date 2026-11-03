"""Builders for the frozen protocol object."""

from __future__ import annotations

from beyond_retrieval.grid import analysis_spec_hash
from beyond_retrieval.hashing import content_hash
from beyond_retrieval.models import BudgetCaps, InferenceSettings, ProtocolFreeze, QueryLimits
from beyond_retrieval.tools import TOOL_SCHEMA


def make_protocol(
    *,
    sealed: bool,
    guidance_hash: str = "unbound",
    lesson_set_hash: str = "none",
    rubric_hashes: dict[str, str] | None = None,
    caps: BudgetCaps | None = None,
    backbone_ids: tuple[str, ...] = ("backbone-a", "backbone-b"),
) -> ProtocolFreeze:
    return ProtocolFreeze(
        freeze_id="sealed" if sealed else "draft",
        sealed=sealed,
        backbone_ids=backbone_ids,
        inference=InferenceSettings(temperature=0.0, top_p=1.0, reasoning_effort="unset"),
        caps=caps or BudgetCaps(),
        query_limits=QueryLimits(),
        guidance_hash=guidance_hash,
        lesson_set_hash=lesson_set_hash,
        analysis_hash=analysis_spec_hash(),
        tool_schema_hash=content_hash(TOOL_SCHEMA),
        rubric_hashes=rubric_hashes or {},
    )
