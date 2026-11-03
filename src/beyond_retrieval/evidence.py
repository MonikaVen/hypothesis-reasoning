"""Recorded-data snapshots and the agent-facing mount.

V0, V1, V2, V4, and V5 are pure functions of a curator bundle plus a seed.
V3 is refused here. A regenerated pair is a new recording, not a transform
of an old one. The database handed to a script contains only the rows the
current stage is allowed to read.
"""

from __future__ import annotations

import json
import random
import re
import sqlite3
from pathlib import Path

from beyond_retrieval.errors import CounterfactualRequiresRegeneration, InvalidVariant
from beyond_retrieval.hashing import content_hash
from beyond_retrieval.models import (
    FORBIDDEN_AGENT_KEYS,
    AgentRecord,
    CaseBundle,
    EntityMapping,
    Event,
    Rubric,
    V4Spec,
    V5Spec,
    Variant,
)

class EvidenceMount:
    """Visible events for one stage. `events` is the complete visible set."""

    def __init__(self, events: list[Event], variant: Variant, label: str) -> None:
        self.events = events
        self.variant = variant
        self.label = label
        self.content_hash = content_hash([_event_payload(event) for event in events])
        self._by_id = {event.record_id: event for event in events}

    def ids(self) -> set[str]:
        return set(self._by_id)

    def agent_records(self) -> list[AgentRecord]:
        return [event.agent_record(_record_hash(event)) for event in self.events]

    def materialize(self, path: Path) -> None:
        """Write a sqlite file that contains only visible rows."""
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            path.unlink()
        connection = sqlite3.connect(path)
        try:
            connection.execute(
                """
                create table events (
                    record_id text primary key,
                    source text not null,
                    event_type text not null,
                    timestamp text not null,
                    timezone text not null,
                    entities_json text not null,
                    raw text not null,
                    content_hash text not null,
                    origin_offset integer not null,
                    parser_status text not null
                )
                """
            )
            for event in self.events:
                dumped = event.agent_record(_record_hash(event))
                connection.execute(
                    """
                    insert into events (
                        record_id, source, event_type, timestamp, timezone,
                        entities_json, raw, content_hash, origin_offset, parser_status
                    ) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        dumped.record_id,
                        dumped.source,
                        dumped.event_type,
                        dumped.timestamp,
                        dumped.timezone,
                        _stable_entities(dumped.entities),
                        dumped.raw,
                        dumped.content_hash,
                        dumped.origin_offset,
                        dumped.parser_status,
                    ),
                )
            connection.commit()
        finally:
            connection.close()


def prepare_mount(
    case: CaseBundle,
    variant: Variant,
    seed: int = 0,
    stage: str | None = None,
) -> EvidenceMount:
    if variant is Variant.V3:
        raise CounterfactualRequiresRegeneration(
            "V3 is a regenerated paired recording, not a transform of this snapshot"
        )
    if variant not in case.eligible_variants:
        raise InvalidVariant(f"{case.record.case_id} is not eligible for {variant.value}")

    if variant is Variant.V0:
        events = list(case.events)
    elif variant is Variant.V1:
        if case.v1_mapping is None:
            raise InvalidVariant("V1 requires an entity mapping")
        events = rename_events(list(case.events), case.v1_mapping)
    elif variant is Variant.V2:
        events = _with_distractors(case, seed)
    elif variant is Variant.V4:
        if case.v4 is None or variant not in case.rubrics:
            raise InvalidVariant("V4 requires a cut and a rubric")
        validate_v4(case.events, case.rubrics[Variant.V0], case.v4)
        cut = set(case.v4.cut_record_ids)
        events = [event for event in case.events if event.record_id not in cut]
    elif variant is Variant.V5:
        if case.v5 is None:
            raise InvalidVariant("V5 requires a delayed-release spec")
        _validate_v5(case.v5, case.events)
        if stage == "stage1":
            visible = set(case.v5.stage1_ids)
        elif stage == "stage2":
            visible = set(case.v5.stage1_ids) | set(case.v5.stage2_ids)
        else:
            raise InvalidVariant("V5 mount requires stage='stage1' or stage='stage2'")
        events = [event for event in case.events if event.record_id in visible]
    else:
        raise InvalidVariant(f"unsupported variant {variant.value}")

    mount = EvidenceMount(events, variant, stage or variant.value)
    audit_mount(mount, case.leak_markers)
    return mount


def rubric_for(case: CaseBundle, variant: Variant, stage: str | None = None) -> Rubric:
    if variant is Variant.V1:
        if case.v1_mapping is None or Variant.V0 not in case.rubrics:
            raise InvalidVariant("V1 rubric requires a mapping and a V0 rubric")
        return rename_rubric(case.rubrics[Variant.V0], case.v1_mapping)
    if variant is Variant.V5:
        if stage == "stage1":
            if case.provisional_rubric is None:
                raise InvalidVariant("V5 is missing a provisional rubric")
            return case.provisional_rubric
        if stage == "stage2":
            if variant not in case.rubrics:
                raise InvalidVariant("V5 is missing a final rubric")
            return case.rubrics[variant]
        raise InvalidVariant("V5 scoring requires a stage")
    if variant not in case.rubrics:
        raise InvalidVariant(f"no rubric for {variant.value}")
    return case.rubrics[variant]


def validate_v4(events: tuple[Event, ...] | list[Event], v0_rubric: Rubric, spec: V4Spec) -> None:
    finding = v0_rubric.finding(spec.blocked_finding_id)
    if not finding.evidence_paths:
        raise InvalidVariant(f"{spec.blocked_finding_id} has no evidence paths to cut")
    present = {event.record_id for event in events}
    cut = set(spec.cut_record_ids)
    if not cut:
        raise InvalidVariant("V4 cut is empty")
    visible = present - cut
    for path in finding.evidence_paths:
        if not set(path) <= present:
            raise InvalidVariant(f"path {path} is not in the base snapshot")
        if set(path) <= visible:
            raise InvalidVariant(
                f"V4 cut leaves a support path for {spec.blocked_finding_id}: {path}"
            )
        if set(path).isdisjoint(cut):
            raise InvalidVariant(f"path {path} does not intersect the cut")


def rename_events(events: list[Event], mapping: EntityMapping) -> list[Event]:
    _require_bijective(mapping)
    renamed: list[Event] = []
    for event in events:
        entities = {key: _map_token(value, mapping) for key, value in event.entities.items()}
        renamed.append(
            event.model_copy(
                update={
                    "entities": entities,
                    "raw": apply_replacements(event.raw, mapping),
                }
            )
        )
    return renamed


def rename_rubric(rubric: Rubric, mapping: EntityMapping) -> Rubric:
    _require_bijective(mapping)
    mandatory = tuple(
        item.model_copy(update={"description": apply_replacements(item.description, mapping)})
        for item in rubric.mandatory
    )
    return rubric.model_copy(update={"variant": Variant.V1, "mandatory": mandatory})


def apply_replacements(text: str, mapping: EntityMapping) -> str:
    """Replace mapped identifiers. Preserved command strings stay literal."""
    masked, masks = _mask(text, mapping.preserve_exact)
    pairs = sorted(mapping.pairs(), key=lambda item: len(item[0]), reverse=True)
    for source, target in pairs:
        masked = re.sub(
            rf"(?<![A-Za-z0-9_-]){re.escape(source)}(?![A-Za-z0-9_-])",
            target,
            masked,
        )
    for token, placeholder in masks:
        masked = masked.replace(placeholder, token)
    return masked


def audit_mount(mount: EvidenceMount, leak_markers: tuple[str, ...] = ()) -> None:
    for record in mount.agent_records():
        payload = record.model_dump(mode="json")
        leaked = FORBIDDEN_AGENT_KEYS.intersection(payload)
        if leaked:
            raise InvalidVariant(f"agent record exposes {sorted(leaked)}")
        blob = str(payload)
        for marker in leak_markers:
            if marker and marker in blob:
                raise InvalidVariant(f"agent mount contains leak marker {marker!r}")


def _with_distractors(case: CaseBundle, seed: int) -> list[Event]:
    if not case.events:
        raise InvalidVariant("V2 base snapshot is empty")
    environment = case.events[0].environment
    if any(event.environment != environment for event in case.events):
        raise InvalidVariant("V2 base events span more than one environment")
    base_ids = {event.record_id for event in case.events}
    eligible = []
    for event in case.distractors:
        if event.environment != environment or not event.unrelated:
            continue
        if event.record_id in base_ids:
            continue
        eligible.append(event)
    target = len(case.events)  # additions that bring the total to 2x when available
    ordered = list(eligible)
    random.Random(seed).shuffle(ordered)
    chosen = ordered[:target]
    return list(case.events) + chosen


def _validate_v5(spec: V5Spec, events: tuple[Event, ...]) -> None:
    present = {event.record_id for event in events}
    stage1 = set(spec.stage1_ids)
    stage2 = set(spec.stage2_ids)
    if not stage2:
        raise InvalidVariant("V5 decisive release is empty")
    if stage1 & stage2:
        raise InvalidVariant("V5 stages overlap")
    if not stage1 <= present or not stage2 <= present:
        raise InvalidVariant("V5 cites records that are not in the snapshot")


def _require_bijective(mapping: EntityMapping) -> None:
    pairs = mapping.pairs()
    sources = [source for source, _ in pairs]
    targets = [target for _, target in pairs]
    if len(sources) != len(set(sources)):
        raise InvalidVariant("entity mapping repeats a source identifier")
    if len(targets) != len(set(targets)):
        raise InvalidVariant("entity mapping is not bijective")
    if set(sources) & set(targets):
        raise InvalidVariant("entity mapping reuses an original identifier as a new one")


def _map_token(value: str, mapping: EntityMapping) -> str:
    table = dict(mapping.pairs())
    return table.get(value, value)


def _mask(text: str, preserved: tuple[str, ...]) -> tuple[str, list[tuple[str, str]]]:
    masks: list[tuple[str, str]] = []
    masked = text
    for index, token in enumerate(sorted(preserved, key=len, reverse=True)):
        placeholder = f"\x00PRESERVED{index}\x00"
        if token in masked:
            masked = masked.replace(token, placeholder)
            masks.append((token, placeholder))
    return masked, masks


def _event_payload(event: Event) -> dict[str, object]:
    record = event.agent_record(_record_hash(event))
    return record.model_dump(mode="json")


def _record_hash(event: Event) -> str:
    body = {
        "record_id": event.record_id,
        "source": event.source,
        "event_type": event.event_type,
        "timestamp": event.timestamp,
        "timezone": event.timezone,
        "entities": event.entities,
        "raw": event.raw,
        "origin_offset": event.origin_offset,
        "parser_status": event.parser_status,
    }
    return content_hash(body)


def _stable_entities(entities: dict[str, str]) -> str:
    return json.dumps(entities, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
