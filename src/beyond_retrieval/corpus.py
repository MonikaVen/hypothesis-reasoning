"""Partitions, guidance audits, and lesson freeze.

Splits are properties of source environments, not of individual events.
Lessons may cite only training cases, and a playbook may only reorder
propositions that already appear in the shared domain reference.
"""

from __future__ import annotations

from beyond_retrieval.errors import LessonLeak, ProtocolError
from beyond_retrieval.hashing import content_hash
from beyond_retrieval.models import (
    CaseRecord,
    DomainReference,
    FrozenLessons,
    Lesson,
    Partition,
    Playbook,
)

# Whole AIT environments. Proposed freeze from the protocol, not a claim
# that the quotas are already filled.
AIT_ENVIRONMENTS: dict[str, tuple[str, ...]] = {
    Partition.TRAIN.value: ("fox", "harrison"),
    Partition.DEVELOPMENT.value: ("russellmitchell", "santos"),
    Partition.TEST.value: ("shaw", "wardbeck", "wheeler", "wilson"),
}


def audit_environment_splits(records: list[CaseRecord]) -> None:
    """One environment stays in one partition. Campaigns do not cross."""
    env_partition: dict[str, Partition] = {}
    campaign_partition: dict[str, Partition] = {}
    for record in records:
        previous = env_partition.get(record.environment)
        if previous is not None and previous != record.partition:
            raise ProtocolError(
                f"environment {record.environment} is in both {previous.value} and {record.partition.value}"
            )
        env_partition[record.environment] = record.partition
        camp = campaign_partition.get(record.campaign_id)
        if camp is not None and camp != record.partition:
            raise ProtocolError(
                f"campaign {record.campaign_id} crosses partitions {camp.value} and {record.partition.value}"
            )
        campaign_partition[record.campaign_id] = record.partition


def audit_reference(reference: DomainReference, banned_tokens: list[str]) -> None:
    """The shared fact list cannot carry incident-specific names."""
    ids: set[str] = set()
    for prop in reference.propositions:
        if prop.prop_id in ids:
            raise ProtocolError(f"duplicate proposition id {prop.prop_id}")
        ids.add(prop.prop_id)
        for token in banned_tokens:
            if token and token in prop.text:
                raise ProtocolError(
                    f"proposition {prop.prop_id} contains incident-specific token {token!r}"
                )


def audit_playbook(playbook: Playbook, reference: DomainReference) -> None:
    known = reference.by_id()
    for index, step in enumerate(playbook.steps):
        if step.kind == "fact" and not step.proposition_ids:
            raise ProtocolError(f"playbook fact step {index} has no proposition id")
        for prop_id in step.proposition_ids:
            if prop_id not in known:
                raise ProtocolError(f"playbook cites unknown proposition {prop_id}")


def freeze_lessons(lessons: list[Lesson], records: list[CaseRecord]) -> FrozenLessons:
    by_id = {record.case_id: record for record in records}
    for lesson in lessons:
        if not lesson.provenance_case_ids:
            raise LessonLeak(f"lesson {lesson.lesson_id} has no training provenance")
        for case_id in lesson.provenance_case_ids:
            record = by_id.get(case_id)
            if record is None:
                raise LessonLeak(f"lesson {lesson.lesson_id} cites unknown case {case_id}")
            if record.partition is not Partition.TRAIN:
                raise LessonLeak(
                    f"lesson {lesson.lesson_id} cites {case_id} in partition {record.partition.value}"
                )
    frozen = tuple(lessons)
    return FrozenLessons(
        lessons=frozen,
        content_hash=content_hash([item.model_dump(mode="json") for item in frozen]),
    )
