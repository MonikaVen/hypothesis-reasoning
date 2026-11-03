"""Execute a synthetic campaign and write run records.

The default backend is the local investigator. A hosted model is used only
when the caller passes one. Records keep ESR pending until adjudication.
"""

from __future__ import annotations

import json
from pathlib import Path

from beyond_retrieval.backends import OpenAICompatibleBackend
from beyond_retrieval.corpus import freeze_lessons
from beyond_retrieval.evidence import EvidenceMount
from beyond_retrieval.fixtures import domain_reference, manifest_records, outbound_case
from beyond_retrieval.hashing import content_hash
from beyond_retrieval.investigator import COUNT_SCRIPT, LocalInvestigator, ModelInvestigator
from beyond_retrieval.models import Capabilities, ConditionId, Lesson, RunRecord, RunSpec, Variant
from beyond_retrieval.protocol import make_protocol
from beyond_retrieval.runner import Clock, ManualClock, execute_run, protocol_hash
from beyond_retrieval.tools import SplitValidator

EXPERIMENTS: dict[str, dict[str, tuple]] = {
    "pilot": {
        "variants": (Variant.V0, Variant.V4),
        "conditions": (ConditionId.A, ConditionId.H),
    },
    "scripts": {
        "variants": (Variant.V0,),
        "conditions": (ConditionId.AS, ConditionId.HS),
    },
    "synthetic": {
        "variants": (Variant.V0, Variant.V1, Variant.V2, Variant.V4, Variant.V5),
        "conditions": (ConditionId.A, ConditionId.H, ConditionId.AS, ConditionId.HS),
    },
}


def run_campaign(
    *,
    experiment: str = "pilot",
    out: Path,
    backend: OpenAICompatibleBackend | None = None,
    repeats: int = 1,
    clock: Clock | None = None,
) -> list[RunRecord]:
    if experiment not in EXPERIMENTS:
        raise ValueError(f"unknown experiment {experiment}")
    if repeats < 1:
        raise ValueError("repeats must be at least 1")
    case = outbound_case()
    reference = domain_reference()
    selected = EXPERIMENTS[experiment]
    needs_scripts = any(
        Capabilities.for_condition(condition).custom_scripts for condition in selected["conditions"]
    )
    out.mkdir(parents=True, exist_ok=True)
    validator = _script_validator(out / "validator") if needs_scripts else None
    backbone = "local-baseline" if backend is None else backend.name
    lessons = _training_lessons()
    protocol = make_protocol(
        sealed=False,
        guidance_hash=content_hash(reference.model_dump(mode="json")),
        lesson_set_hash=lessons.content_hash,
        rubric_hashes={
            variant.value: content_hash(rubric.model_dump(mode="json"))
            for variant, rubric in case.rubrics.items()
        },
        backbone_ids=(backbone,),
    )
    clock = clock or ManualClock()
    records: list[RunRecord] = []
    for variant in selected["variants"]:
        for condition in selected["conditions"]:
            for repeat_index in range(repeats):
                policy = _policy(condition, backend)
                bound_lessons = lessons if Capabilities.for_condition(condition).experience_lessons else None
                record = execute_run(
                    RunSpec(
                        experiment=experiment,
                        case_id=case.record.case_id,
                        variant=variant,
                        condition=condition,
                        backbone_id=backbone,
                        repeat_index=repeat_index,
                    ),
                    case,
                    policy,
                    protocol,
                    reference,
                    lessons=bound_lessons,
                    clock=clock,
                    script_validator=validator,
                    workspace=out / "mounts",
                )
                records.append(record)
                path = out / f"{variant.value}-{condition.value}-r{repeat_index}.json"
                path.write_text(record.model_dump_json(indent=2), encoding="utf-8")
    manifest = {
        "experiment": experiment,
        "backend": backbone,
        "case_id": case.record.case_id,
        "protocol_hash": protocol_hash(protocol),
        "repeats": repeats,
        "records": [
            {
                "file": f"{item.spec.variant.value}-{item.spec.condition.value}-r{item.spec.repeat_index}.json",
                "status": item.status.value,
                "mechanical_passed": None if item.mechanical is None else item.mechanical.passed,
                "esr": None if item.esr is None else item.esr.status,
                "tool_calls": item.budget.tool_calls,
            }
            for item in records
        ],
        "note": (
            "Mechanical checks are not evidence-supported resolution. "
            "ESR stays pending until blinded adjudication. "
            "local-baseline is a deterministic tool user, not a confirmatory model result."
        ),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return records


def _training_lessons():
    return freeze_lessons(
        [
            Lesson(
                lesson_id="L-outbound",
                applicability="an outbound volume alert behind a translated address",
                recommended_test="compare translation rows with the scheduled job window and the initiating process",
                failure_conditions="treating destination reputation as authorization",
                provenance_case_ids=("fox-001",),
            )
        ],
        manifest_records(),
    )


def _policy(condition: ConditionId, backend: OpenAICompatibleBackend | None):
    capabilities = Capabilities.for_condition(condition)
    if backend is None:
        return LocalInvestigator(capabilities)
    return ModelInvestigator(capabilities, backend)


def _script_validator(directory: Path) -> SplitValidator:
    directory.mkdir(parents=True, exist_ok=True)
    database = directory / "fixture.sqlite"
    event = outbound_case().events[0].model_copy(update={"record_id": "fix-1"})
    EvidenceMount([event], Variant.V0, "fixture").materialize(database)
    # The known-answer fixture has one row. The case mount is checked separately.
    _ = COUNT_SCRIPT
    return SplitValidator(
        database,
        [{"finding_id": "sources", "n": 1, "record_id": "fix-1"}],
    )
