"""Command line for allocation planning and the synthetic harness check."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from beyond_retrieval.fixtures import domain_reference, investigation_policy, outbound_case
from beyond_retrieval.grid import (
    e1_ceiling,
    e2_ceiling,
    e3_new_runs,
    e4_ceiling,
    no_evidence_diagnostic,
    optional_regeneration_runs,
    pilot_script_runs,
    pilot_structure_runs,
    planned_ceiling,
    secondary_rp_runs,
)
from beyond_retrieval.hashing import content_hash
from beyond_retrieval.models import ConditionId, Variant
from beyond_retrieval.protocol import make_protocol
from beyond_retrieval.runner import ManualClock, execute_run, protocol_hash


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="beyond-retrieval")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("plan", help="Print the protocol's run ceilings")
    dry = sub.add_parser("dry-run", help="Run the synthetic harness check")
    dry.add_argument("--out", default="runs/dry-run", help="Directory for run records")
    args = parser.parse_args(argv)
    if args.command == "plan":
        _print_plan()
        return 0
    _dry_run(Path(args.out))
    return 0


def _print_plan() -> None:
    print("Beyond Retrieval run ceilings. These are allocation limits, not results.")
    print(f"E1 core grid ceiling: {e1_ceiling()}")
    print(f"E2 common-evidence synthesis: {e2_ceiling()}")
    print(f"E3 validation ablation, new runs: {e3_new_runs()}")
    print(f"E4 external transfer: {e4_ceiling()}")
    print(f"Planned ceiling E1–E4: {planned_ceiling()}")
    print(f"Secondary R/P subset: {secondary_rp_runs()}")
    print(f"No-evidence diagnostic: {no_evidence_diagnostic()}")
    print(f"Pilot 1 (A/H, V0+V4): {pilot_structure_runs()}")
    print(f"Pilot 2 (AS/HS, V0): {pilot_script_runs()}")
    print(f"Optional regenerated pairs, outside the ceiling: {optional_regeneration_runs()}")
    print("Actual E1 size is 4 conditions × 2 backbones × 3 repeats × eligible case-versions.")
    print("Ineligible variants are omitted. They are not scored as failures.")


def _dry_run(out: Path) -> None:
    case = outbound_case()
    reference = domain_reference()
    protocol = make_protocol(
        sealed=False,
        guidance_hash=content_hash(reference.model_dump(mode="json")),
        rubric_hashes={
            variant.value: content_hash(rubric.model_dump(mode="json"))
            for variant, rubric in case.rubrics.items()
        },
    )
    out.mkdir(parents=True, exist_ok=True)
    digest = protocol_hash(protocol)
    print("Synthetic harness check. Not an experiment result. ESR stays pending adjudication.")
    print(f"Protocol hash: {digest}")
    for variant in (Variant.V0, Variant.V4):
        for condition in (ConditionId.A, ConditionId.H):
            record = execute_run(
                _spec(case.record.case_id, variant, condition),
                case,
                investigation_policy(condition, variant),
                protocol,
                reference,
                clock=ManualClock(),
                workspace=out / "mounts",
            )
            path = out / f"{variant.value}-{condition.value}.json"
            path.write_text(record.model_dump_json(indent=2), encoding="utf-8")
            mechanical = "pending" if record.mechanical is None else str(record.mechanical.passed).lower()
            esr = "none" if record.esr is None else record.esr.status
            print(
                f"{variant.value} {condition.value}: status={record.status.value} "
                f"mechanical={mechanical} esr={esr} calls={record.budget.tool_calls} "
                f"snapshot={record.snapshot_hash[:12]}"
            )
    summary = {
        "kind": "synthetic_harness_check",
        "protocol_hash": digest,
        "note": "Mechanical checks only. No model was called. Do not cite these rows as study outcomes.",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote records under {out}")


def _spec(case_id: str, variant: Variant, condition: ConditionId):
    from beyond_retrieval.models import RunSpec

    return RunSpec(
        experiment="PILOT1",
        case_id=case_id,
        variant=variant,
        condition=condition,
        backbone_id="scripted-harness",
        repeat_index=0,
    )


if __name__ == "__main__":
    raise SystemExit(main())
