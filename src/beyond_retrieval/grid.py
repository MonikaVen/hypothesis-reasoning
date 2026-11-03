"""Run allocations.

The grid enumerates work. It does not execute it and it does not score it.
Ceilings match the protocol. Ineligible variants are omitted, not scored
as failures. Exploratory techniques stay outside the primary ceiling.
"""

from __future__ import annotations

from beyond_retrieval.errors import ProtocolError
from beyond_retrieval.hashing import content_hash
from beyond_retrieval.models import (
    CORE_CONDITIONS,
    RECORDED_VARIANTS,
    Ablation,
    CaseEligibility,
    ConditionId,
    Contrast,
    RunMode,
    RunSpec,
    Variant,
)

CONTRASTS: tuple[Contrast, ...] = (
    Contrast(
        hypothesis="H1",
        treatment="H",
        control="A",
        variant="V0",
        endpoint="esr",
        experiment="E1",
    ),
    Contrast(
        hypothesis="H2",
        treatment="structured_synthesis",
        control="standard_synthesis",
        variant="V0",
        endpoint="esr",
        experiment="E2",
    ),
    Contrast(
        hypothesis="H3",
        treatment="HS",
        control="AS",
        variant="V0",
        endpoint="esr",
        experiment="E1",
    ),
    Contrast(
        hypothesis="H4",
        treatment="HS",
        control="HS_no_validation",
        variant="V0",
        endpoint="esr",
        experiment="E3",
    ),
)

EXPLORATORY_TECHNIQUES: tuple[str, ...] = (
    "hypothesis_tree",
    "independent_critique",
    "retrieval_guided_planning",
    "adaptive_replanning",
    "independent_candidates",
    "matched_self_review",
)

PRIMARY_EXPERIMENTS = frozenset({"E1", "E2", "E3", "E4"})


def analysis_spec_hash() -> str:
    return content_hash([item.model_dump(mode="json") for item in CONTRASTS])


def e1_ceiling(n_cases: int = 80, n_versions: int = 5, n_backbones: int = 2, n_repeats: int = 3) -> int:
    """80 × 5 × 4 × 2 × 3 = 9,600 when every case supports every recorded version."""
    return n_cases * n_versions * len(CORE_CONDITIONS) * n_backbones * n_repeats


def e1_actual(eligible_case_versions: int, n_backbones: int = 2, n_repeats: int = 3) -> int:
    """The protocol's factor of 24 is 4 conditions × 2 backbones × 3 repeats."""
    return eligible_case_versions * len(CORE_CONDITIONS) * n_backbones * n_repeats


def e2_ceiling(n_cases: int = 80, n_backbones: int = 2, n_repeats: int = 3) -> int:
    """Two synthesis styles on independently collected V0 bundles."""
    return n_cases * 2 * n_backbones * n_repeats


def e3_new_runs(n_cases: int = 40, n_backbones: int = 2, n_repeats: int = 3) -> int:
    """Unvalidated HS only. Matched full HS runs are reused from E1."""
    return n_cases * n_backbones * n_repeats


def e4_ceiling(n_cases: int = 48, n_backbones: int = 2, n_repeats: int = 3) -> int:
    """24 OTRF + 24 OpTC, four conditions, V0 only."""
    return n_cases * len(CORE_CONDITIONS) * n_backbones * n_repeats


def planned_ceiling() -> int:
    return e1_ceiling() + e2_ceiling() + e3_new_runs() + e4_ceiling()


def secondary_rp_runs(n_cases: int = 20, n_backbones: int = 2, n_repeats: int = 3) -> int:
    return 2 * n_cases * n_backbones * n_repeats


def no_evidence_diagnostic(n_cases: int = 80, n_backbones: int = 2, n_repeats: int = 3) -> int:
    return n_cases * n_backbones * n_repeats


def pilot_structure_runs(
    n_cases: int = 10,
    n_versions: int = 2,
    n_backbones: int = 1,
    n_repeats: int = 2,
) -> int:
    """Development cases, V0 and eligible V4, conditions A and H."""
    return n_cases * n_versions * 2 * n_backbones * n_repeats


def pilot_script_runs(n_cases: int = 10, n_backbones: int = 1, n_repeats: int = 2) -> int:
    """AS and HS on V0."""
    return n_cases * 2 * n_backbones * n_repeats


def optional_regeneration_runs(
    n_pairs: int = 24,
    n_backbones: int = 2,
    n_repeats: int = 3,
) -> int:
    """A and H on both members of each regenerated pair. Not part of the 11,952."""
    return n_pairs * 2 * 2 * n_backbones * n_repeats


def count_eligible(cases: list[CaseEligibility]) -> int:
    total = 0
    for case in cases:
        for variant in case.variants:
            if variant is Variant.V3:
                raise ProtocolError("V3 is not a recorded-data eligibility entry")
            if variant not in RECORDED_VARIANTS:
                raise ProtocolError(f"{variant.value} is outside the recorded-data grid")
        total += len(case.variants)
    return total


def iter_e1(
    cases: list[CaseEligibility],
    backbone_ids: list[str],
    repeats: int = 3,
) -> list[RunSpec]:
    specs: list[RunSpec] = []
    for case in cases:
        for variant in case.variants:
            if variant is Variant.V3 or variant not in RECORDED_VARIANTS:
                raise ProtocolError(f"E1 cannot include {variant.value}")
            for condition in CORE_CONDITIONS:
                for backbone_id in backbone_ids:
                    for repeat_index in range(repeats):
                        specs.append(
                            RunSpec(
                                experiment="E1",
                                case_id=case.case_id,
                                variant=variant,
                                condition=condition,
                                backbone_id=backbone_id,
                                repeat_index=repeat_index,
                                seed=repeat_index,
                            )
                        )
    return specs


def iter_e2(case_ids: list[str], backbone_ids: list[str], repeats: int = 3) -> list[RunSpec]:
    specs: list[RunSpec] = []
    styles = (
        (ConditionId.A, "standard"),
        (ConditionId.H, "structured"),
    )
    for case_id in case_ids:
        for condition, _style in styles:
            for backbone_id in backbone_ids:
                for repeat_index in range(repeats):
                    specs.append(
                        RunSpec(
                            experiment="E2",
                            case_id=case_id,
                            variant=Variant.V0,
                            condition=condition,
                            backbone_id=backbone_id,
                            repeat_index=repeat_index,
                            mode=RunMode.SYNTHESIS,
                            seed=repeat_index,
                        )
                    )
    return specs


def iter_e3(case_ids: list[str], backbone_ids: list[str], repeats: int = 3) -> list[RunSpec]:
    specs: list[RunSpec] = []
    for case_id in case_ids:
        for backbone_id in backbone_ids:
            for repeat_index in range(repeats):
                specs.append(
                    RunSpec(
                        experiment="E3",
                        case_id=case_id,
                        variant=Variant.V0,
                        condition=ConditionId.HS,
                        backbone_id=backbone_id,
                        repeat_index=repeat_index,
                        ablation=Ablation.NO_SCRIPT_VALIDATION,
                        seed=repeat_index,
                    )
                )
    return specs


def iter_pilot_structure(case_ids: list[str], backbone_id: str, repeats: int = 2) -> list[RunSpec]:
    specs: list[RunSpec] = []
    for case_id in case_ids:
        for variant in (Variant.V0, Variant.V4):
            for condition in (ConditionId.A, ConditionId.H):
                for repeat_index in range(repeats):
                    specs.append(
                        RunSpec(
                            experiment="PILOT1",
                            case_id=case_id,
                            variant=variant,
                            condition=condition,
                            backbone_id=backbone_id,
                            repeat_index=repeat_index,
                            seed=repeat_index,
                        )
                    )
    return specs


def assert_primary_scope(experiment: str, technique: str | None = None) -> None:
    if technique in EXPLORATORY_TECHNIQUES and experiment in PRIMARY_EXPERIMENTS:
        raise ProtocolError(
            f"{technique} is exploratory and is outside the primary run ceiling"
        )
