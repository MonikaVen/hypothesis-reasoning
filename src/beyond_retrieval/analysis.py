"""Effect estimates for the confirmatory contrasts.

Repeats are averaged inside a case. Cases are weighted equally. Environments
are also averaged equally, which is a different number when clusters are
uneven. Intervals resample whole campaigns. With four test environments the
randomization distribution has 16 assignments, so this module does not treat
a conventional significance threshold as attainable, and it never promotes a
finite-corpus contrast into a population claim.
"""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class Trial:
    case_id: str
    environment: str
    campaign_id: str
    arm: str
    repeat_index: int
    value: float


@dataclass(frozen=True)
class EffectEstimate:
    treatment: str
    control: str
    n_cases: int
    n_environments: int
    n_campaigns: int
    case_weighted_difference: float
    environment_macro_difference: float
    bootstrap_low: float
    bootstrap_high: float
    population_inference: bool
    conventional_significance_attainable: bool
    randomization_assignments: int


def holm(p_values: list[float]) -> list[float]:
    """Holm step-down adjusted p-values, returned in the input order."""
    count = len(p_values)
    order = sorted(range(count), key=lambda index: (p_values[index], index))
    adjusted = [0.0] * count
    running = 0.0
    for rank, index in enumerate(order):
        value = min(1.0, (count - rank) * p_values[index])
        running = max(running, value)
        adjusted[index] = running
    return adjusted


def effect_estimate(
    trials: list[Trial],
    treatment: str,
    control: str,
    *,
    seed: int = 0,
    draws: int = 200,
) -> EffectEstimate:
    paired, environments, campaigns = _paired(trials, treatment, control)
    if not paired:
        raise ValueError("no case has both arms")
    point = _mean([item[0] for item in paired])
    macro = _environment_macro(paired)
    low, high = _bootstrap(paired, seed, draws)
    n_env = len(environments)
    # Binary label swaps at the environment level: 2^k assignments.
    assignments = 2 ** n_env
    return EffectEstimate(
        treatment=treatment,
        control=control,
        n_cases=len(paired),
        n_environments=n_env,
        n_campaigns=len(campaigns),
        case_weighted_difference=point,
        environment_macro_difference=macro,
        bootstrap_low=low,
        bootstrap_high=high,
        population_inference=False,
        conventional_significance_attainable=n_env > 4,
        randomization_assignments=assignments,
    )


def unmeasured_main_table() -> list[dict[str, str]]:
    """Reporting template. Cells stay unmeasured until a real run fills them."""
    rows = []
    for label in (
        "R Fixed retrieval",
        "A Standard agent",
        "P Fixed playbook",
        "H Hypothesis agent",
        "AS Agent + scripts",
        "HS Full method",
    ):
        rows.append(
            {
                "condition": label,
                "esr": "—",
                "cps": "—",
                "unsupported_claims": "—",
                "calls": "—",
                "cost_per_run": "—",
            }
        )
    return rows


def _paired(
    trials: list[Trial], treatment: str, control: str
) -> tuple[list[tuple[float, str, str]], set[str], set[str]]:
    grouped: dict[tuple[str, str], list[float]] = {}
    env_of: dict[str, str] = {}
    camp_of: dict[str, str] = {}
    for trial in trials:
        if trial.arm not in {treatment, control}:
            continue
        grouped.setdefault((trial.case_id, trial.arm), []).append(trial.value)
        env_of[trial.case_id] = trial.environment
        camp_of[trial.case_id] = trial.campaign_id
    cases = {case_id for case_id, _arm in grouped}
    paired: list[tuple[float, str, str]] = []
    environments: set[str] = set()
    campaigns: set[str] = set()
    for case_id in sorted(cases):
        left = grouped.get((case_id, treatment))
        right = grouped.get((case_id, control))
        if not left or not right:
            continue
        paired.append((_mean(left) - _mean(right), env_of[case_id], camp_of[case_id]))
        environments.add(env_of[case_id])
        campaigns.add(camp_of[case_id])
    return paired, environments, campaigns


def _environment_macro(paired: list[tuple[float, str, str]]) -> float:
    by_env: dict[str, list[float]] = {}
    for difference, environment, _campaign in paired:
        by_env.setdefault(environment, []).append(difference)
    return _mean([_mean(values) for values in by_env.values()])


def _bootstrap(
    paired: list[tuple[float, str, str]], seed: int, draws: int
) -> tuple[float, float]:
    by_campaign: dict[str, list[float]] = {}
    for difference, _environment, campaign in paired:
        by_campaign.setdefault(campaign, []).append(difference)
    keys = sorted(by_campaign)
    if draws < 1:
        point = _mean([item[0] for item in paired])
        return point, point
    rng = random.Random(seed)
    samples: list[float] = []
    for _ in range(draws):
        chosen = [rng.choice(keys) for _key in keys]
        values = [value for key in chosen for value in by_campaign[key]]
        samples.append(_mean(values))
    samples.sort()
    low_index = int(0.025 * (len(samples) - 1))
    high_index = int(0.975 * (len(samples) - 1))
    return samples[low_index], samples[high_index]


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)
