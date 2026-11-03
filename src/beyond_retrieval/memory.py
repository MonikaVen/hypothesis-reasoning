"""Optional representation experiment.

Section 8.7 compares renderings of the same facts. A rendering may change
order and labels. It may not add a fact id. Edge corruption is a seeded
swap or deletion that keeps every derived edge marked unverified.
"""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class RelationFact:
    fact_id: str
    text: str
    source_id: str
    target_id: str
    relation: str
    derived: bool = True


def render_representation(facts: list[RelationFact], representation: str) -> str:
    known = {"ledger", "relation_list", "typed_graph"}
    if representation not in known:
        raise ValueError(f"unknown representation {representation}")
    ordered = sorted(facts, key=lambda item: item.fact_id)
    lines = [f"fact_id:{item.fact_id}" for item in ordered]
    if representation == "ledger":
        lines.extend(item.text for item in ordered)
    elif representation == "relation_list":
        lines.extend(
            f"{item.source_id} {item.relation} {item.target_id} | {item.text}" for item in ordered
        )
    else:
        lines.extend(
            f"edge\t{item.source_id}\t{item.relation}\t{item.target_id}\t{item.text}"
            for item in ordered
        )
    return "\n".join(lines)


def fact_ids(rendered: str) -> set[str]:
    return {line.removeprefix("fact_id:") for line in rendered.splitlines() if line.startswith("fact_id:")}


def corrupt_edges(
    facts: list[RelationFact],
    rate: float,
    seed: int,
    mode: str = "swap",
) -> list[RelationFact]:
    if rate not in {0.0, 0.1, 0.2}:
        raise ValueError("corruption rate must be 0, 0.1, or 0.2")
    if mode not in {"swap", "remove"}:
        raise ValueError("corruption mode must be swap or remove")
    eligible = [item for item in facts if item.derived]
    count = int(round(len(eligible) * rate))
    rng = random.Random(seed)
    chosen = {item.fact_id for item in rng.sample(eligible, count)} if count else set()
    output: list[RelationFact] = []
    for item in facts:
        if item.fact_id not in chosen:
            output.append(item)
            continue
        if mode == "remove":
            continue
        output.append(
            RelationFact(
                fact_id=item.fact_id,
                text=item.text,
                source_id=item.target_id,
                target_id=item.source_id,
                relation=item.relation,
                derived=True,
            )
        )
    return output
