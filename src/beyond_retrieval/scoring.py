"""Mechanical checks and blinded packets.

Identifier, timestamp, and citation checks are automatic. Entailment and
material invention stay with blinded reviewers. An ESR pass requires both.
A missing adjudication is pending, not a pass and not a silent exclusion.
"""

from __future__ import annotations

from beyond_retrieval.models import (
    FORBIDDEN_AGENT_KEYS,
    Adjudication,
    AgentRecord,
    ESRResult,
    FinalAnswer,
    MechanicalScore,
    Rubric,
    RunRecord,
    RunStatus,
    Verdict,
)

def mechanical_score(
    answer: FinalAnswer | None,
    rubric: Rubric,
    visible_ids: set[str],
    retrieved_ids: set[str],
) -> MechanicalScore:
    if answer is None or _empty(answer):
        return MechanicalScore(
            passed=False,
            admissible_verdict=False,
            missing_findings=[item.finding_id for item in rubric.mandatory],
            unknown_citations=[],
            unretrieved_citations=[],
            empty_answer=True,
            probability_sum_ok=False,
        )
    admissible = answer.verdict in rubric.admissible
    asserted = {item.finding_id for item in answer.findings}
    missing = [item.finding_id for item in rubric.mandatory if item.finding_id not in asserted]
    cited: list[str] = []
    for finding in answer.findings:
        cited.extend(finding.record_ids)
    unknown = sorted({record_id for record_id in cited if record_id not in visible_ids})
    unretrieved = sorted(
        {
            record_id
            for record_id in cited
            if record_id in visible_ids and record_id not in retrieved_ids
        }
    )
    total = sum(answer.probabilities.values())
    categories = {item.value for item in Verdict}
    probability_ok = (
        abs(total - 1.0) <= 1e-3
        and set(answer.probabilities) == categories
        and all(value >= 0.0 for value in answer.probabilities.values())
    )
    passed = admissible and not missing and not unknown and not answer_is_blank(answer) and probability_ok
    return MechanicalScore(
        passed=passed,
        admissible_verdict=admissible,
        missing_findings=missing,
        unknown_citations=unknown,
        unretrieved_citations=unretrieved,
        empty_answer=answer_is_blank(answer),
        probability_sum_ok=probability_ok,
        needs_human=True,
    )


def answer_is_blank(answer: FinalAnswer) -> bool:
    return not answer.findings and not answer.limitations.strip()


def esr_result(
    status: RunStatus,
    mechanical: MechanicalScore | None,
    adjudication: Adjudication | None,
    *,
    scored: bool = True,
) -> ESRResult:
    if status is RunStatus.INFRA_OUTAGE:
        return ESRResult(status="missing", passed=False)
    if status is RunStatus.AGENT_ERROR:
        return ESRResult(status="fail", passed=False)
    if not scored:
        return ESRResult(status="not_scored", passed=False)
    if mechanical is None or not mechanical.passed:
        return ESRResult(status="fail", passed=False)
    if adjudication is None:
        return ESRResult(status="pending_adjudication", passed=False)
    if adjudication.support_holds and not adjudication.material_unsupported:
        return ESRResult(status="pass", passed=True)
    return ESRResult(status="fail", passed=False)


def apply_adjudication(record: RunRecord, adjudication: Adjudication) -> RunRecord:
    """Attach a review. Fixture doubles are allowed only so the harness can be tested."""
    updated = record.model_copy(update={"adjudication": adjudication})
    updated.esr = esr_result(record.status, record.mechanical, adjudication)
    return updated


def blind_packet(record: RunRecord, records_by_id: dict[str, AgentRecord]) -> dict[str, object]:
    """Reviewer view. Condition labels and action contracts are removed."""
    answer = record.final_answer
    cited: list[str] = []
    findings: list[dict[str, object]] = []
    if answer is not None:
        for finding in answer.findings:
            cited.extend(finding.record_ids)
            findings.append(
                {
                    "finding_id": finding.finding_id,
                    "text": finding.text,
                    "record_ids": list(finding.record_ids),
                }
            )
    evidence = []
    for record_id in sorted(set(cited)):
        item = records_by_id.get(record_id)
        if item is None:
            evidence.append({"record_id": record_id, "available": False})
        else:
            evidence.append({"record_id": record_id, "available": True, "raw": item.raw, "timestamp": item.timestamp})
    packet: dict[str, object] = {
        "packet_id": record.snapshot_hash[:12] + ":" + record.started_at,
        "question_case": record.spec.case_id,
        "verdict": None if answer is None else answer.verdict.value,
        "probabilities": None if answer is None else dict(answer.probabilities),
        "findings": findings,
        "limitations": "" if answer is None else answer.limitations,
        "cited_records": evidence,
    }
    for key in FORBIDDEN_AGENT_KEYS:
        if key in packet:
            raise RuntimeError(f"blind packet contains {key}")
    if {"condition", "ablation", "contract"} & set(packet):
        raise RuntimeError("blind packet retained a condition field")
    return packet


def revision_label(
    provisional: MechanicalScore | None,
    final: MechanicalScore | None,
    provisional_verdict: str | None,
    final_verdict: str | None,
) -> str:
    """Mechanical revision tag. Confirmatory revision still needs adjudication."""
    if provisional is None or final is None:
        return "unavailable"
    if provisional.passed and final.passed and provisional_verdict == final_verdict:
        return "stable_pass"
    if (not provisional.passed) and final.passed:
        return "mechanical_correction"
    if provisional.passed and not final.passed:
        return "mechanical_harmful_reversal"
    if provisional_verdict != final_verdict and not final.passed:
        return "changed_without_pass"
    if provisional.passed and final.passed and provisional_verdict != final_verdict:
        return "mechanical_resolution"
    return "unresolved"


def _empty(answer: FinalAnswer) -> bool:
    return answer_is_blank(answer) and answer.verdict is Verdict.INSUFFICIENT and not answer.unresolved
