"""Invariants from the protocol, checked against the synthetic harness."""

from __future__ import annotations

import ast
import json
import sqlite3
from pathlib import Path

import pytest
from pydantic import ValidationError

from beyond_retrieval.agent import EvidenceGraph, validate_contract
from beyond_retrieval.analysis import Trial, effect_estimate, holm, unmeasured_main_table
from beyond_retrieval.corpus import (
    audit_environment_splits,
    audit_playbook,
    audit_reference,
    freeze_lessons,
)
from beyond_retrieval.errors import (
    CapabilityError,
    CounterfactualRequiresRegeneration,
    InvalidVariant,
    LessonLeak,
    ProtocolError,
    UnsealedProtocol,
)
from beyond_retrieval.evidence import prepare_mount, rubric_for
from beyond_retrieval.fixtures import (
    domain_reference,
    entity_mapping,
    investigation_policy,
    manifest_records,
    outbound_case,
    playbook,
    train_case,
)
from beyond_retrieval.grid import (
    CONTRASTS,
    assert_primary_scope,
    count_eligible,
    e1_actual,
    e1_ceiling,
    e2_ceiling,
    e3_new_runs,
    e4_ceiling,
    iter_e1,
    iter_e2,
    iter_e3,
    no_evidence_diagnostic,
    optional_regeneration_runs,
    pilot_script_runs,
    pilot_structure_runs,
    planned_ceiling,
    secondary_rp_runs,
)
from beyond_retrieval.hashing import content_hash
from beyond_retrieval.memory import RelationFact, corrupt_edges, fact_ids, render_representation
from beyond_retrieval.models import (
    ActionContract,
    ActionKind,
    Ablation,
    BudgetCaps,
    Capabilities,
    CaseEligibility,
    ConditionId,
    Event,
    HypothesisPrediction,
    Lesson,
    Partition,
    RunMode,
    RunSpec,
    RunStatus,
    ScriptContract,
    Variant,
    Verdict,
)
from beyond_retrieval.protocol import make_protocol
from beyond_retrieval.runner import ManualClock, execute_run
from beyond_retrieval.scoring import apply_adjudication, blind_packet, esr_result, mechanical_score, revision_label
from beyond_retrieval.scripted import ScriptedPolicy
from beyond_retrieval.tools import NOT_FOUND, TOOL_SCHEMA, QueryService, SplitValidator
from beyond_retrieval.models import Adjudication, FinalAnswer, FindingAssertion, Proposal

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "beyond_retrieval"

COUNT_SCRIPT = """
import json, sqlite3, sys
con = sqlite3.connect(sys.argv[1])
rows = list(con.execute("select record_id from events order by record_id"))
json.dump([{"finding_id": "rowcount", "n": len(rows), "record_id": rows[0][0]}], sys.stdout)
"""


def test_allocation_ceilings_match_the_protocol():
    assert e1_ceiling() == 9600
    assert e1_actual(80 * 5) == 9600
    assert e2_ceiling() == 960
    assert e3_new_runs() == 240
    assert e4_ceiling() == 1152
    assert planned_ceiling() == 11952
    assert secondary_rp_runs() == 240
    assert no_evidence_diagnostic() == 480
    assert pilot_structure_runs() == 80
    assert pilot_script_runs() == 40
    assert optional_regeneration_runs() == 576
    assert [item.hypothesis for item in CONTRASTS] == ["H1", "H2", "H3", "H4"]


def test_e1_omits_ineligible_versions_and_rejects_v3():
    cases = [
        CaseEligibility(
            case_id="c1",
            environment="shaw",
            campaign_id="camp",
            variants=(Variant.V0, Variant.V4),
        )
    ]
    specs = iter_e1(cases, ["b1", "b2"], repeats=3)
    assert len(specs) == e1_actual(2, n_backbones=2, n_repeats=3)
    assert {item.variant for item in specs} == {Variant.V0, Variant.V4}
    assert count_eligible(cases) == 2
    with pytest.raises(ProtocolError):
        count_eligible(
            [
                CaseEligibility(
                    case_id="c1",
                    environment="shaw",
                    campaign_id="camp",
                    variants=(Variant.V3,),
                )
            ]
        )
    with pytest.raises(ProtocolError):
        assert_primary_scope("E1", "independent_critique")


def test_e2_and_e3_shapes():
    e2 = iter_e2(["c1"], ["b1"], repeats=1)
    assert len(e2) == 2
    assert all(item.mode is RunMode.SYNTHESIS for item in e2)
    e3 = iter_e3(["c1"], ["b1", "b2"], repeats=3)
    assert len(e3) == 6
    assert all(item.ablation.value == "no_script_validation" for item in e3)
    assert all(item.condition is ConditionId.HS for item in e3)


def test_capability_matrix_does_not_grant_or_leak():
    h = Capabilities.for_condition(ConditionId.H)
    a = Capabilities.for_condition(ConditionId.A)
    hs = Capabilities.for_condition(ConditionId.HS)
    assert h.evidence_graph and h.action_contracts and not h.experience_lessons
    assert not h.custom_scripts and not a.custom_scripts and not a.action_contracts
    assert hs.custom_scripts and hs.script_validation and hs.experience_lessons
    ablated = Capabilities.for_condition(ConditionId.A, Ablation.NO_SCRIPT_VALIDATION)
    assert ablated.custom_scripts is False
    no_val = Capabilities.for_condition(ConditionId.HS, Ablation.NO_SCRIPT_VALIDATION)
    assert no_val.custom_scripts and not no_val.script_validation
    no_lessons = Capabilities.for_condition(ConditionId.HS, Ablation.NO_LESSONS)
    assert not no_lessons.experience_lessons
    flat = Capabilities.for_condition(ConditionId.HS, Ablation.FLAT_MEMORY)
    assert flat.evidence_graph and not flat.graph_organization
    synthesis = h.for_mode(RunMode.SYNTHESIS)
    assert synthesis.adaptive_queries is False and synthesis.custom_scripts is False


def test_contracts_reject_vague_predictions_and_accept_an_open_set():
    vague = _contract(
        [
            HypothesisPrediction(hypothesis_id="h1", explanation="backup", predicted_observation="useful information"),
            HypothesisPrediction(hypothesis_id="h2", explanation="theft", predicted_observation="useful information"),
        ]
    )
    assert validate_contract(vague)
    same = _contract(
        [
            HypothesisPrediction(hypothesis_id="h1", explanation="backup", predicted_observation="A job overlaps 14:22."),
            HypothesisPrediction(hypothesis_id="h2", explanation="theft", predicted_observation="A job overlaps 14:22."),
        ]
    )
    assert any("distinguishable" in item for item in validate_contract(same))
    opened = _contract(
        [HypothesisPrediction(hypothesis_id="h1", explanation="backup", predicted_observation="A job overlaps 14:22.")],
        open_set=True,
    )
    assert validate_contract(opened) == []


def test_partitions_guidance_and_lessons():
    records = manifest_records()
    audit_environment_splits(records)
    audit_reference(domain_reference(), ["srv-app-03", "jdoe", "10.0.0.5"])
    audit_playbook(playbook(), domain_reference())
    leaked = domain_reference().model_copy(
        update={
            "propositions": domain_reference().propositions
            + (
                domain_reference().propositions[0].model_copy(
                    update={"prop_id": "p9", "text": "srv-app-03 is always the victim."}
                ),
            )
        }
    )
    with pytest.raises(ProtocolError):
        audit_reference(leaked, ["srv-app-03"])
    bad_step = playbook().model_copy(
        update={"steps": playbook().steps + (playbook().steps[0].model_copy(update={"proposition_ids": ("missing",)}),)}
    )
    with pytest.raises(ProtocolError):
        audit_playbook(bad_step, domain_reference())
    good = Lesson(
        lesson_id="L1",
        applicability="outbound volume behind a translated address",
        recommended_test="compare translation rows with the job window",
        failure_conditions="destination reputation used as authorization",
        provenance_case_ids=("fox-001",),
    )
    frozen = freeze_lessons([good], records)
    assert frozen.content_hash
    leaked_lesson = good.model_copy(update={"provenance_case_ids": ("synthetic-outbound-01",)})
    with pytest.raises(LessonLeak):
        freeze_lessons([leaked_lesson], records)
    crossed = train_case().model_copy(update={"environment": "synthetic-lab", "partition": Partition.TRAIN})
    with pytest.raises(ProtocolError):
        audit_environment_splits([outbound_case().record, crossed])


def test_v1_is_bijective_and_preserves_commands():
    case = outbound_case()
    mount = prepare_mount(case, Variant.V1, seed=1)
    blob = "\n".join(event.raw for event in mount.events)
    assert "srv-app-03" not in blob
    assert "jdoe" not in blob
    assert "10.0.0.5" not in blob
    assert "host-7f3a" in blob
    assert "curl" in blob and "backup.sh" in blob
    rubric = rubric_for(case, Variant.V1)
    text = " ".join(item.description for item in rubric.mandatory)
    assert "srv-app-03" not in text and "host-7f3a" in text
    nat = next(event for event in mount.events if event.record_id == "nat-1")
    assert nat.entities["host"] == "host-7f3a"
    assert nat.entities["external_ip"] == "10.9.1.5"
    colliding = entity_mapping().model_copy(update={"hosts": {"srv-app-03": "host-7f3a", "srv-backup-01": "host-7f3a"}})
    broken = case.model_copy(update={"v1_mapping": colliding})
    with pytest.raises(InvalidVariant):
        prepare_mount(broken, Variant.V1)


def test_v2_stays_inside_the_environment_and_v4_blocks_every_path():
    case = outbound_case()
    mount = prepare_mount(case, Variant.V2, seed=0)
    ids = {event.record_id for event in mount.events}
    assert len(mount.events) == 8
    assert "foreign-1" not in ids
    assert "related-1" not in ids
    assert ids.issuperset({"flow-1", "nat-1", "proc-1", "job-1"})
    again = prepare_mount(case, Variant.V2, seed=0)
    assert mount.content_hash == again.content_hash
    partial = case.model_copy(
        update={"v4": case.v4.model_copy(update={"cut_record_ids": ("nat-1",)})}  # type: ignore[union-attr]
    )
    with pytest.raises(InvalidVariant):
        prepare_mount(partial, Variant.V4)
    blocked = prepare_mount(case, Variant.V4)
    assert {"nat-1", "proc-1"}.isdisjoint(blocked.ids())
    assert "flow-1" in blocked.ids() and "job-1" in blocked.ids()


def test_v5_stage_file_cannot_see_the_later_release_and_v3_is_refused():
    case = outbound_case()
    stage1 = prepare_mount(case, Variant.V5, stage="stage1")
    stage2 = prepare_mount(case, Variant.V5, stage="stage2")
    assert stage1.ids() == {"flow-1", "job-1"}
    assert stage2.ids() == {"flow-1", "job-1", "nat-1", "proc-1"}
    path = Path("/tmp/br-stage1-test.sqlite")
    stage1.materialize(path)
    present = {
        row[0]
        for row in sqlite3.connect(path).execute("select record_id from events")
    }
    assert present == {"flow-1", "job-1"}
    service = QueryService(stage1.agent_records())
    hidden = service.execute("get_records", {"record_ids": ["nat-1"]})
    unknown = service.execute("get_records", {"record_ids": ["does-not-exist"]})
    assert hidden.error == unknown.error == NOT_FOUND
    with pytest.raises(CounterfactualRequiresRegeneration):
        prepare_mount(case, Variant.V3)


def test_query_order_pagination_truncation_and_parser_failures():
    case = outbound_case()
    records = prepare_mount(case, Variant.V0).agent_records()
    failed = records[0].model_copy(update={"record_id": "fail-1", "parser_status": "failed", "timestamp": "2024-04-11T01:00:00Z"})
    service = QueryService([failed, *records], limits=_limits(page_size=1, max_chars=12_000))
    first = service.execute("search_events", {"limit": 1})
    assert first.records[0].record_id == "fail-1"
    assert first.records[0].parser_status == "failed"
    second = service.execute("search_events", {"limit": 1, "cursor": first.cursor})
    assert second.records[0].record_id != "fail-1"
    narrow = QueryService(records, limits=_limits(page_size=200, max_chars=20))
    truncated = narrow.execute("search_events", {})
    assert truncated.truncated is True
    assert len(truncated.records) == 1
    rejected = service.execute("aggregate_events", {"group_by": "source; drop table events"})
    assert rejected.status.value == "failed_tool"
    invented = service.execute("search_events", {"query": "outbound"})
    assert invented.status.value == "failed_tool"
    assert invented.error is not None and "unknown parameters: query" in invented.error
    counted = QueryService(records).execute("aggregate_events", {"group_by": "source"})
    assert "netflow\t1" in counted.summary


def test_agent_mount_rejects_labels():
    case = outbound_case()
    dirty = case.events[0].model_copy(update={"raw": case.events[0].raw + " ATTACK_LABEL_EXFIL"})
    events = (dirty, *case.events[1:])
    leaked = case.model_copy(update={"events": events})
    with pytest.raises(InvalidVariant):
        prepare_mount(leaked, Variant.V0)
    payload = case.events[0].model_dump()
    payload["attack_label"] = "exfil"
    with pytest.raises(ValidationError):
        Event.model_validate(payload)


def test_graph_edges_are_assertions_and_observations_are_immutable():
    graph = EvidenceGraph()
    graph.add_observation("flow-1", "flow", "hash-a")
    graph.add_observation("flow-1", "flow", "hash-a")
    with pytest.raises(CapabilityError):
        graph.add_observation("flow-1", "changed", "hash-b")
    graph.add_hypothesis("h1", "scheduled job")
    graph.add_hypothesis("h1", "scheduled job")
    graph.add_hypothesis("h1", "unscheduled curl")
    assert "hyp:h1" in graph.nodes
    assert any(node_id.startswith("hyp:h1:") for node_id in graph.nodes)
    graph.add_edge("hyp:h1", "obs:flow-1", "support", "agent")  # type: ignore[arg-type]
    assert graph.edges[0].script_validated is False
    assert not hasattr(graph.edges[0], "relation_true")
    organized = graph.render(organized=True)
    flat = graph.render(organized=False)
    assert "support" in organized
    assert "support" not in flat
    assert "flow" in flat


def test_runner_separates_structure_from_scoring(tmp_path: Path):
    case = outbound_case()
    reference = domain_reference()
    protocol = make_protocol(sealed=False, guidance_hash=content_hash(reference.model_dump(mode="json")))
    clock = ManualClock()
    hidden = execute_run(
        _spec(ConditionId.H, Variant.V0),
        case,
        investigation_policy(ConditionId.H, Variant.V0),
        protocol,
        reference,
        clock=clock,
        workspace=tmp_path,
    )
    standard = execute_run(
        _spec(ConditionId.A, Variant.V0),
        case,
        investigation_policy(ConditionId.A, Variant.V0),
        protocol,
        reference,
        clock=clock,
        workspace=tmp_path,
    )
    assert hidden.mechanical is not None and hidden.mechanical.passed
    assert standard.mechanical is not None and standard.mechanical.passed
    assert hidden.esr is not None and hidden.esr.status == "pending_adjudication"
    assert hidden.falsification_attempt_present is True
    assert standard.falsification_attempt_present is False
    assert any(node.kind.value == "hypothesis" for node in hidden.graph_nodes)
    assert all(node.kind.value != "hypothesis" for node in standard.graph_nodes)
    assert hidden.mechanical.passed == standard.mechanical.passed
    packet = blind_packet(hidden, {item.record_id: item for item in prepare_mount(case, Variant.V0).agent_records()})
    assert set(packet).isdisjoint({"condition", "ablation", "contract", "backbone_id"})
    blinded = json.dumps(packet)
    assert "h_backup" not in blinded
    assert "action contract" not in blinded


def test_v4_insufficient_can_pass_mechanically_and_a_wrong_host_cannot(tmp_path: Path):
    case = outbound_case()
    protocol = make_protocol(sealed=False)
    structured = execute_run(
        _spec(ConditionId.H, Variant.V4),
        case,
        investigation_policy(ConditionId.H, Variant.V4),
        protocol,
        domain_reference(),
        clock=ManualClock(),
        workspace=tmp_path,
    )
    guessed = execute_run(
        _spec(ConditionId.A, Variant.V4),
        case,
        investigation_policy(ConditionId.A, Variant.V4),
        protocol,
        domain_reference(),
        clock=ManualClock(),
        workspace=tmp_path,
    )
    assert structured.mechanical is not None and structured.mechanical.passed
    assert guessed.mechanical is not None and not guessed.mechanical.passed
    assert "proc-1" in guessed.mechanical.unknown_citations
    decided = apply_adjudication(
        structured,
        Adjudication(reviewer_ids=["r1", "r2"], support_holds=True, material_unsupported=False, source="fixture_double"),
    )
    assert decided.esr is not None and decided.esr.passed is True
    rejected = apply_adjudication(
        structured,
        Adjudication(reviewer_ids=["r1"], support_holds=False, material_unsupported=True, source="fixture_double"),
    )
    assert rejected.esr is not None and rejected.esr.passed is False


def test_v5_release_is_scored_as_resolution_not_a_free_pass(tmp_path: Path):
    case = outbound_case()
    record = execute_run(
        _spec(ConditionId.H, Variant.V5),
        case,
        investigation_policy(ConditionId.H, Variant.V5),
        make_protocol(sealed=False),
        domain_reference(),
        clock=ManualClock(),
        workspace=tmp_path,
    )
    assert record.provisional_mechanical is not None and record.provisional_mechanical.passed
    assert record.mechanical is not None and record.mechanical.passed
    assert record.provisional_answer is not None and record.final_answer is not None
    label = revision_label(
        record.provisional_mechanical,
        record.mechanical,
        record.provisional_answer.verdict.value,
        record.final_answer.verdict.value,
    )
    assert label == "mechanical_resolution"
    assert record.stage_snapshot_hashes["stage1"] != record.stage_snapshot_hashes["stage2"]


def test_budget_outage_and_seal_are_not_exclusions(tmp_path: Path):
    case = outbound_case()
    reference = domain_reference()
    stopped = ScriptedPolicy(
        Capabilities.for_condition(ConditionId.A),
        [[]],
        [_gap_answer_unused()],
    )
    exhausted = execute_run(
        _spec(ConditionId.A, Variant.V0),
        case,
        stopped,
        make_protocol(sealed=False, caps=BudgetCaps(max_tool_calls=0)),
        reference,
        clock=ManualClock(),
        workspace=tmp_path,
    )
    assert exhausted.status.value == "budget_exhausted"
    assert exhausted.counts_as_missing is False
    assert exhausted.steps == []

    class Boom:
        def start(self, view):
            return None

        def propose(self, observation):
            raise RuntimeError("agent failed")

        def finalize(self):
            raise AssertionError("finalize should not run after a crashed proposal")

    failed = execute_run(
        _spec(ConditionId.A, Variant.V0),
        case,
        Boom(),
        make_protocol(sealed=False),
        reference,
        clock=ManualClock(),
        workspace=tmp_path,
    )
    assert failed.status.value == "agent_error"
    assert failed.counts_as_missing is False
    assert failed.esr is not None and failed.esr.status == "fail"

    outage = execute_run(
        _spec(ConditionId.A, Variant.V0),
        case,
        investigation_policy(ConditionId.A, Variant.V0),
        make_protocol(sealed=False),
        reference,
        clock=ManualClock(),
        workspace=tmp_path,
        raise_infra=True,
    )
    assert outage.status.value == "infra_outage"
    assert outage.counts_as_missing is True
    assert outage.esr is not None and outage.esr.status == "missing"
    assert outage.final_answer is None

    with pytest.raises(UnsealedProtocol):
        execute_run(
            _spec(ConditionId.A, Variant.V0),
            case,
            stopped,
            make_protocol(sealed=False),
            reference,
            clock=ManualClock(),
            confirmatory=True,
            workspace=tmp_path,
        )
    sealed = execute_run(
        _spec(ConditionId.A, Variant.V0),
        case,
        ScriptedPolicy(Capabilities.for_condition(ConditionId.A), [[]], [_gap_answer_unused()]),
        make_protocol(sealed=True),
        reference,
        clock=ManualClock(),
        confirmatory=True,
        workspace=tmp_path,
    )
    assert sealed.protocol_hash
    with pytest.raises(CounterfactualRequiresRegeneration):
        execute_run(
            _spec(ConditionId.A, Variant.V3),
            case,
            stopped,
            make_protocol(sealed=False),
            reference,
            clock=ManualClock(),
            workspace=tmp_path,
        )


def test_lessons_attach_only_to_hs(tmp_path: Path):
    records = manifest_records()
    lessons = freeze_lessons(
        [
            Lesson(
                lesson_id="L1",
                applicability="translated outbound volume",
                recommended_test="join translation to the job window",
                failure_conditions="reputation used as proof",
                provenance_case_ids=("fox-001",),
            )
        ],
        records,
    )
    case = outbound_case()
    protocol = make_protocol(sealed=False, lesson_set_hash=lessons.content_hash)
    with pytest.raises(CapabilityError):
        execute_run(
            _spec(ConditionId.H, Variant.V0),
            case,
            ScriptedPolicy(Capabilities.for_condition(ConditionId.H), [[]], [_gap_answer_unused()]),
            protocol,
            domain_reference(),
            lessons=lessons,
            clock=ManualClock(),
            workspace=tmp_path,
        )
    with pytest.raises(CapabilityError):
        execute_run(
            _spec(ConditionId.HS, Variant.V0),
            case,
            ScriptedPolicy(Capabilities.for_condition(ConditionId.HS), [[]], [_gap_answer_unused()]),
            protocol,
            domain_reference(),
            clock=ManualClock(),
            workspace=tmp_path,
        )
    allowed = execute_run(
        _spec(ConditionId.HS, Variant.V0),
        case,
        ScriptedPolicy(Capabilities.for_condition(ConditionId.HS), [[]], [_gap_answer_unused()]),
        protocol,
        domain_reference(),
        lessons=lessons,
        clock=ManualClock(),
        workspace=tmp_path,
    )
    assert allowed.status.value == "completed"


def test_script_validation_gates_admission(tmp_path: Path):
    case = outbound_case()
    fixture_db = tmp_path / "fixture.sqlite"
    prepare_one(fixture_db)
    expected = [{"finding_id": "rowcount", "n": 1, "record_id": "fix-1"}]
    validator = SplitValidator(fixture_db, expected)
    protocol = make_protocol(sealed=False)
    hs = execute_run(
        _spec(ConditionId.HS, Variant.V0),
        case,
        _script_policy(ConditionId.HS),
        protocol,
        domain_reference(),
        lessons=_one_lesson(),
        script_validator=validator,
        clock=ManualClock(),
        workspace=tmp_path / "hs",
    )
    assert hs.steps[0].script_validation == "pass"
    assert hs.budget.tool_calls == 2
    assert any(node.kind.value == "artifact" for node in hs.graph_nodes)
    assert hs.graph_edges and hs.graph_edges[0].script_validated is True

    closed = execute_run(
        _spec(ConditionId.HS, Variant.V0),
        case,
        _script_policy(ConditionId.HS),
        protocol,
        domain_reference(),
        lessons=_one_lesson(),
        script_validator=None,
        clock=ManualClock(),
        workspace=tmp_path / "closed",
    )
    assert closed.steps[0].script_validation == "fail"
    assert not any(node.kind.value == "artifact" for node in closed.graph_nodes)

    standard = execute_run(
        _spec(ConditionId.AS, Variant.V0),
        case,
        _script_policy(ConditionId.AS),
        protocol,
        domain_reference(),
        script_validator=validator,
        clock=ManualClock(),
        workspace=tmp_path / "as",
    )
    assert standard.steps[0].script_validation == "not_applicable"
    assert standard.budget.tool_calls == 1
    assert not any(node.kind.value == "artifact" for node in standard.graph_nodes)

    denied = execute_run(
        _spec(ConditionId.A, Variant.V0),
        case,
        _script_policy(ConditionId.A),
        protocol,
        domain_reference(),
        clock=ManualClock(),
        workspace=tmp_path / "a",
    )
    assert denied.steps[0].status == "denied"


def test_contract_repair_consumes_budget_and_synthesis_has_no_tools(tmp_path: Path):
    case = outbound_case()
    bad = Proposal(
        kind=ActionKind.QUERY,
        tool="list_sources",
        contract=_contract(
            [
                HypothesisPrediction(hypothesis_id="h1", explanation="backup", predicted_observation="useful information"),
                HypothesisPrediction(hypothesis_id="h2", explanation="theft", predicted_observation="The tool will provide useful information."),
            ]
        ),
        generated_tokens=3,
    )
    good = investigation_policy(ConditionId.H, Variant.V0).stages[0][0]
    policy = ScriptedPolicy(Capabilities.for_condition(ConditionId.H), [[bad, good]], [_gap_answer_unused()])
    record = execute_run(
        _spec(ConditionId.H, Variant.V0),
        case,
        policy,
        make_protocol(sealed=False),
        domain_reference(),
        clock=ManualClock(),
        workspace=tmp_path,
    )
    assert record.steps[0].status == "contract_rejected"
    assert record.budget.tool_calls >= 2

    seen = ScriptedPolicy(Capabilities.for_condition(ConditionId.A), [[]], [_gap_answer_unused()])
    synthesis = execute_run(
        _spec(ConditionId.A, Variant.V0, mode=RunMode.SYNTHESIS),
        case,
        seen,
        make_protocol(sealed=False),
        domain_reference(),
        clock=ManualClock(),
        workspace=tmp_path,
    )
    assert seen.views[0].tools == ()
    assert synthesis.steps == []
    assert synthesis.esr is not None and synthesis.esr.status == "not_scored"


def test_holm_and_paired_estimates():
    assert holm([0.01, 0.04, 0.03, 0.20]) == [0.04, 0.09, 0.09, 0.20]
    trials = [
        Trial("c1", "e1", "k1", "H", 0, 1),
        Trial("c1", "e1", "k1", "H", 1, 1),
        Trial("c1", "e1", "k1", "A", 0, 0),
        Trial("c1", "e1", "k1", "A", 1, 0),
        Trial("c2", "e1", "k1", "H", 0, 1),
        Trial("c2", "e1", "k1", "H", 1, 0),
        Trial("c2", "e1", "k1", "A", 0, 1),
        Trial("c2", "e1", "k1", "A", 1, 1),
        Trial("c3", "e2", "k2", "H", 0, 0),
        Trial("c3", "e2", "k2", "A", 0, 0),
    ]
    effect = effect_estimate(trials, "H", "A", seed=3, draws=40)
    assert effect.case_weighted_difference == pytest.approx(1 / 6)
    assert effect.environment_macro_difference == pytest.approx(0.125)
    assert effect.population_inference is False
    assert effect.conventional_significance_attainable is False
    assert effect.randomization_assignments == 4
    wider = trials + [
        Trial("c4", "e3", "k3", "H", 0, 0),
        Trial("c4", "e3", "k3", "A", 0, 0),
        Trial("c5", "e4", "k4", "H", 0, 0),
        Trial("c5", "e4", "k4", "A", 0, 0),
        Trial("c6", "e5", "k5", "H", 0, 0),
        Trial("c6", "e5", "k5", "A", 0, 0),
    ]
    broader = effect_estimate(wider, "H", "A", draws=10)
    assert broader.n_environments == 5
    assert broader.conventional_significance_attainable is True
    table = unmeasured_main_table()
    assert len(table) == 6
    assert all(row["esr"] == "—" for row in table)


def test_representations_share_facts_and_corruption_is_seeded():
    facts = [
        RelationFact("f1", "flow supports host", "flow-1", "host", "support"),
        RelationFact("f2", "job contradicts window", "job-1", "flow-1", "contradict"),
        RelationFact("f3", "process derives parent", "proc-1", "flow-1", "derivation"),
        RelationFact("f4", "nat depends on flow", "nat-1", "flow-1", "dependency"),
        RelationFact("f5", "gap remains", "q1", "flow-1", "support"),
    ]
    rendered = {name: render_representation(facts, name) for name in ("ledger", "relation_list", "typed_graph")}
    assert all(fact_ids(text) == {item.fact_id for item in facts} for text in rendered.values())
    swapped = corrupt_edges(facts, 0.2, seed=7, mode="swap")
    assert len(swapped) == len(facts)
    assert [item.relation for item in swapped] == [item.relation for item in facts]
    assert all(item.derived for item in swapped)
    again = corrupt_edges(facts, 0.2, seed=7, mode="swap")
    assert [item.source_id for item in swapped] == [item.source_id for item in again]
    removed = corrupt_edges(facts, 0.2, seed=7, mode="remove")
    assert len(removed) == len(facts) - 1
    assert {item.source_id for item in removed} <= {item.source_id for item in facts} | {
        item.target_id for item in facts
    }


def test_import_layers_keep_policies_away_from_tools():
    allowed = {
        "models": set(),
        "hashing": set(),
        "errors": set(),
        "memory": set(),
        "analysis": set(),
        "corpus": {"errors", "hashing", "models"},
        "evidence": {"errors", "hashing", "models"},
        "tools": {"hashing", "models"},
        "agent": {"errors", "hashing", "models"},
        "grid": {"errors", "hashing", "models"},
        "scoring": {"models"},
    }
    for name, permit in allowed.items():
        imports = _internal_imports(SRC / f"{name}.py")
        assert imports <= permit, f"{name} imports {imports - permit}"
        assert "runner" not in imports
        assert "fixtures" not in imports


def test_tool_schema_is_stable():
    assert content_hash(TOOL_SCHEMA) == content_hash(dict(TOOL_SCHEMA))
    assert set(TOOL_SCHEMA) >= {
        "list_sources",
        "describe_schema",
        "search_events",
        "get_records",
        "aggregate_events",
    }


def test_cli_dry_run(tmp_path: Path):
    from beyond_retrieval.cli import main

    code = main(["dry-run", "--out", str(tmp_path)])
    assert code == 0
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["kind"] == "synthetic_harness_check"
    record = json.loads((tmp_path / "V0-H.json").read_text(encoding="utf-8"))
    assert record["esr"]["status"] == "pending_adjudication"
    assert record["mechanical"]["passed"] is True


def _spec(condition: ConditionId, variant: Variant, mode: RunMode = RunMode.INVESTIGATION) -> RunSpec:
    return RunSpec(
        experiment="PILOT1",
        case_id="synthetic-outbound-01",
        variant=variant,
        condition=condition,
        backbone_id="scripted-harness",
        repeat_index=0,
        mode=mode,
    )


def _contract(hypotheses: list[HypothesisPrediction], open_set: bool = False) -> ActionContract:
    return ActionContract(
        question="Which explanation does this query separate?",
        hypotheses=hypotheses,
        open_set_alternative=open_set,
        tool="search_events",
        parameters={"source": "jobs"},
        discrimination="The two predicted rows cannot both be present.",
        resource_estimate="1 call",
    )


def _limits(page_size: int, max_chars: int):
    from beyond_retrieval.models import QueryLimits

    return QueryLimits(page_size=page_size, max_chars=max_chars)


def _gap_answer_unused() -> FinalAnswer:
    return FinalAnswer(
        probabilities={"malicious": 0.2, "benign": 0.2, "insufficient_evidence": 0.6},
        verdict=Verdict.INSUFFICIENT,
        findings=[
            FindingAssertion(finding_id="F_FLOW", text="Transfer seen.", record_ids=["flow-1"]),
            FindingAssertion(finding_id="F_GAP", text="Host unresolved.", record_ids=["flow-1"]),
        ],
        limitations="Harness placeholder.",
    )


def _one_lesson():
    return freeze_lessons(
        [
            Lesson(
                lesson_id="L1",
                applicability="translated outbound volume",
                recommended_test="join translation to the job window",
                failure_conditions="reputation used as proof",
                provenance_case_ids=("fox-001",),
            )
        ],
        manifest_records(),
    )


def _script_policy(condition: ConditionId) -> ScriptedPolicy:
    contract = ActionContract(
        question="Does the visible mount contain the transfer rows?",
        hypotheses=[
            HypothesisPrediction(
                hypothesis_id="h_backup",
                explanation="Authorized backup.",
                predicted_observation="The row count equals the number of scheduled jobs only.",
            ),
            HypothesisPrediction(
                hypothesis_id="h_unauth",
                explanation="A larger visible episode.",
                predicted_observation="The row count includes flow, translation, process, and job rows.",
            ),
        ],
        tool="script",
        parameters={},
        discrimination="The count either matches the four-row episode or it does not.",
        resource_estimate="1 script",
    )
    proposal = Proposal(
        kind=ActionKind.SCRIPT,
        script_source=COUNT_SCRIPT,
        script_contract=ScriptContract(
            inputs=["events"],
            join_keys=["record_id"],
            units={"n": "count"},
            output_schema={"n": "int", "record_id": "str"},
            assumptions=["each visible row is one event"],
        ),
        contract=contract if condition is not ConditionId.A else None,
        generated_tokens=4,
    )
    return ScriptedPolicy(Capabilities.for_condition(condition), [[proposal]], [_gap_answer_unused()])


def prepare_one(path: Path) -> None:
    from beyond_retrieval.evidence import EvidenceMount

    event = outbound_case().events[0].model_copy(update={"record_id": "fix-1"})
    EvidenceMount([event], Variant.V0, "fixture").materialize(path)


def _internal_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("beyond_retrieval"):
            found.add(node.module.removeprefix("beyond_retrieval.").split(".")[0])
    return found


def test_esr_missing_is_not_a_mechanical_fail_label():
    missing = esr_result(RunStatus.INFRA_OUTAGE, None, None)
    assert missing.status == "missing"
    pending = mechanical_score(
        FinalAnswer(
            probabilities={"malicious": 1.0, "benign": 0.0, "insufficient_evidence": 0.0},
            verdict=Verdict.MALICIOUS,
            findings=[FindingAssertion(finding_id="F_HOST", text="host", record_ids=["gone"])],
            limitations="cited a missing row",
        ),
        outbound_case().rubrics[Variant.V0],
        {"flow-1"},
        {"flow-1"},
    )
    assert pending.passed is False
    assert pending.unknown_citations == ["gone"]
