"""Synthetic full-stream integration evidence, never a research/model run."""

from dataclasses import asdict, fields, replace
from importlib import import_module

import pytest

from automated_phishing_detection import hypothesis_evaluation, policy_replay
from automated_phishing_detection.paired_evaluation import BinaryPrediction
from automated_phishing_detection.policy_replay import MonitorScore, PairedProbabilities

RULES = {
    "stage1_threshold": 0.5,
    "transformer_threshold": 0.75,
    "half_width": 0.125,
    "monitor_boundary": 1.0,
}


@pytest.fixture
def stream():
    return import_module("automated_phishing_detection.evaluation_stream")


def inputs(stream, roles, *, nll=2.0):
    metadata = tuple(
        stream.OutcomeMetadata(
            f"synthetic-{i:04d}",
            f"d{i}.test",
            {"gold": 1, "certified": 0, "tranco": None, "secondary": None}[role],
            role,
        )
        for i, role in enumerate(roles)
    )
    return {
        "metadata": metadata,
        "probabilities": tuple(
            PairedProbabilities(row.record_id, 0.125, 0.75) for row in metadata
        ),
        "monitor_scores": tuple(MonitorScore(row.record_id, nll) for row in metadata),
        "length_predictions": tuple(
            BinaryPrediction(row.record_id, 0) for row in metadata
        ),
    }


def build(stream, evidence, **rules):
    return stream.build_external_evidence(**evidence, **{**RULES, **rules})


def test_complete_label_blind_stream_is_replayed_once_before_stratification(
    stream, monkeypatch
):
    evidence = inputs(
        stream, ["secondary", "tranco"] * 128 + ["gold", "gold", "certified"]
    )
    expected = policy_replay.replay_policy(
        evidence["probabilities"], evidence["monitor_scores"], **RULES
    )
    calls = []
    original = policy_replay.replay_policy

    def observe(probabilities, monitor_scores, **rules):
        calls.append((probabilities, monitor_scores, rules))
        return original(probabilities, monitor_scores, **rules)

    monkeypatch.setattr(policy_replay, "replay_policy", observe)
    result = build(stream, evidence)

    assert calls == [(evidence["probabilities"], evidence["monitor_scores"], RULES)]
    assert calls[0][0] is evidence["probabilities"]
    assert calls[0][1] is evidence["monitor_scores"]
    assert result.replay == expected
    assert all(
        row.policy_decision == row.fixed_decision == 0
        for row in result.replay.rows[:256]
    )
    assert all(row.policy_decision == 1 for row in result.replay.rows[256:])
    assert result.external_windows == hypothesis_evaluation.WindowCounts(1, 1)
    gold = result.populations["gold"]
    assert [row.label for row in gold.records] == [1, 1]
    assert [row.decision for row in gold.predictions["policy"]] == [1, 1]
    assert [row.decision for row in gold.predictions["cascade"]] == [0, 0]
    assert result.role_counts == {
        "gold": 2,
        "certified": 1,
        "tranco": 128,
        "secondary": 128,
    }
    assert tuple(result.populations) == ("gold", "certified")


@pytest.mark.parametrize(
    "count,windows", [(0, 0), (255, 0), (256, 1), (257, 1), (319, 1), (320, 2)]
)
def test_complete_window_denominator_includes_secondary_rows(stream, count, windows):
    result = build(stream, inputs(stream, ["secondary"] * count))

    assert len(result.replay.rows) == count
    assert result.external_windows == hypothesis_evaluation.WindowCounts(
        windows, windows
    )
    assert result.role_counts == {
        "gold": 0,
        "certified": 0,
        "tranco": 0,
        "secondary": count,
    }
    assert all(population.records == () for population in result.populations.values())
    assert result.controls.record_ids == ()


def test_monitor_equality_is_nonalert_without_changing_routing(stream):
    result = build(stream, inputs(stream, ["secondary"] * 256 + ["gold"], nll=1.0))

    assert result.external_windows == hypothesis_evaluation.WindowCounts(0, 1)
    assert result.populations["gold"].predictions["policy"][0].decision == 0


def test_all_models_keep_stratum_order_and_use_frozen_thresholds(stream):
    evidence = inputs(stream, ["gold", "certified", "gold", "tranco", "secondary"])
    pairs = [(0.5, 0.5), (0.9, 0.75), (0.375, 0.75), (0.125, 0.75), (0.9, 0.9)]
    evidence["probabilities"] = tuple(
        PairedProbabilities(row.record_id, stage1, transformer)
        for row, (stage1, transformer) in zip(evidence["metadata"], pairs, strict=True)
    )
    evidence["length_predictions"] = tuple(
        BinaryPrediction(row.record_id, int(i == 0))
        for i, row in enumerate(evidence["metadata"])
    )
    result = build(stream, evidence)

    expected = {
        "gold": {
            "length_only": [1, 0],
            "logistic_l1": [1, 0],
            "transformer": [0, 1],
            "cascade": [0, 1],
            "policy": [0, 1],
        },
        "certified": {
            "length_only": [0],
            "logistic_l1": [1],
            "transformer": [1],
            "cascade": [1],
            "policy": [1],
        },
    }
    for role, decisions in expected.items():
        population = result.populations[role]
        selected = [row for row in evidence["metadata"] if row.role == role]
        assert tuple(row.record_id for row in population.records) == tuple(
            row.record_id for row in selected
        )
        assert tuple(row.registrable_domain for row in population.records) == tuple(
            row.registrable_domain for row in selected
        )
        assert set(population.predictions) == hypothesis_evaluation.MODELS
        for model, values in decisions.items():
            predictions = population.predictions[model]
            assert [row.decision for row in predictions] == values
            assert [row.record_id for row in predictions] == [
                row.record_id for row in selected
            ]

    assert result.controls.record_ids == (evidence["metadata"][3].record_id,)
    assert set(result.controls.predictions) == {"cascade", "transformer"}
    assert [row.decision for row in result.controls.predictions["cascade"]] == [0]
    assert [row.decision for row in result.controls.predictions["transformer"]] == [1]
    assert {field.name for field in fields(result.controls)} == {
        "record_ids",
        "predictions",
    }


def test_saved_output_feeds_existing_primary_arithmetic_without_operational_claims(
    stream,
):
    evidence = inputs(
        stream, ["secondary", "tranco"] * 128 + ["gold", "gold", "certified"]
    )
    probabilities = list(evidence["probabilities"])
    probabilities[-1] = replace(probabilities[-1], transformer_probability=0.0)
    evidence["probabilities"] = tuple(probabilities)
    result = build(stream, evidence)
    primary = hypothesis_evaluation.evaluate_primary(
        populations=result.populations,
        controls=result.controls,
        external_windows=result.external_windows,
    )

    difference = primary.contrasts["gold.policy_minus_cascade"]
    assert difference.estimate == difference.lower == difference.upper == 1.0
    h2 = {gate.name: gate for gate in primary.hypotheses["H2"].gates}
    assert h2["external_window_alerts"].status == "pass"
    assert h2["gold.policy_minus_cascade"].status == "pass"
    assert h2["certified.policy.fpr"].estimate == 0.0
    assert h2["audit_window_alerts"].status == "pending"
    assert primary.hypotheses["H2"].complete is False
    assert primary.hypotheses["H2"].decision == "undecided"
    assert primary.metrics["tranco.transformer"].denominator == 128


@pytest.mark.parametrize("label", [0, 1, None])
def test_secondary_publisher_labels_never_promote_rows_to_primary_strata(stream, label):
    evidence = inputs(stream, ["secondary"])
    evidence["metadata"] = (replace(evidence["metadata"][0], is_phishing=label),)
    result = build(stream, evidence)

    assert result.role_counts["secondary"] == 1
    assert len(result.replay.rows) == 1
    assert all(population.records == () for population in result.populations.values())
    assert result.controls.record_ids == ()


def forbid_replay(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("routing preceded complete stream validation")

    monkeypatch.setattr(policy_replay, "replay_policy", forbidden)


@pytest.mark.parametrize(
    "field", ["metadata", "probabilities", "monitor_scores", "length_predictions"]
)
@pytest.mark.parametrize(
    "change", ["missing", "extra", "reordered", "different", "mapping"]
)
def test_rejects_incomplete_untyped_or_misaligned_stream_before_routing(
    stream, monkeypatch, field, change
):
    evidence = inputs(stream, ["gold", "certified", "tranco"])
    rows = evidence[field]
    if change == "missing":
        rows = rows[:-1]
    elif change == "extra":
        rows = (*rows, replace(rows[-1], record_id="extra"))
    elif change == "reordered":
        rows = rows[::-1]
    elif change == "different":
        rows = (*rows[:-1], replace(rows[-1], record_id="different"))
    else:
        rows = (*rows[:-1], asdict(rows[-1]))
    evidence[field] = rows
    forbid_replay(monkeypatch)
    with pytest.raises(stream.EvaluationStreamError, match="counts|typed|IDs or order"):
        build(stream, evidence)


@pytest.mark.parametrize(
    "field", ["metadata", "probabilities", "monitor_scores", "length_predictions"]
)
@pytest.mark.parametrize("bad", [None, "records", {}, iter(())])
def test_requires_materialized_sequences_before_routing(
    stream, monkeypatch, field, bad
):
    evidence = inputs(stream, [])
    evidence[field] = bad
    forbid_replay(monkeypatch)
    with pytest.raises(stream.EvaluationStreamError, match="ordered sequences"):
        build(stream, evidence)


@pytest.mark.parametrize(
    "identity", ["", "with space", "line\n", "hidden\x00", None, 7]
)
def test_rejects_invalid_ids_even_when_every_input_agrees(
    stream, monkeypatch, identity
):
    evidence = inputs(stream, ["secondary"])
    evidence = {
        key: (replace(rows[0], record_id=identity),) for key, rows in evidence.items()
    }
    forbid_replay(monkeypatch)
    with pytest.raises(stream.EvaluationStreamError, match="record ID"):
        build(stream, evidence)


def test_rejects_duplicate_ids_across_outcome_roles(stream, monkeypatch):
    evidence = inputs(stream, ["gold", "tranco"])
    evidence = {
        key: (rows[0], replace(rows[1], record_id=rows[0].record_id))
        for key, rows in evidence.items()
    }
    forbid_replay(monkeypatch)
    with pytest.raises(stream.EvaluationStreamError, match="unique"):
        build(stream, evidence)


@pytest.mark.parametrize("role", ["unknown", "Gold", "", None, []])
def test_rejects_unknown_roles_before_routing(stream, monkeypatch, role):
    evidence = inputs(stream, ["secondary"] * 256 + ["gold"])
    evidence["metadata"] = (
        *evidence["metadata"][:-1],
        replace(evidence["metadata"][-1], role=role),
    )
    forbid_replay(monkeypatch)
    with pytest.raises(stream.EvaluationStreamError, match="role"):
        build(stream, evidence)


@pytest.mark.parametrize(
    "role,label",
    [
        ("gold", 0),
        ("gold", None),
        ("certified", 1),
        ("certified", None),
        ("tranco", 0),
        ("tranco", 1),
        ("gold", True),
        ("certified", False),
        ("secondary", True),
        ("secondary", 0.0),
        ("secondary", "1"),
        ("secondary", 2),
    ],
)
def test_rejects_invalid_or_role_inconsistent_labels_before_routing(
    stream, monkeypatch, role, label
):
    evidence = inputs(stream, ["secondary"] * 256 + [role])
    evidence["metadata"] = (
        *evidence["metadata"][:-1],
        replace(evidence["metadata"][-1], is_phishing=label),
    )
    forbid_replay(monkeypatch)
    with pytest.raises(stream.EvaluationStreamError, match="label|is_phishing"):
        build(stream, evidence)


@pytest.mark.parametrize(
    "domain",
    [
        None,
        "",
        "UPPER.test",
        "trailing.test.",
        "with space.test",
        "https://example.test",
        "-bad.test",
        "127.0.0.1",
        "127.1",
        "0x7f000001",
        "::1",
        "xn--.test",
    ],
)
def test_validates_all_domain_syntax_without_repairing_before_routing(
    stream, monkeypatch, domain
):
    evidence = inputs(stream, ["secondary"] * 256 + ["tranco"])
    evidence["metadata"] = (
        *evidence["metadata"][:-1],
        replace(evidence["metadata"][-1], registrable_domain=domain),
    )
    forbid_replay(monkeypatch)
    with pytest.raises(stream.EvaluationStreamError, match="domain"):
        build(stream, evidence)


@pytest.mark.parametrize("decision", [True, False, None, 0.0, "0", -1, 2])
def test_validates_secondary_length_decisions_before_routing(
    stream, monkeypatch, decision
):
    evidence = inputs(stream, ["gold"] * 256 + ["secondary"])
    evidence["length_predictions"] = (
        *evidence["length_predictions"][:-1],
        replace(evidence["length_predictions"][-1], decision=decision),
    )
    forbid_replay(monkeypatch)
    with pytest.raises(stream.EvaluationStreamError, match="binary integer"):
        build(stream, evidence)


@pytest.mark.parametrize(
    "field,attribute,bad",
    [
        ("probabilities", "stage1_probability", float("nan")),
        ("probabilities", "transformer_probability", 1.1),
        ("monitor_scores", "negative_log_likelihood", float("inf")),
    ],
)
def test_validates_scores_before_routing(stream, monkeypatch, field, attribute, bad):
    evidence = inputs(stream, ["gold"] * 256 + ["secondary"])
    rows = evidence[field]
    evidence[field] = (*rows[:-1], replace(rows[-1], **{attribute: bad}))
    forbid_replay(monkeypatch)
    with pytest.raises(stream.EvaluationStreamError, match=attribute):
        build(stream, evidence)


@pytest.mark.parametrize(
    "field,bad",
    [
        ("stage1_threshold", True),
        ("transformer_threshold", 1.1),
        ("half_width", -0.1),
        ("monitor_boundary", float("nan")),
    ],
)
def test_empty_stream_still_uses_existing_configuration_validation(stream, field, bad):
    with pytest.raises(stream.EvaluationStreamError, match=field):
        build(stream, inputs(stream, []), **{field: bad})
