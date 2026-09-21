"""Frozen primary decisions from synthetic saved evidence, never model runs."""

from dataclasses import replace
from importlib import import_module

import numpy as np
import pytest

from automated_phishing_detection.paired_evaluation import (
    BinaryPrediction,
    EvaluationRecord,
    RecallDifference,
)


@pytest.fixture
def evaluator():
    return import_module("automated_phishing_detection.hypothesis_evaluation")


def population(module, labels, decisions, prefix="row", domain_size=1):
    rows = tuple(
        EvaluationRecord(f"{prefix}-{i}", f"d{i // domain_size}.test", label)
        for i, label in enumerate(labels)
    )
    predictions = {
        model: tuple(
            BinaryPrediction(row.record_id, value)
            for row, value in zip(rows, values, strict=True)
        )
        for model, values in decisions.items()
    }
    return module.SavedPopulation(rows, predictions)


def passing_evidence(module):
    positive = {
        "length_only": [0, 0, 0, 0],
        "logistic_l1": [1, 0, 0, 0],
        "cascade": [1, 1, 0, 0],
        "transformer": [1, 1, 0, 0],
        "policy": [1, 1, 1, 0],
    }
    # Identical improvements in each of two domains give strict positive bounds.
    positive = {name: values * 2 for name, values in positive.items()}
    internal = population(
        module,
        [1] * 8 + [0] * 100,
        {
            name: values + [0] * 100
            for name, values in positive.items()
            if name != "policy"
        },
        "internal",
        domain_size=4,
    )
    gold = population(module, [1] * 8, positive, "gold", domain_size=4)
    certified = population(
        module, [0] * 100, {name: [0] * 100 for name in positive}, "certified"
    )
    ids = tuple(f"control-{i}" for i in range(100))
    controls = module.SavedControls(
        ids,
        {
            name: tuple(BinaryPrediction(identity, 0) for identity in ids)
            for name in ("cascade", "transformer")
        },
    )
    return {
        "populations": {"internal": internal, "gold": gold, "certified": certified},
        "controls": controls,
        "external_windows": module.WindowCounts(4, 5),
        "audit_windows": module.WindowCounts(1, 20),
        "reference": module.ReferenceInvocations(10000, 3000, 3000, 10000, 0),
        "http": module.PrimaryHttpSummary((10000,) * 5, 64, 49, 200.0),
    }


def gates(result, hypothesis):
    return {gate.name: gate for gate in result.hypotheses[hypothesis].gates}


def test_all_missing_stays_pending(evaluator):
    result = evaluator.evaluate_primary()
    assert len(result.contrasts) == 6
    assert all(value is None for value in result.contrasts.values())
    assert [len(result.hypotheses[h].gates) for h in ("H1", "H2", "H3")] == [10, 4, 8]
    for hypothesis in result.hypotheses.values():
        assert hypothesis.decision == "undecided"
        assert hypothesis.complete is False
        assert all(gate.status == "pending" for gate in hypothesis.gates)


def test_six_fixed_contrasts_and_original_non_support(evaluator):
    evidence = passing_evidence(evaluator)
    evidence["audit_windows"] = evaluator.WindowCounts(28, 252)
    result = evaluator.evaluate_primary(**evidence)
    assert set(result.contrasts) == {
        "internal.logistic_l1_minus_length_only",
        "internal.cascade_minus_logistic_l1",
        "gold.logistic_l1_minus_length_only",
        "gold.cascade_minus_logistic_l1",
        "gold.policy_minus_cascade",
        "gold.cascade_minus_transformer",
    }
    assert result.hypotheses["H1"].decision == "supported"
    assert result.hypotheses["H3"].decision == "supported"
    assert result.hypotheses["H2"].decision == "not_supported"
    assert all(h.complete for h in result.hypotheses.values())
    assert gates(result, "H2")["audit_window_alerts"].status == "fail"
    assert gates(result, "H2")["gold.policy_minus_cascade"].estimate == 0.25


def test_all_hypotheses_can_pass_only_when_every_required_gate_passes(evaluator):
    result = evaluator.evaluate_primary(**passing_evidence(evaluator))
    assert all(
        h.decision == "supported" and h.complete for h in result.hypotheses.values()
    )


def test_identical_policy_predictions_fail_strict_improvement(evaluator):
    evidence = passing_evidence(evaluator)
    gold = evidence["populations"]["gold"]
    evidence["populations"]["gold"] = replace(
        gold, predictions={**gold.predictions, "policy": gold.predictions["cascade"]}
    )
    result = evaluator.evaluate_primary(**evidence)
    gate = gates(result, "H2")["gold.policy_minus_cascade"]
    assert gate.estimate == 0.0 and gate.status == "fail"


@pytest.mark.parametrize("role", ["internal", "certified"])
@pytest.mark.parametrize("model", ["length_only", "logistic_l1", "cascade"])
def test_each_of_six_h1_fpr_gates_is_required(evaluator, role, model):
    evidence = passing_evidence(evaluator)
    saved = evidence["populations"][role]
    predictions = dict(saved.predictions)
    negative_indices = [i for i, row in enumerate(saved.records) if row.label == 0]
    predictions[model] = tuple(
        replace(prediction, decision=1) if i in negative_indices[:2] else prediction
        for i, prediction in enumerate(predictions[model])
    )
    evidence["populations"][role] = replace(saved, predictions=predictions)
    result = evaluator.evaluate_primary(**evidence)
    assert gates(result, "H1")[f"{role}.{model}.fpr"].status == "fail"
    assert result.hypotheses["H1"].decision == "not_supported"


def test_route_complete_stream_before_selecting_gold_outcomes(evaluator):
    from automated_phishing_detection.policy_replay import (
        MonitorScore,
        PairedProbabilities,
        replay_policy,
    )

    # The first 256 secondary/control rows cause an alert; later gold rows benefit.
    ids = tuple(f"stream-{i}" for i in range(258))
    probabilities = tuple(PairedProbabilities(identity, 0.0, 1.0) for identity in ids)
    monitor = tuple(MonitorScore(identity, 2.0) for identity in ids)
    rules = dict(
        stage1_threshold=0.5,
        transformer_threshold=0.5,
        half_width=0.0,
        monitor_boundary=1.0,
    )
    full = replay_policy(probabilities, monitor, **rules)
    filtered_first = replay_policy(probabilities[-2:], monitor[-2:], **rules)
    gold = tuple(
        EvaluationRecord(identity, f"gold{i}.test", 1)
        for i, identity in enumerate(ids[-2:])
    )

    def evaluate_rows(rows):
        return evaluator.evaluate_primary(
            populations={
                "gold": evaluator.SavedPopulation(
                    gold,
                    {
                        "cascade": tuple(
                            BinaryPrediction(row.record_id, row.fixed_decision)
                            for row in rows
                        ),
                        "policy": tuple(
                            BinaryPrediction(row.record_id, row.policy_decision)
                            for row in rows
                        ),
                    },
                )
            }
        )

    correct = evaluate_rows(full.rows[-2:])
    incorrect = evaluate_rows(filtered_first.rows)
    assert gates(correct, "H2")["gold.policy_minus_cascade"].estimate == 1.0
    assert gates(incorrect, "H2")["gold.policy_minus_cascade"].status == "fail"
    assert len(full.windows) == 1 and len(filtered_first.windows) == 0


def test_known_audit_failure_decides_h2_without_claiming_completion(evaluator):
    result = evaluator.evaluate_primary(audit_windows=evaluator.WindowCounts(28, 252))
    assert result.hypotheses["H2"].decision == "not_supported"
    assert result.hypotheses["H2"].complete is False
    assert result.hypotheses["H1"].decision == "undecided"


@pytest.mark.parametrize("false_positives,status", [(1, "pass"), (2, "fail")])
def test_final_fpr_gate_uses_observed_rate_not_cp_upper(
    evaluator, false_positives, status
):
    data = population(
        evaluator,
        [0] * 100,
        {"cascade": [1] * false_positives + [0] * (100 - false_positives)},
    )
    result = evaluator.evaluate_primary(populations={"certified": data})
    gate = gates(result, "H3")["certified.cascade.fpr"]
    assert gate.status == status
    assert gate.estimate == false_positives / 100
    assert gate.upper_95 > 0.01
    assert gate.numerator == false_positives and gate.denominator == 100


@pytest.mark.parametrize("model", ["cascade", "transformer"])
def test_either_tranco_safeguard_can_prevent_h3(evaluator, model):
    evidence = passing_evidence(evaluator)
    control = evidence["controls"]
    predictions = dict(control.predictions)
    predictions[model] = tuple(
        replace(row, decision=int(i < 2)) for i, row in enumerate(predictions[model])
    )
    evidence["controls"] = replace(control, predictions=predictions)
    result = evaluator.evaluate_primary(**evidence)
    assert result.hypotheses["H3"].decision == "not_supported"
    assert gates(result, "H3")[f"tranco.{model}.alert_rate"].status == "fail"
    assert (
        len([g for g in result.hypotheses["H3"].gates if g.upper_95 is not None]) == 4
    )


@pytest.mark.parametrize(
    "field,counts,status",
    [
        ("external_windows", (4, 5), "pass"),
        ("external_windows", (3, 5), "fail"),
        ("audit_windows", (1, 20), "pass"),
        ("audit_windows", (2, 20), "fail"),
        ("external_windows", (0, 0), "not_estimable"),
    ],
)
def test_window_integer_boundaries_no_binomial_interval(
    evaluator, field, counts, status
):
    result = evaluator.evaluate_primary(**{field: evaluator.WindowCounts(*counts)})
    name = (
        "external_window_alerts"
        if field == "external_windows"
        else "audit_window_alerts"
    )
    gate = gates(result, "H2")[name]
    assert gate.status == status
    assert gate.upper_95 is None
    assert gate.estimate == (counts[0] / counts[1] if counts[1] else None)


@pytest.mark.parametrize("attempts,status", [(3000, "pass"), (3001, "fail")])
def test_actual_reference_invocation_boundary(evaluator, attempts, status):
    result = evaluator.evaluate_primary(
        reference=evaluator.ReferenceInvocations(10000, attempts, attempts, 10000, 0)
    )
    assert gates(result, "H3")["reference_transformer_invocations"].status == status


def test_incomplete_execution_cannot_pass_invocation_gate(evaluator):
    result = evaluator.evaluate_primary(
        reference=evaluator.ReferenceInvocations(10000, 1, 0, 9999, 1)
    )
    gate = gates(result, "H3")["reference_transformer_invocations"]
    assert gate.status == "not_estimable"
    assert gate.reason == "reference_execution_incomplete"


@pytest.mark.parametrize("errors,status", [(49, "pass"), (50, "fail")])
def test_http_strict_error_boundary(evaluator, errors, status):
    result = evaluator.evaluate_primary(
        http=evaluator.PrimaryHttpSummary((10000,) * 5, 64, errors, 200.0)
    )
    gate = gates(result, "H3")["http_request_errors"]
    assert gate.status == status
    assert gate.denominator == 50000
    assert gates(result, "H3")["http_pooled_p95_ms"].status == "pass"


def test_latency_above_unrounded_boundary_fails(evaluator):
    result = evaluator.evaluate_primary(
        http=evaluator.PrimaryHttpSummary(
            (10000,) * 5, 64, 0, float(np.nextafter(200.0, np.inf))
        )
    )
    assert gates(result, "H3")["http_pooled_p95_ms"].status == "fail"


@pytest.mark.parametrize(
    "lower,margin,status",
    [
        (0.0, 0.0, "fail"),
        (1e-12, 0.0, "pass"),
        (-0.02, -0.02, "pass"),
        (float(np.nextafter(-0.02, -np.inf)), -0.02, "fail"),
    ],
)
def test_recall_gate_uses_unrounded_lower_endpoint(evaluator, lower, margin, status):
    difference = RecallDifference(
        "estimated", None, 100, 2, 50, 50, 0.0, lower, 1.0, 2000
    )
    assert evaluator._recall_gate("contrast", difference, margin).status == status


@pytest.mark.parametrize("rows", [0, 1])
def test_empty_and_single_domain_never_pass_recall_gate(evaluator, rows):
    data = population(
        evaluator, [1] * rows, {"cascade": [1] * rows, "transformer": [0] * rows}
    )
    result = evaluator.evaluate_primary(populations={"gold": data})
    assert (
        gates(result, "H3")["gold.cascade_minus_transformer"].status == "not_estimable"
    )


def test_empty_negative_stratum_is_not_zero_fpr(evaluator):
    data = population(evaluator, [1, 1], {"cascade": [1, 1]})
    result = evaluator.evaluate_primary(populations={"internal": data})
    assert gates(result, "H1")["internal.cascade.fpr"].status == "not_estimable"


@pytest.mark.parametrize("role,label", [("gold", 0), ("certified", 1)])
def test_rejects_wrong_population_labels(evaluator, role, label):
    data = population(evaluator, [label], {"cascade": [1]})
    with pytest.raises(ValueError):
        evaluator.evaluate_primary(populations={role: data})


@pytest.mark.parametrize(
    "overlap", ["gold_certified", "gold_tranco", "certified_tranco"]
)
def test_external_outcome_strata_cannot_share_record_ids(evaluator, overlap):
    gold = population(evaluator, [1], {"cascade": [1]}, "gold")
    certified = population(evaluator, [0], {"cascade": [0]}, "certified")
    control_id = "tranco-0"
    if overlap == "gold_certified":
        certified = population(evaluator, [0], {"cascade": [0]}, "gold")
    else:
        control_id = "gold-0" if overlap == "gold_tranco" else "certified-0"
    controls = evaluator.SavedControls(
        (control_id,), {"cascade": (BinaryPrediction(control_id, 0),)}
    )
    with pytest.raises(ValueError, match="overlap"):
        evaluator.evaluate_primary(
            populations={"gold": gold, "certified": certified}, controls=controls
        )


@pytest.mark.parametrize(
    "change",
    [
        "unknown_population",
        "unknown_model",
        "empty_models",
        "reordered",
        "omitted_row",
        "bool_decision",
        "duplicate_id",
    ],
)
def test_rejects_malformed_saved_evidence(evaluator, change):
    data = population(evaluator, [0, 1], {"cascade": [0, 1]})
    role = "internal"
    if change == "unknown_population":
        role = "validation"
    elif change == "unknown_model":
        data = replace(data, predictions={"random_forest": data.predictions["cascade"]})
    elif change == "empty_models":
        data = replace(data, predictions={})
    elif change == "reordered":
        data = replace(data, predictions={"cascade": data.predictions["cascade"][::-1]})
    elif change == "omitted_row":
        data = replace(data, predictions={"cascade": data.predictions["cascade"][:-1]})
    elif change == "bool_decision":
        data = replace(
            data,
            predictions={
                "cascade": (
                    replace(data.predictions["cascade"][0], decision=True),
                    data.predictions["cascade"][1],
                )
            },
        )
    else:
        data = replace(data, records=(data.records[0], data.records[0]))
    with pytest.raises(ValueError):
        evaluator.evaluate_primary(populations={role: data})


@pytest.mark.parametrize("counts", [(True, 1), (1.0, 2), (-1, 2), (3, 2), (0, -1)])
def test_rejects_malformed_window_counts(evaluator, counts):
    with pytest.raises(ValueError):
        evaluator.evaluate_primary(audit_windows=evaluator.WindowCounts(*counts))


@pytest.mark.parametrize(
    "values",
    [
        ((10000,) * 4, 64, 0, 1.0),
        ((11000,) * 5, 64, 0, 1.0),
        ((10000,) * 5, 32, 0, 1.0),
        ((10000,) * 5, 64, True, 1.0),
        ((10000,) * 5, 64, 50001, 1.0),
        ((10000,) * 5, 64, 0, float("nan")),
        ((10000,) * 5, 64, 0, float("inf")),
        ((10000,) * 5, 64, 0, -1.0),
        ((10000,) * 5, 64, 0, True),
    ],
)
def test_rejects_invalid_primary_http_summaries(evaluator, values):
    with pytest.raises(ValueError):
        evaluator.evaluate_primary(http=evaluator.PrimaryHttpSummary(*values))


@pytest.mark.parametrize(
    "values",
    [
        (9999, 0, 0, 9999, 0),
        (10000, True, True, 10000, 0),
        (10000, 2, 3, 10000, 0),
        (10000, 1, 0, 10000, 0),
        (10000, 0, 0, 9999, 0),
        (10000, 10001, 10001, 10000, 0),
    ],
)
def test_rejects_invalid_or_logical_only_invocation_counts(evaluator, values):
    with pytest.raises(ValueError):
        evaluator.evaluate_primary(reference=evaluator.ReferenceInvocations(*values))
