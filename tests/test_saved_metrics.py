"""Known-count checks for metrics from immutable saved predictions only."""

from dataclasses import FrozenInstanceError, replace
from importlib import import_module

import numpy as np
import pytest

from automated_phishing_detection.paired_evaluation import (
    BinaryPrediction,
    EvaluationRecord,
)


def evaluator():
    return import_module("automated_phishing_detection.saved_metrics")


def evidence(labels, decisions, domains=None):
    domains = domains if domains is not None else ["a.test"] * len(labels)
    records = tuple(
        EvaluationRecord(f"row-{index}", domain, label)
        for index, (domain, label) in enumerate(zip(domains, labels, strict=True))
    )
    predictions = tuple(
        BinaryPrediction(record.record_id, decision)
        for record, decision in zip(records, decisions, strict=True)
    )
    return records, predictions


@pytest.mark.parametrize(
    ("numerator", "denominator", "estimate", "upper"),
    [
        (0, 400, 0.0, 1.0 - 0.05 ** (1.0 / 400.0)),
        (1, 400, 1 / 400, 0.01180430445398703),
        (400, 400, 1.0, 1.0),
        (0, 1, 0.0, 0.95),
        (1, 1, 1.0, 1.0),
    ],
)
def test_exact_rate_known_one_sided_95_percent_bounds(
    numerator, denominator, estimate, upper
):
    module = evaluator()
    result = module.exact_rate(numerator, denominator)
    assert isinstance(result, module.RateEstimate)
    assert result.numerator == numerator
    assert result.denominator == denominator
    assert result.estimate == estimate
    assert result.upper_95 == pytest.approx(upper, rel=1e-12)
    assert result.status == "estimated"


def test_empty_rate_is_not_estimable_instead_of_zero():
    module = evaluator()
    assert module.exact_rate(0, 0) == module.RateEstimate(
        numerator=0,
        denominator=0,
        estimate=None,
        upper_95=None,
        status="not_estimable",
    )


@pytest.mark.parametrize("position", [0, 1])
@pytest.mark.parametrize(
    "bad_count",
    [True, False, 1.0, -1, "1", None, float("nan"), float("inf"), np.int64(1)],
)
def test_exact_rate_rejects_non_builtin_or_negative_counts(position, bad_count):
    module = evaluator()
    counts = [1, 2]
    counts[position] = bad_count
    with pytest.raises(module.SavedMetricsError, match="nonnegative integer"):
        module.exact_rate(*counts)


@pytest.mark.parametrize("counts", [(1, 0), (3, 2)])
def test_exact_rate_rejects_numerator_above_denominator(counts):
    module = evaluator()
    with pytest.raises(module.SavedMetricsError, match="must not exceed"):
        module.exact_rate(*counts)


def test_rate_results_are_immutable():
    result = evaluator().exact_rate(1, 2)
    with pytest.raises(FrozenInstanceError):
        result.estimate = 0.0


def test_detection_metrics_keep_mixed_binary_labels_and_known_counts():
    module = evaluator()
    rows, predictions = evidence([1, 1, 1, 0, 0, 0, 0], [1, 0, 1, 1, 0, 0, 0])
    result = module.detection_metrics(rows, predictions)
    assert isinstance(result, module.DetectionMetrics)
    assert (
        result.true_positives,
        result.false_negatives,
        result.false_positives,
        result.true_negatives,
    ) == (2, 1, 1, 3)
    assert result.recall == module.exact_rate(2, 3)
    assert result.fpr == module.exact_rate(1, 4)
    with pytest.raises(FrozenInstanceError):
        result.true_positives = 0


@pytest.mark.parametrize("labels", [[], [0, 0], [1, 1]])
def test_absent_classes_are_not_estimable_and_do_not_drop_other_class(labels):
    module = evaluator()
    result = module.detection_metrics(*evidence(labels, [1] * len(labels)))
    assert result.recall == module.exact_rate(labels.count(1), labels.count(1))
    assert result.fpr == module.exact_rate(labels.count(0), labels.count(0))


def test_control_alert_rate_needs_ids_and_decisions_without_outcome_labels():
    module = evaluator()
    identities = ("control-0", "control-1", "control-2")
    predictions = tuple(
        BinaryPrediction(identity, decision)
        for identity, decision in zip(identities, [1, 0, 1], strict=True)
    )
    result = module.control_alert_rate(identities, predictions)
    assert result == module.exact_rate(2, 3)
    assert not hasattr(result, "fpr")
    assert module.control_alert_rate([], []) == module.exact_rate(0, 0)


@pytest.mark.parametrize("kind", ["detection", "control"])
def test_aligned_reordering_preserves_metrics_without_mutating_evidence(kind):
    module = evaluator()
    rows, predictions = evidence([1, 0, 1], [0, 1, 1])
    if kind == "control":
        rows = tuple(row.record_id for row in rows)
    function = getattr(
        module, f"{kind}_metrics" if kind == "detection" else "control_alert_rate"
    )
    original_rows, original_predictions = list(rows), list(predictions)
    assert function(rows, predictions) == function(rows[::-1], predictions[::-1])
    function(original_rows, original_predictions)
    assert original_rows == list(rows)
    assert original_predictions == list(predictions)


@pytest.mark.parametrize("kind", ["detection", "control"])
@pytest.mark.parametrize(
    "change", ["missing", "extra", "reordered", "wrong_id", "duplicate"]
)
def test_refuses_misalignment_without_dropping_or_sorting_rows(kind, change):
    module = evaluator()
    rows, predictions = evidence([1, 0], [0, 1])
    if change == "missing":
        predictions = predictions[:-1]
    elif change == "extra":
        predictions = (*predictions, predictions[0])
    elif change == "reordered":
        predictions = predictions[::-1]
    elif change == "wrong_id":
        predictions = (replace(predictions[0], record_id="wrong"), predictions[1])
    else:
        rows = (rows[0], replace(rows[1], record_id=rows[0].record_id))
        predictions = (
            predictions[0],
            replace(predictions[1], record_id=rows[0].record_id),
        )
    with pytest.raises(module.SavedMetricsError):
        if kind == "control":
            module.control_alert_rate(tuple(row.record_id for row in rows), predictions)
        else:
            module.detection_metrics(rows, predictions)


@pytest.mark.parametrize("kind", ["label", "detection_decision", "control_decision"])
@pytest.mark.parametrize(
    "value",
    [True, False, 1.0, -1, 2, "1", None, float("nan"), float("inf"), np.int64(1)],
)
def test_requires_exact_binary_integer_labels_and_decisions(kind, value):
    module = evaluator()
    rows, predictions = evidence([1], [1])
    if kind == "label":
        rows = (replace(rows[0], label=value),)
    else:
        predictions = (replace(predictions[0], decision=value),)
    with pytest.raises(module.SavedMetricsError, match="binary integer"):
        if kind == "control_decision":
            module.control_alert_rate((rows[0].record_id,), predictions)
        else:
            module.detection_metrics(rows, predictions)


@pytest.mark.parametrize(
    "domain",
    [
        "",
        None,
        12,
        "A.test",
        "a.test.",
        "a test",
        "https://a.test/",
        "a..test",
        "127.0.0.1",
        "0x7f.1",
        "::1",
        "b\u00fccher.test",
        "a\u200b.test",
        "xn--a.test",
    ],
)
def test_detection_requires_canonical_non_ip_domains(domain):
    module = evaluator()
    with pytest.raises(module.SavedMetricsError, match="domain"):
        module.detection_metrics(*evidence([1], [1], [domain]))


def test_accepts_canonical_idna_domains_and_printable_unicode_ids():
    module = evaluator()
    identity = "r\u00e9cord-0"
    rows = [EvaluationRecord(identity, "xn--bcher-kva.test", 1)]
    predictions = [BinaryPrediction(identity, 1)]
    assert module.detection_metrics(rows, predictions).recall == module.exact_rate(1, 1)
    assert module.control_alert_rate([identity], predictions) == module.exact_rate(1, 1)


@pytest.mark.parametrize("kind", ["detection", "control"])
@pytest.mark.parametrize("target", ["record", "prediction", "both"])
@pytest.mark.parametrize(
    "identity",
    [None, "", 12, " bad", "bad ", "line\nbreak", "a\u00a0b", "a\u200bb", "a\x00b"],
)
def test_requires_stable_printable_ids_without_whitespace(kind, target, identity):
    module = evaluator()
    row_id = identity if target in ("record", "both") else "row-0"
    prediction_id = identity if target in ("prediction", "both") else "row-0"
    predictions = [BinaryPrediction(prediction_id, 1)]
    with pytest.raises(module.SavedMetricsError, match="record ID"):
        if kind == "control":
            module.control_alert_rate([row_id], predictions)
        else:
            module.detection_metrics(
                [EvaluationRecord(row_id, "a.test", 1)], predictions
            )


@pytest.mark.parametrize("kind", ["detection", "control"])
@pytest.mark.parametrize("position", [0, 1])
@pytest.mark.parametrize("bad_sequence", [None, "", b"", {}, set(), iter(())])
def test_refuses_unordered_or_unmaterialized_sequences(kind, position, bad_sequence):
    module = evaluator()
    inputs = [(), ()]
    inputs[position] = bad_sequence
    with pytest.raises(module.SavedMetricsError, match="ordered sequences"):
        if kind == "control":
            module.control_alert_rate(*inputs)
        else:
            module.detection_metrics(*inputs)


@pytest.mark.parametrize("kind", ["detection", "control"])
@pytest.mark.parametrize("position", [0, 1])
@pytest.mark.parametrize("bad_item", [None, {}, "row-0", 1])
def test_refuses_untyped_evidence_items(kind, position, bad_item):
    module = evaluator()
    if kind == "control" and position == 0 and bad_item == "row-0":
        bad_item = EvaluationRecord("row-0", "a.test", 0)
    rows, predictions = evidence([1], [1])
    inputs = [rows, predictions]
    if kind == "control":
        inputs[0] = (rows[0].record_id,)
    inputs[position] = [bad_item]
    with pytest.raises(module.SavedMetricsError):
        if kind == "control":
            module.control_alert_rate(*inputs)
        else:
            module.detection_metrics(*inputs)
