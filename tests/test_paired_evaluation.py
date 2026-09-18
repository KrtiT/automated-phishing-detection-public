"""Known-answer checks for paired, domain-clustered recall differences."""

import json
from dataclasses import replace
from importlib import import_module
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def evaluator():
    return import_module("automated_phishing_detection.paired_evaluation")


def evidence(module, domains, candidate, reference, labels=None):
    labels = labels if labels is not None else [1] * len(domains)
    records = tuple(
        module.EvaluationRecord(f"row-{index}", domain, label)
        for index, (domain, label) in enumerate(zip(domains, labels, strict=True))
    )
    left = tuple(
        module.BinaryPrediction(row.record_id, decision)
        for row, decision in zip(records, candidate, strict=True)
    )
    right = tuple(
        module.BinaryPrediction(row.record_id, decision)
        for row, decision in zip(records, reference, strict=True)
    )
    return records, left, right


def test_unequal_domains_keep_url_weighting_and_paired_rows(evaluator):
    inputs = evidence(
        evaluator,
        ["a.test"] * 3 + ["b.test"],
        [1, 1, 1, 0],
        [0, 0, 0, 1],
    )
    result = evaluator.paired_recall_difference(*inputs)
    assert result.status == "estimated"
    assert result.reason is None
    assert result.positive_count == 4
    assert result.domain_count == 2
    assert result.candidate_true_positives == 3
    assert result.reference_true_positives == 1
    assert result.estimate == 0.5  # Not the unweighted domain mean, zero.
    assert (result.lower, result.upper) == (-1.0, 1.0)
    assert result.bootstrap_replicates == 2000


def test_two_cluster_resamples_have_hand_computable_ratios(evaluator):
    distribution = evaluator._bootstrap_differences(
        np.array([3, -1], dtype=np.int64), np.array([3, 1], dtype=np.int64)
    )
    # AA, AB/BA and BB replicate all rows of each selected domain together.
    assert set(distribution) == {-1.0, 0.5, 1.0}
    rng = np.random.Generator(np.random.PCG64(20260816))
    expected = []
    for _ in range(2000):
        draw = rng.integers(0, 2, size=2, dtype=np.int64)
        expected.append(
            {(0, 0): 1.0, (0, 1): 0.5, (1, 0): 0.5, (1, 1): -1.0}[tuple(draw)]
        )
    np.testing.assert_array_equal(distribution, expected)


def test_interval_matches_row_expansion_reference_and_linear_quantiles(evaluator):
    sizes = [1, 2, 3, 4, 5, 6, 7, 8]
    changes = [1, -1, 0, 1, 0, -1, 1, 0]
    domains = [
        f"d{index}.test" for index, size in enumerate(sizes) for _ in range(size)
    ]
    candidate = [
        int(change == 1) for change, size in zip(changes, sizes) for _ in range(size)
    ]
    reference = [
        int(change == -1) for change, size in zip(changes, sizes) for _ in range(size)
    ]
    inputs = evidence(evaluator, domains, candidate, reference)
    result = evaluator.paired_recall_difference(*inputs)
    rng = np.random.Generator(np.random.PCG64(20260816))
    samples = []
    for _ in range(2000):
        chosen = rng.integers(0, len(sizes), size=len(sizes), dtype=np.int64)
        rows = [changes[index] for index in chosen for _ in range(sizes[index])]
        samples.append(sum(rows) / len(rows))
    lower, upper = np.quantile(samples, [0.025, 0.975], method="linear")
    assert result.estimate == (sum(candidate) - sum(reference)) / len(domains)
    assert result.lower == lower
    assert result.upper == upper


def test_refuses_rows_outside_the_already_selected_positive_stratum(evaluator):
    inputs = evidence(
        evaluator,
        ["a.test", "b.test", "negative.test"],
        [1, 0, 1],
        [0, 1, 0],
        labels=[1, 1, 0],
    )
    with pytest.raises(evaluator.PairedEvaluationError, match="positive stratum"):
        evaluator.paired_recall_difference(*inputs)


def test_complete_aligned_reordering_and_global_rng_do_not_change_result(evaluator):
    inputs = evidence(
        evaluator, ["b.test", "a.test", "b.test", "c.test"], [1, 0, 1, 0], [0, 1, 0, 0]
    )
    first = evaluator.paired_recall_difference(*inputs)
    np.random.seed(999)
    reordered = tuple(tuple(items[index] for index in [3, 2, 0, 1]) for items in inputs)
    assert evaluator.paired_recall_difference(*reordered) == first


def test_reversing_contrast_negates_estimate_and_interval(evaluator):
    rows, left, right = evidence(
        evaluator, ["a.test", "b.test", "c.test"], [1, 1, 0], [0, 0, 1]
    )
    forward = evaluator.paired_recall_difference(rows, left, right)
    reverse = evaluator.paired_recall_difference(rows, right, left)
    assert reverse.estimate == -forward.estimate
    assert reverse.lower == pytest.approx(-forward.upper)
    assert reverse.upper == pytest.approx(-forward.lower)


def test_absent_positive_stratum_is_not_zero_recall(evaluator):
    result = evaluator.paired_recall_difference((), (), ())
    assert result.status == "not_estimable"
    assert result.reason == "no_positive_rows"
    assert result.positive_count == result.domain_count == 0
    assert result.estimate is result.lower is result.upper is None
    assert result.bootstrap_replicates == 0


def test_one_domain_has_point_estimate_but_no_cluster_interval(evaluator):
    result = evaluator.paired_recall_difference(
        *evidence(evaluator, ["a.test"] * 2, [1, 1], [0, 1])
    )
    assert result.status == "not_estimable"
    assert result.reason == "insufficient_domain_clusters"
    assert result.estimate == 0.5
    assert result.lower is result.upper is None
    assert result.bootstrap_replicates == 0


def test_identical_predictions_report_degenerate_zero_interval(evaluator):
    result = evaluator.paired_recall_difference(
        *evidence(evaluator, ["a.test", "b.test"], [1, 0], [1, 0])
    )
    assert result.status == "estimated"
    assert result.estimate == result.lower == result.upper == 0.0


@pytest.mark.parametrize(
    "change", ["missing", "extra", "reordered", "wrong_id", "duplicate_record"]
)
def test_refuses_misaligned_evidence(evaluator, change):
    rows, left, right = evidence(evaluator, ["a.test", "b.test"], [1, 0], [0, 1])
    if change == "missing":
        left = left[:-1]
    elif change == "extra":
        right = (*right, right[0])
    elif change == "reordered":
        right = right[::-1]
    elif change == "wrong_id":
        left = (replace(left[0], record_id="unknown"), left[1])
    else:
        rows = (rows[0], replace(rows[1], record_id=rows[0].record_id))
    with pytest.raises(evaluator.PairedEvaluationError):
        evaluator.paired_recall_difference(rows, left, right)


@pytest.mark.parametrize(
    "bad_value", [True, 1.0, -1, 2, "1", float("nan"), float("inf"), None]
)
@pytest.mark.parametrize("field", ["label", "candidate", "reference"])
def test_refuses_nonbinary_or_noninteger_values(evaluator, bad_value, field):
    rows, left, right = evidence(evaluator, ["a.test"], [1], [0])
    if field == "label":
        rows = (replace(rows[0], label=bad_value),)
    elif field == "candidate":
        left = (replace(left[0], decision=bad_value),)
    else:
        right = (replace(right[0], decision=bad_value),)
    with pytest.raises(evaluator.PairedEvaluationError, match="binary integer"):
        evaluator.paired_recall_difference(rows, left, right)


@pytest.mark.parametrize(
    "domain",
    [
        "",
        None,
        "A.test",
        "a.test.",
        "a test",
        "https://a.test/",
        "a..test",
        "127.0.0.1",
    ],
)
def test_requires_canonical_non_ip_domain_metadata(evaluator, domain):
    inputs = evidence(evaluator, [domain], [1], [0])
    with pytest.raises(evaluator.PairedEvaluationError, match="domain"):
        evaluator.paired_recall_difference(*inputs)


@pytest.mark.parametrize("record_id", [None, "", 12, " bad", "line\nbreak"])
def test_requires_stable_nonempty_record_identity(evaluator, record_id):
    rows, left, right = evidence(evaluator, ["a.test"], [1], [0])
    rows = (replace(rows[0], record_id=record_id),)
    left = (replace(left[0], record_id=record_id),)
    right = (replace(right[0], record_id=record_id),)
    with pytest.raises(evaluator.PairedEvaluationError, match="record ID"):
        evaluator.paired_recall_difference(rows, left, right)


@pytest.mark.parametrize(
    "inputs", [(None, (), ()), ("rows", (), ()), ([{}], [{}], [{}])]
)
def test_refuses_untyped_or_unordered_evidence(evaluator, inputs):
    with pytest.raises(evaluator.PairedEvaluationError):
        evaluator.paired_recall_difference(*inputs)


def test_contract_states_exact_procedure_and_incomplete_execution_freeze(evaluator):
    path = Path(__file__).resolve().parents[1] / "data/evaluation-contract-v1.json"
    contract = json.loads(path.read_text())
    paired = contract["paired_recall"]
    assert contract["status"] == "prospective_incomplete"
    assert contract["protected_evaluation_ready"] is False
    assert (
        contract["matrix_sha256"]
        == "aad6b7cf8d8416bfb37ec19503dbed031c15767ea96d9b76486ca3f3fedebe1b"
    )
    assert paired["replicates"] == evaluator.BOOTSTRAP_REPLICATES == 2000
    assert paired["seed"] == evaluator.BOOTSTRAP_SEED == 20260816
    assert paired["numpy_version"] == "2.2.6"
    assert paired["rng"] == "numpy.random.Generator(numpy.random.PCG64(seed))"
    assert (
        paired["draw_per_replicate"] == "rng.integers(0, D, size=D, dtype=numpy.int64)"
    )
    assert paired["quantile_levels"] == list(evaluator.QUANTILE_LEVELS)
    assert paired["quantile_method"] == evaluator.QUANTILE_METHOD == "linear"
    assert paired["domain_order"] == "ascending canonical ASCII registrable domain"
    assert paired["estimand"] == "URL-weighted recall(candidate) - recall(reference)"
    assert paired["resample_routing"] is False
    assert paired["empty_stratum"] == "not_estimable; estimate and bounds null"
    assert paired["single_domain"] == "not_estimable; retain estimate, bounds null"
    assert set(contract["remaining_freeze"]) == {
        "manifest_sampling",
        "http_measurement",
        "numerical_runtime",
        "secondary_analyses",
        "artifact_and_execution_bindings",
    }
