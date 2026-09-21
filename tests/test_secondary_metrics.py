"""Synthetic known answers for secondary saved-evidence descriptions."""

from dataclasses import FrozenInstanceError, replace
from importlib import import_module

import numpy as np
import pytest

from automated_phishing_detection.paired_evaluation import (
    BinaryPrediction,
    EvaluationRecord,
)


def module():
    return import_module("automated_phishing_detection.secondary_metrics")


def evidence(labels, probabilities, decisions):
    rows = tuple(
        EvaluationRecord(f"row-{i}", f"d{i % 3}.test", label)
        for i, label in enumerate(labels)
    )
    scores = tuple(
        module().ScorePrediction(row.record_id, p)
        for row, p in zip(rows, probabilities, strict=True)
    )
    binary = tuple(
        BinaryPrediction(row.record_id, d)
        for row, d in zip(rows, decisions, strict=True)
    )
    return rows, scores, binary


def test_known_confusion_ranking_calibration_and_prevalence():
    result = module().secondary_metrics(
        *evidence([1, 1, 0, 0], [0.9, 0.4, 0.8, 0.1], [1, 0, 1, 0])
    )
    assert (result.row_count, result.domain_count) == (4, 3)
    assert result.precision.value == 0.5
    assert result.f2.value == 0.5
    assert result.mcc.value == 0
    assert result.balanced_accuracy.value == 0.5
    assert result.average_precision.value == pytest.approx(5 / 6)
    assert result.roc_auc.value == 0.75
    assert result.brier.value == pytest.approx(0.255)
    assert result.calibration_error.value == pytest.approx(0.4)
    assert len(result.calibration_bins) == 10
    assert sum(b.count for b in result.calibration_bins) == 4
    assert result.recall_at_fpr.recall.value == 0.5
    assert result.recall_at_fpr.threshold == 0.9
    assert result.recall_at_fpr.false_positives == 0
    assert result.recall_at_fpr.true_positives == 1
    assert result.recall_at_fpr.negative_count == 2
    assert result.recall_at_fpr.positive_count == 2
    assert [p.prevalence for p in result.prevalence_projections] == [0.001, 0.01, 0.05]
    projection = result.prevalence_projections[1]
    assert projection.alerts.value == 5000
    assert projection.misses.value == 50
    assert projection.false_alerts.value == 4950
    assert result.analysis_role == "descriptive_secondary_not_primary"
    with pytest.raises(FrozenInstanceError):
        result.row_count = 0


def test_decisions_are_not_reconstructed_from_score_thresholds():
    result = module().secondary_metrics(*evidence([1, 0], [0.1, 0.9], [1, 0]))
    assert result.precision.value == result.f2.value == result.mcc.value == 1
    assert result.roc_auc.value == 0
    assert result.recall_at_fpr.recall.value == 0
    assert result.recall_at_fpr.threshold == np.nextafter(0.9, np.inf)


def test_ece_equal_width_boundaries_include_zero_one_and_keep_empty_bins():
    result = module().secondary_metrics(
        *evidence([0, 1, 0, 1], [0.0, 0.1, 0.9, 1.0], [0, 0, 1, 1])
    )
    bins = result.calibration_bins
    assert [b.count for b in bins] == [1, 1, 0, 0, 0, 0, 0, 0, 0, 2]
    assert bins[2].mean_probability is None
    assert bins[2].positive_fraction is None
    assert bins[9].probability_sum == 1.9
    assert bins[9].positive_count == 1
    assert bins[9].squared_error_sum == pytest.approx(0.81)
    assert result.calibration_error.value == pytest.approx(0.45)


def test_ece_does_not_round_a_score_below_point_nine_into_the_last_bin():
    result = module().secondary_metrics(
        *evidence([1, 0], [np.nextafter(0.9, 0.0), 0.95], [1, 1])
    )
    assert result.calibration_bins[8].count == 1
    assert result.calibration_bins[9].count == 1
    assert result.calibration_error.value == pytest.approx(0.525)


@pytest.mark.parametrize("boundary_index", range(1, 10))
def test_ece_explicit_half_open_boundary_and_adjacent_float_values(boundary_index):
    boundary = np.float64(boundary_index) / 10
    result = module().secondary_metrics(
        *evidence(
            [0, 1, 0],
            [np.nextafter(boundary, 0.0), boundary, np.nextafter(boundary, 1.0)],
            [0, 1, 0],
        )
    )
    assert result.calibration_bins[boundary_index - 1].count == 1
    assert result.calibration_bins[boundary_index].count == 2


@pytest.mark.parametrize("negative_count, expected", [(99, 0.5), (100, 1.0)])
def test_recall_curve_uses_exact_integer_fpr_boundary(negative_count, expected):
    result = module().secondary_metrics(
        *evidence(
            [1, 1] + [0] * negative_count,
            [1.0, 0.5, 0.5] + [0.0] * (negative_count - 1),
            [1, 1] + [0] * negative_count,
        )
    )
    assert result.recall_at_fpr.recall.value == expected
    assert 100 * result.recall_at_fpr.false_positives <= negative_count


def test_tied_scores_are_not_split_or_interpolated():
    result = module().secondary_metrics(*evidence([1, 0], [0.5, 0.5], [1, 1]))
    assert result.average_precision.value == result.roc_auc.value == 0.5
    assert result.recall_at_fpr.recall.value == 0
    assert result.recall_at_fpr.threshold == np.nextafter(0.5, np.inf)


def test_recall_curve_tie_prefers_lower_fpr_then_higher_threshold():
    result = module().secondary_metrics(
        *evidence([1] + [0] * 100, [0.9, 0.8] + [0.1] * 99, [1] + [0] * 100)
    )
    assert result.recall_at_fpr.threshold == 0.9
    assert result.recall_at_fpr.false_positives == 0


@pytest.mark.parametrize("labels", [[], [0, 0], [1, 1]])
def test_empty_or_single_class_rankings_and_projections_are_explicit(labels):
    result = module().secondary_metrics(
        *evidence(labels, [0.2] * len(labels), [0] * len(labels))
    )
    for estimate in (
        result.average_precision,
        result.roc_auc,
        result.balanced_accuracy,
    ):
        assert estimate.value is None
        assert estimate.reason == "both_classes_required"
    assert result.recall_at_fpr.recall.reason == "both_classes_required"
    assert result.recall_at_fpr.threshold is None
    assert result.precision.reason == "no_predicted_positives"
    assert result.mcc.reason == "zero_mcc_denominator"
    assert result.prevalence_projections[0].alerts.reason == "both_classes_required"
    assert (result.brier.value is None) == (not labels)
    assert (result.calibration_error.value is None) == (not labels)
    if not labels:
        assert (
            result.brier.reason == result.calibration_error.reason == "empty_population"
        )
        assert result.f2.reason == "zero_f2_denominator"


@pytest.mark.parametrize(
    "bad", [float("nan"), float("inf"), -0.1, 1.01, True, "0.5", None]
)
def test_scores_must_be_finite_probabilities(bad):
    rows, scores, binary = evidence([1, 0], [0.1, 0.9], [1, 0])
    scores = (replace(scores[0], probability=bad), scores[1])
    with pytest.raises(module().SecondaryMetricsError, match="probability"):
        module().secondary_metrics(rows, scores, binary)


@pytest.mark.parametrize("target", ["scores", "decisions", "records"])
@pytest.mark.parametrize("change", ["missing", "extra", "reversed", "untyped"])
def test_rejects_misaligned_or_untyped_evidence(target, change):
    values = list(evidence([1, 0], [0.1, 0.9], [1, 0]))
    index = {"records": 0, "scores": 1, "decisions": 2}[target]
    original = values[index]
    values[index] = {
        "missing": original[:-1],
        "extra": (*original, original[0]),
        "reversed": original[::-1],
        "untyped": ({}, original[1]),
    }[change]
    with pytest.raises(module().SecondaryMetricsError):
        module().secondary_metrics(*values)


@pytest.mark.parametrize("bad", [None, {}, "", iter(())])
@pytest.mark.parametrize("index", [0, 1, 2])
def test_requires_materialized_ordered_sequences(bad, index):
    values = [(), (), ()]
    values[index] = bad
    with pytest.raises(module().SecondaryMetricsError):
        module().secondary_metrics(*values)


@pytest.mark.parametrize("field", ["label", "decision"])
@pytest.mark.parametrize("bad", [True, 1.0, 2, None])
def test_rejects_noninteger_binary_labels_and_decisions(field, bad):
    rows, scores, binary = evidence([1], [0.5], [1])
    if field == "label":
        rows = (replace(rows[0], label=bad),)
    else:
        binary = (replace(binary[0], decision=bad),)
    with pytest.raises(module().SecondaryMetricsError):
        module().secondary_metrics(rows, scores, binary)


def test_duplicate_identity_is_rejected_even_when_all_three_inputs_agree():
    rows, scores, binary = evidence([1], [0.5], [1])
    with pytest.raises(module().SecondaryMetricsError, match="unique"):
        module().secondary_metrics(rows * 2, scores * 2, binary * 2)


def mcnemar_evidence(b, c, both_correct=0, both_wrong=0):
    size = b + c + both_correct + both_wrong
    rows, _, candidate = evidence(
        [1] * size,
        [0.5] * size,
        [1] * b + [0] * c + [1] * both_correct + [0] * both_wrong,
    )
    reference = tuple(
        BinaryPrediction(row.record_id, decision)
        for row, decision in zip(
            rows, [0] * b + [1] * c + [1] * both_correct + [0] * both_wrong, strict=True
        )
    )
    return rows, candidate, reference


def test_exact_mcnemar_known_discordance_counts_and_two_sided_tail():
    result = module().exact_mcnemar(*mcnemar_evidence(3, 0, 2, 1))
    assert result.pvalue.value == 0.25
    assert result.candidate_only_correct == 3
    assert result.reference_only_correct == 0
    assert result.both_correct == 2
    assert result.both_incorrect == 1
    assert result.row_count == 6
    assert result.domain_count == 3
    assert result.analysis_role == "nominal_dependence_limited_secondary_not_primary"


def test_mcnemar_uses_correctness_for_mixed_labels_not_alert_status():
    rows, _, candidate = evidence([0, 1], [0.5, 0.5], [0, 1])
    reference = tuple(
        BinaryPrediction(row.record_id, 1 - p.decision)
        for row, p in zip(rows, candidate, strict=True)
    )
    result = module().exact_mcnemar(rows, candidate, reference)
    assert result.candidate_only_correct == 2
    assert result.reference_only_correct == 0
    assert result.pvalue.value == 0.5


def test_mcnemar_no_discordance_is_one_but_empty_is_unavailable():
    assert module().exact_mcnemar(*mcnemar_evidence(0, 0, 1)).pvalue.value == 1
    empty = module().exact_mcnemar(*mcnemar_evidence(0, 0))
    assert empty.pvalue.value is None
    assert empty.pvalue.reason == "empty_population"


def test_mcnemar_refuses_misaligned_predictions():
    rows, candidate, reference = mcnemar_evidence(1, 1)
    with pytest.raises(module().SecondaryMetricsError):
        module().exact_mcnemar(rows, candidate, reference[::-1])


def test_holm_fixed_four_cells_known_adjustment_and_stable_order():
    m = module()
    values = [m.exact_mcnemar(*mcnemar_evidence(b, 0)) for b in [3, 4, 5, 6]]
    family = dict(zip(m.ABLATION_FAMILY, values, strict=True))
    result = m.holm_ablation_family(family)
    assert result.complete
    assert result.family_size == 4
    assert [cell.test_id for cell in result.cells] == list(m.ABLATION_FAMILY)
    assert [cell.adjusted_pvalue.value for cell in result.cells] == [
        0.25,
        0.25,
        0.1875,
        0.125,
    ]
    assert result == m.holm_ablation_family(dict(reversed(list(family.items()))))


def test_holm_retains_missing_and_nonestimable_cells_without_pvalue_replacement():
    m = module()
    family = {name: None for name in m.ABLATION_FAMILY}
    family[m.ABLATION_FAMILY[0]] = m.exact_mcnemar(*mcnemar_evidence(6, 0))
    family[m.ABLATION_FAMILY[1]] = m.exact_mcnemar(*mcnemar_evidence(0, 0))
    result = m.holm_ablation_family(family)
    assert not result.complete
    assert result.family_size == 4
    assert result.cells[0].adjusted_pvalue.value == 0.125
    assert result.cells[1].raw_pvalue.reason == "empty_population"
    assert result.cells[1].adjusted_pvalue.value is None
    assert result.cells[2].raw_pvalue.reason == "missing_evidence"
    assert result.cells[2].adjusted_pvalue.value is None


@pytest.mark.parametrize("change", ["missing", "extra", "untyped"])
def test_holm_refuses_changed_family_or_untyped_results(change):
    m = module()
    family = {name: None for name in m.ABLATION_FAMILY}
    if change == "missing":
        family.pop(m.ABLATION_FAMILY[0])
    elif change == "extra":
        family["chosen_after_results"] = None
    else:
        family[m.ABLATION_FAMILY[0]] = 0.01
    with pytest.raises(m.SecondaryMetricsError):
        m.holm_ablation_family(family)


@pytest.mark.parametrize(
    "change",
    [
        {"row_count": 2},
        {"row_count": True},
        {"domain_count": 0},
        {"domain_count": 7},
        {"positive_count": -1},
        {"both_correct": 1},
        {"candidate_only_correct": 1.0},
        {"pvalue": "0.25"},
        {"analysis_role": "primary"},
    ],
)
def test_holm_rejects_internally_inconsistent_typed_summaries(change):
    m = module()
    result = replace(m.exact_mcnemar(*mcnemar_evidence(3, 0)), **change)
    family = {name: None for name in m.ABLATION_FAMILY}
    family[m.ABLATION_FAMILY[0]] = result
    with pytest.raises(m.SecondaryMetricsError):
        m.holm_ablation_family(family)


@pytest.mark.parametrize("bad", [0.01, float("nan"), 1.1, None, True])
def test_holm_rejects_pvalues_not_matching_discordance_counts(bad):
    m = module()
    result = replace(
        m.exact_mcnemar(*mcnemar_evidence(3, 0)),
        pvalue=m.MetricEstimate(bad),
    )
    family = {name: None for name in m.ABLATION_FAMILY}
    family[m.ABLATION_FAMILY[0]] = result
    with pytest.raises(m.SecondaryMetricsError):
        m.holm_ablation_family(family)


def test_holm_positive_stratum_family_rejects_mixed_label_mcnemar():
    m = module()
    rows, _, candidate = evidence([1, 0], [0.5, 0.5], [1, 0])
    result = m.exact_mcnemar(rows, candidate, candidate)
    family = {name: None for name in m.ABLATION_FAMILY}
    family[m.ABLATION_FAMILY[0]] = result
    with pytest.raises(m.SecondaryMetricsError, match="positive"):
        m.holm_ablation_family(family)


def test_holm_no_evidence_retains_four_unavailable_cells():
    m = module()
    result = m.holm_ablation_family({name: None for name in m.ABLATION_FAMILY})
    assert len(result.cells) == result.family_size == 4
    assert not result.complete
    assert all(
        cell.adjusted_pvalue.reason == "missing_evidence" for cell in result.cells
    )


def test_holm_caps_adjusted_pvalues_and_breaks_equal_pvalues_deterministically():
    m = module()
    result = m.exact_mcnemar(*mcnemar_evidence(1, 1))
    family = {name: result for name in m.ABLATION_FAMILY}
    adjusted = m.holm_ablation_family(family)
    assert all(cell.adjusted_pvalue.value == 1 for cell in adjusted.cells)
    assert adjusted == m.holm_ablation_family(dict(reversed(list(family.items()))))


def test_perfect_inverse_decisions_have_negative_mcc_and_maximum_brier():
    result = module().secondary_metrics(*evidence([0, 1], [1.0, 0.0], [1, 0]))
    assert result.mcc.value == -1
    assert result.brier.value == result.calibration_error.value == 1
    assert result.balanced_accuracy.value == result.f2.value == 0


def test_real_numpy_probabilities_are_accepted_without_inferring_labels():
    result = module().secondary_metrics(
        *evidence([0, 1], [np.float32(0), np.float64(1)], [0, 1])
    )
    assert result.average_precision.value == result.roc_auc.value == 1
    assert result.brier.value == result.calibration_error.value == 0
