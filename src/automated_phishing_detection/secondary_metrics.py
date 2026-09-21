"""Secondary descriptions from aligned saved evidence, never model selection.

Score curves and probability descriptions use the saved routed probabilities.
Confusion metrics use independent saved decisions: a cascade can have different
component thresholds. Nominal McNemar/Holm results do not correct dependence
between URLs sharing a domain or routing history and never decide primary gates.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Real

import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import average_precision_score, roc_auc_score

from .paired_evaluation import BinaryPrediction, EvaluationRecord
from .saved_metrics import DetectionMetrics, SavedMetricsError, detection_metrics

ABLATION_FAMILY = (
    "internal_logistic_minus_length",
    "internal_cascade_minus_logistic",
    "external_gold_logistic_minus_length",
    "external_gold_cascade_minus_logistic",
)
NOMINAL_ROLE = "nominal_dependence_limited_secondary_not_primary"


class SecondaryMetricsError(ValueError):
    """Secondary evidence is invalid, misaligned or outside the frozen family."""


@dataclass(frozen=True)
class ScorePrediction:
    record_id: str
    probability: float


@dataclass(frozen=True)
class MetricEstimate:
    value: float | None
    reason: str | None = None


@dataclass(frozen=True)
class CalibrationBin:
    index: int
    count: int
    positive_count: int
    probability_sum: float
    squared_error_sum: float
    mean_probability: float | None
    positive_fraction: float | None


@dataclass(frozen=True)
class RecallAtFPR:
    recall: MetricEstimate
    threshold: float | None
    true_positives: int | None
    false_positives: int | None
    positive_count: int
    negative_count: int
    analysis_role: str = "descriptive_score_curve_not_deployment_threshold"


@dataclass(frozen=True)
class PrevalenceProjection:
    prevalence: float
    alerts: MetricEstimate
    misses: MetricEstimate
    false_alerts: MetricEstimate
    reference_requests: int = 10000
    assumption: str = "class_conditional_rates_transport_to_assumed_prevalence"


@dataclass(frozen=True)
class SecondaryMetrics:
    row_count: int
    domain_count: int
    counts: DetectionMetrics
    precision: MetricEstimate
    f2: MetricEstimate
    mcc: MetricEstimate
    balanced_accuracy: MetricEstimate
    average_precision: MetricEstimate
    roc_auc: MetricEstimate
    brier: MetricEstimate
    calibration_error: MetricEstimate
    calibration_bins: tuple[CalibrationBin, ...]
    recall_at_fpr: RecallAtFPR
    prevalence_projections: tuple[PrevalenceProjection, ...]
    analysis_role: str = "descriptive_secondary_not_primary"


@dataclass(frozen=True)
class McNemarResult:
    row_count: int
    domain_count: int
    positive_count: int
    negative_count: int
    both_correct: int
    candidate_only_correct: int
    reference_only_correct: int
    both_incorrect: int
    pvalue: MetricEstimate
    analysis_role: str = NOMINAL_ROLE


@dataclass(frozen=True)
class HolmCell:
    test_id: str
    raw_pvalue: MetricEstimate
    adjusted_pvalue: MetricEstimate


@dataclass(frozen=True)
class HolmFamily:
    cells: tuple[HolmCell, ...]
    complete: bool
    family_size: int = 4
    analysis_role: str = NOMINAL_ROLE


def _counts(
    records: Sequence[EvaluationRecord], decisions: Sequence[BinaryPrediction]
) -> DetectionMetrics:
    try:
        return detection_metrics(records, decisions)
    except SavedMetricsError as exc:
        raise SecondaryMetricsError(str(exc)) from exc


def _probabilities(
    records: Sequence[EvaluationRecord], scores: Sequence[ScorePrediction]
) -> np.ndarray:
    if not isinstance(scores, Sequence) or isinstance(scores, (str, bytes)):
        raise SecondaryMetricsError("scores must be ordered sequences")
    if len(records) != len(scores):
        raise SecondaryMetricsError("score and record counts must match")
    result = []
    for record, score in zip(records, scores, strict=True):
        if not isinstance(score, ScorePrediction):
            raise SecondaryMetricsError("evidence must use typed scores")
        if score.record_id != record.record_id:
            raise SecondaryMetricsError("score record IDs or order do not match")
        probability = score.probability
        if (
            isinstance(probability, (bool, np.bool_))
            or not isinstance(probability, Real)
            or not math.isfinite(probability)
            or not 0 <= probability <= 1
        ):
            raise SecondaryMetricsError("probability must be finite and within [0, 1]")
        result.append(float(probability))
    return np.asarray(result, dtype=np.float64)


def _ratio(numerator: float, denominator: float, reason: str) -> MetricEstimate:
    return (
        MetricEstimate(numerator / denominator)
        if denominator
        else MetricEstimate(None, reason)
    )


def _calibration(
    labels: np.ndarray, probabilities: np.ndarray
) -> tuple[MetricEstimate, MetricEstimate, tuple[CalibrationBin, ...]]:
    boundaries = np.arange(1, 10, dtype=np.float64) / 10
    assignments = np.searchsorted(boundaries, probabilities, side="right")
    bins = []
    for index in range(10):
        selected = assignments == index
        p = probabilities[selected]
        y = labels[selected]
        count = len(p)
        positives = int(y.sum())
        probability_sum = math.fsum(p)
        error_sum = math.fsum(
            (float(a) - int(b)) ** 2 for a, b in zip(p, y, strict=True)
        )
        bins.append(
            CalibrationBin(
                index,
                count,
                positives,
                probability_sum,
                error_sum,
                probability_sum / count if count else None,
                positives / count if count else None,
            )
        )
    count = len(labels)
    if not count:
        absent = MetricEstimate(None, "empty_population")
        return absent, absent, tuple(bins)
    brier = math.fsum(b.squared_error_sum for b in bins) / count
    ece = math.fsum(abs(b.positive_count - b.probability_sum) for b in bins) / count
    return MetricEstimate(brier), MetricEstimate(ece), tuple(bins)


def _recall_curve(labels: np.ndarray, probabilities: np.ndarray) -> RecallAtFPR:
    positives = int(labels.sum())
    negatives = len(labels) - positives
    if not positives or not negatives:
        return RecallAtFPR(
            MetricEstimate(None, "both_classes_required"),
            None,
            None,
            None,
            positives,
            negatives,
        )
    grouped: dict[float, list[int]] = {}
    for label, probability in zip(labels, probabilities, strict=True):
        counts = grouped.setdefault(float(probability), [0, 0])
        counts[int(label)] += 1
    threshold = float(np.nextafter(max(grouped), np.inf))
    best = (0, 0, threshold)
    tp = fp = 0
    for threshold in sorted(grouped, reverse=True):
        negative, positive = grouped[threshold]
        tp += positive
        fp += negative
        candidate = (tp, -fp, threshold)
        if 100 * fp <= negatives and candidate > best:
            best = candidate
    tp, negative_fp, threshold = best
    return RecallAtFPR(
        MetricEstimate(tp / positives),
        threshold,
        tp,
        -negative_fp,
        positives,
        negatives,
    )


def _projections(counts: DetectionMetrics) -> tuple[PrevalenceProjection, ...]:
    recall, fpr = counts.recall.estimate, counts.fpr.estimate
    projections = []
    for prevalence in (0.001, 0.01, 0.05):
        if recall is None or fpr is None:
            absent = MetricEstimate(None, "both_classes_required")
            projections.append(PrevalenceProjection(prevalence, absent, absent, absent))
        else:
            projections.append(
                PrevalenceProjection(
                    prevalence,
                    MetricEstimate(
                        10000 * (prevalence * recall + (1 - prevalence) * fpr)
                    ),
                    MetricEstimate(10000 * prevalence * (1 - recall)),
                    MetricEstimate(10000 * (1 - prevalence) * fpr),
                )
            )
    return tuple(projections)


def secondary_metrics(
    records: Sequence[EvaluationRecord],
    scores: Sequence[ScorePrediction],
    decisions: Sequence[BinaryPrediction],
) -> SecondaryMetrics:
    """Describe a declared labeled population without filtering or threshold fitting.

    Records must already be selected after full-stream routing. Scores are not
    assumed calibrated. Score-curve thresholds are descriptive and cannot be
    substituted for the separately supplied frozen binary decisions.
    """
    counts = _counts(records, decisions)
    probabilities = _probabilities(records, scores)
    labels = np.asarray([r.label for r in records], dtype=np.int64)
    tp, fn = counts.true_positives, counts.false_negatives
    fp, tn = counts.false_positives, counts.true_negatives
    absent = MetricEstimate(None, "both_classes_required")
    if tp + fn and fp + tn:
        ap = MetricEstimate(float(average_precision_score(labels, probabilities)))
        auc = MetricEstimate(float(roc_auc_score(labels, probabilities)))
        balanced = MetricEstimate((tp / (tp + fn) + tn / (fp + tn)) / 2)
    else:
        ap = auc = balanced = absent
    brier, ece, bins = _calibration(labels, probabilities)
    return SecondaryMetrics(
        row_count=len(records),
        domain_count=len({record.registrable_domain for record in records}),
        counts=counts,
        precision=_ratio(tp, tp + fp, "no_predicted_positives"),
        f2=_ratio(5 * tp, 5 * tp + 4 * fn + fp, "zero_f2_denominator"),
        mcc=_ratio(
            tp * tn - fp * fn,
            math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)),
            "zero_mcc_denominator",
        ),
        balanced_accuracy=balanced,
        average_precision=ap,
        roc_auc=auc,
        brier=brier,
        calibration_error=ece,
        calibration_bins=bins,
        recall_at_fpr=_recall_curve(labels, probabilities),
        prevalence_projections=_projections(counts),
    )


def exact_mcnemar(
    records: Sequence[EvaluationRecord],
    candidate: Sequence[BinaryPrediction],
    reference: Sequence[BinaryPrediction],
) -> McNemarResult:
    """Calculate nominal exact paired-error p-values, not cluster-valid inference."""
    counts = _counts(records, candidate)
    _counts(records, reference)
    table = {(True, True): 0, (True, False): 0, (False, True): 0, (False, False): 0}
    for record, left, right in zip(records, candidate, reference, strict=True):
        table[(left.decision == record.label, right.decision == record.label)] += 1
    both, b, c, neither = (
        table[key]
        for key in ((True, True), (True, False), (False, True), (False, False))
    )
    pvalue = _mcnemar_pvalue(len(records), b, c)
    return McNemarResult(
        len(records),
        len({r.registrable_domain for r in records}),
        counts.recall.denominator,
        counts.fpr.denominator,
        both,
        b,
        c,
        neither,
        pvalue,
    )


def _mcnemar_pvalue(rows: int, b: int, c: int) -> MetricEstimate:
    if not rows:
        return MetricEstimate(None, "empty_population")
    return MetricEstimate(
        float(binomtest(b, b + c, p=0.5, alternative="two-sided").pvalue)
        if b + c
        else 1.0
    )


def _validate_ablation_result(result: McNemarResult) -> None:
    counts = (
        result.row_count,
        result.domain_count,
        result.positive_count,
        result.negative_count,
        result.both_correct,
        result.candidate_only_correct,
        result.reference_only_correct,
        result.both_incorrect,
    )
    if any(type(count) is not int or count < 0 for count in counts):
        raise SecondaryMetricsError("McNemar counts must be nonnegative integers")
    if (
        sum(counts[4:]) != result.row_count
        or result.positive_count + result.negative_count != result.row_count
        or result.domain_count > result.row_count
        or bool(result.domain_count) != bool(result.row_count)
    ):
        raise SecondaryMetricsError(
            "McNemar support and paired counts are inconsistent"
        )
    if result.negative_count:
        raise SecondaryMetricsError("ablation family requires positive-only strata")
    if result.analysis_role != NOMINAL_ROLE:
        raise SecondaryMetricsError("McNemar evidence must remain nominal secondary")
    if not isinstance(result.pvalue, MetricEstimate) or (
        result.pvalue.value is not None and type(result.pvalue.value) is not float
    ):
        raise SecondaryMetricsError("McNemar pvalue must be a typed estimate")
    expected = _mcnemar_pvalue(
        result.row_count, result.candidate_only_correct, result.reference_only_correct
    )
    if result.pvalue != expected:
        raise SecondaryMetricsError("McNemar pvalue does not match paired counts")


def holm_ablation_family(
    evidence: Mapping[str, McNemarResult | None],
) -> HolmFamily:
    """Retain all four positive-stratum ablation slots, including unavailable ones.

    Missing p-values reserve their place in the family multiplicity but are never
    exposed as invented p-values or tested hypotheses. Holm cannot fix dependence
    within a constituent McNemar test.
    """
    if not isinstance(evidence, Mapping) or set(evidence) != set(ABLATION_FAMILY):
        raise SecondaryMetricsError("must supply the exact fixed four-cell family")
    raw = {}
    for name in ABLATION_FAMILY:
        result = evidence[name]
        if result is None:
            raw[name] = MetricEstimate(None, "missing_evidence")
        elif not isinstance(result, McNemarResult):
            raise SecondaryMetricsError(
                "family evidence must use typed McNemar results"
            )
        else:
            _validate_ablation_result(result)
            raw[name] = result.pvalue
    available = sorted(
        (name for name in ABLATION_FAMILY if raw[name].value is not None),
        key=lambda name: (raw[name].value, name),
    )
    adjusted = {}
    running = 0.0
    for index, name in enumerate(available):
        running = min(1.0, max(running, (4 - index) * raw[name].value))
        adjusted[name] = MetricEstimate(running)
    return HolmFamily(
        tuple(
            HolmCell(name, raw[name], adjusted.get(name, raw[name]))
            for name in ABLATION_FAMILY
        ),
        complete=len(available) == 4,
    )
