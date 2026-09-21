"""Exact rates from saved binary predictions, without model execution."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from . import protocol_preflight
from .baselines import clopper_pearson_upper
from .paired_evaluation import BinaryPrediction, EvaluationRecord


class SavedMetricsError(ValueError):
    """Saved evidence is invalid or cannot be aligned exactly."""


@dataclass(frozen=True)
class RateEstimate:
    """An observed proportion and its one-sided 95% exact binomial upper bound."""

    numerator: int
    denominator: int
    estimate: float | None
    upper_95: float | None
    status: str


@dataclass(frozen=True)
class DetectionMetrics:
    """Binary-label confusion counts with class-specific denominators."""

    true_positives: int
    false_negatives: int
    false_positives: int
    true_negatives: int
    recall: RateEstimate
    fpr: RateEstimate


def exact_rate(numerator: int, denominator: int) -> RateEstimate:
    """Keep an absent denominator distinct from an observed zero rate."""
    for name, value in (("numerator", numerator), ("denominator", denominator)):
        if type(value) is not int or value < 0:
            raise SavedMetricsError(f"{name} must be a nonnegative integer")
    if numerator > denominator:
        raise SavedMetricsError("numerator must not exceed denominator")
    if denominator == 0:
        return RateEstimate(numerator, denominator, None, None, "not_estimable")
    return RateEstimate(
        numerator,
        denominator,
        numerator / denominator,
        clopper_pearson_upper(numerator, denominator),
        "estimated",
    )


def _validate_sequences(*values: object) -> None:
    if not all(
        isinstance(value, Sequence) and not isinstance(value, (str, bytes))
        for value in values
    ):
        raise SavedMetricsError("evidence must be ordered sequences")


def _validate_identity(identity: object) -> None:
    if (
        not isinstance(identity, str)
        or not identity
        or any(
            character.isspace() or not character.isprintable() for character in identity
        )
    ):
        raise SavedMetricsError("record ID must be a nonempty stable string")


def _binary_integer(value: object, name: str) -> None:
    if type(value) is not int or value not in (0, 1):
        raise SavedMetricsError(f"{name} must be a binary integer")


def _validate_predictions(
    record_ids: Sequence[str], predictions: Sequence[BinaryPrediction]
) -> None:
    _validate_sequences(record_ids, predictions)
    if len(record_ids) != len(predictions):
        raise SavedMetricsError("prediction and record counts must match")
    seen = set()
    for identity, prediction in zip(record_ids, predictions, strict=True):
        _validate_identity(identity)
        if identity in seen:
            raise SavedMetricsError("record IDs must be unique")
        seen.add(identity)
        if not isinstance(prediction, BinaryPrediction):
            raise SavedMetricsError("evidence must use typed predictions")
        _validate_identity(prediction.record_id)
        if prediction.record_id != identity:
            raise SavedMetricsError("prediction record IDs or order do not match")
        _binary_integer(prediction.decision, "decision")


def detection_metrics(
    records: Sequence[EvaluationRecord], predictions: Sequence[BinaryPrediction]
) -> DetectionMetrics:
    """Count already-selected labeled rows without fitting, routing, or filtering.

    Domains must come from prepared metadata. As in paired evaluation, canonical
    domain syntax is checked here, not the provenance of the metadata.
    """
    _validate_sequences(records, predictions)
    record_ids = []
    for record in records:
        if not isinstance(record, EvaluationRecord):
            raise SavedMetricsError("evidence must use typed records")
        _binary_integer(record.label, "label")
        try:
            domain = protocol_preflight._ascii_domain(record.registrable_domain)
            protocol_preflight._reject_ip_literal(domain)
        except protocol_preflight.PreflightError as exc:
            raise SavedMetricsError(f"invalid registrable domain: {exc}") from exc
        if domain != record.registrable_domain:
            raise SavedMetricsError("registrable domain must already be canonical")
        record_ids.append(record.record_id)
    _validate_predictions(record_ids, predictions)
    counts = {(1, 1): 0, (1, 0): 0, (0, 1): 0, (0, 0): 0}
    for record, prediction in zip(records, predictions, strict=True):
        counts[(record.label, prediction.decision)] += 1
    tp, fn, fp, tn = (counts[key] for key in ((1, 1), (1, 0), (0, 1), (0, 0)))
    return DetectionMetrics(
        tp, fn, fp, tn, exact_rate(tp, tp + fn), exact_rate(fp, fp + tn)
    )


def control_alert_rate(
    record_ids: Sequence[str], predictions: Sequence[BinaryPrediction]
) -> RateEstimate:
    """Report alerts among unlabeled controls, never a false-positive rate."""
    _validate_predictions(record_ids, predictions)
    return exact_rate(
        sum(prediction.decision for prediction in predictions), len(record_ids)
    )
