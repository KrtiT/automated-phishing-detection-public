"""Paired recall differences from saved predictions, without model execution."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from . import protocol_preflight

BOOTSTRAP_SEED = 20260816
BOOTSTRAP_REPLICATES = 2000
QUANTILE_LEVELS = (0.025, 0.975)
QUANTILE_METHOD = "linear"


class PairedEvaluationError(ValueError):
    """Prediction evidence is invalid or cannot be paired exactly."""


@dataclass(frozen=True)
class EvaluationRecord:
    """Metadata from a prepared stratum; domain extraction occurs upstream."""

    record_id: str
    registrable_domain: str
    label: int


@dataclass(frozen=True)
class BinaryPrediction:
    record_id: str
    decision: int


@dataclass(frozen=True)
class RecallDifference:
    """Candidate minus reference recall; bounds use unrounded probabilities."""

    status: str
    reason: str | None
    positive_count: int
    domain_count: int
    candidate_true_positives: int
    reference_true_positives: int
    estimate: float | None
    lower: float | None
    upper: float | None
    bootstrap_replicates: int


def _binary_integer(value: object, name: str) -> None:
    if type(value) is not int or value not in (0, 1):
        raise PairedEvaluationError(f"{name} must be a binary integer")


def _validate_pairs(
    records: Sequence[EvaluationRecord],
    candidate: Sequence[BinaryPrediction],
    reference: Sequence[BinaryPrediction],
) -> None:
    if not all(
        isinstance(value, Sequence) and not isinstance(value, (str, bytes))
        for value in (records, candidate, reference)
    ):
        raise PairedEvaluationError("evidence must be ordered sequences")
    if len(records) != len(candidate) or len(records) != len(reference):
        raise PairedEvaluationError("prediction and record counts must match")
    seen = set()
    for record, left, right in zip(records, candidate, reference, strict=True):
        if not isinstance(record, EvaluationRecord) or not all(
            isinstance(value, BinaryPrediction) for value in (left, right)
        ):
            raise PairedEvaluationError(
                "evidence must use typed records and predictions"
            )
        identity = record.record_id
        if (
            not isinstance(identity, str)
            or not identity
            or any(
                character.isspace() or not character.isprintable()
                for character in identity
            )
        ):
            raise PairedEvaluationError("record ID must be a nonempty stable string")
        if identity in seen:
            raise PairedEvaluationError("record IDs must be unique")
        seen.add(identity)
        if left.record_id != identity or right.record_id != identity:
            raise PairedEvaluationError("prediction record IDs or order do not match")
        _binary_integer(record.label, "label")
        _binary_integer(left.decision, "candidate decision")
        _binary_integer(right.decision, "reference decision")
        if record.label != 1:
            raise PairedEvaluationError("recall evidence must be a positive stratum")
        try:
            domain = protocol_preflight._ascii_domain(record.registrable_domain)
            protocol_preflight._reject_ip_literal(domain)
        except protocol_preflight.PreflightError as exc:
            raise PairedEvaluationError(f"invalid registrable domain: {exc}") from exc
        if domain != record.registrable_domain:
            raise PairedEvaluationError("registrable domain must already be canonical")


def _bootstrap_differences(
    cluster_differences: np.ndarray, cluster_sizes: np.ndarray
) -> np.ndarray:
    rng = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
    domain_count = len(cluster_sizes)
    distribution = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    for index in range(BOOTSTRAP_REPLICATES):
        draw = rng.integers(0, domain_count, size=domain_count, dtype=np.int64)
        # Repeated domains repeat all their paired rows, including the denominator.
        distribution[index] = (
            cluster_differences[draw].sum() / cluster_sizes[draw].sum()
        )
    return distribution


def paired_recall_difference(
    records: Sequence[EvaluationRecord],
    candidate: Sequence[BinaryPrediction],
    reference: Sequence[BinaryPrediction],
) -> RecallDifference:
    """Compare an already selected positive stratum of immutable predictions.

    For policy evidence, route the full original stream before selecting this
    stratum. Bootstrap saved outcomes only; do not reroute resampled rows.
    Domains must come from the prepared, pinned-PSL metadata. This function
    checks canonical domain syntax, not the provenance of that metadata.
    """
    _validate_pairs(records, candidate, reference)
    counts: dict[str, list[int]] = {}
    candidate_tp = reference_tp = 0
    for record, left, right in zip(records, candidate, reference, strict=True):
        cluster = counts.setdefault(record.registrable_domain, [0, 0])
        cluster[0] += left.decision - right.decision
        cluster[1] += 1
        candidate_tp += left.decision
        reference_tp += right.decision

    row_count = len(records)
    domain_count = len(counts)
    estimate = (candidate_tp - reference_tp) / row_count if row_count else None
    if domain_count < 2:
        return RecallDifference(
            status="not_estimable",
            reason="no_positive_rows"
            if not row_count
            else "insufficient_domain_clusters",
            positive_count=row_count,
            domain_count=domain_count,
            candidate_true_positives=candidate_tp,
            reference_true_positives=reference_tp,
            estimate=estimate,
            lower=None,
            upper=None,
            bootstrap_replicates=0,
        )

    ordered = np.asarray([counts[domain] for domain in sorted(counts)], dtype=np.int64)
    distribution = _bootstrap_differences(ordered[:, 0], ordered[:, 1])
    lower, upper = np.quantile(distribution, QUANTILE_LEVELS, method=QUANTILE_METHOD)
    return RecallDifference(
        status="estimated",
        reason=None,
        positive_count=row_count,
        domain_count=domain_count,
        candidate_true_positives=candidate_tp,
        reference_true_positives=reference_tp,
        estimate=estimate,
        lower=float(lower),
        upper=float(upper),
        bootstrap_replicates=BOOTSTRAP_REPLICATES,
    )
