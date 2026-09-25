"""Shared primary receipt consistency checks without proving source provenance."""

from dataclasses import asdict
from hashlib import sha256

from . import fixed_cascade, phishvn
from ._external_input_records import _retained_positions
from ._external_primary_progress import primary_rows_bytes
from ._external_secondary_validation import (
    SecondaryInputError,
    binary,
    probability,
    require,
    validate_secondary_scoring,
)
from .evaluation_producer import _json_bytes
from .external_primary import ExternalPrimaryScores
from .primary_scores import PrimaryURLScores
from .selective_inference import InferenceCounts

__all__ = ["validate_primary_phase", "validate_secondary_scoring"]
_THRESHOLDS = (
    "length_only",
    "logistic_l1",
    "transformer",
    "half_width",
    "monitor_boundary",
)


def _counts(counts: InferenceCounts, expected: int) -> None:
    require(type(counts) is InferenceCounts)
    require(
        fixed_cascade._matches_exactly(
            asdict(counts), asdict(InferenceCounts(expected, expected, expected, 0))
        )
    )


def _thresholds(primary: ExternalPrimaryScores) -> dict[str, float]:
    values = primary.thresholds
    require(type(values) is tuple and len(values) == len(_THRESHOLDS))
    require(all(type(item) is tuple and len(item) == 2 for item in values))
    require(tuple(name for name, unused_value in values) == _THRESHOLDS)
    result = dict(values)
    for name in _THRESHOLDS[:3]:
        fixed_cascade._threshold(result[name], "primary_threshold")
    require(fixed_cascade._finite_number(result["half_width"], "half_width") >= 0)
    fixed_cascade._finite_number(result["monitor_boundary"], "monitor_boundary")
    return result


def _score(score: PrimaryURLScores) -> None:
    require(type(score) is PrimaryURLScores)
    for name in ("length", "stage1", "transformer", "cascade"):
        probability(getattr(score, f"{name}_probability"))
        binary(getattr(score, f"{name}_decision"))
    probability(score.monitor_probability)
    fixed_cascade._finite_number(
        score.negative_log_likelihood, "negative_log_likelihood"
    )
    require(type(score.band_selected) is bool)
    require(type(score.features) is tuple and len(score.features) == 25)
    for value in score.features:
        fixed_cascade._finite_number(value, "structural_feature")
    require(type(score.length_scoring_audit_json) is str)
    require(type(score.stage1_scoring_audit_json) is str)
    _counts(score.inference_counts, 1)


def _receipt(primary: ExternalPrimaryScores, thresholds: dict) -> bytes:
    retained = b"".join(phishvn._json_bytes(asdict(row)) for row in primary.records)
    return _json_bytes(
        {
            "schema_version": 1,
            "phase": "external_primary",
            "row_count": len(primary.records),
            "retained_test_sha256": sha256(retained).hexdigest(),
            "primary_scores_sha256": sha256(primary.checkpoint_bytes).hexdigest(),
            "thresholds": thresholds,
            "inference_counts": asdict(primary.inference_counts),
        }
    )


def _validate_primary_phase(primary: ExternalPrimaryScores) -> None:
    """Validate typed phase fields, canonical bytes and exact counts, not arithmetic."""
    require(type(primary) is ExternalPrimaryScores)
    require(type(primary.records) is tuple and type(primary.scores) is tuple)
    require(len(primary.records) == len(primary.scores))
    require(all(type(row) is phishvn.PreparedExternalRow for row in primary.records))
    final_position = primary.records[-1].file_position if primary.records else 0
    _retained_positions(primary.records, {"test": final_position})
    for score in primary.scores:
        _score(score)
    _counts(primary.inference_counts, len(primary.records))
    thresholds = _thresholds(primary)
    require(
        type(primary.checkpoint_bytes) is bytes and type(primary.receipt_bytes) is bytes
    )
    require(
        primary.checkpoint_bytes == primary_rows_bytes(primary.records, primary.scores)
    )
    require(primary.receipt_bytes == _receipt(primary, thresholds))


def validate_primary_phase(primary: ExternalPrimaryScores) -> None:
    """Validate primary consistency without replaying its numerical operations."""
    try:
        _validate_primary_phase(primary)
    except Exception:
        raise SecondaryInputError("invalid_external_secondary_evidence") from None
