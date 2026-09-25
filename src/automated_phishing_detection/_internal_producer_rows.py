"""Validate immutable completed primary values before retaining a scientific prefix."""

import json
from dataclasses import asdict

from . import fixed_cascade
from ._checkpoint_codec import canonical_bytes
from ._saved_score_validation import _features, _primary
from .selective_inference import InferenceCounts


def _audit(value: object) -> None:
    if type(value) is not str:
        raise ValueError("invalid_internal_primary_audit")
    parsed = json.loads(
        value,
        object_pairs_hook=fixed_cascade._object_without_duplicate_keys,
        parse_constant=fixed_cascade._reject_json_constant,
    )
    if canonical_bytes(parsed).decode("ascii").rstrip("\n") != value:
        raise ValueError("invalid_internal_primary_audit")


def validate_primary_values(row, thresholds: dict) -> None:
    if (
        type(row.features) is not tuple
        or type(row.inference_counts) is not InferenceCounts
        or type(row.secondary_tabular) is not tuple
        or type(row.secondary_seeds) is not tuple
        or row.secondary_tabular
        or row.secondary_seeds
    ):
        raise ValueError("invalid_internal_primary_values")
    if not fixed_cascade._matches_exactly(
        asdict(row.inference_counts), asdict(InferenceCounts(1, 1, 1, 0))
    ):
        raise ValueError("invalid_internal_primary_counts")
    values = asdict(row)
    values["features"] = list(row.features)
    _features(values)
    _primary(values, thresholds)
    _audit(row.length_scoring_audit_json)
    _audit(row.stage1_scoring_audit_json)
