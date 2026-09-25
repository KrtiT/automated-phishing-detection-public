"""Reservation-agnostic internal score checkpoints over supplied immutable values."""

from dataclasses import asdict
from hashlib import sha256

from . import bound_secondary, fixed_cascade
from ._checkpoint_codec import canonical_bytes, column_bytes, project_member_bindings
from ._external_secondary_validation import validate_column, validate_secondary_scoring

COLUMN_NAMES = (
    *(f"secondary-tabular-{name}.json" for name in bound_secondary._TABULAR_NAMES),
    *(f"secondary-seed-{seed}.json" for seed in bound_secondary._SEEDS),
)
PRODUCER_CHECKPOINT_NAMES = (
    "bindings.json",
    "manifests.json",
    "primary-scores.jsonl",
    "primary-completion.json",
    *COLUMN_NAMES,
    "predictions.jsonl",
    "routing.json",
)


def primary_completion_bytes(
    bindings_bytes: bytes,
    primary_bytes: bytes,
    row_count: int,
    inference_counts,
    partition_sha256: str,
) -> bytes:
    return canonical_bytes(
        {
            "schema_version": 1,
            "phase": "internal_primary",
            "row_count": row_count,
            "partition_sha256": partition_sha256,
            "bindings_sha256": sha256(bindings_bytes).hexdigest(),
            "primary_scores_sha256": sha256(primary_bytes).hexdigest(),
            "inference_counts": asdict(inference_counts),
        }
    )


def validated_column_index(state, column) -> int:
    index = len(state.columns)
    validate_column(column, index, len(state.record_ids))
    return index


def completed_column_bytes(
    state, column, secondary: dict, index: int
) -> tuple[str, bytes]:
    members = project_member_bindings(secondary)
    content = column_bytes(
        sha256(state.outputs["primary-scores.jsonl"]).hexdigest(),
        state.record_ids,
        members[index],
        column,
    )
    return COLUMN_NAMES[index], content


def validate_completed_scoring(state, scoring) -> None:
    if len(state.columns) != len(COLUMN_NAMES):
        raise ValueError("incomplete_internal_secondary_columns")
    validate_secondary_scoring(scoring, len(state.rows))
    expected = bound_secondary._scoring_result(state.columns, len(state.rows))
    if not fixed_cascade._matches_exactly(asdict(scoring), asdict(expected)):
        raise ValueError("internal_secondary_columns_differ")
