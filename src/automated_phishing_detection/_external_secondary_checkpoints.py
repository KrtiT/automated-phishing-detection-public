"""Canonical completed-column identities and receipts from retained bound state."""

from dataclasses import asdict
from hashlib import sha256

from ._checkpoint_codec import column_bytes as column_bytes
from ._checkpoint_codec import project_member_bindings as project_member_bindings
from ._external_secondary_validation import SEEDS, TABULAR_NAMES
from .bound_secondary import (
    BoundSecondary,
    CompletedSeedColumn,
    CompletedTabularColumn,
    SecondaryScoring,
)
from .evaluation_producer import _json_bytes, _secondary_binding

CompletedColumn = CompletedTabularColumn | CompletedSeedColumn
SECONDARY_CHECKPOINTS = (
    *(f"secondary-tabular-{name}.json" for name in TABULAR_NAMES),
    *(f"secondary-seed-{seed}.json" for seed in SEEDS),
)


def member_bindings(bound: BoundSecondary) -> tuple[dict, ...]:
    return project_member_bindings(_secondary_binding(bound, bound.stage1_threshold))


def completion_bytes(
    primary_hash: str, scoring: SecondaryScoring, private_outputs: dict[str, bytes]
) -> bytes:
    return _json_bytes(
        {
            "schema_version": 1,
            "phase": "external_secondary",
            "row_count": len(scoring.rows),
            "primary_scores_sha256": primary_hash,
            "secondary_scores_sha256": sha256(_json_bytes(asdict(scoring))).hexdigest(),
            "checkpoint_sha256": {
                name: sha256(private_outputs[name]).hexdigest()
                for name in SECONDARY_CHECKPOINTS
            },
            "inference_counts": asdict(scoring.counts),
        }
    )
