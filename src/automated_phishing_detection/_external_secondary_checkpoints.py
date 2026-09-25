"""Canonical completed-column identities and receipts from retained bound state."""

from dataclasses import asdict
from hashlib import sha256

from ._external_secondary_validation import SEEDS, TABULAR_NAMES
from .bound_secondary import (
    BoundSecondary,
    CompletedSeedColumn,
    CompletedTabularColumn,
    SecondaryScoring,
)
from .evaluation_producer import _json_bytes

CompletedColumn = CompletedTabularColumn | CompletedSeedColumn
SECONDARY_CHECKPOINTS = (
    *(f"secondary-tabular-{name}.json" for name in TABULAR_NAMES),
    *(f"secondary-seed-{seed}.json" for seed in SEEDS),
)


def member_bindings(bound: BoundSecondary) -> tuple[dict, ...]:
    reports = dict(bound.report_hashes)
    tabular = tuple(
        {
            "kind": "tabular",
            "name": member.name,
            "artifact_sha256": member.artifact_sha256,
            "threshold": member.threshold,
            "accepted_report_sha256": reports["tabular"],
        }
        for member in bound.tabular
    )
    seeds = tuple(
        {
            "kind": "seed",
            "seed": member.seed,
            "weights_sha256": member.weights_sha256,
            "transformer_threshold": member.transformer_threshold,
            "half_width": member.half_width,
            "stage1_threshold": bound.stage1_threshold,
            "reuses_primary": member.reuses_primary,
            "vocabulary_sha256": sha256(bound.vocabulary_bytes).hexdigest(),
            "device_type": bound.device_type,
            "accepted_report_sha256": reports["seeds"],
        }
        for member in bound.seeds
    )
    return (*tabular, *seeds)


def column_bytes(
    primary_hash: str,
    record_ids: tuple[str, ...],
    binding: dict,
    column: CompletedColumn,
) -> bytes:
    return _json_bytes(
        {
            "schema_version": 1,
            "primary_scores_sha256": primary_hash,
            "record_ids": record_ids,
            "binding": binding,
            "column": asdict(column),
        }
    )


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
