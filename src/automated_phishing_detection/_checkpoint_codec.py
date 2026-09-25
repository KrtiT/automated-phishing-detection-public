"""Dependency-neutral canonical checkpoint bytes and secondary binding projections."""

import json
from dataclasses import asdict
from hashlib import sha256

from .bound_secondary import BoundSecondary, CompletedSeedColumn, CompletedTabularColumn


def canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("ascii")


def _tabular_binding(member) -> dict:
    return {
        "name": member.name,
        "artifact_sha256": member.artifact_sha256,
        "threshold": member.threshold,
    }


def _seed_binding(member) -> dict:
    return {
        "seed": member.seed,
        "weights_sha256": member.weights_sha256,
        "transformer_threshold": member.transformer_threshold,
        "half_width": member.half_width,
        "reuses_primary": member.reuses_primary,
    }


def secondary_binding(bound: BoundSecondary, stage1_threshold: float) -> dict:
    if type(bound) is not BoundSecondary or bound.stage1_threshold != stage1_threshold:
        raise ValueError("secondary binding differs from primary")
    return {
        "accepted_report_sha256": dict(bound.report_hashes),
        "device_type": bound.device_type,
        "stage1_threshold": bound.stage1_threshold,
        "vocabulary_sha256": sha256(bound.vocabulary_bytes).hexdigest(),
        "tabular": [_tabular_binding(member) for member in bound.tabular],
        "seeds": [_seed_binding(member) for member in bound.seeds],
    }


def _seed_checkpoint_binding(member: dict, secondary: dict) -> dict:
    return {
        "kind": "seed",
        "seed": member["seed"],
        "weights_sha256": member["weights_sha256"],
        "transformer_threshold": member["transformer_threshold"],
        "half_width": member["half_width"],
        "reuses_primary": member["reuses_primary"],
        "stage1_threshold": secondary["stage1_threshold"],
        "vocabulary_sha256": secondary["vocabulary_sha256"],
        "device_type": secondary["device_type"],
        "accepted_report_sha256": secondary["accepted_report_sha256"]["seeds"],
    }


def project_member_bindings(secondary: dict) -> tuple[dict, ...]:
    reports = secondary["accepted_report_sha256"]
    tabular = tuple(
        {
            "kind": "tabular",
            "name": member["name"],
            "artifact_sha256": member["artifact_sha256"],
            "threshold": member["threshold"],
            "accepted_report_sha256": reports["tabular"],
        }
        for member in secondary["tabular"]
    )
    seeds = tuple(
        _seed_checkpoint_binding(member, secondary) for member in secondary["seeds"]
    )
    return (*tabular, *seeds)


def column_bytes(
    primary_hash: str,
    record_ids: tuple[str, ...],
    binding: dict,
    column: CompletedTabularColumn | CompletedSeedColumn,
) -> bytes:
    return canonical_bytes(
        {
            "schema_version": 1,
            "primary_scores_sha256": primary_hash,
            "record_ids": record_ids,
            "binding": binding,
            "column": asdict(column),
        }
    )
