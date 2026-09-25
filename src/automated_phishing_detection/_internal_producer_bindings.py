"""Freeze carried internal model identities and retained replay bytes before scoring."""

import base64
from hashlib import sha256

from . import bound_secondary
from ._checkpoint_codec import canonical_bytes, secondary_binding


def _replay_artifacts(models) -> dict[str, str]:
    artifacts = {
        "length-only.json": models.length_only._artifact_bytes,
        "logistic-l1.json": models.cascade.stage1_model._artifact_bytes,
        "gmm.json": models.gmm_artifact_bytes,
    }
    hashes = dict(models.artifact_hashes)
    if any(
        type(content) is not bytes or sha256(content).hexdigest() != hashes.get(name)
        for name, content in artifacts.items()
    ):
        raise ValueError("retained replay artifact differs from binding")
    return {
        name: base64.b64encode(content).decode("ascii")
        for name, content in artifacts.items()
    }


def _secondary(bound, models, stage1_threshold: float) -> dict:
    bound_secondary._validate_bound(bound)
    if any(type(member.seed) is not int for member in bound.seeds) or any(
        type(member.model.artifact_bytes) is not bytes
        or sha256(member.model.artifact_bytes).hexdigest() != member.artifact_sha256
        for member in bound.tabular
    ):
        raise ValueError("secondary artifact bytes differ from binding")
    value = secondary_binding(bound, stage1_threshold)
    artifacts = dict(models.artifact_hashes)
    if (
        "transformer-weights.npz" in artifacts
        and bound.seeds[0].weights_sha256 != artifacts["transformer-weights.npz"]
    ):
        raise ValueError("primary seed identity differs")
    if (
        "vocabulary.json" in artifacts
        and value["vocabulary_sha256"] != artifacts["vocabulary.json"]
    ):
        raise ValueError("primary vocabulary identity differs")
    return value


def binding_bytes(prepared, session, thresholds: dict) -> bytes:
    models = session.primary.models
    return canonical_bytes(
        {
            "schema_version": 3,
            "partition_sha256": prepared.partition_sha256,
            "source_csv_sha256": prepared.source_csv_sha256,
            "suffix_rules_sha256": prepared.suffix_rules_sha256,
            "artifact_hashes": dict(models.artifact_hashes),
            "thresholds": thresholds,
            "secondary": _secondary(
                session.secondary, models, thresholds["logistic_l1"]
            ),
            "gmm_audit": {
                "alert_count": models.audit_alert_count,
                "window_count": models.audit_window_count,
            },
            "replay_artifacts": _replay_artifacts(models),
        }
    )
