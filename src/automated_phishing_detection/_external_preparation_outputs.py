"""Retain supplied preparation and loaded bindings before external inference.

These are consistency checks on caller-supplied objects. The official wrapper
must authenticate the publisher, accepted artifacts and execution reservation.
"""

import base64
from dataclasses import asdict
from hashlib import sha256

from . import evaluation_producer, external_monitors
from ._external_inputs import validate_prepared_external
from .bound_drift import BoundDrift
from .bound_external_runtime import BoundExternalSession
from .external_primary import _validate_session
from .external_secondary import _validate_binding
from .phishvn import PreparedExternal
from .retained_drift import RetainedDriftReference
from .retained_external_drift import restore_external_drift

_PUBLIC_INPUTS = (
    (
        "reports/secondary-development-correction-v2-summary.json",
        "drift-accepted-report.json",
    ),
    ("reports/phiusiil-preparation-summary.json", "drift-preparation-summary.json"),
    ("data/sources.json", "drift-source-spec.json"),
)


def _require(condition: bool) -> None:
    if not condition:
        raise ValueError("invalid_external_preparation_binding")


def _drift_buffers(drift: BoundDrift) -> dict[str, bytes]:
    _require(
        type(drift) is BoundDrift and type(drift.reference) is RetainedDriftReference
    )
    inputs = drift.public_inputs
    _require(type(inputs) is tuple and len(inputs) == 3)
    _require(all(type(item) is tuple and len(item) == 2 for item in inputs))
    _require(
        tuple(name for name, unused in inputs)
        == tuple(name for name, unused in _PUBLIC_INPUTS)
    )
    outputs = {
        target: inputs[index][1]
        for index, (unused, target) in enumerate(_PUBLIC_INPUTS)
    }
    outputs.update(
        {
            "training-reference.json": drift.training_reference_bytes,
            "validation-audit.json": drift.validation_audit_bytes,
        }
    )
    _require(all(type(content) is bytes for content in outputs.values()))
    return outputs


def _drift_outputs(drift: BoundDrift, report_hash: str) -> dict[str, bytes]:
    outputs = _drift_buffers(drift)
    expected = {
        "training-reference.json": drift.reference.training_reference_sha256,
        "validation-audit.json": drift.reference.validation_audit_sha256,
        "drift-preparation-summary.json": drift.reference.preparation_summary_sha256,
        "drift-accepted-report.json": report_hash,
    }
    _require(
        all(
            sha256(outputs[name]).hexdigest() == digest
            for name, digest in expected.items()
        )
    )
    external_monitors.replay_external_monitors((), drift.reference)
    restored = restore_external_drift(
        drift.training_reference_bytes,
        drift.validation_audit_bytes,
        drift.public_inputs,
    )
    _require(restored.reference == drift.reference)
    return outputs


def _replay_artifacts(session: BoundExternalSession) -> dict[str, str]:
    models = session.evaluation.primary.models
    contents = {
        "length-only.json": models.length_only._artifact_bytes,
        "logistic-l1.json": models.cascade.stage1_model._artifact_bytes,
        "gmm.json": models.gmm_artifact_bytes,
    }
    hashes = dict(models.artifact_hashes)
    _require(
        all(
            type(value) is bytes and sha256(value).hexdigest() == hashes.get(name)
            for name, value in contents.items()
        )
    )
    return {
        name: base64.b64encode(value).decode("ascii")
        for name, value in contents.items()
    }


def _audit(session: BoundExternalSession) -> dict[str, int]:
    models = session.evaluation.primary.models
    audit = {
        "alert_count": models.audit_alert_count,
        "window_count": models.audit_window_count,
    }
    _require(
        tuple(audit.values()) == (28, 252)
        and all(type(value) is int for value in audit.values())
    )
    return audit


def _bindings(
    prepared: PreparedExternal, session: BoundExternalSession, drift: dict
) -> bytes:
    models = session.evaluation.primary.models
    thresholds = evaluation_producer._thresholds(session.evaluation.primary)
    return evaluation_producer._json_bytes(
        {
            "schema_version": 1,
            "source_binding": "caller_supplied_preparation_only",
            "preparation": prepared.public_summary,
            "artifact_hashes": dict(models.artifact_hashes),
            "thresholds": thresholds,
            "secondary": evaluation_producer._secondary_binding(
                session.evaluation.secondary, thresholds["logistic_l1"]
            ),
            "gmm_audit": _audit(session),
            "replay_artifacts": _replay_artifacts(session),
            "drift": {
                "pins": asdict(session.drift.reference.pins),
                "portable_state_sha256": session.drift.reference.portable_state_sha256,
                "private_sha256": {
                    name: sha256(content).hexdigest() for name, content in drift.items()
                },
            },
        }
    )


def preparation_outputs(
    prepared: PreparedExternal, session: BoundExternalSession
) -> dict[str, bytes]:
    validate_prepared_external(prepared)
    _validate_session(session)
    thresholds = evaluation_producer._thresholds(session.evaluation.primary)
    _validate_binding(session.evaluation.secondary, thresholds["logistic_l1"])
    drift = _drift_outputs(
        session.drift, dict(session.evaluation.secondary.report_hashes)["tabular"]
    )
    outputs = {
        name: prepared.private_outputs[name]
        for name in ("retained-test.jsonl", "quarantine.jsonl", "inventory.json")
    }
    outputs["preparation-summary.json"] = evaluation_producer._json_bytes(
        prepared.public_summary
    )
    outputs["bindings.json"] = _bindings(prepared, session, drift)
    outputs.update(drift)
    return outputs
