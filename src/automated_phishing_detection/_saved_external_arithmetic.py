"""Verify retained portable arithmetic without source or inference-model access."""

from hashlib import sha256

from . import saved_evidence, secondary_development
from .retained_drift import RetainedDriftReference


def verify_retained_arithmetic(
    rows: tuple[dict, ...], bindings: dict, reference: RetainedDriftReference
) -> None:
    artifacts = bindings["artifact_hashes"]
    saved_evidence._require(
        reference.pins.logistic_l1_artifact_sha256 == artifacts["logistic-l1.json"]
        and reference.pins.gmm_artifact_sha256 == artifacts["gmm.json"],
        "saved_external_drift_artifacts_differ",
    )
    length, stage1, gmm = saved_evidence._replay_models(bindings)
    scaler = secondary_development._accepted_states(
        stage1, gmm, reference.pins, reference.psi.training_row_count
    )
    portable = secondary_development._json_bytes(
        secondary_development._portable_snapshot(stage1)
    )
    saved_evidence._require(
        scaler == (reference.scaler_mean, reference.scaler_scale)
        and sha256(portable).hexdigest() == reference.portable_state_sha256,
        "saved_external_drift_state_differs",
    )
    saved_evidence._verify_loaded_monitor_path(rows, length, stage1, gmm)
