"""Byte-valid invented reconstruction fixtures exercise real retained arithmetic."""

import importlib
import importlib.util
import json
from dataclasses import asdict
from hashlib import sha256
from types import ModuleType

import pytest

from automated_phishing_detection import saved_evidence, secondary_development


def fixture_module() -> ModuleType:
    name = "saved_external_fixtures"
    assert importlib.util.find_spec(name), "missing byte-valid external fixture"
    return importlib.import_module(name)


@pytest.mark.parametrize("count", [0, 1, 5, 255, 256, 319, 320])
def test_bundle_retains_every_completed_output_and_nullable_controls(
    monkeypatch: pytest.MonkeyPatch, count: int
) -> None:
    bundle = fixture_module().saved_external_bundle(monkeypatch, count)
    assert len(bundle.produced.private_outputs) == 30
    assert len(bundle.produced.replay.rows) == count
    assert all(
        row.record.is_phishing is None
        for row in bundle.produced.replay.rows
        if row.record.role == "tranco"
    )
    assert bundle.produced.public_summary["protected_evaluation_authorized"] is False


def test_retained_artifacts_run_existing_arithmetic_and_crossbind_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = fixture_module().saved_external_bundle(monkeypatch)
    outputs = bundle.produced.private_outputs
    bindings = json.loads(outputs["bindings.json"])
    rows = tuple(
        {"record": asdict(row.record), **asdict(row.primary)}
        for row in bundle.produced.replay.rows
    )
    saved_evidence._verify_monitor_path(rows, bindings)
    unused_length, logistic, gmm = saved_evidence._replay_models(bindings)
    reference = bundle.session.drift.reference
    scaler = secondary_development._accepted_states(
        logistic, gmm, reference.pins, reference.psi.training_row_count
    )
    snapshot = secondary_development._portable_snapshot(logistic)
    digest = sha256(secondary_development._json_bytes(snapshot)).hexdigest()
    assert scaler == (reference.scaler_mean, reference.scaler_scale)
    assert digest == reference.portable_state_sha256


def test_fixture_does_not_patch_primary_loaders_or_scoring_arithmetic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from automated_phishing_detection import (
        fixed_cascade,
        gmm_monitor,
        length_inference,
    )

    functions = (
        (fixed_cascade, "_load_logistic_l1_artifact_bytes"),
        (fixed_cascade, "score_logistic_l1_authoritative"),
        (length_inference, "_load_length_only_artifact_bytes"),
        (length_inference, "score_length_only_authoritative"),
        (gmm_monitor, "load_gmm_artifact_bytes"),
        (gmm_monitor, "score_feature_matrix"),
    )
    originals = tuple(getattr(module, name) for module, name in functions)
    fixture_module().saved_external_bundle(monkeypatch, 1)
    assert tuple(getattr(module, name) for module, name in functions) == originals


def test_fixture_does_not_read_source_or_model_files_or_fit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import builtins
    from pathlib import Path

    from sklearn.linear_model import LogisticRegression

    from automated_phishing_detection import gmm_monitor, secondary_drift

    module = fixture_module()

    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail("fixture attempted file access or fitting")

    for target, name in (
        (builtins, "open"),
        (Path, "open"),
        (Path, "read_bytes"),
        (Path, "read_text"),
        (LogisticRegression, "fit"),
        (gmm_monitor.GaussianMixture, "fit"),
        (gmm_monitor.StandardScaler, "fit"),
        (secondary_drift, "fit_mmd_reference"),
        (secondary_drift, "fit_psi_reference"),
    ):
        monkeypatch.setattr(target, name, forbidden)
    assert len(module.saved_external_bundle(monkeypatch).produced.replay.rows) == 5
