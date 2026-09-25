"""Real retained-snapshot restoration behind isolated public/model boundaries."""

import json
from hashlib import sha256
from types import SimpleNamespace

import pytest
from test_retained_drift import _rebind, _snapshots

from automated_phishing_detection import bound_drift, retained_drift


def _retained_fixture() -> tuple[dict, dict]:
    data = _snapshots()
    reference = json.loads(data["training_reference"])
    audit = json.loads(data["validation_audit"])
    snapshot = {"fixture": "portable model state"}
    reference["portable_state_sha256"] = sha256(
        bound_drift.secondary_development._json_bytes(snapshot)
    ).hexdigest()
    return (
        _rebind(
            reference,
            audit,
            data["expected_drift_summary"],
            data["preparation_summary"],
            data["pins"],
        ),
        snapshot,
    )


def _isolate_public_boundaries(monkeypatch, data: dict) -> None:
    contents = {
        bound_drift._REPORT: b"{}",
        bound_drift._PREPARATION: data["preparation_summary"],
        bound_drift._SOURCE: b"source fixture",
    }
    monkeypatch.setattr(bound_drift, "_public_inputs", lambda _: contents)
    monkeypatch.setattr(bound_drift, "_pins", lambda *args: data["pins"])
    monkeypatch.setattr(bound_drift, "_preparation", lambda *args: 256)
    monkeypatch.setattr(
        bound_drift.seed_probe_execution,
        "_retained_drift",
        lambda *args: (
            bound_drift.seed_probe_execution._canonical(data["expected_drift_summary"]),
            data["expected_reference_sha256"],
            data["expected_audit_sha256"],
        ),
    )


@pytest.fixture
def restore_fixture(tmp_path, monkeypatch):
    data, snapshot = _retained_fixture()
    _isolate_public_boundaries(monkeypatch, data)
    monkeypatch.setattr(
        bound_drift.secondary_development,
        "_accepted_states",
        lambda *args: ((0.0,) * 26, (1.0,) * 26),
    )
    monkeypatch.setattr(
        bound_drift.secondary_development, "_portable_snapshot", lambda _: snapshot
    )
    paths = bound_drift.DriftArtifactPaths(
        tmp_path / "training-reference.json", tmp_path / "validation-audit.json"
    )
    paths.training_reference.write_bytes(data["training_reference"])
    paths.validation_audit.write_bytes(data["validation_audit"])
    binding = SimpleNamespace(source_hashes=((bound_drift._SOURCE, "a" * 64),))
    models = SimpleNamespace(cascade=SimpleNamespace(stage1_model=object()), gmm={})
    return binding, paths, models, data


def test_canonical_report_summary_restores_real_retained_snapshots(restore_fixture):
    binding, paths, models, data = restore_fixture
    result = bound_drift.load_bound_drift(binding, paths, models)
    assert type(result.reference) is retained_drift.RetainedDriftReference
    assert result.training_reference_bytes == data["training_reference"]
    assert result.validation_audit_bytes == data["validation_audit"]
    assert result.reference.psi.training_row_count == 256
    assert result.reference.scaler_mean == (0.0,) * 26
    assert len(result.reference.mmd.values) == 256
    assert (
        result.reference.training_reference_sha256 == data["expected_reference_sha256"]
    )


@pytest.mark.parametrize("field", ["training_reference", "validation_audit"])
def test_real_restore_rejects_changed_private_snapshot(restore_fixture, field):
    binding, paths, models, _ = restore_fixture
    path = getattr(paths, field)
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(
        bound_drift.BoundDriftError, match="^invalid_bound_drift_evidence$"
    ):
        bound_drift.load_bound_drift(binding, paths, models)
