import importlib
import json
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import pytest

from automated_phishing_detection import secondary_development


@pytest.fixture
def drift():
    source = Path(__file__).resolve().parents[1] / (
        "src/automated_phishing_detection/bound_drift.py"
    )
    assert source.is_file(), "missing accepted external drift binding"
    return importlib.import_module("automated_phishing_detection.bound_drift")


def encode(value):
    return (json.dumps(value, sort_keys=True) + "\n").encode()


@pytest.fixture
def prepared(drift, tmp_path, monkeypatch):
    reference_bytes, audit_bytes = b"reference fixture", b"audit fixture"
    source_bytes = b"source fixture"
    preparation_bytes = b"preparation fixture"
    pins = secondary_development.DevelopmentPins(
        train_sha256="a" * 64,
        validation_sha256="b" * 64,
        source_csv_sha256="c" * 64,
        preparation_summary_sha256=sha256(preparation_bytes).hexdigest(),
        suffix_rules_sha256="d" * 64,
        logistic_l1_artifact_sha256="e" * 64,
        baseline_contract_sha256="f" * 64,
        gmm_artifact_sha256="1" * 64,
        gmm_contract_sha256="2" * 64,
    )
    report_bytes = encode({"completion": {"execution": {"pins": asdict(pins)}}})
    records = {
        "reports/secondary-development-correction-v2-summary.json": report_bytes,
        "reports/phiusiil-preparation-summary.json": preparation_bytes,
        "data/sources.json": source_bytes,
    }
    for name, content in records.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    reference_path, audit_path = tmp_path / "reference", tmp_path / "audit"
    reference_path.write_bytes(reference_bytes)
    audit_path.write_bytes(audit_bytes)
    hashes = {name: sha256(content).hexdigest() for name, content in records.items()}
    hashes.update(
        {
            "data/rq1-baseline-contract-v2.json": pins.baseline_contract_sha256,
            "data/rq2-gmm-development-contract-v1.json": pins.gmm_contract_sha256,
        }
    )
    binding = SimpleNamespace(root=tmp_path, source_hashes=tuple(hashes.items()))
    paths = drift.DriftArtifactPaths(reference_path, audit_path)
    models = SimpleNamespace(
        cascade=SimpleNamespace(stage1_model=object()),
        gmm=object(),
        artifact_hashes=(
            ("logistic-l1.json", pins.logistic_l1_artifact_sha256),
            ("gmm.json", pins.gmm_artifact_sha256),
        ),
    )
    snapshot = {"fixture": "portable"}
    portable_digest = sha256(secondary_development._json_bytes(snapshot)).hexdigest()
    retained = SimpleNamespace(
        scaler_mean=(0.0,) * 26,
        scaler_scale=(1.0,) * 26,
        portable_state_sha256=portable_digest,
        psi=SimpleNamespace(training_row_count=256),
    )
    events = []
    monkeypatch.setattr(drift, "_REPORT_SHA256", sha256(report_bytes).hexdigest())

    def public_preparation(source, preparation, source_hash, supplied_pins):
        assert (source, preparation) == (source_bytes, preparation_bytes)
        assert source_hash == sha256(source_bytes).hexdigest()
        assert supplied_pins == pins
        events.append("public_preparation")
        return 256

    def accepted_report(report, supplied_pins):
        assert report == json.loads(report_bytes)
        assert supplied_pins == pins
        events.append("accepted_report")
        return (
            "{}",
            sha256(reference_bytes).hexdigest(),
            sha256(audit_bytes).hexdigest(),
        )

    def accepted_models(stage1, gmm, supplied_pins, count):
        assert stage1 is models.cascade.stage1_model and gmm is models.gmm
        assert supplied_pins == pins and count == 256
        events.append("accepted_models")
        return (0.0,) * 26, (1.0,) * 26

    def load(reference, audit, **kwargs):
        assert reference == reference_bytes and audit == audit_bytes
        assert kwargs["pins"] == pins
        assert kwargs["preparation_summary"] == preparation_bytes
        events.append("retained_reference")
        return retained

    monkeypatch.setattr(drift, "_preparation", public_preparation)
    monkeypatch.setattr(drift.seed_probe_execution, "_retained_drift", accepted_report)
    monkeypatch.setattr(
        drift.secondary_development, "_accepted_states", accepted_models
    )
    monkeypatch.setattr(
        drift.secondary_development, "_portable_snapshot", lambda _: snapshot
    )
    monkeypatch.setattr(drift.retained_drift, "load_retained_drift_reference", load)
    return SimpleNamespace(
        binding=binding,
        paths=paths,
        models=models,
        retained=retained,
        records=records,
        events=events,
        reference_bytes=reference_bytes,
        audit_bytes=audit_bytes,
    )


def test_loads_bound_snapshots_after_public_and_primary_checks(drift, prepared):
    result = drift.load_bound_drift(prepared.binding, prepared.paths, prepared.models)
    assert result.reference is prepared.retained
    assert result.training_reference_bytes == prepared.reference_bytes
    assert result.validation_audit_bytes == prepared.audit_bytes
    assert prepared.events == [
        "accepted_report",
        "public_preparation",
        "accepted_models",
        "retained_reference",
    ]


@pytest.mark.parametrize(
    "name",
    [
        "reports/secondary-development-correction-v2-summary.json",
        "reports/phiusiil-preparation-summary.json",
        "data/sources.json",
    ],
)
def test_public_tampering_rejects_before_private_access(drift, prepared, name):
    (prepared.binding.root / name).write_bytes(b"changed")
    prepared.paths.training_reference.unlink()
    with pytest.raises(drift.BoundDriftError, match="public"):
        drift.load_bound_drift(prepared.binding, prepared.paths, prepared.models)
    assert not prepared.events


@pytest.mark.parametrize(
    "field,value",
    [
        ("scaler_mean", (1.0,) * 26),
        ("scaler_scale", (2.0,) * 26),
        ("portable_state_sha256", "9" * 64),
    ],
)
def test_loaded_drift_must_match_primary_state(drift, prepared, field, value):
    setattr(prepared.retained, field, value)
    with pytest.raises(drift.BoundDriftError, match="primary_drift_state_mismatch"):
        drift.load_bound_drift(prepared.binding, prepared.paths, prepared.models)
