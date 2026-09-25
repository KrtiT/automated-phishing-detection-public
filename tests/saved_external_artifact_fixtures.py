"""Invented byte-valid primary artifacts bound to retained drift snapshots."""

import json
from dataclasses import asdict, replace
from hashlib import sha256
from types import SimpleNamespace

from retained_external_drift_fixtures import snapshot_chain
from test_fixed_cascade import _artifact as logistic_artifact
from test_length_inference import _artifact as length_artifact
from test_retained_drift import _rebind
from test_secondary_development import _fixture as development_fixture

from automated_phishing_detection import (
    baselines,
    fixed_cascade,
    gmm_monitor,
    length_inference,
    secondary_development,
)


def _baseline_bytes(artifact: dict, data: dict) -> bytes:
    pins = data["pins"]
    contract = fixed_cascade.OFFICIAL_BASELINE_CONTRACT_SHA256
    artifact["contract_sha256"] = contract
    artifact["scaler"]["n_samples_seen"] = 256
    artifact["software_versions"] = baselines._software_versions()
    artifact["input_hashes"] = {
        "train": pins.train_sha256,
        "validation": pins.validation_sha256,
        "preparation_summary": pins.preparation_summary_sha256,
        "contract": contract,
    }
    return secondary_development._json_bytes(artifact)


def _models(
    data: dict,
) -> tuple[length_inference.LoadedLengthOnly, fixed_cascade.PortableLogisticL1]:
    artifact = logistic_artifact()
    artifact["classifier"]["intercept"] = [-3.5]
    logistic_bytes = _baseline_bytes(artifact, data)
    length_bytes = _baseline_bytes(length_artifact(), data)
    logistic = fixed_cascade._load_logistic_l1_artifact_bytes(
        logistic_bytes, expected_sha256=sha256(logistic_bytes).hexdigest()
    )
    length = length_inference._load_length_only_artifact_bytes(
        length_bytes, expected_sha256=sha256(length_bytes).hexdigest()
    )
    return length, logistic


def _gmm(
    data: dict, logistic: fixed_cascade.PortableLogisticL1
) -> tuple[dict, bytes, secondary_development.DevelopmentPins]:
    inputs = development_fixture(secondary_development)["arguments"]
    state = inputs["gmm_state"]
    pins = replace(
        data["pins"],
        logistic_l1_artifact_sha256=logistic.artifact_sha256,
        baseline_contract_sha256=logistic.contract_sha256,
    )
    state["input_hashes"] = secondary_development._input_hashes(pins)
    content = gmm_monitor._canonical_json_bytes(state)
    return gmm_monitor.load_gmm_artifact_bytes(content), content, pins


def _snapshots(
    data: dict,
    logistic: fixed_cascade.PortableLogisticL1,
    gmm: dict,
    content: bytes,
    pins: secondary_development.DevelopmentPins,
) -> dict:
    pins = replace(pins, gmm_artifact_sha256=sha256(content).hexdigest())
    reference = json.loads(data["training_reference"])
    audit = json.loads(data["validation_audit"])
    public = data["expected_drift_summary"]
    for document in (reference, audit, public):
        document["input_hashes"] = asdict(pins)
    reference["scaler"] = {
        "mean": gmm["scaler"]["mean"],
        "scale": gmm["scaler"]["scale"],
    }
    snapshot = secondary_development._portable_snapshot(logistic)
    reference["portable_state_sha256"] = sha256(
        secondary_development._json_bytes(snapshot)
    ).hexdigest()
    return _rebind(reference, audit, public, data["preparation_summary"], pins)


def artifact_state() -> SimpleNamespace:
    data, source = snapshot_chain()
    length, logistic = _models(data)
    gmm, content, pins = _gmm(data, logistic)
    snapshots = _snapshots(data, logistic, gmm, content, pins)
    return SimpleNamespace(
        snapshots=snapshots,
        source=source,
        length=length,
        logistic=logistic,
        gmm=gmm,
        gmm_bytes=content,
    )
