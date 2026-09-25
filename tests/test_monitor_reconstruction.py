"""Replay monitor evidence from invented retained artifacts and real scorers."""

import base64
import builtins
import json
import math
from copy import deepcopy
from hashlib import sha256
from pathlib import Path

import pytest
import threadpoolctl
from sklearn.linear_model import LogisticRegression
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from test_fixed_cascade import _artifact as logistic_artifact
from test_length_inference import _artifact as length_artifact

from automated_phishing_detection import (
    fixed_cascade,
    gmm_monitor,
    length_inference,
    saved_evidence,
)
from automated_phishing_detection.url_features import extract_url_features


def _artifact_bytes(artifact):
    return (json.dumps(artifact, indent=2, sort_keys=True) + "\n").encode("ascii")


def _baseline_bytes(factory):
    artifact = factory()
    contract = fixed_cascade.OFFICIAL_BASELINE_CONTRACT_SHA256
    artifact["contract_sha256"] = contract
    artifact["input_hashes"]["contract"] = contract
    return _artifact_bytes(artifact)


@pytest.fixture
def monitor_evidence(monkeypatch):
    if gmm_monitor._numpy_build_configuration()["name"] != "scipy-openblas":
        monkeypatch.setattr(gmm_monitor, "_require_runtime", lambda: None)
    artifacts = {
        "length-only.json": _baseline_bytes(length_artifact),
        "logistic-l1.json": _baseline_bytes(logistic_artifact),
        "gmm.json": _artifact_bytes(
            {
                "schema_version": 1,
                "contract_id": gmm_monitor.CONTRACT_ID,
                "features": list(gmm_monitor.GMM_FEATURE_NAMES),
                "dtype": "float64",
                "scaler": {
                    "mean": [0.0] * 26,
                    "scale": [1.0] * 26,
                    "variance": [1.0] * 26,
                    "n_samples_seen": 20,
                    "n_features_in": 26,
                },
                "mixture": {
                    "components": 1,
                    "covariance_type": "diag",
                    "weights": [1.0],
                    "means": [[0.0] * 26],
                    "variances": [[1.0] * 26],
                    "precisions": [[1.0] * 26],
                    "precisions_cholesky": [[1.0] * 26],
                    "converged": True,
                    "n_iter": 1,
                    "lower_bound": -1.0,
                },
                "input_hashes": {
                    "logistic_l1_artifact": sha256(
                        _baseline_bytes(logistic_artifact)
                    ).hexdigest()
                },
            }
        ),
    }
    bindings = {
        "artifact_hashes": {
            name: sha256(content).hexdigest() for name, content in artifacts.items()
        },
        "replay_artifacts": {
            name: base64.b64encode(content).decode("ascii")
            for name, content in artifacts.items()
        },
    }
    length = length_inference._load_length_only_artifact_bytes(
        artifacts["length-only.json"],
        expected_sha256=bindings["artifact_hashes"]["length-only.json"],
    )
    stage1 = fixed_cascade._load_logistic_l1_artifact_bytes(
        artifacts["logistic-l1.json"],
        expected_sha256=bindings["artifact_hashes"]["logistic-l1.json"],
    )
    mixture = gmm_monitor.load_gmm_artifact_bytes(artifacts["gmm.json"])
    rows = []
    for raw_url in (
        "https://replay-one.test/a",
        "http://replay-two.test:8080/account?q=23&name=synthetic#fragment",
    ):
        urls = (raw_url,)
        length_scores, length_audit = length_inference.score_length_only_authoritative(
            length, urls
        )
        stage1_scores, stage1_audit = fixed_cascade.score_logistic_l1_authoritative(
            stage1, urls
        )
        monitor_scores = stage1.score_urls(urls)
        features = extract_url_features(raw_url)
        nll = gmm_monitor.score_feature_matrix(
            ((*features, monitor_scores[0]),), mixture
        )
        rows.append(
            {
                "record": {"raw_url": raw_url},
                "features": list(features),
                "length_probability": length_scores[0],
                "stage1_probability": stage1_scores[0],
                "monitor_probability": monitor_scores[0],
                "negative_log_likelihood": float(nll[0]),
                "length_scoring_audit_json": saved_evidence._json_bytes(length_audit)
                .decode("ascii")
                .rstrip("\n"),
                "stage1_scoring_audit_json": saved_evidence._json_bytes(stage1_audit)
                .decode("ascii")
                .rstrip("\n"),
            }
        )
    return tuple(rows), bindings


def test_replays_real_monitor_scores_without_files_or_fitting(
    monitor_evidence, monkeypatch
):
    rows, bindings = monitor_evidence

    def forbidden(*args, **kwargs):
        pytest.fail("retained monitor replay attempted file access or fitting")

    for owner, name in (
        (builtins, "open"),
        (Path, "open"),
        (Path, "read_bytes"),
        (Path, "read_text"),
        (fixed_cascade, "_read_regular_file"),
        (fixed_cascade, "load_logistic_l1_artifact"),
        (length_inference, "load_length_only_artifact"),
        (StandardScaler, "fit"),
        (LogisticRegression, "fit"),
        (GaussianMixture, "fit"),
        (gmm_monitor, "fit_training_mixture"),
    ):
        monkeypatch.setattr(owner, name, forbidden)

    saved_evidence._verify_monitor_path(rows, bindings)


@pytest.mark.parametrize("fail", [False, True])
def test_replay_uses_and_restores_one_thread_for_every_scorer(
    monitor_evidence, monkeypatch, fail
):
    rows, bindings = monitor_evidence
    observed = []

    def observe(owner, name):
        original = getattr(owner, name)

        def wrapped(*args, **kwargs):
            pools = threadpoolctl.threadpool_info()
            assert pools and all(pool["num_threads"] == 1 for pool in pools)
            observed.append(name)
            if fail:
                raise saved_evidence.SavedEvidenceError("injected replay failure")
            return original(*args, **kwargs)

        monkeypatch.setattr(owner, name, wrapped)

    for owner, name in (
        (length_inference, "score_length_only_authoritative"),
        (fixed_cascade, "score_logistic_l1_authoritative"),
        (fixed_cascade.PortableLogisticL1, "score_urls"),
        (gmm_monitor, "score_feature_matrix"),
    ):
        observe(owner, name)
    with threadpoolctl.threadpool_limits(limits=4):
        before = threadpoolctl.threadpool_info()
        if fail:
            with pytest.raises(saved_evidence.SavedEvidenceError, match="injected"):
                saved_evidence._verify_monitor_path(rows, bindings)
        else:
            saved_evidence._verify_monitor_path(rows, bindings)
        assert threadpoolctl.threadpool_info() == before
    assert observed


@pytest.mark.parametrize(
    "field",
    [
        "length_probability",
        "stage1_probability",
        "monitor_probability",
        "negative_log_likelihood",
    ],
)
def test_rejects_one_ulp_score_mutations(monitor_evidence, field):
    rows, bindings = monitor_evidence
    changed = deepcopy(rows)
    changed[1][field] = math.nextafter(changed[1][field], math.inf)

    with pytest.raises(saved_evidence.SavedEvidenceError, match="artifact replay"):
        saved_evidence._verify_monitor_path(changed, bindings)


@pytest.mark.parametrize(
    "field", ["length_scoring_audit_json", "stage1_scoring_audit_json"]
)
def test_rejects_canonical_audit_mutation(monitor_evidence, field):
    rows, bindings = monitor_evidence
    changed = deepcopy(rows)
    audit = json.loads(changed[0][field])
    audit["max_absolute_probability_difference"] += 1e-10
    changed[0][field] = saved_evidence._json_bytes(audit).decode("ascii").rstrip("\n")

    with pytest.raises(saved_evidence.SavedEvidenceError, match="artifact replay"):
        saved_evidence._verify_monitor_path(changed, bindings)


@pytest.mark.parametrize("name", ["length-only.json", "logistic-l1.json", "gmm.json"])
@pytest.mark.parametrize("mutation", ["digest", "bytes", "encoding", "type"])
def test_rejects_retained_artifact_mutations(monitor_evidence, name, mutation):
    rows, bindings = monitor_evidence
    changed = deepcopy(bindings)
    encoded = changed["replay_artifacts"][name]
    if mutation == "digest":
        changed["artifact_hashes"][name] = "f" * 64
    elif mutation == "bytes":
        content = base64.b64decode(encoded) + b" "
        changed["replay_artifacts"][name] = base64.b64encode(content).decode("ascii")
    elif mutation == "encoding":
        changed["replay_artifacts"][name] = encoded + "\n"
    else:
        changed["replay_artifacts"][name] = encoded.encode("ascii")

    with pytest.raises(ValueError):
        saved_evidence._verify_monitor_path(rows, changed)


@pytest.mark.parametrize("mutation", ["missing", "additional"])
def test_rejects_changed_retained_artifact_inventory(monitor_evidence, mutation):
    rows, bindings = monitor_evidence
    changed = deepcopy(bindings)
    if mutation == "missing":
        del changed["replay_artifacts"]["gmm.json"]
    else:
        changed["replay_artifacts"]["extra.json"] = "e30="

    with pytest.raises(saved_evidence.SavedEvidenceError, match="artifact inventory"):
        saved_evidence._verify_monitor_path(rows, changed)
