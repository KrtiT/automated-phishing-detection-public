"""Invented model artifacts and sessions, retaining original scientific bodies."""

import json
from contextlib import contextmanager
from dataclasses import replace
from hashlib import sha256

import saved_external_artifact_fixtures as artifacts
import saved_external_fixtures as scientific
from prepared_external_drift_fixtures import aligned_drift
from prepared_external_fixtures import _bound_baseline, _bound_gmm
from saved_external_scorer_fixtures import FixturePrimaryScorer

from automated_phishing_detection import (
    evaluation_producer,
    external_source_completion,
    external_source_runner,
    gmm_monitor,
    length_inference,
    saved_evidence,
    source_runner,
)
from automated_phishing_detection.bound_external_runtime import BoundExternalSession
from automated_phishing_detection.bound_runtime import (
    BoundEvaluationSession,
    BoundSession,
)
from automated_phishing_detection.bound_secondary import score_bound_secondary
from automated_phishing_detection.gmm_monitor import score_feature_matrix
from automated_phishing_detection.length_inference import (
    score_length_only_authoritative,
)


def install_science(case, monkeypatch, *, write_public=True):
    _restore_original_scorers(monkeypatch)
    source = (case.base.root / "data/sources.json").read_bytes()
    report = (case.base.root / "reports/phiusiil-preparation-summary.json").read_bytes()
    monkeypatch.setattr(
        artifacts, "snapshot_chain", lambda: aligned_drift(source, report)
    )
    _bound_baseline(monkeypatch)
    _bound_gmm(monkeypatch)
    state = artifacts.artifact_state()
    drift = scientific._drift(state, monkeypatch)
    secondary = scientific._secondary(drift, monkeypatch)
    models = _models(state, secondary)
    session = BoundEvaluationSession(
        BoundSession(models, FixturePrimaryScorer(models.cascade)), secondary
    )
    scientific._secondary_scorers(monkeypatch, [])
    _retain_public_models(case, drift, write_public)
    _expected_bindings(session, monkeypatch)
    _session_owners(session, drift, monkeypatch)
    return session


def _models(state, secondary):
    original = scientific._models(state, secondary)
    additional = tuple(
        (name, sha256(f"invented {name}".encode()).hexdigest())
        for name in ("cascade.json", "transformer.json")
    )
    return replace(original, artifact_hashes=original.artifact_hashes + additional)


def _restore_original_scorers(monkeypatch):
    monkeypatch.setattr(gmm_monitor, "score_feature_matrix", score_feature_matrix)
    monkeypatch.setattr(
        length_inference,
        "score_length_only_authoritative",
        score_length_only_authoritative,
    )
    monkeypatch.setattr(
        evaluation_producer, "score_bound_secondary", score_bound_secondary
    )


def _retain_public_models(case, drift, write_public):
    if write_public:
        for name, content in drift.public_inputs:
            (case.base.root / name).write_bytes(content)
    pins = dict(case.base.source_hashes)
    pins.update(
        (name, scientific.sha256(content).hexdigest())
        for name, content in drift.public_inputs
    )
    case.base = replace(case.base, source_hashes=tuple(sorted(pins.items())))


def _session_owners(session, drift, monkeypatch):
    monkeypatch.setattr(source_runner, "recheck_binding", lambda binding: None)
    monkeypatch.setattr(external_source_runner, "recheck_binding", lambda binding: None)
    monkeypatch.setattr(
        external_source_completion, "recheck_binding", lambda binding: None
    )
    monkeypatch.setattr(
        source_runner, "open_bound_evaluation_session", lambda *args: owner(session)
    )
    monkeypatch.setattr(
        external_source_runner,
        "open_bound_external_session",
        lambda *args: owner(BoundExternalSession(session, drift)),
    )


def _expected_bindings(session, monkeypatch):
    bindings = {
        "artifact_hashes": dict(session.primary.models.artifact_hashes),
        "thresholds": evaluation_producer._thresholds(session.primary),
        "secondary": evaluation_producer._secondary_binding(session.secondary, 0.5),
        "gmm_audit": {"alert_count": 28, "window_count": 252},
    }
    monkeypatch.setattr(
        saved_evidence, "_EXPECTED_BINDING_CORE", json.loads(json.dumps(bindings))
    )


@contextmanager
def owner(session):
    yield session
