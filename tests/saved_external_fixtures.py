"""Complete invented external bundles with real retained primary arithmetic."""

import json
from dataclasses import replace
from hashlib import sha256
from types import SimpleNamespace

import pytest
from external_composition_fixtures import _secondary_scorers
from external_producer_fixtures import prepared_external
from retained_external_drift_fixtures import accepted_report
from saved_external_artifact_fixtures import artifact_state
from saved_external_scorer_fixtures import FixturePrimaryScorer
from test_evaluation_producer import _bound_secondary_fixture

from automated_phishing_detection import bound_secondary, saved_evidence
from automated_phishing_detection.bound_drift import BoundDrift
from automated_phishing_detection.bound_external_runtime import BoundExternalSession
from automated_phishing_detection.bound_models import BoundModels
from automated_phishing_detection.bound_runtime import (
    BoundEvaluationSession,
    BoundSession,
)
from automated_phishing_detection.external_producer import produce_external_evidence
from automated_phishing_detection.retained_drift import load_retained_drift_reference


def _drift(state: SimpleNamespace, monkeypatch: pytest.MonkeyPatch) -> BoundDrift:
    report = accepted_report(state.snapshots)
    public = (
        ("reports/secondary-development-correction-v2-summary.json", report),
        (
            "reports/phiusiil-preparation-summary.json",
            state.snapshots["preparation_summary"],
        ),
        ("data/sources.json", state.source),
    )
    monkeypatch.setitem(
        bound_secondary.PUBLIC_REPORTS,
        "tabular",
        (public[0][0], sha256(report).hexdigest()),
    )
    return BoundDrift(
        load_retained_drift_reference(**state.snapshots),
        state.snapshots["training_reference"],
        state.snapshots["validation_audit"],
        public,
    )


def _models(
    state: SimpleNamespace, secondary: bound_secondary.BoundSecondary
) -> BoundModels:
    cascade = SimpleNamespace(
        stage1_model=state.logistic,
        stage1_threshold=0.5,
        transformer_threshold=0.5,
        half_width=0.11,
    )
    hashes = (
        ("length-only.json", state.length.artifact_sha256),
        ("logistic-l1.json", state.logistic.artifact_sha256),
        ("gmm.json", sha256(state.gmm_bytes).hexdigest()),
        ("transformer-weights.npz", secondary.seeds[0].weights_sha256),
        ("vocabulary.json", sha256(secondary.vocabulary_bytes).hexdigest()),
    )
    return BoundModels(
        state.length, cascade, state.gmm, 3.0, hashes, state.gmm_bytes, 28, 252
    )


def _secondary(
    drift: BoundDrift, monkeypatch: pytest.MonkeyPatch
) -> bound_secondary.BoundSecondary:
    bound = _bound_secondary_fixture()
    reports = (
        ("tabular", sha256(drift.public_inputs[0][1]).hexdigest()),
        ("seeds", "2" * 64),
    )
    monkeypatch.setitem(
        bound_secondary.PUBLIC_REPORTS,
        "seeds",
        (bound_secondary.PUBLIC_REPORTS["seeds"][0], reports[1][1]),
    )
    return replace(bound, report_hashes=reports)


def saved_external_bundle(
    monkeypatch: pytest.MonkeyPatch, count: int = 5
) -> SimpleNamespace:
    state = artifact_state()
    drift = _drift(state, monkeypatch)
    secondary = _secondary(drift, monkeypatch)
    models = _models(state, secondary)
    primary = BoundSession(models, FixturePrimaryScorer(models.cascade))
    session = BoundExternalSession(BoundEvaluationSession(primary, secondary), drift)
    _secondary_scorers(monkeypatch, [])
    prepared = prepared_external(count)
    produced = produce_external_evidence(prepared, session)
    bindings = json.loads(produced.private_outputs["bindings.json"])
    monkeypatch.setattr(
        saved_evidence,
        "_EXPECTED_BINDING_CORE",
        {
            name: bindings[name]
            for name in ("artifact_hashes", "thresholds", "secondary", "gmm_audit")
        },
    )
    return SimpleNamespace(produced=produced, prepared=prepared, session=session)
