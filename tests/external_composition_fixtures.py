"""Invented complete external session; no protected source or model files."""

import importlib
import importlib.util
from dataclasses import replace
from hashlib import sha256
from types import SimpleNamespace

from external_producer_fixtures import prepared_external
from retained_external_drift_fixtures import accepted_report, snapshot_chain
from test_external_primary import external_session

from automated_phishing_detection import bound_secondary
from automated_phishing_detection.bound_drift import BoundDrift
from automated_phishing_detection.retained_drift import load_retained_drift_reference


def producer_module():
    name = "automated_phishing_detection.external_producer"
    assert importlib.util.find_spec(name) is not None, "external composition missing"
    return importlib.import_module(name)


def _load_seed(weights, vocabulary, *, seed, device):
    return SimpleNamespace(
        seed=seed,
        device=device,
        weights_sha256=sha256(weights).hexdigest(),
        vocabulary_sha256=sha256(vocabulary).hexdigest(),
    )


def _secondary_scorers(monkeypatch, calls):
    def tabular(model, urls):
        calls.append((model.artifact_bytes.decode(), len(urls)))
        return tuple(0.25 for unused in urls)

    def transformer(model, urls):
        calls.append((model.seed, len(urls)))
        return tuple(0.75 for unused in urls)

    monkeypatch.setattr(
        bound_secondary.SecondaryModel, "score_urls_singleton_ordered", tabular
    )
    monkeypatch.setattr(
        bound_secondary.secondary_transformer,
        "load_secondary_transformer_bytes",
        _load_seed,
    )
    monkeypatch.setattr(
        bound_secondary.secondary_transformer,
        "score_secondary_transformer_urls",
        transformer,
    )


def _drift(monkeypatch):
    snapshots, source = snapshot_chain()
    report = accepted_report(snapshots)
    reference = load_retained_drift_reference(**snapshots)
    public_inputs = (
        (
            "reports/secondary-development-correction-v2-summary.json",
            report,
        ),
        ("reports/phiusiil-preparation-summary.json", snapshots["preparation_summary"]),
        ("data/sources.json", source),
    )
    monkeypatch.setitem(
        bound_secondary.PUBLIC_REPORTS,
        "tabular",
        (public_inputs[0][0], sha256(report).hexdigest()),
    )
    return BoundDrift(
        reference,
        snapshots["training_reference"],
        snapshots["validation_audit"],
        public_inputs,
    )


def composition_inputs(monkeypatch, count=5):
    session, primary_calls = external_session(monkeypatch)
    drift = _drift(monkeypatch)
    secondary = replace(
        session.evaluation.secondary,
        report_hashes=(
            ("tabular", sha256(drift.public_inputs[0][1]).hexdigest()),
            ("seeds", "2" * 64),
        ),
    )
    session = replace(
        session,
        drift=drift,
        evaluation=replace(session.evaluation, secondary=secondary),
    )
    secondary_calls = []
    _secondary_scorers(monkeypatch, secondary_calls)
    return prepared_external(count), session, primary_calls, secondary_calls
