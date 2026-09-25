"""Real invented preparation/restoration/scoring, with declared unit lineage."""

import json
from dataclasses import fields, replace
from hashlib import sha256
from types import SimpleNamespace

import pytest
import saved_external_artifact_fixtures as artifacts
import saved_external_fixtures as scientific
from external_completion_lineage_fixtures import unit_handoff
from prepared_external_drift_fixtures import aligned_drift
from study_preparation_runner_fixtures import (
    inputs,
    preparation_api,
    preparation_case,
    runner,
)
from test_external_source_runner import fresh_session, session_owner

from automated_phishing_detection import external_source_completion as completion
from automated_phishing_detection import external_source_runner as worker
from automated_phishing_detection import gmm_monitor, length_inference
from automated_phishing_detection import retained_study_preparation as restoration
from automated_phishing_detection import study_preparation_context as context
from automated_phishing_detection._checkpoint_codec import canonical_bytes
from automated_phishing_detection._external_source_records import (
    PreparedExternalRunPaths,
)
from automated_phishing_detection._internal_handoff_validation import (
    OVERLAP_NAME,
    SOURCE_LINKS,
)
from automated_phishing_detection.bound_drift import DriftArtifactPaths
from automated_phishing_detection.gmm_monitor import score_feature_matrix as genuine_gmm
from automated_phishing_detection.internal_external_handoff import (
    InternalHandoffPayloads,
)
from automated_phishing_detection.length_inference import (
    score_length_only_authoritative as genuine_length,
)

__all__ = ["inputs", "preparation_api", "preparation_case", "runner", "prepared_case"]


def _restore(case, snapshot):
    source, report = (
        case.binding.root / name
        for name in ("data/sources.json", "reports/phiusiil-preparation-summary.json")
    )
    complete = snapshot.payload("preparation-complete.json")
    return restoration.restore_study_preparation(
        snapshot,
        expected_identity=json.loads(complete)["execution"],
        expected_reservation_sha256=snapshot.reservation_sha256,
        expected_completion_sha256=sha256(complete).hexdigest(),
        source_spec_bytes=source.read_bytes(),
        preparation_summary_bytes=report.read_bytes(),
    )


def _handoff(case):
    original = unit_handoff(
        case.binding,
        case.preparation.payload("suffix-rules.dat"),
        case.preparation.execution["source_csv_sha256"],
    )
    envelope = json.loads(original.handoff_bytes)
    envelope["execution"].update(case.preparation.scoring_source)
    envelope["execution"]["partition_sha256"] = case.preparation.execution[
        "partition_sha256"
    ]
    for name, path in SOURCE_LINKS.items():
        envelope["snapshot_sha256"][path] = envelope["execution"][name]
    overlap = case.preparation.payload("source-overlap.json")
    envelope["snapshot_sha256"][OVERLAP_NAME] = sha256(overlap).hexdigest()
    return InternalHandoffPayloads(canonical_bytes(envelope), overlap)


def _scientific(case, monkeypatch):
    monkeypatch.setattr(gmm_monitor, "score_feature_matrix", genuine_gmm)
    monkeypatch.setattr(
        length_inference, "score_length_only_authoritative", genuine_length
    )
    buffers = {
        name: (case.binding.root / name).read_bytes()
        for name in ("data/sources.json", "reports/phiusiil-preparation-summary.json")
    }
    monkeypatch.setattr(
        artifacts, "snapshot_chain", lambda: aligned_drift(*buffers.values())
    )
    _bound_baseline(monkeypatch)
    _bound_gmm(monkeypatch)
    monkeypatch.setattr(
        scientific, "prepared_external", lambda _: case.preparation.external
    )
    case.bundle = scientific.saved_external_bundle(monkeypatch, 1)
    case.produced = case.bundle.produced
    case.binding = replace(
        case.binding,
        source_hashes=tuple(
            (name, sha256(content).hexdigest())
            for name, content in case.bundle.session.drift.public_inputs
        ),
    )
    case.session = fresh_session(case)


def _bound_baseline(monkeypatch):
    baseline = artifacts._baseline_bytes

    def bound_baseline(artifact, data):
        value = json.loads(baseline(artifact, data))
        value["scaler"]["n_samples_seen"] = json.loads(data["preparation_summary"])[
            "splits"
        ]["train"]["row_count"]
        return canonical_bytes(value)

    monkeypatch.setattr(artifacts, "_baseline_bytes", bound_baseline)


def _bound_gmm(monkeypatch):
    original = artifacts._gmm

    def adjusted(data, logistic):
        unused, content, pins = original(data, logistic)
        value = json.loads(content)
        value["scaler"]["n_samples_seen"] = json.loads(data["preparation_summary"])[
            "splits"
        ]["train"]["row_count"]
        content = gmm_monitor._canonical_json_bytes(value)
        return gmm_monitor.load_gmm_artifact_bytes(content), content, pins

    monkeypatch.setattr(artifacts, "_gmm", adjusted)


def _configure(case, monkeypatch):
    for module in (worker.body, completion, context):
        monkeypatch.setattr(
            module, "resolve_external_source_profile", lambda _: case.profile
        )
    monkeypatch.setattr(
        worker, "open_bound_external_session", lambda *args: session_owner(case, *args)
    )
    monkeypatch.setattr(
        worker, "recheck_binding", lambda _: case.events.append("binding_checked")
    )
    monkeypatch.setattr(completion, "recheck_binding", lambda _: None)


@pytest.fixture
def prepared_case(preparation_api, preparation_case, tmp_path, monkeypatch):
    original = preparation_case
    snapshot = preparation_api._run_bound_preparation(original.binding, original.paths)
    case = SimpleNamespace(
        binding=original.binding,
        profile=original.profile,
        original=original,
        preparation=_restore(original, snapshot),
        events=[],
        active=False,
    )
    _scientific(case, monkeypatch)
    case.paths = PreparedExternalRunPaths(
        original.paths.attempt,
        _artifact_paths(tmp_path),
        _secondary_paths(tmp_path),
        DriftArtifactPaths(
            *(tmp_path / member.name for member in fields(DriftArtifactPaths))
        ),
        tmp_path / "external-attempt",
        tmp_path / "external-public.json",
    )
    case.handoff = _handoff(case)
    _configure(case, monkeypatch)
    return case


def _artifact_paths(root):
    from automated_phishing_detection.bound_models import ArtifactPaths

    return ArtifactPaths(*(root / member.name for member in fields(ArtifactPaths)))


def _secondary_paths(root):
    from automated_phishing_detection.bound_secondary import SecondaryArtifactPaths

    return SecondaryArtifactPaths(
        *(root / member.name for member in fields(SecondaryArtifactPaths))
    )
