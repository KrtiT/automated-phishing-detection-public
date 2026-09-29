"""Genuine scientific owners with invented models and declared unit exit lineage."""

import sys
from contextlib import contextmanager
from dataclasses import fields
from types import SimpleNamespace

import prepared_external_fixtures as external_fixture
import pytest
from phishvn_source_fixtures import bundle, members, record
from prepared_internal_fixtures import bind_saved_fixture, forbid_preparation
from retained_study_derivation_fixtures import derivation_case, run
from study_preparation_runner_fixtures import (
    inputs,
    preparation_api,
    preparation_case,
    profile,
    runner,
)
from test_external_source_runner import fresh_session

from automated_phishing_detection import (
    bound_secondary,
    evaluation_producer,
    external_source_completion,
    external_source_runner,
    prepared_internal_runner,
    source_completion,
    source_runner,
)
from automated_phishing_detection._prepared_external_records import (
    PreparedExternalRunPaths,
)
from automated_phishing_detection._prepared_internal_records import (
    PreparedInternalRunPaths,
)
from automated_phishing_detection.bound_drift import DriftArtifactPaths
from automated_phishing_detection.internal_external_handoff import (
    build_internal_handoff,
)
from automated_phishing_detection.internal_process_handoff import (
    ObservedInternalCompletion,
)
from automated_phishing_detection.owned_worker import observe_worker

__all__ = ["inputs", "preparation_api", "preparation_case", "runner", "scientific_case"]


def _rows():
    mappings = (
        ("tinnhiemmang", "gold", "phishing"),
        ("tinnhiem_web", "gold", "benign"),
        ("tranco", "silver", "benign"),
        ("tinnhiemmang", "silver", "phishing"),
        ("openphish", "bronze", "phishing"),
    )
    return [
        record(
            f"amended-{index}",
            source=source,
            tier=tier,
            label=label,
            url=f"bare{index}.invalid/Original",
            url_norm=f"HTTPS://Host{index}.Amendment{index}.com/PaTh?ToKeN=MiXeD",
        )
        for index, (source, tier, label) in enumerate(mappings)
    ]


def _derived(original, preparation_api, monkeypatch):
    archive = bundle(members(_rows()))
    original.paths.archive.write_bytes(archive.content)
    original.profile = profile(
        original.binding, original.paths.suffix_rules.read_bytes(), archive
    )
    monkeypatch.setattr(
        preparation_api.body,
        "resolve_external_source_profile",
        lambda _: original.profile,
    )
    case = derivation_case.__wrapped__(original, preparation_api, monkeypatch)
    snapshot = run(case)
    prepared = external_fixture._restore(case, snapshot)
    return (
        case,
        prepared,
        preparation_api.body.resolve_external_source_profile(case.binding),
    )


@pytest.fixture
def scientific_case(preparation_case, preparation_api, inputs, tmp_path, monkeypatch):
    derived, prepared, candidate = _derived(
        preparation_case, preparation_api, monkeypatch
    )
    original = inputs[1]
    case = SimpleNamespace(
        binding=derived.binding,
        profile=candidate,
        preparation=prepared,
        prior=derived.prior,
        events=[],
        active=False,
        expected_urls=tuple(row["url_norm"] for row in _rows()),
        feature_urls=[],
    )
    external_fixture._scientific(case, monkeypatch)
    _scoring_paths(case, derived.paths.attempt, original, tmp_path)
    external_fixture._configure(case, monkeypatch)
    return case


def _scoring_paths(case, preparation, original, root):
    artifacts = preparation, original.artifacts, original.secondary_artifacts
    drift = DriftArtifactPaths(
        *(root / value.name for value in fields(DriftArtifactPaths))
    )
    case.paths = PreparedExternalRunPaths(
        *artifacts, drift, root / "external-scoring", root / "external-scoring.json"
    )
    case.internal_paths = PreparedInternalRunPaths(
        *artifacts, root / "internal-scoring", root / "internal-scoring.json"
    )


@contextmanager
def _internal_owner(case, *arguments):
    yield fresh_session(case).evaluation


def _internal(case, monkeypatch):
    monkeypatch.setattr(
        evaluation_producer,
        "score_bound_secondary",
        bound_secondary.score_bound_secondary,
    )
    monkeypatch.setattr(
        source_runner,
        "open_bound_evaluation_session",
        lambda *arguments: _internal_owner(case, *arguments),
    )
    prepared_internal_runner._run_bound_prepared_internal(
        case.binding, case.internal_paths, case.preparation
    )
    bind_saved_fixture(SimpleNamespace(paths=case.internal_paths), monkeypatch)
    return source_completion.verify_prepared_internal_completion_snapshot(
        case.binding,
        case.internal_paths,
        preparation=case.preparation,
        producer_exit_code=0,
    )


def _capture_inputs(case, monkeypatch):
    supplied = []
    original = bound_secondary.SecondaryModel.score_urls_singleton_ordered
    extract = evaluation_producer.extract_url_features

    def tabular(model, urls):
        supplied.append(tuple(urls))
        return original(model, urls)

    def features(url):
        case.feature_urls.append(url)
        return extract(url)

    monkeypatch.setattr(
        bound_secondary.SecondaryModel, "score_urls_singleton_ordered", tabular
    )
    monkeypatch.setattr(evaluation_producer, "extract_url_features", features)
    return supplied


def score_and_verify(case, monkeypatch):
    forbid_preparation(monkeypatch)
    internal = _internal(case, monkeypatch)
    command = (sys.executable, "-c", "pass")
    case.handoff = build_internal_handoff(
        ObservedInternalCompletion(observe_worker(command), internal)
    )
    supplied = _capture_inputs(case, monkeypatch)
    external_source_runner._run_bound_prepared_external(
        case.binding, case.paths, handoff=case.handoff, preparation=case.preparation
    )
    external = external_source_completion.verify_external_completion_snapshot(
        case.binding,
        case.paths,
        expected_handoff=case.handoff,
        worker=observe_worker(command),
        command=command,
        expected_preparation=case.preparation,
    )
    return internal, external, supplied
