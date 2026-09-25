"""Pair fixture keeps both actual children and each independent saved verifier."""

from types import SimpleNamespace

import pytest
from prepared_external_worker_fixtures import child_command as external_command
from prepared_internal_worker_fixtures import child_command as internal_command
from test_evaluation_producer import synthetic_session
from test_prepared_internal_process_integration import expected_models

from automated_phishing_detection import evaluation_producer, prepared_internal_runner
from automated_phishing_detection import prepared_external_process as pair
from automated_phishing_detection._prepared_internal_records import (
    PreparedInternalRunPaths,
)


def internal_paths(case, inputs):
    original = inputs[1]
    return PreparedInternalRunPaths(
        case.paths.preparation,
        original.artifacts,
        original.secondary_artifacts,
        original.attempt.parent / "internal-scoring",
        original.attempt.parent / "internal-scoring.json",
    )


def internal_observed(binding, paths, *, preparation):
    with pytest.MonkeyPatch.context() as monkeypatch:
        session, *_ = synthetic_session(evaluation_producer, monkeypatch)
        case = SimpleNamespace(
            binding=binding,
            paths=paths,
            preparation=preparation,
            identity=preparation.execution,
            session=session,
        )
        expected_models(case, monkeypatch)
        command = internal_command(case, "success")
        monkeypatch.setattr(
            prepared_internal_runner, "_worker_command", lambda *args, **kwargs: command
        )
        return prepared_internal_runner._run_observed_prepared_internal(
            binding, paths, preparation=preparation
        )


def configure_pair(case, monkeypatch, mode="success"):
    for name in ("source_csv", "suffix_rules", "archive"):
        getattr(case.original.paths, name).unlink()
    monkeypatch.setattr(pair, "_run_observed_prepared_internal", internal_observed)
    monkeypatch.setattr(
        pair.process, "resolve_external_source_profile", lambda _: case.profile
    )
    monkeypatch.setattr(
        pair.process,
        "_prepared_worker_command",
        lambda binding, paths, transport, preparation: external_command(
            case, transport, mode
        ),
    )


def run_pair(case, paths):
    return pair._run_held_sources(
        case.binding,
        paths,
        case.paths,
        case.preparation.reservation_sha256,
        case.preparation.completion_sha256,
    )
