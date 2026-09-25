"""Actual retained-input children, observed exits and independent saved replay."""

import json

import pytest
from prepared_external_fixtures import (
    inputs,
    preparation_api,
    preparation_case,
    prepared_case,
    runner,
)
from prepared_external_worker_fixtures import child_command

from automated_phishing_detection import external_source_process as process
from automated_phishing_detection._prepared_external_runtime import held_preparation

__all__ = ["inputs", "preparation_api", "preparation_case", "prepared_case", "runner"]


def run_owned(case, monkeypatch, mode):
    for name in ("source_csv", "suffix_rules", "archive"):
        getattr(case.original.paths, name).unlink()
    monkeypatch.setattr(
        process, "resolve_external_source_profile", lambda _: case.profile
    )
    monkeypatch.setattr(
        process,
        "_prepared_worker_command",
        lambda binding, paths, transport, preparation: child_command(
            case, transport, mode
        ),
    )
    with held_preparation(
        case.binding,
        case.paths,
        case.preparation.reservation_sha256,
        case.preparation.completion_sha256,
    ) as preparation:
        return process._run_observed_external(
            case.binding, case.paths, case.handoff, preparation=preparation
        )


def test_actual_external_producer_child_has_verified_saved_completion(
    prepared_case, monkeypatch
):
    case = prepared_case
    result = run_owned(case, monkeypatch, "success")
    assert result.worker.exit.exit_observed is True
    assert result.worker.exit.exit_code == 0
    assert len(result.snapshot.payloads) == 76
    assert result.snapshot.payload(
        "attempt/evidence/retained-test.jsonl"
    ) == case.preparation.payload("retained-test.jsonl")


def test_published_child_then_nonzero_exit_is_rejected(prepared_case, monkeypatch):
    case = prepared_case
    monkeypatch.setattr(
        process,
        "verify_external_completion_snapshot",
        lambda *args, **kwargs: pytest.fail("nonzero worker reached acceptance"),
    )
    with pytest.raises(
        process.source_runner.SourceExecutionError, match="worker_exit"
    ) as caught:
        run_owned(case, monkeypatch, "nonzero")
    assert case.paths.public_summary.is_file()
    assert caught.value.external_failure.worker.exit.exit_code == 17
    assert (
        caught.value.external_failure.preparation.completion_sha256
        == case.preparation.completion_sha256
    )


def test_child_teardown_preserves_science_without_publication(
    prepared_case, monkeypatch
):
    case = prepared_case
    with pytest.raises(process.source_runner.SourceExecutionError, match="worker_exit"):
        run_owned(case, monkeypatch, "teardown")
    assert not case.paths.public_summary.exists()
    failure = json.loads((case.paths.attempt / "external-failure.json").read_bytes())
    assert failure["cleanup_failed"] is True
    assert len(failure["completed"]["private_outputs_base64"]) == 30
