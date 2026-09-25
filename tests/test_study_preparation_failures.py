"""No source or persistence failure can manufacture an eligible population zero."""

import json
from contextlib import contextmanager
from dataclasses import replace

import pytest
from study_preparation_runner_fixtures import (
    inputs,
    preparation_api,
    preparation_case,
    runner,
)

__all__ = ["inputs", "preparation_api", "preparation_case", "runner"]


@pytest.mark.parametrize(
    "error", [ValueError("private-canary"), KeyboardInterrupt(), SystemExit(0)]
)
def test_external_failure_retains_internal_predecessors(
    preparation_api, preparation_case, monkeypatch, error
):
    case = preparation_case

    def failed(*args, **kwargs):
        raise error

    monkeypatch.setattr(preparation_api.body, "prepare_external", failed)
    with pytest.raises(BaseException) as captured:
        preparation_api._run_bound_preparation(case.binding, case.paths)
    if not isinstance(error, Exception):
        assert captured.value is error
    else:
        assert str(captured.value) == "study_preparation_failed"
    assert (case.paths.attempt / "source-reconstruction.json").is_file()
    assert not (case.paths.attempt / "feasibility.json").exists()
    assert (
        json.loads((case.paths.attempt / "outcome.json").read_bytes())["status"]
        == "failed"
    )
    assert type(captured.value.preparation_progress) is bytes
    assert len(case.session.primary.scorer.urls) == 0


def test_bad_archive_is_integrity_failure_not_shortage(
    preparation_api, preparation_case
):
    case = preparation_case
    case.paths.archive.write_bytes(b"invalid-private-archive")
    with pytest.raises(preparation_api.StudyPreparationError):
        preparation_api._run_bound_preparation(case.binding, case.paths)
    assert (case.paths.attempt / "source-overlap.json").is_file()
    assert not (case.paths.attempt / "publisher-source.json").exists()
    assert not (case.paths.attempt / "feasibility.json").exists()


def test_final_binding_failure_retains_feasibility_without_completion(
    preparation_api, preparation_case, monkeypatch
):
    case, calls = preparation_case, []

    def checked(binding):
        calls.append(binding)
        if len(calls) == 2:
            raise ValueError("changed revision")

    monkeypatch.setattr(preparation_api, "recheck_binding", checked)
    with pytest.raises(preparation_api.StudyPreparationError):
        preparation_api._run_bound_preparation(case.binding, case.paths)
    assert (case.paths.attempt / "feasibility.json").is_file()
    assert not (case.paths.attempt / "preparation-complete.json").exists()


@pytest.mark.parametrize("field", ["source_csv", "suffix_rules", "archive"])
def test_input_cannot_be_inside_attempt(preparation_api, preparation_case, field):
    case = preparation_case
    paths = replace(case.paths, **{field: case.paths.attempt / "input"})
    with pytest.raises(preparation_api.StudyPreparationError):
        preparation_api._run_bound_preparation(case.binding, paths)
    assert not paths.attempt.exists()


def test_interruption_survives_retention_teardown_failure(
    preparation_api, preparation_case, monkeypatch
):
    case, interruption = preparation_case, KeyboardInterrupt()
    original = preparation_api.retain_study_preparation

    @contextmanager
    def failing_exit(*args, **kwargs):
        try:
            with original(*args, **kwargs) as writer:
                yield writer
        finally:
            raise ValueError("private teardown failure")

    def failed(*args, **kwargs):
        raise interruption

    monkeypatch.setattr(preparation_api, "retain_study_preparation", failing_exit)
    monkeypatch.setattr(preparation_api.body, "prepare_external", failed)
    with pytest.raises(KeyboardInterrupt) as captured:
        preparation_api._run_bound_preparation(case.binding, case.paths)
    assert captured.value is interruption


def test_new_failure_retention_interruption_is_not_swallowed(
    preparation_api, preparation_case, monkeypatch
):
    case, interruption = preparation_case, KeyboardInterrupt()

    def body_failed(*args, **kwargs):
        raise ValueError("original private failure")

    def interrupted(*args, **kwargs):
        raise interruption

    monkeypatch.setattr(preparation_api.body, "prepare_external", body_failed)
    monkeypatch.setattr(preparation_api, "record_failure", interrupted)
    with pytest.raises(KeyboardInterrupt) as captured:
        preparation_api._run_bound_preparation(case.binding, case.paths)
    assert captured.value is interruption
