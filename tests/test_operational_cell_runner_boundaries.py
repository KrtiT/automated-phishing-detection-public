"""No cell substitution, input rewrites, publication retries or late acceptance."""

from contextlib import contextmanager
from dataclasses import replace

import pytest
from operational_cell_runner_fixtures import execute, orchestration, setup
from operational_input_fixtures import candidates, manifests

from automated_phishing_detection.evaluation_producer import ManifestOutcome
from automated_phishing_detection.operational_cell_inputs import (
    OperationalCapacityError,
)

__all__ = ["candidates", "manifests"]


def test_capacity_shortage_retains_exact_outcome_before_reservation(
    tmp_path, manifests, monkeypatch
):
    case = setup(tmp_path, manifests, monkeypatch, 21)
    outcome = ManifestOutcome("insufficient_capacity", None, 1, 100, 9)
    snapshot = replace(
        case.accepted.internal.snapshot, manifest_outcomes=((100, outcome),)
    )
    internal = replace(case.accepted.internal, snapshot=snapshot)
    case.accepted = replace(case.accepted, internal=internal)
    with pytest.raises(OperationalCapacityError) as caught:
        execute(case)
    assert caught.value.manifest_outcome is outcome
    assert caught.value.cell is case.cell and not case.paths.attempt.exists()
    assert caught.value.operational_failure.attempt is None


@pytest.mark.parametrize(
    "member", ["accepted_bytes", "descriptor_bytes", "binding_bytes", "manifest_bytes"]
)
def test_parent_compares_each_original_buffer_before_launch(
    tmp_path, manifests, monkeypatch, member
):
    case = setup(tmp_path, manifests, monkeypatch)
    orchestration(case, monkeypatch)
    original = case.module.hold_operational_inputs

    @contextmanager
    def changed(*arguments, **keywords):
        with original(*arguments, **keywords) as inputs:
            yield replace(inputs, **{member: b"different"})

    monkeypatch.setattr(case.module, "hold_operational_inputs", changed)
    with pytest.raises(case.module.OperationalCellExecutionError):
        execute(case)
    assert "observe" not in case.events


def test_mutable_writer_exits_before_completion(tmp_path, manifests, monkeypatch):
    case = setup(tmp_path, manifests, monkeypatch)
    orchestration(case, monkeypatch)
    original = case.module.held_attempt_writer

    @contextmanager
    def ordered(*arguments, **keywords):
        with original(*arguments, **keywords) as writer:
            yield writer
        case.events.append("writer_exit")

    monkeypatch.setattr(case.module, "held_attempt_writer", ordered)
    execute(case)
    assert case.events.index("writer_exit") < case.events.index("complete")


def test_failure_after_publication_starts_never_records_failure(
    tmp_path, manifests, monkeypatch
):
    case = setup(tmp_path, manifests, monkeypatch)
    orchestration(case, monkeypatch)

    def ambiguous(**keywords):
        case.completer.publishing = True
        case.completer.working = object()
        raise OSError("invented ambiguous publication")

    case.completer.complete = ambiguous
    with pytest.raises(case.module.OperationalCellExecutionError) as caught:
        execute(case)
    failure = caught.value.operational_failure
    assert failure.publishing and failure.working is case.completer.working
    assert failure.observation is case.observation and failure.candidate is None
    assert not (case.paths.attempt / "outcome.json").exists()
    assert case.events.count("observe") == 1


def test_late_input_mutation_rejects_completed_candidate(
    tmp_path, manifests, monkeypatch
):
    case = setup(tmp_path, manifests, monkeypatch)
    orchestration(case, monkeypatch)
    original = case.completer.complete

    def changed(**keywords):
        result = original(**keywords)
        (case.paths.cell_input_directory / "manifest").write_bytes(b"changed")
        return result

    case.completer.complete = changed
    with pytest.raises(case.module.OperationalCellExecutionError) as caught:
        execute(case)
    assert caught.value.operational_failure.candidate is case.snapshot
    assert caught.value.operational_failure.observation is case.observation
    assert not (case.paths.attempt / "outcome.json").exists()


def test_later_specific_failure_is_not_overwritten(tmp_path, manifests, monkeypatch):
    case = setup(tmp_path, manifests, monkeypatch)
    orchestration(case, monkeypatch)
    error = ValueError("late")
    error.operational_failure = object()

    def failed(**keywords):
        raise error

    case.completer.complete = failed
    with pytest.raises(case.module.OperationalCellExecutionError) as caught:
        execute(case)
    assert caught.value.operational_failure is error.operational_failure
