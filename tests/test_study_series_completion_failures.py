"""Isolated IO failures never retry, publish missing science, or hide interrupts."""

import signal

import pytest
from study_series_completion_fixtures import holder, module, setup, write_working

from automated_phishing_detection._operational_cell_protocol import WORKING_NAMES


def _failure(monkeypatch, phase):
    target = module().receipt if phase.startswith("publish_") else module()
    name = "publish_completion" if phase.startswith("publish_") else phase
    original, calls = getattr(target, name), []

    def reject(*arguments, **keywords):
        calls.append(phase)
        if phase == "publish_after":
            original(*arguments, **keywords)
        raise ValueError("invented private failure")

    monkeypatch.setattr(target, name, reject)
    return calls


@pytest.mark.parametrize(
    "phase",
    [
        "_verify_working",
        "_build_public",
        "publish_before",
        "publish_after",
        "_verify_published",
    ],
)
def test_failed_completion_cannot_retry_inside_or_after_holder(
    tmp_path, monkeypatch, phase
):
    case = setup(tmp_path, monkeypatch)
    calls = _failure(monkeypatch, phase)
    with pytest.raises(module().OperationalCellCompletionError):
        with holder(case) as completer:
            write_working(case)
            for unused in range(2):
                with pytest.raises(module().OperationalCellCompletionError):
                    completer.complete(**case.options)
    with pytest.raises(module().OperationalCellCompletionError):
        completer.complete(**case.options)
    assert calls == [phase]
    assert completer.candidate is None
    assert (completer.working is None) is (phase == "_verify_working")
    assert completer.publishing is (phase not in ("_verify_working", "_build_public"))
    assert case.public.exists() is (phase in ("publish_after", "_verify_published"))


def test_scientific_failure_retains_only_exact_working_inventory(tmp_path, monkeypatch):
    case = setup(tmp_path, monkeypatch)
    _failure(monkeypatch, "_verify_working")
    with pytest.raises(module().OperationalCellCompletionError):
        with holder(case) as completer:
            write_working(case)
            completer.complete(**case.options)
    assert set(path.name for path in case.attempt.directory.iterdir()) == set(
        WORKING_NAMES
    )
    assert not case.public.exists()
    assert not (case.attempt.directory / "finalize.claim").exists()


@pytest.mark.parametrize("first", [KeyboardInterrupt("first"), SystemExit(7)])
@pytest.mark.parametrize(
    "later", [OSError("cleanup"), KeyboardInterrupt("cleanup"), SystemExit(9)]
)
def test_original_interruption_object_survives_holder_failure(
    tmp_path, monkeypatch, first, later
):
    case = setup(tmp_path, monkeypatch)
    first.progress = b"original immutable progress"

    def fail(*arguments):
        raise later

    with pytest.raises(BaseException) as caught:
        with holder(case) as completer:
            monkeypatch.setattr(module().storage.HeldCellFiles, "check", fail)
            raise first
    assert caught.value is first
    assert caught.value.progress == b"original immutable progress"
    with pytest.raises(module().OperationalCellCompletionError):
        completer.complete(**case.options)
    assert not case.calls


def test_late_interruption_carries_original_failure_context_and_candidate(
    tmp_path, monkeypatch
):
    case = setup(tmp_path, monkeypatch)
    first, later = ValueError("body"), KeyboardInterrupt("cleanup")
    first.progress = b"retained process bytes"
    first.operational_failure = object()

    def fail(*arguments):
        raise later

    with pytest.raises(KeyboardInterrupt) as caught:
        with holder(case) as completer:
            write_working(case)
            candidate = completer.complete(**case.options)
            monkeypatch.setattr(module().storage.HeldCellFiles, "check", fail)
            raise first
    assert caught.value is later
    assert caught.value.progress is first.progress
    assert caught.value.operational_failure is first.operational_failure
    assert completer.candidate is candidate and completer.working is not None
    assert case.public.exists()


def test_holder_failure_keeps_candidate_but_closes_completion(tmp_path, monkeypatch):
    case = setup(tmp_path, monkeypatch)

    def fail(*arguments):
        raise OSError("held identity changed")

    with pytest.raises(module().OperationalCellCompletionError):
        with holder(case) as completer:
            write_working(case)
            candidate = completer.complete(**case.options)
            monkeypatch.setattr(module().storage.HeldCellFiles, "check", fail)
    with pytest.raises(module().OperationalCellCompletionError):
        completer.complete(**case.options)
    assert completer.candidate is candidate
    assert completer.publishing and case.public.exists()
    assert [call[0] for call in case.calls] == ["working", "published"]


def test_partial_child_failure_keeps_original_exception_and_bytes(
    tmp_path, monkeypatch
):
    case = setup(tmp_path, monkeypatch)
    original = ValueError("invented child failure")
    with pytest.raises(ValueError) as caught:
        with holder(case) as completer:
            (case.attempt.directory / "run.json").write_bytes(b"partial")
            raise original
    assert caught.value is original
    assert (case.attempt.directory / "run.json").read_bytes() == b"partial"
    assert completer.working is completer.candidate is None
    assert not completer.publishing and not case.public.exists()


@pytest.mark.parametrize("phase", ["_verify_working", "_verify_published"])
def test_scientific_interruption_is_not_wrapped_or_retryable(
    tmp_path, monkeypatch, phase
):
    case = setup(tmp_path, monkeypatch)
    interruption = KeyboardInterrupt("invented science interruption")

    def interrupt(*arguments, **keywords):
        raise interruption

    monkeypatch.setattr(module(), phase, interrupt)
    with pytest.raises(KeyboardInterrupt) as caught:
        with holder(case) as completer:
            write_working(case)
            completer.complete(**case.options)
    assert caught.value is interruption
    with pytest.raises(module().OperationalCellCompletionError):
        completer.complete(**case.options)


@pytest.mark.parametrize("phase", ["_verify_working", "_verify_published"])
def test_actual_signal_interrupts_scientific_validation_immediately(
    tmp_path, monkeypatch, phase
):
    case = setup(tmp_path, monkeypatch)
    reached = []

    def interrupt(*arguments, **keywords):
        signal.raise_signal(signal.SIGINT)
        reached.append("signal deferred past validation")

    monkeypatch.setattr(module(), phase, interrupt)
    with pytest.raises(KeyboardInterrupt):
        with holder(case) as completer:
            write_working(case)
            completer.complete(**case.options)
    assert not reached
    assert completer.publishing is (phase == "_verify_published")
