"""Outer-holder failures retain prior evidence without asserting acceptance."""

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
from test_prepared_external_holder import holder_case
from test_prepared_external_pair import paths

from automated_phishing_detection import _prepared_external_runtime as runtime
from automated_phishing_detection import prepared_external_process as process


@pytest.mark.parametrize(
    "name",
    [
        "worker_failure",
        "external_failure",
        "source_internal",
        "progress",
        "preparation_progress",
    ],
)
def test_replacing_holder_interruption_keeps_body_evidence(monkeypatch, name):
    binding, run_paths = holder_case(monkeypatch)
    original, interruption, retained = (
        ValueError("body"),
        KeyboardInterrupt("cleanup"),
        object(),
    )
    setattr(original, name, retained)

    @contextmanager
    def restore(*args, **kwargs):
        try:
            yield object()
        finally:
            raise interruption

    monkeypatch.setattr(runtime, "hold_study_preparation", restore)
    with pytest.raises(KeyboardInterrupt) as caught:
        with runtime.held_preparation(binding, run_paths, "a" * 64, "b" * 64):
            raise original
    assert caught.value is interruption
    assert getattr(interruption, name, None) is retained


def completed_pair():
    external = SimpleNamespace(
        worker=SimpleNamespace(command_sha256="a" * 64), snapshot=object()
    )
    return SimpleNamespace(
        preparation=object(), internal=object(), external=external, handoff=object()
    )


def test_late_holder_rejection_retains_completed_pair(monkeypatch):
    binding, internal_paths, external_paths = paths()
    first, candidate = ValueError("changed"), completed_pair()

    @contextmanager
    def held(*args):
        yield candidate.preparation
        raise first

    monkeypatch.setattr(process, "held_preparation", held)
    monkeypatch.setattr(
        process, "_run_observed_prepared_sources", lambda *args: candidate
    )
    assert callable(getattr(process, "_run_held_sources", None))
    with pytest.raises(ValueError) as caught:
        process._run_held_sources(
            binding, internal_paths, external_paths, "a" * 64, "b" * 64
        )
    assert caught.value is first
    assert first.source_internal is candidate.internal
    assert first.external_failure.worker is candidate.external.worker
    assert first.external_failure.candidate_snapshot is candidate.external.snapshot
    assert first.external_failure.stage == "preparation_finalization"


def replacement_case(monkeypatch, later_specific):
    first = ValueError("internal")
    later, preparation = KeyboardInterrupt("holder"), object()
    first.worker_failure = object()
    retained = object() if later_specific else preparation
    if later_specific:
        later.study_preparation = retained

    @contextmanager
    def held(*args):
        try:
            yield preparation
        finally:
            raise later

    def internal(*args, **kwargs):
        raise first

    monkeypatch.setattr(runtime, "_reader", held)
    monkeypatch.setattr(process, "_run_observed_prepared_internal", internal)
    return first, later, preparation, retained


@pytest.mark.parametrize("later_specific", [False, True])
def test_internal_failure_keeps_preparation_after_holder_interruption(
    monkeypatch, later_specific
):
    binding, internal_paths, external_paths = paths()
    first, later, preparation, retained = replacement_case(monkeypatch, later_specific)
    with pytest.raises(KeyboardInterrupt) as caught:
        process._run_held_sources(
            binding, internal_paths, external_paths, "a" * 64, "b" * 64
        )
    assert caught.value is later
    assert first.study_preparation is preparation
    assert later.worker_failure is first.worker_failure
    assert getattr(later, "study_preparation", None) is retained


def test_failed_holder_entry_does_not_invent_preparation(monkeypatch):
    binding, internal_paths, external_paths = paths()
    first = ValueError("entry")

    @contextmanager
    def failed(*args):
        raise first
        yield

    monkeypatch.setattr(runtime, "_reader", failed)
    with pytest.raises(ValueError) as caught:
        process._run_held_sources(
            binding, internal_paths, external_paths, "a" * 64, "b" * 64
        )
    assert caught.value is first
    assert "study_preparation" not in vars(first)


def test_completed_pair_keeps_later_specific_external_failure(monkeypatch):
    binding, internal_paths, external_paths = paths()
    first, retained, candidate = ValueError("cleanup"), object(), completed_pair()
    first.external_failure = retained

    @contextmanager
    def held(*args):
        yield candidate.preparation
        raise first

    monkeypatch.setattr(process, "held_preparation", held)
    monkeypatch.setattr(
        process, "_run_observed_prepared_sources", lambda *args: candidate
    )
    with pytest.raises(ValueError) as caught:
        process._run_held_sources(
            binding, internal_paths, external_paths, "a" * 64, "b" * 64
        )
    assert caught.value is first
    assert first.external_failure is retained
    assert first.study_preparation is candidate.preparation
    assert first.source_internal is candidate.internal
