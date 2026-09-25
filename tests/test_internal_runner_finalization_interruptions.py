"""Failure persistence cannot erase a newly observed interruption."""

import asyncio
import json
from types import ModuleType

import pytest
from test_source_runner import inputs as inputs
from test_source_runner import runner as runner


def _fail_production(
    monkeypatch: pytest.MonkeyPatch, runner: ModuleType, error: BaseException
) -> None:
    def fail(*args: object, **kwargs: object) -> None:
        raise error

    monkeypatch.setattr(runner.evaluation_producer, "produce_internal_evidence", fail)


@pytest.mark.parametrize(
    "kind", [asyncio.CancelledError, KeyboardInterrupt, SystemExit]
)
@pytest.mark.parametrize("target", ["retain_failure_progress", "record_failure"])
@pytest.mark.parametrize("after_install", [False, True])
def test_first_interruption_during_failure_persistence_is_preserved(
    runner: ModuleType,
    inputs: tuple,
    monkeypatch: pytest.MonkeyPatch,
    kind: type[BaseException],
    target: str,
    after_install: bool,
) -> None:
    binding, paths, unused_session, unused_events = inputs
    interruption = kind("private-finalization-canary")
    original = getattr(runner, target)
    calls = []
    _fail_production(monkeypatch, runner, ValueError("private-scoring-canary"))

    def interrupted(*args: object, **kwargs: object) -> None:
        calls.append(target)
        if after_install:
            original(*args, **kwargs)
        raise interruption

    monkeypatch.setattr(runner, target, interrupted)
    with pytest.raises(kind) as caught:
        runner._run_bound_internal(binding, paths)
    assert caught.value is interruption
    assert calls == [target]
    assert "private" not in str(caught.value)
    assert json.loads(caught.value.progress)["status"] == "failed"
    assert not paths.public_summary.exists()


@pytest.mark.parametrize("target", ["retain_failure_progress", "record_failure"])
@pytest.mark.parametrize(
    "kind", [asyncio.CancelledError, KeyboardInterrupt, SystemExit]
)
def test_existing_interruption_wins_over_later_persistence_interruption(
    runner: ModuleType,
    inputs: tuple,
    monkeypatch: pytest.MonkeyPatch,
    target: str,
    kind: type[BaseException],
) -> None:
    binding, paths, unused_session, unused_events = inputs
    original = kind("private-original-canary")
    later = KeyboardInterrupt("private-later-canary")
    calls = []
    _fail_production(monkeypatch, runner, original)

    def interrupted(*args: object, **kwargs: object) -> None:
        calls.append(target)
        raise later

    monkeypatch.setattr(runner, target, interrupted)
    with pytest.raises(kind) as caught:
        runner._run_bound_internal(binding, paths)
    assert caught.value is original
    assert calls == [target]
    assert "private" not in str(caught.value)
    assert json.loads(caught.value.progress)["status"] == "failed"
    assert not paths.public_summary.exists()


@pytest.mark.parametrize(
    "kind", [asyncio.CancelledError, KeyboardInterrupt, SystemExit]
)
def test_publication_interruption_never_starts_replacement_failure_finalization(
    runner: ModuleType,
    inputs: tuple,
    monkeypatch: pytest.MonkeyPatch,
    kind: type[BaseException],
) -> None:
    binding, paths, unused_session, unused_events = inputs
    interruption = kind("private-publication-canary")
    publications = []

    def interrupt_publication(*args: object, **kwargs: object) -> None:
        publications.append(True)
        raise interruption

    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail("publication failure started replacement finalization")

    monkeypatch.setattr(runner, "publish_completion", interrupt_publication)
    monkeypatch.setattr(runner, "retain_failure_progress", forbidden)
    monkeypatch.setattr(runner, "record_failure", forbidden)
    with pytest.raises(kind) as caught:
        runner._run_bound_internal(binding, paths)
    assert caught.value is interruption
    assert publications == [True]
    assert "private" not in str(caught.value)
    assert not (paths.attempt / "failure-progress.json").exists()
    assert not paths.public_summary.exists()
