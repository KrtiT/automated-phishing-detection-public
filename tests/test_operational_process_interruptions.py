"""Observed child cleanup must not replace the first parent interruption."""

import asyncio
import json

import pytest
from test_operational_process import assert_reaped, inputs, operational

__all__ = ["operational"]


@pytest.mark.parametrize("first", [KeyboardInterrupt("first"), SystemExit(17)])
@pytest.mark.parametrize("later", [OSError("later"), KeyboardInterrupt("later")])
def test_cleanup_failure_preserves_first_interruption_and_actual_exits(
    operational, tmp_path, monkeypatch, first, later
):
    attempt, options = inputs(tmp_path)
    original_run, original_cleanup = operational._run, operational._finish_cleanup

    async def interrupted(*arguments):
        await original_run(*arguments)
        raise first

    async def failed_cleanup(*arguments):
        await original_cleanup(*arguments)
        raise later

    monkeypatch.setattr(operational, "_run", interrupted)
    monkeypatch.setattr(operational, "_finish_cleanup", failed_cleanup)
    with pytest.raises(BaseException) as caught:
        asyncio.run(operational.observe_process_pair(attempt, **options))
    assert caught.value is first
    progress = json.loads(operational.process_progress(caught.value))
    assert progress["status"] == "failed"
    assert progress["client"]["exit_code"] == progress["service"]["exit_code"] == 0
    assert_reaped(progress)


@pytest.mark.parametrize("first", [KeyboardInterrupt("first"), SystemExit(17)])
@pytest.mark.parametrize("later", [OSError("later"), KeyboardInterrupt("later")])
def test_descriptor_cleanup_preserves_first_interruption_and_progress(
    operational, tmp_path, monkeypatch, first, later
):
    attempt, options = inputs(tmp_path)
    original_run = operational._run

    def close():
        raise later

    async def interrupted(children, *arguments):
        children.stack.callback(close)
        await original_run(children, *arguments)
        raise first

    monkeypatch.setattr(operational, "_run", interrupted)
    with pytest.raises(BaseException) as caught:
        asyncio.run(operational.observe_process_pair(attempt, **options))
    assert caught.value is first
    assert_reaped(json.loads(operational.process_progress(caught.value)))


@pytest.mark.parametrize("first", [KeyboardInterrupt("first"), SystemExit(17)])
@pytest.mark.parametrize("later", [OSError("later"), KeyboardInterrupt("later")])
def test_failed_setup_still_closes_listener_and_preserves_first_interruption(
    operational, tmp_path, monkeypatch, first, later
):
    attempt, options = inputs(tmp_path)
    listeners = []

    def close():
        raise later

    def interrupt(children):
        listeners.append(children.listener)
        children.stack.callback(close)
        raise first

    monkeypatch.setattr(operational.OwnedChildren, "_pipe", interrupt)
    with pytest.raises(BaseException) as caught:
        asyncio.run(operational.observe_process_pair(attempt, **options))
    assert caught.value is first and listeners[0].fileno() == -1
    progress = json.loads(operational.process_progress(caught.value))
    assert progress["failure"] == "process_setup_failed"
    assert progress["service"]["pid"] is progress["client"]["pid"] is None
