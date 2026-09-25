"""Real SIGINT between stream allocation and registration retains ownership."""

import asyncio
import json
import os
import signal

import pytest
from test_operational_process import assert_reaped, inputs, operational

from automated_phishing_detection import _operational_process_children as children
from automated_phishing_detection._operational_process_records import Observations

__all__ = ["operational"]


@pytest.mark.parametrize("allocation", [1, 2])
def test_signal_after_temporary_stream_allocation_cannot_leak_stream(
    operational, tmp_path, monkeypatch, allocation
):
    attempt, options = inputs(tmp_path)
    original, streams = children.tempfile.TemporaryFile, []

    def interrupt(*arguments, **keywords):
        stream = original(*arguments, **keywords)
        streams.append(stream)
        if len(streams) == allocation:
            signal.raise_signal(signal.SIGINT)
        return stream

    monkeypatch.setattr(children.tempfile, "TemporaryFile", interrupt)
    try:
        with pytest.raises(KeyboardInterrupt):
            asyncio.run(operational.observe_process_pair(attempt, **options))
        assert len(streams) == allocation
        assert all(stream.closed for stream in streams)
        assert not (attempt.directory / "service-started.json").exists()
    finally:
        for stream in streams:
            stream.close()


@pytest.mark.parametrize("late_phase", ["finish", "descriptor"])
def test_repeated_real_signals_preserve_first_object_after_owned_cleanup(
    operational, tmp_path, monkeypatch, late_phase
):
    attempt, options = inputs(tmp_path)
    first, later = KeyboardInterrupt("first"), KeyboardInterrupt("later")
    signals = iter((first, later))
    previous = signal.getsignal(signal.SIGINT)

    def handler(number, frame):
        raise next(signals)

    signal.signal(signal.SIGINT, handler)
    install_signal_seams(operational, monkeypatch, late_phase)
    try:
        with pytest.raises(KeyboardInterrupt) as caught:
            asyncio.run(operational.observe_process_pair(attempt, **options))
        assert caught.value is first
        progress = json.loads(operational.process_progress(caught.value))
        assert progress["failure"] == "parent_interrupted"
        assert_reaped(progress)
    finally:
        signal.signal(signal.SIGINT, previous)


def install_signal_seams(operational, monkeypatch, late_phase):
    original_run, original_cleanup = operational._run, operational._finish_cleanup

    async def interrupted(children, *arguments):
        if late_phase == "descriptor":
            children.stack.callback(signal.raise_signal, signal.SIGINT)
        await original_run(children, *arguments)
        signal.raise_signal(signal.SIGINT)

    async def cleanup(*arguments):
        result = await original_cleanup(*arguments)
        if late_phase == "finish":
            signal.raise_signal(signal.SIGINT)
        return result

    monkeypatch.setattr(operational, "_run", interrupted)
    monkeypatch.setattr(operational, "_finish_cleanup", cleanup)


def test_signal_before_pipe_close_cannot_remove_ownership_before_close(
    operational, tmp_path, monkeypatch
):
    attempt, unused = inputs(tmp_path)
    observation = Observations(attempt, operational._record)
    owner = children.OwnedChildren(observation, {})
    descriptors, original = [], os.close
    try:
        with pytest.raises(KeyboardInterrupt):
            with owner:
                descriptors.extend(owner.descriptors)
                descriptors.append(owner.listener.fileno())
                interrupt_pipe_close(monkeypatch, owner.ready_read, original)
        for descriptor in descriptors:
            with pytest.raises(OSError):
                os.fstat(descriptor)
    finally:
        for descriptor in descriptors:
            try:
                original(descriptor)
            except OSError:
                pass


def interrupt_pipe_close(monkeypatch, target, original):
    signalled = []

    def close(descriptor):
        if descriptor == target and not signalled:
            signalled.append(descriptor)
            signal.raise_signal(signal.SIGINT)
        return original(descriptor)

    monkeypatch.setattr(os, "close", close)
