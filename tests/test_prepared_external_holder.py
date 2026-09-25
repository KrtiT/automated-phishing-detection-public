"""Retained reader ownership is registered without deferring pure restoration."""

import os
import signal
from contextlib import contextmanager

import pytest
from test_external_source_process_commands import command_case

from automated_phishing_detection import _prepared_external_runtime as runtime
from automated_phishing_detection._external_source_records import (
    PreparedExternalRunPaths,
)


def holder_case(monkeypatch):
    binding, old, unused = command_case()
    paths = PreparedExternalRunPaths(
        old.archive,
        old.artifacts,
        old.secondary_artifacts,
        old.drift_artifacts,
        old.attempt,
        old.public_summary,
    )
    buffers = {
        "data/sources.json": b"source",
        "reports/phiusiil-preparation-summary.json": b"report",
    }
    monkeypatch.setattr(
        runtime, "bound_preparation_context", lambda _: ({}, {}, buffers, None)
    )
    return binding, paths


def test_sigint_inside_pure_restoration_is_immediate(monkeypatch):
    binding, paths = holder_case(monkeypatch)
    calls, first = [], KeyboardInterrupt("invented")

    @contextmanager
    def restore(*args, **kwargs):
        try:
            os.kill(os.getpid(), signal.SIGINT)
            calls.append("continued_restoration")
            yield object()
        finally:
            calls.append("closed")

    def interrupted(signum, frame):
        raise first

    monkeypatch.setattr(runtime, "hold_study_preparation", restore)
    previous = signal.signal(signal.SIGINT, interrupted)
    try:
        with pytest.raises(KeyboardInterrupt) as captured:
            with runtime.held_preparation(binding, paths, "a" * 64, "b" * 64):
                pytest.fail("interrupted restoration yielded")
        assert captured.value is first
    finally:
        signal.signal(signal.SIGINT, previous)
    assert calls == ["closed"]


def test_failed_entry_preserves_original_failure(monkeypatch):
    binding, paths = holder_case(monkeypatch)
    first = ValueError("original")

    @contextmanager
    def restore(*args, **kwargs):
        raise first
        yield

    monkeypatch.setattr(runtime, "hold_study_preparation", restore)
    with pytest.raises(ValueError) as captured:
        with runtime.held_preparation(binding, paths, "a" * 64, "b" * 64):
            pytest.fail("failed restoration yielded")
    assert captured.value is first
