"""Verifier unit lineage tests keep saved pure work immediately interruptible."""

import os
import signal
import sys

import pytest
from prepared_external_fixtures import (
    inputs,
    preparation_api,
    preparation_case,
    prepared_case,
    runner,
)
from test_prepared_external_completion import published
from test_prepared_external_io import _closed, _interrupt_open

from automated_phishing_detection import external_source_completion as completion
from automated_phishing_detection.owned_worker import observe_worker

__all__ = ["inputs", "preparation_api", "preparation_case", "prepared_case", "runner"]


def verify(case, observed, command):
    return completion.verify_external_completion_snapshot(
        case.binding,
        case.paths,
        expected_handoff=case.handoff,
        worker=observed,
        command=command,
        expected_preparation=case.preparation,
    )


def test_verifier_sigint_after_descriptor_open_closes_owned_fd(
    prepared_case, monkeypatch
):
    case = prepared_case
    published(case)
    command = (sys.executable, "-c", "pass")
    observed = observe_worker(command)
    first = KeyboardInterrupt("invented")
    descriptors, handler = _interrupt_open(case, monkeypatch, 1, first)
    previous = signal.signal(signal.SIGINT, handler)
    try:
        with pytest.raises(KeyboardInterrupt) as caught:
            verify(case, observed, command)
        assert caught.value is first
    finally:
        signal.signal(signal.SIGINT, previous)
    assert descriptors and all(_closed(descriptor) for descriptor in descriptors)


@pytest.mark.parametrize(
    "name", ["verify_external_provenance", "reconstruct_external_evidence"]
)
def test_saved_pure_reconstruction_never_defers_interrupt(
    prepared_case, monkeypatch, name
):
    case, calls = prepared_case, []
    published(case)
    command = (sys.executable, "-c", "pass")
    observed = observe_worker(command)
    first = KeyboardInterrupt("invented")

    def pure(*args, **kwargs):
        os.kill(os.getpid(), signal.SIGINT)
        calls.append("continued")

    def interrupted(signum, frame):
        raise first

    monkeypatch.setattr(completion, name, pure)
    previous = signal.signal(signal.SIGINT, interrupted)
    try:
        with pytest.raises(KeyboardInterrupt) as caught:
            verify(case, observed, command)
        assert caught.value is first
    finally:
        signal.signal(signal.SIGINT, previous)
    assert calls == []
