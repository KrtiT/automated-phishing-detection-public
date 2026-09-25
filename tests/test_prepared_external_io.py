"""Real interruption checks at borrowed descriptor-acquisition boundaries."""

import os
import signal

import pytest
from prepared_external_fixtures import (
    inputs,
    preparation_api,
    preparation_case,
    prepared_case,
    runner,
)

from automated_phishing_detection import execution_receipt as receipt
from automated_phishing_detection import external_source_runner as worker

__all__ = ["inputs", "preparation_api", "preparation_case", "prepared_case", "runner"]


def _interrupt_open(case, monkeypatch, ordinal, interruption):
    original, calls, descriptors = receipt._open_directory, [], []

    def opened(path):
        descriptor = original(path)
        if path == case.paths.attempt:
            calls.append(path)
            if len(calls) == ordinal:
                descriptors.append(descriptor)
                os.kill(os.getpid(), signal.SIGINT)
        return descriptor

    def delivered(signum, frame):
        raise interruption

    monkeypatch.setattr(receipt, "_open_directory", opened)
    return descriptors, delivered


def _closed(descriptor):
    try:
        os.fstat(descriptor)
    except OSError:
        return True
    os.close(descriptor)
    return False


@pytest.mark.parametrize("ordinal", [1, 2, 3])
def test_prepared_io_owns_fd_before_real_sigint_delivery(
    prepared_case, monkeypatch, ordinal
):
    case, interruption = prepared_case, KeyboardInterrupt("invented")
    descriptors, handler = _interrupt_open(case, monkeypatch, ordinal, interruption)
    previous = signal.signal(signal.SIGINT, handler)
    try:
        with pytest.raises(KeyboardInterrupt) as caught:
            worker._run_bound_prepared_external(
                case.binding,
                case.paths,
                handoff=case.handoff,
                preparation=case.preparation,
            )
        assert caught.value is interruption
    finally:
        signal.signal(signal.SIGINT, previous)
    assert descriptors and all(_closed(descriptor) for descriptor in descriptors)
    assert (case.paths.attempt / "outcome.json").is_file()
    assert not case.paths.public_summary.exists()
