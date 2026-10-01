"""New channel setup inherits deferred-signal and exception-preserving cleanup."""

import os
import signal

import pytest
from study_series_admission_fixtures import api, frame


@pytest.mark.parametrize("boundary", ["pipe", "set_blocking", "write"])
def test_setup_interrupt_closes_every_allocated_end(monkeypatch, boundary):
    allocated = []
    original_pipe = os.pipe
    original = getattr(os, boundary)

    def pipe():
        descriptors = original_pipe()
        allocated.extend(descriptors)
        if boundary == "pipe":
            signal.raise_signal(signal.SIGINT)
        return descriptors

    def interrupt(*args):
        result = original(*args)
        signal.raise_signal(signal.SIGINT)
        return result

    monkeypatch.setattr(os, "pipe", pipe)
    if boundary != "pipe":
        monkeypatch.setattr(os, boundary, interrupt)
    with pytest.raises(KeyboardInterrupt):
        with api().SeriesParentAdmission(frame()):
            pytest.fail("interrupted setup launched a child")
    for descriptor in allocated:
        with pytest.raises(OSError):
            os.fstat(descriptor)


def test_mode_setup_failure_closes_both_ends(monkeypatch):
    allocated = []
    original = os.pipe

    def pipe():
        created = original()
        allocated.extend(created)
        return created

    def failure(*args):
        raise OSError("invented mode error")

    monkeypatch.setattr(os, "pipe", pipe)
    monkeypatch.setattr(os, "set_blocking", failure)
    with pytest.raises(OSError, match="invented mode error"):
        with api().SeriesParentAdmission(frame()):
            pytest.fail("failed setup launched a child")
    for descriptor in allocated:
        with pytest.raises(OSError):
            os.fstat(descriptor)


@pytest.mark.parametrize("error", [KeyboardInterrupt(), SystemExit(17)])
def test_callback_interrupt_retains_first_error_and_closes_channel(error):
    def interrupt(unused):
        raise error

    admission = api().SeriesParentAdmission(frame(), on_launched=interrupt)
    with pytest.raises(BaseException) as caught:
        with admission:
            admission.launched(os.getpid() + 100000)
    assert caught.value is error
    assert admission.pid == os.getpid() + 100000
    assert admission.pipe.descriptors == set()
