"""Diagnostic association must not swallow a newly delivered interruption."""

import os
import signal
from types import SimpleNamespace

import pytest

from automated_phishing_detection import _study_run_failure as failure


@pytest.mark.parametrize("phase", ["first", "second"])
def test_new_association_signal_is_not_replaced_by_ordinary_rejection(phase):
    original, marker = ValueError("ordinary"), object()
    calls = []

    def snapshot(error):
        calls.append(error)
        if len(calls) == (1 if phase == "first" else 2):
            os.kill(os.getpid(), signal.SIGINT)
        if phase == "second" and len(calls) == 1:
            raise ValueError("association unavailable")
        return marker

    state = SimpleNamespace(original=original, attempt=None, snapshot=snapshot)
    with pytest.raises(KeyboardInterrupt):
        failure.reject(state, original)


@pytest.mark.parametrize("original", [KeyboardInterrupt("first"), SystemExit(13)])
def test_first_nonexception_survives_every_later_association_signal(original):
    def snapshot(error):
        os.kill(os.getpid(), signal.SIGINT)

    state = SimpleNamespace(original=original, attempt=None, snapshot=snapshot)
    with pytest.raises(BaseException) as caught:
        failure.reject(state, original)
    assert caught.value is original
