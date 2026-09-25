"""Failure bookkeeping does not inspect arbitrary exception behavior."""

import json
from types import SimpleNamespace

import pytest

from automated_phishing_detection.internal_failure import InternalFailureState


class PrivateError(ValueError):
    def __bool__(self):
        raise AssertionError("private exception truthiness must not run")


def _state():
    progress = SimpleNamespace(
        snapshot=lambda: b"{}", observe_counts=lambda scorer, **kwargs: None
    )
    return InternalFailureState(progress)


def _snapshot(state, error):
    attempt = SimpleNamespace(reservation_sha256="a" * 64)
    return json.loads(state.snapshot(attempt, {}, "scoring", error))


def test_snapshot_never_evaluates_original_exception_truthiness():
    state = _state()
    original = PrivateError("private-canary")
    with pytest.raises(PrivateError):
        with state.capture_body(None):
            raise original
    value = _snapshot(state, original)
    assert value["failure"] == "execution_failed"
    assert value["cleanup_failed"] is False


@pytest.mark.parametrize("session_closed", [False, True])
def test_cleanup_failure_requires_an_unclosed_session(session_closed):
    state = _state()
    with state.capture_body(None):
        pass
    state.session_closed = session_closed
    value = _snapshot(state, ValueError("private-canary"))
    assert value["cleanup_failed"] is not session_closed


@pytest.mark.parametrize("prior_interrupt", [False, True])
def test_first_counter_interruption_is_not_cleanup(prior_interrupt):
    original = KeyboardInterrupt() if prior_interrupt else ValueError()
    counter_error = KeyboardInterrupt()
    flags = []

    def observe(scorer, *, suppress_interruptions=False):
        flags.append(suppress_interruptions)
        if not suppress_interruptions:
            raise counter_error

    state = InternalFailureState(
        SimpleNamespace(snapshot=lambda: b"{}", observe_counts=observe)
    )
    expected = original if prior_interrupt else counter_error
    with pytest.raises(type(expected)) as caught:
        with state.capture_body(None):
            raise original
    assert caught.value is expected
    assert flags == [prior_interrupt]
    assert state.original_error is expected
    assert _snapshot(state, expected)["cleanup_failed"] is False
