"""Cleanup preserves the first interruption while attempting every callback."""

import asyncio
from importlib import import_module

import pytest

KINDS = (OSError, KeyboardInterrupt, asyncio.CancelledError, SystemExit)


def fail(error, calls, name):
    calls.append(name)
    raise error


@pytest.mark.parametrize("original_kind", KINDS)
@pytest.mark.parametrize("cleanup_kind", KINDS)
def test_cleanup_preserves_exception_priority(original_kind, cleanup_kind):
    module = import_module("automated_phishing_detection._exception_cleanup")
    original, later, calls = original_kind("first"), cleanup_kind("later"), []
    with pytest.raises(BaseException) as caught:
        with module.preserve_cleanup(lambda: fail(later, calls, "close")):
            raise original
    expected = later if isinstance(original, Exception) else original
    assert caught.value is expected
    assert calls == ["close"]


@pytest.mark.parametrize("cleanup_kind", KINDS)
def test_cleanup_failure_after_success_still_propagates(cleanup_kind):
    module = import_module("automated_phishing_detection._exception_cleanup")
    later, calls = cleanup_kind("later"), []
    with pytest.raises(BaseException) as caught:
        with module.preserve_cleanup(lambda: fail(later, calls, "close")):
            pass
    assert caught.value is later
    assert calls == ["close"]


@pytest.mark.parametrize("original_kind", (None, *KINDS))
def test_exit_stack_attempts_all_callbacks_and_keeps_first_interrupt(original_kind):
    module = import_module("automated_phishing_detection._exception_cleanup")
    original = original_kind("body") if original_kind else None
    interrupt, repeated, calls = KeyboardInterrupt("first"), SystemExit(12), []
    with pytest.raises(BaseException) as caught:
        with module.CleanupStack() as stack:
            stack.callback(fail, OSError("last"), calls, "last")
            stack.callback(fail, repeated, calls, "repeated")
            stack.callback(fail, interrupt, calls, "first")
            if original is not None:
                raise original
    expected = (
        original if original and not isinstance(original, Exception) else interrupt
    )
    assert caught.value is expected
    assert calls == ["first", "repeated", "last"]


def test_close_stack_attempts_all_callbacks():
    module = import_module("automated_phishing_detection._exception_cleanup")
    original, calls = SystemExit(9), []
    stack = module.CleanupStack()
    stack.callback(fail, OSError("later"), calls, "later")
    stack.callback(fail, original, calls, "first")
    with pytest.raises(BaseException) as caught:
        stack.close()
    assert caught.value is original
    assert calls == ["first", "later"]


def test_successful_cleanup_does_not_change_body_or_return():
    module = import_module("automated_phishing_detection._exception_cleanup")
    calls = []
    with module.preserve_cleanup(lambda: calls.append("closed")):
        calls.append("body")
    assert calls == ["body", "closed"]


class UnstableSuppression:
    def __init__(self, mode):
        self.mode, self.calls = mode, 0

    def __bool__(self):
        self.calls += 1
        if self.mode == "raises":
            raise OSError("suppression truthiness")
        return self.calls > 1


@pytest.mark.parametrize(
    "kind", (KeyboardInterrupt, asyncio.CancelledError, SystemExit)
)
@pytest.mark.parametrize("mode", ("raises", "changes"))
def test_exit_return_cannot_erase_interruption(kind, mode):
    module = import_module("automated_phishing_detection._exception_cleanup")
    original, suppression, calls = kind("body"), UnstableSuppression(mode), []
    with pytest.raises(BaseException) as caught:
        with module.CleanupStack() as stack:
            stack.callback(calls.append, "last callback")
            stack.push(lambda *arguments: suppression)
            raise original
    assert caught.value is original
    assert suppression.calls == 0
    assert calls == ["last callback"]


def test_ordinary_exception_interprets_suppression_once():
    module = import_module("automated_phishing_detection._exception_cleanup")
    original, suppression = ValueError("body"), UnstableSuppression("changes")
    with pytest.raises(ValueError) as caught:
        with module.CleanupStack() as stack:
            stack.push(lambda *arguments: suppression)
            raise original
    assert caught.value is original
    assert suppression.calls == 1


def test_ordinary_exception_can_still_be_suppressed():
    module = import_module("automated_phishing_detection._exception_cleanup")
    calls = []
    with module.CleanupStack() as stack:
        stack.callback(calls.append, "last callback")
        stack.push(lambda *arguments: True)
        raise ValueError("ordinary body error")
    assert calls == ["last callback"]
