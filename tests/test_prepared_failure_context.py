"""Copy existing private associations without callbacks or replacement semantics."""

import importlib
import importlib.util

import pytest

NAMES = (
    "worker_failure",
    "external_failure",
    "source_internal",
    "progress",
    "preparation_progress",
)


def carry(error, original):
    name = "automated_phishing_detection._prepared_failure_context"
    assert importlib.util.find_spec(name), "missing private failure context copier"
    importlib.import_module(name).carry_failure_context(error, original)


@pytest.mark.parametrize("name", NAMES)
def test_missing_existing_context_is_copied_by_identity(name):
    original, later, value = ValueError(), KeyboardInterrupt(), object()
    setattr(original, name, value)
    original.unrelated = "not copied"
    carry(later, original)
    assert getattr(later, name) is value
    assert not hasattr(later, "unrelated")


@pytest.mark.parametrize("name", NAMES)
def test_existing_later_context_is_never_overwritten(name):
    original, later, value = ValueError(), SystemExit(), object()
    setattr(original, name, object())
    setattr(later, name, value)
    carry(later, original)
    assert getattr(later, name) is value


def test_exception_property_hooks_are_not_probed():
    class HostileError(ValueError):
        def __getattribute__(self, name):
            raise AssertionError("exception property was probed")

    original, later, value = HostileError(), HostileError(), object()
    descriptor = BaseException.__dict__["__dict__"]
    descriptor.__get__(original)["worker_failure"] = value
    carry(later, original)
    assert descriptor.__get__(later)["worker_failure"] is value


def test_no_body_failure_or_same_error_is_unchanged():
    original = KeyboardInterrupt()
    original.progress = b"immutable"
    carry(original, None)
    carry(original, original)
    assert original.progress == b"immutable"
