"""Real SIGINT cannot strand new preparation resources before registration."""

import os
import signal
from contextlib import contextmanager

import pytest
from study_preparation_runner_fixtures import (
    inputs,
    preparation_api,
    preparation_case,
    runner,
)

__all__ = ["inputs", "preparation_api", "preparation_case", "runner"]


def close_remaining(descriptors, close):
    for descriptor in descriptors:
        try:
            os.fstat(descriptor)
        except OSError:
            continue
        close(descriptor)


@contextmanager
def signal_after_open(monkeypatch, target, enabled=None):
    original_open, original_close = os.open, os.close
    opened, closed = [], []

    def interrupted_open(name, flags, *arguments, **keywords):
        descriptor = original_open(name, flags, *arguments, **keywords)
        if name == target and not opened and (enabled is None or enabled()):
            opened.append(descriptor)
            signal.raise_signal(signal.SIGINT)
        return descriptor

    def tracked_close(descriptor):
        result = original_close(descriptor)
        if descriptor in opened:
            closed.append(descriptor)
        return result

    with monkeypatch.context() as guard:
        guard.setattr(os, "open", interrupted_open)
        guard.setattr(os, "close", tracked_close)
        try:
            yield opened, closed
        finally:
            close_remaining(opened, original_close)


def recheck_signal_scope(api, monkeypatch, source):
    if source not in {"recheck_entry", "recheck_final"}:
        return None
    active, calls = [], []
    selected = 1 if source == "recheck_entry" else 2

    def recheck(binding):
        calls.append(binding)
        active.append(len(calls) == selected)
        try:
            api.body.source_runner._read_file_once(binding.root / "data/sources.json")
        finally:
            active.pop()

    monkeypatch.setattr(api, "recheck_binding", recheck)
    return lambda: bool(active and active[-1])


@pytest.mark.parametrize(
    "source",
    [
        "suffix_rules",
        "source_csv",
        "archive",
        "public",
        "recheck_entry",
        "recheck_final",
    ],
)
def test_sigint_after_open_closes_root_preparation_input(
    preparation_api, preparation_case, monkeypatch, source
):
    case = preparation_case
    enabled = recheck_signal_scope(preparation_api, monkeypatch, source)
    target = (
        "sources.json"
        if source in {"public", "recheck_entry", "recheck_final"}
        else getattr(case.paths, source).name
    )
    with signal_after_open(monkeypatch, target, enabled) as (opened, closed):
        with pytest.raises(KeyboardInterrupt):
            preparation_api._run_bound_preparation(case.binding, case.paths)
        assert len(opened) == 1
        assert opened[0] in closed
        with pytest.raises(OSError):
            os.fstat(opened[0])
    assert "enter" not in case.events
    assert not case.session.primary.scorer.urls


def test_sigint_after_reservation_return_keeps_attempt_for_failure_record(
    preparation_api, preparation_case, monkeypatch
):
    case = preparation_case
    original = preparation_api.body.receipt.reserve_attempt
    reserved = []

    def interrupted_reservation(*arguments, **keywords):
        attempt = original(*arguments, **keywords)
        reserved.append(attempt)
        signal.raise_signal(signal.SIGINT)
        return attempt

    monkeypatch.setattr(
        preparation_api.body.receipt, "reserve_attempt", interrupted_reservation
    )
    with pytest.raises(KeyboardInterrupt):
        preparation_api._run_bound_preparation(case.binding, case.paths)
    assert len(reserved) == 1
    assert (case.paths.attempt / "reservation.json").is_file()
    assert (case.paths.attempt / "outcome.json").is_file()
    assert not (case.paths.attempt / "suffix-rules.dat").exists()


class InterruptedEntry:
    def __init__(self, manager):
        self.manager = manager
        self.writer = None

    def __enter__(self):
        self.writer = self.manager.__enter__()
        signal.raise_signal(signal.SIGINT)
        return self.writer

    def __exit__(self, *arguments):
        return self.manager.__exit__(*arguments)


def test_sigint_after_writer_entry_closes_and_retains_progress(
    preparation_api, preparation_case, monkeypatch
):
    case = preparation_case
    original = preparation_api.retain_study_preparation
    contexts = []

    def interrupted_entry(*arguments, **keywords):
        context = InterruptedEntry(original(*arguments, **keywords))
        contexts.append(context)
        return context

    monkeypatch.setattr(preparation_api, "retain_study_preparation", interrupted_entry)
    try:
        with pytest.raises(KeyboardInterrupt) as caught:
            preparation_api._run_bound_preparation(case.binding, case.paths)
        writer = contexts[0].writer
        assert writer._closed
        with pytest.raises(OSError):
            os.fstat(writer._directory.descriptor)
        assert type(caught.value.preparation_progress) is bytes
        assert not (case.paths.attempt / "suffix-rules.dat").exists()
    finally:
        for context in contexts:
            if context.writer is not None and not context.writer._closed:
                context.__exit__(KeyboardInterrupt, KeyboardInterrupt(), None)
