"""Invented live pipe frames cannot be replaced by retained files or metadata."""

import os
import sys
import time
from importlib import import_module
from importlib.util import find_spec
from types import SimpleNamespace

import pytest


def api():
    name = "automated_phishing_detection.study_series_supervision"
    assert find_spec(name), "missing live physical supervision consumer"
    return import_module(name)


def arguments():
    return SimpleNamespace(
        expected_revision="a" * 40,
        expected_profile_sha256="b" * 64,
        expected_envelope_sha256="c" * 64,
    )


def channel(monkeypatch):
    module = api()
    controls = import_module(
        "automated_phishing_detection._study_series_supervisor_channel"
    )
    selected = controls._open()
    consumer_descriptor = os.dup(selected.read_fd)
    selected.launched()
    frame = controls.admission(arguments(), os.getpid(), (sys.executable, *sys.argv))
    frame["parent_pid"] = os.getppid()
    monkeypatch.setenv(module.LOCATOR, str(consumer_descriptor))
    return module, selected, frame


def test_missing_channel_fails_without_any_public_binding(monkeypatch):
    module = api()
    monkeypatch.delenv(module.LOCATOR, raising=False)
    with pytest.raises(ValueError):
        with module.consume_series_supervision(arguments()):
            pytest.fail("missing live channel yielded")


def test_live_channel_drains_five_minutes_of_heartbeats_before_age_check(monkeypatch):
    module, selected, frame = channel(monkeypatch)
    try:
        frame["monotonic_ns"] = time.monotonic_ns() - 360_000_000_000
        selected.release(frame)
        for offset in range(1, 13):
            selected.heartbeat(frame["monotonic_ns"] + offset * 30_000_000_000)
        with module.consume_series_supervision(arguments()) as held:
            held.check()
    finally:
        selected.close()


@pytest.mark.parametrize(
    "field",
    (
        "root_pid",
        "parent_pid",
        "profile_sha256",
        "envelope_sha256",
        "revision",
        "command_sha256",
    ),
)
def test_every_initial_identity_is_required(monkeypatch, field):
    module, selected, frame = channel(monkeypatch)
    try:
        frame[field] = (
            frame[field] + 1 if type(frame[field]) is int else "0" * len(frame[field])
        )
        selected.release(frame)
        with pytest.raises(ValueError):
            with module.consume_series_supervision(arguments()):
                pytest.fail("mismatched live identity yielded")
    finally:
        selected.close()


def test_closed_writer_is_rejected_even_with_queued_fresh_heartbeat(monkeypatch):
    module, selected, frame = channel(monkeypatch)
    try:
        selected.release(frame)
        selected.heartbeat(time.monotonic_ns())
        selected.revoke()
        with pytest.raises(ValueError):
            with module.consume_series_supervision(arguments()):
                pytest.fail("closed supervisor yielded")
    finally:
        selected.close()


def test_late_channel_revocation_fails_holder_exit(monkeypatch):
    module, selected, frame = channel(monkeypatch)
    try:
        selected.release(frame)
        with pytest.raises(ValueError):
            with module.consume_series_supervision(arguments()):
                selected.revoke()
    finally:
        selected.close()


@pytest.mark.parametrize(
    "mutation", ("extra_field", "boolean_pid", "stale", "legacy_env")
)
def test_malformed_or_stale_context_rejects(monkeypatch, mutation):
    module, selected, frame = channel(monkeypatch)
    try:
        if mutation == "extra_field":
            frame["additional"] = True
        elif mutation == "boolean_pid":
            frame["root_pid"] = True
        elif mutation == "stale":
            frame["monotonic_ns"] -= 100_000_000_000
        else:
            monkeypatch.setenv("APD_STUDY_ADMISSION_FD", "123")
        selected.release(frame)
        with pytest.raises(ValueError):
            with module.consume_series_supervision(arguments()):
                pytest.fail("invalid supervision yielded")
    finally:
        selected.close()


def test_full_pipe_fails_without_blocking(monkeypatch):
    module, selected, frame = channel(monkeypatch)
    try:
        selected.release(frame)
        for unused in range(65536):
            try:
                selected.heartbeat(time.monotonic_ns())
            except ValueError:
                break
        else:
            pytest.fail("bounded nonblocking pipe never rejected")
    finally:
        selected.close()
        os.close(int(os.environ[module.LOCATOR]))
