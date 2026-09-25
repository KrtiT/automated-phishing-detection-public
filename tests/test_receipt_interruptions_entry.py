"""Descriptor ownership survives interrupted entry and stream construction."""

import asyncio
import os
from importlib import import_module

import pytest


def watch_files(monkeypatch, names):
    opened, closed, owned = [], [], set()
    open_file, close = os.open, os.close

    def observed_open(name, flags, *args, **kwargs):
        descriptor = open_file(name, flags, *args, **kwargs)
        if name in names:
            opened.append(descriptor)
            owned.add(descriptor)
        return descriptor

    def observed_close(descriptor):
        if descriptor in owned:
            owned.remove(descriptor)
            closed.append(descriptor)
        close(descriptor)

    monkeypatch.setattr(os, "open", observed_open)
    monkeypatch.setattr(os, "close", observed_close)
    return opened, closed


@pytest.mark.parametrize(
    "kind", (KeyboardInterrupt, asyncio.CancelledError, SystemExit)
)
def test_open_directory_closes_owned_parent_after_interruption(
    tmp_path, monkeypatch, kind
):
    receipt = import_module("automated_phishing_detection.execution_receipt")
    original, opened, closed = kind("open"), [], []
    open_file, close = os.open, os.close

    def interrupted(path, flags, **kwargs):
        if opened:
            raise original
        descriptor = open_file(path, flags, **kwargs)
        opened.append(descriptor)
        return descriptor

    def observed(descriptor):
        closed.append(descriptor)
        close(descriptor)

    monkeypatch.setattr(os, "open", interrupted)
    monkeypatch.setattr(os, "close", observed)
    with pytest.raises(BaseException) as caught:
        receipt._open_directory(tmp_path)
    assert caught.value is original
    assert closed == opened


def test_open_directory_close_failure_closes_child_without_retry(tmp_path, monkeypatch):
    receipt = import_module("automated_phishing_detection.execution_receipt")
    original, opened, closed = KeyboardInterrupt("close"), [], []
    open_file, close = os.open, os.close

    def observed_open(path, flags, **kwargs):
        descriptor = open_file(path, flags, **kwargs)
        opened.append(descriptor)
        return descriptor

    def interrupted_close(descriptor):
        closed.append(descriptor)
        close(descriptor)
        if len(closed) == 1:
            raise original
        raise OSError("later close")

    monkeypatch.setattr(os, "open", observed_open)
    monkeypatch.setattr(os, "close", interrupted_close)
    with pytest.raises(BaseException) as caught:
        receipt._open_directory(tmp_path)
    assert caught.value is original
    assert len(opened) == 2 and closed == opened


@pytest.mark.parametrize("operation", ("write", "read"))
def test_stream_construction_failure_closes_owned_descriptor(
    tmp_path, monkeypatch, operation
):
    receipt = import_module("automated_phishing_detection.execution_receipt")
    attempt = receipt.reserve_attempt(tmp_path / "attempt", identity={"fixture": 1})
    original = KeyboardInterrupt("fdopen")
    with receipt._directory(attempt.directory) as directory:

        def broken(*args, **kwargs):
            raise original

        with monkeypatch.context() as patch:
            opened, closed = watch_files(patch, {"new", "reservation.json"})
            patch.setattr(os, "fdopen", broken)
            with pytest.raises(BaseException) as caught:
                if operation == "write":
                    receipt._write_file(directory, "new", b"invented", 0o600)
                else:
                    receipt._read_reservation(directory)
        assert caught.value is original
        assert len(opened) == 1 and closed == opened
