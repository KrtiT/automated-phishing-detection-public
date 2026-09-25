"""Interruption precedence at real invented-file stream boundaries."""

import asyncio
import os
from importlib import import_module

import pytest
from test_receipt_interruptions_entry import watch_files


class BrokenStream:
    def __init__(self, stream, original, cleanup):
        self.stream, self.original, self.cleanup = stream, original, cleanup

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.stream.close()
        raise self.cleanup

    def fileno(self):
        return self.stream.fileno()

    def read(self):
        raise self.original

    def write(self, content):
        raise self.original


@pytest.mark.parametrize(
    "kind", (KeyboardInterrupt, asyncio.CancelledError, SystemExit)
)
@pytest.mark.parametrize("operation", ("write", "reservation", "source"))
def test_stream_interruption_survives_close_and_releases_descriptor(
    tmp_path, monkeypatch, kind, operation
):
    receipt = import_module("automated_phishing_detection.execution_receipt")
    runner = import_module("automated_phishing_detection.source_runner")
    attempt = receipt.reserve_attempt(tmp_path / "attempt", identity={"fixture": 1})
    original, later, fdopen = kind("io"), KeyboardInterrupt("close"), os.fdopen
    with receipt._directory(attempt.directory) as directory:
        with monkeypatch.context() as patch:
            opened, closed = watch_files(patch, {"new", "reservation.json"})
            patch.setattr(
                os,
                "fdopen",
                lambda *args, **kwargs: BrokenStream(
                    fdopen(*args, **kwargs), original, later
                ),
            )
            with pytest.raises(BaseException) as caught:
                if operation == "write":
                    receipt._write_file(directory, "new", b"invented", 0o600)
                elif operation == "reservation":
                    receipt._read_reservation(directory)
                else:
                    runner._read_file_once(attempt.directory / "reservation.json")
        assert caught.value is original
        assert len(opened) == 1 and closed == opened


@pytest.mark.parametrize(
    "kind", (KeyboardInterrupt, asyncio.CancelledError, SystemExit)
)
def test_directory_identity_check_preserves_interrupted_stat(
    tmp_path, monkeypatch, kind
):
    receipt = import_module("automated_phishing_detection.execution_receipt")
    original, closed = kind("identity"), []
    close = os.close
    with receipt._directory(tmp_path) as directory:

        def interrupted(value):
            raise original

        def broken_close(descriptor):
            close(descriptor)
            closed.append(descriptor)
            if len(closed) > len(tmp_path.parts) - 1:
                raise OSError("close")

        with monkeypatch.context() as patch:
            patch.setattr(receipt, "_identity", interrupted)
            patch.setattr(os, "close", broken_close)
            with pytest.raises(BaseException) as caught:
                directory.check()
        assert caught.value is original
        assert len(closed) == len(tmp_path.parts)
