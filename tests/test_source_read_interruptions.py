"""Once-read source descriptors preserve interruptions using invented files."""

import asyncio
import os
from importlib import import_module

import pytest
from test_receipt_interruptions_entry import watch_files


def interrupt_file_close(monkeypatch, closed):
    close = os.close

    def broken(descriptor):
        before = len(closed)
        close(descriptor)
        if len(closed) != before:
            raise OSError("close")

    monkeypatch.setattr(os, "close", broken)


@pytest.mark.parametrize(
    "kind", (KeyboardInterrupt, asyncio.CancelledError, SystemExit)
)
def test_source_read_interruption_wins_over_descriptor_close(
    tmp_path, monkeypatch, kind
):
    runner = import_module("automated_phishing_detection.source_runner")
    path = tmp_path / "invented.txt"
    path.write_bytes(b"invented")
    original = kind("read")
    opened, closed = watch_files(monkeypatch, {path.name})

    def broken_stream(*args, **kwargs):
        raise original

    monkeypatch.setattr(os, "fdopen", broken_stream)
    interrupt_file_close(monkeypatch, closed)
    with pytest.raises(BaseException) as caught:
        runner._read_file_once(path)
    assert caught.value is original
    assert closed == opened
