"""Authentication interruption cannot be replaced while opening an attempt."""

import asyncio
from contextlib import contextmanager

import pytest
from test_external_source_checkpoints import provenance
from test_external_source_checkpoints import writer as writer

from automated_phishing_detection import execution_receipt


@pytest.mark.parametrize(
    "kind", [asyncio.CancelledError, KeyboardInterrupt, SystemExit]
)
def test_authentication_interruption_survives_directory_cleanup(
    writer, monkeypatch, kind
):
    original = execution_receipt._directory
    first = kind("first private interruption")

    @contextmanager
    def broken(*arguments):
        with original(*arguments) as value:
            try:
                yield value
            finally:
                raise OSError("second private failure")

    def interrupt(*arguments):
        raise first

    monkeypatch.setattr(execution_receipt, "_directory", broken)
    monkeypatch.setattr(execution_receipt, "_authenticate", interrupt)
    with pytest.raises(BaseException) as caught:
        writer.begin(provenance())
    assert caught.value is first
