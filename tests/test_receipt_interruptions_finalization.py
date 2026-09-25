"""Cleanup changes never remove an installed reservation or public marker."""

import asyncio
import os
from importlib import import_module

import pytest


def interrupt_after_install(receipt, monkeypatch, target, original, calls):
    publish, close = receipt._publish, os.close
    installed = False

    def broken_publish(source, name, destination, filename):
        nonlocal installed
        publish(source, name, destination, filename)
        if filename == target:
            installed = True
            raise original

    def broken_close(descriptor):
        close(descriptor)
        if installed:
            calls.append(descriptor)
            raise KeyboardInterrupt("repeated cleanup")

    monkeypatch.setattr(receipt, "_publish", broken_publish)
    monkeypatch.setattr(os, "close", broken_close)


@pytest.mark.parametrize(
    "kind", (KeyboardInterrupt, asyncio.CancelledError, SystemExit)
)
def test_reservation_interruption_keeps_occupied_attempt(tmp_path, monkeypatch, kind):
    receipt = import_module("automated_phishing_detection.execution_receipt")
    original, closed = kind("reservation"), []
    with monkeypatch.context() as patch:
        interrupt_after_install(receipt, patch, "attempt", original, closed)
        with pytest.raises(BaseException) as caught:
            receipt.reserve_attempt(tmp_path / "attempt", identity={"fixture": 1})
    assert caught.value is original
    assert len(closed) == 2
    assert (tmp_path / "attempt/reservation.json").is_file()
    with pytest.raises(receipt.ExecutionReceiptError):
        receipt.reserve_attempt(tmp_path / "attempt", identity={"fixture": 1})


@pytest.mark.parametrize(
    "kind", (KeyboardInterrupt, asyncio.CancelledError, SystemExit)
)
def test_public_interruption_preserves_all_installed_evidence(
    tmp_path, monkeypatch, kind
):
    receipt = import_module("automated_phishing_detection.execution_receipt")
    attempt = receipt.reserve_attempt(tmp_path / "attempt", identity={"fixture": 1})
    original, closed = kind("publication"), []
    with monkeypatch.context() as patch:
        interrupt_after_install(receipt, patch, "summary.json", original, closed)
        with pytest.raises(BaseException) as caught:
            receipt.publish_completion(
                attempt,
                private_outputs={"one": b"invented"},
                public_summary={"fixture": 1},
                public_path=tmp_path / "summary.json",
            )
    assert caught.value is original
    assert len(closed) == 3
    assert (tmp_path / "summary.json").is_file()
    assert (attempt.directory / "evidence/one").read_bytes() == b"invented"
    assert (attempt.directory / "outcome.json").is_file()
    with pytest.raises(receipt.ExecutionReceiptError):
        receipt.record_failure(attempt, stage="late", error_type="interrupted")
