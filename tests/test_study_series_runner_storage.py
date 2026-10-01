"""Thin series writers retain generic original no-follow publication semantics."""

from importlib import import_module
from importlib.util import find_spec

import pytest

from automated_phishing_detection import execution_receipt as receipt
from automated_phishing_detection._checkpoint_codec import canonical_bytes


def api():
    name = "automated_phishing_detection._study_series_runner_storage"
    assert find_spec(name), "missing held series root writer"
    return import_module(name)


def test_real_held_writer_keeps_prefix_and_publishes_original_receipt(tmp_path):
    identity = {"protocol": "invented-series-test"}
    attempt = receipt.reserve_attempt(tmp_path / "attempt", identity=identity)
    public = tmp_path / "summary.json"
    content = canonical_bytes({"invented": "prefix"})
    with api().hold_root(attempt, public, identity) as writer:
        writer.append("history-import.json", content)
        result = writer.complete({"accounting.json": content}, {"status": "invented"})
        assert writer.held.publishing and writer.candidate is result
    assert public.read_bytes() == receipt._json_bytes({"status": "invented"}, "test")
    assert dict(result)["attempt/history-import.json"] == content
    assert dict(result)["attempt/evidence/accounting.json"] == content


def test_writer_failure_preserves_original_interrupt_and_owned_files(tmp_path):
    identity = {"protocol": "invented-series-test"}
    attempt = receipt.reserve_attempt(tmp_path / "attempt", identity=identity)
    error = KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt) as caught:
        with api().hold_root(attempt, tmp_path / "summary.json", identity) as writer:
            writer.append("history-import.json", canonical_bytes({"invented": True}))
            raise error
    assert caught.value is error
    assert (attempt.directory / "history-import.json").is_file()
    assert not (attempt.directory / "finalize.claim").exists()
