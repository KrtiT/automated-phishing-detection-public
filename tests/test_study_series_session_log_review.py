"""Concurrent appends remain valid; previously observed log bytes cannot change."""

import os
import signal

import pytest
from study_series_session_fixtures import api


def test_same_size_external_log_rewrite_is_rejected(tmp_path):
    module = api("_study_series_session_records")
    tmp_path.chmod(0o700)
    path = tmp_path / "physical"
    with pytest.raises(ValueError):
        with module.hold_records(path) as records:
            records.streams["stdout.log"].write(b"original\n")
            records.check()
            with (path / "stdout.log").open("r+b") as target:
                target.write(b"altered!\n")
            records.check()


def test_owned_log_descriptor_always_appends_even_after_seek(tmp_path):
    module = api("_study_series_session_records")
    tmp_path.chmod(0o700)
    path = tmp_path / "physical"
    with module.hold_records(path) as records:
        stream = records.streams["stdout.log"]
        stream.write(b"first\n")
        stream.seek(0)
        stream.write(b"second\n")
    assert (path / "stdout.log").read_bytes() == b"first\nsecond\n"


def test_append_between_fd_and_path_stats_is_valid(monkeypatch, tmp_path):
    module = api("_study_series_session_records")
    tmp_path.chmod(0o700)
    with module.hold_records(tmp_path / "physical") as records:
        original = module.receipt._entry
        appended = []

        def append_then_stat(directory, name):
            if name == "stdout.log" and not appended:
                appended.append(True)
                records.streams[name].write(b"concurrent child output\n")
            return original(directory, name)

        monkeypatch.setattr(module.receipt, "_entry", append_then_stat)
        records.check()
        assert appended == [True]


def test_prefix_rewrite_with_growth_is_rejected(tmp_path):
    module = api("_study_series_session_records")
    tmp_path.chmod(0o700)
    path = tmp_path / "physical"
    with pytest.raises(ValueError):
        with module.hold_records(path) as records:
            records.streams["stdout.log"].write(b"original\n")
            records.check()
            with (path / "stdout.log").open("r+b") as target:
                target.write(b"altered!\nmore\n")
                os.fsync(target.fileno())
            records.check()


def test_interrupt_cannot_split_fixed_record_publication(monkeypatch, tmp_path):
    module = api("_study_series_session_records")
    tmp_path.chmod(0o700)
    path = tmp_path / "physical"
    with module.hold_records(path) as records:
        original, received = os.fsync, []

        def interrupt_after_sync(descriptor):
            original(descriptor)
            if not received:
                received.append(True)
                os.kill(os.getpid(), signal.SIGINT)

        monkeypatch.setattr(os, "fsync", interrupt_after_sync)
        with pytest.raises(KeyboardInterrupt):
            records.record("pre.json", {"complete": True})
        records.check()
    assert (path / "pre.json").read_bytes() == b'{"complete":true}\n'
