"""Receipt cleanup faults use only temporary directories and invented bytes."""

import asyncio
import os
from importlib import import_module

import pytest

INTERRUPTIONS = (KeyboardInterrupt, asyncio.CancelledError, SystemExit)


def fail(error):
    raise error


def faulty_close(monkeypatch, descriptor, later, calls):
    close = os.close

    def broken(current):
        close(current)
        if current == descriptor:
            calls.append(current)
            raise later

    monkeypatch.setattr(os, "close", broken)


@pytest.mark.parametrize("kind", INTERRUPTIONS)
@pytest.mark.parametrize("later_kind", (OSError, KeyboardInterrupt))
def test_directory_close_preserves_body_interruption(
    tmp_path, monkeypatch, kind, later_kind
):
    receipt = import_module("automated_phishing_detection.execution_receipt")
    original, later, calls = kind("first"), later_kind("later"), []
    with pytest.raises(BaseException) as caught:
        with receipt._directory(tmp_path) as directory:
            faulty_close(monkeypatch, directory.descriptor, later, calls)
            raise original
    assert caught.value is original
    assert len(calls) == 1


@pytest.mark.parametrize("kind", INTERRUPTIONS)
def test_staging_closes_descriptor_even_when_cleanup_is_interrupted(
    tmp_path, monkeypatch, kind
):
    receipt = import_module("automated_phishing_detection.execution_receipt")
    original, later = kind("first"), KeyboardInterrupt("later")
    entry = receipt._entry
    with receipt._directory(tmp_path) as parent:

        def broken_entry(directory, name):
            if name.startswith(".fixture"):
                raise later
            return entry(directory, name)

        with pytest.raises(BaseException) as caught:
            with receipt._staging_directory(parent, "fixture") as (staging, unused):
                monkeypatch.setattr(receipt, "_entry", broken_entry)
                raise original
        assert caught.value is original
        with pytest.raises(OSError):
            os.fstat(staging.descriptor)


@pytest.mark.parametrize("kind", INTERRUPTIONS)
def test_staging_entry_failure_wins_over_descriptor_close(tmp_path, monkeypatch, kind):
    receipt = import_module("automated_phishing_detection.execution_receipt")
    original, calls = kind("entry"), []
    check = receipt._Directory.check
    with receipt._directory(tmp_path) as parent:

        def broken(directory):
            if directory.path != tmp_path:
                faulty_close(monkeypatch, directory.descriptor, OSError("close"), calls)
                raise original
            return check(directory)

        monkeypatch.setattr(receipt._Directory, "check", broken)
        with pytest.raises(BaseException) as caught:
            with receipt._staging_directory(parent, "fixture"):
                pytest.fail("entry must fail")
        assert caught.value is original
    assert len(calls) == 1


@pytest.mark.parametrize("kind", INTERRUPTIONS)
def test_publication_unlink_preserves_original_and_consumed_claim(
    tmp_path, monkeypatch, kind
):
    receipt = import_module("automated_phishing_detection.execution_receipt")
    attempt = receipt.reserve_attempt(tmp_path / "attempt", identity={"fixture": 1})
    original, later = kind("publication"), KeyboardInterrupt("unlink")
    publish, unlink = receipt._publish, os.unlink

    def broken_publish(source, name, destination, target):
        if target == "evidence":
            raise original
        return publish(source, name, destination, target)

    def broken_unlink(name, **kwargs):
        if name.startswith(".summary.json.tmp-"):
            raise later
        return unlink(name, **kwargs)

    monkeypatch.setattr(receipt, "_publish", broken_publish)
    monkeypatch.setattr(os, "unlink", broken_unlink)
    with pytest.raises(BaseException) as caught:
        receipt.publish_completion(
            attempt,
            private_outputs={"one": b"invented"},
            public_summary={"fixture": 1},
            public_path=tmp_path / "summary.json",
        )
    assert caught.value is original
    assert (attempt.directory / "finalize.claim").is_file()
    assert not (tmp_path / "summary.json").exists()
