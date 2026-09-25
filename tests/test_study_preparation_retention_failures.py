import base64
import json
import os
import stat
from contextlib import contextmanager

import pytest
from test_study_preparation_retention import (
    IDENTITY,
    ORDER,
    append_all,
    module,
    outputs,
    reserved,
)

from automated_phishing_detection import execution_receipt as receipt


@pytest.mark.parametrize(
    "error", [OSError("private"), KeyboardInterrupt(), SystemExit(0)]
)
def test_partial_file_and_pending_bytes_survive(tmp_path, monkeypatch, error):
    retained = module()
    attempt = reserved(tmp_path)
    original = retained.files.write_file

    def partial(directory, name, content, mode):
        original(directory, name, content[:3], mode)
        raise error

    monkeypatch.setattr(retained.files, "write_file", partial)
    expected = (
        retained.StudyPreparationRetentionError
        if isinstance(error, Exception)
        else type(error)
    )
    with pytest.raises(expected) as caught:
        with retained.retain_study_preparation(attempt, identity=IDENTITY) as writer:
            writer.append(*outputs()[0])
    if not isinstance(error, Exception):
        assert caught.value is error
    assert (attempt.directory / ORDER[0]).read_bytes() == outputs()[0][1][:3]
    progress = json.loads(writer.snapshot())
    assert progress["confirmed_sha256"] == {}
    assert progress["pending_checkpoint_bytes"] == {
        ORDER[0]: base64.b64encode(outputs()[0][1]).decode("ascii")
    }


def test_readback_must_match_requested_bytes(tmp_path, monkeypatch):
    retained = module()
    attempt = reserved(tmp_path)
    original = retained.files.write_file

    def changed(directory, name, content, mode):
        return original(directory, name, b"changed", mode)

    monkeypatch.setattr(retained.files, "write_file", changed)
    with pytest.raises(retained.StudyPreparationRetentionError):
        with retained.retain_study_preparation(attempt, identity=IDENTITY) as writer:
            writer.append(*outputs()[0])
    assert (attempt.directory / ORDER[0]).read_bytes() == b"changed"
    assert json.loads(writer.snapshot())["confirmed_sha256"] == {}


def test_directory_sync_failure_does_not_confirm_or_remove_file(tmp_path, monkeypatch):
    retained = module()
    attempt = reserved(tmp_path)

    def fail(unused):
        raise OSError("private")

    monkeypatch.setattr(receipt, "_sync_directory", fail)
    with pytest.raises(retained.StudyPreparationRetentionError):
        with retained.retain_study_preparation(attempt, identity=IDENTITY) as writer:
            writer.append(*outputs()[0])
    assert (attempt.directory / ORDER[0]).read_bytes() == outputs()[0][1]
    assert json.loads(writer.snapshot())["confirmed_sha256"] == {}


@pytest.mark.parametrize("error", [KeyboardInterrupt(), SystemExit(0)])
@pytest.mark.parametrize("cleanup_error", [OSError("cleanup"), KeyboardInterrupt()])
def test_original_interruption_survives_directory_cleanup(
    tmp_path, monkeypatch, error, cleanup_error
):
    retained = module()
    attempt = reserved(tmp_path)
    original = receipt._directory

    @contextmanager
    def broken_cleanup(path):
        try:
            with original(path) as directory:
                yield directory
        finally:
            raise cleanup_error

    monkeypatch.setattr(receipt, "_directory", broken_cleanup)
    with pytest.raises(type(error)) as caught:
        with retained.retain_study_preparation(attempt, identity=IDENTITY) as writer:
            writer.append(*outputs()[0])
            raise error
    assert caught.value is error
    assert (attempt.directory / ORDER[0]).read_bytes() == outputs()[0][1]


def test_caught_append_failure_cannot_retry(tmp_path):
    retained = module()
    attempt = reserved(tmp_path)
    with pytest.raises(retained.StudyPreparationRetentionError):
        with retained.retain_study_preparation(attempt, identity=IDENTITY) as writer:
            with pytest.raises(retained.StudyPreparationRetentionError):
                writer.append(ORDER[1], b"out of order")
            writer.append(*outputs()[0])
    assert not (attempt.directory / ORDER[0]).exists()


def test_no_destructive_or_scientific_finalization_calls(tmp_path, monkeypatch):
    retained = module()
    attempt = reserved(tmp_path)

    def forbidden(*unused, **unused_keywords):
        pytest.fail("destructive or scientific finalization call")

    for name in (
        "_clean_staging",
        "_staging_directory",
        "publish_completion",
        "_claim",
    ):
        monkeypatch.setattr(receipt, name, forbidden)
    with retained.retain_study_preparation(attempt, identity=IDENTITY) as writer:
        append_all(writer)
        writer.finish()


def test_same_bytes_replacement_during_file_sync_rejected(tmp_path, monkeypatch):
    retained = module()
    attempt = reserved(tmp_path)
    original = receipt.transformer_pipeline._sync_descriptor
    target = attempt.directory / ORDER[0]

    def replace_file(descriptor):
        original(descriptor)
        if stat.S_ISREG(os.fstat(descriptor).st_mode) and target.exists():
            target.rename(tmp_path / "original")
            target.write_bytes(outputs()[0][1])
            target.chmod(0o600)

    monkeypatch.setattr(receipt.transformer_pipeline, "_sync_descriptor", replace_file)
    with pytest.raises(RuntimeError, match="test_stop"):
        with retained.retain_study_preparation(attempt, identity=IDENTITY) as writer:
            with pytest.raises(retained.StudyPreparationRetentionError):
                writer.append(*outputs()[0])
            raise RuntimeError("test_stop")
    assert (tmp_path / "original").read_bytes() == outputs()[0][1]
    assert target.read_bytes() == outputs()[0][1]
