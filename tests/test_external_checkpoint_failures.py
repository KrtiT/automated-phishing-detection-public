"""Failed or ambiguous external retention never creates a retry permission."""

import asyncio
import base64
import json
from contextlib import contextmanager

import pytest
from test_external_source_checkpoints import checkpoint_module, provenance, science
from test_external_source_checkpoints import writer as writer

from automated_phishing_detection import execution_receipt


@pytest.mark.parametrize("after_install", [False, True])
@pytest.mark.parametrize(
    "kind", [OSError, asyncio.CancelledError, KeyboardInterrupt, SystemExit]
)
def test_append_failure_retains_attempted_bytes_and_refuses_retry(
    writer, monkeypatch, after_install, kind
):
    module = checkpoint_module()
    original = module.checkpoint_io._install_record
    failure, calls = kind("private-canary"), []
    writer.begin(provenance())

    def fail(directory, name, content):
        calls.append(name)
        if after_install:
            original(directory, name, content)
        raise failure

    monkeypatch.setattr(module.checkpoint_io, "_install_record", fail)
    name = module.SCIENTIFIC_ORDER[0]
    expected = module.ExternalCheckpointError if kind is OSError else kind
    with pytest.raises(expected) as caught:
        writer(name, b"retained private bytes")
    assert caught.value is failure or "private-canary" not in str(caught.value)
    with pytest.raises(module.ExternalCheckpointError):
        writer(name, b"retry forbidden")
    assert calls == [name]
    progress = json.loads(writer.snapshot())
    assert len(progress["confirmed_sha256"]) == 6
    assert base64.b64decode(progress["pending_checkpoint_bytes"][name]) == (
        b"retained private bytes"
    )


@pytest.mark.parametrize("after_install", [False, True])
def test_begin_failure_preserves_all_six_attempted_payloads(
    writer, monkeypatch, after_install
):
    module, original = checkpoint_module(), execution_receipt._publish

    def fail(*arguments):
        if after_install:
            original(*arguments)
        raise OSError("private-canary")

    monkeypatch.setattr(execution_receipt, "_publish", fail)
    with pytest.raises(module.ExternalCheckpointError, match="checkpoint"):
        writer.begin(provenance())
    with pytest.raises(module.ExternalCheckpointError):
        writer.begin(provenance())
    progress = json.loads(writer.snapshot())
    assert progress["confirmed_sha256"] == {}
    assert {
        name: base64.b64decode(content)
        for name, content in progress["pending_checkpoint_bytes"].items()
    } == provenance()


@pytest.mark.parametrize(
    "name", ["finalize.claim", "outcome.json", "failure-progress.json", "unexpected"]
)
def test_finalized_or_changed_attempt_refuses_later_append(writer, name):
    module = checkpoint_module()
    writer.begin(provenance())
    (writer.attempt.directory / name).write_bytes(b"stopped")
    with pytest.raises(module.ExternalCheckpointError):
        writer(module.SCIENTIFIC_ORDER[0], b"forbidden")
    assert not (
        writer.attempt.directory / "checkpoints" / module.SCIENTIFIC_ORDER[0]
    ).exists()


@pytest.mark.parametrize("change", ["rename", "symlink", "mode", "extra"])
def test_changed_checkpoint_directory_rejects_before_append(writer, change):
    module = checkpoint_module()
    writer.begin(provenance())
    directory = writer.attempt.directory / "checkpoints"
    if change in {"rename", "symlink"}:
        saved = writer.attempt.directory.parent / "saved-checkpoints"
        directory.rename(saved)
        if change == "rename":
            directory.mkdir(mode=0o700)
        else:
            directory.symlink_to(saved, target_is_directory=True)
    elif change == "mode":
        directory.chmod(0o755)
    else:
        (directory / "unexpected").write_bytes(b"changed")
    with pytest.raises(module.ExternalCheckpointError):
        writer(module.SCIENTIFIC_ORDER[0], b"forbidden")


def test_wrong_execution_identity_stops_before_checkpoint_creation(tmp_path):
    module = checkpoint_module()
    attempt = execution_receipt.reserve_attempt(
        tmp_path / "attempt", identity={"fixture": 1}
    )
    writer = module.ExternalCheckpointWriter(attempt, identity={"changed": True})
    with pytest.raises(module.ExternalCheckpointError):
        writer.begin(provenance())
    assert not (attempt.directory / "checkpoints").exists()


def test_finalization_after_last_callback_cannot_be_completed(writer):
    module, outputs = checkpoint_module(), science()
    writer.begin(provenance())
    for name, content in outputs.items():
        writer(name, content)
    (writer.attempt.directory / "finalize.claim").write_bytes(b"claimed")
    with pytest.raises(module.ExternalCheckpointError):
        writer.complete(outputs)


@pytest.mark.parametrize(
    "kind", [asyncio.CancelledError, KeyboardInterrupt, SystemExit]
)
@pytest.mark.parametrize("cleanup_kind", [OSError, KeyboardInterrupt])
@pytest.mark.parametrize("operation", ["begin", "append"])
def test_original_interruption_survives_staging_cleanup_failure(
    writer, monkeypatch, kind, cleanup_kind, operation
):
    original = execution_receipt._staging_directory
    first = kind("first private interruption")
    if operation == "append":
        writer.begin(provenance())

    @contextmanager
    def broken(*arguments):
        with original(*arguments) as value:
            try:
                yield value
            finally:
                raise cleanup_kind("second private failure")

    def interrupt(*arguments):
        raise first

    monkeypatch.setattr(execution_receipt, "_staging_directory", broken)
    monkeypatch.setattr(execution_receipt, "_write_file", interrupt)
    with pytest.raises(BaseException) as caught:
        _retain(writer, operation)
    assert caught.value is first
    assert json.loads(writer.snapshot())["status"] == "failed"


def _retain(writer, operation):
    if operation == "begin":
        writer.begin(provenance())
    else:
        writer(checkpoint_module().SCIENTIFIC_ORDER[0], b"attempted bytes")
