"""Closed failure envelopes reject malformed inventories before any file access."""

import json

import pytest
from test_external_source_checkpoints import writer as writer
from test_external_source_failure import failure_module
from test_external_source_failure import state as state
from test_external_source_failure_io import snapshot

from automated_phishing_detection._checkpoint_codec import canonical_bytes


@pytest.mark.parametrize(
    "field, value",
    [
        ("schema_version", True),
        ("schema_version", 2),
        ("protocol_id", "external-source-checkpoints-v1"),
        ("status", "complete"),
        ("reservation_sha256", "A" * 64),
        ("execution", {}),
        ("execution", []),
        ("stage", "secret/path"),
        ("failure", "secret diagnostic"),
        ("cleanup_failed", 1),
        ("producer_progress_base64", "YQ="),
        ("producer_progress_base64", "YR=="),
        ("producer_progress_base64", 0),
        ("completed", {}),
        ("completed", {"composition": {}, "private_outputs_base64": {}}),
        ("unknown", 1),
    ],
)
def test_invalid_envelope_never_opens_attempt(state, writer, monkeypatch, field, value):
    from automated_phishing_detection import execution_receipt

    record = json.loads(snapshot(state, writer))
    record[field] = value

    def forbidden(*args):
        pytest.fail("invalid schema must fail before opening attempt")

    monkeypatch.setattr(execution_receipt, "_attempt_directory", forbidden)
    with pytest.raises(ValueError, match="^invalid_external_source_failure$"):
        failure_module().retain_external_failure(
            writer.attempt, canonical_bytes(record)
        )


@pytest.mark.parametrize(
    "mutation", ["missing", "duplicate", "whitespace", "nonfinite", "not_bytes"]
)
def test_encoding_is_strict_and_canonical(state, writer, mutation):
    content = snapshot(state, writer)
    if mutation == "missing":
        record = json.loads(content)
        del record["completed"]
        content = canonical_bytes(record)
    elif mutation == "duplicate":
        content = content.replace(b"{", b'{"status":"failed",', 1)
    elif mutation == "whitespace":
        content += b"\n"
    elif mutation == "nonfinite":
        content = content.replace(b'"cleanup_failed":false', b'"cleanup_failed":NaN')
    else:
        content = bytearray(content)
    with pytest.raises(ValueError, match="^invalid_external_source_failure$"):
        failure_module().retain_external_failure(writer.attempt, content)


@pytest.mark.parametrize(
    "field, value",
    [
        ("schema_version", True),
        ("protocol_id", "external-source-failure-v1"),
        ("reservation_sha256", "0" * 64),
        ("status", "complete"),
        ("confirmed_sha256", {"retained-test.jsonl": "0" * 64}),
        ("pending_checkpoint_bytes", {"secret": "YQ=="}),
        ("pending_checkpoint_bytes", {"publisher-source.json": "YQ="}),
        ("pending_checkpoint_bytes", {"publisher-source.json": "YQ=="}),
        ("unknown", 1),
    ],
)
def test_checkpoint_shape_is_bound_and_closed(state, writer, field, value):
    state.writer = writer
    record = json.loads(snapshot(state, writer))
    record["checkpoints"][field] = value
    with pytest.raises(ValueError, match="^invalid_external_source_failure$"):
        failure_module().retain_external_failure(
            writer.attempt, canonical_bytes(record)
        )


def test_writer_identity_must_match_envelope(state, writer):
    writer.identity = {"fixture": 2}
    state.writer = writer
    with pytest.raises(ValueError, match="^invalid_external_source_failure$"):
        state.snapshot(writer.attempt, {"fixture": 1}, "scoring", OSError())


@pytest.mark.parametrize("progress", [bytearray(b"secret"), "secret", None])
def test_nonbytes_progress_remains_unavailable(state, writer, progress):
    original = OSError()
    original.progress = progress
    with pytest.raises(OSError), state.capture_body():
        raise original
    assert json.loads(snapshot(state, writer))["producer_progress_base64"] is None
