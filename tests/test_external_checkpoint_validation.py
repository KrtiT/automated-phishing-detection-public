"""Retention rejects malformed mappings and replaced attempts before appending."""

import json

import pytest
from test_external_source_checkpoints import checkpoint_module, provenance
from test_external_source_checkpoints import writer as writer

from automated_phishing_detection import execution_receipt


@pytest.mark.parametrize("change", ["missing", "extra", "mutable", "subclass_key"])
def test_invalid_initial_mapping_is_latched_before_filesystem_work(writer, change):
    class Named(str):
        pass

    module, contents = checkpoint_module(), provenance()
    name = module.PROVENANCE_ORDER[0]
    if change == "missing":
        contents.pop(name)
    elif change == "extra":
        contents["unknown"] = b"unknown"
    elif change == "mutable":
        contents[name] = bytearray(contents[name])
    else:
        content = contents.pop(name)
        contents[Named(name)] = content
    with pytest.raises(module.ExternalCheckpointError):
        writer.begin(contents)
    with pytest.raises(module.ExternalCheckpointError):
        writer.begin(provenance())
    assert not (writer.attempt.directory / "checkpoints").exists()
    assert json.loads(writer.snapshot())["status"] == "failed"


def test_recreated_identical_reservation_cannot_replace_original_attempt(writer):
    module = checkpoint_module()
    writer.begin(provenance())
    directory = writer.attempt.directory
    directory.rename(directory.with_name("original"))
    replacement = execution_receipt.reserve_attempt(directory, identity={"fixture": 1})
    assert replacement.reservation_sha256 == writer.attempt.reservation_sha256
    (directory / "checkpoints").mkdir(mode=0o700)
    with pytest.raises(module.ExternalCheckpointError):
        writer(module.SCIENTIFIC_ORDER[0], b"not installed")
    assert list((directory / "checkpoints").iterdir()) == []


def test_constructor_copies_execution_identity(tmp_path):
    module, identity = checkpoint_module(), {"fixture": {"value": 1}}
    attempt = execution_receipt.reserve_attempt(tmp_path / "attempt", identity=identity)
    writer = module.ExternalCheckpointWriter(attempt, identity=identity)
    identity["fixture"]["value"] = 2
    writer.begin(provenance())
    assert len(json.loads(writer.snapshot())["confirmed_sha256"]) == 6


@pytest.mark.parametrize(
    "identity", [None, [], {"value": float("nan")}, {"value": object()}]
)
def test_constructor_rejects_malformed_identity_symbolically(tmp_path, identity):
    module = checkpoint_module()
    attempt = execution_receipt.reserve_attempt(
        tmp_path / "attempt", identity={"ok": 1}
    )
    with pytest.raises(
        module.ExternalCheckpointError, match="invalid_external_checkpoint"
    ):
        module.ExternalCheckpointWriter(attempt, identity=identity)
