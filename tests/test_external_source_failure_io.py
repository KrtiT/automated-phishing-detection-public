"""Invented private failure sidecars are authenticated and create-only."""

import json
import os
import stat

import pytest
from test_external_source_checkpoints import provenance
from test_external_source_checkpoints import writer as writer
from test_external_source_failure import failure_module
from test_external_source_failure import state as state


def snapshot(state, writer):
    return state.snapshot(writer.attempt, writer.identity, "scoring", OSError("secret"))


def test_private_create_only_sidecar_does_not_finalize(state, writer):
    writer.begin(provenance())
    content = snapshot(state, writer)
    failure_module().retain_external_failure(writer.attempt, content)
    sidecar = writer.attempt.directory / "external-failure.json"
    assert sidecar.read_bytes() == content
    assert stat.S_IMODE(sidecar.stat().st_mode) == 0o600
    assert sidecar.stat().st_nlink == 1
    assert set(os.listdir(writer.attempt.directory)) == {
        "reservation.json",
        "checkpoints",
        "external-failure.json",
    }
    with pytest.raises(ValueError, match="^invalid_external_source_failure$"):
        failure_module().retain_external_failure(writer.attempt, content)
    assert sidecar.read_bytes() == content


@pytest.mark.parametrize(
    "collision", ["regular", "directory", "symlink", "dangling", "hardlink"]
)
def test_collisions_are_occupied_and_never_changed(state, writer, tmp_path, collision):
    sidecar = writer.attempt.directory / "external-failure.json"
    outside = tmp_path / "outside"
    outside.write_bytes(b"unchanged")
    if collision == "regular":
        sidecar.write_bytes(b"occupied")
    elif collision == "directory":
        sidecar.mkdir()
    elif collision == "hardlink":
        sidecar.hardlink_to(outside)
    else:
        sidecar.symlink_to(outside if collision == "symlink" else tmp_path / "absent")
    before = sidecar.lstat()
    with pytest.raises(ValueError, match="^invalid_external_source_failure$"):
        failure_module().retain_external_failure(
            writer.attempt, snapshot(state, writer)
        )
    assert sidecar.lstat() == before
    assert outside.read_bytes() == b"unchanged"


@pytest.mark.parametrize(
    "name", ["finalize.claim", "evidence", "outcome.json", "unknown"]
)
def test_finalization_or_unknown_entries_block_failure_retention(state, writer, name):
    (writer.attempt.directory / name).write_bytes(b"occupied")
    with pytest.raises(ValueError, match="^invalid_external_source_failure$"):
        failure_module().retain_external_failure(
            writer.attempt, snapshot(state, writer)
        )
    assert not (writer.attempt.directory / "external-failure.json").exists()


@pytest.mark.parametrize(
    "target, mode",
    [("attempt", 0o755), ("reservation.json", 0o644), ("checkpoints", 0o755)],
)
def test_private_modes_are_required(state, writer, target, mode):
    writer.begin(provenance())
    path = (
        writer.attempt.directory
        if target == "attempt"
        else writer.attempt.directory / target
    )
    path.chmod(mode)
    with pytest.raises(ValueError, match="^invalid_external_source_failure$"):
        failure_module().retain_external_failure(
            writer.attempt, snapshot(state, writer)
        )


@pytest.mark.parametrize("target", ["attempt", "reservation.json", "checkpoints"])
def test_symbolic_links_are_rejected(state, writer, tmp_path, target):
    writer.begin(provenance())
    path = (
        writer.attempt.directory
        if target == "attempt"
        else writer.attempt.directory / target
    )
    moved = tmp_path / "moved"
    path.rename(moved)
    path.symlink_to(moved)
    with pytest.raises(ValueError, match="^invalid_external_source_failure$"):
        failure_module().retain_external_failure(
            writer.attempt, snapshot(state, writer)
        )


@pytest.mark.parametrize("mutation", ["identity", "reservation", "hash"])
def test_reservation_authentication_binds_exact_execution(state, writer, mutation):
    from dataclasses import replace

    from automated_phishing_detection._checkpoint_codec import canonical_bytes

    attempt, content = writer.attempt, snapshot(state, writer)
    if mutation == "hash":
        attempt = replace(attempt, reservation_sha256="0" * 64)
    elif mutation == "reservation":
        (attempt.directory / "reservation.json").write_bytes(b"changed")
    else:
        record = json.loads(content)
        record["execution"] = {"fixture": 2}
        content = canonical_bytes(record)
    with pytest.raises(ValueError, match="^invalid_external_source_failure$"):
        failure_module().retain_external_failure(attempt, content)
    assert not (writer.attempt.directory / "external-failure.json").exists()
