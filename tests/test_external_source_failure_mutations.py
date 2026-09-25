"""Failures leave private attempts occupied without retries or public output."""

import os

import pytest
from test_external_source_checkpoints import writer as writer
from test_external_source_failure import failure_module
from test_external_source_failure import state as state
from test_external_source_failure_io import snapshot

from automated_phishing_detection import _external_source_failure_io as failure_io
from automated_phishing_detection import execution_receipt as receipt


def test_reservation_alias_is_rejected(state, writer, tmp_path):
    (tmp_path / "alias").hardlink_to(writer.attempt.directory / "reservation.json")
    with pytest.raises(ValueError, match="^invalid_external_source_failure$"):
        failure_module().retain_external_failure(
            writer.attempt, snapshot(state, writer)
        )


@pytest.mark.parametrize("target", ["attempt", "reservation"])
def test_replacement_during_installation_is_rejected_without_deletion(
    state, writer, monkeypatch, tmp_path, target
):
    original = failure_io._install
    attempt = writer.attempt.directory
    moved = tmp_path / "original"

    def replace(directory, content):
        identity = original(directory, content)
        if target == "attempt":
            attempt.rename(moved)
            attempt.mkdir(mode=0o700)
            (attempt / "marker").write_bytes(b"replacement")
        else:
            reservation = attempt / "reservation.json"
            reservation.rename(moved)
            reservation.write_bytes(moved.read_bytes())
            reservation.chmod(0o600)
        return identity

    monkeypatch.setattr(failure_io, "_install", replace)
    with pytest.raises(ValueError, match="^invalid_external_source_failure$"):
        failure_module().retain_external_failure(
            writer.attempt, snapshot(state, writer)
        )
    assert moved.exists()
    if target == "attempt":
        assert (attempt / "marker").read_bytes() == b"replacement"
        assert (moved / "external-failure.json").exists()


def test_ambiguous_post_install_sync_leaves_sidecar_occupied(
    state, writer, monkeypatch
):
    original = receipt._sync_directory

    def fail_after_publish(directory):
        if (directory.path / "external-failure.json").exists():
            raise OSError("private-diagnostic")
        original(directory)

    monkeypatch.setattr(receipt, "_sync_directory", fail_after_publish)
    content = snapshot(state, writer)
    with pytest.raises(ValueError, match="^invalid_external_source_failure$"):
        failure_module().retain_external_failure(writer.attempt, content)
    assert (writer.attempt.directory / "external-failure.json").read_bytes() == content
    with pytest.raises(ValueError, match="^invalid_external_source_failure$"):
        failure_module().retain_external_failure(writer.attempt, content)
    assert set(os.listdir(writer.attempt.directory)) == {
        "reservation.json",
        "external-failure.json",
    }


@pytest.mark.parametrize(
    "cleanup_error", [OSError("secret"), KeyboardInterrupt(), SystemExit(9)]
)
def test_install_interruption_survives_staging_cleanup(
    state, writer, monkeypatch, cleanup_error
):
    original = KeyboardInterrupt("first")
    cleaned = []
    cleanup = receipt._clean_staging

    def interrupt(*args):
        raise original

    def fail_cleanup(*args):
        cleaned.append(True)
        cleanup(*args)
        raise cleanup_error

    monkeypatch.setattr(receipt, "_write_file", interrupt)
    monkeypatch.setattr(receipt, "_clean_staging", fail_cleanup)
    with pytest.raises(KeyboardInterrupt) as caught:
        failure_module().retain_external_failure(
            writer.attempt, snapshot(state, writer)
        )
    assert caught.value is original
    assert cleaned == [True]
    assert set(os.listdir(writer.attempt.directory)) == {"reservation.json"}


@pytest.mark.parametrize("mutation", ["bytes", "inode", "symlink"])
def test_installed_sidecar_mutations_are_rejected(
    state, writer, monkeypatch, tmp_path, mutation
):
    original = receipt._sync_directory
    sidecar = writer.attempt.directory / "external-failure.json"
    changed = []

    def mutate(directory):
        original(directory)
        if sidecar.exists() and not changed:
            changed.append(True)
            if mutation == "bytes":
                sidecar.write_bytes(b"changed")
            else:
                displaced = tmp_path / "displaced"
                sidecar.rename(displaced)
                if mutation == "inode":
                    sidecar.write_bytes(displaced.read_bytes())
                    sidecar.chmod(0o600)
                else:
                    sidecar.symlink_to(displaced)

    monkeypatch.setattr(receipt, "_sync_directory", mutate)
    with pytest.raises(ValueError, match="^invalid_external_source_failure$"):
        failure_module().retain_external_failure(
            writer.attempt, snapshot(state, writer)
        )
    assert sidecar.exists()


def test_readback_uses_the_created_inode(state, writer, monkeypatch, tmp_path):
    original = failure_io._private_file
    sidecar = writer.attempt.directory / "external-failure.json"
    checks = []

    def swap_after_stat(directory, name):
        result = original(directory, name)
        if name == "external-failure.json":
            checks.append(name)
            if len(checks) == 2:
                sidecar.rename(tmp_path / "original")
                sidecar.write_bytes((tmp_path / "original").read_bytes())
                sidecar.chmod(0o600)
        return result

    monkeypatch.setattr(failure_io, "_private_file", swap_after_stat)
    with pytest.raises(ValueError, match="^invalid_external_source_failure$"):
        failure_module().retain_external_failure(
            writer.attempt, snapshot(state, writer)
        )
