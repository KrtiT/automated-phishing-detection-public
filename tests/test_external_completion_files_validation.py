"""Invented file aliases, permission changes and concurrent tree mutations reject."""

import asyncio
import os
import shutil

import pytest
import test_external_completion_files as fixtures

from automated_phishing_detection import source_runner

files_api = fixtures.files_api
completed_tree = fixtures.completed_tree
_FILES = (
    "attempt/reservation.json",
    "attempt/finalize.claim",
    "attempt/outcome.json",
    "attempt/checkpoints/publisher-source.json",
    "attempt/evidence/bindings.json",
    "public-summary.json",
)


@pytest.mark.parametrize("logical", _FILES)
@pytest.mark.parametrize("change", ["mode", "symlink", "hardlink", "fifo"])
def test_unsafe_file_rejected_before_first_read(
    files_api, completed_tree, monkeypatch, logical, change
):
    attempt, public = completed_tree
    target = fixtures.expected_files(attempt, public)[logical]
    if change == "mode":
        target.chmod(0o666)
    else:
        saved = public.parent / "saved"
        target.rename(saved)
        if change == "symlink":
            target.symlink_to(saved)
        elif change == "hardlink":
            os.link(saved, target)
        else:
            os.mkfifo(target, 0o600)

    def forbidden(*args, **kwargs):
        pytest.fail("unsafe initial file reached content read")

    monkeypatch.setattr(source_runner, "_read_file_once", forbidden)
    with pytest.raises(files_api.ExternalCompletionFileError) as rejected:
        with files_api.snapshot_external_files(attempt, public):
            pytest.fail("unsafe file yielded")
    assert str(rejected.value) == "invalid_external_completion_files"


@pytest.mark.parametrize("logical", _FILES)
@pytest.mark.parametrize("change", ["bytes", "mode", "inode"])
def test_final_file_state_checked_after_reconstruction(
    files_api, completed_tree, logical, change
):
    attempt, public = completed_tree
    target = fixtures.expected_files(attempt, public)[logical]
    with pytest.raises(files_api.ExternalCompletionFileError):
        with files_api.snapshot_external_files(attempt, public):
            if change == "mode":
                target.chmod(0o400)
            elif change == "bytes":
                content = target.read_bytes()
                target.write_bytes(b"!" + content[1:])
            else:
                replacement = public.parent / "replacement"
                shutil.copy2(target, replacement)
                replacement.replace(target)


@pytest.mark.parametrize("directory", ["", "checkpoints", "evidence"])
@pytest.mark.parametrize("change", ["mode", "extra", "inode"])
def test_final_directory_identity_inventory_and_modes_checked(
    files_api, completed_tree, directory, change
):
    attempt, public = completed_tree
    target = attempt / directory
    with pytest.raises(files_api.ExternalCompletionFileError):
        with files_api.snapshot_external_files(attempt, public):
            if change == "mode":
                target.chmod(0o755)
            elif change == "extra":
                (target / "unknown").write_bytes(b"unexpected")
            else:
                moved = public.parent / "moved"
                target.rename(moved)
                shutil.copytree(moved, target)


def test_parent_alias_is_rejected(files_api, completed_tree):
    attempt, public = completed_tree
    alias = public.parent / "alias"
    alias.symlink_to(attempt, target_is_directory=True)
    with pytest.raises(files_api.ExternalCompletionFileError):
        with files_api.snapshot_external_files(alias, public):
            pytest.fail("alias yielded")


def test_no_public_marker_inside_attempt(files_api, completed_tree, monkeypatch):
    attempt, public = completed_tree

    def forbidden(*args, **kwargs):
        pytest.fail("invalid output paths inspected")

    monkeypatch.setattr(source_runner, "_read_file_once", forbidden)
    with pytest.raises(files_api.ExternalCompletionFileError):
        with files_api.snapshot_external_files(attempt, attempt / "public.json"):
            pytest.fail("contained public marker yielded")


@pytest.mark.parametrize(
    "error",
    [
        OSError("fixture"),
        RuntimeError("fixture"),
        KeyboardInterrupt(),
        asyncio.CancelledError(),
    ],
)
def test_consumer_error_identity_survives_context(files_api, completed_tree, error):
    attempt, public = completed_tree
    with pytest.raises(type(error)) as rejected:
        with files_api.snapshot_external_files(attempt, public):
            raise error
    assert rejected.value is error


def test_snapshot_repr_does_not_expose_payloads(files_api, completed_tree):
    attempt, public = completed_tree
    with files_api.snapshot_external_files(attempt, public) as snapshot:
        assert "bindings.json" not in repr(snapshot)
        assert "invented" not in repr(snapshot)
