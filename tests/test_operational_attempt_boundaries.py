"""Private growing attempts reject mutation without deleting partial evidence."""

import os
import signal

import pytest
from test_operational_attempt_io import module

from automated_phishing_detection import _study_preparation_files as files
from automated_phishing_detection import execution_receipt as receipt


def attempt(tmp_path):
    return receipt.reserve_attempt(tmp_path / "attempt", identity={"kind": "fixture"})


@pytest.mark.parametrize(
    "name", ["unknown", "finalize.claim", "outcome.json", "evidence"]
)
def test_unknown_or_finalized_inventory_rejects_before_write(tmp_path, name):
    reserved = attempt(tmp_path)
    (reserved.directory / name).write_bytes(b"existing")
    with pytest.raises(ValueError):
        with module().held_attempt_writer(reserved, names=("client-role.json",)):
            pytest.fail("invalid attempt yielded")
    assert not (reserved.directory / "client-role.json").exists()


@pytest.mark.parametrize("change", ["bytes", "inode", "mode", "link"])
def test_completed_owned_record_changes_reject_on_exit(tmp_path, change):
    reserved = attempt(tmp_path)
    with pytest.raises(ValueError):
        with module().held_attempt_writer(
            reserved, names=("client-role.json",)
        ) as writer:
            writer.retain("client-role.json", b"original")
            path = reserved.directory / "client-role.json"
            if change == "bytes":
                path.write_bytes(b"changed")
            elif change == "inode":
                path.rename(reserved.directory.parent / "retained")
                path.write_bytes(b"original")
                path.chmod(0o600)
            elif change == "mode":
                path.chmod(0o644)
            else:
                os.link(path, reserved.directory.parent / "alias")
    assert path.exists()


def test_write_failure_preserves_partial_and_never_retries(tmp_path, monkeypatch):
    reserved, original = attempt(tmp_path), files.write_file

    def interrupted(*args):
        original(*args)
        raise OSError("after partial retention")

    monkeypatch.setattr(files, "write_file", interrupted)
    with pytest.raises(OSError):
        with module().held_attempt_writer(reserved, names=("run.json",)) as writer:
            writer.retain("run.json", b"partial")
    assert (reserved.directory / "run.json").read_bytes() == b"partial"
    with pytest.raises(ValueError):
        with module().held_attempt_writer(reserved, names=("run.json",)):
            pytest.fail("partial attempt resumed")


def test_sigint_after_file_open_closes_fd_and_keeps_retained_bytes(
    tmp_path, monkeypatch
):
    reserved, original, opened = attempt(tmp_path), os.open, []

    def interrupted(name, flags, *args, **kwargs):
        descriptor = original(name, flags, *args, **kwargs)
        if name == "client-role.json" and flags & os.O_CREAT:
            opened.append(descriptor)
            os.kill(os.getpid(), signal.SIGINT)
        return descriptor

    monkeypatch.setattr(os, "open", interrupted)
    with pytest.raises(KeyboardInterrupt):
        with module().held_attempt_writer(
            reserved, names=("client-role.json",)
        ) as writer:
            writer.retain("client-role.json", b"retained")
    assert (reserved.directory / "client-role.json").read_bytes() == b"retained"
    assert len(opened) == 1
    with pytest.raises(OSError):
        os.fstat(opened[0])


def test_public_check_closes_fd_when_sigint_follows_directory_open(
    tmp_path, monkeypatch
):
    reserved, original, opened = attempt(tmp_path), os.open, []

    def interrupted(name, flags, *args, **kwargs):
        descriptor = original(name, flags, *args, **kwargs)
        if name == "attempt" and not opened:
            opened.append(descriptor)
            os.kill(os.getpid(), signal.SIGINT)
        return descriptor

    with pytest.raises(KeyboardInterrupt):
        with module().held_attempt_writer(reserved, names=("run.json",)) as writer:
            with monkeypatch.context() as patch:
                patch.setattr(os, "open", interrupted)
                writer.check()
    assert len(opened) == 1
    try:
        with pytest.raises(OSError):
            os.fstat(opened[0])
    finally:
        try:
            os.close(opened[0])
        except OSError:
            pass
