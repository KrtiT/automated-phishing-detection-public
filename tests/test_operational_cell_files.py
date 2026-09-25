import os

import pytest
from operational_cell_io_fixtures import holder, module, setup, write_working

from automated_phishing_detection._operational_cell_protocol import WORKING_NAMES


@pytest.mark.parametrize("name", WORKING_NAMES)
@pytest.mark.parametrize("damage", ["missing", "mode", "symlink", "hardlink"])
def test_working_inventory_rejects_missing_modes_and_aliases(
    tmp_path, monkeypatch, name, damage
):
    case = setup(tmp_path, monkeypatch)
    with pytest.raises(module().OperationalCellCompletionError):
        with holder(case) as completer:
            write_working(case)
            path = case.attempt.directory / name
            if damage == "missing":
                path.unlink()
            elif damage == "mode":
                path.chmod(0o644)
            else:
                target = tmp_path / "outside"
                path.rename(target)
                if damage == "symlink":
                    path.symlink_to(target)
                else:
                    os.link(target, path)
            completer.complete(**case.options)
    assert not case.public.exists()
    assert not case.calls


@pytest.mark.parametrize("damage", ["extra", "directory_mode", "directory_swap"])
def test_working_root_changes_reject_before_pure_verification(
    tmp_path, monkeypatch, damage
):
    case = setup(tmp_path, monkeypatch)
    with pytest.raises(module().OperationalCellCompletionError):
        with holder(case) as completer:
            write_working(case)
            if damage == "extra":
                (case.attempt.directory / "unexpected").write_bytes(b"invented")
            elif damage == "directory_mode":
                case.attempt.directory.chmod(0o755)
            else:
                case.attempt.directory.rename(tmp_path / "old")
                case.attempt.directory.mkdir(mode=0o700)
            completer.complete(**case.options)
    assert not case.calls


def test_initial_reservation_inode_cannot_be_replaced_during_final_capture(
    tmp_path, monkeypatch
):
    case = setup(tmp_path, monkeypatch)
    original = module().storage.capture
    calls = []

    def capture(directory, name, mode=0o600):
        if name == "reservation.json":
            calls.append(name)
            if len(calls) == 2:
                path = directory.path / name
                content = path.read_bytes()
                path.unlink()
                path.write_bytes(content)
                path.chmod(0o600)
        return original(directory, name, mode)

    with pytest.raises(module().OperationalCellCompletionError):
        with holder(case) as completer:
            write_working(case)
            monkeypatch.setattr(module().storage, "capture", capture)
            completer.complete(**case.options)
    assert not case.calls


def test_every_working_state_precedes_first_read(tmp_path, monkeypatch):
    case = setup(tmp_path, monkeypatch)
    original, captured = module().storage.capture, set()
    original_read = module().storage.read_file

    def capture(directory, name, mode=0o600):
        captured.add(name)
        return original(directory, name, mode)

    def read(directory, name, initial):
        assert set(WORKING_NAMES) <= captured
        return original_read(directory, name, initial)

    monkeypatch.setattr(module().storage, "capture", capture)
    monkeypatch.setattr(module().storage, "read_file", read)
    with holder(case) as completer:
        write_working(case)
        completer.complete(**case.options)


@pytest.mark.parametrize(
    "name", ["public", "run.json", "copied_run", "extra", "evidence_mode"]
)
def test_late_publication_changes_reject_with_candidate_retained(
    tmp_path, monkeypatch, name
):
    case = setup(tmp_path, monkeypatch)
    with pytest.raises(module().OperationalCellCompletionError):
        with holder(case) as completer:
            write_working(case)
            result = completer.complete(**case.options)
            if name == "public":
                case.public.write_bytes(b"changed")
            elif name == "copied_run":
                (case.attempt.directory / "evidence/run.json").write_bytes(b"changed")
            elif name == "evidence_mode":
                (case.attempt.directory / "evidence").chmod(0o755)
            else:
                (case.attempt.directory / name).write_bytes(b"changed")
    assert completer.candidate is result
    assert completer.publishing


def test_existing_public_marker_rejected_before_body(tmp_path, monkeypatch):
    case = setup(tmp_path, monkeypatch)
    case.public.write_bytes(b"original")
    with pytest.raises(module().OperationalCellCompletionError):
        with holder(case):
            pytest.fail("existing public marker entered")
    assert case.public.read_bytes() == b"original"


def test_public_marker_cannot_reside_inside_attempt(tmp_path, monkeypatch):
    case = setup(tmp_path, monkeypatch)
    case.public = case.attempt.directory / "public.json"
    with pytest.raises(module().OperationalCellCompletionError):
        with holder(case):
            pytest.fail("private/public alias entered")
