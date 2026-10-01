"""Real temporary file identities surround isolated, explicitly stubbed science."""

import os

import pytest
from study_series_completion_fixtures import holder, module, setup, write_working

from automated_phishing_detection._operational_cell_protocol import (
    SNAPSHOT_NAMES,
    WORKING_NAMES,
)


def _damage(path, damage, target):
    if damage == "missing":
        path.unlink()
    elif damage == "mode":
        path.chmod(0o644)
    else:
        path.rename(target)
        if damage == "symlink":
            path.symlink_to(target)
        else:
            os.link(target, path)


@pytest.mark.parametrize("name", WORKING_NAMES)
@pytest.mark.parametrize("damage", ["missing", "mode", "symlink", "hardlink"])
def test_every_working_file_rejects_missing_permission_or_alias(
    tmp_path, monkeypatch, name, damage
):
    case = setup(tmp_path, monkeypatch)
    with pytest.raises(module().OperationalCellCompletionError):
        with holder(case) as completer:
            write_working(case)
            _damage(case.attempt.directory / name, damage, tmp_path / "outside")
            completer.complete(**case.options)
    assert not case.calls and not case.public.exists()


def test_exact17_to36_reads_every_retained_path_once(tmp_path, monkeypatch):
    case = setup(tmp_path, monkeypatch)
    original, reads = module().storage.read_file, []

    def read(directory, name, initial):
        reads.append((directory.path, name))
        return original(directory, name, initial)

    monkeypatch.setattr(module().storage, "read_file", read)
    with holder(case) as completer:
        write_working(case)
        result = completer.complete(**case.options)
        assert len(completer.working.payloads) == len(WORKING_NAMES) == 17
        assert len(result.payloads) == len(SNAPSHOT_NAMES) == 36
    assert len(reads) == len(set(reads)) == 36
    assert {name for path, name in reads if path == case.attempt.directory} == set(
        WORKING_NAMES
    ) | {"finalize.claim", "outcome.json"}


def test_all_working_states_are_captured_before_first_read(tmp_path, monkeypatch):
    case = setup(tmp_path, monkeypatch)
    capture, read = module().storage.capture, module().storage.read_file
    captured = set()

    def remember(directory, name, mode=0o600):
        captured.add(name)
        return capture(directory, name, mode)

    def check(directory, name, initial):
        assert set(WORKING_NAMES) <= captured
        return read(directory, name, initial)

    monkeypatch.setattr(module().storage, "capture", remember)
    monkeypatch.setattr(module().storage, "read_file", check)
    with holder(case) as completer:
        write_working(case)
        completer.complete(**case.options)


@pytest.mark.parametrize("name", ["reservation.json", "run.json", "process-pair.json"])
def test_inplace_change_after_read_rejects_before_science(tmp_path, monkeypatch, name):
    case = setup(tmp_path, monkeypatch)
    original = module().storage.read_file

    def read(directory, selected, initial):
        content = original(directory, selected, initial)
        if selected == name:
            (directory.path / selected).write_bytes(b"x" * len(content))
        return content

    monkeypatch.setattr(module().storage, "read_file", read)
    with pytest.raises(module().OperationalCellCompletionError):
        with holder(case) as completer:
            write_working(case)
            completer.complete(**case.options)
    assert not case.calls and not case.public.exists()


@pytest.mark.parametrize("location", ["public", "run.json", "evidence/run.json"])
@pytest.mark.parametrize("damage", ["inplace", "mode", "symlink"])
def test_late_file_mutation_rejects_with_candidate_preserved(
    tmp_path, monkeypatch, location, damage
):
    case = setup(tmp_path, monkeypatch)
    with pytest.raises(module().OperationalCellCompletionError):
        with holder(case) as completer:
            write_working(case)
            candidate = completer.complete(**case.options)
            path = (
                case.public
                if location == "public"
                else case.attempt.directory / location
            )
            if damage == "inplace":
                path.write_bytes(b"x" * path.stat().st_size)
            elif damage == "mode":
                path.chmod(0o600 if location == "public" else 0o644)
            else:
                path.rename(tmp_path / "outside")
                path.symlink_to(tmp_path / "outside")
    assert completer.candidate is candidate and completer.publishing


@pytest.mark.parametrize("damage", ["extra", "mode", "replacement"])
def test_held_directory_changes_reject_before_working_validation(
    tmp_path, monkeypatch, damage
):
    case = setup(tmp_path, monkeypatch)
    with pytest.raises(module().OperationalCellCompletionError):
        with holder(case) as completer:
            write_working(case)
            if damage == "extra":
                (case.attempt.directory / "extra").write_bytes(b"invented")
            elif damage == "mode":
                case.attempt.directory.chmod(0o755)
            else:
                case.attempt.directory.rename(tmp_path / "old")
                case.attempt.directory.mkdir(mode=0o700)
            completer.complete(**case.options)
    assert not case.calls and not case.public.exists()


def test_final_check_failure_keeps_returned_candidate_and_blocks_retry(
    tmp_path, monkeypatch
):
    case = setup(tmp_path, monkeypatch)
    original, candidates = module()._verify_published, []

    def fail(*arguments):
        raise OSError("invented final held-file failure")

    def publish(*arguments, **keywords):
        candidate = original(*arguments, **keywords)
        candidates.append(candidate)
        monkeypatch.setattr(module().storage.HeldCellFiles, "check", fail)
        return candidate

    monkeypatch.setattr(module(), "_verify_published", publish)
    with pytest.raises(module().OperationalCellCompletionError):
        with holder(case) as completer:
            write_working(case)
            completer.complete(**case.options)
    assert completer.candidate is candidates[0]
    assert len(candidates) == 1 and case.public.exists()
    with pytest.raises(module().OperationalCellCompletionError):
        completer.complete(**case.options)
