import os
from hashlib import sha256

import pytest
from study_preparation_transport_fixtures import (
    inputs,
    module,
    preparation_api,
    preparation_case,
    retained_case,
    runner,
)

__all__ = ["inputs", "preparation_api", "preparation_case", "retained_case", "runner"]


def rejected(retained):
    api = module()
    with pytest.raises(api.StudyPreparationTransportError) as captured:
        with api.hold_study_preparation(retained.directory, **retained.expected):
            pytest.fail("invalid preparation accepted")
    assert str(captured.value) == "invalid_study_preparation_transport"
    assert retained.directory.is_dir()


@pytest.mark.parametrize(
    "name", ["expected_reservation_sha256", "expected_completion_sha256"]
)
@pytest.mark.parametrize("value", ["f" * 64, "F" * 64, "invalid", None, True])
def test_reader_rejects_independent_digest_mismatch(retained_case, name, value):
    retained_case.expected[name] = value
    rejected(retained_case)


@pytest.mark.parametrize(
    "mutation",
    ["missing", "extra", "directory_mode", "file_mode", "symlink", "hardlink"],
)
def test_reader_rejects_unsafe_inventory(retained_case, mutation, tmp_path):
    retained = retained_case
    selected = retained.directory / "suffix-rules.dat"
    if mutation == "missing":
        selected.unlink()
    elif mutation == "extra":
        (retained.directory / "extra").write_bytes(b"invented")
    elif mutation == "directory_mode":
        retained.directory.chmod(0o750)
    elif mutation == "file_mode":
        selected.chmod(0o644)
    elif mutation == "hardlink":
        os.link(selected, tmp_path / "linked")
    else:
        moved = tmp_path / "moved"
        selected.rename(moved)
        selected.symlink_to(moved)
    rejected(retained)


def test_reader_authenticates_reservation_before_pure_restoration(
    retained_case, monkeypatch
):
    retained, api = retained_case, module()
    selected = retained.directory / "reservation.json"
    content = selected.read_bytes() + b"\n"
    selected.write_bytes(content)
    retained.expected["expected_reservation_sha256"] = sha256(content).hexdigest()
    monkeypatch.setattr(
        api, "_restore", lambda *_: pytest.fail("invalid reservation reached restore")
    )
    rejected(retained)


def test_reader_rejects_identity_substitution(retained_case):
    retained_case.expected["expected_identity"]["archive_sha256"] = "f" * 64
    rejected(retained_case)


@pytest.mark.parametrize("mutation", ["rewrite", "replace", "extra", "mode"])
def test_reader_rejects_changes_during_held_body(retained_case, mutation):
    retained, api = retained_case, module()
    selected = retained.directory / "suffix-rules.dat"
    entered = False
    with pytest.raises(api.StudyPreparationTransportError):
        with api.hold_study_preparation(retained.directory, **retained.expected):
            entered = True
            if mutation == "rewrite":
                selected.write_bytes(selected.read_bytes())
            elif mutation == "replace":
                original = selected.read_bytes()
                selected.unlink()
                selected.write_bytes(original)
                selected.chmod(0o600)
            elif mutation == "mode":
                selected.chmod(0o644)
            else:
                (retained.directory / "extra").write_bytes(b"invented")
    assert entered and retained.directory.is_dir()


def test_reader_rejects_replacement_between_capture_and_open(
    retained_case, monkeypatch
):
    retained, api = retained_case, module()
    original = api._read_file

    def substituted(directory, name, initial):
        if name == "suffix-rules.dat":
            selected = retained.directory / name
            content = selected.read_bytes()
            selected.unlink()
            selected.write_bytes(content)
            selected.chmod(0o600)
        return original(directory, name, initial)

    monkeypatch.setattr(api, "_read_file", substituted)
    rejected(retained)


def test_reader_rejects_held_directory_replacement(retained_case):
    retained, api = retained_case, module()
    original = retained.directory.with_name("retained-original")
    entered = False
    with pytest.raises(api.StudyPreparationTransportError):
        with api.hold_study_preparation(retained.directory, **retained.expected):
            entered = True
            retained.directory.rename(original)
            retained.directory.mkdir(mode=0o700)
    assert entered and original.is_dir() and retained.directory.is_dir()
