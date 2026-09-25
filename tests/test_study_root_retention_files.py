import os
from dataclasses import replace

import pytest
from study_root_retention_fixtures import api, append_all, complete, manager, root_case

from automated_phishing_detection import _study_root_files as storage


def damage(path, kind):
    if kind == "mode":
        path.chmod(0o644)
    elif kind == "hardlink":
        os.link(path, path.parent / "hardlink")
    elif kind == "replace":
        content = path.read_bytes()
        path.unlink()
        path.write_bytes(content)
        path.chmod(0o600)
    elif kind == "symlink":
        path.unlink()
        path.symlink_to("missing")
    elif kind == "bytes":
        path.write_bytes(b"modified")
    elif kind == "fifo":
        path.unlink()
        os.mkfifo(path, 0o600)
    else:
        path.unlink()


@pytest.mark.parametrize(
    "kind", ["mode", "hardlink", "replace", "symlink", "bytes", "fifo", "missing"]
)
@pytest.mark.parametrize("name", ["reservation.json", "study-intent.json"])
def test_original_file_mutations_reject_before_publication(tmp_path, kind, name):
    module, case = api(), root_case(tmp_path)
    with pytest.raises(module.StudyRootRetentionError):
        with manager(module, case) as writer:
            append_all(writer, case)
            damage(case.attempt.directory / name, kind)
            complete(writer, case)
    assert not writer.publishing
    assert not case.public_path.exists()


@pytest.mark.parametrize(
    "kind", ["mode", "replace", "symlink", "bytes", "fifo", "missing"]
)
@pytest.mark.parametrize("selected", ["evidence", "outcome", "public"])
def test_published_files_remain_pinned_through_holder_exit(tmp_path, kind, selected):
    module, case = api(), root_case(tmp_path)
    with pytest.raises(module.StudyRootRetentionError):
        with manager(module, case) as writer:
            append_all(writer, case)
            candidate = complete(writer, case)
            path = {
                "evidence": case.attempt.directory / "evidence/study-intent.json",
                "outcome": case.attempt.directory / "outcome.json",
                "public": case.public_path,
            }[selected]
            if kind == "mode" and selected == "public":
                path.chmod(0o600)
            else:
                damage(path, kind)
    assert writer.candidate is candidate
    assert writer.publishing


@pytest.mark.parametrize("phase", ["entry", "appended", "published"])
def test_unknown_inventory_is_never_whitelisted(tmp_path, phase):
    module, case = api(), root_case(tmp_path)
    if phase == "entry":
        (case.attempt.directory / "foreign").write_bytes(b"invented")
    with pytest.raises(module.StudyRootRetentionError):
        with manager(module, case) as writer:
            append_all(writer, case)
            if phase == "published":
                complete(writer, case)
            (case.attempt.directory / "foreign").write_bytes(b"invented")


@pytest.mark.parametrize(
    "bad", ["identity", "hash", "public_inside", "public_exists", "root_mode"]
)
def test_prelaunch_boundary_rejects_before_body(tmp_path, bad):
    module, case = api(), root_case(tmp_path)
    if bad == "identity":
        case.identity = {"kind": "different"}
    elif bad == "hash":
        case.attempt = replace(case.attempt, reservation_sha256="0" * 64)
    elif bad == "public_inside":
        case.public_path = case.attempt.directory / "public.json"
    elif bad == "public_exists":
        case.public_path.write_bytes(b"occupied")
    else:
        case.attempt.directory.chmod(0o755)
    with pytest.raises(module.StudyRootRetentionError):
        with manager(module, case):
            pytest.fail("invalid boundary entered")


def test_capture_all_published_states_before_first_new_read(tmp_path, monkeypatch):
    module, case = api(), root_case(tmp_path)
    captures, reads = [], []
    original_capture, original_read = storage.capture, storage.read_file

    def capture(directory, name, mode=0o600):
        captures.append((directory.path, name))
        return original_capture(directory, name, mode)

    def read(directory, name, initial):
        if not reads:
            expected = {
                (case.attempt.directory, "finalize.claim"),
                (case.attempt.directory, "outcome.json"),
                (case.public_path.parent, case.public_path.name),
            }
            expected |= {
                (case.attempt.directory / "evidence", name) for name in case.contents
            }
            assert expected <= set(captures)
        reads.append((directory.path, name))
        return original_read(directory, name, initial)

    monkeypatch.setattr(storage, "capture", capture)
    monkeypatch.setattr(storage, "read_file", read)
    with manager(module, case) as writer:
        append_all(writer, case)
        complete(writer, case)
    assert len(reads) == len(set(reads)) == 6


def test_shared_public_parent_is_permitted(tmp_path):
    module, case = api(), root_case(tmp_path)
    case.public_path.parent.chmod(0o755)
    (case.public_path.parent / "unrelated").write_bytes(b"invented")
    with manager(module, case) as writer:
        append_all(writer, case)
        complete(writer, case)
    assert case.public_path.exists()
