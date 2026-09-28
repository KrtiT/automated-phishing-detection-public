"""Child root metadata is hash-authenticated and held before source access."""

import importlib
import importlib.util
from hashlib import sha256
from types import SimpleNamespace

import pytest


def api():
    name = "automated_phishing_detection._study_child_root"
    assert importlib.util.find_spec(name), "missing held child root context"
    return importlib.import_module(name)


def setup(tmp_path, monkeypatch, role="internal"):
    module = api()
    directory = tmp_path / "root"
    directory.mkdir(mode=0o700)
    values = {
        "reservation.json": b'{"name":"reservation"}\n',
        "study-intent.json": b'{"name":"intent"}\n',
        "prediction-barrier.json": b'{"name":"barrier"}\n',
    }
    if role in ("service", "client"):
        values["source-results.json"] = b'{"name":"sources"}\n'
    for name, content in values.items():
        (directory / name).write_bytes(content)
        (directory / name).chmod(0o600)
    frame = SimpleNamespace(
        role=role,
        root_reservation_sha256=sha256(values["reservation.json"]).hexdigest(),
        intent_sha256=sha256(values["study-intent.json"]).hexdigest(),
        barrier_sha256=sha256(values["prediction-barrier.json"]).hexdigest(),
        predecessor_sha256=sha256(values.get("source-results.json", b"")).hexdigest(),
    )
    auth = SimpleNamespace(paths=SimpleNamespace(attempt=directory))
    seen = []
    monkeypatch.setattr(module, "validate_child_root", lambda *args: seen.append(args))
    return SimpleNamespace(**locals())


@pytest.mark.parametrize("role", ["internal", "external", "service", "client"])
def test_exact_root_prefix_reaches_pure_join_once(tmp_path, monkeypatch, role):
    case = setup(tmp_path, monkeypatch, role)
    with case.module.hold_child_root(case.auth, case.frame) as payloads:
        assert payloads == case.values
    assert case.seen == [(case.auth, case.frame, case.values)]


def test_hash_mismatch_rejects_before_parsing(tmp_path, monkeypatch):
    case = setup(tmp_path, monkeypatch)
    (case.directory / "prediction-barrier.json").write_bytes(b"not json")
    with pytest.raises(ValueError):
        with case.module.hold_child_root(case.auth, case.frame):
            pytest.fail("changed barrier admitted")
    assert case.seen == []


def test_late_root_mutation_rejects_on_holder_exit(tmp_path, monkeypatch):
    case = setup(tmp_path, monkeypatch)
    with pytest.raises(ValueError):
        with case.module.hold_child_root(case.auth, case.frame):
            (case.directory / "study-intent.json").write_bytes(b"changed")


def test_added_root_entry_is_not_ignored(tmp_path, monkeypatch):
    case = setup(tmp_path, monkeypatch)
    with pytest.raises(ValueError):
        with case.module.hold_child_root(case.auth, case.frame):
            (case.directory / "old-resume.json").write_bytes(b"{}")


def test_unsafe_root_file_mode_rejects(tmp_path, monkeypatch):
    case = setup(tmp_path, monkeypatch)
    (case.directory / "prediction-barrier.json").chmod(0o644)
    with pytest.raises(ValueError):
        with case.module.hold_child_root(case.auth, case.frame):
            pytest.fail("unsafe root file admitted")


def test_root_failure_does_not_replace_first_interruption(tmp_path, monkeypatch):
    case = setup(tmp_path, monkeypatch)
    original = KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt) as caught:
        with case.module.hold_child_root(case.auth, case.frame):
            (case.directory / "study-intent.json").write_bytes(b"changed")
            raise original
    assert caught.value is original
