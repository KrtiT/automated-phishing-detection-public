"""Live admission and complete approval precede root and scientific input access."""

import importlib
import importlib.util
from contextlib import contextmanager
from types import SimpleNamespace

import pytest


def api():
    name = "automated_phishing_detection._study_child_context"
    assert importlib.util.find_spec(name), "missing adopted child context"
    return importlib.import_module(name)


def install(module, monkeypatch, events, child, auth):
    monkeypatch.setattr(
        module, "_environment", lambda role: {"APD_STUDY_ADMISSION_FD": "7"}
    )
    monkeypatch.setattr(
        module,
        "consume_child_admission",
        lambda *args, **kwargs: (events.append("consume"), child)[1],
    )
    monkeypatch.setattr(
        module,
        "bind_study_execution",
        lambda *args, **kwargs: (events.append("bind"), auth)[1],
    )
    monkeypatch.setattr(
        module, "recheck_study_execution", lambda value: events.append("recheck")
    )

    @contextmanager
    def root(*args):
        events.append("root")
        yield {"invented": b"root"}
        events.append("root_closed")

    monkeypatch.setattr(module, "hold_child_root", root)


def setup(monkeypatch):
    module, events = api(), []
    frame = SimpleNamespace(profile_sha256="a" * 64, envelope_sha256="b" * 64)
    auth = SimpleNamespace(profile_sha256="a" * 64, envelope_sha256="b" * 64)
    child = SimpleNamespace(
        frame=frame,
        check=lambda: events.append("live"),
        close=lambda: events.append("close"),
    )
    arguments = SimpleNamespace(
        role="internal",
        repo_root="unused",
        expected_revision="unused",
        envelope="unused",
        expected_envelope_sha256="b" * 64,
    )
    install(module, monkeypatch, events, child, auth)
    return SimpleNamespace(**locals())


def test_lifetime_surrounds_complete_bound_child_context(monkeypatch):
    case = setup(monkeypatch)
    with case.module.held_authorization(case.arguments) as held:
        assert held.authorization is case.auth and held.admission is case.child
        case.events.append("body")
    assert (
        case.events.index("consume")
        < case.events.index("bind")
        < case.events.index("root")
    )
    assert (
        case.events.index("body")
        < case.events.index("root_closed")
        < case.events.index("close")
    )
    assert "recheck" in case.events


def test_missing_live_admission_prevents_even_approval_file_read(monkeypatch):
    case = setup(monkeypatch)

    def fail(*args, **kwargs):
        raise ValueError("missing_admission")

    monkeypatch.setattr(case.module, "consume_child_admission", fail)
    with pytest.raises(ValueError):
        with case.module.held_authorization(case.arguments):
            pytest.fail("missing admission reached body")
    assert "bind" not in case.events and "root" not in case.events


@pytest.mark.parametrize("field", ["profile_sha256", "envelope_sha256"])
def test_frame_profile_mismatch_prevents_root_read(monkeypatch, field):
    case = setup(monkeypatch)
    setattr(case.frame, field, "c" * 64)
    with pytest.raises(ValueError):
        with case.module.held_authorization(case.arguments):
            pytest.fail("wrong approval reached body")
    assert "root" not in case.events and "close" in case.events


def test_unknown_apd_environment_rejects_before_consumption(monkeypatch):
    module = api()
    for name in tuple(module.os.environ):
        if name.startswith("APD_"):
            monkeypatch.delenv(name)
    monkeypatch.setenv("APD_STUDY_ADMISSION_FD", "7")
    monkeypatch.setenv("APD_ALLOW_RETRY", "true")
    with pytest.raises(ValueError):
        module._environment("internal")
