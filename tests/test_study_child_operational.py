"""Adopted operational roles preserve strict old context and numerical bodies."""

import asyncio
import importlib
import importlib.util
from contextlib import contextmanager
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import pytest


def api():
    name = "automated_phishing_detection._study_child_operational"
    assert importlib.util.find_spec(name), "missing admitted operational role"
    return importlib.import_module(name)


def held_context(auth, seen):
    frame = SimpleNamespace(
        cell_binding_sha256="a" * 64,
        accepted_inputs_sha256=sha256(b"accepted").hexdigest(),
    )
    held = SimpleNamespace(
        authorization=auth,
        admission=SimpleNamespace(frame=frame, check=lambda: seen.append("live")),
        environment={
            "APD_STUDY_ADMISSION_FD": "7",
            "APD_ATTEMPT_DIRECTORY": "/invented/cells/cell-001-attempt",
            "APD_RESERVATION_SHA256": "b" * 64,
            "APD_BASE_URL": "http://127.0.0.1:12345",
        },
    )
    return frame, held


def runtime_context(runtime, seen):
    @contextmanager
    def context(binding, profile, role, environment, paths, digest, *, lifecycle_check):
        assert "APD_STUDY_ADMISSION_FD" not in environment
        assert callable(lifecycle_check)
        seen.append("held")
        yield runtime
        seen.append("closed")

    return context


def setup(monkeypatch, role="client"):
    module, seen = api(), []
    paths = SimpleNamespace(
        cells_directory=Path("/invented/cells"),
        accepted_inputs_directory=Path("/invented/accepted"),
        internal=SimpleNamespace(artifacts=object()),
    )
    auth = SimpleNamespace(
        base=SimpleNamespace(root=Path("/invented/repo")),
        operational=object(),
        paths=paths,
    )
    frame, held = held_context(auth, seen)
    args = SimpleNamespace(role=role, cell_ordinal=1, expected_binding_sha256="a" * 64)
    inputs = SimpleNamespace(
        cell=SimpleNamespace(ordinal=1), accepted_bytes=b"accepted"
    )
    runtime = SimpleNamespace(
        inputs=inputs, retain=lambda name, content: seen.append((name, content))
    )
    monkeypatch.setattr(module.context, "_held", runtime_context(runtime, seen))
    monkeypatch.setattr(
        module, "recheck_held_child", lambda value: seen.append("study_recheck")
    )
    return SimpleNamespace(**locals())


def test_client_reuses_original_replay_and_retains_original_run(monkeypatch):
    case = setup(monkeypatch)

    async def replay(value):
        assert value is case.runtime and case.seen == ["held", "live", "study_recheck"]
        return b"invented run"

    monkeypatch.setattr(case.module.client, "_replay", replay)
    asyncio.run(case.module.run_operational(case.held, case.args))
    assert case.seen[-2:] == [("run.json", b"invented run"), "closed"]


def test_service_reuses_original_owner_and_serve_body(monkeypatch):
    case = setup(monkeypatch, "service")

    async def serve(runtime, binding, artifacts, *, lifecycle_check):
        assert runtime is case.runtime and binding is case.auth.base
        assert artifacts is case.paths.internal.artifacts
        assert callable(lifecycle_check)
        case.seen.append("serve")

    monkeypatch.setattr(case.module.service, "_serve", serve)
    asyncio.run(case.module.run_operational(case.held, case.args))
    assert case.seen == ["held", "live", "serve", "closed"]


def test_wrong_cell_binding_rejects_before_held_input_read(monkeypatch):
    case = setup(monkeypatch)
    case.args.expected_binding_sha256 = "c" * 64
    with pytest.raises(ValueError):
        asyncio.run(case.module.run_operational(case.held, case.args))
    assert case.seen == []


def test_wrong_attempt_environment_rejects_before_held_input_read(monkeypatch):
    case = setup(monkeypatch)
    case.held.environment["APD_ATTEMPT_DIRECTORY"] = "/invented/other"
    with pytest.raises(ValueError):
        asyncio.run(case.module.run_operational(case.held, case.args))
    assert case.seen == []


def test_wrong_accepted_context_starts_no_replay(monkeypatch):
    case = setup(monkeypatch)
    case.inputs.accepted_bytes = b"swapped"
    monkeypatch.setattr(
        case.module.client, "_replay", lambda value: pytest.fail("replay started")
    )
    with pytest.raises(ValueError):
        asyncio.run(case.module.run_operational(case.held, case.args))
