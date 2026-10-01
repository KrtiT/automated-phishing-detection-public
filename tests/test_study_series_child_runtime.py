"""Only original operational bodies execute inside the joined series holders."""

import asyncio
from contextlib import contextmanager
from dataclasses import fields
from importlib import import_module
from importlib.util import find_spec
from pathlib import Path
from types import SimpleNamespace

import pytest
from study_series_adoption_fixtures import make_case

from automated_phishing_detection._checkpoint_codec import canonical_bytes
from automated_phishing_detection.bound_models import ArtifactPaths


def api():
    name = "automated_phishing_detection._study_series_child_runtime"
    assert find_spec(name), "missing series child role dispatcher"
    return import_module(name)


def setup(monkeypatch, role="client"):
    module, events, runtime = api(), [], object()
    profile = canonical_bytes(make_case().profile)
    held = SimpleNamespace(
        authorization=SimpleNamespace(base=object(), profile_bytes=profile)
    )

    @contextmanager
    def authorized(arguments):
        events.append("authorized")
        yield held
        events.append("authorization_closed")

    @contextmanager
    def operational(value, arguments):
        assert value is held and arguments.role == role
        events.append("operational")
        yield runtime
        events.append("operational_closed")

    monkeypatch.setattr(module, "held_authorization", authorized)
    monkeypatch.setattr(module, "held_operational", operational)
    monkeypatch.setattr(
        module, "recheck_held_child", lambda value: events.append("recheck")
    )
    return SimpleNamespace(**locals())


def test_service_uses_original_serve_and_embedded_original_artifacts(monkeypatch):
    case = setup(monkeypatch, "service")

    async def serve(runtime, binding, artifacts, *, lifecycle_check):
        assert runtime is case.runtime and binding is case.held.authorization.base
        expected = make_case().profile["origin"]["profile"]["paths"]
        assert type(artifacts) is ArtifactPaths
        for selected in fields(artifacts):
            assert getattr(artifacts, selected.name) == Path(
                expected[selected.name.replace("_", "-")]
            )
        lifecycle_check()
        case.events.append("serve")

    monkeypatch.setattr(case.module.service, "_serve", serve)
    asyncio.run(case.module.run_child(SimpleNamespace(role="service")))
    assert case.events == [
        "authorized",
        "operational",
        "recheck",
        "serve",
        "operational_closed",
        "authorization_closed",
    ]


def test_client_uses_original_replay_between_lifecycle_checks(monkeypatch):
    case = setup(monkeypatch)
    runtime = SimpleNamespace(
        retain=lambda name, content: case.events.append((name, content))
    )

    async def replay(selected):
        assert selected is runtime and case.events == ["recheck"]
        case.events.append("replay")
        return b"invented run"

    monkeypatch.setattr(case.module.client, "_replay", replay)
    asyncio.run(case.module._client(runtime, lambda: case.events.append("recheck")))
    assert case.events == [
        "recheck",
        "replay",
        "recheck",
        ("run.json", b"invented run"),
    ]


@pytest.mark.parametrize(
    "error", (ValueError("failure"), KeyboardInterrupt("stop"), SystemExit(9))
)
def test_client_retains_original_partial_progress_and_original_exception(
    monkeypatch, error
):
    case = setup(monkeypatch)
    error.progress = b"invented original progress"
    runtime = SimpleNamespace(
        retain=lambda name, content: case.events.append((name, content))
    )

    async def replay(selected):
        raise error

    monkeypatch.setattr(case.module.client, "_replay", replay)
    with pytest.raises(type(error)) as caught:
        asyncio.run(case.module._client(runtime, lambda: None))
    assert caught.value is error
    assert case.events == [("client-failure.json", error.progress)]


def test_no_dispatch_after_admission_failure(monkeypatch):
    module = api()

    @contextmanager
    def rejected(arguments):
        raise ValueError("no_live_admission")
        yield

    monkeypatch.setattr(module, "held_authorization", rejected)
    monkeypatch.setattr(
        module, "run_operational", lambda *args: pytest.fail("role started")
    )
    with pytest.raises(ValueError, match="no_live_admission"):
        asyncio.run(module.run_child(SimpleNamespace(role="service")))


@pytest.mark.parametrize("role", ("internal", "external", "arbitrary"))
def test_dispatcher_has_no_other_roles(monkeypatch, role):
    case = setup(monkeypatch, role)
    with pytest.raises(ValueError):
        asyncio.run(case.module.run_child(SimpleNamespace(role=role)))
    assert "operational" not in case.events
