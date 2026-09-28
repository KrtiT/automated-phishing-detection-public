"""Only the complete held admission reaches any scientific child role."""

import asyncio
import importlib
import importlib.util
from contextlib import contextmanager
from types import SimpleNamespace

import pytest


def api():
    name = "automated_phishing_detection._study_child_runtime"
    assert importlib.util.find_spec(name), "missing study child runtime"
    return importlib.import_module(name)


@pytest.mark.parametrize("role", ["internal", "external", "service", "client"])
def test_each_body_is_inside_held_authorization(monkeypatch, role):
    module, events, held = api(), [], object()

    @contextmanager
    def context(arguments):
        events.append("held")
        yield held
        events.append("closed")

    def source(*args):
        assert events == ["held"] and args[0] is held
        events.append("body")

    async def operational(*args):
        source(*args)

    monkeypatch.setattr(module, "held_authorization", context)
    monkeypatch.setattr(module, "run_internal", source)
    monkeypatch.setattr(module, "run_external", source)
    monkeypatch.setattr(module, "run_operational", operational)
    asyncio.run(module.run_child(SimpleNamespace(role=role)))
    assert events == ["held", "body", "closed"]


def test_context_failure_starts_no_scientific_role(monkeypatch):
    module = api()

    @contextmanager
    def context(arguments):
        raise ValueError("invented_admission_failure")
        yield

    monkeypatch.setattr(module, "held_authorization", context)
    monkeypatch.setattr(
        module, "run_internal", lambda *args: pytest.fail("body reached")
    )
    with pytest.raises(ValueError, match="invented_admission_failure"):
        asyncio.run(module.run_child(SimpleNamespace(role="internal")))
