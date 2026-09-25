"""Public child gates precede every supplied private path and environment read."""

import asyncio

import pytest
from operational_child_fixtures import child_context, child_module, options, ready

__all__ = ["child_context"]


@pytest.mark.parametrize("role", ["service", "client"])
@pytest.mark.parametrize("closed", ["binding", "profile", "digest"])
def test_public_gate_rejects_before_bound_body(
    child_context, monkeypatch, role, closed
):
    api = child_module(role)
    binding, profile = ready(closed != "binding"), ready(closed != "profile")
    monkeypatch.setattr(
        child_context, "bind_execution", lambda *args, **kwargs: binding
    )
    monkeypatch.setattr(
        child_context, "resolve_operational_profile", lambda value: profile
    )
    supplied = options(role)
    if closed == "digest":
        supplied["expected_operational_profile_sha256"] = "f" * 64

    async def forbidden(*args, **kwargs):
        pytest.fail("public gate reached private child context")

    monkeypatch.setattr(api, f"_run_bound_{role}", forbidden)
    with pytest.raises(child_context.OperationalChildError):
        asyncio.run(getattr(api, f"run_operational_{role}")(object(), **supplied))


@pytest.mark.parametrize("role", ["service", "client"])
def test_binding_failure_remains_first_and_does_not_resolve_profile(
    child_context, monkeypatch, role
):
    error = KeyboardInterrupt("binding")

    def interrupted(*args, **kwargs):
        raise error

    def forbidden(*args):
        pytest.fail("profile resolved after binding failure")

    monkeypatch.setattr(child_context, "bind_execution", interrupted)
    monkeypatch.setattr(child_context, "resolve_operational_profile", forbidden)
    with pytest.raises(KeyboardInterrupt) as caught:
        asyncio.run(
            getattr(child_module(role), f"run_operational_{role}")(
                object(), **options(role)
            )
        )
    assert caught.value is error
