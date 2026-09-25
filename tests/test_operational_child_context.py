"""Actual reservation and held immutable inputs precede per-role writes."""

import os
import sys

import pytest
from operational_bound_child_fixtures import child_case, keywords
from operational_child_fixtures import child_context
from operational_transport_integration_fixtures import (
    accepted,
    candidates,
    case,
    manifests,
)

__all__ = ["accepted", "candidates", "case", "manifests", "child_context"]


def test_real_held_context_joins_reservation_and_actual_command(
    child_context, accepted, case, tmp_path, monkeypatch
):
    fixture = child_case(tmp_path, accepted, case, monkeypatch)
    monkeypatch.setattr(child_context, "recheck_binding", lambda binding: None)
    with child_context.held_child(
        fixture.binding, fixture.profile, "client", **keywords(fixture)
    ) as held:
        assert held.inputs.requests == fixture.requests
        assert held.role_context.pid == os.getpid()
        assert held.role_context.command == (sys.executable, *sys.argv)
        held.retain("client-role.json", b"invented\n")
    assert (
        fixture.attempt.directory / "client-role.json"
    ).read_bytes() == b"invented\n"


@pytest.mark.parametrize("change", ["reservation", "profile", "endpoint", "extra_env"])
def test_context_rejects_before_role_retention(
    child_context, accepted, case, tmp_path, monkeypatch, change
):
    fixture = child_case(tmp_path, accepted, case, monkeypatch)
    monkeypatch.setattr(child_context, "recheck_binding", lambda binding: None)
    if change == "reservation":
        monkeypatch.setenv("APD_RESERVATION_SHA256", "f" * 64)
    elif change == "profile":
        fixture.profile.profile_sha256 = "f" * 64
    elif change == "endpoint":
        monkeypatch.setenv("APD_BASE_URL", "http://localhost:1234")
    else:
        monkeypatch.setenv("APD_UNKNOWN", "extra")
    with pytest.raises(Exception):
        with child_context.held_child(
            fixture.binding, fixture.profile, "client", **keywords(fixture)
        ):
            pytest.fail("mismatched inherited context yielded")
    assert set(path.name for path in fixture.attempt.directory.iterdir()) == {
        "reservation.json"
    }


@pytest.mark.parametrize(
    "name", ["service-role.json", "outcome.json", "../escape", "unknown"]
)
def test_client_writer_has_closed_ownership(
    child_context, accepted, case, tmp_path, monkeypatch, name
):
    fixture = child_case(tmp_path, accepted, case, monkeypatch)
    monkeypatch.setattr(child_context, "recheck_binding", lambda binding: None)
    with pytest.raises(Exception):
        with child_context.held_child(
            fixture.binding, fixture.profile, "client", **keywords(fixture)
        ) as held:
            held.retain(name, b"invented")
    assert set(path.name for path in fixture.attempt.directory.iterdir()) == {
        "reservation.json"
    }


def test_actual_reservation_is_authenticated_before_retained_input_read(
    child_context, accepted, case, tmp_path, monkeypatch
):
    fixture = child_case(tmp_path, accepted, case, monkeypatch)
    monkeypatch.setattr(child_context, "recheck_binding", lambda binding: None)
    monkeypatch.setenv("APD_RESERVATION_SHA256", "f" * 64)

    def forbidden(*args, **kwargs):
        pytest.fail("input read preceded actual reservation authentication")

    monkeypatch.setattr(child_context, "hold_operational_inputs", forbidden)
    with pytest.raises(ValueError):
        with child_context.held_child(
            fixture.binding, fixture.profile, "client", **keywords(fixture)
        ):
            pytest.fail("invalid reservation yielded")
