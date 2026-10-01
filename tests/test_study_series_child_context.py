"""The live parent, current public binding and final policy precede private IO."""

import json
import sys
from dataclasses import replace

import pytest
from study_series_child_context_fixtures import api, change_frame, setup
from study_series_execution_fixtures import public_case

from automated_phishing_detection._checkpoint_codec import canonical_bytes

__all__ = ["public_case"]


def test_missing_live_channel_prevents_even_public_binding(public_case, monkeypatch):
    case = setup(public_case, monkeypatch)

    def reject(*args, **kwargs):
        raise ValueError("missing_live_channel")

    monkeypatch.setattr(case.module, "consume_series_admission", reject)
    with pytest.raises(ValueError, match="missing_live_channel"):
        with case.module.held_authorization(case.arguments):
            pytest.fail("unadmitted body")
    assert "bind" not in case.events and "prefix" not in case.events


def test_current_candidate_rejects_before_private_prefix(public_case, monkeypatch):
    case = setup(public_case, monkeypatch, allow_execution=False)
    candidate = json.loads(case.binding.policy_bytes) | {
        "status": "development_candidate_header_only"
    }
    case.binding = replace(case.binding, policy_bytes=canonical_bytes(candidate))
    with pytest.raises(ValueError, match="series_child_execution_not_adopted"):
        with case.module.held_authorization(case.arguments):
            pytest.fail("candidate reached private IO")
    assert "prefix" not in case.events and case.events[-1] == "close"


@pytest.mark.parametrize("role", ("service", "client"))
def test_live_public_prefix_lifetimes_are_nested(public_case, monkeypatch, role):
    case = setup(public_case, monkeypatch, role)
    with case.module.held_authorization(case.arguments) as held:
        assert held.authorization is case.binding and held.admission is case.child
        assert held.prefix_payloads == (("invented", b"prefix"),)
        case.events.append("body")
        case.module.recheck_held_child(held)
    assert case.events.index("consume") < case.events.index("bind")
    assert case.events.index("public_recheck") < case.events.index("prefix")
    assert case.events.index("body") < case.events.index("prefix_closed")
    assert case.events[-1] == "close"
    assert case.events.count("public_recheck") >= 3


@pytest.mark.parametrize(
    "field", ("profile_sha256", "envelope_sha256", "cell_binding_sha256")
)
def test_frame_pin_substitutions_prevent_prefix_read(public_case, monkeypatch, field):
    case = setup(public_case, monkeypatch)
    change_frame(case, **{field: "c" * 64})
    with pytest.raises(ValueError):
        with case.module.held_authorization(case.arguments):
            pytest.fail("substituted frame admitted")
    assert "prefix" not in case.events and case.events[-1] == "close"


@pytest.mark.parametrize("change", ("ordinal", "argv", "binding"))
def test_exact_command_and_cell_are_rechecked(public_case, monkeypatch, change):
    case = setup(public_case, monkeypatch)
    if change == "ordinal":
        change_frame(case, cell_ordinal=74)
    elif change == "argv":
        monkeypatch.setattr(sys, "argv", [*sys.argv, "--timeout", "999"])
    else:
        case.arguments.expected_binding_sha256 = "c" * 64
    with pytest.raises(ValueError):
        with case.module.held_authorization(case.arguments):
            pytest.fail("changed command admitted")
    assert "prefix" not in case.events


@pytest.mark.parametrize("role", ("internal", "external", "parent", None))
def test_only_operational_roles_are_accepted(monkeypatch, role):
    with pytest.raises(ValueError):
        api()._environment(role)


@pytest.mark.parametrize(
    "name", ("APD_STUDY_ADMISSION_FD", "APD_ALLOW_RETRY", "APD_READY_FD")
)
def test_extra_environment_rejects_before_admission(public_case, monkeypatch, name):
    case = setup(public_case, monkeypatch)
    monkeypatch.setenv(name, "8")
    with pytest.raises(ValueError):
        with case.module.held_authorization(case.arguments):
            pytest.fail("extra environment admitted")
    assert case.events == []


@pytest.mark.parametrize("value", ("2", "08", "+8", "-8", "9"))
def test_bad_service_descriptor_rejects_before_consume(public_case, monkeypatch, value):
    case = setup(public_case, monkeypatch, "service")
    monkeypatch.setenv("APD_LISTENER_FD", value)
    with pytest.raises(ValueError):
        with case.module.held_authorization(case.arguments):
            pytest.fail("invalid descriptor admitted")
    assert "consume" not in case.events
