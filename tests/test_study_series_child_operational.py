"""The unchanged runtime receives exact admitted computations and actual ownership."""

import os
import sys
from dataclasses import replace

import pytest
from study_series_child_inputs_fixtures import (
    candidates,
    child_case,
    manifests,
    series_case,
)
from study_series_child_operational_fixtures import hold, setup

from automated_phishing_detection._checkpoint_codec import canonical_bytes
from automated_phishing_detection.operational_cell_inputs import RestoredOperationalCell

__all__ = ["candidates", "child_case", "manifests", "series_case"]


def test_actual_holder_yields_original_computational_type_and_owned_writer(
    child_case, series_case, tmp_path, monkeypatch
):
    case = setup(child_case, series_case, tmp_path, monkeypatch)
    with hold(case) as runtime:
        assert type(runtime.inputs) is RestoredOperationalCell
        assert runtime.inputs.binding_bytes == case.selected.binding
        assert runtime.inputs.requests == child_case.expected.computational.requests
        assert runtime.role_context.pid == os.getpid()
        assert runtime.role_context.command == (sys.executable, *sys.argv)
        assert runtime.role_context.base_url == case.held.environment["APD_BASE_URL"]
        assert runtime.handles == ()
        runtime.retain("client-role.json", b"invented role")
    assert (
        case.attempt.directory / "client-role.json"
    ).read_bytes() == b"invented role"
    assert case.events == ["recheck", "recheck", "recheck"]


@pytest.mark.parametrize("change", ("binding", "ordinal", "attempt", "role"))
def test_scalar_rejection_precedes_private_holders(
    child_case, series_case, tmp_path, monkeypatch, change
):
    case = setup(child_case, series_case, tmp_path, monkeypatch)
    if change == "binding":
        case.arguments.expected_binding_sha256 = "f" * 64
    elif change == "ordinal":
        case.arguments.cell_ordinal += 1
    elif change == "role":
        case.arguments.role = "external"
    else:
        case.held.environment["APD_ATTEMPT_DIRECTORY"] += "-other"
    monkeypatch.setattr(
        case.module, "held_writer", lambda *args: pytest.fail("writer opened")
    )
    with pytest.raises(ValueError):
        with hold(case):
            pytest.fail("invalid scalar context yielded")


@pytest.mark.parametrize(
    "content", (b"{}\n", canonical_bytes({"origin_metadata_sha256": "f" * 64}))
)
def test_missing_or_wrong_origin_metadata_pin_rejects_before_runtime(
    child_case, series_case, tmp_path, monkeypatch, content
):
    case = setup(child_case, series_case, tmp_path, monkeypatch)
    case.held.prefix_payloads = (("segment/history-import.json", content),)
    with pytest.raises(ValueError):
        with hold(case):
            pytest.fail("origin metadata not joined")


@pytest.mark.parametrize("change", ("profile", "execution", "reservation"))
def test_current_runtime_and_real_reservation_must_join(
    child_case, series_case, tmp_path, monkeypatch, change
):
    case = setup(child_case, series_case, tmp_path, monkeypatch)
    if change == "profile":
        case.auth.operational.profile_sha256 = "f" * 64
    elif change == "execution":
        case.auth.base = replace(case.auth.base, revision="f" * 40)
    else:
        case.held.environment["APD_RESERVATION_SHA256"] = "f" * 64
        monkeypatch.setattr(
            case.module,
            "hold_series_child_inputs",
            lambda *args, **kwargs: pytest.fail(
                "inputs read before reservation authenticated"
            ),
        )
    with pytest.raises(ValueError):
        with hold(case):
            pytest.fail("mismatched execution yielded")


def test_final_public_recheck_carries_earlier_failure_progress(
    child_case, series_case, tmp_path, monkeypatch
):
    case = setup(child_case, series_case, tmp_path, monkeypatch)
    original, later = ValueError("body"), KeyboardInterrupt("cleanup")
    original.progress = b"invented prior failure"

    def recheck(held):
        case.events.append("recheck")
        if len(case.events) == 3:
            raise later

    monkeypatch.setattr(case.module, "recheck_held_child", recheck)
    with pytest.raises(KeyboardInterrupt) as caught:
        with hold(case):
            raise original
    assert caught.value is later and later.progress == original.progress
