"""Original serve, replay and owner composition accept the exact restored type."""

import asyncio
import json
import os

import pytest
from operational_input_signal_fixtures import assert_closed
from operational_service_child_fixtures import inherited
from study_series_child_inputs_fixtures import (
    candidates,
    child_case,
    manifests,
    series_case,
)
from study_series_child_operational_fixtures import setup
from study_series_child_runtime_fixtures import owner, zero_request_service
from test_study_series_child_operational_interrupts import service_context

from automated_phishing_detection import _study_series_child_runtime as runtime

__all__ = ["candidates", "child_case", "manifests", "series_case"]


def test_original_serve_and_owner_keep_real_holder_and_lifecycle_checks(
    child_case, series_case, tmp_path, monkeypatch
):
    case = setup(child_case, series_case, tmp_path, monkeypatch)
    monkeypatch.setattr(
        runtime, "recheck_held_child", lambda held: case.events.append("boundary")
    )
    monkeypatch.setattr(runtime.service, "serve_service", zero_request_service)
    owner(case, monkeypatch)
    with inherited(monkeypatch) as controls:
        service_context(case)
        asyncio.run(runtime.run_operational(case.held, case.arguments))
        assert os.read(controls[2], 6) == b"ready\n"
        assert_closed(controls[3])
    role = json.loads((case.attempt.directory / "service-role.json").read_bytes())
    assert role["binding_sha256"] == case.arguments.expected_binding_sha256
    assert role["pid"] == os.getpid()
    assert case.events == [
        "recheck",
        "recheck",
        "boundary",
        "session_open",
        "boundary",
        "session_closed",
        "boundary",
        "recheck",
    ]


def test_original_replay_receives_exact_requests_and_retains_original_role(
    child_case, series_case, tmp_path, monkeypatch
):
    case = setup(child_case, series_case, tmp_path, monkeypatch)

    async def replay(base_url, requests, **kwargs):
        assert requests == child_case.expected.computational.requests
        assert kwargs["warmup_count"] == 1000
        return object()

    async def shift(base_url, plan, **kwargs):
        assert plan.requests == child_case.expected.computational.requests
        return object()

    monkeypatch.setattr(runtime, "recheck_held_child", lambda held: None)
    monkeypatch.setattr(runtime.client, "replay_run", replay)
    monkeypatch.setattr(runtime.client, "replay_shift_run", shift)
    monkeypatch.setattr(
        runtime.client, "encode_http_run", lambda value: b"invented run"
    )
    monkeypatch.setattr(
        runtime.client, "encode_shift_run", lambda value: b"invented run"
    )
    asyncio.run(runtime.run_operational(case.held, case.arguments))
    role = json.loads((case.attempt.directory / "client-role.json").read_bytes())
    assert role["binding_sha256"] == case.arguments.expected_binding_sha256
    assert (case.attempt.directory / "run.json").read_bytes() == b"invented run"


def test_real_writer_preserves_partial_failure_without_fabricating_completion(
    child_case, series_case, tmp_path, monkeypatch
):
    case = setup(child_case, series_case, tmp_path, monkeypatch)
    error = ValueError("invented replay failure")
    error.progress = b"invented original partial progress"

    async def replay(*args):
        raise error

    monkeypatch.setattr(runtime, "recheck_held_child", lambda held: None)
    monkeypatch.setattr(runtime.client, "_replay", replay)
    with pytest.raises(ValueError) as caught:
        asyncio.run(runtime.run_operational(case.held, case.arguments))
    assert caught.value is error
    assert (
        case.attempt.directory / "client-failure.json"
    ).read_bytes() == error.progress
    assert not (case.attempt.directory / "run.json").exists()
