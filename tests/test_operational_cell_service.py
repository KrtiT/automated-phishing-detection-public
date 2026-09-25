"""The inherited endpoint becomes ready only after durable bound role records."""

import asyncio
import json
import os

import httpx
import pytest
from operational_bound_child_fixtures import child_case, keywords
from operational_child_fixtures import child_context, child_module
from operational_service_child_fixtures import inherited, synthetic_primary
from operational_transport_integration_fixtures import (
    accepted,
    candidates,
    case,
    manifests,
)

from automated_phishing_detection.bound_models import ArtifactPaths

__all__ = ["accepted", "candidates", "case", "manifests", "child_context"]


async def exchange(api, fixture, artifacts, controls):
    base_url, stop_write, ready_read, unused = controls
    task = asyncio.create_task(
        api._run_bound_service(
            fixture.binding, fixture.profile, artifacts=artifacts, **keywords(fixture)
        )
    )
    try:
        assert (
            await asyncio.wait_for(asyncio.to_thread(os.read, ready_read, 6), 5)
            == b"ready\n"
        )
        ready = json.loads(
            (fixture.attempt.directory / "service-ready.json").read_bytes()
        )
        assert ready["port"] == int(base_url.rsplit(":", 1)[1])
        assert (fixture.attempt.directory / "service-role.json").is_file()
        async with httpx.AsyncClient(trust_env=False) as client:
            response = await client.post(
                base_url + "/v1/scan",
                json={"request_id": "invented", "url": "https://stage2.test"},
            )
            assert response.status_code == 200
        os.write(stop_write, b"stop\n")
        await asyncio.wait_for(task, 5)
    finally:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


def test_real_service_retains_then_notifies_and_closes_handles(
    child_context, accepted, case, tmp_path, monkeypatch
):
    api = child_module("service")
    fixture = child_case(tmp_path, accepted, case, monkeypatch, 1)
    monkeypatch.setattr(child_context, "recheck_binding", lambda binding: None)
    scorer = synthetic_primary(
        monkeypatch, json.loads(accepted.metadata_bytes)["primary"]
    )
    artifacts = ArtifactPaths(
        *(tmp_path / name for name in ("length", "logistic", "transformer", "gmm"))
    )
    with inherited(monkeypatch) as controls:
        asyncio.run(exchange(api, fixture, artifacts, controls))
        for descriptor in controls[3]:
            with pytest.raises(OSError):
                os.fstat(descriptor)
    assert scorer.urls == ["https://stage2.test"]
    cleanup = json.loads(
        (fixture.attempt.directory / "service-cleanup.json").read_bytes()
    )
    assert cleanup["status"] == "clean"
    assert [name for name, unused in scorer.events][-1] == "exit"


def test_endpoint_rejection_closes_all_inherited_descriptors(
    child_context, accepted, case, tmp_path, monkeypatch
):
    fixture = child_case(tmp_path, accepted, case, monkeypatch, 1)
    monkeypatch.setattr(child_context, "recheck_binding", lambda binding: None)
    with inherited(monkeypatch) as controls:
        monkeypatch.setenv("APD_BASE_URL", "http://127.0.0.1:1")
        with pytest.raises(Exception):
            with child_context.held_child(
                fixture.binding, fixture.profile, "service", **keywords(fixture)
            ):
                pytest.fail("different endpoint was accepted")
        for descriptor in controls[3]:
            with pytest.raises(OSError):
                os.fstat(descriptor)
