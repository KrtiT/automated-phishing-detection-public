"""Real singleton inference and live monitoring share one artifact owner."""

import asyncio
import json
import threading
from dataclasses import replace

import pytest
from operational_real_owner_fixtures import real_case, retain
from operational_runtime_fixtures import records, runtime
from test_bound_models import fixture, module
from test_operational_role_records import context
from test_operational_runtime import reject_startup
from test_selective_service import client_for

from automated_phishing_detection._checkpoint_codec import canonical_bytes

__all__ = ["fixture", "module", "records", "runtime"]


async def request_one(app, case, ordinal):
    async with app.router.lifespan_context(app), client_for(app) as client:
        assert case.loads == ["length", "transformer"]
        request_id = (
            app.state.owner.plan.request_id("warmup", 0)
            if ordinal == 121
            else "fixture-request"
        )
        response = await client.post(
            "/v1/scan",
            json={"request_id": request_id, "url": case.inputs.requests[0].raw_url},
        )
        assert response.status_code == 200
        assert case.session.counts.completed_requests == 1
        if ordinal == 91:
            assert response.json()["stage2_invoked"] is True
        if ordinal == 121:
            state = (await client.post("/v1/shift/state")).json()
            assert state["manifest_sha256"] == case.inputs.manifest_sha256
            assert state["run_index"] == case.inputs.cell.run_index
            assert state["warmup_count"] == 1000 and len(state["rows"]) == 1


@pytest.mark.parametrize("ordinal", [1, 91, 121])
def test_all_workloads_reuse_real_singleton_owner(
    runtime, records, fixture, monkeypatch, ordinal
):
    case = real_case(fixture, monkeypatch, ordinal)
    app = runtime.create_operational_app(
        case.binding,
        case.paths,
        case.inputs,
        role_context=context(records),
        retain=lambda name, content: retain(case, name, content),
    )
    assert case.loads == [] and case.events == [] and case.session is None
    asyncio.run(request_one(app, case, ordinal))
    assert case.loads == ["length", "transformer"]
    assert [name for name, unused in case.events] == ["load", "enter", "role", "exit"]
    threads = {thread for unused, thread in case.events} | set(case.forwards)
    assert len(threads) == 1 and threading.get_ident() not in threads
    assert not app.state.owner.is_alive
    assert case.session.counts.transformer_forward_attempts == len(case.forwards)
    records.verify_role_record(
        case.retained[0][1],
        inputs=case.inputs,
        context=context(records),
        role="service",
    )


@pytest.mark.parametrize("field", ["artifact_hashes", "thresholds"])
def test_real_loaded_mismatch_never_forwards(
    runtime, records, fixture, monkeypatch, field
):
    case = real_case(fixture, monkeypatch, 91)
    accepted = json.loads(case.inputs.accepted_bytes)
    if field == "artifact_hashes":
        accepted["primary"][field]["cascade.json"] = "f" * 64
    else:
        accepted["primary"][field]["monitor_boundary"] += 1.0
    case.inputs = replace(case.inputs, accepted_bytes=canonical_bytes(accepted))
    app = runtime.create_operational_app(
        case.binding,
        case.paths,
        case.inputs,
        role_context=context(records),
        retain=lambda name, content: retain(case, name, content),
    )
    asyncio.run(reject_startup(app))
    assert case.loads == ["length", "transformer"]
    assert case.forwards == [] and case.retained == []
    assert [name for name, unused in case.events] == ["load", "enter", "exit"]
    assert not app.state.owner.is_alive
