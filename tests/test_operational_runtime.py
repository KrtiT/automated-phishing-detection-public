"""Operational composition owns one lazy session, never a second scoring core."""

import asyncio
import json
import threading

import pytest
from operational_owner_fixtures import owner_case, retain
from operational_runtime_fixtures import ARTIFACTS, THRESHOLDS, records, runtime
from test_operational_role_records import context
from test_selective_service import client_for

__all__ = ["records", "runtime"]


def app_for(api, records, case, writer=None):
    return api.create_operational_app(
        case.binding,
        case.paths,
        case.inputs,
        role_context=context(records),
        retain=writer or (lambda name, content: retain(case, name, content)),
    )


def test_fixed_app_is_lazy_and_all_work_stays_on_one_owner(
    runtime, records, monkeypatch
):
    case = owner_case(monkeypatch)
    app = app_for(runtime, records, case)
    assert case.events == [] and case.scorer is None and case.retained == []

    async def exercise():
        async with app.router.lifespan_context(app), client_for(app) as client:
            assert len(case.retained) == 1 and case.scorer.urls == []
            response = await client.post(
                "/v1/scan", json={"request_id": "one", "url": "https://stage2.test"}
            )
            assert response.status_code == 200

    asyncio.run(exercise())
    assert [event for event, unused in case.events] == [
        "check",
        "load",
        "check",
        "role",
        "check",
    ]
    assert len({thread for unused, thread in case.events + case.scorer.events}) == 1
    assert case.events[0][1] != threading.get_ident()
    assert case.scorer.urls == ["https://stage2.test"] and not app.state.owner.is_alive
    assert case.retained[0][0] == "service-role.json"
    assert json.loads(case.retained[0][1])["artifact_hashes"] == dict(
        case.models.artifact_hashes
    )


async def reject_startup(app):
    with pytest.raises(Exception):
        async with app.router.lifespan_context(app):
            pytest.fail("mismatched owner became available")


@pytest.mark.parametrize("name", ARTIFACTS)
def test_each_loaded_artifact_mismatch_rejects_before_role_or_forward(
    runtime, records, monkeypatch, name
):
    case = owner_case(monkeypatch)
    case.models.artifact_hashes = tuple(
        (key, "f" * 64 if key == name else value)
        for key, value in case.models.artifact_hashes
    )
    app = app_for(runtime, records, case)
    asyncio.run(reject_startup(app))
    assert case.retained == [] and case.scorer.urls == []
    assert not case.active and not app.state.owner.is_alive
    assert [name for name, unused in case.scorer.events] == ["create", "enter", "exit"]


def change_threshold(case, name):
    if name == "length_only":
        case.models.length_only.validation_threshold_record["threshold"] = 0.9
    elif name == "monitor_boundary":
        case.models.monitor_boundary = 99.0
    else:
        target = {
            "logistic_l1": "stage1_threshold",
            "transformer": "transformer_threshold",
            "half_width": "half_width",
        }[name]
        setattr(case.models.cascade, target, 0.9)


@pytest.mark.parametrize("name", THRESHOLDS)
def test_each_loaded_operating_point_mismatch_rejects_before_forward(
    runtime, records, monkeypatch, name
):
    case = owner_case(monkeypatch)
    change_threshold(case, name)
    app = app_for(runtime, records, case)
    asyncio.run(reject_startup(app))
    assert case.retained == [] and case.scorer.urls == []
    assert not case.active and not app.state.owner.is_alive


def test_role_retention_failure_closes_the_same_loaded_session(
    runtime, records, monkeypatch
):
    case = owner_case(monkeypatch)

    def fail(name, content):
        raise OSError("private invented path")

    app = app_for(runtime, records, case, fail)
    asyncio.run(reject_startup(app))
    assert case.scorer.urls == [] and not case.active and not app.state.owner.is_alive
    assert sum(name == "load" for name, unused in case.events) == 1
