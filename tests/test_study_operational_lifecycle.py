"""Supplement checks surround replay and service publication, never measurements."""

import asyncio
import json
import os
from functools import partial
from types import SimpleNamespace

import pytest
from operational_bound_child_fixtures import child_case, keywords
from operational_child_fixtures import child_context as child_context
from operational_service_child_fixtures import inherited, synthetic_primary
from operational_transport_integration_fixtures import accepted as accepted
from operational_transport_integration_fixtures import candidates as candidates
from operational_transport_integration_fixtures import case as case
from operational_transport_integration_fixtures import manifests as manifests
from study_execution_fixtures import bind
from study_execution_fixtures import execution_case as execution_case
from study_lifecycle_fixtures import admitted, invalidate

from automated_phishing_detection import _study_child_operational as child
from automated_phishing_detection import operational_cell_service as service
from automated_phishing_detection._study_child_context import recheck_held_child
from automated_phishing_detection.bound_models import ArtifactPaths
from automated_phishing_detection.operational_service import OperationalServiceError


@pytest.mark.parametrize("boundary", ["envelope", "channel"])
def test_late_client_invalidation_retains_measurements_but_not_run(
    execution_case, monkeypatch, boundary
):
    retained = {}
    runtime = SimpleNamespace(
        retain=lambda name, content: retained.setdefault(name, content)
    )
    with admitted(bind(execution_case)) as (held, state):

        async def replay(actual):
            assert actual is runtime
            runtime.retain("warmup.json", b"invented warmup")
            runtime.retain("measured.json", b"invented measured")
            invalidate(held, state, boundary)
            return b"invented complete run"

        monkeypatch.setattr(child.client, "_replay", replay)
        with pytest.raises(ValueError):
            asyncio.run(child._client(runtime, partial(recheck_held_child, held)))
    assert retained == {
        "warmup.json": b"invented warmup",
        "measured.json": b"invented measured",
    }


@pytest.mark.parametrize(
    "kind", [KeyboardInterrupt, SystemExit, asyncio.CancelledError]
)
def test_client_first_replay_interruption_keeps_progress_and_identity(
    monkeypatch, kind
):
    retained, checks = {}, []
    original = kind("invented interruption")
    original.progress = b"invented partial progress"
    runtime = SimpleNamespace(
        retain=lambda name, content: retained.setdefault(name, content)
    )

    async def replay(actual):
        raise original

    monkeypatch.setattr(child.client, "_replay", replay)

    async def observe():
        with pytest.raises(kind) as caught:
            await child._client(runtime, lambda: checks.append("checked"))
        return caught.value

    assert asyncio.run(observe()) is original
    assert retained == {"client-failure.json": original.progress}
    assert checks == ["checked"]


async def serve_invalidated(runtime, fixture, artifacts, controls, check, stage):
    task = asyncio.create_task(
        service._serve(runtime, fixture.binding, artifacts, lifecycle_check=check)
    )
    try:
        if stage == "cleanup":
            assert (
                await asyncio.wait_for(asyncio.to_thread(os.read, controls[2], 6), 5)
                == b"ready\n"
            )
            os.write(controls[1], b"stop\n")
        with pytest.raises(OperationalServiceError):
            await asyncio.wait_for(task, 5)
    finally:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


def invalidation_check(held, state, stage, boundary):
    checks = []

    def check():
        checks.append(True)
        if len(checks) == (2 if stage == "ready" else 3):
            invalidate(held, state, boundary)
        recheck_held_child(held)

    return check


@pytest.mark.parametrize("stage", ["ready", "cleanup"])
@pytest.mark.parametrize("boundary", ["envelope", "channel"])
def test_service_invalidated_boundary_never_retains_success_checkpoint(
    execution_case,
    child_context,
    accepted,
    case,
    tmp_path,
    monkeypatch,
    stage,
    boundary,
):
    fixture = child_case(tmp_path, accepted, case, monkeypatch, 1)
    monkeypatch.setattr(child_context, "recheck_binding", lambda binding: None)
    synthetic_primary(monkeypatch, json.loads(accepted.metadata_bytes)["primary"])
    artifacts = ArtifactPaths(*(tmp_path / str(index) for index in range(4)))
    with admitted(bind(execution_case)) as (held, state):
        check = invalidation_check(held, state, stage, boundary)
        with inherited(monkeypatch) as controls:
            with child_context.held_child(
                fixture.binding, fixture.profile, "service", **keywords(fixture)
            ) as runtime:
                asyncio.run(
                    serve_invalidated(
                        runtime, fixture, artifacts, controls, check, stage
                    )
                )
    assert not (fixture.attempt.directory / f"service-{stage}.json").exists()
    assert not (fixture.attempt.directory / "service-cleanup.json").exists()
