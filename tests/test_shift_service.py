"""Synthetic owner-thread checks for the serialized live-monitor workload."""

import asyncio
import threading
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_selective_service import client_for

from automated_phishing_detection.http_replay import ReplayRequest
from automated_phishing_detection.selective_inference import (
    InferenceCounts,
    RequestScores,
)


class SyntheticMonitor:
    def __init__(self, *, gate=None, fail_at=None):
        self.counts = InferenceCounts(0, 0, 0, 0)
        self.position = 0
        self.gate = gate
        self.fail_at = fail_at
        self.events = []
        self.started = threading.Event()

    def __enter__(self):
        self.events.append(("enter", threading.get_ident()))
        return self

    def __exit__(self, *args):
        self.events.append(("exit", threading.get_ident()))

    def reset_monitor(self):
        self.position = 0
        self.events.append(("reset", threading.get_ident()))

    def scan_shift(self, raw_url):
        self.events.append(("scan", threading.get_ident()))
        self.started.set()
        if self.gate:
            assert self.gate.wait(5)
        if self.position + 1 == self.fail_at:
            self.counts = replace(
                self.counts, failed_requests=self.counts.failed_requests + 1
            )
            raise ValueError("synthetic monitor failure")
        self.position += 1
        self.counts = replace(
            self.counts, completed_requests=self.counts.completed_requests + 1
        )
        scores = RequestScores(0.2, None, 0, 0, False, False, False, False, {})
        return SimpleNamespace(
            scores=scores, monitor_nll=1.0, position=self.position, window=None
        )


@pytest.fixture
def modules():
    root = Path(__file__).resolve().parents[1]
    assert (root / "src/automated_phishing_detection/shift_service.py").is_file(), (
        "missing shift service"
    )
    from automated_phishing_detection import shift_schema, shift_service

    return shift_schema, shift_service


def make_plan(schema):
    return schema.ShiftPlan(
        "a" * 64,
        1,
        tuple(ReplayRequest(f"row-{i}", f"https://example.test/{i}") for i in range(3)),
        warmup_count=1,
    )


def test_warmup_reset_and_exact_measured_stream(modules):
    schema, service = modules
    scorer = SyntheticMonitor()
    plan = make_plan(schema)

    async def exercise():
        app = service.create_shift_app(lambda: scorer, plan)
        async with app.router.lifespan_context(app), client_for(app) as client:
            initial = (await client.post("/v1/shift/state")).json()
            assert initial["phase"] == "warmup" and initial["rows"] == []
            row = plan.requests[0]
            warmup_id = plan.request_id("warmup", 0)
            assert (
                await client.post(
                    "/v1/scan", json={"request_id": warmup_id, "url": row.raw_url}
                )
            ).status_code == 200
            reset = await client.post(
                "/v1/shift/reset", json={"request_ids": [warmup_id]}
            )
            assert reset.status_code == 200
            state = reset.json()
            assert state["phase"] == "measured" and state["rows"] == []
            assert state["counts"]["completed_requests"] == 1
            for i, row in enumerate(plan.requests):
                response = await client.post(
                    "/v1/scan",
                    json={
                        "request_id": plan.request_id("measured", i),
                        "url": row.raw_url,
                    },
                )
                assert response.status_code == 200
                assert response.json()["admission_sequence"] == i + 2
            state = (await client.post("/v1/shift/state")).json()
            assert state["complete"] is True
            assert [row["position"] for row in state["rows"]] == [1, 2, 3]
            assert state["counts"]["completed_requests"] == 4
            assert (
                await client.post("/v1/shift/reset", json={"request_ids": [warmup_id]})
            ).status_code == 409

    asyncio.run(exercise())
    assert len({thread for _, thread in scorer.events}) == 1
    assert scorer.events[0][1] != threading.get_ident()


@pytest.mark.parametrize("wrong", ["order", "url", "phase"])
def test_invalid_stream_admission_is_irrecoverable(modules, wrong):
    schema, service = modules
    scorer = SyntheticMonitor()
    plan = make_plan(schema)

    async def exercise():
        app = service.create_shift_app(lambda: scorer, plan)
        async with app.router.lifespan_context(app), client_for(app) as client:
            identity = (
                plan.request_id("warmup", 1)
                if wrong == "order"
                else plan.request_id("measured" if wrong == "phase" else "warmup", 0)
            )
            url = (
                "https://different.test" if wrong == "url" else plan.requests[0].raw_url
            )
            assert (
                await client.post("/v1/scan", json={"request_id": identity, "url": url})
            ).status_code == 503
            assert (
                await client.post(
                    "/v1/scan",
                    json={
                        "request_id": plan.request_id("warmup", 0),
                        "url": plan.requests[0].raw_url,
                    },
                )
            ).status_code == 503
            state = (await client.post("/v1/shift/state")).json()
            assert state["broken"] is True and state["counts"]["admitted_requests"] == 0

    asyncio.run(exercise())


def test_monitor_failure_blocks_later_rows_but_keeps_drain_available(modules):
    schema, service = modules
    plan = make_plan(schema)
    scorer = SyntheticMonitor(fail_at=1)

    async def exercise():
        app = service.create_shift_app(lambda: scorer, plan)
        async with app.router.lifespan_context(app), client_for(app) as client:
            response = await client.post(
                "/v1/scan",
                json={
                    "request_id": plan.request_id("warmup", 0),
                    "url": plan.requests[0].raw_url,
                },
            )
            assert response.status_code == 500
            state = (await client.post("/v1/shift/state")).json()
            assert state["broken"] is True
            assert state["counts"]["failed_requests"] == 1
            assert (await client.post("/v1/drain", json={})).status_code == 200
            assert (
                await client.post(
                    "/v1/shift/reset",
                    json={"request_ids": [plan.request_id("warmup", 0)]},
                )
            ).status_code == 409

    asyncio.run(exercise())


def test_second_unresolved_request_is_not_queued(modules):
    schema, service = modules
    plan = make_plan(schema)
    gate = threading.Event()
    scorer = SyntheticMonitor(gate=gate)

    async def exercise():
        app = service.create_shift_app(lambda: scorer, plan)
        async with app.router.lifespan_context(app), client_for(app) as client:
            first = asyncio.create_task(
                client.post(
                    "/v1/scan",
                    json={
                        "request_id": plan.request_id("warmup", 0),
                        "url": plan.requests[0].raw_url,
                    },
                )
            )
            try:
                assert await asyncio.to_thread(scorer.started.wait, 3)
                second = await client.post(
                    "/v1/scan",
                    json={
                        "request_id": plan.request_id("measured", 0),
                        "url": plan.requests[0].raw_url,
                    },
                )
                assert second.status_code == 503
                assert app.state.owner.admitted_requests == 1
            finally:
                gate.set()
                await first

    asyncio.run(exercise())
