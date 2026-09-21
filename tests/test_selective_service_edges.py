"""Synthetic ownership, queue-boundary, and failure-cleanup edge cases."""

import asyncio
import threading
from dataclasses import replace

import pytest
from test_selective_service import (
    SyntheticScorer,
    client_for,
    factory_for,
    wait_until,
)

from automated_phishing_detection import selective_service as service
from automated_phishing_detection.http_schema import ScanRequest


def test_default_queue_accepts_one_active_and_128_waiting_then_rejects():
    async def exercise():
        holder, gate = [], threading.Event()
        app = service.create_app(factory_for(holder, gate=gate))
        async with app.router.lifespan_context(app), client_for(app) as client:
            owner = app.state.owner
            first = asyncio.create_task(
                client.post("/v1/scan", json={"request_id": "active", "url": "active"})
            )
            try:
                await wait_until(holder[0].started.is_set)
                queued = [
                    owner.admit(ScanRequest(request_id=f"q-{i}", url=f"queued-{i}"))
                    for i in range(128)
                ]
                assert owner.queued_requests == 128
                assert owner.admitted_requests == 129
                rejected = await client.post(
                    "/v1/scan", json={"request_id": "overflow", "url": "overflow"}
                )
                assert rejected.status_code == 503
                assert owner.queued_requests == 128
                assert owner.admitted_requests == 129
                assert holder[0].urls == ["active"]
                barrier = owner.drain()
                assert not barrier.done()
            finally:
                gate.set()
            assert (await first).json()["admission_sequence"] == 1
            results = await asyncio.wait_for(
                asyncio.gather(*(asyncio.wrap_future(future) for future in queued)),
                3,
            )
            assert [result.admission_sequence for result in results] == list(
                range(2, 130)
            )
            assert [result.request_id for result in results] == [
                f"q-{i}" for i in range(128)
            ]
            assert (await asyncio.wrap_future(barrier)).model_dump() == {
                "admitted_requests": 129,
                "completed_requests": 129,
                "failed_requests": 0,
                "transformer_forward_attempts": 0,
                "successful_transformer_scores": 0,
            }
            assert holder[0].urls == ["active", *(f"queued-{i}" for i in range(128))]
            assert owner.queued_requests == 0
            accepted = await client.post(
                "/v1/scan", json={"request_id": "overflow", "url": "overflow"}
            )
            assert accepted.status_code == 200
            assert accepted.json()["admission_sequence"] == 130
        assert not owner.is_alive

    asyncio.run(exercise())


class BrokenCounterScorer(SyntheticScorer):
    def __init__(self, ready, release, failure):
        self._break_counts = False
        self._counter_ready = ready
        self._counter_release = release
        self._counter_failure = failure
        super().__init__()

    @property
    def counts(self):
        if not self._break_counts:
            return self._counts
        self.events.append(("counts", threading.get_ident()))
        self._counter_ready.set()
        assert self._counter_release.wait(5), "test did not release counter accessor"
        if self._counter_failure == "accessor":
            raise RuntimeError("private counter failure /secret/path")
        field, value = self._counter_failure
        return replace(self._counts, **{field: value})

    @counts.setter
    def counts(self, value):
        self._counts = value

    def scan(self, url, *, drift_override=False):
        result = super().scan(url, drift_override=drift_override)
        self._break_counts = True
        return result


@pytest.mark.parametrize(
    "failure",
    [
        "accessor",
        ("transformer_forward_attempts", True),
        ("successful_transformer_scores", -1),
    ],
)
def test_bad_drain_counters_fail_active_and_queued_work_then_exit_owner(failure):
    async def exercise():
        holder, ready, release = [], threading.Event(), threading.Event()

        def factory():
            holder.append(BrokenCounterScorer(ready, release, failure))
            return holder[-1]

        app = service.create_app(factory)
        owner = app.state.owner
        with pytest.raises(RuntimeError, match="scoring owner failed"):
            async with app.router.lifespan_context(app), client_for(app) as client:
                assert (
                    await client.post(
                        "/v1/scan", json={"request_id": "first", "url": "stage2-first"}
                    )
                ).status_code == 200
                active_barrier = asyncio.create_task(client.post("/v1/drain"))
                try:
                    await wait_until(ready.is_set)
                    queued = owner.admit(ScanRequest(request_id="queued", url="queued"))
                    queued_barrier = owner.drain()
                    assert not active_barrier.done()
                    assert not queued.done()
                    assert not queued_barrier.done()
                finally:
                    release.set()
                response = await asyncio.wait_for(active_barrier, 3)
                assert response.status_code == 503
                assert response.json() == {"detail": "scoring service unavailable"}
                failures = await asyncio.wait_for(
                    asyncio.gather(
                        asyncio.wrap_future(queued),
                        asyncio.wrap_future(queued_barrier),
                        return_exceptions=True,
                    ),
                    3,
                )
                assert all(
                    isinstance(exc, service.OwnerUnavailable) for exc in failures
                )
                assert not owner.accepting
                assert not owner.healthy
                assert owner.queued_requests == 0
                assert (
                    await client.post(
                        "/v1/scan", json={"request_id": "late", "url": "late"}
                    )
                ).status_code == 503
        assert not owner.is_alive
        assert len(holder) == 1
        assert holder[0].urls == ["stage2-first"]
        assert [event for event, _ in holder[0].events] == [
            "create",
            "enter",
            "scan",
            "counts",
            "exit",
        ]
        assert len({ident for _, ident in holder[0].events}) == 1

    asyncio.run(exercise())


@pytest.mark.parametrize("phase", ["factory", "enter"])
def test_cancelled_blocked_startup_waits_for_owner_context_cleanup(phase):
    async def exercise():
        holder, ready, release = [], threading.Event(), threading.Event()

        def block_startup():
            ready.set()
            assert release.wait(5), "test did not release startup"

        class BlockingStartupScorer(SyntheticScorer):
            def __enter__(self):
                if phase == "enter":
                    block_startup()
                return super().__enter__()

        def factory():
            if phase == "factory":
                block_startup()
            holder.append(BlockingStartupScorer())
            return holder[-1]

        app = service.create_app(factory)
        owner = app.state.owner

        async def lifespan():
            async with app.router.lifespan_context(app):
                pytest.fail("cancelled startup must not yield an available service")

        startup = asyncio.create_task(lifespan())
        try:
            await wait_until(ready.is_set)
            startup.cancel()
            await asyncio.sleep(0.01)
            assert not startup.done()
            assert owner.is_alive
            assert not owner.accepting
            assert not owner.healthy
            async with client_for(app) as client:
                response = await client.post(
                    "/v1/scan", json={"request_id": "during-startup", "url": "raw"}
                )
                assert response.status_code == 503
            assert owner.admitted_requests == 0
        finally:
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(startup, 3)
        assert not owner.is_alive
        assert not owner.accepting
        assert not owner.healthy
        assert len(holder) == 1
        assert holder[0].urls == []
        assert [event for event, _ in holder[0].events] == ["create", "enter", "exit"]
        assert len({ident for _, ident in holder[0].events}) == 1
        assert holder[0].events[0][1] != threading.get_ident()

    asyncio.run(exercise())
