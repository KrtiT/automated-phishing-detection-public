import asyncio
import threading
from dataclasses import replace
from importlib import import_module

import httpx
import pytest

from automated_phishing_detection.selective_inference import (
    InferenceCounts,
    RequestScores,
)


class SyntheticScorer:
    def __init__(self, *, gate=None, outcomes=None, enter_error=None, exit_error=None):
        self.events = [("create", threading.get_ident())]
        self.gate = gate
        self.outcomes = outcomes or {}
        self.enter_error = enter_error
        self.exit_error = exit_error
        self.started = threading.Event()
        self.urls = []
        self.counts = InferenceCounts(0, 0, 0, 0)

    def __enter__(self):
        self.events.append(("enter", threading.get_ident()))
        if self.enter_error:
            raise self.enter_error
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.events.append(("exit", threading.get_ident()))
        if self.exit_error:
            raise self.exit_error

    def scan(self, url, *, drift_override=False):
        assert drift_override is False
        self.events.append(("scan", threading.get_ident()))
        self.urls.append(url)
        self.started.set()
        if self.gate:
            assert self.gate.wait(5), "test failed to release synthetic owner"
        selected = "stage2" in url
        self.counts = replace(
            self.counts,
            transformer_forward_attempts=self.counts.transformer_forward_attempts
            + int(selected),
        )
        result = self.outcomes.get(url)
        if isinstance(result, BaseException):
            self.counts = replace(
                self.counts, failed_requests=self.counts.failed_requests + 1
            )
            raise result
        self.counts = replace(
            self.counts,
            completed_requests=self.counts.completed_requests + 1,
            successful_transformer_scores=self.counts.successful_transformer_scores
            + int(selected),
        )
        return result or RequestScores(
            0.2,
            0.9 if selected else None,
            int(selected),
            int(selected),
            selected,
            False,
            selected,
            selected,
            {"secret": "must not leak"},
        )


@pytest.fixture
def service():
    return import_module("automated_phishing_detection.selective_service")


def factory_for(holder, **kwargs):
    def factory():
        holder.append(SyntheticScorer(**kwargs))
        return holder[-1]

    return factory


def client_for(app):
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    )


async def wait_until(predicate):
    async def poll():
        while not predicate():
            await asyncio.sleep(0.001)

    await asyncio.wait_for(poll(), 3)


def test_factory_context_scan_and_exit_share_one_non_daemon_owner(service):
    async def exercise():
        holder = []
        app = service.create_app(factory_for(holder))
        async with app.router.lifespan_context(app), client_for(app) as client:
            for sequence, url in enumerate(
                ("HTTPS://Example.test:443/raw", "https://stage2.test"), 1
            ):
                response = await client.post(
                    "/v1/scan", json={"request_id": f"r-{sequence}", "url": url}
                )
                assert response.status_code == 200
                assert response.json() == {
                    "request_id": f"r-{sequence}",
                    "admission_sequence": sequence,
                    "action": "alert" if sequence == 2 else "allow",
                    "probability": 0.9 if sequence == 2 else 0.2,
                    "stage2_invoked": sequence == 2,
                }
            assert holder[0].urls == [
                "HTTPS://Example.test:443/raw",
                "https://stage2.test",
            ]
            owner_thread = next(
                thread
                for thread in threading.enumerate()
                if thread.ident == holder[0].events[0][1]
            )
            assert owner_thread.daemon is False
        assert [name for name, _ in holder[0].events] == [
            "create",
            "enter",
            "scan",
            "scan",
            "exit",
        ]
        assert len({ident for _, ident in holder[0].events}) == 1
        assert holder[0].events[0][1] != threading.get_ident()
        assert not app.state.owner.is_alive

    asyncio.run(exercise())


def test_queue_capacity_excludes_active_and_rejection_does_not_take_sequence(service):
    async def exercise():
        holder, gate = [], threading.Event()
        app = service.create_app(factory_for(holder, gate=gate), queue_capacity=1)
        async with app.router.lifespan_context(app), client_for(app) as client:
            first = asyncio.create_task(
                client.post("/v1/scan", json={"request_id": "a", "url": "first"})
            )
            await wait_until(holder[0].started.is_set)
            second = asyncio.create_task(
                client.post("/v1/scan", json={"request_id": "b", "url": "second"})
            )
            await wait_until(lambda: app.state.owner.admitted_requests == 2)
            try:
                rejected = await client.post(
                    "/v1/scan", json={"request_id": "c", "url": "third"}
                )
                assert rejected.status_code == 503
                assert app.state.owner.admitted_requests == 2
            finally:
                gate.set()
            assert [
                response.json()["admission_sequence"]
                for response in await asyncio.gather(first, second)
            ] == [1, 2]
            response = await client.post(
                "/v1/scan", json={"request_id": "c", "url": "third"}
            )
            assert response.json()["admission_sequence"] == 3
        assert holder[0].urls == ["first", "second", "third"]

    asyncio.run(exercise())


def test_validation_precedes_admission_and_unstarted_service_is_unavailable(service):
    async def exercise():
        holder = []
        app = service.create_app(factory_for(holder))
        async with client_for(app) as client:
            assert (
                await client.post("/v1/scan", json={"request_id": "r", "url": "raw"})
            ).status_code == 503
        async with app.router.lifespan_context(app), client_for(app) as client:
            for body in (
                {"request_id": "bad id", "url": "raw"},
                {"request_id": "r", "url": ""},
                {"request_id": "r", "url": "raw", "extra": 1},
            ):
                assert (await client.post("/v1/scan", json=body)).status_code == 422
            assert app.state.owner.admitted_requests == 0
            assert holder[0].urls == []
            assert (await client.get("/docs")).status_code == 404

    asyncio.run(exercise())


def test_ordinary_failure_is_sanitized_and_keeps_actual_attempt_counter(service):
    async def exercise():
        holder = []
        app = service.create_app(
            factory_for(
                holder,
                outcomes={"stage2-fail": RuntimeError("private model /secret/path")},
            )
        )
        async with app.router.lifespan_context(app), client_for(app) as client:
            failed = await client.post(
                "/v1/scan", json={"request_id": "a", "url": "stage2-fail"}
            )
            assert failed.status_code == 500
            assert "secret" not in failed.text and "private" not in failed.text
            assert (
                await client.post(
                    "/v1/scan", json={"request_id": "b", "url": "stage2-ok"}
                )
            ).status_code == 200
            drain = await client.post("/v1/drain")
            assert drain.status_code == 200
            assert drain.json() == {
                "admitted_requests": 2,
                "completed_requests": 1,
                "failed_requests": 1,
                "transformer_forward_attempts": 2,
                "successful_transformer_scores": 1,
            }
            assert app.state.owner.healthy

    asyncio.run(exercise())


@pytest.mark.parametrize("cancel_index", [0, 1])
def test_cancellation_never_drops_admitted_work_and_drain_bypasses_full_queue(
    service, cancel_index
):
    async def exercise():
        holder, gate = [], threading.Event()
        app = service.create_app(factory_for(holder, gate=gate), queue_capacity=1)
        async with app.router.lifespan_context(app), client_for(app) as client:
            first = asyncio.create_task(
                client.post("/v1/scan", json={"request_id": "a", "url": "stage2-first"})
            )
            await wait_until(holder[0].started.is_set)
            second = asyncio.create_task(
                client.post(
                    "/v1/scan", json={"request_id": "b", "url": "stage2-second"}
                )
            )
            await wait_until(lambda: app.state.owner.admitted_requests == 2)
            requests = [first, second]
            requests[cancel_index].cancel()
            with pytest.raises(asyncio.CancelledError):
                await requests[cancel_index]
            drain = asyncio.create_task(client.post("/v1/drain"))
            await asyncio.sleep(0.01)
            assert not drain.done()
            gate.set()
            assert (await requests[1 - cancel_index]).status_code == 200
            assert (await drain).json() == {
                "admitted_requests": 2,
                "completed_requests": 2,
                "failed_requests": 0,
                "transformer_forward_attempts": 2,
                "successful_transformer_scores": 2,
            }
            assert holder[0].urls == ["stage2-first", "stage2-second"]

    asyncio.run(exercise())


def test_drain_snapshot_excludes_requests_admitted_after_barrier(service):
    async def exercise():
        holder, gate = [], threading.Event()
        app = service.create_app(factory_for(holder, gate=gate))
        async with app.router.lifespan_context(app):
            owner = app.state.owner
            request = import_module(
                "automated_phishing_detection.http_schema"
            ).ScanRequest
            first = owner.admit(request(request_id="a", url="stage2-first"))
            await wait_until(holder[0].started.is_set)
            snapshot = owner.drain()
            second = owner.admit(request(request_id="b", url="stage2-second"))
            gate.set()
            result = await asyncio.wrap_future(snapshot)
            assert result.model_dump() == {
                "admitted_requests": 1,
                "completed_requests": 1,
                "failed_requests": 0,
                "transformer_forward_attempts": 1,
                "successful_transformer_scores": 1,
            }
            await asyncio.gather(
                asyncio.wrap_future(first), asyncio.wrap_future(second)
            )
            assert (await asyncio.wrap_future(owner.drain())).admitted_requests == 2

    asyncio.run(exercise())


@pytest.mark.parametrize("phase", ["factory", "enter", "exit"])
def test_startup_and_cleanup_failure_mark_owner_unhealthy_and_join(service, phase):
    async def exercise():
        holder = []

        def factory():
            if phase == "factory":
                raise RuntimeError("private startup")
            holder.append(
                SyntheticScorer(**{f"{phase}_error": RuntimeError("private context")})
            )
            return holder[-1]

        app = service.create_app(factory)
        with pytest.raises(RuntimeError):
            async with app.router.lifespan_context(app):
                assert phase == "exit"
        assert not app.state.owner.is_alive
        assert not app.state.owner.healthy
        assert not app.state.owner.accepting
        async with client_for(app) as client:
            response = await client.post(
                "/v1/scan", json={"request_id": "a", "url": "raw"}
            )
            assert response.status_code == 503
            assert "private" not in response.text

    asyncio.run(exercise())


def test_baseexception_fails_active_queued_and_barrier_without_restart(service):
    async def exercise():
        holder, gate = [], threading.Event()
        app = service.create_app(
            factory_for(
                holder, gate=gate, outcomes={"fatal": SystemExit("private fatal")}
            )
        )
        with pytest.raises(RuntimeError):
            async with app.router.lifespan_context(app), client_for(app) as client:
                first = asyncio.create_task(
                    client.post("/v1/scan", json={"request_id": "a", "url": "fatal"})
                )
                await wait_until(holder[0].started.is_set)
                second = asyncio.create_task(
                    client.post("/v1/scan", json={"request_id": "b", "url": "queued"})
                )
                await wait_until(lambda: app.state.owner.admitted_requests == 2)
                drain = asyncio.create_task(client.post("/v1/drain"))
                await asyncio.sleep(0.01)
                gate.set()
                responses = await asyncio.wait_for(
                    asyncio.gather(first, second, drain), 3
                )
                assert [response.status_code for response in responses] == [
                    503,
                    503,
                    503,
                ]
                assert all("private" not in response.text for response in responses)
                assert not app.state.owner.accepting
                assert not app.state.owner.healthy
        assert len(holder) == 1
        assert holder[0].urls == ["fatal"]
        assert holder[0].events[-1][0] == "exit"
        assert not app.state.owner.is_alive

    asyncio.run(exercise())


def test_shutdown_closes_admission_then_drains_fifo_before_context_exit(service):
    async def exercise():
        holder, gate = [], threading.Event()
        app = service.create_app(factory_for(holder, gate=gate))
        async with app.router.lifespan_context(app), client_for(app) as client:
            first = asyncio.create_task(
                client.post("/v1/scan", json={"request_id": "a", "url": "first"})
            )
            await wait_until(holder[0].started.is_set)
            second = asyncio.create_task(
                client.post("/v1/scan", json={"request_id": "b", "url": "second"})
            )
            await wait_until(lambda: app.state.owner.admitted_requests == 2)
            shutdown = asyncio.create_task(app.state.owner.shutdown())
            await wait_until(lambda: not app.state.owner.accepting)
            try:
                assert (
                    await client.post(
                        "/v1/scan", json={"request_id": "c", "url": "late"}
                    )
                ).status_code == 503
                assert (await client.post("/v1/drain")).status_code == 503
                assert not shutdown.done()
                assert holder[0].events[-1][0] == "scan"
            finally:
                gate.set()
            assert [
                response.status_code for response in await asyncio.gather(first, second)
            ] == [200, 200]
            await shutdown
        assert holder[0].urls == ["first", "second"]
        assert holder[0].events[-1][0] == "exit"

    asyncio.run(exercise())


def test_shutdown_timeout_fails_every_pending_future_without_claiming_thread_exit(
    service,
):
    async def exercise():
        holder, gate = [], threading.Event()
        app = service.create_app(factory_for(holder, gate=gate))
        owner = app.state.owner
        await owner.start()
        request = import_module("automated_phishing_detection.http_schema").ScanRequest
        first = owner.admit(request(request_id="a", url="active"))
        await wait_until(holder[0].started.is_set)
        second = owner.admit(request(request_id="b", url="queued"))
        drain = owner.drain()
        try:
            with pytest.raises(RuntimeError, match="still alive"):
                await owner.shutdown(timeout=0.01)
            assert owner.is_alive
            assert not owner.healthy
            assert not owner.accepting
            for future in (first, second, drain):
                assert future.done(), (
                    "shutdown timeout must fail active and queued futures"
                )
                assert isinstance(future.exception(), service.OwnerUnavailable)
        finally:
            gate.set()
            with pytest.raises(RuntimeError):
                await owner.shutdown(timeout=3)
        assert not owner.is_alive
        assert holder[0].urls == ["active"]
        assert holder[0].events[-1][0] == "exit"

    asyncio.run(exercise())


@pytest.mark.parametrize(
    "changes",
    [
        {"decision": True},
        {"decision": 2},
        {"fixed_decision": 1},
        {"stage1_probability": float("nan")},
        {"stage1_probability": True},
        {
            "stage1_probability": None,
            "transformer_probability": 0.8,
            "band_selected": True,
            "logical_stage2_selected": True,
            "transformer_evaluated": True,
        },
        {"transformer_probability": 0.8},
        {"transformer_evaluated": 1},
        {"drift_override": True},
        {"logical_stage2_selected": True},
        {"stage1_probability": 1.01},
        {"stage1_probability": "0.2"},
    ],
)
def test_inconsistent_scorer_result_is_sanitized_failure(service, changes):
    async def exercise():
        normal = RequestScores(0.2, None, 0, 0, False, False, False, False, {})
        holder = []
        app = service.create_app(
            factory_for(holder, outcomes={"bad": replace(normal, **changes)})
        )
        async with app.router.lifespan_context(app), client_for(app) as client:
            response = await client.post(
                "/v1/scan", json={"request_id": "r", "url": "bad"}
            )
            assert response.status_code == 500
            assert response.json() == {"detail": "request scoring failed"}
            assert (await client.post("/v1/drain")).json()["failed_requests"] == 1

    asyncio.run(exercise())


def test_drain_seals_phase_ids_against_delayed_admission(service):
    async def exercise():
        holder = []
        app = service.create_app(factory_for(holder))
        async with app.router.lifespan_context(app), client_for(app) as client:
            started, release = asyncio.Event(), asyncio.Event()

            async def delayed_body():
                started.set()
                await release.wait()
                yield b'{"request_id":"warmup-late","url":"late-body"}'

            delayed = asyncio.create_task(
                client.post(
                    "/v1/scan",
                    content=delayed_body(),
                    headers={"content-type": "application/json"},
                )
            )
            await started.wait()
            assert (
                await client.post(
                    "/v1/scan",
                    json={"request_id": "warmup-done", "url": "stage2-warmup"},
                )
            ).status_code == 200
            sealed = await client.post(
                "/v1/drain", json={"request_ids": ["warmup-done", "warmup-late"]}
            )
            assert sealed.status_code == 200
            assert sealed.json()["admitted_requests"] == 1
            release.set()
            assert (await delayed).status_code == 503
            measured = await client.post(
                "/v1/scan", json={"request_id": "measured-1", "url": "stage2-measured"}
            )
            assert measured.status_code == 200
            assert measured.json()["admission_sequence"] == 2
            assert (await client.post("/v1/drain")).json()["admitted_requests"] == 2
            assert holder[0].urls == ["stage2-warmup", "stage2-measured"]

    asyncio.run(exercise())


def test_duplicate_admitted_ids_rejected_without_scoring_or_sequence_gap(service):
    async def exercise():
        holder, gate = [], threading.Event()
        app = service.create_app(factory_for(holder, gate=gate))
        async with app.router.lifespan_context(app), client_for(app) as client:
            first = asyncio.create_task(
                client.post("/v1/scan", json={"request_id": "same", "url": "first"})
            )
            await wait_until(holder[0].started.is_set)
            try:
                assert (
                    await client.post(
                        "/v1/scan", json={"request_id": "same", "url": "duplicate"}
                    )
                ).status_code == 503
            finally:
                gate.set()
            assert (await first).status_code == 200
            assert (
                await client.post(
                    "/v1/scan", json={"request_id": "same", "url": "duplicate"}
                )
            ).status_code == 503
            assert (
                await client.post(
                    "/v1/scan", json={"request_id": "next", "url": "second"}
                )
            ).json()["admission_sequence"] == 2
            assert holder[0].urls == ["first", "second"]

    asyncio.run(exercise())


def test_invalid_drain_seal_does_not_close_ids(service):
    async def exercise():
        app = service.create_app(factory_for([]))
        async with app.router.lifespan_context(app), client_for(app) as client:
            response = await client.post(
                "/v1/drain", json={"request_ids": ["valid", "valid"]}
            )
            assert response.status_code == 422
            assert (
                await client.post(
                    "/v1/scan", json={"request_id": "valid", "url": "raw"}
                )
            ).status_code == 200

    asyncio.run(exercise())


def test_cancelling_shutdown_still_waits_for_owner_cleanup(service):
    async def exercise():
        holder, gate = [], threading.Event()
        app = service.create_app(factory_for(holder, gate=gate))
        owner = app.state.owner
        await owner.start()
        request = import_module("automated_phishing_detection.http_schema").ScanRequest
        request_future = owner.admit(request(request_id="r", url="active"))
        await wait_until(holder[0].started.is_set)
        shutdown = asyncio.create_task(owner.shutdown(timeout=3))
        await wait_until(lambda: not owner.accepting)
        shutdown.cancel()
        try:
            await asyncio.sleep(0.01)
            assert not shutdown.done(), (
                "shutdown cannot return before cleanup on cancellation"
            )
        finally:
            gate.set()
            try:
                await shutdown
            except asyncio.CancelledError:
                pass
            await owner.shutdown(timeout=3)
        assert not owner.is_alive
        assert holder[0].events[-1][0] == "exit"
        assert request_future.result().admission_sequence == 1

    asyncio.run(exercise())


def test_real_selective_cascade_cpu_fixture_preserves_counts_and_restores_context(
    service, tmp_path, monkeypatch
):
    import torch
    from test_transformer_inference import _build_fixture, _load_fixture

    from automated_phishing_detection import fixed_cascade
    from automated_phishing_detection.selective_inference import SelectiveCascade

    original_mode = (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
    )
    owner_threads, forwards = [], []

    def stage1(model, urls):
        return (0.5 if "stage2" in urls[0] else 0.2,), {"synthetic": True}

    monkeypatch.setattr(fixed_cascade, "score_logistic_l1_authoritative", stage1)

    def factory():
        owner_threads.append(threading.get_ident())
        loaded = replace(
            _load_fixture(_build_fixture(tmp_path)),
            stage1_threshold=0.5,
            transformer_threshold=0.5,
            half_width=0.125,
        )
        original_forward = loaded._model.forward

        def forward(*args, **kwargs):
            forwards.append(threading.get_ident())
            assert torch.get_num_threads() == 1
            assert torch.is_inference_mode_enabled()
            if len(forwards) == 2:
                raise RuntimeError("synthetic forward failure")
            return original_forward(*args, **kwargs)

        monkeypatch.setattr(loaded._model, "forward", forward)
        return SelectiveCascade(loaded, _fixture_cpu=True)

    async def exercise():
        app = service.create_app(factory)
        async with app.router.lifespan_context(app), client_for(app) as client:
            skipped = await client.post(
                "/v1/scan",
                json={"request_id": "outside", "url": "https://safe.example"},
            )
            assert skipped.status_code == 200
            assert skipped.json()["stage2_invoked"] is False
            selected = await client.post(
                "/v1/scan",
                json={"request_id": "inside", "url": "https://stage2.example"},
            )
            assert selected.status_code == 200
            assert selected.json()["stage2_invoked"] is True
            failed = await client.post(
                "/v1/scan",
                json={"request_id": "failed", "url": "https://stage2-failed.example"},
            )
            assert failed.status_code == 500
            assert (await client.post("/v1/drain")).json() == {
                "admitted_requests": 3,
                "completed_requests": 2,
                "failed_requests": 1,
                "transformer_forward_attempts": 2,
                "successful_transformer_scores": 1,
            }
        assert not app.state.owner.is_alive

    asyncio.run(exercise())
    assert forwards == owner_threads * 2
    assert (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
    ) == original_mode


@pytest.mark.parametrize("capacity", [0, -1, True, 1.0, "1"])
def test_queue_capacity_requires_positive_strict_integer(service, capacity):
    with pytest.raises(ValueError, match="queue_capacity"):
        service.create_app(factory_for([]), queue_capacity=capacity)
