"""Service and replay integration over actual loopback TCP connections."""

import asyncio
import threading
from dataclasses import replace

import httpx
import pytest
import torch
from test_http_replay import loopback_server
from test_selective_service import factory_for, wait_until
from test_transformer_inference import _build_fixture, _load_fixture

from automated_phishing_detection import fixed_cascade
from automated_phishing_detection.http_replay import ReplayRequest, replay_run
from automated_phishing_detection.selective_inference import SelectiveCascade
from automated_phishing_detection.selective_service import create_app

SHA = "a" * 64


async def replay(url, records, *, concurrency=1):
    return await replay_run(
        url,
        records,
        manifest_sha256=SHA,
        prevalence_basis_points=100,
        concurrency=concurrency,
        run_index=1,
        warmup_count=1,
    )


def test_socket_replay_keeps_scorer_lifecycle_on_single_owner_and_routes_normally():
    holder = []
    app = create_app(factory_for(holder))
    records = (
        ReplayRequest("outside", "HTTPS://Example.test:443/a%2Fb?x=raw"),
        ReplayRequest("inside", "https://stage2.test/a"),
        ReplayRequest("outside-again", "https://other.test/b"),
    )
    with loopback_server(app) as url:
        result = asyncio.run(replay(url, records, concurrency=8))
        assert app.state.owner.healthy
    scorer = holder[0]
    assert len(holder) == 1
    assert [event for event, _ in scorer.events] == [
        "create",
        "enter",
        "scan",
        "scan",
        "scan",
        "scan",
        "exit",
    ]
    owner_ids = {ident for _, ident in scorer.events}
    assert len(owner_ids) == 1
    assert threading.get_ident() not in owner_ids
    assert scorer.urls[0] == records[0].raw_url
    assert sorted(scorer.urls[1:]) == sorted(row.raw_url for row in records)
    assert not app.state.owner.is_alive
    assert [row.error for row in result.measured] == [None, None, None]
    assert [row.response.stage2_invoked for row in result.measured] == [
        False,
        True,
        False,
    ]
    assert [row.response.action for row in result.measured] == [
        "allow",
        "alert",
        "allow",
    ]
    assert [row.response.probability for row in result.measured] == [0.2, 0.9, 0.2]
    assert result.initial.admitted_requests == 0
    assert result.after_warmup.admitted_requests == 1
    assert result.after_warmup.transformer_forward_attempts == 0
    assert result.after_measured.model_dump() == {
        "admitted_requests": 4,
        "completed_requests": 4,
        "failed_requests": 0,
        "transformer_forward_attempts": 1,
        "successful_transformer_scores": 1,
    }


def test_socket_queue_overload_returns_503_without_losing_accepted_work():
    holder, gate = [], threading.Event()
    app = create_app(factory_for(holder, gate=gate), queue_capacity=1)

    async def exercise(url):
        async with httpx.AsyncClient(base_url=url, trust_env=False) as client:
            first = asyncio.create_task(
                client.post(
                    "/v1/scan", json={"request_id": "first", "url": "stage2-first"}
                )
            )
            await wait_until(holder[0].started.is_set)
            second = asyncio.create_task(
                client.post(
                    "/v1/scan", json={"request_id": "second", "url": "stage2-second"}
                )
            )
            await wait_until(lambda: app.state.owner.admitted_requests == 2)
            try:
                assert app.state.owner.queued_requests == 1
                overloaded = await client.post(
                    "/v1/scan", json={"request_id": "third", "url": "not-admitted"}
                )
                assert overloaded.status_code == 503
                assert overloaded.json() == {"detail": "scoring service unavailable"}
                drain = asyncio.create_task(client.post("/v1/drain"))
                await asyncio.sleep(0.02)
                assert not drain.done()
                assert app.state.owner.admitted_requests == 2
            finally:
                gate.set()
            accepted = await asyncio.gather(first, second)
            assert [row.status_code for row in accepted] == [200, 200]
            assert [row.json()["admission_sequence"] for row in accepted] == [1, 2]
            assert (await drain).json() == {
                "admitted_requests": 2,
                "completed_requests": 2,
                "failed_requests": 0,
                "transformer_forward_attempts": 2,
                "successful_transformer_scores": 2,
            }

    with loopback_server(app) as url:
        asyncio.run(exercise(url))
    assert holder[0].urls == ["stage2-first", "stage2-second"]
    assert not app.state.owner.is_alive


@pytest.mark.parametrize("failed_forward", [False, True])
def test_socket_real_cascade_timeout_finishes_before_phase_barrier(
    tmp_path, monkeypatch, failed_forward
):
    gate, forward_started = threading.Event(), threading.Event()
    events, forwards = [], []
    original_mode = (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
    )
    records = (
        ReplayRequest("inside", "HTTPS://stage2.example:443/a%2Fb"),
        ReplayRequest("outside", "https://safe.example/path"),
    )

    def stage1(model, urls):
        return (0.5 if "stage2" in urls[0] else 0.2,), {"synthetic": True}

    monkeypatch.setattr(fixed_cascade, "score_logistic_l1_authoritative", stage1)

    class ObservedCascade(SelectiveCascade):
        def __enter__(self):
            events.append(("enter", threading.get_ident()))
            return super().__enter__()

        def scan(self, raw_url, *, drift_override=False):
            events.append(("scan", threading.get_ident()))
            return super().scan(raw_url, drift_override=drift_override)

        def __exit__(self, *args):
            result = super().__exit__(*args)
            events.append(("exit", threading.get_ident()))
            return result

    def factory():
        events.append(("create", threading.get_ident()))
        loaded = replace(
            _load_fixture(_build_fixture(tmp_path)),
            stage1_threshold=0.5,
            transformer_threshold=0.5,
            half_width=0.125,
        )
        original_forward = loaded._model.forward

        def forward(*args, **kwargs):
            forwards.append(threading.get_ident())
            assert torch.is_inference_mode_enabled()
            assert torch.get_num_threads() == 1
            if len(forwards) == 1:
                forward_started.set()
                assert gate.wait(5), "test did not release timed-out forward"
                if failed_forward:
                    raise RuntimeError("synthetic post-timeout forward failure")
            return original_forward(*args, **kwargs)

        monkeypatch.setattr(loaded._model, "forward", forward)
        return ObservedCascade(loaded, _fixture_cpu=True)

    app = create_app(factory)

    async def exercise(url):
        run = asyncio.create_task(replay(url, records))
        try:
            await wait_until(forward_started.is_set)
            # Exceed the real harness deadline, not a test-only timeout override.
            await asyncio.sleep(2.15)
            assert not run.done()
            assert app.state.owner.admitted_requests == 1
            assert [event for event, _ in events].count("scan") == 1
        finally:
            gate.set()
        return await asyncio.wait_for(run, 5)

    with loopback_server(app) as url:
        result = asyncio.run(exercise(url))
    assert result.warmup[0].error == "timeout"
    assert result.warmup[0].elapsed_ms >= 2000
    assert result.warmup[0].response is None
    assert [row.error for row in result.measured] == [None, None]
    assert [row.response.stage2_invoked for row in result.measured] == [True, False]
    assert result.measured[0].response.action == (
        "alert" if result.measured[0].response.probability >= 0.5 else "allow"
    )
    assert result.measured[1].response.action == "allow"
    assert result.after_warmup.model_dump() == {
        "admitted_requests": 1,
        "completed_requests": int(not failed_forward),
        "failed_requests": int(failed_forward),
        "transformer_forward_attempts": 1,
        "successful_transformer_scores": int(not failed_forward),
    }
    assert result.after_measured.model_dump() == {
        "admitted_requests": 3,
        "completed_requests": 3 - int(failed_forward),
        "failed_requests": int(failed_forward),
        "transformer_forward_attempts": 2,
        "successful_transformer_scores": 2 - int(failed_forward),
    }
    assert [event for event, _ in events] == [
        "create",
        "enter",
        "scan",
        "scan",
        "scan",
        "exit",
    ]
    owner_ids = {ident for _, ident in events}
    assert len(owner_ids) == 1
    assert threading.get_ident() not in owner_ids
    assert set(forwards) == owner_ids
    assert not app.state.owner.is_alive
    assert (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
    ) == original_mode


def test_socket_drain_fences_late_streaming_body_without_blocking_new_phase():
    holder = []
    app = create_app(factory_for(holder))

    async def exercise(url):
        body_started, release_body = asyncio.Event(), asyncio.Event()

        async def slow_body():
            yield b'{"request_id":"late-warmup",'
            body_started.set()
            await release_body.wait()
            yield b'"url":"must-never-score"}'

        async with httpx.AsyncClient(base_url=url, trust_env=False) as client:
            delayed = asyncio.create_task(
                client.post(
                    "/v1/scan",
                    content=slow_body(),
                    headers={"content-type": "application/json"},
                )
            )
            await asyncio.wait_for(body_started.wait(), 3)
            try:
                snapshot = await client.post(
                    "/v1/drain", json={"request_ids": ["late-warmup"]}
                )
                assert snapshot.status_code == 200
                assert all(value == 0 for value in snapshot.json().values())
                measured = await client.post(
                    "/v1/scan",
                    json={"request_id": "measured", "url": "stage2-measured"},
                )
                assert measured.status_code == 200
                assert measured.json()["admission_sequence"] == 1
                assert not delayed.done()
            finally:
                release_body.set()
            assert (await delayed).status_code == 503
            assert (await client.post("/v1/drain")).json() == {
                "admitted_requests": 1,
                "completed_requests": 1,
                "failed_requests": 0,
                "transformer_forward_attempts": 1,
                "successful_transformer_scores": 1,
            }

    with loopback_server(app) as url:
        asyncio.run(exercise(url))
    assert holder[0].urls == ["stage2-measured"]
    assert not app.state.owner.is_alive
