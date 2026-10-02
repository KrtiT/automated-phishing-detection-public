import asyncio
import json
from importlib import import_module

import httpx
import pytest
from test_http_replay import fixture_app, loopback_server
from test_selective_service import factory_for

from automated_phishing_detection import http_replay
from automated_phishing_detection.http_run_codec import (
    HttpRunCodecError,
    encode_http_run,
)
from automated_phishing_detection.selective_service import create_app


def extension():
    assert hasattr(http_replay, "_replay_run"), "extension core is not implemented"
    return import_module("automated_phishing_detection.worker_connection_replay")


def records(count=32):
    return tuple(
        http_replay.ReplayRequest(f"record-{index}", f"https://safe{index}.example/a")
        for index in range(count)
    )


async def replay(url, **kwargs):
    return await extension().replay_worker_connections(
        url,
        records(),
        manifest_sha256="a" * 64,
        prevalence_basis_points=100,
        concurrency=8,
        run_index=1,
        warmup_count=8,
        **kwargs,
    )


def test_worker_connections_preserve_responses_and_mark_separate_protocol():
    extension()
    holder = []
    app = create_app(factory_for(holder))
    checkpoints = []
    with loopback_server(app) as url:
        result = asyncio.run(
            replay(
                url, retain=lambda name, content: checkpoints.append((name, content))
            )
        )
    assert result.protocol == "worker-connection-replay-v1"
    run = result.run
    assert [outcome.record_id for outcome in run.measured] == [
        row.record_id for row in records()
    ]
    assert all(outcome.error is None for outcome in run.measured)
    assert all(outcome.response.action == "allow" for outcome in run.measured)
    assert run.after_measured.completed_requests == 40
    assert run.after_measured.transformer_forward_attempts == 0
    assert not app.state.owner.is_alive
    assert len(checkpoints) == 2
    assert [name for name, _ in checkpoints] == ["warmup.json", "measured.json"]
    for _, content in checkpoints:
        payload = json.loads(content)
        assert payload["protocol"] == result.protocol
        assert payload["progress"]["concurrency"] == 8
    with pytest.raises(http_replay.ReplayError, match="typed HTTP runs"):
        http_replay.primary_http_summary([result] * 5)
    with pytest.raises(HttpRunCodecError):
        encode_http_run(result)


def test_each_worker_keeps_one_distinct_client_across_both_phases(monkeypatch):
    extension()
    original_post = httpx.AsyncClient.post
    original_enter = httpx.AsyncClient.__aenter__
    original_exit = httpx.AsyncClient.__aexit__
    clients, exited, worker_ids, connections = [], [], {}, {}

    async def enter(client):
        clients.append(client)
        return await original_enter(client)

    async def leave(client, *args):
        exited.append(client)
        return await original_exit(client, *args)

    async def post(client, url, **kwargs):
        response = await original_post(client, url, **kwargs)
        if url == "/v1/scan":
            request_id = kwargs["json"]["request_id"]
            phase = request_id.split(".")[-2]
            worker_ids.setdefault(phase, set()).add(id(client))
            stream = response.extensions["network_stream"]
            local_address = stream.get_extra_info("client_addr")
            connections.setdefault(id(client), set()).add(local_address)
        return response

    monkeypatch.setattr(httpx.AsyncClient, "__aenter__", enter)
    monkeypatch.setattr(httpx.AsyncClient, "__aexit__", leave)
    monkeypatch.setattr(httpx.AsyncClient, "post", post)
    app, _ = fixture_app()
    with loopback_server(app) as url:
        asyncio.run(replay(url))
    assert len(clients) == len(exited) == 8
    assert worker_ids["warmup"] == worker_ids["measured"] == set(connections)
    assert len(connections) == 8
    assert all(len(addresses) == 1 for addresses in connections.values())
    assert len({next(iter(addresses)) for addresses in connections.values()}) == 8
    assert all(client.is_closed for client in clients)


@pytest.mark.parametrize(
    "mode,error",
    [
        ("status", "http_status"),
        ("json", "invalid_json"),
        ("schema", "invalid_schema"),
        ("correlation", "correlation"),
    ],
)
def test_worker_client_preserves_terminal_errors_without_retries(mode, error):
    extension()
    app, state = fixture_app(mode)
    with loopback_server(app) as url:
        result = asyncio.run(replay(url))
    assert len(result.run.measured) == 32
    assert all(outcome.error == error for outcome in result.run.measured)
    assert len(state["calls"]) == 40
    assert len({call["request_id"] for call in state["calls"]}) == 40


def test_partial_client_setup_failure_closes_open_clients_and_retains_identity(
    monkeypatch,
):
    module = extension()
    original_enter = httpx.AsyncClient.__aenter__
    clients = []

    async def enter(client):
        clients.append(client)
        if len(clients) == 3:
            raise RuntimeError("synthetic enter failure")
        return await original_enter(client)

    monkeypatch.setattr(httpx.AsyncClient, "__aenter__", enter)
    with pytest.raises(http_replay.ReplayError) as caught:
        asyncio.run(replay("http://127.0.0.1:12345"))
    progress = json.loads(caught.value.progress)
    assert progress["protocol"] == module.PROTOCOL
    assert progress["progress"]["stage"] == "client_start"
    assert all(client.is_closed for client in clients)
    assert len(clients) == 3


def test_checkpoint_failure_preserves_extension_evidence_without_retry():
    module = extension()
    checkpoints = []

    def retain(name, content):
        checkpoints.append(name)
        raise RuntimeError("synthetic storage failure")

    app, _ = fixture_app()
    with loopback_server(app) as url:
        with pytest.raises(http_replay.ReplayError) as caught:
            asyncio.run(replay(url, retain=retain))
    assert checkpoints == ["warmup.json"]
    payload = json.loads(caught.value.progress)
    assert payload["protocol"] == module.PROTOCOL
    assert payload["progress"]["stage"] == "warmup_checkpoint"
    assert payload["progress"]["measured_started"] == [False] * 32


def test_external_cancellation_retains_identity_and_closes_clients(monkeypatch):
    module = extension()
    original_enter = httpx.AsyncClient.__aenter__
    clients = []

    async def enter(client):
        clients.append(client)
        return await original_enter(client)

    monkeypatch.setattr(httpx.AsyncClient, "__aenter__", enter)

    async def exercise(url, state):
        task = asyncio.create_task(replay(url))
        for _ in range(1000):
            if any(".measured." in call["request_id"] for call in state["calls"]):
                break
            await asyncio.sleep(0.001)
        else:
            pytest.fail("measured requests did not begin")
        task.cancel()
        with pytest.raises(asyncio.CancelledError) as caught:
            await task
        return json.loads(http_replay.replay_progress(caught.value))

    app, state = fixture_app("timeout")
    with loopback_server(app) as url:
        payload = asyncio.run(exercise(url, state))
    assert payload["protocol"] == module.PROTOCOL
    assert payload["progress"]["stage"] == "measured"
    assert any(payload["progress"]["measured_started"])
    assert len(clients) == 8
    assert all(client.is_closed for client in clients)
