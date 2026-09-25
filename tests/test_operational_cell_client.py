"""A restored client records its actual role and invokes one existing replayer."""

import asyncio

import pytest
from operational_bound_child_fixtures import child_case, keywords
from operational_child_fixtures import child_context, child_module
from operational_transport_integration_fixtures import (
    accepted,
    candidates,
    case,
    manifests,
)

__all__ = ["accepted", "candidates", "case", "manifests", "child_context"]


@pytest.mark.parametrize("ordinal", [1, 91, 121])
def test_one_replay_uses_only_restored_inputs(
    child_context, accepted, case, tmp_path, monkeypatch, ordinal
):
    api = child_module("client")
    fixture = child_case(tmp_path, accepted, case, monkeypatch, ordinal)
    monkeypatch.setattr(child_context, "recheck_binding", lambda binding: None)
    calls, result = [], object()

    async def replay(base_url, inputs, **kwargs):
        assert (fixture.attempt.directory / "client-role.json").is_file()
        calls.append((base_url, inputs, kwargs))
        kwargs["retain"]("warmup.json", b"first\n")
        kwargs["retain"]("measured.json", b"second\n")
        return result

    monkeypatch.setattr(api, "replay_run", replay)
    monkeypatch.setattr(api, "replay_shift_run", replay)
    monkeypatch.setattr(
        api, "encode_http_run", lambda run: b"complete\n" if run is result else None
    )
    monkeypatch.setattr(
        api, "encode_shift_run", lambda run: b"complete\n" if run is result else None
    )
    asyncio.run(
        api._run_bound_client(fixture.binding, fixture.profile, **keywords(fixture))
    )
    assert len(calls) == 1 and calls[0][0] == "http://127.0.0.1:1234"
    actual = calls[0][1].requests if ordinal == 121 else calls[0][1]
    assert actual == fixture.requests
    assert (fixture.attempt.directory / "run.json").read_bytes() == b"complete\n"


def test_failure_retains_existing_progress_without_retry(
    child_context, accepted, case, tmp_path, monkeypatch
):
    api = child_module("client")
    fixture = child_case(tmp_path, accepted, case, monkeypatch)
    monkeypatch.setattr(child_context, "recheck_binding", lambda binding: None)
    error = KeyboardInterrupt("first")
    error.progress = b'{"partial":"invented"}\n'
    calls = []

    async def interrupted(*args, **kwargs):
        calls.append(1)
        raise error

    monkeypatch.setattr(api, "replay_shift_run", interrupted)
    with pytest.raises(KeyboardInterrupt) as caught:
        asyncio.run(
            api._run_bound_client(fixture.binding, fixture.profile, **keywords(fixture))
        )
    assert caught.value is error and calls == [1]
    assert (
        fixture.attempt.directory / "client-failure.json"
    ).read_bytes() == error.progress
    assert not (fixture.attempt.directory / "run.json").exists()
