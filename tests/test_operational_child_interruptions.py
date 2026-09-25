"""Direct replay interruptions must retain their own in-flight evidence."""

import asyncio
import os
import signal

import pytest

from automated_phishing_detection import http_replay, shift_replay
from automated_phishing_detection.http_replay import ReplayRequest
from automated_phishing_detection.shift_schema import ShiftPlan


class Client:
    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False


def interrupted_phase(captured):
    async def phase(*args):
        progress = args[-2]
        progress.stage = "warmup"
        try:
            os.kill(os.getpid(), signal.SIGINT)
        except KeyboardInterrupt as error:
            captured.append((error, progress.snapshot()))
            raise

    return phase


def replay(role):
    rows = (ReplayRequest("one", "https://invented.test"),)
    return (
        http_replay.replay_run(
            "http://127.0.0.1:1",
            rows,
            manifest_sha256="a" * 64,
            prevalence_basis_points=100,
            concurrency=1,
            run_index=1,
            warmup_count=1,
        )
        if role == "http"
        else shift_replay.replay_shift_run(
            "http://127.0.0.1:1", ShiftPlan("a" * 64, 1, rows, 1)
        )
    )


@pytest.mark.parametrize("role", ["http", "shift"])
@pytest.mark.parametrize(
    "cleanup", [None, OSError("cleanup"), KeyboardInterrupt("later")]
)
def test_actual_sigint_retains_exact_replay_progress(monkeypatch, role, cleanup):
    captured = []
    target = http_replay if role == "http" else shift_replay
    monkeypatch.setattr(target, "_replay_phases", interrupted_phase(captured))
    monkeypatch.setattr(http_replay.httpx, "AsyncClient", Client)
    if cleanup is not None:

        async def leave(*args):
            raise cleanup

        monkeypatch.setattr(Client, "__aexit__", leave)
    with pytest.raises(KeyboardInterrupt) as caught:
        asyncio.run(replay(role))
    assert caught.value is captured[0][0]
    assert vars(caught.value).get("progress") == captured[0][1]
