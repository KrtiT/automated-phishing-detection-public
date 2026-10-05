"""Explicit follow-up client, separate from the frozen shared-pool HTTP protocol."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass

from . import http_replay

PROTOCOL = "worker-connection-replay-v1"


@dataclass(frozen=True)
class WorkerConnectionRun:
    run: http_replay.HttpRun
    protocol: str = PROTOCOL


def _envelope(content: bytes) -> bytes:
    return (
        json.dumps(
            {"protocol": PROTOCOL, "progress": http_replay._json(content)},
            allow_nan=False,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("ascii")


async def replay_worker_connections(
    base_url: str,
    requests: tuple[http_replay.ReplayRequest, ...] | list[http_replay.ReplayRequest],
    *,
    manifest_sha256: str,
    prevalence_basis_points: int,
    concurrency: int,
    run_index: int,
    warmup_count: int = 1000,
    workload: str = "fixed_cascade",
    retain: Callable[[str, bytes], None] | None = None,
) -> WorkerConnectionRun:
    """Keep one persistent client per worker without changing measurement rules.

    This is an extension measurement, not replacement evidence for original H3.
    Scientific data access and execution approval are the caller's responsibility.
    """
    if retain is not None and not callable(retain):
        raise http_replay.ReplayError("retain must be a callable or None")

    def wrapped_retain(name, content):
        if retain is not None:
            retain(name, _envelope(content))

    try:
        run = await http_replay._replay_run(
            base_url,
            requests,
            manifest_sha256=manifest_sha256,
            prevalence_basis_points=prevalence_basis_points,
            concurrency=concurrency,
            run_index=run_index,
            warmup_count=warmup_count,
            workload=workload,
            retain=wrapped_retain,
            worker_connections=True,
        )
    except BaseException as error:
        progress = getattr(error, "progress", None)
        if type(progress) is bytes:
            error.progress = _envelope(progress)
        raise
    return WorkerConnectionRun(run)
