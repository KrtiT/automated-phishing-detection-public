"""Execute the fixed 80-arm extension, retaining every owned attempt and failure."""

import argparse
import asyncio
import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import uvicorn
from run_followup_detection import digest, verify_execution, write_json

from automated_phishing_detection import _study_series_session_conditions as host
from automated_phishing_detection import followup_service as comparison
from automated_phishing_detection import (
    http_replay,
    http_run_codec,
    worker_connection_replay,
)
from automated_phishing_detection._checkpoint_codec import canonical_bytes
from automated_phishing_detection.selective_service import create_app


class ConditionsError(RuntimeError):
    pass


def thermal_violation(content):
    expected = {
        "Note: No thermal warning level has been recorded",
        "Note: No performance warning level has been recorded",
        "Note: No CPU power status has been recorded",
    }
    if set(content.strip().splitlines()) == expected:
        return None
    return "thermal_or_performance_not_normal"


def retain_bytes(path, content):
    with path.open("xb") as output:
        os.chmod(path, 0o600)
        output.write(content)
        output.flush()
        os.fsync(output.fileno())


def record_conditions(path, inhibitor):
    observation = host.capture()
    problem = (
        host.violation(observation)
        or thermal_violation(observation["thermal"])
        or host.inhibition(observation, inhibitor)
    )
    observation["violation"] = problem
    with path.open("a") as output:
        output.write(json.dumps(observation, sort_keys=True, allow_nan=False) + "\n")
        output.flush()
    if problem:
        raise ConditionsError(problem)
    return observation


async def monitor_conditions(path, inhibitor):
    while True:
        await asyncio.to_thread(record_conditions, path, inhibitor)
        await asyncio.sleep(5)


async def wait_ready(process, base_url):
    deadline = time.monotonic() + 45
    async with httpx.AsyncClient(trust_env=False, timeout=1) as client:
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(
                    f"owned service exited during startup: {process.returncode}"
                )
            try:
                response = await client.post(base_url + "/v1/drain", json={})
            except httpx.TransportError:
                await asyncio.sleep(0.05)
                continue
            if response.status_code != 200 or any(response.json().values()):
                raise RuntimeError("service readiness did not confirm fresh counters")
            return
    raise RuntimeError("owned service startup deadline exceeded")


@asynccontextmanager
async def owned_server(command, directory):
    with socket.socket() as listener, (directory / "server.log").open("xb") as log:
        listener.bind(("127.0.0.1", 0))
        listener.listen(128)
        base_url = f"http://127.0.0.1:{listener.getsockname()[1]}"
        process = subprocess.Popen(
            [*command, "--socket-fd", str(listener.fileno())],
            pass_fds=(listener.fileno(),),
            stdin=subprocess.PIPE,
            stdout=log,
            stderr=subprocess.STDOUT,
            env={
                **os.environ,
                "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "VECLIB_MAXIMUM_THREADS": "1",
            },
        )
        forced = False
        try:
            write_json(
                directory / "launch.json",
                {
                    "pid": process.pid,
                    "parent_pid": os.getpid(),
                    "base_url": base_url,
                    "started_at": host.stamp(),
                },
            )
            await wait_ready(process, base_url)
            yield base_url
        finally:
            process.stdin.close()
            try:
                await asyncio.to_thread(process.wait, 65)
            except subprocess.TimeoutExpired:
                forced = True
                process.terminate()
                try:
                    await asyncio.to_thread(process.wait, 10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    await asyncio.to_thread(process.wait)
            write_json(
                directory / "exit.json",
                {
                    "pid": process.pid,
                    "exit_code": process.returncode,
                    "forced": forced,
                    "ended_at": host.stamp(),
                },
            )
            if forced or process.returncode != 0:
                raise RuntimeError(
                    f"owned service cleanup unsuccessful: {process.returncode}; forced={forced}"
                )


def serve(arguments):
    verify_execution(arguments.context_root, arguments.manifest)
    baseline = (
        arguments.context_root
        / "gwu_working/study-only-v1-reviewed-inputs/primary/logistic-l1.json"
    )
    app = create_app(
        lambda: comparison.scorer_session(arguments.workload, baseline),
        queue_capacity=128,
    )
    server = uvicorn.Server(
        uvicorn.Config(
            app,
            host="127.0.0.1",
            port=0,
            loop="asyncio",
            http="h11",
            access_log=False,
            log_level="warning",
            timeout_keep_alive=5,
            timeout_graceful_shutdown=60,
        )
    )

    def shutdown_on_parent_eof():
        sys.stdin.buffer.read()
        server.should_exit = True

    threading.Thread(target=shutdown_on_parent_eof, daemon=True).start()
    with socket.socket(fileno=arguments.socket_fd) as listener:
        server.run(sockets=[listener])


def wire_index(pair):
    return (pair - 1) % 5 + 1


def arm_name(arm):
    return (
        f"{arm['workload']}-c{arm['concurrency']}-pair{arm['pair']:02d}-{arm['client']}"
    )


async def measure_arm(arguments, root, arm, requests, manifest_hash):
    directory = root / arm_name(arm)
    directory.mkdir(mode=0o700)
    write_json(
        directory / "intent.json",
        {
            "protocol": comparison.PROTOCOL,
            "arm": arm,
            "started_at": host.stamp(),
            "synthetic_manifest_sha256": manifest_hash,
        },
    )
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--context-root",
        str(arguments.context_root),
        "--manifest",
        str(arguments.manifest),
        "--serve",
        "--workload",
        arm["workload"],
    ]

    def retain(name, content):
        retain_bytes(
            directory / name,
            canonical_bytes(
                {
                    "protocol": comparison.PROTOCOL,
                    "arm": arm,
                    "retained_progress": json.loads(content),
                }
            ),
        )

    try:
        async with owned_server(command, directory) as base_url:
            replay = (
                http_replay.replay_run
                if arm["client"] == "shared"
                else worker_connection_replay.replay_worker_connections
            )
            result = await replay(
                base_url,
                requests,
                manifest_sha256=manifest_hash,
                prevalence_basis_points=100,
                concurrency=arm["concurrency"],
                run_index=wire_index(arm["pair"]),
                warmup_count=1000,
                workload="fixed_cascade",
                retain=retain,
            )
            run = result if arm["client"] == "shared" else result.run
            record = {
                "protocol": comparison.PROTOCOL,
                "arm": arm,
                "synthetic_class_prevalence": None,
                "legacy_transport_record": json.loads(
                    http_run_codec.encode_http_run(run)
                ),
            }
            retain_bytes(directory / "run.json", canonical_bytes(record))
        write_json(
            directory / "completion.json",
            {
                "arm": arm,
                "run_sha256": digest(directory / "run.json"),
                "summary": comparison.describe_run(run),
                "completed_at": host.stamp(),
            },
        )
        print(f"completed {arm_name(arm)}", flush=True)
    except BaseException as error:
        progress = http_replay.replay_progress(error)
        if progress is not None:
            retain("interrupted.json", progress)
        write_json(
            directory / "failure.json",
            {"type": type(error).__name__, "message": str(error), "at": host.stamp()},
        )
        raise


async def measure_schedule(arguments, root, inhibitor, requests, manifest_hash):
    for arm in comparison.schedule():
        await asyncio.to_thread(record_conditions, root / "conditions.jsonl", inhibitor)
        await measure_arm(arguments, root, arm, requests, manifest_hash)
        await asyncio.to_thread(record_conditions, root / "conditions.jsonl", inhibitor)


async def guarded_schedule(arguments, root, inhibitor, requests, manifest_hash):
    measurement = asyncio.create_task(
        measure_schedule(arguments, root, inhibitor, requests, manifest_hash)
    )
    monitoring = asyncio.create_task(
        monitor_conditions(root / "conditions.jsonl", inhibitor)
    )
    try:
        done, _ = await asyncio.wait(
            (measurement, monitoring), return_when=asyncio.FIRST_COMPLETED
        )
        if monitoring in done:
            monitoring.result()
            raise ConditionsError("condition monitor exited unexpectedly")
        measurement.result()
    finally:
        measurement.cancel()
        monitoring.cancel()
        await asyncio.gather(measurement, monitoring, return_exceptions=True)


def load_run(root, arm, requests, manifest_hash):
    directory = root / arm_name(arm)
    complete = json.loads((directory / "completion.json").read_bytes())
    if complete["arm"] != arm or complete["run_sha256"] != digest(
        directory / "run.json"
    ):
        raise ValueError("arm provenance changed")
    exit_record = json.loads((directory / "exit.json").read_bytes())
    if exit_record["exit_code"] != 0 or exit_record["forced"]:
        raise ValueError("owned service did not exit successfully")
    record = json.loads((directory / "run.json").read_bytes())
    if (
        record["protocol"] != comparison.PROTOCOL
        or record["arm"] != arm
        or record["synthetic_class_prevalence"] is not None
    ):
        raise ValueError("wrong service extension identity")
    run = http_run_codec.decode_http_run(
        canonical_bytes(record["legacy_transport_record"]),
        expected_manifest_sha256=manifest_hash,
        expected_requests=requests,
        expected_prevalence_basis_points=100,
        expected_concurrency=arm["concurrency"],
        expected_run_index=wire_index(arm["pair"]),
        expected_workload="fixed_cascade",
    )
    if comparison.describe_run(run) != complete["summary"]:
        raise ValueError("arm summary does not recompute")
    return run


def reduce_schedule(root, requests, manifest_hash):
    arms, pairs, primary_pairs, latencies = [], [], [], []
    errors, agreement = 0, 0
    scheduled = comparison.schedule()
    for offset in range(0, len(scheduled), 2):
        selected = scheduled[offset : offset + 2]
        runs = {
            arm["client"]: load_run(root, arm, requests, manifest_hash)
            for arm in selected
        }
        summaries = {
            client: comparison.describe_run(run) for client, run in runs.items()
        }
        for arm in selected:
            arms.append({**arm, "summary": summaries[arm["client"]]})
        identity = {
            key: selected[0][key] for key in ("workload", "concurrency", "pair")
        }
        matching = comparison.response_agreement(runs["shared"], runs["worker"])
        pairs.append({**identity, "response_agreement": matching})
        if (
            identity["workload"] == "structural_detector"
            and identity["concurrency"] == 64
        ):
            primary_pairs.append(
                {
                    "pair": identity["pair"],
                    **{
                        f"{client}_p95_ms": summary["success_latency"]["p95_ms"]
                        for client, summary in summaries.items()
                    },
                }
            )
            latencies.extend(
                row.elapsed_ms for row in runs["worker"].measured if row.error is None
            )
            errors += summaries["worker"]["request_errors"]
            agreement += matching["exact_except_request_id"]
    return {
        "protocol": comparison.PROTOCOL,
        "arm_count": len(arms),
        "arms": arms,
        "pairs": pairs,
        "primary": comparison.primary_summary(
            primary_pairs, latencies, errors=errors, exact_agreement=agreement
        ),
        "original_hypothesis_decisions_changed": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--workload", choices=comparison.WORKLOADS)
    parser.add_argument("--socket-fd", type=int)
    arguments = parser.parse_args()
    arguments.context_root = arguments.context_root.resolve(strict=True)
    arguments.manifest = arguments.manifest.resolve(strict=True)
    if arguments.serve:
        return serve(arguments)
    manifest, followup, _ = verify_execution(arguments.context_root, arguments.manifest)
    root = followup / "service-comparison-v1"
    root.mkdir(mode=0o700)
    requests = comparison.synthetic_requests()
    content = canonical_bytes(
        {
            "protocol": comparison.PROTOCOL,
            "class_prevalence": None,
            "requests": [
                {"record_id": row.record_id, "raw_url": row.raw_url} for row in requests
            ],
        }
    )
    manifest_hash = hashlib.sha256(content).hexdigest()
    retain_bytes(root / "synthetic-manifest.json", content)
    write_json(
        root / "intent.json",
        {
            "started_at": host.stamp(),
            "manifest": manifest,
            "execution_manifest_sha256": digest(arguments.manifest),
            "synthetic_manifest_sha256": manifest_hash,
            "schedule": comparison.schedule(),
            "hardware": host.command(
                "/usr/sbin/sysctl",
                "hw.model",
                "hw.memsize",
                "hw.ncpu",
                "machdep.cpu.brand_string",
            ),
            "platform": platform.platform(),
        },
    )
    retain_bytes(root / "conditions.jsonl", b"")
    try:
        with subprocess.Popen(
            ["/usr/bin/caffeinate", "-dims", "-w", str(os.getpid())]
        ) as inhibitor:
            try:
                time.sleep(0.5)
                record_conditions(root / "conditions.jsonl", inhibitor)
                asyncio.run(
                    guarded_schedule(
                        arguments, root, inhibitor, requests, manifest_hash
                    )
                )
                record_conditions(root / "conditions.jsonl", inhibitor)
            finally:
                inhibitor.terminate()
                inhibitor.wait(timeout=10)
                write_json(
                    root / "sleep-inhibitor-exit.json",
                    {
                        "pid": inhibitor.pid,
                        "exit_code": inhibitor.returncode,
                        "ended_at": host.stamp(),
                    },
                )
        result = reduce_schedule(root, requests, manifest_hash)
        result["completed_at"] = host.stamp()
        write_json(root / "completion.json", result)
        print(json.dumps(result["primary"], indent=2))
    except BaseException as error:
        write_json(
            root / "failure.json",
            {"type": type(error).__name__, "message": str(error), "at": host.stamp()},
        )
        raise


if __name__ == "__main__":
    main()
