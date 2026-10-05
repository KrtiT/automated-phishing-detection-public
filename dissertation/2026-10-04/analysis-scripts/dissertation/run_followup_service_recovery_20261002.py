"""One explicitly authorized full service schedule after environmental interruption."""

import argparse
import asyncio
import hashlib
import json
import os
import platform
import subprocess
import time
from pathlib import Path

import run_followup_service as runner


def retained_hash(path):
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def verify_recovery(arguments, followup):
    if runner.digest(arguments.authorization) != arguments.authorization_sha256:
        raise ValueError("recovery authorization changed")
    authorization = json.loads(arguments.authorization.read_bytes())
    required = {
        "attempt_name": "service-comparison-v2",
        "maximum_new_schedules": 1,
        "minimum_stable_seconds": 180,
        "sample_interval_seconds": 5,
        "pool_interrupted_attempt": False,
        "launcher_sha256": runner.digest(Path(__file__)),
        "test_sha256": runner.digest(Path(__file__).with_name("test_followup_service_recovery_20261002.py")),
    }
    if any(authorization.get(key) != value for key, value in required.items()):
        raise ValueError("recovery scope or implementation changed")
    names = {"execution-manifest-v1.json", "comparison-specification-v1.md",
             "service-recovery-amendment-v2.md", "service-v1-preservation.json"}
    if set(authorization["files"]) != names:
        raise ValueError("recovery evidence binding is incomplete")
    if arguments.manifest.resolve() != (followup / "execution-manifest-v1.json").resolve():
        raise ValueError("recovery must use the original execution manifest")
    for name, expected in authorization["files"].items():
        if runner.digest(followup / name) != expected:
            raise ValueError(f"bound evidence changed: {name}")
    receipt = json.loads((followup / "service-v1-preservation.json").read_bytes())
    if (len(receipt["completed_arms"]) != 45
            or receipt["primary_structural_c64_arms_completed"] != 0
            or receipt["pooling_into_recovery_permitted"] is not False):
        raise ValueError("interrupted attempt eligibility changed")
    previous = followup / "service-comparison-v1"
    observed = {str(path.relative_to(previous)): retained_hash(path)
                for path in sorted(previous.rglob("*")) if path.is_file()}
    if observed != receipt["source_sha256"]:
        raise ValueError("interrupted evidence inventory changed")
    return authorization


def stable_preflight(directory, inhibitor):
    started = time.monotonic()
    while True:
        runner.record_conditions(directory / "conditions.jsonl", inhibitor)
        if time.monotonic() - started >= 180:
            return
        time.sleep(5)


def execute(arguments):
    manifest, followup, _ = runner.verify_execution(arguments.context_root, arguments.manifest)
    authorization = verify_recovery(arguments, followup)
    root = followup / "service-comparison-v2"
    if root.exists():
        raise FileExistsError("the single authorized recovery schedule already exists")
    preflight = followup / "service-recovery-preflight-v2"
    preflight.mkdir(mode=0o700)
    active = preflight
    runner.write_json(preflight / "intent.json", {
        "started_at": runner.host.stamp(), "authorization_sha256": arguments.authorization_sha256,
        "minimum_stable_seconds": 180, "sample_interval_seconds": 5,
    })
    inhibitor = None
    try:
        inhibitor = subprocess.Popen(["/usr/bin/caffeinate", "-dims", "-w", str(os.getpid())])
        time.sleep(0.5)
        stable_preflight(preflight, inhibitor)
        runner.write_json(preflight / "completion.json", {
            "completed_at": runner.host.stamp(),
            "conditions_sha256": runner.digest(preflight / "conditions.jsonl"),
        })
        runner.record_conditions(preflight / "transition.jsonl", inhibitor)
        root.mkdir(mode=0o700)
        active = root
        requests = runner.comparison.synthetic_requests()
        content = runner.canonical_bytes({
            "protocol": runner.comparison.PROTOCOL, "class_prevalence": None,
            "requests": [{"record_id": row.record_id, "raw_url": row.raw_url} for row in requests],
        })
        manifest_hash = hashlib.sha256(content).hexdigest()
        runner.retain_bytes(root / "synthetic-manifest.json", content)
        runner.write_json(root / "intent.json", {
            "started_at": runner.host.stamp(), "manifest": manifest,
            "execution_manifest_sha256": runner.digest(arguments.manifest),
            "authorization_sha256": arguments.authorization_sha256,
            "recovery_authorization": authorization,
            "preflight_completion_sha256": runner.digest(preflight / "completion.json"),
            "preflight_transition_sha256": runner.digest(preflight / "transition.jsonl"),
            "synthetic_manifest_sha256": manifest_hash, "schedule": runner.comparison.schedule(),
            "hardware": runner.host.command("/usr/sbin/sysctl", "hw.model", "hw.memsize", "hw.ncpu", "machdep.cpu.brand_string"),
            "platform": platform.platform(),
        })
        runner.retain_bytes(root / "conditions.jsonl", b"")
        runner.record_conditions(root / "conditions.jsonl", inhibitor)
        asyncio.run(runner.guarded_schedule(arguments, root, inhibitor, requests, manifest_hash))
        runner.record_conditions(root / "conditions.jsonl", inhibitor)
        inhibitor.terminate()
        inhibitor.wait(timeout=10)
        result = runner.reduce_schedule(root, requests, manifest_hash)
        result["completed_at"] = runner.host.stamp()
        result["recovery_authorization_sha256"] = arguments.authorization_sha256
        result["interrupted_schedule_pooled"] = False
        runner.write_json(root / "completion.json", result)
        print(json.dumps(result["primary"], indent=2), flush=True)
    except BaseException as error:
        runner.write_json(active / "failure.json", {
            "type": type(error).__name__, "message": str(error), "at": runner.host.stamp(),
        })
        raise
    finally:
        if inhibitor is not None:
            if inhibitor.returncode is None:
                inhibitor.terminate()
                inhibitor.wait(timeout=10)
            runner.write_json(active / "sleep-inhibitor-exit.json", {
                "pid": inhibitor.pid, "exit_code": inhibitor.returncode,
                "ended_at": runner.host.stamp(),
            })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--authorization-sha256", required=True)
    arguments = parser.parse_args()
    for name in ("context_root", "manifest", "authorization"):
        setattr(arguments, name, getattr(arguments, name).resolve(strict=True))
    execute(arguments)


if __name__ == "__main__":
    main()
