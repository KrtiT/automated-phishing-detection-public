"""Invented sampled host history; no actual system observations or research IO."""

import json
from hashlib import sha256
from types import SimpleNamespace

from stopped_study_authorization_fixtures import (
    make_stopped,
    refresh_accounting,
    verify,
)

from automated_phishing_detection._checkpoint_codec import canonical_bytes

FILES = (
    "run_urlnorm_attempt4.py",
    "urlnorm_session_conditions.py",
    "urlnorm_attempt4_records.py",
)
ASSERTIONS = (
    "PreventSystemSleep",
    "PreventUserIdleSystemSleep",
    "PreventUserIdleDisplaySleep",
)


def observation(seconds, parent, *, next_pid=None, battery=False):
    rows = [
        "PID PPID %CPU %MEM COMM",
        "901 1 0.0 0.1 python",
        "902 901 0.0 0.1 caffeinate",
        f"{parent} 901 0.0 0.1 python",
    ]
    if next_pid is not None:
        rows.append(f"{next_pid} {parent} 0.0 0.1 python")
    return {
        "observed_at": f"2026-01-01T00:00:{seconds:02d}+00:00",
        "battery": f"Now drawing from '{'Battery' if battery else 'AC'} Power'\n",
        "power_settings": "Battery Power:\n powermode 0\nAC Power:\n powermode 0\n",
        "thermal": "Note: No thermal warning level has been recorded\n",
        "assertions": "\n".join(f"pid 902(caffeinate): {kind}" for kind in ASSERTIONS),
        "processes": "\n".join(rows) + "\n",
        "competing_known_workload_pids": [],
        "interference_scope": "Known-command detection plus retained process observations; operator reservation remains required.",
    }


def make_timeline(prepared, manifests):
    root = make_stopped(prepared, manifests, prefix=1, stopped_admissions=1)
    root.accounting["authorization_ledger"]["admissions"][-1]["launched_pid"] = 903
    refresh_accounting(root)
    authority = verify(root)
    next_pid = root.accounting["authorization_ledger"]["admissions"][-1]["launched_pid"]
    hashes = {name: sha256(name.encode()).hexdigest() for name in FILES}
    common = dict(
        profile_sha256=authority.profile_sha256,
        envelope_sha256=authority.envelope_sha256,
        supervisor_files_sha256=hashes,
    )
    pre = observation(0, authority.parent_pid)
    launch = _launch(authority, common)
    values = _values(authority, next_pid) | {
        "pre.json": pre | dict(initial=pre, caffeinate_pid=902, **common),
        "launch.json": launch,
    }
    case = SimpleNamespace(root=root, authority=authority, values=values, hashes=hashes)
    refresh_timeline(case)
    return case


def _launch(authority, common):
    profile = json.loads(authority.profile_bytes)
    arguments = [
        "/invented/python",
        "scripts/run_adopted_study.py",
        "--repo-root",
        "/invented/checkout",
        "--expected-revision",
        profile["execution"]["revision"],
        "--envelope",
        "/invented/envelope.json",
        "--expected-envelope-sha256",
        authority.envelope_sha256,
    ]
    return dict(
        root_pid=authority.parent_pid,
        supervisor_pid=901,
        caffeinate_pid=902,
        launched_at="2026-01-01T00:00:01+00:00",
        cwd="/invented/checkout",
        arguments=arguments,
        automatic_retry=False,
        workload_cutoff_seconds=None,
        **common,
    )


def _values(authority, next_pid):
    samples = [
        observation(2, authority.parent_pid),
        observation(30, authority.parent_pid, next_pid=next_pid),
        observation(40, authority.parent_pid, next_pid=next_pid),
        observation(45, authority.parent_pid, battery=True),
    ]
    post = observation(50, authority.parent_pid, battery=True) | dict(
        ended_at="2026-01-01T00:00:51+00:00",
        root_pid=authority.parent_pid,
        root_exit_code=130,
        session_violation="ac_power_absent",
        supervisor_error_type=None,
    )
    return {
        "conditions.jsonl": samples,
        "post.json": post,
        "sleep-cleanup.json": dict(
            recorded_at="2026-01-01T00:00:52+00:00",
            caffeinate_pid=902,
            exit_code=-15,
            assertions="",
        ),
    }


def refresh_timeline(case):
    case.payloads = tuple(
        (
            name,
            b"".join(map(canonical_bytes, value))
            if name == "conditions.jsonl"
            else canonical_bytes(value),
        )
        for name, value in case.values.items()
    )
    case.pins = {name: sha256(content).hexdigest() for name, content in case.payloads}


def verify_timeline(case, **overrides):
    from automated_phishing_detection.stopped_study_timeline import (
        verify_stopped_study_timeline,
    )

    arguments = dict(
        expected_profile_sha256=case.authority.profile_sha256,
        expected_envelope_sha256=case.authority.envelope_sha256,
        expected_snapshot_sha256=case.root.pins,
        expected_observation_sha256=case.pins,
        expected_supervisor_sha256=case.hashes,
    )
    return verify_stopped_study_timeline(
        case.root.snapshot, case.payloads, **(arguments | overrides)
    )
