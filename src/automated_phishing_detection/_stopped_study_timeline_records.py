"""Closed supervisor records and original identity joins using only bytes."""

import re
from hashlib import sha256
from pathlib import PurePosixPath

from . import _operational_input_schema as schema
from . import _stopped_study_host as host
from ._operational_cell_publication_records import inventory

MEMBERS = frozenset(
    {"pre.json", "launch.json", "conditions.jsonl", "post.json", "sleep-cleanup.json"}
)
SUPERVISOR_NAMES = frozenset(
    {
        "run_urlnorm_attempt4.py",
        "urlnorm_session_conditions.py",
        "urlnorm_attempt4_records.py",
    }
)
IDENTITY = frozenset({"profile_sha256", "envelope_sha256", "supervisor_files_sha256"})
LAUNCH = IDENTITY | {
    "arguments",
    "cwd",
    "root_pid",
    "supervisor_pid",
    "caffeinate_pid",
    "launched_at",
    "automatic_retry",
    "workload_cutoff_seconds",
}


def authenticate(payloads, expected):
    values = inventory(payloads, MEMBERS)
    schema.keys(expected, MEMBERS)
    for name, content in values.items():
        schema.digest(expected[name])
        schema.require(sha256(content).hexdigest() == expected[name])
    parsed = {
        name: schema.loads(content)
        for name, content in values.items()
        if name != "conditions.jsonl"
    }
    stream = values["conditions.jsonl"]
    schema.require(stream and stream.endswith(b"\n"))
    parsed["conditions.jsonl"] = tuple(
        schema.loads(line) for line in stream.splitlines(keepends=True)
    )
    return parsed, tuple(sorted(expected.items()))


def identity(value, authority, supervisor_hashes):
    schema.keys(supervisor_hashes, SUPERVISOR_NAMES)
    for digest in supervisor_hashes.values():
        schema.digest(digest)
    schema.require(value["profile_sha256"] == authority.profile_sha256)
    schema.require(value["envelope_sha256"] == authority.envelope_sha256)
    schema.same(value["supervisor_files_sha256"], supervisor_hashes)


def launch_record(value, authority, supervisor_hashes):
    schema.keys(value, LAUNCH)
    identity(value, authority, supervisor_hashes)
    for key in ("root_pid", "supervisor_pid", "caffeinate_pid"):
        host.pid(value[key])
    schema.require(
        len({value[key] for key in ("root_pid", "supervisor_pid", "caffeinate_pid")})
        == 3
    )
    schema.require(value["root_pid"] == authority.parent_pid)
    schema.require(
        value["automatic_retry"] is False and value["workload_cutoff_seconds"] is None
    )
    host.timestamp(value["launched_at"])
    profile = schema.loads(authority.profile_bytes)
    schema.require(value["cwd"] == profile["paths"]["repo-root"])
    arguments(value["arguments"], profile, authority.envelope_sha256)


def arguments(values, profile, envelope_pin):
    schema.require(type(values) is list and len(values) == 10)
    schema.require(all(type(value) is str for value in values))
    schema.require(
        PurePosixPath(values[0]).is_absolute()
        and PurePosixPath(values[7]).is_absolute()
    )
    expected = [
        values[0],
        "scripts/run_adopted_study.py",
        "--repo-root",
        profile["paths"]["repo-root"],
        "--expected-revision",
        profile["execution"]["revision"],
        "--envelope",
        values[7],
        "--expected-envelope-sha256",
        envelope_pin,
    ]
    schema.require(values == expected)


def pre_record(value, launch, authority, supervisor_hashes):
    schema.keys(value, host.FIELDS | IDENTITY | {"initial", "caffeinate_pid"})
    identity(value, authority, supervisor_hashes)
    schema.require(
        type(value["caffeinate_pid"]) is int
        and value["caffeinate_pid"] == launch["caffeinate_pid"]
    )
    host.clean(value["initial"])
    observed = {key: value[key] for key in host.FIELDS}
    host.clean(observed, launch["caffeinate_pid"])
    host.owned_processes(observed, launch, root=False)
    schema.require(
        host.timestamp(value["initial"]["observed_at"])
        <= host.timestamp(value["observed_at"])
        < host.timestamp(launch["launched_at"])
    )


def post_record(value, launch, last):
    schema.keys(
        value,
        host.FIELDS
        | {
            "ended_at",
            "root_pid",
            "root_exit_code",
            "session_violation",
            "supervisor_error_type",
        },
    )
    host.sample({key: value[key] for key in host.FIELDS})
    schema.require(
        type(value["root_pid"]) is int and value["root_pid"] == launch["root_pid"]
    )
    schema.require(
        type(value["root_exit_code"]) is int and value["root_exit_code"] == 130
    )
    schema.require(
        value["session_violation"] == "ac_power_absent"
        and value["supervisor_error_type"] is None
    )
    schema.require(
        host.timestamp(last["observed_at"])
        < host.timestamp(value["observed_at"])
        <= host.timestamp(value["ended_at"])
    )


def cleanup_record(value, launch, post):
    schema.keys(value, {"recorded_at", "caffeinate_pid", "exit_code", "assertions"})
    schema.require(
        type(value["caffeinate_pid"]) is int
        and value["caffeinate_pid"] == launch["caffeinate_pid"]
    )
    schema.require(type(value["exit_code"]) is int and value["exit_code"] == -15)
    schema.require(type(value["assertions"]) is str)
    schema.require(
        re.search(
            rf"pid\s+{launch['caffeinate_pid']}\(caffeinate\):", value["assertions"]
        )
        is None
    )
    schema.require(
        host.timestamp(post["ended_at"]) <= host.timestamp(value["recorded_at"])
    )
