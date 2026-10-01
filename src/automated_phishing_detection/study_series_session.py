"""Launch exactly one fixed series root under live physical supervision."""

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from . import _study_series_session_conditions as host
from . import _study_series_session_monitor as monitor
from ._exception_cleanup import CleanupStack
from ._process_support import _defer_interrupt, _InterruptGuard, command_hash
from ._study_authorized_cli import keywords
from ._study_series_session_records import hold_records
from ._study_series_supervisor_channel import LOCATOR, admission, supervisor_channel
from .study_series_execution import (
    bind_series_public_execution,
    recheck_series_public_execution,
)
from .study_series_runner import require_final_policy


def root_command(arguments):
    return [
        sys.executable,
        "scripts/run_study_series.py",
        "--repo-root",
        str(arguments.repo_root),
        "--expected-revision",
        arguments.expected_revision,
        "--expected-profile-sha256",
        arguments.expected_profile_sha256,
        "--envelope",
        str(arguments.envelope),
        "--expected-envelope-sha256",
        arguments.expected_envelope_sha256,
    ]


def _preflight(arguments):
    public = bind_series_public_execution(
        arguments.repo_root,
        **keywords(arguments),
        expected_profile_sha256=arguments.expected_profile_sha256,
    )
    require_final_policy(public)
    monitor.require(
        not any(name.startswith("APD_") for name in os.environ),
        "unexpected_admission_environment",
    )
    initial = host.capture()
    selected = host.violation(initial)
    monitor.require(selected is None, selected)
    return public, initial


def _prepare(state):
    with _defer_interrupt():
        state.inhibitor = subprocess.Popen(
            ["/usr/bin/caffeinate", "-dims", "-w", str(os.getpid())]
        )
    recheck_series_public_execution(state.public)
    require_final_policy(state.public)
    observed, state.observed_ns = monitor.ready(state.inhibitor)
    state.records.record(
        "pre.json",
        observed
        | {
            "initial": state.initial,
            "caffeinate_pid": state.inhibitor.pid,
            "profile_sha256": state.public.profile_sha256,
            "envelope_sha256": state.public.envelope_sha256,
            "revision": state.public.base.revision,
            "sampling_interval_seconds": 30,
            "physical_evidence_scope": "sampled_non_atomic_operator_reserved_session",
        },
    )


def _spawn(state):
    root = state.public.base.root
    environment = dict(os.environ, PYTHONPATH=str(root / "src"))
    environment[LOCATOR] = str(state.channel.read_fd)
    with _defer_interrupt():
        state.process = subprocess.Popen(
            state.command,
            cwd=root,
            env=environment,
            stdout=state.records.streams["stdout.log"],
            stderr=state.records.streams["stderr.log"],
            pass_fds=(state.channel.read_fd,),
            close_fds=True,
            start_new_session=True,
        )
    state.channel.launched()
    _launch_record(state)
    state.channel.release(
        admission(
            state.arguments,
            state.process.pid,
            state.command,
            observed_ns=state.observed_ns,
        )
    )


def _launch_record(state):
    state.records.record(
        "launch.json",
        {
            "started_at": host.stamp(),
            "root_pid": state.process.pid,
            "supervisor_pid": os.getpid(),
            "caffeinate_pid": state.inhibitor.pid,
            "command": state.command,
            "command_sha256": command_hash(state.command),
            "profile_sha256": state.public.profile_sha256,
            "envelope_sha256": state.public.envelope_sha256,
        },
    )


def _begin(state):
    _prepare(state)
    _spawn(state)
    monitor.monitor(state)


def _owned(state):
    with _InterruptGuard(lambda: {}) as guard, CleanupStack() as cleanup:
        cleanup.callback(monitor.stop_inhibitor, state)
        cleanup.callback(monitor.post, state)
        cleanup.callback(monitor.stop_root, state)
        try:
            guard.run(_begin, state)
        except BaseException as error:
            state.failure = type(error).__name__
            raise
    return state.process.returncode or (2 if state.selected else 0)


def run_series_session(arguments):
    """Bind public identity first; never open research inputs in the supervisor."""
    public, initial = _preflight(arguments)
    profile = json.loads(public.profile_bytes)
    path = Path(profile["paths"]["physical_records_dir"])
    with hold_records(path) as records, supervisor_channel() as channel:
        state = SimpleNamespace(
            arguments=arguments,
            public=public,
            initial=initial,
            records=records,
            channel=channel,
            command=root_command(arguments),
            inhibitor=None,
            process=None,
            observed_ns=None,
            selected=None,
            failure=None,
        )
        return _owned(state)
