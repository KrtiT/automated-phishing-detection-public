"""Sample one session, revoke on violations, and retain actual postconditions."""

import os
import signal
import subprocess
import time

from . import _study_series_session_conditions as host
from ._checkpoint_codec import canonical_bytes
from ._study_execution_policy import DEADLINES


def require(condition, message):
    if not condition:
        raise ValueError(message)


def ready(inhibitor):
    deadline = time.monotonic() + 5
    while True:
        observed_ns = time.monotonic_ns()
        observed = host.capture()
        selected = host.violation(observed)
        require(selected is None, selected)
        if host.inhibition(observed, inhibitor) is None:
            return observed, observed_ns
        require(inhibitor.poll() is None, "owned_sleep_inhibitor_exited")
        require(time.monotonic() < deadline, "owned_sleep_assertions_unavailable")
        time.sleep(0.1)


def monitor(state):
    while state.process.poll() is None:
        observed_ns = time.monotonic_ns()
        observed = host.capture()
        state.records.check()
        stream = state.records.streams["conditions.jsonl"]
        content = canonical_bytes(observed) + b"\n"
        require(stream.write(content) == len(content), "conditions_write_failed")
        stream.flush()
        current = host.violation(observed) or host.inhibition(observed, state.inhibitor)
        if current is not None and state.selected is None:
            state.selected = current
            stop_root(state)
            return
        elif state.selected is None:
            heartbeat(state, observed_ns)
        try:
            state.process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            pass


def heartbeat(state, observed_ns):
    try:
        state.channel.heartbeat(observed_ns)
    except ValueError:
        if state.process.poll() is None:
            raise


def stop_root(state):
    if state.process is not None and state.process.poll() is None:
        state.selected = state.selected or "supervisor_interrupted"
        state.channel.revoke()
        state.process.send_signal(signal.SIGINT)
        try:
            state.process.wait(timeout=DEADLINES["shutdown"])
        except subprocess.TimeoutExpired:
            _escalate_root(state)


def _group_signal(process_id, number):
    require(process_id != os.getpgrp(), "cannot_signal_supervisor_group")
    try:
        os.killpg(process_id, number)
        return True
    except ProcessLookupError:
        return False


def _escalate_root(state):
    state.shutdown_escalated = True
    _group_signal(state.process.pid, signal.SIGTERM)
    try:
        state.process.wait(timeout=DEADLINES["terminate"])
    except subprocess.TimeoutExpired:
        pass
    _group_signal(state.process.pid, signal.SIGKILL)
    state.process.wait(timeout=DEADLINES["kill"])
    deadline = time.monotonic() + DEADLINES["kill"]
    while _group_signal(state.process.pid, 0):
        require(time.monotonic() < deadline, "owned_root_group_cleanup_unconfirmed")
        time.sleep(0.05)


def post(state):
    try:
        final = host.capture()
        current = host.violation(final)
        if state.inhibitor is not None:
            current = current or host.inhibition(final, state.inhibitor)
    except BaseException as error:
        final = {"postcondition_capture_error_type": type(error).__name__}
        current = "postconditions_unavailable"
    state.selected = state.selected or current
    state.records.record(
        "post.json",
        final
        | {
            "ended_at": host.stamp(),
            "root_pid": None if state.process is None else state.process.pid,
            "root_exit_code": None if state.process is None else state.process.poll(),
            "session_violation": state.selected,
            "supervisor_error_type": state.failure,
            "root_shutdown_escalated": getattr(state, "shutdown_escalated", False),
        },
    )


def stop_inhibitor(state):
    if state.inhibitor is None:
        return
    if state.inhibitor.poll() is None:
        state.inhibitor.terminate()
    try:
        state.inhibitor.wait(timeout=DEADLINES["terminate"])
    except subprocess.TimeoutExpired:
        state.inhibitor.kill()
        state.inhibitor.wait(timeout=DEADLINES["kill"])
    observed = {
        "recorded_at": host.stamp(),
        "caffeinate_pid": state.inhibitor.pid,
        "exit_code": state.inhibitor.returncode,
    }
    try:
        observed["assertions"] = host.command("/usr/bin/pmset", "-g", "assertions")
    except BaseException as error:
        observed["assertions_capture_error_type"] = type(error).__name__
        raise
    finally:
        state.records.record("sleep-cleanup.json", observed)
