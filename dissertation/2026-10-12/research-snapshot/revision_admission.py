"""Single-process live authority for the bounded P/F/R/C revision schedule."""

import hashlib
import os
import signal
import stat
import subprocess
import threading
import time
from contextlib import contextmanager

from revision_population import _bytes, _write

from automated_phishing_detection import _study_series_session_conditions as host

TASKS = ("P", "F", "R", "C")


class AdmissionError(ValueError):
    pass


def fingerprint(path):
    observed = os.lstat(path)
    if not stat.S_ISREG(observed.st_mode):
        raise AdmissionError("bound_input_not_regular_file")
    return (
        observed.st_dev,
        observed.st_ino,
        observed.st_size,
        observed.st_mtime_ns,
        observed.st_ctime_ns,
    )


def file_hash(path):
    before = fingerprint(path)
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    if fingerprint(path) != before:
        raise AdmissionError("bound_file_changed_during_read")
    return digest.hexdigest()


class LiveParent:
    def __init__(self, profile, root, session_check, code_check=None):
        self.profile = profile
        self.root = root
        self.session_check = session_check
        self.code_check = code_check
        self.owner = (os.getpid(), threading.get_ident())
        self.started = False
        self.closed = False
        self.active = None
        self.completed = []
        self.identities = {}

    def _owner(self):
        if self.owner != (os.getpid(), threading.get_ident()):
            raise AdmissionError("parent_owner_changed")

    def verify_files(self, *, full=False):
        if full and self.code_check is not None:
            self.code_check()
        for path, expected in self.profile["files"].items():
            if path in self.identities and fingerprint(path) != self.identities[path]:
                raise AdmissionError("bound_file_changed")
            if full and file_hash(path) != expected:
                raise AdmissionError("bound_file_hash_changed")

    def start(self):
        self._owner()
        if self.started or self.closed:
            raise AdmissionError("attempt_already_used")
        if self.profile["readiness"] != dict.fromkeys(TASKS, True):
            raise AdmissionError("whole_study_not_ready")
        self.session_check()
        self.verify_files(full=True)
        self.identities = {path: fingerprint(path) for path in self.profile["files"]}
        self.root.mkdir(mode=0o700)
        _write(self.root / "profile.json", _bytes(self.profile))
        _write(
            self.root / "parent-started.json",
            _bytes(
                {
                    "pid": os.getpid(),
                    "observed_at": host.stamp(),
                    "tasks": TASKS,
                    "profile_sha256": hashlib.sha256(_bytes(self.profile)).hexdigest(),
                }
            ),
        )
        self.started = True

    def _check(self, name):
        self._owner()
        if self.closed or not self.started or self.active != name:
            raise AdmissionError("task_not_live")
        self.session_check()
        self.verify_files()

    def admit(self, name):
        self._owner()
        if self.closed or not self.started or self.active is not None:
            raise AdmissionError("parent_not_available")
        if len(self.completed) >= len(TASKS) or name != TASKS[len(self.completed)]:
            raise AdmissionError("task_order")
        self.session_check()
        self.verify_files(full=True)
        _write(
            self.root / f"{name}-admitted.json",
            _bytes(
                {
                    "observed_at": host.stamp(),
                    "owner_pid": os.getpid(),
                    "task": name,
                }
            ),
        )
        self.active = name
        return lambda: self._check(name)

    def complete(self, name):
        self._check(name)
        self.verify_files(full=True)
        _write(
            self.root / f"{name}-completed.json", _bytes({"observed_at": host.stamp()})
        )
        self.completed.append(name)
        self.active = None

    def finish(self):
        self._owner()
        if self.closed or self.active is not None or self.completed != list(TASKS):
            raise AdmissionError("whole_study_not_complete")
        self.session_check()
        self.verify_files(full=True)
        _write(
            self.root / "parent-completed.json",
            _bytes(
                {
                    "observed_at": host.stamp(),
                    "completed": self.completed,
                }
            ),
        )
        self.closed = True

    def stop(self, error):
        self.closed = True
        if self.started:
            _write(
                self.root / "parent-failure.json",
                _bytes(
                    {
                        "observed_at": host.stamp(),
                        "error_type": type(error).__name__,
                        "reason": str(error),
                        "completed": self.completed,
                        "active": self.active,
                        "unstarted": [
                            name
                            for name in TASKS
                            if name not in self.completed and name != self.active
                        ],
                    }
                ),
            )


def verify_thermal(value):
    required = (
        "No thermal warning level has been recorded",
        "No performance warning level has been recorded",
        "No CPU power status has been recorded",
    )
    if any(marker not in value for marker in required):
        raise AdmissionError("thermal_state_not_normal")


class MonitorState:
    def __init__(self):
        self.last_good = None
        self.reason = None

    def observe(self):
        self.last_good = time.monotonic()

    def revoke(self, reason):
        if self.reason is None:
            self.reason = reason

    def check(self):
        if self.reason:
            raise AdmissionError(self.reason)
        if self.last_good is None or time.monotonic() - self.last_good > 20:
            raise AdmissionError("session_observation_stale")


@contextmanager
def guarded_session(log_path, *, stable_seconds=30):
    state = MonitorState()
    done = threading.Event()
    inhibitor = subprocess.Popen(
        ["/usr/bin/caffeinate", "-dims", "-w", str(os.getpid())]
    )
    previous = {
        number: signal.getsignal(number) for number in (signal.SIGTERM, signal.SIGINT)
    }
    thread = None
    descriptor = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)

    def interrupted(number, frame):
        state.revoke(state.reason or "operator_or_process_interruption")
        raise AdmissionError(state.reason)

    try:
        for number in previous:
            signal.signal(number, interrupted)
        with os.fdopen(descriptor, "wb") as stream:

            def sample():
                observation = host.capture()
                stream.write(_bytes(observation))
                stream.flush()
                os.fsync(stream.fileno())
                reason = host.violation(observation) or host.inhibition(
                    observation, inhibitor
                )
                if reason:
                    raise AdmissionError(reason)
                verify_thermal(observation["thermal"])
                state.observe()

            time.sleep(1)
            started = time.monotonic()
            while True:
                sample()
                if time.monotonic() - started >= stable_seconds:
                    break
                time.sleep(2)

            def monitor():
                while not done.wait(2):
                    try:
                        sample()
                    except BaseException as error:
                        state.revoke(str(error))
                        os.kill(os.getpid(), signal.SIGTERM)
                        return

            thread = threading.Thread(target=monitor, daemon=True)
            thread.start()
            try:
                yield state.check
                state.check()
            finally:
                done.set()
                thread.join(timeout=90)
                if thread.is_alive():
                    raise AdmissionError("session_monitor_did_not_stop")
    finally:
        done.set()
        if thread is not None and thread.is_alive():
            thread.join(timeout=90)
        for number, handler in previous.items():
            signal.signal(number, handler)
        inhibitor.terminate()
        inhibitor.wait(timeout=10)
