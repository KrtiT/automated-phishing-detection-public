"""Observe one owned worker without accepting cached subprocess exit status."""

import json
import os
import signal
import subprocess
import tempfile
from contextlib import ExitStack
from dataclasses import dataclass
from hashlib import sha256

from ._owned_process_exit import OwnedProcessExit, observe_owned_exit
from ._process_support import _defer_interrupt, _InterruptGuard, command_hash
from ._study_admission_parent import validate_launch_admission


class WorkerExecutionError(ValueError):
    def __init__(self, check_id, *, progress=None):
        self.check_id, self.progress = check_id, progress
        super().__init__(check_id)


@dataclass(frozen=True)
class WorkerObservation:
    command_sha256: str
    exit: OwnedProcessExit
    stdout_sha256: str
    stderr_sha256: str


def _stream_hash(stream):
    stream.seek(0)
    digest = sha256()
    while chunk := stream.read(1024 * 1024):
        digest.update(chunk)
    return digest.hexdigest()


class _Worker:
    def __init__(self, admission=None):
        self.stack, self.streams = ExitStack(), ()
        self.admission = admission
        self.process, self.terminal = None, None
        self.stage = "worker_setup_failed"
        self.value = {
            "command_sha256": None,
            "pid": None,
            "exit_observed": False,
            "exit_code": None,
            "stdout_sha256": None,
            "stderr_sha256": None,
            "failure": None,
            "cleanup_failures": [],
            "signals": [],
        }

    def snapshot(self):
        value = self.value.copy()
        if self.process is not None:
            value["pid"] = self.process.pid
        if self.terminal is not None:
            value.update(
                exit_observed=self.terminal.exit_observed,
                exit_code=self.terminal.exit_code,
            )
        return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("ascii")

    def interruption_progress(self):
        self.value["failure"] = "parent_interrupted"
        return self.snapshot()

    def observe(self, *, block):
        """Disarm Python 3.10's destructor after losing child ownership."""
        if self.terminal is None:
            self.terminal = observe_owned_exit(self.process, block=block)
        if self.terminal is not None and not self.terminal.exit_observed:
            self.process._child_created = False
        if self.terminal is not None and self.admission is not None:
            self.admission.observed(
                self.terminal.exit_observed, self.terminal.exit_code
            )
        return self.terminal

    def run(self, command):
        self.value["command_sha256"] = command_hash(command)
        with _defer_interrupt():
            if self.admission is not None:
                validate_launch_admission(
                    self.admission, ("internal", "external"), command
                )
                self.stack.enter_context(self.admission)
            self.streams = tuple(
                self.stack.enter_context(tempfile.TemporaryFile()) for _ in range(2)
            )
        self.stage = "worker_launch_failed"
        self.launch(command)
        self.stage = "worker_wait_failed"
        if self.observe(block=True) is None:
            raise ValueError("worker_exit_pending")
        self.stage = "worker_diagnostics_failed"
        self.value["stdout_sha256"] = _stream_hash(self.streams[0])
        self.value["stderr_sha256"] = _stream_hash(self.streams[1])
        return WorkerObservation(
            self.value["command_sha256"],
            self.terminal,
            self.value["stdout_sha256"],
            self.value["stderr_sha256"],
        )

    def launch(self, command):
        options = {}
        if self.admission is not None:
            environment = {
                key: value
                for key, value in os.environ.items()
                if not key.startswith("APD_")
            }
            environment.update(self.admission.environment)
            options = dict(env=environment, pass_fds=(self.admission.read_fd,))
        with _defer_interrupt():
            self.process = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=self.streams[0],
                stderr=self.streams[1],
                close_fds=True,
                **options,
            )
            if self.admission is not None:
                self.admission.launched(self.process.pid)

    def stop(self):
        if self.process is None:
            return
        with _defer_interrupt():
            if self.observe(block=False) is not None:
                return
            self.value["signals"].append(int(signal.SIGKILL))
            try:
                os.kill(self.process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            self.observe(block=True)

    def finish(self, failure, interrupts):
        for action in (self.stop, self.stack.close):
            try:
                action()
            except BaseException as error:
                interrupts.record(error)
                self.value["cleanup_failures"].append("worker_cleanup_failed")
                if failure is None or (
                    isinstance(failure, Exception) and not isinstance(error, Exception)
                ):
                    failure = error
                    self.value["failure"] = (
                        "worker_cleanup_failed"
                        if isinstance(error, Exception)
                        else "parent_interrupted"
                    )
        return failure


def observe_worker(command: tuple[str, ...]) -> WorkerObservation:
    return _observe_owned_worker(command, _Worker())


def _observe_study_worker(command, admission):
    return _observe_owned_worker(command, _Worker(admission))


def _observe_owned_worker(command, worker):
    failure = None
    with _InterruptGuard(worker.interruption_progress) as interrupts:
        try:
            observed = interrupts.run(worker.run, command)
        except BaseException as error:
            failure = error
            worker.value["failure"] = (
                worker.stage if isinstance(error, Exception) else "parent_interrupted"
            )
        failure = worker.finish(failure, interrupts)
        failure = interrupts.select(failure)
        if failure is not None:
            if isinstance(failure, Exception):
                raise WorkerExecutionError(
                    worker.value["failure"], progress=worker.snapshot()
                ) from None
            try:
                failure.progress = worker.interruption_progress()
            except BaseException:
                pass
            raise failure from None
        return observed
