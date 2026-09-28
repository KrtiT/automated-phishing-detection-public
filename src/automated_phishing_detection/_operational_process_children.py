"""Own one service/client pair and close only their descriptors and processes."""

import asyncio
import os
import signal
import socket
import subprocess
import tempfile

from ._exception_cleanup import CleanupStack, preserve_cleanup
from ._operational_process_records import OperationalProcessError, command_hash
from ._process_support import _defer_interrupt
from ._study_admission_parent import StudyAdmissions, validate_launch_admission


class OwnedChildren:
    def __init__(
        self, observations, deadlines, study_admissions: StudyAdmissions | None = None
    ):
        self.observations, self.deadlines = observations, deadlines
        self.stack = CleanupStack()
        self.descriptors = set()
        self.processes, self.streams = {}, {}
        self.study_admissions, self.admissions = study_admissions, {}
        self.admission_failure = None

    def __enter__(self):
        try:
            with _defer_interrupt():
                self.listener = self.stack.enter_context(socket.socket())
                self.listener.bind(("127.0.0.1", 0))
                self.port = self.listener.getsockname()[1]
                self.stop_read, self.stop_write = self._pipe()
                self.ready_read, self.ready_write = self._pipe()
                os.set_blocking(self.ready_read, False)
            return self
        except BaseException as error:
            self.observations.fail("process_setup_failed")
            selected = (
                OperationalProcessError("process_setup_failed")
                if isinstance(error, Exception)
                else error
            )
            selected.progress = self.observations.snapshot()
            with preserve_cleanup(self.stack.close):
                raise selected from None

    def __exit__(self, *args):
        return self.stack.__exit__(*args)

    def _pipe(self):
        descriptors = os.pipe()
        for descriptor in descriptors:
            self.descriptors.add(descriptor)
            self.stack.callback(self.close_fd, descriptor)
        return descriptors

    def close_fd(self, descriptor):
        with _defer_interrupt():
            if descriptor in self.descriptors:
                self.descriptors.remove(descriptor)
                os.close(descriptor)

    def _environment(self, role):
        environment = {
            key: value
            for key, value in os.environ.items()
            if not key.startswith("APD_")
        }
        environment.update(
            APD_BASE_URL=f"http://127.0.0.1:{self.port}",
            APD_ATTEMPT_DIRECTORY=str(self.observations.attempt.directory),
            APD_RESERVATION_SHA256=self.observations.attempt.reservation_sha256,
        )
        if role == "service":
            environment.update(
                APD_LISTENER_FD=str(self.listener.fileno()),
                APD_STOP_FD=str(self.stop_read),
                APD_READY_FD=str(self.ready_write),
            )
        return environment

    def launch(self, role, command):
        self.observations.install(
            f"{role}-intent.json", {"command_sha256": command_hash(command)}
        )
        streams = tuple(self._stream() for _ in range(2))
        descriptors, environment = self._launch_options(role, command)
        with _defer_interrupt():
            process = subprocess.Popen(
                command,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=streams[0],
                stderr=streams[1],
                pass_fds=descriptors,
                close_fds=True,
            )
            self.processes[role], self.streams[role] = process, streams
            self.observations.value[role]["pid"] = process.pid
            if role in self.admissions:
                self.admissions[role].launched(process.pid)
        if role == "service":
            self.listener.close()
            self.close_fd(self.stop_read)
            self.close_fd(self.ready_write)
        self.observations.install(f"{role}-started.json", {"pid": process.pid})

    def _launch_options(self, role, command):
        descriptors = (
            (self.listener.fileno(), self.stop_read, self.ready_write)
            if role == "service"
            else ()
        )
        environment = self._environment(role)
        if self.study_admissions is not None:
            with _defer_interrupt():
                admission = self.study_admissions(role, command)
                validate_launch_admission(admission, (role,), command)
                self.admissions[role] = self.stack.enter_context(admission)
            descriptors += (admission.read_fd,)
            environment.update(admission.environment)
        return descriptors, environment

    def _stream(self):
        with _defer_interrupt():
            return self.stack.enter_context(tempfile.TemporaryFile())

    async def wait_exit(self, role, timeout):
        async def wait():
            while not self.exited(role):
                await asyncio.sleep(0.01)

        await asyncio.wait_for(wait(), timeout)

    def exited(self, role):
        with _defer_interrupt():
            exited = self.observations.exited(role, self.processes[role])
            if exited and role in self.admissions:
                self._record_admission_exit(role)
            return exited

    def _record_admission_exit(self, role):
        observed = self.observations.value[role]
        try:
            self.admissions[role].observed(
                observed["exit_observed"], observed["exit_code"]
            )
        except BaseException as error:
            if self.admission_failure is None or (
                isinstance(self.admission_failure, Exception)
                and not isinstance(error, Exception)
            ):
                self.admission_failure = error
            self.observations.fail(
                "admission_observation_failed"
                if isinstance(error, Exception)
                else "parent_interrupted"
            )

    async def force(self, role):
        process = self.processes[role]
        for number, limit in (
            (signal.SIGTERM, self.deadlines["terminate"]),
            (signal.SIGKILL, self.deadlines["kill"]),
        ):
            if self.exited(role):
                return
            observed = self.observations.value[role]
            observed["forced"] = True
            observed["signals"].append(int(number))
            try:
                os.kill(process.pid, number)
            except ProcessLookupError:
                pass
            try:
                await self.wait_exit(role, limit)
                return
            except asyncio.TimeoutError:
                pass
        self.observations.fail(f"{role}_exit_unobserved")

    async def cleanup(self):
        for role in ("client", "service"):
            try:
                if role == "service":
                    await self.stop_service()
                elif role in self.processes and not self.exited(role):
                    await self.force(role)
            except Exception:
                self.observations.fail(f"{role}_cleanup_failed")

    async def stop_service(self):
        if "service" not in self.processes:
            return
        if not self.exited("service"):
            self.observations.try_install("service-stop.json", {"status": "requested"})
            try:
                self.observations.value["stop_sent"] = (
                    os.write(self.stop_write, b"stop\n") == 5
                )
            except OSError:
                self.observations.fail("stop_control_failed")
            self.close_fd(self.stop_write)
            try:
                await self.wait_exit("service", self.deadlines["shutdown"])
            except asyncio.TimeoutError:
                await self.force("service")
