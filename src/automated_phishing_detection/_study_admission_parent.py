"""Keep a study admission writer alive through owned child cleanup."""

import os
from collections.abc import Callable

from ._exception_cleanup import preserve_cleanup
from ._process_support import _defer_interrupt, command_hash
from ._study_admission_frame import (
    AdmissionFrame,
    StudyAdmissionError,
    validate_admission_frame,
)
from ._study_admission_io import LOCATOR, AdmissionPipe


class ParentAdmission:
    def __init__(
        self,
        frame: AdmissionFrame,
        *,
        on_launched: Callable[[int], None] | None = None,
        on_observed: Callable[[bool, int | None], None] | None = None,
    ):
        validate_admission_frame(frame)
        for callback in (on_launched, on_observed):
            if callback is not None and not callable(callback):
                raise StudyAdmissionError("invalid_admission_callback")
        self.frame = frame
        self.on_launched, self.on_observed = on_launched, on_observed
        self.pipe = AdmissionPipe()
        self.read_fd, self.environment = None, {}
        self.pid, self.exit_observed, self.exit_code = None, False, None
        self.entered, self.recorded = False, False

    def __enter__(self):
        if self.entered or self.frame.parent_pid != os.getpid():
            raise StudyAdmissionError("invalid_admission_parent")
        self.entered = True
        try:
            with _defer_interrupt():
                self.read_fd, _ = self.pipe.open(self.frame.canonical_bytes)
                self.environment = {LOCATOR: str(self.read_fd)}
            return self
        except BaseException:
            with preserve_cleanup(self.pipe.close):
                raise

    def __exit__(self, *arguments):
        return self.pipe.stack.__exit__(*arguments)

    def launched(self, pid):
        if self.pid is not None or type(pid) is not int or pid <= 0 or not self.entered:
            raise StudyAdmissionError("invalid_admission_launch")
        self.pid = pid
        self.pipe.close_fd(self.read_fd)
        if self.on_launched is not None:
            self.on_launched(pid)

    def observed(self, exit_observed, exit_code):
        if self.recorded:
            return
        if self.pid is None or type(exit_observed) is not bool:
            raise StudyAdmissionError("invalid_admission_observation")
        valid = type(exit_code) is int if exit_observed else exit_code is None
        if not valid:
            raise StudyAdmissionError("invalid_admission_observation")
        self.recorded, self.exit_observed, self.exit_code = (
            True,
            exit_observed,
            exit_code,
        )
        if self.on_observed is not None:
            self.on_observed(exit_observed, exit_code)


def validate_launch_admission(admission, roles, command):
    if (
        type(admission) is not ParentAdmission
        or admission.frame.role not in roles
        or admission.frame.command_sha256 != command_hash(command)
    ):
        raise StudyAdmissionError("admission_command_mismatch")


StudyAdmissions = Callable[[str, tuple[str, ...]], ParentAdmission]
