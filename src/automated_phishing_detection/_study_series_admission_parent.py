"""Keep a separate series writer alive through actual owned child cleanup."""

import os

from ._exception_cleanup import preserve_cleanup
from ._process_support import _defer_interrupt, command_hash
from ._study_admission_io import AdmissionPipe
from ._study_series_admission_frame import require, validate_series_frame

LOCATOR = "APD_STUDY_SERIES_ADMISSION_FD"


class SeriesParentAdmission:
    def __init__(self, frame, *, on_launched=None, on_observed=None):
        validate_series_frame(frame)
        for callback in (on_launched, on_observed):
            require(callback is None or callable(callback))
        self.frame = frame
        self.on_launched, self.on_observed = on_launched, on_observed
        self.pipe = AdmissionPipe()
        self.read_fd, self.environment = None, {}
        self.pid, self.exit_observed, self.exit_code = None, False, None
        self.entered, self.recorded = False, False

    def __enter__(self):
        require(not self.entered and self.frame.parent_pid == os.getpid())
        self.entered = True
        try:
            with _defer_interrupt():
                self.read_fd, unused = self.pipe.open(self.frame.canonical_bytes)
                self.environment = {LOCATOR: str(self.read_fd)}
            return self
        except BaseException:
            with preserve_cleanup(self.pipe.close):
                raise

    def __exit__(self, *arguments):
        return self.pipe.stack.__exit__(*arguments)

    def launched(self, pid):
        require(self.entered and self.pid is None)
        require(self.read_fd in self.pipe.descriptors)
        require(type(pid) is int and pid > 0 and pid != self.frame.parent_pid)
        self.pid = pid
        self.pipe.close_fd(self.read_fd)
        if self.on_launched is not None:
            self.on_launched(pid)

    def observed(self, exit_observed, exit_code):
        if self.recorded:
            return
        require(self.pid is not None and type(exit_observed) is bool)
        require(type(exit_code) is int if exit_observed else exit_code is None)
        self.recorded, self.exit_observed, self.exit_code = (
            True,
            exit_observed,
            exit_code,
        )
        if self.on_observed is not None:
            self.on_observed(exit_observed, exit_code)


def validate_series_launch(admission, role, command):
    require(type(admission) is SeriesParentAdmission)
    validate_series_frame(admission.frame)
    require(admission.frame.role == role)
    require(admission.frame.command_sha256 == command_hash(command))
