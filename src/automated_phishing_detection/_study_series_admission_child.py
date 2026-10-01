"""Consume a live series channel before any historical or model access."""

import os

from ._exception_cleanup import preserve_cleanup
from ._process_support import _defer_interrupt, command_hash
from ._study_admission_io import (
    LOCATOR as LEGACY_LOCATOR,
)
from ._study_admission_io import check_writer, descriptor_identity, read_frame
from ._study_series_admission_frame import decode_series_admission, require
from ._study_series_admission_parent import LOCATOR


class SeriesChildAdmission:
    def __init__(self, descriptor, frame, identity):
        self.read_fd, self.frame, self.identity = descriptor, frame, identity

    def check(self):
        require(
            self.frame.parent_pid == os.getppid(), "series_admission_parent_changed"
        )
        require(descriptor_identity(self.read_fd) == self.identity)
        check_writer(self.read_fd)

    def close(self):
        with _defer_interrupt():
            descriptor, self.read_fd = self.read_fd, None
            if descriptor is not None:
                os.close(descriptor)


def _locator(environment):
    require(LEGACY_LOCATOR not in environment)
    value = environment.get(LOCATOR)
    require(type(value) is str and value.isascii() and value.isdecimal())
    descriptor = int(value)
    require(str(descriptor) == value)
    return descriptor


def consume_series_admission(role, command, *, environment=None, separated_fds=()):
    environment = os.environ if environment is None else environment
    descriptor = _locator(environment)
    identity = descriptor_identity(descriptor, separated_fds)
    try:
        frame = decode_series_admission(read_frame(descriptor))
        require(frame.role == role and frame.command_sha256 == command_hash(command))
        admission = SeriesChildAdmission(descriptor, frame, identity)
        admission.check()
        return admission
    except BaseException:
        with preserve_cleanup(lambda: os.close(descriptor)):
            raise
