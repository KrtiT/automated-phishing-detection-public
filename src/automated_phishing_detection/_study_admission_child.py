"""Consume a complete live frame before opening any study input."""

import os

from ._exception_cleanup import preserve_cleanup
from ._process_support import _defer_interrupt, command_hash
from ._study_admission_frame import StudyAdmissionError, decode_admission_frame
from ._study_admission_io import (
    LOCATOR,
    check_writer,
    descriptor_identity,
    read_frame,
)


class ChildAdmission:
    def __init__(self, descriptor, frame, identity):
        self.read_fd, self.frame, self.identity = descriptor, frame, identity

    def check(self):
        if self.frame.parent_pid != os.getppid():
            raise StudyAdmissionError("admission_parent_changed")
        if descriptor_identity(self.read_fd) != self.identity:
            raise StudyAdmissionError("invalid_admission_descriptor")
        check_writer(self.read_fd)

    def close(self):
        with _defer_interrupt():
            descriptor, self.read_fd = self.read_fd, None
            if descriptor is not None:
                os.close(descriptor)


def _locator(environment):
    value = environment.get(LOCATOR)
    if type(value) is not str or not value.isascii() or not value.isdecimal():
        raise StudyAdmissionError("invalid_admission_locator")
    descriptor = int(value)
    if str(descriptor) != value:
        raise StudyAdmissionError("invalid_admission_locator")
    return descriptor


def consume_child_admission(role, command, *, environment=None, separated_fds=()):
    environment = os.environ if environment is None else environment
    descriptor = _locator(environment)
    identity = descriptor_identity(descriptor, separated_fds)
    try:
        frame = decode_admission_frame(read_frame(descriptor))
        if frame.role != role or frame.command_sha256 != command_hash(command):
            raise StudyAdmissionError("admission_command_mismatch")
        admission = ChildAdmission(descriptor, frame, identity)
        admission.check()
        return admission
    except BaseException:
        with preserve_cleanup(lambda: os.close(descriptor)):
            raise
