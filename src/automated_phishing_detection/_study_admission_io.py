"""Bounded nonblocking pipe ownership and reads for study admissions."""

import fcntl
import os
import stat

from ._exception_cleanup import CleanupStack, preserve_cleanup
from ._process_support import _defer_interrupt
from ._study_admission_frame import StudyAdmissionError

LOCATOR = "APD_STUDY_ADMISSION_FD"


class AdmissionPipe:
    def __init__(self):
        self.stack, self.descriptors = CleanupStack(), set()

    def open(self, content):
        try:
            with _defer_interrupt():
                read_fd, write_fd = os.pipe()
                for descriptor in (read_fd, write_fd):
                    self.descriptors.add(descriptor)
                    self.stack.callback(self.close_fd, descriptor)
                for descriptor in (read_fd, write_fd):
                    os.set_blocking(descriptor, False)
                wire = len(content).to_bytes(4, "big") + content
                if os.write(write_fd, wire) != len(wire):
                    raise StudyAdmissionError("admission_preload_failed")
            return read_fd, write_fd
        except BaseException:
            with preserve_cleanup(self.close):
                raise

    def close_fd(self, descriptor):
        with _defer_interrupt():
            if descriptor in self.descriptors:
                self.descriptors.remove(descriptor)
                os.close(descriptor)

    def close(self):
        self.stack.close()


def descriptor_identity(descriptor, separated_fds=()):
    if type(descriptor) is not int or descriptor < 3 or descriptor in separated_fds:
        raise StudyAdmissionError("invalid_admission_descriptor")
    try:
        status = os.fstat(descriptor)
        flags = fcntl.fcntl(descriptor, fcntl.F_GETFL)
        valid = stat.S_ISFIFO(status.st_mode) and flags & os.O_ACCMODE == os.O_RDONLY
        if not valid or not flags & os.O_NONBLOCK:
            raise StudyAdmissionError("invalid_admission_descriptor")
        identity = (status.st_dev, status.st_ino)
        for other in separated_fds:
            other_status = os.fstat(other)
            if identity == (other_status.st_dev, other_status.st_ino):
                raise StudyAdmissionError("invalid_admission_descriptor")
        return identity
    except OSError:
        raise StudyAdmissionError("invalid_admission_descriptor") from None


def read_exact(descriptor, size):
    content = bytearray()
    for _ in range(size):
        try:
            chunk = os.read(descriptor, size - len(content))
        except OSError:
            raise StudyAdmissionError("incomplete_admission_frame") from None
        if not chunk:
            raise StudyAdmissionError("incomplete_admission_frame")
        content.extend(chunk)
        if len(content) == size:
            return bytes(content)
    raise StudyAdmissionError("incomplete_admission_frame")


def check_writer(descriptor):
    try:
        extra = os.read(descriptor, 1)
    except BlockingIOError:
        return
    except OSError:
        raise StudyAdmissionError("invalid_admission_descriptor") from None
    raise StudyAdmissionError("extra_admission_bytes" if extra else "admission_closed")


def read_frame(descriptor):
    size = int.from_bytes(read_exact(descriptor, 4), "big")
    if not 0 < size <= 4096:
        raise StudyAdmissionError("invalid_admission_length")
    return read_exact(descriptor, size)
