"""Per-role create-only files inside a held, concurrently growing attempt."""

import fcntl
import os
import socket
import stat
from contextlib import contextmanager

from . import _study_preparation_files as files
from ._exception_cleanup import CleanupStack
from ._operational_attempt_io import CLIENT_NAMES, held_attempt_writer, require
from ._operational_attempt_io import OperationalAttemptError as OperationalChildError

__all__ = ["OperationalChildError"]


def owned_names(role):
    require(role in ("service", "client"))
    return (
        ("service-role.json", "service-ready.json", "service-cleanup.json")
        if role == "service"
        else CLIENT_NAMES
    )


def held_writer(attempt, role):
    return held_attempt_writer(attempt, names=owned_names(role))


def _close_handle(owned, descriptor):
    handle = owned.pop(descriptor)
    if handle is None:
        os.close(descriptor)
    else:
        handle.close()


def _handles(cleanup, descriptors, base_url):
    owned = dict.fromkeys(descriptors)
    for descriptor in descriptors:
        cleanup.callback(files.deferred, _close_handle, owned, descriptor)
    listener_fd, stop_fd, ready_fd = descriptors
    listener = socket.socket(fileno=listener_fd)
    owned[listener_fd] = listener
    require(listener.family == socket.AF_INET and listener.type == socket.SOCK_STREAM)
    require(listener.getsockname()[0] == "127.0.0.1")
    require(base_url == f"http://127.0.0.1:{listener.getsockname()[1]}")
    for descriptor, mode in ((stop_fd, os.O_RDONLY), (ready_fd, os.O_WRONLY)):
        require(stat.S_ISFIFO(os.fstat(descriptor).st_mode))
        require(fcntl.fcntl(descriptor, fcntl.F_GETFL) & os.O_ACCMODE == mode)
    return listener, stop_fd, ready_fd


@contextmanager
def service_handles(descriptors, base_url):
    with CleanupStack() as cleanup:
        yield files.deferred(_handles, cleanup, descriptors, base_url)
