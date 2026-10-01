"""Small live supervisor frames; retained metadata cannot replace the pipe."""

import json
import os
import re
import struct
import time
from contextlib import contextmanager
from dataclasses import dataclass

from ._exception_cleanup import preserve_cleanup
from ._process_support import _defer_interrupt, command_hash

LOCATOR = "APD_STUDY_SERIES_SUPERVISION_FD"
PROTOCOL = "study-series-supervision-v1"
HEARTBEAT = struct.Struct("!QQ")
MAX_AGE_NS = 75_000_000_000
MAX_DRAIN_BYTES = 65536
FIELDS = {
    "protocol",
    "parent_pid",
    "root_pid",
    "profile_sha256",
    "envelope_sha256",
    "revision",
    "command_sha256",
    "monotonic_ns",
}


def require(condition):
    if not condition:
        raise ValueError("invalid_series_supervision")


def encoded(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("ascii")


def admission(arguments, root_pid, command, *, observed_ns=None):
    return dict(
        protocol=PROTOCOL,
        parent_pid=os.getpid(),
        root_pid=root_pid,
        profile_sha256=arguments.expected_profile_sha256,
        envelope_sha256=arguments.expected_envelope_sha256,
        revision=arguments.expected_revision,
        command_sha256=command_hash(command),
        monotonic_ns=time.monotonic_ns() if observed_ns is None else observed_ns,
    )


def decode(content):
    require(type(content) is bytes and 0 < len(content) <= 4096)
    value = json.loads(content)
    require(type(value) is dict and set(value) == FIELDS)
    require(encoded(value) == content and value["protocol"] == PROTOCOL)
    for name in ("parent_pid", "root_pid", "monotonic_ns"):
        require(type(value[name]) is int and value[name] > 0)
    require(value["parent_pid"] != value["root_pid"])
    for name in ("profile_sha256", "envelope_sha256", "command_sha256", "revision"):
        size = 40 if name == "revision" else 64
        require(
            type(value[name]) is str
            and re.fullmatch(rf"[0-9a-f]{{{size}}}", value[name])
        )
    return value


def _write(channel, content):
    require(channel.write_fd is not None)
    try:
        require(len(content) <= os.fpathconf(channel.write_fd, "PC_PIPE_BUF"))
        require(os.write(channel.write_fd, content) == len(content))
    except OSError:
        raise ValueError("series_supervision_delivery_failed") from None


@dataclass
class SupervisorChannel:
    read_fd: int | None = None
    write_fd: int | None = None
    sequence: int = 0
    released: bool = False

    def release(self, frame):
        require(not self.released)
        content = encoded(frame)
        require(len(content) <= 4096)
        _write(self, len(content).to_bytes(4, "big") + content)
        self.released = True

    def heartbeat(self, observed_ns):
        require(self.released and type(observed_ns) is int and observed_ns > 0)
        _write(self, HEARTBEAT.pack(self.sequence + 1, observed_ns))
        self.sequence += 1

    def launched(self):
        with _defer_interrupt():
            descriptor, self.read_fd = self.read_fd, None
            if descriptor is not None:
                os.close(descriptor)

    def revoke(self):
        with _defer_interrupt():
            descriptor, self.write_fd = self.write_fd, None
            if descriptor is not None:
                os.close(descriptor)

    def close(self):
        with preserve_cleanup(self.revoke):
            self.launched()


def _open():
    channel = SupervisorChannel()
    try:
        with _defer_interrupt():
            channel.read_fd, channel.write_fd = os.pipe()
            os.set_blocking(channel.read_fd, False)
            os.set_blocking(channel.write_fd, False)
        return channel
    except BaseException:
        with preserve_cleanup(channel.close):
            raise


@contextmanager
def supervisor_channel():
    channel = _open()
    with preserve_cleanup(channel.close):
        yield channel
