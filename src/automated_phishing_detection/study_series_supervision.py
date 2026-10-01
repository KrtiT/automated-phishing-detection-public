"""Require an actual live physical supervisor before public binding or research."""

import os
import select
import sys
import time
from contextlib import contextmanager

from ._exception_cleanup import CleanupStack
from ._process_support import command_hash
from ._study_admission_io import descriptor_identity, read_frame
from ._study_preparation_files import deferred
from ._study_series_supervisor_channel import (
    HEARTBEAT,
    LOCATOR,
    MAX_AGE_NS,
    MAX_DRAIN_BYTES,
    decode,
    require,
)


def _descriptor():
    value = os.environ.get(LOCATOR)
    require(type(value) is str and value.isascii() and value.isdecimal())
    descriptor = int(value)
    require(str(descriptor) == value)
    return descriptor, descriptor_identity(descriptor)


def _initial(descriptor, arguments):
    require({name for name in os.environ if name.startswith("APD_")} == {LOCATOR})
    require(select.select([descriptor], [], [], 60)[0] == [descriptor])
    value = decode(read_frame(descriptor))
    require(value["parent_pid"] == os.getppid() and value["root_pid"] == os.getpid())
    require(value["command_sha256"] == command_hash((sys.executable, *sys.argv)))
    for name, expected in (
        ("profile_sha256", arguments.expected_profile_sha256),
        ("envelope_sha256", arguments.expected_envelope_sha256),
        ("revision", arguments.expected_revision),
    ):
        require(value[name] == expected)
    return value


class SeriesSupervision:
    def __init__(self, descriptor, identity, frame):
        self.descriptor, self.identity, self.frame = descriptor, identity, frame
        self.sequence, self.latest_ns = 0, frame["monotonic_ns"]
        self.failed = False

    def _drain(self):
        pending, total = bytearray(), 0
        while True:
            try:
                content = os.read(self.descriptor, 4096)
            except BlockingIOError:
                break
            require(bool(content))
            pending.extend(content)
            total += len(content)
            require(total <= MAX_DRAIN_BYTES)
        require(len(pending) % HEARTBEAT.size == 0)
        for offset in range(0, len(pending), HEARTBEAT.size):
            sequence, observed_ns = HEARTBEAT.unpack_from(pending, offset)
            require(sequence == self.sequence + 1 and observed_ns >= self.latest_ns)
            self.sequence, self.latest_ns = sequence, observed_ns

    def check(self):
        try:
            require(not self.failed and self.frame["parent_pid"] == os.getppid())
            require(descriptor_identity(self.descriptor) == self.identity)
            self._drain()
            require(0 <= time.monotonic_ns() - self.latest_ns <= MAX_AGE_NS)
        except BaseException:
            self.failed = True
            raise


@contextmanager
def consume_series_supervision(arguments):
    with CleanupStack() as cleanup:
        descriptor, identity = deferred(_descriptor)
        cleanup.callback(deferred, os.close, descriptor)
        frame = _initial(descriptor, arguments)
        held = SeriesSupervision(descriptor, identity, frame)
        deferred(held.check)
        cleanup.callback(deferred, held.check)
        yield held
