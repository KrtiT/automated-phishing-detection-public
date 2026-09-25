"""Small shared primitives for private command identity and resource assignment."""

import json
import signal
import sys
import threading
from contextlib import contextmanager
from hashlib import sha256


def command_hash(command):
    content = json.dumps(list(command), ensure_ascii=True, separators=(",", ":"))
    return sha256(content.encode("ascii")).hexdigest()


@contextmanager
def _defer_interrupt():
    handler = signal.getsignal(signal.SIGINT)
    received = []
    if threading.current_thread() is not threading.main_thread() or not callable(
        handler
    ):
        yield received
        return
    signal.signal(signal.SIGINT, lambda number, frame: received.append((number, frame)))
    try:
        yield received
    finally:
        signal.signal(signal.SIGINT, handler)
        if received:
            handler(*received[0])


class _InterruptGuard:
    """Keep the first interruption while an owner completes resource cleanup."""

    def __init__(self, progress):
        self.progress = progress
        self.first, self.running, self.handling = None, False, False

    def __enter__(self):
        self.handler = signal.getsignal(signal.SIGINT)
        self.enabled = (
            threading.current_thread() is threading.main_thread()
            and callable(self.handler)
        )
        if self.enabled:
            signal.signal(signal.SIGINT, self._receive)
        return self

    def __exit__(self, exception_type, error, traceback):
        try:
            if self.enabled:
                signal.signal(signal.SIGINT, self.handler)
            selected = self.select(error)
            if selected is not error and selected is not None:
                raise selected from None
        except BaseException as failure:
            selected = self.select(failure)
            try:
                selected.progress = self.progress()
            except BaseException:
                pass
            raise selected from None

    def select(self, error):
        return self.first if self.first is not None else error

    def record(self, error):
        if self.first is None and not isinstance(error, Exception):
            self.first = error

    def _receive(self, number, frame):
        active = sys.exc_info()[1]
        if (
            self.first is None
            and active is not None
            and not isinstance(active, Exception)
        ):
            self.first = active
        if self.first is not None or self.handling:
            return
        self.handling = True
        try:
            try:
                self.handler(number, frame)
            except BaseException as error:
                self.first = error
                if self.running:
                    raise
        finally:
            self.handling = False

    def run(self, function, *arguments):
        self.running = True
        try:
            if self.first is not None:
                raise self.first
            return function(*arguments)
        except BaseException as error:
            self.record(error)
            raise
        finally:
            self.running = False
