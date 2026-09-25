import os
import signal


def assert_closed(descriptors):
    leaked = []
    for descriptor in set(descriptors):
        try:
            os.fstat(descriptor)
        except OSError:
            continue
        os.close(descriptor)
        leaked.append(descriptor)
    assert not leaked, "owned descriptors survived interruption"


def watch_open(monkeypatch, selected, *, interrupt=False, before_close=False):
    original_open, original_close = os.open, os.close
    descriptors, signaled = [], []

    def opened(path, flags, *arguments, **keywords):
        descriptor = original_open(path, flags, *arguments, **keywords)
        if path == selected:
            descriptors.append(descriptor)
            if interrupt and not signaled:
                signaled.append(True)
                signal.raise_signal(signal.SIGINT)
        return descriptor

    def close(descriptor):
        if (
            before_close
            and descriptors
            and descriptor == descriptors[0]
            and not signaled
        ):
            signaled.append(True)
            signal.raise_signal(signal.SIGINT)
        original_close(descriptor)

    monkeypatch.setattr(os, "open", opened)
    monkeypatch.setattr(os, "close", close)
    return descriptors


def interrupt_fdopen(monkeypatch):
    original, descriptors, signaled = os.fdopen, [], []

    def opened(descriptor, *arguments, **keywords):
        stream = original(descriptor, *arguments, **keywords)
        descriptors.append(descriptor)
        if not signaled:
            signaled.append(True)
            signal.raise_signal(signal.SIGINT)
        return stream

    monkeypatch.setattr(os, "fdopen", opened)
    return descriptors
