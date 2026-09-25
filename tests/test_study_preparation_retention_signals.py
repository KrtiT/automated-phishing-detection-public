import os
import signal

import pytest
from test_study_preparation_retention import IDENTITY, append_all, module, reserved

from automated_phishing_detection import execution_receipt as receipt


def assert_closed(descriptors):
    leaked = []
    for descriptor in descriptors:
        try:
            os.fstat(descriptor)
        except OSError:
            continue
        os.close(descriptor)
        leaked.append(descriptor)
    assert not leaked, "owned descriptors survived interruption"


def interrupt_open(monkeypatch, name):
    original, descriptors = os.open, []

    def interrupted(path, flags, *arguments, **keywords):
        descriptor = original(path, flags, *arguments, **keywords)
        if path == name:
            descriptors.append(descriptor)
            signal.raise_signal(signal.SIGINT)
        return descriptor

    monkeypatch.setattr(os, "open", interrupted)
    return descriptors


@pytest.mark.parametrize("operation", ["readback", "write_file"])
def test_real_sigint_after_open_closes_owned_file(tmp_path, monkeypatch, operation):
    files = module().files
    attempt = reserved(tmp_path)
    with receipt._directory(attempt.directory) as directory:
        if operation == "readback":
            initial = files.write_file(directory, "invented", b"bytes", 0o600)
        with monkeypatch.context() as guard:
            descriptors = interrupt_open(guard, "invented")
            with pytest.raises(KeyboardInterrupt):
                if operation == "readback":
                    files.readback(directory, "invented", b"bytes", initial)
                else:
                    files.write_file(directory, "invented", b"bytes", 0o600)
        assert len(descriptors) == 1
        assert_closed(descriptors)


def test_real_sigint_during_attempt_entry_closes_directory(tmp_path, monkeypatch):
    retained = module()
    attempt = reserved(tmp_path)
    with monkeypatch.context() as guard:
        descriptors = interrupt_open(guard, attempt.directory.name)
        with pytest.raises(KeyboardInterrupt):
            with retained.retain_study_preparation(attempt, identity=IDENTITY):
                pytest.fail("interrupted entry yielded")
    assert len(descriptors) == 1
    assert_closed(descriptors)


def interrupt_close(monkeypatch, descriptor):
    original, interrupted = os.close, []

    def close(current):
        if current == descriptor and not interrupted:
            interrupted.append(current)
            signal.raise_signal(signal.SIGINT)
        original(current)

    monkeypatch.setattr(os, "close", close)


@pytest.mark.parametrize("operation", ["readback", "write_file"])
def test_real_sigint_before_file_close_still_closes(tmp_path, monkeypatch, operation):
    files = module().files
    attempt = reserved(tmp_path)
    original, descriptors = os.open, []

    def observe(path, flags, *arguments, **keywords):
        descriptor = original(path, flags, *arguments, **keywords)
        if path == "invented":
            descriptors.append(descriptor)
            interrupt_close(monkeypatch, descriptor)
        return descriptor

    with receipt._directory(attempt.directory) as directory:
        if operation == "readback":
            initial = files.write_file(directory, "invented", b"bytes", 0o600)
        with monkeypatch.context() as guard:
            guard.setattr(os, "open", observe)
            with pytest.raises(KeyboardInterrupt):
                if operation == "readback":
                    files.readback(directory, "invented", b"bytes", initial)
                else:
                    files.write_file(directory, "invented", b"bytes", 0o600)
        assert_closed(descriptors)


def test_real_sigint_before_attempt_close_still_closes(tmp_path, monkeypatch):
    retained = module()
    attempt = reserved(tmp_path)
    with pytest.raises(KeyboardInterrupt):
        with retained.retain_study_preparation(attempt, identity=IDENTITY) as writer:
            append_all(writer)
            writer.finish()
            descriptor = writer._directory.descriptor
            interrupt_close(monkeypatch, descriptor)
    assert_closed([descriptor])


@pytest.mark.parametrize("operation", ["check", "authenticate"])
def test_real_sigint_during_owned_metadata_check_closes(
    tmp_path, monkeypatch, operation
):
    files = module().files
    attempt = reserved(tmp_path)
    with receipt._directory(attempt.directory) as directory:
        states = files.authenticate(directory, attempt, IDENTITY)
        name = attempt.directory.name if operation == "check" else "reservation.json"
        with monkeypatch.context() as guard:
            descriptors = interrupt_open(guard, name)
            with pytest.raises(KeyboardInterrupt):
                if operation == "check":
                    files.check(directory, states)
                else:
                    files.authenticate(directory, attempt, IDENTITY)
        assert len(descriptors) == 1
        assert_closed(descriptors)
