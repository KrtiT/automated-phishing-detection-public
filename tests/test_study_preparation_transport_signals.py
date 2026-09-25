import os
import signal

import pytest
from study_preparation_transport_fixtures import (
    inputs,
    module,
    preparation_api,
    preparation_case,
    retained_case,
    runner,
)
from test_study_preparation_retention_signals import assert_closed, interrupt_open

__all__ = ["inputs", "preparation_api", "preparation_case", "retained_case", "runner"]


@pytest.mark.parametrize(
    "name",
    ["directory", "reservation.json", "suffix-rules.dat", "preparation-complete.json"],
)
def test_sigint_after_open_closes_every_owned_descriptor(
    retained_case, monkeypatch, name
):
    retained, api = retained_case, module()
    selected = retained.directory.name if name == "directory" else name
    with monkeypatch.context() as guard:
        descriptors = interrupt_open(guard, selected)
        with pytest.raises(KeyboardInterrupt):
            with api.hold_study_preparation(retained.directory, **retained.expected):
                pytest.fail("interrupted acquisition yielded")
    assert descriptors
    assert_closed(descriptors)


def test_sigint_after_fdopen_closes_stream_and_descriptor(retained_case, monkeypatch):
    retained, api = retained_case, module()
    original, streams = os.fdopen, []

    def interrupted(*arguments, **keywords):
        stream = original(*arguments, **keywords)
        streams.append((stream, stream.fileno()))
        signal.raise_signal(signal.SIGINT)
        return stream

    with monkeypatch.context() as guard:
        guard.setattr(os, "fdopen", interrupted)
        with pytest.raises(KeyboardInterrupt):
            with api.hold_study_preparation(retained.directory, **retained.expected):
                pytest.fail("interrupted stream acquisition yielded")
    assert streams and all(stream.closed for stream, _ in streams)
    assert_closed([descriptor for _, descriptor in streams])


@pytest.mark.parametrize("failure", [KeyboardInterrupt(), SystemExit(9)])
def test_original_interruption_survives_final_state_failure(retained_case, failure):
    retained, api = retained_case, module()
    with pytest.raises(type(failure)) as captured:
        with api.hold_study_preparation(retained.directory, **retained.expected):
            (retained.directory / "suffix-rules.dat").chmod(0o644)
            raise failure
    assert captured.value is failure


def test_final_io_interruption_overrides_ordinary_body_error(
    retained_case, monkeypatch
):
    retained, api = retained_case, module()
    interruption = KeyboardInterrupt()

    def interrupted(*_):
        raise interruption

    with pytest.raises(KeyboardInterrupt) as captured:
        with api.hold_study_preparation(retained.directory, **retained.expected):
            monkeypatch.setattr(api.files, "check", interrupted)
            raise RuntimeError("invented")
    assert captured.value is interruption


def test_sigint_before_held_directory_close_still_releases(retained_case, monkeypatch):
    retained, api = retained_case, module()
    original_open, original_close, descriptors = os.open, os.close, []
    interrupted = []

    def observed(name, flags, *arguments, **keywords):
        descriptor = original_open(name, flags, *arguments, **keywords)
        if name == retained.directory.name and not descriptors:
            descriptors.append(descriptor)
        return descriptor

    def closed(descriptor):
        if descriptor == descriptors[0] and not interrupted:
            interrupted.append(descriptor)
            signal.raise_signal(signal.SIGINT)
        original_close(descriptor)

    with monkeypatch.context() as guard:
        guard.setattr(os, "open", observed)
        with pytest.raises(KeyboardInterrupt):
            with api.hold_study_preparation(retained.directory, **retained.expected):
                guard.setattr(os, "close", closed)
    assert interrupted and descriptors
    assert_closed(descriptors)
