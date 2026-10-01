"""Inherited bounded pipe primitives fail closed for the new protocol."""

import os

import pytest
from study_series_admission_fixtures import api, command, frame
from test_study_admission_rejection import channel


def consume(descriptor, **kwargs):
    return api().consume_series_admission(
        "service",
        command(),
        environment={"APD_STUDY_SERIES_ADMISSION_FD": str(descriptor)},
        **kwargs,
    )


def wire():
    content = frame(parent_pid=os.getppid()).canonical_bytes
    return len(content).to_bytes(4, "big") + content


@pytest.mark.parametrize(
    "content", [b"", b"\0", b"\0\0\0\0", b"\0\0\x10\x01", b"\0\0\0\x04{}"]
)
def test_missing_partial_and_oversized_frames_never_wait(content):
    with channel(content) as descriptor:
        with pytest.raises(ValueError):
            consume(descriptor)


@pytest.mark.parametrize("suffix,closed", [(b"extra", False), (b"", True)])
def test_exact_wire_and_live_writer_required(suffix, closed):
    with channel(wire() + suffix, closed=closed) as descriptor:
        with pytest.raises(ValueError):
            consume(descriptor)


def test_nonblocking_reader_and_separate_control_descriptors_required():
    with channel(wire(), blocking=True) as descriptor:
        with pytest.raises(ValueError):
            consume(descriptor)
    with channel(wire()) as descriptor:
        with pytest.raises(ValueError):
            consume(descriptor, separated_fds=(descriptor,))


def test_distinct_numbers_cannot_alias_one_pipe():
    with channel(wire()) as descriptor:
        duplicate = os.dup(descriptor)
        try:
            with pytest.raises(ValueError):
                consume(descriptor, separated_fds=(duplicate,))
        finally:
            os.close(duplicate)


@pytest.mark.parametrize("descriptor", [-1, 0, 1, 2, "03", "missing", "٣"])
def test_invalid_locators_reject(descriptor):
    with pytest.raises(ValueError):
        consume(descriptor)


def test_regular_file_and_write_end_reject(tmp_path):
    path = tmp_path / "not-a-pipe"
    path.write_bytes(wire())
    with path.open("rb") as stream:
        with pytest.raises(ValueError):
            consume(stream.fileno())
    read_fd, write_fd = os.pipe()
    try:
        os.set_blocking(write_fd, False)
        with pytest.raises(ValueError):
            consume(write_fd)
    finally:
        os.close(read_fd)
        os.close(write_fd)


def test_simultaneous_legacy_locator_never_consumes_either_pipe():
    with channel(wire()) as descriptor:
        environment = dict(
            APD_STUDY_SERIES_ADMISSION_FD=str(descriptor),
            APD_STUDY_ADMISSION_FD=str(descriptor),
        )
        with pytest.raises(ValueError):
            api().consume_series_admission(
                "service", command(), environment=environment
            )
        assert consume(descriptor).frame.cell_ordinal == 73


def test_parent_change_prevents_consumption():
    content = frame(parent_pid=os.getpid()).canonical_bytes
    with channel(len(content).to_bytes(4, "big") + content) as descriptor:
        with pytest.raises(ValueError):
            consume(descriptor)
