import os
from contextlib import contextmanager

import pytest
from study_admission_fixtures import command, frame, module


@contextmanager
def channel(content, *, closed=False, blocking=False):
    read_fd, write_fd = os.pipe()
    try:
        os.set_blocking(read_fd, blocking)
        if content:
            os.write(write_fd, content)
        if closed:
            os.close(write_fd)
        yield read_fd
    finally:
        for descriptor in (read_fd, write_fd):
            try:
                os.close(descriptor)
            except OSError:
                pass


def consume(descriptor, **kwargs):
    api = module()
    assert hasattr(api, "consume_child_admission"), "missing bounded child admission"
    return api.consume_child_admission(
        "internal",
        command(),
        environment={"APD_STUDY_ADMISSION_FD": str(descriptor)},
        **kwargs,
    )


def wire():
    content = frame(parent_pid=os.getppid()).canonical_bytes
    return len(content).to_bytes(4, "big") + content


@pytest.mark.parametrize(
    "content", [b"", b"\0", b"\0\0\0\x00", b"\0\0\x10\x01", b"\0\0\0\x04{}"]
)
def test_missing_short_or_oversized_frame_never_waits(content):
    with channel(content) as descriptor:
        with pytest.raises(module().StudyAdmissionError):
            consume(descriptor)


@pytest.mark.parametrize("suffix,closed", [(b"extra", False), (b"", True)])
def test_extra_bytes_or_closed_writer_is_rejected(suffix, closed):
    with channel(wire() + suffix, closed=closed) as descriptor:
        with pytest.raises(module().StudyAdmissionError):
            consume(descriptor)


def test_nonblocking_fifo_and_distinct_service_handles_are_required():
    with channel(wire(), blocking=True) as descriptor:
        with pytest.raises(module().StudyAdmissionError):
            consume(descriptor)
    with channel(wire()) as descriptor:
        with pytest.raises(module().StudyAdmissionError):
            consume(descriptor, separated_fds=(descriptor,))


@pytest.mark.parametrize("descriptor", (-1, 0, 1, 2, "03", "missing"))
def test_invalid_locator_is_rejected(descriptor):
    with pytest.raises(module().StudyAdmissionError):
        consume(descriptor)


def test_regular_file_and_write_end_are_rejected(tmp_path):
    path = tmp_path / "not-a-channel"
    path.write_bytes(wire())
    with path.open("rb") as stream:
        with pytest.raises(module().StudyAdmissionError):
            consume(stream.fileno())
    read_fd, write_fd = os.pipe()
    try:
        os.set_blocking(write_fd, False)
        with pytest.raises(module().StudyAdmissionError):
            consume(write_fd)
    finally:
        os.close(read_fd)
        os.close(write_fd)


def test_child_checks_parent_identity_before_accepting_frame():
    content = frame(parent_pid=os.getpid()).canonical_bytes
    with channel(len(content).to_bytes(4, "big") + content) as descriptor:
        with pytest.raises(module().StudyAdmissionError):
            consume(descriptor)


def test_duplicate_service_handle_identity_is_rejected():
    with channel(wire()) as descriptor:
        duplicate = os.dup(descriptor)
        try:
            with pytest.raises(module().StudyAdmissionError):
                consume(descriptor, separated_fds=(duplicate,))
        finally:
            os.close(duplicate)
