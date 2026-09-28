"""Invented approval and actual pipe liveness at private lifecycle boundaries."""

import os
from contextlib import contextmanager
from types import SimpleNamespace

from study_admission_fixtures import command, frame

from automated_phishing_detection._study_admission import consume_child_admission
from automated_phishing_detection._study_child_context import HeldStudyChild


@contextmanager
def admitted(authorization):
    read_fd, write_fd = os.pipe()
    os.set_blocking(read_fd, False)
    admission_frame = frame(
        parent_pid=os.getppid(),
        profile_sha256=authorization.profile_sha256,
        envelope_sha256=authorization.envelope_sha256,
    )
    content = admission_frame.canonical_bytes
    os.write(write_fd, len(content).to_bytes(4, "big") + content)
    child = consume_child_admission(
        "internal", command(), environment={"APD_STUDY_ADMISSION_FD": str(read_fd)}
    )
    state = SimpleNamespace(write_fd=write_fd)
    try:
        yield HeldStudyChild(authorization, child, {}, {}), state
    finally:
        child.close()
        if state.write_fd is not None:
            os.close(state.write_fd)


def invalidate(held, state, boundary):
    if boundary == "envelope":
        held.authorization.envelope_path.write_bytes(b"{}")
    else:
        os.close(state.write_fd)
        state.write_fd = None


def late_check(held, state, events, closed, boundary):
    from automated_phishing_detection._study_child_context import recheck_held_child

    def check():
        if closed in events:
            invalidate(held, state, boundary)
        recheck_held_child(held)

    return check
