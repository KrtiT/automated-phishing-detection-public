"""Independent protocol and descriptor checks with invented admission bytes."""

import os

import pytest
from study_admission_fixtures import frame as original_frame
from study_series_admission_fixtures import api, command, frame
from test_study_admission_rejection import channel

from automated_phishing_detection._study_admission_child import consume_child_admission
from automated_phishing_detection._study_admission_frame import StudyAdmissionError


def wire(content):
    return len(content).to_bytes(4, "big") + content


def consume(descriptor):
    return api().consume_series_admission(
        "service",
        command(),
        environment={"APD_STUDY_SERIES_ADMISSION_FD": str(descriptor)},
    )


def test_deep_json_wire_is_symbolic_and_closes_owned_reader():
    content = b"[" * 1100 + b"0" + b"]" * 1100
    assert len(content) < 4096
    with channel(wire(content)) as descriptor:
        with pytest.raises(StudyAdmissionError, match="invalid_series_admission_frame"):
            consume(descriptor)
        with pytest.raises(OSError):
            os.fstat(descriptor)


@pytest.mark.parametrize("recipient", ("series", "legacy"))
def test_opposite_protocol_rejects_and_closes_owned_reader(recipient):
    producer = original_frame if recipient == "series" else frame
    content = producer(parent_pid=os.getppid()).canonical_bytes
    with channel(wire(content)) as descriptor:
        with pytest.raises(StudyAdmissionError):
            if recipient == "series":
                consume(descriptor)
            else:
                consume_child_admission(
                    "service",
                    command(),
                    environment={"APD_STUDY_ADMISSION_FD": str(descriptor)},
                )
        with pytest.raises(OSError):
            os.fstat(descriptor)


@pytest.mark.parametrize("mismatch", ("role", "command", "parent"))
def test_consume_rejection_closes_verified_reader(mismatch):
    changes = {"parent_pid": os.getppid()}
    changes.update(
        {
            "role": {"role": "client"},
            "command": {"command_sha256": "0" * 64},
            "parent": {"parent_pid": os.getpid()},
        }[mismatch]
    )
    with channel(wire(frame(**changes).canonical_bytes)) as descriptor:
        with pytest.raises(StudyAdmissionError):
            consume(descriptor)
        with pytest.raises(OSError):
            os.fstat(descriptor)


def test_recheck_rejects_replacement_pipe_at_same_descriptor():
    content = wire(frame(parent_pid=os.getppid()).canonical_bytes)
    with channel(content) as descriptor, channel(b"") as replacement:
        admission = consume(descriptor)
        os.dup2(replacement, descriptor)
        with pytest.raises(StudyAdmissionError):
            admission.check()
        admission.close()
        admission.close()


def test_recheck_rejects_reader_changed_to_blocking():
    content = wire(frame(parent_pid=os.getppid()).canonical_bytes)
    with channel(content) as descriptor:
        admission = consume(descriptor)
        os.set_blocking(descriptor, True)
        with pytest.raises(StudyAdmissionError):
            admission.check()
        admission.close()


def test_recheck_rejects_parent_identity_change(monkeypatch):
    content = wire(frame(parent_pid=os.getppid()).canonical_bytes)
    with channel(content) as descriptor:
        admission = consume(descriptor)
        monkeypatch.setattr(os, "getppid", lambda: os.getpid())
        with pytest.raises(
            StudyAdmissionError, match="series_admission_parent_changed"
        ):
            admission.check()
        admission.close()


def test_parent_writer_is_not_inheritable_or_advertised():
    with api().SeriesParentAdmission(frame()) as admission:
        writer = next(
            descriptor
            for descriptor in admission.pipe.descriptors
            if descriptor != admission.read_fd
        )
        assert not os.get_inheritable(writer)
        assert admission.environment == {
            "APD_STUDY_SERIES_ADMISSION_FD": str(admission.read_fd)
        }
        admission.launched(os.getpid() + 100000)
        assert admission.pipe.descriptors == {writer}
        os.fstat(writer)
    with pytest.raises(OSError):
        os.fstat(writer)
