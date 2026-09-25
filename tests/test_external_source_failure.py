"""Private failure state uses retained bytes, never new numerical observations."""

import asyncio
import base64
import importlib
import importlib.util
import json

import pytest
from test_external_source_checkpoints import provenance, science
from test_external_source_checkpoints import writer as writer


def failure_module():
    name = "automated_phishing_detection.external_source_failure"
    assert importlib.util.find_spec(name) is not None, "missing external failure state"
    return importlib.import_module(name)


@pytest.fixture
def state():
    return failure_module().ExternalSourceFailureState()


@pytest.mark.parametrize(
    "original", [KeyboardInterrupt(), SystemExit(7), asyncio.CancelledError()]
)
@pytest.mark.parametrize(
    "later", [OSError("private"), KeyboardInterrupt(), SystemExit(4)]
)
def test_first_body_interruption_survives_later_errors(state, original, later):
    with pytest.raises(type(original)) as caught, state.capture_body():
        raise original
    assert caught.value is original
    assert state.original_error is original
    assert state.body_exited and not state.session_closed
    assert state.selected_error(later) is original


def test_cleanup_interruption_supersedes_ordinary_body_error(state):
    with pytest.raises(ValueError), state.capture_body():
        raise ValueError("private")
    later = KeyboardInterrupt()
    assert state.selected_error(later) is later


def test_progress_is_captured_once_as_immutable_bytes(state, writer):
    original = KeyboardInterrupt("private-message")
    original.progress = b"\x00already retained private bytes\xff"
    with pytest.raises(KeyboardInterrupt), state.capture_body():
        raise original
    original.progress = b"replaced"
    record = json.loads(
        state.snapshot(writer.attempt, writer.identity, "scoring", OSError())
    )
    assert (
        base64.b64decode(record["producer_progress_base64"])
        == b"\x00already retained private bytes\xff"
    )
    assert record["failure"] == "keyboard_interrupt"
    assert record["cleanup_failed"] is True
    assert b"private-message" not in state.snapshot(
        writer.attempt, writer.identity, "scoring", original
    )


def test_missing_progress_does_not_evaluate_exception_property(state, writer):
    class Error(ValueError):
        @property
        def progress(self):
            pytest.fail("failure capture must not query a progress property")

    original = Error()
    with pytest.raises(Error), state.capture_body():
        raise original
    record = json.loads(
        state.snapshot(
            writer.attempt, writer.identity, "source_reconstruction", original
        )
    )
    assert record["producer_progress_base64"] is None
    assert record["completed"] is None
    assert record["checkpoints"] is None
    assert record["cleanup_failed"] is False


def test_writer_snapshot_preserves_pending_and_confirmed_bytes(
    state, writer, monkeypatch
):
    from automated_phishing_detection import _external_checkpoint_io as checkpoint_io

    writer.begin(provenance())
    name, content = next(iter(science().items()))

    def fail(*args):
        raise OSError("private")

    monkeypatch.setattr(checkpoint_io, "append", fail)
    state.writer = writer
    with pytest.raises(ValueError) as caught, state.capture_body():
        writer(name, content)
    record = json.loads(
        state.snapshot(writer.attempt, writer.identity, "scoring", caught.value)
    )
    assert record["protocol_id"] == "external-source-failure-v1"
    assert record["checkpoints"]["protocol_id"] == "external-source-checkpoints-v1"
    assert len(record["checkpoints"]["confirmed_sha256"]) == 6
    assert (
        base64.b64decode(record["checkpoints"]["pending_checkpoint_bytes"][name])
        == content
    )


@pytest.mark.parametrize(
    "error, kind",
    [
        (ValueError("secret"), "execution_failed"),
        (KeyboardInterrupt(), "keyboard_interrupt"),
        (SystemExit(), "system_exit"),
        (asyncio.CancelledError(), "cancelled"),
        (BaseException(), "interrupted"),
    ],
)
def test_symbolic_failure_and_closed_envelope(state, writer, error, kind):
    record = json.loads(
        state.snapshot(writer.attempt, writer.identity, "model_loading", error)
    )
    assert set(record) == {
        "schema_version",
        "protocol_id",
        "status",
        "reservation_sha256",
        "execution",
        "stage",
        "failure",
        "cleanup_failed",
        "checkpoints",
        "producer_progress_base64",
        "completed",
    }
    assert record["failure"] == kind
    assert record["execution"] == writer.identity
    assert record["reservation_sha256"] == writer.attempt.reservation_sha256


@pytest.mark.parametrize("stage", ["../private", "private message", "", 1])
def test_invalid_stage_fails_without_private_diagnostics(state, writer, stage):
    with pytest.raises(ValueError, match="^invalid_external_source_failure$"):
        state.snapshot(writer.attempt, writer.identity, stage, OSError("secret"))
