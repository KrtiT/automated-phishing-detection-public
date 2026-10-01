"""Actual live descriptors and signals cross the composed child admission path."""

import os
import signal
from contextlib import contextmanager

import pytest
from study_series_child_context_fixtures import change_frame, setup
from study_series_child_context_pipe_fixtures import live_pipe
from study_series_execution_fixtures import public_case

__all__ = ["public_case"]


def test_actual_pipe_is_held_through_prefix_and_closed_at_exit(
    public_case, monkeypatch
):
    case = setup(public_case, monkeypatch)
    with live_pipe(case, monkeypatch) as (descriptor, unused, channel):
        with case.module.held_authorization(case.arguments) as held:
            assert held.admission.read_fd == descriptor
            held.admission.check()
        with pytest.raises(OSError):
            os.fstat(descriptor)


@pytest.mark.parametrize("change", ("parent", "command"))
def test_real_wrong_parent_or_command_prevents_public_binding(
    public_case, monkeypatch, change
):
    case = setup(public_case, monkeypatch)
    change_frame(
        case,
        **(
            {"parent_pid": os.getpid()}
            if change == "parent"
            else {"command_sha256": "f" * 64}
        ),
    )
    with live_pipe(case, monkeypatch) as (descriptor, unused, channel):
        with pytest.raises(ValueError):
            with case.module.held_authorization(case.arguments):
                pytest.fail("wrong actual context admitted")
        with pytest.raises(OSError):
            os.fstat(descriptor)
    assert "bind" not in case.events and "prefix" not in case.events


def test_writer_death_during_public_binding_prevents_prefix(public_case, monkeypatch):
    case = setup(public_case, monkeypatch)
    with live_pipe(case, monkeypatch) as (descriptor, writer, channel):

        def bind(*args, **kwargs):
            channel.close_fd(writer)
            return case.binding

        monkeypatch.setattr(case.module, "bind_series_public_execution", bind)
        with pytest.raises(ValueError, match="admission_closed"):
            with case.module.held_authorization(case.arguments):
                pytest.fail("orphan reached private inputs")
        with pytest.raises(OSError):
            os.fstat(descriptor)
    assert "prefix" not in case.events


def test_real_sigint_after_consume_registers_and_closes_owned_fd(
    public_case, monkeypatch
):
    case = setup(public_case, monkeypatch)
    with live_pipe(case, monkeypatch) as (descriptor, unused, channel):
        original = case.module.consume_series_admission

        def consume(*args, **kwargs):
            child = original(*args, **kwargs)
            signal.raise_signal(signal.SIGINT)
            return child

        monkeypatch.setattr(case.module, "consume_series_admission", consume)
        with pytest.raises(KeyboardInterrupt):
            with case.module.held_authorization(case.arguments):
                pytest.fail("interrupted acquisition yielded")
        with pytest.raises(OSError):
            os.fstat(descriptor)
    assert "bind" not in case.events


def test_prefix_cleanup_interruption_preserves_body_and_closes_pipe(
    public_case, monkeypatch
):
    case = setup(public_case, monkeypatch)
    original = SystemExit(7)
    original.progress = b"invented original evidence"

    @contextmanager
    def prefix(*args):
        try:
            yield ()
        finally:
            signal.raise_signal(signal.SIGINT)

    monkeypatch.setattr(case.module, "hold_series_child_prefix", prefix)
    with live_pipe(case, monkeypatch) as (descriptor, unused, channel):
        with pytest.raises(SystemExit) as caught:
            with case.module.held_authorization(case.arguments):
                raise original
        assert caught.value is original
        with pytest.raises(OSError):
            os.fstat(descriptor)


def test_admission_fd_may_not_alias_a_service_control_pipe(public_case, monkeypatch):
    case = setup(public_case, monkeypatch, "service")
    with live_pipe(case, monkeypatch) as (descriptor, unused, channel):
        monkeypatch.setenv("APD_STOP_FD", str(descriptor))
        with pytest.raises(ValueError):
            with case.module.held_authorization(case.arguments):
                pytest.fail("aliased control admitted")
    assert "bind" not in case.events
