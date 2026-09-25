"""Pure restoration stays interruptible while actual retained handles are owned."""

import os
import signal
from contextlib import contextmanager, nullcontext

import pytest
from operational_bound_child_fixtures import child_case, keywords
from operational_child_fixtures import child_context
from operational_service_child_fixtures import inherited
from operational_transport_integration_fixtures import (
    accepted,
    candidates,
    case,
    manifests,
)

from automated_phishing_detection import operational_input_transport as transport

__all__ = ["accepted", "candidates", "case", "manifests", "child_context"]


@pytest.mark.parametrize("role", ["service", "client"])
def test_signal_inside_pure_restore_is_immediate_and_closes_handles(
    child_context, accepted, case, tmp_path, monkeypatch, role
):
    fixture = child_case(tmp_path, accepted, case, monkeypatch)
    monkeypatch.setattr(child_context, "recheck_binding", lambda binding: None)
    completed = []

    def interrupted(*args):
        os.kill(os.getpid(), signal.SIGINT)
        completed.append(True)

    monkeypatch.setattr(transport, "_restore", interrupted)
    with inherited(monkeypatch) if role == "service" else nullcontext() as controls:
        with pytest.raises(KeyboardInterrupt):
            with child_context.held_child(
                fixture.binding, fixture.profile, role, **keywords(fixture)
            ):
                pytest.fail("interrupted restoration yielded")
        if controls is not None:
            for descriptor in controls[3]:
                with pytest.raises(OSError):
                    os.fstat(descriptor)
    assert completed == []


@pytest.mark.parametrize("body", [OSError("body"), KeyboardInterrupt("first")])
def test_late_holder_replacement_carries_original_progress(
    child_context, accepted, case, tmp_path, monkeypatch, body
):
    fixture = child_case(tmp_path, accepted, case, monkeypatch)
    monkeypatch.setattr(child_context, "recheck_binding", lambda binding: None)
    original, later = (
        child_context.hold_operational_inputs,
        KeyboardInterrupt("cleanup"),
    )
    body.progress = b'{"partial":"invented"}\n'

    @contextmanager
    def held(*args, **kwargs):
        with original(*args, **kwargs) as inputs:
            try:
                yield inputs
            finally:
                raise later

    monkeypatch.setattr(child_context, "hold_operational_inputs", held)
    with pytest.raises(KeyboardInterrupt) as caught:
        with child_context.held_child(
            fixture.binding, fixture.profile, "client", **keywords(fixture)
        ):
            raise body
    assert caught.value is (later if isinstance(body, Exception) else body)
    assert caught.value.progress == body.progress


def test_original_restore_progress_survives_final_binding_interruption(
    child_context, accepted, case, tmp_path, monkeypatch
):
    fixture = child_case(tmp_path, accepted, case, monkeypatch)
    original, later = ValueError("restore"), KeyboardInterrupt("final binding")
    original.progress = b"existing restoration evidence"
    checks = []

    def restore(*arguments):
        raise original

    def recheck(binding):
        checks.append(binding)
        if len(checks) == 2:
            raise later

    monkeypatch.setattr(transport, "_restore", restore)
    monkeypatch.setattr(child_context, "recheck_binding", recheck)
    with pytest.raises(KeyboardInterrupt) as caught:
        with child_context.held_child(
            fixture.binding, fixture.profile, "client", **keywords(fixture)
        ):
            pytest.fail("failed restoration yielded")
    assert caught.value is later
    assert caught.value.progress == original.progress
