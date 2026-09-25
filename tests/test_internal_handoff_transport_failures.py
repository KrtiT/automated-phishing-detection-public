"""First interruptions survive partial setup and owned-resource cleanup failures."""

import asyncio
import os
import signal
import tempfile

import pytest
import test_internal_handoff_transport as fixtures

inputs = fixtures.inputs
published = fixtures.published
runner = fixtures.runner
verifier = fixtures.verifier
observed_worker = fixtures.observed_worker
completion = fixtures.completion
handoff_api = fixtures.handoff_api
transport_api = fixtures.transport_api
payloads = fixtures.payloads
temporary_parent = fixtures.temporary_parent


@pytest.fixture
def resources(transport_api, monkeypatch):
    owner_type = transport_api.transport_io.RetainedFiles
    original, owners = owner_type.__init__, []

    def initialize(owner):
        original(owner)
        owners.append(owner)

    monkeypatch.setattr(owner_type, "__init__", initialize)
    return owners


def assert_closed(owner):
    for descriptor in owner.descriptors:
        with pytest.raises(OSError):
            os.fstat(descriptor)


@pytest.mark.parametrize("after_write", [False, True])
@pytest.mark.parametrize("kind", [OSError, KeyboardInterrupt, asyncio.CancelledError])
def test_partial_file_creation_does_not_retry_and_closes_owned_resources(
    transport_api, payloads, temporary_parent, resources, monkeypatch, after_write, kind
):
    original, calls, failure = os.write, [], kind("private-canary")

    def interrupted(descriptor, content):
        calls.append(descriptor)
        if after_write:
            original(descriptor, content)
        raise failure

    monkeypatch.setattr(os, "write", interrupted)
    expected = transport_api.InternalTransportError if kind is OSError else kind
    with pytest.raises(expected) as caught:
        with transport_api.retain_internal_handoff(payloads):
            pytest.fail("partial transport was yielded")
    assert len(calls) == 1
    assert_closed(resources[0])
    assert resources[0].path.is_dir()
    if kind is not OSError:
        assert caught.value is failure
    else:
        assert "private-canary" not in str(caught.value)


def test_signal_before_temporary_path_assignment_closes_owned_descriptors(
    transport_api, payloads, temporary_parent, resources, monkeypatch
):
    original = tempfile.mkdtemp

    def interrupted(*args, **kwargs):
        path = original(*args, **kwargs)
        os.kill(os.getpid(), signal.SIGINT)
        return path

    monkeypatch.setattr(tempfile, "mkdtemp", interrupted)
    with pytest.raises(KeyboardInterrupt):
        with transport_api.retain_internal_handoff(payloads):
            pytest.fail("interrupted setup yielded a transport")
    assert_closed(resources[0])
    assert resources[0].path.is_dir()


@pytest.mark.parametrize(
    "kind", [KeyboardInterrupt, SystemExit, asyncio.CancelledError]
)
@pytest.mark.parametrize("cleanup_kind", [OSError, KeyboardInterrupt, SystemExit])
def test_body_interruption_survives_cleanup_failure(
    transport_api,
    payloads,
    temporary_parent,
    resources,
    monkeypatch,
    kind,
    cleanup_kind,
):
    original = kind("original")

    def broken(*args, **kwargs):
        raise cleanup_kind("cleanup")

    with pytest.raises(kind) as caught:
        with transport_api.retain_internal_handoff(payloads):
            monkeypatch.setattr(type(resources[0]), "_check_retained", broken)
            raise original
    assert caught.value is original
    assert_closed(resources[0])


def test_cleanup_failure_cannot_report_success(
    transport_api, payloads, temporary_parent, resources, monkeypatch
):
    def broken(*args, **kwargs):
        raise OSError("private-canary")

    with pytest.raises(transport_api.InternalTransportError) as caught:
        with transport_api.retain_internal_handoff(payloads) as retained:
            monkeypatch.setattr(type(resources[0]), "_check_retained", broken)
    assert "private-canary" not in str(caught.value)
    assert retained.payloads is payloads
    assert_closed(resources[0])


def test_first_cleanup_interruption_survives_deferred_sigint(
    transport_api, payloads, temporary_parent, resources, monkeypatch
):
    first, original_close, delivered = KeyboardInterrupt(), os.close, []

    def fail_cleanup(owner):
        raise first

    def close_with_signal(descriptor):
        original_close(descriptor)
        if not delivered:
            delivered.append(True)
            os.kill(os.getpid(), signal.SIGINT)

    with pytest.raises(KeyboardInterrupt) as caught:
        with transport_api.retain_internal_handoff(payloads):
            monkeypatch.setattr(type(resources[0]), "_check_retained", fail_cleanup)
            monkeypatch.setattr(os, "close", close_with_signal)
    assert caught.value is first
    assert delivered == [True]
    assert_closed(resources[0])


@pytest.mark.parametrize("operation", ["chmod", "fchmod"])
def test_assignment_interruption_survives_buffered_sigint(
    transport_api, payloads, temporary_parent, resources, monkeypatch, operation
):
    first, original = KeyboardInterrupt(), getattr(os, operation)

    def interrupted(*args, **kwargs):
        original(*args, **kwargs)
        os.kill(os.getpid(), signal.SIGINT)
        raise first

    monkeypatch.setattr(os, operation, interrupted)
    with pytest.raises(KeyboardInterrupt) as caught:
        with transport_api.retain_internal_handoff(payloads):
            pytest.fail("interrupted setup yielded transport")
    assert caught.value is first
    assert_closed(resources[0])
    assert resources[0].path.is_dir()
