"""Read-side interruption and pre-access validation preserve owned boundaries."""

import asyncio
import os

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


@pytest.mark.parametrize("expected", [None, True, "invalid"])
def test_bad_expected_digest_stops_before_any_path_access(
    transport_api, tmp_path, monkeypatch, expected
):
    def forbidden(*args, **kwargs):
        pytest.fail("invalid digest inspected a supplied path")

    monkeypatch.setattr(os, "open", forbidden)
    with pytest.raises(transport_api.InternalTransportError):
        transport_api.read_internal_handoff_transport(
            tmp_path, expected_handoff_sha256=expected
        )


@pytest.mark.parametrize(
    "kind", [KeyboardInterrupt, SystemExit, asyncio.CancelledError]
)
@pytest.mark.parametrize("cleanup_kind", [OSError, KeyboardInterrupt])
def test_read_interruption_survives_final_state_failure(
    transport_api, payloads, temporary_parent, monkeypatch, kind, cleanup_kind
):
    original = kind("original")

    def interrupted(*args, **kwargs):
        raise original

    def broken(*args, **kwargs):
        raise cleanup_kind("cleanup")

    with transport_api.retain_internal_handoff(payloads) as retained:
        with monkeypatch.context() as patch:
            patch.setattr(transport_api, "verify_internal_handoff", interrupted)
            patch.setattr(transport_api.transport_io, "_recheck", broken)
            with pytest.raises(kind) as caught:
                fixtures.read(transport_api, retained)
    assert caught.value is original


@pytest.mark.parametrize("name", sorted(fixtures.NAMES))
def test_missing_payload_rejects_without_recreating_it(
    transport_api, payloads, temporary_parent, name
):
    with pytest.raises(transport_api.InternalTransportError):
        with transport_api.retain_internal_handoff(payloads) as retained:
            path = retained.directory / name
            path.unlink()
            with pytest.raises(transport_api.InternalTransportError):
                fixtures.read(transport_api, retained)
            assert not path.exists()
    assert retained.directory.is_dir()


@pytest.mark.parametrize("name", sorted(fixtures.NAMES))
def test_symlink_payload_is_not_read_or_deleted(
    transport_api, payloads, temporary_parent, name
):
    with pytest.raises(transport_api.InternalTransportError):
        with transport_api.retain_internal_handoff(payloads) as retained:
            path = retained.directory / name
            detached = temporary_parent / "detached"
            path.rename(detached)
            path.symlink_to(detached)
            with pytest.raises(transport_api.InternalTransportError):
                fixtures.read(transport_api, retained)
    assert path.is_symlink() and detached.is_file()


def test_restrictive_umask_does_not_change_transport_modes(
    transport_api, payloads, temporary_parent
):
    original = os.umask(0o777)
    try:
        with transport_api.retain_internal_handoff(payloads) as retained:
            assert fixtures.read(transport_api, retained) == payloads
    finally:
        os.umask(original)
