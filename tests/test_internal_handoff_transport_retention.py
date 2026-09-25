"""Closing transport never deletes pathname replacements or retained evidence."""

import os

import pytest
import test_internal_handoff_transport as fixtures
from test_internal_handoff_transport_failures import assert_closed
from test_internal_handoff_transport_failures import resources as resources

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


def test_replacement_after_entry_observation_is_never_deleted(
    transport_api, payloads, temporary_parent, resources, monkeypatch
):
    receipt = transport_api.transport_io.receipt
    original, changed = receipt._entry, []

    def substitute(directory, name):
        metadata = original(directory, name)
        if name == "internal-source-overlap.json" and not changed:
            target = directory.path / name
            target.rename(temporary_parent / "retained-original")
            target.write_bytes(b"unowned replacement")
            changed.append(target)
        return metadata

    try:
        with transport_api.retain_internal_handoff(payloads):
            monkeypatch.setattr(receipt, "_entry", substitute)
    except transport_api.InternalTransportError:
        pass
    assert len(changed) == 1
    assert changed[0].read_bytes() == b"unowned replacement"
    assert_closed(resources[0])


@pytest.mark.parametrize("partial", [False, True])
def test_transport_closure_never_calls_unlink_or_rmdir(
    transport_api, payloads, temporary_parent, resources, monkeypatch, partial
):
    def forbidden(*args, **kwargs):
        pytest.fail("transport attempted destructive pathname cleanup")

    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt()

    monkeypatch.setattr(os, "unlink", forbidden)
    monkeypatch.setattr(os, "rmdir", forbidden)
    if partial:
        monkeypatch.setattr(os, "write", interrupted)
        with pytest.raises(KeyboardInterrupt):
            with transport_api.retain_internal_handoff(payloads):
                pytest.fail("interrupted transport yielded")
    else:
        with transport_api.retain_internal_handoff(payloads) as retained:
            assert fixtures.read(transport_api, retained) == payloads
    assert resources[0].path.is_dir()
    assert_closed(resources[0])
