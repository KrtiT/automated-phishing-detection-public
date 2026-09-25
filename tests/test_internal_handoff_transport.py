"""Temporary handoff transport uses invented, verified internal snapshots only."""

import importlib
import importlib.util
import json
import stat
import tempfile
from collections import Counter
from dataclasses import FrozenInstanceError
from hashlib import sha256

import pytest
import test_internal_external_handoff as fixtures

from automated_phishing_detection.internal_external_handoff import (
    InternalHandoffPayloads,
)

inputs = fixtures.inputs
published = fixtures.published
runner = fixtures.runner
verifier = fixtures.verifier
observed_worker = fixtures.observed_worker
completion = fixtures.completion
handoff_api = fixtures.handoff_api

NAMES = {"internal-source-handoff.json", "internal-source-overlap.json"}


@pytest.fixture
def transport_api():
    name = "automated_phishing_detection.internal_handoff_transport"
    assert importlib.util.find_spec(name), "missing private internal handoff transport"
    return importlib.import_module(name)


@pytest.fixture
def payloads(completion, handoff_api):
    return handoff_api.build_internal_handoff(completion)


@pytest.fixture
def temporary_parent(tmp_path, monkeypatch):
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path))
    return tmp_path


def read(api, transport, *, expected=None):
    return api.read_internal_handoff_transport(
        transport.directory,
        expected_handoff_sha256=expected or transport.expected_handoff_sha256,
    )


def test_private_roundtrip_retains_parent_bytes_after_descriptor_closure(
    transport_api, payloads, temporary_parent
):
    with transport_api.retain_internal_handoff(payloads) as retained:
        assert retained.payloads is payloads
        assert (
            retained.expected_handoff_sha256
            == sha256(payloads.handoff_bytes).hexdigest()
        )
        assert retained.directory.parent == temporary_parent
        assert {path.name for path in retained.directory.iterdir()} == NAMES
        assert stat.S_IMODE(retained.directory.stat().st_mode) == 0o700
        for path in retained.directory.iterdir():
            metadata = path.stat()
            assert stat.S_ISREG(metadata.st_mode) and metadata.st_nlink == 1
            assert stat.S_IMODE(metadata.st_mode) == 0o600
        assert read(transport_api, retained) == payloads
        with pytest.raises(FrozenInstanceError):
            retained.directory = temporary_parent
    assert retained.directory.is_dir()
    assert retained.payloads is payloads
    assert (
        retained.expected_handoff_sha256 == sha256(payloads.handoff_bytes).hexdigest()
    )


def test_each_transport_payload_is_read_once(
    transport_api, payloads, temporary_parent, monkeypatch
):
    from automated_phishing_detection import source_runner

    original, reads = source_runner._read_file_once, Counter()

    def observed(path, **kwargs):
        reads[path.name] += 1
        return original(path, **kwargs)

    with transport_api.retain_internal_handoff(payloads) as retained:
        monkeypatch.setattr(source_runner, "_read_file_once", observed)
        assert read(transport_api, retained) == payloads
    assert reads == Counter(dict.fromkeys(NAMES, 1))


def test_canonical_temporary_parent_precedes_creation(
    transport_api, payloads, tmp_path, monkeypatch
):
    target = tmp_path / "actual"
    target.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(target, target_is_directory=True)
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(alias))
    with transport_api.retain_internal_handoff(payloads) as retained:
        assert retained.directory.parent == target
        assert read(transport_api, retained) == payloads


def test_wrong_parent_hash_rejects_before_json_parsing(
    transport_api, payloads, temporary_parent, monkeypatch
):
    def forbidden(*args, **kwargs):
        pytest.fail("transport parsed before checking the observing parent digest")

    with transport_api.retain_internal_handoff(payloads) as retained:
        monkeypatch.setattr(json, "loads", forbidden)
        with pytest.raises(transport_api.InternalTransportError):
            read(transport_api, retained, expected="0" * 64)


@pytest.mark.parametrize(
    "value",
    [None, {}, InternalHandoffPayloads(b"invalid", b"invalid")],
)
def test_invalid_payloads_never_create_temporary_files(
    transport_api, value, monkeypatch
):
    def forbidden(*args, **kwargs):
        pytest.fail("invalid handoff reached temporary directory creation")

    monkeypatch.setattr(tempfile, "mkdtemp", forbidden)
    with pytest.raises(transport_api.InternalTransportError):
        with transport_api.retain_internal_handoff(value):
            pytest.fail("invalid handoff yielded transport")
