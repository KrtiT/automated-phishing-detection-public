"""Entirely invented accepted-row bytes; no original sources or process authority."""

import importlib
import importlib.util
from dataclasses import asdict
from hashlib import sha256

import pytest
from external_producer_fixtures import prepared_external

from automated_phishing_detection.phishvn import _json_bytes


def module():
    name = "automated_phishing_detection.external_replay_codec"
    assert importlib.util.find_spec(name), "missing retained external replay codec"
    return importlib.import_module(name)


@pytest.fixture(scope="module")
def sample():
    return prepared_external(1001)


def encode(rows):
    return b"".join(_json_bytes(asdict(row)) for row in rows)


def decode(content, digest=None):
    return module().decode_external_replay_manifest(
        content,
        expected_sha256=sha256(content).hexdigest() if digest is None else digest,
    )
