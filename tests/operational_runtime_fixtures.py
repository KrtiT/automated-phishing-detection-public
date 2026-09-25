"""Invented consistency inputs, not accepted source or process authority."""

import importlib
import importlib.util
from hashlib import sha256
from pathlib import Path

import pytest

from automated_phishing_detection._checkpoint_codec import canonical_bytes
from automated_phishing_detection.execution_preflight import ExecutionBinding
from automated_phishing_detection.http_replay import ReplayRequest
from automated_phishing_detection.operational_schedule import cell_for_ordinal

ARTIFACTS = (
    "cascade.json",
    "gmm.json",
    "length-only.json",
    "logistic-l1.json",
    "transformer-weights.npz",
    "transformer.json",
    "vocabulary.json",
)
THRESHOLDS = (
    "length_only",
    "logistic_l1",
    "transformer",
    "half_width",
    "monitor_boundary",
)


def primary():
    return {
        "artifact_hashes": {
            name: sha256(name.encode()).hexdigest() for name in ARTIFACTS
        },
        "thresholds": dict(zip(THRESHOLDS, (0.1, 0.5, 0.6, 0.2, -1.0), strict=True)),
    }


def binding():
    return ExecutionBinding(
        Path("/invented"), "a" * 40, "b" * 64, (("data/sources.json", "c" * 64),), "{}"
    )


def inputs(ordinal=1):
    from automated_phishing_detection.operational_cell_inputs import (
        RestoredOperationalCell,
    )

    execution = {
        "revision": "a" * 40,
        "execution_contract_sha256": "b" * 64,
        "runtime_sha256": sha256(b"{}").hexdigest(),
        "source_spec_sha256": "c" * 64,
    }
    accepted = canonical_bytes(
        {
            "primary": primary(),
            "execution": execution,
            "operational_profile_sha256": "d" * 64,
        }
    )
    descriptor = canonical_bytes({"manifest_sha256": "e" * 64})
    requests = tuple(
        ReplayRequest(f"row-{index}", f"https://invented.test/{index}")
        for index in range(1000 if ordinal > 120 else 10000)
    )
    return RestoredOperationalCell(
        accepted, descriptor, b"{}\n", b"invented", cell_for_ordinal(ordinal), requests
    )


@pytest.fixture
def records():
    name = "automated_phishing_detection.operational_role_records"
    assert importlib.util.find_spec(name), "missing closed operational role records"
    return importlib.import_module(name)


@pytest.fixture
def runtime():
    name = "automated_phishing_detection.operational_runtime"
    assert importlib.util.find_spec(name), "missing single-load operational runtime"
    return importlib.import_module(name)
