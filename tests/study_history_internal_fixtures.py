"""Historical science fixtures read only temporary invented producer outputs."""

import importlib
import importlib.util
import json
from hashlib import sha256
from types import SimpleNamespace

import pytest
from test_source_completion import published
from test_source_runner import inputs, runner

from automated_phishing_detection import execution_receipt
from automated_phishing_detection._checkpoint_codec import canonical_bytes
from automated_phishing_detection._internal_handoff_validation import SNAPSHOT_NAMES
from automated_phishing_detection._internal_scientific_protocol import (
    SCIENTIFIC_CHECKPOINT_NAMES,
    completion_bytes,
    expected_counts,
)

__all__ = ["published", "inputs", "runner"]


def api():
    name = "automated_phishing_detection.study_history_internal"
    assert importlib.util.find_spec(name), "missing pure historical internal science"
    return importlib.import_module(name)


@pytest.fixture
def historical_internal(published):
    binding, paths, unused = published
    payloads = {}
    for name in SNAPSHOT_NAMES:
        if name == "public-summary.json":
            path = paths.public_summary
        elif name.startswith("source/"):
            path = binding.root / name.removeprefix("source/")
        else:
            path = paths.attempt / name.removeprefix("attempt/")
        payloads[name] = path.read_bytes()
    execution = json.loads(payloads["public-summary.json"])["execution"]
    return SimpleNamespace(
        payloads=payloads,
        pins={name: sha256(content).hexdigest() for name, content in payloads.items()},
        execution=execution,
        sources=dict(binding.source_hashes),
        attempt=str(paths.attempt),
    )


def verify(case, **changes):
    arguments = dict(
        expected_snapshot_sha256=case.pins,
        expected_execution=case.execution,
        expected_source_sha256=case.sources,
        expected_attempt_directory=case.attempt,
    )
    return api().verify_historical_internal_science(
        case.payloads, **(arguments | changes)
    )


def repin(case):
    case.pins = {
        name: sha256(content).hexdigest() for name, content in case.payloads.items()
    }


def republish(case, public):
    content = execution_receipt._json_bytes(public, "invented_history")
    case.payloads["public-summary.json"] = content
    outcome = json.loads(case.payloads["attempt/outcome.json"])
    outcome.update(
        public_summary_sha256=sha256(content).hexdigest(),
        private_sha256=public["private_sha256"],
    )
    case.payloads["attempt/outcome.json"] = execution_receipt._json_bytes(
        outcome, "invented_history"
    )
    repin(case)


def republish_private(case, name, value, *, public_value=None):
    content = canonical_bytes(value)
    for directory in ("evidence", "scientific-checkpoints"):
        case.payloads[f"attempt/{directory}/{name}"] = content
    public = json.loads(case.payloads["public-summary.json"])
    public["private_sha256"][name] = sha256(content).hexdigest()
    if public_value is not None:
        public[name.removesuffix(".json")] = public_value
    scientific = {
        member: case.payloads[f"attempt/scientific-checkpoints/{member}"]
        for member in SCIENTIFIC_CHECKPOINT_NAMES - {"completion.json"}
    }
    case.payloads["attempt/scientific-checkpoints/completion.json"] = completion_bytes(
        scientific,
        case.execution["reservation_sha256"],
        *expected_counts(public["row_count"]),
    )
    republish(case, public)
