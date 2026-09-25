"""One external attempt retains provenance and the unchanged producer inventory."""

import importlib
import importlib.util
import json
import stat
from hashlib import sha256

import pytest

from automated_phishing_detection import execution_receipt
from automated_phishing_detection._saved_external_bindings import PRIVATE_OUTPUTS

PROVENANCE = (
    "publisher-source.json",
    "publisher-summary.json",
    "suffix-rules.dat",
    "internal-source-overlap.json",
    "internal-source-handoff.json",
    "external-source-reconstruction.json",
)


def checkpoint_module():
    name = "automated_phishing_detection._external_source_checkpoints"
    assert importlib.util.find_spec(name) is not None, "missing external writer"
    return importlib.import_module(name)


@pytest.fixture
def writer(tmp_path):
    module = checkpoint_module()
    attempt = execution_receipt.reserve_attempt(
        tmp_path / "external", identity={"fixture": 1}
    )
    return module.ExternalCheckpointWriter(attempt, identity={"fixture": 1})


def provenance():
    return {name: name.encode("ascii") for name in PROVENANCE}


def science():
    module = checkpoint_module()
    return {name: name.encode("ascii") for name in module.SCIENTIFIC_ORDER}


def test_exact_provenance_and_existing_scientific_inventory(writer):
    module = checkpoint_module()
    assert module.PROVENANCE_ORDER == PROVENANCE
    assert frozenset(module.SCIENTIFIC_ORDER) == PRIVATE_OUTPUTS
    outputs = science()
    writer.begin(provenance())
    for name, content in outputs.items():
        writer(name, content)
    retained = writer.complete(outputs)
    assert retained == provenance() | outputs
    assert len(retained) == 36
    directory = writer.attempt.directory / "checkpoints"
    assert stat.S_IMODE(directory.stat().st_mode) == 0o700
    assert {entry.name for entry in directory.iterdir()} == set(retained)
    for name, content in retained.items():
        assert (directory / name).read_bytes() == content
        assert stat.S_IMODE((directory / name).stat().st_mode) == 0o600
    progress = json.loads(writer.snapshot())
    assert progress["status"] == "complete"
    assert progress["confirmed_sha256"] == {
        name: sha256(content).hexdigest() for name, content in retained.items()
    }
    retained.clear()
    assert len(writer.complete(outputs)) == 36


@pytest.mark.parametrize("name", ["../private", "bindings.json", "unknown"])
def test_unexpected_callback_is_latched_before_installation(writer, name):
    module = checkpoint_module()
    writer.begin(provenance())
    with pytest.raises(module.ExternalCheckpointError):
        writer(name, b"private-canary")
    with pytest.raises(module.ExternalCheckpointError):
        writer(module.SCIENTIFIC_ORDER[0], b"not retried")
    assert set((writer.attempt.directory / "checkpoints").iterdir()) == {
        writer.attempt.directory / "checkpoints" / name for name in PROVENANCE
    }
    assert json.loads(writer.snapshot())["status"] == "failed"


def test_callback_before_begin_never_creates_directory(writer):
    module = checkpoint_module()
    with pytest.raises(module.ExternalCheckpointError):
        writer(module.SCIENTIFIC_ORDER[0], b"data")
    with pytest.raises(module.ExternalCheckpointError):
        writer.begin(provenance())
    assert not (writer.attempt.directory / "checkpoints").exists()


def test_begin_copies_provenance_and_rejects_a_second_begin(writer):
    module = checkpoint_module()
    inputs = provenance()
    writer.begin(inputs)
    inputs.clear()
    with pytest.raises(module.ExternalCheckpointError):
        writer.begin(provenance())
    assert len(json.loads(writer.snapshot())["confirmed_sha256"]) == 6


@pytest.mark.parametrize("mutation", ["missing", "extra", "changed"])
def test_complete_rejects_nonidentical_producer_outputs(writer, mutation):
    module = checkpoint_module()
    outputs = science()
    writer.begin(provenance())
    for name, content in outputs.items():
        writer(name, content)
    if mutation == "missing":
        outputs.pop("bindings.json")
    else:
        outputs["unknown" if mutation == "extra" else "bindings.json"] = b"changed"
    with pytest.raises(module.ExternalCheckpointError):
        writer.complete(outputs)
    assert json.loads(writer.snapshot())["status"] == "failed"
