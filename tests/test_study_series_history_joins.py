"""Retained transport joins use authenticated original bytes, never new inputs."""

import base64
import json
from types import SimpleNamespace

import pytest

from automated_phishing_detection._checkpoint_codec import canonical_bytes
from automated_phishing_detection._study_history_snapshot_records import digest
from automated_phishing_detection._study_series_history_cells import _inputs
from automated_phishing_detection._study_series_history_files import HistoryFiles
from automated_phishing_detection._study_series_history_origin import _origin


def test_original_context_decodes_verified_nested_bytes_without_representation_changes():
    profile = canonical_bytes({"paths": {}})
    scientific = canonical_bytes({"cells": []})
    accounting = canonical_bytes(
        {"scientific_accounting_bytes": base64.b64encode(scientific).decode()}
    )
    metadata = {"internal": {}, "external": {}}
    authority = SimpleNamespace(
        profile_bytes=profile,
        profile_sha256=digest(profile),
        accounting_bytes=accounting,
        source_results_bytes=canonical_bytes({"accepted_inputs": metadata}),
    )
    origin = _origin("snapshot", authority, {"pin": "original"})
    assert origin.metadata_bytes == canonical_bytes(metadata)
    assert origin.metadata == metadata and origin.scientific == {"cells": []}
    assert origin.accounting == json.loads(accounting)
    assert origin.authority is authority


def input_case(tmp_path):
    manifest = b"invented retained manifest"
    descriptor = canonical_bytes({"manifest_sha256": digest(manifest)})
    binding = canonical_bytes({"descriptor_sha256": digest(descriptor)})
    for name, content in (
        ("descriptor.json", descriptor),
        ("binding.json", binding),
        ("manifest", manifest),
    ):
        (tmp_path / name).write_bytes(content)
    entry = {
        name: base64.b64encode(content).decode()
        for name, content in (
            ("descriptor_bytes", descriptor),
            ("binding_bytes", binding),
        )
    }
    return entry, descriptor, binding


def test_cell_input_hashes_are_taken_from_original_acceptance(tmp_path):
    entry, descriptor, binding = input_case(tmp_path)
    result = _inputs(
        HistoryFiles(),
        SimpleNamespace(metadata_bytes=b"original"),
        entry,
        SimpleNamespace(cell_input_directory=tmp_path),
    )
    assert result == dict(
        accepted_metadata_bytes=b"original",
        descriptor_bytes=descriptor,
        binding_bytes=binding,
        expected_descriptor_sha256=digest(descriptor),
        expected_binding_sha256=digest(binding),
    )


@pytest.mark.parametrize("name", ["descriptor.json", "binding.json", "manifest"])
def test_cell_input_replacement_cannot_self_pin(tmp_path, name):
    entry, descriptor, binding = input_case(tmp_path)
    (tmp_path / name).write_bytes(b"replacement")
    with pytest.raises(ValueError):
        _inputs(
            HistoryFiles(),
            SimpleNamespace(metadata_bytes=b"original"),
            entry,
            SimpleNamespace(cell_input_directory=tmp_path),
        )
