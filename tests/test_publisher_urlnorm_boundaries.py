"""Exercise retained-only restoration boundaries with invented source snapshots."""

import builtins
import csv
import json
import zipfile
from dataclasses import replace
from hashlib import sha256
from pathlib import Path

import pytest
from test_publisher_urlnorm import encoded, module, restore, sample

from automated_phishing_detection import (
    _external_provenance_payloads,
    _phishvn_archive,
    phishvn,
    phishvn_source,
)


def forbidden(*args, **kwargs):
    pytest.fail("retained representation attempted source I/O or scientific work")


def test_derivation_and_restore_perform_no_original_source_or_scientific_work(
    monkeypatch,
):
    parent, derived, source, summary, pins = sample()
    for owner, name in (
        (builtins, "open"),
        (Path, "open"),
        (Path, "read_bytes"),
        (zipfile, "ZipFile"),
        (csv, "reader"),
        (csv, "writer"),
        (_phishvn_archive, "authenticated_members"),
        (phishvn_source, "authenticated_members"),
        (phishvn_source, "decode_phishvn_archive"),
        (phishvn_source, "_tables"),
        (phishvn_source, "_csv_table"),
        (phishvn_source, "_result"),
        (phishvn, "prepare_external_rows"),
    ):
        monkeypatch.setattr(owner, name, forbidden)
    assert module().derive_publisher_url_norm(parent) == derived
    assert restore(source, summary, pins) == derived


@pytest.mark.parametrize("pins", [None, {}, "private-canary"])
def test_restore_requires_original_archive_pin_type(pins):
    _, _, source, summary, _ = sample()
    with pytest.raises(phishvn_source.PhishVNSourceError):
        restore(source, summary, pins)


@pytest.mark.parametrize(
    "field,value",
    [
        ("archive_sha256", "a" * 64),
        ("archive_size_bytes", 1),
        ("archive_size_bytes", True),
        ("archive_size_bytes", 1.0),
    ],
)
def test_original_archive_pin_mismatch_rejects(field, value):
    _, _, source, summary, pins = sample()
    with pytest.raises(phishvn_source.PhishVNSourceError):
        restore(source, summary, replace(pins, **{field: value}))


@pytest.mark.parametrize("parent", [None, {}, "private-canary"])
def test_derivation_requires_original_decoded_type(parent):
    with pytest.raises(phishvn_source.PhishVNSourceError):
        module().derive_publisher_url_norm(parent)


def test_empty_original_snapshot_retains_empty_splits():
    parent, derived, source, summary, pins = sample([])
    assert not parent.rows and not derived.rows
    assert derived.published_split_counts == {"train": 0, "val": 0, "test": 0}
    assert restore(source, summary, pins) == derived


def test_existing_provenance_decoder_preserves_derived_rows_and_exact_source_bytes():
    _, derived, source, summary, _ = sample()
    restored, payloads = _external_provenance_payloads.decoder_inputs(derived)
    assert restored == derived
    assert payloads == {
        "publisher-source.json": encoded(source),
        "publisher-summary.json": encoded(summary),
    }


@pytest.mark.parametrize("change", ["position", "mapping", "missing_header"])
def test_parent_v1_validation_survives_consistently_rehashed_derived_records(change):
    parent, _, source, summary, pins = sample()
    original_source = json.loads(parent.private_outputs["publisher-source.json"])
    original_summary = json.loads(encoded(parent.public_summary))
    for document in (source, original_source):
        if change == "position":
            document["rows"][0]["file_position"] = 2
        elif change == "mapping":
            document["rows"][0]["normalization_status"] = "undefined_mapping"
        else:
            del document["headers"]["data/dataset_url.csv"]
    parent_digest = sha256(encoded(original_source)).hexdigest()
    original_summary["private_sha256"]["publisher-source.json"] = parent_digest
    for document in (source, summary):
        document["parent_source_sha256"] = parent_digest
        document["parent_summary_sha256"] = sha256(
            encoded(original_summary)
        ).hexdigest()
    summary["private_sha256"]["publisher-source.json"] = sha256(
        encoded(source)
    ).hexdigest()
    with pytest.raises(phishvn_source.PhishVNSourceError):
        restore(source, summary, pins)


@pytest.mark.parametrize("target", ["source", "summary"])
@pytest.mark.parametrize(
    "convert",
    [
        memoryview,
        lambda value: value.decode("ascii"),
        lambda value: b"null\n",
        lambda value: b"[]\n",
        lambda value: value.replace(
            b'"schema_version":2', b'"schema_version":2,"schema_version":2'
        ),
        lambda value: value.replace(b'"schema_version":2', b'"schema_version":NaN'),
        lambda value: b"{private-canary",
        lambda value: b"\xffprivate-canary",
    ],
)
def test_invalid_json_and_duplicate_properties_reject_without_leaking_cells(
    target, convert
):
    _, _, source, summary, pins = sample()
    arguments = {"source": encoded(source), "summary": encoded(summary)}
    arguments[target] = convert(arguments[target])
    with pytest.raises(phishvn_source.PhishVNSourceError) as rejected:
        module().restore_derived_publisher(
            arguments["source"], arguments["summary"], pins=pins
        )
    assert "private-canary" not in str(rejected.value)
