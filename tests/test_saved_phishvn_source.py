"""Restore invented retained publisher cells without original source access."""

import builtins
import csv
import json
import zipfile
from dataclasses import asdict
from hashlib import sha256
from importlib import import_module
from importlib.util import find_spec
from pathlib import Path

import pytest
from phishvn_source_fixtures import HEADER, csv_bytes, decode, members, prepare, record

from automated_phishing_detection import _phishvn_archive, phishvn, phishvn_source


def module():
    name = "automated_phishing_detection.saved_phishvn_source"
    assert find_spec(name) is not None, "missing saved publisher restoration"
    return import_module(name)


def encoded(value):
    return phishvn_source._json_bytes(value)


def saved_sample(rows=None):
    decoded = decode(members(rows))
    source = json.loads(decoded.private_outputs["publisher-source.json"])
    summary = json.loads(encoded(decoded.public_summary))
    pins = phishvn_source.PhishVNSourcePins(**summary["input_archive"])
    return decoded, source, summary, pins


def restore(source, summary, pins):
    return module().restore_phishvn_source(encoded(source), encoded(summary), pins=pins)


def rehash(source, summary):
    summary["private_sha256"]["publisher-source.json"] = sha256(
        encoded(source)
    ).hexdigest()


@pytest.mark.parametrize(
    "rows",
    [[], [record("test-only")], [record("val-only", "val")], None],
)
def test_restoration_preserves_all_split_counts_exact_bytes_and_fresh_summary(rows):
    original, source, summary, pins = saved_sample(rows)
    restored = restore(source, summary, pins)
    assert restored == original
    assert restored.public_summary is not original.public_summary
    assert restored.published_split_counts is not original.published_split_counts
    assert restored.public_summary["source_binding"] == "caller_supplied_pins_only"
    assert restored.public_summary["protected_evaluation_authorized"] is False
    assert "rows=" not in repr(restored) and "private_outputs=" not in repr(restored)


@pytest.mark.parametrize(
    "source,tier,label,role,outcome",
    [
        ("tinnhiemmang", "gold", "phishing", "gold", 1),
        ("tinnhiemmang", "silver", "phishing", "secondary", 1),
        ("tinnhiem_web", "gold", "benign", "certified", 0),
        ("tinnhiem_org", "gold", "benign", "certified", 0),
        ("chongluadao", "bronze", "phishing", "secondary", 1),
        ("openphish", "bronze", "phishing", "secondary", 1),
        ("tranco", "silver", "benign", "tranco", None),
        ("tranco_vn", "silver", "benign", "tranco", None),
    ],
)
def test_every_publisher_mapping_remains_preparation_compatible(
    source, tier, label, role, outcome
):
    original = record("retained", source=source, tier=tier, label=label)
    decoded, saved, summary, pins = saved_sample([original])
    restored = restore(saved, summary, pins)
    assert prepare(restored) == prepare(decoded)
    assert [(row.role, row.is_phishing) for row in prepare(restored).retained] == [
        (role, outcome)
    ]


@pytest.mark.parametrize("channel", ["", "qr", "social", "opaque-\u2603"])
@pytest.mark.parametrize("changes", [{}, {"source": ""}, {"label": "undefined"}])
def test_opaque_channels_and_unknown_mappings_are_preserved(channel, changes):
    rows = [record("private-canary", channel=channel, **changes)]
    decoded, source, summary, pins = saved_sample(rows)
    restored = restore(source, summary, pins)
    assert restored == decoded
    assert prepare(restored) == prepare(decoded)
    assert "private-canary" not in json.dumps(restored.public_summary)


def test_restoration_keeps_realized_headers_and_raw_cells_in_logical_split_order():
    header = HEADER[::-1] + ("opaque_extra",)
    originals = [
        record(split, split, opaque_extra="two\nlines")
        for split in ("test", "train", "val")
    ]
    contents = members(originals)
    contents["data/dataset_url.csv"] = csv_bytes(originals, header)
    for split in ("train", "val", "test"):
        contents[f"data/splits/url_{split}.csv"] = csv_bytes(
            [row for row in originals if row["split"] == split], header
        )
    decoded = decode(contents)
    pins = phishvn_source.PhishVNSourcePins(**decoded.public_summary["input_archive"])
    source_bytes = decoded.private_outputs["publisher-source.json"]
    restored = module().restore_phishvn_source(
        source_bytes, encoded(decoded.public_summary), pins=pins
    )
    assert restored == decoded
    assert restored.private_outputs["publisher-source.json"] is source_bytes
    assert [(row.source_split, row.file_position) for row in restored.rows] == [
        ("train", 1),
        ("val", 1),
        ("test", 1),
    ]


def test_restore_forbids_archive_csv_path_preparation_or_scoring_work(monkeypatch):
    original, source, summary, pins = saved_sample()
    api = module()

    def forbidden(*args, **kwargs):
        pytest.fail("saved restoration attempted source I/O or non-restoration work")

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
    assert (
        api.restore_phishvn_source(encoded(source), encoded(summary), pins=pins)
        == original
    )
    assert asdict(pins) == summary["input_archive"]
