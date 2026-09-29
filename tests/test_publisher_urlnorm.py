"""Derive exact publisher-normalized inputs from invented retained records."""

import json
from dataclasses import replace
from hashlib import sha256
from importlib import import_module
from importlib.util import find_spec

import pytest
from phishvn_source_fixtures import HEADER, csv_bytes, decode, members, prepare, record

from automated_phishing_detection import phishvn_source, saved_phishvn_source


def module():
    name = "automated_phishing_detection.publisher_urlnorm"
    assert find_spec(name) is not None, "missing retained publisher URL amendment"
    return import_module(name)


def encoded(value):
    return phishvn_source._json_bytes(value)


def sample(rows=None):
    parent = decode(members(rows))
    pins = phishvn_source.PhishVNSourcePins(**parent.public_summary["input_archive"])
    derived = module().derive_publisher_url_norm(parent)
    source = json.loads(derived.private_outputs["publisher-source.json"])
    summary = json.loads(encoded(derived.public_summary))
    return parent, derived, source, summary, pins


def restore(source, summary, pins):
    return module().restore_derived_publisher(
        encoded(source), encoded(summary), pins=pins
    )


def test_derivation_preserves_parent_cells_and_binds_both_canonical_parent_records():
    parent, derived, source, summary, pins = sample()
    original_source = parent.private_outputs["publisher-source.json"]
    original_summary = encoded(parent.public_summary)
    for document in (source, summary):
        assert document["schema_version"] == 2
        assert document["algorithm_id"] == "phishvn-publisher-urlnorm-v1"
        assert document["representation"] == "publisher_url_norm_v1"
        assert document["parent_source_sha256"] == sha256(original_source).hexdigest()
        assert document["parent_summary_sha256"] == sha256(original_summary).hexdigest()
    for field in ("headers", "rows", "member_inventory", "input_archive"):
        assert source[field] == json.loads(original_source)[field]
    for field in ("published_split_counts", "full_table_rows", "mapping_counts"):
        assert summary[field] == parent.public_summary[field]
    assert (
        summary["private_sha256"]["publisher-source.json"]
        == sha256(encoded(source)).hexdigest()
    )
    assert restore(source, summary, pins) == derived
    assert parent.private_outputs["publisher-source.json"] == original_source
    assert encoded(parent.public_summary) == original_summary


@pytest.mark.parametrize("split", ["train", "val", "test"])
@pytest.mark.parametrize(
    "source,tier,label",
    [
        ("tinnhiemmang", "gold", "phishing"),
        ("tinnhiemmang", "silver", "phishing"),
        ("tinnhiem_web", "gold", "benign"),
        ("tinnhiem_org", "gold", "benign"),
        ("chongluadao", "bronze", "phishing"),
        ("openphish", "bronze", "phishing"),
        ("tranco", "silver", "benign"),
        ("tranco_vn", "silver", "benign"),
        ("unknown", "unknown", "unknown"),
        ("", "", ""),
    ],
)
def test_exact_normalized_cell_is_selected_for_every_split_and_mapping(
    split, source, tier, label
):
    normalized = " HTTPS://WWW.Mixed.Example/PaTh?KeY=VaL "
    rows = [
        record("row", split, source=source, tier=tier, label=label, url_norm=normalized)
    ]
    parent, derived, _, _, _ = sample(rows)
    assert derived.rows == (replace(parent.rows[0], raw_url=normalized),)


@pytest.mark.parametrize("normalized", ["", "bare.example", "not a URL"])
def test_invalid_normalized_cell_is_retained_and_reaches_original_quarantine(
    normalized,
):
    parent, derived, _, _, _ = sample([record("row", url_norm=normalized)])
    assert parent.rows[0].raw_url == "https://row.example/path"
    assert derived.rows[0].raw_url == normalized
    assert not prepare(derived).retained
    assert "invalid_or_missing_url" in prepare(derived).quarantine[0].reason_codes


def test_bare_raw_url_uses_only_exact_publisher_value_in_existing_preparation():
    parent, derived, source, summary, pins = sample(
        [record("row", url="row.example", url_norm="https://row.example/Case")]
    )
    assert not prepare(parent).retained
    assert prepare(derived).retained[0].raw_url == "https://row.example/Case"
    assert (
        saved_phishvn_source.restore_phishvn_source(
            encoded(source), encoded(summary), pins=pins
        )
        == derived
    )


def test_missing_normalized_header_rejects_even_without_any_rows():
    header = tuple(name for name in HEADER if name != "url_norm")
    contents = members([])
    for name in (
        "data/dataset_url.csv",
        *(f"data/splits/url_{split}.csv" for split in ("train", "val", "test")),
    ):
        contents[name] = csv_bytes([], header)
    with pytest.raises(
        phishvn_source.PhishVNSourceError, match="missing_publisher_url_norm"
    ):
        module().derive_publisher_url_norm(decode(contents))


def test_header_order_cells_and_logical_split_order_remain_exact():
    rows = [
        record(split, split, url_norm=f"https://{split}.example/Case")
        for split in ("test", "train", "val")
    ]
    contents = members(rows)
    for name in (
        "data/dataset_url.csv",
        *(f"data/splits/url_{split}.csv" for split in ("train", "val", "test")),
    ):
        selected = (
            rows
            if name == "data/dataset_url.csv"
            else [row for row in rows if f"url_{row['split']}.csv" in name]
        )
        contents[name] = csv_bytes(selected, HEADER[::-1])
    parent = decode(contents)
    derived = module().derive_publisher_url_norm(parent)
    assert [
        (row.source_split, row.file_position, row.raw_url) for row in derived.rows
    ] == [
        (split, 1, f"https://{split}.example/Case")
        for split in ("train", "val", "test")
    ]
    original = json.loads(parent.private_outputs["publisher-source.json"])
    result = json.loads(derived.private_outputs["publisher-source.json"])
    assert result["headers"] == original["headers"]
    assert result["rows"] == original["rows"]
