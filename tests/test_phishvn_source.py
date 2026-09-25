"""Decode invented archive buffers and preserve the unchanged external policy."""

import json

import pytest
from phishvn_source_fixtures import HEADER, csv_bytes, decode, members, prepare, record


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
def test_exact_publisher_mappings_preserve_policy(source, tier, label, role, outcome):
    original = record("original", source=source, tier=tier, label=label)
    decoded = decode(members([original]))
    retained = prepare(decoded).retained
    assert [(row.role, row.is_phishing) for row in retained] == [(role, outcome)]
    private = json.loads(decoded.private_outputs["publisher-source.json"])
    assert private["rows"][0]["cells"] == [original[name] for name in HEADER]
    assert decoded.rows[0].raw_url == original["url"]


@pytest.mark.parametrize("channel", ["url", "social", "qr", "unknown-channel", ""])
def test_channel_is_opaque_provenance_without_an_eligibility_filter(channel):
    decoded = decode(members([record("channel-row", channel=channel)]))
    assert len(prepare(decoded).retained) == 1
    private = json.loads(decoded.private_outputs["publisher-source.json"])
    assert private["rows"][0]["cells"][HEADER.index("channel")] == channel


@pytest.mark.parametrize(
    "changes,reason",
    [
        ({"source": "ncsc"}, "undefined_phishvn_mapping"),
        (
            {"source": "tranco", "tier": "gold", "label": "benign"},
            "undefined_phishvn_mapping",
        ),
        ({"tier": "gold "}, "undefined_phishvn_mapping"),
        ({"label": "1"}, "undefined_phishvn_mapping"),
        ({"label": "NaN"}, "undefined_phishvn_mapping"),
        ({"source": "null"}, "undefined_phishvn_mapping"),
        ({"source": ""}, "missing_phishvn_mapping_field"),
        ({"label": ""}, "missing_phishvn_mapping_field"),
        ({"tier": ""}, "missing_phishvn_mapping_field"),
    ],
)
def test_unmapped_rows_reach_existing_whole_canonical_group_quarantine(changes, reason):
    original = record("original")
    conflicting = record("second", **(changes | {"url": original["url"]}))
    decoded = decode(members([original, conflicting]))
    prepared = prepare(decoded)
    assert len(decoded.rows) == len(prepared.quarantine) == 2
    assert prepared.retained == ()
    assert any(reason in item.reason_codes for item in prepared.quarantine)
    assert all(
        "canonical_url_invalid_mapping" in item.reason_codes
        for item in prepared.quarantine
    )


@pytest.mark.parametrize("url", ["bare.example", "", " https://space.example/"])
def test_raw_url_is_never_replaced_by_valid_url_norm(url):
    decoded = decode(members([record("invalid-url", url=url)]))
    assert decoded.rows[0].raw_url == url
    assert "invalid_or_missing_url" in prepare(decoded).quarantine[0].reason_codes


def test_positions_follow_logical_csv_records_and_actual_headers():
    original = [
        record("z-last", status="two\nlines"),
        record("a-first", "train"),
        record("b-second"),
    ]
    decoded = decode(members(original))
    assert [
        (row.source_split, row.file_position, row.published_id) for row in decoded.rows
    ] == [
        ("train", 1, "a-first"),
        ("test", 1, "z-last"),
        ("test", 2, "b-second"),
    ]
    assert decoded.published_split_counts == {"train": 1, "val": 0, "test": 2}
    private = json.loads(decoded.private_outputs["publisher-source.json"])
    assert private["headers"]["data/splits/url_test.csv"] == list(HEADER)
    assert private["rows"][1]["cells"][HEADER.index("status")] == "two\nlines"


def test_bad_published_split_is_preserved_for_existing_domain_quarantine():
    original = record("split-row", split="unexpected")
    contents = members([])
    contents["data/dataset_url.csv"] = csv_bytes([original])
    contents["data/splits/url_test.csv"] = csv_bytes([original])
    decoded = decode(contents)
    assert decoded.rows[0].published_split == "unexpected"
    assert "invalid_published_split" in prepare(decoded).quarantine[0].reason_codes


@pytest.mark.parametrize(
    "field,reason",
    [
        ("id", "missing_or_invalid_published_id"),
        ("split", "invalid_published_split"),
    ],
)
def test_empty_identity_cells_are_preserved_without_imputation(field, reason):
    original = record("missing-value", **{field: ""})
    contents = members([])
    for name in ("data/dataset_url.csv", "data/splits/url_test.csv"):
        contents[name] = csv_bytes([original])
    decoded = decode(contents)
    assert (
        getattr(decoded.rows[0], "published_id" if field == "id" else "published_split")
        == ""
    )
    assert reason in prepare(decoded).quarantine[0].reason_codes


def test_all_split_positions_and_cross_split_domains_reach_existing_quarantine():
    original = [
        record("test-record"),
        record("validation-record", "val"),
        record("train-record", "train"),
    ]
    for row in original:
        row["url"] = "https://shared.example/path"
    decoded = decode(members(original))
    assert [(row.source_split, row.file_position) for row in decoded.rows] == [
        ("train", 1),
        ("val", 1),
        ("test", 1),
    ]
    assert all(
        "domain_crosses_published_splits" in row.reason_codes
        for row in prepare(decoded).quarantine
    )
