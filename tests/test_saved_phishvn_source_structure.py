"""Invented saved header, position, normalization, and aggregate forgeries."""

import pytest
from phishvn_source_fixtures import record
from test_saved_phishvn_source import rehash, restore, saved_sample

from automated_phishing_detection.phishvn_source import PhishVNSourceError


@pytest.mark.parametrize("change", ["missing", "extra", "wrong_header", "wrong_type"])
def test_exact_four_headers_and_equality_are_required(change):
    _, source, summary, pins = saved_sample()
    headers = source["headers"]
    if change == "missing":
        headers.pop("data/dataset_url.csv")
    elif change == "extra":
        headers["other"] = headers["data/dataset_url.csv"]
    elif change == "wrong_header":
        headers["data/dataset_url.csv"].reverse()
    else:
        source["headers"] = []
    rehash(source, summary)
    with pytest.raises(PhishVNSourceError):
        restore(source, summary, pins)


@pytest.mark.parametrize(
    "missing", ["id", "url", "source", "label", "tier", "split", "channel"]
)
def test_required_column_names_cannot_be_removed_consistently(missing):
    _, source, summary, pins = saved_sample()
    index = source["headers"]["data/dataset_url.csv"].index(missing)
    for header in source["headers"].values():
        header.pop(index)
    for row in source["rows"]:
        row["cells"].pop(index)
    rehash(source, summary)
    with pytest.raises(PhishVNSourceError):
        restore(source, summary, pins)


@pytest.mark.parametrize("header", [[], "header", ["id", "id"], [""], [True], [None]])
def test_invalid_header_shapes_fail_even_when_all_four_match(header):
    _, source, summary, pins = saved_sample()
    source["headers"] = dict.fromkeys(source["headers"], header)
    rehash(source, summary)
    with pytest.raises(PhishVNSourceError):
        restore(source, summary, pins)


@pytest.mark.parametrize(
    "field,value",
    [
        ("file_position", True),
        ("file_position", 1.0),
        ("file_position", "1"),
        ("file_position", 0),
        ("file_position", -1),
        ("file_position", 2),
        ("source_member", "data/dataset_url.csv"),
        ("source_member", "unexpected"),
        ("source_member", None),
        ("source_member", []),
        ("normalization_status", "undefined_mapping"),
        ("normalization_status", "private-canary"),
        ("normalization_status", 1),
        ("cells", []),
        ("cells", "private-canary"),
        ("cells", {}),
    ],
)
def test_row_fields_and_derived_status_reject_rehashed_forgery(field, value):
    _, source, summary, pins = saved_sample()
    source["rows"][0][field] = value
    rehash(source, summary)
    with pytest.raises(PhishVNSourceError) as rejected:
        restore(source, summary, pins)
    assert "private-canary" not in str(rejected.value)


@pytest.mark.parametrize("value", [None, 1, True, [], {}, 1.0])
def test_every_cell_requires_string_type(value):
    _, source, summary, pins = saved_sample()
    source["rows"][0]["cells"][0] = value
    rehash(source, summary)
    with pytest.raises(PhishVNSourceError):
        restore(source, summary, pins)


@pytest.mark.parametrize(
    "change", ["missing", "extra", "rows_type", "row_type", "order", "duplicate"]
)
def test_row_inventory_schema_and_split_order_are_closed(change):
    _, source, summary, pins = saved_sample()
    if change == "missing":
        source["rows"][0].pop("cells")
    elif change == "extra":
        source["rows"][0]["extra"] = "private-canary"
    elif change == "rows_type":
        source["rows"] = {}
    elif change == "row_type":
        source["rows"][0] = []
    elif change == "order":
        source["rows"].reverse()
    else:
        source["rows"].insert(1, source["rows"][0])
    rehash(source, summary)
    with pytest.raises(PhishVNSourceError):
        restore(source, summary, pins)


@pytest.mark.parametrize("position", [1, 3])
def test_positions_are_sequential_within_each_split(position):
    _, source, summary, pins = saved_sample([record("first"), record("second")])
    source["rows"][1]["file_position"] = position
    rehash(source, summary)
    with pytest.raises(PhishVNSourceError):
        restore(source, summary, pins)


@pytest.mark.parametrize(
    "path,value",
    [
        (("full_table_rows",), 3),
        (("full_table_rows",), 2.0),
        (("published_split_counts", "train"), True),
        (("published_split_counts", "test"), 0),
        (("mapping_counts", "mapped"), 1),
        (("mapping_counts", "undefined_mapping"), True),
        (("private_sha256", "publisher-source.json"), "0" * 64),
        (("member_inventory", "LICENSE", "size_bytes"), 0),
        (("published_split_counts", "extra"), 0),
    ],
)
def test_summary_projection_requires_exact_counts_types_and_linkage(path, value):
    _, source, summary, pins = saved_sample()
    target = summary
    for field in path[:-1]:
        target = target[field]
    target[path[-1]] = value
    with pytest.raises(PhishVNSourceError):
        restore(source, summary, pins)


def test_changed_mapping_cells_cannot_hide_behind_updated_source_hash():
    _, source, summary, pins = saved_sample()
    source["rows"][0]["cells"][0] = ""
    rehash(source, summary)
    with pytest.raises(PhishVNSourceError):
        restore(source, summary, pins)


def test_changed_raw_cell_requires_matching_retained_source_identity():
    _, source, summary, pins = saved_sample()
    source["rows"][0]["cells"][1] = "changed-identity"
    with pytest.raises(PhishVNSourceError):
        restore(source, summary, pins)
