"""Reject malformed saved publisher JSON and caller consistency bindings."""

import json
from dataclasses import replace

import pytest
from test_saved_phishvn_source import encoded, module, rehash, restore, saved_sample

from automated_phishing_detection._phishvn_archive import EXPECTED_MEMBERS
from automated_phishing_detection.phishvn_source import PhishVNSourceError


@pytest.mark.parametrize("target", ["source", "summary"])
@pytest.mark.parametrize(
    "convert",
    [
        bytearray,
        memoryview,
        lambda value: value.decode("ascii"),
        lambda value: value.rstrip(b"\n"),
        lambda value: b" " + value,
        lambda value: json.dumps(json.loads(value), indent=2).encode(),
        lambda value: b"[]\n",
        lambda value: b"null\n",
        lambda value: b'{"private-canary":NaN}\n',
        lambda value: b'{"private-canary":Infinity}\n',
        lambda value: b"\xffprivate-canary",
        lambda value: b"{private-canary",
        lambda value: value.replace(
            b'"schema_version":1', b'"schema_version":1,"schema_version":1'
        ),
        lambda value: value.replace(
            b'"size_bytes":', b'"size_bytes":0,"size_bytes":', 1
        ),
    ],
)
def test_only_exact_canonical_json_bytes_are_accepted(target, convert):
    _, source, summary, pins = saved_sample()
    arguments = {"source": encoded(source), "summary": encoded(summary)}
    arguments[target] = convert(arguments[target])
    with pytest.raises(PhishVNSourceError) as rejected:
        module().restore_phishvn_source(
            arguments["source"], arguments["summary"], pins=pins
        )
    assert "private-canary" not in str(rejected.value)


@pytest.mark.parametrize("role", ["source", "summary"])
@pytest.mark.parametrize("change", ["missing", "extra"])
def test_top_level_layout_is_closed(role, change):
    _, source, summary, pins = saved_sample()
    target = source if role == "source" else summary
    if change == "extra":
        target["private-canary"] = "unexpected"
    else:
        target.pop("schema_version")
    rehash(source, summary)
    with pytest.raises(PhishVNSourceError):
        restore(source, summary, pins)


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_version", True),
        ("schema_version", 1.0),
        ("algorithm_id", "other"),
        ("protected_evaluation_authorized", 0),
        ("protected_evaluation_authorized", True),
        ("source_binding", "authorized"),
        ("expected_format", {}),
        ("input_archive", {}),
    ],
)
def test_consistently_rehashed_identity_forgery_is_rejected(field, value):
    _, source, summary, pins = saved_sample()
    source[field] = summary[field] = value
    rehash(source, summary)
    with pytest.raises(PhishVNSourceError):
        restore(source, summary, pins)


@pytest.mark.parametrize(
    "field,value",
    [
        ("archive_sha256", "f" * 64),
        ("archive_sha256", "F" * 64),
        ("archive_sha256", None),
        ("archive_sha256", "short"),
        ("archive_size_bytes", 1),
        ("archive_size_bytes", True),
        ("archive_size_bytes", 1.0),
        ("archive_size_bytes", 0),
        ("archive_size_bytes", -1),
        ("archive_size_bytes", "1"),
    ],
)
def test_caller_archive_pin_mismatch_or_bad_shape_is_rejected(field, value):
    _, source, summary, pins = saved_sample()
    with pytest.raises(PhishVNSourceError):
        restore(source, summary, replace(pins, **{field: value}))


@pytest.mark.parametrize("pins", [None, {}, "private-canary"])
def test_caller_pins_require_exact_declared_type(pins):
    _, source, summary, _ = saved_sample()
    with pytest.raises(PhishVNSourceError):
        restore(source, summary, pins)


@pytest.mark.parametrize("change", ["missing", "extra", "type"])
def test_member_inventory_is_the_exact_fifteen_names(change):
    _, source, summary, pins = saved_sample()
    inventory = source["member_inventory"]
    if change == "missing":
        inventory.pop("LICENSE")
    elif change == "extra":
        inventory["other"] = inventory["LICENSE"]
    else:
        inventory = []
    source["member_inventory"] = summary["member_inventory"] = inventory
    rehash(source, summary)
    with pytest.raises(PhishVNSourceError):
        restore(source, summary, pins)


@pytest.mark.parametrize("name", sorted(EXPECTED_MEMBERS))
@pytest.mark.parametrize(
    "metadata",
    [
        {},
        [],
        {"sha256": "0" * 64},
        {"sha256": "0" * 64, "size_bytes": 1, "extra": 1},
        {"sha256": "0" * 63, "size_bytes": 1},
        {"sha256": "F" * 64, "size_bytes": 1},
        {"sha256": 0, "size_bytes": 1},
        {"sha256": "0" * 64, "size_bytes": True},
        {"sha256": "0" * 64, "size_bytes": 1.0},
        {"sha256": "0" * 64, "size_bytes": -1},
    ],
)
def test_every_member_requires_closed_metadata_shapes(name, metadata):
    _, source, summary, pins = saved_sample()
    source["member_inventory"][name] = metadata
    summary["member_inventory"][name] = metadata
    rehash(source, summary)
    with pytest.raises(PhishVNSourceError):
        restore(source, summary, pins)
