import json
from dataclasses import asdict

import pytest
from external_replay_codec_fixtures import decode, encode, module, sample

from automated_phishing_detection.phishvn import PreparedExternalRow, _json_bytes

__all__ = ["sample"]

_STRINGS = tuple(
    name
    for name in PreparedExternalRow.__dataclass_fields__
    if name not in ("file_position", "is_phishing")
)


def replace_first(sample, row):
    return _json_bytes(row) + encode(sample.retained[1:])


@pytest.mark.parametrize("field", PreparedExternalRow.__dataclass_fields__)
def test_every_row_field_is_required(sample, field):
    api, row = module(), asdict(sample.retained[0])
    del row[field]
    with pytest.raises(api.ExternalReplayCodecError):
        decode(replace_first(sample, row))


@pytest.mark.parametrize("field", _STRINGS)
@pytest.mark.parametrize("value", [None, True, 7, []])
def test_all_ten_string_fields_have_exact_string_types(sample, field, value):
    api, row = module(), asdict(sample.retained[0])
    row[field] = value
    with pytest.raises(api.ExternalReplayCodecError):
        decode(replace_first(sample, row))


@pytest.mark.parametrize(
    "field,value",
    [
        ("extra", None),
        ("file_position", True),
        ("file_position", 1.0),
        ("file_position", 0),
        ("file_position", -1),
        ("file_position", "1"),
        ("is_phishing", True),
        ("is_phishing", 1.0),
        ("is_phishing", 2),
        ("is_phishing", "1"),
        ("is_phishing", None),
        ("source_split", "train"),
        ("published_split", "val"),
        ("record_id", "has space"),
        ("role", "secondary"),
        ("source_group", "unknown"),
        ("source_class", "legitimate"),
        ("confidence_tier", "silver"),
        ("canonical_url_sha256", "0" * 64),
        ("registrable_domain", "different.example"),
        ("raw_url", "invalid"),
    ],
)
def test_rehashed_row_semantic_forgery_rejected(sample, field, value):
    api, row = module(), asdict(sample.retained[0])
    row[field] = value
    with pytest.raises(api.ExternalReplayCodecError):
        decode(replace_first(sample, row))


@pytest.mark.parametrize("mutation", ["id", "canonical", "position", "order"])
def test_unique_identities_and_strict_original_order(sample, mutation):
    api = module()
    rows = [asdict(row) for row in sample.retained]
    if mutation == "order":
        rows.reverse()
    elif mutation == "canonical":
        for name in ("raw_url", "canonical_url_sha256", "registrable_domain"):
            rows[1][name] = rows[0][name]
    else:
        name = "record_id" if mutation == "id" else "file_position"
        rows[1][name] = rows[0][name]
    with pytest.raises(api.ExternalReplayCodecError):
        decode(b"".join(_json_bytes(row) for row in rows))


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_newline",
        "blank_line",
        "spaces",
        "duplicate_key",
        "float",
        "unicode",
        "crlf",
        "array",
        "scalar",
    ],
)
def test_rehashed_noncanonical_rows_are_rejected(sample, mutation):
    api, content = module(), encode(sample.retained)
    first, remaining = content.split(b"\n", 1)
    variants = {
        "missing_newline": content[:-1],
        "blank_line": b"\n" + content,
        "spaces": b" " + content,
        "duplicate_key": first[:-1] + b',"role":"gold"}\n' + remaining,
        "float": first.replace(b'"is_phishing":1', b'"is_phishing":NaN')
        + b"\n"
        + remaining,
        "unicode": first.replace(b'"record_id":"row-0"', b'"record_id":"r\\u006fw-0"')
        + b"\n"
        + remaining,
        "crlf": content.replace(b"\n", b"\r\n"),
        "array": _json_bytes([json.loads(first)]) + remaining,
        "scalar": b"7\n" + remaining,
    }
    with pytest.raises(api.ExternalReplayCodecError):
        decode(variants[mutation])
