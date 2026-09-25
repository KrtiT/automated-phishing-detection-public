"""Retained strings must be possible outputs of the original UTF-8 decoder."""

import pytest
from phishvn_source_fixtures import HEADER, csv_bytes, decode, members, record
from test_saved_phishvn_source import encoded, module, rehash, restore, saved_sample

from automated_phishing_detection.phishvn_source import (
    PhishVNSourceError,
    PhishVNSourcePins,
)


@pytest.mark.parametrize("invalid", ["\x00", "\ud800", "\udfff"])
@pytest.mark.parametrize("location", ["header", "channel", "url"])
def test_rehashed_impossible_source_text_is_rejected(invalid, location):
    _, source, summary, pins = saved_sample([record("example")])
    text = f"private-canary{invalid}text"
    if location == "header":
        for header in source["headers"].values():
            header[HEADER.index("status")] = text
    else:
        source["rows"][0]["cells"][HEADER.index(location)] = text
    rehash(source, summary)
    with pytest.raises(PhishVNSourceError) as rejected:
        restore(source, summary, pins)
    assert "private-canary" not in str(rejected.value)


@pytest.mark.parametrize("invalid", ["\x00", "\ud800", "\udfff"])
def test_original_decoder_also_rejects_impossible_source_text(invalid):
    contents = members([record("example", channel="sentinel")])
    for name in ("data/dataset_url.csv", "data/splits/url_test.csv"):
        contents[name] = contents[name].replace(
            b"sentinel", invalid.encode("utf-8", "surrogatepass")
        )
    with pytest.raises(PhishVNSourceError):
        decode(contents)


@pytest.mark.parametrize(
    "text", ["\ufeff", "\U0001f642", "\uffff", "\U0010ffff", "\x01", "\n\r"]
)
def test_valid_unicode_cells_are_preserved_exactly(text):
    original, source, summary, pins = saved_sample(
        [record("example", channel=text, url=f"https://example.test/{text}")]
    )
    assert restore(source, summary, pins) == original


def _quoted_first_header_contents(text):
    header = (f"{text}opaque", *HEADER)
    original = record("example", **{header[0]: "opaque-value"})
    contents = members([])
    for name in (
        "data/dataset_url.csv",
        "data/splits/url_train.csv",
        "data/splits/url_val.csv",
        "data/splits/url_test.csv",
    ):
        rows = (
            [original]
            if name in ("data/dataset_url.csv", "data/splits/url_test.csv")
            else []
        )
        content = csv_bytes(rows, header)
        contents[name] = b'"' + content.replace(b",", b'",', 1)
    return contents


@pytest.mark.parametrize(
    "text", ["\ufeff", "\U0001f642", "\uffff", "\U0010ffff", "\x01"]
)
def test_valid_quoted_first_header_can_include_bom_or_other_unicode(text):
    decoded = decode(_quoted_first_header_contents(text))
    restored = module().restore_phishvn_source(
        decoded.private_outputs["publisher-source.json"],
        encoded(decoded.public_summary),
        pins=PhishVNSourcePins(**decoded.public_summary["input_archive"]),
    )
    assert restored == decoded


def test_bom_before_original_csv_header_remains_rejected():
    contents = _quoted_first_header_contents("\ufeff")
    contents["data/dataset_url.csv"] = (
        b"\xef\xbb\xbf" + contents["data/dataset_url.csv"]
    )
    with pytest.raises(PhishVNSourceError, match="invalid_csv_encoding"):
        decode(contents)
