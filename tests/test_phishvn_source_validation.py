"""Structural failures and privacy boundaries using invented CSV bytes only."""

import builtins
import json
from hashlib import sha256
from pathlib import Path

import pytest
from phishvn_source_fixtures import (
    HEADER,
    csv_bytes,
    decode,
    decoder_module,
    members,
    record,
)

from automated_phishing_detection import phishvn


@pytest.mark.parametrize(
    "content",
    [
        b"",
        b"id,url\n",
        b"id,id,url,source,label,tier,split,channel\n",
        b"id,url,source,label,tier,split,channel,\n",
        b"\xef\xbb\xbfid,url,source,label,tier,split,channel\n",
        b"id,url,source,label,tier,split,channel\nshort,row\n",
        b'id,url,source,label,tier,split,channel\n"unterminated-secret-canary\n',
        b"\xffprivate-secret-canary",
        b"id,url,source,label,tier,split,channel\x00\n",
    ],
)
def test_bad_csv_is_a_symbolic_structural_rejection(content):
    contents = members()
    contents["data/splits/url_test.csv"] = content
    with pytest.raises(decoder_module().PhishVNSourceError) as rejected:
        decode(contents)
    assert "secret-canary" not in str(rejected.value)


@pytest.mark.parametrize("change", ["missing", "extra", "header", "cell"])
def test_full_table_and_complete_split_inventory_must_agree(change):
    contents = members()
    rows = [record("test")]
    if change == "missing":
        rows = []
    elif change == "extra":
        rows.append(record("additional"))
    elif change == "cell":
        rows[0]["status"] = "changed"
    contents["data/splits/url_test.csv"] = csv_bytes(
        rows, HEADER[::-1] if change == "header" else HEADER
    )
    with pytest.raises(
        decoder_module().PhishVNSourceError, match="source_table_mismatch"
    ):
        decode(contents)


def test_summary_stays_aggregate_and_decoder_does_no_io_or_preparation(monkeypatch):
    contents = members([record("secret-canary", status="private-status-canary")])
    api = decoder_module()

    def forbidden(*args, **kwargs):
        pytest.fail("decoder attempted I/O or preparation")

    for owner, name in (
        (builtins, "open"),
        (Path, "open"),
        (Path, "read_bytes"),
        (phishvn, "prepare_external_rows"),
    ):
        monkeypatch.setattr(owner, name, forbidden)
    decoded = decode(contents)
    public = json.dumps(decoded.public_summary)
    assert "secret-canary" not in public and "private-status-canary" not in public
    assert decoded.public_summary["source_binding"] == "caller_supplied_pins_only"
    assert decoded.public_summary["protected_evaluation_authorized"] is False
    assert decoded.public_summary["private_sha256"] == {
        name: sha256(content).hexdigest()
        for name, content in decoded.private_outputs.items()
    }
    assert "rows=" not in repr(decoded) and "private_outputs=" not in repr(decoded)
    assert isinstance(decoded, api.DecodedPhishVNSource)


@pytest.mark.parametrize(
    "missing", ["id", "url", "source", "label", "tier", "split", "channel"]
)
def test_every_required_named_column_is_structural(missing):
    contents = members()
    header = tuple(name for name in HEADER if name != missing)
    contents["data/splits/url_test.csv"] = csv_bytes([record("test")], header)
    with pytest.raises(decoder_module().PhishVNSourceError, match="invalid_csv_header"):
        decode(contents)


def test_complete_realized_headers_and_inventory_are_retained_without_public_values():
    header = HEADER[::-1] + ("private-header-canary",)
    original = record("identity", **{"private-header-canary": "private-cell-canary"})
    contents = members([])
    for name in ("data/dataset_url.csv", "data/splits/url_test.csv"):
        contents[name] = csv_bytes([original], header)
    for split in ("train", "val"):
        contents[f"data/splits/url_{split}.csv"] = csv_bytes([], header)
    decoded = decode(contents)
    private = json.loads(decoded.private_outputs["publisher-source.json"])
    assert private["headers"]["data/dataset_url.csv"] == list(header)
    assert private["rows"][0]["cells"] == [original[name] for name in header]
    for name, content in contents.items():
        assert decoded.public_summary["member_inventory"][name] == {
            "sha256": sha256(content).hexdigest(),
            "size_bytes": len(content),
        }
    assert "canary" not in json.dumps(decoded.public_summary)


def test_member_manifest_verification_precedes_any_csv_parsing(monkeypatch):
    import csv

    from phishvn_source_fixtures import bundle, manifest

    contents = members()
    prior_manifest = manifest(contents)
    contents["data/vn_compphish.csv"] += b"opaque feature bytes"
    fixture = bundle(contents, supplied_manifest=prior_manifest)
    api = decoder_module()
    monkeypatch.setattr(csv, "reader", lambda *args, **kwargs: pytest.fail("early CSV"))
    with pytest.raises(api.PhishVNSourceError, match="manifest_mismatch"):
        api.decode_phishvn_archive(
            fixture.content, pins=api.PhishVNSourcePins(**fixture.pins)
        )
