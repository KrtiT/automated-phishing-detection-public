"""In-memory invented publisher archives shared by source decoder tests."""

import csv
import io
import stat
import zipfile
from hashlib import sha256
from importlib import import_module
from importlib.util import find_spec
from types import SimpleNamespace

import pytest

from automated_phishing_detection import phishvn, protocol_preflight

COPIED = (
    "data/dataset_url.csv",
    "data/vn_compphish.csv",
    "data/splits/url_train.csv",
    "data/splits/url_val.csv",
    "data/splits/url_test.csv",
    "data/attacks/p3_paraphrase.csv",
    "data/attacks/p3_paraphrase_band.csv",
    "data/abuse_type.csv",
    "docs/datasheet.md",
    "docs/schema.md",
    "docs/data_sources.md",
    "LICENSE",
    "CITATION.cff",
)
HEADER = (
    "source",
    "id",
    "url",
    "label",
    "channel",
    "tier",
    "split",
    "url_norm",
    "status",
)


def record(identity, split="test", **changes):
    return {
        "source": "tinnhiemmang",
        "id": identity,
        "url": f"https://{identity}.example/path",
        "label": "phishing",
        "channel": "url",
        "tier": "gold",
        "split": split,
        "url_norm": "https://unused.example/",
        "status": "invented-status",
    } | changes


def csv_bytes(rows, header=HEADER):
    stream = io.StringIO(newline="")
    writer = csv.writer(stream)
    writer.writerow(header)
    writer.writerows(tuple(row.get(name, "") for name in header) for row in rows)
    return stream.getvalue().encode("utf-8")


def members(rows=None):
    rows = rows if rows is not None else [record("training", "train"), record("test")]
    contents = {name: b"invented opaque payload\n" for name in COPIED}
    contents["data/dataset_url.csv"] = csv_bytes(rows)
    for split in ("train", "val", "test"):
        contents[f"data/splits/url_{split}.csv"] = csv_bytes(
            [row for row in rows if row["split"] == split]
        )
    contents["README.md"] = b"invented public README\n"
    return contents


def manifest(contents):
    lines = ["PhishVN v3.1.0 — MANIFEST (SHA-256)", ""]
    lines.extend(
        f"{sha256(contents[name]).hexdigest()}  {name}  ({len(contents[name])} bytes)"
        for name in COPIED
    )
    return ("\n".join(lines) + "\n").encode("utf-8")


def _entries(archive, contents, symlink):
    for name, content in contents.items():
        info = zipfile.ZipInfo(name)
        info.compress_type = zipfile.ZIP_DEFLATED
        if name == symlink:
            info.external_attr = (stat.S_IFLNK | 0o777) << 16
        archive.writestr(info, content)


def bundle(contents=None, *, supplied_manifest=None, duplicate=None, symlink=None):
    contents = dict(members() if contents is None else contents)
    contents["MANIFEST.txt"] = (
        manifest(contents) if supplied_manifest is None else supplied_manifest
    )
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        _entries(archive, contents, symlink)
        if duplicate is not None:
            with pytest.warns(UserWarning, match="Duplicate name"):
                archive.writestr(duplicate, contents[duplicate])
    content = stream.getvalue()
    return SimpleNamespace(
        content=content,
        contents=contents,
        pins={
            "archive_sha256": sha256(content).hexdigest(),
            "archive_size_bytes": len(content),
        },
    )


def decoder_module():
    name = "automated_phishing_detection.phishvn_source"
    assert find_spec(name) is not None, "missing byte-only publisher decoder"
    return import_module(name)


def decode(contents=None):
    fixture = bundle(contents)
    api = decoder_module()
    return api.decode_phishvn_archive(
        fixture.content, pins=api.PhishVNSourcePins(**fixture.pins)
    )


def prepare(decoded):
    return phishvn.prepare_external_rows(
        decoded.rows,
        published_split_counts=decoded.published_split_counts,
        test_split="test",
        suffix_rules=protocol_preflight.parse_suffix_rules("example\n"),
        phiusiil_domains=frozenset(),
    )
