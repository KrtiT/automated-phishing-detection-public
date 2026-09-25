"""Validate caller-pinned in-memory archives against the publisher writer format."""

import io
import re
import stat
import struct
import zipfile
import zlib
from dataclasses import dataclass
from hashlib import sha256

COPIED_MEMBERS = (
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
EXPECTED_MEMBERS = frozenset((*COPIED_MEMBERS, "README.md", "MANIFEST.txt"))
EXPECTED_FORMAT = {
    "dataset_doi": "10.17632/b97hxbxtpd.4",
    "release": "3.1.0",
    "publisher_commit": "bdebc126097d18e0837253f6367726ceb7303202",
    "archive_filename": "PhishVN_v3.1.0_open.zip",
    "archive_sha256": "308351e9d0c0fca13a81f2f63524a7dee08ccba6c775848d125b3da5525d0ab5",
    "archive_size_bytes": 5813309,
}


class PhishVNSourceError(ValueError):
    """A symbolic source rejection without protected values or parser diagnostics."""


@dataclass(frozen=True)
class PhishVNSourcePins:
    archive_sha256: str
    archive_size_bytes: int


def require(condition, reason):
    if not condition:
        raise PhishVNSourceError(reason)


def _authenticate(content, pins):
    require(type(pins) is PhishVNSourcePins, "invalid_source_pins")
    require(
        type(pins.archive_sha256) is str
        and re.fullmatch(r"[0-9a-f]{64}", pins.archive_sha256) is not None
        and type(pins.archive_size_bytes) is int
        and pins.archive_size_bytes > 0,
        "invalid_source_pins",
    )
    require(type(content) is bytes, "invalid_archive_bytes")
    require(len(content) == pins.archive_size_bytes, "archive_size_mismatch")
    require(sha256(content).hexdigest() == pins.archive_sha256, "archive_hash_mismatch")


def _inventory(archive):
    entries = archive.infolist()
    require(
        len(entries) == len(EXPECTED_MEMBERS)
        and {entry.filename for entry in entries} == EXPECTED_MEMBERS,
        "archive_inventory_mismatch",
    )
    for entry in entries:
        require(
            entry.orig_filename == entry.filename
            and not entry.is_dir()
            and stat.S_IFMT(entry.external_attr >> 16) in (0, stat.S_IFREG)
            and not entry.flag_bits & 1
            and entry.compress_type == zipfile.ZIP_DEFLATED,
            "unsupported_archive_member",
        )
    return entries


def _verify_manifest(contents):
    lines = ["PhishVN v3.1.0 — MANIFEST (SHA-256)", ""]
    lines.extend(
        f"{sha256(contents[name]).hexdigest()}  {name}  ({len(contents[name])} bytes)"
        for name in COPIED_MEMBERS
    )
    expected = ("\n".join(lines) + "\n").encode("utf-8")
    require(contents["MANIFEST.txt"] == expected, "manifest_mismatch")


def authenticated_members(content: bytes, pins: PhishVNSourcePins) -> dict[str, bytes]:
    """Check all member bytes; expected pins do not authorize protected access."""
    try:
        _authenticate(content, pins)
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            entries = _inventory(archive)
            contents = {entry.filename: archive.read(entry) for entry in entries}
        _verify_manifest(contents)
        return contents
    except PhishVNSourceError:
        raise
    except (
        ValueError,
        TypeError,
        RuntimeError,
        EOFError,
        zipfile.BadZipFile,
        struct.error,
        zlib.error,
    ):
        raise PhishVNSourceError("archive_decode_failed") from None
