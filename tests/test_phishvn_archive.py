"""Invented publisher archives; no external data or paths are read."""

import zipfile
from hashlib import sha256
from importlib import import_module
from importlib.util import find_spec

import pytest
from phishvn_source_fixtures import COPIED, bundle, manifest, members


def module():
    name = "automated_phishing_detection._phishvn_archive"
    assert find_spec(name) is not None, "missing byte-only publisher archive validator"
    return import_module(name)


def verify(fixture):
    api = module()
    return api.authenticated_members(
        fixture.content, api.PhishVNSourcePins(**fixture.pins)
    )


def test_complete_archive_verifies_all_members_without_parsing_opaque_tables():
    fixture = bundle()
    assert verify(fixture) == fixture.contents


@pytest.mark.parametrize(
    "field,value", [("archive_sha256", "f" * 64), ("archive_size_bytes", 1)]
)
def test_outer_authentication_precedes_zip_open(monkeypatch, field, value):
    fixture = bundle()
    fixture.pins[field] = value
    api = module()
    monkeypatch.setattr(
        zipfile, "ZipFile", lambda *args: pytest.fail("ZIP before authentication")
    )
    with pytest.raises(api.PhishVNSourceError, match="archive_.*_mismatch"):
        verify(fixture)


@pytest.mark.parametrize(
    "change", ["extra", "missing", "duplicate", "alias", "symlink"]
)
def test_rejects_ambiguous_or_nonregular_inventory(change):
    contents = members()
    original_manifest = manifest(contents)
    kwargs = {}
    if change == "extra":
        contents["extra.txt"] = b"extra"
    elif change == "missing":
        del contents["LICENSE"]
    elif change == "alias":
        contents["../LICENSE"] = contents.pop("LICENSE")
    elif change == "duplicate":
        kwargs["duplicate"] = "LICENSE"
    else:
        kwargs["symlink"] = "LICENSE"
    fixture = bundle(contents, supplied_manifest=original_manifest, **kwargs)
    with pytest.raises(module().PhishVNSourceError):
        verify(fixture)


@pytest.mark.parametrize(
    "change", ["hash", "size", "version", "missing", "extra", "order"]
)
def test_manifest_is_exact_writer_output(change):
    contents = members()
    lines = manifest(contents).decode("utf-8").splitlines()
    if change == "hash":
        lines[2] = "0" * 64 + lines[2][64:]
    elif change == "size":
        lines[2] = lines[2].replace(" bytes)", "0 bytes)")
    elif change == "version":
        lines[0] = lines[0].replace("3.1.0", "3.0.0")
    elif change == "missing":
        lines.pop()
    elif change == "extra":
        lines.append(lines[-1])
    else:
        lines[2], lines[3] = lines[3], lines[2]
    fixture = bundle(contents, supplied_manifest=("\n".join(lines) + "\n").encode())
    with pytest.raises(module().PhishVNSourceError, match="manifest_mismatch"):
        verify(fixture)


@pytest.mark.parametrize(
    "field,value", [("archive_sha256", "bad"), ("archive_size_bytes", True)]
)
def test_pin_fields_require_exact_valid_types(field, value):
    fixture = bundle()
    fixture.pins[field] = value
    with pytest.raises(module().PhishVNSourceError):
        verify(fixture)


def test_malformed_archive_errors_hide_content_and_parser_context():
    api = module()
    content = b"private-secret-canary-not-a-zip"
    pins = api.PhishVNSourcePins(sha256(content).hexdigest(), len(content))
    with pytest.raises(api.PhishVNSourceError) as rejected:
        api.authenticated_members(content, pins)
    assert "secret-canary" not in str(rejected.value)
    assert rejected.value.__suppress_context__


@pytest.mark.parametrize("name", COPIED)
def test_every_copied_member_is_checked_against_the_manifest(name):
    contents = members()
    original_manifest = manifest(contents)
    contents[name] += b"private-tamper-canary"
    fixture = bundle(contents, supplied_manifest=original_manifest)
    with pytest.raises(module().PhishVNSourceError, match="manifest_mismatch"):
        verify(fixture)


@pytest.mark.parametrize(
    "convert", [bytearray, memoryview, lambda value: value.decode("latin1")]
)
def test_mutable_and_nonbyte_archives_are_rejected(convert):
    fixture = bundle()
    fixture.content = convert(fixture.content)
    with pytest.raises(module().PhishVNSourceError, match="invalid_archive_bytes"):
        verify(fixture)


def test_invalid_pin_object_is_rejected_before_zip_open(monkeypatch):
    api = module()
    monkeypatch.setattr(zipfile, "ZipFile", lambda *args: pytest.fail("opened ZIP"))
    with pytest.raises(api.PhishVNSourceError, match="invalid_source_pins"):
        api.authenticated_members(b"invented", None)
