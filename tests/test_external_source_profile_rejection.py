"""Candidate resolution authenticates before parsing and never grants access."""

import json
from hashlib import sha256

import pytest
import test_external_source_profile as profile_fixtures
from test_external_source_profile import (
    IMPLEMENTATIONS,
    PACKAGE,
    api,
    resolve,
)

from automated_phishing_detection import execution_preflight as preflight
from automated_phishing_detection import phiusiil

profile_case = profile_fixtures.profile_case


@pytest.mark.parametrize("name", (*IMPLEMENTATIONS, "data/sources.json"))
def test_missing_required_bound_identity_stops_before_source_read(profile_case, name):
    hashes = profile_case.hashes.copy()
    hashes.pop(name if name == "data/sources.json" else PACKAGE + name)
    with pytest.raises(api().ExternalSourceProfileError):
        resolve(profile_case, source_hashes=tuple(sorted(hashes.items())))
    assert "public_source_read" not in profile_case.events


@pytest.mark.parametrize("value", [None, {}, [], "binding", True])
def test_only_exact_execution_binding_is_accepted(profile_case, value):
    with pytest.raises(api().ExternalSourceProfileError):
        api().resolve_external_source_profile(value)
    assert profile_case.events == []


@pytest.mark.parametrize(
    "entries",
    [
        [],
        {},
        (("data/sources.json",),),
        ((True, "a" * 64),),
        (("unknown", True),),
        (("unknown", "A" * 64),),
        (["unknown", "a" * 64],),
    ],
)
def test_bound_source_entries_have_exact_types_and_shapes(profile_case, entries):
    with pytest.raises(api().ExternalSourceProfileError):
        resolve(profile_case, source_hashes=entries)
    assert "public_source_read" not in profile_case.events


@pytest.mark.parametrize("different", [False, True])
def test_duplicate_source_entries_are_rejected_even_with_matching_values(
    profile_case, different
):
    entry = profile_case.binding.source_hashes[0]
    duplicate = (entry[0], "f" * 64 if different else entry[1])
    with pytest.raises(api().ExternalSourceProfileError):
        resolve(
            profile_case, source_hashes=(*profile_case.binding.source_hashes, duplicate)
        )
    assert "public_source_read" not in profile_case.events


@pytest.mark.parametrize(
    "field,value",
    [
        ("revision", "invalid"),
        ("revision", True),
        ("contract_sha256", "F" * 64),
        ("runtime_json", {}),
        ("runtime_json", "NaN"),
        ("runtime_json", "[]"),
        ("runtime_json", '{"duplicate":1,"duplicate":1}'),
    ],
)
def test_execution_identity_shape_cannot_enter_profile(profile_case, field, value):
    with pytest.raises(api().ExternalSourceProfileError):
        resolve(profile_case, **{field: value})


def test_source_hash_is_verified_before_existing_source_parser(
    profile_case, monkeypatch
):
    profile_case.source += b"private-canary"
    monkeypatch.setattr(
        phiusiil,
        "_load_source_spec",
        lambda *args: pytest.fail("parse before hash authentication"),
    )
    with pytest.raises(api().ExternalSourceProfileError) as rejected:
        resolve(profile_case)
    assert "private-canary" not in str(rejected.value)


@pytest.mark.parametrize("change", ["missing", "extra", "commit", "hash", "duplicate"])
def test_hash_bound_public_source_still_requires_existing_schema(profile_case, change):
    source = json.loads(profile_case.source)
    if change == "missing":
        source["public_suffix_list"].pop("version")
    elif change == "extra":
        source["public_suffix_list"]["private-canary"] = True
    elif change == "commit":
        source["public_suffix_list"]["commit"] = "a" * 40
    elif change == "hash":
        source["public_suffix_list"]["sha256"] = "invalid"
    content = json.dumps(source).encode()
    if change == "duplicate":
        content = content.replace(
            b'"schema_version": 2', b'"schema_version": 2,"schema_version": 2'
        )
    profile_case.source = content
    hashes = profile_case.hashes | {"data/sources.json": sha256(content).hexdigest()}
    with pytest.raises(api().ExternalSourceProfileError):
        resolve(profile_case, source_hashes=tuple(sorted(hashes.items())))


@pytest.mark.parametrize("failed_check", [1, 2])
def test_binding_is_rechecked_before_read_and_after_construction(
    profile_case, monkeypatch, failed_check
):
    observed = []

    def recheck(binding):
        observed.append(binding)
        if len(observed) == failed_check:
            raise preflight.ExecutionPreflightError("private-canary")

    monkeypatch.setattr(preflight, "recheck_binding", recheck)
    with pytest.raises(api().ExternalSourceProfileError) as rejected:
        resolve(profile_case)
    assert len(observed) == failed_check
    assert ("public_source_read" in profile_case.events) is (failed_check == 2)
    assert "private-canary" not in str(rejected.value)


def test_source_spec_bytes_change_profile_identity_even_when_psl_is_unchanged(
    profile_case,
):
    original = resolve(profile_case)
    profile_case.source += b"\n"
    hashes = profile_case.hashes | {
        "data/sources.json": sha256(profile_case.source).hexdigest()
    }
    changed = resolve(profile_case, source_hashes=tuple(sorted(hashes.items())))
    assert changed.profile_sha256 != original.profile_sha256
    assert changed.suffix_rules_sha256 == original.suffix_rules_sha256


@pytest.mark.parametrize(
    "name",
    [
        "data/sources.json",
        PACKAGE + "_external_source_profile.py",
        "unrelated-public-file",
    ],
)
@pytest.mark.parametrize("value", [True, None, "F" * 64, "short"])
def test_invalid_hash_rejects_even_with_all_required_entries(profile_case, name, value):
    hashes = profile_case.hashes | {name: value}
    with pytest.raises(api().ExternalSourceProfileError):
        resolve(profile_case, source_hashes=tuple(sorted(hashes.items())))
    assert "public_source_read" not in profile_case.events


def test_valid_psl_digest_comes_only_from_authenticated_public_source(profile_case):
    source = json.loads(profile_case.source)
    source["public_suffix_list"]["sha256"] = "f" * 64
    profile_case.source = json.dumps(source).encode()
    hashes = profile_case.hashes | {
        "data/sources.json": sha256(profile_case.source).hexdigest()
    }
    profile = resolve(profile_case, source_hashes=tuple(sorted(hashes.items())))
    assert profile.suffix_rules_sha256 == "f" * 64
