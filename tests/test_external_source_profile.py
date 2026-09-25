"""Closed candidate metadata from invented bindings and public source metadata."""

import json
from dataclasses import replace
from hashlib import sha256
from importlib import import_module
from importlib.util import find_spec
from pathlib import Path
from types import SimpleNamespace

import pytest

from automated_phishing_detection import execution_preflight as preflight
from automated_phishing_detection._checkpoint_codec import canonical_bytes
from automated_phishing_detection._external_checkpoint_protocol import (
    DIRECTORY,
    PROTOCOL,
    PROVENANCE_ORDER,
    SCIENTIFIC_ORDER,
)
from automated_phishing_detection._phishvn_archive import (
    COPIED_MEMBERS,
    EXPECTED_FORMAT,
    EXPECTED_MEMBERS,
    PhishVNSourcePins,
)

PACKAGE = "src/automated_phishing_detection/"
IMPLEMENTATIONS = (
    "_external_source_profile.py",
    "execution_preflight.py",
    "phiusiil.py",
    "protocol_preflight.py",
    "_phishvn_archive.py",
    "phishvn_source.py",
    "saved_phishvn_source.py",
    "phishvn.py",
    "internal_external_handoff.py",
    "_internal_handoff_validation.py",
    "external_source_provenance.py",
    "_external_provenance_payloads.py",
    "_external_source_checkpoints.py",
    "_external_checkpoint_protocol.py",
    "_external_checkpoint_io.py",
    "_external_secondary_checkpoints.py",
    "bound_external_runtime.py",
    "external_producer.py",
    "saved_external_evidence.py",
    "_saved_external_bindings.py",
    "_checkpoint_codec.py",
)


def api():
    name = "automated_phishing_detection._external_source_profile"
    assert find_spec(name) is not None, "missing closed external candidate profile"
    return import_module(name)


@pytest.fixture
def profile_case(monkeypatch):
    source = (Path(__file__).parents[1] / "data/sources.json").read_bytes()
    hashes = {
        PACKAGE + name: sha256(name.encode()).hexdigest() for name in IMPLEMENTATIONS
    }
    hashes["data/sources.json"] = sha256(source).hexdigest()
    binding = preflight.ExecutionBinding(
        Path("/invented/public-checkout"),
        "a" * 40,
        "b" * 64,
        tuple(sorted(hashes.items())),
        '{"invented_runtime":true}',
    )
    case = SimpleNamespace(binding=binding, source=source, hashes=hashes, events=[])

    def read(root, relative):
        assert root == binding.root and relative == "data/sources.json"
        case.events.append("public_source_read")
        return case.source

    monkeypatch.setattr(preflight, "_read_regular", read)
    monkeypatch.setattr(
        preflight, "recheck_binding", lambda value: case.events.append("recheck")
    )
    return case


def resolve(case, **changes):
    return api().resolve_external_source_profile(replace(case.binding, **changes))


def test_closed_profile_binds_fixed_publisher_suffix_and_checkpoint_contracts(
    profile_case,
):
    profile = resolve(profile_case)
    projection = profile.projection()
    assert profile.canonical_bytes == canonical_bytes(projection)
    assert profile.profile_sha256 == sha256(profile.canonical_bytes).hexdigest()
    assert profile.archive_pins == PhishVNSourcePins(
        EXPECTED_FORMAT["archive_sha256"], EXPECTED_FORMAT["archive_size_bytes"]
    )
    assert projection["publisher"] == {
        "expected_format": EXPECTED_FORMAT,
        "archive_members": sorted(EXPECTED_MEMBERS),
        "manifest_copied_order": list(COPIED_MEMBERS),
        "table_members": [
            "data/dataset_url.csv",
            *(f"data/splits/url_{split}.csv" for split in ("train", "val", "test")),
        ],
    }
    assert (
        projection["public_suffix_list"]
        == json.loads(profile_case.source)["public_suffix_list"]
    )
    assert profile.suffix_rules_sha256 == projection["public_suffix_list"]["sha256"]
    assert projection["retention"] == {
        "protocol": PROTOCOL,
        "directory": DIRECTORY,
        "provenance_order": list(PROVENANCE_ORDER),
        "scientific_order": list(SCIENTIFIC_ORDER),
    }


def test_profile_closes_identity_and_implementation_projection(profile_case):
    profile = resolve(profile_case)
    projection = profile.projection()
    assert set(projection) == {
        "schema_version",
        "profile_id",
        "status",
        "protected_evaluation_ready",
        "protected_evaluation_authorized",
        "execution",
        "publisher",
        "public_suffix_list",
        "retention",
        "implementation_sha256",
    }
    assert projection["schema_version"] == 1
    assert projection["profile_id"] == "external-source-candidate-v1"
    assert projection["status"] == "specified_closed_candidate"
    assert profile.protected_evaluation_ready is False
    assert projection["protected_evaluation_ready"] is False
    assert projection["protected_evaluation_authorized"] is False


def test_profile_execution_and_implementation_identity(profile_case):
    projection = resolve(profile_case).projection()
    assert projection["execution"] == {
        "revision": profile_case.binding.revision,
        "execution_contract_sha256": profile_case.binding.contract_sha256,
        "runtime_sha256": sha256(
            profile_case.binding.runtime_json.encode()
        ).hexdigest(),
        "source_spec_sha256": profile_case.hashes["data/sources.json"],
    }
    assert projection["implementation_sha256"] == {
        PACKAGE + name: profile_case.hashes[PACKAGE + name] for name in IMPLEMENTATIONS
    }
    assert profile_case.events == ["recheck", "public_source_read", "recheck"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("revision", "c" * 40),
        ("contract_sha256", "d" * 64),
        ("runtime_json", '{"invented_runtime":false}'),
    ],
)
def test_each_authenticated_execution_identity_changes_profile_digest(
    profile_case, field, value
):
    original = resolve(profile_case)
    assert (
        resolve(profile_case, **{field: value}).profile_sha256
        != original.profile_sha256
    )


@pytest.mark.parametrize("name", IMPLEMENTATIONS)
def test_every_relevant_implementation_digest_is_bound(profile_case, name):
    original = resolve(profile_case)
    hashes = profile_case.hashes | {PACKAGE + name: "e" * 64}
    assert (
        resolve(
            profile_case, source_hashes=tuple(sorted(hashes.items()))
        ).profile_sha256
        != original.profile_sha256
    )
