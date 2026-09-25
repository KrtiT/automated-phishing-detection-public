"""Cross-stage execution linkage cannot be supplied by an unchecked profile."""

import json
from dataclasses import replace

import pytest
import test_external_source_records as fixtures

from automated_phishing_detection._checkpoint_codec import canonical_bytes
from automated_phishing_detection._external_source_profile import (
    CandidateExternalProfile,
)

inputs = fixtures.inputs
published = fixtures.published
runner = fixtures.runner
verifier = fixtures.verifier
observed_worker = fixtures.observed_worker
completion = fixtures.completion
handoff_api = fixtures.handoff_api
records_case = fixtures.records_case


def identity(case, **changes):
    values = {"binding": case.binding, "profile": case.profile, "handoff": case.handoff}
    return fixtures.api().external_identity(**(values | changes))


@pytest.mark.parametrize(
    "field",
    [
        "revision",
        "execution_contract_sha256",
        "runtime_sha256",
        "source_spec_sha256",
        "suffix_rules_sha256",
        "preparation_summary_sha256",
    ],
)
def test_valid_internal_handoff_must_match_external_expected_chain(records_case, field):
    case = records_case
    projection = case.profile.projection()
    binding = case.binding
    pins = dict(binding.source_hashes)
    if field == "revision":
        binding = replace(binding, revision="a" * 40)
        projection["execution"][field] = binding.revision
    elif field == "execution_contract_sha256":
        binding = replace(binding, contract_sha256="a" * 64)
        projection["execution"][field] = binding.contract_sha256
    elif field == "runtime_sha256":
        binding = replace(binding, runtime_json='{"changed":true}')
        projection["execution"][field] = fixtures.digest(binding.runtime_json.encode())
    elif field == "suffix_rules_sha256":
        projection["public_suffix_list"]["sha256"] = "a" * 64
    else:
        name = (
            "data/sources.json"
            if field == "source_spec_sha256"
            else "reports/phiusiil-preparation-summary.json"
        )
        pins[name] = "a" * 64
        binding = replace(binding, source_hashes=tuple(pins.items()))
        if field == "source_spec_sha256":
            projection["execution"][field] = pins[name]
    profile = CandidateExternalProfile(canonical_bytes(projection))
    with pytest.raises(fixtures.api().ExternalSourceExecutionError):
        identity(case, binding=binding, profile=profile)


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_version", True),
        ("schema_version", 2),
        ("profile_id", "unknown"),
        ("status", "authorized"),
        ("protected_evaluation_ready", True),
        ("protected_evaluation_ready", 0),
        ("protected_evaluation_authorized", True),
        ("protected_evaluation_authorized", 0),
    ],
)
def test_profile_requires_exact_false_candidate_metadata(records_case, field, value):
    projection = records_case.profile.projection()
    projection[field] = value
    with pytest.raises(fixtures.api().ExternalSourceExecutionError):
        identity(
            records_case, profile=CandidateExternalProfile(canonical_bytes(projection))
        )


@pytest.mark.parametrize("change", ["wrong_revision", "missing", "extra", "wrong_type"])
def test_profile_execution_matches_exact_binding_projection(records_case, change):
    projection = records_case.profile.projection()
    if change == "wrong_revision":
        projection["execution"]["revision"] = "e" * 40
    elif change == "missing":
        projection["execution"].pop("runtime_sha256")
    elif change == "extra":
        projection["execution"]["extra"] = True
    else:
        projection["execution"] = []
    with pytest.raises(fixtures.api().ExternalSourceExecutionError):
        identity(
            records_case, profile=CandidateExternalProfile(canonical_bytes(projection))
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("archive_sha256", "invalid"),
        ("archive_size_bytes", True),
        ("archive_size_bytes", 1.0),
        ("archive_size_bytes", 0),
        ("archive_size_bytes", -1),
    ],
)
def test_profile_archive_pins_have_exact_valid_shapes(records_case, field, value):
    projection = records_case.profile.projection()
    projection["publisher"]["expected_format"][field] = value
    with pytest.raises(fixtures.api().ExternalSourceExecutionError):
        identity(
            records_case, profile=CandidateExternalProfile(canonical_bytes(projection))
        )


@pytest.mark.parametrize("field", ["binding", "profile", "handoff"])
def test_records_reject_untyped_context(records_case, field):
    with pytest.raises(fixtures.api().ExternalSourceExecutionError):
        identity(records_case, **{field: None})


@pytest.mark.parametrize(
    "field,value",
    [
        ("handoff_bytes", b"private-canary"),
        ("overlap_bytes", b"private-canary"),
        ("handoff_bytes", bytearray(b"private-canary")),
    ],
)
def test_retained_handoff_bytes_are_verified_before_projection(
    records_case, field, value
):
    handoff = replace(records_case.handoff, **{field: value})
    with pytest.raises(fixtures.api().ExternalSourceExecutionError) as rejected:
        identity(records_case, handoff=handoff)
    assert "private-canary" not in str(rejected.value)


def test_noncanonical_profile_bytes_are_rejected(records_case):
    content = json.dumps(records_case.profile.projection(), indent=2).encode()
    with pytest.raises(fixtures.api().ExternalSourceExecutionError):
        identity(records_case, profile=CandidateExternalProfile(content))
