"""Full original membership remains bound through the byte-only projection."""

import json
from dataclasses import replace
from hashlib import sha256

import pytest
import test_internal_external_handoff as fixtures
import test_internal_external_handoff_validation as validation

from automated_phishing_detection._checkpoint_codec import canonical_bytes

completion = fixtures.completion
handoff_api = fixtures.handoff_api
inputs = fixtures.inputs
observed_worker = fixtures.observed_worker
published = fixtures.published
runner = fixtures.runner
verifier = fixtures.verifier
payloads = validation.payloads


def verify_overlap(api, payloads, overlap, *, rehash=True):
    envelope = json.loads(payloads.handoff_bytes)
    content = canonical_bytes(overlap)
    if rehash:
        envelope["snapshot_sha256"]["attempt/checkpoints/source-overlap.json"] = sha256(
            content
        ).hexdigest()
    changed = replace(
        payloads, handoff_bytes=canonical_bytes(envelope), overlap_bytes=content
    )
    return fixtures.verified_domains(api, changed)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema_version", True),
        ("schema_version", 1.0),
        ("algorithm_id", "retained-only"),
        ("scope", "retained_domains"),
        ("extra", True),
        ("input_hashes", {}),
        ("reconstructed_output_sha256", {}),
        ("rows", {}),
        ("domains", []),
    ],
)
def test_rehashed_overlap_schema_mutations_rejected(
    handoff_api, payloads, field, value
):
    overlap = json.loads(payloads.overlap_bytes)
    overlap[field] = value
    with pytest.raises(ValueError):
        verify_overlap(handoff_api, payloads, overlap)


@pytest.mark.parametrize(
    "field",
    [
        "source_csv_sha256",
        "suffix_rules_sha256",
        "source_spec_sha256",
        "preparation_summary_sha256",
    ],
)
def test_rehashed_overlap_identity_must_match_execution(handoff_api, payloads, field):
    overlap = json.loads(payloads.overlap_bytes)
    overlap["input_hashes"][field] = "e" * 64
    with pytest.raises(ValueError):
        verify_overlap(handoff_api, payloads, overlap)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("source_ordinal", True),
        ("source_ordinal", 1.0),
        ("source_ordinal", 2),
        ("record_id", "changed"),
        ("canonical_url_sha256", "X" * 64),
        ("registrable_domain", "UPPERCASE.COM"),
        ("status", "retained_only"),
        ("extra", None),
    ],
)
def test_rehashed_original_row_membership_is_checked(
    handoff_api, payloads, field, value
):
    overlap = json.loads(payloads.overlap_bytes)
    overlap["rows"][0][field] = value
    with pytest.raises(ValueError):
        verify_overlap(handoff_api, payloads, overlap)


@pytest.mark.parametrize(
    "change",
    ["drop_domain", "domain_order", "duplicate_domain", "row_order", "row_gap"],
)
def test_domains_must_equal_all_original_valid_membership(
    handoff_api, payloads, change
):
    overlap = json.loads(payloads.overlap_bytes)
    if change == "drop_domain":
        overlap["domains"].remove("quarantined-label.com")
    elif change == "domain_order":
        overlap["domains"].reverse()
    elif change == "duplicate_domain":
        overlap["domains"].append(overlap["domains"][-1])
    elif change == "row_order":
        overlap["rows"].reverse()
    else:
        overlap["rows"].pop(0)
    with pytest.raises(ValueError):
        verify_overlap(handoff_api, payloads, overlap)


def test_overlap_change_rejected_without_parent_identity_change(handoff_api, payloads):
    overlap = json.loads(payloads.overlap_bytes)
    overlap["domains"].remove("quarantined-label.com")
    with pytest.raises(ValueError):
        verify_overlap(handoff_api, payloads, overlap, rehash=False)


def test_quarantined_only_omission_cannot_reuse_parent_digest(handoff_api, payloads):
    overlap = json.loads(payloads.overlap_bytes)
    overlap["domains"].remove("quarantined-label.com")
    content = canonical_bytes(overlap)
    envelope = json.loads(payloads.handoff_bytes)
    envelope["snapshot_sha256"]["attempt/checkpoints/source-overlap.json"] = sha256(
        content
    ).hexdigest()
    with pytest.raises(ValueError, match="handoff_identity_mismatch"):
        handoff_api.verify_internal_handoff(
            canonical_bytes(envelope),
            content,
            expected_handoff_sha256=sha256(payloads.handoff_bytes).hexdigest(),
        )


def test_reconstructed_partition_hash_is_execution_bound(handoff_api, payloads):
    overlap = json.loads(payloads.overlap_bytes)
    overlap["reconstructed_output_sha256"]["group_test.jsonl"] = "d" * 64
    with pytest.raises(ValueError):
        verify_overlap(handoff_api, payloads, overlap)
