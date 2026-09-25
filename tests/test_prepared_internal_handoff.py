"""Retained preparation is a second exact internal identity, not optional keys."""

import pytest

from automated_phishing_detection import _internal_handoff_validation as validation

LINKS = (
    "study_preparation_reservation_sha256",
    "study_preparation_complete_sha256",
)


def identity():
    hashes = {name: "a" * 64 for name in validation.SNAPSHOT_NAMES}
    value = {
        **validation.EXECUTION_CONSTANTS,
        **{name: "a" * 64 for name in validation.EXECUTION_HASHES},
        "revision": "b" * 40,
        "source_interface": "retained_study_preparation_v1",
        **{name: "c" * 64 for name in LINKS},
    }
    return value, hashes


def test_retained_identity_preserves_existing_snapshot_inventory():
    value, hashes = identity()
    validation.execution(value, hashes)
    assert len(hashes) == 35


@pytest.mark.parametrize("field", LINKS)
@pytest.mark.parametrize("value", [None, True, 1, "A" * 64, "x" * 64, "a" * 63])
def test_retained_identity_requires_both_exact_digest_fields(field, value):
    execution, hashes = identity()
    execution[field] = value
    with pytest.raises(validation.InternalHandoffError):
        validation.execution(execution, hashes)


@pytest.mark.parametrize("field", LINKS)
def test_retained_identity_rejects_missing_link(field):
    execution, hashes = identity()
    execution.pop(field)
    with pytest.raises(validation.InternalHandoffError):
        validation.execution(execution, hashes)


@pytest.mark.parametrize("variant", ["original_csv_reconstruction_v1", "unknown"])
def test_preparation_links_cannot_be_attached_to_other_interfaces(variant):
    execution, hashes = identity()
    execution["source_interface"] = variant
    with pytest.raises(validation.InternalHandoffError):
        validation.execution(execution, hashes)


@pytest.mark.parametrize("field", ["extra", "partition_sha256", "reservation_sha256"])
def test_retained_identity_keeps_closed_schema_and_scientific_links(field):
    execution, hashes = identity()
    execution[field] = "d" * 64
    with pytest.raises(validation.InternalHandoffError):
        validation.execution(execution, hashes)
