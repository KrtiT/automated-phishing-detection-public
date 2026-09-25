import json
from dataclasses import replace
from hashlib import sha256

import pytest
from retained_study_preparation_fixtures import api, changed, restore, retained_case

from automated_phishing_detection._checkpoint_codec import canonical_bytes

__all__ = ["api", "retained_case"]


@pytest.mark.parametrize(
    "field", ["expected_reservation_sha256", "expected_completion_sha256"]
)
@pytest.mark.parametrize("value", ["0" * 64, "F" * 64, "short", None, 1, True])
def test_independent_digests_are_required(api, retained_case, field, value):
    with pytest.raises(api.StudyPreparationRestoreError):
        restore(api, retained_case, **{field: value})


@pytest.mark.parametrize(
    "change",
    [
        "list",
        "pair_list",
        "missing",
        "extra",
        "duplicate",
        "reordered",
        "bytearray",
        "wrong_reservation",
    ],
)
def test_closed_snapshot_layout(api, retained_case, change):
    snapshot = retained_case.snapshot
    payloads = snapshot.payloads
    variants = {
        "list": list(payloads),
        "pair_list": (list(payloads[0]), *payloads[1:]),
        "missing": payloads[:-1],
        "extra": (*payloads, ("extra", b"extra")),
        "duplicate": (payloads[0], *payloads[:-1]),
        "reordered": tuple(reversed(payloads)),
        "bytearray": ((payloads[0][0], bytearray(payloads[0][1])), *payloads[1:]),
        "wrong_reservation": payloads,
    }
    retained_case.snapshot = replace(snapshot, payloads=variants[change])
    if change == "wrong_reservation":
        retained_case.snapshot = replace(
            retained_case.snapshot, reservation_sha256="0" * 64
        )
    with pytest.raises(api.StudyPreparationRestoreError):
        restore(api, retained_case)


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_version", True),
        ("schema_version", 1.0),
        ("schema_version", 2),
        ("protocol", "other"),
        ("status", "scoring_complete"),
        ("protected_evaluation_authorized", True),
        ("protected_evaluation_authorized", 0),
        ("scoring_authorized", True),
        ("scoring_authorized", 0),
        ("reservation_sha256", "0" * 64),
        ("execution", {}),
        ("extra", False),
    ],
)
def test_rehashed_completion_schema_rejected(api, retained_case, field, value):
    receipt = json.loads(retained_case.snapshot.payload("preparation-complete.json"))
    receipt[field] = value
    candidate = changed(
        retained_case,
        {"preparation-complete.json": canonical_bytes(receipt)},
        repin=True,
    )
    with pytest.raises(api.StudyPreparationRestoreError):
        restore(api, candidate)


@pytest.mark.parametrize(
    "field",
    [
        "kind",
        "protocol",
        "revision",
        "execution_contract_sha256",
        "runtime_sha256",
        "source_spec_sha256",
        "source_profile_sha256",
        "preparation_summary_sha256",
        "source_csv_sha256",
        "suffix_rules_sha256",
        "partition_sha256",
        "archive_sha256",
        "archive_size_bytes",
    ],
)
def test_missing_expected_identity_field_rejected(api, retained_case, field):
    identity = dict(retained_case.identity)
    del identity[field]
    with pytest.raises(api.StudyPreparationRestoreError):
        restore(api, retained_case, expected_identity=identity)


@pytest.mark.parametrize(
    "field,value",
    [
        ("kind", "scoring"),
        ("protocol", "other"),
        ("revision", "a" * 64),
        ("source_csv_sha256", "A" * 64),
        ("archive_size_bytes", True),
        ("archive_size_bytes", 1.0),
        ("archive_size_bytes", 0),
        ("extra", False),
    ],
)
def test_invalid_identity_types_rejected(api, retained_case, field, value):
    identity = dict(retained_case.identity) | {field: value}
    with pytest.raises(api.StudyPreparationRestoreError):
        restore(api, retained_case, expected_identity=identity)


@pytest.mark.parametrize("field", ["source_spec_bytes", "preparation_summary_bytes"])
@pytest.mark.parametrize(
    "value", [b"private-secret-canary", bytearray(b"mutable"), None]
)
def test_public_buffers_are_independently_bound(api, retained_case, field, value):
    with pytest.raises(api.StudyPreparationRestoreError) as caught:
        restore(api, retained_case, **{field: value})
    assert "private-secret" not in str(caught.value)


def test_consistent_rehash_cannot_replace_parent_completion(api, retained_case):
    candidate = changed(retained_case, {"feasibility.json": b"{}\n"})
    with pytest.raises(api.StudyPreparationRestoreError):
        restore(api, candidate)


@pytest.mark.parametrize(
    "content", [b'{"schema_version":1,"schema_version":1}\n', b"{}", b'{"value":NaN}\n']
)
def test_noncanonical_or_duplicate_completion_rejected(api, retained_case, content):
    payloads = tuple(
        (name, content if name == "preparation-complete.json" else value)
        for name, value in retained_case.snapshot.payloads
    )
    retained_case.snapshot = replace(retained_case.snapshot, payloads=payloads)
    retained_case.completion = sha256(content).hexdigest()
    with pytest.raises(api.StudyPreparationRestoreError):
        restore(api, retained_case)
