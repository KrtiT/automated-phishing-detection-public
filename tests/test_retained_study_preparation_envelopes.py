import json
from dataclasses import replace
from hashlib import sha256

import pytest
from retained_study_preparation_fixtures import api, restore, retained_case

from automated_phishing_detection._checkpoint_codec import canonical_bytes
from automated_phishing_detection._study_preparation_records import (
    PreparedStudySnapshot,
)
from automated_phishing_detection.study_preparation_retention import PREPARATION_ORDER

__all__ = ["api", "retained_case"]


def receipt_candidate(case, content):
    outputs = dict(case.snapshot.payloads) | {"preparation-complete.json": content}
    case.snapshot = replace(
        case.snapshot,
        payloads=tuple((name, outputs[name]) for name in PREPARATION_ORDER),
    )
    case.completion = sha256(content).hexdigest()
    return case


@pytest.mark.parametrize(
    "field",
    [
        "schema_version",
        "protocol",
        "status",
        "protected_evaluation_authorized",
        "scoring_authorized",
        "execution",
        "reservation_sha256",
        "input_sha256",
    ],
)
def test_every_completion_field_is_required(api, retained_case, field):
    receipt = json.loads(retained_case.snapshot.payload("preparation-complete.json"))
    del receipt[field]
    with pytest.raises(api.StudyPreparationRestoreError):
        restore(api, receipt_candidate(retained_case, canonical_bytes(receipt)))


@pytest.mark.parametrize("name", PREPARATION_ORDER[:-1])
@pytest.mark.parametrize("mutation", ["missing", "bool", "wrong"])
def test_each_committed_payload_hash_is_required(api, retained_case, name, mutation):
    receipt = json.loads(retained_case.snapshot.payload("preparation-complete.json"))
    if mutation == "missing":
        del receipt["input_sha256"][name]
    else:
        receipt["input_sha256"][name] = True if mutation == "bool" else "0" * 64
    with pytest.raises(api.StudyPreparationRestoreError):
        restore(api, receipt_candidate(retained_case, canonical_bytes(receipt)))


@pytest.mark.parametrize("content", [None, True, [], {}, {"extra": "0" * 64}])
def test_hash_inventory_has_no_alternative_schema(api, retained_case, content):
    receipt = json.loads(retained_case.snapshot.payload("preparation-complete.json"))
    receipt["input_sha256"] = content
    with pytest.raises(api.StudyPreparationRestoreError):
        restore(api, receipt_candidate(retained_case, canonical_bytes(receipt)))


@pytest.mark.parametrize("mutation", ["whitespace", "newline", "duplicate", "order"])
def test_complete_receipt_has_one_exact_canonical_encoding(
    api, retained_case, mutation
):
    content = retained_case.snapshot.payload("preparation-complete.json")
    receipt = json.loads(content)
    variants = {
        "whitespace": json.dumps(receipt, indent=2).encode() + b"\n",
        "newline": content[:-1],
        "duplicate": content.replace(
            b'{"execution":', b'{"protocol":"study-preparation-v1","execution":', 1
        ),
        "order": json.dumps(
            dict(reversed(tuple(receipt.items()))), separators=(",", ":")
        ).encode()
        + b"\n",
    }
    with pytest.raises(api.StudyPreparationRestoreError):
        restore(api, receipt_candidate(retained_case, variants[mutation]))


@pytest.mark.parametrize("kind", [str, bytes, tuple, dict, PreparedStudySnapshot])
def test_subclasses_are_not_accepted_as_exact_records(api, retained_case, kind):
    subtype = type("Subclass", (kind,), {})
    changes = {}
    if kind is dict:
        changes["expected_identity"] = subtype(retained_case.identity)
    elif kind is str:
        changes["expected_reservation_sha256"] = subtype(retained_case.reservation)
    elif kind is bytes:
        changes["source_spec_bytes"] = subtype(
            retained_case.source.buffers["source_spec_bytes"]
        )
    elif kind is tuple:
        retained_case.snapshot = replace(
            retained_case.snapshot, payloads=subtype(retained_case.snapshot.payloads)
        )
    else:
        retained_case.snapshot = subtype(
            retained_case.reservation, retained_case.snapshot.payloads
        )
    with pytest.raises(api.StudyPreparationRestoreError):
        restore(api, retained_case, **changes)
