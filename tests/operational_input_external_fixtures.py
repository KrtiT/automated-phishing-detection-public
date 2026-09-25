"""Invented external lineage metadata, not a claimed observed research process."""

from hashlib import sha256

from external_producer_fixtures import prepared_external
from external_replay_codec_fixtures import encode

from automated_phishing_detection import saved_evidence
from automated_phishing_detection._checkpoint_codec import canonical_bytes
from automated_phishing_detection._external_checkpoint_protocol import PROTOCOL
from automated_phishing_detection._external_completion_records import _LOGICAL_NAMES
from automated_phishing_detection._operational_input_schema import (
    EXECUTION_FIELDS,
    PREPARATION_FIELDS,
)
from automated_phishing_detection.external_evidence_types import ScoredExternalRow
from automated_phishing_detection.external_source_handoff import (
    ObservedExternalCompletion,
    VerifiedExternalSnapshot,
)
from automated_phishing_detection.internal_external_handoff import (
    build_internal_handoff,
)


def digest(content):
    return sha256(content).hexdigest()


def _identity(original, handoff, profile, payloads):
    return {
        **{name: original[name] for name in EXECUTION_FIELDS},
        "kind": "external_evaluation",
        "source_interface": original["source_interface"],
        "checkpoint_protocol": PROTOCOL,
        "source_profile_sha256": digest(profile),
        "archive_sha256": "4" * 64,
        "archive_size_bytes": 1234,
        "suffix_rules_sha256": original["suffix_rules_sha256"],
        "internal_handoff_sha256": digest(handoff.handoff_bytes),
        "internal_overlap_sha256": digest(handoff.overlap_bytes),
        "internal_reservation_sha256": original["reservation_sha256"],
        "reservation_sha256": digest(payloads["attempt/reservation.json"]),
        **{name: original[name] for name in PREPARATION_FIELDS},
    }


def _snapshot(payloads, profile, rows):
    return VerifiedExternalSnapshot(
        tuple(sorted(payloads.items())),
        profile,
        tuple(
            ScoredExternalRow(row, None, (), (), (), 0.0, 0, False, False)
            for row in rows
        ),
        (),
        (),
        (),
        None,
        None,
        (),
        (),
    )


def external_case(observed, count, worker):
    handoff = build_internal_handoff(observed)
    original = observed.public_summary["execution"]
    profile = canonical_bytes(
        {"execution": {name: original[name] for name in EXECUTION_FIELDS}}
    )
    payloads = {name: name.encode() for name in _LOGICAL_NAMES}
    execution = _identity(original, handoff, profile, payloads)
    rows = prepared_external(count).retained
    for name, content in (
        ("internal-source-handoff.json", handoff.handoff_bytes),
        ("internal-source-overlap.json", handoff.overlap_bytes),
        ("retained-test.jsonl", encode(rows)),
        ("bindings.json", canonical_bytes(saved_evidence._EXPECTED_BINDING_CORE)),
    ):
        for directory in ("checkpoints", "evidence"):
            payloads[f"attempt/{directory}/{name}"] = content
    payloads["public-summary.json"] = canonical_bytes({"execution": execution})
    return ObservedExternalCompletion(worker, _snapshot(payloads, profile, rows))
