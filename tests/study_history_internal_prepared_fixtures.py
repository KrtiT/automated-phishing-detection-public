"""Republish invented science under an original retained-preparation context."""

import json
from hashlib import sha256

from study_history_internal_fixtures import republish

from automated_phishing_detection import execution_receipt
from automated_phishing_detection._checkpoint_codec import canonical_bytes
from automated_phishing_detection._internal_scientific_protocol import (
    SCIENTIFIC_CHECKPOINT_NAMES,
    SOURCE_CHECKPOINT_NAMES,
    completion_bytes,
    expected_counts,
)


def _receipt(case, name, **changes):
    value = json.loads(case.payloads[name]) | changes
    case.payloads[name] = execution_receipt._json_bytes(value, "invented_history")


def _checkpoint_context(case, identity, reservation, public):
    name = "attempt/checkpoints/source-reconstruction.json"
    value = json.loads(case.payloads[name])
    value.update(execution=identity, reservation_sha256=reservation)
    case.payloads[name] = canonical_bytes(value)
    public["checkpoint_sha256"] = {
        member: sha256(case.payloads[f"attempt/checkpoints/{member}"]).hexdigest()
        for member in SOURCE_CHECKPOINT_NAMES
    }
    name = "attempt/scientific-checkpoints/context.json"
    value = json.loads(case.payloads[name])
    value.update(
        execution=identity,
        reservation_sha256=reservation,
        source_checkpoint_sha256=public["checkpoint_sha256"],
    )
    case.payloads[name] = canonical_bytes(value)


def as_prepared_history(case):
    identity = {
        name: value
        for name, value in case.execution.items()
        if name != "reservation_sha256"
    }
    identity.update(
        source_interface="retained_study_preparation_v1",
        study_preparation_reservation_sha256="a" * 64,
        study_preparation_complete_sha256="b" * 64,
    )
    _receipt(case, "attempt/reservation.json", identity=identity)
    reservation = sha256(case.payloads["attempt/reservation.json"]).hexdigest()
    case.execution = identity | {"reservation_sha256": reservation}
    public = json.loads(case.payloads["public-summary.json"])
    public["execution"] = case.execution
    _checkpoint_context(case, identity, reservation, public)
    _completion(case, reservation, public["row_count"])
    for name in ("attempt/finalize.claim", "attempt/outcome.json"):
        _receipt(case, name, reservation_sha256=reservation)
    republish(case, public)
    return case


def _completion(case, reservation, count):
    preceding = {
        member: case.payloads[f"attempt/scientific-checkpoints/{member}"]
        for member in SCIENTIFIC_CHECKPOINT_NAMES - {"completion.json"}
    }
    case.payloads["attempt/scientific-checkpoints/completion.json"] = completion_bytes(
        preceding, reservation, *expected_counts(count)
    )
