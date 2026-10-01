"""Invented source receipt bytes; scientific payloads are intentionally opaque."""

from hashlib import sha256

from automated_phishing_detection import _internal_handoff_validation as internal
from automated_phishing_detection import execution_receipt as receipt
from automated_phishing_detection._checkpoint_codec import canonical_bytes
from automated_phishing_detection._external_completion_records import _LOGICAL_NAMES
from automated_phishing_detection._internal_scientific_protocol import (
    SCIENTIFIC_OUTPUT_NAMES,
    SOURCE_CHECKPOINT_NAMES,
)
from automated_phishing_detection.source_completion import _PUBLIC_NAMES


def digest(content):
    return sha256(content).hexdigest()


def hashes(values):
    return {name: digest(content) for name, content in values.items()}


def encoded(value):
    return receipt._json_bytes(value, "invented_history")


def reserve(values, execution, path):
    identity = {
        name: value for name, value in execution.items() if name != "reservation_sha256"
    }
    values["attempt/reservation.json"] = encoded(
        dict(schema_version=1, status="reserved", directory=path, identity=identity)
    )
    execution["reservation_sha256"] = digest(values["attempt/reservation.json"])


def finish(values, public, private):
    values["public-summary.json"] = encoded(public)
    common = dict(
        schema_version=1, reservation_sha256=public["execution"]["reservation_sha256"]
    )
    values["attempt/finalize.claim"] = encoded(common | {"operation": "completion"})
    values["attempt/outcome.json"] = encoded(
        common
        | {
            "status": "completion_prepared",
            "public_summary_sha256": digest(values["public-summary.json"]),
            "private_sha256": hashes(private),
        }
    )


def internal_source(metadata, path):
    values = {name: name.encode() for name in internal.SNAPSHOT_NAMES}
    for name in SCIENTIFIC_OUTPUT_NAMES:
        for directory in ("evidence", "scientific-checkpoints"):
            values[f"attempt/{directory}/{name}"] = b"opaque science " + name.encode()
    execution = metadata["execution"]
    reserve(values, execution, path)
    private = {
        name: values[f"attempt/evidence/{name}"] for name in SCIENTIFIC_OUTPUT_NAMES
    }
    public = dict.fromkeys(_PUBLIC_NAMES)
    public.update(
        schema_version=4,
        status="internal_evidence_published",
        source_binding="authenticated_public_preparation",
        protected_evaluation_authorized=False,
        execution=execution,
        private_sha256=hashes(private),
        checkpoint_sha256={
            name: digest(values[f"attempt/checkpoints/{name}"])
            for name in SOURCE_CHECKPOINT_NAMES
        },
    )
    finish(values, public, private)
    metadata["snapshot_sha256"] = hashes(values)
    return values


def external_source(metadata, original, original_payloads, path, source_profile):
    values = {name: name.encode() for name in _LOGICAL_NAMES}
    outputs = {
        name.removeprefix("attempt/evidence/"): b"opaque science " + name.encode()
        for name in values
        if name.startswith("attempt/evidence/")
    }
    outputs["internal-source-handoff.json"] = canonical_bytes(original)
    outputs["internal-source-overlap.json"] = original_payloads[internal.OVERLAP_NAME]
    for name, content in outputs.items():
        for directory in ("checkpoints", "evidence"):
            values[f"attempt/{directory}/{name}"] = content
    execution = metadata["execution"]
    execution.update(
        internal_handoff_sha256=digest(outputs["internal-source-handoff.json"]),
        internal_overlap_sha256=digest(outputs["internal-source-overlap.json"]),
        internal_reservation_sha256=original["execution"]["reservation_sha256"],
    )
    reserve(values, execution, path)
    public = _external_public(execution, source_profile, outputs)
    finish(values, public, outputs)
    metadata["snapshot_sha256"] = hashes(values)
    return values


def _external_public(execution, source_profile, outputs):
    return dict(
        schema_version=1,
        status="external_evidence_published",
        source_binding="authenticated_retained_preparation_with_parent_declared_internal_handoff",
        protected_evaluation_authorized=False,
        execution=execution,
        source_profile=source_profile,
        publisher=None,
        composition=None,
        checkpoint_sha256=hashes(outputs),
        private_sha256=hashes(outputs),
    )
