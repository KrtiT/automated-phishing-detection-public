"""Coherently relink invented saved files without publishing or rerunning work."""

import json
from hashlib import sha256

from automated_phishing_detection import execution_receipt
from automated_phishing_detection._checkpoint_codec import canonical_bytes
from automated_phishing_detection._external_provenance_payloads import (
    PREPARED_NAMES,
    PROVENANCE_NAMES,
)
from automated_phishing_detection._external_source_records import build_external_public
from automated_phishing_detection._saved_external_bindings import PRIVATE_OUTPUTS


def digest(content):
    return sha256(content).hexdigest()


def encoded(value):
    return execution_receipt._json_bytes(value, "invented")


def replace_private(case, name, content, *, link_provenance=False):
    outputs = case.filesinputs.copy()
    outputs[name] = content
    if link_provenance:
        receipt = json.loads(outputs["external-source-reconstruction.json"])
        receipt["provenance_sha256"] = {
            member: digest(outputs[member])
            for member in PROVENANCE_NAMES - {"external-source-reconstruction.json"}
        }
        receipt["prepared_sha256"] = {
            member: digest(outputs[member]) for member in PREPARED_NAMES
        }
        outputs["external-source-reconstruction.json"] = canonical_bytes(receipt)
    composition = json.loads(canonical_bytes(case.produced.public_summary))
    composition["private_sha256"] = {
        member: digest(outputs[member]) for member in PRIVATE_OUTPUTS
    }
    public = build_external_public(
        case.binding,
        case.profile,
        case.identity,
        case.attempt.reservation_sha256,
        outputs,
        composition,
    )
    rewrite(case, outputs, public)


def rewrite(case, outputs, public):
    for directory in ("checkpoints", "evidence"):
        for name, content in outputs.items():
            (case.paths.attempt / directory / name).write_bytes(content)
    content = encoded(public)
    case.paths.public_summary.write_bytes(content)
    outcome = {
        "schema_version": 1,
        "status": "completion_prepared",
        "reservation_sha256": case.attempt.reservation_sha256,
        "public_summary_sha256": digest(content),
        "private_sha256": {name: digest(payload) for name, payload in outputs.items()},
    }
    (case.paths.attempt / "outcome.json").write_bytes(encoded(outcome))
