"""Invented stopped prefixes built exclusively from synthetic profile fixtures."""

import json
from hashlib import sha256
from types import SimpleNamespace

from adopted_study_profile_fixtures import (
    _authorization,
    _contents,
    _root,
    make_case,
)

from automated_phishing_detection import _adopted_study_records as records
from automated_phishing_detection import execution_receipt as receipt
from automated_phishing_detection._checkpoint_codec import canonical_bytes
from automated_phishing_detection._study_root_records import StudyRootSnapshot


def make_stopped(prepared, manifests, *, prefix=1, stopped_admissions=0):
    original = make_case(prepared, manifests)
    authorization = _authorization(original)
    reserved, execution = _root(original, authorization)
    contents = _contents(
        original, authorization, execution, original.external.profile_sha256
    )
    accounting = json.loads(contents["study-accounting.json"])
    scientific = json.loads(records.decoded(accounting["scientific_accounting_bytes"]))
    _truncate(accounting, scientific, prefix, stopped_admissions)
    case = SimpleNamespace(
        authorization=authorization,
        execution=execution,
        contents=contents,
        accounting=accounting,
        scientific=scientific,
        payloads={"attempt/reservation.json": reserved},
    )
    case.payloads.update(_finalization(execution["reservation_sha256"]))
    refresh_accounting(case)
    return case


def _truncate(accounting, scientific, prefix, stopped_admissions):
    accounting["status"] = "failed"
    scientific.update(status="failed", stage="cell_execution")
    for index, projection in enumerate(scientific["cells"]):
        if index < prefix:
            continue
        for name in set(projection) - {"cell", "status"}:
            projection[name] = None
        projection["status"] = "stopped" if index == prefix else "unattempted"
        if index == prefix:
            projection["stage"] = "cell_execution"
    ledger = accounting["authorization_ledger"]
    completed = 2 + 2 * prefix
    ledger["admissions"] = ledger["admissions"][: completed + stopped_admissions]
    for entry in ledger["admissions"][completed:]:
        entry["accepted"] = False
    ledger["cell_acceptances"] = ledger["cell_acceptances"][:prefix]


def _finalization(reservation):
    common = {"schema_version": 1, "reservation_sha256": reservation}
    return {
        "attempt/finalize.claim": receipt._json_bytes(
            common | {"operation": "failure"}, "fixture"
        ),
        "attempt/outcome.json": receipt._json_bytes(
            common
            | {
                "status": "failed",
                "stage": "cell_execution",
                "error_type": "cancelled",
            },
            "fixture",
        ),
    }


def refresh_accounting(case):
    case.accounting["scientific_accounting_bytes"] = records.encoded(
        canonical_bytes(case.scientific)
    )
    case.contents["study-accounting.json"] = canonical_bytes(case.accounting)
    case.payloads.update(
        {f"attempt/{name}": content for name, content in case.contents.items()}
    )
    refresh_snapshot(case)


def refresh_snapshot(case):
    case.snapshot = StudyRootSnapshot(
        case.execution["reservation_sha256"], tuple(case.payloads.items())
    )
    case.pins = {
        name: sha256(content).hexdigest() for name, content in case.payloads.items()
    }


def verify(case, **overrides):
    from automated_phishing_detection.stopped_study_authorization import (
        verify_stopped_study_authorization,
    )

    arguments = dict(
        expected_profile_sha256=case.authorization.profile_sha256,
        expected_envelope_sha256=case.authorization.envelope_sha256,
        expected_snapshot_sha256=case.pins,
    )
    return verify_stopped_study_authorization(case.snapshot, **(arguments | overrides))


def change_frame(case, position, **changes):
    entry = case.accounting["authorization_ledger"]["admissions"][position]
    value = json.loads(records.decoded(entry["frame_bytes"])) | changes
    content = receipt._json_bytes(value, "fixture")
    entry.update(
        frame_bytes=records.encoded(content), frame_sha256=sha256(content).hexdigest()
    )
    refresh_accounting(case)


def repin_evidence(case, payload_key, file_name, value):
    content = canonical_bytes(value)
    ledger = case.accounting["authorization_ledger"]
    ledger["cell_acceptances"][0][payload_key] = records.encoded(content)
    projection = case.scientific["cells"][0]
    for prefix in ("attempt", "attempt/evidence"):
        projection["snapshot_sha256"][f"{prefix}/{file_name}"] = sha256(
            content
        ).hexdigest()
    if payload_key == "observation_bytes":
        projection["observation_sha256"] = sha256(content).hexdigest()
    refresh_accounting(case)
