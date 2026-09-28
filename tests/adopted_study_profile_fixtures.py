"""Canonical saved authorization simulations, never historical execution proof."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

from adopted_study_ledger_fixtures import complete_ledger
from adopted_study_profile_evidence_fixtures import accounting, digest, sources
from study_execution_fixtures import seal
from study_execution_profile_fixtures import profile

from automated_phishing_detection import _adopted_study_records as records
from automated_phishing_detection import _study_root_records as retention
from automated_phishing_detection import execution_receipt as receipt
from automated_phishing_detection._checkpoint_codec import canonical_bytes
from automated_phishing_detection._operational_profile import (
    CandidateOperationalProfile,
    _projection,
)
from automated_phishing_detection._study_admission import decode_admission_frame
from automated_phishing_detection._study_execution_policy import (
    CONTRACT_SHA256,
    DEADLINES,
    policy_bytes,
)
from automated_phishing_detection.execution_preflight import ExecutionBinding
from automated_phishing_detection.study_reduction import ReducedStudyBytes


def make_case(prepared, manifests, *, changed_scope=False):
    original = complete_ledger(prepared, manifests)
    intent = json.loads(original.contents["study-intent.json"])
    source = json.loads(original.contents["source-results.json"])
    base = ExecutionBinding(
        Path("/invented/checkout"),
        original.execution["revision"],
        CONTRACT_SHA256,
        tuple(sorted(intent["operational_profile"]["bound_file_sha256"].items())),
        "{}",
    )
    case = SimpleNamespace(
        original=original,
        base=base,
        root=base.root,
        policy=policy_bytes(),
        operational=CandidateOperationalProfile(canonical_bytes(_projection(base))),
        external=SimpleNamespace(
            profile_sha256=source["accepted_inputs"]["external"]["execution"][
                "source_profile_sha256"
            ]
        ),
    )
    case.profile = profile(case)
    if changed_scope:
        case.profile["source_artifact_scope"]["pyproject.toml"] = "0" * 64
    return case


def _authorization(case):
    profile_bytes = canonical_bytes(case.profile)
    envelope_bytes = canonical_bytes(seal(case, case.profile))
    return SimpleNamespace(
        base=case.base,
        operational=case.operational,
        paths=SimpleNamespace(attempt=Path(case.profile["paths"]["attempt"])),
        profile_bytes=profile_bytes,
        envelope_bytes=envelope_bytes,
        profile_sha256=digest(profile_bytes),
        envelope_sha256=digest(envelope_bytes),
        policy_sha256=digest(case.policy),
    )


def _root(case, authorization):
    identity = records.study_identity(authorization)
    content = receipt._json_bytes(
        {
            "schema_version": 1,
            "status": "reserved",
            "directory": str(authorization.paths.attempt),
            "identity": identity,
        },
        "fixture",
    )
    execution = identity | {"reservation_sha256": digest(content)}
    return content, execution


def _intent(case, authorization, execution):
    scientific = json.loads(case.original.contents["study-intent.json"])
    scientific.update(
        execution=records.scientific_execution(execution),
        operational_profile=case.operational.projection(),
        protective_deadlines_seconds=dict(DEADLINES),
    )
    return canonical_bytes(
        records.envelope("intent", execution)
        | {
            "scientific_intent_bytes": records.encoded(canonical_bytes(scientific)),
            "policy_bytes": records.encoded(case.policy),
            "profile_bytes": records.encoded(authorization.profile_bytes),
            "envelope_bytes": records.encoded(authorization.envelope_bytes),
        }
    )


def _contents(case, authorization, execution, external_pin):
    original = copy.deepcopy(case.original)
    scientific = records.scientific_execution(execution)
    barrier = json.loads(original.contents["prediction-barrier.json"])
    contents = {
        "study-intent.json": _intent(case, authorization, execution),
        "prediction-barrier.json": canonical_bytes(barrier | {"execution": scientific}),
        "source-results.json": sources(
            original.contents["source-results.json"], scientific, external_pin
        ),
    }
    contents["study-accounting.json"] = accounting(original, execution, contents)
    return contents


def _snapshot(authorization, reserved, execution, contents):
    reduced = ReducedStudyBytes(
        canonical_bytes({"groups": []}), canonical_bytes({"primary": {}})
    )
    public = receipt._json_bytes(
        records.public_summary(execution, tuple(contents.items()), reduced), "fixture"
    )
    outputs = contents | dict(
        zip(retention.EXTRA_NAMES, (reduced.operational_bytes, reduced.study_bytes))
    )
    attempt = receipt.Attempt(
        authorization.paths.attempt, execution["reservation_sha256"]
    )
    payloads = {"attempt/reservation.json": reserved, "public-summary.json": public}
    payloads.update({f"attempt/{name}": content for name, content in contents.items()})
    payloads.update(
        {f"attempt/evidence/{name}": content for name, content in outputs.items()}
    )
    payloads.update(
        {
            f"attempt/{name}": content
            for name, content in retention.publication(attempt, outputs, public).items()
        }
    )
    return retention.StudyRootSnapshot(
        attempt.reservation_sha256, tuple(payloads.items())
    )


def saved(case, *, external_pin=None):
    authorization = _authorization(case)
    reserved, execution = _root(case, authorization)
    contents = _contents(
        case, authorization, execution, external_pin or case.external.profile_sha256
    )
    snapshot = _snapshot(authorization, reserved, execution, contents)
    return authorization, snapshot


def child_context(snapshot, role):
    payloads = {
        name: snapshot.payload(f"attempt/{name}")
        for name in (
            "reservation.json",
            "study-intent.json",
            "prediction-barrier.json",
            "source-results.json",
        )
    }
    accounting = json.loads(snapshot.payload("attempt/study-accounting.json"))
    entry = next(
        entry
        for entry in accounting["authorization_ledger"]["admissions"]
        if entry["role"] == role
    )
    frame = decode_admission_frame(records.decoded(entry["frame_bytes"]))
    return frame, payloads
