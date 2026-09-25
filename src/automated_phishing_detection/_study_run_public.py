"""Join root projection bytes without re-running scientific or preparation work."""

from hashlib import sha256

from . import _study_root_records as retention
from . import _study_run_schema as schema
from ._checkpoint_codec import canonical_bytes

_BARRIER = {
    "study_preparation_reservation_sha256",
    "study_preparation_complete_sha256",
    "feasibility",
    "feasibility_sha256",
    "predictions_started",
}
_ACCOUNTING = {"stage", "internal_status", "external_status", "cells"}


def _checkpoints(checkpoints, success):
    schema.require(type(checkpoints) is tuple)
    schema.require(
        all(type(member) is tuple and len(member) == 2 for member in checkpoints)
    )
    names = retention.ORDER if success else retention.HOLD_ORDER
    schema.require(tuple(name for name, content in checkpoints) == names)
    schema.require(
        all(
            type(name) is str and type(content) is bytes
            for name, content in checkpoints
        )
    )
    return dict(checkpoints)


def _intent(content, execution):
    value = schema.record(
        content, execution, {"operational_profile", "protective_deadlines_seconds"}
    )
    schema.require(value["status"] == "intent")
    profile = value["operational_profile"]
    schema.require(type(profile) is dict)
    schema.require(
        sha256(canonical_bytes(profile)).hexdigest()
        == execution["operational_profile_sha256"]
    )
    schema.same(
        profile["execution"], {name: execution[name] for name in schema.EXECUTION}
    )
    schema.deadlines(value["protective_deadlines_seconds"])


def _barrier(content, execution, success):
    value = schema.record(content, execution, _BARRIER)
    expected = "necessary_capacity_present" if success else "whole_study_hold"
    schema.require(
        value["status"] == expected and value["predictions_started"] is False
    )
    for name in (
        "study_preparation_reservation_sha256",
        "study_preparation_complete_sha256",
    ):
        schema.operational.digest(value[name])
    feasibility = value["feasibility"]
    schema.feasibility(feasibility)
    schema.require(bool(feasibility["shortages"]) is not success)
    schema.require(
        value["feasibility_sha256"] == sha256(canonical_bytes(feasibility)).hexdigest()
    )
    return value


def _sources(content, execution, barrier):
    value = schema.record(
        content, execution, {"accepted_inputs", "accepted_inputs_sha256"}
    )
    schema.require(value["status"] == "sources_accepted")
    metadata = value["accepted_inputs"]
    schema.accepted_metadata(metadata, execution)
    schema.require(
        value["accepted_inputs_sha256"] == sha256(canonical_bytes(metadata)).hexdigest()
    )
    original = metadata["internal"]["execution"]
    for name in (
        "study_preparation_reservation_sha256",
        "study_preparation_complete_sha256",
    ):
        schema.require(original[name] == barrier[name])


def _reductions(reduced):
    from .study_reduction import ReducedStudyBytes

    schema.require(type(reduced) is ReducedStudyBytes)
    return {
        "operational-summary.json": reduced.operational_bytes,
        "study-evidence.json": reduced.study_bytes,
    }


def public_summary(execution, checkpoints, reduced):
    schema.execution(execution)
    success = reduced is not None
    contents = _checkpoints(checkpoints, success)
    _intent(contents[retention.ORDER[0]], execution)
    barrier = _barrier(contents[retention.ORDER[1]], execution, success)
    accounting = schema.record(contents[retention.ORDER[3]], execution, _ACCOUNTING)
    schema.accounting(accounting)
    schema.require(
        accounting["status"] == ("matrix_accepted" if success else "whole_study_hold")
    )
    extra = _reductions(reduced) if success else {}
    if success:
        _sources(contents[retention.ORDER[2]], execution, barrier)
    result = schema.envelope(
        "study_evidence_published" if success else "whole_study_hold", execution
    )
    result.update(
        accounting_sha256=sha256(contents[retention.ORDER[3]]).hexdigest(),
        private_sha256=retention.hashes(contents | extra),
    )
    result.update(
        {
            "operational": schema.load(extra[retention.EXTRA_NAMES[0]]),
            "study": schema.load(extra[retention.EXTRA_NAMES[1]]),
        }
        if success
        else {"feasibility": barrier["feasibility"]}
    )
    return schema.load(canonical_bytes(result))
