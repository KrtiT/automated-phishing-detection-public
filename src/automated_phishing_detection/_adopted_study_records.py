"""Versioned authorization wrappers leave original scientific records unchanged."""

import base64
from hashlib import sha256

from . import _study_root_records as retention
from . import _study_run_records as original
from . import _study_run_schema as schema
from ._checkpoint_codec import canonical_bytes

PROTOCOL = "adopted-study-root-v1"
PINS = {"study_profile_sha256", "adoption_envelope_sha256", "study_policy_sha256"}


def encoded(content):
    schema.require(type(content) is bytes)
    return base64.b64encode(content).decode("ascii")


def decoded(value):
    schema.require(type(value) is str)
    content = base64.b64decode(value, validate=True)
    schema.require(encoded(content) == value)
    return content


def scientific_execution(execution, *, reserved=True):
    schema.require(type(execution) is dict and execution.get("protocol") == PROTOCOL)
    schema.require(PINS <= set(execution))
    for name in PINS:
        schema.operational.digest(execution[name])
    result = {name: value for name, value in execution.items() if name not in PINS}
    result["protocol"] = schema.PROTOCOL
    schema.execution(result, reserved=reserved)
    return result


def study_identity(authorization):
    return original.study_identity(authorization.base, authorization.operational) | {
        "protocol": PROTOCOL,
        "study_profile_sha256": authorization.profile_sha256,
        "adoption_envelope_sha256": authorization.envelope_sha256,
        "study_policy_sha256": authorization.policy_sha256,
    }


def envelope(status, execution):
    scientific_execution(execution)
    return {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "status": status,
        "execution": execution,
    }


def study_intent(authorization, attempt):
    identity = study_identity(authorization)
    retention.reservation(attempt, identity)
    execution = identity | {"reservation_sha256": attempt.reservation_sha256}
    legacy = schema.envelope("intent", scientific_execution(execution)) | {
        "operational_profile": authorization.operational.projection(),
        "protective_deadlines_seconds": authorization.deadlines,
    }
    return canonical_bytes(
        envelope("intent", execution)
        | {
            "scientific_intent_bytes": encoded(canonical_bytes(legacy)),
            "policy_bytes": encoded(authorization.policy_bytes),
            "profile_bytes": encoded(authorization.profile_bytes),
            "envelope_bytes": encoded(authorization.envelope_bytes),
        }
    )


def study_accounting(state, status, error=None):
    internal, external = state.source_statuses(error)
    scientific = original.study_accounting(
        state.cells(error),
        execution=state.execution,
        stage=state.stage,
        status=status,
        internal_status=internal,
        external_status=external,
    )
    return canonical_bytes(
        envelope(status, state.adopted_execution)
        | {
            "scientific_accounting_bytes": encoded(scientific),
            "authorization_ledger": schema.load(state.admissions.snapshot()),
        }
    )


def unwrap(content, execution, extra):
    value = schema.load(content)
    schema.keys(value, {"schema_version", "protocol", "status", "execution", *extra})
    schema.require(
        type(value["schema_version"]) is int and value["schema_version"] == 1
    )
    schema.require(value["protocol"] == PROTOCOL)
    schema.same(value["execution"], execution)
    return value


def original_checkpoints(contents, execution):
    intent = unwrap(
        contents[retention.ORDER[0]],
        execution,
        {
            "scientific_intent_bytes",
            "policy_bytes",
            "profile_bytes",
            "envelope_bytes",
        },
    )
    accounting = unwrap(
        contents[retention.ORDER[3]],
        execution,
        {
            "scientific_accounting_bytes",
            "authorization_ledger",
        },
    )
    schema.require(intent["status"] == "intent")
    restored = dict(contents)
    restored[retention.ORDER[0]] = decoded(intent["scientific_intent_bytes"])
    restored[retention.ORDER[3]] = decoded(accounting["scientific_accounting_bytes"])
    schema.require(
        schema.load(restored[retention.ORDER[3]])["status"] == accounting["status"]
    )
    return tuple(restored.items())


def public_summary(execution, checkpoints, reduced=None):
    from ._adopted_study_ledger_validation import validate_ledger

    contents = dict(checkpoints)
    restored = original_checkpoints(contents, execution)
    public = original.root_public_summary(
        execution=scientific_execution(execution),
        checkpoints=restored,
        reduced=reduced,
    )
    validate_ledger(
        schema.load(contents[retention.ORDER[3]])["authorization_ledger"],
        execution,
        contents,
        reduced is not None,
    )
    extra = (
        {}
        if reduced is None
        else {
            "operational-summary.json": reduced.operational_bytes,
            "study-evidence.json": reduced.study_bytes,
        }
    )
    return public | {
        "protocol": PROTOCOL,
        "execution": execution,
        "accounting_sha256": sha256(contents[retention.ORDER[3]]).hexdigest(),
        "private_sha256": retention.hashes(contents | extra),
    }


def public_bytes(attempt, identity, contents, outputs, success, public):
    from .study_reduction import ReducedStudyBytes

    execution = identity | {"reservation_sha256": attempt.reservation_sha256}
    reduced = (
        ReducedStudyBytes(*(outputs[name] for name in retention.EXTRA_NAMES))
        if success
        else None
    )
    expected = public_summary(execution, tuple(contents.items()), reduced)
    content = retention.receipt._json_bytes(public, "study_public")
    retention.require(
        content == retention.receipt._json_bytes(expected, "study_public")
    )
    return content
