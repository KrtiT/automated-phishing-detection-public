"""Retain operator-directed rebinding without inventing advisor or later review."""

from datetime import datetime
from hashlib import sha256

from . import _study_execution_policy as original
from . import _study_execution_schema as schema
from . import _study_policy_variants as variants
from ._checkpoint_codec import canonical_bytes

DECISIONS = ("method", "advisor_requirement_waiver", "profile", "access")


def _digest(value):
    return sha256(canonical_bytes(value)).hexdigest()


def _recorded_at(value):
    schema.text(value)
    parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
    schema.require(parsed.strftime("%Y-%m-%dT%H:%M:%SZ") == value)
    return parsed


def _profile_policy(value, policy):
    schema.profile(value)
    schema.require(value["policy_sha256"] == _digest(policy))
    schema.require(value["method_sha256"] == _digest(policy["method"]))


def _source_scope(source, effective):
    _profile_policy(source, original.policy_projection())
    for name in ("paths", "session"):
        schema.require(source[name] == effective[name])
    schema.require(
        source["execution"]["contract_sha256"]
        == effective["execution"]["contract_sha256"]
    )
    previous, current = (
        source["source_artifact_scope"],
        effective["source_artifact_scope"],
    )
    schema.require(previous.keys() <= current.keys())
    changed = {name for name in current if current[name] != previous.get(name)}
    schema.require(
        changed <= variants.GOVERNANCE_MODULES | {variants.OPERATOR_POLICY_PATH}
    )


def _directive(value, effective):
    schema.closed(
        value,
        (
            "operator",
            "decision_text",
            "recorded_at",
            "source_profile",
            "source_profile_sha256",
        ),
    )
    schema.require(value["operator"] == effective["session"]["operator"])
    schema.text(value["decision_text"])
    recorded = _recorded_at(value["recorded_at"])
    source = value["source_profile"]
    _source_scope(source, effective)
    schema.require(value["source_profile_sha256"] == _digest(source))
    return recorded


def _rebind(value, directive, effective, directive_recorded):
    schema.closed(
        value,
        (
            "recorded_by",
            "recorded_at",
            "directive_sha256",
            "profile_sha256",
            "scope",
            "advisor_decision_obtained",
        ),
    )
    schema.text(value["recorded_by"])
    schema.require(_recorded_at(value["recorded_at"]) >= directive_recorded)
    schema.require(value["directive_sha256"] == _digest(directive))
    schema.require(value["profile_sha256"] == _digest(effective))
    schema.require(value["scope"] == "governance_only")
    schema.require(value["advisor_decision_obtained"] is False)


def _decisions(value, directive, effective):
    schema.closed(value, DECISIONS)
    for name, decision in value.items():
        schema.closed(
            decision,
            (
                "operator",
                "status",
                "directive_sha256",
                "subject_kind",
                "subject_sha256",
            ),
        )
        schema.require(decision["operator"] == directive["operator"])
        schema.require(decision["directive_sha256"] == _digest(directive))
        status = (
            "waived_by_operator"
            if name == "advisor_requirement_waiver"
            else "authorized_by_operator_directive"
        )
        schema.require(decision["status"] == status)
        kind, digest = (
            ("method", effective["method_sha256"])
            if name == "method"
            else ("study-profile", _digest(effective))
        )
        schema.require(
            decision["subject_kind"] == kind and decision["subject_sha256"] == digest
        )


def envelope(value):
    schema.closed(
        value,
        (
            "schema_version",
            "envelope_id",
            "profile",
            "decisions",
            "revoked",
            "operator_directive",
            "technical_rebind",
        ),
    )
    schema.require(
        type(value["schema_version"]) is int and value["schema_version"] == 2
    )
    schema.require(value["envelope_id"] == "study-adoption-envelope-v2")
    schema.require(value["revoked"] is False)
    effective, directive = value["profile"], value["operator_directive"]
    _profile_policy(effective, variants.operator_policy_projection())
    recorded = _directive(directive, effective)
    _rebind(value["technical_rebind"], directive, effective, recorded)
    _decisions(value["decisions"], directive, effective)
    return effective
