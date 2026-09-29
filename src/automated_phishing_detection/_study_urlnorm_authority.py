"""Bind the actual scientific directive without inventing a future human review."""

from hashlib import sha256

from . import _study_execution_schema as schema
from . import _study_urlnorm_policy as policy
from ._study_operator_authority import _recorded_at
from ._study_urlnorm_scope import digest

DIRECTIVE_FIELDS = (
    "schema_version",
    "operator",
    "decision",
    "decision_text",
    "question_text",
    "recorded_at",
    "reviewed_proposal_text",
    "reviewed_proposal_sha256",
    "prior_profile_sha256",
    "held_public_summary_sha256",
    "diagnostic_sha256",
    "advisor_decision_obtained",
    "scope",
    "personally_reviewed_future_code_or_profile_hash_claimed",
)


def _proposal(value):
    content = value["reviewed_proposal_text"]
    schema.require(type(content) is str and 0 < len(content) <= 32768)
    schema.require(bool(content.strip()) and "\0" not in content)
    schema.require(
        sha256(content.encode()).hexdigest() == value["reviewed_proposal_sha256"]
    )


def _directive(value, effective):
    schema.closed(value, DIRECTIVE_FIELDS)
    schema.require(
        type(value["schema_version"]) is int and value["schema_version"] == 1
    )
    schema.require(value["decision"] == "approved" and value["scope"] == policy.SCOPE)
    schema.require(value["operator"] == effective["session"]["operator"])
    schema.require(value["advisor_decision_obtained"] is False)
    schema.require(
        value["personally_reviewed_future_code_or_profile_hash_claimed"] is False
    )
    for name in ("decision_text", "question_text"):
        schema.text(value[name])
    _proposal(value)
    prior = effective["continuation"]
    for name, expected in (
        ("prior_profile_sha256", "prior_profile_sha256"),
        ("held_public_summary_sha256", "prior_public_summary_sha256"),
        ("diagnostic_sha256", "diagnostic_sha256"),
    ):
        schema.require(value[name] == prior[expected])
    return _recorded_at(value["recorded_at"])


def _rebind(value, directive, effective, recorded):
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
    schema.require(_recorded_at(value["recorded_at"]) >= recorded)
    schema.require(value["scope"] == policy.SCOPE)
    schema.require(value["advisor_decision_obtained"] is False)
    schema.require(value["directive_sha256"] == digest(directive))
    schema.require(value["profile_sha256"] == digest(effective))


def _decisions(value, directive, effective):
    schema.closed(value, ("method", "profile", "access"))
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
        schema.require(decision["status"] == "authorized_by_operator_directive")
        schema.require(decision["directive_sha256"] == digest(directive))
        kind, subject = (
            ("method", effective["method_sha256"])
            if name == "method"
            else ("study-profile", digest(effective))
        )
        schema.require(decision["subject_kind"] == kind)
        schema.require(decision["subject_sha256"] == subject)


def envelope(value):
    schema.closed(
        value,
        (
            "schema_version",
            "envelope_id",
            "profile",
            "operator_directive",
            "technical_rebind",
            "decisions",
            "revoked",
        ),
    )
    schema.require(
        type(value["schema_version"]) is int and value["schema_version"] == 3
    )
    schema.require(value["envelope_id"] == "study-adoption-envelope-v3")
    schema.require(value["revoked"] is False)
    effective, directive = value["profile"], value["operator_directive"]
    schema.profile(effective)
    schema.require(effective["profile_id"] == "study-urlnorm-profile-v1")
    schema.require(effective["policy_sha256"] == digest(policy.policy_projection()))
    schema.require(
        effective["method_sha256"] == digest(policy.policy_projection()["method"])
    )
    recorded = _directive(directive, effective)
    _rebind(value["technical_rebind"], directive, effective, recorded)
    _decisions(value["decisions"], directive, effective)
    return effective
