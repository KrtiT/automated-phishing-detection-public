"""Validate separately retained execution directives without inventing consent."""

from hashlib import sha256

from . import _study_execution_schema as legacy
from . import _study_series_adoption_schema as schema
from ._study_operator_authority import _recorded_at


def _directive(value, profile):
    schema.closed(value, "directive")
    schema.version(value["schema_version"])
    legacy.require(value["operator"] == profile["operator"])
    legacy.require(value["decision"] == "approved")
    legacy.require(value["scope"] == schema.EXECUTION_SCOPE)
    legacy.require(value["advisor_decision_obtained"] is False)
    legacy.require(
        value["personally_reviewed_future_code_or_profile_hash_claimed"] is False
    )
    for name in ("decision_text", "question_text", "reviewed_proposal_text"):
        legacy.text(value[name])
    schema.pins(value)
    proposal_hash = sha256(value["reviewed_proposal_text"].encode("utf-8")).hexdigest()
    legacy.require(value["reviewed_proposal_sha256"] == proposal_hash)
    return _recorded_at(value["recorded_at"])


def _rebind(value, directive, profile, recorded):
    schema.closed(value, "rebind")
    legacy.text(value["recorded_by"])
    legacy.require(_recorded_at(value["recorded_at"]) >= recorded)
    legacy.require(value["scope"] == schema.EXECUTION_SCOPE)
    legacy.require(value["advisor_decision_obtained"] is False)
    schema.pins(value)
    legacy.require(value["directive_sha256"] == schema.digest(directive))
    legacy.require(value["profile_sha256"] == schema.digest(profile))
    legacy.require(value["history_index_sha256"] == profile["history"]["index_sha256"])


def _decisions(value, directive, profile):
    schema.closed(value, "decisions")
    for name, decision in value.items():
        schema.closed(decision, "decision")
        legacy.require(decision["operator"] == directive["operator"])
        legacy.require(decision["status"] == "authorized_by_explicit_series_directive")
        schema.pins(decision)
        legacy.require(decision["directive_sha256"] == schema.digest(directive))
        kind, digest = (
            ("series-amendment", profile["amendment_sha256"])
            if name == "amendment"
            else ("study-series-profile", schema.digest(profile))
        )
        legacy.require(
            decision["subject_kind"] == kind and decision["subject_sha256"] == digest
        )


def envelope(value, profile):
    schema.closed(value, "envelope")
    schema.version(value["schema_version"])
    legacy.require(value["envelope_id"] == "study-series-adoption-envelope-v1")
    legacy.require(value["revoked"] is False)
    legacy.require(schema.digest(value["profile"]) == schema.digest(profile))
    directive = value["operator_directive"]
    recorded = _directive(directive, profile)
    _rebind(value["technical_rebind"], directive, profile, recorded)
    _decisions(value["decisions"], directive, profile)
