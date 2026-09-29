"""Invented retained-data amendment decisions, not real study authority."""

import copy
import json
from hashlib import sha256

from study_operator_fixtures import effective_profile

from automated_phishing_detection._checkpoint_codec import canonical_bytes

OUTPUTS = (
    "preparation-attempt",
    "internal-attempt",
    "internal-public-summary",
    "external-attempt",
    "external-public-summary",
    "attempt",
    "public-summary",
    "accepted-inputs-dir",
    "cells-dir",
)
PINS = (
    "prior_envelope_sha256",
    "prior_root_reservation_sha256",
    "prior_public_summary_sha256",
    "prior_preparation_reservation_sha256",
    "prior_preparation_complete_sha256",
    "publisher_source_sha256",
    "publisher_summary_sha256",
    "diagnostic_sha256",
)


def digest(value):
    content = value if isinstance(value, bytes) else canonical_bytes(value)
    return sha256(content).hexdigest()


def policy_bytes():
    from automated_phishing_detection import _study_urlnorm_policy

    return _study_urlnorm_policy.policy_bytes()


def profile(case):
    prior = effective_profile(case)
    current = copy.deepcopy(prior)
    current.update(schema_version=2, profile_id="study-urlnorm-profile-v1")
    current["policy_sha256"] = digest(policy_bytes())
    current["method_sha256"] = digest(json.loads(policy_bytes())["method"])
    current["session"]["session_id"] += "-urlnorm"
    for name in OUTPUTS:
        current["paths"][name] += "-urlnorm"
    current["continuation"] = {
        "representation": "publisher_url_norm_v1",
        "prior_profile": prior,
        "prior_profile_sha256": digest(prior),
        **{name: digest(name.encode()) for name in PINS},
    }
    return current


def directive(current):
    prior = current["continuation"]
    proposal = "Invented approved publisher-url representation design."
    return {
        "schema_version": 1,
        "operator": current["session"]["operator"],
        "decision": "approved",
        "decision_text": "Invented approval of the disclosed URL amendment.",
        "question_text": "Invented question about publisher url_norm.",
        "recorded_at": "2026-09-28T20:00:00Z",
        "reviewed_proposal_text": proposal,
        "reviewed_proposal_sha256": digest(proposal.encode()),
        "prior_profile_sha256": prior["prior_profile_sha256"],
        "held_public_summary_sha256": prior["prior_public_summary_sha256"],
        "diagnostic_sha256": prior["diagnostic_sha256"],
        "advisor_decision_obtained": False,
        "scope": "publisher_url_norm_retained_data_continuation",
        "personally_reviewed_future_code_or_profile_hash_claimed": False,
    }


def refresh(value):
    selected = value["profile"]
    instructions = value["operator_directive"]
    value["technical_rebind"].update(
        directive_sha256=digest(instructions), profile_sha256=digest(selected)
    )
    for name, decision in value["decisions"].items():
        decision.update(
            directive_sha256=digest(instructions),
            subject_sha256=selected["method_sha256"]
            if name == "method"
            else digest(selected),
        )
    return value


def seal(current):
    operator = current["session"]["operator"]
    return refresh(
        {
            "schema_version": 3,
            "envelope_id": "study-adoption-envelope-v3",
            "profile": current,
            "operator_directive": directive(current),
            "technical_rebind": {
                "recorded_by": "Invented technical recorder",
                "recorded_at": "2026-09-28T21:00:00Z",
                "scope": "publisher_url_norm_retained_data_continuation",
                "advisor_decision_obtained": False,
            },
            "decisions": {
                name: {
                    "operator": operator,
                    "status": "authorized_by_operator_directive",
                    "subject_kind": "method" if name == "method" else "study-profile",
                }
                for name in ("method", "profile", "access")
            },
            "revoked": False,
        }
    )
