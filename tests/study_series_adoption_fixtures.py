"""Invented series metadata; no actual approval or protected input is used."""

import json
from hashlib import sha256
from importlib import import_module
from importlib.util import find_spec
from types import SimpleNamespace

from study_series_adoption_profile_fixtures import profile

from automated_phishing_detection._checkpoint_codec import canonical_bytes


def digest(value):
    return sha256(
        value if isinstance(value, bytes) else canonical_bytes(value)
    ).hexdigest()


def api():
    name = "automated_phishing_detection.study_series_adoption"
    assert find_spec(name), "missing pure series adoption header"
    return import_module(name)


def directive(value):
    proposal = "Invented one-segment checkpoint proposal for tests only."
    return {
        "schema_version": 1,
        "operator": value["operator"],
        "decision": "approved",
        "decision_text": "Invented execution and historical-access adoption, not real consent.",
        "question_text": "Invented one-segment execution question.",
        "recorded_at": "2026-09-30T00:00:00Z",
        "reviewed_proposal_text": proposal,
        "reviewed_proposal_sha256": digest(proposal.encode()),
        "scope": "checkpointed_series_v1_one_segment_execution",
        "development_directive_sha256": digest(b"invented development directive"),
        "advisor_decision_obtained": False,
        "personally_reviewed_future_code_or_profile_hash_claimed": False,
    }


def refresh(case):
    value = case.envelope
    selected, instructions = value["profile"], value["operator_directive"]
    value["technical_rebind"].update(
        profile_sha256=digest(selected),
        directive_sha256=digest(instructions),
        history_index_sha256=selected["history"]["index_sha256"],
    )
    for name, decision in value["decisions"].items():
        decision.update(
            directive_sha256=digest(instructions),
            subject_sha256=(
                selected["amendment_sha256"]
                if name == "amendment"
                else digest(selected)
            ),
        )
    return case


def make_case():
    api()
    policy_module = import_module("automated_phishing_detection._study_series_policy")
    policy = json.loads(policy_module.policy_bytes())
    selected = profile(policy)
    instructions = directive(selected)
    envelope = {
        "schema_version": 1,
        "envelope_id": "study-series-adoption-envelope-v1",
        "profile": selected,
        "operator_directive": instructions,
        "revoked": False,
        "technical_rebind": {
            "recorded_by": "Invented recorder",
            "recorded_at": "2026-09-30T01:00:00Z",
            "scope": instructions["scope"],
            "advisor_decision_obtained": False,
            "implementation_review_sha256": digest(b"invented review"),
            "verification_evidence_sha256": digest(b"invented tests"),
        },
        "decisions": decisions(selected["operator"]),
    }
    return refresh(SimpleNamespace(policy=policy, profile=selected, envelope=envelope))


def decisions(operator):
    return {
        name: {
            "operator": operator,
            "status": "authorized_by_explicit_series_directive",
            "subject_kind": "series-amendment"
            if name == "amendment"
            else "study-series-profile",
        }
        for name in ("amendment", "historical_access", "segment_execution")
    }


def validate(case, **overrides):
    arguments = dict(
        policy_bytes=canonical_bytes(case.policy),
        profile_bytes=canonical_bytes(case.profile),
        envelope_bytes=canonical_bytes(case.envelope),
        expected_profile_sha256=digest(case.profile),
        expected_envelope_sha256=digest(case.envelope),
    )
    return api().validate_series_adoption_header(**(arguments | overrides))


def selected(value, path):
    for name in path.split("."):
        value = value[name]
    return value
