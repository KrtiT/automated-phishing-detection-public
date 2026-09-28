"""Invented operator instructions; these fixtures authorize no real activity."""

import copy
import json
from hashlib import sha256

from study_execution_fixtures import profile

from automated_phishing_detection import _study_execution_policy as original
from automated_phishing_detection._checkpoint_codec import canonical_bytes

POLICY_PATH = "data/study-execution-policy-v2.json"
GOVERNANCE = (
    "Require actual method, exact-profile and protected-access authorization from "
    "the named operator's retained directive. That directive explicitly waives "
    "fresh advisor concurrence for a governance-only rebind of the retained "
    "reviewed profile. Record the original scope and the agent's effective-profile "
    "binding without claiming advisor approval or personal review of a later "
    "digest. Preserve all scientific and session requirements."
)


def digest(content):
    return sha256(content).hexdigest()


def policy_bytes():
    value = original.policy_projection()
    value.update(schema_version=2, policy_id="study-execution-policy-v2")
    value["method"][-1] = GOVERNANCE
    value["decisions"] = ["method", "advisor_requirement_waiver", "profile", "access"]
    value["authorization_basis"] = "operator_directive_governance_only_rebind"
    return canonical_bytes(value)


def effective_profile(case):
    value = profile(case)
    value["policy_sha256"] = digest(policy_bytes())
    value["method_sha256"] = digest(
        canonical_bytes(json.loads(policy_bytes())["method"])
    )
    return value


def source_profile(current):
    value = copy.deepcopy(current)
    value["execution"]["revision"] = "b" * 40
    value["invocation"]["arguments"]["expected-revision"] = "b" * 40
    value["policy_sha256"] = digest(original.policy_bytes())
    value["method_sha256"] = digest(
        canonical_bytes(original.policy_projection()["method"])
    )
    value["components"] = {"external": "3" * 64, "operational": "4" * 64}
    return value


def refresh(value):
    directive = value["operator_directive"]
    directive["source_profile_sha256"] = digest(
        canonical_bytes(directive["source_profile"])
    )
    directive_hash = digest(canonical_bytes(directive))
    profile_hash = digest(canonical_bytes(value["profile"]))
    value["technical_rebind"].update(
        directive_sha256=directive_hash, profile_sha256=profile_hash
    )
    for name, decision in value["decisions"].items():
        decision.update(
            directive_sha256=directive_hash,
            subject_sha256=value["profile"]["method_sha256"]
            if name == "method"
            else profile_hash,
        )
    return value


def _decisions(operator):
    return {
        name: {
            "operator": operator,
            "status": "waived_by_operator"
            if name == "advisor_requirement_waiver"
            else "authorized_by_operator_directive",
            "subject_kind": "method" if name == "method" else "study-profile",
        }
        for name in ("method", "advisor_requirement_waiver", "profile", "access")
    }


def seal(current, source=None):
    operator = current["session"]["operator"]
    value = {
        "schema_version": 2,
        "envelope_id": "study-adoption-envelope-v2",
        "profile": current,
        "revoked": False,
        "operator_directive": {
            "operator": operator,
            "decision_text": "Invented instruction: proceed; waive advisor concurrence.",
            "recorded_at": "2026-01-01T00:00:00Z",
            "source_profile": source or source_profile(current),
        },
        "technical_rebind": {
            "recorded_by": "Invented fixture recorder",
            "recorded_at": "2026-01-02T00:00:00Z",
            "scope": "governance_only",
            "advisor_decision_obtained": False,
        },
        "decisions": _decisions(operator),
    }
    return refresh(value)


def install_policy(case, monkeypatch):
    from automated_phishing_detection import execution_preflight as preflight

    read = preflight._read_regular
    case.policy = policy_bytes()

    def selected(root, relative):
        return (
            case.policy
            if root == case.root and relative == POLICY_PATH
            else read(root, relative)
        )

    def git(root, *arguments):
        if arguments[0] == "ls-tree":
            assert arguments[-1] == POLICY_PATH
            return b"100644 blob " + b"c" * 40 + b"\t" + POLICY_PATH.encode() + b"\0"
        assert arguments == ("cat-file", "blob", "c" * 40)
        return case.policy

    monkeypatch.setattr(preflight, "_read_regular", selected)
    monkeypatch.setattr(preflight, "_git", git)
