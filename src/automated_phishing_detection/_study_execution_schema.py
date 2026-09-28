"""Closed canonical approval records; no names or booleans establish consent."""

import json
import re
from datetime import datetime
from hashlib import sha256
from pathlib import Path

from . import _study_execution_policy as policy
from ._checkpoint_codec import canonical_bytes
from ._study_cli_protocol import PATH_ARGUMENTS
from ._study_policy_variants import select_policy

PROFILE_FIELDS = (
    "schema_version",
    "profile_id",
    "execution",
    "policy_sha256",
    "method_sha256",
    "components",
    "invocation",
    "paths",
    "source_artifact_scope",
    "session",
)


class StudyExecutionError(ValueError):
    """The exact study authorization or its identity cannot be authenticated."""


def require(condition):
    if not condition:
        raise StudyExecutionError("invalid_study_execution_authorization")


def closed(value, names):
    require(type(value) is dict and set(value) == set(names))


def digest(value, size=64):
    require(type(value) is str and re.fullmatch(rf"[0-9a-f]{{{size}}}", value))


def text(value):
    require(type(value) is str and value.strip() == value and 0 < len(value) <= 32768)
    require("\0" not in value)


def parse(content, expected_sha256):
    digest(expected_sha256)
    require(type(content) is bytes and sha256(content).hexdigest() == expected_sha256)
    value = json.loads(content)
    require(canonical_bytes(value) == content)
    return value


def lexical_path(value):
    text(value)
    result = Path(value)
    require(result.anchor == "/" and ".." not in result.parts and str(result) == value)
    return result


def _invocation(value):
    closed(value["invocation"], ("script", "arguments"))
    require(value["invocation"]["script"] == policy.ROOT_SCRIPT)
    require(
        value["invocation"]["arguments"]
        == {
            "repo-root": value["paths"]["repo-root"],
            "expected-revision": value["execution"]["revision"],
        }
    )


def _session(value):
    closed(value, ("operator", "session_id", "requirements", "operator_commitment"))
    text(value["operator"])
    text(value["session_id"])
    require(value["requirements"] == policy.SESSION_REQUIREMENTS)
    require(
        value["operator_commitment"]
        == "exclusive_session_conditions_and_pre_post_records_required"
    )


def profile(value):
    closed(value, PROFILE_FIELDS)
    require(type(value["schema_version"]) is int and value["schema_version"] == 1)
    require(value["profile_id"] == "study-execution-profile-v1")
    closed(value["execution"], ("revision", "contract_sha256"))
    digest(value["execution"]["revision"], 40)
    require(value["execution"]["contract_sha256"] == policy.CONTRACT_SHA256)
    digest(value["policy_sha256"])
    digest(value["method_sha256"])
    closed(value["components"], ("external", "operational"))
    for member in value["components"].values():
        digest(member)
    closed(value["paths"], PATH_ARGUMENTS)
    for member in value["paths"].values():
        lexical_path(member)
    require(
        type(value["source_artifact_scope"]) is dict and value["source_artifact_scope"]
    )
    for name, member in value["source_artifact_scope"].items():
        text(name)
        digest(member)
    _invocation(value)
    _session(value["session"])


def _decision(value, allowed):
    closed(
        value,
        (
            "status",
            "approver",
            "recorded_at",
            "subject_kind",
            "subject_sha256",
            "decision_text",
        ),
    )
    require(value["status"] == "approved")
    for name in ("approver", "recorded_at", "subject_kind", "decision_text"):
        text(value[name])
    digest(value["subject_sha256"])
    parsed = datetime.strptime(value["recorded_at"], "%Y-%m-%dT%H:%M:%SZ")
    require(parsed.strftime("%Y-%m-%dT%H:%M:%SZ") == value["recorded_at"])
    require((value["subject_kind"], value["subject_sha256"]) in allowed)


def envelope(content, expected_sha256):
    value = parse(content, expected_sha256)
    if type(value) is dict and value.get("envelope_id") == "study-adoption-envelope-v2":
        from ._study_operator_authority import envelope as operator_envelope

        return operator_envelope(value)
    closed(value, ("schema_version", "envelope_id", "profile", "decisions", "revoked"))
    require(type(value["schema_version"]) is int and value["schema_version"] == 1)
    require(value["envelope_id"] == "study-adoption-envelope-v1")
    require(value["revoked"] is False)
    profile(value["profile"])
    payload = value["profile"]
    payload_sha256 = sha256(canonical_bytes(payload)).hexdigest()
    closed(value["decisions"], ("method", "advisor", "profile", "access"))
    for name, member in value["decisions"].items():
        allowed = {("study-profile", payload_sha256)}
        if name in ("method", "advisor"):
            allowed |= {
                ("method", payload["method_sha256"]),
                ("policy", payload["policy_sha256"]),
            }
        _decision(member, allowed)
    return payload


def joined(
    policy_content, profile_content, envelope_content, profile_pin, envelope_pin
):
    value = envelope(envelope_content, envelope_pin)
    retained = parse(profile_content, profile_pin)
    require(canonical_bytes(value) == profile_content and retained == value)
    selected_policy = parse(policy_content, value["policy_sha256"])
    require(policy_content == select_policy(value["policy_sha256"])[1])
    require(
        json.loads(envelope_content)["schema_version"]
        == selected_policy["schema_version"]
    )
    require(
        value["method_sha256"]
        == sha256(canonical_bytes(selected_policy["method"])).hexdigest()
    )
    return value
