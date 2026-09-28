"""Explicit operator waiver is distinct from advisor approval or later hash review."""

import copy
import json
from pathlib import Path

import pytest
from study_execution_fixtures import execution_case as execution_case
from study_operator_fixtures import (
    digest,
    effective_profile,
    policy_bytes,
    refresh,
    seal,
)

from automated_phishing_detection import _study_execution_policy as original
from automated_phishing_detection import _study_execution_schema as schema
from automated_phishing_detection._checkpoint_codec import canonical_bytes
from automated_phishing_detection.study_execution import validate_study_execution_bytes


def verify(value):
    profile_content, envelope_content = (
        canonical_bytes(value["profile"]),
        canonical_bytes(value),
    )
    return validate_study_execution_bytes(
        policy_bytes(),
        profile_content,
        envelope_content,
        expected_profile_sha256=digest(profile_content),
        expected_envelope_sha256=digest(envelope_content),
    )


def test_explicit_directive_and_rebind_are_accepted_without_advisor_claim(
    execution_case,
):
    value = seal(effective_profile(execution_case))
    assert verify(value) == value["profile"]
    assert "advisor" not in value["decisions"]
    assert value["technical_rebind"]["advisor_decision_obtained"] is False


def test_committed_v2_policy_changes_only_governance():
    path = Path(__file__).parents[1] / "data/study-execution-policy-v2.json"
    assert path.is_file(), "missing explicit operator-only policy variant"
    content = path.read_bytes()
    assert content == policy_bytes()
    assert (
        digest(original.policy_bytes())
        == "40cbb7c5365c40d0c02e75259d535408219948fb7d62f2a177bcf6b05cdbb6e2"
    )
    value = json.loads(content)
    assert value["method"][:-1] == original.policy_projection()["method"][:-1]
    for name in (
        "base_contract",
        "protective_deadlines_seconds",
        "session_requirements",
        "scripts",
        "admission_roles",
        "legacy_readiness",
    ):
        assert value[name] == original.policy_projection()[name]


@pytest.mark.parametrize(
    "slot", ["method", "advisor_requirement_waiver", "profile", "access"]
)
@pytest.mark.parametrize(
    "field,value",
    [
        ("operator", "Wrong operator"),
        ("status", "approved"),
        ("directive_sha256", "0" * 64),
        ("subject_sha256", "0" * 64),
        ("subject_kind", "policy"),
    ],
)
def test_each_decision_requires_exact_operator_directive_status_and_scope(
    execution_case, slot, field, value
):
    envelope = seal(effective_profile(execution_case))
    envelope["decisions"][slot][field] = value
    with pytest.raises(ValueError):
        verify(envelope)


@pytest.mark.parametrize(
    "change",
    [
        "missing_waiver",
        "advisor_substitute",
        "extra",
        "claimed_advisor",
        "wrong_source",
        "wrong_directive",
        "wrong_effective",
        "backdated",
        "revoked",
    ],
)
def test_false_or_incomplete_rebind_cannot_authorize(execution_case, change):
    value = seal(effective_profile(execution_case))
    directive, rebind = value["operator_directive"], value["technical_rebind"]
    changes = {
        "claimed_advisor": (rebind, "advisor_decision_obtained", True),
        "wrong_source": (directive, "source_profile_sha256", "0" * 64),
        "wrong_directive": (directive, "decision_text", "Changed directive"),
        "wrong_effective": (rebind, "profile_sha256", "0" * 64),
        "backdated": (rebind, "recorded_at", "2025-01-01T00:00:00Z"),
        "extra": (directive, "approval_override", True),
        "revoked": (value, "revoked", True),
    }
    if change in ("missing_waiver", "advisor_substitute"):
        waived = value["decisions"].pop("advisor_requirement_waiver")
        if change == "advisor_substitute":
            value["decisions"]["advisor"] = waived | {"status": "approved"}
    else:
        target, key, selected = changes[change]
        target[key] = selected
    with pytest.raises(ValueError):
        verify(value)


@pytest.mark.parametrize(
    "change",
    [
        "path",
        "session",
        "operator",
        "scientific_hash",
        "scientific_addition",
        "removed_file",
        "prior_policy",
        "prior_method",
    ],
)
def test_rehashed_scientific_or_reviewed_scope_expansion_rejects(
    execution_case, change
):
    value = seal(effective_profile(execution_case))
    source = value["operator_directive"]["source_profile"]
    if change == "path":
        source["paths"]["source-csv"] += "-other"
    elif change in ("session", "operator"):
        source["session"]["session_id" if change == "session" else "operator"] += (
            "-other"
        )
    elif change == "scientific_hash":
        source["source_artifact_scope"]["data/sources.json"] = "0" * 64
    elif change == "scientific_addition":
        value["profile"]["source_artifact_scope"]["invented-model.json"] = "0" * 64
    elif change == "removed_file":
        del value["profile"]["source_artifact_scope"]["data/sources.json"]
    else:
        source["policy_sha256" if change == "prior_policy" else "method_sha256"] = (
            "0" * 64
        )
    with pytest.raises(ValueError):
        verify(refresh(value))


def test_only_fixed_governance_module_changes_are_permitted(execution_case):
    value = seal(effective_profile(execution_case))
    paths = (
        "_study_policy_variants",
        "_study_operator_authority",
        "_study_execution_schema",
        "_study_execution_io",
    )
    for name in paths:
        value["profile"]["source_artifact_scope"][
            f"src/automated_phishing_detection/{name}.py"
        ] = "0" * 64
    assert verify(refresh(value)) == value["profile"]


def test_original_envelope_cannot_carry_operator_policy(execution_case):
    from study_execution_fixtures import seal as original_seal

    value = original_seal(execution_case, effective_profile(execution_case))
    with pytest.raises(ValueError):
        verify(value)


def test_v2_envelope_rejects_original_policy(execution_case):
    value = seal(effective_profile(execution_case))
    value["profile"] = copy.deepcopy(value["operator_directive"]["source_profile"])
    with pytest.raises(ValueError):
        schema.envelope(canonical_bytes(refresh(value)), digest(canonical_bytes(value)))
