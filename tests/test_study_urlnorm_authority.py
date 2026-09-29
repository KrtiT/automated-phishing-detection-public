"""Scientific amendment authority cannot masquerade as a governance-only rebind."""

from importlib.util import find_spec
from pathlib import Path

import pytest
from study_execution_fixtures import execution_case as execution_case
from study_urlnorm_fixtures import digest, policy_bytes, profile, refresh, seal

from automated_phishing_detection._checkpoint_codec import canonical_bytes
from automated_phishing_detection.study_execution import validate_study_execution_bytes


def verify(value):
    return validate_study_execution_bytes(
        policy_bytes(),
        canonical_bytes(value["profile"]),
        canonical_bytes(value),
        expected_profile_sha256=digest(value["profile"]),
        expected_envelope_sha256=digest(value),
    )


def test_retained_urlnorm_policy_is_explicitly_versioned():
    assert find_spec("automated_phishing_detection._study_urlnorm_policy")
    path = Path(__file__).parents[1] / "data/study-execution-policy-v3.json"
    assert path.read_bytes() == policy_bytes()


def test_scientific_directive_accepts_only_the_new_explicit_profile(execution_case):
    value = seal(profile(execution_case))
    assert verify(value) == value["profile"]
    assert "advisor" not in value["decisions"]


def test_literal_multiline_proposal_preserves_terminal_newline(execution_case):
    value = seal(profile(execution_case))
    instructions = value["operator_directive"]
    instructions["reviewed_proposal_text"] = (
        "Invented proposal\n\nPreserved literal text.\n"
    )
    instructions["reviewed_proposal_sha256"] = digest(
        instructions["reviewed_proposal_text"].encode()
    )
    assert verify(refresh(value)) == value["profile"]


@pytest.mark.parametrize("field", ["source-csv", "gmm", "random-forest"])
def test_original_source_and_model_locators_cannot_change(execution_case, field):
    value = seal(profile(execution_case))
    value["profile"]["paths"][field] += "-replacement"
    with pytest.raises(ValueError):
        verify(refresh(value))


@pytest.mark.parametrize("field", ["attempt", "preparation-attempt", "cells-dir"])
def test_old_output_paths_are_not_resumable(execution_case, field):
    value = seal(profile(execution_case))
    current = value["profile"]
    current["paths"][field] = current["continuation"]["prior_profile"]["paths"][field]
    with pytest.raises(ValueError):
        verify(refresh(value))


@pytest.mark.parametrize("change", ["model", "contract", "operator", "session"])
def test_scientific_and_session_invariants_are_not_expanded(execution_case, change):
    value = seal(profile(execution_case))
    current = value["profile"]
    if change == "model":
        current["source_artifact_scope"]["data/sources.json"] = "0" * 64
    elif change == "contract":
        current["execution"]["contract_sha256"] = "0" * 64
    elif change == "operator":
        current["session"]["operator"] += "-other"
    else:
        current["session"]["session_id"] = current["continuation"]["prior_profile"][
            "session"
        ]["session_id"]
    with pytest.raises(ValueError):
        verify(refresh(value))


@pytest.mark.parametrize("change", ["consent", "scope", "parent", "advisor", "later"])
def test_rehashed_false_directive_rejects(execution_case, change):
    value = seal(profile(execution_case))
    instructions = value["operator_directive"]
    if change == "consent":
        instructions["decision"] = "pending"
    elif change == "scope":
        instructions["scope"] = "governance_only"
    elif change == "parent":
        instructions["prior_profile_sha256"] = "0" * 64
    elif change == "advisor":
        instructions["advisor_decision_obtained"] = True
    else:
        instructions["personally_reviewed_future_code_or_profile_hash_claimed"] = True
    with pytest.raises(ValueError):
        verify(refresh(value))


@pytest.mark.parametrize("field", ["representation", "prior_profile_sha256"])
def test_unrecognized_or_changed_ancestry_rejects(execution_case, field):
    value = seal(profile(execution_case))
    value["profile"]["continuation"][field] = "unrecognized"
    with pytest.raises(ValueError):
        verify(refresh(value))


@pytest.mark.parametrize("version", [1, 2])
def test_legacy_envelopes_never_authorize_scientific_amendment(execution_case, version):
    import study_execution_fixtures as original
    import study_operator_fixtures as operator

    from automated_phishing_detection import _study_execution_policy as policy
    from automated_phishing_detection import _study_policy_variants as variants

    current = profile(execution_case)
    selected = (
        policy.policy_bytes() if version == 1 else variants.operator_policy_bytes()
    )
    import json

    current.update(
        policy_sha256=digest(selected),
        method_sha256=digest(json.loads(selected)["method"]),
    )
    value = (
        original.seal(execution_case, changed=current)
        if version == 1
        else operator.seal(current)
    )
    with pytest.raises(ValueError):
        validate_study_execution_bytes(
            selected,
            canonical_bytes(current),
            canonical_bytes(value),
            expected_profile_sha256=digest(current),
            expected_envelope_sha256=digest(value),
        )
