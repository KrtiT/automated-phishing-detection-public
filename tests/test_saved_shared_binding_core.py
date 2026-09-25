"""The same frozen core is reusable without inventing internal envelope fields."""

from copy import deepcopy

import pytest

from automated_phishing_detection import saved_evidence


def validate(value):
    checker = getattr(saved_evidence, "_validate_binding_core", None)
    assert callable(checker), "missing shared frozen-core validator"
    return checker(value)


def test_frozen_core_needs_no_internal_partition_or_source_fields():
    core = deepcopy(saved_evidence._EXPECTED_BINDING_CORE)
    original = deepcopy(core)
    assert validate(core) is None
    assert core == original


@pytest.mark.parametrize(
    "fault",
    ["half_width_type", "seed_type", "reuse_type", "audit_type", "artifact", "extra"],
)
def test_shared_core_rejects_value_and_exact_type_changes(fault):
    core = deepcopy(saved_evidence._EXPECTED_BINDING_CORE)
    if fault == "half_width_type":
        core["thresholds"]["half_width"] = 0
    elif fault == "seed_type":
        core["secondary"]["seeds"][0]["seed"] = 42.0
    elif fault == "reuse_type":
        core["secondary"]["seeds"][0]["reuses_primary"] = 1
    elif fault == "audit_type":
        core["gmm_audit"]["alert_count"] = 28.0
    elif fault == "artifact":
        core["artifact_hashes"]["gmm.json"] = "f" * 64
    else:
        core["thresholds"]["private"] = "sensitive-value"
    with pytest.raises(saved_evidence.SavedEvidenceError) as caught:
        validate(core)
    assert "sensitive-value" not in str(caught.value)


@pytest.mark.parametrize("value", [None, [], {}, {"thresholds": {}}])
def test_shared_core_rejects_missing_or_malformed_core(value):
    with pytest.raises(saved_evidence.SavedEvidenceError):
        validate(value)
