import inspect
from dataclasses import FrozenInstanceError

import pytest

from automated_phishing_detection.proposed_label_contract import (
    LabelDecision,
    evaluate_phishvn_policy,
    map_phiusiil_label,
)


@pytest.mark.parametrize(
    ("native_label", "expected"),
    [
        pytest.param(
            0,
            LabelDecision("include", 1, "development_internal", "phiusiil_native_zero"),
            id="native-zero-is-phishing",
        ),
        pytest.param(
            1,
            LabelDecision("include", 0, "development_internal", "phiusiil_native_one"),
            id="native-one-is-legitimate",
        ),
    ],
)
def test_maps_defined_phiusiil_native_labels(native_label, expected):
    assert map_phiusiil_label(native_label) == expected


@pytest.mark.parametrize(
    "native_label",
    [
        pytest.param(None, id="none"),
        pytest.param(True, id="boolean"),
        pytest.param(2, id="out-of-range-integer"),
        pytest.param("0", id="numeric-string"),
    ],
)
def test_quarantines_invalid_phiusiil_native_labels(native_label):
    assert map_phiusiil_label(native_label) == LabelDecision(
        "quarantine", None, None, "invalid_phiusiil_native_label"
    )


@pytest.mark.parametrize(
    ("source_group", "confidence_tier", "designation", "expected"),
    [
        pytest.param(
            "ncsc",
            "gold",
            "phishing",
            LabelDecision("include", 1, "primary_external", "ncsc_gold_phishing"),
            id="ncsc-gold-phishing",
        ),
        pytest.param(
            "trusted_registry",
            "certified",
            "legitimate",
            LabelDecision(
                "include",
                0,
                "primary_external",
                "trusted_registry_certified_legitimate",
            ),
            id="trusted-registry-certified-legitimate",
        ),
        pytest.param(
            "ncsc",
            "silver",
            "phishing",
            LabelDecision(
                "include",
                1,
                "secondary_or_sensitivity",
                "ncsc_silver_phishing",
            ),
            id="ncsc-silver-phishing",
        ),
        pytest.param(
            "chongluadao_openphish",
            "bronze",
            "phishing",
            LabelDecision(
                "include",
                1,
                "secondary_or_sensitivity",
                "chongluadao_openphish_bronze_phishing",
            ),
            id="chongluadao-openphish-bronze-phishing",
        ),
        pytest.param(
            "tranco",
            "control",
            "reference_negative",
            LabelDecision(
                "control",
                None,
                "reference_negative_control",
                "tranco_reference_negative_control",
            ),
            id="tranco-reference-negative-control",
        ),
    ],
)
def test_applies_defined_phishvn_policy_combinations(
    source_group, confidence_tier, designation, expected
):
    assert (
        evaluate_phishvn_policy(source_group, confidence_tier, designation) == expected
    )


@pytest.mark.parametrize(
    ("source_group", "confidence_tier", "designation"),
    [
        pytest.param(None, "gold", "phishing", id="missing-source-group"),
        pytest.param("", "gold", "phishing", id="blank-source-group"),
        pytest.param("ncsc", None, "phishing", id="missing-confidence-tier"),
        pytest.param("ncsc", "", "phishing", id="blank-confidence-tier"),
        pytest.param("ncsc", "gold", None, id="missing-designation"),
        pytest.param("ncsc", "gold", "", id="blank-designation"),
    ],
)
def test_quarantines_missing_or_blank_phishvn_mapping_fields(
    source_group, confidence_tier, designation
):
    assert evaluate_phishvn_policy(
        source_group, confidence_tier, designation
    ) == LabelDecision("quarantine", None, None, "missing_phishvn_mapping_field")


@pytest.mark.parametrize(
    ("source_group", "confidence_tier", "designation"),
    [
        pytest.param("ncsc", "gold", "legitimate", id="mismatched-designation"),
        pytest.param(
            "trusted_registry",
            "certified",
            "phishing",
            id="reverse-mismatched-designation",
        ),
        pytest.param("unknown", "gold", "phishing", id="unknown-source-group"),
        pytest.param("NCSC", "gold", "phishing", id="case-varied-source-group"),
        pytest.param("ncsc", "Gold", "phishing", id="case-varied-tier"),
        pytest.param("ncsc", "gold", "Phishing", id="case-varied-designation"),
        pytest.param(" ncsc", "gold", "phishing", id="whitespace-varied-source"),
    ],
)
def test_quarantines_undefined_exact_phishvn_combinations(
    source_group, confidence_tier, designation
):
    assert evaluate_phishvn_policy(
        source_group, confidence_tier, designation
    ) == LabelDecision("quarantine", None, None, "undefined_phishvn_mapping")


def test_returns_deterministic_decisions_for_repeated_inputs():
    phiusiil_decisions = [map_phiusiil_label(0) for _ in range(3)]
    phishvn_decisions = [
        evaluate_phishvn_policy("ncsc", "gold", "phishing") for _ in range(3)
    ]

    assert phiusiil_decisions == [phiusiil_decisions[0]] * 3
    assert phishvn_decisions == [phishvn_decisions[0]] * 3


def test_label_decision_values_are_frozen():
    decision = LabelDecision("include", 1, "primary_external", "defined_mapping")

    with pytest.raises(FrozenInstanceError):
        decision.disposition = "quarantine"


def test_phishvn_policy_signature_exposes_only_normalized_policy_inputs():
    signature = inspect.signature(evaluate_phishvn_policy)

    assert list(signature.parameters) == [
        "source_group",
        "confidence_tier",
        "designation",
    ]
