"""Map normalized proposed-policy inputs.

No PhishVN file schema is parsed or assumed.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class LabelDecision:
    disposition: str
    is_phishing: int | None
    evidence_role: str | None
    reason_code: str


_INVALID_PHIUSIIL_LABEL = LabelDecision(
    "quarantine", None, None, "invalid_phiusiil_native_label"
)
_MISSING_PHISHVN_FIELD = LabelDecision(
    "quarantine", None, None, "missing_phishvn_mapping_field"
)
_UNDEFINED_PHISHVN_MAPPING = LabelDecision(
    "quarantine", None, None, "undefined_phishvn_mapping"
)

_PHIUSIIL_LABEL_DECISIONS = {
    0: LabelDecision("include", 1, "development_internal", "phiusiil_native_zero"),
    1: LabelDecision("include", 0, "development_internal", "phiusiil_native_one"),
}

_PHISHVN_POLICY_DECISIONS = {
    ("ncsc", "gold", "phishing"): LabelDecision(
        "include", 1, "primary_external", "ncsc_gold_phishing"
    ),
    ("trusted_registry", "certified", "legitimate"): LabelDecision(
        "include",
        0,
        "primary_external",
        "trusted_registry_certified_legitimate",
    ),
    ("ncsc", "silver", "phishing"): LabelDecision(
        "include",
        1,
        "secondary_or_sensitivity",
        "ncsc_silver_phishing",
    ),
    ("chongluadao_openphish", "bronze", "phishing"): LabelDecision(
        "include",
        1,
        "secondary_or_sensitivity",
        "chongluadao_openphish_bronze_phishing",
    ),
    ("tranco", "control", "reference_negative"): LabelDecision(
        "control",
        None,
        "reference_negative_control",
        "tranco_reference_negative_control",
    ),
}


def map_phiusiil_label(native_label: object) -> LabelDecision:
    if type(native_label) is not int:
        return _INVALID_PHIUSIIL_LABEL
    return _PHIUSIIL_LABEL_DECISIONS.get(native_label, _INVALID_PHIUSIIL_LABEL)


def evaluate_phishvn_policy(
    source_group: object, confidence_tier: object, designation: object
) -> LabelDecision:
    if (
        not isinstance(source_group, str)
        or source_group == ""
        or not isinstance(confidence_tier, str)
        or confidence_tier == ""
        or not isinstance(designation, str)
        or designation == ""
    ):
        return _MISSING_PHISHVN_FIELD

    policy_key = (source_group, confidence_tier, designation)
    return _PHISHVN_POLICY_DECISIONS.get(policy_key, _UNDEFINED_PHISHVN_MAPPING)
