"""Synthetic profile and composition projections for byte-only outer records."""

from hashlib import sha256

from automated_phishing_detection._saved_external_bindings import PRIVATE_OUTPUTS


def profile_projection(internal, publisher):
    execution = {
        name: internal[name]
        for name in (
            "revision",
            "execution_contract_sha256",
            "runtime_sha256",
            "source_spec_sha256",
        )
    }
    return {
        "schema_version": 1,
        "profile_id": "external-source-candidate-v1",
        "status": "specified_closed_candidate",
        "protected_evaluation_ready": False,
        "protected_evaluation_authorized": False,
        "execution": execution,
        "publisher": {"expected_format": publisher.public_summary["input_archive"]},
        "public_suffix_list": {"sha256": internal["suffix_rules_sha256"]},
    }


def composition_summary(outputs):
    return {
        "schema_version": 1,
        "status": "external_evidence_composed",
        "protected_evaluation_authorized": False,
        "source_binding": "caller_supplied_preparation_only",
        "private_sha256": {
            name: sha256(outputs[name]).hexdigest() for name in PRIVATE_OUTPUTS
        },
        "row_count": 0,
        "invented_nested": {"unchanged": [1, None, False]},
    }
