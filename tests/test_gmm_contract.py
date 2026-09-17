"""Check the prospective GMM recipe without opening research observations."""

import json
from hashlib import sha256
from pathlib import Path

import pytest
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from automated_phishing_detection import fixed_cascade
from automated_phishing_detection.url_features import FEATURE_NAMES

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "data" / "rq2-gmm-development-contract-v1.json"
TRANSFORMER = ROOT / "data" / "rq1-transformer-cascade-contract-v2.json"


@pytest.fixture
def contract():
    assert CONTRACT.is_file(), "prospective GMM contract must exist before execution"
    return json.loads(CONTRACT.read_text(encoding="utf-8"))


def test_gmm_contract_exists_before_any_execution():
    assert CONTRACT.is_file(), "prospective GMM contract must exist before execution"


def test_gmm_freeze_identity_and_unchanged_transformer_inputs(contract):
    transformer = json.loads(TRANSFORMER.read_text(encoding="utf-8"))
    assert contract["contract_id"] == "rq2-gmm-development-v1"
    assert contract["schema_version"] == 1
    assert contract["protocol_version"] == "1.10"
    assert contract["date"] == "2026-09-17"
    assert contract["status"] == "frozen_not_run"
    assert (
        contract["inputs"]["accepted_roles"] == transformer["inputs"]["accepted_roles"]
    )
    assert (
        contract["inputs"]["forbidden_roles"]
        == transformer["inputs"]["forbidden_roles"]
    )
    assert contract["cli"] == {
        "accepted_path_roles": [
            "train",
            "validation",
            "preparation_summary",
            "contract",
            "logistic_l1_artifact",
            "gmm_contract",
            "output_directory",
            "summary",
        ],
        "tuning_arguments": False,
    }
    assert contract["execution"]["fit_performed"] is False
    assert contract["execution"]["result_claimed"] is False
    assert sha256(TRANSFORMER.read_bytes()).hexdigest() == (
        "686c0d86b33b8a6c2e09cd6e174003db0bd2f7c30b087faf5470e6a270524213"
    )


def test_gmm_preserves_feature_order_and_training_only_scaler(contract):
    assert contract["features"]["names"] == [
        *FEATURE_NAMES,
        "logistic_l1_phishing_probability",
    ]
    assert contract["features"]["dtype"] == "float64"
    assert contract["features"]["stage1"] == {
        "source_role": "logistic_l1_artifact",
        "loader": "fixed_cascade.load_logistic_l1_artifact",
        "scoring": "PortableLogisticL1.score_urls",
        "refit": False,
    }
    assert callable(fixed_cascade.load_logistic_l1_artifact)
    assert callable(fixed_cascade.PortableLogisticL1.score_urls)
    scaler = contract["scaler"]
    assert scaler["fit_partition"] == "entire_training_26_column_matrix_only"
    assert scaler["variance_ddof"] == 0
    assert scaler["constant_column_scale"] == 1.0
    assert scaler["parameters"] == StandardScaler().get_params()


def test_validation_allocation_is_label_blind_and_fully_determined(contract):
    assert contract["validation_allocation"] == {
        "unit": "unique_normalized_ascii_registrable_domain",
        "namespace": "rq2-gmm-validation-v1",
        "seed": 20260816,
        "hash": "sha256",
        "hash_input": "namespace_utf8 + NUL + seed_ascii + NUL + domain_ascii",
        "sort_key": ["digest_bytes", "domain_ascii_bytes"],
        "calibration": "first_floor_domain_count_divided_by_2",
        "audit": "remaining_domains",
        "stream_row_order": "preserve_pinned_validation_input_order",
        "uses_labels": False,
        "uses_class_counts": False,
        "reroll": False,
    }


def test_all_six_gmm_estimators_have_exact_pinned_configuration(contract):
    mixture = contract["mixture"]
    assert mixture["candidate_components"] == [1, 2, 3, 4, 5, 6]
    expected = GaussianMixture().get_params()
    expected.pop("n_components")
    expected.update(covariance_type="diag", max_iter=500, n_init=5, random_state=42)
    assert mixture["parameters"] == expected
    assert mixture["selection"] == {
        "metric": "training_BIC",
        "objective": "minimum",
        "exact_tie_break": "smaller_component_count",
        "validation_used": False,
    }
    assert mixture["candidate_failure"] == "stop_entire_run_no_skip_no_retry"
    assert mixture["require_all_converged"] is True
    assert mixture["require_finite_parameters"] is True
    assert mixture["require_positive_weights_and_variances"] is True
    assert mixture["require_normalized_weights"] is True


def test_gmm_runtime_is_cpu_float64_without_warning_allowlist(contract):
    assert contract["runtime"] == {
        "device": "cpu",
        "dtype": "float64",
        "versions": {
            "numpy": "2.2.6",
            "scipy": "1.15.3",
            "scikit-learn": "1.7.2",
            "threadpoolctl": "3.6.0",
        },
        "threadpoolctl_limits": 1,
        "numpy_blas": {
            "name": "scipy-openblas",
            "version": "0.3.29",
            "version_match": "release_or_build_suffix",
            "preflight": "before_input_reads",
        },
        "macos_arm64_numpy_wheel": {
            "filename": "numpy-2.2.6-cp310-cp310-macosx_11_0_arm64.whl",
            "url": (
                "https://files.pythonhosted.org/packages/22/c2/"
                "4b9221495b2a132cc9d2eb862e21d42a009f5a60e45fc44b00118c174bff/"
                "numpy-2.2.6-cp310-cp310-macosx_11_0_arm64.whl"
            ),
            "sha256": "8e41fd67c52b86603a91c1a505ebaef50b3314de0213461c7a6e99c9a3beff90",
        },
        "warnings": "fatal",
        "numpy_errors": {"over": "raise", "invalid": "raise", "divide": "raise"},
        "underflow": "ignore_inside_mixture_fit_bic_and_likelihood_scoring",
        "nonfinite": "fatal",
        "accelerate_warning_allowlist_extension": False,
    }


def test_complete_windows_quantile_and_audit_gate_are_exact(contract):
    assert contract["windows"] == {
        "length": 256,
        "stride": 64,
        "score": "float64_mean_negative_log_likelihood",
        "complete_only": True,
        "minimum_complete_windows_per_stream": 1,
        "overlapping_windows_count_separately": True,
    }
    assert contract["calibration"] == {
        "source": "validation_calibration_stream_only",
        "quantile": 0.95,
        "numpy_quantile_method": "linear",
        "alert_rule": "score > boundary",
    }
    assert contract["audit"] == {
        "source": "validation_audit_stream_only",
        "gate": "20 * alert_windows <= complete_windows",
        "retune_boundary": False,
        "binomial_confidence_interval": False,
        "failed_gate_status": "completed_development_validation",
        "failed_gate_false_alert_gate_met": False,
    }
    assert contract["execution"]["hypotheses"] == dict.fromkeys(
        ("H1", "H2", "H3"), "undecided"
    )


def test_private_membership_and_parameters_are_not_public(contract):
    artifacts = contract["artifacts"]
    assert artifacts["private"]["files"] == {
        "gmm.json": "fitted_scaler_and_mixture",
        "validation-audit.json": "domain_membership_and_window_traces",
        "SHA256SUMS": "sha256_manifest",
    }
    assert artifacts["private"]["directory_mode"] == "0700"
    assert artifacts["private"]["file_mode"] == "0600"
    assert artifacts["private"]["sha256sums"]["files"] == [
        "gmm.json",
        "validation-audit.json",
    ]
    assert artifacts["public_summary"]["forbidden_content"] == [
        "URLs",
        "records",
        "domains",
        "record_ids",
        "fitted_parameter_arrays",
        "predictions",
        "scaler_parameters",
        "mixture_parameters",
        "window_traces",
        "membership",
    ]
    assert (
        "all_six_BIC_values_and_iteration_counts"
        in artifacts["public_summary"]["allowlisted_content"]
    )
    assert (
        "numpy_build_configuration"
        in artifacts["public_summary"]["allowlisted_content"]
    )
    assert artifacts["public_summary"]["access"] == {
        "group_test_accessed": False,
        "external_accessed": False,
        "phishvn_accessed": False,
        "scope": "this_process_only",
    }


def test_publication_inherits_private_first_public_last_v2_semantics(contract):
    transformer = json.loads(TRANSFORMER.read_text(encoding="utf-8"))
    assert contract["publication"] == transformer["publication"]
    assert (
        contract["artifacts"]["private"]["canonical_json"]
        == transformer["artifacts"]["private"]["canonical_json"]
    )
