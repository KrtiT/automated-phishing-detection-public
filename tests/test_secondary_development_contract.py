import json
from hashlib import sha256
from pathlib import Path

import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from automated_phishing_detection import baselines, secondary_probes, url_features

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "data/secondary-development-contract-v1.json"


def contract():
    return json.loads(CONTRACT.read_bytes())


def test_secondary_development_contract_has_fixed_bytes():
    assert sha256(CONTRACT.read_bytes()).hexdigest() == (
        "f592352593ae64b178a468d5800e780267275a62046a05b31b952f69e424f44f"
    )


def test_secondary_development_is_separate_from_primary_and_protected_execution():
    value = contract()
    assert value["contract_id"] == "secondary-development-v1"
    assert value["protected_evaluation_ready"] is False
    assert value["research_execution_ready"] is False
    assert value["research_fits_run"] is False
    assert value["primary_changes"] == {
        "artifacts": False,
        "cutoffs": False,
        "hypothesis_gates": False,
        "gmm_audit": False,
    }
    assert value["additional_transformer_fits"] == 0


def test_secondary_contract_preserves_all_linked_public_evidence():
    for name, digest in contract()["public_file_sha256"].items():
        assert sha256((ROOT / name).read_bytes()).hexdigest() == digest


def test_secondary_input_pins_are_development_only():
    value = contract()["inputs"]
    assert set(value["partitions"]) == {"train", "validation"}
    assert value["partitions"]["train"] == baselines._OFFICIAL_TRAIN_SHA256
    assert value["partitions"]["validation"] == baselines._OFFICIAL_VALIDATION_SHA256
    assert value["forbidden_roles"] == ["group_test", "external", "PhishVN"]


def test_secondary_logistic_parameters_match_the_declared_baseline():
    params = {
        key: value
        for key, value in baselines._CLASSIFIER_CONFIG.items()
        if key != "class"
    }
    assert contract()["tabular"]["logistic_parameters"] == LogisticRegression(
        **params
    ).get_params(deep=False)
    assert contract()["tabular"]["scaler_parameters"] == StandardScaler().get_params(
        deep=False
    )


def test_secondary_feature_names_and_permutation_family_are_fixed():
    value = contract()["tabular"]
    assert value["formatting_features"] == list(
        secondary_probes.FORMATTING_FEATURE_NAMES
    )
    assert value["structural_features"] == list(url_features.FEATURE_NAMES)
    assert value["permutation"]["seeds"] == [42, 43, 44, 45, 46]
    assert value["permutation"]["generator"] == "PCG64"
    assert value["permutation"]["solver_seed"] == 42
    assert value["permutation"]["validation_labels_permuted"] is False
    assert value["permutation"]["permutation_test_pvalue"] is False


def test_secondary_parameters_and_versions_match_the_implementation():
    from automated_phishing_detection import secondary_tabular

    value = contract()
    assert (
        secondary_tabular._parameters("random_forest")
        == value["tabular"]["random_forest_parameters"]
    )
    expected = {
        "classifier": value["tabular"]["logistic_parameters"],
        "scaler": value["tabular"]["scaler_parameters"],
    }
    for kind in ("formatting", "permutation"):
        assert secondary_tabular._parameters(kind) == expected
    assert secondary_tabular._VERSIONS == value["runtime"]["versions"]


def test_existing_execution_profile_is_not_silently_extended():
    path = ROOT / "data/execution-binding-contract-v2.json"
    assert sha256(path.read_bytes()).hexdigest() == (
        "887f771381927dfe1b9268a45f4e605baf3e9a7caee2b7005cdfe68b1be516e1"
    )
    profile = json.loads(path.read_bytes())
    assert (
        "data/secondary-development-contract-v1.json"
        not in profile["public_file_sha256"]
    )
    assert profile["protected_evaluation_ready"] is False


def test_ci_uses_the_exact_recorded_python_patch_version():
    workflow = (ROOT / ".github/workflows/ci.yml").read_text()
    python = contract()["runtime"]["versions"]["python"]
    assert f'python-version: "{python}"' in workflow
    assert f"uv sync --locked --python {python}\n" in workflow


def test_random_forest_records_every_effective_parameter():
    model = RandomForestClassifier(
        n_estimators=100,
        criterion="gini",
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=1,
    )
    assert contract()["tabular"]["random_forest_parameters"] == model.get_params(
        deep=False
    )


def test_drift_reuses_original_scaler_and_allocation_without_rescue_gate():
    value = contract()["drift"]
    assert value["scaler_refit"] is False
    assert value["gmm_refit"] is False
    assert value["probability_source"] == "PortableLogisticL1.score_urls"
    original = json.loads(
        (ROOT / "data/rq2-gmm-development-contract-v1.json").read_bytes()
    )
    assert value["validation_allocation"] == original["validation_allocation"]
    assert value["hypothesis_role"] == "secondary_descriptive_only"
    assert value["window_length"] == 256
    assert value["window_stride"] == 64


@pytest.mark.parametrize("kind", ["drift", "tabular"])
def test_secondary_contract_carries_no_primary_evaluation_authorization(kind):
    value = contract()
    assert (
        value[kind]["research_input_access"] == "separate_authenticated_runner_required"
    )
    assert value["runtime"]["numerical_threads"] == 1
    assert value["runtime"]["scoring_batch_size"] == 1
    assert value["runtime"]["device"] == "cpu"
    assert value["runtime"]["numpy_blas"] == {
        "name": "scipy-openblas",
        "version": "0.3.29",
    }
