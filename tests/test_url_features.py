import inspect
import json
import math
from hashlib import sha256
from pathlib import Path

import pytest

from automated_phishing_detection import url_features

ROOT = Path(__file__).resolve().parents[1]
CONTRACT_V1 = ROOT / "data" / "rq1-baseline-contract.json"
CONTRACT = ROOT / "data" / "rq1-baseline-contract-v2.json"
PROTOCOL = ROOT / "docs" / "advisor-approval" / "2026-08-16-realignment-matrix.md"
STATUS = ROOT / "docs" / "advisor-approval" / "approval-status.md"
EVIDENCE_OUTLINE = ROOT / "docs" / "research-evidence-outline.md"

FEATURE_NAMES = (
    "raw_url_codepoint_length",
    "raw_url_utf8_byte_length",
    "hostname_ascii_length",
    "path_codepoint_length",
    "query_codepoint_length",
    "fragment_codepoint_length",
    "hostname_label_count",
    "hostname_ascii_digit_count",
    "hostname_hyphen_count",
    "hostname_punycode_label_count",
    "path_segment_count",
    "query_parameter_count",
    "raw_url_ascii_letter_count",
    "raw_url_ascii_digit_count",
    "raw_url_other_codepoint_count",
    "raw_url_ascii_digit_ratio",
    "raw_url_other_codepoint_ratio",
    "raw_url_unique_codepoint_count",
    "raw_url_utf8_byte_entropy_bits",
    "percent_escape_count",
    "is_https",
    "has_userinfo",
    "has_explicit_port",
    "has_query_delimiter",
    "has_fragment_delimiter",
)


def _contract(path=CONTRACT):
    return json.loads(path.read_text(encoding="utf-8"))


def test_v1_contract_remains_byte_identical_and_v2_preserves_method_boundaries():
    assert sha256(CONTRACT_V1.read_bytes()).hexdigest() == (
        "594a66769dee3bf23c4133020dcf9b7d57c105590e5007832ac4249def6a33d4"
    )

    v1 = _contract(CONTRACT_V1)
    v2 = _contract()
    for field in (
        "input",
        "output",
        "predictor_policy",
        "features",
        "partition_use",
        "score",
        "threshold_selection",
    ):
        assert v2[field] == v1[field]
    assert v2["models"]["length-only"] == v1["models"]["length-only"]
    assert v2["models"]["Logistic-L1"] == v1["models"]["Logistic-L1"]
    assert v2["models"]["search"] == v1["models"]["search"]


def test_feature_contract_freezes_order_types_and_denominators():
    contract = _contract()

    assert contract["contract_id"] == "rq1-baselines-v2"
    assert contract["schema_version"] == 2
    assert contract["protocol_version"] == "1.7"
    assert contract["output"] == {
        "container": "ordered_vector",
        "dtype": "float64",
        "feature_count": 25,
    }

    features = contract["features"]
    assert tuple(feature["name"] for feature in features) == FEATURE_NAMES
    assert [feature["position"] for feature in features] == list(range(1, 26))
    assert all(
        set(feature) == {"position", "name", "definition", "dtype", "denominator"}
        for feature in features
    )
    assert all(feature["dtype"] == "float64" for feature in features)
    assert all(feature["definition"] for feature in features)
    assert all(feature["denominator"] for feature in features)
    assert features[15]["denominator"] == "raw_url_codepoint_length"
    assert features[16]["denominator"] == "raw_url_codepoint_length"
    assert features[18]["denominator"] == "raw_url_utf8_byte_length"


def test_contract_freezes_input_policy_and_forbidden_predictors():
    contract = _contract()

    assert contract["input"] == {
        "field": "raw_url",
        "validation": "canonical-url-v1 preparation rules",
        "missing_or_invalid_action": "error",
        "error_message": "raw_url is missing or invalid under canonical-url-v1",
        "imputation": False,
    }
    assert contract["predictor_policy"]["basis"] == "raw_url only"
    forbidden = set(contract["predictor_policy"]["forbidden"])
    assert {
        "label",
        "native_label",
        "is_phishing",
        "split",
        "record_id",
        "canonical_url",
        "canonical_url_sha256",
        "registrable_domain",
        "source",
        "source_class",
        "confidence_tier",
        "published_split",
        "published_record_id",
    } <= forbidden


def test_contract_freezes_model_and_threshold_selection():
    contract = _contract()
    common = contract["models"]["common_pipeline"]

    assert common == {
        "scaler": {
            "class": "StandardScaler",
            "fit_partition": "train",
            "with_mean": True,
            "with_std": True,
        },
        "classifier": {
            "class": "LogisticRegression",
            "penalty": "l1",
            "solver": "saga",
            "C": 1.0,
            "class_weight": "balanced",
            "fit_intercept": True,
            "max_iter": 5000,
            "tol": 1e-4,
            "random_state": 42,
        },
        "convergence_warning_action": "error",
    }
    assert contract["models"]["length-only"]["features"] == ["raw_url_codepoint_length"]
    assert tuple(contract["models"]["Logistic-L1"]["features"]) == FEATURE_NAMES
    assert contract["models"]["search"] == "none"

    assert contract["score"] == {
        "value": "P(is_phishing=1)",
        "alert_rule": "score >= threshold",
    }
    assert contract["partition_use"] == {
        "fit_scaler_and_classifier": "train only",
        "select_threshold": "validation only",
    }
    assert contract["threshold_selection"] == {
        "candidate_thresholds": (
            "unique validation scores plus the finite no-alert threshold "
            "nextafter(maximum validation score, +infinity)"
        ),
        "objective": "maximize validation recall",
        "constraint": (
            "exact one-sided 95% Clopper-Pearson false-positive-rate upper "
            "confidence bound <= 0.01"
        ),
        "tie_breaks": [
            "smaller false-positive-rate upper confidence bound",
            "higher threshold",
        ],
        "no_feasible_candidate": "target_not_met",
    }


def test_contract_freezes_scoring_integrity_policy():
    assert _contract()["scoring_integrity"] == {
        "policy_id": "rq1-scoring-integrity-v1",
        "stages": ["decision_function", "predict_proba"],
        "default_warning_action": "error",
        "allowed_warning": {
            "environment": {
                "sys_platform": "darwin",
                "platform_machine": "arm64",
                "numpy_blas_name": "accelerate",
            },
            "category": "RuntimeWarning",
            "module": "sklearn.utils.extmath",
            "messages": [
                "divide by zero encountered in matmul",
                "overflow encountered in matmul",
                "invalid value encountered in matmul",
            ],
            "action": "capture-and-verify",
        },
        "reference": {
            "dtype": "float64",
            "decision": (
                "np.einsum('ij,j->i', scaled, coef[0], optimize=False)+intercept"
            ),
            "probability": "[1-expit(decision), expit(decision)]",
            "finite_required": True,
            "rtol": 1e-12,
            "atol": 1e-12,
        },
        "authoritative_threshold_input": ("sklearn predict_proba class-1 column"),
        "emitted_aggregate_fields": [
            "platform_identity",
            "warning_records",
            "max_absolute_decision_difference",
            "max_absolute_probability_difference",
        ],
    }


def test_extractor_api_is_pure_and_uses_one_raw_url_argument():
    assert url_features.FEATURE_NAMES == FEATURE_NAMES
    assert tuple(inspect.signature(url_features.extract_url_features).parameters) == (
        "raw_url",
    )


def test_extracts_exact_unicode_authority_and_component_features():
    raw_url = "https://user:pw@bücher.example:8443/a//β?q=1&x=%2f#frag"

    actual = url_features.extract_url_features(raw_url)

    expected = (
        55.0,
        57.0,
        21.0,
        5.0,
        9.0,
        4.0,
        2.0,
        0.0,
        3.0,
        1.0,
        2.0,
        2.0,
        31.0,
        6.0,
        18.0,
        6.0 / 55.0,
        18.0 / 55.0,
        33.0,
        4.9095708828824485,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
    )
    assert actual == pytest.approx(expected, rel=0.0, abs=1e-15)
    assert all(type(value) is float and math.isfinite(value) for value in actual)


def test_character_counts_exactly_partition_raw_url_length():
    features = dict(
        zip(
            FEATURE_NAMES,
            url_features.extract_url_features(
                "https://user:pw@bücher.example:8443/a//β?q=1&x=%2f#frag"
            ),
        )
    )

    assert features["raw_url_codepoint_length"] == (
        features["raw_url_ascii_letter_count"]
        + features["raw_url_ascii_digit_count"]
        + features["raw_url_other_codepoint_count"]
    )


def test_empty_query_and_fragment_delimiters_are_distinct_from_content():
    features = dict(
        zip(FEATURE_NAMES, url_features.extract_url_features("http://example.com?#"))
    )

    assert features["query_codepoint_length"] == 0.0
    assert features["fragment_codepoint_length"] == 0.0
    assert features["query_parameter_count"] == 0.0
    assert features["has_query_delimiter"] == 1.0
    assert features["has_fragment_delimiter"] == 1.0
    assert features["is_https"] == 0.0
    assert features["has_userinfo"] == 0.0
    assert features["has_explicit_port"] == 0.0


def test_percent_escape_count_includes_only_valid_raw_percent_triplets():
    features = dict(
        zip(
            FEATURE_NAMES,
            url_features.extract_url_features(
                "https://u%41:p%42@example.com:443/a%2fb?q=%2F#%aa"
            ),
        )
    )

    assert features["percent_escape_count"] == 5.0
    assert features["has_userinfo"] == 1.0
    assert features["has_explicit_port"] == 1.0


@pytest.mark.parametrize(
    "raw_url",
    (
        None,
        "",
        17,
        "example.com/path",
        "ftp://example.com/path",
        "https://example.com/a%ZZ",
        "https://127.0.0.1/path",
        "https://example.com/white space",
        "https://example.com/\ud800",
    ),
)
def test_missing_or_invalid_urls_raise_the_frozen_error(raw_url):
    with pytest.raises(
        url_features.FeatureExtractionError,
        match="^raw_url is missing or invalid under canonical-url-v1$",
    ):
        url_features.extract_url_features(raw_url)


def test_v1_contract_sha_is_recorded_in_each_v14_research_record():
    expected = sha256(CONTRACT_V1.read_bytes()).hexdigest()
    marker = f"`{expected}`"

    for path in (PROTOCOL, STATUS, EVIDENCE_OUTLINE):
        text = path.read_text(encoding="utf-8")
        assert "1.4" in text
        assert marker in text
