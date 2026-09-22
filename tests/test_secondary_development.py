"""Invented in-memory development partitions and loaded synthetic model state."""

import inspect
import json
from dataclasses import FrozenInstanceError, asdict, replace
from hashlib import sha256

import numpy as np
import pytest
from test_fixed_cascade import _artifact
from test_transformer_pipeline import SOURCE_SHA256, _preparation_summary, _records

from automated_phishing_detection import fixed_cascade, gmm_monitor, secondary_drift
from automated_phishing_detection.url_features import extract_url_features


def _bytes(value):
    return (
        json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()


@pytest.fixture
def development():
    from automated_phishing_detection import secondary_development

    return secondary_development


def _fixture(
    development,
    *,
    train_count=256,
    validation_count=640,
    mutate_train=None,
    mutate_validation=None,
    mutate_summary=None,
    mutate_gmm=None,
):
    train = _records(
        "train",
        negatives=train_count // 2,
        positives=train_count - train_count // 2,
        ordinal_start=1,
    )
    validation = _records(
        "validation",
        negatives=validation_count // 2,
        positives=validation_count - validation_count // 2,
        ordinal_start=1000,
    )
    for rows in (train, validation):
        for index, row in enumerate(rows):
            row["raw_url"] += "a" * (index % 17) + f"?q={index * index}"
            row["canonical_url_sha256"] = sha256(row["raw_url"].encode()).hexdigest()
    if mutate_train:
        mutate_train(train)
    if mutate_validation:
        mutate_validation(validation)
    train_content = b"".join(_bytes(row) for row in train)
    validation_content = b"".join(_bytes(row) for row in validation)
    suffix_rules = b"example\n"
    summary = _preparation_summary(train, validation, train_content, validation_content)
    for split, rows in (("train", train), ("validation", validation)):
        summary["splits"][split]["domain_count"] = len(
            {row["registrable_domain"] for row in rows}
        )
    summary["declared_sources"]["public_suffix_list"]["sha256"] = sha256(
        suffix_rules
    ).hexdigest()
    if mutate_summary:
        mutate_summary(summary)
    preparation = _bytes(summary)
    baseline_hash = "b" * 64
    logistic_artifact = _artifact(baseline_hash)
    logistic_artifact["scaler"]["n_samples_seen"] = train_count
    logistic_artifact["input_hashes"].update(
        train=sha256(train_content).hexdigest(),
        validation=sha256(validation_content).hexdigest(),
        preparation_summary=sha256(preparation).hexdigest(),
    )
    logistic_bytes = _bytes(logistic_artifact)
    logistic = fixed_cascade._load_logistic_l1_artifact_bytes(
        logistic_bytes,
        expected_sha256=sha256(logistic_bytes).hexdigest(),
        expected_contract_sha256=baseline_hash,
    )
    input_hashes = {
        "train": sha256(train_content).hexdigest(),
        "validation": sha256(validation_content).hexdigest(),
        "preparation_summary": sha256(preparation).hexdigest(),
        "baseline_contract": baseline_hash,
        "logistic_l1_artifact": logistic.artifact_sha256,
        "gmm_contract": "c" * 64,
    }
    scale = [3.0 + index / 7 for index in range(26)]
    gmm = {
        "schema_version": 1,
        "contract_id": gmm_monitor.CONTRACT_ID,
        "features": list(gmm_monitor.GMM_FEATURE_NAMES),
        "dtype": "float64",
        "input_hashes": dict(input_hashes),
        "scaler": {
            "mean": [0.1 + index / 11 for index in range(26)],
            "scale": scale,
            "variance": [value * value for value in scale],
            "n_samples_seen": train_count,
            "n_features_in": 26,
        },
        "mixture": {
            "components": 1,
            "covariance_type": "diag",
            "weights": [1.0],
            "means": [[0.0] * 26],
            "variances": [[1.0] * 26],
            "precisions": [[1.0] * 26],
            "precisions_cholesky": [[1.0] * 26],
            "converged": True,
            "n_iter": 1,
            "lower_bound": -1.0,
        },
    }
    if mutate_gmm:
        mutate_gmm(gmm)
    pins = development.DevelopmentPins(
        train_sha256=input_hashes["train"],
        validation_sha256=input_hashes["validation"],
        source_csv_sha256=SOURCE_SHA256,
        preparation_summary_sha256=input_hashes["preparation_summary"],
        suffix_rules_sha256=sha256(suffix_rules).hexdigest(),
        logistic_l1_artifact_sha256=logistic.artifact_sha256,
        baseline_contract_sha256=baseline_hash,
        gmm_artifact_sha256=sha256(gmm_monitor._canonical_json_bytes(gmm)).hexdigest(),
        gmm_contract_sha256=input_hashes["gmm_contract"],
    )
    return {
        "train": train,
        "validation": validation,
        "validation_content": validation_content,
        "arguments": {
            "train_content": train_content,
            "preparation_summary": preparation,
            "suffix_rules": suffix_rules,
            "logistic_l1": logistic,
            "gmm_state": gmm,
            "pins": pins,
        },
    }


def test_training_constructor_accepts_no_validation_record_argument(development):
    assert (
        "validation_content"
        not in inspect.signature(development.build_training_reference).parameters
    )
    data = _fixture(development)
    reference = development.build_training_reference(**data["arguments"])
    assert reference.mmd.reason is None
    assert reference.psi.training_row_count == 256
    assert reference.training_record_ids == tuple(
        row["record_id"] for row in data["train"]
    )
    with pytest.raises(FrozenInstanceError):
        reference.scaler_mean = ()


def test_standardization_matches_original_subtract_then_divide_exactly(development):
    data = _fixture(development)
    arguments = data["arguments"]
    reference = development.build_training_reference(**arguments)
    urls = tuple(row["raw_url"] for row in data["train"])
    raw = np.column_stack(
        (
            np.asarray([extract_url_features(url) for url in urls], dtype=np.float64),
            arguments["logistic_l1"].score_urls(urls),
        )
    )
    mean = np.asarray(arguments["gmm_state"]["scaler"]["mean"], dtype=np.float64)
    scale = np.asarray(arguments["gmm_state"]["scaler"]["scale"], dtype=np.float64)
    expected = (raw - mean) / scale
    assert np.any(expected != (raw - mean) * (1 / scale))
    by_id = {row["record_id"]: index for index, row in enumerate(data["train"])}
    actual = np.asarray(reference.mmd.values)
    selected = [by_id[identity] for identity in reference.mmd.stable_ids]
    np.testing.assert_array_equal(actual, expected[selected])
    assert reference.psi == secondary_drift.fit_psi_reference(expected)


def test_validation_reuses_fixed_references_original_allocation_and_private_order(
    development,
):
    data = _fixture(development)
    reference = development.build_training_reference(**data["arguments"])
    result = development.evaluate_validation(reference, data["validation_content"])
    assert set(result.private_outputs) == {
        "training-reference.json",
        "validation-audit.json",
    }
    assert (
        result.private_outputs["training-reference.json"] == reference.private_payload
    )
    audit = json.loads(result.private_outputs["validation-audit.json"])
    allocation = gmm_monitor.allocate_validation_domains(
        [row["registrable_domain"] for row in data["validation"]]
    )
    for stream, indices in allocation.items():
        trace = audit["streams"][stream]
        assert trace["input_row_positions"] == list(indices)
        assert trace["record_ids"] == [
            data["validation"][i]["record_id"] for i in indices
        ]
        assert trace["domains"] == [
            data["validation"][i]["registrable_domain"] for i in indices
        ]
        assert trace["window_end_positions"] == [256, 320]
        assert len(trace["psi"]["feature_scores"]) == 2
    for method in ("mmd", "psi"):
        calibration = audit["streams"]["calibration"][method]["scores"]
        scores = audit["streams"]["audit"][method]["scores"]
        boundary = float(np.quantile(calibration, 0.95, method="linear"))
        summary = result.public_summary["methods"][method]
        assert summary["status"] == "estimated"
        assert summary["threshold"] == boundary
        assert summary["audit_alert_count"] == sum(score > boundary for score in scores)


def test_evaluation_never_refits_or_calls_detector_or_gmm_likelihood(
    development, monkeypatch
):
    data = _fixture(development)
    reference = development.build_training_reference(**data["arguments"])

    def forbidden(*args, **kwargs):
        pytest.fail("unexpected fit, detector scorer, or primary likelihood")

    monkeypatch.setattr(secondary_drift, "fit_mmd_reference", forbidden)
    monkeypatch.setattr(secondary_drift, "fit_psi_reference", forbidden)
    monkeypatch.setattr(fixed_cascade, "score_logistic_l1_authoritative", forbidden)
    monkeypatch.setattr(gmm_monitor, "fit_training_mixture", forbidden)
    monkeypatch.setattr(gmm_monitor, "score_feature_matrix", forbidden)
    result = development.evaluate_validation(reference, data["validation_content"])
    assert result.public_summary["status"] == "completed_development_validation"


def test_loaded_input_mutation_cannot_change_reference_or_validation_scores(
    development,
):
    data = _fixture(development)
    reference = development.build_training_reference(**data["arguments"])
    before = development.evaluate_validation(reference, data["validation_content"])
    data["arguments"]["gmm_state"]["scaler"]["mean"][0] += 99
    data["train"][0]["raw_url"] = "not used"
    after = development.evaluate_validation(reference, data["validation_content"])
    assert before == after


@pytest.mark.parametrize(
    "role", ["train_content", "preparation_summary", "suffix_rules"]
)
def test_input_hash_mismatch_stops_before_portable_scoring(
    development, monkeypatch, role
):
    data = _fixture(development)
    data["arguments"][role] += b" "
    monkeypatch.setattr(
        fixed_cascade.PortableLogisticL1,
        "score_urls",
        lambda *args: pytest.fail("scored invalid input"),
    )
    with pytest.raises(development.SecondaryDevelopmentError):
        development.build_training_reference(**data["arguments"])


def test_validation_hash_mismatch_stops_before_scoring(development, monkeypatch):
    data = _fixture(development)
    reference = development.build_training_reference(**data["arguments"])
    monkeypatch.setattr(
        fixed_cascade.PortableLogisticL1,
        "score_urls",
        lambda *args: pytest.fail("scored invalid input"),
    )
    with pytest.raises(development.SecondaryDevelopmentError):
        development.evaluate_validation(reference, data["validation_content"] + b" ")


@pytest.mark.parametrize(
    "mutation",
    [
        "order",
        "duplicate_id",
        "source",
        "split",
        "canonical",
        "domain",
        "bool_label",
        "extra",
    ],
)
def test_partition_schema_identity_order_and_psl_are_validated(development, mutation):
    def mutate(rows):
        if mutation == "order":
            rows[0], rows[1] = rows[1], rows[0]
        elif mutation == "duplicate_id":
            rows[1]["record_id"] = rows[0]["record_id"]
        elif mutation == "source":
            rows[0]["record_id"] = rows[0]["record_id"].replace(SOURCE_SHA256, "9" * 64)
        elif mutation == "split":
            rows[0]["split"] = "group_test"
        elif mutation == "canonical":
            rows[0]["canonical_url_sha256"] = "9" * 64
        elif mutation == "domain":
            rows[0]["registrable_domain"] = "invented.example"
        elif mutation == "bool_label":
            rows[0]["is_phishing"] = False
        else:
            rows[0]["unrecognized"] = 1

    data = _fixture(development, mutate_train=mutate)
    with pytest.raises(development.SecondaryDevelopmentError):
        development.build_training_reference(**data["arguments"])


@pytest.mark.parametrize("kind", ["domain", "identifier"])
def test_validation_overlap_is_rejected_before_probability_scoring(
    development, monkeypatch, kind
):
    def mutate(rows):
        if kind == "identifier":
            rows[0]["record_id"] = f"phiusiil-row-v1:{SOURCE_SHA256}:{1:016x}"
        else:
            rows[0]["raw_url"] = "https://tr-0001.example/different"
            rows[0]["canonical_url_sha256"] = sha256(
                rows[0]["raw_url"].encode()
            ).hexdigest()
            rows[0]["registrable_domain"] = "tr-0001.example"

    data = _fixture(development, mutate_validation=mutate)
    reference = development.build_training_reference(**data["arguments"])
    monkeypatch.setattr(
        fixed_cascade.PortableLogisticL1,
        "score_urls",
        lambda *args: pytest.fail("scored overlap"),
    )
    with pytest.raises(development.SecondaryDevelopmentError):
        development.evaluate_validation(reference, data["validation_content"])


@pytest.mark.parametrize(
    "kind", ["input_hash", "scale_zero", "sample_count", "feature_order"]
)
def test_gmm_scaler_artifact_and_original_training_chain_are_bound(development, kind):
    def mutate(gmm):
        if kind == "input_hash":
            gmm["input_hashes"]["train"] = "9" * 64
        elif kind == "scale_zero":
            gmm["scaler"]["scale"][0] = 0
        elif kind == "sample_count":
            gmm["scaler"]["n_samples_seen"] += 1
        else:
            gmm["features"][0], gmm["features"][1] = (
                gmm["features"][1],
                gmm["features"][0],
            )

    data = _fixture(development, mutate_gmm=mutate)
    with pytest.raises(development.SecondaryDevelopmentError):
        development.build_training_reference(**data["arguments"])


def test_logistic_model_requires_loaded_marker_and_exact_hash(development):
    data = _fixture(development)
    original = data["arguments"]["logistic_l1"]
    for altered in (
        replace(original, _loader_marker=None),
        replace(original, artifact_sha256="9" * 64),
    ):
        data["arguments"]["logistic_l1"] = altered
        with pytest.raises(development.SecondaryDevelopmentError):
            development.build_training_reference(**data["arguments"])


def test_nonestimable_reference_and_short_streams_remain_explicit(development):
    data = _fixture(development, train_count=16)
    reference = development.build_training_reference(**data["arguments"])
    result = development.evaluate_validation(reference, data["validation_content"])
    assert result.public_summary["status"] == "partially_estimable"
    assert (
        result.public_summary["methods"]["mmd"]["reason"]
        == "fewer_than_256_training_domains"
    )
    assert result.public_summary["methods"]["mmd"]["threshold"] is None
    small = _fixture(development, validation_count=16)
    reference = development.build_training_reference(**small["arguments"])
    result = development.evaluate_validation(reference, small["validation_content"])
    assert result.public_summary["status"] == "not_estimable"
    for method in result.public_summary["methods"].values():
        assert method["threshold"] is None
        assert method["audit_alert_fraction"] is None
        assert method["reason"] == "no_complete_256_row_window"


def test_public_summary_is_aggregate_only_and_hashes_match_private_payloads(
    development,
):
    data = _fixture(development)
    result = development.evaluate_validation(
        development.build_training_reference(**data["arguments"]),
        data["validation_content"],
    )
    public_text = json.dumps(result.public_summary)
    assert result.public_summary["contract_id"] == "secondary-development-v1"
    assert result.public_summary["protected_evaluation_authorized"] is False
    for forbidden in (
        "raw_url",
        "record_ids",
        "domains",
        "is_phishing",
        "false_alert_gate_met",
        "H2",
    ):
        assert f'"{forbidden}"' not in public_text
    assert "https://" not in public_text
    for name, content in result.private_outputs.items():
        assert (
            result.public_summary["private_sha256"][name] == sha256(content).hexdigest()
        )
        assert b'"is_phishing"' not in content
    assert result.public_summary["input_hashes"] == asdict(data["arguments"]["pins"])


def test_portable_nonfinite_probability_is_rejected(development, monkeypatch):
    data = _fixture(development)
    monkeypatch.setattr(
        fixed_cascade.PortableLogisticL1,
        "score_urls",
        lambda self, urls: (float("nan"),) * len(urls),
    )
    with pytest.raises(development.SecondaryDevelopmentError):
        development.build_training_reference(**data["arguments"])


def test_runtime_failure_prevents_scoring(development, monkeypatch):
    data = _fixture(development)
    monkeypatch.setattr(
        gmm_monitor,
        "_require_runtime",
        lambda: (_ for _ in ()).throw(ValueError("wrong runtime")),
    )
    monkeypatch.setattr(
        fixed_cascade.PortableLogisticL1,
        "score_urls",
        lambda *args: pytest.fail("scored with wrong runtime"),
    )
    with pytest.raises(development.SecondaryDevelopmentError):
        development.build_training_reference(**data["arguments"])


def test_unavailable_private_results_preserve_actual_windows_and_original_reason(
    development,
):
    data = _fixture(development, train_count=16)
    reference = development.build_training_reference(**data["arguments"])
    result = development.evaluate_validation(reference, data["validation_content"])
    private = json.loads(result.private_outputs["validation-audit.json"])["results"][
        "mmd"
    ]
    reason = "fewer_than_256_training_domains"
    assert private["calibration"]["reason"] == reason
    assert private["calibration"]["calibration_window_count"] == 2
    assert private["audit"]["window_count"] == 2
    assert private["audit"]["reason"] == reason
    assert private["audit"]["alert_count"] is None


def test_validation_labels_do_not_change_allocation_or_probe_scores(development):
    def flip(rows):
        for row in rows:
            row["is_phishing"] = 1 - row["is_phishing"]

    original = _fixture(development)
    flipped = _fixture(development, mutate_validation=flip)
    first = development.build_training_reference(**original["arguments"])
    second = development.build_training_reference(**flipped["arguments"])
    assert first.mmd == second.mmd
    assert first.psi == second.psi
    results = [
        development.evaluate_validation(ref, data["validation_content"])
        for ref, data in ((first, original), (second, flipped))
    ]
    traces = [
        json.loads(result.private_outputs["validation-audit.json"])["streams"]
        for result in results
    ]
    assert traces[0] == traces[1]
    assert results[0].public_summary["methods"] == results[1].public_summary["methods"]


def test_audit_only_changes_never_retune_calibration_or_training_references(
    development,
):
    original = _fixture(development)
    indices = set(
        gmm_monitor.allocate_validation_domains(
            [row["registrable_domain"] for row in original["validation"]]
        )["audit"]
    )

    def change_audit(rows):
        for index in indices:
            row = rows[index]
            row["raw_url"] += "z" * 100
            row["canonical_url_sha256"] = sha256(row["raw_url"].encode()).hexdigest()

    changed = _fixture(development, mutate_validation=change_audit)
    first = development.build_training_reference(**original["arguments"])
    second = development.build_training_reference(**changed["arguments"])
    assert first.mmd == second.mmd
    assert first.psi == second.psi
    before = development.evaluate_validation(first, original["validation_content"])
    after = development.evaluate_validation(second, changed["validation_content"])
    for method in ("mmd", "psi"):
        assert (
            before.public_summary["methods"][method]["threshold"]
            == after.public_summary["methods"][method]["threshold"]
        )
    before_traces = json.loads(before.private_outputs["validation-audit.json"])[
        "streams"
    ]
    after_traces = json.loads(after.private_outputs["validation-audit.json"])["streams"]
    assert before_traces["calibration"] == after_traces["calibration"]
    assert (
        before_traces["audit"]["mmd"]["scores"]
        != after_traces["audit"]["mmd"]["scores"]
    )


def test_composition_uses_strict_calibration_boundary(development, monkeypatch):
    data = _fixture(development)
    reference = development.build_training_reference(**data["arguments"])
    calls = []

    def controlled_scores(*args):
        scores = (1.0, 1.0) if not calls else (1.0, float(np.nextafter(1.0, np.inf)))
        calls.append(True)
        return secondary_drift.DriftWindowScores((256, 320), scores)

    monkeypatch.setattr(secondary_drift, "mmd_window_scores", controlled_scores)
    result = development.evaluate_validation(reference, data["validation_content"])
    assert result.public_summary["methods"]["mmd"]["threshold"] == 1.0
    assert result.public_summary["methods"]["mmd"]["audit_alert_count"] == 1


def test_reference_tampering_is_rejected(development):
    data = _fixture(development)
    reference = development.build_training_reference(**data["arguments"])
    for altered in (
        replace(reference, scaler_mean=(99.0,) + reference.scaler_mean[1:]),
        replace(
            reference,
            portable_model=replace(reference.portable_model, _loader_marker=None),
        ),
    ):
        with pytest.raises(development.SecondaryDevelopmentError):
            development.evaluate_validation(altered, data["validation_content"])
