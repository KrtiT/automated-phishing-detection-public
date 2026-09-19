import copy
import importlib.util
import json
import os
import subprocess
import sys
import textwrap
from hashlib import sha256
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/inference_compatibility.py"


@pytest.fixture
def bridge():
    assert SCRIPT.is_file(), "the no-fit compatibility bridge is not implemented"
    spec = importlib.util.spec_from_file_location("inference_compatibility", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _lanes():
    reference = {
        "record_ids": ["PRIVATE_ROW_0", "PRIVATE_ROW_1"],
        "labels": [0, 1],
        "probabilities": {
            "length-only": [0.2, 0.8],
            "Logistic-L1": [0.2, 0.8],
            "transformer": [0.2, 0.8],
        },
        "decisions": {
            name: [0, 1]
            for name in ("length-only", "Logistic-L1", "transformer", "cascade")
        },
        "band_selected": [False, True],
    }
    candidate = copy.deepcopy(reference)
    audit = {
        "warning_records": [],
        "max_absolute_decision_difference": 0.0,
        "max_absolute_probability_difference": 0.0,
    }
    reference["execution_counts"] = {
        "reference_batch_size": 512,
        "length_scoring_audit": audit,
        "stage1_scoring_audit": audit,
    }
    candidate["execution_counts"] = {
        "transformer_forward_attempts": 2,
        "successful_transformer_scores": 2,
        "completed_requests": 2,
        "failed_requests": 0,
    }
    candidate["gmm"] = {
        "reference_reproduces_saved_traces": True,
        "reference_nll": [-1.0, 1.0],
        "candidate_nll": [-1.0, 1.0],
        "threshold": 1.0,
        "streams": {
            name: {"reference_scores": [1.0, 2.0], "candidate_scores": [1.0, 2.0]}
            for name in ("calibration", "audit")
        },
    }
    counts = {
        name: {
            "true_negative": 1,
            "false_positive": 0,
            "false_negative": 0,
            "true_positive": 1,
            "negative": 1,
            "positive": 1,
        }
        for name in reference["decisions"]
    }
    return reference, candidate, counts


def test_exact_binary_band_and_strict_window_agreement(bridge):
    reference, candidate, counts = _lanes()
    candidate["probabilities"]["transformer"][0] += 1e-15
    candidate["gmm"]["candidate_nll"][0] += 1e-15
    result = bridge._compare_lanes(reference, candidate, counts)
    assert result["status"] == "equivalent_on_development_validation"
    assert result["protected_evaluation_ready"] is False
    assert result["band_mismatch_count"] == 0
    assert result["models"]["transformer"]["max_absolute_probability_difference"] > 0
    assert result["gmm"]["max_absolute_nll_difference"] > 0
    assert result["gmm"]["streams"]["audit"]["reference_alert_count"] == 1
    assert (
        result["comparison_execution"]["candidate"]["transformer_forward_attempts"] == 2
    )
    assert (
        result["comparison_execution"]["reference_scoring_audits"]["Logistic-L1"][
            "warning_count"
        ]
        == 0
    )
    assert all(row["decision_mismatch_count"] == 0 for row in result["models"].values())
    serialized = json.dumps(result)
    for private in (
        "PRIVATE",
        "record_ids",
        "labels",
        "candidate_nll",
        "reference_scores",
    ):
        assert private not in serialized


@pytest.mark.parametrize(
    "change", ["decision", "band", "calibration", "audit", "reference"]
)
def test_any_exact_mismatch_is_preserved_as_not_equivalent(bridge, change):
    reference, candidate, counts = _lanes()
    if change == "decision":
        candidate["decisions"]["cascade"][0] = 1
    elif change == "band":
        candidate["band_selected"][0] = True
    elif change in ("calibration", "audit"):
        candidate["gmm"]["streams"][change]["candidate_scores"][0] = float(
            np.nextafter(1.0, 2.0)
        )
    else:
        counts["transformer"]["true_positive"] = 0
    result = bridge._compare_lanes(reference, candidate, counts)
    assert result["status"] == "not_equivalent"
    assert result["protected_evaluation_ready"] is False


@pytest.mark.parametrize(
    "change",
    [
        "order",
        "duplicate",
        "label",
        "bool_label",
        "nan",
        "shape",
        "binary",
        "bool_band",
    ],
)
def test_invalid_or_misaligned_lane_results_fail_closed(bridge, change):
    reference, candidate, counts = _lanes()
    if change == "order":
        candidate["record_ids"].reverse()
    elif change == "duplicate":
        reference["record_ids"][1] = reference["record_ids"][0]
    elif change == "label":
        candidate["labels"][0] = 1
    elif change == "bool_label":
        reference["labels"][0] = False
    elif change == "nan":
        candidate["probabilities"]["transformer"][0] = float("nan")
    elif change == "shape":
        candidate["decisions"]["cascade"].pop()
    elif change == "binary":
        candidate["decisions"]["cascade"][0] = 0.0
    else:
        candidate["band_selected"][0] = 0
    with pytest.raises(ValueError):
        bridge._compare_lanes(reference, candidate, counts)


def test_reference_transformer_uses_original_no_grad_helper_and_final_batch(
    bridge, monkeypatch
):
    import torch

    from automated_phishing_detection import character_transformer

    observed = []

    def original(model, loader, device):
        assert isinstance(loader, torch.utils.data.DataLoader)
        assert loader.batch_size == 512 and loader.drop_last is False
        assert loader.num_workers == 0
        assert isinstance(loader.sampler, torch.utils.data.SequentialSampler)
        for tokens, padding, labels in loader:
            observed.append(len(tokens))
        return 0.5, tuple([0.25] * 513)

    monkeypatch.setattr(character_transformer, "_evaluate_validation", original)
    tokens = torch.zeros((513, 256), dtype=torch.int64)
    padding = torch.ones_like(tokens, dtype=torch.bool)
    labels = torch.tensor([0] * 256 + [1] * 257, dtype=torch.float32)
    assert bridge._reference_transformer(
        object(), tokens, padding, labels, torch.device("cpu")
    ) == tuple([0.25] * 513)
    assert observed == [512, 1]


def test_authenticate_before_decoding_and_reject_symlinks(bridge, tmp_path):
    source = tmp_path / "fixture.json"
    content = b"not even json"
    source.write_bytes(content)
    assert bridge._read_verified(source, sha256(content).hexdigest()) == content
    with pytest.raises(ValueError, match="SHA-256"):
        bridge._read_verified(source, "0" * 64)
    link = tmp_path / "link.json"
    link.symlink_to(source)
    with pytest.raises(ValueError, match="regular"):
        bridge._read_verified(link, sha256(content).hexdigest())


def test_input_whitelist_has_no_training_or_protected_partitions(bridge):
    names = list(bridge.INPUTS)
    assert "data/processed/phiusiil-v1/validation.jsonl" in names
    assert not any(
        "train.jsonl" in name or "group_test" in name or "phishvn" in name.lower()
        for name in names
    )
    assert len(names) == len(set(names))
    assert all(len(digest) == 64 for digest in bridge.INPUTS.values())


def test_private_outputs_are_exclusive_and_mode_600(bridge, tmp_path):
    output = tmp_path / "reference.json"
    bridge._write_private(output, {"PRIVATE": "fixture"})
    assert output.stat().st_mode & 0o777 == 0o600
    with pytest.raises(FileExistsError):
        bridge._write_private(output, {"PRIVATE": "replacement"})
    assert json.loads(output.read_bytes()) == {"PRIVATE": "fixture"}


def test_receipt_publication_does_not_replace_existing_results(bridge, tmp_path):
    output = tmp_path / "receipt.json"
    bridge._publish(output, {"status": "not_equivalent"})
    assert output.stat().st_mode & 0o777 == 0o644
    with pytest.raises(FileExistsError):
        bridge._publish(output, {"status": "equivalent_on_development_validation"})
    assert json.loads(output.read_bytes()) == {"status": "not_equivalent"}
    assert list(tmp_path.iterdir()) == [output]


def test_private_output_rejects_nonfinite_before_publication(bridge, tmp_path):
    output = tmp_path / "reference.json"
    with pytest.raises(ValueError):
        bridge._write_private(output, {"score": float("nan")})
    assert not output.exists()


def test_contract_requires_exact_bridge_rules_and_hash(bridge, tmp_path):
    value = {
        "protected_evaluation_ready": False,
        "runtime_candidate": bridge.RUNTIME_REQUIREMENTS,
        "compatibility_bridge": bridge.BRIDGE_REQUIREMENTS,
    }
    path = tmp_path / "contract.json"
    content = json.dumps(value).encode()
    path.write_bytes(content)
    bridge._verify_contract(path, sha256(content).hexdigest())
    with pytest.raises(ValueError, match="SHA-256"):
        bridge._verify_contract(path, "0" * 64)
    value["runtime_candidate"] = dict(
        value["runtime_candidate"], inference_batch_size=512
    )
    content = json.dumps(value).encode()
    path.write_bytes(content)
    with pytest.raises(ValueError, match="rules"):
        bridge._verify_contract(path, sha256(content).hexdigest())


def _gmm_fixture():
    from automated_phishing_detection import gmm_monitor as gm

    partition = {
        "raw_urls": ("https://a.example/",) * 256 + ("https://b.example/",) * 256,
        "features": np.ones((512, 25), dtype=np.float64),
        "domains": ("a.example",) * 256 + ("b.example",) * 256,
        "record_ids": tuple(f"PRIVATE_{i}" for i in range(512)),
    }

    class Stage1:
        def score_urls(self, urls):
            return (0.5,) * len(urls)

    artifact = {
        "schema_version": 1,
        "contract_id": gm.CONTRACT_ID,
        "features": list(gm.GMM_FEATURE_NAMES),
        "dtype": "float64",
        "input_hashes": {},
        "scaler": {
            "mean": [0.0] * 26,
            "scale": [1.0] * 26,
            "variance": [1.0] * 26,
            "n_samples_seen": 20,
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
    scores = gm.score_feature_matrix(gm._build_features(partition, Stage1()), artifact)
    traces = {}
    for name, indices in gm.allocate_validation_domains(partition["domains"]).items():
        ends, window = gm.window_scores(scores[list(indices)])
        traces[name] = {
            "input_row_positions": list(indices),
            "record_ids": [partition["record_ids"][i] for i in indices],
            "domains": sorted({partition["domains"][i] for i in indices}),
            "window_end_positions": list(ends),
            "window_scores": list(window),
        }
    summary = {
        "threshold": traces["calibration"]["window_scores"][0],
        "audit_window_count": 1,
        "calibration_window_count": 1,
        "audit_alert_count": 0,
    }
    return partition, Stage1(), artifact, traces, summary


def test_gmm_bridge_uses_saved_disjoint_order_and_portable_stage_one(
    bridge, monkeypatch
):
    from automated_phishing_detection import gmm_monitor as gm

    fixture = _gmm_fixture()
    original = gm.score_feature_matrix
    shapes = []

    def observe(features, artifact):
        shapes.append(features.shape)
        return original(features, artifact)

    monkeypatch.setattr(gm, "score_feature_matrix", observe)
    result = bridge._gmm_comparison(*fixture)
    assert result["reference_reproduces_saved_traces"] is True
    assert shapes == [(512, 26)] + [(1, 26)] * 512
    assert result["candidate_nll"] == result["reference_nll"]
    assert (
        result["streams"]["audit"]["reference_scores"]
        == result["streams"]["audit"]["candidate_scores"]
    )


def test_changed_saved_gmm_scores_are_not_rescued(bridge, monkeypatch):
    from automated_phishing_detection import gmm_monitor as gm

    fixture = _gmm_fixture()
    fixture[3]["calibration"]["window_scores"][0] += 0.1
    original = gm.score_feature_matrix
    shapes = []

    def observe(features, artifact):
        shapes.append(features.shape)
        return original(features, artifact)

    monkeypatch.setattr(gm, "score_feature_matrix", observe)
    result = bridge._gmm_comparison(*fixture)
    assert result["reference_reproduces_saved_traces"] is False
    assert shapes == [(512, 26)]


@pytest.mark.parametrize(
    "field", ["input_row_positions", "record_ids", "domains", "window_end_positions"]
)
def test_gmm_saved_membership_alignment_is_required(bridge, field):
    fixture = _gmm_fixture()
    fixture[3]["calibration"][field] = []
    with pytest.raises(ValueError, match="alignment"):
        bridge._gmm_comparison(*fixture)


def test_partition_loader_checks_declared_counts_and_order(bridge):
    from test_transformer_pipeline import (
        _canonical_json_bytes,
        _preparation_summary,
        _records,
    )

    from automated_phishing_detection.baselines import _validate_preparation_summary

    train = _records("train", negatives=2, positives=2, ordinal_start=1)
    rows = _records("validation", negatives=2, positives=2, ordinal_start=10)
    train_bytes = b"".join(map(_canonical_json_bytes, train))
    content = b"".join(map(_canonical_json_bytes, rows))
    preparation = _validate_preparation_summary(
        _preparation_summary(train, rows, train_bytes, content)
    )
    result = bridge._partition(content, preparation)
    assert result["record_ids"] == tuple(row["record_id"] for row in rows)
    assert result["labels"].tolist() == [row["is_phishing"] for row in rows]
    with pytest.raises(ValueError):
        bridge._partition(
            b"".join(map(_canonical_json_bytes, reversed(rows))), preparation
        )


def test_orchestrator_preflights_both_roles_and_cleans_private_rows_on_success(
    bridge, tmp_path, monkeypatch
):
    _orchestrator_fixture(bridge, tmp_path, monkeypatch)
    reference, candidate, counts = _lanes()
    observed, outputs = [], []

    def child(role, interpreter, contract_hash, output=None):
        observed.append((role, output is None))
        if output is None:
            return {"role": role}, {"role": role, "exit_code": 0}
        assert output.parent.stat().st_mode & 0o777 == 0o700
        outputs.append(output)
        bridge._write_private(output, reference if role == "reference" else candidate)
        return None, {"role": role, "exit_code": 0}

    result = bridge._run_bridge(
        "python1", "python2", "f" * 64, _root=tmp_path, _invoke=child
    )
    assert observed == [
        ("reference", True),
        ("candidate", True),
        ("reference", False),
        ("candidate", False),
    ]
    assert result["status"] == "equivalent_on_development_validation"
    assert all(not path.exists() and not path.parent.exists() for path in outputs)
    assert json.loads((tmp_path / bridge.REPORT).read_bytes()) == result


def _orchestrator_fixture(bridge, root, monkeypatch):
    (root / "reports").mkdir()
    monkeypatch.setattr(bridge, "_verify_contract", lambda *args: {})
    monkeypatch.setattr(
        bridge,
        "_source_bindings",
        lambda *args: {"head": "a" * 40, "source_sha256": {}},
    )
    monkeypatch.setattr(bridge, "_accepted_counts", lambda *args: _lanes()[2])
    monkeypatch.setattr(bridge, "_accepted_band_count", lambda *args: 1)


def test_environment_failure_prevents_any_lane_input_read_and_preserves_receipt(
    bridge, tmp_path, monkeypatch
):
    _orchestrator_fixture(bridge, tmp_path, monkeypatch)
    calls = []

    def child(role, interpreter, contract_hash, output=None):
        calls.append((role, output))
        if role == "candidate":
            raise ValueError("PRIVATE_ROW_URL_CANARY")
        return {"role": role}, {"role": role, "exit_code": 0}

    result = bridge._run_bridge(
        "python1", "python2", "f" * 64, _root=tmp_path, _invoke=child
    )
    assert result["status"] == "failed"
    assert result["failure_stage"] == "candidate_environment_preflight"
    assert all(output is None for _, output in calls)
    assert "PRIVATE" not in json.dumps(result)
    assert result["protected_evaluation_ready"] is False
    assert (tmp_path / bridge.REPORT).exists()


def test_mismatch_is_preserved_without_retry_and_private_files_removed(
    bridge, tmp_path, monkeypatch
):
    _orchestrator_fixture(bridge, tmp_path, monkeypatch)
    reference, candidate, counts = _lanes()
    candidate["band_selected"][0] = True
    outputs = []

    def child(role, interpreter, contract_hash, output=None):
        if output is not None:
            outputs.append(output)
            bridge._write_private(
                output, reference if role == "reference" else candidate
            )
        return {"role": role}, {"role": role, "exit_code": 0}

    result = bridge._run_bridge(
        "python1", "python2", "f" * 64, _root=tmp_path, _invoke=child
    )
    assert result["status"] == "not_equivalent"
    assert len(outputs) == 2 and all(not output.exists() for output in outputs)
    with pytest.raises(FileExistsError):
        bridge._run_bridge(
            "python1", "python2", "f" * 64, _root=tmp_path, _invoke=child
        )
    assert len(outputs) == 2


def test_child_exception_cleans_all_private_files_without_leaking_error(
    bridge, tmp_path, monkeypatch
):
    _orchestrator_fixture(bridge, tmp_path, monkeypatch)
    outputs = []

    def child(role, interpreter, contract_hash, output=None):
        if output is not None:
            outputs.append(output)
            bridge._write_private(output, _lanes()[0])
            if role == "candidate":
                raise RuntimeError("PRIVATE_URL_CANARY")
        return {"role": role}, {"role": role, "exit_code": 0}

    result = bridge._run_bridge(
        "python1", "python2", "f" * 64, _root=tmp_path, _invoke=child
    )
    assert result["status"] == "failed"
    assert result["failure_stage"] == "candidate_scoring"
    assert len(outputs) == 2 and all(not output.parent.exists() for output in outputs)
    assert "PRIVATE" not in json.dumps(result)


def test_bad_original_counts_stop_before_candidate_scoring(
    bridge, tmp_path, monkeypatch
):
    _orchestrator_fixture(bridge, tmp_path, monkeypatch)
    reference, _, _ = _lanes()
    reference["decisions"]["length-only"][0] = 1
    calls = []

    def child(role, interpreter, contract_hash, output=None):
        if output is not None:
            calls.append(role)
            assert role == "reference"
            bridge._write_private(output, reference)
        return {"role": role}, {"role": role, "exit_code": 0}

    result = bridge._run_bridge(
        "python1", "python2", "f" * 64, _root=tmp_path, _invoke=child
    )
    assert result["status"] == "not_equivalent"
    assert result["reason"] == "original_validation_counts_not_reproduced"
    assert calls == ["reference"]


def test_unchanged_confusion_counts_do_not_hide_changed_original_band(
    bridge, tmp_path, monkeypatch
):
    _orchestrator_fixture(bridge, tmp_path, monkeypatch)
    reference, _, _ = _lanes()
    reference["band_selected"] = [False, False]
    calls = []

    def child(role, interpreter, contract_hash, output=None):
        if output is not None:
            calls.append(role)
            assert role == "reference"
            bridge._write_private(output, reference)
        return {"role": role}, {"role": role, "exit_code": 0}

    result = bridge._run_bridge(
        "python1", "python2", "f" * 64, _root=tmp_path, _invoke=child
    )
    assert result["status"] == "not_equivalent"
    assert result["reason"] == "original_band_selection_count_not_reproduced"
    assert result["accepted_band_selected_count"] == 1
    assert calls == ["reference"]


def test_original_source_identity_checks_function_bodies(bridge, monkeypatch, tmp_path):
    monkeypatch.setattr(bridge, "REFERENCE_FUNCTIONS", {"module.py": ("score",)})
    monkeypatch.setattr(
        bridge, "_git", lambda *args: b"def score(x):\n    return x + 1\n"
    )
    path = tmp_path / "module.py"
    path.write_text("def score(x):\n    return x + 1\n")
    assert "module.py:score" in bridge._reference_fidelity(tmp_path)
    path.write_text("def score(x):\n    return x + 2\n")
    with pytest.raises(ValueError, match="original"):
        bridge._reference_fidelity(tmp_path)


def test_candidate_lane_records_actual_core_masks_and_decisions(
    bridge, tmp_path, monkeypatch
):
    from dataclasses import replace

    from test_length_inference import _load
    from test_transformer_inference import _build_fixture, _load_fixture

    from automated_phishing_detection.selective_inference import SelectiveCascade

    loaded = _load_fixture(_build_fixture(tmp_path))
    partition = {
        "raw_urls": ("https://a.example/", "https://b.example/long/path"),
        "record_ids": ("PRIVATE_0", "PRIVATE_1"),
        "labels": np.array([0, 1]),
    }
    original = SelectiveCascade.score_all
    observed = []

    def injected(self, raw_url):
        row = original(self, raw_url)
        row = replace(
            row,
            fixed_decision=1 - row.fixed_decision,
            band_selected=not row.band_selected,
        )
        observed.append(row)
        return row

    monkeypatch.setattr(SelectiveCascade, "score_all", injected)
    result = bridge._score_models(
        partition, loaded, _load(), "candidate", _fixture_cpu=True
    )
    assert result["band_selected"] == [row.band_selected for row in observed]
    assert result["decisions"]["cascade"] == [row.fixed_decision for row in observed]


def test_actual_synthetic_model_lanes_score_without_any_fit(
    bridge, tmp_path, monkeypatch
):
    import torch
    from sklearn.linear_model import LogisticRegression
    from test_length_inference import _load
    from test_transformer_inference import _build_fixture, _load_fixture

    from automated_phishing_detection import character_transformer, transformer_pipeline

    loaded = _load_fixture(_build_fixture(tmp_path))
    partition = {
        "raw_urls": ("https://a.example/", "https://b.example/long/path"),
        "record_ids": ("PRIVATE_0", "PRIVATE_1"),
        "labels": np.array([0, 1]),
    }

    def forbidden(*args, **kwargs):
        raise AssertionError("fitting must not be called")

    monkeypatch.setattr(LogisticRegression, "fit", forbidden)
    monkeypatch.setattr(character_transformer, "fit_character_transformer", forbidden)
    monkeypatch.setattr(transformer_pipeline, "_train_transformer", forbidden)
    previous = torch.are_deterministic_algorithms_enabled()
    try:
        reference = bridge._score_models(
            partition, loaded, _load(), "reference", _fixture_cpu=True
        )
        candidate = bridge._score_models(
            partition, loaded, _load(), "candidate", _fixture_cpu=True
        )
    finally:
        torch.use_deterministic_algorithms(previous)
    assert (
        reference["record_ids"] == candidate["record_ids"] == ["PRIVATE_0", "PRIVATE_1"]
    )
    assert reference["labels"] == candidate["labels"] == [0, 1]
    assert reference["decisions"] == candidate["decisions"]
    assert set(reference["probabilities"]) == set(bridge.PROBABILITY_MODELS)
    assert candidate["execution_counts"]["transformer_forward_attempts"] == 2
    assert candidate["execution_counts"]["completed_requests"] == 2


def test_environment_check_rejects_mps_fallback_without_input_access(
    bridge, monkeypatch
):
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    with pytest.raises(ValueError, match="fallback"):
        bridge._environment("candidate")


def test_command_rejects_arbitrary_data_paths_without_exposing_arguments(
    bridge, capsys
):
    assert bridge.main(["--validation", "PRIVATE_PROTECTED_CANARY"]) == 2
    output = capsys.readouterr()
    assert "PRIVATE" not in output.out + output.err
    assert json.loads(output.out)["status"] == "failed"


def test_candidate_thread_limit_failure_precedes_any_lane_input_read(
    bridge, monkeypatch, tmp_path
):
    import threadpoolctl
    import torch

    from automated_phishing_detection import gmm_monitor as gm

    monkeypatch.setattr(bridge, "_verify_contract", lambda *args: {})
    monkeypatch.setattr(bridge, "_source_bindings", lambda *args: {})
    monkeypatch.setattr(bridge.sys, "platform", "darwin")
    monkeypatch.setattr(bridge.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(bridge.platform, "python_version", lambda: "3.10.19")
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setattr(gm, "_require_runtime", lambda: None)
    monkeypatch.setattr(
        gm,
        "_software_versions",
        lambda: {
            "numpy": "2.2.6",
            "scipy": "1.15.3",
            "scikit-learn": "1.7.2",
            "threadpoolctl": "3.6.0",
        },
    )
    monkeypatch.setattr(torch, "__version__", "2.7.1")
    monkeypatch.setattr(threadpoolctl, "threadpool_info", lambda: [{"num_threads": 2}])
    reads = []
    monkeypatch.setattr(bridge, "_read_verified", lambda *args: reads.append(args))
    with pytest.raises(ValueError, match="one numerical thread"):
        bridge._execute_lane("candidate", "f" * 64, tmp_path / "candidate.json")
    assert reads == []


def test_candidate_preflight_initializes_torch_before_limiting_in_fresh_process():
    program = textwrap.dedent(
        """
        import importlib.util
        import json
        import sys
        from types import SimpleNamespace

        import torch
        from automated_phishing_detection import gmm_monitor as gm

        spec = importlib.util.spec_from_file_location("bridge", sys.argv[1])
        bridge = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(bridge)
        # Isolate the real thread-pool probe from hardware/version eligibility.
        bridge.sys = SimpleNamespace(platform="darwin", executable=sys.executable)
        bridge.platform = SimpleNamespace(python_version=lambda: "3.10.19",
                                         machine=lambda: "arm64", platform=lambda: "synthetic")
        torch.backends.mps.is_available = lambda: True
        torch.__version__ = "2.7.1"
        gm._require_runtime = lambda: None
        gm._software_versions = lambda: {"numpy": "2.2.6", "scipy": "1.15.3",
                                        "scikit-learn": "1.7.2", "threadpoolctl": "3.6.0"}
        original_get = torch.get_num_threads
        calls = []
        def observe_get():
            value = original_get()
            calls.append(value)
            return value
        torch.get_num_threads = observe_get
        result = bridge._environment("candidate")
        print(json.dumps({"entry": result["torch_intraop_threads_before_scoring"],
                          "calls": calls, "restored": original_get()}))
        """
    )
    child = subprocess.run(
        [sys.executable, "-s", "-c", program, str(SCRIPT)],
        env=dict(os.environ, PYTHONPATH=str(SCRIPT.parents[1] / "src")),
        capture_output=True,
        text=True,
        check=False,
    )
    assert child.returncode == 0, child.stderr
    result = json.loads(child.stdout)
    assert result["entry"] >= 1
    assert result["calls"] == [result["entry"], 1]
    assert result["restored"] == result["entry"]


def test_child_stage_is_sanitized_in_process_failure_record(bridge, monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr(
        bridge.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=2,
            stdout=bridge._json_bytes(
                {
                    "status": "failed",
                    "failure_stage": "gmm_reference_and_candidate",
                    "failure_type": "ValueError",
                }
            ),
            stderr=b"PRIVATE_URL_CANARY",
        ),
    )
    with pytest.raises(ChildProcessError) as caught:
        bridge._invoke_child(
            "candidate", bridge.sys.executable, "f" * 64, Path("/tmp/private.json")
        )
    record = caught.value.process_record
    assert record["failure_stage"] == "gmm_reference_and_candidate"
    assert "PRIVATE" not in json.dumps(record)


def test_unknown_child_stage_is_not_echoed(bridge, monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr(
        bridge.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=2,
            stdout=bridge._json_bytes(
                {
                    "status": "failed",
                    "failure_stage": "PRIVATE_URL_CANARY",
                    "failure_type": "ValueError",
                }
            ),
            stderr=b"PRIVATE_URL_CANARY",
        ),
    )
    with pytest.raises(ChildProcessError) as caught:
        bridge._invoke_child(
            "candidate", bridge.sys.executable, "f" * 64, Path("/tmp/private.json")
        )
    assert caught.value.process_record["failure_stage"] == "unknown_child_stage"
    assert "PRIVATE" not in json.dumps(caught.value.process_record)


def _correction_fixture(bridge, tmp_path, monkeypatch):
    _orchestrator_fixture(bridge, tmp_path, monkeypatch)
    prior = {
        "schema_version": 1,
        "contract_id": bridge.BRIDGE_REQUIREMENTS["id"],
        "head": "6977cec7acaa5491caaad8b1bba140b4134a7fcf",
        "contract_sha256": "0a42fc0f27abc611431cbbf1263bb9c2f02264942332ed8478b8a305e0dae9b9",
        "status": "failed",
        "failure_stage": "candidate_environment_preflight",
        "fit_performed": False,
        "protected_evaluation_ready": False,
        "processes": [
            {"role": "reference", "phase": "preflight", "exit_code": 0},
            {
                "role": "candidate",
                "phase": "preflight",
                "exit_code": 2,
                "failure_stage": "environment_preflight",
            },
        ],
    }
    content = bridge._json_bytes(prior)
    policy = dict(
        bridge.CORRECTION_REQUIREMENTS, prior_receipt_sha256=sha256(content).hexdigest()
    )
    monkeypatch.setattr(bridge, "CORRECTION_REQUIREMENTS", policy)
    monkeypatch.setattr(
        bridge,
        "_verify_contract",
        lambda *args: {"compatibility_bridge": {"preflight_correction": policy}},
    )
    (tmp_path / bridge.REPORT).write_bytes(content)
    return prior, content, policy


def test_explicit_preflight_correction_preserves_original_and_has_own_single_receipt(
    bridge, tmp_path, monkeypatch
):
    _, prior_content, policy = _correction_fixture(bridge, tmp_path, monkeypatch)
    reference, candidate, _ = _lanes()
    calls = []

    def child(role, interpreter, contract_hash, output=None):
        calls.append((role, output is None))
        if output is not None:
            bridge._write_private(
                output, reference if role == "reference" else candidate
            )
        return {"role": role}, {"role": role, "exit_code": 0}

    result = bridge._run_bridge(
        "python1",
        "python2",
        "f" * 64,
        preflight_correction=True,
        _root=tmp_path,
        _invoke=child,
    )
    assert result["status"] == "equivalent_on_development_validation"
    assert result["contract_id"] == policy["id"]
    assert (
        result["preflight_correction"]["prior_receipt_sha256"]
        == policy["prior_receipt_sha256"]
    )
    assert (
        result["preflight_correction"]["change"]
        == "initialize_torch_thread_runtime_before_limiting"
    )
    assert (tmp_path / policy["report"]).exists()
    assert (tmp_path / bridge.REPORT).read_bytes() == prior_content
    for correction in (False, True):
        with pytest.raises(FileExistsError):
            bridge._run_bridge(
                "python1",
                "python2",
                "f" * 64,
                preflight_correction=correction,
                _root=tmp_path,
                _invoke=child,
            )
    assert len(calls) == 4


@pytest.mark.parametrize(
    "mutation",
    [
        "hash",
        "stage",
        "head",
        "contract",
        "phase",
        "process_count",
        "comparison",
        "fit",
        "contract_exception",
    ],
)
def test_correction_rejects_invalid_predecessor_before_any_environment_or_data_call(
    bridge, tmp_path, monkeypatch, mutation
):
    prior, _, policy = _correction_fixture(bridge, tmp_path, monkeypatch)
    if mutation in ("hash", "stage"):
        prior["failure_stage"] = "candidate_scoring"
    elif mutation == "head":
        prior["head"] = "1" * 40
    elif mutation == "contract":
        prior["contract_sha256"] = "1" * 64
    elif mutation == "phase":
        prior["processes"][1]["phase"] = "scoring"
    elif mutation == "process_count":
        prior["processes"].append(prior["processes"][1])
    elif mutation == "comparison":
        prior["models"] = {}
    elif mutation == "fit":
        prior["fit_performed"] = True
    else:
        monkeypatch.setattr(bridge, "_verify_contract", lambda *args: {})
    content = bridge._json_bytes(prior)
    (tmp_path / bridge.REPORT).write_bytes(content)
    if mutation != "hash":
        policy["prior_receipt_sha256"] = sha256(content).hexdigest()
    calls = []
    with pytest.raises(ValueError):
        bridge._run_bridge(
            "python1",
            "python2",
            "f" * 64,
            preflight_correction=True,
            _root=tmp_path,
            _invoke=lambda *args: calls.append(args),
        )
    assert calls == []
    assert not (tmp_path / policy["report"]).exists()


def test_correction_failure_is_preserved_and_never_automatically_retried(
    bridge, tmp_path, monkeypatch
):
    _, prior_content, policy = _correction_fixture(bridge, tmp_path, monkeypatch)
    calls = []

    def child(*args):
        calls.append(args)
        raise ValueError("synthetic failure")

    result = bridge._run_bridge(
        "python1",
        "python2",
        "f" * 64,
        preflight_correction=True,
        _root=tmp_path,
        _invoke=child,
    )
    assert result["status"] == "failed"
    assert len(calls) == 1
    with pytest.raises(FileExistsError):
        bridge._run_bridge(
            "python1",
            "python2",
            "f" * 64,
            preflight_correction=True,
            _root=tmp_path,
            _invoke=child,
        )
    assert len(calls) == 1
    assert (tmp_path / bridge.REPORT).read_bytes() == prior_content


def test_public_correction_flag_is_explicit_and_parent_only(
    bridge, monkeypatch, capsys
):
    calls = []

    def execute(*args, **kwargs):
        calls.append(kwargs)
        return {"status": "equivalent_on_development_validation"}

    monkeypatch.setattr(bridge, "_run_bridge", execute)
    arguments = [
        "--reference-python",
        "python1",
        "--candidate-python",
        "python2",
        "--contract-sha256",
        "f" * 64,
    ]
    assert bridge.main(arguments + ["--preflight-correction"]) == 0
    assert calls == [{"preflight_correction": True}]
    capsys.readouterr()


def test_correction_contract_may_include_hash_bound_explanatory_prose(
    bridge, tmp_path, monkeypatch
):
    _, _, policy = _correction_fixture(bridge, tmp_path, monkeypatch)
    contract = {
        "compatibility_bridge": {
            "preflight_correction": dict(policy, reason="Synthetic explanation")
        }
    }
    result = bridge._verify_preflight_correction(tmp_path, contract)
    assert result["prior_receipt_sha256"] == policy["prior_receipt_sha256"]
