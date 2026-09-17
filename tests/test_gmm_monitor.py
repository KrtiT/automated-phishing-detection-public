import importlib
import inspect
import json
import os
import stat
import warnings
from dataclasses import replace
from hashlib import sha256
from pathlib import Path

import numpy as np
import pytest
from sklearn.mixture import GaussianMixture
from test_transformer_pipeline import (
    _canonical_json_bytes,
    _logistic_artifact,
    _preparation_summary,
    _records,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "data/rq2-gmm-development-contract-v1.json"


def module():
    return importlib.import_module("automated_phishing_detection.gmm_monitor")


def fixture_paths(tmp_path):
    gm = module()
    train = _records("train", negatives=16, positives=16, ordinal_start=1)
    validation = _records("validation", negatives=600, positives=40, ordinal_start=1000)
    for rows in (train, validation):
        for i, row in enumerate(rows):
            row["raw_url"] += "/a" * (i % 19) + "?n=" + str(i * i) + "z" * (i % 7)
            row["canonical_url_sha256"] = sha256(row["raw_url"].encode()).hexdigest()
    train_bytes = b"".join(_canonical_json_bytes(row) for row in train)
    validation_bytes = b"".join(_canonical_json_bytes(row) for row in validation)
    baseline = (ROOT / "data/rq1-baseline-contract-v2.json").read_bytes()
    contents = {
        "train": train_bytes,
        "validation": validation_bytes,
        "preparation_summary": _canonical_json_bytes(
            _preparation_summary(train, validation, train_bytes, validation_bytes)
        ),
        "baseline_contract": baseline,
        "logistic_l1_artifact": _canonical_json_bytes(
            _logistic_artifact(
                np.array([r["is_phishing"] for r in validation]),
                baseline_contract_sha256=sha256(baseline).hexdigest(),
            )
        ),
        "gmm_contract": CONTRACT.read_bytes(),
    }
    paths = {key + "_path": tmp_path / (key + ".json") for key in contents}
    for key, content in contents.items():
        paths[key + "_path"].write_bytes(content)
    paths["output_dir"] = tmp_path / "private-gmm"
    paths["summary_path"] = tmp_path / "summary.json"
    paths["_input_hash_policy"] = gm._InputHashPolicy(
        **{
            key + "_sha256": sha256(value).hexdigest()
            for key, value in contents.items()
        }
    )
    return paths


def test_allocation_is_label_blind_domain_disjoint_and_keeps_row_order():
    gm = module()
    domains = (
        "C.EXAMPLE.",
        "a.example",
        "b.example",
        "c.example",
        "d.example",
        "e.example",
    )
    result = gm.allocate_validation_domains(domains)
    normalized = [d.lower().removesuffix(".") for d in domains]
    ranked = sorted(
        set(normalized),
        key=lambda d: (
            sha256(
                ("rq2-gmm-validation-v1\0" + "20260816\0" + d).encode("ascii")
            ).digest(),
            d,
        ),
    )
    expected = set(ranked[: len(ranked) // 2])
    assert result == {
        "calibration": tuple(i for i, d in enumerate(normalized) if d in expected),
        "audit": tuple(i for i, d in enumerate(normalized) if d not in expected),
    }
    assert len(set(normalized[i] for i in result["calibration"])) == 2
    assert all(tuple(sorted(indices)) == indices for indices in result.values())


def test_complete_windows_quantile_strict_alert_and_exact_gate_boundary():
    gm = module()
    nll = np.arange(383, dtype=np.float64)
    ends, scores = gm.window_scores(nll)
    assert ends == (256, 320)
    assert scores == (127.5, 191.5)
    result = gm.calibrate_and_audit([0.0, 20.0], [19.0] * 19 + [20.0])
    assert result["threshold"] == 19.0
    assert result["audit_alert_count"] == 1
    assert result["audit_window_count"] == 20
    assert result["false_alert_gate_met"] is True
    assert (
        gm.calibrate_and_audit([0.0, 20.0], [20.0] + [19.0] * 18)[
            "false_alert_gate_met"
        ]
        is False
    )
    for values in ([], [float("nan")], [float("inf")]):
        with pytest.raises(gm.GMMMonitorError):
            gm.window_scores(values)
    with pytest.raises(gm.GMMMonitorError, match="complete window"):
        gm.window_scores(np.ones(255))


def test_actual_six_candidate_fits_training_scaler_and_portable_roundtrip():
    gm = module()
    matrix = np.random.default_rng(20260816).normal(size=(32, 26))
    matrix[:, 25] = 0.5
    artifact, candidates = gm.fit_training_mixture(matrix)
    assert [row["components"] for row in candidates] == list(range(1, 7))
    assert all(
        row["converged"] and row["n_iter"] > 0 and np.isfinite(row["bic"])
        for row in candidates
    )
    assert (
        artifact["mixture"]["components"]
        == min(candidates, key=lambda c: (c["bic"], c["components"]))["components"]
    )
    np.testing.assert_allclose(artifact["scaler"]["mean"], matrix.mean(axis=0))
    assert artifact["scaler"]["scale"][-1] == 1.0
    assert artifact["scaler"]["variance"][-1] == 0.0
    loaded = gm.load_gmm_artifact_bytes(gm._canonical_json_bytes(artifact))
    np.testing.assert_array_equal(
        gm.score_feature_matrix(matrix, loaded),
        gm.score_feature_matrix(matrix, artifact),
    )
    state = artifact["mixture"]
    reference = GaussianMixture(
        n_components=state["components"], covariance_type="diag"
    )
    reference.weights_ = np.array(state["weights"])
    reference.means_ = np.array(state["means"])
    reference.covariances_ = np.array(state["variances"])
    reference.precisions_cholesky_ = np.array(state["precisions_cholesky"])
    scaled = (matrix - np.array(artifact["scaler"]["mean"])) / np.array(
        artifact["scaler"]["scale"]
    )
    np.testing.assert_allclose(
        gm.score_feature_matrix(matrix, loaded),
        -reference.score_samples(scaled),
        rtol=1e-12,
        atol=1e-12,
    )


@pytest.mark.parametrize(
    "failure",
    ["warning", "nonconvergence", "nonfinite", "zero_weight", "negative_variance"],
)
def test_any_candidate_failure_stops_without_retry(monkeypatch, failure):
    gm = module()
    original = gm.GaussianMixture.fit
    calls = []

    def fit(self, matrix):
        calls.append(self.n_components)
        result = original(self, matrix)
        if self.n_components == 2:
            if failure == "warning":
                warnings.warn("fixture warning", RuntimeWarning)
            elif failure == "nonconvergence":
                self.converged_ = False
            elif failure == "nonfinite":
                self.means_[0, 0] = np.nan
            elif failure == "zero_weight":
                self.weights_[0] = 0.0
            else:
                self.covariances_[0, 0] = -1.0
        return result

    monkeypatch.setattr(gm.GaussianMixture, "fit", fit)
    with pytest.raises(gm.GMMMonitorError):
        gm.fit_training_mixture(np.random.default_rng(1).normal(size=(32, 26)))
    assert calls == [1, 2]


def test_real_fixture_pipeline_private_traces_and_public_aggregate_only(tmp_path):
    gm = module()
    paths = fixture_paths(tmp_path)
    result = gm._fit_gmm_monitor(**paths)
    assert result["status"] == "completed_development_validation"
    assert result["hypothesis_status"] == {
        "H1": "undecided",
        "H2": "undecided",
        "H3": "undecided",
    }
    assert result["access"] == {
        "group_test_accessed": False,
        "external_accessed": False,
        "phishvn_accessed": False,
        "scope": "this_process_only",
    }
    assert len(result["candidates"]) == 6
    assert json.loads(paths["summary_path"].read_bytes()) == result
    directory = paths["output_dir"]
    assert set(p.name for p in directory.iterdir()) == {
        "gmm.json",
        "validation-audit.json",
        "SHA256SUMS",
    }
    assert stat.S_IMODE(directory.stat().st_mode) == 0o700
    for path in directory.iterdir():
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
    audit = json.loads((directory / "validation-audit.json").read_bytes())
    model = gm.load_gmm_artifact_bytes((directory / "gmm.json").read_bytes())
    assert model["scaler"]["n_samples_seen"] == 32
    assert model["scaler"]["n_features_in"] == 26
    assert model["scaler"]["mean"][-1] == 0.5
    assert len(audit["calibration"]["record_ids"]) == 320
    assert len(audit["audit"]["record_ids"]) == 320
    assert not set(audit["calibration"]["domains"]) & set(audit["audit"]["domains"])
    assert audit["calibration"]["window_end_positions"] == [256, 320]
    assert audit["audit"]["window_end_positions"] == [256, 320]
    for name, expected in result["artifact_hashes"].items():
        assert sha256((directory / name).read_bytes()).hexdigest() == expected
    public = paths["summary_path"].read_text()
    for private in (
        '"record_ids"',
        '"raw_url"',
        '"registrable_domain"',
        '"weights"',
        "va-",
        "row-v1:",
    ):
        assert private not in public


def test_hash_guard_runs_before_parse_or_fit(tmp_path, monkeypatch):
    gm = module()
    paths = fixture_paths(tmp_path)
    paths["train_path"].write_bytes(b"not json")
    monkeypatch.setattr(
        gm, "fit_training_mixture", lambda *_: pytest.fail("fit reached")
    )
    with pytest.raises(gm.GMMMonitorError, match="SHA-256 mismatch"):
        gm._fit_gmm_monitor(**paths)
    assert not paths["output_dir"].exists()
    assert not paths["summary_path"].exists()


@pytest.mark.parametrize("kind", ["symlink", "hardlink", "alias_output", "incomplete"])
def test_path_guards_preserve_existing_destinations(tmp_path, kind):
    gm = module()
    paths = fixture_paths(tmp_path)
    if kind == "symlink":
        replacement = tmp_path / "train-link.json"
        replacement.symlink_to(paths["train_path"])
        paths["train_path"] = replacement
    elif kind == "hardlink":
        paths["validation_path"].unlink()
        os.link(paths["train_path"], paths["validation_path"])
    elif kind == "alias_output":
        paths["summary_path"] = paths["train_path"]
    else:
        paths["output_dir"].mkdir()
        (paths["output_dir"] / "keep").write_text("existing")
    with pytest.raises(gm.GMMMonitorError):
        gm._fit_gmm_monitor(**paths)
    assert paths["train_path"].exists()
    if kind == "incomplete":
        assert (paths["output_dir"] / "keep").read_text() == "existing"


def test_contract_semantic_drift_rejected_under_fixture_policy(tmp_path):
    gm = module()
    paths = fixture_paths(tmp_path)
    contract = json.loads(paths["gmm_contract_path"].read_bytes())
    contract["contract_id"] = "retuned"
    content = _canonical_json_bytes(contract)
    paths["gmm_contract_path"].write_bytes(content)
    paths["_input_hash_policy"] = replace(
        paths["_input_hash_policy"], gmm_contract_sha256=sha256(content).hexdigest()
    )
    with pytest.raises(gm.GMMMonitorError, match="frozen"):
        gm._fit_gmm_monitor(**paths)


def test_summary_failure_rolls_back_private_and_preserves_competitor(
    tmp_path, monkeypatch
):
    gm = module()
    paths = fixture_paths(tmp_path)
    publish = gm._publish_path_without_replace

    def fail(source, destination):
        if destination == paths["summary_path"]:
            destination.write_text("competitor")
            raise OSError("injected publication failure")
        publish(source, destination)

    monkeypatch.setattr(gm, "_publish_path_without_replace", fail)
    with pytest.raises(OSError, match="injected"):
        gm._fit_gmm_monitor(**paths)
    assert paths["summary_path"].read_text() == "competitor"
    assert not paths["output_dir"].exists()
    assert not list(tmp_path.glob(".*.tmp-*"))


def test_public_entry_point_has_only_frozen_path_parameters(monkeypatch):
    gm = module()
    parameters = inspect.signature(gm.fit_gmm_monitor).parameters
    assert set(parameters) == {
        "train_path",
        "validation_path",
        "preparation_summary_path",
        "baseline_contract_path",
        "logistic_l1_artifact_path",
        "gmm_contract_path",
        "output_dir",
        "summary_path",
    }
    observed = {}

    def run(**kwargs):
        observed.update(kwargs)
        return {"status": "fixture"}

    monkeypatch.setattr(gm, "_fit_gmm_monitor", run)
    assert gm.fit_gmm_monitor(**{key: Path(key) for key in parameters}) == {
        "status": "fixture"
    }
    assert observed["_input_hash_policy"] is gm._OFFICIAL_INPUT_HASH_POLICY


def test_selected_window_end_routes_only_future_requests():
    from automated_phishing_detection.proposed_routing_policy import (
        drift_override_requests,
    )

    gm = module()
    ends, scores = gm.window_scores(np.ones(600))
    audit = gm.calibrate_and_audit([0.0], scores)
    alert_ends = [end for end, score in zip(ends, scores) if score > audit["threshold"]]
    assert alert_ends[0] == 256
    assert drift_override_requests(600, alert_ends[:1]) == tuple(range(257, 513))
    assert drift_override_requests(600, alert_ends) == tuple(range(257, 601))


def test_exact_bic_ties_choose_smaller_k_and_freeze_all_fit_parameters(monkeypatch):
    gm = module()
    original = gm.GaussianMixture.fit
    calls = []

    def fit(self, matrix):
        params = self.get_params()
        assert params == {"n_components": self.n_components, **gm._MIXTURE_PARAMETERS}
        assert matrix.dtype == np.float64
        assert np.geterr() == {
            "divide": "raise",
            "over": "raise",
            "under": "ignore",
            "invalid": "raise",
        }
        assert all(
            pool["num_threads"] == 1 for pool in gm.threadpoolctl.threadpool_info()
        )
        calls.append(self.n_components)
        return original(self, matrix)

    monkeypatch.setattr(gm.GaussianMixture, "fit", fit)
    monkeypatch.setattr(gm.GaussianMixture, "bic", lambda *_: 42.0)
    artifact, _ = gm.fit_training_mixture(
        np.random.default_rng(2).normal(size=(32, 26))
    )
    assert artifact["mixture"]["components"] == 1
    assert calls == list(range(1, 7))


@pytest.mark.parametrize(
    "field,value", [("mean", True), ("scale", "1"), ("variance", False)]
)
def test_portable_loader_rejects_coerced_nonnumeric_state(field, value):
    gm = module()
    artifact, _ = gm.fit_training_mixture(
        np.random.default_rng(2).normal(size=(32, 26))
    )
    artifact["scaler"][field][0] = value
    with pytest.raises(gm.GMMMonitorError, match="numeric"):
        gm.load_gmm_artifact_bytes(gm._canonical_json_bytes(artifact))


def test_public_summary_rejects_unallowlisted_row_fields_before_publication(
    tmp_path, monkeypatch
):
    gm = module()
    paths = fixture_paths(tmp_path)
    original = gm.fit_training_mixture

    def fit(features):
        artifact, candidates = original(features)
        candidates[0]["record_id"] = "fixture-private-id"
        return artifact, candidates

    monkeypatch.setattr(gm, "fit_training_mixture", fit)
    with pytest.raises(gm.GMMMonitorError, match="public summary"):
        gm._fit_gmm_monitor(**paths)
    assert not paths["output_dir"].exists()
    assert not paths["summary_path"].exists()


def test_failed_audit_is_still_published_with_hypotheses_undecided(
    tmp_path, monkeypatch
):
    gm = module()
    paths = fixture_paths(tmp_path)
    records = [
        json.loads(line) for line in paths["validation_path"].read_bytes().splitlines()
    ]
    indices = gm.allocate_validation_domains(
        [record["registrable_domain"] for record in records]
    )
    nll = np.zeros(len(records), dtype=np.float64)
    nll[list(indices["audit"])] = 1.0
    monkeypatch.setattr(gm, "score_feature_matrix", lambda *_: nll)
    result = gm._fit_gmm_monitor(**paths)
    assert result["status"] == "completed_development_validation"
    assert result["false_alert_gate_met"] is False
    assert result["audit_alert_count"] == result["audit_window_count"] == 2
    assert set(result["hypothesis_status"].values()) == {"undecided"}


def test_80_row_backend_regression_never_suppresses_accelerate_errors():
    gm = module()
    matrix = np.random.default_rng(20260816).normal(size=(80, 26))
    matrix[:, 25] = 0.5
    backend = np.__config__.CONFIG["Build Dependencies"]["blas"]["name"]
    if backend == "accelerate":
        with pytest.raises(
            gm.GMMMonitorError,
            match="(divide by zero|overflow|invalid value|Accelerate|OpenBLAS)",
        ):
            gm.fit_training_mixture(matrix)
    else:
        _, candidates = gm.fit_training_mixture(matrix)
        assert len(candidates) == 6


def test_official_policy_binds_all_six_files_and_exact_contract():
    gm = module()
    contract = json.loads(CONTRACT.read_bytes())
    assert gm._validate_contract(contract) == contract
    assert gm.OFFICIAL_GMM_CONTRACT_SHA256 == sha256(CONTRACT.read_bytes()).hexdigest()
    actual = {
        key.removesuffix("_sha256"): value
        for key, value in vars(gm._OFFICIAL_INPUT_HASH_POLICY).items()
    }
    assert actual.pop("gmm_contract") == gm.OFFICIAL_GMM_CONTRACT_SHA256
    actual["contract"] = actual.pop("baseline_contract")
    assert actual == contract["inputs"]["accepted_roles"]


@pytest.mark.parametrize(
    "name,version",
    [
        ("accelerate", "unknown"),
        ("scipy-openblas", "0.3.290"),
        ("scipy-openblas", "0.3.28"),
    ],
)
def test_unsupported_blas_is_rejected_before_any_input_reads(
    tmp_path, monkeypatch, name, version
):
    gm = module()
    paths = fixture_paths(tmp_path)
    monkeypatch.setitem(
        np.__config__.CONFIG["Build Dependencies"],
        "blas",
        {"name": name, "version": version},
    )
    monkeypatch.setattr(
        gm.baselines,
        "_open_input_streams",
        lambda *_: pytest.fail("read inputs before BLAS preflight"),
    )
    with pytest.raises(gm.GMMMonitorError, match="OpenBLAS"):
        gm._fit_gmm_monitor(**paths)
    assert not paths["output_dir"].exists()


def test_permission_failure_cleans_own_temporary_directory(tmp_path, monkeypatch):
    gm = module()
    chmod = gm.os.chmod

    def fail(path, mode):
        if mode == 0o700:
            raise PermissionError("injected permission failure")
        return chmod(path, mode)

    monkeypatch.setattr(gm.os, "chmod", fail)
    with pytest.raises(PermissionError, match="injected"):
        gm._publish_artifacts(
            output_dir=tmp_path / "private",
            summary_path=tmp_path / "summary.json",
            contents={},
            summary={},
        )
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "stage", ["write", "private_sync", "summary_sync", "interrupt"]
)
def test_publication_failure_never_leaves_completed_outputs(
    tmp_path, monkeypatch, stage
):
    gm = module()
    output = tmp_path / "private"
    summary = tmp_path / "summary.json"
    contents = {
        "gmm.json": b"{}\n",
        "validation-audit.json": b"{}\n",
        "SHA256SUMS": b"",
    }
    if stage == "write":

        def fail(*_):
            raise OSError("injected write")

        monkeypatch.setattr(gm, "_write_file", fail)
    else:
        original = gm._fsync_directory

        def fail(path):
            if (stage == "private_sync" and output.exists()) or (
                stage == "summary_sync" and summary.exists()
            ):
                raise OSError("injected durability")
            if stage == "interrupt" and output.exists():
                raise KeyboardInterrupt("injected interrupt")
            return original(path)

        monkeypatch.setattr(gm, "_fsync_directory", fail)
    with pytest.raises((OSError, KeyboardInterrupt), match="injected"):
        gm._publish_artifacts(
            output_dir=output, summary_path=summary, contents=contents, summary={}
        )
    assert list(tmp_path.iterdir()) == []


def test_identical_fixture_inputs_produce_byte_identical_artifacts(tmp_path):
    gm = module()
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    second.mkdir()
    paths_one, paths_two = fixture_paths(first), fixture_paths(second)
    assert gm._fit_gmm_monitor(**paths_one) == gm._fit_gmm_monitor(**paths_two)
    for name in ("gmm.json", "validation-audit.json", "SHA256SUMS"):
        assert (paths_one["output_dir"] / name).read_bytes() == (
            paths_two["output_dir"] / name
        ).read_bytes()


@pytest.mark.parametrize(
    "field,match",
    [("registrable_domain", "domain crosses"), ("record_id", "identifier crosses")],
)
def test_cross_partition_identity_is_rejected_before_fit(
    tmp_path, monkeypatch, field, match
):
    gm = module()
    paths = fixture_paths(tmp_path)
    train = [json.loads(line) for line in paths["train_path"].read_bytes().splitlines()]
    validation = [
        json.loads(line) for line in paths["validation_path"].read_bytes().splitlines()
    ]
    validation[0][field] = train[0][field]
    content = b"".join(_canonical_json_bytes(row) for row in validation)
    paths["validation_path"].write_bytes(content)
    preparation = json.loads(paths["preparation_summary_path"].read_bytes())
    preparation["output_hashes"]["validation.jsonl"] = sha256(content).hexdigest()
    preparation_bytes = _canonical_json_bytes(preparation)
    paths["preparation_summary_path"].write_bytes(preparation_bytes)
    paths["_input_hash_policy"] = replace(
        paths["_input_hash_policy"],
        validation_sha256=sha256(content).hexdigest(),
        preparation_summary_sha256=sha256(preparation_bytes).hexdigest(),
    )
    monkeypatch.setattr(
        gm, "fit_training_mixture", lambda *_: pytest.fail("fit must not start")
    )
    with pytest.raises(gm.GMMMonitorError, match=match):
        gm._fit_gmm_monitor(**paths)
    assert not paths["output_dir"].exists()


def test_underflow_is_not_ignored_during_scaler_arithmetic(monkeypatch):
    gm = module()

    def invalid_scaler(*_):
        return np.multiply(np.full((32, 26), 1e-300), 1e-300)

    monkeypatch.setattr(gm.StandardScaler, "fit_transform", invalid_scaler)
    with pytest.raises(gm.GMMMonitorError, match="underflow"):
        gm.fit_training_mixture(np.ones((32, 26)))
