import inspect
import json
from dataclasses import FrozenInstanceError, replace
from hashlib import sha256
from importlib import import_module
from pathlib import Path

import pytest
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from test_length_inference import _artifact as length_artifact
from test_transformer_inference import _build_fixture

from automated_phishing_detection import (
    baselines,
    fixed_cascade,
    gmm_monitor,
    length_inference,
    transformer_inference,
    transformer_pipeline,
)

ROOT = Path(__file__).resolve().parents[1]
PUBLIC_PATHS = {
    "baseline": "reports/rq1-baseline-v2-summary.json",
    "transformer": "reports/rq1-transformer-cascade-v2-summary.json",
    "gmm": "reports/rq2-gmm-development-v1-summary.json",
}


def encode(value):
    return transformer_pipeline._canonical_json_bytes(value)


@pytest.fixture
def module():
    return import_module("automated_phishing_detection.bound_models")


@pytest.fixture
def fixture(module, tmp_path, monkeypatch):
    private = tmp_path / "private"
    private.mkdir()
    bundle = _build_fixture(private)
    summaries = {
        name: json.loads((ROOT / path).read_bytes())
        for name, path in PUBLIC_PATHS.items()
    }
    summaries["transformer"] = json.loads(bundle.summary_path.read_bytes())
    stage1 = json.loads(bundle.stage1_path.read_bytes())
    length = length_artifact()
    length["contract_sha256"] = stage1["contract_sha256"]
    length["input_hashes"] = stage1["input_hashes"]
    length["validation_threshold"] = stage1["validation_threshold"]
    length_path = private / "length-only.json"
    length_path.write_bytes(encode(length))
    baseline = summaries["baseline"]
    baseline["input_hashes"] = stage1["input_hashes"]
    baseline["input_counts"] = {
        partition: {
            key: value for key, value in counts.items() if key != "domain_count"
        }
        for partition, counts in summaries["transformer"]["input_counts"].items()
    }
    for name, artifact, content in (
        ("length-only", length, length_path.read_bytes()),
        ("Logistic-L1", stage1, bundle.stage1_path.read_bytes()),
    ):
        baseline["models"][name].update(
            artifact_sha256=sha256(content).hexdigest(),
            validation_threshold=artifact["validation_threshold"],
            validation_scoring_audit=artifact["validation_scoring_audit"],
            n_iter=artifact["classifier"]["n_iter"],
        )
    gmm_summary = summaries["gmm"]
    gmm_summary["input_hashes"] = {
        key: value
        for key, value in summaries["transformer"]["input_hashes"].items()
        if key != "transformer_contract"
    } | {"gmm_contract": gmm_summary["contract"]["sha256"]}
    gmm_summary["selected_component_count"] = 1
    artifact = {
        "schema_version": 1,
        "contract_id": gmm_monitor.CONTRACT_ID,
        "features": list(gmm_monitor.GMM_FEATURE_NAMES),
        "dtype": "float64",
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
        "input_hashes": gmm_summary["input_hashes"],
    }
    gmm_path = private / "gmm.json"
    gmm_path.write_bytes(encode(artifact))
    gmm_summary["artifact_hashes"]["gmm.json"] = sha256(
        gmm_path.read_bytes()
    ).hexdigest()
    paths = module.ArtifactPaths(
        length_path, bundle.stage1_path, bundle.bundle_dir, gmm_path
    )
    root = tmp_path / "public"
    (root / "reports").mkdir(parents=True)
    policy = {}
    for name, summary in summaries.items():
        content = encode(summary)
        (root / PUBLIC_PATHS[name]).write_bytes(content)
        policy[name] = (PUBLIC_PATHS[name], sha256(content).hexdigest())
    monkeypatch.setattr(module, "_PUBLIC_SUMMARIES", policy)
    monkeypatch.setattr(
        module,
        "_ARTIFACT_DIGESTS",
        {
            "length-only.json": baseline["models"]["length-only"]["artifact_sha256"],
            "logistic-l1.json": baseline["models"]["Logistic-L1"]["artifact_sha256"],
            "gmm.json": gmm_summary["artifact_hashes"]["gmm.json"],
        },
    )
    calls = []

    def load_length(path):
        calls.append("length")
        return length_inference._load_length_only_artifact_bytes(
            fixed_cascade._read_regular_file(path),
            expected_sha256=baseline["models"]["length-only"]["artifact_sha256"],
            expected_contract_sha256=stage1["contract_sha256"],
        )

    def load_transformer(
        bundle_dir,
        summary_path,
        logistic_path,
        *,
        expected_public_summary_sha256,
        device,
    ):
        calls.append("transformer")
        assert device == torch.device("mps")
        assert expected_public_summary_sha256 == policy["transformer"][1]
        return transformer_inference._load_transformer_cascade_bundle(
            bundle_dir,
            summary_path,
            logistic_path,
            _hash_policy=transformer_inference._BundleHashPolicy(
                **(
                    bundle.policy
                    | {"public_summary_sha256": expected_public_summary_sha256}
                )
            ),
            _device=torch.device("cpu"),
            _fixture_cpu=True,
        )

    monkeypatch.setattr(length_inference, "load_length_only_artifact", load_length)
    monkeypatch.setattr(
        transformer_inference, "load_transformer_cascade_bundle", load_transformer
    )
    return root, paths, summaries, calls


def rewrite_public(module, fixture, name, mutate):
    root, _, summaries, _ = fixture
    mutate(summaries[name])
    content = encode(summaries[name])
    path, _ = module._PUBLIC_SUMMARIES[name]
    (root / path).write_bytes(content)
    module._PUBLIC_SUMMARIES[name] = (path, sha256(content).hexdigest())


def test_public_api_has_fixed_hashes_and_no_device_or_hash_overrides(module):
    assert list(inspect.signature(module.load_bound_models).parameters) == [
        "root",
        "paths",
    ]
    assert module._PUBLIC_SUMMARIES == {
        "baseline": (
            PUBLIC_PATHS["baseline"],
            "bf5b3a6f0fc705d26852da4dd0053c6111ffc3e500d7a2e95dfba5ad859b279c",
        ),
        "transformer": (
            PUBLIC_PATHS["transformer"],
            "41499aa388babe60442de7231b4087f67a53f96f340568a7cc58a3268606a2fd",
        ),
        "gmm": (
            PUBLIC_PATHS["gmm"],
            "6f695138a302e854e1e5af590152e289486affe8ccdf75510ca9a5dcaad3b523",
        ),
    }
    assert (
        module._ARTIFACT_DIGESTS["gmm.json"]
        == "a01a8b143c2423df57a153462cc47e79822ad1c6768213dc69e03683d4009745"
    )


def test_fixed_public_summaries_validate_without_private_paths(module):
    module._validate_public(module._read_public(ROOT))


def test_bound_models_preserve_thresholds_all_provenance_and_do_not_fit(
    module, fixture, monkeypatch
):
    root, paths, summaries, calls = fixture

    def forbidden(*args, **kwargs):
        pytest.fail("binding must not fit, calibrate, score, or read an audit trace")

    for owner, name in (
        (StandardScaler, "fit"),
        (LogisticRegression, "fit"),
        (GaussianMixture, "fit"),
        (gmm_monitor, "fit_training_mixture"),
        (gmm_monitor, "calibrate_and_audit"),
        (baselines, "select_validation_threshold"),
    ):
        monkeypatch.setattr(owner, name, forbidden)
    original_read = fixed_cascade._read_regular_file
    read_paths = []

    def read(path):
        assert "audit" not in path.name
        read_paths.append(path)
        return original_read(path)

    monkeypatch.setattr(fixed_cascade, "_read_regular_file", read)
    result = module.load_bound_models(root, paths)
    assert isinstance(result.length_only, length_inference.LoadedLengthOnly)
    assert isinstance(result.cascade, transformer_inference.LoadedTransformerCascade)
    assert (
        result.monitor_boundary == summaries["gmm"]["threshold"] == -67.45792380813624
    )
    assert result.gmm["input_hashes"] == summaries["gmm"]["input_hashes"]
    assert result.gmm["mixture"]["components"] == 1
    assert (
        result.length_only.validation_threshold_record
        == summaries["baseline"]["models"]["length-only"]["validation_threshold"]
    )
    assert (
        result.cascade.stage1_model.validation_threshold_record
        == summaries["baseline"]["models"]["Logistic-L1"]["validation_threshold"]
    )
    assert result.artifact_hashes == tuple(
        sorted(
            (
                module._ARTIFACT_DIGESTS | summaries["transformer"]["artifact_hashes"]
            ).items()
        )
    )
    assert set(calls) == {"length", "transformer"}
    assert set(read_paths[:3]) == {root / path for path in PUBLIC_PATHS.values()}
    with pytest.raises(FrozenInstanceError):
        result.monitor_boundary = 0.0
    with pytest.raises(FrozenInstanceError):
        paths.gmm = root


def test_binding_retains_gmm_bytes_from_its_single_read(module, fixture, monkeypatch):
    root, paths, _, _ = fixture
    original_read = fixed_cascade._read_regular_file
    captured = []

    def read(path):
        content = original_read(path)
        if path == paths.gmm:
            captured.append(content)
        return content

    monkeypatch.setattr(fixed_cascade, "_read_regular_file", read)
    result = module.load_bound_models(root, paths)

    assert len(captured) == 1
    assert result.gmm_artifact_bytes is captured[0]
    assert "gmm_artifact_bytes=" not in repr(result)


def test_binding_derives_audit_counts_from_authenticated_summary(module, fixture):
    root, paths, _, _ = fixture

    def update_audit(summary):
        summary.update(
            audit_alert_count=2,
            audit_window_count=20,
            audit_alert_fraction=0.1,
            false_alert_gate_met=False,
        )
        summary["input_counts"]["audit"]["complete_windows"] = 20

    rewrite_public(module, fixture, "gmm", update_audit)
    result = module.load_bound_models(root, paths)

    assert result.audit_alert_count == 2
    assert result.audit_window_count == 20


@pytest.mark.parametrize("name", PUBLIC_PATHS)
def test_public_hash_failure_precedes_every_private_read(
    module, fixture, monkeypatch, name
):
    root, paths, _, calls = fixture
    (root / PUBLIC_PATHS[name]).write_bytes(b"{}")
    original_read = fixed_cascade._read_regular_file

    def read(path):
        assert path.is_relative_to(root), "private read preceded public authentication"
        return original_read(path)

    monkeypatch.setattr(fixed_cascade, "_read_regular_file", read)
    with pytest.raises(module.BoundModelsError, match="SHA-256"):
        module.load_bound_models(root, paths)
    assert calls == []


@pytest.mark.parametrize(
    "name,mutate",
    [
        ("baseline", lambda value: value.update(schema_version=True)),
        (
            "baseline",
            lambda value: value["models"]["length-only"].update(
                artifact_sha256="0" * 64
            ),
        ),
        (
            "baseline",
            lambda value: value["models"]["Logistic-L1"].update(
                validation_threshold={}
            ),
        ),
        ("transformer", lambda value: value["input_hashes"].update(train="0" * 64)),
        ("transformer", lambda value: value["cascade"].update(accepted_cascade=False)),
        ("gmm", lambda value: value.update(threshold=True)),
        ("gmm", lambda value: value.update(selected_component_count=True)),
        ("gmm", lambda value: value.update(audit_alert_count=True)),
        ("gmm", lambda value: value.update(audit_alert_count=-1)),
        ("gmm", lambda value: value.update(audit_alert_count=253)),
        ("gmm", lambda value: value.update(audit_window_count=True)),
        ("gmm", lambda value: value.update(audit_window_count=0)),
        ("gmm", lambda value: value.update(audit_alert_fraction=0.0)),
        ("gmm", lambda value: value.update(false_alert_gate_met=True)),
        (
            "gmm",
            lambda value: value["input_counts"]["audit"].update(complete_windows=251),
        ),
        (
            "gmm",
            lambda value: value["input_hashes"].update(logistic_l1_artifact="0" * 64),
        ),
        (
            "gmm",
            lambda value: value["artifact_hashes"].update(**{"gmm.json": "0" * 64}),
        ),
        ("gmm", lambda value: value.update(extra={})),
        ("baseline", lambda value: value.update(pipeline=[])),
        (
            "baseline",
            lambda value: value["models"]["length-only"].update(n_iter=[True]),
        ),
        ("transformer", lambda value: value["transformer"].update(threshold={})),
        ("transformer", lambda value: value["configuration"].update(extra=True)),
        ("transformer", lambda value: value["contract"].update(extra=True)),
        ("transformer", lambda value: value["cascade"].update(extra=True)),
        ("gmm", lambda value: value["contract"].update(extra=True)),
    ],
)
def test_malformed_or_cross_bound_public_summary_fails_before_private_reads(
    module, fixture, monkeypatch, name, mutate
):
    root, paths, _, calls = fixture
    rewrite_public(module, fixture, name, mutate)
    original_read = fixed_cascade._read_regular_file

    def read(path):
        assert path.is_relative_to(root), "private read preceded public validation"
        return original_read(path)

    monkeypatch.setattr(fixed_cascade, "_read_regular_file", read)
    with pytest.raises(module.BoundModelsError):
        module.load_bound_models(root, paths)
    assert calls == []


@pytest.mark.parametrize(
    "content", [b"\xff", b"[]", b'{"a":1,"a":2}', b'{"threshold":NaN}']
)
def test_authenticated_malformed_json_still_fails_before_private_read(
    module, fixture, content
):
    root, paths, _, calls = fixture
    path, _ = module._PUBLIC_SUMMARIES["gmm"]
    (root / path).write_bytes(content)
    module._PUBLIC_SUMMARIES["gmm"] = (path, sha256(content).hexdigest())
    with pytest.raises(module.BoundModelsError):
        module.load_bound_models(root, paths)
    assert calls == []


def test_gmm_hash_is_checked_before_decoder(module, fixture, monkeypatch):
    root, paths, _, _ = fixture
    paths.gmm.write_bytes(b"malformed private JSON")

    def forbidden(content):
        pytest.fail("unauthenticated GMM reached decoder")

    monkeypatch.setattr(gmm_monitor, "load_gmm_artifact_bytes", forbidden)
    with pytest.raises(module.BoundModelsError, match="GMM.*SHA-256"):
        module.load_bound_models(root, paths)


@pytest.mark.parametrize("field", ["input_hashes", "components"])
def test_authenticated_gmm_must_match_exact_public_provenance(module, fixture, field):
    root, paths, summaries, _ = fixture
    artifact = json.loads(paths.gmm.read_bytes())
    if field == "input_hashes":
        artifact["input_hashes"]["extra"] = "f" * 64
    else:
        summaries["gmm"]["selected_component_count"] = 2
    content = encode(artifact)
    paths.gmm.write_bytes(content)
    digest = sha256(content).hexdigest()
    module._ARTIFACT_DIGESTS["gmm.json"] = digest
    rewrite_public(
        module,
        fixture,
        "gmm",
        lambda value: value["artifact_hashes"].update(**{"gmm.json": digest}),
    )
    with pytest.raises(module.BoundModelsError, match="GMM.*(input|component)"):
        module.load_bound_models(root, paths)


@pytest.mark.parametrize(
    "field", ["length_only", "logistic_l1", "transformer_bundle", "gmm"]
)
def test_private_symlinks_are_refused_by_existing_regular_read_helpers(
    module, fixture, field
):
    root, paths, _, _ = fixture
    target = getattr(paths, field)
    alias = target.parent / f"alias-{field}"
    alias.symlink_to(target, target_is_directory=target.is_dir())
    with pytest.raises(module.BoundModelsError):
        module.load_bound_models(root, replace(paths, **{field: alias}))


@pytest.mark.parametrize(
    "which", ["length", "stage1", "transformer_hashes", "transformer_summary"]
)
def test_loaded_model_provenance_cannot_escape_authenticated_public_summary(
    module, fixture, monkeypatch, which
):
    root, paths, _, _ = fixture
    if which == "length":
        original = length_inference.load_length_only_artifact
        monkeypatch.setattr(
            length_inference,
            "load_length_only_artifact",
            lambda path: replace(original(path), artifact_sha256="0" * 64),
        )
    else:
        original = transformer_inference.load_transformer_cascade_bundle

        def load(*args, **kwargs):
            model = original(*args, **kwargs)
            if which == "stage1":
                return replace(model, stage1_threshold=0.125)
            if which == "transformer_summary":
                return replace(model, public_summary_sha256="0" * 64)
            return replace(model, artifact_hashes=())

        monkeypatch.setattr(
            transformer_inference, "load_transformer_cascade_bundle", load
        )
    with pytest.raises(module.BoundModelsError):
        module.load_bound_models(root, paths)


@pytest.mark.parametrize("name", PUBLIC_PATHS)
def test_public_symlink_refused_before_any_private_loader(module, fixture, name):
    root, paths, _, calls = fixture
    public_path = root / PUBLIC_PATHS[name]
    target = public_path.with_suffix(".target")
    public_path.rename(target)
    public_path.symlink_to(target)
    with pytest.raises(module.BoundModelsError, match="regular"):
        module.load_bound_models(root, paths)
    assert calls == []


@pytest.mark.parametrize("role", ["length_only", "logistic_l1", "transformer_bundle"])
def test_corrupted_private_artifacts_rejected_by_existing_loaders(
    module, fixture, role
):
    root, paths, _, _ = fixture
    path = getattr(paths, role)
    if role == "transformer_bundle":
        path /= "transformer-weights.npz"
    path.write_bytes(b"corrupted synthetic artifact")
    with pytest.raises(module.BoundModelsError):
        module.load_bound_models(root, paths)


def test_noncanonical_transformer_summary_fails_before_private_read(
    module, fixture, monkeypatch
):
    root, paths, summaries, calls = fixture
    path, _ = module._PUBLIC_SUMMARIES["transformer"]
    content = json.dumps(summaries["transformer"], indent=2, sort_keys=True).encode()
    (root / path).write_bytes(content)
    module._PUBLIC_SUMMARIES["transformer"] = path, sha256(content).hexdigest()
    original_read = fixed_cascade._read_regular_file

    def read(path):
        assert path.is_relative_to(root), (
            "noncanonical public bytes reached private reads"
        )
        return original_read(path)

    monkeypatch.setattr(fixed_cascade, "_read_regular_file", read)
    with pytest.raises(module.BoundModelsError, match="canonical"):
        module.load_bound_models(root, paths)
    assert calls == []
