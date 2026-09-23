import json
import os
import shutil
import zipfile
from dataclasses import dataclass, replace
from hashlib import sha256
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
import torch

from automated_phishing_detection import (
    baselines,
    character_sequence,
    character_transformer,
    transformer_inference,
    transformer_pipeline,
)
from automated_phishing_detection.url_features import FEATURE_NAMES

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = json.loads(
    (ROOT / "data" / "rq1-transformer-cascade-contract-v2.json").read_text()
)


@dataclass(frozen=True)
class FixtureBundle:
    bundle_dir: Path
    summary_path: Path
    stage1_path: Path
    policy: object
    source_model: character_transformer.CharacterTransformer


def _canonical_json(value):
    return transformer_pipeline._canonical_json_bytes(value)


def _threshold_record(threshold=0.5):
    negative = 400
    return {
        "status": "selected",
        "threshold": threshold,
        "candidate_count": 3,
        "counts": {
            "true_positive": 20,
            "false_positive": 0,
            "true_negative": negative,
            "false_negative": 0,
            "positive": 20,
            "negative": negative,
        },
        "recall": 1.0,
        "observed_fpr": 0.0,
        "fpr_upper_95": baselines.clopper_pearson_upper(0, negative),
    }


def _stage1_artifact(contract_sha256):
    width = len(FEATURE_NAMES)
    coefficients = [0.0] * width
    coefficients[0] = 0.01
    return {
        "access": {"group_test_accessed": False, "phishvn_accessed": False},
        "analysis_stage": "development_validation_only",
        "artifact_type": "rq1-baseline-model",
        "classes": [0, 1],
        "classifier": {
            "coefficients": [coefficients],
            "config": {
                "C": 1.0,
                "class": "LogisticRegression",
                "class_weight": "balanced",
                "fit_intercept": True,
                "max_iter": 5000,
                "penalty": "l1",
                "random_state": 42,
                "solver": "saga",
                "tol": 0.0001,
            },
            "intercept": [0.0],
            "n_iter": [12],
        },
        "contract_id": "rq1-baselines-v2",
        "contract_sha256": contract_sha256,
        "features": list(FEATURE_NAMES),
        "input_hashes": {
            "contract": contract_sha256,
            "preparation_summary": "3" * 64,
            "train": "1" * 64,
            "validation": "2" * 64,
        },
        "model_name": "Logistic-L1",
        "scaler": {
            "config": {
                "class": "StandardScaler",
                "fit_partition": "train",
                "with_mean": True,
                "with_std": True,
            },
            "mean": [0.0] * width,
            "n_samples_seen": 20,
            "scale": [1.0] * width,
            "variance": [1.0] * width,
        },
        "schema_version": 2,
        "software_versions": baselines._software_versions(),
        "validation_scoring_audit": {
            "max_absolute_decision_difference": 0.0,
            "max_absolute_probability_difference": 0.0,
            "platform_identity": baselines._platform_identity(),
            "warning_records": [],
        },
        "validation_threshold": _threshold_record(),
    }


def _partition(prefix):
    negative = 400
    positive = 20
    rows = negative + positive
    urls = tuple(f"https://{prefix}-{index}.example/path" for index in range(rows))
    return transformer_pipeline._Partition(
        raw_urls=urls,
        labels=np.asarray([0] * negative + [1] * positive, dtype=np.int8),
        domains=frozenset(f"{prefix}-{index}.example" for index in range(rows)),
        ordinals=frozenset(range(rows)),
        class_counts={"0": negative, "1": positive},
    )


def _cascade_record():
    return {
        "schema_version": 1,
        "status": "selected",
        "accepted_cascade": True,
        "reason": "constraints_met",
        "threshold_statuses": {"stage1": "selected", "transformer": "selected"},
        "candidate_count": 4,
        "half_width": 0.1,
        "transformer_invocations": 40,
        "transformer_invocation_rate": 40 / 420,
        "counts": {
            "true_positive": 20,
            "false_positive": 0,
            "true_negative": 400,
            "false_negative": 0,
            "positive": 20,
            "negative": 400,
        },
        "recall": 1.0,
        "observed_fpr": 0.0,
        "fpr_upper_95": baselines.clopper_pearson_upper(0, 400),
        "minimum_recall": 0.98,
        "maximum_fpr_upper_95": 0.01,
    }


def _build_fixture(tmp_path):
    bundle_dir = tmp_path / "transformer-bundle"
    bundle_dir.mkdir(mode=0o700)
    summary_path = tmp_path / "summary.json"
    stage1_path = tmp_path / "logistic-l1.json"

    baseline_contract_sha256 = "b" * 64
    transformer_contract_sha256 = "c" * 64
    stage1_bytes = (
        json.dumps(_stage1_artifact(baseline_contract_sha256), indent=2, sort_keys=True)
        + "\n"
    ).encode("ascii")
    stage1_path.write_bytes(stage1_bytes)
    stage1_sha256 = sha256(stage1_bytes).hexdigest()

    vocabulary = character_sequence.CharacterVocabulary(
        tuple(sorted(set("https://safe-phish.example/path0123456789"), key=ord))
    )
    torch.manual_seed(20260917)
    model = character_transformer.CharacterTransformer(vocabulary.size)
    fit = character_transformer.TransformerFit(
        model=model,
        history=(character_transformer.EpochRecord(1, 0.5, 0.75),),
        best_epoch=1,
        best_validation_average_precision=0.75,
        epochs_completed=1,
        stopped_early=False,
        positive_class_weight=1.0,
        validation_probabilities=(0.1, 0.2, 0.8, 0.9),
    )
    weights_bytes, _ = transformer_pipeline._serialize_state_dict_npz(model)
    vocabulary_bytes = vocabulary.to_json().encode("utf-8")
    input_hashes = {
        "train": "1" * 64,
        "validation": "2" * 64,
        "preparation_summary": "3" * 64,
        "baseline_contract": baseline_contract_sha256,
        "logistic_l1_artifact": stage1_sha256,
        "transformer_contract": transformer_contract_sha256,
    }
    transformer = transformer_pipeline._transformer_metadata(
        fit=fit,
        threshold=_threshold_record(),
        input_hashes=input_hashes,
        contract=CONTRACT,
        vocabulary_sha256=sha256(vocabulary_bytes).hexdigest(),
        weights_sha256=sha256(weights_bytes).hexdigest(),
        training_device=torch.device("cpu"),
    )
    transformer_bytes = _canonical_json(transformer)
    cascade = _cascade_record()
    cascade_metadata = transformer_pipeline._cascade_metadata(
        calibration=cascade,
        input_hashes=input_hashes,
        transformer_sha256=sha256(transformer_bytes).hexdigest(),
        stage1_warnings=[],
    )
    contents = {
        "cascade.json": _canonical_json(cascade_metadata),
        "transformer-weights.npz": weights_bytes,
        "transformer.json": transformer_bytes,
        "vocabulary.json": vocabulary_bytes,
    }
    artifact_hashes = {
        name: sha256(content).hexdigest() for name, content in contents.items()
    }
    contents["SHA256SUMS"] = "".join(
        f"{artifact_hashes[name]}  {name}\n" for name in sorted(artifact_hashes)
    ).encode("ascii")
    for name, content in contents.items():
        path = bundle_dir / name
        path.write_bytes(content)
        path.chmod(0o600)

    summary = transformer_pipeline._public_summary(
        train=_partition("train"),
        validation=_partition("validation"),
        vocabulary=vocabulary,
        transformer_fit=fit,
        transformer_threshold=_threshold_record(),
        cascade=cascade,
        contract=CONTRACT,
        input_hashes=input_hashes,
        artifact_hashes=artifact_hashes,
        stage1_warnings=[],
    )
    summary_bytes = _canonical_json(summary)
    summary_path.write_bytes(summary_bytes)
    policy = {
        "public_summary_sha256": sha256(summary_bytes).hexdigest(),
        "train_sha256": "1" * 64,
        "validation_sha256": "2" * 64,
        "preparation_summary_sha256": "3" * 64,
        "transformer_contract_sha256": transformer_contract_sha256,
        "baseline_contract_sha256": baseline_contract_sha256,
        "logistic_l1_artifact_sha256": stage1_sha256,
    }
    return FixtureBundle(bundle_dir, summary_path, stage1_path, policy, model)


def _copy_fixture(source, destination):
    shutil.copytree(source.bundle_dir, destination / "transformer-bundle")
    shutil.copy2(source.summary_path, destination / "summary.json")
    shutil.copy2(source.stage1_path, destination / "logistic-l1.json")
    return FixtureBundle(
        destination / "transformer-bundle",
        destination / "summary.json",
        destination / "logistic-l1.json",
        source.policy,
        source.source_model,
    )


def _read_json(path):
    return json.loads(path.read_text())


def _rewrite_bound_bundle(
    fixture,
    *,
    transformer=None,
    cascade=None,
    summary=None,
    weights_bytes=None,
):
    transformer = (
        _read_json(fixture.bundle_dir / "transformer.json")
        if transformer is None
        else transformer
    )
    cascade = (
        _read_json(fixture.bundle_dir / "cascade.json") if cascade is None else cascade
    )
    summary = _read_json(fixture.summary_path) if summary is None else summary
    if weights_bytes is not None:
        (fixture.bundle_dir / "transformer-weights.npz").write_bytes(weights_bytes)
        transformer["weights_sha256"] = sha256(weights_bytes).hexdigest()

    transformer_bytes = _canonical_json(transformer)
    (fixture.bundle_dir / "transformer.json").write_bytes(transformer_bytes)
    return _refresh_bindings(fixture, cascade=cascade, summary=summary)


def _refresh_bindings(fixture, *, cascade=None, summary=None):
    transformer_bytes = (fixture.bundle_dir / "transformer.json").read_bytes()
    cascade = (
        _read_json(fixture.bundle_dir / "cascade.json") if cascade is None else cascade
    )
    summary = _read_json(fixture.summary_path) if summary is None else summary
    cascade["transformer_metadata_sha256"] = sha256(transformer_bytes).hexdigest()
    (fixture.bundle_dir / "cascade.json").write_bytes(_canonical_json(cascade))

    artifact_hashes = {
        name: sha256((fixture.bundle_dir / name).read_bytes()).hexdigest()
        for name in (
            "cascade.json",
            "transformer-weights.npz",
            "transformer.json",
            "vocabulary.json",
        )
    }
    manifest = "".join(
        f"{artifact_hashes[name]}  {name}\n" for name in sorted(artifact_hashes)
    ).encode("ascii")
    (fixture.bundle_dir / "SHA256SUMS").write_bytes(manifest)
    summary["artifact_hashes"] = artifact_hashes
    summary_bytes = _canonical_json(summary)
    fixture.summary_path.write_bytes(summary_bytes)
    policy = dict(fixture.policy)
    policy["public_summary_sha256"] = sha256(summary_bytes).hexdigest()
    return replace(fixture, policy=policy)


def _rewrite_cascade_calibration(fixture, change):
    cascade = _read_json(fixture.bundle_dir / "cascade.json")
    summary = _read_json(fixture.summary_path)
    change(cascade["calibration"])
    summary["cascade"] = transformer_pipeline._public_cascade_record(
        cascade["calibration"]
    )
    return _rewrite_bound_bundle(fixture, cascade=cascade, summary=summary)


def _weights_with_changed_first_array(content, change, *, allow_pickle=False):
    source = zipfile.ZipFile(BytesIO(content))
    members = [(info, source.read(info)) for info in source.infolist()]
    source.close()
    first_info, first_content = members[0]
    array = np.load(BytesIO(first_content), allow_pickle=False)
    changed = BytesIO()
    np.lib.format.write_array(
        changed, change(array), version=(1, 0), allow_pickle=allow_pickle
    )
    members[0] = (first_info, changed.getvalue())

    destination = BytesIO()
    with zipfile.ZipFile(
        destination, mode="w", compression=zipfile.ZIP_STORED
    ) as archive:
        for info, member in members:
            archive.writestr(info, member)
    return destination.getvalue()


def _weights_with_changed_member_set(content, change):
    source = zipfile.ZipFile(BytesIO(content))
    members = [(info, source.read(info)) for info in source.infolist()]
    source.close()
    if change == "missing":
        members.pop()
    else:
        extra_info = zipfile.ZipInfo(
            filename="unexpected.npy", date_time=(1980, 1, 1, 0, 0, 0)
        )
        extra_info.compress_type = zipfile.ZIP_STORED
        extra_info.create_system = 3
        extra_info.external_attr = 0o600 << 16
        members.append((extra_info, members[0][1]))

    destination = BytesIO()
    with zipfile.ZipFile(
        destination, mode="w", compression=zipfile.ZIP_STORED
    ) as archive:
        for info, member in members:
            archive.writestr(info, member)
    return destination.getvalue()


def _weights_with_changed_named_array(content, member_name, change):
    source = zipfile.ZipFile(BytesIO(content))
    members = [(info, source.read(info)) for info in source.infolist()]
    source.close()
    changed_members = []
    found = False
    for info, member in members:
        if info.filename == member_name:
            array = np.load(BytesIO(member), allow_pickle=False)
            changed = BytesIO()
            np.lib.format.write_array(
                changed, change(array), version=(1, 0), allow_pickle=False
            )
            member = changed.getvalue()
            found = True
        changed_members.append((info, member))
    assert found

    destination = BytesIO()
    with zipfile.ZipFile(
        destination, mode="w", compression=zipfile.ZIP_STORED
    ) as archive:
        for info, member in changed_members:
            archive.writestr(info, member)
    return destination.getvalue()


@pytest.fixture
def fixture_bundle(tmp_path):
    return _build_fixture(tmp_path)


def _load_fixture(fixture):
    assert hasattr(transformer_inference, "_BundleHashPolicy")
    policy = transformer_inference._BundleHashPolicy(**fixture.policy)
    return transformer_inference._load_transformer_cascade_bundle(
        fixture.bundle_dir,
        fixture.summary_path,
        fixture.stage1_path,
        _hash_policy=policy,
        _device=torch.device("cpu"),
        _fixture_cpu=True,
    )


def test_loads_complete_hash_bound_bundle_without_fitting(fixture_bundle):
    loaded = _load_fixture(fixture_bundle)

    assert loaded.device == torch.device("cpu")
    assert loaded.stage1_threshold == 0.5
    assert loaded.transformer_threshold == 0.5
    assert loaded.half_width == 0.1
    assert loaded._model.training is False
    assert all(not parameter.requires_grad for parameter in loaded._model.parameters())


def test_rejects_public_summary_hash_mismatch(fixture_bundle):
    fixture_bundle.summary_path.write_bytes(b"{}\n")

    with pytest.raises(
        transformer_inference.TransformerInferenceError,
        match="public summary SHA-256 mismatch",
    ):
        _load_fixture(fixture_bundle)


@pytest.mark.parametrize("change", ["missing", "extra"])
def test_rejects_bundle_file_set_changes(fixture_bundle, change):
    if change == "missing":
        (fixture_bundle.bundle_dir / "vocabulary.json").unlink()
    else:
        (fixture_bundle.bundle_dir / "notes.txt").write_text("not part of the bundle")

    with pytest.raises(
        transformer_inference.TransformerInferenceError,
        match="bundle files do not match the frozen schema",
    ):
        _load_fixture(fixture_bundle)


def test_rejects_symlinked_bundle_member(fixture_bundle):
    vocabulary_path = fixture_bundle.bundle_dir / "vocabulary.json"
    target = fixture_bundle.bundle_dir.parent / "vocabulary-target.json"
    vocabulary_path.replace(target)
    vocabulary_path.symlink_to(target)

    with pytest.raises(
        transformer_inference.TransformerInferenceError,
        match="regular file, not an alias",
    ):
        _load_fixture(fixture_bundle)


def test_rejects_hard_linked_bundle_members(fixture_bundle):
    cascade_path = fixture_bundle.bundle_dir / "cascade.json"
    cascade_path.unlink()
    os.link(fixture_bundle.bundle_dir / "transformer.json", cascade_path)

    with pytest.raises(
        transformer_inference.TransformerInferenceError,
        match="must not alias one another",
    ):
        _load_fixture(fixture_bundle)


def test_rejects_corrupt_bundle_member(fixture_bundle):
    weights_path = fixture_bundle.bundle_dir / "transformer-weights.npz"
    weights_path.write_bytes(weights_path.read_bytes() + b"corrupt")

    with pytest.raises(
        transformer_inference.TransformerInferenceError,
        match="SHA256SUMS does not match the bundle",
    ):
        _load_fixture(fixture_bundle)


def test_rejects_noncanonical_private_metadata_even_when_rebound(fixture_bundle):
    transformer_path = fixture_bundle.bundle_dir / "transformer.json"
    transformer = _read_json(transformer_path)
    noncanonical = (json.dumps(transformer, indent=2, sort_keys=True) + "\n").encode()
    transformer_path.write_bytes(noncanonical)
    fixture_bundle = _refresh_bindings(fixture_bundle)

    with pytest.raises(
        transformer_inference.TransformerInferenceError,
        match="transformer.json is not canonical JSON",
    ):
        _load_fixture(fixture_bundle)


def test_rejects_public_private_fit_projection_difference(fixture_bundle):
    summary = _read_json(fixture_bundle.summary_path)
    summary["transformer"]["fit"]["best_epoch"] = 2
    fixture_bundle = _rewrite_bound_bundle(fixture_bundle, summary=summary)

    with pytest.raises(
        transformer_inference.TransformerInferenceError,
        match="transformer fit projection differs",
    ):
        _load_fixture(fixture_bundle)


def test_rejects_malformed_nested_public_summary(fixture_bundle):
    summary = _read_json(fixture_bundle.summary_path)
    summary["transformer"] = None
    fixture_bundle = _rewrite_bound_bundle(fixture_bundle, summary=summary)

    with pytest.raises(
        transformer_inference.TransformerInferenceError,
        match="public transformer fields do not match the frozen schema",
    ):
        _load_fixture(fixture_bundle)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema_version", 2),
        ("hypothesis_status", {"H1": "accepted", "H2": "undecided", "H3": "undecided"}),
    ],
)
def test_rejects_public_summary_status_drift(fixture_bundle, field, value):
    summary = _read_json(fixture_bundle.summary_path)
    summary[field] = value
    fixture_bundle = _rewrite_bound_bundle(fixture_bundle, summary=summary)

    with pytest.raises(
        transformer_inference.TransformerInferenceError,
        match="public summary identity is invalid",
    ):
        _load_fixture(fixture_bundle)


def test_rejects_validation_count_that_differs_from_calibration(fixture_bundle):
    summary = _read_json(fixture_bundle.summary_path)
    summary["input_counts"]["validation"] = {
        "0": 400,
        "1": 19,
        "domain_count": 419,
        "rows": 419,
    }
    fixture_bundle = _rewrite_bound_bundle(fixture_bundle, summary=summary)

    with pytest.raises(
        transformer_inference.TransformerInferenceError,
        match="counts differ from public validation counts",
    ):
        _load_fixture(fixture_bundle)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("candidate_count", 421, "cascade candidate_count is invalid"),
        ("transformer_invocations", 0, "cascade transformer_invocations"),
        ("transformer_invocation_rate", 0.5, "cascade invocation rate is inconsistent"),
        ("recall", 0.9, "cascade calibration.recall is inconsistent with counts"),
        (
            "observed_fpr",
            0.1,
            "cascade calibration.observed_fpr is inconsistent with counts",
        ),
        (
            "fpr_upper_95",
            0.001,
            "cascade calibration.fpr_upper_95 is inconsistent with counts",
        ),
        ("minimum_recall", 0.5, "cascade minimum recall is inconsistent"),
        ("maximum_fpr_upper_95", 0.02, "cascade maximum FPR gate is invalid"),
    ],
)
def test_rejects_inconsistent_selected_cascade(fixture_bundle, field, value, message):
    fixture_bundle = _rewrite_cascade_calibration(
        fixture_bundle,
        lambda calibration: calibration.__setitem__(field, value),
    )

    with pytest.raises(
        transformer_inference.TransformerInferenceError,
        match=message,
    ):
        _load_fixture(fixture_bundle)


def test_rejects_selected_cascade_count_algebra_drift(fixture_bundle):
    def change(calibration):
        calibration["counts"]["true_positive"] = 19

    fixture_bundle = _rewrite_cascade_calibration(fixture_bundle, change)

    with pytest.raises(
        transformer_inference.TransformerInferenceError,
        match="cascade calibration.positive counts are inconsistent",
    ):
        _load_fixture(fixture_bundle)


def test_rejects_selected_cascade_above_clopper_pearson_gate(fixture_bundle):
    def change(calibration):
        calibration["counts"]["false_positive"] = 1
        calibration["counts"]["true_negative"] = 399
        calibration["observed_fpr"] = 1 / 400
        calibration["fpr_upper_95"] = baselines.clopper_pearson_upper(1, 400)

    fixture_bundle = _rewrite_cascade_calibration(fixture_bundle, change)

    with pytest.raises(
        transformer_inference.TransformerInferenceError,
        match="cascade calibration does not satisfy the frozen FPR gate",
    ):
        _load_fixture(fixture_bundle)


def test_rejects_rebound_development_input_hash(fixture_bundle):
    transformer = _read_json(fixture_bundle.bundle_dir / "transformer.json")
    cascade = _read_json(fixture_bundle.bundle_dir / "cascade.json")
    summary = _read_json(fixture_bundle.summary_path)
    for value in (transformer, cascade, summary):
        value["input_hashes"]["train"] = "9" * 64
    fixture_bundle = _rewrite_bound_bundle(
        fixture_bundle,
        transformer=transformer,
        cascade=cascade,
        summary=summary,
    )

    with pytest.raises(
        transformer_inference.TransformerInferenceError,
        match="artifact input hashes violate the active policy",
    ):
        _load_fixture(fixture_bundle)


def test_rejects_fit_summary_that_does_not_match_private_history(fixture_bundle):
    transformer = _read_json(fixture_bundle.bundle_dir / "transformer.json")
    transformer["fit"]["history"][0]["validation_average_precision"] = 0.25
    fixture_bundle = _rewrite_bound_bundle(
        fixture_bundle,
        transformer=transformer,
    )

    with pytest.raises(
        transformer_inference.TransformerInferenceError,
        match="transformer fit summary does not match its history",
    ):
        _load_fixture(fixture_bundle)


def test_rejects_rebound_configuration_that_drifted_from_contract(fixture_bundle):
    transformer = _read_json(fixture_bundle.bundle_dir / "transformer.json")
    summary = _read_json(fixture_bundle.summary_path)
    transformer["configuration"]["architecture"]["encoder"]["layers"] = 5
    summary["configuration"]["architecture"]["encoder"]["layers"] = 5
    fixture_bundle = _rewrite_bound_bundle(
        fixture_bundle,
        transformer=transformer,
        summary=summary,
    )

    with pytest.raises(
        transformer_inference.TransformerInferenceError,
        match="configuration does not match the frozen contract",
    ):
        _load_fixture(fixture_bundle)


def test_rejects_rebound_weight_with_wrong_state_shape(fixture_bundle):
    weights_path = fixture_bundle.bundle_dir / "transformer-weights.npz"
    weights = _weights_with_changed_first_array(
        weights_path.read_bytes(), lambda array: array.reshape(-1)[:-1]
    )
    fixture_bundle = _rewrite_bound_bundle(fixture_bundle, weights_bytes=weights)

    with pytest.raises(
        transformer_inference.TransformerInferenceError,
        match="invalid shape, dtype, or values",
    ):
        _load_fixture(fixture_bundle)


def test_rejects_rebound_nonzero_padding_embedding(fixture_bundle):
    weights_path = fixture_bundle.bundle_dir / "transformer-weights.npz"

    def change_padding_row(array):
        changed = array.copy()
        changed[0, 0] = 1.0
        return changed

    weights = _weights_with_changed_named_array(
        weights_path.read_bytes(),
        "token_embedding.weight.npy",
        change_padding_row,
    )
    fixture_bundle = _rewrite_bound_bundle(fixture_bundle, weights_bytes=weights)

    with pytest.raises(
        transformer_inference.TransformerInferenceError,
        match="token embedding padding row must remain exactly zero",
    ):
        _load_fixture(fixture_bundle)


def test_translates_model_construction_failure(fixture_bundle, monkeypatch):
    class BrokenTransformer:
        def __init__(self, vocabulary_size):
            raise character_transformer.TransformerTrainingError("construction failed")

    monkeypatch.setattr(
        character_transformer, "CharacterTransformer", BrokenTransformer
    )

    with pytest.raises(
        transformer_inference.TransformerInferenceError,
        match="weights archive is invalid: construction failed",
    ):
        _load_fixture(fixture_bundle)


@pytest.mark.parametrize("change", ["missing", "extra"])
def test_rejects_rebound_weight_key_changes(fixture_bundle, change):
    weights_path = fixture_bundle.bundle_dir / "transformer-weights.npz"
    weights = _weights_with_changed_member_set(weights_path.read_bytes(), change)
    fixture_bundle = _rewrite_bound_bundle(fixture_bundle, weights_bytes=weights)

    with pytest.raises(
        transformer_inference.TransformerInferenceError,
        match="weights keys do not match the model",
    ):
        _load_fixture(fixture_bundle)


def test_rejects_rebound_empty_weight_member(fixture_bundle):
    weights_path = fixture_bundle.bundle_dir / "transformer-weights.npz"
    destination = BytesIO()
    with zipfile.ZipFile(BytesIO(weights_path.read_bytes())) as source:
        with zipfile.ZipFile(destination, "w") as changed:
            for index, info in enumerate(source.infolist()):
                changed.writestr(info, b"" if index == 0 else source.read(info))
    fixture_bundle = _rewrite_bound_bundle(
        fixture_bundle, weights_bytes=destination.getvalue()
    )

    with pytest.raises(
        transformer_inference.TransformerInferenceError,
        match="weights archive is invalid",
    ):
        _load_fixture(fixture_bundle)


@pytest.mark.parametrize(
    ("change", "allow_pickle", "message"),
    [
        (lambda array: array.astype("<f8"), False, "invalid shape, dtype, or values"),
        (
            lambda array: np.full(array.shape, np.nan, dtype="<f4"),
            False,
            "invalid shape, dtype, or values",
        ),
        (
            lambda array: np.full(array.shape, "pickled", dtype=object),
            True,
            "weights archive is invalid",
        ),
    ],
    ids=("float64", "nonfinite", "pickled-object"),
)
def test_rejects_rebound_weight_with_unsafe_array(
    fixture_bundle, change, allow_pickle, message
):
    weights_path = fixture_bundle.bundle_dir / "transformer-weights.npz"
    weights = _weights_with_changed_first_array(
        weights_path.read_bytes(), change, allow_pickle=allow_pickle
    )
    fixture_bundle = _rewrite_bound_bundle(fixture_bundle, weights_bytes=weights)

    with pytest.raises(
        transformer_inference.TransformerInferenceError,
        match=message,
    ):
        _load_fixture(fixture_bundle)


def test_preserves_synthetic_predictions_without_fitting(fixture_bundle, monkeypatch):
    def fail_if_called(*args, **kwargs):
        raise AssertionError("inference must not call a fitting path")

    monkeypatch.setattr(
        character_transformer, "fit_character_transformer", fail_if_called
    )
    loaded = _load_fixture(fixture_bundle)
    encoded = tuple(
        character_sequence.encode_character_url(raw_url, loaded.vocabulary)
        for raw_url in (
            "https://safe.example/account",
            "https://signin.example/verify",
        )
    )
    token_ids = torch.tensor([row.token_ids for row in encoded], dtype=torch.int64)
    padding_mask = torch.tensor([row.padding_mask for row in encoded], dtype=torch.bool)
    fixture_bundle.source_model.eval()
    with torch.inference_mode():
        before = fixture_bundle.source_model(token_ids, padding_mask)
    with torch.inference_mode():
        after = loaded._model(token_ids, padding_mask)

    # Fixed CPU fixture outputs also detect changes to forward semantics.
    expected = torch.tensor([0.6071698665618896, 0.6393680572509766])
    torch.testing.assert_close(before, expected, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(after, expected, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(after, before, rtol=0.0, atol=0.0)
    assert loaded._model.training is False
    assert all(not parameter.requires_grad for parameter in loaded._model.parameters())


def _fixture_byte_arguments(fixture):
    bundle, summary, stage1 = transformer_inference._snapshot_files(
        fixture.bundle_dir, fixture.summary_path, fixture.stage1_path
    )
    return {
        "bundle": bundle,
        "summary_bytes": summary,
        "stage1_bytes": stage1,
        "_hash_policy": transformer_inference._BundleHashPolicy(**fixture.policy),
        "_device": torch.device("cpu"),
        "_fixture_cpu": True,
    }


def test_byte_loader_matches_path_loader_without_reading_or_fitting(
    fixture_bundle, monkeypatch
):
    arguments = _fixture_byte_arguments(fixture_bundle)
    original_bundle = dict(arguments["bundle"])
    expected = _load_fixture(fixture_bundle)

    def forbidden(*args, **kwargs):
        pytest.fail("byte-only loading attempted filesystem access or fitting")

    monkeypatch.setattr(transformer_inference, "_snapshot_files", forbidden)
    monkeypatch.setattr(transformer_inference, "_read_regular_file", forbidden)
    monkeypatch.setattr(character_transformer, "fit_character_transformer", forbidden)
    actual = transformer_inference._load_transformer_cascade_bytes(**arguments)
    assert actual == expected
    assert arguments["bundle"] == original_bundle
    for name, value in expected._model.state_dict().items():
        torch.testing.assert_close(
            actual._model.state_dict()[name], value, rtol=0, atol=0
        )
    encoded = tuple(
        character_sequence.encode_character_url(url, actual.vocabulary)
        for url in ("https://safe.example/account", "https://signin.example/verify")
    )
    tokens = torch.tensor([row.token_ids for row in encoded], dtype=torch.int64)
    mask = torch.tensor([row.padding_mask for row in encoded], dtype=torch.bool)
    with torch.inference_mode():
        torch.testing.assert_close(
            actual._model(tokens, mask), expected._model(tokens, mask), rtol=0, atol=0
        )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda args: args["bundle"].pop("vocabulary.json"),
        lambda args: args["bundle"].update({"extra.json": b"{}\n"}),
        lambda args: args.update(bundle=list(args["bundle"].items())),
        lambda args: args["bundle"].update(
            {"vocabulary.json": bytearray(args["bundle"]["vocabulary.json"])}
        ),
        lambda args: args.update(summary_bytes=bytearray(args["summary_bytes"])),
        lambda args: args.update(stage1_bytes=args["stage1_bytes"].decode()),
        lambda args: args.update(summary_bytes=args["summary_bytes"] + b" "),
        lambda args: args.update(stage1_bytes=args["stage1_bytes"] + b" "),
        lambda args: args["bundle"].update({"SHA256SUMS": b"bad manifest"}),
        lambda args: args["bundle"].update({"transformer-weights.npz": b"bad weights"}),
    ],
)
def test_byte_loader_rejects_invalid_shape_types_or_tampering_before_weights(
    fixture_bundle, monkeypatch, mutation
):
    arguments = _fixture_byte_arguments(fixture_bundle)
    mutation(arguments)
    monkeypatch.setattr(
        transformer_inference,
        "_load_model",
        lambda *args: pytest.fail("invalid bytes reached weight loading"),
    )
    with pytest.raises(transformer_inference.TransformerInferenceError):
        transformer_inference._load_transformer_cascade_bytes(**arguments)


@pytest.mark.parametrize(
    "override",
    [
        {"_hash_policy": None},
        {"_device": "cpu"},
        {"_device": torch.device("cpu:0")},
        {"_fixture_cpu": False},
        {"_fixture_cpu": 1},
    ],
)
def test_byte_loader_preserves_hash_policy_and_device_gates(
    fixture_bundle, monkeypatch, override
):
    arguments = _fixture_byte_arguments(fixture_bundle)
    arguments.update(override)
    monkeypatch.setattr(
        transformer_inference,
        "_load_model",
        lambda *args: pytest.fail("invalid policy/device reached weight loading"),
    )
    with pytest.raises(transformer_inference.TransformerInferenceError):
        transformer_inference._load_transformer_cascade_bytes(**arguments)


def test_byte_loader_retains_public_private_projection_validation(fixture_bundle):
    summary = _read_json(fixture_bundle.summary_path)
    summary["transformer"]["fit"]["best_epoch"] = 2
    fixture_bundle = _rewrite_bound_bundle(fixture_bundle, summary=summary)
    with pytest.raises(
        transformer_inference.TransformerInferenceError,
        match="transformer fit projection differs",
    ):
        transformer_inference._load_transformer_cascade_bytes(
            **_fixture_byte_arguments(fixture_bundle)
        )


def test_path_loader_validates_before_snapshot_then_delegates_exact_bytes(
    fixture_bundle, monkeypatch
):
    arguments = _fixture_byte_arguments(fixture_bundle)
    events = []
    sentinel = object()
    validate_policy = transformer_inference._validate_hash_policy
    validate_device = transformer_inference._validate_device

    def policy(value):
        events.append("policy")
        return validate_policy(value)

    def device(value, fixture_cpu):
        events.append("device")
        return validate_device(value, fixture_cpu)

    def snapshot(*paths):
        assert paths == (
            fixture_bundle.bundle_dir,
            fixture_bundle.summary_path,
            fixture_bundle.stage1_path,
        )
        events.append("snapshot")
        return (
            arguments["bundle"],
            arguments["summary_bytes"],
            arguments["stage1_bytes"],
        )

    def load_bytes(bundle, summary_bytes, stage1_bytes, **kwargs):
        events.append("bytes")
        assert bundle is arguments["bundle"]
        assert summary_bytes is arguments["summary_bytes"]
        assert stage1_bytes is arguments["stage1_bytes"]
        assert kwargs == {
            key: value for key, value in arguments.items() if key.startswith("_")
        }
        return sentinel

    monkeypatch.setattr(transformer_inference, "_validate_hash_policy", policy)
    monkeypatch.setattr(transformer_inference, "_validate_device", device)
    monkeypatch.setattr(transformer_inference, "_snapshot_files", snapshot)
    monkeypatch.setattr(
        transformer_inference, "_load_transformer_cascade_bytes", load_bytes
    )
    assert _load_fixture(fixture_bundle) is sentinel
    assert events == ["policy", "device", "snapshot", "bytes"]


@pytest.mark.parametrize("override", [{"_hash_policy": None}, {"_fixture_cpu": False}])
def test_path_loader_rejects_invalid_policy_or_device_before_reading(
    fixture_bundle, monkeypatch, override
):
    arguments = _fixture_byte_arguments(fixture_bundle)
    options = {key: value for key, value in arguments.items() if key.startswith("_")}
    options.update(override)
    monkeypatch.setattr(
        transformer_inference,
        "_snapshot_files",
        lambda *args: pytest.fail("invalid policy/device reached private reads"),
    )
    with pytest.raises(transformer_inference.TransformerInferenceError):
        transformer_inference._load_transformer_cascade_bundle(
            fixture_bundle.bundle_dir,
            fixture_bundle.summary_path,
            fixture_bundle.stage1_path,
            **options,
        )
