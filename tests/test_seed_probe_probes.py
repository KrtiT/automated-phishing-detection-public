"""Invented byte bundles and score fixtures only; no research inputs or fits."""

import importlib
import warnings
from dataclasses import replace
from hashlib import sha256
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from test_fixed_cascade import _artifact as logistic_artifact
from test_fixed_cascade import _threshold_record
from test_length_inference import _artifact as length_artifact
from test_secondary_development import _fixture
from test_transformer_inference import CONTRACT, _cascade_record

from automated_phishing_detection import (
    bound_models,
    character_sequence,
    character_transformer,
    fixed_cascade,
    gmm_monitor,
    length_inference,
    probe_replay,
    secondary_drift,
    transformer_inference,
    transformer_pipeline,
)
from automated_phishing_detection import (
    secondary_development as development,
)


def module():
    return importlib.import_module("automated_phishing_detection.seed_probe_probes")


def fixture_data(train_count=256):
    source = _fixture(development, train_count=train_count, validation_count=640)
    arguments = dict(source["arguments"])
    pins = arguments["pins"]
    threshold = _threshold_record(positive=320, negative=320)
    logistic = logistic_artifact(pins.baseline_contract_sha256)
    logistic["scaler"]["n_samples_seen"] = train_count
    logistic["validation_threshold"] = threshold
    logistic["input_hashes"].update(
        train=pins.train_sha256,
        validation=pins.validation_sha256,
        preparation_summary=pins.preparation_summary_sha256,
    )
    logistic_bytes = development._json_bytes(logistic)
    pins = replace(pins, logistic_l1_artifact_sha256=sha256(logistic_bytes).hexdigest())
    arguments["logistic_l1"] = fixed_cascade._load_logistic_l1_artifact_bytes(
        logistic_bytes,
        expected_sha256=pins.logistic_l1_artifact_sha256,
        expected_contract_sha256=pins.baseline_contract_sha256,
    )
    arguments["gmm_state"]["input_hashes"]["logistic_l1_artifact"] = (
        pins.logistic_l1_artifact_sha256
    )
    gmm_bytes = development._json_bytes(arguments["gmm_state"])
    arguments["pins"] = pins = replace(
        pins, gmm_artifact_sha256=sha256(gmm_bytes).hexdigest()
    )
    reference = development.build_training_reference(**arguments)
    drift = development.evaluate_validation(reference, source["validation_content"])
    length = length_artifact()
    length["contract_sha256"] = pins.baseline_contract_sha256
    length["input_hashes"] = dict(logistic["input_hashes"])
    length["scaler"]["n_samples_seen"] = train_count
    length["validation_threshold"] = threshold
    vocabulary = character_sequence.build_character_vocabulary(
        row["raw_url"] for row in source["train"]
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        model = character_transformer.CharacterTransformer(vocabulary.size)
    fit = character_transformer.TransformerFit(
        model,
        (character_transformer.EpochRecord(1, 0.5, 0.75),),
        1,
        0.75,
        1,
        False,
        1.0,
        (0.5,) * 640,
    )
    weights, _ = transformer_pipeline._serialize_state_dict_npz(model)
    vocabulary_bytes = vocabulary.to_json().encode()
    hashes = {
        "train": pins.train_sha256,
        "validation": pins.validation_sha256,
        "preparation_summary": pins.preparation_summary_sha256,
        "baseline_contract": pins.baseline_contract_sha256,
        "logistic_l1_artifact": pins.logistic_l1_artifact_sha256,
        "transformer_contract": "c" * 64,
    }
    metadata = transformer_pipeline._transformer_metadata(
        fit=fit,
        threshold=threshold,
        input_hashes=hashes,
        contract=CONTRACT,
        vocabulary_sha256=sha256(vocabulary_bytes).hexdigest(),
        weights_sha256=sha256(weights).hexdigest(),
        training_device=torch.device("cpu"),
    )
    metadata_bytes = development._json_bytes(metadata)
    cascade = _cascade_record()
    cascade.update(
        counts=threshold["counts"],
        fpr_upper_95=threshold["fpr_upper_95"],
        transformer_invocation_rate=40 / 640,
    )
    cascade_metadata = transformer_pipeline._cascade_metadata(
        calibration=cascade,
        input_hashes=hashes,
        transformer_sha256=sha256(metadata_bytes).hexdigest(),
        stage1_warnings=[],
    )
    artifacts = {
        "length-only.json": development._json_bytes(length),
        "logistic-l1.json": logistic_bytes,
        "gmm.json": gmm_bytes,
        "transformer-weights.npz": weights,
        "vocabulary.json": vocabulary_bytes,
        "transformer.json": metadata_bytes,
        "cascade.json": development._json_bytes(cascade_metadata),
    }

    def partition(rows):
        labels = np.asarray([row["is_phishing"] for row in rows], dtype=np.int8)
        return transformer_pipeline._Partition(
            tuple(row["raw_url"] for row in rows),
            labels,
            frozenset(row["registrable_domain"] for row in rows),
            frozenset(range(len(rows))),
            {"0": int(sum(labels == 0)), "1": int(sum(labels == 1))},
        )

    summary = transformer_pipeline._public_summary(
        train=partition(source["train"]),
        validation=partition(source["validation"]),
        vocabulary=vocabulary,
        transformer_fit=fit,
        transformer_threshold=threshold,
        cascade=cascade,
        contract=CONTRACT,
        input_hashes=hashes,
        artifact_hashes={
            name: sha256(artifacts[name]).hexdigest()
            for name in transformer_inference._HASHED_FILENAMES
        },
        stage1_warnings=[],
    )
    summary_bytes = development._json_bytes(summary)
    binding = SimpleNamespace(
        pins=pins,
        preparation_bytes=arguments["preparation_summary"],
        primary_artifact_hashes=tuple(
            sorted(
                (name, sha256(content).hexdigest())
                for name, content in artifacts.items()
            )
        ),
        public_operating_points_json='{"length_threshold":0.5,"monitor_boundary":0.0,"stage1_threshold":0.5}',
        retained_drift_summary_json=development._json_bytes(
            drift.public_summary
        ).decode(),
        training_reference_sha256=sha256(
            drift.private_outputs["training-reference.json"]
        ).hexdigest(),
        validation_audit_sha256=sha256(
            drift.private_outputs["validation-audit.json"]
        ).hexdigest(),
        transformer_summary_bytes=summary_bytes,
        methods_sha256="a" * 64,
        profile_sha256="b" * 64,
        base=SimpleNamespace(
            source_hashes=(
                (
                    bound_models._PUBLIC_SUMMARIES["transformer"][0],
                    sha256(summary_bytes).hexdigest(),
                ),
            )
        ),
    )
    return SimpleNamespace(
        binding=binding,
        artifacts=artifacts,
        arguments={
            "validation_bytes": source["validation_content"],
            "suffix_rules_bytes": arguments["suffix_rules"],
            "artifacts": artifacts,
            "drift_reference_bytes": drift.private_outputs["training-reference.json"],
            "drift_audit_bytes": drift.private_outputs["validation-audit.json"],
        },
    )


def synthetic_scorer(calls):
    def make(length, cascade):
        def score(row):
            calls.append(row)
            return probe_replay.PrimaryScores(
                row.record_id,
                row.raw_url,
                0.2,
                0.3,
                0.8,
                '{"fixture":true}',
                '{"fixture":true}',
            )

        return score

    return make


@pytest.fixture(scope="module")
def sample():
    api = module()
    data = fixture_data()
    auxiliary, calls = {}, []
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(probe_replay, "make_primary_scorer", synthetic_scorer(calls))
        outputs, summary = api._run_probe_stage(
            data.binding,
            **data.arguments,
            retain=auxiliary.__setitem__,
            _fixture_cpu=True,
        )
    return SimpleNamespace(
        api=api,
        data=data,
        outputs=outputs,
        summary=summary,
        auxiliary=auxiliary,
        calls=calls,
    )


def verify(sample, *, outputs=None, auxiliary=None):
    return sample.api._verify_probe_stage(
        sample.data.binding,
        outputs=sample.outputs if outputs is None else outputs,
        auxiliary=sample.auxiliary if auxiliary is None else auxiliary,
        _fixture_cpu=True,
    )


def test_saved_probe_roundtrip_exact_names_scope_rows_windows_and_no_weights(sample):
    assert verify(sample) == sample.summary
    assert set(sample.outputs) == {"comparison.json"}
    assert (
        sample.summary["verification_scope"]
        == "saved_monitor_and_routing_arithmetic_no_independent_primary_or_source_rescoring"
    )
    assert len(sample.calls) == 4 * 320
    assert not any(name.endswith(".npz") for name in sample.auxiliary)
    assert len(sample.auxiliary) == 8 + 4 * 320 + 4
    record = development._json(sample.auxiliary["score-row-00-000001.json"])
    assert record["phase"] == "scored_pre_routing"
    assert record["stream_name"] == "original"
    assert "label" not in record["row"] and "is_phishing" not in record["row"]
    completed = development._json(sample.auxiliary["stream-00.json"])
    assert completed["phase"] == "completed_stream"
    assert completed["stream"]["rows"][256]["drift_override"] is True
    assert [
        window["end_position"]
        for window in completed["stream"]["monitors"][0]["windows"]
    ] == [256, 320]
    assert all(
        len(window["feature_scores"]) == 26
        for window in completed["stream"]["monitors"][2]["windows"]
    )
    encoded = sample.outputs["comparison.json"]
    for private in (b"raw_url", b"record_id", b"https://", b"is_phishing"):
        assert private not in encoded


def test_verification_never_reads_files_refits_or_rescores_primary_models(
    sample, monkeypatch
):
    def forbidden(*args, **kwargs):
        pytest.fail("saved verification touched sources, a fit, or primary inference")

    for owner, name in (
        (transformer_inference, "_load_model"),
        (transformer_inference, "_snapshot_files"),
        (length_inference, "score_length_only_authoritative"),
        (fixed_cascade, "score_logistic_l1_authoritative"),
        (character_transformer, "fit_character_transformer"),
        (secondary_drift, "fit_mmd_reference"),
        (secondary_drift, "fit_psi_reference"),
        (gmm_monitor, "fit_training_mixture"),
    ):
        monkeypatch.setattr(owner, name, forbidden)
    assert verify(sample) == sample.summary


@pytest.mark.parametrize(
    "name", ["input-gmm.json", "score-row-00-000001.json", "stream-03.json"]
)
def test_saved_inventory_requires_every_input_row_and_stream(sample, name):
    auxiliary = dict(sample.auxiliary)
    del auxiliary[name]
    with pytest.raises(sample.api.ProbeStageError):
        verify(sample, auxiliary=auxiliary)


def test_extra_output_or_auxiliary_member_is_not_ignored(sample):
    with pytest.raises(sample.api.ProbeStageError):
        verify(sample, auxiliary=sample.auxiliary | {"extra.json": b"{}\n"})
    with pytest.raises(sample.api.ProbeStageError):
        verify(sample, outputs=sample.outputs | {"extra.json": b"{}\n"})


@pytest.mark.parametrize(
    "change",
    [
        lambda row: row["mapping"].update(record_id="different"),
        lambda row: row["mapping"].update(validation_position=True),
        lambda row: row["mapping"].update(stream_position=2),
        lambda row: row["mapping"].update(changed=True),
        lambda row: row["mapping"].update(output_url="https://wrong.example/path"),
        lambda row: row.update(portable_monitor_probability=row["probabilities"][1]),
        lambda row: row.update(
            negative_log_likelihood=row["negative_log_likelihood"] + 1
        ),
        lambda row: row["standardized_monitor_features"].__setitem__(25, 0.0),
        lambda row: row["structural_features"].__setitem__(0, 0.0),
        lambda row: row["decisions"].__setitem__(0, True),
        lambda row: row.update(drift_override=True),
        lambda row: row.update(stage1_scoring_audit_json='{"x":1,"x":2}'),
    ],
)
def test_saved_scored_rows_are_exactly_aligned_and_recomputed(sample, change):
    auxiliary = dict(sample.auxiliary)
    name = "score-row-00-000001.json"
    record = development._json(auxiliary[name])
    change(record["row"])
    auxiliary[name] = development._json_bytes(record)
    with pytest.raises(sample.api.ProbeStageError):
        verify(sample, auxiliary=auxiliary)


def test_transformed_stream_cannot_compose_operators_or_inherit_wrong_mapping(sample):
    auxiliary = dict(sample.auxiliary)
    name = "score-row-02-000001.json"
    record = development._json(auxiliary[name])
    record["row"]["mapping"]["output_url"] = record["row"]["mapping"][
        "output_url"
    ].upper()
    auxiliary[name] = development._json_bytes(record)
    with pytest.raises(sample.api.ProbeStageError):
        verify(sample, auxiliary=auxiliary)


def test_saved_psi_features_and_aggregate_are_verified(sample):
    auxiliary = dict(sample.auxiliary)
    stream = development._json(auxiliary["stream-00.json"])
    stream["stream"]["monitors"][2]["windows"][0]["feature_scores"].pop()
    auxiliary["stream-00.json"] = development._json_bytes(stream)
    with pytest.raises(sample.api.ProbeStageError):
        verify(sample, auxiliary=auxiliary)
    value = development._json(sample.outputs["comparison.json"])
    value["result"]["streams"][0]["row_count"] -= 1
    with pytest.raises(sample.api.ProbeStageError):
        verify(sample, outputs={"comparison.json": development._json_bytes(value)})


def test_hash_mismatch_precedes_scoring_and_retention(sample, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("unbound artifact reached scoring or retention")

    monkeypatch.setattr(probe_replay, "make_primary_scorer", forbidden)
    supplied = sample.data.arguments | {
        "artifacts": sample.data.artifacts | {"gmm.json": b"{}\n"}
    }
    with pytest.raises(sample.api.ProbeStageError, match="hash"):
        sample.api._run_probe_stage(
            sample.data.binding, **supplied, retain=forbidden, _fixture_cpu=True
        )


def test_retained_inputs_and_raw_prefix_survive_scorer_failure(sample, monkeypatch):
    auxiliary, calls = {}, []

    def make(*args):
        base = synthetic_scorer(calls)(*args)

        def score(row):
            if len(calls) == 1:
                raise RuntimeError("invented primary failure")
            return base(row)

        return score

    monkeypatch.setattr(probe_replay, "make_primary_scorer", make)
    with pytest.raises(sample.api.ProbeStageError):
        sample.api._run_probe_stage(
            sample.data.binding,
            **sample.data.arguments,
            retain=auxiliary.__setitem__,
            _fixture_cpu=True,
        )
    assert len(calls) == 1
    assert len(auxiliary) == 9
    assert "score-row-00-000001.json" in auxiliary
    assert "stream-00.json" not in auxiliary


def test_public_entry_never_accepts_fixture_cpu_policy(sample, monkeypatch):
    monkeypatch.setattr(
        transformer_inference,
        "_load_model",
        lambda *args: pytest.fail("fixture reached public model loading"),
    )
    with pytest.raises(sample.api.ProbeStageError):
        sample.api.run_probe_stage(
            sample.data.binding, **sample.data.arguments, retain=lambda *args: None
        )


def test_safe_failed_checks_never_echo_private_exception_text():
    api = module()
    assert type(api.SAFE_CHECKS) is frozenset
    known = api.ProbeStageError("probe_row_mismatch")
    assert known.check_id == str(known) == "probe_row_mismatch"
    private = api.ProbeStageError("https://private.example/secret?token=credential")
    assert private.check_id == str(private) == "unclassified_check"


def test_underlying_private_scoring_error_is_safely_classified(sample, monkeypatch):
    def make(*args):
        def score(row):
            raise probe_replay.ProbeReplayError(
                "private URL https://private.example/token"
            )

        return score

    monkeypatch.setattr(probe_replay, "make_primary_scorer", make)
    with pytest.raises(sample.api.ProbeStageError) as caught:
        sample.api._run_probe_stage(
            sample.data.binding,
            **sample.data.arguments,
            retain=lambda *args: None,
            _fixture_cpu=True,
        )
    assert caught.value.check_id == "probe_replay"
    assert "private" not in str(caught.value)


@pytest.mark.parametrize(
    ("failure", "expected_scores", "expected_streams"),
    [("model", 0, 0), ("monitor", 320, 0), ("retention", 1, 0), ("aggregate", 1280, 4)],
)
def test_probe_stage_retains_prefix_before_each_later_failure(
    sample, monkeypatch, failure, expected_scores, expected_streams
):
    auxiliary, calls = {}, []
    monkeypatch.setattr(probe_replay, "make_primary_scorer", synthetic_scorer(calls))

    def stop(*args, **kwargs):
        raise RuntimeError("private URL https://private.example/credential")

    if failure == "model":
        monkeypatch.setattr(
            transformer_inference, "_load_transformer_cascade_bytes", stop
        )
    elif failure == "monitor":
        monkeypatch.setattr(secondary_drift, "mmd_window_scores", stop)
    elif failure == "aggregate":
        monkeypatch.setattr(sample.api, "_aggregate", stop)

    def retain(name, content):
        assert name not in auxiliary
        auxiliary[name] = content
        if failure == "retention" and name.startswith("score-row-"):
            stop()

    with pytest.raises(sample.api.ProbeStageError) as caught:
        sample.api._run_probe_stage(
            sample.data.binding,
            **sample.data.arguments,
            retain=retain,
            _fixture_cpu=True,
        )
    assert caught.value.check_id == (
        "probe_retention" if failure == "retention" else "probe_replay"
    )
    assert len(calls) == expected_scores
    assert sum(name.startswith("input-") for name in auxiliary) == 8
    assert sum(name.startswith("score-row-") for name in auxiliary) == expected_scores
    assert sum(name.startswith("stream-") for name in auxiliary) == expected_streams


@pytest.mark.parametrize(
    ("name", "mutate"),
    [
        ("score-row-00-000001.json", lambda value: value.update(schema_version=True)),
        (
            "score-row-00-000001.json",
            lambda value: value["row"]["mapping"].update(eligible=1),
        ),
        (
            "score-row-00-000001.json",
            lambda value: value["row"]["probabilities"].__setitem__(0, True),
        ),
        ("score-row-00-000001.json", lambda value: value["row"].update(label=1)),
        ("stream-00.json", lambda value: value.update(schema_version=1.0)),
        (
            "stream-00.json",
            lambda value: value["stream"]["monitors"][0]["windows"][0].update(alert=1),
        ),
        (
            "stream-00.json",
            lambda value: value["stream"]["monitors"][1].update(
                calibration_window_count=True
            ),
        ),
    ],
)
def test_reencoded_records_cannot_relax_exact_schema_or_types(sample, name, mutate):
    auxiliary = dict(sample.auxiliary)
    value = development._json(auxiliary[name])
    mutate(value)
    auxiliary[name] = development._json_bytes(value)
    with pytest.raises(sample.api.ProbeStageError):
        verify(sample, auxiliary=auxiliary)


def test_stage_roundtrip_preserves_unavailable_accepted_reference(monkeypatch):
    api = module()
    data = fixture_data(train_count=8)
    auxiliary = {}
    monkeypatch.setattr(probe_replay, "make_primary_scorer", synthetic_scorer([]))
    outputs, summary = api._run_probe_stage(
        data.binding, **data.arguments, retain=auxiliary.__setitem__, _fixture_cpu=True
    )
    assert (
        api._verify_probe_stage(
            data.binding, outputs=outputs, auxiliary=auxiliary, _fixture_cpu=True
        )
        == summary
    )
    for stream in summary["result"]["streams"]:
        mmd = stream["monitors"]["mmd"]
        assert mmd["reason"] == "fewer_than_256_training_domains"
        assert mmd["window_count"] == 2
        assert mmd["alert_count"] is None
        assert (
            stream["paired_with_original"]["monitors"]["mmd"]["score_differences"]
            is None
        )


def test_producer_monitor_warnings_are_fatal_with_scored_prefix_and_state_restored(
    sample, monkeypatch
):
    auxiliary, calls, numerical_states = {}, [], []
    original_monitor = secondary_drift.mmd_window_scores

    def make(*args):
        score = synthetic_scorer(calls)(*args)

        def capture(row):
            numerical_states.append(np.geterr())
            return score(row)

        return capture

    def warning_monitor(*args):
        warnings.warn("invented monitor warning", RuntimeWarning)
        return original_monitor(*args)

    monkeypatch.setattr(probe_replay, "make_primary_scorer", make)
    monkeypatch.setattr(secondary_drift, "mmd_window_scores", warning_monitor)
    torch_state = (
        torch.get_num_threads(),
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
    )
    with warnings.catch_warnings(record=True), np.errstate(all="ignore"):
        filters = list(warnings.filters)
        with pytest.raises(sample.api.ProbeStageError) as caught:
            sample.api._run_probe_stage(
                sample.data.binding,
                **sample.data.arguments,
                retain=auxiliary.__setitem__,
                _fixture_cpu=True,
            )
        assert caught.value.check_id == "probe_replay"
        assert np.geterr() == dict.fromkeys(
            ("divide", "over", "under", "invalid"), "ignore"
        )
        assert warnings.filters == filters
    assert len(calls) == 320
    assert all(
        state == dict.fromkeys(("divide", "over", "under", "invalid"), "raise")
        for state in numerical_states
    )
    assert len(auxiliary) == 8 + 320
    assert "score-row-00-000320.json" in auxiliary
    assert "stream-00.json" not in auxiliary
    assert torch_state == (
        torch.get_num_threads(),
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
    )
