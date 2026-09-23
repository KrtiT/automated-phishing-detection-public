"""Byte-only probe production and saved-evidence arithmetic verification.

The supervisor authenticates the binding, source reads, writer and worker exit.
Saved original URLs cannot independently prove source membership. Verification
recomputes monitor and routing arithmetic, never primary inference or a fit.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, fields
from hashlib import sha256

import torch

from . import (
    bound_models,
    development_probes,
    fixed_cascade,
    gmm_monitor,
    length_inference,
    probe_replay,
    retained_drift,
    transformer_inference,
    transformer_pipeline,
)
from . import (
    secondary_development as development,
)
from .selective_inference import SelectiveCascade

SAFE_CHECKS = frozenset(
    {
        "input_validation",
        "artifact_hash_mismatch",
        "reference_hash_mismatch",
        "official_primary_binding_mismatch",
        "probe_inventory_mismatch",
        "probe_metadata",
        "probe_row_mismatch",
        "probe_stream_mismatch",
        "probe_aggregate_mismatch",
        "probe_replay",
        "probe_retention",
        "probe_verification",
        "unclassified_check",
    }
)
_METADATA_NAMES = (
    "length-only.json",
    "logistic-l1.json",
    "gmm.json",
    "transformer.json",
    "cascade.json",
    "vocabulary.json",
)
_ARTIFACT_NAMES = frozenset((*_METADATA_NAMES, "transformer-weights.npz"))
_REFERENCE_NAMES = ("input-training-reference.json", "input-validation-audit.json")
_INPUT_NAMES = frozenset(
    (*[f"input-{name}" for name in _METADATA_NAMES], *_REFERENCE_NAMES)
)
_VERIFICATION_SCOPE = (
    "saved_monitor_and_routing_arithmetic_no_independent_primary_or_source_rescoring"
)


class ProbeStageError(ValueError):
    """Only fixed symbolic checks, never a URL or underlying exception message."""

    def __init__(self, check_id):
        self.check_id = (
            check_id
            if type(check_id) is str and check_id in SAFE_CHECKS
            else "unclassified_check"
        )
        super().__init__(self.check_id)


@dataclass(frozen=True)
class _Inputs:
    policy: transformer_inference._BundleHashPolicy
    length: length_inference.LoadedLengthOnly
    stage1: fixed_cascade.PortableLogisticL1
    gmm: dict
    retained: retained_drift.RetainedDriftReference
    points: probe_replay.OperatingPoints


def _require(condition, check_id):
    if not condition:
        raise ProbeStageError(check_id)


def _json_bytes(value):
    return development._json_bytes(value)


def _saved_json(content):
    _require(type(content) is bytes, "input_validation")
    value = development._json(content)
    _require(_json_bytes(value) == content, "input_validation")
    return value


def _fields(value, expected, check_id):
    _require(type(value) is dict and set(value) == set(expected), check_id)
    return value


def _bound(content, digest, check_id):
    _require(type(content) is bytes, check_id)
    fixed_cascade._lowercase_sha256(digest, "expected digest")
    _require(sha256(content).hexdigest() == digest, check_id)


def _authenticate(binding, artifacts, reference_bytes, audit_bytes, *, weights):
    hashes = dict(binding.primary_artifact_hashes)
    _require(
        len(binding.primary_artifact_hashes) == len(_ARTIFACT_NAMES)
        and set(hashes) == _ARTIFACT_NAMES,
        "input_validation",
    )
    expected = _ARTIFACT_NAMES if weights else set(_METADATA_NAMES)
    _fields(artifacts, expected, "probe_inventory_mismatch")
    for name, content in artifacts.items():
        _bound(content, hashes[name], "artifact_hash_mismatch")
    _require(
        hashes["logistic-l1.json"] == binding.pins.logistic_l1_artifact_sha256
        and hashes["gmm.json"] == binding.pins.gmm_artifact_sha256,
        "artifact_hash_mismatch",
    )
    _bound(
        binding.preparation_bytes,
        binding.pins.preparation_summary_sha256,
        "input_validation",
    )
    _bound(
        reference_bytes, binding.training_reference_sha256, "reference_hash_mismatch"
    )
    _bound(audit_bytes, binding.validation_audit_sha256, "reference_hash_mismatch")
    summary_path = bound_models._PUBLIC_SUMMARIES["transformer"][0]
    _bound(
        binding.transformer_summary_bytes,
        dict(binding.base.source_hashes)[summary_path],
        "artifact_hash_mismatch",
    )
    return hashes


def _policy(binding, summary, fixture_cpu):
    _require(type(fixture_cpu) is bool, "input_validation")
    pins = binding.pins
    policy = transformer_inference._BundleHashPolicy(
        sha256(binding.transformer_summary_bytes).hexdigest(),
        pins.train_sha256,
        pins.validation_sha256,
        pins.preparation_summary_sha256,
        summary["input_hashes"]["transformer_contract"],
        pins.baseline_contract_sha256,
        pins.logistic_l1_artifact_sha256,
    )
    transformer_inference._validate_hash_policy(policy)
    if not fixture_cpu:
        official = transformer_pipeline._OFFICIAL_INPUT_HASH_POLICY
        expected = transformer_inference._BundleHashPolicy(
            bound_models._PUBLIC_SUMMARIES["transformer"][1],
            official.train_sha256,
            official.validation_sha256,
            official.preparation_summary_sha256,
            transformer_pipeline.OFFICIAL_TRANSFORMER_CONTRACT_SHA256,
            fixed_cascade.OFFICIAL_BASELINE_CONTRACT_SHA256,
            fixed_cascade.OFFICIAL_LOGISTIC_L1_SHA256,
        )
        _require(
            policy == expected
            and all(
                dict(binding.primary_artifact_hashes)[name] == digest
                for name, digest in bound_models._ARTIFACT_DIGESTS.items()
            ),
            "official_primary_binding_mismatch",
        )
    return policy


def _load_inputs(binding, artifacts, reference_bytes, audit_bytes, fixture_cpu):
    """Validate accepted metadata without constructing or evaluating a transformer."""
    try:
        summary = transformer_inference._canonical_json(
            binding.transformer_summary_bytes, "transformer summary"
        )
        policy = _policy(binding, summary, fixture_cpu)
        hashes = dict(binding.primary_artifact_hashes)
        length = length_inference._load_length_only_artifact_bytes(
            artifacts["length-only.json"],
            expected_sha256=hashes["length-only.json"],
            expected_contract_sha256=policy.baseline_contract_sha256,
        )
        stage1 = fixed_cascade._load_logistic_l1_artifact_bytes(
            artifacts["logistic-l1.json"],
            expected_sha256=policy.logistic_l1_artifact_sha256,
            expected_contract_sha256=policy.baseline_contract_sha256,
        )
        gmm = gmm_monitor.load_gmm_artifact_bytes(artifacts["gmm.json"])
        transformer, threshold = transformer_inference._validate_transformer_metadata(
            transformer_inference._canonical_json(
                artifacts["transformer.json"], "transformer metadata"
            ),
            policy=policy,
            artifact_hashes=hashes,
            device=torch.device("cpu" if fixture_cpu else "mps"),
        )
        cascade, calibration = transformer_inference._validate_cascade_metadata(
            transformer_inference._canonical_json(
                artifacts["cascade.json"], "cascade metadata"
            ),
            policy=policy,
            artifact_hashes=hashes,
            transformer_metadata=transformer,
            transformer_threshold=threshold,
        )
        counts = transformer_inference._validate_public_projection(
            summary,
            artifact_hashes={
                name: hashes[name] for name in transformer_inference._HASHED_FILENAMES
            },
            transformer=transformer,
            transformer_threshold=threshold,
            cascade=cascade,
            calibration=calibration,
            vocabulary=transformer_inference._load_vocabulary(
                artifacts["vocabulary.json"]
            ),
        )
        expected_inputs = {
            "train": binding.pins.train_sha256,
            "validation": binding.pins.validation_sha256,
            "preparation_summary": binding.pins.preparation_summary_sha256,
            "contract": binding.pins.baseline_contract_sha256,
        }
        for name, model in (("length-only.json", length), ("logistic-l1.json", stage1)):
            _require(
                development._json(artifacts[name])["input_hashes"] == expected_inputs,
                "probe_metadata",
            )
            selected = transformer_inference._validate_threshold(
                model.validation_threshold_record, "primary threshold"
            )
            _require(
                selected["counts"]["negative"] == counts["0"]
                and selected["counts"]["positive"] == counts["1"],
                "probe_metadata",
            )
        retained = retained_drift.load_retained_drift_reference(
            reference_bytes,
            audit_bytes,
            expected_reference_sha256=binding.training_reference_sha256,
            expected_audit_sha256=binding.validation_audit_sha256,
            pins=binding.pins,
            preparation_summary=binding.preparation_bytes,
            expected_drift_summary=development._json(
                binding.retained_drift_summary_json
            ),
        )
        mean, scale = development._accepted_states(
            stage1, gmm, binding.pins, retained.psi.training_row_count
        )
        _require(
            mean == retained.scaler_mean
            and scale == retained.scaler_scale
            and sha256(_json_bytes(development._portable_snapshot(stage1))).hexdigest()
            == retained.portable_state_sha256,
            "probe_metadata",
        )
        public_points = _fields(
            development._json(binding.public_operating_points_json),
            {"length_threshold", "stage1_threshold", "monitor_boundary"},
            "probe_metadata",
        )
        _require(
            public_points["length_threshold"]
            == length.validation_threshold_record["threshold"]
            and public_points["stage1_threshold"]
            == stage1.validation_threshold_record["threshold"],
            "probe_metadata",
        )
        points = probe_replay.OperatingPoints(
            public_points["length_threshold"],
            public_points["stage1_threshold"],
            threshold["threshold"],
            calibration["half_width"],
            public_points["monitor_boundary"],
        )
        probe_replay._validate_operating_points(points)
        return _Inputs(policy, length, stage1, gmm, retained, points)
    except ProbeStageError:
        raise
    except Exception:
        raise ProbeStageError("probe_metadata") from None


def _row_name(stream_index, position):
    return f"score-row-{stream_index:02d}-{position:06d}.json"


def _row_record(name, row):
    return {
        "schema_version": 1,
        "phase": "scored_pre_routing",
        "stream_name": name,
        "row": asdict(row),
    }


def _stream_record(stream):
    return {"schema_version": 1, "phase": "completed_stream", "stream": asdict(stream)}


def _aggregate(binding, points, result):
    return {
        "schema_version": 1,
        "status": "completed_secondary_probe_replay",
        "analysis_stage": "development_validation_only",
        "analysis_role": "secondary_descriptive_only",
        "protected_evaluation_authorized": False,
        "verification_scope": _VERIFICATION_SCOPE,
        "primary_artifact_sha256": dict(binding.primary_artifact_hashes),
        "input_hashes": asdict(binding.pins),
        "training_reference_sha256": binding.training_reference_sha256,
        "validation_audit_sha256": binding.validation_audit_sha256,
        "operating_points": asdict(points),
        "result": result.public_summary,
    }


def run_probe_stage(
    binding,
    *,
    validation_bytes,
    suffix_rules_bytes,
    artifacts,
    drift_reference_bytes,
    drift_audit_bytes,
    retain: Callable[[str, bytes], None],
):
    """Production MPS entry; accepts already-bound bytes, never input paths."""
    return _run_probe_stage(
        binding,
        validation_bytes=validation_bytes,
        suffix_rules_bytes=suffix_rules_bytes,
        artifacts=artifacts,
        drift_reference_bytes=drift_reference_bytes,
        drift_audit_bytes=drift_audit_bytes,
        retain=retain,
        _fixture_cpu=False,
    )


def _run_probe_stage(
    binding,
    *,
    validation_bytes,
    suffix_rules_bytes,
    artifacts,
    drift_reference_bytes,
    drift_audit_bytes,
    retain,
    _fixture_cpu=False,
):
    """Private synthetic CPU seam; retain every scored prefix before later work."""
    try:
        _require(callable(retain), "input_validation")
        hashes = _authenticate(
            binding, artifacts, drift_reference_bytes, drift_audit_bytes, weights=True
        )
        _bound(validation_bytes, binding.pins.validation_sha256, "input_validation")
        _bound(suffix_rules_bytes, binding.pins.suffix_rules_sha256, "input_validation")

        def save(name, content):
            try:
                retain(name, content)
            except Exception:
                raise ProbeStageError("probe_retention") from None

        for name in _METADATA_NAMES:
            save(f"input-{name}", artifacts[name])
        save(_REFERENCE_NAMES[0], drift_reference_bytes)
        save(_REFERENCE_NAMES[1], drift_audit_bytes)
        inputs = _load_inputs(
            binding, artifacts, drift_reference_bytes, drift_audit_bytes, _fixture_cpu
        )
        rows = development_probes.prepare_audit_probe_rows(
            validation_bytes=validation_bytes,
            suffix_rules_bytes=suffix_rules_bytes,
            preparation_summary=binding.preparation_bytes,
            retained=inputs.retained,
        )
        bundle = {
            name: artifacts[name] for name in transformer_inference._HASHED_FILENAMES
        }
        bundle["SHA256SUMS"] = "".join(
            f"{hashes[name]}  {name}\n"
            for name in transformer_inference._HASHED_FILENAMES
        ).encode("ascii")
        loaded = transformer_inference._load_transformer_cascade_bytes(
            bundle,
            binding.transformer_summary_bytes,
            artifacts["logistic-l1.json"],
            _hash_policy=inputs.policy,
            _device=torch.device("cpu" if _fixture_cpu else "mps"),
            _fixture_cpu=_fixture_cpu,
        )

        def row_callback(name, row):
            save(
                _row_name(
                    probe_replay.STREAM_NAMES.index(name), row.mapping.stream_position
                ),
                _json_bytes(_row_record(name, row)),
            )

        def stream_callback(stream):
            save(
                f"stream-{probe_replay.STREAM_NAMES.index(stream.name):02d}.json",
                _json_bytes(_stream_record(stream)),
            )

        # Initialize the owner pools before imposing the strict monitor context.
        with (
            SelectiveCascade(loaded, _fixture_cpu=_fixture_cpu) as cascade,
            development._numerical_context(),
        ):
            result = probe_replay.replay_probes(
                rows,
                primary_scorer=probe_replay.make_primary_scorer(inputs.length, cascade),
                stage1_model=inputs.stage1,
                gmm_artifact=inputs.gmm,
                operating_points=inputs.points,
                mmd_reference=inputs.retained.mmd,
                mmd_calibration=inputs.retained.mmd_calibration,
                psi_reference=inputs.retained.psi,
                psi_calibration=inputs.retained.psi_calibration,
                row_callback=row_callback,
                stream_callback=stream_callback,
            )
        aggregate = _aggregate(binding, inputs.points, result)
        return {"comparison.json": _json_bytes(aggregate)}, aggregate
    except ProbeStageError:
        raise
    except Exception as exc:
        if type(exc.__cause__) is ProbeStageError:
            raise ProbeStageError(exc.__cause__.check_id) from None
        raise ProbeStageError("probe_replay") from None


def _restore_row(value):
    _fields(
        value,
        (field.name for field in fields(probe_replay.ProbeRow)),
        "probe_row_mismatch",
    )
    mapping = _fields(
        value["mapping"],
        (field.name for field in fields(probe_replay.ProbeMapping)),
        "probe_row_mismatch",
    )
    row = dict(value)
    row["mapping"] = probe_replay.ProbeMapping(**mapping)
    for name, width in (
        ("probabilities", 5),
        ("decisions", 5),
        ("structural_features", 25),
        ("standardized_monitor_features", 26),
    ):
        _require(
            type(row[name]) is list and len(row[name]) == width, "probe_row_mismatch"
        )
        row[name] = tuple(row[name])
    return probe_replay.ProbeRow(**row)


def _restore_stream(value):
    _fields(
        value,
        (field.name for field in fields(probe_replay.ProbeStream)),
        "probe_stream_mismatch",
    )
    _require(
        type(value["rows"]) is list and type(value["monitors"]) is list,
        "probe_stream_mismatch",
    )
    monitors = []
    for value_monitor in value["monitors"]:
        monitor = dict(
            _fields(
                value_monitor,
                (field.name for field in fields(probe_replay.MonitorReplay)),
                "probe_stream_mismatch",
            )
        )
        _require(type(monitor["windows"]) is list, "probe_stream_mismatch")
        windows = []
        for value_window in monitor["windows"]:
            window = dict(
                _fields(
                    value_window,
                    (field.name for field in fields(probe_replay.MonitorWindow)),
                    "probe_stream_mismatch",
                )
            )
            _require(type(window["feature_scores"]) is list, "probe_stream_mismatch")
            window["feature_scores"] = tuple(window["feature_scores"])
            windows.append(probe_replay.MonitorWindow(**window))
        monitor["windows"] = tuple(windows)
        monitors.append(probe_replay.MonitorReplay(**monitor))
    return probe_replay.ProbeStream(
        value["name"],
        tuple(_restore_row(row) for row in value["rows"]),
        tuple(monitors),
    )


def verify_probe_stage(binding, *, outputs, auxiliary):
    """Verify saved bytes only; no independent source/primary prediction check."""
    return _verify_probe_stage(
        binding, outputs=outputs, auxiliary=auxiliary, _fixture_cpu=False
    )


def _verify_probe_stage(binding, *, outputs, auxiliary, _fixture_cpu=False):
    try:
        _fields(outputs, {"comparison.json"}, "probe_inventory_mismatch")
        _require(
            type(auxiliary) is dict and _INPUT_NAMES <= set(auxiliary),
            "probe_inventory_mismatch",
        )
        artifacts = {name: auxiliary[f"input-{name}"] for name in _METADATA_NAMES}
        reference_bytes, audit_bytes = (auxiliary[name] for name in _REFERENCE_NAMES)
        _authenticate(binding, artifacts, reference_bytes, audit_bytes, weights=False)
        inputs = _load_inputs(
            binding, artifacts, reference_bytes, audit_bytes, _fixture_cpu
        )
        count = len(inputs.retained.audit_record_ids)
        expected_names = (
            _INPUT_NAMES
            | {
                _row_name(index, position)
                for index in range(4)
                for position in range(1, count + 1)
            }
            | {f"stream-{index:02d}.json" for index in range(4)}
        )
        _require(set(auxiliary) == expected_names, "probe_inventory_mismatch")
        raw_streams = []
        for index, name in enumerate(probe_replay.STREAM_NAMES):
            rows = []
            for position in range(1, count + 1):
                value = _fields(
                    _saved_json(auxiliary[_row_name(index, position)]),
                    {"schema_version", "phase", "stream_name", "row"},
                    "probe_row_mismatch",
                )
                row = _restore_row(value["row"])
                _require(
                    _json_bytes(value) == _json_bytes(_row_record(name, row)),
                    "probe_row_mismatch",
                )
                _require(
                    type(row.mapping.validation_position) is int
                    and type(row.mapping.stream_position) is int
                    and row.mapping.record_id
                    == inputs.retained.audit_record_ids[position - 1]
                    and row.mapping.validation_position
                    == inputs.retained.audit_validation_positions[position - 1]
                    and row.mapping.stream_position == position,
                    "probe_row_mismatch",
                )
                rows.append(row)
            raw_streams.append(tuple(rows))
        original_inputs = tuple(
            probe_replay.AuditInput(
                row.mapping.record_id,
                row.mapping.validation_position,
                row.mapping.original_url,
            )
            for row in raw_streams[0]
        )
        mappings = probe_replay._prepare_mappings(original_inputs)
        completed = []
        with development._numerical_context():
            for index, (name, rows, expected_mappings) in enumerate(
                zip(probe_replay.STREAM_NAMES, raw_streams, mappings, strict=True)
            ):
                for row, mapping in zip(rows, expected_mappings, strict=True):

                    def saved_scores(request):
                        return probe_replay.PrimaryScores(
                            request.record_id,
                            request.raw_url,
                            *row.probabilities[:3],
                            row.length_scoring_audit_json,
                            row.stage1_scoring_audit_json,
                        )

                    expected_row = probe_replay._score_mapping(
                        mapping, saved_scores, inputs.stage1, inputs.gmm, inputs.points
                    )
                    _require(
                        _json_bytes(asdict(row)) == _json_bytes(asdict(expected_row)),
                        "probe_row_mismatch",
                    )
                replayed = probe_replay._replay_stream(
                    name,
                    rows,
                    inputs.points,
                    inputs.retained.mmd,
                    inputs.retained.mmd_calibration,
                    inputs.retained.psi,
                    inputs.retained.psi_calibration,
                )
                value = _fields(
                    _saved_json(auxiliary[f"stream-{index:02d}.json"]),
                    {"schema_version", "phase", "stream"},
                    "probe_stream_mismatch",
                )
                restored = _restore_stream(value["stream"])
                _require(
                    _json_bytes(value) == _json_bytes(_stream_record(restored))
                    and _json_bytes(asdict(restored)) == _json_bytes(asdict(replayed)),
                    "probe_stream_mismatch",
                )
                completed.append(replayed)
        aggregate = _aggregate(
            binding, inputs.points, probe_replay.ProbeReplay(tuple(completed))
        )
        _require(
            _json_bytes(_saved_json(outputs["comparison.json"]))
            == _json_bytes(aggregate),
            "probe_aggregate_mismatch",
        )
        return aggregate
    except ProbeStageError:
        raise
    except Exception:
        raise ProbeStageError("probe_verification") from None
