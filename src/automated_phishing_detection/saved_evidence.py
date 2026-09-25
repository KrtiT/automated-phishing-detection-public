"""Reconstruct internal scientific evidence from retained bytes only."""

from __future__ import annotations

import base64
import json
import math
from dataclasses import asdict, dataclass
from hashlib import sha256

import threadpoolctl

from . import (
    evaluation_manifest,
    fixed_cascade,
    gmm_monitor,
    hypothesis_evaluation,
    length_inference,
    policy_replay,
    secondary_metrics,
)
from ._saved_score_validation import validate_score_values
from .bound_secondary import SecondaryInferenceCounts
from .evaluation_manifest import ManifestRecord
from .evaluation_producer import ManifestOutcome, ScoredInternalRow
from .hypothesis_evaluation import PrimaryEvaluation, SavedPopulation, WindowCounts
from .paired_evaluation import BinaryPrediction, EvaluationRecord
from .selective_inference import InferenceCounts

_TABULAR_NAMES = (
    "formatting",
    "permutation_42",
    "permutation_43",
    "permutation_44",
    "permutation_45",
    "permutation_46",
    "random_forest",
)
_SEEDS = (42, 43, 44, 45, 46)
_PREVALENCES = (10, 100, 500)
_ROW_FIELDS = frozenset(ScoredInternalRow.__dataclass_fields__)
_RECORD_FIELDS = frozenset(ManifestRecord.__dataclass_fields__)
_EXPECTED_BINDING_CORE = {
    "gmm_audit": {"alert_count": 28, "window_count": 252},
    "artifact_hashes": {
        "cascade.json": "7ac88c784dbc299d436a029904f0c6bde5a8cf741e62028a7635e8672e55119c",
        "gmm.json": "a01a8b143c2423df57a153462cc47e79822ad1c6768213dc69e03683d4009745",
        "length-only.json": "b8b92cfbe29160e769e5e7d80712becc8fc0680cdfd45a44b839ef9bada87799",
        "logistic-l1.json": "71a3e24a0283a31ba188bc7dd60b18c1b708370b9ca275d5ab1a1004680c968a",
        "transformer-weights.npz": "1d4cdef31cb23cb84f093ca61c0afe0318142fa45ae559acd78cf10b49ee5de7",
        "transformer.json": "a13b6b7d554db6a9ee5b1689ecef44e2ad8069fcd25967f931101bdd3b256727",
        "vocabulary.json": "68bda780006d07b3b849abc366fbe3ccffe8a29e8984b092396d04f4eef43579",
    },
    "thresholds": {
        "length_only": 0.7612031186147,
        "logistic_l1": 0.2670846328466124,
        "transformer": 0.033976949751377106,
        "half_width": 0.0,
        "monitor_boundary": -67.45792380813624,
    },
    "secondary": {
        "accepted_report_sha256": {
            "tabular": "663f1117cd33f949b70c35c56810764193f69cae3a37505004d0db27641d829d",
            "seeds": "d63a85792088871cfb667e7e5cbe86c6148dc3a6c7430ca2e4db29d788ab8e23",
        },
        "device_type": "mps",
        "stage1_threshold": 0.2670846328466124,
        "vocabulary_sha256": "68bda780006d07b3b849abc366fbe3ccffe8a29e8984b092396d04f4eef43579",
        "tabular": [
            {
                "name": "formatting",
                "artifact_sha256": "f7d9b6623e116c11a270ab51e60126e0b9685bd7958c3870e4030ba2ebb58119",
                "threshold": 0.999925571711023,
            },
            {
                "name": "permutation_42",
                "artifact_sha256": "b6f28d37c59575ffc42cc08dd6b942f4831256280ceea0c3094c5a3460215792",
                "threshold": 0.5042071644591366,
            },
            {
                "name": "permutation_43",
                "artifact_sha256": "f3cf518d102f4719be2b9b86519358c131a0b1619b2af298a5811a40ae884d16",
                "threshold": 0.5031540848581733,
            },
            {
                "name": "permutation_44",
                "artifact_sha256": "6fd1f0218b8e667d274bb0d7c11859d4fc3ff764776990dbd8dd603c9573de67",
                "threshold": 0.5058518481954868,
            },
            {
                "name": "permutation_45",
                "artifact_sha256": "8aba0d802558a3d3c29f0b3ff457e8157353078b47c0783e82c570c605c79ac6",
                "threshold": 0.5087053417015189,
            },
            {
                "name": "permutation_46",
                "artifact_sha256": "6cd9d33e49da967a558c6e74e061dbf812d5313642f28955d41dd491d58d1fb9",
                "threshold": 0.5014918824187501,
            },
            {
                "name": "random_forest",
                "artifact_sha256": "fb134cfb5da65fb5d16b1503595c69e616ebef6699de6f27ef993a0e8ee3a17f",
                "threshold": 0.2,
            },
        ],
        "seeds": [
            {
                "seed": 42,
                "weights_sha256": "1d4cdef31cb23cb84f093ca61c0afe0318142fa45ae559acd78cf10b49ee5de7",
                "transformer_threshold": 0.03397693857550621,
                "half_width": 0.0,
                "reuses_primary": True,
            },
            {
                "seed": 43,
                "weights_sha256": "9b8e676547fcc6883b945c41c422819ee31cbaa7cae839b0492640877ec1c881",
                "transformer_threshold": 0.17120759189128876,
                "half_width": 0.0,
                "reuses_primary": False,
            },
            {
                "seed": 44,
                "weights_sha256": "8a67b72fe86eaffd0d1336daeb54fa5f6398aac5ed82a9125db801ffbdbfa533",
                "transformer_threshold": 0.03427749499678612,
                "half_width": 0.0,
                "reuses_primary": False,
            },
            {
                "seed": 45,
                "weights_sha256": "decec41908400fb7cbb775a3390dc0118af148272da3d68961aaacdf9f70f5aa",
                "transformer_threshold": 0.05094735324382782,
                "half_width": 0.0,
                "reuses_primary": False,
            },
            {
                "seed": 46,
                "weights_sha256": "bac18ebed42a8577a52ed84b7dd1d7fce3939043105b3ea379d804d20d9928c4",
                "transformer_threshold": 0.02175699733197689,
                "half_width": 0.0,
                "reuses_primary": False,
            },
        ],
    },
}


class SavedEvidenceError(ValueError):
    """Retained evidence is noncanonical, malformed, or internally inconsistent."""


@dataclass(frozen=True)
class ReconstructedInternalEvidence:
    row_count: int
    domain_count: int
    class_counts: dict[str, int]
    inference_counts: InferenceCounts
    secondary_inference_counts: SecondaryInferenceCounts
    manifests: dict[int, ManifestOutcome]
    primary: PrimaryEvaluation
    secondary: dict


def _require(condition: object, message: str) -> None:
    if not condition:
        raise SavedEvidenceError(message)


def _json_bytes(value: object) -> bytes:
    try:
        return (
            json.dumps(
                value,
                allow_nan=False,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise SavedEvidenceError(
            "saved evidence must be finite canonical JSON"
        ) from exc


def _loads(content: bytes, role: str):
    _require(type(content) is bytes, f"{role} must be exact bytes")
    try:
        value = json.loads(
            content,
            object_pairs_hook=fixed_cascade._object_without_duplicate_keys,
            parse_constant=fixed_cascade._reject_json_constant,
        )
    except (ValueError, UnicodeError) as exc:
        raise SavedEvidenceError(f"invalid {role} JSON") from exc
    _require(_json_bytes(value) == content, f"{role} must be canonical JSON")
    return value


def _number(value: object, name: str) -> float:
    _require(
        type(value) in (int, float) and math.isfinite(value),
        f"{name} must be finite",
    )
    return float(value)


def _digest(value: object, name: str) -> str:
    try:
        return fixed_cascade._lowercase_sha256(value, name)
    except fixed_cascade.FixedCascadeError as exc:
        raise SavedEvidenceError(str(exc)) from exc


def _threshold(value: object, name: str) -> float:
    try:
        return fixed_cascade._threshold(value, name)
    except fixed_cascade.FixedCascadeError as exc:
        raise SavedEvidenceError(str(exc)) from exc


def _bindings(content: bytes) -> dict:
    value = _loads(content, "bindings")
    _require(
        type(value) is dict
        and set(value)
        == {
            "schema_version",
            "partition_sha256",
            "source_csv_sha256",
            "suffix_rules_sha256",
            "artifact_hashes",
            "thresholds",
            "secondary",
            "gmm_audit",
            "replay_artifacts",
        }
        and type(value["schema_version"]) is int
        and value["schema_version"] == 3,
        "invalid bindings schema",
    )
    for name in ("partition_sha256", "source_csv_sha256", "suffix_rules_sha256"):
        _digest(value[name], name)
    artifacts = value["artifact_hashes"]
    _require(type(artifacts) is dict and bool(artifacts), "invalid primary artifacts")
    for name, digest in artifacts.items():
        _require(type(name) is str and bool(name), "invalid primary artifact name")
        _digest(digest, f"primary artifact {name}")
    thresholds = value["thresholds"]
    _require(
        type(thresholds) is dict
        and set(thresholds)
        == {
            "length_only",
            "logistic_l1",
            "transformer",
            "half_width",
            "monitor_boundary",
        },
        "invalid primary threshold inventory",
    )
    for name in ("length_only", "logistic_l1", "transformer"):
        thresholds[name] = _threshold(thresholds[name], name)
    thresholds["half_width"] = _number(thresholds["half_width"], "half_width")
    thresholds["monitor_boundary"] = _number(
        thresholds["monitor_boundary"], "monitor_boundary"
    )
    _require(thresholds["half_width"] >= 0, "half_width must be nonnegative")
    secondary = value["secondary"]
    _require(
        type(secondary) is dict
        and set(secondary)
        == {
            "accepted_report_sha256",
            "device_type",
            "stage1_threshold",
            "vocabulary_sha256",
            "tabular",
            "seeds",
        },
        "invalid secondary binding schema",
    )
    reports = secondary["accepted_report_sha256"]
    _require(
        type(reports) is dict and set(reports) == {"tabular", "seeds"},
        "invalid accepted report inventory",
    )
    for role, digest in reports.items():
        _digest(digest, f"{role} accepted report")
    _require(secondary["device_type"] == "mps", "invalid secondary device")
    _digest(secondary["vocabulary_sha256"], "secondary vocabulary")
    secondary["stage1_threshold"] = _threshold(
        secondary["stage1_threshold"], "secondary stage1 threshold"
    )
    _require(
        secondary["stage1_threshold"] == thresholds["logistic_l1"],
        "secondary stage1 threshold differs from primary",
    )
    tabular = secondary["tabular"]
    _require(
        type(tabular) is list
        and len(tabular) == len(_TABULAR_NAMES)
        and all(type(item) is dict for item in tabular),
        "invalid secondary tabular inventory",
    )
    for expected_name, item in zip(_TABULAR_NAMES, tabular, strict=True):
        _require(
            set(item) == {"name", "artifact_sha256", "threshold"}
            and item["name"] == expected_name,
            "invalid secondary tabular member",
        )
        _digest(item["artifact_sha256"], f"{expected_name} artifact")
        item["threshold"] = _threshold(item["threshold"], f"{expected_name} threshold")
    seeds = secondary["seeds"]
    _require(
        type(seeds) is list
        and len(seeds) == len(_SEEDS)
        and all(type(item) is dict for item in seeds),
        "invalid secondary seed inventory",
    )
    for expected_seed, item in zip(_SEEDS, seeds, strict=True):
        _require(
            set(item)
            == {
                "seed",
                "weights_sha256",
                "transformer_threshold",
                "half_width",
                "reuses_primary",
            }
            and type(item["seed"]) is int
            and item["seed"] == expected_seed
            and item["reuses_primary"] is (expected_seed == 42),
            "invalid secondary seed member",
        )
        _digest(item["weights_sha256"], f"seed {expected_seed} weights")
        item["transformer_threshold"] = _threshold(
            item["transformer_threshold"],
            f"seed {expected_seed} transformer threshold",
        )
        item["half_width"] = _number(
            item["half_width"], f"seed {expected_seed} half_width"
        )
        _require(item["half_width"] >= 0, "seed half_width must be nonnegative")
    if "transformer-weights.npz" in artifacts:
        _require(
            seeds[0]["weights_sha256"] == artifacts["transformer-weights.npz"],
            "seed 42 weights differ from primary",
        )
    if "vocabulary.json" in artifacts:
        _require(
            secondary["vocabulary_sha256"] == artifacts["vocabulary.json"],
            "secondary vocabulary differs from primary",
        )
    audit = value["gmm_audit"]
    _require(
        type(audit) is dict
        and set(audit) == {"alert_count", "window_count"}
        and all(type(count) is int for count in audit.values())
        and 0 <= audit["alert_count"] <= audit["window_count"]
        and audit["window_count"] > 0,
        "invalid authenticated audit counts",
    )
    _validate_binding_core(value)
    return value


def _validate_binding_core(value: dict) -> None:
    """Compare the raw frozen core without assuming an internal source envelope."""
    _require(type(value) is dict, "invalid saved model binding")
    core = {name: value.get(name) for name in _EXPECTED_BINDING_CORE}
    _require(
        fixed_cascade._matches_exactly(core, _EXPECTED_BINDING_CORE),
        "saved model binding differs from frozen evidence profile",
    )


def _replay_models(bindings):
    retained = bindings["replay_artifacts"]
    _require(
        type(retained) is dict
        and set(retained) == {"length-only.json", "logistic-l1.json", "gmm.json"},
        "invalid retained replay artifact inventory",
    )
    decoded = {}
    for name, encoded in retained.items():
        _require(type(encoded) is str, "invalid retained artifact encoding")
        content = base64.b64decode(encoded, validate=True)
        _require(
            base64.b64encode(content).decode("ascii") == encoded
            and sha256(content).hexdigest() == bindings["artifact_hashes"].get(name),
            "retained replay artifact differs from binding",
        )
        decoded[name] = content
    return (
        length_inference._load_length_only_artifact_bytes(
            decoded["length-only.json"],
            expected_sha256=bindings["artifact_hashes"]["length-only.json"],
        ),
        fixed_cascade._load_logistic_l1_artifact_bytes(
            decoded["logistic-l1.json"],
            expected_sha256=bindings["artifact_hashes"]["logistic-l1.json"],
        ),
        gmm_monitor.load_gmm_artifact_bytes(decoded["gmm.json"]),
    )


def _verify_monitor_path(rows, bindings):
    length_model, stage1_model, gmm = _replay_models(bindings)
    _verify_loaded_monitor_path(rows, length_model, stage1_model, gmm)


def _verify_loaded_monitor_path(rows, length_model, stage1_model, gmm):
    """Replay already-loaded retained models under the existing single-thread guard."""
    with threadpoolctl.threadpool_limits(limits=1):
        pools = threadpoolctl.threadpool_info()
        _require(
            bool(pools) and all(pool.get("num_threads") == 1 for pool in pools),
            "saved replay requires one numerical thread",
        )
        _verify_monitor_rows(rows, length_model, stage1_model, gmm)


def _verify_monitor_rows(rows, length_model, stage1_model, gmm):
    for row in rows:
        urls = (row["record"]["raw_url"],)
        length, length_audit = length_inference.score_length_only_authoritative(
            length_model, urls
        )
        stage1, stage1_audit = fixed_cascade.score_logistic_l1_authoritative(
            stage1_model, urls
        )
        portable = stage1_model.score_urls(urls)
        nll = gmm_monitor.score_feature_matrix(((*row["features"], portable[0]),), gmm)
        _require(
            len(length) == len(stage1) == len(portable) == len(nll) == 1
            and row["length_probability"] == float(length[0])
            and row["stage1_probability"] == float(stage1[0])
            and row["monitor_probability"] == float(portable[0])
            and row["negative_log_likelihood"] == float(nll[0])
            and row["length_scoring_audit_json"]
            == _json_bytes(length_audit).decode("ascii").rstrip("\n")
            and row["stage1_scoring_audit_json"]
            == _json_bytes(stage1_audit).decode("ascii").rstrip("\n"),
            "saved monitor path differs from retained artifact replay",
        )


def _routing(rows, bindings):
    thresholds = bindings["thresholds"]
    return policy_replay.replay_policy(
        tuple(
            policy_replay.PairedProbabilities(
                row["record"]["record_id"],
                row["stage1_probability"],
                row["transformer_probability"],
            )
            for row in rows
        ),
        tuple(
            policy_replay.MonitorScore(
                row["record"]["record_id"], row["negative_log_likelihood"]
            )
            for row in rows
        ),
        stage1_threshold=thresholds["logistic_l1"],
        transformer_threshold=thresholds["transformer"],
        half_width=thresholds["half_width"],
        monitor_boundary=thresholds["monitor_boundary"],
    )


def _audit(value: object, name: str) -> None:
    _require(type(value) is str and bool(value), f"invalid {name}")
    try:
        parsed = json.loads(
            value,
            object_pairs_hook=fixed_cascade._object_without_duplicate_keys,
            parse_constant=fixed_cascade._reject_json_constant,
        )
        canonical = json.dumps(
            parsed,
            allow_nan=False,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (ValueError, TypeError, UnicodeError) as exc:
        raise SavedEvidenceError(f"invalid {name}") from exc
    _require(canonical == value, f"noncanonical {name}")


def _validate_score_row(row: dict, bindings: dict) -> None:
    """Check saved score relations without imposing internal record or label policy."""
    try:
        validate_score_values(row, bindings)
        _audit(row["length_scoring_audit_json"], "length audit")
        _audit(row["stage1_scoring_audit_json"], "stage1 audit")
        _require(
            fixed_cascade._matches_exactly(
                row["inference_counts"], asdict(InferenceCounts(1, 1, 1, 0))
            ),
            "saved row inference counts differ",
        )
    except Exception:
        raise SavedEvidenceError("invalid saved score row") from None


def _internal_record(row: dict, source_hash: str) -> ManifestRecord:
    record_value = row["record"]
    _require(
        type(record_value) is dict and set(record_value) == _RECORD_FIELDS,
        "invalid record schema",
    )
    record = ManifestRecord(**record_value)
    try:
        observed_source = evaluation_manifest._validate_record(record)
    except (TypeError, ValueError, UnicodeError) as exc:
        raise SavedEvidenceError("invalid saved record") from exc
    _require(observed_source == source_hash, "saved record source differs")
    return record


def _parse_rows(content: bytes, bindings: dict) -> tuple[dict, ...]:
    _require(type(content) is bytes and bool(content), "predictions must be bytes")
    rows = []
    previous_id = ""
    for line in content.splitlines(keepends=True):
        row = _loads(line, "prediction row")
        _require(type(row) is dict and set(row) == _ROW_FIELDS, "invalid row schema")
        record = _internal_record(row, bindings["source_csv_sha256"])
        _require(record.record_id > previous_id, "saved record order differs")
        previous_id = record.record_id
        _validate_score_row(row, bindings)
        rows.append(row)
    _require(bool(rows), "predictions must contain rows")
    partition = b"".join(_json_bytes(row["record"]) for row in rows)
    _require(
        sha256(partition).hexdigest() == bindings["partition_sha256"],
        "saved records differ from partition binding",
    )
    return tuple(rows)


def _manifests(records: tuple[ManifestRecord, ...]) -> dict[int, ManifestOutcome]:
    outcomes = {}
    for prevalence in _PREVALENCES:
        try:
            manifest = evaluation_manifest.build_manifest(
                records, prevalence_basis_points=prevalence
            )
        except evaluation_manifest.InsufficientStratum as exc:
            outcomes[prevalence] = ManifestOutcome(
                "insufficient_capacity",
                insufficient_label=exc.label,
                required=exc.required,
                available=exc.available,
            )
        else:
            outcomes[prevalence] = ManifestOutcome("prepared", manifest=manifest)
    return outcomes


def _metric(records, rows, scores, decisions):
    return asdict(
        secondary_metrics.secondary_metrics(
            records,
            tuple(
                secondary_metrics.ScorePrediction(row["record"]["record_id"], score)
                for row, score in zip(rows, scores, strict=True)
            ),
            tuple(
                BinaryPrediction(row["record"]["record_id"], decision)
                for row, decision in zip(rows, decisions, strict=True)
            ),
        )
    )


def _secondary(rows, population):
    records = population.records
    metrics = {}
    primary_columns = {
        "length_only": ("length_probability", "length_decision"),
        "logistic_l1": ("stage1_probability", "stage1_decision"),
        "transformer": ("transformer_probability", "transformer_decision"),
        "cascade": ("cascade_probability", "cascade_decision"),
    }
    for name, (score_name, decision_name) in primary_columns.items():
        metrics[name] = _metric(
            records,
            rows,
            (row[score_name] for row in rows),
            (row[decision_name] for row in rows),
        )
    for index, name in enumerate(_TABULAR_NAMES):
        metrics[f"tabular.{name}"] = _metric(
            records,
            rows,
            (row["secondary_tabular"][index]["probability"] for row in rows),
            (row["secondary_tabular"][index]["decision"] for row in rows),
        )
    for index, seed in enumerate(_SEEDS):
        for role in ("transformer", "cascade"):
            metrics[f"seed_{seed}.{role}"] = _metric(
                records,
                rows,
                (row["secondary_seeds"][index][f"{role}_probability"] for row in rows),
                (row["secondary_seeds"][index][f"{role}_decision"] for row in rows),
            )
    positives = tuple(row for row in records if row.label == 1)
    positive_ids = {row.record_id for row in positives}
    predictions = {
        model: tuple(row for row in values if row.record_id in positive_ids)
        for model, values in population.predictions.items()
    }
    contrasts = {
        "internal_logistic_minus_length": secondary_metrics.exact_mcnemar(
            positives, predictions["logistic_l1"], predictions["length_only"]
        ),
        "internal_cascade_minus_logistic": secondary_metrics.exact_mcnemar(
            positives, predictions["cascade"], predictions["logistic_l1"]
        ),
        "external_gold_logistic_minus_length": None,
        "external_gold_cascade_minus_logistic": None,
    }
    return {
        "schema_version": 2,
        "metrics": metrics,
        "mcnemar": {
            name: asdict(value) if value is not None else None
            for name, value in contrasts.items()
        },
        "holm": asdict(secondary_metrics.holm_ablation_family(contrasts)),
    }


def reconstruct_internal_evidence(
    predictions: bytes, manifests: bytes, bindings: bytes, routing: bytes
) -> ReconstructedInternalEvidence:
    """Recompute internal evidence without opening source or model paths."""
    return reconstruct_internal_evidence_and_population(
        predictions, manifests, bindings, routing
    )[0]


def _validated_internal_inputs(predictions, manifests, bindings, routing):
    bound = _bindings(bindings)
    rows = _parse_rows(predictions, bound)
    _verify_monitor_path(rows, bound)
    _require(
        _json_bytes(_loads(routing, "routing"))
        == _json_bytes(asdict(_routing(rows, bound))),
        "saved routing differs from reconstructed routing",
    )
    records = tuple(ManifestRecord(**row["record"]) for row in rows)
    outcomes = _manifests(records)
    saved_manifests = _loads(manifests, "manifests")
    _require(
        saved_manifests == {str(key): asdict(value) for key, value in outcomes.items()},
        "saved manifests differ from reconstructed manifests",
    )
    return bound, rows, records, outcomes


def _internal_population(rows, records):
    evaluation_records = tuple(
        EvaluationRecord(
            record.record_id, record.registrable_domain, record.is_phishing
        )
        for record in records
    )
    columns = {
        "length_only": "length_decision",
        "logistic_l1": "stage1_decision",
        "transformer": "transformer_decision",
        "cascade": "cascade_decision",
    }
    return SavedPopulation(
        evaluation_records,
        {
            model: tuple(
                BinaryPrediction(row["record"]["record_id"], row[column])
                for row in rows
            )
            for model, column in columns.items()
        },
    )


def _internal_result(rows, records, outcomes, primary, population):
    count = len(rows)
    secondary_counts = SecondaryInferenceCounts(
        tuple((name, count) for name in _TABULAR_NAMES),
        tuple((seed, 0 if seed == 42 else count) for seed in _SEEDS),
        count,
    )
    return ReconstructedInternalEvidence(
        count,
        len({record.registrable_domain for record in records}),
        {
            "0": sum(record.is_phishing == 0 for record in records),
            "1": sum(record.is_phishing == 1 for record in records),
        },
        InferenceCounts(count, count, count, 0),
        secondary_counts,
        outcomes,
        primary,
        _secondary(rows, population),
    )


def reconstruct_internal_evidence_and_population(
    predictions: bytes, manifests: bytes, bindings: bytes, routing: bytes
) -> tuple[ReconstructedInternalEvidence, SavedPopulation]:
    """Return summaries and their population after the full saved-byte verification."""
    try:
        bound, rows, records, outcomes = _validated_internal_inputs(
            predictions, manifests, bindings, routing
        )
        population = _internal_population(rows, records)
        primary = hypothesis_evaluation.evaluate_primary(
            populations={"internal": population},
            audit_windows=WindowCounts(
                bound["gmm_audit"]["alert_count"], bound["gmm_audit"]["window_count"]
            ),
        )
        return _internal_result(
            rows, records, outcomes, primary, population
        ), population
    except SavedEvidenceError:
        raise
    except (TypeError, ValueError, KeyError, OverflowError) as exc:
        raise SavedEvidenceError("saved evidence reconstruction failed") from exc
