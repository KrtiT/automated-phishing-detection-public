"""Compose one in-memory prepared partition into complete saved evidence.

No path, file opener, model loader, fitting method, or protected-data command is
provided here. Supplied source/PSL pins are caller claims, not an authorization
or an official source binding. A future reviewed runner must reserve its attempt,
read the source once, authenticate those pins and the full execution freeze, and
publish the returned payloads without replacement. Tests use synthetic bytes.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace
from functools import partial
from hashlib import sha256

from . import (
    evaluation_manifest,
    fixed_cascade,
    gmm_monitor,
    hypothesis_evaluation,
    length_inference,
    phiusiil,
    policy_replay,
    protocol_preflight,
)
from ._checkpoint_codec import canonical_bytes, secondary_binding
from ._internal_producer_bindings import binding_bytes
from ._internal_producer_checkpoints import (
    COLUMN_NAMES,
    completed_column_bytes,
    primary_completion_bytes,
    validate_completed_scoring,
    validated_column_index,
)
from ._internal_producer_progress import InternalProgress
from ._internal_producer_rows import validate_primary_values
from .bound_runtime import BoundEvaluationSession, BoundSession
from .bound_secondary import (
    BoundSecondary,
    SecondaryInferenceCounts,
    SecondaryScoring,
    SecondarySeedScore,
    SecondaryTabularScore,
    score_bound_secondary,
)
from .evaluation_manifest import ManifestRecord, ReplayManifest
from .hypothesis_evaluation import PrimaryEvaluation, SavedPopulation, WindowCounts
from .paired_evaluation import BinaryPrediction, EvaluationRecord
from .primary_scores import PrimaryURLScores
from .selective_inference import InferenceCounts, RequestScores
from .url_features import extract_url_features

_RECORD_FIELDS = frozenset(ManifestRecord.__dataclass_fields__)
_PREVALENCES = (10, 100, 500)
_ZERO_COUNTS = InferenceCounts(0, 0, 0, 0)


class EvaluationProducerError(ValueError):
    """Source claims or composed evidence fail their structural checks."""


@dataclass(frozen=True)
class PreparedInternal:
    records: tuple[ManifestRecord, ...]
    partition_sha256: str
    source_csv_sha256: str
    suffix_rules_sha256: str
    domain_count: int
    class_counts: tuple[int, int]
    _parsed_state: tuple = field(repr=False, compare=False)


@dataclass(frozen=True)
class ScoredInternalRow:
    record: ManifestRecord
    features: tuple[float, ...]
    length_probability: float
    stage1_probability: float
    transformer_probability: float
    cascade_probability: float
    length_decision: int
    stage1_decision: int
    transformer_decision: int
    cascade_decision: int
    band_selected: bool
    monitor_probability: float
    negative_log_likelihood: float
    length_scoring_audit_json: str
    stage1_scoring_audit_json: str
    inference_counts: InferenceCounts
    secondary_tabular: tuple[SecondaryTabularScore, ...] = ()
    secondary_seeds: tuple[SecondarySeedScore, ...] = ()


@dataclass(frozen=True)
class ManifestOutcome:
    status: str
    manifest: ReplayManifest | None = None
    insufficient_label: int | None = None
    required: int | None = None
    available: int | None = None


@dataclass(frozen=True)
class ProducedInternal:
    rows: tuple[ScoredInternalRow, ...]
    population: SavedPopulation
    manifests: dict[int, ManifestOutcome]
    primary: PrimaryEvaluation
    inference_counts: InferenceCounts
    secondary_inference_counts: SecondaryInferenceCounts
    private_outputs: dict[str, bytes]
    public_summary: dict


def _json_bytes(value: object) -> bytes:
    try:
        return canonical_bytes(value)
    except (TypeError, ValueError, UnicodeError) as exc:
        raise EvaluationProducerError("evidence must be finite JSON") from exc


def _hash(value: object, field_name: str) -> str:
    try:
        return phiusiil._validate_sha256(value, field_name)
    except phiusiil.PreparationError as exc:
        raise EvaluationProducerError(str(exc)) from exc


def _count(value: object, field_name: str) -> int:
    if type(value) is not int or value < 0:
        raise EvaluationProducerError(f"{field_name} must be a nonnegative integer")
    return value


def _parse_records(content: bytes) -> tuple[ManifestRecord, ...]:
    records = []
    for line in content.splitlines(keepends=True):
        try:
            row = json.loads(
                line.decode("utf-8"),
                object_pairs_hook=fixed_cascade._object_without_duplicate_keys,
                parse_constant=fixed_cascade._reject_json_constant,
            )
        except (ValueError, UnicodeError) as exc:
            raise EvaluationProducerError("invalid partition JSONL") from exc
        if type(row) is not dict or set(row) != _RECORD_FIELDS:
            raise EvaluationProducerError("partition fields differ from preparation")
        if _json_bytes(row) != line:
            raise EvaluationProducerError("partition must use canonical JSONL")
        records.append(ManifestRecord(**row))
    if not records:
        raise EvaluationProducerError("partition must not be empty")
    return tuple(records)


def _validate_records(records, source_csv_sha256, suffix_rules):
    previous_id = ""
    canonical_hashes = set()
    domains = set()
    classes = [0, 0]
    for record in records:
        try:
            source_hash = evaluation_manifest._validate_record(record)
            domain = protocol_preflight.registrable_domain_for_url(
                record.raw_url, suffix_rules
            )
        except (TypeError, ValueError, UnicodeError) as exc:
            raise EvaluationProducerError("invalid prepared row metadata") from exc
        if source_hash != source_csv_sha256:
            raise EvaluationProducerError("record ID differs from source CSV claim")
        if record.record_id <= previous_id:
            raise EvaluationProducerError("record IDs must be strictly increasing")
        if record.canonical_url_sha256 in canonical_hashes:
            raise EvaluationProducerError("duplicate canonical URL in partition")
        if domain != record.registrable_domain:
            raise EvaluationProducerError(
                "registrable domain differs from supplied PSL"
            )
        previous_id = record.record_id
        canonical_hashes.add(record.canonical_url_sha256)
        domains.add(domain)
        classes[record.is_phishing] += 1
    return len(domains), tuple(classes)


def parse_internal_partition(
    content: bytes,
    *,
    expected_sha256: str,
    source_csv_sha256: str,
    suffix_rules: protocol_preflight.SuffixRules,
    suffix_rules_sha256: str,
    expected_row_count: int,
    expected_domain_count: int,
    expected_class_counts: dict[str, int],
) -> PreparedInternal:
    """Authenticate supplied bytes once, then validate without dropping or sorting.

    The supplied PSL hash identifies the caller's claim; a parsed SuffixRules
    value alone cannot prove which original PSL bytes produced it. The future
    official wrapper must authenticate that relationship before calling here.
    """
    if type(content) is not bytes:
        raise EvaluationProducerError("partition content must be exact bytes")
    _hash(expected_sha256, "expected_sha256")
    _hash(source_csv_sha256, "source_csv_sha256")
    _hash(suffix_rules_sha256, "suffix_rules_sha256")
    observed_hash = sha256(content).hexdigest()
    if observed_hash != expected_sha256:
        raise EvaluationProducerError("partition SHA-256 differs from supplied pin")
    if type(suffix_rules) is not protocol_preflight.SuffixRules:
        raise EvaluationProducerError("suffix_rules must be parsed SuffixRules")
    _count(expected_row_count, "expected_row_count")
    _count(expected_domain_count, "expected_domain_count")
    if type(expected_class_counts) is not dict or set(expected_class_counts) != {
        "0",
        "1",
    }:
        raise EvaluationProducerError("expected class counts must contain 0 and 1")
    declared_classes = tuple(
        _count(expected_class_counts[key], "expected class count") for key in ("0", "1")
    )
    records = _parse_records(content)
    domain_count, classes = _validate_records(records, source_csv_sha256, suffix_rules)
    if (
        len(records) != expected_row_count
        or domain_count != expected_domain_count
        or classes != declared_classes
        or 0 in classes
    ):
        raise EvaluationProducerError(
            "partition counts differ or lack a required class"
        )
    state = (
        records,
        observed_hash,
        source_csv_sha256,
        suffix_rules_sha256,
        domain_count,
        classes,
    )
    return PreparedInternal(*state, _parsed_state=state)


def _probability(value: object, name: str) -> float:
    try:
        result = fixed_cascade._finite_number(value, name)
    except fixed_cascade.FixedCascadeError as exc:
        raise EvaluationProducerError(str(exc)) from exc
    if not 0.0 <= result <= 1.0:
        raise EvaluationProducerError(f"{name} must be a probability")
    return result


def _singleton(values, name: str) -> float:
    if len(values) != 1:
        raise EvaluationProducerError(f"{name} must return one singleton score")
    return _probability(values[0], name)


def _expected_counts(scorer, count: int) -> InferenceCounts:
    observed = scorer.counts
    expected = InferenceCounts(count, count, count, 0)
    if type(observed) is not InferenceCounts or observed != expected:
        raise EvaluationProducerError(
            "physical inference counters differ from row count"
        )
    if any(type(value) is not int for value in asdict(observed).values()):
        raise EvaluationProducerError(
            "physical inference counts must be exact integers"
        )
    return observed


def _thresholds(session: BoundSession) -> dict:
    model = session.models.cascade
    length = session.models.length_only.validation_threshold_record
    if length.get("status") != "selected":
        raise EvaluationProducerError("length-only operating point is not selected")
    try:
        values = {
            "length_only": fixed_cascade._threshold(
                length["threshold"], "length threshold"
            ),
            "logistic_l1": fixed_cascade._threshold(
                model.stage1_threshold, "stage1 threshold"
            ),
            "transformer": fixed_cascade._threshold(
                model.transformer_threshold, "transformer threshold"
            ),
            "half_width": fixed_cascade._finite_number(model.half_width, "half_width"),
            "monitor_boundary": fixed_cascade._finite_number(
                session.models.monitor_boundary, "monitor_boundary"
            ),
        }
    except (KeyError, fixed_cascade.FixedCascadeError) as exc:
        raise EvaluationProducerError(
            "invalid carried-forward operating point"
        ) from exc
    if values["half_width"] < 0:
        raise EvaluationProducerError("half_width must be nonnegative")
    return values


def _detector_scores(raw_url: str, session: BoundSession, thresholds: dict) -> tuple:
    length_values, length_audit = length_inference.score_length_only_authoritative(
        session.models.length_only, (raw_url,)
    )
    length = _singleton(length_values, "length probability")
    scores = session.scorer.score_all(raw_url)
    if type(scores) is not RequestScores:
        raise EvaluationProducerError("scorer must return typed RequestScores")
    stage1 = _probability(scores.stage1_probability, "stage1 probability")
    transformer = _probability(
        scores.transformer_probability, "transformer probability"
    )
    fixed = fixed_cascade.score_fixed_cascade(
        (stage1,),
        (transformer,),
        stage1_threshold=thresholds["logistic_l1"],
        transformer_threshold=thresholds["transformer"],
        half_width=thresholds["half_width"],
    )
    band, decision = fixed.transformer_invoked[0], fixed.decisions[0]
    _validate_score_flags(scores, band, decision)
    return (
        length,
        stage1,
        transformer,
        band,
        decision,
        length_audit,
        scores.stage1_scoring_audit,
    )


def _validate_score_flags(scores: RequestScores, band: bool, decision: int) -> None:
    flags = (
        scores.band_selected,
        scores.drift_override,
        scores.logical_stage2_selected,
        scores.transformer_evaluated,
    )
    if (
        any(type(flag) is not bool for flag in flags)
        or flags != (band, False, band, True)
        or type(scores.fixed_decision) is not int
        or type(scores.decision) is not int
        or scores.fixed_decision != decision
        or scores.decision != decision
    ):
        raise EvaluationProducerError("scorer decisions differ from frozen cascade")


def _monitor_scores(raw_url: str, session: BoundSession) -> tuple:
    try:
        features = tuple(
            float(value)
            for value in gmm_monitor._finite_array(
                extract_url_features(raw_url), (25,), "structural features"
            )
        )
    except gmm_monitor.GMMMonitorError as exc:
        raise EvaluationProducerError(str(exc)) from exc
    portable = _singleton(
        session.models.cascade.stage1_model.score_urls((raw_url,)),
        "portable monitor probability",
    )
    nll = gmm_monitor.score_feature_matrix(((*features, portable),), session.models.gmm)
    try:
        score = float(
            gmm_monitor._finite_array(nll, (1,), "negative log likelihood")[0]
        )
    except gmm_monitor.GMMMonitorError as exc:
        raise EvaluationProducerError(str(exc)) from exc
    return features, portable, score


def score_primary_url(
    raw_url: str, session: BoundSession, thresholds: dict, position: int
) -> PrimaryURLScores:
    """Score one raw URL without constructing or consulting a scientific label."""
    length, stage1, transformer, band, decision, length_audit, stage1_audit = (
        _detector_scores(raw_url, session, thresholds)
    )
    _expected_counts(session.scorer, position)
    features, portable, score = _monitor_scores(raw_url, session)
    return PrimaryURLScores(
        features,
        length,
        stage1,
        transformer,
        transformer if band else stage1,
        int(length >= thresholds["length_only"]),
        int(stage1 >= thresholds["logistic_l1"]),
        int(transformer >= thresholds["transformer"]),
        decision,
        band,
        portable,
        score,
        _json_bytes(length_audit).decode("ascii").rstrip("\n"),
        _json_bytes(stage1_audit).decode("ascii").rstrip("\n"),
        InferenceCounts(1, 1, 1, 0),
    )


def _score_row(record: ManifestRecord, session: BoundSession, thresholds, position):
    scores = score_primary_url(record.raw_url, session, thresholds, position)
    return ScoredInternalRow(record, **vars(scores))


def _manifests(records):
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


def _manifest_summary(outcome):
    if outcome.manifest is None:
        return asdict(outcome)
    manifest = outcome.manifest
    return {
        "status": outcome.status,
        "sha256": manifest.sha256,
        "measured_count": len(manifest.records),
        "warmup_count": len(manifest.warmup_records),
        "prevalence_basis_points": manifest.prevalence_basis_points,
    }


def _secondary_binding(bound: BoundSecondary, stage1_threshold: float) -> dict:
    try:
        return secondary_binding(bound, stage1_threshold)
    except ValueError as exc:
        raise EvaluationProducerError("secondary binding differs from primary") from exc


def _validate_internal_inputs(prepared, session, state):
    if (
        type(prepared) is not PreparedInternal
        or type(session) is not BoundEvaluationSession
        or type(session.primary) is not BoundSession
    ):
        raise EvaluationProducerError(
            "use typed prepared input and BoundEvaluationSession"
        )
    snapshot = (
        prepared.records,
        prepared.partition_sha256,
        prepared.source_csv_sha256,
        prepared.suffix_rules_sha256,
        prepared.domain_count,
        prepared.class_counts,
    )
    if snapshot != prepared._parsed_state:
        raise EvaluationProducerError("prepared state differs from parsed snapshot")
    primary_session = session.primary
    primary_session.scorer._require_owner()
    if primary_session.scorer.counts != _ZERO_COUNTS:
        raise EvaluationProducerError("producer requires a fresh inference session")
    _expected_counts(primary_session.scorer, 0)
    state.record_ids = tuple(record.record_id for record in prepared.records)
    state.observe_counts(primary_session.scorer)


def _initial_internal_checkpoints(prepared, session, state, retain):
    thresholds = _thresholds(session.primary)
    content = binding_bytes(prepared, session, thresholds)
    state.stage = "binding_retention"
    state.store("bindings.json", content, retain)
    state.stage = "manifest_preparation"
    manifests = _manifests(prepared.records)
    state.stage = "manifest_retention"
    state.store(
        "manifests.json",
        _json_bytes({str(bp): asdict(value) for bp, value in manifests.items()}),
        retain,
    )
    return thresholds, manifests


def _retain_primary(prepared, session, state, retain):
    counts = _expected_counts(session.scorer, len(state.rows))
    content = b"".join(_json_bytes(asdict(row)) for row in state.rows)
    receipt = primary_completion_bytes(
        state.outputs["bindings.json"],
        content,
        len(state.rows),
        counts,
        prepared.partition_sha256,
    )
    state.stage = "primary_phase_retention"
    state.store("primary-scores.jsonl", content, retain)
    state.store("primary-completion.json", receipt, retain)


def _score_primary_rows(prepared, session, thresholds, state, retain):
    for position, record in enumerate(prepared.records, start=1):
        state.stage = "primary_scoring"
        state.started_primary_position = position
        row = _score_row(record, session, thresholds, position)
        _validate_completed_primary(row, record, thresholds)
        state.rows.append(row)
        state.started_primary_position = None
        state.observe_counts(session.scorer)
    _retain_primary(prepared, session, state, retain)


def _validate_completed_primary(row, record, thresholds):
    try:
        if type(row) is not ScoredInternalRow or row.record is not record:
            raise ValueError("invalid_internal_primary_type")
        validate_primary_values(row, thresholds)
    except Exception:
        raise EvaluationProducerError("invalid completed primary row") from None


def _retain_secondary_column(state, retain, binding, column):
    state.stage = "secondary_column_validation"
    position = validated_column_index(state, column)
    state.columns.append(column)
    index = len(state.columns)
    state.next_expected_member = (
        COLUMN_NAMES[index] if index < len(COLUMN_NAMES) else None
    )
    state.stage = "secondary_column_encoding"
    name, content = completed_column_bytes(state, column, binding, position)
    state.stage = "secondary_column_retention"
    state.store(name, content, retain)
    state.stage = "secondary_scoring"


def _score_secondary_rows(session, state, retain) -> SecondaryScoring:
    state.stage = "secondary_scoring"
    state.next_expected_member = COLUMN_NAMES[0]
    binding = json.loads(state.outputs["bindings.json"])["secondary"]
    secondary = score_bound_secondary(
        session.secondary,
        tuple(row.record.raw_url for row in state.rows),
        tuple(row.stage1_probability for row in state.rows),
        tuple(row.transformer_probability for row in state.rows),
        on_completed_column=partial(_retain_secondary_column, state, retain, binding),
    )
    state.stage = "secondary_completion_validation"
    validate_completed_scoring(state, secondary)
    return secondary


def _joined_internal_rows(state, secondary, retain):
    rows = tuple(
        replace(
            row,
            secondary_tabular=secondary_row.tabular,
            secondary_seeds=secondary_row.seeds,
        )
        for row, secondary_row in zip(state.rows, secondary.rows, strict=True)
    )
    state.stage = "prediction_retention"
    state.store(
        "predictions.jsonl",
        b"".join(_json_bytes(asdict(row)) for row in rows),
        retain,
    )
    return rows


def _internal_population(rows):
    decisions = {
        "length_only": "length_decision",
        "logistic_l1": "stage1_decision",
        "transformer": "transformer_decision",
        "cascade": "cascade_decision",
    }
    return SavedPopulation(
        tuple(
            EvaluationRecord(
                row.record.record_id,
                row.record.registrable_domain,
                row.record.is_phishing,
            )
            for row in rows
        ),
        {
            model: tuple(
                BinaryPrediction(row.record.record_id, getattr(row, attribute))
                for row in rows
            )
            for model, attribute in decisions.items()
        },
    )


def _internal_routing(rows, thresholds):
    return policy_replay.replay_policy(
        tuple(
            policy_replay.PairedProbabilities(
                row.record.record_id,
                row.stage1_probability,
                row.transformer_probability,
            )
            for row in rows
        ),
        tuple(
            policy_replay.MonitorScore(
                row.record.record_id, row.negative_log_likelihood
            )
            for row in rows
        ),
        stage1_threshold=thresholds["logistic_l1"],
        transformer_threshold=thresholds["transformer"],
        half_width=thresholds["half_width"],
        monitor_boundary=thresholds["monitor_boundary"],
    )


def _internal_summary(prepared, result):
    return {
        "schema_version": 3,
        "status": "internal_evidence_composed",
        "protected_evaluation_authorized": False,
        "source_binding": "caller_supplied_pins_only",
        "row_count": len(result.rows),
        "domain_count": prepared.domain_count,
        "class_counts": dict(zip(("0", "1"), prepared.class_counts)),
        "offline_inference_counts": asdict(result.inference_counts),
        "offline_secondary_inference_counts": asdict(result.secondary_inference_counts),
        "manifests": {
            str(bp): _manifest_summary(value) for bp, value in result.manifests.items()
        },
        "primary": asdict(result.primary),
        "private_sha256": {
            name: sha256(content).hexdigest()
            for name, content in result.private_outputs.items()
        },
    }


def _derived_internal(rows, thresholds, state, retain):
    state.stage = "derived"
    population = _internal_population(rows)
    audit = json.loads(state.outputs["bindings.json"])["gmm_audit"]
    primary = hypothesis_evaluation.evaluate_primary(
        populations={"internal": population},
        audit_windows=WindowCounts(audit["alert_count"], audit["window_count"]),
    )
    routing = _internal_routing(rows, thresholds)
    state.stage = "routing_retention"
    state.store("routing.json", _json_bytes(asdict(routing)), retain)
    return population, primary


def _finish_internal(
    prepared, session, state, manifests, rows, secondary, population, primary
):
    state.stage = "final_binding"
    if (
        binding_bytes(prepared, session, _thresholds(session.primary))
        != state.outputs["bindings.json"]
    ):
        raise EvaluationProducerError("internal binding changed during scoring")
    counts = _expected_counts(session.primary.scorer, len(rows))
    private = {
        name: state.outputs[name]
        for name in (
            "predictions.jsonl",
            "routing.json",
            "manifests.json",
            "bindings.json",
        )
    }
    result = ProducedInternal(
        rows, population, manifests, primary, counts, secondary.counts, private, {}
    )
    state.stage = "summary"
    summary = _internal_summary(prepared, result)
    _json_bytes(summary)
    state.status, state.stage = "complete", "complete"
    return replace(result, public_summary=summary)


def _produce_internal(prepared, session, state, retain):
    _validate_internal_inputs(prepared, session, state)
    thresholds, manifests = _initial_internal_checkpoints(
        prepared, session, state, retain
    )
    _score_primary_rows(prepared, session.primary, thresholds, state, retain)
    secondary = _score_secondary_rows(session, state, retain)
    rows = _joined_internal_rows(state, secondary, retain)
    population, primary = _derived_internal(rows, thresholds, state, retain)
    return _finish_internal(
        prepared, session, state, manifests, rows, secondary, population, primary
    )


def _begin_internal_progress(progress, retain):
    state = InternalProgress() if progress is None else progress
    if (
        type(state) is not InternalProgress
        or retain is not None
        and not callable(retain)
    ):
        raise EvaluationProducerError("invalid internal retention inputs")
    try:
        state.begin()
    except ValueError as exc:
        raise EvaluationProducerError("internal progress must be fresh") from exc
    return state


def produce_internal_evidence(
    prepared: PreparedInternal,
    session: BoundEvaluationSession,
    *,
    retain=None,
    progress=None,
) -> ProducedInternal:
    """Score once, retaining complete prefixes without authorizing or resuming an attempt."""
    state = _begin_internal_progress(progress, retain)
    try:
        return _produce_internal(prepared, session, state, retain)
    except BaseException as error:
        state.status = "failed"
        if (
            type(session) is BoundEvaluationSession
            and type(session.primary) is BoundSession
        ):
            state.observe_counts(
                session.primary.scorer,
                suppress_interruptions=not isinstance(error, Exception),
            )
        raise
