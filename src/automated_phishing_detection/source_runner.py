"""Authenticated internal file-to-evidence composition, with a closed access gate.

The current execution profile is not a complete pre-access freeze. The public
entry therefore stops before inspecting any supplied input/output path. The
private composition is exercised on temporary fixtures, not research records.
External and operational process composition remain separate work.
"""

from __future__ import annotations

import json
import os
import stat
import sys
from dataclasses import asdict, dataclass, fields
from hashlib import sha256
from pathlib import Path

from . import (
    baselines,
    evaluation_producer,
    execution_receipt,
    fixed_cascade,
    phiusiil,
    protocol_preflight,
    secondary_metrics,
    source_overlap,
)
from ._owned_process_exit import OwnedProcessExit
from ._process_support import command_hash
from .bound_models import ArtifactPaths
from .bound_runtime import open_bound_evaluation_session
from .bound_secondary import SecondaryArtifactPaths
from .execution_preflight import ExecutionBinding, bind_execution, recheck_binding
from .execution_receipt import publish_completion, record_failure, reserve_attempt
from .internal_failure import InternalFailureState, failure_kind, propagate_interruption
from .internal_process_handoff import ObservedInternalCompletion, retain_worker_failure
from .internal_scientific_checkpoints import (
    SCIENTIFIC_CHECKPOINT_PROTOCOL,
    ScientificCheckpointWriter,
    retain_failure_progress,
)
from .owned_worker import WorkerObservation, observe_worker
from .paired_evaluation import BinaryPrediction
from .source_checkpoints import retain_source_checkpoints

_SECONDARY_TABULAR_NAMES = (
    "formatting",
    "permutation_42",
    "permutation_43",
    "permutation_44",
    "permutation_45",
    "permutation_46",
    "random_forest",
)
_SECONDARY_SEEDS = (42, 43, 44, 45, 46)

_SOURCE = "data/sources.json"
_PREPARATION = "reports/phiusiil-preparation-summary.json"
_PUBLIC_FIELDS = frozenset(
    {
        "schema_version",
        "status",
        "protected_evaluation_authorized",
        "source_binding",
        "row_count",
        "domain_count",
        "class_counts",
        "offline_inference_counts",
        "offline_secondary_inference_counts",
        "manifests",
        "primary",
        "private_sha256",
    }
)


class SourceExecutionError(ValueError):
    """A symbolic execution failure; never includes private URLs or error text."""


@dataclass(frozen=True)
class InternalRunPaths:
    source_csv: Path
    suffix_rules: Path
    artifacts: ArtifactPaths
    secondary_artifacts: SecondaryArtifactPaths
    attempt: Path
    public_summary: Path


def _json(content):
    return json.loads(
        content,
        object_pairs_hook=fixed_cascade._object_without_duplicate_keys,
        parse_constant=fixed_cascade._reject_json_constant,
    )


def _file_state(value):
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
        value.st_nlink,
        value.st_mode,
    )


def _read_file_once(path: Path, *, expected_state: tuple | None = None) -> bytes:
    """Read one pinned descriptor, checking the pathname without reopening bytes."""
    try:
        path = execution_receipt._absolute_path(path)
        with execution_receipt._directory(path.parent) as parent:
            before_path = execution_receipt._entry(parent, path.name)
            descriptor = os.open(
                path.name,
                os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                dir_fd=parent.descriptor,
            )
            try:
                before = os.fstat(descriptor)
                if (
                    before_path is None
                    or not stat.S_ISREG(before.st_mode)
                    or before.st_nlink != 1
                    or _file_state(before_path) != _file_state(before)
                    or (
                        expected_state is not None
                        and _file_state(before) != expected_state
                    )
                ):
                    raise SourceExecutionError("unsafe_input_file")
                with os.fdopen(descriptor, "rb", closefd=False) as stream:
                    content = stream.read()
                after = os.fstat(descriptor)
                after_path = execution_receipt._entry(parent, path.name)
                parent.check()
                if (
                    after_path is None
                    or _file_state(before) != _file_state(after)
                    or _file_state(before) != _file_state(after_path)
                    or len(content) != before.st_size
                ):
                    raise SourceExecutionError("input_changed_during_read")
                return content
            finally:
                os.close(descriptor)
    except (OSError, execution_receipt.ExecutionReceiptError):
        raise SourceExecutionError("unsafe_input_path") from None


def _public_sources(binding):
    pins = dict(binding.source_hashes)
    contents = {}
    for relative in (_SOURCE, _PREPARATION):
        content = _read_file_once(binding.root / relative)
        if relative not in pins or sha256(content).hexdigest() != pins[relative]:
            raise SourceExecutionError("public_source_hash_mismatch")
        contents[relative] = content
    source = phiusiil._load_source_spec(contents[_SOURCE])
    summary = _json(contents[_PREPARATION])
    preparation = baselines._validate_preparation_summary(summary)
    if summary["source_spec_sha256"] != pins[_SOURCE] or not phiusiil._matches_exactly(
        summary["declared_sources"], source
    ):
        raise SourceExecutionError("public_source_chain_mismatch")
    split = preparation["splits"]["group_test"]
    if split["domain_count"] > split["row_count"]:
        raise SourceExecutionError("invalid_domain_count")
    return {
        "expected_sha256": preparation["output_hashes"]["group_test.jsonl"],
        "source_csv_sha256": preparation["source_csv_sha256"],
        "suffix_rules_sha256": source["public_suffix_list"]["sha256"],
        "expected_row_count": split["row_count"],
        "expected_domain_count": split["domain_count"],
        "expected_class_counts": dict(split["class_counts"]),
    }, contents


def _output_paths(binding, paths):
    if (
        type(paths) is not InternalRunPaths
        or type(paths.artifacts) is not ArtifactPaths
        or type(paths.secondary_artifacts) is not SecondaryArtifactPaths
    ):
        raise SourceExecutionError("invalid_run_paths")
    for path in (paths.attempt, paths.public_summary):
        absolute = execution_receipt._absolute_path(path)
        if absolute.is_relative_to(binding.root):
            raise SourceExecutionError("outputs_must_be_outside_checkout")
        with execution_receipt._directory(absolute.parent) as parent:
            execution_receipt._require_absent(parent, absolute.name)
    if paths.public_summary.absolute().is_relative_to(paths.attempt.absolute()):
        raise SourceExecutionError("public_summary_inside_attempt")


def _secondary(produced):
    population = produced.population
    columns = {
        "length_only": "length_probability",
        "logistic_l1": "stage1_probability",
        "transformer": "transformer_probability",
        "cascade": "cascade_probability",
    }
    metrics = {
        model: asdict(
            secondary_metrics.secondary_metrics(
                population.records,
                tuple(
                    secondary_metrics.ScorePrediction(
                        row.record.record_id, getattr(row, column)
                    )
                    for row in produced.rows
                ),
                population.predictions[model],
            )
        )
        for model, column in columns.items()
    }
    if any(
        tuple(value.name for value in row.secondary_tabular) != _SECONDARY_TABULAR_NAMES
        or tuple(value.seed for value in row.secondary_seeds) != _SECONDARY_SEEDS
        for row in produced.rows
    ):
        raise SourceExecutionError("secondary_row_inventory_mismatch")

    def saved_metric(scores, decisions):
        return asdict(
            secondary_metrics.secondary_metrics(
                population.records,
                tuple(
                    secondary_metrics.ScorePrediction(row.record.record_id, score)
                    for row, score in zip(produced.rows, scores, strict=True)
                ),
                tuple(
                    BinaryPrediction(row.record.record_id, decision)
                    for row, decision in zip(produced.rows, decisions, strict=True)
                ),
            )
        )

    for index, name in enumerate(_SECONDARY_TABULAR_NAMES):
        metrics[f"tabular.{name}"] = saved_metric(
            (row.secondary_tabular[index].probability for row in produced.rows),
            (row.secondary_tabular[index].decision for row in produced.rows),
        )
    for index, seed in enumerate(_SECONDARY_SEEDS):
        metrics[f"seed_{seed}.transformer"] = saved_metric(
            (
                row.secondary_seeds[index].transformer_probability
                for row in produced.rows
            ),
            (row.secondary_seeds[index].transformer_decision for row in produced.rows),
        )
        metrics[f"seed_{seed}.cascade"] = saved_metric(
            (row.secondary_seeds[index].cascade_probability for row in produced.rows),
            (row.secondary_seeds[index].cascade_decision for row in produced.rows),
        )
    positives = tuple(row for row in population.records if row.label == 1)
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
            key: asdict(value) if value is not None else None
            for key, value in contrasts.items()
        },
        "holm": asdict(secondary_metrics.holm_ablation_family(contrasts)),
    }


def _run_bound_internal(binding: ExecutionBinding, paths: InternalRunPaths) -> Path:
    """Compose the boundary on fixtures; this helper grants no research access."""
    attempt = None
    publishing = False
    stage = "public_preflight"
    progress = evaluation_producer.InternalProgress()
    failures = InternalFailureState(progress)
    identity = None
    try:
        recheck_binding(binding)
        source, source_buffers = _public_sources(binding)
        _output_paths(binding, paths)
        pins = dict(binding.source_hashes)
        identity = {
            "kind": "internal_evaluation",
            "source_interface": "original_csv_reconstruction_v1",
            "scientific_checkpoint_protocol": SCIENTIFIC_CHECKPOINT_PROTOCOL,
            "revision": binding.revision,
            "execution_contract_sha256": binding.contract_sha256,
            "source_spec_sha256": pins[_SOURCE],
            "preparation_summary_sha256": pins[_PREPARATION],
            "runtime_sha256": sha256(binding.runtime_json.encode()).hexdigest(),
            "partition_sha256": source["expected_sha256"],
            "source_csv_sha256": source["source_csv_sha256"],
            "suffix_rules_sha256": source["suffix_rules_sha256"],
        }
        stage = "reservation"
        attempt = reserve_attempt(paths.attempt, identity=identity)
        stage = "model_loading"
        with (
            open_bound_evaluation_session(
                binding, paths.artifacts, paths.secondary_artifacts
            ) as session,
            failures.capture_body(session.primary.scorer),
        ):
            stage = "suffix_rules"
            suffix_bytes = _read_file_once(paths.suffix_rules)
            if sha256(suffix_bytes).hexdigest() != source["suffix_rules_sha256"]:
                raise SourceExecutionError("suffix_hash_mismatch")
            stage = "source_csv"
            content = _read_file_once(paths.source_csv)
            stage = "source_reconstruction"
            reconstructed = source_overlap.reconstruct_source_overlap(
                content,
                suffix_bytes,
                source_buffers[_SOURCE],
                source_buffers[_PREPARATION],
                pins=source_overlap.SourceOverlapPins(
                    source["source_csv_sha256"],
                    source["suffix_rules_sha256"],
                    pins[_SOURCE],
                    pins[_PREPARATION],
                ),
            )
            stage = "source_checkpoints"
            checkpoint_hashes = retain_source_checkpoints(
                attempt, identity, reconstructed
            )
            failures.source_checkpoint_sha256 = checkpoint_hashes
            stage = "partition"
            rules = protocol_preflight.parse_suffix_rules(suffix_bytes.decode("utf-8"))
            prepared = evaluation_producer.parse_internal_partition(
                reconstructed.group_test_bytes,
                suffix_rules=rules,
                **source,
            )
            failures.writer = ScientificCheckpointWriter(
                attempt,
                identity=identity,
                source_checkpoint_sha256=checkpoint_hashes,
                record_ids=tuple(record.record_id for record in prepared.records),
            )
            stage = "scoring"
            produced = evaluation_producer.produce_internal_evidence(
                prepared, session, retain=failures.writer, progress=progress
            )
            secondary = _secondary(produced)
            failures.writer(
                "secondary.json", evaluation_producer._json_bytes(secondary)
            )
            failures.writer.complete(
                inference_counts=produced.inference_counts,
                secondary_inference_counts=produced.secondary_inference_counts,
            )
        failures.session_closed = True
        if failures.original_error is not None:
            raise failures.original_error
        stage = "final_binding"
        recheck_binding(binding)
        stage = "summary"
        if set(produced.public_summary) != _PUBLIC_FIELDS or set(
            produced.private_outputs
        ) != {"predictions.jsonl", "manifests.json", "bindings.json", "routing.json"}:
            raise SourceExecutionError("unexpected_public_fields")
        private = dict(produced.private_outputs)
        private["secondary.json"] = evaluation_producer._json_bytes(secondary)
        public = {
            "schema_version": 4,
            "source_reconstruction": reconstructed.public_summary,
            "checkpoint_sha256": checkpoint_hashes,
            "row_count": len(produced.rows),
            "domain_count": prepared.domain_count,
            "class_counts": dict(zip(("0", "1"), prepared.class_counts)),
            "offline_inference_counts": asdict(produced.inference_counts),
            "offline_secondary_inference_counts": asdict(
                produced.secondary_inference_counts
            ),
            "manifests": {
                str(bp): evaluation_producer._manifest_summary(value)
                for bp, value in produced.manifests.items()
            },
            "primary": asdict(produced.primary),
        }
        public.update(
            {
                "status": "internal_evidence_published",
                "source_binding": "authenticated_public_preparation",
                "protected_evaluation_authorized": binding.protected_evaluation_ready,
                "execution": {
                    **identity,
                    "reservation_sha256": attempt.reservation_sha256,
                },
                "secondary": secondary,
                "private_sha256": {
                    name: sha256(value).hexdigest() for name, value in private.items()
                },
            }
        )
        public = _json(evaluation_producer._json_bytes(public))
        stage = "publication"
        publishing = True
        return publish_completion(
            attempt,
            private_outputs=private,
            public_summary=public,
            public_path=paths.public_summary,
        )
    except BaseException as exc:
        selected = failures.selected_error(exc)
        private_progress = None
        persistence_failed = False
        if attempt is not None and not publishing:
            try:
                private_progress = failures.snapshot(attempt, identity, stage, exc)
                retain_failure_progress(attempt, private_progress)
            except BaseException as persistence_error:
                persistence_failed = True
                if isinstance(selected, Exception) and not isinstance(
                    persistence_error, Exception
                ):
                    selected = persistence_error
            try:
                record_failure(attempt, stage=stage, error_type=failure_kind(selected))
            except BaseException as persistence_error:
                persistence_failed = True
                if isinstance(selected, Exception) and not isinstance(
                    persistence_error, Exception
                ):
                    selected = persistence_error
        if not isinstance(selected, Exception):
            propagate_interruption(selected, private_progress)
        symbol = (
            "failure_record_incomplete" if persistence_failed else "execution_failed"
        )
        raise SourceExecutionError(f"{stage}: {symbol}") from None


def run_internal_evaluation(
    root: Path,
    *,
    expected_revision: str,
    expected_contract_sha256: str,
    paths: InternalRunPaths,
) -> Path:
    """Require a complete frozen profile before any supplied-path inspection.

    The current profile always returns False. There is no override parameter;
    complete source/process/output coverage requires a separately reviewed
    execution-profile change before this command can process records.
    """
    binding = bind_execution(
        root,
        expected_revision=expected_revision,
        expected_contract_sha256=expected_contract_sha256,
    )
    if not binding.protected_evaluation_ready:
        raise SourceExecutionError("pre_access_freeze_incomplete")
    return _run_bound_internal(binding, paths)


def run_internal_process(
    root: Path,
    *,
    expected_revision: str,
    expected_contract_sha256: str,
    paths: InternalRunPaths,
) -> dict:
    """Return public evidence from the same owned observation and verified snapshot."""
    return run_internal_process_with_evidence(
        root,
        expected_revision=expected_revision,
        expected_contract_sha256=expected_contract_sha256,
        paths=paths,
    ).public_summary


def run_internal_process_with_evidence(
    root: Path,
    *,
    expected_revision: str,
    expected_contract_sha256: str,
    paths: InternalRunPaths,
) -> ObservedInternalCompletion:
    """Keep owned exit and once-read evidence together in the observing parent."""
    binding = bind_execution(
        root,
        expected_revision=expected_revision,
        expected_contract_sha256=expected_contract_sha256,
    )
    if not binding.protected_evaluation_ready:
        raise SourceExecutionError("pre_access_freeze_incomplete")
    return _run_observed_internal(binding, paths)


def _worker_options(paths):
    return (
        ("source-csv", paths.source_csv),
        ("suffix-rules", paths.suffix_rules),
        *(
            (member.name.replace("_", "-"), getattr(group, member.name))
            for group in (paths.artifacts, paths.secondary_artifacts)
            for member in fields(group)
        ),
        ("attempt", paths.attempt),
        ("public-summary", paths.public_summary),
    )


def _worker_command(binding, paths):
    if (
        type(paths) is not InternalRunPaths
        or type(paths.artifacts) is not ArtifactPaths
        or type(paths.secondary_artifacts) is not SecondaryArtifactPaths
    ):
        raise SourceExecutionError("invalid_run_paths")
    return (
        sys.executable,
        str(binding.root / "scripts/run_internal_evaluation.py"),
        "--worker",
        "--repo-root",
        str(binding.root),
        "--expected-revision",
        binding.revision,
        "--expected-contract-sha256",
        binding.contract_sha256,
        *(
            argument
            for name, path in _worker_options(paths)
            for argument in (f"--{name}", str(path))
        ),
    )


def _require_successful_worker(observed, command):
    if (
        type(observed) is not WorkerObservation
        or type(observed.exit) is not OwnedProcessExit
    ):
        raise SourceExecutionError("invalid_worker_observation")
    if observed.command_sha256 != command_hash(command):
        raise SourceExecutionError("worker_command_mismatch")
    if type(observed.exit.pid) is not int or observed.exit.pid <= 0:
        raise SourceExecutionError("invalid_worker_pid")
    if observed.exit.exit_observed is not True:
        raise SourceExecutionError("worker_exit_unobserved")
    if type(observed.exit.exit_code) is not int or observed.exit.exit_code != 0:
        raise SourceExecutionError("worker_exit_not_successful")
    for digest in (observed.stdout_sha256, observed.stderr_sha256):
        if (
            type(digest) is not str
            or execution_receipt._SHA256.fullmatch(digest) is None
        ):
            raise SourceExecutionError("invalid_worker_diagnostic_hash")


def _run_observed_internal(binding, paths):
    from .source_completion import verify_internal_completion_snapshot

    command = _worker_command(binding, paths)
    stage, observed = "worker_acceptance", None
    try:
        observed = observe_worker(command)
        _require_successful_worker(observed, command)
        stage = "completion_verification"
        snapshot = verify_internal_completion_snapshot(
            binding, paths, producer_exit_code=observed.exit.exit_code
        )
        return ObservedInternalCompletion(observed, snapshot)
    except BaseException as error:
        if observed is not None:
            retain_worker_failure(error, observed, binding, stage)
        raise
