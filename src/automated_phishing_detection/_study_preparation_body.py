"""Ordered source preparation; no models, predictions or scientific publication."""

from contextlib import contextmanager
from hashlib import sha256

from . import execution_receipt as receipt
from . import source_checkpoints, source_runner, study_preparation_inputs
from ._checkpoint_codec import canonical_bytes
from ._exception_cleanup import CleanupStack
from ._external_records_validation import context
from ._external_source_profile import resolve_external_source_profile
from ._process_support import _defer_interrupt
from ._study_preparation_records import StudyPreparationError, StudyPreparationPaths
from .source_overlap import SourceOverlapPins
from .study_feasibility import assess_preparation_feasibility

SOURCE = "data/sources.json"
PREPARATION = "reports/phiusiil-preparation-summary.json"


@contextmanager
def deferred_io():
    """Deliver SIGINT after one borrowed I/O operation has released its resources."""
    with CleanupStack() as cleanup:
        cleanup.enter_context(_defer_interrupt())
        yield


def _read_once(path):
    with deferred_io():
        return source_runner._read_file_once(path)


def _paths(binding, paths):
    if type(paths) is not StudyPreparationPaths:
        raise StudyPreparationError("invalid_preparation_paths")
    selected = tuple(
        receipt._absolute_path(value)
        for value in (
            paths.source_csv,
            paths.suffix_rules,
            paths.archive,
            paths.attempt,
        )
    )
    if len(set(selected)) != 4 or any(
        value.is_relative_to(selected[-1]) for value in selected[:-1]
    ):
        raise StudyPreparationError("invalid_preparation_paths")
    if selected[-1].is_relative_to(binding.root):
        raise StudyPreparationError("outputs_must_be_outside_checkout")
    with deferred_io(), receipt._directory(selected[-1].parent) as parent:
        receipt._require_absent(parent, selected[-1].name)


def preflight(state):
    with deferred_io():
        state.profile = resolve_external_source_profile(state.binding)
        projection, pins = context(state.binding, state.profile)
        state.source, state.source_buffers = source_runner._public_sources(
            state.binding
        )
    if state.source["suffix_rules_sha256"] != state.profile.suffix_rules_sha256:
        raise StudyPreparationError("preparation_suffix_pin_mismatch")
    _paths(state.binding, state.paths)
    state.identity = {
        "kind": "study_preparation_only",
        "protocol": "study-preparation-v1",
        **projection["execution"],
        "source_profile_sha256": state.profile.profile_sha256,
        "preparation_summary_sha256": pins[PREPARATION],
        "source_csv_sha256": state.source["source_csv_sha256"],
        "suffix_rules_sha256": state.source["suffix_rules_sha256"],
        "partition_sha256": state.source["expected_sha256"],
        "archive_sha256": state.profile.archive_pins.archive_sha256,
        "archive_size_bytes": state.profile.archive_pins.archive_size_bytes,
    }
    _reserve(state)


def _reserve(state):
    state.stage = "reservation"
    with deferred_io():
        state.attempt = receipt.reserve_attempt(
            state.paths.attempt, identity=state.identity
        )


def retain(state, name, content):
    state.outputs[name] = content
    state.writer.append(name, content)


def prepare_internal(state):
    state.stage = "suffix_rules"
    suffix = _read_once(state.paths.suffix_rules)
    if sha256(suffix).hexdigest() != state.source["suffix_rules_sha256"]:
        raise StudyPreparationError("preparation_suffix_hash_mismatch")
    retain(state, "suffix-rules.dat", suffix)
    state.stage = "internal_source"
    content = _read_once(state.paths.source_csv)
    state.stage = "internal_preparation"
    pins = dict(state.binding.source_hashes)
    reconstructed, prepared = study_preparation_inputs.prepare_internal_inputs(
        content,
        suffix,
        state.source_buffers[SOURCE],
        state.source_buffers[PREPARATION],
        pins=SourceOverlapPins(
            state.source["source_csv_sha256"],
            state.source["suffix_rules_sha256"],
            pins[SOURCE],
            pins[PREPARATION],
        ),
        source=state.source,
    )
    _retain_internal(state, reconstructed)
    return prepared, reconstructed.overlap_domains, suffix


def _retain_internal(state, reconstructed):
    state.stage = "internal_retention"
    outputs = source_checkpoints._contents(state.attempt, state.identity, reconstructed)
    for name in (
        "group_test.jsonl",
        "source-overlap.json",
        "source-reconstruction.json",
    ):
        retain(state, name, outputs[name])


def prepare_external(state, domains, suffix):
    state.stage = "external_source"
    content = _read_once(state.paths.archive)
    state.stage = "external_decode"
    decoded = study_preparation_inputs.decode_external_inputs(
        content,
        suffix,
        archive_pins=state.profile.archive_pins,
        suffix_rules_sha256=state.profile.suffix_rules_sha256,
    )
    state.stage = "publisher_retention"
    retain(
        state, "publisher-source.json", decoded.private_outputs["publisher-source.json"]
    )
    retain(state, "publisher-summary.json", canonical_bytes(decoded.public_summary))
    state.stage = "external_preparation"
    prepared = study_preparation_inputs.prepare_external_inputs(
        decoded, suffix, overlap_domains=domains
    )
    state.stage = "external_retention"
    for name in ("retained-test.jsonl", "quarantine.jsonl", "inventory.json"):
        retain(state, name, prepared.private_outputs[name])
    retain(state, "preparation-summary.json", canonical_bytes(prepared.public_summary))
    return prepared


def assess(state, internal, external):
    state.stage = "feasibility"
    content = assess_preparation_feasibility(internal.records, external)
    retain(state, "feasibility.json", content)


def complete(state):
    state.stage = "preparation_completion"
    content = canonical_bytes(
        {
            "schema_version": 1,
            "protocol": "study-preparation-v1",
            "status": "preparation_only",
            "protected_evaluation_authorized": False,
            "scoring_authorized": False,
            "execution": state.identity,
            "reservation_sha256": state.attempt.reservation_sha256,
            "input_sha256": {
                name: sha256(value).hexdigest() for name, value in state.outputs.items()
            },
        }
    )
    retain(state, "preparation-complete.json", content)
    return state.writer.finish()
