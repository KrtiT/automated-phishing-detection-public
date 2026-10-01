"""Invented prefixes, callback observations and complete synthetic publications."""

import json
from importlib import import_module
from importlib.util import find_spec
from types import SimpleNamespace

import pytest
from operational_cell_process_fixtures import process_records
from study_series_cell_fixtures import _publication, _run
from study_series_input_fixtures import descriptor
from study_series_prefix_fixtures import (
    candidates,
    manifests,
    prefix_case,
    reserve,
    series_case,
)

from automated_phishing_detection._checkpoint_codec import canonical_bytes
from automated_phishing_detection._study_execution_policy import DEADLINES
from automated_phishing_detection._study_history_snapshot_records import digest
from automated_phishing_detection.operational_cell_inputs import bind_cell_descriptor
from automated_phishing_detection.study_series_cell import SeriesCellScience
from automated_phishing_detection.study_series_cell_acceptance import (
    series_cell_identity,
)
from automated_phishing_detection.study_series_inputs import restore_series_cell_inputs

__all__ = ["candidates", "manifests", "series_case", "prefix_case", "ledger_cell"]


def api():
    name = "automated_phishing_detection.study_series_ledger"
    assert find_spec(name) is not None, "missing ordered live series ledger"
    return import_module(name)


def ledger(prefix, **changes):
    arguments = dict(
        public_binding=prefix.binding,
        series_attempt=prefix.series,
        segment_attempt=prefix.segment,
        intent_bytes=prefix.intent,
        import_bytes=prefix.imported,
        metadata_bytes=prefix.metadata,
    )
    return api().SeriesAdmissionLedger(**(arguments | changes))


def prepared(prefix, ordinal=None):
    source = prefix.source
    ordinal = source.profile["segment"]["start_ordinal"] if ordinal is None else ordinal
    selected = descriptor(source, ordinal, content=prefix.metadata)
    identity = series_cell_identity(
        prefix.metadata,
        selected.descriptor_bytes,
        expected_metadata_sha256=digest(prefix.metadata),
    )
    directory = source.profile["paths"]["cells_dir"] + f"/cell-{ordinal:03d}-attempt"
    attempt, reservation = reserve(identity, directory)
    binding = bind_cell_descriptor(
        selected.descriptor_bytes, cell_reservation_sha256=attempt.reservation_sha256
    )
    inputs = _restore(prefix, selected, binding, attempt)
    return SimpleNamespace(
        prefix=prefix,
        attempt=attempt,
        reservation=reservation,
        inputs=inputs,
        selected=selected,
        binding=binding,
        identity=identity,
        cell=inputs.computational.cell,
    )


def _restore(prefix, selected, binding, attempt):
    return restore_series_cell_inputs(
        prefix.metadata,
        prefix.binding.profile_bytes,
        prefix.source.internal,
        prefix.source.external,
        selected.descriptor_bytes,
        binding,
        selected.manifest_bytes,
        expected_metadata_sha256=digest(prefix.metadata),
        expected_profile_sha256=prefix.binding.profile_sha256,
        expected_binding_sha256=digest(binding),
        expected_cell_reservation_sha256=attempt.reservation_sha256,
    )


def commands(selected):
    from automated_phishing_detection._study_series_child_commands import (
        series_child_command,
    )

    return tuple(
        series_child_command(
            selected.prefix.binding,
            role,
            cell_ordinal=selected.cell.ordinal,
            cell_binding_sha256=digest(selected.binding),
        )
        for role in ("service", "client")
    )


def candidate(selected):
    inputs = selected.inputs.computational
    values, observation = process_records(
        inputs, selected.attempt.reservation_sha256, commands(selected), dict(DEADLINES)
    )
    unused_run, encoded, summary, checkpoints = _run(inputs)
    values.update(
        checkpoints, **{"reservation.json": selected.reservation, "run.json": encoded}
    )
    payloads = _publication(
        values, inputs, selected.attempt, selected.identity, summary
    )
    result = SeriesCellScience(
        tuple(payloads.items()),
        selected.inputs,
        canonical_bytes(summary),
        selected.attempt.reservation_sha256,
    )
    return result, observation, values["process-pair-intent.json"]


@pytest.fixture(scope="module")
def ledger_cell(prefix_case):
    selected = prepared(prefix_case)
    selected.candidate, selected.observation, selected.pair = candidate(selected)
    return selected


def start(current, selected):
    return current.start_cell(
        selected.cell,
        selected.attempt,
        selected.selected.descriptor_bytes,
        selected.binding,
    )


def observed(current, selected):
    service, client = commands(selected)
    with current.issue("service", service) as first:
        first.launched(321)
        with current.issue("client", client) as second:
            second.launched(654)
            second.observed(True, 0)
        first.observed(True, 0)


def snapshot(current):
    return json.loads(api().snapshot_series_ledger(current))
