"""Constructed mixed-origin125 inputs, explicitly not verified publications."""

import json
from dataclasses import replace
from functools import lru_cache
from types import SimpleNamespace

import pytest
from study_reduction_fixtures import full_case
from study_series_input_fixtures import joined_profile, metadata, original_case
from test_evaluation_manifest import candidates, manifests

from automated_phishing_detection import study_operational_records
from automated_phishing_detection._checkpoint_codec import canonical_bytes
from automated_phishing_detection._operational_cell_protocol import SNAPSHOT_NAMES
from automated_phishing_detection._operational_cell_records import encode_descriptor
from automated_phishing_detection.http_replay import ReplayRequest
from automated_phishing_detection.operational_cell_inputs import (
    RestoredOperationalCell,
    bind_cell_descriptor,
)
from automated_phishing_detection.study_history_cell import HistoricalCellScience


def digest(content):
    from hashlib import sha256

    return sha256(content).hexdigest()


def _context(complete, source):
    accepted = complete.accepted
    source.internal, source.external = accepted.internal, accepted.external
    origin = json.loads(accepted.metadata_bytes)
    profile = joined_profile(source, origin)
    selected = SimpleNamespace(
        origin_bytes=accepted.metadata_bytes,
        profile=profile,
        internal=accepted.internal.snapshot,
        external=accepted.external.snapshot,
    )
    current = metadata(selected)
    return selected, current


def _inputs(record, selected, current, internal, external, pool):
    from automated_phishing_detection import _study_series_input_cells as cells

    content = selected if record.cell.ordinal < 73 else current
    context = json.loads(content)
    view = context if record.cell.ordinal < 73 else cells.view(context)
    key = (content, record.cell.prevalence_basis_points)
    if key not in pool:
        pool[key] = cells.selection(internal, external, record.cell)
    manifest, rows = pool[key]
    descriptor = encode_descriptor(content, view, record.cell, manifest, rows)
    reservation = json.loads(record.binding_bytes)["cell_reservation_sha256"]
    binding = bind_cell_descriptor(descriptor, cell_reservation_sha256=reservation)
    return RestoredOperationalCell(
        content,
        descriptor,
        binding,
        manifest,
        record.cell,
        tuple(ReplayRequest(row.record_id, row.raw_url) for row in rows),
    ), reservation


def _cells(complete, selected, current):
    from automated_phishing_detection.study_series_cell import SeriesCellScience
    from automated_phishing_detection.study_series_inputs import SeriesOperationalCell

    values, pool = [], {}
    for slot in complete.slots:
        record = slot.accepted
        restored, reservation = _inputs(
            record,
            selected.origin_bytes,
            current,
            selected.internal,
            selected.external,
            pool,
        )
        payloads, summary = _payloads(record)
        constructor = HistoricalCellScience
        if record.cell.ordinal >= 73:
            restored = SeriesOperationalCell(
                origin_metadata_bytes=selected.origin_bytes, computational=restored
            )
            constructor = SeriesCellScience
        values.append(constructor(payloads, restored, summary, reservation))
    return tuple(values[:72]), tuple(values[72:])


def _payloads(record):
    payloads = dict.fromkeys(SNAPSHOT_NAMES, b"constructed publication precondition")
    payloads["attempt/run.json"] = record.run_bytes
    payloads["public-summary.json"] = record.public_bytes
    summary = canonical_bytes(json.loads(record.public_bytes)["summary"])
    return tuple(sorted(payloads.items())), summary


@pytest.fixture(scope="session")
@lru_cache(maxsize=1)
def matrix():
    from test_study_series_reduction import api

    api()
    source = original_case(manifests.__wrapped__(candidates.__wrapped__()))
    complete = full_case(study_operational_records, source)
    selected, current = _context(complete, source)
    historical, fresh = _cells(complete, selected, current)
    arguments = {
        "selected_metadata_bytes": selected.origin_bytes,
        "current_context_bytes": current,
        "profile_bytes": canonical_bytes(selected.profile),
        "expected_profile_sha256": digest(canonical_bytes(selected.profile)),
        "expected_selected_metadata_sha256": digest(selected.origin_bytes),
        "expected_current_context_sha256": digest(current),
        "internal_snapshot": selected.internal,
        "external_snapshot": selected.external,
    }
    return SimpleNamespace(
        historical=historical,
        fresh=fresh,
        arguments=arguments,
        runs=complete.runs,
    )


def reduce(matrix, *, historical=None, fresh=None, **changes):
    from test_study_series_reduction import api

    return api().reduce_series_science(
        matrix.historical if historical is None else historical,
        matrix.fresh if fresh is None else fresh,
        **(matrix.arguments | changes),
    )


def replace_inputs(record, **changes):
    if type(record) is HistoricalCellScience:
        return replace(record, inputs=replace(record.inputs, **changes))
    return replace(
        record,
        inputs=replace(
            record.inputs,
            computational=replace(record.inputs.computational, **changes),
        ),
    )


def prepare(matrix):
    from automated_phishing_detection import _study_series_reduction_records as records

    arguments = matrix.arguments
    names = ("selected_metadata", "current_context", "profile")
    return records.prepare(
        matrix.historical,
        matrix.fresh,
        tuple(arguments[f"{name}_bytes"] for name in names),
        tuple(arguments[f"expected_{name}_sha256"] for name in names),
        arguments["internal_snapshot"],
        arguments["external_snapshot"],
    )
