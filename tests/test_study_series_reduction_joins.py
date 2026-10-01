"""Context pins, cached source selection and retained summaries remain exact."""

import json
from dataclasses import replace

import pytest
from study_series_reduction_fixtures import (
    digest,
    matrix,
    prepare,
    reduce,
    replace_inputs,
)
from test_study_series_reduction import api
from test_study_series_reduction_boundaries import no_runs

from automated_phishing_detection._checkpoint_codec import canonical_bytes

__all__ = ["matrix", "no_runs"]


def test_rehashed_profile_cutover_must_equal_supplied_prefix(matrix, no_runs):
    profile = json.loads(matrix.arguments["profile_bytes"])
    profile["segment"]["start_ordinal"] += 1
    profile_bytes = canonical_bytes(profile)
    context = json.loads(matrix.arguments["current_context_bytes"])
    context["profile_sha256"] = digest(profile_bytes)
    context_bytes = canonical_bytes(context)
    with pytest.raises(api().SeriesReductionError):
        reduce(
            matrix,
            profile_bytes=profile_bytes,
            expected_profile_sha256=digest(profile_bytes),
            current_context_bytes=context_bytes,
            expected_current_context_sha256=digest(context_bytes),
        )


def test_independently_repinned_original_cannot_replace_context_origin(matrix, no_runs):
    origin = json.loads(matrix.arguments["selected_metadata_bytes"])
    origin["root_reservation_sha256"] = "0" * 64
    content = canonical_bytes(origin)
    with pytest.raises(api().SeriesReductionError):
        reduce(
            matrix,
            selected_metadata_bytes=content,
            expected_selected_metadata_sha256=digest(content),
        )


@pytest.mark.parametrize("position", [0, 72])
def test_input_request_tuple_cannot_be_replaced_by_mutable_view(
    matrix, no_runs, position
):
    values = list(matrix.historical + matrix.fresh)
    record = values[position]
    original = record.inputs if position == 0 else record.inputs.computational
    values[position] = replace_inputs(record, requests=list(original.requests))
    with pytest.raises(api().SeriesReductionError):
        reduce(matrix, historical=tuple(values[:72]), fresh=tuple(values[72:]))


@pytest.mark.parametrize("changed", ["bool", "ordinal", "reference"])
def test_frozen_schedule_identity_rejects_coercion_and_substitution(
    matrix, no_runs, changed
):
    record = matrix.historical[0]
    cell = record.inputs.cell
    updates = {"ordinal": True} if changed == "bool" else {"ordinal": 2}
    if changed == "reference":
        updates = {"run_index": 2}
    record = replace_inputs(record, cell=replace(cell, **updates))
    with pytest.raises(api().SeriesReductionError):
        reduce(matrix, historical=(record,) + matrix.historical[1:])


def test_summary_join_rejects_an_altered_late_repeat(matrix):
    from automated_phishing_detection import _study_series_reduction_records as records

    values = matrix.historical + matrix.fresh
    summaries = tuple(
        json.loads(value.summary_bytes) | {"cell_ordinal": index + 1}
        for index, value in enumerate(values)
    )
    operational = {"groups": [{"run_summaries": summaries}]}
    records.match_summaries(operational, values)
    changed = replace(values[-1], summary_bytes=canonical_bytes({"request_errors": 0}))
    with pytest.raises(ValueError):
        records.match_summaries(operational, values[:-1] + (changed,))


def test_context_and_source_pair_authenticate_only_once(matrix, monkeypatch):
    from automated_phishing_detection import _study_series_reduction_records as records

    calls, original = [], records.context.authenticate

    def counted(*arguments, **keywords):
        calls.append(arguments)
        return original(*arguments, **keywords)

    monkeypatch.setattr(records.context, "authenticate", counted)
    prepare(matrix)
    assert len(calls) == 1


def test_selection_cache_separates_identical_prevalence_across_contexts(
    matrix, monkeypatch
):
    from automated_phishing_detection import _study_series_reduction_records as records

    pool, calls, original = {}, [], records.selection.selection

    def counted(*arguments):
        calls.append(arguments[-1])
        return original(*arguments)

    monkeypatch.setattr(records.selection, "selection", counted)
    cell = matrix.historical[0].inputs.cell
    for content in (b"old", b"new", b"old", b"new"):
        records._selected(
            pool,
            content,
            cell,
            matrix.arguments["internal_snapshot"],
            matrix.arguments["external_snapshot"],
        )
    assert len(calls) == 2
