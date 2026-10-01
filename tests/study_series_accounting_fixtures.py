"""Synthetic accounting contexts; no original research or retained files."""

import copy
from importlib import import_module
from importlib.util import find_spec
from types import SimpleNamespace

import pytest
import study_series_history_index_fixtures as history
from study_series_ledger_fixtures import candidate, ledger, observed, prepared, start
from study_series_prefix_fixtures import make_prefix_case

from automated_phishing_detection._checkpoint_codec import canonical_bytes
from automated_phishing_detection._study_history_snapshot_records import digest
from automated_phishing_detection.study_series_history_index import (
    validate_series_history_index,
)
from automated_phishing_detection.study_series_ledger import snapshot_series_ledger


def api():
    name = "automated_phishing_detection.study_series_accounting"
    assert find_spec(name), "missing pure saved series accounting"
    return import_module(name)


def history_index(profile):
    value = history.make_case(124).index
    value.update(
        attempts=history.attempts(profile),
        **history.origin_records(profile),
        accepted_sources={
            "internal": history.sources(profile, "internal", history.SNAPSHOT_NAMES),
            "external": history.sources(profile, "external", history._LOGICAL_NAMES),
        },
        accepted_cells=[history.cell(profile, ordinal) for ordinal in range(1, 125)],
        stopped_cell=history.cell(profile, 125, True),
    )
    value["interruption_review"] = value["attempts"][-1]["interruption_review"]
    profile["history"]["index_sha256"] = digest(canonical_bytes(value))
    return canonical_bytes(value)


@pytest.fixture(scope="module")
def accounting_case(series_case):
    source = SimpleNamespace(**vars(series_case))
    source.profile = copy.deepcopy(source.profile)
    source.profile["segment"]["start_ordinal"] = 125
    index_bytes = history_index(source.profile)
    prefix = make_prefix_case(source)
    index = validate_series_history_index(
        index_bytes,
        prefix.binding.profile_bytes,
        expected_index_sha256=digest(index_bytes),
        expected_profile_sha256=prefix.binding.profile_sha256,
    )
    selected = prepared(prefix)
    selected.candidate, selected.observation, selected.pair = candidate(selected)
    return SimpleNamespace(prefix=prefix, selected=selected, index=index)


def prefix_arguments(prefix):
    return dict(
        import_bytes=prefix.imported,
        intent_bytes=prefix.intent,
        metadata_bytes=prefix.metadata,
    )


def complete_ledger(selected):
    current = ledger(selected.prefix)
    start(current, selected)
    observed(current, selected)
    current.accept_cell(selected.candidate, selected.observation, selected.pair)
    return snapshot_series_ledger(current)


def segment(prefix, *, ledger_bytes=None, status="failed", stage="cell_execution"):
    return api().segment_accounting_bytes(
        prefix.binding,
        prefix.series,
        prefix.segment,
        status=status,
        stage=stage,
        ledger_bytes=ledger_bytes,
        **prefix_arguments(prefix),
    )


def verify_segment(prefix, content, **changes):
    return api().verify_segment_accounting(
        content,
        prefix.binding,
        prefix.series,
        prefix.segment,
        expected_sha256=digest(content),
        **(prefix_arguments(prefix) | changes),
    )


def series(case, *, status="failed", segment_bytes=None):
    prefix = case.prefix
    arguments = (
        {}
        if segment_bytes is None
        else dict(
            segment_attempt=prefix.segment,
            segment_accounting_bytes=segment_bytes,
            **prefix_arguments(prefix),
        )
    )
    return api().series_accounting_bytes(
        prefix.binding,
        prefix.series,
        case.index,
        status=status,
        **arguments,
    )


@pytest.fixture(scope="module")
def completed_accounting(accounting_case):
    return segment(
        accounting_case.prefix,
        ledger_bytes=complete_ledger(accounting_case.selected),
        status="complete",
        stage="segment_finalization",
    )
