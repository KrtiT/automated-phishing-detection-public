"""Explicitly synthetic held-history preconditions with real root reservations."""

import asyncio
import copy
from contextlib import contextmanager
from types import SimpleNamespace

import study_series_history_index_fixtures as history
from study_series_input_fixtures import candidates, manifests, series_case
from study_series_prefix_fixtures import binding
from test_study_series_runner import api

from automated_phishing_detection._checkpoint_codec import canonical_bytes
from automated_phishing_detection._study_history_snapshot_records import digest
from automated_phishing_detection.study_series_history_index import (
    validate_series_history_index,
)

__all__ = ["candidates", "manifests", "series_case"]


def setup(tmp_path, series_case, monkeypatch):
    source = SimpleNamespace(**vars(series_case))
    source.profile = copy.deepcopy(source.profile)
    source.profile["paths"].update(
        {
            name: str(tmp_path / name)
            for name in source.profile["paths"]
            if name != "repo_root"
        }
    )
    content = index_bytes(source.profile)
    public = binding(source)
    index = validate_series_history_index(
        content,
        public.profile_bytes,
        expected_index_sha256=digest(content),
        expected_profile_sha256=public.profile_sha256,
    )
    case = SimpleNamespace(source=source, public=public, index=index, events=[])
    case.history = history_record(case)
    monkeypatch.setattr(api(), "require_final_policy", lambda public: None)
    monkeypatch.setattr(
        api(),
        "recheck_series_public_execution",
        lambda public: case.events.append("public_check"),
    )
    return case


def history_record(case):
    return SimpleNamespace(
        index=case.index,
        origin_metadata_bytes=case.source.origin_bytes,
        historical_prefix=tuple(range(72)),
        internal_snapshot=case.source.internal,
        external_snapshot=case.source.external,
        check=lambda: case.events.append("history_check"),
    )


def index_bytes(profile):
    value = history.make_case(72).index
    value.update(
        attempts=history.attempts(profile),
        **history.origin_records(profile),
        accepted_sources={
            "internal": history.sources(profile, "internal", history.SNAPSHOT_NAMES),
            "external": history.sources(profile, "external", history._LOGICAL_NAMES),
        },
        accepted_cells=[history.cell(profile, ordinal) for ordinal in range(1, 73)],
        stopped_cell=history.cell(profile, 73, True),
    )
    value["interruption_review"] = value["attempts"][-1]["interruption_review"]
    content = canonical_bytes(value)
    profile["history"]["index_sha256"] = digest(content)
    return content


@contextmanager
def loader(case, public):
    assert public is case.public
    case.events.append("history_open")
    try:
        yield case.history
    finally:
        case.events.append("history_closed")


def run(case, **changes):
    arguments = dict(
        hold_history=lambda public: loader(case, public),
        lifecycle_check=lambda: case.events.append("supervisor_check"),
    )
    return asyncio.run(api()._run_series_bound(case.public, **(arguments | changes)))
