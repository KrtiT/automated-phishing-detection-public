"""Independent synthetic sessions emit identical scientific payloads for both routes."""

from contextlib import contextmanager
from dataclasses import replace

from prepared_internal_fixtures import restored_case
from study_preparation_runner_fixtures import (
    inputs,
    preparation_api,
    preparation_case,
    runner,
)
from test_evaluation_producer import synthetic_session
from test_prepared_internal_runner import module

from automated_phishing_detection import evaluation_producer, source_runner

__all__ = ["inputs", "preparation_api", "preparation_case", "runner"]


@contextmanager
def session_owner(session, *args):
    yield session


def test_original_and_prepared_routes_emit_identical_scientific_bytes(
    preparation_api, preparation_case, inputs, monkeypatch
):
    binding, original, unused_session, unused_events = inputs
    baseline = replace(
        original,
        attempt=original.attempt.parent / "baseline",
        public_summary=original.attempt.parent / "baseline.json",
    )
    source_runner._run_bound_internal(binding, baseline)
    case = restored_case(preparation_api, preparation_case, inputs)
    fresh, *_ = synthetic_session(evaluation_producer, monkeypatch)
    monkeypatch.setattr(
        source_runner,
        "open_bound_evaluation_session",
        lambda *args: session_owner(fresh, *args),
    )
    module()._run_bound_prepared_internal(binding, case.paths, case.preparation)
    for name in (
        "predictions.jsonl",
        "manifests.json",
        "bindings.json",
        "routing.json",
        "secondary.json",
    ):
        assert (baseline.attempt / "evidence" / name).read_bytes() == (
            case.paths.attempt / "evidence" / name
        ).read_bytes()
