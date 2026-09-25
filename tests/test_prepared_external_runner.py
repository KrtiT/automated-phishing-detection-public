"""Real preparation and scientific production use retained bytes only."""

import json

import pytest
from prepared_external_fixtures import (
    inputs,
    preparation_api,
    preparation_case,
    prepared_case,
    runner,
)

from automated_phishing_detection import external_source_runner as worker

__all__ = ["inputs", "preparation_api", "preparation_case", "prepared_case", "runner"]


def test_real_restored_inputs_score_without_original_acquisition(
    prepared_case, monkeypatch
):
    case = prepared_case

    def forbidden(*args, **kwargs):
        pytest.fail("prepared worker reacquired original input")

    monkeypatch.setattr(worker.body, "_source", forbidden)
    monkeypatch.setattr(worker.body, "decode_phishvn_archive", forbidden)
    result = worker._run_bound_prepared_external(
        case.binding, case.paths, handoff=case.handoff, preparation=case.preparation
    )
    public = json.loads(result.read_bytes())
    assert len(public["private_sha256"]) == 36
    assert (
        public["execution"]["study_preparation_complete_sha256"]
        == case.preparation.completion_sha256
    )
    assert (
        public["execution"]["reservation_sha256"] != case.preparation.reservation_sha256
    )
    assert len(case.session.evaluation.primary.scorer.urls) == 1
    for name, expected in case.produced.private_outputs.items():
        assert (case.paths.attempt / "evidence" / name).read_bytes() == expected
