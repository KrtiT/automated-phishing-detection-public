"""Reserve before source access and retain provenance before the first score."""

import json
from dataclasses import replace

import pytest
import test_external_source_runner as fixtures

from automated_phishing_detection.internal_external_handoff import (
    InternalHandoffPayloads,
)

runner_api = fixtures.runner_api
runner_case = fixtures.runner_case


def test_one_read_inputs_and_provenance_precede_real_producer(
    runner_api, runner_case, monkeypatch
):
    case, reads = runner_case, []
    original_read = runner_api.body._read_file_once
    original_produce = runner_api.body.produce_external_evidence

    def read(path):
        assert case.active and (case.paths.attempt / "reservation.json").is_file()
        reads.append(path)
        return original_read(path)

    def produce(prepared, session, *, retain):
        names = {path.name for path in (case.paths.attempt / "checkpoints").iterdir()}
        assert names == runner_api.body.PROVENANCE_NAMES
        assert case.active and len(session.evaluation.primary.scorer.urls) == 0
        return original_produce(prepared, session, retain=retain)

    monkeypatch.setattr(runner_api.body, "_read_file_once", read)
    monkeypatch.setattr(runner_api.body, "produce_external_evidence", produce)
    fixtures.run(runner_api, case)
    assert reads == [case.paths.suffix_rules, case.paths.archive]


def test_invalid_handoff_precedes_any_supplied_path_inspection(
    runner_api, runner_case, monkeypatch
):
    def forbidden(*args):
        pytest.fail("invalid handoff reached supplied paths")

    monkeypatch.setattr(runner_api.body, "_output_paths", forbidden)
    handoff = InternalHandoffPayloads(b"invalid", b"private-canary")
    with pytest.raises(runner_api.ExternalSourceExecutionError) as caught:
        fixtures.run(runner_api, runner_case, handoff=handoff)
    assert "private-canary" not in str(caught.value)
    assert not runner_case.paths.attempt.exists()


@pytest.mark.parametrize("name", ["suffix_rules", "archive"])
def test_bad_source_bytes_stop_before_scoring(
    runner_api, runner_case, monkeypatch, name
):
    case, reads = runner_case, []
    original = runner_api.body._read_file_once
    getattr(case.paths, name).write_bytes(b"private-canary")

    def read(path):
        reads.append(path)
        return original(path)

    monkeypatch.setattr(runner_api.body, "_read_file_once", read)
    with pytest.raises(runner_api.ExternalSourceExecutionError) as caught:
        fixtures.run(runner_api, case)
    assert "private-canary" not in str(caught.value)
    expected = [case.paths.suffix_rules]
    assert reads == expected + ([] if name == "suffix_rules" else [case.paths.archive])
    assert not case.session.evaluation.primary.scorer.urls
    assert not case.paths.public_summary.exists()
    assert (
        json.loads((case.paths.attempt / "outcome.json").read_bytes())["status"]
        == "failed"
    )


@pytest.mark.parametrize(
    "change", ["paths", "artifacts", "secondary_artifacts", "drift_artifacts"]
)
def test_wrong_path_record_types_stop_before_reservation(
    runner_api, runner_case, change
):
    paths = None if change == "paths" else replace(runner_case.paths, **{change: None})
    with pytest.raises(runner_api.ExternalSourceExecutionError):
        fixtures.run(runner_api, runner_case, paths=paths)
    assert not runner_case.paths.attempt.exists()


@pytest.mark.parametrize("name", ["attempt", "public_summary"])
def test_existing_output_is_never_reused(runner_api, runner_case, name):
    target = getattr(runner_case.paths, name)
    target.write_bytes(b"keep existing")
    with pytest.raises(runner_api.ExternalSourceExecutionError):
        fixtures.run(runner_api, runner_case)
    assert target.read_bytes() == b"keep existing"
    assert not runner_case.events


def test_output_inside_checkout_is_rejected_before_reservation(runner_api, runner_case):
    paths = replace(runner_case.paths, attempt=runner_case.binding.root / "attempt")
    with pytest.raises(runner_api.ExternalSourceExecutionError):
        fixtures.run(runner_api, runner_case, paths=paths)
    assert not runner_case.events
