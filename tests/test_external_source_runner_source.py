"""Invented full-split preparation and once-only retention boundaries."""

import json

import phishvn_source_fixtures as publisher
import pytest
import test_external_source_runner as fixtures
from external_completion_lineage_fixtures import fixture_profile

runner_api = fixtures.runner_api
runner_case = fixtures.runner_case


def test_verified_overlap_quarantines_all_splits_before_any_score(
    runner_api, runner_case
):
    case = runner_case
    rows = [
        publisher.record("internal-test", url="https://internal-only.com/test"),
        publisher.record(
            "internal-train", "train", url="https://internal-only.com/train"
        ),
        publisher.record("validation", "val", url="https://unscored.com/val"),
        publisher.record("retained", url="https://safe.com/path"),
    ]
    archive = publisher.bundle(publisher.members(rows))
    case.profile = fixture_profile(case.binding, archive.pins, case.suffix)
    case.paths.archive.write_bytes(archive.content)
    fixtures.run(runner_api, case)
    assert case.session.evaluation.primary.scorer.urls == ["https://safe.com/path"]
    checkpoint = case.paths.attempt / "checkpoints"
    rows = [
        json.loads(line)
        for line in (checkpoint / "quarantine.jsonl").read_bytes().splitlines()
    ]
    assert {row["source_split"] for row in rows} == {"train", "test"}
    assert all("phiusiil_domain_overlap" in row["reason_codes"] for row in rows)
    inventory = json.loads((checkpoint / "inventory.json").read_bytes())
    assert inventory["declared_split_counts"] == {"train": 1, "val": 1, "test": 2}


@pytest.mark.parametrize("name", ["archive", "suffix_rules"])
def test_source_alias_is_rejected_without_scoring(runner_api, runner_case, name):
    path = getattr(runner_case.paths, name)
    target = path.with_suffix(".original")
    path.rename(target)
    path.symlink_to(target)
    with pytest.raises(runner_api.ExternalSourceExecutionError):
        fixtures.run(runner_api, runner_case)
    assert not runner_case.session.evaluation.primary.scorer.urls
    assert not runner_case.paths.public_summary.exists()


@pytest.mark.parametrize("boundary", ["begin", "complete"])
def test_retention_failure_never_retries_or_loses_completed_producer(
    runner_api, runner_case, monkeypatch, boundary
):
    owner = runner_api.body.ExternalCheckpointWriter
    original, calls = getattr(owner, boundary), []
    first = KeyboardInterrupt("private-canary")

    def interrupted(writer, *args):
        calls.append(True)
        original(writer, *args)
        raise first

    monkeypatch.setattr(owner, boundary, interrupted)
    with pytest.raises(KeyboardInterrupt) as caught:
        fixtures.run(runner_api, runner_case)
    assert caught.value is first and calls == [True]
    record = json.loads(caught.value.progress)
    expected = 6 if boundary == "begin" else 36
    assert len(record["checkpoints"]["confirmed_sha256"]) == expected
    assert (record["completed"] is None) is (boundary == "begin")
    assert len(runner_case.session.evaluation.primary.scorer.urls) == (
        0 if boundary == "begin" else 5
    )
    assert not runner_case.paths.public_summary.exists()
