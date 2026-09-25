"""Every authenticated predecessor is durable before the next source-only stage."""

import json

import pytest
from study_preparation_runner_fixtures import (
    inputs,
    preparation_api,
    preparation_case,
    runner,
)

__all__ = ["inputs", "preparation_api", "preparation_case", "runner"]


def test_sources_read_once_after_reservation_and_internal_retention(
    preparation_api, preparation_case, monkeypatch
):
    case, reads = preparation_case, []
    original = preparation_api.body.source_runner._read_file_once

    def observed(path, **kwargs):
        if path in (case.paths.source_csv, case.paths.suffix_rules, case.paths.archive):
            assert (case.paths.attempt / "reservation.json").is_file()
            reads.append(path)
        if path == case.paths.archive:
            for name in (
                "group_test.jsonl",
                "source-overlap.json",
                "source-reconstruction.json",
            ):
                assert (case.paths.attempt / name).is_file()
        return original(path, **kwargs)

    monkeypatch.setattr(preparation_api.body.source_runner, "_read_file_once", observed)
    preparation_api._run_bound_preparation(case.binding, case.paths)
    assert reads == [case.paths.suffix_rules, case.paths.source_csv, case.paths.archive]


def test_external_preparation_follows_durable_publisher_bytes(
    preparation_api, preparation_case, monkeypatch
):
    case = preparation_case
    helper = preparation_api.body.study_preparation_inputs
    original = helper.prepare_external_inputs

    def observed(decoded, suffix, **kwargs):
        assert (case.paths.attempt / "publisher-source.json").read_bytes() == (
            decoded.private_outputs["publisher-source.json"]
        )
        summary = json.loads(
            (case.paths.attempt / "publisher-summary.json").read_bytes()
        )
        assert summary == decoded.public_summary
        overlap = json.loads((case.paths.attempt / "source-overlap.json").read_bytes())
        assert kwargs["overlap_domains"] == frozenset(overlap["domains"])
        assert "quarantined-label.com" in kwargs["overlap_domains"]
        return original(decoded, suffix, **kwargs)

    monkeypatch.setattr(helper, "prepare_external_inputs", observed)
    preparation_api._run_bound_preparation(case.binding, case.paths)


def test_feasibility_follows_all_prepared_bytes(
    preparation_api, preparation_case, monkeypatch
):
    case = preparation_case
    original = preparation_api.body.assess_preparation_feasibility

    def observed(internal, external):
        for name, content in external.private_outputs.items():
            assert (case.paths.attempt / name).read_bytes() == content
        assert (case.paths.attempt / "preparation-summary.json").is_file()
        return original(internal, external)

    monkeypatch.setattr(
        preparation_api.body, "assess_preparation_feasibility", observed
    )
    preparation_api._run_bound_preparation(case.binding, case.paths)


def test_second_public_gate_is_independently_closed(
    preparation_api, preparation_case, monkeypatch
):
    binding = preparation_case.binding
    monkeypatch.setattr(
        type(binding), "protected_evaluation_ready", property(lambda _: True)
    )
    monkeypatch.setattr(
        preparation_api, "bind_execution", lambda *args, **kwargs: binding
    )

    def forbidden(*args, **kwargs):
        pytest.fail("external closed gate reached source preparation")

    monkeypatch.setattr(preparation_api, "_run_bound_preparation", forbidden)
    with pytest.raises(
        preparation_api.StudyPreparationError, match="pre_access_freeze"
    ):
        preparation_api.run_study_preparation(
            binding.root,
            expected_revision=binding.revision,
            expected_contract_sha256=binding.contract_sha256,
            paths=object(),
        )
