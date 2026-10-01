"""Full original numerical reducers over a fixed historical/fresh split."""

from dataclasses import asdict

import pytest
from study_series_reduction_fixtures import matrix, reduce
from test_study_series_reduction import api

from automated_phishing_detection._checkpoint_codec import canonical_bytes
from automated_phishing_detection.http_replay import (
    primary_http_summary,
    reference_invocations,
)
from automated_phishing_detection.operational_summary import summarize_operational_runs
from automated_phishing_detection.study_evidence import reduce_study_evidence

__all__ = ["matrix"]


def _guarded_reduction(matrix, monkeypatch):
    import builtins
    import os
    import subprocess

    from automated_phishing_detection import _study_reduction_runs, operational_inputs

    def forbidden(*arguments, **keywords):
        pytest.fail("reduction attempted IO or same-parent construction")

    with monkeypatch.context() as guard:
        for module, name in (
            (builtins, "open"),
            (os, "open"),
            (subprocess, "Popen"),
            (operational_inputs, "build_accepted_inputs"),
            (_study_reduction_runs, "restore_runs"),
        ):
            guard.setattr(module, name, forbidden)
        return reduce(matrix)


def test_full125_cross_origin_reduction_matches_all_original_kernels(
    matrix, monkeypatch
):
    result = _guarded_reduction(matrix, monkeypatch)
    study = reduce_study_evidence(
        internal=matrix.arguments["internal_snapshot"].population,
        external=matrix.arguments["external_snapshot"].replay.evidence,
        reference=reference_invocations(matrix.runs[0]),
        http=primary_http_summary(matrix.runs[20:25]),
    )
    assert result.operational_bytes == canonical_bytes(
        summarize_operational_runs(matrix.runs)
    )
    assert result.study_bytes == canonical_bytes(asdict(study))
    assert result.study["ablation_family"]["family_size"] == 4
    assert result.study["primary"]["hypotheses"]["H2"]["decision"] == "not_supported"
    assert result.operational["groups"][0]["request_errors"] == 49996
    assert result.operational["groups"][-1]["request_errors"] == 10


def test_each_origin_decodes_each_run_once_without_mutable_cached_views(
    matrix, monkeypatch
):
    from automated_phishing_detection import study_history_cell, study_series_cell

    counts = {"old": 0, "new": 0}
    for module, key in ((study_history_cell, "old"), (study_series_cell, "new")):
        original = module.decode_run

        def counted(*arguments, original=original, key=key):
            counts[key] += 1
            return original(*arguments)

        monkeypatch.setattr(module, "decode_run", counted)
    result = reduce(matrix)
    assert counts == {"old": 72, "new": 53}
    result.operational["groups"].clear()
    result.study.clear()
    assert len(result.operational["groups"]) == 25 and result.study


def test_reduction_does_not_relabel_original_input_bytes(matrix):
    old = matrix.historical[0].inputs
    new = matrix.fresh[0].inputs
    matrix.historical[0].run.after_measured.admitted_requests = 0
    matrix.fresh[0].run.after_measured.admitted_requests = 0
    result = reduce(matrix)
    assert old.accepted_bytes == matrix.arguments["selected_metadata_bytes"]
    assert new.origin_metadata_bytes == old.accepted_bytes
    assert new.computational.accepted_bytes == matrix.arguments["current_context_bytes"]
    assert new.computational.accepted_bytes != old.accepted_bytes
    assert len(result.operational["groups"]) == 25


def test_original_keyboard_interrupt_is_not_normalized(matrix, monkeypatch):
    interruption = KeyboardInterrupt("invented interruption")

    def interrupted(*arguments):
        raise interruption

    monkeypatch.setattr(api(), "reference_invocations", interrupted)
    with pytest.raises(KeyboardInterrupt) as caught:
        reduce(matrix)
    assert caught.value is interruption
