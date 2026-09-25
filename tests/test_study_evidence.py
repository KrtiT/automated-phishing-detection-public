"""Joint primary gates and the unchanged four-cell nominal ablation family."""

import importlib.util
from dataclasses import FrozenInstanceError, asdict, replace

import pytest
from study_evidence_fixtures import (
    cells,
    empty_external,
    evidence,
    internal_population,
    without_model,
)
from study_evidence_fixtures import study as study

from automated_phishing_detection import hypothesis_evaluation as evaluator
from automated_phishing_detection import secondary_metrics


def test_joint_reducer_is_available():
    assert importlib.util.find_spec("automated_phishing_detection.study_evidence"), (
        "missing joint study evidence reducer"
    )


def test_complete_reduction_matches_existing_primary_and_frozen_audit(study):
    supplied = evidence()
    external = supplied["external"]
    result = study.reduce_study_evidence(**supplied)
    assert result.primary == evaluator.evaluate_primary(
        populations={"internal": supplied["internal"], **external.populations},
        controls=external.controls,
        external_windows=external.external_windows,
        audit_windows=evaluator.WindowCounts(28, 252),
        reference=supplied["reference"],
        http=supplied["http"],
    )
    assert [
        result.primary.hypotheses[name].decision for name in ("H1", "H2", "H3")
    ] == ["supported", "not_supported", "supported"]
    assert result.ablation_family.complete
    assert tuple(cells(result)) == secondary_metrics.ABLATION_FAMILY
    assert result.ablation_family.family_size == 4
    assert all(cell.raw_pvalue.value == 0.5 for cell in result.ablation_family.cells)
    assert all(
        cell.adjusted_pvalue.value == 1.0 for cell in result.ablation_family.cells
    )
    with pytest.raises(FrozenInstanceError):
        result.primary = None


def test_absent_evidence_stays_missing_but_retains_the_audit_failure(study):
    result = study.reduce_study_evidence()
    assert result.primary.hypotheses["H2"].decision == "not_supported"
    assert all(
        result.primary.hypotheses[name].decision == "undecided" for name in ("H1", "H3")
    )
    assert len(result.ablation_family.cells) == result.ablation_family.family_size == 4
    assert not result.ablation_family.complete
    for cell in result.ablation_family.cells:
        assert (
            cell.raw_pvalue
            == cell.adjusted_pvalue
            == secondary_metrics.MetricEstimate(None, "missing_evidence")
        )
    assert len(result.primary.contrasts) == 6


def test_supplied_empty_populations_remain_distinct_from_absent_evidence(study):
    result = study.reduce_study_evidence(
        internal=internal_population(()), external=empty_external()
    )
    assert not result.ablation_family.complete
    assert all(
        cell.raw_pvalue.reason == "empty_population"
        for cell in result.ablation_family.cells
    )
    assert all(cell.raw_pvalue.value is None for cell in result.ablation_family.cells)
    gate = next(
        gate
        for gate in result.primary.hypotheses["H2"].gates
        if gate.name == "external_window_alerts"
    )
    assert gate.status == "not_estimable" and gate.reason == "zero_denominator"


@pytest.mark.parametrize("labels", [(1,) * 16, (0,) * 16, (1,) * 16 + (0,) * 5])
def test_only_positive_internal_rows_enter_mcnemar_and_missing_slots_count(
    study, labels
):
    result = study.reduce_study_evidence(internal=internal_population(labels))
    family = cells(result)
    logistic = family["internal_logistic_minus_length"]
    cascade = family["internal_cascade_minus_logistic"]
    if 1 in labels:
        assert logistic.raw_pvalue.value == 2**-15
        assert logistic.adjusted_pvalue.value == 4 * 2**-15
        assert cascade.raw_pvalue.value == cascade.adjusted_pvalue.value == 1.0
    else:
        assert (
            logistic.raw_pvalue.reason
            == cascade.raw_pvalue.reason
            == "empty_population"
        )
    for name in secondary_metrics.ABLATION_FAMILY[2:]:
        assert family[name].raw_pvalue == secondary_metrics.MetricEstimate(
            None, "missing_evidence"
        )
    assert (
        not result.ablation_family.complete and result.ablation_family.family_size == 4
    )


def test_missing_models_reserve_their_slots_without_hiding_other_cells(study):
    supplied = evidence()
    supplied["internal"] = without_model(supplied["internal"], "length_only")
    external = supplied["external"]
    populations = external.populations | {
        "gold": without_model(external.populations["gold"], "cascade")
    }
    supplied["external"] = replace(external, populations=populations)
    family = cells(study.reduce_study_evidence(**supplied))
    assert (
        family["internal_logistic_minus_length"].raw_pvalue.reason == "missing_evidence"
    )
    assert (
        family["external_gold_cascade_minus_logistic"].raw_pvalue.reason
        == "missing_evidence"
    )
    assert family["internal_cascade_minus_logistic"].raw_pvalue.value is not None
    assert family["external_gold_logistic_minus_length"].raw_pvalue.value is not None
    assert tuple(family) == secondary_metrics.ABLATION_FAMILY


def test_operational_and_external_inputs_are_forwarded_without_invention(
    study, monkeypatch
):
    supplied = evidence()
    observed = []
    original = evaluator.evaluate_primary

    def evaluate(**kwargs):
        observed.append(kwargs)
        return original(**kwargs)

    monkeypatch.setattr(evaluator, "evaluate_primary", evaluate)
    result = study.reduce_study_evidence(**supplied)
    assert observed[0]["reference"] is supplied["reference"]
    assert observed[0]["http"] is supplied["http"]
    assert observed[0]["controls"] is supplied["external"].controls
    assert observed[0]["external_windows"] is supplied["external"].external_windows
    assert observed[0]["audit_windows"] == evaluator.WindowCounts(28, 252)
    assert "private" not in str(asdict(result))


def test_missing_control_model_preserves_existing_pending_semantics(study):
    supplied = evidence()
    external = supplied["external"]
    supplied["external"] = replace(
        external, controls=without_model(external.controls, "transformer")
    )
    result = study.reduce_study_evidence(**supplied)
    gate = next(
        gate
        for gate in result.primary.hypotheses["H3"].gates
        if gate.name == "tranco.transformer.alert_rate"
    )
    assert gate.status == "pending" and gate.reason == "missing_evidence"
    assert result.ablation_family.complete


def test_summary_does_not_disclose_record_ids_or_domains(study):
    supplied = evidence()
    rendered = str(asdict(study.reduce_study_evidence(**supplied)))
    populations = (supplied["internal"], *supplied["external"].populations.values())
    assert all(
        row.record_id not in rendered and row.registrable_domain not in rendered
        for population in populations
        for row in population.records
    )
