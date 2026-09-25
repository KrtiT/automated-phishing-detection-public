"""Fixed schedule selection preserves explicit shortages and all original rows."""

from dataclasses import replace

import pytest
from operational_input_fixtures import build, candidates, case, manifests, source_case
from test_operational_cell_inputs import module
from test_operational_inputs import module as inputs_module

from automated_phishing_detection.evaluation_producer import ManifestOutcome
from automated_phishing_detection.operational_schedule import (
    cell_for_ordinal,
    planned_cells,
)

__all__ = ["candidates", "manifests", "case"]


def test_all125_cells_use_the_same_exact_original_manifest(case, monkeypatch):
    api = module()
    accepted = build(inputs_module(), case)
    from automated_phishing_detection import evaluation_manifest

    monkeypatch.setattr(
        evaluation_manifest,
        "build_manifest",
        lambda *args, **kwargs: pytest.fail("resampled"),
    )
    identities = {}
    for cell in planned_cells():
        result = api.build_cell_descriptor(accepted, cell)
        key = cell.prevalence_basis_points
        assert (
            identities.setdefault(key, result.manifest_bytes) == result.manifest_bytes
        )
        expected = (
            case.external.snapshot.rows
            if key is None
            else case.internal.snapshot.manifests[key].manifest.records
        )
        records = tuple(row.record for row in expected) if key is None else expected
        assert tuple((row.record_id, row.raw_url) for row in result.requests) == tuple(
            (row.record_id, row.raw_url) for row in records
        )
        assert cell.reference_invocations_member == (cell.ordinal == 1)
        assert cell.primary_http_member == (21 <= cell.ordinal <= 25)


def test_internal_shortage_retains_original_outcome(case):
    api = module()
    outcome = ManifestOutcome(
        "insufficient_capacity", insufficient_label=1, required=100, available=7
    )
    outcomes = dict(case.internal.snapshot.manifest_outcomes) | {100: outcome}
    snapshot = replace(
        case.internal.snapshot, manifest_outcomes=tuple(sorted(outcomes.items()))
    )
    accepted = replace(
        build(inputs_module(), case), internal=replace(case.internal, snapshot=snapshot)
    )
    with pytest.raises(api.OperationalCapacityError) as failure:
        api.build_cell_descriptor(accepted, cell_for_ordinal(1))
    assert failure.value.manifest_outcome is outcome
    assert (failure.value.required, failure.value.available) == (100, 7)


@pytest.mark.parametrize("count", [0, 1, 999])
def test_external_shortage_is_not_fabricated_zero(manifests, count):
    api = module()
    accepted = build(inputs_module(), source_case(manifests, count))
    with pytest.raises(api.OperationalCapacityError) as failure:
        api.build_cell_descriptor(accepted, cell_for_ordinal(121))
    assert failure.value.manifest_outcome is None
    assert (failure.value.required, failure.value.available) == (1000, count)


@pytest.mark.parametrize(
    "field,value",
    [
        ("ordinal", True),
        ("ordinal", 1.0),
        ("concurrency", True),
        ("prevalence_basis_points", 100.0),
        ("run_index", 2),
        ("workload", "shift_period"),
    ],
)
def test_forged_cells_are_rejected(case, field, value):
    with pytest.raises(ValueError):
        module().build_cell_descriptor(
            build(inputs_module(), case), replace(cell_for_ordinal(1), **{field: value})
        )


def test_boolean_insufficient_label_is_integrity_error_not_capacity(case):
    api = module()
    outcome = ManifestOutcome(
        "insufficient_capacity", insufficient_label=True, required=100, available=7
    )
    snapshot = replace(case.internal.snapshot, manifest_outcomes=((100, outcome),))
    accepted = replace(
        build(inputs_module(), case), internal=replace(case.internal, snapshot=snapshot)
    )
    with pytest.raises(api.OperationalInputError):
        api.build_cell_descriptor(accepted, cell_for_ordinal(1))


@pytest.mark.parametrize("label,required", [(1, 1), (0, 100), (1, 101), (0, 9901)])
def test_insufficient_quota_must_equal_frozen_manifest_capacity(case, label, required):
    api = module()
    outcome = ManifestOutcome(
        "insufficient_capacity",
        insufficient_label=label,
        required=required,
        available=0,
    )
    snapshot = replace(case.internal.snapshot, manifest_outcomes=((100, outcome),))
    accepted = replace(
        build(inputs_module(), case), internal=replace(case.internal, snapshot=snapshot)
    )
    with pytest.raises(api.OperationalInputError):
        api.build_cell_descriptor(accepted, cell_for_ordinal(1))


@pytest.mark.parametrize("count", [999, 1002])
def test_parent_descriptor_rejects_external_snapshot_replaced_after_projection(
    case, manifests, count
):
    api = module()
    accepted = build(inputs_module(), case)
    changed = replace(accepted, external=source_case(manifests, count).external)
    with pytest.raises(api.OperationalInputError):
        api.build_cell_descriptor(changed, cell_for_ordinal(121))


def test_short_typed_view_cannot_report_capacity_against_long_retained_bytes(case):
    api = module()
    accepted = build(inputs_module(), case)
    snapshot = replace(case.external.snapshot, rows=case.external.snapshot.rows[:999])
    changed = replace(accepted, external=replace(case.external, snapshot=snapshot))
    with pytest.raises(api.OperationalInputError):
        api.build_cell_descriptor(changed, cell_for_ordinal(121))
