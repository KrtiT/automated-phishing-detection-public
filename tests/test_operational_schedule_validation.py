"""A caller cannot replace a planned cell or change designated membership."""

from dataclasses import replace

import pytest
from test_operational_schedule import schedule_api as schedule_api


class IntegerSubclass(int):
    pass


class StringSubclass(str):
    pass


@pytest.mark.parametrize(
    "ordinal", [True, False, 0, -1, 126, 1.0, "1", None, IntegerSubclass(1)]
)
def test_lookup_rejects_invalid_or_nonexact_ordinals(schedule_api, ordinal):
    with pytest.raises(ValueError, match="^invalid_operational_ordinal$"):
        schedule_api.cell_for_ordinal(ordinal)


@pytest.mark.parametrize(
    "field, value",
    [
        ("ordinal", True),
        ("ordinal", 1.0),
        ("ordinal", 0),
        ("ordinal", 126),
        ("ordinal", IntegerSubclass(1)),
        ("ordinal", 2),
        ("workload", "transformer_only"),
        ("workload", "unknown"),
        ("workload", StringSubclass("fixed_cascade")),
        ("prevalence_basis_points", True),
        ("prevalence_basis_points", 100.0),
        ("prevalence_basis_points", IntegerSubclass(100)),
        ("prevalence_basis_points", 10),
        ("prevalence_basis_points", None),
        ("concurrency", True),
        ("concurrency", 1.0),
        ("concurrency", IntegerSubclass(1)),
        ("concurrency", 8),
        ("run_index", True),
        ("run_index", 1.0),
        ("run_index", IntegerSubclass(1)),
        ("run_index", 2),
    ],
)
def test_altered_cell_is_rejected_without_substitution(schedule_api, field, value):
    cell = replace(schedule_api.cell_for_ordinal(1), **{field: value})
    with pytest.raises(ValueError, match="^invalid_operational_cell$"):
        schedule_api.validate_cell(cell)
    for membership in ("reference_invocations_member", "primary_http_member"):
        with pytest.raises(ValueError, match="^invalid_operational_cell$"):
            getattr(cell, membership)


@pytest.mark.parametrize("cell", [None, {}, (1, "fixed_cascade", 100, 1, 1), object()])
def test_validation_requires_the_exact_record_type(schedule_api, cell):
    with pytest.raises(ValueError, match="^invalid_operational_cell$"):
        schedule_api.validate_cell(cell)


def test_record_subclass_is_not_a_scheduled_cell(schedule_api):
    class CellSubclass(schedule_api.OperationalCell):
        pass

    cell = CellSubclass(1, "fixed_cascade", 100, 1, 1)
    with pytest.raises(ValueError, match="^invalid_operational_cell$"):
        schedule_api.validate_cell(cell)


@pytest.mark.parametrize(
    "ordinal, field, value",
    [
        (91, "prevalence_basis_points", 10),
        (91, "prevalence_basis_points", 500),
        (121, "prevalence_basis_points", 0),
        (121, "prevalence_basis_points", 100),
        (121, "concurrency", 8),
    ],
)
def test_comparator_and_shift_are_not_alternative_primary_cells(
    schedule_api, ordinal, field, value
):
    cell = replace(schedule_api.cell_for_ordinal(ordinal), **{field: value})
    with pytest.raises(ValueError, match="^invalid_operational_cell$"):
        schedule_api.validate_cell(cell)


def test_exact_caller_constructed_cell_is_consistency_not_authority(schedule_api):
    cell = schedule_api.OperationalCell(121, "shift_period", None, 1, 1)
    assert schedule_api.validate_cell(cell) == cell
    assert not cell.reference_invocations_member and not cell.primary_http_member


def test_forced_record_mutation_cannot_change_later_lookups(schedule_api):
    cell = schedule_api.cell_for_ordinal(1)
    object.__setattr__(cell, "run_index", 2)
    with pytest.raises(ValueError, match="^invalid_operational_cell$"):
        schedule_api.validate_cell(cell)
    assert schedule_api.cell_for_ordinal(1).run_index == 1
    assert schedule_api.planned_cells()[0].run_index == 1
