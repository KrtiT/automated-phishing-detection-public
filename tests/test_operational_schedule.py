"""The complete operational schedule is fixed independently of outcomes."""

import importlib
import importlib.util
import json
from dataclasses import FrozenInstanceError, astuple, fields
from pathlib import Path

import pytest


@pytest.fixture
def schedule_api():
    name = "automated_phishing_detection.operational_schedule"
    assert importlib.util.find_spec(name) is not None, "missing pure schedule"
    return importlib.import_module(name)


def test_exact_fixed_transformer_then_shift_order(schedule_api):
    expected = [
        ("fixed_cascade", prevalence, concurrency, repeat)
        for prevalence in (100, 10, 500)
        for concurrency in (1, 8, 16, 32, 64, 128)
        for repeat in (1, 2, 3, 4, 5)
    ]
    expected.extend(
        ("transformer_only", 100, concurrency, repeat)
        for concurrency in (1, 8, 16, 32, 64, 128)
        for repeat in (1, 2, 3, 4, 5)
    )
    expected.extend(("shift_period", None, 1, repeat) for repeat in range(1, 6))
    cells = schedule_api.planned_cells()
    assert type(cells) is tuple and len(cells) == len(set(cells)) == 125
    assert tuple(astuple(cell) for cell in cells) == tuple(
        (ordinal, *specification)
        for ordinal, specification in enumerate(expected, start=1)
    )


def test_schedule_matches_unchanged_public_contracts(schedule_api):
    directory = Path(__file__).resolve().parents[1] / "data"
    http = json.loads((directory / "http-replay-contract-v1.json").read_bytes())
    workloads = json.loads((directory / "operational-workloads-v1.json").read_bytes())
    cells = schedule_api.planned_cells()
    for workload in workloads["execution_order"][:2]:
        contract = http["runs"] if workload == "fixed_cascade" else workloads[workload]
        prevalences = contract.get("prevalence_order_basis_points", [100])
        expected = tuple(
            (workload, prevalence, concurrency, repeat)
            for prevalence in prevalences
            for concurrency in contract["concurrency_order"]
            for repeat in contract["run_indices"]
        )
        assert (
            tuple(astuple(cell)[1:] for cell in cells if cell.workload == workload)
            == expected
        )
    shift = workloads["shift_period"]
    assert tuple(astuple(cell)[1:] for cell in cells[120:]) == tuple(
        ("shift_period", None, shift["concurrency"], repeat)
        for repeat in shift["run_indices"]
    )
    assert http["protected_evaluation_ready"] is False
    assert workloads["protected_evaluation_ready"] is False


@pytest.mark.parametrize(
    "ordinal, expected",
    [
        (1, ("fixed_cascade", 100, 1, 1)),
        (21, ("fixed_cascade", 100, 64, 1)),
        (25, ("fixed_cascade", 100, 64, 5)),
        (30, ("fixed_cascade", 100, 128, 5)),
        (31, ("fixed_cascade", 10, 1, 1)),
        (61, ("fixed_cascade", 500, 1, 1)),
        (90, ("fixed_cascade", 500, 128, 5)),
        (91, ("transformer_only", 100, 1, 1)),
        (120, ("transformer_only", 100, 128, 5)),
        (121, ("shift_period", None, 1, 1)),
        (125, ("shift_period", None, 1, 5)),
    ],
)
def test_prospective_one_based_boundary_ordinals(schedule_api, ordinal, expected):
    cell = schedule_api.cell_for_ordinal(ordinal)
    assert astuple(cell) == (ordinal, *expected)
    assert cell == schedule_api.planned_cells()[ordinal - 1]
    assert schedule_api.validate_cell(cell) == cell


def test_designated_reference_and_primary_are_distinct(schedule_api):
    cells = schedule_api.planned_cells()
    assert [cell.ordinal for cell in cells if cell.reference_invocations_member] == [1]
    assert [cell.ordinal for cell in cells if cell.primary_http_member] == list(
        range(21, 26)
    )
    assert all(type(cell.reference_invocations_member) is bool for cell in cells)
    assert all(type(cell.primary_http_member) is bool for cell in cells)


def test_cells_have_only_frozen_planned_fields(schedule_api):
    cell = schedule_api.cell_for_ordinal(1)
    assert [member.name for member in fields(cell)] == [
        "ordinal",
        "workload",
        "prevalence_basis_points",
        "concurrency",
        "run_index",
    ]
    with pytest.raises(FrozenInstanceError):
        cell.run_index = 2


def test_schedule_is_parameter_free_and_deterministic(schedule_api):
    assert schedule_api.planned_cells() == schedule_api.planned_cells()
    with pytest.raises(TypeError):
        schedule_api.planned_cells((1,))
    with pytest.raises(TypeError):
        schedule_api.planned_cells(prevalences=(100,))


def test_pure_schedule_never_reads_inputs(schedule_api, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("pure scheduling inspected a supplied path or input")

    monkeypatch.setattr("builtins.open", forbidden)
    monkeypatch.setattr(Path, "read_bytes", forbidden)
    monkeypatch.setattr(Path, "read_text", forbidden)
    for cell in schedule_api.planned_cells():
        assert schedule_api.validate_cell(cell) == cell
        assert cell == schedule_api.cell_for_ordinal(cell.ordinal)
