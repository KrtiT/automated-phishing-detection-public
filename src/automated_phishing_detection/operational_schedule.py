"""Pure fixed workload schedule with prospectively declared one-based ordinals.

Cells establish planned membership, not source authenticity or access authority.
Capacity, reservations, process ownership and outcomes are separate concerns.
"""

from dataclasses import dataclass
from typing import Literal

_PREVALENCES = (100, 10, 500)
_CONCURRENCIES = (1, 8, 16, 32, 64, 128)


@dataclass(frozen=True)
class OperationalCell:
    ordinal: int
    workload: Literal["fixed_cascade", "transformer_only", "shift_period"]
    prevalence_basis_points: int | None
    concurrency: int
    run_index: int

    @property
    def reference_invocations_member(self) -> bool:
        return validate_cell(self).ordinal == 1

    @property
    def primary_http_member(self) -> bool:
        return validate_cell(self).ordinal in range(21, 26)


def cell_for_ordinal(ordinal: int) -> OperationalCell:
    """Return the specified planned cell, never an alternative or replacement."""
    if type(ordinal) is not int or not 1 <= ordinal <= 125:
        raise ValueError("invalid_operational_ordinal")
    position = ordinal - 1
    if position < 90:
        workload = "fixed_cascade"
        prevalence = _PREVALENCES[position // 30]
        position %= 30
    elif position < 120:
        workload = "transformer_only"
        prevalence = 100
        position -= 90
    else:
        return OperationalCell(ordinal, "shift_period", None, 1, position - 119)
    return OperationalCell(
        ordinal, workload, prevalence, _CONCURRENCIES[position // 5], position % 5 + 1
    )


def planned_cells() -> tuple[OperationalCell, ...]:
    """Return all 125 cells in contract order, independently of available inputs."""
    return tuple(cell_for_ordinal(ordinal) for ordinal in range(1, 126))


def validate_cell(cell: OperationalCell) -> OperationalCell:
    """Check exact schedule consistency; a matching caller record proves no exit."""
    if (
        type(cell) is not OperationalCell
        or type(cell.workload) is not str
        or any(
            type(value) is not int
            for value in (cell.ordinal, cell.concurrency, cell.run_index)
        )
        or not 1 <= cell.ordinal <= 125
        or (
            cell.prevalence_basis_points is not None
            and type(cell.prevalence_basis_points) is not int
        )
    ):
        raise ValueError("invalid_operational_cell")
    if cell != cell_for_ordinal(cell.ordinal):
        raise ValueError("invalid_operational_cell")
    return cell
