"""The entire fixed schedule and all original accounting schemas are mandatory."""

import copy

import pytest
from operational_input_fixtures import candidates, manifests
from stopped_study_authorization_fixtures import (
    make_stopped,
    refresh_accounting,
    verify,
)
from study_run_record_fixtures import prepared

__all__ = ["candidates", "manifests", "prepared"]


@pytest.mark.parametrize(
    "change",
    [
        "duplicate",
        "short",
        "skipped",
        "later_accepted",
        "unpacked",
        "absent_stop",
        "second_stop",
        "zero_prefix",
        "full_prefix",
        "unknown_cell",
        "bool_ordinal",
    ],
)
def test_prefix_and_suffix_are_exact(prepared, manifests, change):
    case = make_stopped(prepared, manifests, prefix=2)
    cells = case.scientific["cells"]
    if change == "duplicate":
        cells[1] = copy.deepcopy(cells[0])
    elif change == "short":
        cells.pop()
    elif change == "skipped":
        cells[0], cells[1] = cells[1], cells[0]
    elif change == "later_accepted":
        cells[3] = copy.deepcopy(cells[0])
    elif change == "unpacked":
        cells[0].update(
            retention="unpacked", snapshot_sha256=None, reservation_sha256=None
        )
    elif change == "absent_stop":
        cells[2].update(status="unattempted", stage=None)
    elif change == "second_stop":
        cells[3].update(status="stopped", stage="cell_execution")
    elif change in ("zero_prefix", "full_prefix"):
        case = make_stopped(
            prepared, manifests, prefix=0 if change == "zero_prefix" else 125
        )
    elif change == "unknown_cell":
        cells[0]["unexpected"] = True
    else:
        cells[0]["cell"]["ordinal"] = True
    refresh_accounting(case)
    with pytest.raises(ValueError):
        verify(case)


@pytest.mark.parametrize(
    "level,field,value",
    [
        ("adopted", "status", "matrix_accepted"),
        ("adopted", "schema_version", True),
        ("adopted", "unknown", True),
        ("scientific", "status", "matrix_accepted"),
        ("scientific", "stage", "cell_compaction"),
        ("scientific", "internal_status", "unattempted"),
        ("scientific", "external_status", "stopped"),
        ("scientific", "unknown", True),
    ],
)
def test_accounting_closed_schema_and_failure_state(
    prepared, manifests, level, field, value
):
    case = make_stopped(prepared, manifests)
    target = case.accounting if level == "adopted" else case.scientific
    target[field] = value
    refresh_accounting(case)
    with pytest.raises(ValueError):
        verify(case)


@pytest.mark.parametrize("level", ["adopted", "scientific"])
def test_accounting_execution_must_match_reserved_root(prepared, manifests, level):
    case = make_stopped(prepared, manifests)
    target = case.accounting if level == "adopted" else case.scientific
    target["execution"]["reservation_sha256"] = "0" * 64
    refresh_accounting(case)
    with pytest.raises(ValueError):
        verify(case)
