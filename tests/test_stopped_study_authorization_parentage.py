"""Historical admissions cannot describe a child as its own recorded parent."""

import pytest
from operational_input_fixtures import candidates, manifests
from stopped_study_authorization_fixtures import change_frame, make_stopped, verify
from study_run_record_fixtures import prepared

__all__ = ["candidates", "manifests", "prepared"]


@pytest.mark.parametrize(
    "position", [0, 1, 2, 3], ids=["internal", "external", "service", "client"]
)
def test_accepted_child_cannot_alias_the_common_parent(prepared, manifests, position):
    case = make_stopped(prepared, manifests)
    entries = case.accounting["authorization_ledger"]["admissions"]
    alias = entries[position]["launched_pid"]
    for index in range(len(entries)):
        change_frame(case, index, parent_pid=alias)
    with pytest.raises(ValueError, match="^invalid_stopped_study_authorization$"):
        verify(case)


@pytest.mark.parametrize("position", [4, 5], ids=["stopped_service", "stopped_client"])
def test_stopped_child_cannot_alias_the_common_parent(prepared, manifests, position):
    case = make_stopped(prepared, manifests, stopped_admissions=2)
    entries = case.accounting["authorization_ledger"]["admissions"]
    entries[position]["launched_pid"] = 98765
    for index in range(len(entries)):
        change_frame(case, index, parent_pid=98765)
    with pytest.raises(ValueError, match="^invalid_stopped_study_authorization$"):
        verify(case)


def test_unlaunched_stopped_service_has_no_child_pid_alias(prepared, manifests):
    case = make_stopped(prepared, manifests, stopped_admissions=1)
    entry = case.accounting["authorization_ledger"]["admissions"][-1]
    entry.update(launched_pid=None, exit_observed=False, exit_code=None)
    change_frame(case, -1)
    assert verify(case).accepted_ordinals == (1,)
