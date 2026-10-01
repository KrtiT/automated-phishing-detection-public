"""Independent snapshot review keeps original authority declarations closed."""

import pytest
from operational_input_fixtures import candidates, manifests
from stopped_study_authorization_fixtures import verify as verify_root
from study_history_snapshot_fixtures import complete_history, verify
from study_history_snapshot_mutation_fixtures import replace_source_record
from study_run_record_fixtures import prepared

__all__ = ["candidates", "manifests", "prepared"]


@pytest.mark.parametrize("role", ["internal", "external"])
def test_republished_source_cannot_claim_protected_authority(prepared, manifests, role):
    case = complete_history(prepared, manifests)
    replace_source_record(
        case,
        role,
        "public-summary.json",
        lambda value: value.update(protected_evaluation_authorized=True),
    )
    verify_root(case)
    with pytest.raises(ValueError, match="invalid_study_history_snapshots"):
        verify(case)
