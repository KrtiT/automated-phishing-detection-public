"""Re-read accounting schemas without turning hashes into owned process proof."""

import json

import pytest
from study_run_record_fixtures import api, prepared
from test_study_run_public import hold_checkpoints

from automated_phishing_detection._checkpoint_codec import canonical_bytes

__all__ = ["prepared"]


@pytest.mark.parametrize(
    "change",
    ["order", "bool", "extra", "missing", "unknown_status", "invented_reference"],
)
def test_public_rejects_malformed_cell_accounting(prepared, change):
    checkpoints = list(hold_checkpoints(prepared))
    value = json.loads(checkpoints[-1][1])
    first = value["cells"][0]
    if change == "order":
        value["cells"] = list(reversed(value["cells"]))
    elif change == "bool":
        first["cell"]["ordinal"] = True
    elif change == "extra":
        first["accepted"] = True
    elif change == "missing":
        first.pop("retention")
    elif change == "unknown_status":
        first["status"] = "success"
    else:
        first["observation_sha256"] = "f" * 64
    checkpoints[-1] = ("study-accounting.json", canonical_bytes(value))
    with pytest.raises(ValueError):
        api().root_public_summary(
            execution=prepared.execution, checkpoints=tuple(checkpoints)
        )
