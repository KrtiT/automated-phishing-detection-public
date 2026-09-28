"""Regression isolated to real capacity-success manifest JSON reconstruction."""

from dataclasses import asdict, dataclass

import pytest
from study_feasibility_fixtures import internal_rows

from automated_phishing_detection import saved_evidence
from automated_phishing_detection._checkpoint_codec import canonical_bytes


@dataclass
class RoutingFixture:
    invented: bool = True


@pytest.mark.parametrize("change", [None, "reverse", "drop", "url"])
def test_complete_manifest_json_roundtrip_preserves_ordered_records(
    monkeypatch, change
):
    records = internal_rows(negatives=9990, positives=500)
    rows = tuple({"record": asdict(record)} for record in records)
    outcomes = saved_evidence._manifests(records)
    assert all(outcome.status == "prepared" for outcome in outcomes.values())
    manifests = _manifest_bytes(outcomes, change)
    monkeypatch.setattr(saved_evidence, "_bindings", lambda content: {})
    monkeypatch.setattr(saved_evidence, "_parse_rows", lambda content, bound: rows)
    monkeypatch.setattr(saved_evidence, "_verify_monitor_path", lambda *args: None)
    monkeypatch.setattr(saved_evidence, "_routing", lambda *args: RoutingFixture())
    arguments = (
        b"invented prevalidated predictions",
        manifests,
        b"{}",
        canonical_bytes({"invented": True}),
    )
    if change is not None:
        with pytest.raises(
            saved_evidence.SavedEvidenceError, match="saved manifests differ"
        ):
            saved_evidence._validated_internal_inputs(*arguments)
        return
    restored = saved_evidence._validated_internal_inputs(*arguments)
    assert restored[2] == records
    assert restored[3] == outcomes


def _manifest_bytes(outcomes, change):
    values = {str(key): asdict(value) for key, value in outcomes.items()}
    records = values["100"]["manifest"]["records"]
    if change == "reverse":
        values["100"]["manifest"]["records"] = records[::-1]
    elif change == "drop":
        values["100"]["manifest"]["records"] = records[1:]
    elif change == "url":
        records[0]["raw_url"] = "https://changed.test/invented"
    return canonical_bytes(values)
