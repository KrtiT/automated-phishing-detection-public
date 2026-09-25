"""Secondary failures retain completed columns without retries or invented counts."""

import asyncio
import json
from dataclasses import replace
from types import ModuleType, SimpleNamespace

import pytest
from bound_secondary_column_fixtures import scoring as scoring
from external_secondary_fixtures import CHECKPOINTS
from external_secondary_fixtures import module as module
from external_secondary_fixtures import phase as phase

from automated_phishing_detection import bound_secondary


@pytest.mark.parametrize("fail_name", (*CHECKPOINTS, "secondary-completion.json"))
def test_writer_failure_keeps_exact_ambiguous_bytes_and_never_retries(
    module: ModuleType,
    phase: SimpleNamespace,
    fail_name: str,
) -> None:
    writes = []

    def retain(name: str, content: bytes) -> None:
        writes.append((name, content))
        if name == fail_name:
            raise OSError("private-storage-path")

    with pytest.raises(module.ExternalSecondaryError) as caught:
        module.score_external_secondary(phase.primary, phase.bound, retain=retain)
    progress = json.loads(caught.value.progress)
    assert progress["stage"] == "retention"
    assert progress["retention_status"] == "failed_or_ambiguous"
    assert progress["current_checkpoint"] == fail_name
    assert {
        name: content.encode("ascii")
        for name, content in progress["completed_checkpoints"].items()
    } == dict(writes)
    completed = min(len(writes), 12)
    assert progress["unattempted_members"] == list(CHECKPOINTS[completed:])
    assert (
        len(writes) == (*CHECKPOINTS, "secondary-completion.json").index(fail_name) + 1
    )
    assert "private-storage" not in str(caught.value)


@pytest.mark.parametrize(
    "member,completed", [("formatting", 0), ("permutation_44", 3), (44, 9)]
)
def test_mid_member_failure_marks_unknown_counts_and_exact_unattempted_inventory(
    module: ModuleType,
    phase: SimpleNamespace,
    member: str | int,
    completed: int,
) -> None:
    phase.fail_at = ("tabular" if isinstance(member, str) else "score", member)
    with pytest.raises(module.ExternalSecondaryError) as caught:
        module.score_external_secondary(phase.primary, phase.bound)
    progress = json.loads(caught.value.progress)
    assert progress["stage"] == "scoring"
    assert progress["current_member"] == CHECKPOINTS[completed]
    assert set(progress["completed_checkpoints"]) == set(CHECKPOINTS[:completed])
    assert progress["unattempted_members"] == list(CHECKPOINTS[completed + 1 :])
    assert progress["current_member_physical_counts"] is None
    assert progress["current_member_counts_reason"] == "physical_counts_unavailable"
    assert "secondary-completion.json" not in progress["completed_checkpoints"]


@pytest.mark.parametrize(
    "exception_type", [KeyboardInterrupt, SystemExit, asyncio.CancelledError]
)
def test_cancellation_preserves_original_exception_and_completed_bytes(
    module: ModuleType,
    phase: SimpleNamespace,
    exception_type: type[BaseException],
) -> None:
    cancellation = exception_type("private-source-value")

    def retain(name: str, content: bytes) -> None:
        raise cancellation

    with pytest.raises(exception_type) as caught:
        module.score_external_secondary(phase.primary, phase.bound, retain=retain)
    assert caught.value is cancellation
    assert "private-source" not in str(caught.value)
    progress = json.loads(cancellation.progress)
    assert set(progress["completed_checkpoints"]) == {CHECKPOINTS[0]}
    assert phase.events == [("tabular", "formatting")]


def test_final_result_must_agree_with_every_completed_column(
    module: ModuleType,
    phase: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = bound_secondary.score_bound_secondary

    def corrupt(*args: object, **kwargs: object) -> bound_secondary.SecondaryScoring:
        result = original(*args, **kwargs)
        first = result.rows[0]
        changed = replace(first.tabular[0], probability=0.99)
        row = replace(first, tabular=(changed, *first.tabular[1:]))
        return replace(result, rows=(row, *result.rows[1:]))

    monkeypatch.setattr(bound_secondary, "score_bound_secondary", corrupt)
    with pytest.raises(module.ExternalSecondaryError) as caught:
        module.score_external_secondary(phase.primary, phase.bound)
    progress = json.loads(caught.value.progress)
    assert progress["stage"] == "completion_validation"
    assert set(progress["completed_checkpoints"]) == set(CHECKPOINTS)
