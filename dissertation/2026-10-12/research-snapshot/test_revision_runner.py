"""Runner sequencing tests never open research inputs."""

import json

import pytest
import revision_runner as runner
from revision_admission import TASKS
from test_revision_admission import make_parent


def test_runner_calls_every_task_once_and_finalizes(tmp_path):
    parent, unused = make_parent(tmp_path)
    called = []

    def task(callback):
        callback()
        called.append(parent.active)

    runner.execute(parent, dict.fromkeys(TASKS, task))
    assert called == list(TASKS)
    assert (parent.root / "parent-completed.json").exists()


def test_task_error_preserves_prefix_and_does_not_continue(tmp_path):
    parent, unused = make_parent(tmp_path)
    called = []

    def task(callback):
        callback()
        called.append(parent.active)
        if parent.active == "F":
            raise RuntimeError("fixture")

    with pytest.raises(RuntimeError):
        runner.execute(parent, dict.fromkeys(TASKS, task))
    assert called == ["P", "F"]
    failure = json.loads((parent.root / "parent-failure.json").read_bytes())
    assert failure["completed"] == ["P"] and failure["unstarted"] == ["R", "C"]


def test_missing_task_never_opens_parent(tmp_path):
    parent, unused = make_parent(tmp_path)
    with pytest.raises(ValueError, match="all_task_implementations_required"):
        runner.execute(parent, {"P": lambda callback: pytest.fail("ran")})
    assert not parent.root.exists()
