import json
import stat
from dataclasses import FrozenInstanceError

import pytest
from study_root_retention_fixtures import api, append_all, complete, manager, root_case


@pytest.mark.parametrize("success, count", [(False, 10), (True, 14)])
def test_actual_receipt_roundtrip_preserves_root_bytes(tmp_path, success, count):
    module, case = api(), root_case(tmp_path, success=success)
    with manager(module, case) as writer:
        append_all(writer, case)
        snapshot = complete(writer, case)
        assert writer.publishing is True
        assert writer.candidate is snapshot
        assert writer.payloads == tuple(case.contents.items())
    assert snapshot.reservation_sha256 == case.attempt.reservation_sha256
    assert len(snapshot.payloads) == count
    assert json.loads(snapshot.payload("public-summary.json")) == case.public
    for name, content in case.contents.items():
        assert snapshot.payload("attempt/" + name) == content
        assert snapshot.payload("attempt/evidence/" + name) == content
    assert stat.S_IMODE(case.public_path.stat().st_mode) == 0o644
    with pytest.raises(FrozenInstanceError):
        snapshot.payloads = ()


@pytest.mark.parametrize("completed", [False, True])
def test_closed_writer_rejects_reuse(tmp_path, completed):
    module, case = api(), root_case(tmp_path)
    with manager(module, case) as writer:
        append_all(writer, case)
        complete(writer, case)
        if completed:
            with pytest.raises(module.StudyRootRetentionError):
                complete(writer, case)
    with pytest.raises(module.StudyRootRetentionError):
        writer.append("study-intent.json", b"again")


def test_original_failure_retains_partial_checkpoint_without_finalizing(tmp_path):
    module, case = api(), root_case(tmp_path)
    original = RuntimeError("invented")
    with pytest.raises(RuntimeError) as caught:
        with manager(module, case) as writer:
            writer.append("study-intent.json", case.contents["study-intent.json"])
            raise original
    assert caught.value is original
    assert set(path.name for path in case.attempt.directory.iterdir()) == {
        "reservation.json",
        "study-intent.json",
    }
    assert not case.public_path.exists()
    assert writer.publishing is False


def test_normal_exit_requires_candidate(tmp_path):
    module, case = api(), root_case(tmp_path)
    with pytest.raises(module.StudyRootRetentionError):
        with manager(module, case):
            pass
