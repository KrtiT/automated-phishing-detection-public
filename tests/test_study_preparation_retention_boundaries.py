import json
import os
from contextlib import contextmanager

import pytest
from test_study_preparation_retention import (
    IDENTITY,
    ORDER,
    append_all,
    module,
    outputs,
    reserved,
)

from automated_phishing_detection import execution_receipt as receipt


@pytest.mark.parametrize("target", ["root", "reservation"])
@pytest.mark.parametrize("mode", [0o755, 0o644, 0o777])
def test_initial_modes_are_exact(tmp_path, target, mode):
    retained = module()
    attempt = reserved(tmp_path)
    path = (
        attempt.directory
        if target == "root"
        else attempt.directory / "reservation.json"
    )
    path.chmod(mode)
    with pytest.raises(retained.StudyPreparationRetentionError):
        with retained.retain_study_preparation(attempt, identity=IDENTITY):
            pytest.fail("insecure initial mode accepted")


@pytest.mark.parametrize(
    "kind", ["root_symlink", "reservation_symlink", "reservation_link"]
)
def test_initial_aliases_rejected(tmp_path, kind):
    retained = module()
    attempt = reserved(tmp_path)
    reservation = attempt.directory / "reservation.json"
    if kind == "root_symlink":
        attempt.directory.rename(tmp_path / "moved")
        attempt.directory.symlink_to(tmp_path / "moved", target_is_directory=True)
    elif kind == "reservation_symlink":
        reservation.rename(tmp_path / "moved")
        reservation.symlink_to(tmp_path / "moved")
    else:
        os.link(reservation, tmp_path / "alias")
    with pytest.raises(retained.StudyPreparationRetentionError):
        with retained.retain_study_preparation(attempt, identity=IDENTITY):
            pytest.fail("initial alias accepted")


@pytest.mark.parametrize(
    "operation", ["duplicate", "append_after_finish", "finish_twice"]
)
def test_writer_never_repeats_an_operation(tmp_path, operation):
    retained = module()
    attempt = reserved(tmp_path)
    with pytest.raises(retained.StudyPreparationRetentionError):
        with retained.retain_study_preparation(attempt, identity=IDENTITY) as writer:
            if operation == "duplicate":
                writer.append(*outputs()[0])
            else:
                append_all(writer)
                writer.finish()
            with pytest.raises(retained.StudyPreparationRetentionError):
                if operation == "finish_twice":
                    writer.finish()
                else:
                    writer.append(*outputs()[0])
    assert (attempt.directory / ORDER[0]).read_bytes() == outputs()[0][1]


@pytest.mark.parametrize("fail", [False, True])
def test_attempt_descriptor_is_held_then_closed(tmp_path, fail):
    retained = module()
    attempt = reserved(tmp_path)
    try:
        with retained.retain_study_preparation(attempt, identity=IDENTITY) as writer:
            descriptor = writer._directory.descriptor
            expected = attempt.directory.stat().st_ino
            writer.append(*outputs()[0])
            assert os.fstat(descriptor).st_ino == expected
            if fail:
                raise RuntimeError("invented_stop")
            for name, content in outputs()[1:]:
                writer.append(name, content)
            writer.finish()
            assert os.fstat(descriptor).st_ino == expected
    except RuntimeError:
        assert fail
    with pytest.raises(OSError):
        os.fstat(descriptor)


def test_ordinary_cleanup_failure_invalidates_finished_writer(tmp_path, monkeypatch):
    retained = module()
    attempt = reserved(tmp_path)
    original = receipt._directory

    @contextmanager
    def failed_cleanup(path):
        with original(path) as directory:
            yield directory
        raise OSError("private_cleanup_detail")

    monkeypatch.setattr(receipt, "_directory", failed_cleanup)
    with pytest.raises(retained.StudyPreparationRetentionError) as caught:
        with retained.retain_study_preparation(attempt, identity=IDENTITY) as writer:
            append_all(writer)
            writer.finish()
    assert "private_cleanup_detail" not in str(caught.value)
    assert json.loads(writer.snapshot())["status"] == "failed"
    assert all((attempt.directory / name).is_file() for name in ORDER)


def test_partial_next_write_keeps_confirmed_predecessors(tmp_path, monkeypatch):
    retained = module()
    attempt = reserved(tmp_path)
    original = retained.files.write_file

    def partial(directory, name, content, mode):
        original(directory, name, content[:2], mode)
        raise OSError("invented_failure")

    with pytest.raises(retained.StudyPreparationRetentionError):
        with retained.retain_study_preparation(attempt, identity=IDENTITY) as writer:
            writer.append(*outputs()[0])
            monkeypatch.setattr(retained.files, "write_file", partial)
            writer.append(*outputs()[1])
    progress = json.loads(writer.snapshot())
    assert set(progress["confirmed_sha256"]) == {ORDER[0]}
    assert set(progress["pending_checkpoint_bytes"]) == {ORDER[1]}
    assert (attempt.directory / ORDER[0]).read_bytes() == outputs()[0][1]
    assert (attempt.directory / ORDER[1]).read_bytes() == outputs()[1][1][:2]
