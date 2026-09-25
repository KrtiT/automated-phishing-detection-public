import os

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


@pytest.mark.parametrize("name", [ORDER[1], "../unsafe", "", 1, b"name"])
def test_invalid_order_or_name_consumes_writer(tmp_path, name):
    retained = module()
    with pytest.raises(retained.StudyPreparationRetentionError):
        with retained.retain_study_preparation(
            reserved(tmp_path), identity=IDENTITY
        ) as writer:
            with pytest.raises(retained.StudyPreparationRetentionError):
                writer.append(name, b"invented")
    assert json_snapshot(writer)["status"] == "failed"


def json_snapshot(writer):
    import json

    return json.loads(writer.snapshot())


@pytest.mark.parametrize("content", ["text", bytearray(b"x"), memoryview(b"x"), None])
def test_nonexact_bytes_rejected(tmp_path, content):
    retained = module()
    with pytest.raises(retained.StudyPreparationRetentionError):
        with retained.retain_study_preparation(
            reserved(tmp_path), identity=IDENTITY
        ) as writer:
            with pytest.raises(retained.StudyPreparationRetentionError):
                writer.append(ORDER[0], content)


@pytest.mark.parametrize("identity", [{}, {"kind": "wrong"}, [], None, True])
def test_reservation_identity_must_match(tmp_path, identity):
    retained = module()
    with pytest.raises(retained.StudyPreparationRetentionError):
        with retained.retain_study_preparation(reserved(tmp_path), identity=identity):
            pytest.fail("invalid identity accepted")


@pytest.mark.parametrize("name", ["extra", ORDER[0], "finalize.claim", "outcome.json"])
def test_nonempty_attempt_never_resumes(tmp_path, name):
    retained = module()
    attempt = reserved(tmp_path)
    (attempt.directory / name).write_bytes(b"previous")
    with pytest.raises(retained.StudyPreparationRetentionError):
        with retained.retain_study_preparation(attempt, identity=IDENTITY):
            pytest.fail("existing file accepted")
    assert (attempt.directory / name).read_bytes() == b"previous"


def mutate(attempt, kind):
    path = attempt.directory / ORDER[0]
    if kind == "bytes":
        path.write_bytes(b"changed")
    elif kind == "mode":
        path.chmod(0o644)
    elif kind == "hardlink":
        os.link(path, attempt.directory.parent / "alias")
    elif kind == "symlink":
        path.unlink()
        path.symlink_to(attempt.directory / "reservation.json")
    elif kind == "missing":
        path.unlink()
    elif kind == "extra":
        (attempt.directory / "extra").write_bytes(b"unexpected")
    elif kind == "root_mode":
        attempt.directory.chmod(0o755)
    elif kind == "reservation":
        (attempt.directory / "reservation.json").write_bytes(b"changed")
    elif kind == "root_swap":
        attempt.directory.rename(attempt.directory.parent / "moved")
        attempt.directory.mkdir(mode=0o700)


@pytest.mark.parametrize(
    "kind",
    [
        "bytes",
        "mode",
        "hardlink",
        "symlink",
        "missing",
        "extra",
        "root_mode",
        "reservation",
        "root_swap",
    ],
)
@pytest.mark.parametrize("when", ["append", "finish", "exit"])
def test_changed_retained_state_rejected(tmp_path, kind, when):
    retained = module()
    attempt = reserved(tmp_path)
    with pytest.raises(retained.StudyPreparationRetentionError):
        with retained.retain_study_preparation(attempt, identity=IDENTITY) as writer:
            if when == "append":
                writer.append(*outputs()[0])
                mutate(attempt, kind)
                with pytest.raises(retained.StudyPreparationRetentionError):
                    writer.append(*outputs()[1])
            else:
                append_all(writer)
                if when == "exit":
                    writer.finish()
                mutate(attempt, kind)
                if when == "finish":
                    with pytest.raises(retained.StudyPreparationRetentionError):
                        writer.finish()


@pytest.mark.parametrize("operation", ["append", "finish"])
def test_writer_cannot_be_used_after_context_exit(tmp_path, operation):
    retained = module()
    with retained.retain_study_preparation(
        reserved(tmp_path), identity=IDENTITY
    ) as writer:
        append_all(writer)
        writer.finish()
    with pytest.raises(retained.StudyPreparationRetentionError):
        if operation == "append":
            writer.append(*outputs()[0])
        else:
            writer.finish()


def test_wrong_attempt_type_is_symbolic(tmp_path):
    retained = module()
    with pytest.raises(retained.StudyPreparationRetentionError):
        with retained.retain_study_preparation(tmp_path, identity=IDENTITY):
            pytest.fail("wrong attempt accepted")


def test_wrong_reservation_hash_rejected(tmp_path):
    retained = module()
    attempt = reserved(tmp_path)
    changed = receipt.Attempt(attempt.directory, "0" * 64)
    with pytest.raises(retained.StudyPreparationRetentionError):
        with retained.retain_study_preparation(changed, identity=IDENTITY):
            pytest.fail("wrong hash accepted")
