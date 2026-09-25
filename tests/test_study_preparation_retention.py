import importlib
import json
import stat
from hashlib import sha256

import pytest

from automated_phishing_detection import execution_receipt as receipt

ORDER = (
    "suffix-rules.dat",
    "group_test.jsonl",
    "source-overlap.json",
    "source-reconstruction.json",
    "publisher-source.json",
    "publisher-summary.json",
    "retained-test.jsonl",
    "quarantine.jsonl",
    "inventory.json",
    "preparation-summary.json",
    "feasibility.json",
    "preparation-complete.json",
)
IDENTITY = {"kind": "invented_preparation", "authorized": False}


def module():
    return importlib.import_module(
        "automated_phishing_detection.study_preparation_retention"
    )


def reserved(tmp_path):
    return receipt.reserve_attempt(tmp_path / "attempt", identity=IDENTITY)


def outputs():
    return tuple((name, f"invented:{name}\n".encode()) for name in ORDER)


def append_all(writer):
    for name, content in outputs():
        writer.append(name, content)


def test_fixed_order_and_immutable_finish(tmp_path):
    retained = module()
    attempt = reserved(tmp_path)
    assert retained.PREPARATION_ORDER == ORDER
    with retained.retain_study_preparation(attempt, identity=IDENTITY) as writer:
        append_all(writer)
        snapshot = writer.finish()
        assert type(snapshot) is tuple and snapshot == outputs()
        assert json.loads(writer.snapshot())["status"] == "complete"
    assert set(path.name for path in attempt.directory.iterdir()) == {
        "reservation.json",
        *ORDER,
    }
    for name, content in snapshot:
        assert (attempt.directory / name).read_bytes() == content
        assert stat.S_IMODE((attempt.directory / name).stat().st_mode) == 0o600
    assert not (attempt.directory / "finalize.claim").exists()


def test_pending_snapshot_is_private_bytes_only(tmp_path):
    retained = module()
    attempt = reserved(tmp_path)
    with pytest.raises(RuntimeError, match="invented_stop"):
        with retained.retain_study_preparation(attempt, identity=IDENTITY) as writer:
            writer.append(*outputs()[0])
            progress = json.loads(writer.snapshot())
            assert progress["confirmed_sha256"] == {
                ORDER[0]: sha256(outputs()[0][1]).hexdigest()
            }
            assert progress["pending_checkpoint_bytes"] == {}
            assert progress["reservation_sha256"] == attempt.reservation_sha256
            raise RuntimeError("invented_stop")
    assert json.loads(writer.snapshot())["status"] == "failed"
    assert (attempt.directory / ORDER[0]).is_file()


def test_empty_bytes_are_retained_unchanged(tmp_path):
    retained = module()
    attempt = reserved(tmp_path)
    with retained.retain_study_preparation(attempt, identity=IDENTITY) as writer:
        for name in ORDER:
            writer.append(name, b"")
        assert writer.finish() == tuple((name, b"") for name in ORDER)


def test_identity_is_copied_before_caller_mutation(tmp_path):
    retained = module()
    identity = dict(IDENTITY)
    with retained.retain_study_preparation(
        reserved(tmp_path), identity=identity
    ) as writer:
        identity["kind"] = "mutated"
        append_all(writer)
        assert writer.finish() == outputs()


@pytest.mark.parametrize("operation", ["finish", "leave"])
def test_incomplete_normal_context_rejected(tmp_path, operation):
    retained = module()
    with pytest.raises(retained.StudyPreparationRetentionError):
        with retained.retain_study_preparation(
            reserved(tmp_path), identity=IDENTITY
        ) as writer:
            if operation == "finish":
                writer.finish()
