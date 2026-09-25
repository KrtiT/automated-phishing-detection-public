"""Create-only source checkpoints installed before any internal scoring."""

from hashlib import sha256

from . import execution_receipt
from .evaluation_producer import _json_bytes

CHECKPOINT_NAMES = frozenset(
    {"group_test.jsonl", "source-overlap.json", "source-reconstruction.json"}
)


def _hashes(contents):
    return {name: sha256(content).hexdigest() for name, content in contents.items()}


def _contents(attempt, identity, reconstructed):
    outputs = {
        "group_test.jsonl": reconstructed.group_test_bytes,
        "source-overlap.json": reconstructed.private_outputs["source-overlap.json"],
    }
    outputs["source-reconstruction.json"] = _json_bytes(
        {
            "schema_version": 1,
            "execution": identity,
            "reservation_sha256": attempt.reservation_sha256,
            "reconstruction": reconstructed.public_summary,
            "checkpoint_sha256": _hashes(outputs),
        }
    )
    return outputs


def retain_source_checkpoints(attempt, identity, reconstructed):
    outputs = _contents(attempt, identity, reconstructed)
    with execution_receipt._attempt_directory(attempt) as directory:
        execution_receipt._require_absent(directory, "checkpoints")
        with execution_receipt._staging_directory(directory, "checkpoints") as (
            staging,
            name,
        ):
            for filename, content in sorted(outputs.items()):
                execution_receipt._write_file(staging, filename, content, 0o600)
            execution_receipt._sync_directory(staging)
            execution_receipt._publish(directory, name, directory, "checkpoints")
            execution_receipt._sync_directory(directory)
    return _hashes(outputs)
