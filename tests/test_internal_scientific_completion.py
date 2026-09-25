"""Successful source acceptance includes the exact scientific checkpoint boundary."""

import os
from collections import Counter

import pytest
from internal_scientific_fixtures import ORDER
from test_source_completion import (
    inputs as inputs,
)
from test_source_completion import (
    published as published,
)
from test_source_completion import (
    runner as runner,
)
from test_source_completion import (
    verifier as verifier,
)


def test_every_scientific_file_is_safely_read_once(
    verifier: object, runner: object, published: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    binding, paths, unused = published
    original = runner._read_file_once
    reads = Counter()

    def observe(path: object, **kwargs: object) -> bytes:
        reads[path] += 1
        return original(path, **kwargs)

    monkeypatch.setattr(runner, "_read_file_once", observe)
    verifier.verify_internal_completion(binding, paths, producer_exit_code=0)
    assert all(
        reads[paths.attempt / "scientific-checkpoints" / name] == 1 for name in ORDER
    )
    assert all(count == 1 for count in reads.values())


@pytest.mark.parametrize(
    "fault",
    [
        "missing",
        "extra",
        "symlink",
        "hardlink",
        "mode",
        "directory_mode",
        "failure_sidecar",
    ],
)
def test_scientific_inventory_and_descriptor_guards(
    verifier: object, published: tuple, fault: str
) -> None:
    binding, paths, unused = published
    directory = paths.attempt / "scientific-checkpoints"
    target = directory / "primary-scores.jsonl"
    if fault == "missing":
        target.unlink()
    elif fault == "extra":
        (directory / "unexpected").write_bytes(b"private")
    elif fault in ("symlink", "hardlink"):
        target.rename(paths.attempt.parent / "original")
        if fault == "symlink":
            target.symlink_to(paths.attempt.parent / "original")
        else:
            os.link(paths.attempt.parent / "original", target)
    elif fault == "mode":
        target.chmod(0o644)
    elif fault == "directory_mode":
        directory.chmod(0o755)
    else:
        (paths.attempt / "failure-progress.json").write_bytes(b"{}\n")
    with pytest.raises(verifier.CompletionVerificationError):
        verifier.verify_internal_completion(binding, paths, producer_exit_code=0)
