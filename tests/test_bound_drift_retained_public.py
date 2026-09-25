"""The pre-owner loader retains its already-read public buffers unchanged."""

from types import ModuleType, SimpleNamespace

import pytest
from test_bound_drift import drift as drift
from test_bound_drift import prepared as prepared


def test_bound_loader_retains_existing_reads_in_exact_public_order(
    drift: ModuleType, prepared: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    reads = []
    original = drift.execution_preflight._read_regular

    def observe(root: object, relative: str) -> bytes:
        content = original(root, relative)
        reads.append((relative, content))
        return content

    monkeypatch.setattr(drift.execution_preflight, "_read_regular", observe)
    result = drift.load_bound_drift(prepared.binding, prepared.paths, prepared.models)
    assert tuple(name for name, content in reads) == (
        drift._REPORT,
        drift._PREPARATION,
        drift._SOURCE,
    )
    assert result.public_inputs == tuple(reads)
    assert all(
        retained is original
        for (_, retained), (_, original) in zip(
            result.public_inputs, reads, strict=True
        )
    )
