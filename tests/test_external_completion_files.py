"""Once-read external completion trees contain invented bytes, not measurements."""

import importlib
import importlib.util
import os
from collections import Counter
from dataclasses import FrozenInstanceError

import pytest

from automated_phishing_detection import execution_receipt, source_runner
from automated_phishing_detection._external_checkpoint_protocol import (
    PROVENANCE_ORDER,
    SCIENTIFIC_ORDER,
)
from automated_phishing_detection._external_source_checkpoints import (
    ExternalCheckpointWriter,
)


def test_external_completion_snapshot_api_exists():
    assert importlib.util.find_spec(
        "automated_phishing_detection._external_completion_files"
    ), "missing external completion snapshot"


@pytest.fixture
def files_api():
    name = "automated_phishing_detection._external_completion_files"
    assert importlib.util.find_spec(name), "missing external completion snapshot"
    return importlib.import_module(name)


@pytest.fixture
def completed_tree(tmp_path):
    identity = {"fixture": "external-completion-files"}
    attempt = execution_receipt.reserve_attempt(tmp_path / "attempt", identity=identity)
    writer = ExternalCheckpointWriter(attempt, identity=identity)
    writer.begin({name: name.encode() for name in PROVENANCE_ORDER})
    scientific = {name: name.encode() for name in SCIENTIFIC_ORDER}
    for name, content in scientific.items():
        writer(name, content)
    public = tmp_path / "public.json"
    execution_receipt.publish_completion(
        attempt,
        private_outputs=writer.complete(scientific),
        public_summary={"status": "invented"},
        public_path=public,
    )
    return attempt.directory, public


def expected_files(attempt, public):
    return {
        "attempt/reservation.json": attempt / "reservation.json",
        "attempt/finalize.claim": attempt / "finalize.claim",
        "attempt/outcome.json": attempt / "outcome.json",
        "public-summary.json": public,
        **{
            f"attempt/{directory}/{name}": attempt / directory / name
            for directory in ("checkpoints", "evidence")
            for name in (*PROVENANCE_ORDER, *SCIENTIFIC_ORDER)
        },
    }


def test_exact_immutable_snapshot_reads_each_file_once(
    files_api, completed_tree, monkeypatch
):
    attempt, public = completed_tree
    expected = expected_files(attempt, public)
    reader = source_runner._read_file_once
    reads = Counter()

    def counted(filename, **options):
        reads[filename] += 1
        assert options["expected_state"] is not None
        return reader(filename, **options)

    monkeypatch.setattr(source_runner, "_read_file_once", counted)
    with files_api.snapshot_external_files(attempt, public) as snapshot:
        assert type(snapshot) is files_api.ExternalFileSnapshot
        assert type(snapshot.payloads) is tuple
        assert dict(snapshot.payloads) == {
            name: filename.read_bytes() for name, filename in expected.items()
        }
        with pytest.raises(FrozenInstanceError):
            snapshot.payloads = ()
    assert reads == Counter({filename: 1 for filename in expected.values()})
    assert len(snapshot.payloads) == 76


@pytest.mark.parametrize("directory", ["", "checkpoints", "evidence"])
@pytest.mark.parametrize("change", ["missing", "extra", "permissions"])
def test_initial_tree_rejects_before_any_payload_read(
    files_api, completed_tree, monkeypatch, directory, change
):
    attempt, public = completed_tree
    target = attempt / directory
    if change == "permissions":
        target.chmod(0o755)
    elif change == "extra":
        (target / "unexpected").write_bytes(b"not accepted")
    else:
        (target / ("outcome.json" if not directory else "bindings.json")).unlink()

    def forbidden(*args, **kwargs):
        pytest.fail("invalid tree reached payload read")

    monkeypatch.setattr(source_runner, "_read_file_once", forbidden)
    with pytest.raises(files_api.ExternalCompletionFileError):
        with files_api.snapshot_external_files(attempt, public):
            pytest.fail("invalid tree yielded")


def test_public_parent_may_be_shared(files_api, completed_tree):
    attempt, public = completed_tree
    public.parent.chmod(0o755)
    (public.parent / "unrelated").write_bytes(b"keep")
    with files_api.snapshot_external_files(attempt, public) as snapshot:
        assert len(snapshot.payloads) == 76
    assert (public.parent / "unrelated").read_bytes() == b"keep"


def test_directory_descriptors_remain_open_until_final_checks(
    files_api, completed_tree, monkeypatch
):
    attempt, public = completed_tree
    opened = []
    original = execution_receipt._open_directory

    def tracked(filename):
        descriptor = original(filename)
        opened.append((filename, descriptor))
        return descriptor

    monkeypatch.setattr(execution_receipt, "_open_directory", tracked)
    with files_api.snapshot_external_files(attempt, public):
        for directory in (attempt, attempt / "checkpoints", attempt / "evidence"):
            descriptor = next(value for name, value in opened if name == directory)
            assert os.fstat(descriptor).st_ino == directory.stat().st_ino
    for directory in (attempt, attempt / "checkpoints", attempt / "evidence"):
        descriptor = next(value for name, value in opened if name == directory)
        with pytest.raises(OSError):
            os.fstat(descriptor)
