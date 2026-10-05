import hashlib
import importlib.util
import io
import json
import tarfile
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/research_archive.py"


def load_script():
    spec = importlib.util.spec_from_file_location("research_archive", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_scientific_bytes_preserved_and_duplicates_mapped(tmp_path):
    module = load_script()
    source = tmp_path / "source"
    source.mkdir()
    payload = b'{"score":0.125,"raw_url":"https://fixture.example/a"}\n'
    (source / "predictions.jsonl").write_bytes(payload)
    (source / "copy.jsonl").write_bytes(payload)
    archive = tmp_path / "family.tar.gz"
    result = module.build(source, ["predictions.jsonl", "copy.jsonl"], archive)
    assert result["files"] == 2
    assert result["blobs"] == 1
    with tarfile.open(archive) as reader:
        assert (
            result["inventory_sha256"]
            == hashlib.sha256(reader.extractfile("inventory.jsonl").read()).hexdigest()
        )
    restored = tmp_path / "restored"
    assert module.verify(archive, restored)["files"] == 2
    assert (restored / "predictions.jsonl").read_bytes() == payload


def test_duplicate_materialization_reads_each_blob_only_twice(tmp_path, monkeypatch):
    module = load_script()
    for name in ("first", "second", "third"):
        (tmp_path / name).write_text("same scientific bytes")
    archive = tmp_path / "family.tar.gz"
    module.build(tmp_path, ["first", "second", "third"], archive)
    original_extract = tarfile.TarFile.extractfile
    calls = []

    def extract(reader, member):
        calls.append(member if isinstance(member, str) else member.name)
        return original_extract(reader, member)

    monkeypatch.setattr(tarfile.TarFile, "extractfile", extract)
    module.verify(archive, tmp_path / "restored")
    assert sum(name.startswith("blobs/") for name in calls) == 2


def test_metadata_projection_preserves_values_and_records_source_hash(tmp_path):
    module = load_script()
    source = tmp_path / "source"
    source.mkdir()
    payload = json.dumps(
        {
            "score": 0.4,
            "frame_bytes": "c2VjcmV0",
            "path": "/Users/person/private",
            "processes": [{"command": "private"}],
        }
    ).encode()
    (source / "accounting.json").write_bytes(payload)
    archive = tmp_path / "family.tar.gz"
    module.build(source, ["accounting.json"], archive)
    with tarfile.open(archive) as reader:
        entry = json.loads(reader.extractfile("inventory.jsonl").read())
        assert entry["original_sha256"] == hashlib.sha256(payload).hexdigest()
        assert entry["status"] == "administrative_projection"
        public = reader.extractfile("blobs/" + entry["public_sha256"]).read()
        assert b"/Users/" not in public and b"c2VjcmV0" not in public
        assert json.loads(public)["score"] == 0.4


def test_private_logs_are_accounted_for_not_published(tmp_path):
    module = load_script()
    source = tmp_path / "source"
    source.mkdir()
    (source / "stderr.log").write_text("private host details")
    archive = tmp_path / "family.tar.gz"
    module.build(source, ["stderr.log"], archive)
    assert module.verify(archive)["withheld"] == 1


def test_local_path_keys_are_projected_without_losing_distinct_entries(tmp_path):
    module = load_script()
    context = Path("/Users/person/project/.context")
    payload = json.dumps(
        {
            str(context / "result.json"): {"score": 0.5},
            "/Users/person/first": 1,
            "/Users/person/second": 2,
        }
    ).encode()
    output, status, _ = module.public_bytes(Path("inventory.json"), payload, context)
    projected = json.loads(output)
    assert status == "administrative_projection"
    assert b"/Users/" not in output
    assert projected["archive://result.json"] == {"score": 0.5}
    assert len(projected) == 3


def test_projected_key_collision_is_rejected():
    module = load_script()
    context = Path("/Users/person/project/.context")
    with pytest.raises(ValueError, match="collision"):
        module.project({str(context / "same"): 1, "archive://same": 2}, context)


def test_materialization_rejects_inventory_symlink_before_writing(tmp_path):
    module = load_script()
    source = tmp_path / "source"
    source.mkdir()
    (source / "result.json").write_text('{"result":1}')
    archive = tmp_path / "family.tar.gz"
    module.build(source, ["result.json"], archive)
    destination, outside = tmp_path / "restored", tmp_path / "outside"
    destination.mkdir()
    outside.mkdir()
    (destination / ".archive-inventories").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        module.verify(archive, destination)
    assert list(outside.iterdir()) == []
    assert not (destination / "result.json").exists()


def test_materialization_never_overwrites_existing_files(tmp_path):
    module = load_script()
    (tmp_path / "result.json").write_text("new")
    archive = tmp_path / "family.tar.gz"
    module.build(tmp_path, ["result.json"], archive)
    destination = tmp_path / "restored"
    destination.mkdir()
    (destination / "result.json").write_text("old")
    with pytest.raises(ValueError, match="already exists"):
        module.verify(archive, destination)
    assert (destination / "result.json").read_text() == "old"


def test_hyphenated_url_slug_is_not_an_api_credential(tmp_path):
    module = load_script()
    payload = json.dumps(
        {"raw_url": "https://fixture.example/sk-" + "word-" * 18}
    ).encode()
    output, status, _ = module.public_bytes(Path("prediction.json"), payload, tmp_path)
    assert output == payload and status == "exact"


def test_real_credential_shape_requires_manual_review(tmp_path):
    module = load_script()
    with pytest.raises(ValueError, match="Credential-like"):
        module.public_bytes(
            Path("metadata.json"), b'{"token":"ghp_' + b"A" * 36 + b'"}', tmp_path
        )


@pytest.mark.parametrize(
    "key",
    [
        "argv",
        "command_line",
        "cmdline",
        "additional_live_parent_lookup",
        "exclusive_setup_question",
    ],
)
def test_process_fields_trigger_projection_without_a_local_path(tmp_path, key):
    module = load_script()
    output, status, _ = module.public_bytes(
        Path("metadata.json"), json.dumps({key: "private process"}).encode(), tmp_path
    )
    assert status == "administrative_projection"
    assert json.loads(output)[key]["withheld"] == "private execution metadata"


@pytest.mark.parametrize("name", ["../escape", "/absolute", "./result.json", "."])
def test_unsafe_inventory_paths_never_materialize(tmp_path, name):
    module = load_script()
    payload = b"value"
    digest = hashlib.sha256(payload).hexdigest()
    entry = {
        "path": name,
        "status": "exact",
        "original_sha256": digest,
        "public_sha256": digest,
        "original_size": 5,
        "public_size": 5,
    }
    archive = tmp_path / "bad.tar.gz"
    with tarfile.open(archive, "w:gz") as writer:
        module.add_bytes(writer, "inventory.jsonl", json.dumps(entry).encode())
        module.add_bytes(writer, "blobs/" + digest, payload)
    with pytest.raises(ValueError, match="Unsafe"):
        module.verify(archive, tmp_path / "restored")
    assert not (tmp_path / "restored").exists()


def test_duplicate_archive_members_are_rejected(tmp_path):
    module = load_script()
    archive = tmp_path / "bad.tar.gz"
    with tarfile.open(archive, "w:gz") as writer:
        module.add_bytes(writer, "inventory.jsonl", b"")
        module.add_bytes(writer, "inventory.jsonl", b"")
    with pytest.raises(ValueError, match="Duplicate"):
        module.verify(archive)


@pytest.mark.parametrize(
    "names",
    [
        ["a", "a/b"],
        ["A", "a"],
        ["e\u0301", "é"],
        [".archive-inventories/foreign.jsonl"],
    ],
)
def test_materialization_path_graph_rejected_before_writes(tmp_path, names):
    module = load_script()
    payload = b"value"
    digest = hashlib.sha256(payload).hexdigest()
    entries = [
        {
            "path": name,
            "status": "exact",
            "original_sha256": digest,
            "public_sha256": digest,
            "original_size": len(payload),
            "public_size": len(payload),
        }
        for name in names
    ]
    archive = tmp_path / "bad.tar.gz"
    with tarfile.open(archive, "w:gz") as writer:
        module.add_bytes(
            writer,
            "inventory.jsonl",
            b"\n".join(json.dumps(entry).encode() for entry in entries),
        )
        module.add_bytes(writer, "blobs/" + digest, payload)
    with pytest.raises(ValueError, match="path|namespace"):
        module.verify(archive, tmp_path / "restored")
    assert not (tmp_path / "restored").exists()


def test_unlisted_archive_is_rejected_before_materialization(tmp_path):
    module = load_script()
    (tmp_path / "result.json").write_text("data")
    archive = tmp_path / "family.tar.gz"
    record = module.build(tmp_path, ["result.json"], archive)
    catalog = tmp_path / "catalog.json"
    catalog.write_text(json.dumps({"families": {archive.name: record}}))
    extra = tmp_path / "stale.tar.gz"
    extra.write_bytes(archive.read_bytes())
    with pytest.raises(ValueError, match="Archive set"):
        module.verify_release([archive, extra], catalog, tmp_path / "restored")
    assert not (tmp_path / "restored").exists()


def test_cross_archive_collision_rejected_before_materialization(tmp_path):
    module = load_script()
    (tmp_path / "result.json").write_text("data")
    archives = [tmp_path / name for name in ("first.tar.gz", "second.tar.gz")]
    families = {
        archive.name: module.build(tmp_path, ["result.json"], archive)
        for archive in archives
    }
    catalog = tmp_path / "catalog.json"
    catalog.write_text(json.dumps({"families": families}))
    with pytest.raises(ValueError, match="path"):
        module.verify_release(archives, catalog, tmp_path / "restored")
    assert not (tmp_path / "restored").exists()


@pytest.mark.parametrize("name", ["../escape", "/absolute", "folder/../escape"])
def test_rejects_unsafe_source_paths(tmp_path, name):
    with pytest.raises(ValueError):
        load_script().build(tmp_path, [name], tmp_path / "bad.tar.gz")


def test_rejects_symlink_and_missing_source(tmp_path):
    module = load_script()
    (tmp_path / "link").symlink_to(tmp_path / "missing")
    for name in ("link", "missing"):
        with pytest.raises((ValueError, FileNotFoundError)):
            module.build(tmp_path, [name], tmp_path / "bad.tar.gz")


def test_tampered_blob_is_rejected_before_materialization(tmp_path):
    module = load_script()
    payload = b"actual"
    digest = hashlib.sha256(b"expected").hexdigest()
    entry = {
        "path": "result.json",
        "status": "exact",
        "original_sha256": digest,
        "public_sha256": digest,
        "public_size": 6,
        "original_size": 6,
    }
    archive = tmp_path / "bad.tar.gz"
    with tarfile.open(archive, "w:gz") as writer:
        for name, data in [
            ("inventory.jsonl", json.dumps(entry).encode()),
            ("blobs/" + digest, payload),
        ]:
            member = tarfile.TarInfo(name)
            member.size = len(data)
            writer.addfile(member, io.BytesIO(data))
    destination = tmp_path / "restored"
    with pytest.raises(ValueError):
        module.verify(archive, destination)
    assert not (destination / "result.json").exists()
