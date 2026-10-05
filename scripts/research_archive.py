"""Package and verify retained research files without executing research code."""

import argparse
import gzip
import hashlib
import io
import json
import re
import shutil
import tarfile
import unicodedata
from collections import Counter
from pathlib import Path, PurePosixPath

HASH = re.compile(r"[0-9a-f]{64}")
CATALOG = (
    Path(__file__).resolve().parents[1] / "research-archive/2026-10-04/catalog.json"
)
PRIVATE_KEYS = {
    "processes",
    "command",
    "argv",
    "command_line",
    "cmdline",
    "additional_live_parent_lookup",
    "exclusive_setup_question",
}
PRIVATE_MARKERS = (
    b"/Users/",
    b"/private/var/",
    *(json.dumps(key).encode() for key in sorted(PRIVATE_KEYS)),
    b'_bytes"',
)
SECRET = re.compile(
    rb"(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|sk-(?:proj-|svca-)[A-Za-z0-9_-]{35,}|sk-[A-Za-z0-9]{48}(?![A-Za-z0-9]))"
)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def safe_name(name):
    path = PurePosixPath(name)
    if (
        not name
        or not path.parts
        or path.is_absolute()
        or ".." in path.parts
        or "\\" in name
        or str(path) != name
    ):
        raise ValueError("Unsafe archive path")
    return path


def validate_paths(names):
    normalized = set()
    for name in names:
        safe_name(name)
        canonical = unicodedata.normalize("NFC", name).casefold()
        if PurePosixPath(canonical).parts[0] == ".archive-inventories":
            raise ValueError("Reserved inventory namespace")
        if canonical in normalized:
            raise ValueError("Duplicate or filesystem-normalized path collision")
        normalized.add(canonical)
    for name in normalized:
        if any(str(parent) in normalized for parent in PurePosixPath(name).parents):
            raise ValueError("Ancestor/descendant path collision")


def project(value, context):
    if isinstance(value, dict):
        result = {}
        for key, child in value.items():
            public_key = project(key, context)
            if public_key in result:
                raise ValueError("Projected metadata key collision")
            if key in PRIVATE_KEYS or (
                key.endswith("_bytes") and isinstance(child, str)
            ):
                encoded = json.dumps(child, sort_keys=True).encode()
                result[public_key] = {
                    "withheld": "private execution metadata",
                    "value_sha256": digest(encoded),
                }
            else:
                result[public_key] = project(child, context)
        return result
    if isinstance(value, list):
        return [project(child, context) for child in value]
    if isinstance(value, str):
        value = value.replace(str(context) + "/", "archive://")
        return re.sub(
            r"/(?:Users|private/var)/[^\s\"']+",
            lambda match: (
                "<private-local-path-sha256:" + digest(match[0].encode()) + ">"
            ),
            value,
        )
    return value


def public_bytes(path, original, context):
    if path.suffix == ".log" or path.name == ".DS_Store":
        return (
            None,
            "hash_only",
            "Host/process logs are private; scientific failure records are retained separately.",
        )
    if SECRET.search(original):
        raise ValueError(f"Credential-like content requires manual review: {path.name}")
    if any(marker in original for marker in PRIVATE_MARKERS):
        try:
            if path.suffix == ".jsonl":
                values = [
                    project(json.loads(line), context) for line in original.splitlines()
                ]
                output = b"".join(
                    json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
                    + b"\n"
                    for value in values
                )
            else:
                output = (
                    json.dumps(
                        project(json.loads(original), context),
                        sort_keys=True,
                        indent=2,
                        ensure_ascii=False,
                    )
                    + "\n"
                ).encode()
        except (json.JSONDecodeError, UnicodeError):
            return (
                None,
                "hash_only",
                "Unstructured local execution metadata; retained privately.",
            )
        if any(marker in output for marker in (b"/Users/", b"/private/var/")):
            raise ValueError("Local path survived projection")
        return (
            output,
            "administrative_projection",
            "Local paths, process command lines and encoded execution frames projected; original file hash retained.",
        )
    return original, "exact", "Byte-identical retained file."


def add_bytes(writer, name, payload):
    member = tarfile.TarInfo(name)
    member.size = len(payload)
    member.mode = 0o644
    writer.addfile(member, io.BytesIO(payload))


def collect(context, roots):
    files = {}
    for name in roots:
        safe_name(name)
        path = context / name
        if not path.exists() or path.is_symlink():
            raise ValueError(f"Missing or symbolic source: {name}")
        members = sorted(path.rglob("*")) if path.is_dir() else [path]
        for member in members:
            if member.is_symlink():
                raise ValueError("Symbolic research source")
            if member.is_file():
                relative = member.relative_to(context).as_posix()
                safe_name(relative)
                if not member.resolve().is_relative_to(context.resolve()):
                    raise ValueError("Source escaped research root")
                files[relative] = member
    return files


def build(context, roots, destination):
    context, destination = Path(context).resolve(), Path(destination)
    files = collect(context, roots)
    entries, seen, counts, transformed = [], set(), Counter(), {}
    with destination.open("xb") as raw:
        with gzip.GzipFile(
            filename="", mode="wb", fileobj=raw, mtime=0, compresslevel=6
        ) as compressed:
            with tarfile.open(fileobj=compressed, mode="w|") as writer:
                for name, path in sorted(files.items()):
                    original = path.read_bytes()
                    source_hash = digest(original)
                    cache_key = (source_hash, path.suffix, path.name == ".DS_Store")
                    if cache_key not in transformed:
                        payload, status, reason = public_bytes(path, original, context)
                        public_hash = digest(payload) if payload is not None else None
                        transformed[cache_key] = (
                            public_hash,
                            len(payload) if payload is not None else None,
                            status,
                            reason,
                        )
                        if payload is not None and public_hash not in seen:
                            add_bytes(writer, "blobs/" + public_hash, payload)
                            seen.add(public_hash)
                    public_hash, public_size, status, reason = transformed[cache_key]
                    entries.append(
                        {
                            "path": name,
                            "original_sha256": source_hash,
                            "original_size": len(original),
                            "public_sha256": public_hash,
                            "public_size": public_size,
                            "status": status,
                            "reason": reason,
                        }
                    )
                    counts[status] += 1
                inventory_bytes = b"".join(
                    json.dumps(entry, sort_keys=True).encode() + b"\n"
                    for entry in entries
                )
                add_bytes(writer, "inventory.jsonl", inventory_bytes)
    return {
        "files": len(entries),
        "blobs": len(seen),
        "statuses": dict(counts),
        "sha256": digest(destination.read_bytes()),
        "inventory_sha256": digest(inventory_bytes),
        "size": destination.stat().st_size,
    }


def verify(archive, destination=None):
    with tarfile.open(archive, "r:gz") as reader:
        members = reader.getmembers()
        names = [member.name for member in members]
        if len(names) != len(set(names)) or names.count("inventory.jsonl") != 1:
            raise ValueError("Duplicate members or missing inventory")
        for member in members:
            safe_name(member.name)
            if not member.isfile() or (
                member.name != "inventory.jsonl"
                and not re.fullmatch(r"blobs/[0-9a-f]{64}", member.name)
            ):
                raise ValueError("Unsupported archive member")
        entries = [json.loads(line) for line in reader.extractfile("inventory.jsonl")]
        validate_paths(entry["path"] for entry in entries)
        paths, blobs = set(), set()
        for entry in entries:
            name = entry["path"]
            safe_name(name)
            if name in paths or not HASH.fullmatch(entry["original_sha256"]):
                raise ValueError("Duplicate path or invalid original hash")
            paths.add(name)
            if entry["status"] == "hash_only":
                if entry["public_sha256"] is not None:
                    raise ValueError("Withheld content cannot have a public blob")
                continue
            public_hash = entry["public_sha256"]
            if entry["status"] not in {
                "exact",
                "administrative_projection",
            } or not HASH.fullmatch(public_hash):
                raise ValueError("Invalid public identity")
            if entry["status"] == "exact" and (
                public_hash != entry["original_sha256"]
                or entry["public_size"] != entry["original_size"]
            ):
                raise ValueError("Exact identity differs")
            member = reader.getmember("blobs/" + public_hash)
            if member.size != entry["public_size"]:
                raise ValueError("Size mismatch")
            if public_hash not in blobs:
                hasher = hashlib.sha256()
                with reader.extractfile(member) as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        hasher.update(chunk)
                if hasher.hexdigest() != public_hash:
                    raise ValueError("Blob digest mismatch")
                blobs.add(public_hash)
        if set(names) != {"inventory.jsonl", *("blobs/" + value for value in blobs)}:
            raise ValueError("Uninventoried blob")
        if destination is not None:
            destination = Path(destination).absolute()
            inventory = (
                destination / ".archive-inventories" / (Path(archive).name + ".jsonl")
            )
            if inventory.exists() or any(
                parent.is_symlink() for parent in (inventory, *inventory.parents)
            ):
                raise ValueError("Inventory already exists or contains symlink")
            outputs = []
            for entry in entries:
                if entry["status"] == "hash_only":
                    continue
                output = destination / entry["path"]
                if output.exists() or any(
                    parent.is_symlink() for parent in (output, *output.parents)
                ):
                    raise ValueError("Destination already exists or contains symlink")
                outputs.append((entry, output))
            materialized_blobs = {}
            for entry, output in outputs:
                output.parent.mkdir(parents=True, exist_ok=True)
                public_hash = entry["public_sha256"]
                with (
                    (
                        materialized_blobs[public_hash].open("rb")
                        if public_hash in materialized_blobs
                        else reader.extractfile("blobs/" + public_hash)
                    ) as source,
                    output.open("xb") as target,
                ):
                    shutil.copyfileobj(source, target)
                materialized_blobs[public_hash] = output
            inventory.parent.mkdir(parents=True, exist_ok=True)
            with inventory.open("x") as stream:
                for entry in entries:
                    stream.write(json.dumps(entry, sort_keys=True) + "\n")
        return {
            "files": len(entries),
            "blobs": len(blobs),
            "withheld": sum(entry["status"] == "hash_only" for entry in entries),
        }


def verify_release(archives, catalog_path, destination=None):
    catalog = json.loads(Path(catalog_path).read_bytes())["families"]
    names = [archive.name for archive in archives]
    if len(set(names)) != len(names) or set(names) != set(catalog):
        raise ValueError("Archive set differs from the committed catalog")
    if destination is not None:
        destination = Path(destination).absolute()
        if destination.exists() or any(
            parent.is_symlink() for parent in (destination, *destination.parents)
        ):
            raise ValueError(
                "Release destination must be unused and contain no symlink"
            )
    paths, results = [], []
    for archive in archives:
        expected = catalog[archive.name]
        if digest(archive.read_bytes()) != expected["sha256"]:
            raise ValueError("Archive hash differs from the committed catalog")
        with tarfile.open(archive, "r:gz") as reader:
            inventory = reader.extractfile("inventory.jsonl").read()
        if digest(inventory) != expected["inventory_sha256"]:
            raise ValueError("Inventory hash differs from the committed catalog")
        paths.extend(json.loads(line)["path"] for line in inventory.splitlines())
        results.append({"archive": archive.name, **verify(archive)})
    validate_paths(paths)
    if destination is not None:
        for archive in archives:
            verify(archive, destination)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="operation", required=True)
    create = subparsers.add_parser("build")
    create.add_argument("--context", type=Path, required=True)
    create.add_argument("--plan", type=Path, required=True)
    create.add_argument("--output", type=Path, required=True)
    check = subparsers.add_parser("verify")
    check.add_argument("archives", nargs="+", type=Path)
    check.add_argument("--materialize", type=Path)
    check.add_argument("--catalog", type=Path, default=CATALOG)
    arguments = parser.parse_args()
    if arguments.operation == "verify":
        for result in verify_release(
            arguments.archives, arguments.catalog, arguments.materialize
        ):
            print(
                json.dumps(result),
                flush=True,
            )
        return
    arguments.output.mkdir(parents=True, exist_ok=False)
    catalog = {"schema_version": 1, "families": {}}
    plan = json.loads(arguments.plan.read_text())
    for family, roots in plan.items():
        safe_name(family)
        if "/" in family:
            raise ValueError("Family must be a basename")
        archive = arguments.output / (family + ".tar.gz")
        record = build(arguments.context, roots, archive)
        record["roots"] = roots
        catalog["families"][archive.name] = record
        print(json.dumps({"archive": archive.name, **record}), flush=True)
    (arguments.output / "catalog.json").write_text(json.dumps(catalog, indent=2) + "\n")
    (arguments.output / "SHA256SUMS.txt").write_text(
        "".join(
            f"{record['sha256']}  {name}\n"
            for name, record in catalog["families"].items()
        )
    )


if __name__ == "__main__":
    main()
