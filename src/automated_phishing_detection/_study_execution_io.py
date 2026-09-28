"""Read only reviewed envelope and committed public policy regular-file bytes."""

from hashlib import sha256
from pathlib import Path

from . import execution_preflight as preflight
from ._study_execution_policy import CHILD_SCRIPT, ROOT_SCRIPT
from ._study_execution_schema import lexical_path, require
from ._study_policy_variants import select_policy


def read_envelope(path):
    selected = lexical_path(str(path))
    return preflight._read_regular(
        Path(selected.anchor), selected.relative_to(selected.anchor).as_posix()
    )


def committed_policy(base, expected_sha256):
    policy_path, unused = select_policy(expected_sha256)
    content = preflight._read_regular(base.root, policy_path)
    require(sha256(content).hexdigest() == expected_sha256)
    tree = preflight._git(base.root, "ls-tree", "-z", base.revision, "--", policy_path)
    entries = tree.split(b"\0")
    require(len(entries) == 2 and not entries[1])
    metadata, filename = entries[0].split(b"\t", 1)
    mode, kind, object_id = metadata.decode("ascii").split()
    require(mode in ("100644", "100755") and kind == "blob")
    require(filename.decode("utf-8") == policy_path)
    require(preflight._git(base.root, "cat-file", "blob", object_id) == content)
    require({ROOT_SCRIPT, CHILD_SCRIPT} <= dict(base.source_hashes).keys())
    return content
