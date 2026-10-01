"""Read the pinned envelope and separately committed public series policy only."""

from hashlib import sha256

from . import _study_execution_io as original
from . import _study_execution_schema as schema
from . import _study_series_policy as policy
from . import execution_preflight as preflight

read_envelope = original.read_envelope


def committed_policy(base, expected_sha256):
    schema.digest(expected_sha256)
    content = preflight._read_regular(base.root, policy.POLICY_PATH)
    schema.require(sha256(content).hexdigest() == expected_sha256)
    schema.require(content == policy.policy_bytes())
    tree = preflight._git(
        base.root, "ls-tree", "-z", base.revision, "--", policy.POLICY_PATH
    )
    entries = tree.split(b"\0")
    schema.require(len(entries) == 2 and not entries[1])
    metadata, filename = entries[0].split(b"\t", 1)
    mode, kind, object_id = metadata.decode("ascii").split()
    schema.require(mode in ("100644", "100755") and kind == "blob")
    schema.digest(object_id, 40)
    schema.require(filename.decode("utf-8") == policy.POLICY_PATH)
    schema.require(preflight._git(base.root, "cat-file", "blob", object_id) == content)
    return content
