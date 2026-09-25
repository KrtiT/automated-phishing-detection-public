"""Verify source checkpoints and final predictions using retained bytes only.

These checks authenticate producer-retained identities and structure, not an
independent extraction of quarantined domains from the unopened original CSV.
"""

from hashlib import sha256

from ._source_checkpoint_structure import _MEMBERSHIP_FIELDS as _MEMBERSHIP_FIELDS
from ._source_checkpoint_structure import _json as _json
from ._source_checkpoint_structure import _manifest as _manifest
from ._source_checkpoint_structure import _membership as _membership
from ._source_checkpoint_structure import _require as _require
from ._source_checkpoint_structure import _same as _same
from ._source_checkpoint_structure import _summary as _summary
from .evaluation_producer import _json_bytes
from .source_checkpoints import CHECKPOINT_NAMES, _hashes


def _partition(contents, predictions, source, valid):
    partition = contents["group_test.jsonl"]
    _require(sha256(partition).hexdigest() == source["expected_sha256"])
    rows = [_json(line) for line in predictions.splitlines()]
    _require(partition == b"".join(_json_bytes(row["record"]) for row in rows))
    for row in rows:
        record = row["record"]
        _require(
            valid.get(record["record_id"])
            == (record["canonical_url_sha256"], record["registrable_domain"])
        )


def verify_source_checkpoints(
    contents, public, source, report, identity, reservation_hash, predictions
):
    _require(set(contents) == CHECKPOINT_NAMES)
    hashes = _hashes(contents)
    _require(_same(public["checkpoint_sha256"], hashes))
    common, valid, domains = _manifest(contents, identity, report)
    summary = _summary(common, valid, domains, report, hashes["source-overlap.json"])
    _require(_same(public["source_reconstruction"], summary))
    expected = {
        "schema_version": 1,
        "execution": identity,
        "reservation_sha256": reservation_hash,
        "reconstruction": summary,
        "checkpoint_sha256": {
            name: digest
            for name, digest in hashes.items()
            if name != "source-reconstruction.json"
        },
    }
    _require(contents["source-reconstruction.json"] == _json_bytes(expected))
    _partition(contents, predictions, source, valid)
    return frozenset(domains)
