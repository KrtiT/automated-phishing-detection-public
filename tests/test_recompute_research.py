import importlib.util
import json
from hashlib import sha256
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/recompute_research.py"


def module():
    spec = importlib.util.spec_from_file_location("recompute_research", SCRIPT)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def fixture_archive(root, payload, status="exact"):
    (root / ".archive-inventories").mkdir()
    (root / "result.json").write_bytes(payload)
    (root / ".archive-inventories/family.jsonl").write_text(
        json.dumps(
            {
                "path": "result.json",
                "status": status,
                "public_sha256": sha256(payload).hexdigest(),
            }
        )
        + "\n"
    )
    catalog = root / "catalog.json"
    catalog.write_text(
        json.dumps(
            {
                "families": {
                    "family": {
                        "inventory_sha256": sha256(
                            (root / ".archive-inventories/family.jsonl").read_bytes()
                        ).hexdigest()
                    }
                }
            }
        )
    )
    return catalog


def test_bound_inputs_reject_tampering(tmp_path):
    catalog = fixture_archive(tmp_path, b'{"count":2}')
    inputs = module().ArchiveInputs(tmp_path, catalog)
    assert inputs.read("result.json") == {"count": 2}
    (tmp_path / "result.json").write_text('{"count":3}')
    with pytest.raises(ValueError, match="hash"):
        inputs.read("result.json")


def test_scientific_inputs_must_be_exact(tmp_path):
    catalog = fixture_archive(tmp_path, b'{"count":2}', "administrative_projection")
    inputs = module().ArchiveInputs(tmp_path, catalog)
    with pytest.raises(ValueError, match="exact"):
        inputs.read("result.json")
    assert inputs.read("result.json", exact=False) == {"count": 2}


def test_unlisted_inputs_are_not_read(tmp_path):
    catalog = fixture_archive(tmp_path, b'{"count":2}')
    with pytest.raises(ValueError, match="inventoried"):
        module().ArchiveInputs(tmp_path, catalog).read("missing.json")


def test_coordinated_payload_inventory_change_is_rejected(tmp_path):
    catalog = fixture_archive(tmp_path, b'{"count":2}')
    payload = b'{"count":3}'
    (tmp_path / "result.json").write_bytes(payload)
    inventory = tmp_path / ".archive-inventories/family.jsonl"
    entry = json.loads(inventory.read_text())
    entry["public_sha256"] = sha256(payload).hexdigest()
    inventory.write_text(json.dumps(entry) + "\n")
    with pytest.raises(ValueError, match="Inventory hash"):
        module().ArchiveInputs(tmp_path, catalog)


def test_unlisted_inventory_is_rejected(tmp_path):
    catalog = fixture_archive(tmp_path, b'{"count":2}')
    (tmp_path / ".archive-inventories/stale.jsonl").write_text("")
    with pytest.raises(ValueError, match="Inventory set"):
        module().ArchiveInputs(tmp_path, catalog)


def test_retained_reader_consumes_the_authenticated_bytes(tmp_path):
    catalog = fixture_archive(tmp_path, b'{"count":2}')
    adapter = module()
    inputs = adapter.ArchiveInputs(tmp_path, catalog)
    path = adapter.RetainedPath(inputs, "result.json")
    stream = path.open()
    (tmp_path / "result.json").write_text('{"count":3}')
    assert stream.read() == '{"count":2}'
    with pytest.raises(ValueError, match="hash"):
        path.open()


@pytest.mark.parametrize(
    "rows,bins,names",
    [
        (0, 430, ["gmm", "mmd", "psi"]),
        (153, 0, ["gmm", "mmd", "psi"]),
        (153, 430, []),
        (153, 430, ["gmm"] * 3),
    ],
)
def test_incomplete_secondary_coverage_is_rejected(rows, bins, names):
    with pytest.raises(ValueError, match="Secondary coverage"):
        module().require_secondary_coverage(rows, bins, names)
