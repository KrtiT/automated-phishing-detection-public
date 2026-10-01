"""Real temporary catalogs and closed external-review metadata joins."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from automated_phishing_detection import _study_series_history_context as context
from automated_phishing_detection._checkpoint_codec import canonical_bytes
from automated_phishing_detection._study_history_snapshot_records import digest
from automated_phishing_detection._study_series_history_files import HistoryFiles


def save(path, value):
    content = canonical_bytes(value)
    path.write_bytes(content)
    return {"path": str(path), "sha256": digest(content)}


def catalog(tmp_path):
    root = tmp_path / "retained"
    root.mkdir()
    payload = save(root / "retained.json", {"synthetic": True})
    sidecar = save(tmp_path / "study-session-invented-pre.json", {"observation": True})
    selected = {
        "profile": save(
            tmp_path / "profile.json", {"paths": {"attempt": str(root / "study-root")}}
        ),
        "envelope": save(tmp_path / "envelope.json", {"synthetic": True}),
        "inventory": save(
            tmp_path / "inventory.json",
            {
                "schema_version": 1,
                "inventory_id": "study-series-retained-inventory-v1",
                "files": {
                    "tree/retained.json": payload,
                    "sidecars/study-session-invented-pre.json": sidecar,
                },
            },
        ),
        "root_reservation_sha256": "a" * 64,
        "disposition": "preserve_excluded",
    }
    review = {
        "catalog_sha256": selected["inventory"]["sha256"],
        "catalog_files": sorted([payload["path"], sidecar["path"]]),
        "root_reservation_sha256": selected["root_reservation_sha256"],
        "disposition": selected["disposition"],
    }
    selected["interruption_review"] = save(tmp_path / "review.json", review)
    return selected, review, root


def test_complete_catalog_with_authenticated_review_and_real_files(tmp_path):
    selected, review, root = catalog(tmp_path)
    reader = HistoryFiles()
    context._catalog(reader, selected, reviewed=True)
    reader.check()
    assert root in reader.trees and len(reader.files) == 6


@pytest.mark.parametrize(
    "mutation",
    ["catalog_sha256", "root_reservation_sha256", "disposition", "catalog_files"],
)
def test_rehashed_wrong_review_cannot_join_current_catalog(tmp_path, mutation):
    selected, review, root = catalog(tmp_path)
    review[mutation] = [] if mutation == "catalog_files" else "wrong"
    selected["interruption_review"] = save(tmp_path / "review.json", review)
    with pytest.raises(ValueError):
        context._catalog(HistoryFiles(), selected, reviewed=True)


def test_unlisted_current_tree_file_rejects_even_if_catalog_hashes_match(tmp_path):
    selected, review, root = catalog(tmp_path)
    (root / "omitted.json").write_bytes(b"not in the pinned inventory")
    with pytest.raises(ValueError):
        context._catalog(HistoryFiles(), selected, reviewed=True)


def eligibility():
    index = SimpleNamespace(
        index_sha256="a" * 64, accepted_ordinals=(1, 2), stopped_ordinal=3
    )
    profile = {
        "origin": {"root_reservation_sha256": "b" * 64},
        "scientific_pins": {
            kind + "_bindings_sha256": "c" * 64 for kind in ("internal", "external")
        },
    }
    report = {"path": "/invented/report.json", "sha256": "d" * 64}
    review = {
        "schema_version": 1,
        "review_id": "study-series-eligible-prefix-review-v1",
        "index_sha256": index.index_sha256,
        "selected_root_reservation_sha256": "b" * 64,
        "accepted_ordinals": [1, 2],
        "stopped_ordinal": 3,
        "source_bindings_sha256": {kind: "c" * 64 for kind in ("internal", "external")},
        "custody_review": report,
        "scientific_report": report,
        "interruption_cause": "supervisor_ac_power_loss_after_prefix",
        "physical_evidence_scope": "sampled_prefix_with_next_child_witness_not_continuous_power",
        "limitations": ["Synthetic sampled observations only."],
        "status": "eligible_entire_prefix",
        "authorizes_execution": False,
    }
    return review, profile, index


def test_review_accepts_only_exact_factual_prefix():
    review, profile, index = eligibility()
    context._review_identity(review, profile, index)
    context._review_result(review)


@pytest.mark.parametrize("ordinals", [[True, 2], [1.0, 2], [2, 1], [1], [1, 2, 3]])
def test_review_rejects_wrong_types_order_or_subset(ordinals):
    review, profile, index = eligibility()
    review["accepted_ordinals"] = deepcopy(ordinals)
    with pytest.raises(ValueError):
        context._review_identity(review, profile, index)
