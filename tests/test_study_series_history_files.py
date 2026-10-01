"""Actual temporary retained-file identity and digest boundaries."""

from hashlib import sha256
from importlib import import_module

import pytest


def api():
    return import_module("automated_phishing_detection._study_series_history_files")


def retained(tmp_path):
    path = tmp_path / "retained.json"
    path.write_bytes(b"retained")
    return path, sha256(b"retained").hexdigest()


def test_authentic_read_and_hash_only_have_same_stability_checks(tmp_path):
    path, expected = retained(tmp_path)
    reader = api().HistoryFiles()
    assert reader.read(path, expected) == b"retained"
    assert reader.read(path, expected, retain=False) is None
    reader.check()
    path.write_bytes(b"modified")
    with pytest.raises(ValueError):
        reader.check()


@pytest.mark.parametrize("kind", ("digest", "symlink", "parent_link", "hardlink"))
def test_unsafe_or_unpinned_input_rejected(tmp_path, kind):
    path, expected = retained(tmp_path)
    if kind == "digest":
        expected = "0" * 64
    elif kind == "symlink":
        alias = tmp_path / "alias"
        alias.symlink_to(path)
        path = alias
    elif kind == "parent_link":
        alias = tmp_path / "parent"
        alias.symlink_to(tmp_path, target_is_directory=True)
        path = alias / path.name
    else:
        import os

        os.link(path, tmp_path / "hardlink")
    with pytest.raises(ValueError):
        api().HistoryFiles().read(path, expected)


def test_catalog_tree_roster_cannot_silently_gain_or_lose_files(tmp_path):
    path, expected = retained(tmp_path)
    reader = api().HistoryFiles()
    reader.read(path, expected)
    reader.tree(tmp_path, (str(path),))
    reader.check()
    (tmp_path / "unexpected").write_bytes(b"new")
    with pytest.raises(ValueError):
        reader.check()


def test_replacement_parent_is_detected_even_with_same_content(tmp_path):
    parent = tmp_path / "parent"
    parent.mkdir()
    path, expected = retained(parent)
    reader = api().HistoryFiles()
    reader.read(path, expected)
    parent.rename(tmp_path / "old")
    parent.mkdir()
    (parent / path.name).write_bytes(b"retained")
    with pytest.raises(ValueError):
        reader.check()
