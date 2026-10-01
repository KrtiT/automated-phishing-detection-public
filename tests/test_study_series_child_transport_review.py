"""Independent specification probes for pre-access and pre-yield held identities."""

from dataclasses import replace

import pytest
from study_series_child_inputs_fixtures import (
    candidates,
    child_case,
    manifests,
    series_case,
)
from study_series_child_transport_fixtures import api, hold, written
from test_study_series_child_transport_files import NAMES, mutate, selected

__all__ = ["candidates", "child_case", "manifests", "series_case"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("origin_reservation_sha256", "0" * 64),
        ("history_index_sha256", "0" * 64),
        ("cell_ordinal", 72),
    ],
)
def test_valid_shaped_wrong_frame_context_never_opens_inputs(
    tmp_path, child_case, monkeypatch, field, value
):
    def forbidden(*arguments, **keywords):
        pytest.fail("mismatched pre-access frame opened protected inputs")

    monkeypatch.setattr(api().storage, "hold", forbidden)
    frame = replace(child_case.frame, **{field: value})
    with pytest.raises(api().OperationalInputTransportError):
        with hold((tmp_path / "missing", tmp_path / "other"), child_case, frame=frame):
            pytest.fail("mismatched pre-access frame yielded")


@pytest.mark.parametrize("name", NAMES)
def test_change_during_pure_restore_is_rechecked_before_yield(
    tmp_path, child_case, monkeypatch, name
):
    paths = written(tmp_path, child_case)
    original, restored = api().restore_series_child_inputs, []

    def changing(*arguments, **keywords):
        result = original(*arguments, **keywords)
        restored.append(result)
        mutate(selected(paths, name), "bytes")
        return result

    monkeypatch.setattr(api(), "restore_series_child_inputs", changing)
    with pytest.raises(api().OperationalInputTransportError):
        with hold(paths, child_case):
            pytest.fail("changed input identity escaped post-restoration check")
    assert restored == [child_case.expected]


@pytest.mark.parametrize("name", NAMES)
def test_byte_identical_inode_replacement_fails_held_exit(tmp_path, child_case, name):
    paths = written(tmp_path, child_case)
    with pytest.raises(api().OperationalInputTransportError):
        with hold(paths, child_case):
            path = selected(paths, name)
            content = path.read_bytes()
            path.rename(tmp_path / "retained-original")
            path.write_bytes(content)
            path.chmod(0o600)
    assert (tmp_path / "retained-original").read_bytes() == content


@pytest.mark.parametrize("interruption", [SystemExit(7), GeneratorExit("invented")])
def test_non_exception_body_survives_mutation_cleanup(
    tmp_path, child_case, interruption
):
    paths = written(tmp_path, child_case)
    with pytest.raises(BaseException) as caught:
        with hold(paths, child_case):
            mutate(selected(paths, "manifest"), "bytes")
            raise interruption
    assert caught.value is interruption


def test_cleanup_rejection_carries_original_body_progress(tmp_path, child_case):
    paths = written(tmp_path, child_case)
    original = ValueError("invented body failure")
    original.progress = b"original progress association"
    with pytest.raises(api().OperationalInputTransportError) as caught:
        with hold(paths, child_case):
            mutate(selected(paths, "manifest"), "bytes")
            raise original
    assert caught.value.progress is original.progress


@pytest.mark.parametrize("relation", ["equal", "child", "parent"])
def test_lexical_overlap_rejects_before_storage_open(
    tmp_path, child_case, monkeypatch, relation
):
    root = tmp_path / "root"
    pairs = {
        "equal": (root, root),
        "child": (root, root / "child"),
        "parent": (root / "child", root),
    }

    def forbidden(*arguments, **keywords):
        pytest.fail("overlapping input locators opened storage")

    monkeypatch.setattr(api().storage, "hold", forbidden)
    with pytest.raises(api().OperationalInputTransportError):
        with hold(pairs[relation], child_case):
            pytest.fail("overlapping input locators yielded")
