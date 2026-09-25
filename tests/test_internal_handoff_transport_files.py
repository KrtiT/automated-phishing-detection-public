"""Reject aliases, mutable file states and inconsistent private transport trees."""

import os

import pytest
import test_internal_handoff_transport as fixtures

inputs = fixtures.inputs
published = fixtures.published
runner = fixtures.runner
verifier = fixtures.verifier
observed_worker = fixtures.observed_worker
completion = fixtures.completion
handoff_api = fixtures.handoff_api
transport_api = fixtures.transport_api
payloads = fixtures.payloads
temporary_parent = fixtures.temporary_parent


@pytest.mark.parametrize("name", sorted(fixtures.NAMES))
@pytest.mark.parametrize("change", ["mode", "content", "hardlink"])
def test_transport_files_are_private_intact_and_unaliased(
    transport_api, payloads, temporary_parent, name, change
):
    with pytest.raises(transport_api.InternalTransportError):
        with transport_api.retain_internal_handoff(payloads) as retained:
            path = retained.directory / name
            alias = temporary_parent / "alias"
            if change == "mode":
                path.chmod(0o644)
            elif change == "content":
                path.write_bytes(b"private-canary")
            else:
                os.link(path, alias)
            with pytest.raises(transport_api.InternalTransportError) as caught:
                fixtures.read(transport_api, retained)
            assert "private-canary" not in str(caught.value)


def test_directory_must_have_private_mode(transport_api, payloads, temporary_parent):
    with pytest.raises(transport_api.InternalTransportError):
        with transport_api.retain_internal_handoff(payloads) as retained:
            retained.directory.chmod(0o755)
            with pytest.raises(transport_api.InternalTransportError):
                fixtures.read(transport_api, retained)


def test_reader_rejects_symlink_components(transport_api, payloads, temporary_parent):
    with transport_api.retain_internal_handoff(payloads) as retained:
        alias = temporary_parent / "alias"
        alias.symlink_to(retained.directory, target_is_directory=True)
        with pytest.raises(transport_api.InternalTransportError):
            transport_api.read_internal_handoff_transport(
                alias, expected_handoff_sha256=retained.expected_handoff_sha256
            )


def test_reader_rechecks_first_file_after_reading_second(
    transport_api, payloads, temporary_parent, monkeypatch
):
    from automated_phishing_detection import source_runner

    original, paths = source_runner._read_file_once, []

    def changed(path, **kwargs):
        content = original(path, **kwargs)
        paths.append(path)
        if len(paths) == 2:
            paths[0].write_bytes(b"changed after first safe read")
        return content

    with pytest.raises(transport_api.InternalTransportError):
        with transport_api.retain_internal_handoff(payloads) as retained:
            monkeypatch.setattr(source_runner, "_read_file_once", changed)
            with pytest.raises(transport_api.InternalTransportError):
                fixtures.read(transport_api, retained)


def test_unknown_entries_are_never_recursively_deleted(
    transport_api, payloads, temporary_parent
):
    with pytest.raises(transport_api.InternalTransportError):
        with transport_api.retain_internal_handoff(payloads) as retained:
            unknown = retained.directory / "unknown"
            unknown.mkdir()
            (unknown / "unowned").write_bytes(b"keep")
            with pytest.raises(transport_api.InternalTransportError):
                fixtures.read(transport_api, retained)
    assert (unknown / "unowned").read_bytes() == b"keep"


def test_replaced_directory_is_not_deleted(transport_api, payloads, temporary_parent):
    with pytest.raises(transport_api.InternalTransportError):
        with transport_api.retain_internal_handoff(payloads) as retained:
            moved = retained.directory.with_name("moved")
            retained.directory.rename(moved)
            retained.directory.mkdir(mode=0o700)
            replacement = retained.directory / "replacement"
            replacement.write_bytes(b"keep")
    assert replacement.read_bytes() == b"keep"
    assert {path.name for path in moved.iterdir()} == fixtures.NAMES
