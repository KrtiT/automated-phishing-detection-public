"""Create-only physical observations allow owned logs to grow, not be replaced."""

import json
import os

import pytest
from study_series_session_fixtures import api


def test_records_allow_stream_growth_and_preserve_all_seven_files(tmp_path):
    module = api("_study_series_session_records")
    tmp_path.chmod(0o700)
    path = tmp_path / "physical"
    with module.hold_records(path) as records:
        records.record("pre.json", {"synthetic": True})
        records.streams["stdout.log"].write(b"one\n")
        records.check()
        records.streams["conditions.jsonl"].write(b"{}\n")
        records.streams["stdout.log"].write(b"two\n")
        records.check()
        records.record("post.json", {"root_exit_code": 130})
    assert set(member.name for member in path.iterdir()) == set(module.NAMES)
    assert (path / "stdout.log").read_bytes() == b"one\ntwo\n"
    assert json.loads((path / "post.json").read_bytes())["root_exit_code"] == 130
    assert all(member.stat().st_mode & 0o777 == 0o600 for member in path.iterdir())


def test_repeat_session_never_overwrites_existing_records(tmp_path):
    module = api("_study_series_session_records")
    tmp_path.chmod(0o700)
    path = tmp_path / "physical"
    with module.hold_records(path) as records:
        records.record("pre.json", {"original": True})
    with pytest.raises(ValueError):
        with module.hold_records(path):
            pytest.fail("repeated destination admitted")
    assert json.loads((path / "pre.json").read_bytes()) == {"original": True}


@pytest.mark.parametrize(
    "mutation",
    ["replace", "symlink", "hardlink", "extra", "truncate", "rewrite_record"],
)
def test_mutated_physical_records_reject(tmp_path, mutation):
    module = api("_study_series_session_records")
    tmp_path.chmod(0o700)
    path = tmp_path / "physical"
    with pytest.raises(ValueError):
        with module.hold_records(path) as records:
            records.record("pre.json", {"original": True})
            records.streams["stdout.log"].write(b"original log\n")
            records.check()
            target = path / "stdout.log"
            if mutation in ("replace", "symlink"):
                target.unlink()
                if mutation == "replace":
                    target.write_bytes(b"changed")
                else:
                    target.symlink_to(path / "pre.json")
            elif mutation == "hardlink":
                os.link(target, tmp_path / "alias")
            elif mutation == "extra":
                (path / "extra").write_bytes(b"unknown")
            elif mutation == "truncate":
                target.write_bytes(b"")
            else:
                (path / "pre.json").write_bytes(b'{"original":false}\n')
            records.check()


def test_fixed_record_cannot_be_written_twice(tmp_path):
    module = api("_study_series_session_records")
    tmp_path.chmod(0o700)
    with module.hold_records(tmp_path / "physical") as records:
        records.record("pre.json", {"first": True})
        with pytest.raises(ValueError):
            records.record("pre.json", {"second": True})


def test_symlink_parent_is_rejected_without_creating_output(tmp_path):
    module = api("_study_series_session_records")
    tmp_path.chmod(0o700)
    alias = tmp_path / "alias"
    alias.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises((ValueError, OSError)):
        with module.hold_records(alias / "physical"):
            pytest.fail("aliased parent admitted")
    assert not (tmp_path / "physical").exists()
