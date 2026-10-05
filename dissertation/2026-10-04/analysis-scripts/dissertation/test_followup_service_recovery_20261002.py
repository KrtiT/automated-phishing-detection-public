import importlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.recovery = importlib.import_module("run_followup_service_recovery_20261002")
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.followup = Path(self.temporary.name)
        self.old = self.followup / "service-comparison-v1"
        self.old.mkdir()
        (self.old / "failure.json").write_text('{"message":"ac_power_absent"}')
        self.manifest = self.followup / "execution-manifest-v1.json"
        self.manifest.write_text("{}")
        self.specification = self.followup / "comparison-specification-v1.md"
        self.specification.write_text("frozen science")
        (self.followup / "service-recovery-amendment-v2.md").write_text("approved recovery")
        receipt = {
            "completed_arms": [{}] * 45,
            "primary_structural_c64_arms_completed": 0,
            "pooling_into_recovery_permitted": False,
            "source_sha256": {"failure.json": self.recovery.runner.digest(self.old / "failure.json")},
        }
        (self.followup / "service-v1-preservation.json").write_text(json.dumps(receipt))
        self.authorization = self.followup / "service-recovery-authorization-v2.json"
        value = {
            "attempt_name": "service-comparison-v2",
            "maximum_new_schedules": 1,
            "minimum_stable_seconds": 180,
            "sample_interval_seconds": 5,
            "pool_interrupted_attempt": False,
            "launcher_sha256": self.recovery.runner.digest(Path(self.recovery.__file__)),
            "test_sha256": self.recovery.runner.digest(Path(__file__)),
            "files": {name: self.recovery.runner.digest(self.followup / name) for name in (
                "execution-manifest-v1.json", "comparison-specification-v1.md",
                "service-recovery-amendment-v2.md", "service-v1-preservation.json")},
        }
        self.authorization.write_text(json.dumps(value))
        self.arguments = SimpleNamespace(
            context_root=self.followup, manifest=self.manifest,
            authorization=self.authorization,
            authorization_sha256=self.recovery.runner.digest(self.authorization),
        )

    def test_accepts_exact_binding_and_inventory(self):
        self.recovery.verify_recovery(self.arguments, self.followup)

    def test_rejects_changed_authorization(self):
        self.authorization.write_text("{}")
        with self.assertRaises(ValueError):
            self.recovery.verify_recovery(self.arguments, self.followup)

    def test_rejects_changed_retained_file(self):
        (self.old / "failure.json").write_text("changed")
        with self.assertRaises(ValueError):
            self.recovery.verify_recovery(self.arguments, self.followup)

    def test_rejects_uninventoried_retained_file(self):
        (self.old / "extra.txt").write_text("new file")
        with self.assertRaises(ValueError):
            self.recovery.verify_recovery(self.arguments, self.followup)

    def test_rejects_changed_specification(self):
        self.specification.write_text("changed")
        with self.assertRaises(ValueError):
            self.recovery.verify_recovery(self.arguments, self.followup)

    def test_stable_preflight_waits_full_duration_and_samples(self):
        with patch.object(self.recovery.time, "monotonic", side_effect=[0, 0, 90, 180]), patch.object(
            self.recovery.time, "sleep"
        ) as sleep, patch.object(self.recovery.runner, "record_conditions") as capture:
            self.recovery.stable_preflight(self.followup, object())
        self.assertEqual(capture.call_count, 3)
        self.assertEqual([call.args for call in sleep.call_args_list], [(5,), (5,)])

    def test_preflight_violation_does_not_restart(self):
        with patch.object(self.recovery.runner, "record_conditions", side_effect=RuntimeError("power")) as capture:
            with self.assertRaisesRegex(RuntimeError, "power"):
                self.recovery.stable_preflight(self.followup, object())
        self.assertEqual(capture.call_count, 1)

    def execute_with_fakes(self, *, failure=None):
        inhibitor = SimpleNamespace(pid=123, returncode=-15, terminate=lambda:None, wait=lambda timeout:None)
        schedule = AsyncMock(side_effect=failure)
        fake_manifest = {"code_revision":"frozen"}
        with patch.object(self.recovery.runner, "verify_execution", return_value=(fake_manifest, self.followup, None)), patch.object(
            self.recovery, "stable_preflight", side_effect=lambda path, process:(path / "conditions.jsonl").write_text("clean\n")
        ), patch.object(self.recovery.runner, "record_conditions", side_effect=lambda path, process:path.write_text("clean\n")), patch.object(
            self.recovery.subprocess, "Popen", return_value=inhibitor
        ), patch.object(self.recovery.platform, "platform", return_value="test-platform"), patch.object(
            self.recovery.runner.host, "command", return_value="hardware"
        ), patch.object(
            self.recovery.runner, "guarded_schedule", schedule
        ), patch.object(self.recovery.runner, "reduce_schedule", return_value={"primary":{"supported":False}}) as reduce:
            if failure:
                with self.assertRaisesRegex(RuntimeError, "interrupted"):
                    self.recovery.execute(self.arguments)
            else:
                self.recovery.execute(self.arguments)
        return schedule, reduce

    def test_delegates_once_to_frozen_schedule_and_reducer(self):
        schedule, reduce = self.execute_with_fakes()
        self.assertEqual(schedule.await_count, 1)
        self.assertEqual(reduce.call_count, 1)
        root = self.followup / "service-comparison-v2"
        self.assertEqual(schedule.call_args.args[1], root)
        self.assertEqual(len(json.loads((root / "intent.json").read_text())["schedule"]), 80)
        self.assertTrue((root / "completion.json").exists())
        self.assertTrue((root / "sleep-inhibitor-exit.json").exists())

    def test_interruption_retains_failure_and_never_reduces(self):
        schedule, reduce = self.execute_with_fakes(failure=RuntimeError("interrupted"))
        self.assertEqual(schedule.await_count, 1)
        reduce.assert_not_called()
        root = self.followup / "service-comparison-v2"
        self.assertFalse((root / "completion.json").exists())
        self.assertEqual(json.loads((root / "failure.json").read_text())["message"], "interrupted")
        self.assertTrue((root / "sleep-inhibitor-exit.json").exists())

    def test_existing_attempt_is_never_overwritten(self):
        root = self.followup / "service-comparison-v2"
        root.mkdir()
        sentinel = root / "existing"
        sentinel.write_text("preserve")
        with patch.object(self.recovery.runner, "verify_execution", return_value=({}, self.followup, None)), patch.object(
            self.recovery.subprocess, "Popen"
        ) as process:
            with self.assertRaises(FileExistsError):
                self.recovery.execute(self.arguments)
        process.assert_not_called()
        self.assertEqual(sentinel.read_text(), "preserve")

    def test_preflight_failure_retains_receipt_without_creating_attempt(self):
        inhibitor = SimpleNamespace(pid=123, returncode=-15, terminate=lambda:None, wait=lambda timeout:None)
        with patch.object(self.recovery.runner, "verify_execution", return_value=({}, self.followup, None)), patch.object(
            self.recovery.subprocess, "Popen", return_value=inhibitor
        ), patch.object(self.recovery, "stable_preflight", side_effect=RuntimeError("power")), patch.object(
            self.recovery.runner, "guarded_schedule", AsyncMock()
        ) as schedule:
            with self.assertRaisesRegex(RuntimeError, "power"):
                self.recovery.execute(self.arguments)
        self.assertFalse((self.followup / "service-comparison-v2").exists())
        self.assertTrue((self.followup / "service-recovery-preflight-v2/failure.json").exists())
        schedule.assert_not_called()


if __name__ == "__main__":
    unittest.main()
