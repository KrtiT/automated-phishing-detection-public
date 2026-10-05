import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import draw_followup_detection_20261001 as figures


class FollowupFigureTests(unittest.TestCase):
    def test_every_rendered_image_is_in_the_manifest(self):
        with TemporaryDirectory() as temporary:
            original = figures.ROOT
            root = Path(temporary)
            (root / "verified-detection-v1").symlink_to(original / "verified-detection-v1", target_is_directory=True)
            (root / "population-admission-v1-attempt-2").symlink_to(original / "population-admission-v1-attempt-2", target_is_directory=True)
            with patch.object(figures, "ROOT", root):
                figures.main()
            directory = root / "figures"
            manifest = json.loads((directory / "detection-figure-manifest.json").read_text())
            expected = {"followup-calibration.png", "followup-calibration.pdf",
                        "followup-paired-recall.png", "followup-paired-recall.pdf",
                        "followup-source-partitions.png", "followup-source-partitions.pdf"}
            self.assertEqual(set(manifest["outputs"]), expected)
            for name in expected:
                self.assertGreater((directory / name).stat().st_size, 1000)

    def test_source_figure_separates_development_diagnosis_and_evaluation(self):
        admission = json.loads((figures.ROOT / "population-admission-v1-attempt-2/summary.json").read_bytes())
        verification = json.loads((figures.ROOT / "verified-detection-v1/verification.json").read_bytes())
        figure = figures.source_partition_figure(admission, verification)
        try:
            labels = "\n".join(item.get_text() for axis in figure.axes for item in axis.texts)
            for required in ("166,248", "32,695", "34,593", "8,701", "11,430", "2,808",
                             "8,622", "4,651", "3,971", "6,273", "Diagnosis", "Threshold",
                             "Retrospective", "not development data", "Before external scoring"):
                self.assertIn(required, labels)
        finally:
            figures.plt.close(figure)


if __name__ == "__main__":
    unittest.main()
