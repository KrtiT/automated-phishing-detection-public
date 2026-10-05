import importlib
import importlib.util
import unittest
from unittest.mock import patch


class FollowupDeckRenderTests(unittest.TestCase):
    def setUp(self):
        name = "check_followup_deck_render_20261001"
        self.assertIsNotNone(importlib.util.find_spec(name), "Render checker not implemented")
        self.checker = importlib.import_module(name)

    def test_every_slide_and_visible_text_survives_native_export(self):
        report = self.checker.check()
        self.assertEqual(report["slides"], 23)
        self.assertGreater(report["checked_text_items"], 200)
        self.assertEqual(report["missing_text"], [])
        self.assertEqual(report["out_of_bounds_words"], [])

    def test_normalization_preserves_meaningful_numbers(self):
        self.assertEqual(self.checker.normalize("95.36%\nrecall"), "95.36%recall")
        self.assertNotEqual(self.checker.normalize("95.36%"), self.checker.normalize("99.36%"))

    def test_expected_slide_count_is_explicit_and_enforced(self):
        with patch.object(self.checker, "EXPECTED_SLIDES", 25, create=True):
            with self.assertRaisesRegex(ValueError, "Expected all 25 slides"):
                self.checker.check()


if __name__ == "__main__":
    unittest.main()
