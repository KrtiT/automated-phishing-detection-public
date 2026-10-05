import unittest
from pathlib import Path

import integrate_followup_manuscript_20261001 as integration


class FollowupManuscriptTests(unittest.TestCase):
    def test_original_questions_hypotheses_and_measurements_preserved(self):
        original = integration.ORIGINAL.read_text()
        body = integration.compose()
        for start, end in [("## 1.4 Research Questions and Hypotheses", "## 1.5 Contribution Boundary"),
                           ("## 4.1 Completed Evaluation and Populations", "# Chapter 5—Discussion and Conclusions")]:
            protected = original.split(start, 1)[1].split(end, 1)[0]
            self.assertIn(protected.strip(), body)

    def test_working_status_and_real_chronology_are_explicit(self):
        body = integration.compose()
        self.assertIn("after inspection of the initial results", body)
        self.assertIn("Service comparison S has not yet been measured", body)
        self.assertIn("−3.9561", body)
        self.assertIn("8,622 / 8,622", body)
        self.assertNotIn("TODO", body)
        self.assertEqual(body.count("# Chapter "), 5)

    def test_new_figures_references_and_effects_present(self):
        body = integration.compose()
        for value in ["Figure 3.2.", "![Follow-up source partitions]", "Figure 4.1.", "Figure 4.2.", "Table 4.17.",
                      "## 5.2.3", "Arp", "TESSERACT", "97.5%", "251", "184"]:
            self.assertIn(value, body)
        self.assertEqual(body.count("## References"), 1)
        self.assertNotIn("## New References", body)

    def test_output_is_separate_from_preserved_manuscript(self):
        self.assertNotEqual(integration.OUTPUT, integration.ORIGINAL)
        self.assertIn("manuscript-work", integration.OUTPUT.parts)
        self.assertTrue(integration.OUTPUT.is_relative_to(Path(__file__).resolve().parent))

    def test_comparison_decisions_and_completed_regression_result_are_distinct(self):
        body = integration.compose()
        self.assertIn("Table 5.2.", body)
        self.assertIn("13,016 passed, three failed and two skipped", body)
        self.assertIn("all six original interruption assertions pass", body)
        self.assertIn("not relabeled as a green full-suite run", body)


if __name__ == "__main__":
    unittest.main()
