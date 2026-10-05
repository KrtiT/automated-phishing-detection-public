import copy
import unittest

import numpy as np

import verify_followup_detection_20261001 as verifier


class DetectionRecomputationTests(unittest.TestCase):
    def setUp(self):
        self.rows = []
        for index, (label, baseline, candidate, domain) in enumerate([
            (1, 0.2, 0.9, "a.example"),
            (1, 0.8, 0.7, "a.example"),
            (0, 0.6, 0.1, "b.example"),
            (0, 0.1, 0.8, "c.example"),
        ]):
            self.rows.append({
                "record_id": str(index), "is_phishing": label,
                "registrable_domain": domain, "candidate_feature_equal": True,
                "original": {"baseline": baseline, "candidate": candidate},
                "scheme_swap": {"baseline": baseline, "candidate": candidate},
            })
        self.thresholds = {"baseline": 0.5, "candidate": 0.5}

    def test_confusion_and_paired_gain(self):
        result = verifier.recompute(self.rows, self.thresholds, replicates=100)
        self.assertEqual(result["metrics"]["baseline"]["counts"], {
            "positive": 2, "negative": 2, "tp": 1, "fp": 1, "tn": 1, "fn": 1,
        })
        self.assertEqual(result["metrics"]["candidate"]["recall"], 1)
        self.assertEqual(result["paired_uncertainty"]["recall_difference"]["point"], 0.5)
        self.assertFalse(result["D_requirement_met"])

    def test_cluster_resampling_keeps_dependent_rows_together(self):
        result = verifier.recompute(self.rows, self.thresholds, replicates=100)
        paired = result["paired_uncertainty"]
        self.assertEqual(paired["recall_difference"]["interval_97_5"], [0.5, 0.5])
        self.assertGreater(paired["undefined_recall_replicates"], 0)
        self.assertGreater(paired["undefined_fpr_replicates"], 0)
        self.assertEqual(paired["largest_domain_rows"], 2)

    def test_probability_one_is_in_last_bin_and_empty_bins_are_explicit(self):
        rows = copy.deepcopy(self.rows)
        rows[0]["original"]["candidate"] = 1.0
        result = verifier.recompute(rows, self.thresholds, replicates=10)
        bins = result["metrics"]["candidate"]["calibration_bins"]
        self.assertEqual(len(bins), 10)
        self.assertEqual(bins[-1]["count"], 1)
        self.assertIsNone(bins[0]["positive_fraction"])
        self.assertEqual(sum(item["count"] for item in bins), 4)

    def test_duplicate_ids_nonbinary_labels_and_nonfinite_scores_rejected(self):
        for field, value in [("record_id", "0"), ("is_phishing", 2)]:
            rows = copy.deepcopy(self.rows)
            rows[1][field] = value
            with self.assertRaises(ValueError):
                verifier.recompute(rows, self.thresholds, replicates=10)
        rows = copy.deepcopy(self.rows)
        rows[1]["original"]["candidate"] = float("nan")
        with self.assertRaises(ValueError):
            verifier.recompute(rows, self.thresholds, replicates=10)

    def test_no_alert_precision_is_undefined(self):
        result = verifier.recompute(self.rows, {"baseline": 1.0, "candidate": 1.0}, replicates=10)
        self.assertIsNone(result["metrics"]["baseline"]["precision"])

    def test_saved_summary_disagreement_is_detected(self):
        with self.assertRaises(ValueError):
            verifier.compare({"count": 3}, {"count": 4})
        with self.assertRaises(ValueError):
            verifier.compare({"value": False}, {"value": 0})
        verifier.compare({"value": 1.0}, {"value": np.nextafter(1.0, 2.0)})


if __name__ == "__main__":
    unittest.main()
