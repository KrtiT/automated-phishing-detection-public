import importlib
import unittest


class ServiceArithmeticTests(unittest.TestCase):
    def setUp(self):
        self.verifier = importlib.import_module("verify_followup_service_20261002")

    def test_linear_quantiles_and_empty_failure_denominator(self):
        self.assertEqual(self.verifier.quantiles([]), {"count":0,"p50_ms":None,"p95_ms":None,"p99_ms":None})
        result = self.verifier.quantiles([0, 10, 20, 30])
        self.assertEqual(result["count"], 4)
        self.assertEqual(result["p50_ms"], 15)
        self.assertAlmostEqual(result["p95_ms"], 28.5)
        self.assertAlmostEqual(result["p99_ms"], 29.7)

    def test_quantiles_reject_nonfinite_and_negative_latency(self):
        for value in [-1, float("nan"), float("inf")]:
            with self.assertRaises(ValueError):
                self.verifier.quantiles([value])

    def test_strict_response_agreement_keeps_sequence(self):
        shared = [{"error":None,"record_id":"row","response":{"request_id":"a","admission_sequence":1,"action":"allow","probability":.25,"stage2_invoked":False}}]
        worker = [{"error":None,"record_id":"row","response":{"request_id":"b","admission_sequence":2,"action":"allow","probability":.25,"stage2_invoked":False}}]
        result = self.verifier.agreement(shared, worker)
        self.assertEqual(result, {"requested":1,"both_successful":1,"exact_except_request_id":0,"prediction_agreement":1,"noncomparable_errors":0})

    def test_errors_stay_in_agreement_denominator(self):
        row = {"error":"timeout","record_id":"row","response":None}
        result = self.verifier.agreement([row], [row])
        self.assertEqual(result["requested"], 1)
        self.assertEqual(result["noncomparable_errors"], 1)
        self.assertEqual(result["exact_except_request_id"], 0)

    def test_response_order_mismatch_is_rejected(self):
        with self.assertRaises(ValueError):
            self.verifier.agreement([{"record_id":"a"}], [{"record_id":"b"}])

    def test_constant_ratio_bootstrap_and_unchanged_gates(self):
        result = self.verifier.primary([.5] * 10, [100.] * 100000, errors=0, exact=100000)
        self.assertEqual(result["ratio_interval_97_5"], [.5, .5])
        self.assertTrue(result["S_requirement_met"])
        strict = self.verifier.primary([.5] * 10, [100.] * 100000, errors=0, exact=99999)
        self.assertFalse(strict["S_requirement_met"])
        self.assertFalse(strict["requirements"]["exact_paired_response_agreement"])

    def test_error_boundary_is_strict(self):
        result = self.verifier.primary([.5] * 10, [100.] * 99900, errors=100, exact=99900)
        self.assertFalse(result["requirements"]["errors_below_one_per_thousand"])

    def test_incomplete_primary_cannot_be_reduced(self):
        with self.assertRaises(ValueError):
            self.verifier.primary([.5] * 9, [100.] * 100000, errors=0, exact=100000)
        with self.assertRaises(ValueError):
            self.verifier.primary([.5] * 10, [100.] * 99999, errors=0, exact=100000)

    def test_undefined_ratio_is_not_removed(self):
        result = self.verifier.primary([.5] * 9 + [None], [100.] * 100000, errors=0, exact=100000)
        self.assertEqual(result["undefined_pairs"], [10])
        self.assertIsNone(result["median_ratio"])
        self.assertFalse(result["S_requirement_met"])


if __name__ == "__main__":
    unittest.main()
