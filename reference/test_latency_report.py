"""Unit tests for the stdout-only latency report (no model downloads needed)."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from latency_report import report


class LatencyReportTests(unittest.TestCase):
    def test_load_and_intervals(self):
        raw = """weights loaded in 4.400s
prompt ids: 31 23
  id 42  logit 100 0.5  t=0.200s
  id 43  logit 100 0.5  t=0.280s
  id 44  logit 100 0.5  t=0.370s
  id 45  logit 100 0.5  t=0.470s
text: Once upon a time
"""
        r = report(raw)
        self.assertEqual(r["tokens"], 4)
        self.assertEqual(r["load_s"], 4.4)
        self.assertEqual(r["first_generated_ms_since_generation_start"], 200.0)
        self.assertEqual(r["decode_mean_ms"], 90.0)
        self.assertEqual(r["decode_p50_ms"], 90.0)
        self.assertEqual(r["decode_p95_ms"], 100.0)
        self.assertEqual(r["decode_slope_ms_per_position"], 10.0)

    def test_single_token(self):
        r = report("  id 4  logit 1 1.0  t=0.009s")
        self.assertEqual(r["tokens"], 1)
        self.assertIsNone(r["decode_mean_ms"])
        self.assertIsNone(r["decode_slope_ms_per_position"])
        self.assertIsNone(r["load_s"])

    def test_missing_token_times(self):
        with self.assertRaisesRegex(ValueError, "no generated token"):
            report("prompt ids: 2 3\ntext: hello\n")

    def test_nonmonotonic_times(self):
        with self.assertRaisesRegex(ValueError, "not monotonic"):
            report("  id 4  logit 1 1.0 t=1.5s\n  id 5  logit 2 2.0 t=1.0s")

    def test_noisy_output(self):
        r = report("Random t=3.7s\n  id 5  logit 2 2.0 t=1.0s\nignored\n  id 6  logit 3 3.0 t=1.1s")
        self.assertEqual(r["tokens"], 2)
        self.assertAlmostEqual(r["decode_mean_ms"], 100.0)


if __name__ == "__main__":
    unittest.main()
