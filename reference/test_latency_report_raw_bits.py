"""Guard against fabricated latency samples with nonfinite F32 raw bits."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from latency_report import report

class RawBitIntegrityTests(unittest.TestCase):
    def test_nonfinite_f32_bits_rejected(self):
        for bits in (2139095040, 4286578688, 2143289344, 2139095041, 4290772992):
            with self.subTest(bits=bits):
                capture = (f"  id 4 logit {bits} 0.5 t=0.100s\n"
                           "  id 5 logit 1056964608 0.5 t=0.200s")
                with self.assertRaisesRegex(ValueError, "nonfinite generated token logit bits"):
                    report(capture)

    def test_negative_largest_finite_f32_bits_remain_accepted(self):
        capture = ("  id 4 logit 4286578687 -3.4028235e+38 t=0.100s\n"
                   "  id 5 logit 1056964608 0.5 t=0.200s")
        self.assertEqual(report(capture)["tokens"], 2)

if __name__ == "__main__":
    unittest.main()
