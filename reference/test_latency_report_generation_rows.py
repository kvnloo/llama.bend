"""Regression for silently discarded generation rows in captured gen.bend stdout."""
import json
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from latency_report import report

CAPTURE = (ROOT / "docs/media/outputs/gen_17b.txt").read_text(encoding="utf-8")


class GeneratedRowIntegrityTests(unittest.TestCase):
    def test_recorded_capture_still_reports_all_tokens(self):
        result = report(CAPTURE)
        self.assertEqual((result["tokens"], result["load_s"],
                          result["decode_p50_ms"], result["decode_p95_ms"]),
                         (32, 71.2, 400.0, 500.0))

    def test_corrupt_native_token_row_does_not_shrink_distribution(self):
        for bad in ("  token ", "  i d ", "  xid "):
            with self.subTest(bad=bad):
                corrupted = CAPTURE.replace("  id ", bad, 1)
                self.assertNotEqual(corrupted, CAPTURE)
                with self.assertRaisesRegex(ValueError, "unexpected native generation line"):
                    report(corrupted)

    def test_cli_does_not_publish_partial_metrics(self):
        changed = CAPTURE.replace("  id ", "  token ", 1)
        proc = subprocess.run([sys.executable, str(ROOT / "scripts/latency_report.py"), "--json"],
                              input=changed, text=True, capture_output=True, check=False)
        self.assertEqual(proc.returncode, 2, proc.stderr)
        self.assertEqual(proc.stdout, "")
        self.assertIn("unexpected native generation line", proc.stderr)

    def test_token_only_fragment_and_utf8_footer_remain_valid(self):
        fragment = "  id 4 logit 1056964608 0.5 t=0.100s\n"
        self.assertEqual(report("noise\n" + fragment)["tokens"], 1)
        self.assertEqual(report(CAPTURE.replace("text: ", "text: 🌍 café ", 1))["tokens"], 32)


if __name__ == "__main__":
    unittest.main()
