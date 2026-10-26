"""Native Nat/U32 printer contract regressions for PR #2's stdout parser.

Standalone portable suite; real captured 1.7B output remains a historical fixture,
not a new inference measurement. Nat.show and U32.show do not zero-pad.
"""
import json
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from latency_report import report

CAPTURE = (ROOT / 'docs/media/outputs/gen_17b.txt').read_text(encoding='utf-8')


class CanonicalNativePrinterContract(unittest.TestCase):
    def cli(self, raw: str):
        return subprocess.run(
            [sys.executable, str(ROOT / 'scripts/latency_report.py'), '--json'],
            input=raw, text=True, capture_output=True, check=False,
        )

    def test_recorded_native_capture_unmodified(self):
        metrics = report(CAPTURE)
        self.assertEqual((metrics['tokens'], metrics['load_s'],
                          metrics['decode_p50_ms'], metrics['decode_p95_ms']),
                         (32, 71.2, 400.0, 500.0))
        result = self.cli(CAPTURE)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout), metrics)

    def test_native_capture_rejects_noncanonical_ids(self):
        for old, new, err in (
            ('prompt ids: 504 ', 'prompt ids: 0504 ', 'malformed prompt header'),
            ('  id 7042 ', '  id 07042 ', 'malformed generated token row'),
        ):
            with self.subTest(noncanonical=new):
                raw = CAPTURE.replace(old, new, 1)
                self.assertNotEqual(raw, CAPTURE)
                with self.assertRaisesRegex(ValueError, err):
                    report(raw)
                proc = self.cli(raw)
                self.assertEqual(proc.returncode, 2, proc.stderr)
                self.assertEqual(proc.stdout, '')
                self.assertIn(err, proc.stderr)

    def test_zero_bits_or_token_id_remain_valid_without_padding(self):
        raw = ('weights loaded in 0.5s\nprompt ids: 0 \n'
               '  id 0 logit 0 0.0 t=0.100s\ntext: ok\n')
        self.assertEqual(report(raw)['tokens'], 1)
        self.assertEqual(self.cli(raw).returncode, 0)
        for old, new, err in (
            ('prompt ids: 0 ', 'prompt ids: 00 ', 'malformed prompt header'),
            ('  id 0 ', '  id 00 ', 'malformed generated token row'),
            ('logit 0 ', 'logit 00 ', 'malformed generated token row'),
        ):
            with self.subTest(noncanonical=new):
                corrupted = raw.replace(old, new, 1)
                with self.assertRaisesRegex(ValueError, err):
                    report(corrupted)
                proc = self.cli(corrupted)
                self.assertEqual(proc.returncode, 2, proc.stderr)
                self.assertEqual(proc.stdout, '')

    def test_headerless_fragment_rejects_padded_numbers(self):
        self.assertEqual(report('  id 0 logit 0 0.0 t=0.100s\n')['tokens'], 1)
        for row in (
            '  id 00 logit 0 0.0 t=0.100s\n',
            '  id 0 logit 00 0.0 t=0.100s\n',
        ):
            with self.subTest(row=row):
                with self.assertRaisesRegex(ValueError, 'malformed generated token row'):
                    report(row)


    def test_native_millisecond_resolution_cannot_be_forged(self):
        # gen.bend:tsec2 has 1ms granularity; a fourth fractional digit is
        # not native output, even when float conversion silently rounds it.
        raw = ('weights loaded in 0.125s\nprompt ids: 7 \n'
               '  id 4 logit 1056964608 0.5 t=0.100s\n'
               '  id 5 logit 1056964608 0.5 t=0.200s\n'
               'text: ok\n')
        self.assertEqual(report(raw)['decode_p50_ms'], 100.0)
        for old, new, error in (
            ('weights loaded in 0.125s', 'weights loaded in 0.1251s',
             'malformed load duration'),
            ('weights loaded in 0.125s', 'weights loaded in 0.125000s',
             'malformed load duration'),
            ('t=0.100s', 't=0.1009s', 'malformed generated token row'),
            ('t=0.200s', 't=0.2001s', 'malformed generated token row'),
        ):
            with self.subTest(corruption=new):
                changed = raw.replace(old, new, 1)
                self.assertNotEqual(changed, raw)
                with self.assertRaisesRegex(ValueError, error):
                    report(changed)
                child = self.cli(changed)
                self.assertEqual(child.returncode, 2, child.stderr)
                self.assertEqual(child.stdout, '')
                self.assertIn(error, child.stderr)


    def test_native_clock_seconds_cannot_be_zero_padded(self):
        # tsec2 builds its integer seconds with Nat.show, never zero padding.
        # Keep the captured, historically rounded fractional widths valid.
        raw = (
            "weights loaded in 0.125s\nprompt ids: 7 \n"
            "  id 4 logit 1056964608 0.5 t=0.100s\n"
            "  id 5 logit 1056964608 0.5 t=0.200s\n"
            "text: ok\n"
        )
        self.assertEqual(report(raw)['decode_p50_ms'], 100.0)
        self.assertEqual(self.cli(raw).returncode, 0)
        for original, invalid, diagnostic in (
            ('weights loaded in 0.125s', 'weights loaded in 00.125s',
             'malformed load duration'),
            ('t=0.100s', 't=00.100s', 'malformed generated token row'),
            ('t=0.200s', 't=000.200s', 'malformed generated token row'),
        ):
            with self.subTest(invalid=invalid):
                changed = raw.replace(original, invalid, 1)
                self.assertNotEqual(changed, raw)
                with self.assertRaisesRegex(ValueError, diagnostic):
                    report(changed)
                child = self.cli(changed)
                self.assertEqual(child.returncode, 2, child.stderr)
                self.assertEqual(child.stdout, '')
                self.assertIn(diagnostic, child.stderr)

        self.assertEqual(report(CAPTURE)['tokens'], 32)
        for original, invalid, diagnostic in (
            ('weights loaded in 71.2s', 'weights loaded in 071.2s',
             'malformed load duration'),
            ('t=1.2s', 't=01.2s', 'malformed generated token row'),
        ):
            with self.subTest(invalid=invalid):
                changed = CAPTURE.replace(original, invalid, 1)
                self.assertNotEqual(changed, CAPTURE)
                with self.assertRaisesRegex(ValueError, diagnostic):
                    report(changed)
                child = self.cli(changed)
                self.assertEqual(child.returncode, 2, child.stderr)
                self.assertEqual(child.stdout, '')
                self.assertIn(diagnostic, child.stderr)


if __name__ == '__main__':
    unittest.main()
