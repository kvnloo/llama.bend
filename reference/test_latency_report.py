"""Unit tests for the stdout-only latency report (no model downloads needed)."""

import json
import math
import subprocess
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from latency_report import report


class LatencyReportTests(unittest.TestCase):
    def test_load_and_intervals(self):
        raw = """weights loaded in 4.400s
prompt ids: 31 23
  id 42  logit 1056964608 0.5  t=0.200s
  id 43  logit 1056964608 0.5  t=0.280s
  id 44  logit 1056964608 0.5  t=0.370s
  id 45  logit 1056964608 0.5  t=0.470s
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
        r = report("  id 4  logit 1065353216 1.0  t=0.009s")
        self.assertEqual(r["tokens"], 1)
        self.assertIsNone(r["decode_mean_ms"])
        self.assertIsNone(r["decode_slope_ms_per_position"])
        self.assertIsNone(r["load_s"])

    def test_missing_token_times(self):
        with self.assertRaisesRegex(ValueError, "no generated token"):
            report("prompt ids: 2 3\ntext: hello\n")

    def test_nonmonotonic_times(self):
        with self.assertRaisesRegex(ValueError, "not monotonic"):
            report("  id 4  logit 1065353216 1.0 t=1.5s\n  id 5  logit 1073741824 2.0 t=1.0s")

    def test_finite_logit_bits_display_disagreement_is_rejected(self):
        # The two native fields encode one F32, never separate measurements.
        examples = ((1056964608, "0.75"), (1065353216, "-1.0"),
                    (0, "1.0"), (1065353216, "0.0"))
        for bits, shown in examples:
            with self.subTest(bits=bits, shown=shown):
                with self.assertRaisesRegex(ValueError, "bits/display mismatch"):
                    report(f"  id 4 logit {bits} {shown} t=0.100s\n")

    def test_real_captured_f32_bit_renderings_roundtrip(self):
        # Validate the retained actual 1.7B generator trace (historical only).
        import re, struct
        capture = (Path(__file__).resolve().parents[1] /
                   "docs/media/outputs/gen_17b.txt").read_text(encoding="utf-8")
        rows = re.findall(r"\blogit\s+(\d+)\s+([+-]?[0-9.eE+-]+)\s+t=", capture)
        self.assertEqual(len(rows), 32)
        for bits, shown in rows:
            with self.subTest(bits=bits):
                self.assertEqual(struct.unpack("<I", struct.pack("<f", float(shown)))[0],
                                 int(bits))
        self.assertEqual(report(capture)["tokens"], 32)

    def test_pinned_f32_show_representative_boundaries(self):
        # Pinned v2.0.37 C-lane rendering; local source reproduction separately
        # checked, not a native Bend compiler or inference run on this host.
        samples = ((0,"0"),(2147483648,"-0"),(1,"1e-45"),
                   (2147483649,"-1e-45"),(8388607,"1.1754942e-38"),
                   (2155872255,"-1.1754942e-38"),(8388608,"1.1754944e-38"),
                   (2155872256,"-1.1754944e-38"),(2139095039,"3.4028235e+38"),
                   (4286578687,"-3.4028235e+38"),(1065353216,"1"),
                   (3212836864,"-1"),(1056964608,"0.5"),
                   (1078530011,"3.1415927"),(925353388,"0.00001"))
        for bits, shown in samples:
            with self.subTest(bits=bits, shown=shown):
                self.assertEqual(report(f"  id 4 logit {bits} {shown} t=0.100s")["tokens"],1)

    def test_duplicate_prompt_cannot_merge_two_generation_traces(self):
        # Joining two captures must not invent a decode interval across runs.
        capture = ("weights loaded in 0.100s\nprompt ids: 7\n"
                   "  id 8 logit 1056964608 0.5 t=0.200s\n"
                   "prompt ids: 8\n"
                   "  id 9 logit 1056964608 0.5 t=0.300s\n"
                   "text: complete\n")
        with self.assertRaisesRegex(ValueError, "duplicate prompt header"):
            report(capture)

    def test_late_prompt_cannot_reclassify_prior_token(self):
        capture = ("weights loaded in 0.100s\n"
                   "  id 8 logit 1056964608 0.5 t=0.200s\n"
                   "prompt ids: 7\ntext: complete\n")
        with self.assertRaisesRegex(ValueError, "prompt header appears after generated token"):
            report(capture)
        # Token-only diagnostic traces remain valid, and decoded text is inert.
        self.assertEqual(report("  id 8 logit 1056964608 0.5 t=0.200s\n")["tokens"], 1)
        self.assertEqual(report("prompt ids: 7\n  id 8 logit 1056964608 0.5 t=0.200s\n"
                                "text: prompt ids: 9\n")["tokens"], 1)

    def test_duplicate_prompt_header_cli_fails_closed(self):
        # Traces accidentally concatenated by the shell must never be sampled.
        capture = ("weights loaded in 0.100s\nprompt ids: 7\n"
                   "  id 8 logit 1056964608 0.5 t=0.200s\n"
                   "prompt ids: 8\n"
                   "  id 9 logit 1056964608 0.5 t=0.300s\n"
                   "text: complete\n")
        runner = Path(__file__).resolve().parents[1] / "scripts/latency_report.py"
        child = subprocess.run([sys.executable, str(runner), "--json"],
                               input=capture, text=True, capture_output=True,
                               check=False)
        self.assertNotEqual(child.returncode, 0)
        self.assertEqual(child.stdout, "")
        self.assertIn("duplicate prompt header", child.stderr)

    def test_late_prompt_header_cli_fails_closed(self):
        # A late prompt header cannot reclassify a generated token as prefill.
        capture = ("weights loaded in 0.100s\n"
                   "  id 8 logit 1056964608 0.5 t=0.200s\n"
                   "prompt ids: 7\ntext: complete\n")
        runner = Path(__file__).resolve().parents[1] / "scripts/latency_report.py"
        child = subprocess.run([sys.executable, str(runner), "--json"],
                               input=capture, text=True, capture_output=True,
                               check=False)
        self.assertNotEqual(child.returncode, 0)
        self.assertEqual(child.stdout, "")
        self.assertIn("prompt header appears after generated token", child.stderr)

    def test_malformed_prompt_headers_fail_closed(self):
        # Native start2 can only emit decimal Nat IDs separated by spaces.
        # A header that isn't native must not bless a fabricated capture.
        for bad in ("prompt ids: junk", "prompt ids: ",
                    "prompt ids: 7 -8", "prompt ids: 7xyz",
                    "prompt ids: 7\t9", "prompt ids: 7  9"):
            with self.subTest(header=bad):
                capture = ("weights loaded in 0.100s\n" + bad + "\n"
                           "  id 8 logit 1056964608 0.5 t=0.200s\n"
                           "text: complete\n")
                with self.assertRaisesRegex(ValueError, "malformed prompt header"):
                    report(capture)
        # Preserve the exact native trailing-space form and CRLF transports.
        valid = ("weights loaded in 0.100s\r\nprompt ids: 7 8 \r\n"
                 "  id 9 logit 1056964608 0.5 t=0.200s\r\n"
                 "text: complete\r\n")
        self.assertEqual(report(valid)["tokens"], 1)

    def test_indented_prompt_header_cannot_disable_completion_check(self):
        # gen.bend prints headers at column zero. A leading space/tab must
        # not silently turn a full run into a token-only incomplete trace.
        for header in (" prompt ids: 7", "\tprompt ids: 7", "\u00a0prompt ids: 7"):
            with self.subTest(header=header):
                capture = ("weights loaded in 0.100s\n" + header + "\n"
                           "  id 8 logit 1056964608 0.5 t=0.200s\n")
                with self.assertRaisesRegex(ValueError, "malformed prompt header"):
                    report(capture)

    def test_indented_prompt_header_cli_emits_no_metrics(self):
        runner = Path(__file__).resolve().parents[1] / "scripts/latency_report.py"
        capture = ("weights loaded in 0.100s\n prompt ids: 7\n"
                   "  id 8 logit 1056964608 0.5 t=0.200s\n")
        child = subprocess.run([sys.executable, str(runner), "--json"],
                               input=capture, text=True, capture_output=True,
                               check=False)
        self.assertNotEqual(child.returncode, 0)
        self.assertEqual(child.stdout, "")
        self.assertIn("malformed prompt header", child.stderr)

    def test_malformed_prompt_header_cli_has_no_json(self):
        capture = ("weights loaded in 0.100s\nprompt ids: 7xyz\n"
                   "  id 8 logit 1056964608 0.5 t=0.200s\n"
                   "text: complete\n")
        runner = Path(__file__).resolve().parents[1] / "scripts/latency_report.py"
        child = subprocess.run([sys.executable, str(runner), "--json"],
                               input=capture, text=True, capture_output=True,
                               check=False)
        self.assertNotEqual(child.returncode, 0)
        self.assertEqual(child.stdout, "")
        self.assertIn("malformed prompt header", child.stderr)

    def test_full_generation_requires_completion_footer(self):
        # The real generator prints decoded text only after finishing decode.
        partial = ("weights loaded in 1.000s\nprompt ids: 1 2\n"
                   "  id 4 logit 1056964608 0.5 t=0.100s\n"
                   "  id 5 logit 1056964608 0.5 t=0.200s\n")
        for suffix in ("", "text:\n", "text:truncated\n"):
            with self.subTest(suffix=suffix):
                with self.assertRaisesRegex(ValueError, "missing decoded-text footer"):
                    report(partial + suffix)
        self.assertEqual(report(partial + "text: completed\n")["tokens"], 2)
        self.assertEqual(report("  id 4 logit 1065353216 1.0 t=0.100s\n")["tokens"], 1)

    def test_partial_generation_cli_does_not_emit_metrics(self):
        partial = ("prompt ids: 1\n"
                   "  id 4 logit 1056964608 0.5 t=0.100s\n")
        result = subprocess.run(
            [sys.executable, str(Path(__file__).resolve().parents[1]
                                 / "scripts/latency_report.py"), "--json"],
            input=partial, text=True, capture_output=True, check=False,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout, "")
        self.assertIn("missing decoded-text footer", result.stderr)

    def test_captured_generation_output(self):
        # Real 1.7B output tracked for the release video (timestamps may be rounded).
        path = Path(__file__).resolve().parents[1] / "docs/media/outputs/gen_17b.txt"
        r = report(path.read_text(encoding="utf-8"))
        self.assertGreater(r["tokens"], 16)
        self.assertIsNotNone(r["load_s"])
        self.assertGreater(r["decode_p50_ms"], 0)
        self.assertGreater(r["first_generated_ms_since_generation_start"], 0)

    def test_unterminated_native_footer_rejected(self):
        # Native IO.print emits a line terminator. A partial pipe or file
        # capture must not create apparently completed latency metrics.
        prefix = ("weights loaded in 1.0s\nprompt ids: 1\n"
                  "  id 4 logit 1056964608 0.5 t=0.100s\n")
        for suffix in ("text: ", "text: partial", "text: partially decoded 💡", "text: done\r"):
            with self.subTest(suffix=suffix):
                with self.assertRaisesRegex(ValueError, "unterminated decoded-text footer"):
                    report(prefix + suffix)

    def test_terminated_native_footer_keeps_metrics(self):
        prefix = ("weights loaded in 1.0s\nprompt ids: 1\n"
                  "  id 4 logit 1056964608 0.5 t=0.100s\n")
        for suffix in ("text: done\n", "text: done\r\n", "text: \n"):
            with self.subTest(suffix=suffix):
                self.assertEqual(report(prefix + suffix)["tokens"], 1)
        # Content after the first *terminated* footer is decoded text,
        # never additional timestamps, even without an ending newline.
        spliced = prefix + "text: done\n  id 999 logit 0 0.0 t=99s"
        self.assertEqual(report(spliced)["tokens"], 1)

    def test_malformed_generated_token_rows_are_not_silently_skipped(self):
        # Skipping one row manufactures a seemingly valid 100ms decode
        # distribution from a corrupt/incomplete token sequence.
        for invalid in ("t=nans", "t=infs", "t=-0.002s", "t=wrong", ""):
            with self.subTest(token=invalid):
                raw = (
                    "weights loaded in 1.0s\nprompt ids: 1\n"
                    "  id 4 logit 1056964608 0.5 t=0.100s\n"
                    f"  id 5 logit 1056964608 0.5 {invalid}\n"
                    "  id 6 logit 1056964608 0.5 t=0.200s\n"
                    "text: done\n"
                )
                with self.assertRaisesRegex(ValueError, "malformed generated token row"):
                    report(raw)

    def test_truncated_logit_rows_cannot_invent_decode_samples(self):
        # All generated rows have logit bits and an F32 rendering before t=.
        for malformed in (
            "  id 4 logit t=0.100s",
            "  id 4 logit nonsense t=0.100s",
            "  id 4 logit 1099424380 t=0.100s",
            "  id 4 logit 1099379507 16.9 trailing t=0.100s",
            "  id 4 logit 1099424380 nonsense t=0.100s",
            "  id 4 logit 1099424380 nan t=0.100s",
            "  id 4 logit 1099424380 inf t=0.100s",
        ):
            with self.subTest(row=malformed):
                with self.assertRaisesRegex(ValueError, "malformed generated token row"):
                    report(malformed + "\n  id 5 logit 1056964608 0.5 t=0.200s")

    def test_generated_logit_bits_must_fit_u32(self):
        # demo/gen.bend prints U32.show(bits(F32)), never wider integers.
        for bits in ("4294967296", "9999999999999999", "9" * 5000):
            with self.subTest(bits=bits[:32]):
                with self.assertRaisesRegex(ValueError, "invalid generated token logit bits"):
                    report(f"  id 4 logit {bits} 0.5 t=0.100s\n"
                           "  id 5 logit 1058642330 0.6 t=0.200s")

    def test_non_ascii_numerals_are_not_native_generated_token_rows(self):
        # Nat.show, U32.show, F32.show and tsec2 emit ASCII decimal numerals.
        # Python's Unicode \d and float/int accept non-native numeral scripts.
        for row in (
            "  id ٤ logit 1056964608 0.5 t=0.100s",
            "  id 4 logit ١ 0.5 t=0.100s",
            "  id 4 logit 1 ٠.٥ t=0.100s",
            "  id 4 logit 1056964608 0.5 t=٠.100s",
        ):
            with self.subTest(row=row):
                with self.assertRaisesRegex(ValueError, "malformed generated token row"):
                    report(row + "\n  id 5 logit 1056964608 0.5 t=0.200s")

    def test_non_ascii_numerals_are_not_native_load_duration(self):
        with self.assertRaisesRegex(ValueError, "malformed load duration"):
            report("weights loaded in ١.٢s\n  id 4 logit 1056964608 0.5 t=0.100s")

    def test_largest_finite_f32_bit_pattern_is_accepted(self):
        metrics = report("  id 4 logit 2139095039 3.4028235e+38 t=0.100s\n"
                         "  id 5 logit 1056964608 0.5 t=0.200s")
        self.assertEqual(metrics["tokens"], 2)
        self.assertAlmostEqual(metrics["decode_p50_ms"], 100.0)

    def test_nonfinite_raw_f32_logit_bits_cannot_create_false_timing_samples(self):
        # These are syntactically valid U32 bit patterns, but none encodes
        # a finite F32. A finite-looking display must not launder them.
        for bits in (
            2139095040,  # +infinity: 0x7f800000
            4286578688,  # -infinity: 0xff800000
            2143289344,  # quiet NaN: 0x7fc00000
            2139095041,  # signaling NaN payload: 0x7f800001
            4290772992,  # negative quiet NaN: 0xffc00000
        ):
            with self.subTest(bits=bits):
                raw = (f"  id 4 logit {bits} 0.5 t=0.100s\n"
                       "  id 5 logit 1056964608 0.5 t=0.200s")
                with self.assertRaisesRegex(ValueError, "nonfinite generated token logit bits"):
                    report(raw)

    def test_negative_largest_finite_f32_bits_remain_accepted(self):
        metrics = report("  id 4 logit 4286578687 -3.4028235e+38 t=0.100s\n"
                         "  id 5 logit 1056964608 0.5 t=0.200s")
        self.assertEqual(metrics["tokens"], 2)
        self.assertAlmostEqual(metrics["decode_p50_ms"], 100.0)

    def test_overflowing_displayed_f32_cannot_create_false_timing_samples(self):
        # These are valid decimal spellings and used to pass despite being
        # impossible finite F32 values in the generator's output.
        for shown, diagnostic in (
            ("1e309", "nonfinite generated token logit"),
            ("-1e309", "nonfinite generated token logit"),
            ("3.5e38", "generated token logit outside F32 range"),
            ("-3.5e38", "generated token logit outside F32 range"),
        ):
            with self.subTest(shown=shown):
                raw = (f"  id 4 logit 1 {shown} t=0.100s\n"
                       "  id 5 logit 1056964608 0.5 t=0.200s")
                with self.assertRaisesRegex(ValueError, diagnostic):
                    report(raw)

    def test_scientific_logit_rendering_does_not_change_timing(self):
        r = report("  id 4 logit 3077235083 -1.4e-05 t=0.100s\n  id 5 logit 1130758144 2.3e+02 t=0.200s")
        self.assertEqual(r["tokens"], 2)
        self.assertAlmostEqual(r["decode_p50_ms"], 100.0)

    def test_malformed_rows_after_decoded_text_are_ignored(self):
        raw = (
            "  id 4 logit 1056964608 0.5 t=0.100s\n"
            "text: generated text might quote parser lines\n"
            "  id 5 logit 1056964608 0.5 t=nans\n"
        )
        self.assertEqual(report(raw)["tokens"], 1)

    def test_near_miss_prompt_header_cannot_turn_truncated_capture_into_success(self):
        # A corrupt header must not disable the required completion footer.
        capture = (Path(__file__).resolve().parents[1]
                   / "docs/media/outputs/gen_17b.txt").read_text(encoding="utf-8")
        self.assertEqual(report(capture)["tokens"], 32)
        for malformed in ("prompt  ids:", "prompt\tids:", "prompt ids :"):
            with self.subTest(malformed=repr(malformed)):
                truncated = capture.replace("prompt ids:", malformed, 1).split("text: ")[0]
                with self.assertRaisesRegex(ValueError, "malformed prompt header"):
                    report(truncated)

    def test_near_miss_load_header_cannot_silently_erase_load_clock(self):
        capture = (Path(__file__).resolve().parents[1]
                   / "docs/media/outputs/gen_17b.txt").read_text(encoding="utf-8")
        self.assertEqual(report(capture)["load_s"], 71.2)
        for malformed in ("weights  loaded in", "weights\tloaded in", "weights loaded\tin"):
            with self.subTest(malformed=repr(malformed)):
                corrupted = capture.replace("weights loaded in", malformed, 1)
                with self.assertRaisesRegex(ValueError, "malformed load duration"):
                    report(corrupted)

    def test_indented_native_load_header_must_not_be_silently_discarded(self):
        # Header is column-zero in demo/gen.bend. A corrupt/indented load
        # must not turn a complete 1.7B capture into load_s=None metrics.
        capture = (Path(__file__).resolve().parents[1]
                   / "docs/media/outputs/gen_17b.txt").read_text(encoding="utf-8")
        self.assertTrue(capture.startswith("weights loaded in "))
        baseline = report(capture)
        self.assertEqual((baseline["load_s"], baseline["tokens"],
                          baseline["decode_p50_ms"], baseline["decode_p95_ms"]),
                         (71.2, 32, 400.0, 500.0))
        for leading in (" ", "\t", "\u00a0"):
            with self.subTest(leading=repr(leading)):
                with self.assertRaisesRegex(ValueError, "malformed load duration"):
                    report(leading + capture)

    def test_malformed_load_clock_is_not_silently_treated_as_missing(self):
        # A capture can omit the load line (piped generation output), but
        # an explicitly malformed load receipt must invalidate that metric.
        for malformed in ("nans", "infs", "-0.100s", "tbd", ""):
            with self.subTest(malformed=malformed):
                raw = f"weights loaded in {malformed}\n  id 4 logit 1065353216 1.0 t=0.100s\n"
                with self.assertRaisesRegex(ValueError, "malformed load duration"):
                    report(raw)

    def test_load_duration_after_prompt_is_rejected(self):
        # Native generation loads weights before prompt tokenization. A later
        # load line is a mislabeled or spliced timing receipt, not a valid load.
        raw = (
            "prompt ids: 1 2\n"
            "weights loaded in 71.2s\n"
            "  id 4 logit 1056964608 0.5 t=0.100s\n"
            "text: done\n"
        )
        with self.assertRaisesRegex(ValueError, "load duration appears after prompt or token output"):
            report(raw)

    def test_load_duration_after_generated_token_is_rejected(self):
        raw = (
            "  id 4 logit 1056964608 0.5 t=0.100s\n"
            "weights loaded in 71.2s\n"
            "  id 5 logit 1056964608 0.5 t=0.200s\n"
            "text: done\n"
        )
        with self.assertRaisesRegex(ValueError, "load duration appears after prompt or token output"):
            report(raw)

    def test_duplicate_load_clock_does_not_override_first_measurement(self):
        raw = (
            "weights loaded in 9.000s\n"
            "weights loaded in 0.001s\n"
            "  id 4 logit 1065353216 1.0 t=0.100s\n"
        )
        with self.assertRaisesRegex(ValueError, "duplicate load duration"):
            report(raw)

    def test_noisy_output(self):
        r = report("Random t=3.7s\n  id 5  logit 1073741824 2.0 t=1.0s\nignored\n  id 6  logit 1077936128 3.0 t=1.1s")
        self.assertEqual(r["tokens"], 2)
        self.assertAlmostEqual(r["decode_mean_ms"], 100.0)


    def test_timestamp_decimal_overflow_is_rejected(self):
        huge = "9" * 400
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            report(f"  id 4 logit 1065353216 1.0 t={huge}s")
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            report(f"weights loaded in {huge}s\n  id 4 logit 1065353216 1.0 t=0.001s")

    def test_finite_timestamp_millisecond_conversion_overflow_is_rejected(self):
        huge_finite = "1" + "0" * 308
        self.assertTrue(math.isfinite(float(huge_finite)))
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            report(f"  id 4 logit 1065353216 1.0 t={huge_finite}s")



    def test_decoded_text_cannot_spoof_real_capture_timing(self):
        # The decoded text can contain complete log-looking lines on new lines.
        # The real 1.7B capture must keep identical measurements after the text.
        path = Path(__file__).resolve().parents[1] / "docs/media/outputs/gen_17b.txt"
        capture = path.read_text(encoding="utf-8")
        injected = capture + (
            "\n  id 999  logit 0 0.0  t=999.000s\n"
            "weights loaded in 0.005s\n"
            "  id 123  logit 0 0.0  t=0.001s\n"
        )
        self.assertEqual(report(injected), report(capture))

    def test_decoded_text_cannot_invent_generated_tokens(self):
        raw = (
            "weights loaded in 1.000s\n"
            "text: Here is a copied generator log:\n"
            "  id 4  logit 1056964608 0.5  t=0.100s\n"
            "  id 5  logit 1056964608 0.5  t=0.200s\n"
        )
        with self.assertRaisesRegex(ValueError, "no generated token"):
            report(raw)

class LatencyCLIIntegrationTests(unittest.TestCase):
    """Exercise the published CLI boundary, not only the report() helper."""

    def cli(self, *args, data=None):
        script = Path(__file__).resolve().parents[1] / "scripts/latency_report.py"
        return subprocess.run(
            [sys.executable, str(script), "--json", *args],
            input=data, text=True, capture_output=True, check=False,
        )

    def test_indented_load_clock_cli_rejects_false_missing_load_metric(self):
        # Fail closed rather than emitting a plausible decode p50/p95 and
        # silently losing the measured model load time.
        capture = (Path(__file__).resolve().parents[1]
                   / "docs/media/outputs/gen_17b.txt").read_text(encoding="utf-8")
        proc = self.cli(data=" " + capture)
        self.assertEqual(proc.returncode, 2)
        self.assertEqual(proc.stdout, "")
        self.assertIn("malformed load duration", proc.stderr)

    def test_corrupt_prompt_header_truncated_cli_emits_no_metrics(self):
        capture = (Path(__file__).resolve().parents[1]
                   / "docs/media/outputs/gen_17b.txt").read_text(encoding="utf-8")
        partial = capture.replace("prompt ids:", "prompt  ids:", 1).split("text: ")[0]
        proc = self.cli(data=partial)
        self.assertEqual(proc.returncode, 2)
        self.assertEqual(proc.stdout, "")
        self.assertIn("malformed prompt header", proc.stderr)

    def test_real_capture_file_matches_stdin_json(self):
        capture = Path(__file__).resolve().parents[1] / "docs/media/outputs/gen_17b.txt"
        file_run = self.cli(str(capture))
        stdin_run = self.cli(data=capture.read_text(encoding="utf-8"))
        self.assertEqual(file_run.returncode, 0, file_run.stderr)
        self.assertEqual(stdin_run.returncode, 0, stdin_run.stderr)
        self.assertEqual(json.loads(file_run.stdout), json.loads(stdin_run.stdout))
        self.assertEqual(json.loads(file_run.stdout)["tokens"], 32)

    def test_cli_unterminated_footer_emits_no_metrics(self):
        raw = ("prompt ids: 1\n"
               "  id 4 logit 1056964608 0.5 t=0.100s\n"
               "text: partial")
        proc = self.cli(data=raw)
        self.assertEqual(proc.returncode, 2)
        self.assertEqual(proc.stdout, "")
        self.assertIn("unterminated decoded-text footer", proc.stderr)

    def test_finite_f32_mismatch_fails_closed_at_cli(self):
        sample = ("weights loaded in 0.100s\nprompt ids: 7\n"
                  "  id 8 logit 1056964608 0.5 t=0.200s\ntext: done\n")
        valid=self.cli(data=sample)
        self.assertEqual(valid.returncode,0,valid.stderr)
        self.assertEqual(json.loads(valid.stdout)["tokens"],1)
        corrupt=self.cli(data=sample.replace("logit 1056964608 0.5", "logit 1056964608 0.75"))
        self.assertEqual(corrupt.returncode,2)
        self.assertEqual(corrupt.stdout,"")
        self.assertIn("bits/display mismatch",corrupt.stderr)

    def test_invalid_u32_logit_bits_fail_at_cli_boundary(self):
        bad = "  id 4 logit 4294967296 0.5 t=0.100s\n"
        result = self.cli(data=bad)
        self.assertEqual(result.returncode, 2)
        self.assertIn("invalid generated token logit bits", result.stderr)
        self.assertEqual(result.stdout, "")

    def test_decoded_text_spoof_does_not_affect_cli_json(self):
        capture = Path(__file__).resolve().parents[1] / "docs/media/outputs/gen_17b.txt"
        original = capture.read_text(encoding="utf-8")
        baseline = self.cli(data=original)
        injected = self.cli(data=original + "\n  id 999 logit 1056964608 0.5 t=999.000s\n")
        self.assertEqual(baseline.returncode, 0, baseline.stderr)
        self.assertEqual(injected.returncode, 0, injected.stderr)
        self.assertEqual(json.loads(injected.stdout), json.loads(baseline.stdout))

    def test_non_ascii_numbers_fail_at_cli_boundary(self):
        # Python accepts Unicode decimal digits in regex/float/int, but the
        # native Bend generator prints ASCII-only numbers.
        for receipt in (
            "  id ٤ logit 1056964608 0.5 t=0.100s\n",
            "  id 4 logit ١ 0.5 t=0.100s\n",
            "  id 4 logit 1 ٠.٥ t=0.100s\n",
            "  id 4 logit 1056964608 0.5 t=٠.100s\n",
        ):
            with self.subTest(receipt=receipt):
                result = self.cli(data=receipt)
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertIn("malformed generated token row", result.stderr)
                self.assertEqual(result.stdout, "")

if __name__ == "__main__":
    unittest.main()
