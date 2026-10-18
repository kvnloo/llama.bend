"""Numerical and evidence-boundary regressions; no device or model is needed.

Run from the repository root: python3 -m unittest discover -s reference -p test_adreno_gemv.py -v
These tests do not replace the physical three-pattern comparison in run-device.py.
"""
import argparse
import importlib.util
import json
import math
from pathlib import Path
import struct
import sys
import tempfile
import unittest

ADRENO = Path(__file__).resolve().parents[1] / "effects" / "adreno"
sys.path.insert(0, str(ADRENO))
from validation import (ABS_TOL, REL_TOL, ROWS, ValidationError, compare_outputs,
                        compare_values, parse_bits, parse_metadata, within_tolerance)

spec = importlib.util.spec_from_file_location("adreno_device_runner", ADRENO / "run-device.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def word(value):
    return struct.unpack("<I", struct.pack("<f", value))[0]


def line(target="CPU", pattern=0, values=None):
    values = [0.0] * ROWS if values is None else values
    return f"{target}_GEMV_BITS pattern={pattern} values=" + " ".join(str(word(x)) for x in values) + "\n"


METADATA = """tensor_bytes=1327104 rows=576 cols=576 pattern=0
platform=Qualcomm OpenCL
platform_vendor=Qualcomm
platform_version=OpenCL 3.0
vendor=QUALCOMM
device=Adreno GPU
device_version=OpenCL 3.0 Adreno
driver_version=vendor-driver
device_type=4 gpu_required=true cpu_fallback=false
kernel_build_status=0 kernel_build_options=-cl-std=CL1.2 fp_contract=off
enqueue_ndrange_status=0 global=576 local=driver-selected
kernel_event_execution_status=0 blocking_readback=success
"""


class NumericTests(unittest.TestCase):
    def test_all_rows_all_patterns(self):
        for pattern in range(3):
            values = [(-1.0 if row % 2 else 1.0) * row / 16 for row in range(ROWS)]
            result = compare_outputs(line(pattern=pattern, values=values),
                                     METADATA.replace("pattern=0", f"pattern={pattern}")
                                     + line("ADRENO", pattern, values), pattern)
            self.assertTrue(result["acceptance_met"])
            self.assertEqual(result["compared_rows"], ROWS)
            self.assertEqual(result["max_absolute_error"], 0.0)

    def test_wrong_final_row_is_rejected(self):
        cpu = [1.0] * ROWS
        gpu = cpu.copy()
        gpu[-1] = -1.0
        result = compare_outputs(line(values=cpu), METADATA + line("ADRENO", values=gpu), 0)
        self.assertFalse(result["acceptance_met"])
        self.assertEqual(result["failed_rows"], [ROWS - 1])
        self.assertEqual(result["worst_absolute_row"], ROWS - 1)
        self.assertEqual(result["max_absolute_error"], 2.0)

    def test_multiple_wrong_rows_are_not_short_circuited(self):
        gpu = [0.0] * ROWS
        gpu[0], gpu[287], gpu[-1] = 1.0, -2.0, 3.0
        result = compare_values([0.0] * ROWS, gpu)
        self.assertEqual(result["failed_rows"], [0, 287, ROWS - 1])
        self.assertEqual(result["max_absolute_error"], 3.0)

    def test_signed_zeros_are_numerically_equal(self):
        result = compare_outputs(line(values=[0.0] * ROWS),
                                 METADATA + line("ADRENO", values=[-0.0] * ROWS), 0)
        self.assertTrue(result["acceptance_met"])
        self.assertEqual(result["max_relative_error"], 0.0)

    def test_absolute_boundary_is_inclusive(self):
        self.assertTrue(within_tolerance(ABS_TOL, 0.0))
        self.assertTrue(within_tolerance(math.nextafter(ABS_TOL, 0.0), 0.0))
        self.assertFalse(within_tolerance(math.nextafter(ABS_TOL, math.inf), 0.0))

    def test_relative_boundary_is_inclusive_and_symmetric_in_reference_sign(self):
        for reference in (1.0, -1.0, 4096.0, -4096.0):
            boundary = ABS_TOL + REL_TOL * abs(reference)
            self.assertTrue(within_tolerance(boundary, reference))
            self.assertTrue(within_tolerance(math.nextafter(boundary, 0.0), reference))
            self.assertFalse(within_tolerance(math.nextafter(boundary, math.inf), reference))

    def test_relative_policy_uses_cpu_not_gpu_magnitude(self):
        cpu, gpu = [1.0] * ROWS, [1.0] * ROWS
        gpu[-1] = 1.00022002
        result = compare_values(cpu, gpu)
        self.assertFalse(result["acceptance_met"])
        self.assertEqual(result["failed_rows"], [ROWS - 1])

    def test_representable_values_on_both_sides_of_tolerance(self):
        cpu, gpu = [1.0] * ROWS, [1.0] * ROWS
        gpu[-1] = 1.0002
        self.assertTrue(compare_outputs(line(values=cpu), METADATA + line("ADRENO", values=gpu), 0)["acceptance_met"])
        gpu[-1] = 1.0003
        self.assertFalse(compare_outputs(line(values=cpu), METADATA + line("ADRENO", values=gpu), 0)["acceptance_met"])

    def test_relative_error_reporting_for_zero_reference_stays_finite(self):
        gpu = [0.0] * ROWS
        gpu[-1] = ABS_TOL / 2
        result = compare_values([0.0] * ROWS, gpu)
        self.assertTrue(result["acceptance_met"])
        self.assertTrue(math.isfinite(result["max_relative_error"]))
        self.assertEqual(result["worst_relative_row"], ROWS - 1)

    def test_shape_mismatch_on_either_side(self):
        for left, right in ((ROWS - 1, ROWS), (ROWS, ROWS - 1), (ROWS + 1, ROWS)):
            with self.subTest(left=left, right=right), self.assertRaises(ValidationError):
                compare_values([0.0] * left, [0.0] * right)

    def test_nonfinite_values_on_either_side(self):
        for nonfinite in (math.nan, math.inf, -math.inf):
            for side in (0, 1):
                values = [[0.0] * ROWS, [0.0] * ROWS]
                values[side][-1] = nonfinite
                with self.subTest(nonfinite=nonfinite, side=side), self.assertRaises(ValidationError):
                    compare_values(*values)


class ProtocolTests(unittest.TestCase):
    def test_missing_final_row(self):
        with self.assertRaises(ValidationError):
            parse_bits(line(values=[0.0] * (ROWS - 1)), "CPU", 0)

    def test_extra_row(self):
        with self.assertRaises(ValidationError):
            parse_bits(line(values=[0.0] * (ROWS + 1)), "CPU", 0)

    def test_pattern_mismatch(self):
        with self.assertRaises(ValidationError):
            compare_outputs(line(pattern=0), METADATA + line("ADRENO", 1), 0)

    def test_missing_result(self):
        with self.assertRaises(ValidationError):
            parse_bits("", "ADRENO", 0)

    def test_duplicate_result(self):
        with self.assertRaises(ValidationError):
            parse_bits(line() + line(), "CPU", 0)

    def test_malformed_result_header(self):
        for bad in (line().replace("pattern=0", "pattern=3"), line().replace(" values=", " "),
                    line().replace("pattern=0", "pattern=ones")):
            with self.subTest(header=bad[:55]), self.assertRaises(ValidationError):
                parse_bits(bad, "CPU", 0)

    def test_corrupt_final_word(self):
        for bad in ("-1", "4294967296", "1.0", "NaN", "+0", "0x0", "9" * 20):
            corrupt = line().rstrip().rsplit(" ", 1)[0] + " " + bad + "\n"
            with self.subTest(word=bad), self.assertRaises(ValidationError):
                parse_bits(corrupt, "CPU", 0)

    def test_nonfinite_final_word_on_both_targets(self):
        for target in ("CPU", "ADRENO"):
            for value in (math.nan, math.inf, -math.inf):
                values = [0.0] * ROWS
                values[-1] = value
                with self.subTest(target=target, value=value), self.assertRaises(ValidationError):
                    parse_bits(line(target, values=values), target, 0)

    def test_nonfinite_before_final_row(self):
        values = [0.0] * ROWS
        values[191] = math.nan
        with self.assertRaises(ValidationError):
            parse_bits(line(values=values), "CPU", 0)

    def test_metadata_retains_full_identity_with_spaces(self):
        metadata = parse_metadata(METADATA)
        self.assertEqual(metadata["platform"], "Qualcomm OpenCL")
        self.assertEqual(metadata["device_version"], "OpenCL 3.0 Adreno")
        self.assertEqual(metadata["driver_version"], "vendor-driver")

    def test_successful_completion_without_gpu_metadata_is_rejected(self):
        with self.assertRaises(ValidationError):
            compare_outputs(line(), line("ADRENO"), 0)

    def test_cpu_fallback_is_rejected(self):
        for old, new in (("cpu_fallback=false", "cpu_fallback=true"), ("gpu_required=true", "gpu_required=false"),
                         ("device_type=4", "device_type=2"), ("device_type=4", "device_type=6")):
            with self.subTest(change=new), self.assertRaises(ValidationError):
                parse_metadata(METADATA.replace(old, new))

    def test_failed_enqueue_event_readback_or_wrong_shape_is_rejected(self):
        for old, new in (("enqueue_ndrange_status=0", "enqueue_ndrange_status=-5"),
                         ("kernel_event_execution_status=0", "kernel_event_execution_status=-1"),
                         ("blocking_readback=success", "blocking_readback=failed"), ("global=576", "global=575")):
            with self.subTest(change=new), self.assertRaises(ValidationError):
                parse_metadata(METADATA.replace(old, new))

    def test_missing_duplicate_or_empty_identity_is_rejected(self):
        for broken in (METADATA.replace("driver_version=vendor-driver\n", ""),
                       METADATA + "device=other GPU\n", METADATA.replace("vendor=QUALCOMM", "vendor=")):
            with self.assertRaises(ValidationError):
                parse_metadata(broken)

    def test_tensor_shape_and_build_status_are_required(self):
        for old, new in (("tensor_bytes=1327104", "tensor_bytes=1327100"),
                         ("rows=576", "rows=575"), ("cols=576", "cols=575"),
                         ("kernel_build_status=0", "kernel_build_status=-11")):
            with self.subTest(change=new), self.assertRaises(ValidationError):
                parse_metadata(METADATA.replace(old, new))
        for field in ("tensor_bytes=1327104 ", "rows=576 ", "cols=576 ",
                      "kernel_build_status=0 "):
            with self.subTest(missing=field), self.assertRaises(ValidationError):
                parse_metadata(METADATA.replace(field, ""))

    def test_metadata_pattern_must_match_the_requested_case(self):
        for pattern in (1, 2, 3, -1):
            with self.subTest(pattern=pattern), self.assertRaises(ValidationError):
                compare_outputs(line(), METADATA.replace("pattern=0", f"pattern={pattern}")
                                + line("ADRENO"), 0)


class ReceiptTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.current = self.root / "latest.json"

    def receipt(self):
        return json.loads(self.current.read_text())

    def evidence(self, name="attempt"):
        return runner.Evidence(self.root / name, self.current, "regression")

    def test_new_attempt_immediately_replaces_stale_success(self):
        runner.atomic_json(self.current, dict(status="completed", acceptance_met=True))
        evidence = self.evidence()
        self.assertFalse(self.receipt()["acceptance_met"])
        self.assertEqual(self.receipt()["status"], "running")
        evidence.finish(False, "preflight failed")
        self.assertFalse(self.receipt()["acceptance_met"])
        self.assertEqual(self.receipt()["status"], "failed")

    def test_real_nonzero_exit_is_preserved(self):
        evidence = self.evidence()
        result = evidence.invoke("failure", [sys.executable, "-c", "raise SystemExit(7)"], 10)
        self.assertEqual(result["exit_code"], 7)
        self.assertFalse(result["acceptance_met"])
        with self.assertRaises(RuntimeError):
            runner.require_success(result)
        evidence.finish(False)
        self.assertEqual(self.receipt()["commands"][0]["exit_code"], 7)

    def test_real_timeout_retains_partial_output_without_inventing_exit(self):
        evidence = self.evidence()
        command = [sys.executable, "-c", "import os,time; os.write(1,b'partial stdout\\n'); "
                   "os.write(2,b'partial stderr\\n'); time.sleep(60)"]
        result = evidence.invoke("deadline", command, 1)
        self.assertTrue(result["timed_out"])
        self.assertIsNone(result["exit_code"])
        self.assertIn("partial stdout", result["stdout"])
        self.assertIn("partial stderr", result["stderr"])
        self.assertFalse(result["acceptance_met"])
        self.assertFalse(runner.accepted_refusal(result))
        evidence.finish(False)
        self.assertIsNone(self.receipt()["commands"][0]["exit_code"])

    def test_launch_failure_has_no_invented_exit(self):
        evidence = self.evidence()
        result = evidence.invoke("missing", [str(self.root / "absent-executable")], 1)
        self.assertEqual(result["status"], "launch-error")
        self.assertIsNone(result["exit_code"])
        self.assertFalse(result["acceptance_met"])
        self.assertFalse(runner.accepted_refusal(result))

    def test_remote_deadline_exit_is_failure_not_refusal(self):
        evidence = self.evidence()
        for code in (124, 137):
            result = evidence.invoke("remote-deadline", [sys.executable, "-c", f"raise SystemExit({code})"],
                                     10, remote_deadline=True)
            self.assertEqual(result["exit_code"], code)
            self.assertTrue(result["timed_out"])
            self.assertFalse(result["acceptance_met"])
            self.assertFalse(runner.accepted_refusal(result))

    def test_completed_case_survives_later_failure_and_new_attempt(self):
        first = self.evidence("first")
        numerical = compare_values([1.0] * ROWS, [1.0] * ROWS)
        first.data["cases"].append(dict(mode="ones", **numerical))
        first.persist()
        failure = first.invoke("failure", [sys.executable, "-c", "raise SystemExit(5)"], 10)
        first.finish(False)
        self.assertFalse(failure["acceptance_met"])
        self.assertTrue(self.receipt()["cases"][0]["acceptance_met"])
        second = self.evidence("second")
        second.finish(False, "a new preflight failure")
        retained = json.loads((self.root / "first" / "receipt.json").read_text())
        self.assertTrue(retained["cases"][0]["acceptance_met"])
        self.assertFalse(retained["acceptance_met"])
        self.assertNotEqual(retained["attempt_id"], self.receipt()["attempt_id"])

    def test_sha256_mismatch_missing_or_duplicate_device_file_is_rejected(self):
        digest = "a" * 64
        name = runner.REMOTE_PREFIX + "abcdefgh/model/w/h0.q.bin"
        expected = {name: digest}
        runner.validate_remote_hashes(digest + "  " + name + "\n", expected)
        for invalid in ("", "b" * 64 + "  " + name + "\n", (digest + "  " + name + "\n") * 2):
            with self.assertRaises(ValidationError):
                runner.validate_remote_hashes(invalid, expected)

    def test_unowned_or_malicious_remote_path_is_rejected_before_push(self):
        valid = runner.REMOTE_PREFIX + "abcdefgh"
        self.assertEqual(runner.validate_remote_path(valid), valid)
        for invalid in ("/data/local/tmp", runner.REMOTE_PREFIX + "../existing", valid + "/subdir",
                        valid + "\n" + valid, valid + "; touch /data/local/tmp/existing", "", valid + " "):
            with self.subTest(path=invalid), self.assertRaises(ValidationError):
                runner.ownership_command(invalid)

    def test_root_or_unknown_device_uid_is_rejected(self):
        self.assertEqual(runner.validate_device_uid("2000\n"), 2000)
        for invalid in ("0", "0000\n", "", "uid=2000", "2000\n0", "-1"):
            with self.subTest(uid=invalid), self.assertRaises(ValidationError):
                runner.validate_device_uid(invalid)

    def test_implicit_or_malformed_serial_is_rejected(self):
        self.assertEqual(runner.explicit_serial("192.0.2.1:5555"), "192.0.2.1:5555")
        for invalid in ("", " ", "-d", "one two", "one\ntwo"):
            with self.subTest(serial=invalid), self.assertRaises(argparse.ArgumentTypeError):
                runner.explicit_serial(invalid)


if __name__ == "__main__":
    unittest.main()
