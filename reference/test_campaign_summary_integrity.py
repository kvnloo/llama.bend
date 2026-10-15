"""Offline evidence-integrity tests. No model, Bend or benchmark execution."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from campaign_runner import canonical_hash, file_hash, plan_trials
from campaign_summary import summarize


def fixture(root, *, missing=None, zero_control=False, include_group=True):
    root = Path(root)
    stdout = root / "stdout"
    stderr = root / "stderr"
    stdout.write_bytes(b"synthetic fixture\n")
    stderr.write_bytes(b"")
    sha = file_hash(stdout)
    m = {
        "schema": 1, "seed": 7, "warmups": 0, "rounds": 6,
        "label": "synthetic-not-inference", "purpose": "confirmatory",
        "primary_metric": "process_observed_ttft_upper_bound_ms",
        "selection_manifest_sha256": canonical_hash({"frozen": True}),
        "cases": [{"id": "case", "variants": [
            {"id": name, "argv": ["not-executed"], "cwd": str(root),
             "validation": {"kind": "stdout_sha256", "sha256": sha}}
            for name in ("control", "candidate")
        ]}],
    }
    (root / "manifest.json").write_text(json.dumps(m))
    rows = []
    for t in plan_trials(m):
        control = t["variant"] == "control"
        ttft = 80.0 if control else 56.0
        if t["round"] == 5:
            if missing == t["variant"]:
                ttft = None
            elif control and zero_control:
                ttft = 0.0
        row = {k: t[k] for k in ("key", "case", "variant", "phase", "round")}
        row.update({
            "event": "finished", "status": "passed", "manifest_sha256": canonical_hash(m),
            "returncode": 0, "stop_reason": None, "latency": {},
            "wall_ms": 100.0, "peak_rss_kib": 2000 if control else 1000,
            "process_observed_ttft_upper_bound_ms": ttft,
            "process_observed_decoded_text_upper_bound_ms": None,
            "stdout": str(stdout), "stderr": str(stderr),
            "stdout_sha256": sha, "stderr_sha256": file_hash(stderr),
        })
        if include_group:
            row["peak_sampled_process_group_rss_kib"] = 15000 if control else 30000
        rows.append(row)
    save(root, rows)
    return rows


def save(root, rows):
    (Path(root) / "journal.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )


def primary(result):
    return result["cases"]["case"]["paired"]["candidate"]["metrics"][
        "process_observed_ttft_upper_bound_ms"
    ]


class SummaryIntegrityTests(unittest.TestCase):
    def test_six_complete_pairs_can_pass(self):
        with tempfile.TemporaryDirectory() as td:
            fixture(td)
            stats = primary(summarize(td))
            self.assertEqual(stats["paired_n"], 6)
            self.assertTrue(stats["meets_primary_5pct_gate"])

    def test_missing_candidate_clock_cannot_pass(self):
        with tempfile.TemporaryDirectory() as td:
            fixture(td, missing="candidate")
            stats = primary(summarize(td))
            self.assertEqual(stats["paired_n"], 5)
            self.assertFalse(stats["meets_primary_5pct_gate"])

    def test_missing_control_clock_cannot_pass(self):
        with tempfile.TemporaryDirectory() as td:
            fixture(td, missing="control")
            stats = primary(summarize(td))
            self.assertEqual(stats["paired_n"], 5)
            self.assertFalse(stats["meets_primary_5pct_gate"])

    def test_zero_control_clock_cannot_pass(self):
        with tempfile.TemporaryDirectory() as td:
            fixture(td, zero_control=True)
            stats = primary(summarize(td))
            self.assertEqual(stats["paired_n"], 5)
            self.assertFalse(stats["meets_primary_5pct_gate"])

    def test_group_rss_is_not_per_child_rss(self):
        with tempfile.TemporaryDirectory() as td:
            fixture(td)
            p = summarize(td)["cases"]["case"]["paired"]["candidate"]["metrics"]
            self.assertEqual(p["peak_rss_kib"]["median_saving_pct"], 50.0)
            self.assertEqual(p["peak_sampled_process_group_rss_kib"]["median_saving_pct"], -100.0)
            self.assertFalse(p["peak_sampled_process_group_rss_kib"]["meets_primary_5pct_gate"])

    def test_failed_retry_kept_and_blocks_gate(self):
        with tempfile.TemporaryDirectory() as td:
            rows = fixture(td)
            bad = rows[0].copy()
            bad.update(status="failed", error="timeout", stop_reason="timeout", returncode=-15)
            save(td, [bad] + rows)
            result = summarize(td)
            self.assertTrue(result["completion"]["complete"])
            self.assertEqual(len(result["failures"]), 1)
            self.assertFalse(primary(result)["meets_primary_5pct_gate"])

    def test_failed_event_after_success_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            rows = fixture(td)
            bad = rows[0].copy()
            bad.update(status="failed", error="late failure", returncode=-15)
            save(td, rows + [bad])
            with self.assertRaises(ValueError):
                summarize(td)

    def test_unknown_validation_kind_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            rows = fixture(td)
            root = Path(td)
            manifest = json.loads((root / "manifest.json").read_text())
            manifest["cases"][0]["variants"][0]["validation"]["kind"] = "unknown"
            (root / "manifest.json").write_text(json.dumps(manifest))
            lookup = {(row["case"], row["variant"], row["round"]): row for row in rows}
            new = []
            for t in plan_trials(manifest):
                row = lookup[(t["case"], t["variant"], t["round"])].copy()
                row["key"] = t["key"]
                row["manifest_sha256"] = canonical_hash(manifest)
                new.append(row)
            save(td, new)
            with self.assertRaisesRegex(ValueError, "unknown validation kind"):
                summarize(td)

    def test_kernel_latency_injection_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            rows = fixture(td)
            rows[0]["latency"] = {"decode_mean_ms": 0.01}
            save(td, rows)
            with self.assertRaisesRegex(ValueError, "unsupported latency metrics"):
                summarize(td)


if __name__ == "__main__":
    unittest.main()
