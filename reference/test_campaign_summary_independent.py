"""Synthetic summary acceptance tests; never execute a benchmark/model process."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
from campaign_runner import canonical_hash, file_hash, plan_trials
from campaign_summary import summarize


def fixture(root, rounds=5, purpose="exploratory", times=None, warmups=1):
    out=Path(root);stdout=out/"stdout";stderr=out/"stderr"
    stdout.write_bytes(b"fixture\n");stderr.write_bytes(b"")
    sha=file_hash(stdout)
    m={"schema":1,"seed":7,"rounds":rounds,"warmups":warmups,"label":"synthetic-not-inference",
       "purpose":purpose,"primary_metric":"wall_ms","selection_manifest_sha256":canonical_hash({"mock_selection":True}),
       "cases":[{"id":"fixture","variants":[{"id":name,"argv":["not-executed"],"cwd":str(out),
          "validation":{"kind":"stdout_sha256","sha256":sha}} for name in ("control","candidate")]}]}
    (out/"manifest.json").write_text(json.dumps(m))
    rows=[]
    for t in plan_trials(m):
        candidate=t["variant"]=="candidate"
        wall=(times[t["round"]][candidate] if times is not None and t["phase"]=="sample" else (80. if candidate else 100.))
        rows.append({**{k:t[k] for k in ("key","case","variant","phase","round")},"manifest_sha256":canonical_hash(m),
            "event":"finished","status":"passed","returncode":0,"stop_reason":None,"latency":{},
            "wall_ms":wall,"process_observed_ttft_upper_bound_ms":wall,"process_observed_decoded_text_upper_bound_ms":wall,
            "peak_rss_kib":1000 if candidate else 2000,"stdout":str(stdout),"stderr":str(stderr),
            "stdout_sha256":sha,"stderr_sha256":file_hash(stderr)})
    save(out,rows)
    return m,rows


def save(out,rows):
    (out/"journal.jsonl").write_text("".join(json.dumps(row)+"\n" for row in rows))


def pair(result): return result["cases"]["fixture"]["paired"]["candidate"]


class SummaryAuditTests(unittest.TestCase):
    def test_screen_never_promotes_even_with_five_pairs(self):
        with tempfile.TemporaryDirectory() as td:
            fixture(Path(td));p=pair(summarize(td))
            self.assertTrue(p["exploratory"])
            self.assertFalse(any(s["meets_primary_5pct_gate"] for s in p["metrics"].values()))

    def test_fresh_confirmatory_primary_only_and_warmups_excluded(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);_,rows=fixture(root,purpose="confirmatory")
            for row in rows:
                if row["phase"]=="warmup":
                    row["wall_ms"]=1e9 if row["variant"]=="candidate" else 1.
                    row["process_observed_ttft_upper_bound_ms"]=row["wall_ms"]*.5
                    row["process_observed_decoded_text_upper_bound_ms"]=row["wall_ms"]*.8
            save(root,rows);p=pair(summarize(td));s=p["metrics"]["wall_ms"]
            self.assertFalse(p["exploratory"]);self.assertEqual(s["paired_n"],5)
            self.assertEqual(s["median_saving_pct"],20.)
            self.assertTrue(s["meets_primary_5pct_gate"])
            self.assertFalse(p["metrics"]["peak_rss_kib"]["meets_primary_5pct_gate"])

    def test_four_pairs_not_enough(self):
        with tempfile.TemporaryDirectory() as td:
            fixture(Path(td),rounds=4,purpose="confirmatory");p=pair(summarize(td))
            self.assertTrue(p["exploratory"])
            self.assertFalse(p["metrics"]["wall_ms"]["meets_primary_5pct_gate"])

    def test_incomplete_pairs_and_failure_never_become_zero_wins(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);_,rows=fixture(root,purpose="confirmatory")
            missing=next(r for r in rows if r["phase"]=="sample" and r["variant"]=="candidate")
            missing.update(status="failed",error="fixture failure");save(root,rows)
            result=summarize(root);s=pair(result)["metrics"]["wall_ms"]
            self.assertEqual(s["paired_n"],4);self.assertEqual(len(result["failures"]),1)
            self.assertFalse(s["meets_primary_5pct_gate"])

    def test_prespecified_plan_must_complete_before_promotion(self):
        for missing in ("later_sample_rounds", "warmup"):
            with self.subTest(missing=missing),tempfile.TemporaryDirectory() as td:
                root=Path(td)
                _,rows=fixture(root,rounds=10 if missing=="later_sample_rounds" else 5,purpose="confirmatory")
                if missing=="later_sample_rounds":
                    rows=[r for r in rows if r["phase"]=="warmup" or r["round"]<5]
                else:
                    rows=[r for r in rows if r["phase"]!="warmup"]
                save(root,rows);result=summarize(root);s=pair(result)["metrics"]["wall_ms"]
                self.assertEqual(s["paired_n"],5)
                self.assertEqual(result["failures"],[])
                self.assertFalse(result["completion"]["complete"])
                self.assertEqual(len(result["completion"]["missing_trial_keys"]),10 if missing=="later_sample_rounds" else 2)
                self.assertFalse(s["meets_primary_5pct_gate"])

    def test_duplicate_passing_plan_mismatch_and_stderr_rejected(self):
        for mutation in ("duplicate","wrong_round","stderr"):
            with self.subTest(mutation=mutation),tempfile.TemporaryDirectory() as td:
                root=Path(td);_,rows=fixture(root)
                if mutation=="duplicate":rows.append(rows[-1].copy())
                elif mutation=="wrong_round":rows[-1]["round"]+=1
                else:
                    (root/"stderr").write_text("ERROR\n")
                    for r in rows:r["stderr_sha256"]=file_hash(root/"stderr")
                save(root,rows)
                with self.assertRaises(ValueError):summarize(root)

    def test_savings_are_paired_not_difference_of_marginal_medians(self):
        with tempfile.TemporaryDirectory() as td:
            fixture(Path(td),times=[(100.,90.),(200.,100.),(300.,210.),(400.,200.),(500.,450.)])
            p=pair(summarize(td));s=p["metrics"]["wall_ms"]
            self.assertEqual(s["median_saving_pct"],30.)
            self.assertEqual(s["min_saving_pct"],10.)
            self.assertEqual(s["max_saving_pct"],50.)

    def test_regression_never_promotes(self):
        with tempfile.TemporaryDirectory() as td:
            fixture(Path(td),times=[(100.,110.)]*5,purpose="confirmatory")
            s=pair(summarize(td))["metrics"]["wall_ms"]
            self.assertEqual(s["median_saving_pct"],-10.)
            self.assertFalse(s["meets_primary_5pct_gate"])


if __name__ == "__main__": unittest.main()
