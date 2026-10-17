"""Offline summary tests; no inference or performance evidence."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from campaign_runner import run_manifest, canonical_hash
from campaign_summary import summarize
from test_campaign_runner import RAW, EXPECTED

class SummaryTests(unittest.TestCase):
    def fixture(self,root):
        variant={'id':'baseline','argv':[sys.executable,'-c','print('+repr(RAW)+',end="")'],'cwd':str(root),'validation':{'kind':'generation','expected':EXPECTED}}
        m={'schema':1,'seed':7,'warmups':1,'rounds':1,'label':'unit-fixture','lock':str(root/'lock'),'budget':{'timeout_s':2,'memory_mb':256,'min_free_mb':1,'max_output_mb':1},'cases':[{'id':'case','variants':[variant,variant|{'id':'candidate'}]}]}
        run_manifest(m,root/'out');return m,root/'out'
    def test_warmups_excluded_and_exploratory(self):
        with tempfile.TemporaryDirectory() as td:
            m,out=self.fixture(Path(td));r=summarize(out)['cases']['case']
            self.assertEqual(r['variants']['baseline']['samples'],1)
            self.assertTrue(r['paired']['candidate']['exploratory'])
            self.assertFalse(r['paired']['candidate']['metrics']['load_s']['meets_primary_5pct_gate'])
    def test_duplicate_pass_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            m,out=self.fixture(Path(td));p=out/'journal.jsonl';lines=p.read_text().splitlines();p.write_text('\n'.join(lines+[lines[-1]])+'\n')
            with self.assertRaises(ValueError):summarize(out)
    def test_changed_plan_identity_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            m,out=self.fixture(Path(td));p=out/'journal.jsonl';entries=[json.loads(x) for x in p.read_text().splitlines()];entries[-1]['round']=42;p.write_text(''.join(json.dumps(x)+'\n' for x in entries))
            with self.assertRaises(ValueError):summarize(out)

if __name__=='__main__':unittest.main()
