"""Offline adversarial tests; subprocess fixtures are not inference evidence."""
import hashlib
import json
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from campaign_runner import validate_generation, plan_trials, run_manifest, canonical_hash

RAW = 'weights loaded in 0.002s\nprompt ids: 1 2 \n  id 4  logit 1056964608 0.5  t=0.003s\n  id 5  logit 1058642330 0.6  t=0.004s\ntext: ok\n'
EXPECTED = {'prompt_ids': [1, 2], 'generated_ids': [4, 5], 'text': 'ok', 'logit_bits': [1056964608, 1058642330]}

class ValidationTests(unittest.TestCase):
    def test_missing_text_oracle_is_value_error(self):
        invalid = {k: v for k, v in EXPECTED.items() if k != 'text'}
        with self.assertRaisesRegex(ValueError, 'invalid generated-text oracle'):
            validate_generation(RAW, invalid)

    def test_nonstring_text_oracle_is_value_error(self):
        for value in (None, 14, ['ok'], {'text': 'ok'}, True):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, 'invalid generated-text oracle'):
                    validate_generation(RAW, EXPECTED | {'text': value})

    def test_complete(self): self.assertEqual(validate_generation(RAW, EXPECTED)['tokens'], 2)
    def test_strict_latency_parser_receives_complete_capture(self):
        # Clean PR #2 requires the final native text footer. The runner
        # validates that footer against the oracle before invoking the parser.
        from latency_report import report as existing_report
        def require_complete_stdout(capture):
            if '\ntext: ' not in capture:
                raise ValueError('incomplete generated output: missing decoded-text footer')
            return existing_report(capture)
        with patch('campaign_runner.report', side_effect=require_complete_stdout):
            self.assertEqual(validate_generation(RAW, EXPECTED)['tokens'], 2)

    def test_zero_quantized_interval(self): self.assertEqual(validate_generation(RAW.replace('0.004s', '0.003s'), EXPECTED)['decode_mean_ms'], 0)
    def test_missing_tail(self):
        with self.assertRaises(ValueError): validate_generation(RAW.split('text:')[0], EXPECTED)
    def test_wrong_ids(self):
        with self.assertRaises(ValueError): validate_generation(RAW.replace('id 5 ', 'id 6 '), EXPECTED)
    def test_wrong_prompt(self):
        with self.assertRaises(ValueError): validate_generation(RAW.replace('ids: 1 2', 'ids: 1 3'), EXPECTED)
    def test_malformed_token_rejected(self):
        with self.assertRaises(ValueError): validate_generation(RAW.replace('t=0.004s', 't=nans'), EXPECTED)
    def test_duplicate_headers(self):
        with self.assertRaises(ValueError): validate_generation('weights loaded in 0.1s\n'+RAW, EXPECTED)
    def test_infinite_float_rejected(self):
        with self.assertRaises(ValueError): validate_generation(RAW.replace('0.5  t=', 'inf  t='), EXPECTED)
    def test_trailing_error(self):
        with self.assertRaises(ValueError): validate_generation(RAW+'ERROR: incomplete\n', EXPECTED)
    def test_logit_bit_check(self):
        with self.assertRaises(ValueError): validate_generation(RAW, EXPECTED | {'logit_bits': [123, 999]})

    def test_token_looking_decoded_text(self):
        text = 'ok\n  id 5 as text\ntext: quoted\r\u2028'
        self.assertEqual(validate_generation(RAW.replace('text: ok\n', 'text: '+text+'\n'), EXPECTED | {'text': text})['tokens'], 2)
    def test_overflow_time_rejected(self):
        with self.assertRaises(ValueError): validate_generation(RAW.replace('t=0.004s', 't='+'9'*400+'s'), EXPECTED)
    def test_u32_overflow_rejected(self):
        with self.assertRaises(ValueError): validate_generation(RAW.replace('logit 1056964608', 'logit 4294967296'), EXPECTED)
    def test_tokens_after_eos_rejected(self):
        with self.assertRaises(ValueError): validate_generation(RAW.replace('id 4 ', 'id 0 '), EXPECTED | {'generated_ids': [0,5]})

    def test_unknown_output_rejected(self):
        with self.assertRaises(ValueError): validate_generation('ERROR: detected failure\n'+RAW, EXPECTED)
    def test_overrun_despite_eos(self):
        with self.assertRaises(ValueError): validate_generation(RAW.replace('id 5 ', 'id 0 '), EXPECTED | {'generated_ids': [4,0], 'requested_tokens': 1})
    def test_per_token_relative_numerical_gate(self):
        expected = {k:v for k,v in EXPECTED.items() if k != 'logit_bits'}
        with self.assertRaises(ValueError): validate_generation(RAW, expected | {'logits': [0.5,0.1], 'logit_tolerance': 2e-4})

    def test_stage_output(self):
        raw = 'model_load_ms=2\nprompt ids: 1 2\ntokenize_ms=1\n  id 4  logit 1056964608 0.5\nprefill_ms=3\nfirst_forward_ms=2\nresident_first_token_id_ms=5\n  id 5  logit 1058642330 0.6\ntext: ok\n'
        self.assertEqual(validate_generation(raw, EXPECTED, stages=True)['resident_first_token_id_ms'],5)
        with self.assertRaises(ValueError): validate_generation(raw.replace('prefill_ms=3','prefill_ms=4'),EXPECTED,stages=True)


    def test_staged_prompt_rejects_non_native_id_glyphs_and_spacing(self):
        # Nat.show emits ASCII decimals with single spaces; Python int()
        # otherwise accepts Unicode numerals, tabs and doubled spaces.
        raw = ('model_load_ms=2\nprompt ids: 1 2\ntokenize_ms=1\n'
               '  id 4  logit 1056964608 0.5\nprefill_ms=3\n'
               'first_forward_ms=2\nresident_first_token_id_ms=5\n'
               '  id 5  logit 1058642330 0.6\ntext: ok\n')
        for native in ('prompt ids: 1 2', 'prompt ids: 1 2 '):
            with self.subTest(native=native):
                self.assertEqual(validate_generation(
                    raw.replace('prompt ids: 1 2', native),
                    EXPECTED, stages=True)['tokens'], 2)
        for bad in ('prompt ids: ١ ٢', 'prompt ids: 1\t2',
                    'prompt ids: 1  2', 'prompt ids: 1\u00a02'):
            with self.subTest(bad=bad):
                with self.assertRaisesRegex(ValueError, 'malformed prompt IDs'):
                    validate_generation(raw.replace('prompt ids: 1 2', bad),
                                        EXPECTED, stages=True)


class RunnerTests(unittest.TestCase):
    def manifest(self, root, command=None):
        return {'schema': 1, 'seed': 7, 'warmups': 1, 'rounds': 2, 'label': 'test-fixture-not-inference', 'lock': str(root/'lock'), 'budget': {'timeout_s': 5, 'memory_mb': 256, 'min_free_mb': 1, 'max_output_mb': 1}, 'cases': [{'id': 'case', 'variants': [{'id': 'control', 'argv': command or [sys.executable, '-c', 'print('+repr(RAW)+', end="")'], 'cwd': str(root), 'validation': {'kind': 'generation', 'expected': EXPECTED}}]}]}
    def test_dedup(self):
        m=self.manifest(Path('/tmp')); m['cases']*=2
        self.assertEqual(len(plan_trials(m)), 3)
    def test_rounds_interleaved(self):
        m=self.manifest(Path('/tmp')); m['cases'][0]['variants'].append(m['cases'][0]['variants'][0] | {'id': 'candidate'})
        p=plan_trials(m); self.assertEqual([x['phase'] for x in p], ['warmup']*2+['sample']*4)
        self.assertEqual([x['round'] for x in p if x['phase']=='sample'], [0,0,1,1])
    def test_resume_and_tamper(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); m=self.manifest(root); out=root/'out'
            self.assertEqual(run_manifest(m,out),0)
            n=len((out/'journal.jsonl').read_text().splitlines())
            self.assertEqual(run_manifest(m,out),0)
            self.assertEqual(len((out/'journal.jsonl').read_text().splitlines()),n)
            raw=next((out/'raw').glob('*.stdout')); raw.write_text('tampered')
            with self.assertRaises(ValueError): run_manifest(m,out)
    def test_invalid_text_oracle_is_finished_failure_not_abandoned_attempt(self):
        # Malformed frozen text oracle must not leave an abandoned started entry.
        for bad in ('missing', None, 14, ['ok'], {'text':'ok'}, True):
            with self.subTest(bad=bad), tempfile.TemporaryDirectory() as td:
                root=Path(td); m=self.manifest(root); m['warmups']=0; m['rounds']=1
                oracle=m['cases'][0]['variants'][0]['validation']['expected'].copy()
                if bad == 'missing': oracle.pop('text')
                else: oracle['text']=bad
                m['cases'][0]['variants'][0]['validation']['expected']=oracle
                out=root/'out'
                self.assertEqual(run_manifest(m,out),1)
                events=[json.loads(s) for s in (out/'journal.jsonl').read_text().splitlines()]
                self.assertEqual([e['event'] for e in events], ['started','finished'])
                self.assertEqual(events[-1]['status'], 'failed')
                self.assertIn('invalid generated-text oracle', events[-1]['error'])
                self.assertNotIn('latency', events[-1])
    def test_nonzero_is_failure(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); m=self.manifest(root,[sys.executable,'-c','print('+repr(RAW)+',end="");raise SystemExit(7)']); m['warmups']=0;m['rounds']=1
            self.assertEqual(run_manifest(m,root/'out'),1)
            done=[json.loads(s) for s in (root/'out/journal.jsonl').read_text().splitlines() if json.loads(s)['event']=='finished']
            self.assertEqual(done[0]['status'],'failed')
            self.assertNotIn('latency',done[0])
    def test_changed_manifest_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); m=self.manifest(root); run_manifest(m,root/'out'); m['seed']=8
            with self.assertRaises(ValueError): run_manifest(m,root/'out')
    def test_timeout(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);m=self.manifest(root,[sys.executable,'-c','import time;time.sleep(10)']);m['budget']['timeout_s']=0.05;m['warmups']=0;m['rounds']=1
            self.assertEqual(run_manifest(m,root/'out'),1)

if __name__=='__main__': unittest.main()
