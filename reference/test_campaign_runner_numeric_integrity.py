"""New regression tests for boolean masquerading as integer/float metrics."""
import sys, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from campaign_runner import validate_generation, plan_trials
RAW = ('weights loaded in 0.002s\nprompt ids: 1 2 \n'
       '  id 4  logit 1056964608 0.5  t=0.003s\n'
       '  id 5  logit 1058642330 0.6  t=0.004s\ntext: ok\n')
BASE = {'prompt_ids':[1,2], 'generated_ids':[4,5], 'text':'ok'}
class NumericIntegrity(unittest.TestCase):
    def test_boolean_numeric_tolerance_does_not_waive_logit_parity(self):
        # Under Python True == 1, a 50% logit mismatch passes the bad oracle.
        with self.assertRaisesRegex(ValueError,'invalid numerical tolerance'):
            validate_generation(RAW, BASE|{'logits':[0.4,0.4],'logit_tolerance':True})
    def test_boolean_requested_tokens_is_not_valid_oracle(self):
        one = RAW.replace('  id 5  logit 1058642330 0.6  t=0.004s\n', '')
        with self.assertRaisesRegex(ValueError,'generated count outside requested'):
            validate_generation(one,BASE|{'generated_ids':[4],'logit_bits':[1056964608], 'requested_tokens':True})
    def test_boolean_frozen_ids_are_not_natural_numbers(self):
        for field, ids in [('prompt_ids', [True, 2]),
                           ('generated_ids', [True, 5])]:
            with self.subTest(field=field):
                raw = RAW if field == 'prompt_ids' else RAW.replace('id 4 ', 'id 1 ')
                expected = BASE | {'logit_bits': [1056964608, 1058642330], field: ids}
                with self.assertRaisesRegex(ValueError, f'invalid {field} oracle'):
                    validate_generation(raw, expected)

    def test_boolean_f32_bits_cannot_spoof_a_frozen_oracle(self):
        raw = RAW.replace('logit 1056964608 0.5', 'logit 1 1e-45')
        with self.assertRaisesRegex(ValueError, 'invalid logit bits oracle'):
            validate_generation(raw, BASE | {'logit_bits': [True, 1058642330]})

    def test_malformed_numerical_oracles_fail_closed(self):
        for want in ([True, 0.6], ['0.5', 0.6], [10**400, 0.6], None, {}):
            with self.subTest(oracle=repr(want)[:40]):
                with self.assertRaisesRegex(ValueError, 'invalid logit oracle'):
                    validate_generation(RAW, BASE | {'logits': want, 'logit_tolerance': 1})
        for tolerance in (10**400, None):
            with self.subTest(tolerance=repr(tolerance)[:40]):
                with self.assertRaisesRegex(ValueError, 'invalid numerical tolerance'):
                    validate_generation(RAW, BASE | {'logits': [0.5, 0.6], 'logit_tolerance': tolerance})

    def test_valid_numeric_oracles_and_counts_unchanged(self):
        self.assertEqual(validate_generation(RAW, BASE | {
            'logits':[0.5,0.6], 'logit_tolerance':1e-5,
            'requested_tokens':2,
        })['tokens'], 2)
        m = {'schema':1,'seed':1,'warmups':0,'rounds':1,
             'cases':[{'id':'c','variants':[{'id':'v','argv':['x']}]}]}
        self.assertEqual(len(plan_trials(m)),1)
    def test_boolean_trial_counts_are_not_integer_manifest_contract(self):
        m = {'schema':1,'seed':1,'warmups':0,'rounds':1,'cases':[{'id':'c','variants':[{'id':'v','argv':['x']}]}]}
        for field in ('warmups','rounds'):
            for val in (False,True):
                with self.subTest(field=field,val=val):
                    with self.assertRaisesRegex(ValueError,f'invalid {field}'):
                        plan_trials(m | {field:val})

class DisplaySyntaxIntegrity(unittest.TestCase):
    """Native F32.show emits only ASCII digits, including in stage receipts."""

    def test_foreign_script_f32_glyphs_rejected_in_both_paths(self):
        expected = BASE | {'logit_bits': [1056964608, 1058642330]}
        staged = ('model_load_ms=2\nprompt ids: 1 2\ntokenize_ms=1\n'
                  '  id 4  logit 1056964608 0.5\nprefill_ms=3\n'
                  'first_forward_ms=2\nresident_first_token_id_ms=5\n'
                  '  id 5  logit 1058642330 0.6\ntext: ok\n')
        for shown in ('0.٥', '٠.٥', '+٠.٥', '0.５'):
            for raw, is_staged in ((RAW, False), (staged, True)):
                with self.subTest(shown=shown, staged=is_staged):
                    fake = raw.replace('1056964608 0.5', '1056964608 ' + shown)
                    with self.assertRaisesRegex(ValueError, 'malformed generated-token row'):
                        validate_generation(fake, expected, stages=is_staged)

    def test_ascii_f32_spellings_preserve_valid_tokens(self):
        expected = BASE | {'logit_bits': [1056964608, 1058642330]}
        for shown in ('0.5', '5e-1', '+0.50', '.5'):
            with self.subTest(shown=shown):
                raw = RAW.replace('1056964608 0.5', '1056964608 ' + shown)
                self.assertEqual(validate_generation(raw, expected)['tokens'], 2)


class StagedF32DisplayParity(unittest.TestCase):
    def test_staged_logit_display_must_roundtrip_exact_f32_bits(self):
        # The former isclose(abs_tol=1e-6) accepted inconsistent staged receipts.
        raw = ('model_load_ms=2\nprompt ids: 1 2\ntokenize_ms=1\n'
               '  id 4 logit 0 0\nprefill_ms=3\nfirst_forward_ms=2\n'
               'resident_first_token_id_ms=5\n'
               '  id 5 logit 1056964608 0.5\ntext: ok\n')
        expected = {'prompt_ids': [1, 2], 'generated_ids': [4, 5],
                    'logit_bits': [0, 1056964608], 'text': 'ok'}
        self.assertEqual(validate_generation(raw, expected, stages=True)['tokens'], 2)
        for bits, shown in ((0, '0.0000009'),
                            (2147483648, '0'),
                            (1065353216, '1.0000005')):
            with self.subTest(bits=bits, shown=shown):
                forged = raw.replace('logit 0 0', f'logit {bits} {shown}', 1)
                oracle = expected | {'logit_bits': [bits, 1056964608]}
                with self.assertRaisesRegex(ValueError, 'printed logit and bits disagree'):
                    validate_generation(forged, oracle, stages=True)
        # Preserve correctly displayed signed zero.
        signed = raw.replace('logit 0 0', 'logit 2147483648 -0', 1)
        self.assertEqual(validate_generation(
            signed, expected | {'logit_bits': [2147483648, 1056964608]},
            stages=True)['tokens'], 2)

if __name__=='__main__':unittest.main()
