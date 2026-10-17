"""Fail-closed runner plan contracts: synthetic fixtures, not inference evidence."""
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from campaign_runner import plan_trials

VALID = {'schema': 1, 'seed': 7, 'warmups': 0, 'rounds': 2,
         'cases': [{'id': 'c', 'variants': [{'id': 'control', 'argv': ['synthetic']}]}]}

class PlanManifestCompletenessTests(unittest.TestCase):
    def test_empty_manifest_is_not_success_without_trials(self):
        for cases in ([], [{'id':'c', 'variants': []}]):
            with self.subTest(cases=cases), self.assertRaisesRegex(ValueError, 'no configurations'):
                plan_trials(VALID | {'cases': cases})

    def test_noninteger_or_missing_seed_is_not_replayable(self):
        for seed in (None, True, False, 1.5, '7'):
            with self.subTest(seed=seed), self.assertRaisesRegex(ValueError, 'invalid seed'):
                plan_trials(VALID | {'seed': seed})
        missing = dict(VALID); missing.pop('seed')
        with self.assertRaisesRegex(ValueError, 'invalid seed'):
            plan_trials(missing)

    def test_boolean_schema_is_not_v1(self):
        with self.assertRaisesRegex(ValueError, 'unsupported schema'):
            plan_trials(VALID | {'schema': True})

    def test_environment_must_be_a_valid_string_mapping(self):
        invalid = (None, [], {'N': 1}, {1: 'v'}, {'N': True},
                   {'N': None}, {'': 'v'}, {'BAD=KEY': 'v'}, {'N': 'bad\0value'},
                   {'BAD\0KEY': 'value'})
        for env in invalid:
            with self.subTest(env=env):
                cases = [{'id': 'c', 'variants': [
                    {'id': 'control', 'argv': ['synthetic'], 'env': env}]}]
                with self.assertRaisesRegex(ValueError, 'invalid env'):
                    plan_trials(VALID | {'cases': cases})
        cases = [{'id': 'c', 'variants': [
            {'id': 'control', 'argv': ['synthetic'], 'env': {'OMP_NUM_THREADS': '2'}}]}]
        self.assertEqual(len(plan_trials(VALID | {'cases': cases})), 2)

    def test_valid_plan_preserves_stable_trial_keys(self):
        first = plan_trials(VALID)
        self.assertEqual(len(first), 2)
        self.assertEqual(first, plan_trials(VALID))

if __name__ == '__main__': unittest.main()
