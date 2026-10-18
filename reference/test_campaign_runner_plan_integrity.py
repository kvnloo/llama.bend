"""Fail-closed runner plan contracts: synthetic fixtures, not inference evidence."""
import sys
import unittest
import tempfile
import json
import io
from contextlib import redirect_stdout
from unittest import mock
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from campaign_runner import main, plan_trials, run_manifest, validate_disk_floor

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

class DiskFloorBudgetTests(unittest.TestCase):
    def test_invalid_disk_floors_fail_before_creating_results(self):
        invalid = (-1, -0.01, True, False, '0', None, float('nan'),
                   float('inf'), float('-inf'), 10**400, [], {})
        for value in invalid:
            with self.subTest(floor=repr(value)), tempfile.TemporaryDirectory() as td:
                out = Path(td) / 'results'
                m = VALID | {'budget': {'min_free_mb': value}}
                with self.assertRaisesRegex(ValueError, 'invalid min_free_mb'):
                    run_manifest(m, out)
                self.assertFalse(out.exists())

    def test_missing_or_nonmapping_budget_rejected(self):
        for budget in (None, [], 'no-limit', {}):
            with self.subTest(budget=repr(budget)), tempfile.TemporaryDirectory() as td:
                out = Path(td) / 'results'
                with self.assertRaisesRegex(ValueError, 'invalid min_free_mb'):
                    run_manifest(VALID | {'budget': budget}, out)
                self.assertFalse(out.exists())

    def test_zero_and_positive_disk_floors_remain_valid(self):
        # Zero is permitted by existing synthetic resume fixtures, while
        # production manifests should set a nonzero minimum free disk budget.
        for value in (0, 1, 12288, 1.25):
            with self.subTest(floor=value):
                validate_disk_floor(VALID | {'budget': {'min_free_mb': value}})

    def test_plan_only_does_not_approve_invalid_disk_floor(self):
        # Plan-only is a dry-run; success must not imply an unsafe budget.
        for floor in (-1, False, '0', None, float('nan')):
            with self.subTest(floor=repr(floor)), tempfile.TemporaryDirectory() as td:
                path = Path(td) / 'manifest.json'
                out = Path(td) / 'out'
                path.write_text(json.dumps(VALID | {'budget': {'min_free_mb': floor}}))
                with mock.patch.object(sys, 'argv', ['campaign_runner.py', str(path), '--out', str(out), '--plan-only']):
                    with self.assertRaisesRegex(ValueError, 'invalid min_free_mb'):
                        main()
                self.assertFalse(out.exists())

    def test_plan_only_valid_disk_floor_has_no_output_side_effect(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / 'manifest.json'
            out = Path(td) / 'out'
            path.write_text(json.dumps(VALID | {'budget': {'min_free_mb': 1}}))
            with mock.patch.object(sys, 'argv', ['campaign_runner.py', str(path), '--out', str(out), '--plan-only']):
                with redirect_stdout(io.StringIO()) as stdout:
                    self.assertEqual(main(), 0)
            self.assertEqual(json.loads(stdout.getvalue())['state'], 'planned_not_executed')
            self.assertFalse(out.exists())

if __name__ == '__main__': unittest.main()
