"""Fail-closed runner plan contracts: synthetic fixtures, not inference evidence."""
import sys
import unittest
import tempfile
import json
import io
import hashlib
from contextlib import redirect_stdout
from unittest import mock
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from campaign_runner import main, plan_trials, run_manifest

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

    def test_invalid_plan_does_not_create_result_or_lock_files(self):
        # Invalid trial plans must have no filesystem side effects.
        mutations = ({'cases': []},
                     {'cases': [{'id': 'c', 'variants': []}]},
                     {'seed': True}, {'warmups': -1})
        for mutation in mutations:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as td:
                root = Path(td)
                lock = root / 'runner.lock'
                out = root / 'results'
                manifest = VALID | {'lock': str(lock),
                                    'budget': {'min_free_mb': 0}} | mutation
                with self.assertRaises(ValueError):
                    run_manifest(manifest, out)
                self.assertFalse(out.exists())
                self.assertFalse(lock.exists())

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

    def test_disk_floor_boundaries_and_completed_resume(self):
        for floor in (0, 1, 12288, 1.25):
            with self.subTest(floor=floor), tempfile.TemporaryDirectory() as td:
                root = Path(td)
                out = root / 'results'
                m = VALID | {
                    'rounds': 1, 'lock': str(root / 'lock'),
                    'budget': {'timeout_s': 3, 'memory_mb': 256,
                               'max_output_mb': 1, 'min_free_mb': floor},
                    'cases': [{'id': 'fixture', 'variants': [{
                        'id': 'control', 'argv': [sys.executable, '-c', 'pass'],
                        'cwd': str(root), 'validation': {
                            'kind': 'stdout_sha256',
                            'sha256': hashlib.sha256(b'').hexdigest()}}]}]}
                with mock.patch('campaign_runner.shutil.disk_usage') as usage:
                    if floor:
                        usage.return_value.free = int(floor * 1024 * 1024) - 1
                        with mock.patch('campaign_runner.execute') as execute:
                            self.assertEqual(run_manifest(m, out), 2)
                            execute.assert_not_called()
                        self.assertEqual(json.loads((out / 'journal.jsonl').read_text())['reason'],
                                         'free_disk_budget')
                    # Exact equality is allowed, including zero with zero free bytes.
                    usage.return_value.free = int(floor * 1024 * 1024)
                    self.assertEqual(run_manifest(m, out), 0)
                    journal = (out / 'journal.jsonl').read_bytes()
                    usage.return_value.free = 0
                    self.assertEqual(run_manifest(m, out), 0)
                    self.assertEqual((out / 'journal.jsonl').read_bytes(), journal)
                    m['budget']['min_free_mb'] = False
                    with self.assertRaisesRegex(ValueError, 'invalid min_free_mb'):
                        run_manifest(m, out)
                    self.assertEqual((out / 'journal.jsonl').read_bytes(), journal)

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
            # A valid plan obeys the runner's absolute shared-lock/cwd
            # contract; only budget and no-side-effect behavior are under test.
            valid = VALID | {
                'budget': {'min_free_mb': 1},
                'lock': str(Path(td) / 'runner.lock'),
                'cases': [{'id': 'c', 'variants': [
                    {**VALID['cases'][0]['variants'][0], 'cwd': td}]}]}
            path.write_text(json.dumps(valid))
            with mock.patch.object(sys, 'argv', ['campaign_runner.py', str(path), '--out', str(out), '--plan-only']):
                with redirect_stdout(io.StringIO()) as stdout:
                    self.assertEqual(main(), 0)
            self.assertEqual(json.loads(stdout.getvalue())['state'], 'planned_not_executed')
            self.assertFalse(out.exists())

if __name__ == '__main__': unittest.main()
