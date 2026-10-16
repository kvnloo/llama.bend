"""Offline PR #4 replay/resume authenticity regressions; no executable launched."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from campaign_runner import canonical_hash, file_hash, plan_trials, run_manifest


def resumed_fixture(root, *, code=0, reason=None):
    root = Path(root)
    out = root / 'results'; out.mkdir()
    stdout = root / 'stdout'; stdout.write_bytes(b'pass\n')
    stderr = root / 'stderr'; stderr.write_bytes(b'')
    digest = file_hash(stdout)
    manifest = {
        'schema': 1, 'seed': 1, 'warmups': 0, 'rounds': 1,
        'label': 'synthetic-resume-only', 'lock': str(root / 'shared.lock'),
        'budget': {'min_free_mb': 0},
        'cases': [{'id': 'case', 'variants': [
            {'id': 'control', 'argv': ['must-not-run'], 'cwd': str(root),
             'validation': {'kind': 'stdout_sha256', 'sha256': digest}}
        ]}]
    }
    trial = plan_trials(manifest)[0]
    event = {
        'manifest_sha256': canonical_hash(manifest), 'event': 'finished',
        'status': 'passed', 'key': trial['key'],
        'case': trial['case'], 'variant': trial['variant'],
        'phase': trial['phase'], 'round': trial['round'], 'returncode': code,
        'stop_reason': reason, 'stdout': str(stdout), 'stderr': str(stderr),
        'stdout_sha256': digest, 'stderr_sha256': file_hash(stderr)
    }
    (out/'journal.jsonl').write_text(json.dumps(event)+'\n')
    return manifest, out


class ResumeIntegrityTests(unittest.TestCase):
    def test_valid_passing_trial_is_resumed_without_execution(self):
        with tempfile.TemporaryDirectory() as d:
            manifest, out = resumed_fixture(d)
            with patch('campaign_runner.execute', side_effect=AssertionError('must not rerun'), create=True):
                self.assertEqual(run_manifest(manifest, out), 0)

    def test_invalid_falsy_exit_is_not_treated_as_verified(self):
        for bad in (False, 0.0):
            with self.subTest(code=repr(bad)), tempfile.TemporaryDirectory() as d:
                manifest, out = resumed_fixture(d, code=bad)
                with patch('campaign_runner.execute', side_effect=AssertionError('must not rerun'), create=True):
                    with self.assertRaisesRegex(ValueError, 'successful exit evidence'):
                        run_manifest(manifest, out)

    def test_empty_stop_reason_is_not_verified_completion(self):
        with tempfile.TemporaryDirectory() as d:
            manifest, out = resumed_fixture(d, reason='')
            with patch('campaign_runner.execute', side_effect=AssertionError('must not rerun'), create=True):
                with self.assertRaisesRegex(ValueError, 'successful exit evidence'):
                    run_manifest(manifest, out)

    def test_duplicate_passing_receipt_is_not_silently_resumed(self):
        with tempfile.TemporaryDirectory() as d:
            manifest, out = resumed_fixture(d)
            journal = out / 'journal.jsonl'
            row = journal.read_text()
            journal.write_text(row + row)
            with patch('campaign_runner.execute', side_effect=AssertionError('must not rerun'), create=True):
                with self.assertRaisesRegex(ValueError, 'duplicate passing trial'):
                    run_manifest(manifest, out)

    def test_late_failed_receipt_does_not_allow_skipped_trial(self):
        with tempfile.TemporaryDirectory() as d:
            manifest, out = resumed_fixture(d)
            journal = out / 'journal.jsonl'
            previous = json.loads(journal.read_text())
            late = dict(previous, status='failed', error='timeout', returncode=-15, stop_reason='timeout')
            with journal.open('a') as f:
                f.write(json.dumps(late) + '\n')
            with patch('campaign_runner.execute', side_effect=AssertionError('must not rerun'), create=True):
                with self.assertRaisesRegex(ValueError, 'failed attempt after passing trial'):
                    run_manifest(manifest, out)

    def test_trial_identity_mismatch_is_not_resumed(self):
        with tempfile.TemporaryDirectory() as d:
            manifest, out = resumed_fixture(d)
            journal = out / 'journal.jsonl'
            event = json.loads(journal.read_text()); event['round'] = 999
            journal.write_text(json.dumps(event) + '\n')
            with patch('campaign_runner.execute', side_effect=AssertionError('must not rerun'), create=True):
                with self.assertRaisesRegex(ValueError, 'trial identity mismatch'):
                    run_manifest(manifest, out)

    def test_failure_before_valid_retry_remains_resumable(self):
        with tempfile.TemporaryDirectory() as d:
            manifest, out = resumed_fixture(d)
            journal = out / 'journal.jsonl'
            good = json.loads(journal.read_text())
            failed = dict(good, status='failed', error='previous timeout', returncode=-15, stop_reason='timeout')
            journal.write_text(json.dumps(failed) + '\n' + json.dumps(good) + '\n')
            with patch('campaign_runner.execute', side_effect=AssertionError('must not rerun'), create=True):
                self.assertEqual(run_manifest(manifest, out), 0)

    def test_live_execution_rejects_unverified_falsy_exit_receipts(self):
        # An execute() result is untrusted until the exit type and stop reason
        # are checked. JSON False and 0.0 must not become passing receipts.
        for code, reason, should_pass in ((0, None, True), (False, None, False),
                                          (0.0, None, False), (0, '', False)):
            with self.subTest(code=repr(code), reason=repr(reason)), tempfile.TemporaryDirectory() as d:
                manifest, out = resumed_fixture(d)
                (out / 'journal.jsonl').unlink()  # Force new execution rather than resume.
                raw = Path(d) / 'stdout'; err = Path(d) / 'stderr'
                receipt = {'returncode': code, 'stop_reason': reason,
                           'stdout': str(raw), 'stderr': str(err),
                           'stdout_sha256': file_hash(raw), 'stderr_sha256': file_hash(err)}
                with patch('campaign_runner.execute', return_value=receipt) as mocked:
                    self.assertEqual(run_manifest(manifest, out), 0 if should_pass else 1)
                    mocked.assert_called_once()
                finished = [json.loads(line) for line in (out / 'journal.jsonl').read_text().splitlines()
                            if json.loads(line)['event'] == 'finished']
                self.assertEqual(len(finished), 1)
                self.assertEqual(finished[0]['status'], 'passed' if should_pass else 'failed')

if __name__ == '__main__':
    unittest.main()
