"""A failed raw-output setup must never leave a child process running."""
import os
import signal
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from campaign_runner import execute


class SpawnCleanupTests(unittest.TestCase):
    def test_existing_raw_receipt_path_reaps_child(self):
        # Assert Popen was reached; the regression was previously unprotected
        # file creation after the process had already been launched.
        real_popen = subprocess.Popen
        for collision in ('stdout', 'stderr'):
            with self.subTest(collision=collision), tempfile.TemporaryDirectory() as d:
                root = Path(d)
                base = root / 'receipt'
                existing = base.with_suffix('.' + collision)
                existing.write_bytes(b'preserve original receipt\n')
                children = []
                def record(*args, **kwargs):
                    p = real_popen(*args, **kwargs)
                    children.append(p)
                    return p
                spec = {'argv': [sys.executable, '-c', 'import time;time.sleep(20)'],
                        'cwd': d}
                budget = {'timeout_s': 3, 'memory_mb': 1024, 'max_output_mb': 1}
                try:
                    with mock.patch('campaign_runner.subprocess.Popen', side_effect=record):
                        with self.assertRaises(FileExistsError):
                            execute(spec, budget, base)
                    self.assertEqual(len(children), 1)
                    self.assertIsNotNone(children[0].returncode,
                                         'spawned child not waited/reaped after setup exception')
                    self.assertTrue(children[0].stdout.closed)
                    self.assertTrue(children[0].stderr.closed)
                    with self.assertRaises(ProcessLookupError):
                        os.kill(children[0].pid, 0)
                    self.assertEqual(existing.read_bytes(), b'preserve original receipt\n')
                finally:
                    # Also protect the pre-fix negative demonstration from leaks.
                    for p in children:
                        if p.returncode is None:
                            try: os.killpg(p.pid, signal.SIGKILL)
                            except ProcessLookupError: pass
                            try: p.wait(timeout=2)
                            except (ChildProcessError, subprocess.TimeoutExpired): pass

    def test_selector_creation_failure_reaps_child_and_closes_pipes(self):
        real_popen = subprocess.Popen
        with tempfile.TemporaryDirectory() as d:
            base = Path(d) / 'receipt'
            children = []
            def record(*args, **kwargs):
                p = real_popen(*args, **kwargs)
                children.append(p)
                return p
            spec = {'argv': [sys.executable, '-c', 'import time;time.sleep(20)'], 'cwd': d}
            budget = {'timeout_s': 3, 'memory_mb': 1024, 'max_output_mb': 1}
            try:
                with mock.patch('campaign_runner.subprocess.Popen', side_effect=record),                      mock.patch('campaign_runner.selectors.DefaultSelector', side_effect=OSError('injected selector failure')):
                    with self.assertRaisesRegex(OSError, 'selector failure'):
                        execute(spec, budget, base)
                self.assertEqual(len(children), 1)
                self.assertIsNotNone(children[0].returncode, 'child must be reaped')
                self.assertTrue(children[0].stdout.closed)
                self.assertTrue(children[0].stderr.closed)
            finally:
                for p in children:
                    if p.returncode is None:
                        try: os.killpg(p.pid, signal.SIGKILL)
                        except ProcessLookupError: pass
                        p.wait(timeout=2)

    def test_fsync_failure_closes_both_raw_receipts(self):
        # Retain file objects so GC cannot conceal an unclosed handle.
        real_open = Path.open
        with tempfile.TemporaryDirectory() as d:
            base = Path(d) / 'receipt'
            tracked = []
            targets = {base.with_suffix('.stdout'), base.with_suffix('.stderr')}
            def track_open(self, *args, **kwargs):
                opened = real_open(self, *args, **kwargs)
                if self in targets: tracked.append(opened)
                return opened
            spec = {'argv': [sys.executable, '-c', 'print("ok")'], 'cwd': d}
            budget = {'timeout_s': 3, 'memory_mb': 1024, 'max_output_mb': 1}
            try:
                with mock.patch('campaign_runner.Path.open', track_open),                      mock.patch('campaign_runner.os.fsync', side_effect=OSError('injected fsync failure')):
                    with self.assertRaisesRegex(OSError, 'fsync failure'):
                        execute(spec, budget, base)
                self.assertEqual(len(tracked), 2)
                self.assertTrue(all(f.closed for f in tracked), 'all handles must close even if first fsync fails')
            finally:
                for f in tracked:
                    if not f.closed: f.close()

    def test_selector_registration_error_reaps_and_closes(self):
        real_popen = subprocess.Popen
        with tempfile.TemporaryDirectory() as d:
            children = []
            def record(*args, **kwargs):
                p = real_popen(*args, **kwargs)
                children.append(p)
                return p
            class FailRegistration:
                def register(self, *args): raise OSError('injected registration failure')
                def close(self): self.closed = True
            selected = []
            def selector():
                s = FailRegistration(); selected.append(s); return s
            spec = {'argv':[sys.executable,'-c','import time;time.sleep(20)'], 'cwd':d}
            budget = {'timeout_s': 3, 'memory_mb': 1024, 'max_output_mb': 1}
            try:
                with mock.patch('campaign_runner.subprocess.Popen', side_effect=record),                      mock.patch('campaign_runner.selectors.DefaultSelector', side_effect=selector):
                    with self.assertRaisesRegex(OSError, 'registration failure'):
                        execute(spec,budget,Path(d)/'receipt')
                self.assertEqual(len(children),1)
                self.assertIsNotNone(children[0].returncode)
                self.assertTrue(children[0].stdout.closed)
                self.assertTrue(children[0].stderr.closed)
                self.assertTrue(selected[0].closed)
            finally:
                for p in children:
                    if p.returncode is None:
                        try: os.killpg(p.pid,signal.SIGKILL)
                        except ProcessLookupError: pass
                        p.wait(timeout=2)

    def test_normal_success_retains_clean_receipt(self):
        with tempfile.TemporaryDirectory() as d:
            spec={'argv':[sys.executable,'-c','print("ok")'], 'cwd':d}
            budget={'timeout_s':3, 'memory_mb':1024, 'max_output_mb':1}
            receipt=execute(spec,budget,Path(d)/'receipt')
            self.assertEqual(receipt['returncode'],0)
            self.assertIsNone(receipt['stop_reason'])
            self.assertEqual(Path(receipt['stdout']).read_bytes(),b'ok\n')
            self.assertEqual(Path(receipt['stderr']).read_bytes(),b'')

if __name__ == '__main__':
    unittest.main()
