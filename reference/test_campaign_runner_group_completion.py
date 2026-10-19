"""Native Linux PR #4 regression: a reaped leader must not leave work running."""
import os
import signal
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from campaign_runner import execute, live_group_descendants

class GroupCompletionTests(unittest.TestCase):
    def test_closed_pipe_background_child_is_reaped_and_trial_not_passed(self):
        code = ("import subprocess,sys;"
                "subprocess.Popen([sys.executable,'-c','import time;time.sleep(8)'],"
                "stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL);"
                "print('completed',flush=True)")
        with tempfile.TemporaryDirectory() as d:
            recorded=[]; original=subprocess.Popen
            def track(*args,**kwargs):
                child=original(*args,**kwargs);recorded.append(child);return child
            try:
                with patch('campaign_runner.subprocess.Popen',side_effect=track):
                    receipt=execute({'argv':[sys.executable,'-c',code], 'cwd':d},
                                    {'timeout_s':2,'memory_mb':2048,'max_output_mb':1},
                                    Path(d)/'raw')
                self.assertEqual(len(recorded),1)
                self.assertEqual(receipt['returncode'],0)
                self.assertEqual(Path(receipt['stdout']).read_bytes(),b'completed\n')
                self.assertEqual(receipt['stop_reason'],'lingering_descendant')
                self.assertEqual(live_group_descendants(recorded[0].pid),[])
            finally:
                for child in recorded:
                    try:os.killpg(child.pid,signal.SIGKILL)
                    except ProcessLookupError:pass
                    if child.returncode is None:
                        try:child.wait(timeout=1)
                        except (ChildProcessError,subprocess.TimeoutExpired):pass

    def test_normal_single_process_is_not_flagged(self):
        with tempfile.TemporaryDirectory() as d:
            receipt=execute({'argv':[sys.executable,'-c','print("ok")'], 'cwd':d},
                            {'timeout_s':2,'memory_mb':2048,'max_output_mb':1},
                            Path(d)/'raw')
            self.assertEqual(receipt['returncode'],0)
            self.assertIsNone(receipt['stop_reason'])
            self.assertEqual(Path(receipt['stdout']).read_bytes(),b'ok\n')

if __name__ == '__main__':unittest.main()
