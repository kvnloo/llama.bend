"""Keep benchmark arrival observations tied to retained stdout, not discarded bytes."""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from campaign_runner import execute


class OutputBudgetIntegrityTests(unittest.TestCase):
    def test_discarded_decoded_text_must_not_claim_arrival(self):
        with tempfile.TemporaryDirectory() as td:
            spec = {
                'argv': [sys.executable, '-c',
                         "import sys; print('text: synthetic not a model', flush=True)"],
                'cwd': td,
            }
            budget = {'timeout_s': 3, 'memory_mb': 1024, 'max_output_mb': 0}
            receipt = execute(spec, budget, Path(td) / 'trial')
            self.assertEqual(receipt['stop_reason'], 'output_budget')
            self.assertEqual(Path(receipt['stdout']).read_bytes(), b'')
            self.assertIsNone(receipt['process_observed_decoded_text_upper_bound_ms'])
            self.assertIsNone(receipt['process_observed_ttft_upper_bound_ms'])

    def test_discarded_generated_token_must_not_claim_arrival(self):
        with tempfile.TemporaryDirectory() as td:
            spec = {
                'argv': [sys.executable, '-c',
                         "import sys; print('  id 4 logit 1 0.0 t=0.010s', flush=True)"],
                'cwd': td,
            }
            budget = {'timeout_s': 3, 'memory_mb': 1024, 'max_output_mb': 0}
            receipt = execute(spec, budget, Path(td) / 'trial')
            self.assertEqual(receipt['stop_reason'], 'output_budget')
            self.assertEqual(Path(receipt['stdout']).read_bytes(), b'')
            self.assertIsNone(receipt['process_observed_ttft_upper_bound_ms'])


if __name__ == '__main__':
    unittest.main()
