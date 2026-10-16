"""Consumer-visible protocol, optimizer and free-provider refusal boundaries."""
import importlib.util
import json
from pathlib import Path
import struct
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parents[1] / 'effects' / 'adreno' / 'autoresearch'
sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location('adreno_autoresearch', HERE / 'controller.py')
research = importlib.util.module_from_spec(spec)
spec.loader.exec_module(research)


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.kernel = research.generate({'family': 'cooperative', 'lanes': 4, 'local_size': 32})
        self.config = dict(samples=3, warmups=2, batch=8)
        self.meta = dict(pattern=4, rows=576, cols=576, device_type=4, gpu_required=True,
                         cpu_fallback=False, profiling=True, global_size=self.kernel['global_size'],
                         local_size=self.kernel['local_size'], **self.config,
                         platform='QUALCOMM', platform_vendor='QUALCOMM', platform_version='OpenCL3.0',
                         vendor='QUALCOMM', device='Adreno830', device_version='OpenCL3.0', driver_version='test')
        self.samples = [dict(index=index, gpu_ns=1000.0 + 100 * index, wall_ns=16000,
                             events=8, completed=True) for index in range(3)]
        self.values = [.125] * 576

    def output(self, values=None):
        values = self.values if values is None else values
        bits = struct.unpack('<' + 'I' * len(values), struct.pack('<' + 'f' * len(values), *values))
        return ('AUTORESEARCH_META ' + json.dumps(self.meta) + '\n'
                + ''.join('AUTORESEARCH_SAMPLE ' + json.dumps(sample) + '\n' for sample in self.samples)
                + 'AUTORESEARCH_BITS pattern=4 values=' + ' '.join(map(str, bits)) + '\n')

    def parse(self, output=None):
        return research.parse_benchmark(self.output() if output is None else output,
                                        self.kernel, 4, self.config, self.values)

    def test_final_row_corruption_is_not_hidden_by_matching_prefix(self):
        corrupt = self.values.copy()
        corrupt[-1] += 1
        with self.assertRaisesRegex(research.ValidationError, 'numerical oracle mismatch'):
            self.parse(self.output(corrupt))

    def test_missing_final_row_rejected(self):
        with self.assertRaisesRegex(research.ValidationError, '576 U32'):
            self.parse(self.output(self.values[:-1]))

    def test_nan_numeric_output_rejected(self):
        corrupt = self.values.copy()
        corrupt[-1] = float('nan')
        with self.assertRaisesRegex(research.ValidationError, 'nonfinite output'):
            self.parse(self.output(corrupt))

    def test_cpu_or_missing_profiling_cannot_substitute(self):
        for key, value in [('device_type', 2), ('cpu_fallback', True), ('profiling', False), ('device', 'other GPU')]:
            with self.subTest(key=key), patch.dict(self.meta, {key: value}):
                with self.assertRaises(research.ValidationError):
                    self.parse()

    def test_launch_geometry_must_match_measured_candidate(self):
        self.meta['global_size'] = 576
        with self.assertRaisesRegex(research.ValidationError, 'global_size'):
            self.parse()

    def test_boolean_cannot_impersonate_integer_pattern(self):
        self.meta['pattern'] = True
        with self.assertRaises(research.ValidationError):
            self.parse()

    def test_duplicate_metadata_rejected(self):
        with self.assertRaisesRegex(research.ValidationError, 'duplicate'):
            self.parse(self.output() + 'AUTORESEARCH_META ' + json.dumps(self.meta))

    def test_each_timed_dispatch_requires_complete_event_sample(self):
        for key, value in [('index', 9), ('events', 7), ('completed', False)]:
            with self.subTest(key=key), patch.dict(self.samples[-1], {key: value}):
                with self.assertRaises(research.ValidationError):
                    self.parse()

    def test_invalid_durations_never_become_fast_winners(self):
        for value in (0, -1, True, float('nan'), float('inf')):
            with self.subTest(value=value), patch.dict(self.samples[-1], {'gpu_ns': value}):
                with self.assertRaises(research.ValidationError):
                    self.parse()

    def test_device_time_must_fit_actual_enclosing_wall_interval(self):
        self.samples[-1]['wall_ns'] = 1
        with self.assertRaisesRegex(research.ValidationError, 'inconsistent'):
            self.parse()

    def test_batch_wall_time_is_amortized_not_mislabeled_gpu_time(self):
        result = self.parse()
        self.assertEqual(result['gpu_median_ns'], 1100)
        self.assertEqual(result['wall_amortized_median_ns'], 2000)
        self.assertEqual(result['gpu_p95_ns'], 1200)

    def test_both_extended_dense_input_ids_supported(self):
        for pattern in (3, 4):
            values = research.words('CPU_GEMV_BITS pattern=' + str(pattern) + ' values=' + ' '.join(['1065353216'] * 576), 'CPU_GEMV_BITS', pattern)
            self.assertEqual(values[-1], 1)
        with self.assertRaises(research.ValidationError):
            research.words('CPU_GEMV_BITS pattern=3 values=' + ' '.join(['1065353216'] * 576), 'CPU_GEMV_BITS', 4)


class SelectionTests(unittest.TestCase):
    def values(self, medians, tails=None):
        tails = medians if tails is None else tails
        return {index: dict(gpu_median_ns=median, gpu_p95_ns=tails[index]) for index, median in enumerate(medians)}

    def test_geometric_not_arithmetic_cross_pattern_gain(self):
        pair = research.assess_pair(self.values([100]*5), self.values([1, 10000, 100, 100, 100]))
        self.assertAlmostEqual(pair['speedup'], 1)

    def test_missing_input_invalidates_comparison(self):
        fewer = self.values([100]*4)
        with self.assertRaises(research.ValidationError):
            research.assess_pair(self.values([100]*5), fewer)

    def test_one_tail_regression_blocks_otherwise_fast_kernel(self):
        pair = research.assess_pair(self.values([100]*5), self.values([50]*5, [50,50,50,50,120]))
        self.assertGreater(pair['speedup'], 1.05)
        self.assertFalse(research.promotion_gate([pair]*3))

    def test_three_confirmations_and_every_pair_minimum_required(self):
        good = dict(speedup=1.2, tail_gate=True)
        self.assertFalse(research.promotion_gate([good]*2))
        self.assertTrue(research.promotion_gate([good]*3))
        self.assertFalse(research.promotion_gate([good,good,dict(speedup=1.01,tail_gate=True)]))
        self.assertFalse(research.promotion_gate([dict(speedup=1.04,tail_gate=True)]*3))

    def test_untrusted_specs_cannot_inject_source_or_unsafe_launches(self):
        for value in ({'family':'vector4','source':'arbitrary code'}, {'family':'cooperative','local_size':0},
                      {'family':'cooperative','lanes':128,'unroll':8}, {'family':'vector4','unroll':True}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                research.generate(value)


class ModelRefusalTests(unittest.TestCase):
    def test_nonretryable_free_tier_403_does_not_launch_another_round(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = root / 'model-round-initial'
            directory.mkdir()
            (directory / 'free.json').write_text(json.dumps(dict(status='incomplete', usage={'error':{'name':'APIError','data':{'statusCode':403,'isRetryable':False}}})))
            experiment = research.Experiment.__new__(research.Experiment)
            experiment.args = SimpleNamespace(models=True)
            experiment.process = None
            experiment.session = root / 'session'
            experiment.session.mkdir()
            experiment.model_seen = set()
            experiment.model_blocked = None
            experiment.queue, experiment.seen = [], set()
            experiment.last_models = 0
            with patch.object(research, 'SCRATCH', root), patch.object(research.subprocess, 'Popen', side_effect=AssertionError('provider refusal must not cause another request')):
                experiment.models()
            self.assertIn('403', experiment.model_blocked)
            self.assertEqual(experiment.queue, [])


class RecoveryTests(unittest.TestCase):
    def test_changed_cached_oracle_requires_its_original_raw_witness(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prepare = root / 'prepare-original'
            prepare.mkdir()
            raw = 'CPU_GEMV_BITS pattern=4 values=' + ' '.join(['1065353216'] * 576)
            command = dict(label='production-oracle-4', acceptance_met=True, exit_code=0,
                           timed_out=False, stdout=raw)
            (prepare / 'receipt.json').write_text(json.dumps(dict(commands=[command])))
            corrupted = [1.0] * 576
            corrupted[-1] = 2
            with self.assertRaisesRegex(research.ValidationError, 'no matching retained'):
                research.require_oracle_witness(root, 4, corrupted)

    def test_failed_or_missing_cpu_command_cannot_authorize_cached_values(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaises(research.ValidationError):
                research.require_oracle_witness(root, 3, [1.0] * 576)
            prepare = root / 'prepare-failed'
            prepare.mkdir()
            raw = 'CPU_GEMV_BITS pattern=3 values=' + ' '.join(['1065353216'] * 576)
            command = dict(label='production-oracle-3', acceptance_met=False, exit_code=33,
                           timed_out=False, stdout=raw)
            (prepare / 'receipt.json').write_text(json.dumps(dict(commands=[command])))
            with self.assertRaises(research.ValidationError):
                research.require_oracle_witness(root, 3, [1.0] * 576)

    def test_wall_clock_rollback_cannot_extend_process_deadline(self):
        with tempfile.TemporaryDirectory() as temporary:
            experiment = research.Experiment.__new__(research.Experiment)
            experiment.session = Path(temporary)
            experiment.end = 100
            experiment.monotonic_end = 10
            with patch.object(research.time, 'time', return_value=0), patch.object(research.time, 'monotonic', return_value=11):
                with self.assertRaises(research.Deadline):
                    experiment.deadline()


if __name__ == '__main__':
    unittest.main()
