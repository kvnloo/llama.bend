#!/usr/bin/env python3
"""Correctness-gated, fork-local Adreno GEMV autotuning; never a native ! backend."""
import argparse
from datetime import datetime, timezone
import fcntl
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import re
import shlex
import shutil
import signal
import statistics
import struct
import subprocess
import sys
import tempfile
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SCRATCH = ROOT / '.scratch' / 'autoresearch'
BUILD = SCRATCH / 'build'
sys.path.insert(0, str(HERE.parent))
from validation import compare_values, ValidationError
from kernels import generate, enumerate_specs, SPEC_SCHEMA
_spec = importlib.util.spec_from_file_location('adreno_device_tools', HERE.parent / 'run-device.py')
device = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(device)
identity, atomic_json, Evidence = device.identity, device.atomic_json, device.Evidence
TENSOR_SHA = '0f585f0779626028bdddf5cc74949849d90c838b759a3652220a1cd37d505162'
STOPPING = False


class CandidateUnavailable(RuntimeError):
    pass


class Cooling(RuntimeError):
    pass


class Deadline(RuntimeError):
    pass


def now():
    return datetime.now(timezone.utc).isoformat()


def source_ids(bend):
    paths = [path for path in HERE.iterdir() if path.suffix in ('.c', '.bend', '.js', '.py', '.mjs')]
    paths += [HERE.parent / 'validation.py', HERE.parent / 'run-device.py']
    # Includes unchanged production M.mv, pinned TA and Base, not just our wrapper.
    base = device.source_identities(bend)
    return [identity(path) for path in sorted(paths)] + [item for item in base if item['path'].endswith('.bend') or '/names/' in item['path']]


def checked(evidence, label, command, timeout=60, remote=False):
    return device.require_success(evidence.invoke(label, command, timeout, cwd=ROOT, remote_deadline=remote))


def words(stdout, prefix, pattern):
    lines = [line for line in stdout.splitlines() if line.startswith(prefix)]
    if len(lines) != 1:
        raise ValidationError('missing/duplicate result: ' + prefix)
    match = re.fullmatch(re.escape(prefix) + r' pattern=([0-4]) values=(.*)', lines[0])
    if not match or int(match[1]) != pattern:
        raise ValidationError('incorrect result pattern')
    tokens = match[2].split()
    if len(tokens) != 576 or any(not re.fullmatch(r'[0-9]{1,10}', value) for value in tokens):
        raise ValidationError('result must contain exactly 576 U32 words')
    values = [int(value) for value in tokens]
    if any(value > 0xffffffff for value in values):
        raise ValidationError('result word exceeds U32')
    floats = struct.unpack('<576f', struct.pack('<576I', *values))
    if not all(math.isfinite(value) for value in floats):
        raise ValidationError('nonfinite output')
    return list(floats)


def percentile(values, fraction):
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * fraction) - 1)]


def parse_benchmark(stdout, candidate, pattern, config, oracle):
    def records(prefix):
        return [json.loads(line[len(prefix):]) for line in stdout.splitlines() if line.startswith(prefix)]
    metadata = records('AUTORESEARCH_META ')
    if len(metadata) != 1:
        raise ValidationError('missing/duplicate actual profiling metadata')
    metadata = metadata[0]
    expected = dict(pattern=pattern, rows=576, cols=576, device_type=4, gpu_required=True,
                    cpu_fallback=False, profiling=True, global_size=candidate['global_size'],
                    local_size=candidate['local_size'], **config)
    for key, value in expected.items():
        if key not in metadata or type(metadata[key]) is not type(value) or metadata[key] != value:
            raise ValidationError('unaccepted profiling field: ' + key)
    for key in ('platform', 'platform_vendor', 'platform_version', 'vendor', 'device', 'device_version', 'driver_version'):
        if not isinstance(metadata.get(key), str) or not metadata[key].strip():
            raise ValidationError('missing actual device identity: ' + key)
    if 'adreno' not in metadata['device'].lower() or '830' not in metadata['device']:
        raise ValidationError('wrong GPU; this experiment requires Adreno830')
    samples = records('AUTORESEARCH_SAMPLE ')
    if len(samples) != config['samples']:
        raise ValidationError('missing/extra timing samples')
    for index, sample in enumerate(samples):
        if type(sample.get('index')) is not int or sample['index'] != index or sample.get('completed') is not True:
            raise ValidationError('incomplete/unordered sample')
        if type(sample.get('events')) is not int or sample['events'] != config['batch']:
            raise ValidationError('incorrect event count')
        for key in ('gpu_ns', 'wall_ns'):
            value = sample.get(key)
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValidationError('missing/nonpositive/nonfinite duration')
        if sample['gpu_ns'] * config['batch'] > sample['wall_ns'] * 1.05:
            raise ValidationError('GPU durations inconsistent with enclosing wall interval')
    comparison = compare_values(oracle, words(stdout, 'AUTORESEARCH_BITS', pattern))
    if not comparison['acceptance_met']:
        raise ValidationError('numerical oracle mismatch: ' + json.dumps(comparison))
    gpu = [sample['gpu_ns'] for sample in samples]
    wall = [sample['wall_ns'] / config['batch'] for sample in samples]
    return dict(metadata=metadata, samples=samples, comparison=comparison,
                gpu_median_ns=statistics.median(gpu), gpu_p95_ns=percentile(gpu, .95),
                wall_amortized_median_ns=statistics.median(wall), wall_amortized_p95_ns=percentile(wall, .95))


def assess_pair(control, candidate):
    if set(control) != set(range(5)) or set(candidate) != set(range(5)):
        raise ValidationError('all five independently checked patterns required')
    ratios = [control[p]['gpu_median_ns'] / candidate[p]['gpu_median_ns'] for p in range(5)]
    tails = [candidate[p]['gpu_p95_ns'] / control[p]['gpu_p95_ns'] for p in range(5)]
    return dict(speedup=math.exp(statistics.mean(math.log(value) for value in ratios)),
                pattern_speedups=ratios, p95_regression_ratios=tails,
                tail_gate=all(value <= 1.10 for value in tails))


def promotion_gate(pairs):
    if len(pairs) < 3:
        return False
    return (all(pair['tail_gate'] and pair['speedup'] >= 1.03 for pair in pairs)
            and math.exp(statistics.mean(math.log(pair['speedup']) for pair in pairs)) >= 1.05)


def build(args):
    directory = Path(tempfile.mkdtemp(prefix='build-', dir=SCRATCH))
    evidence = Evidence(directory, SCRATCH / 'build-receipt.json', 'autoresearch-build')
    try:
        bend = device.executable(args.bend)
        cc = device.executable(args.cc or os.environ.get('ANDROID_CC', ''))
        sources = source_ids(bend)
        evidence.data.update(sources=sources, bend=identity(bend), compiler=identity(cc))
        version = checked(evidence, 'bend-version', [bend, 'version'])
        if not re.search(r'(?<![0-9.])2\.0\.37(?![0-9.])', version['stdout'] + version['stderr']):
            raise RuntimeError('pinned Bend2.0.37 required')
        checked(evidence, 'compiler-version', [cc, '--version'])
        BUILD.mkdir(parents=True, exist_ok=True)
        for stem, entry in (('oracle', 'oracle.bend'), ('bench', 'bench.bend')):
            output = BUILD / (stem + '.generated.c')
            checked(evidence, 'emit-' + stem, [bend, str(HERE / entry), '-o', str(output)], 300)
            checked(evidence, 'compile-' + stem, [cc, '-I' + str(HERE), '-ffp-contract=off', '-fno-fast-math', str(output), '-o', str(BUILD / (stem + '-android'))], 300)
        checked(evidence, 'emit-refusal', [bend, str(HERE / 'bench.bend'), '-o', str(BUILD / 'bench.js')], 300)
        if sources != source_ids(bend):
            raise RuntimeError('sources changed during build')
        evidence.data['artifacts'] = {name: identity(BUILD / name) for name in ('bench-android', 'oracle-android', 'bench.js')}
        evidence.finish(True)
    except BaseException as error:
        evidence.finish(False, error)
        raise
    print('AUTORESEARCH_BUILD_OK ' + str(evidence.current), flush=True)


def require_oracle_witness(session, pattern, values):
    """Cached numbers require the retained successful production-command evidence."""
    for path in sorted(Path(session).glob('prepare*/receipt.json')):
        receipt = json.loads(path.read_text())
        for command in receipt.get('commands', []):
            if (command.get('label') == 'production-oracle-' + str(pattern)
                    and command.get('acceptance_met') is True
                    and command.get('exit_code') == 0 and command.get('timed_out') is False):
                if words(command.get('stdout', ''), 'CPU_GEMV_BITS', pattern) == values:
                    return
    raise ValidationError('cached oracle has no matching retained production witness')


class Experiment:
    def __init__(self, args):
        self.args = args
        self.config = dict(samples=args.samples, warmups=args.warmups, batch=args.batch)
        self.adb = device.executable(args.adb)
        self.control = generate({'family': 'control'})
        self.process = None
        self.model_log = None
        self.model_round = 0
        self.last_models = 0
        self.model_seen = set()
        self.model_blocked = None
        self.uploaded = set()
        self.queue = []
        self.seen = set()
        self.results = []
        self.best = None
        self.end = time.time() + args.hours * 3600
        self.session = Path(args.resume).resolve(strict=True) if args.resume else Path(tempfile.mkdtemp(prefix='session-', dir=SCRATCH))
        self.state_path = self.session / 'state.json'
        self.remote = None
        self.oracles = {}
        self.build = json.loads((SCRATCH / 'build-receipt.json').read_text())
        if not self.build.get('acceptance_met'):
            raise RuntimeError('no successful current build')
        if self.build['sources'] != source_ids(self.build['bend']['path']):
            raise RuntimeError('source changed since build; rebuild before running')
        for item in self.build['artifacts'].values():
            if identity(item['path']) != item:
                raise RuntimeError('changed build artifact')
        self.tensor = identity(args.tensor)
        if self.tensor['bytes'] != 1327104 or self.tensor['sha256'] != TENSOR_SHA:
            raise RuntimeError('exact real SmolLM2 h0.q tensor required')
        self.expected_hashes = {}
        self.prepared = False
        self.stage = 'initializing'
        self.current_candidate = None
        if args.resume:
            old = json.loads(self.state_path.read_text())
            if old['serial'] != args.serial or old['tensor'] != self.tensor or old['build'] != self.build or old['config'] != self.config:
                raise RuntimeError('resume identity/config mismatch')
            self.remote = old['remote']
            self.oracles = {int(key): value for key, value in old['oracles'].items()}
            self.best, self.results = old['best'], old['results']
            self.seen, self.model_seen = set(old['seen']), set(old.get('model_seen', []))
            self.model_round = old.get('model_round', 0)
            self.model_blocked = old.get('model_blocked')
            self.end = old['deadline_epoch']
        self.monotonic_end = time.monotonic() + max(0, min(args.hours * 3600, self.end - time.time()))
        self.persist()

    def persist(self, error=None):
        state = dict(schema_version=1, updated_at=now(), stage=self.stage, error=str(error) if error else None,
                     session=str(self.session), serial=self.args.serial, tensor=self.tensor, build=self.build,
                     config=self.config, deadline_epoch=self.end, remote=self.remote,
                     oracles=self.oracles, best=self.best, results=self.results, seen=sorted(self.seen),
                     model_seen=sorted(self.model_seen), model_round=self.model_round,
                     model_blocked=self.model_blocked,
                     current_candidate=self.current_candidate,
                     scope='resident 576x576 foreign-effect GEMV; not native !, inference or global optimum')
        atomic_json(self.state_path, state)
        atomic_json(SCRATCH / 'current.json', state)

    def deadline(self):
        if STOPPING or (self.session / 'STOP').exists() or time.time() >= self.end or time.monotonic() >= self.monotonic_end:
            raise Deadline('stop requested or wall-clock deadline reached')

    def invoke(self, evidence, label, command, timeout=60):
        self.deadline()
        return checked(evidence, label, [self.adb, '-s', self.args.serial, 'shell', command], timeout + 10, True)

    def guard(self, evidence):
        self.deadline()
        if shutil.disk_usage(SCRATCH).free < 1024**3:
            raise RuntimeError('INVALID: less than 1GiB local headroom; no data deleted')
        if sum(path.stat().st_size for path in self.session.rglob('*') if path.is_file()) > 1024**3:
            raise RuntimeError('receipt budget1GiB reached; preserving data and stopping')
        item = self.invoke(evidence, 'thermal-preflight', "printf 'temperature='; dumpsys battery get temp; printf 'level='; dumpsys battery get level; printf 'powered='; dumpsys battery get ac; dumpsys thermalservice")
        output = item['stdout']
        def number(pattern):
            found = re.findall(pattern, output, re.M)
            if len(found) != 1:
                raise RuntimeError('INVALID: thermal/battery instrumentation unavailable')
            return int(found[0])
        temperature = number(r'^temperature=([0-9]+)\s*$') / 10
        level = number(r'^level=([0-9]+)\s*$')
        status = number(r'^Thermal Status: ([0-9]+)\s*$')
        powered = re.search(r'^powered=(true|false|0|1)\s*$', output, re.M)
        if not powered or not 0 <= level <= 100 or not 0 < temperature < 80 or not 0 <= status <= 6:
            raise RuntimeError('INVALID: malformed safety instrumentation')
        if level < 25 and powered[1] not in ('true', '1'):
            raise Cooling('unplugged battery below25%; waiting, not weakening thresholds')
        if temperature >= 41.5 or status >= 2:
            raise Cooling(f'thermal pause: battery{temperature}C status{status}')
        return dict(battery_c=temperature, thermal_status=status, level=level, powered=powered[1])

    def verify_remote(self, evidence):
        self.invoke(evidence, 'owned-directory', device.ownership_command(self.remote))
        paths = [self.remote + '/model/w/h0.q.bin', self.remote + '/bench-android', self.remote + '/oracle-android']
        expected = {paths[0]: self.tensor['sha256'], paths[1]: self.build['artifacts']['bench-android']['sha256'], paths[2]: self.build['artifacts']['oracle-android']['sha256']}
        item = self.invoke(evidence, 'remote-byte-identities', 'sha256sum ' + ' '.join(map(shlex.quote, paths)))
        device.validate_remote_hashes(item['stdout'], expected)

    def prepare(self):
        directory = Path(tempfile.mkdtemp(prefix='prepare-', dir=self.session))
        evidence = Evidence(directory, SCRATCH / 'latest.json', 'autoresearch-prepare')
        try:
            self.guard(evidence)
            state = checked(evidence, 'explicit-device', [self.adb, '-s', self.args.serial, 'get-state'])
            if state['stdout'].strip() != 'device':
                raise RuntimeError('explicit physical device unavailable')
            uid = self.invoke(evidence, 'unprivileged-device', 'id -u')
            device.validate_device_uid(uid['stdout'])
            emulated = self.invoke(evidence, 'physical-device', 'getprop ro.kernel.qemu')
            if emulated['stdout'].strip() not in ('', '0'):
                raise RuntimeError('emulator refused')
            model = self.invoke(evidence, 'device-model', 'getprop ro.product.model')
            if model['stdout'].strip().replace('_', '-') != 'SM-S938U':
                raise RuntimeError('wrong phone model')
            if not self.remote:
                item = self.invoke(evidence, 'exclusive-device-directory', 'umask 077; mktemp -d /data/local/tmp/bend-adreno-gemv.XXXXXXXX')
                self.remote = device.validate_remote_path(item['stdout'].strip())
                self.invoke(evidence, 'exclusive-ownership', device.ownership_command(self.remote, empty=True))
                self.invoke(evidence, 'prepare-model-directory', 'mkdir -p ' + shlex.quote(self.remote + '/model/w'))
                uploads = [(self.tensor['path'], self.remote + '/model/w/h0.q.bin')]
                uploads += [(item['path'], self.remote + '/' + name) for name, item in self.build['artifacts'].items() if name.endswith('-android')]
                for local, remote in uploads:
                    self.invoke(evidence, 'verify-before-upload', device.ownership_command(self.remote))
                    checked(evidence, 'upload-' + Path(remote).name, [self.adb, '-s', self.args.serial, 'push', local, remote])
                self.invoke(evidence, 'executable-permissions', 'chmod 700 ' + shlex.quote(self.remote + '/bench-android') + ' ' + shlex.quote(self.remote + '/oracle-android'))
            self.verify_remote(evidence)
            for pattern, values in self.oracles.items():
                require_oracle_witness(self.session, pattern, values)
            for pattern in range(5):
                if pattern in self.oracles:
                    continue
                command = f'timeout -k 5 45 taskset 80 {shlex.quote(self.remote + "/oracle-android")} --threads 1 --gpu off {shlex.quote(self.remote + "/model")} {pattern}'
                item = self.invoke(evidence, 'production-oracle-' + str(pattern), command)
                self.oracles[pattern] = words(item['stdout'], 'CPU_GEMV_BITS', pattern)
                self.persist()
            if set(self.oracles) != set(range(5)):
                raise RuntimeError('incomplete production oracles')
            evidence.finish(True)
            self.prepared = True
            self.stage = 'ready'
            self.persist()
        except BaseException as error:
            evidence.finish(False, error)
            raise

    def measure(self, evidence, candidate, pattern):
        local = self.session / 'kernels' / (candidate['id'] + '.cl')
        local.parent.mkdir(exist_ok=True)
        if local.exists() and local.read_text() != candidate['source']:
            raise RuntimeError('candidate identity collision')
        if not local.exists():
            local.write_text(candidate['source'])
        remote = self.remote + '/' + candidate['id'] + '.cl'
        self.invoke(evidence, 'verify-candidate-directory', device.ownership_command(self.remote))
        if candidate['id'] not in self.uploaded:
            checked(evidence, 'upload-kernel', [self.adb, '-s', self.args.serial, 'push', str(local), remote])
            self.uploaded.add(candidate['id'])
        item = self.invoke(evidence, 'verify-kernel-bytes', 'sha256sum ' + shlex.quote(remote))
        device.validate_remote_hashes(item['stdout'], {remote: identity(local)['sha256']})
        params = ':'.join(map(str, (pattern, candidate['global_size'], candidate['local_size'], self.config['samples'], self.config['warmups'], self.config['batch'])))
        command = 'timeout -k 5 45 taskset 80 ' + shlex.quote(self.remote + '/bench-android') + ' --threads 1 --gpu off ' + ' '.join(map(shlex.quote, (self.remote + '/model/w/h0.q.bin', remote, params)))
        try:
            item = self.invoke(evidence, 'gpu-' + candidate['id'][:12] + '-' + str(pattern), command)
        except RuntimeError as error:
            failure = evidence.data['commands'][-1]
            stderr = failure.get('stderr', '')
            if failure.get('exit_code') in (33, 65):
                raise CandidateUnavailable(stderr[-4096:]) from error
            raise
        return parse_benchmark(item['stdout'], candidate, pattern, self.config, self.oracles[pattern])

    def pair(self, baseline, candidate, index, label):
        directory = Path(tempfile.mkdtemp(prefix=label + '-', dir=self.session))
        evidence = Evidence(directory, SCRATCH / 'latest.json', 'paired-gpu-measurement')
        evidence.data.update(baseline=baseline, candidate=candidate, config=self.config, tensor=self.tensor,
                             build_receipt=str(SCRATCH / 'build-receipt.json'), pattern_order=list(range(5)))
        try:
            before = self.guard(evidence)
            self.verify_remote(evidence)
            values = {'baseline': {}, 'candidate': {}}
            # Alternate order by pair AND input; every input has fresh same-session control.
            for pattern in range(5):
                order = [('baseline', baseline), ('candidate', candidate)]
                if (index + pattern) % 2:
                    order.reverse()
                for target, kernel in order:
                    self.deadline()
                    values[target][pattern] = self.measure(evidence, kernel, pattern)
                    evidence.data['cases'].append(dict(target=target, pattern=pattern, result=values[target][pattern]))
                    evidence.persist()
            after = self.guard(evidence)
            if abs(after['battery_c'] - before['battery_c']) > 2:
                raise Cooling('pair invalidated by >2C battery-temperature drift')
            self.verify_remote(evidence)
            result = assess_pair(values['baseline'], values['candidate'])
            result.update(receipt=str(directory / 'receipt.json'), thermal_before=before, thermal_after=after)
            evidence.data['assessment'] = result
            evidence.finish(True)
            return result
        except BaseException as error:
            evidence.finish(False, error)
            raise

    def model_context(self):
        return dict(schema=SPEC_SCHEMA, control=self.control, space_size=len(enumerate_specs()),
                    objective='geomean of five GPU event medians; every576 rows checked, tail<=1.10x; three alternating confirmations >=1.03x and aggregate>=1.05x',
                    observations=sorted([row for row in self.results if 'score' in row], key=lambda row: -row['score'])[:20],
                    rejected=[row for row in self.results if row['status'] == 'rejected'][-8:], best=self.best)

    def models(self):
        if not self.args.models:
            return
        if self.process and self.process.poll() is not None:
            if self.process.returncode == 2:
                raise RuntimeError('free-provider billing guard failed')
            self.model_log.close()
            self.process, self.model_log = None, None
        rounds = [SCRATCH / 'model-round-initial', *sorted(self.session.glob('models-*'))]
        for directory in rounds:
            for path in directory.glob('*.json'):
                if str(path) in self.model_seen or path.name in ('context.json', 'catalog.json', 'round.json'):
                    continue
                record = json.loads(path.read_text())
                if record.get('status') in ('running', 'starting'):
                    continue
                self.model_seen.add(str(path))
                if 'BILLING_GUARD' in record.get('error', ''):
                    raise RuntimeError('free-provider billing guard failed')
                failure = record.get('usage', {}).get('error') or {}
                if failure.get('data', {}).get('statusCode') == 403:
                    self.model_blocked = 'provider403: free-tier access unavailable even through official OpenCode; no retries/spoofing/paid fallback'
                if record.get('status') != 'completed':
                    continue
                for spec in record.get('candidates', []):
                    try:
                        candidate = generate(spec)
                        if candidate['id'] not in self.seen:
                            self.queue.insert(0, candidate)
                    except ValueError:
                        pass  # Invalid model specs never reach the compiler/device.
        if not self.process and not self.model_blocked and time.time() - self.last_models >= 1800:
            self.model_round += 1
            directory = self.session / ('models-' + str(self.model_round))
            directory.mkdir()
            context = directory / 'context.json'
            atomic_json(context, self.model_context())
            self.model_log = (directory / 'broker.log').open('w')
            self.process = subprocess.Popen([device.executable(self.args.node), str(HERE / 'models.mjs'), str(directory), str(context)], cwd=ROOT, stdout=self.model_log, stderr=subprocess.STDOUT)
            self.last_models = time.time()
            print('FREE_MODELS_RUNNING ' + str(directory), flush=True)

    def evaluate(self, candidate, ordinal):
        self.current_candidate = dict(id=candidate['id'], spec=candidate['spec'])
        self.stage = 'measuring'
        self.persist()
        first = self.pair(self.control, candidate, ordinal, 'screen')
        row = dict(id=candidate['id'], spec=candidate['spec'], status='screened', score=first['speedup'], pairs=[first])
        if first['speedup'] >= 1.05 and first['tail_gate']:
            row['pairs'] += [self.pair(self.control, candidate, ordinal + step, 'confirm') for step in (1, 2)]
            row['score'] = math.exp(statistics.mean(math.log(pair['speedup']) for pair in row['pairs']))
            if promotion_gate(row['pairs']) and (self.best is None or row['score'] > self.best['score'] * 1.02):
                beats_incumbent = True
                if self.best:
                    incumbent = generate(self.best['spec'])
                    head = [self.pair(incumbent, candidate, ordinal + step, 'incumbent') for step in (3, 4)]
                    row['incumbent_pairs'] = head
                    beats_incumbent = all(pair['tail_gate'] and pair['speedup'] >= 1.02 for pair in head)
                if beats_incumbent:
                    row['status'] = 'promoted'
                    self.best = row.copy()
                    atomic_json(self.session / 'best.json', dict(**self.best, kernel=candidate, recorded_at=now()))
                    print('NEW_MEASURED_BEST ' + json.dumps({'spec': row['spec'], 'speedup_vs_control': row['score']}), flush=True)
        self.results.append(row)
        self.seen.add(candidate['id'])
        self.current_candidate = None
        self.persist()
        print('CANDIDATE_COMPLETE ' + json.dumps({'id': candidate['id'], 'score': row['score'], 'status': row['status']}), flush=True)

    def self_check(self):
        # Exercise distinct compile/native-domain/numerical rejection transitions.
        original = '  y[row] = (s0 + s1) + (s2 + s3);'
        checks = [
            ('compile-error', '__kernel void gemv(', CandidateUnavailable),
            ('nonfinite-last-row', self.control['source'].replace(original,
             '  y[row] = row == 575u ? as_float((uint)0x7fc00000u) : (s0+s1)+(s2+s3);'), CandidateUnavailable),
            ('last-row-corruption', self.control['source'].replace(original,
             '  y[row] = (s0+s1)+(s2+s3) + (row == 575u ? 1.0f : 0.0f);'), ValidationError),
        ]
        for label, source, expected in checks:
            bad = dict(self.control, id='deliberate-' + label, source=source)
            directory = Path(tempfile.mkdtemp(prefix='rejection-' + label + '-', dir=self.session))
            evidence = Evidence(directory, SCRATCH / 'latest.json', 'physical-invalid-kernel-rejection')
            try:
                self.measure(evidence, bad, 4)
            except expected as error:
                if expected is ValidationError and 'numerical oracle mismatch' not in str(error):
                    evidence.finish(False, error)
                    raise
                evidence.data['expected_rejection'] = str(error)
                evidence.finish(True)
            except BaseException as error:
                evidence.finish(False, error)
                raise
            else:
                evidence.finish(False, 'invalid kernel incorrectly accepted')
                raise RuntimeError('invalid kernel accepted: ' + label)
        selected = generate({'family': 'cooperative', 'lanes': 4, 'local_size': 32, 'unroll': 1})
        self.evaluate(selected, 0)
        atomic_json(self.session / 'smoke.json', dict(status='PASS', actual_device=True,
                    rejected=[entry[0] for entry in checks], result=self.results[-1], best=self.best))
        print('AUTORESEARCH_SMOKE_PASS ' + str(self.session), flush=True)

    def cool(self, error):
        self.stage = 'cooling'
        self.persist(error)
        print('THERMAL_PAUSE ' + str(error), flush=True)
        for _ in range(60):
            self.deadline()
            time.sleep(1)

    def run(self):
        while True:
            try:
                self.prepare()
                if self.args.mode == 'smoke':
                    self.self_check()
                    self.stage = 'smoke-complete'
                    self.persist()
                    return
                # A complete fresh control is required even after resume/cooling.
                self.pair(self.control, self.control, 0, 'control-sanity')
                break
            except Cooling as error:
                self.cool(error)
        self.queue = [generate(spec) for spec in enumerate_specs() if generate(spec)['id'] != self.control['id']]
        random.Random(20261016).shuffle(self.queue)
        self.queue.insert(0, generate({'family': 'cooperative', 'lanes': 4, 'local_size': 32, 'unroll': 1}))
        print('AUTORESEARCH_READY ' + str(self.session), flush=True)
        ordinal = len(self.results)
        while True:
            try:
                self.deadline()
                self.models()
                if self.build['sources'] != source_ids(self.build['bend']['path']):
                    raise RuntimeError('INVALID: live source changed; stopping rather than mixing candidates')
                candidate = None
                while self.queue and candidate is None:
                    possible = self.queue.pop(0)
                    if possible['id'] not in self.seen:
                        candidate = possible
                if candidate is None:
                    # Retest exploratory leaders, including losses/noise, rather than declaring an optimum.
                    leaders = sorted([row for row in self.results if row['status'] != 'rejected'], key=lambda row: -row['score'])[:20]
                    if not leaders:
                        raise RuntimeError('all candidates unavailable/incorrect')
                    candidate = generate(leaders[ordinal % len(leaders)]['spec'])
                try:
                    self.evaluate(candidate, ordinal)
                except Cooling:
                    self.queue.insert(0, candidate)
                    raise
                except (ValidationError, CandidateUnavailable) as error:
                    self.results.append(dict(id=candidate['id'], spec=candidate['spec'], status='rejected', error=str(error)))
                    self.seen.add(candidate['id'])
                    self.persist(error)
                ordinal += 1
            except Cooling as error:
                self.cool(error)

    def close(self):
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        if self.model_log:
            self.model_log.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('build', 'smoke', 'run'))
    parser.add_argument('--lock', required=True)
    parser.add_argument('--bend', default='bend')
    parser.add_argument('--cc')
    parser.add_argument('--adb', default='adb')
    parser.add_argument('--node', default='node')
    parser.add_argument('--serial')
    parser.add_argument('--tensor')
    parser.add_argument('--hours', type=float, default=8)
    parser.add_argument('--samples', type=int, default=17)
    parser.add_argument('--warmups', type=int, default=8)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--resume')
    parser.add_argument('--models', action='store_true')
    args = parser.parse_args()
    if not math.isfinite(args.hours) or not 0 < args.hours <= 12 or not 3 <= args.samples <= 101 or not 1 <= args.warmups <= 128 or not 1 <= args.batch <= 64:
        parser.error('duration/sample/warmup/batch out of bounded range')
    if args.mode != 'build' and (not args.serial or not args.tensor):
        parser.error('physical --serial and real --tensor are required')
    if args.models and not os.environ.get('OPENCODE_PRIVATE_SETTINGS'):
        parser.error('--models requires OPENCODE_PRIVATE_SETTINGS; no paid fallback')
    os.umask(0o077)
    SCRATCH.mkdir(parents=True, exist_ok=True)
    def stopping(signum, frame):
        global STOPPING
        STOPPING = True
    signal.signal(signal.SIGTERM, stopping)
    signal.signal(signal.SIGINT, stopping)
    with open(args.lock, 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.mode == 'build':
            build(args)
            return 0
        experiment = None
        atomic_json(SCRATCH / 'current.json', dict(stage='initializing', started_at=now(), acceptance_met=False))
        try:
            experiment = Experiment(args)
            experiment.run()
        except Deadline as error:
            if experiment:
                experiment.stage = 'stopped' if STOPPING or (experiment.session / 'STOP').exists() else 'deadline-complete'
                experiment.persist(error)
            return 0
        except BaseException as error:
            if experiment:
                experiment.stage = 'INVALID'
                experiment.persist(error)
            else:
                atomic_json(SCRATCH / 'current.json', dict(stage='INVALID', error=str(error), acceptance_met=False, updated_at=now()))
            print('AUTORESEARCH_INVALID ' + str(error), file=sys.stderr, flush=True)
            return 1
        finally:
            if experiment:
                experiment.close()
    return 0


if __name__ == '__main__':
    sys.exit(main())
