#!/usr/bin/env python3
"""Serialized, resumable CPU trials. Manifest argv runs directly, never through a shell.

No output is a measurement until its exit status and frozen oracle pass. Completed
trials resume only after raw-file hashes are rechecked. Failed/interrupted trials
remain in the append-only journal and are retried with new raw files. This tool
is a measurement mechanism, not evidence that any queued configuration ran.
"""
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import selectors
import shutil
import signal
import struct
import subprocess
import time
import uuid

from latency_report import report

TOKEN = re.compile(r'^\s+id\s+(\d+)\s+logit\s+(\d+)\s+(\S+)\s+t=(\d+(?:\.\d+)?)s\s*$')
STAGE_TOKEN = re.compile(r'^\s+id\s+(\d+)\s+logit\s+(\d+)\s+(\S+)\s*$')
STAGE_NAMES = ('model_load_ms', 'tokenize_ms', 'prefill_ms', 'first_forward_ms', 'resident_first_token_id_ms')
PROMPT = re.compile(r'^prompt ids:\s*(.*?)\s*$')

def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()

def file_hash(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''): h.update(chunk)
    return h.hexdigest()

def validate_generation(raw, expected, stages=False):
    """Strict completion/identity gate around PR2's intentionally diagnostic parser."""
    prefix, separator, suffix = raw.partition('\ntext: ')
    if not separator or not suffix.endswith('\n'):
        raise ValueError('missing complete final text')
    # IO.print adds exactly one newline; all text bytes before it are semantic.
    if suffix[:-1] != expected['text']:
        raise ValueError('final text differs or trailing output present')
    lines = prefix.split('\n')
    loads = [x for x in lines if x.startswith('model_load_ms=' if stages else 'weights loaded in ')]
    stage_values = {}
    if stages:
        for name in STAGE_NAMES:
            matches = [x for x in lines if x.startswith(name+'=')]
            if len(matches) != 1 or not re.fullmatch(name+r'=\d+', matches[0]): raise ValueError('missing/duplicate/malformed stage timer')
            stage_values[name] = int(matches[0].split('=')[1])
        if stage_values['prefill_ms'] + stage_values['first_forward_ms'] != stage_values['resident_first_token_id_ms']:
            raise ValueError('stage timer boundaries disagree')
    prompts = [x for x in lines if x.startswith('prompt ids:')]
    if len(loads) != 1 or len(prompts) != 1:
        raise ValueError('expected exactly one load and prompt header')
    if lines.index(loads[0]) > lines.index(prompts[0]): raise ValueError('load/prompt order')
    ids, bits, logits = [], [], []
    for i,line in enumerate(lines):
        if re.match(r'^\s*id\b', line):
            m = (STAGE_TOKEN if stages else TOKEN).fullmatch(line)
            if not m: raise ValueError('malformed generated-token row')
            value, word = float(m[3]), int(m[2])
            if not math.isfinite(value) or (not stages and not math.isfinite(float(m[4]))): raise ValueError('nonfinite logit/time')
            if not 0 <= word <= 0xffffffff: raise ValueError('logit bits outside U32')
            bitvalue = struct.unpack('<f', struct.pack('<I', word))[0]
            if not math.isfinite(bitvalue) or not math.isclose(bitvalue, value, rel_tol=1e-5, abs_tol=1e-6):
                raise ValueError('printed logit and bits disagree')
            if i <= lines.index(prompts[0]): raise ValueError('token before prompt')
            ids.append(int(m[1])); bits.append(word); logits.append(bitvalue)
        elif stages and any(line == f'{name}={value}' for name,value in stage_values.items()):
            pass
        elif line not in loads and line not in prompts and line.strip():
            raise ValueError('unexpected generation output line')
    try: prompt_ids = [int(x) for x in PROMPT.fullmatch(prompts[0])[1].split()]
    except (TypeError, ValueError): raise ValueError('malformed prompt IDs') from None
    if prompt_ids != expected['prompt_ids']: raise ValueError('prompt IDs differ from frozen oracle')
    if ids != expected['generated_ids']: raise ValueError('generated IDs/count differ from frozen oracle')
    if not ids: raise ValueError('oracle must contain generated IDs')
    if 0 in ids[:-1]: raise ValueError('tokens after EOS')
    if 'requested_tokens' in expected:
        n = expected['requested_tokens']
        if type(n) is not int or n < 1 or len(ids) > n or (len(ids) < n and ids[-1] != 0):
            raise ValueError('generated count outside requested/EOS bounds')
    if 'logit_bits' in expected:
        if bits != expected['logit_bits']: raise ValueError('logit bits differ')
    elif 'logits' in expected:
        want = expected['logits']
        if len(want) != len(logits) or not all(math.isfinite(x) for x in want): raise ValueError('invalid logit oracle')
        tolerance = expected['logit_tolerance']
        if type(tolerance) not in (int, float) or not math.isfinite(tolerance) or tolerance < 0: raise ValueError('invalid numerical tolerance')
        errors = [abs(a-b)/abs(b) if b else (0.0 if a == b else math.inf) for a,b in zip(logits,want)]
        if max(errors) > tolerance:
            raise ValueError('logits differ from numerical oracle')
    else: raise ValueError('numeric oracle required: logit_bits or logits')
    if stages:
        positions = {name: lines.index(f'{name}={value}') for name,value in stage_values.items()}
        token_positions = [i for i,line in enumerate(lines) if STAGE_TOKEN.fullmatch(line)]
        sequence = [positions['model_load_ms'], lines.index(prompts[0]), positions['tokenize_ms'], token_positions[0], positions['prefill_ms'], positions['first_forward_ms'], positions['resident_first_token_id_ms']]
        if sequence != sorted(sequence) or any(i < positions['resident_first_token_id_ms'] for i in token_positions[1:]):
            raise ValueError('stage records out of order')
        return stage_values | {'tokens': len(ids)}
    result = report(prefix)
    if result['load_s'] is None or any(isinstance(v, float) and not math.isfinite(v) for v in result.values()):
        raise ValueError('malformed/nonfinite latency metric')
    return result

def plan_trials(manifest):
    if type(manifest.get('schema')) is not int or manifest['schema'] != 1: raise ValueError('unsupported schema')
    if type(manifest.get('seed')) is not int: raise ValueError('invalid seed')
    for name in ('warmups', 'rounds'):
        if type(manifest.get(name)) is not int or manifest[name] < 0: raise ValueError(f'invalid {name}')
    if manifest['rounds'] < 1: raise ValueError('at least one measured round required')
    unique = {}
    for case in manifest['cases']:
        for variant in case['variants']:
            env = variant.get('env', {})
            if (not isinstance(env, dict) or
                any(not isinstance(k, str) or not isinstance(v, str) or not k or
                    '=' in k or '\0' in k or '\0' in v for k, v in env.items())):
                raise ValueError('invalid env: expected environment string mapping')
            key = (case['id'], variant['id'])
            cfg = {'case': case['id'], 'variant': variant['id'], 'spec': variant}
            if key in unique and unique[key] != cfg: raise ValueError('conflicting duplicate configuration')
            unique[key] = cfg
    if not unique: raise ValueError('no configurations to execute')
    if len(unique) > manifest.get('max_configurations', 420): raise ValueError('configuration budget exceeded')
    rng = random.Random(manifest['seed']); result = []
    # Group each case's variants in every round, yielding paired randomized A/B.
    for phase, n in [('warmup', manifest['warmups']), ('sample', manifest['rounds'])]:
        for r in range(n):
            cases = list(dict.fromkeys(x['case'] for x in unique.values())); rng.shuffle(cases)
            for case in cases:
                group = [x for x in unique.values() if x['case'] == case]; rng.shuffle(group)
                for cfg in group:
                    trial = cfg | {'phase': phase, 'round': r}
                    trial['key'] = canonical_hash(trial)
                    result.append(trial)
    return result

def append_event(path, event):
    event = event | {'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
    with path.open('a', encoding='utf-8') as f:
        f.write(json.dumps(event, sort_keys=True, allow_nan=False) + '\n'); f.flush(); os.fsync(f.fileno())

def host_snapshot():
    def read(path):
        try: return Path(path).read_text().strip()
        except OSError: return None
    return {'affinity': sorted(os.sched_getaffinity(0)), 'loadavg': list(os.getloadavg()),
            'cpu_max': read('/sys/fs/cgroup/cpu.max'), 'meminfo': read('/proc/meminfo'),
            'cpuinfo': read('/proc/cpuinfo'), 'platform': list(os.uname())}

def group_rss_kib(pgid):
    total = 0
    for p in Path('/proc').glob('[0-9]*/status'):
        try:
            if os.getpgid(int(p.parent.name)) != pgid: continue
            row = next((x for x in p.read_text().splitlines() if x.startswith('VmRSS:')), None)
            if row: total += int(row.split()[1])
        except (OSError, ProcessLookupError): pass
    return total

def execute(spec, budget, rawbase):
    """Read stdout/stderr concurrently, monitor only this process group, reap via wait4.

    Bend reserves a large sparse virtual arena, so RLIMIT_AS is inappropriate.
    RSS is sampled every 20ms; the budget is a watchdog, not a kernel hard limit.
    """
    argv = spec['argv']
    if not argv or not all(isinstance(x,str) for x in argv): raise ValueError('argv must be strings')
    env = os.environ.copy(); env.update(spec.get('env', {}))
    stdout_path = rawbase.with_suffix('.stdout'); stderr_path = rawbase.with_suffix('.stderr')
    before = host_snapshot(); start = time.monotonic_ns()
    proc = subprocess.Popen(argv, cwd=spec['cwd'], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
    sel = selectors.DefaultSelector(); files={}
    for stream,name,path in [(proc.stdout,'stdout',stdout_path),(proc.stderr,'stderr',stderr_path)]:
        os.set_blocking(stream.fileno(),False); sel.register(stream,selectors.EVENT_READ,name); files[name]=path.open('xb')
    remaining = b''; first_arrival = None; text_arrival = None; count=0; peak=0; reason=None; exit_info=None; killed_at=None
    try:
        while sel.get_map() or exit_info is None:
            now=time.monotonic_ns(); elapsed=(now-start)/1e9
            if exit_info is None:
                peak=max(peak,group_rss_kib(proc.pid))
                if elapsed > budget['timeout_s'] and reason is None: reason='timeout'
                if peak > budget['memory_mb']*1024 and reason is None: reason='rss_budget'
                if reason and killed_at is None:
                    try: os.killpg(proc.pid,signal.SIGTERM)
                    except ProcessLookupError: pass
                    killed_at=now
                if killed_at and (now-killed_at)>2e9:
                    try: os.killpg(proc.pid,signal.SIGKILL)
                    except ProcessLookupError: pass
                pid,status,usage=os.wait4(proc.pid,os.WNOHANG)
                if pid:
                    proc.returncode=os.waitstatus_to_exitcode(status); exit_info=usage
            for key,_ in sel.select(.02):
                chunk=os.read(key.fd,65536)
                if not chunk: sel.unregister(key.fileobj);key.fileobj.close();continue
                count+=len(chunk)
                if count <= budget['max_output_mb']*1024*1024:
                    files[key.data].write(chunk)
                    # Derive observations only from the bytes retained in the
                    # immutable trial receipt. Discarded output is not evidence.
                    if key.data=='stdout':
                        remaining+=chunk
                        rows=remaining.split(b'\n');remaining=rows.pop()
                        if first_arrival is None and any(re.match(rb'^\s+id\s+\d+\s+logit\b', x) for x in rows): first_arrival=(time.monotonic_ns()-start)/1e6
                        if text_arrival is None and any(x.startswith(b'text: ') for x in rows): text_arrival=(time.monotonic_ns()-start)/1e6
                elif reason is None: reason='output_budget'
            if exit_info is not None and sel.get_map() and elapsed > budget['timeout_s']+3:
                reason=reason or 'descendant_output_timeout'
                try:os.killpg(proc.pid,signal.SIGKILL)
                except ProcessLookupError:pass
                break
    finally:
        if proc.returncode is None:
            try:os.killpg(proc.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            _,status,exit_info=os.wait4(proc.pid,0);proc.returncode=os.waitstatus_to_exitcode(status)
        for f in files.values(): f.flush();os.fsync(f.fileno());f.close()
        sel.close()
    return {'returncode':proc.returncode,'stop_reason':reason,'wall_ms':(time.monotonic_ns()-start)/1e6,
            'process_observed_ttft_upper_bound_ms':first_arrival,
            'process_observed_decoded_text_upper_bound_ms':text_arrival,
            'stdout_buffering':spec.get('stdout_buffering','unverified'),
            'peak_rss_kib':exit_info.ru_maxrss,'peak_sampled_process_group_rss_kib':peak,
            'host_before':before,'host_after':host_snapshot(),
            'stdout':str(stdout_path),'stderr':str(stderr_path),
            'stdout_sha256':file_hash(stdout_path),'stderr_sha256':file_hash(stderr_path)}

def verify_inputs(manifest):
    for item in manifest.get('inputs', []):
        if file_hash(item['path']) != item['sha256']: raise ValueError(f'input hash changed: {item["path"]}')

def run_manifest(manifest, out):
    out=Path(out).resolve();out.mkdir(parents=True,exist_ok=True);(out/'raw').mkdir(exist_ok=True)
    digest=canonical_hash(manifest); plan=plan_trials(manifest)
    lock=Path(manifest['lock']);lock.parent.mkdir(parents=True,exist_ok=True)
    with lock.open('a') as lockfile:
        fcntl.flock(lockfile,fcntl.LOCK_EX)
        saved=out/'manifest.json'
        if saved.exists():
            if canonical_hash(json.loads(saved.read_text())) != digest: raise ValueError('manifest changed; choose a new output directory')
        else:
            with saved.open('x') as f: json.dump(manifest,f,indent=2,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
        verify_inputs(manifest)
        journal=out/'journal.jsonl'; done=set()
        if journal.exists():
            for line in journal.read_text().splitlines():
                event=json.loads(line)  # Truncated/corrupt journal is fail-closed, never silently skipped.
                if event.get('manifest_sha256') != digest: raise ValueError('journal manifest mismatch')
                if event['event']=='finished' and event['status']=='failed' and event.get('key') in done:
                    raise ValueError('failed attempt after passing trial')
                if event['event']=='finished' and event['status']=='passed':
                    if event.get('key') in done:
                        raise ValueError('duplicate passing trial')
                    for name in ('stdout','stderr'):
                        if file_hash(event[name]) != event[name+'_sha256']: raise ValueError('completed raw evidence changed')
                    trial = next((t for t in plan if t['key'] == event['key']), None)
                    if trial is None or any(event.get(k) != trial[k] for k in ('case', 'variant', 'phase', 'round')):
                        raise ValueError('completed trial identity mismatch')
                    if type(event.get('returncode')) is not int or event['returncode'] != 0 or event.get('stop_reason') is not None:
                        raise ValueError('completed trial lacks successful exit evidence')
                    if Path(event['stderr']).read_bytes(): raise ValueError('completed trial has unexpected stderr')
                    rule = trial['spec']['validation']
                    if rule['kind'] in ('generation', 'generation_stages'):
                        derived = validate_generation(Path(event['stdout']).read_bytes().decode('utf-8'), rule['expected'], stages=rule['kind']=='generation_stages')
                        if derived != event.get('latency'): raise ValueError('saved latency differs from raw evidence')
                    elif rule['kind'] == 'stdout_sha256':
                        if event['stdout_sha256'] != rule['sha256']: raise ValueError('completed kernel output differs')
                    else: raise ValueError('unknown validation kind')
                    done.add(event['key'])
        failed=0
        for trial in plan:
            if trial['key'] in done: continue
            if shutil.disk_usage(out).free < manifest['budget']['min_free_mb']*1024*1024:
                append_event(journal,{'event':'blocked','manifest_sha256':digest,'key':trial['key'],'reason':'free_disk_budget'});return 2
            attempt=uuid.uuid4().hex
            common={'manifest_sha256':digest,'key':trial['key'],'attempt':attempt,'case':trial['case'],'variant':trial['variant'],'phase':trial['phase'],'round':trial['round']}
            append_event(journal,common|{'event':'started','argv':trial['spec']['argv']})
            try:
                result=execute(trial['spec'],manifest['budget'],out/'raw'/attempt)
                if type(result.get('returncode')) is not int or result['returncode'] != 0 or result.get('stop_reason') is not None:
                    raise ValueError(f'child failed: {result.get("returncode")}, {result.get("stop_reason")}')
                if Path(result['stderr']).read_bytes(): raise ValueError('unexpected stderr; inspect preserved raw output')
                rule=trial['spec']['validation']; raw=Path(result['stdout']).read_bytes().decode('utf-8')
                if rule['kind'] in ('generation','generation_stages'): latency=validate_generation(raw,rule['expected'],stages=rule['kind']=='generation_stages')
                elif rule['kind']=='stdout_sha256':
                    if result['stdout_sha256'] != rule['sha256']: raise ValueError('kernel output differs from frozen oracle')
                    latency={}
                else: raise ValueError('unknown validation kind')
                append_event(journal,common|result|{'event':'finished','status':'passed','latency':latency})
                print(f'{trial["phase"]} {trial["round"]} {trial["case"]}/{trial["variant"]}: passed',flush=True)
            except (OSError,ValueError) as e:
                append_event(journal,common|locals().get('result',{})|{'event':'finished','status':'failed','error':str(e)})
                failed+=1;print(f'{trial["case"]}/{trial["variant"]}: failed: {e}',flush=True)
            finally: result={}
        return 1 if failed else 0

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('manifest',type=Path);p.add_argument('--out',type=Path,required=True);p.add_argument('--plan-only',action='store_true');args=p.parse_args()
    m=json.loads(args.manifest.read_text())
    if args.plan_only:
        print(json.dumps({'state':'planned_not_executed','manifest_sha256':canonical_hash(m),'trials':plan_trials(m)},indent=2));return 0
    return run_manifest(m,args.out)

if __name__=='__main__': raise SystemExit(main())
