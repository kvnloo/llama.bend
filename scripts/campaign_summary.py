#!/usr/bin/env python3
"""Summarize verified campaign samples and fresh paired differences.

The confidence range is a seeded paired bootstrap of median percentage savings,
not a guarantee. Fewer than five pairs remain exploratory. Negative savings are
regressions. Native diagnostic clocks and process wall/arrival clocks stay apart.
"""
import argparse
import json
import math
from pathlib import Path
import random
import statistics

from campaign_runner import canonical_hash, file_hash, validate_generation, plan_trials


def summarize(out):
    out=Path(out);m=json.loads((out/'manifest.json').read_text());digest=canonical_hash(m)
    specs={(c['id'],v['id']):v for c in m['cases'] for v in c['variants']}
    planned={t['key']:t for t in plan_trials(m)}
    passing_keys=set();runs={};failures=[]
    for line in (out/'journal.jsonl').read_text().splitlines():
        e=json.loads(line)
        if e['manifest_sha256']!=digest: raise ValueError('manifest mismatch')
        if e['event']!='finished':continue
        t=planned.get(e['key'])
        if t is None or any(e[k]!=t[k] for k in ('case','variant','phase','round')): raise ValueError('event differs from trial plan')
        if e['status']!='passed':
            if e['status']!='failed': raise ValueError('unknown trial status')
            if e['key'] in passing_keys: raise ValueError('failed attempt after a passing trial')
            failures.append({'key':e['key'],'case':e['case'],'variant':e['variant'],'error':e['error']})
            continue
        if e['key'] in passing_keys: raise ValueError('duplicate passing trial')
        passing_keys.add(e['key'])
        if type(e.get('returncode')) is not int or e['returncode'] != 0 or e['stop_reason'] is not None:
            raise ValueError('invalid passing exit')
        if Path(e['stderr']).read_bytes():raise ValueError('unexpected stderr')
        for metric in ('wall_ms','peak_rss_kib','peak_sampled_process_group_rss_kib','process_observed_ttft_upper_bound_ms','process_observed_decoded_text_upper_bound_ms'):
            value=e.get(metric)
            if value is not None and (type(value) not in (int, float) or not math.isfinite(value) or value<0):raise ValueError('invalid physical metric')
        if any(e.get(k) is not None and e[k]>e['wall_ms'] for k in ('process_observed_ttft_upper_bound_ms','process_observed_decoded_text_upper_bound_ms')):raise ValueError('arrival after process wall interval')
        for key in ('stdout','stderr'):
            if file_hash(e[key])!=e[key+'_sha256']:raise ValueError('changed raw evidence')
        rule=specs[(e['case'],e['variant'])]['validation']
        if rule['kind'] in ('generation','generation_stages'):
            derived=validate_generation(Path(e['stdout']).read_bytes().decode('utf-8'),rule['expected'],stages=rule['kind']=='generation_stages')
            if derived!=e['latency']:raise ValueError('saved latency mismatch')
        elif rule['kind']=='stdout_sha256':
            if e['stdout_sha256']!=rule['sha256']:raise ValueError('kernel output mismatch')
            if e['latency']!={}:raise ValueError('kernel trial has unsupported latency metrics')
        else:raise ValueError('unknown validation kind')
        if e['phase']=='sample':runs[(e['case'],e['variant'],e['round'])]=e
    def metrics(e):
        observed = {k:e.get(k) for k in (
            'wall_ms', 'process_observed_ttft_upper_bound_ms',
            'process_observed_decoded_text_upper_bound_ms', 'peak_rss_kib',
            'peak_sampled_process_group_rss_kib',
        )}
        return {k:v for k,v in (e['latency']|observed).items()
                if type(v) in (int, float) and k!='tokens'}
    confirmatory=m.get('purpose')=='confirmatory' and bool(m.get('selection_manifest_sha256')) and bool(m.get('primary_metric'))
    missing_keys=sorted(set(planned)-passing_keys)
    complete=not missing_keys
    result={'purpose':m.get('purpose','exploratory'), 'primary_metric':m.get('primary_metric'), 'manifest_sha256':digest,'label':m['label'],'failures':failures,
            'completion':{'complete':complete,'planned_trials':len(planned),'passed_trials':len(passing_keys),'missing_trial_keys':missing_keys},'cases':{}}
    for c in m['cases']:
        variants={};paired={}
        for v in c['variants']:
            rows=[e for (case,var,r),e in runs.items() if case==c['id'] and var==v['id']]
            names=set(k for e in rows for k in metrics(e))
            variants[v['id']]={'samples':len(rows),'metrics':{k:{'median':statistics.median(vals),'min':min(vals),'max':max(vals)} for k in names if (vals:=[metrics(e)[k] for e in rows if k in metrics(e)])}}
        control=c['variants'][0]['id']
        for v in c['variants'][1:]:
            pair_rows=[(runs[(c['id'],control,r)],runs[(c['id'],v['id'],r)]) for r in range(m['rounds']) if (c['id'],control,r) in runs and (c['id'],v['id'],r) in runs]
            names=set(k for a,b in pair_rows for k in metrics(a).keys()&metrics(b).keys())
            stats={}
            for k in sorted(names):
                if k == 'decode_slope_ms_per_position': continue  # Near-zero signed slopes need absolute units, not ratios.
                # An otherwise completed trial plan does not imply full primary-clock coverage.
                # Missing values and zero controls cannot silently select a winner.
                paired_values=((metrics(a).get(k),metrics(b).get(k)) for a,b in pair_rows)
                savings=[100*(a-b)/a for a,b in paired_values if a is not None and b is not None and a>0]
                if not savings:continue
                rng=random.Random(1234);boot=sorted(statistics.median(rng.choices(savings,k=len(savings))) for _ in range(2000))
                ci=[boot[int(.025*len(boot))],boot[int(.975*len(boot))]]
                stats[k]={'paired_n':len(savings),'median_saving_pct':statistics.median(savings),'min_saving_pct':min(savings),'max_saving_pct':max(savings),'paired_bootstrap_95pct':ci,'meets_primary_5pct_gate':confirmatory and complete and not failures and k==m['primary_metric'] and len(savings)>=5 and len(savings)==len(pair_rows)==m['rounds'] and ci[0]>=5}
            paired[v['id']]={'control':control,'exploratory':not confirmatory or len(pair_rows)<5,'metrics':stats}
        result['cases'][c['id']]={'variants':variants,'paired':paired}
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('out',type=Path);a=p.parse_args();print(json.dumps(summarize(a.out),indent=2,sort_keys=True))
