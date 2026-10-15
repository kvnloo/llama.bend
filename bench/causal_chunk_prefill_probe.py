#!/usr/bin/env python3
"""Synthetic one-layer CPU Llama prefill: serial token GEMVs vs causal batch GEMMs.

This is a mechanism probe, NOT llama.bend or full-model latency evidence. It uses
SmolLM2 dimensions but seeded SYNTHETIC weights/embeddings. Do not use its timing
to claim a Bend, model-loading, TTFT, throughput, or numerical-oracle result.
Renan Vinicius owns llama.bend and its verified tensor/causal architecture;
Kevin directed the downstream study. No upstream/Modular implementation copied.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import statistics
import time

import torch
import torch.nn.functional as F

# (hidden, query heads, KV heads, FFN width, layers, vocabulary)
# 135M sizes are from nuxyel/llama.bend NOTES.md; this script uses one layer.
CONFIGS = {
    "135m": (576, 9, 3, 1536, 30, 49152),
}


def weights(config, seed):
    h, heads, kv_heads, ffn, _, vocab = config
    torch.manual_seed(seed)
    kv_width = kv_heads * (h // heads)
    scale = h ** -.5
    def mat(rows, cols):
        return torch.randn(rows, cols, dtype=torch.float32) * scale
    return {
        "q": mat(h, h), "k": mat(kv_width, h), "v": mat(kv_width, h),
        "o": mat(h, h), "gate": mat(ffn, h), "up": mat(ffn, h),
        "down": mat(h, ffn), "norm1": torch.ones(h), "norm2": torch.ones(h),
        "head": mat(vocab, h),  # logits/argmax-only synthetic head; no real vocab weights
    }


def rms(x, gain, eps=1e-5):
    return x * torch.rsqrt((x * x).mean(dim=-1, keepdim=True) + eps) * gain


def rope(x, positions, theta=100000.0):
    """Llama rotate-half, head dimension even; no interleaving."""
    half = x.shape[-1] // 2
    assert x.shape[-1] == 2 * half
    inv = torch.pow(torch.tensor(theta), -torch.arange(half).float() / half)
    a = positions.float().unsqueeze(-1) * inv
    co = torch.cat((a.cos(), a.cos()), -1)[:, None, :]
    si = torch.cat((a.sin(), a.sin()), -1)[:, None, :]
    rotated = torch.cat((-x[..., half:], x[..., :half]), -1)
    return x * co + rotated * si


def serial_prefill(x, w, config):
    """Actual causal growing KV per position; all projections are vector products."""
    h, heads, kv_heads, _, _, _ = config
    hd = h // heads
    group = heads // kv_heads
    keys, vals, outputs = [], [], []
    for t in range(x.shape[0]):
        a = rms(x[t], w["norm1"])
        q = rope((w["q"] @ a).reshape(1, heads, hd), torch.tensor([t]))[0]
        k = rope((w["k"] @ a).reshape(1, kv_heads, hd), torch.tensor([t]))[0]
        v = (w["v"] @ a).reshape(kv_heads, hd)
        keys.append(k)
        vals.append(v)
        kk = torch.stack(keys).repeat_interleave(group, 1)
        vv = torch.stack(vals).repeat_interleave(group, 1)
        scores = torch.einsum("hd,shd->hs", q, kk) * (hd ** -.5)
        p = torch.softmax(scores, -1)
        att = torch.einsum("hs,shd->hd", p, vv).reshape(h)
        x1 = x[t] + w["o"] @ att
        n2 = rms(x1, w["norm2"])
        out = x1 + w["down"] @ (F.silu(w["gate"] @ n2) * (w["up"] @ n2))
        outputs.append(out)
    return torch.stack(outputs)


def chunked_prefill(x, w, config):
    """All positions in one GEMM; triangular mask ensures no future KV is read."""
    h, heads, kv_heads, _, _, _ = config
    hd = h // heads
    group = heads // kv_heads
    t = x.shape[0]
    n1 = rms(x, w["norm1"])
    positions = torch.arange(t)
    q = rope((n1 @ w["q"].T).reshape(t, heads, hd), positions)
    k = rope((n1 @ w["k"].T).reshape(t, kv_heads, hd), positions)
    v = (n1 @ w["v"].T).reshape(t, kv_heads, hd)
    kk = k.repeat_interleave(group, 1)
    vv = v.repeat_interleave(group, 1)
    scores = torch.einsum("thd,shd->hts", q, kk) * (hd ** -.5)
    mask = torch.triu(torch.ones(t, t, dtype=torch.bool), diagonal=1)
    scores.masked_fill_(mask[None, :, :], -torch.inf)
    p = torch.softmax(scores, -1)
    att = torch.einsum("hts,shd->thd", p, vv).reshape(t, h)
    x1 = x + att @ w["o"].T
    n2 = rms(x1, w["norm2"])
    return x1 + (F.silu(n2 @ w["gate"].T) * (n2 @ w["up"].T)) @ w["down"].T


def ensure_parity(x, w, config, *, verbose=False):
    a = serial_prefill(x, w, config)
    b = chunked_prefill(x, w, config)
    assert a.shape == b.shape
    maxabs = (a - b).abs().max().item()
    denom = a.abs().max().item()
    max_rel_scale = maxabs / max(1e-12, denom)
    ok = torch.allclose(a, b, rtol=1e-5, atol=1e-5)
    # A batch cannot affect earlier results when only future positions change.
    modified = x.clone()
    modified[-1].mul_(100)
    b2 = chunked_prefill(modified, w, config)
    causal = torch.equal(b[:-1], b2[:-1])
    la, lb = w["head"] @ a[-1], w["head"] @ b[-1]
    greedy = la.argmax().item() == lb.argmax().item()
    logits_close = torch.allclose(la, lb, rtol=1e-5, atol=1e-5)
    if verbose:
        print(json.dumps({"parity": ok, "causal_future_invariance_exact": causal,
                          "synthetic_logits_close": logits_close, "greedy_token_same": greedy,
                          "max_abs_diff": maxabs, "max_diff_over_peak_abs": max_rel_scale,
                          "greedy_id": la.argmax().item()}, sort_keys=True))
    if not (ok and causal and greedy and logits_close):
        raise AssertionError("numerical, causal or greedy parity failed")
    return {"max_abs_diff": maxabs, "max_diff_over_peak_abs": max_rel_scale,
            "causal_future_invariance_exact": causal, "synthetic_logits_close": logits_close,
            "greedy_token_same": greedy}


def percentile(values, p):
    s = sorted(values)
    return s[min(len(s)-1, max(0, int((len(s)-1)*p+0.5)))]


def timed(fn, x, w, c):
    t0 = time.perf_counter_ns()
    result = fn(x, w, c)
    elapsed = (time.perf_counter_ns()-t0)/1e6
    assert result.shape == x.shape
    return elapsed


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", choices=CONFIGS, default="135m")
    p.add_argument("--tokens", type=int, default=32)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--rounds", type=int, default=8)
    p.add_argument("--warmups", type=int, default=2)
    p.add_argument("--seed", type=int, default=140715)
    p.add_argument("--verify-only", action="store_true")
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    if min(args.tokens, args.threads, args.rounds) < 1 or args.warmups < 0:
        p.error("tokens/threads/rounds must be positive; warmups must be nonnegative")
    torch.set_num_threads(args.threads)
    c = CONFIGS[args.config]
    w = weights(c, args.seed)
    torch.manual_seed(args.seed+1)
    x = torch.randn(args.tokens, c[0])
    with torch.inference_mode():
        parity = ensure_parity(x, w, c)
        if args.verify_only:
            print(json.dumps({"status":"verified_synthetic_only","config":args.config,
                              "tokens":args.tokens,"parity":parity},indent=2))
            return
        # Warm both implementations outside the measured window; stable env.
        for _ in range(args.warmups):
            serial_prefill(x, w, c)
            chunked_prefill(x, w, c)
        rng = random.Random(args.seed)
        order = [("serial", "chunked") if rng.getrandbits(1) else ("chunked", "serial")
                 for _ in range(args.rounds)]
        times = {"serial":[], "chunked":[]}
        funcs = {"serial":serial_prefill,"chunked":chunked_prefill}
        for aa,bb in order:
            for name in (aa,bb):
                times[name].append(timed(funcs[name], x, w, c))
    savings = [100*(a-b)/a for a,b in zip(times["serial"],times["chunked"])]
    summary = {name:{"median_ms":statistics.median(xs), "p95_ms":percentile(xs,.95),
                     "min_ms":min(xs),"max_ms":max(xs)} for name,xs in times.items()}
    source_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result={"scope":"synthetic_single_layer_prefill_only_NOT_Bend_or_model",
            "source_sha256":source_sha,"config":args.config,"shape":dict(zip(
                ("hidden","heads","kv_heads","ffn","layers_not_run","vocab"),c)),
            "tokens":args.tokens,"torch":torch.__version__,"python":platform.python_version(),
            "cpu":platform.processor(),"threads":args.threads,"affinity":sorted(os.sched_getaffinity(0)),
            "seed":args.seed,"warmups_per_arm":args.warmups,"interleaved_pairs":args.rounds,
            "orders":order,"samples_ms":times,"paired_savings_pct":savings,
            "median_paired_saving_pct":statistics.median(savings),"summary":summary,"parity":parity,
            "notes":"Synthetic one-layer Python/PyTorch dispatch and batched kernels; no Bend compiler/model/TTFT."
            }
    out=json.dumps(result,indent=2)
    if args.output:
        args.output.write_text(out+"\n")
    print(out)


if __name__=="__main__":
    main()
