# Negative results and limits

## Resource guard setup

Initial use of a 6 GiB address-space limit prevented Bend/Lean from reserving
virtual memory. This was a harness setup failure, not numerical evidence or a
candidate performance result. Its logs remain in the baseline evidence. Accepted
runs instead use a sampled aggregate process-group RSS watchdog, identically
configured across compared arms.

## Prefill skipping

Skipping logits that are unused during prompt consumption looked promising for
the resident first-ID clock; bit preservation is a separate correctness gate. The five fresh one-thread, 32-token-prompt finalist pairs initially
cleared the selected native first-ID clock criterion. Five additional independent
pairs produced savings of 7.79%, 11.35%, 11.92%, 4.39% and 7.72%: median 7.79%,
with a 4.39% bootstrap lower bound. That does not clear the required 5% lower
bound. The independent batch is not pooled into the earlier batch to hide the
weaker result. The eight-thread candidate did not clear the gate either.

The native first-ID clock begins after model loading and tokenization. It is
not end-to-end time to decoded text. Independent decoded-text arrival savings
were 1.34–4.66%, median 3.18%. The candidate remains isolated and unselected.
Extended prompt checks do not change this performance decision.

## Loader zero-band reuse

The source change removes repeated zero-band allocation. Fresh A/B medians are
favorable on some loader/process measures, but variation prevents the required
primary-metric acceptance. There is no default change. Later attribution used
an instrumented C diagnostic and microprobes. Measured instrumented loads were
5.711–5.837 s; summed `File.read_at` call regions were 43.13–44.34 ms and shared
byte-list packing regions were 2.290–2.327 s. These are process-wide elapsed
regions, not CPU time or exclusive model-load stages. The list packer also serves
ordinary reads, so its 984 calls differ from the 981 `read_at` calls. The raw
scope erratum is retained. Discard/destruction probes were slower than consuming
walks, which makes naive phase subtraction misleading. No further read-size
experiment is justified by these observations alone.

## Shape/layout claims

B's shape sweep uses synthetic kernel fixtures with the real model dimensions.
It does not run the 360M or 1.7B models. C's packed-eight prototype is also kernel
research. Neither may be called a whole-model speedup without isolated integration
and real-model numerical and latency confirmation. The 1.7B large-head work is
deferred for calibrated resources, not declared impossible; the eligible 135M
and 360M head screens are complete.

The eligible 364-configuration sweep is complete. Its three measured rounds per
configuration are exploratory; selecting the smallest observed median would
overstate certainty. Fast head probes have only 40–100 ms of net work on a
millisecond clock. The 28 largest-head configurations remain deferred.

Packed-eight fresh Q/O confirmation retained one eight-thread saving of -33.08%.
Later gate/up and down synthetic batches had favorable five-pair ranges, but the
fixture generator has only 13 operand classes. Synthetic construction savings
are not checkpoint loading savings. Real packed loading, complete word parity,
full model numerical checks and fresh total-wall trials remain separate gates.

## Two-thread real-model scheduling

The single preregistered 135M policy used layer matvec depth 2, tied-head depth 5
and unchanged attention depth 0 at two threads. All numerical, local frozen-bit,
default-regression and emitted-C acceptance gates passed. Every later process
also preserved the complete 128-token local baseline IDs/U32 bits/text.

The primary was total process wall including load, with exactly five randomized
pairs in each of two separately seeded batches. Initial paired savings were
1.117%, 2.417%, 3.527%, -0.266%, 3.614%: median 2.417%, descriptive percentile
bootstrap lower bound -0.266%. Independent replication gave 3.772%, 11.974%,
2.404%, 2.343%, 7.099%: median 3.772%, lower bound 2.343%. Each complete batch
fails its required 5% lower bound. These small-sample bootstrap summaries make
no population-coverage guarantee. Neither pooling nor additional rounds are
used to rescue the policy. The repeated roughly 36.22% per-child peak-RSS
reduction (`wait4` `ru_maxrss`, distinct from sampled group RSS) is a secondary
observation and does not change the latency decision.

## Packed-loader compiler failures

Real packed loading has not yet produced accepted model evidence. Two native
fixture build attempts failed on Bend syntax before any of their 15 planned
children. A smaller loader-only verdict then exposed a linear-ownership error
in a reused scalar. A separately reviewed correction added sharing annotations
only to five pure U32/Nat fields and passed a fresh minimal verdict. A third
native fixture build then failed on fixture-local `U32` versus `Nat` depth
literals before any child. Four explicit Nat literals passed a separate
fixture-only verdict. At the 09:49 UTC checkpoint, all four failed receipts
remain intact; no failed build is counted as a fixture or model execution.
Native fixtures, complete real-weight verification and model gates remain
separate prerequisites to performance trials.
