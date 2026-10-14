# Downstream latency campaign, 2026-10-14

This is Kevin's independent downstream experiment on `kvnloo/llama.bend`.
Renan Vinícius (nuxyel) and the upstream contributors retain credit for the
implementation, proofs and original measurements. No Modular source is copied.
The upstream benchmark numbers are not reproduced by this host.

## Current checkpoint

Execution is active. No performance candidate is selected for integration.
The baseline runtime is unchanged. Prefill, loader and two-thread scheduling
candidates remain isolated and unselected. Packed-eight has accepted synthetic
kernel evidence; its real-loader/model correctness gates are still in progress.

- Runtime: Bend 2.0.37, Lean 4.34.0, Clang 19.1.7, pinned Python dependencies.
- Full model: real pinned SmolLM2-135M revision
  `93efa2f097d58c2a74874c7e644dbc9b0cee75a2` only.
- Host: AMD EPYC 9V74 virtual CPUs, affinity 0–8, about 9.7 GiB RAM, no swap or
  exposed GPU. CPU quota, physical topology, power and thermal state are unknown.
- Timed execution: one shared `control/benchmark.lock`; no builds or other lane
  workloads during the window. One thread uses CPU 0, two use CPUs 0–1,
  and eight use CPUs 0–7.
  A host lock cannot exclude workloads outside this campaign.
- Budget: 6 GiB sampled process-group RSS watchdog; at least 12 GiB free disk.
  There is no address-space cap: Bend reserves a large sparse virtual arena.
- Filesystem cache is warm; every trial launches a fresh process. No privileged
  cache drop or claim about cold storage performance.

## Accepted evidence at the takeover checkpoint

The seven model campaign journals contain 98 accepted subprocess executions:
18 warmups, 24 exploratory A/B samples, 40 fresh finalist samples, 10 independent
replication samples, two unpinned baseline correctness-smoke calls and four
stage-instrumented diagnostic calls. These categories are not interchangeable.
They are not 98 distinct configurations.

The strict numerical suite passed for the real 135M model. The 23 local baseline
files (11 forward, 12 generation) were frozen after that check. They are not a
claimed reproduction of historical upstream bit files. Candidate comparison
against those files is recorded separately.

- D, skip unused prompt logits: first fresh 32-prompt-token, one-thread batch
  cleared its prespecified first-ID clock gate. Independent fresh replication
  did not: median paired saving 7.79%, bootstrap lower bound 4.39%, below 5%.
  First decoded-text arrival improved by a median 3.18% in that replication.
  The batches remain separate. D is unselected.
- E, reuse the zero weight band during loading: lower medians, but the fresh
  uncertainty bound did not clear the gate. E is unselected.
- At that initial 98-process snapshot, B, adaptive depth by matrix shape:
  364 eligible unique thread/depth/shape
  configurations are planned, not yet counted as measured at this checkpoint.
  A further 28 large-head configurations are deferred for resource calibration.
- At that initial snapshot, C, packed-eight kernel: exploratory Q/O observations only. Exact kernel
  checks and fresh family confirmation precede any possible model integration.

## Extended checkpoint: eligible sweep complete

The later `extended-manifest.json` records a separate, independently audited
checkpoint. It does not add kernel, correctness or diagnostic calls to the
original 98 model-journal processes.

- B completed all 364 eligible thread/depth/shape configurations across 13
  synthetic shapes: 2,093 calls, comprising 13 references, 520 warmups and 1,560
  exploratory measurements. The two head-resource pilots add 30 separate calls.
  There are no eligible failures or pending shapes. The 28 large 1.7B-head
  configurations remain deferred, not declared impossible. All full vectors,
  hashes and numerical references passed independent receipt validation.
  `adaptive-settings.csv` contains all 364 settings' descriptive minimum,
  median and maximum clocks/RSS. Nonzero depths have three measured samples;
  depth zero has twelve repeated controls. The CSV is not a ranking or policy.
- The unchanged real-135M baseline completed 20 long-decode processes: four
  untimed smokes, four warmups and twelve characterization samples. Each of the
  128/512-token, one/eight-thread cells produced its full requested count and
  passed the independent F32/eager oracle. These cells ran in separate windows;
  they are not a paired thread comparison. `baseline-long-decode-summary.json`
  retains the raw-journal hashes, ranges and clocks.
- C's fresh synthetic gate/up and down confirmations passed all 16 full-vector
  parity and 96 timing calls (16 warmups, 80 measurements). Setup-subtracted
  five-pair median savings were 28.05% / 12.01% for gate/up and 31.56% / 19.96%
  for down at one/eight threads. The operands have only 13 periodic classes.
  The earlier Q/O eight-thread negative pair remains in the record. These
  results do not measure real checkpoint loading or model inference.
- E's two fixture builds and 22 boundary cases passed, including ten exact
  expected-error controls. Its 28 later diagnostic calls also passed. The
  process-wide reader and list-packer counters have different call populations;
  the retained erratum prevents interpreting their difference as a load-stage
  decomposition. The loader candidate remains unselected.

## Real-model follow-up: two-thread scheduling rejected

The isolated layer-depth-2/head-depth-5 policy preserves attention depth zero
and the original default loader. Its 14 guarded correctness phases contained
98 two-thread model children, covering opt-in and freshly rebuilt default
layer/forward/generation numerics and both sets of 23 local frozen outputs.
Independent acceptance additionally reconstructed all 82 reference files from
pinned unchanged producers, byte-identically, and verified emitted-C routing
and pure fork/join sections. That reconstruction is explicitly post-execution
provenance, not an earlier freeze or a universal numerical proof.

Two separately preregistered batches then tested real 135M greedy generation
for the four-token `Once upon a time` prompt and all 128 requested tokens, on
two threads pinned to CPUs 0–1. Each batch has two untimed smoke calls, two
warmups and ten measured calls forming five randomized pairs. All 28 processes
passed exact token IDs, chosen-logit U32 bits and text. No samples were added.

- Initial total-process-wall paired savings: 1.117%, 2.417%, 3.527%, -0.266%,
  3.614%; median 2.417%, descriptive bootstrap lower bound -0.266%.
- Independent replication: 3.772%, 11.974%, 2.404%, 2.343%, 7.099%; median
  3.772%, descriptive lower bound 2.343%.

Both complete batches fail their separately required 5% lower-bound gate.
They are not pooled. Per-child peak-RSS reductions of about 36.22% (`wait4`
`ru_maxrss`) repeat as a secondary observation; this metric differs from the
sampled aggregate-process-group guard. It does not replace the prespecified
wall-latency primary.
The policy remains unselected. `two-thread-model-result.json` indexes both
summaries, audits, manifests and journals without adding these calls to either
the original 98-process inventory or the synthetic configuration count.

The packed-MLP follow-up remains in staged correctness work. Two native-build
attempts failed before any of their 15 fixture calls, followed by one failed
minimal loader verdict. A bounded five-scalar ownership correction then passed
one loader-only verdict. A third native fixture build later failed on
fixture-local `U32` versus `Nat` depth literals, again before any of its 15
children. Four explicit Nat literals then passed one fixture-only verdict.
Thus the 09:49 UTC checkpoint has three failed native fixture builds, one
failed minimal verdict and two successful checker verdicts. All earlier
failures are retained. Green checker verdicts do not establish native fixture,
real-weight or model correctness.

## Work after the takeover

1. Extended D correctness is complete: exact 128/512-token prompt parity and
   all 23 local frozen baseline outputs passed. A separate strict raw audit
   checked empty stderr and finite selected logits for the four captured
   long-prompt smoke children, plus raw hashes and clean guarded exits.
   Individual smoke returncodes were checked but not stored per child. The
   23-output checker requires successful child exits, but its child stderr is
   not retained separately; both provenance limits are retained. These are
   correctness calls, not extra selection rounds.
2. Serialized B shape batches are complete, with immutable run specifications and
   independently verified receipts. A passing 28-configuration batch requires 161 raw calls:
   one reference, 40 warmups, 120 measured calls, including repeated controls.
3. Unchanged-baseline long decode is complete: 128/512 requested tokens at one/eight threads,
   an independent local F32/eager oracle, one warmup and three samples per cell.
   Actual generation count/EOS is retained. Cells were interleaved with B;
   each cell's timed window was uninterrupted.
4. C's bounded kernel confirmations/family screens and E's attribution diagnostics
   are complete. Their separate evidence categories are recorded above.

## Resume contract

The local control directory holds the frozen manifests and producer scripts;
the results directory holds append-only journals, raw stdout/stderr, summaries,
failures and checkpoint inventories. Reuse the same manifest/output directory
only when all hashes and strict gates revalidate. Never overwrite failed runs,
refresh hashes to conceal source changes, or relabel an unfinished plan as data.

The committed runner instructions are in `../campaign-runner.md`. A portable
control/evidence bundle will be recorded here once the current bounded run is
complete. This checkpoint deliberately does not invent registry IDs, promise a
portable model download, or claim that raw local paths are published artifacts.

See `negative-results.md` for rejected hypotheses and guard failures.

## Reproduce the harness checks

From the repository root, the focused tests need no model or Bend installation:

```sh
python3 -m unittest discover -s reference -p test_latency_report.py
python3 -m unittest discover -s reference -p test_campaign_runner.py
python3 -m unittest discover -s reference -p 'test_campaign_summary*.py'
# These oracle-wrapper checks use the pinned numerical dependencies:
reference/.venv/bin/python -m unittest discover -s reference -p test_reference_guards.py
```

The runner tests create temporary executable/output fixtures, run the real
watchdog/journal/resume paths and verify rejection of corrupt or failed evidence.
Those fixture executions are never included in model or kernel measurement counts.

For full numerical validation, use the repository's existing `make setup`,
`make prep MODEL=smollm2-135m`, and `make check MODEL=smollm2-135m` workflow. Run it
outside any timed campaign window. `scripts/setup.sh` and `reference/models.py`
pin the runtime/dependencies/model; `results/baseline/toolchain-manifest.txt` in
the local evidence bundle records this host's actual versions and compiler.

For a new measurement, follow `../campaign-runner.md`: build before timing,
freeze explicit inputs/oracle/commands/resources in a new manifest, run the
untimed correctness smoke, then invoke the runner. An exact replay needs the
local model, binary and oracle files whose hashes are in that run's manifest.
The compact committed checkpoint is an evidence index; it intentionally omits
weights and bulky raw traces and is not itself a substitute for those files.

This draft is stacked on the fork's diagnostic-parser PR #2. It proposes only
fail-closed reference checks, the measurement mechanism and bounded evidence;
no candidate runtime, proof, API, default branch or release changes are included.

## Compact reviewed receipts

- `manifest.json`: seven-journal takeover inventory and SHA256 receipts, with
  exact counts by evidence type.
- `host-limits.json` and `toolchain-manifest.txt`: observed host and pinned tools.
- `validation.txt`: 49 focused harness test methods passed on the final guards.
- `loader-final-summary.json`, `prefill-final-summary.json`, and
  `prefill-replication-f-summary.json`: raw-output-revalidated summaries. Each
  reports completeness explicitly; the independent prefill batch stays separate.

After the takeover checkpoint, B's first 135M Q/O shape completed 28 unique
configurations with 161 verified calls (one reference, 40 warmups, 120 measured).
That is exploratory kernel evidence and is not added to the seven model-journal
counts. Remaining eligible B configurations at that milestone: 336; deferred: 28.
