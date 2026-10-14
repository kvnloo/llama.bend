# Downstream latency campaign, 2026-10-14

This is Kevin's independent downstream experiment on `kvnloo/llama.bend`.
Renan Vinícius (nuxyel) and the upstream contributors retain credit for the
implementation, proofs and original measurements. No Modular source is copied.
The upstream benchmark numbers are not reproduced by this host.

## Current checkpoint

Execution is active. No performance candidate is selected for integration.
The baseline runtime is unchanged. Prefill and loader source candidates remain
isolated; packed-eight layout is a kernel-only prototype.

- Runtime: Bend 2.0.37, Lean 4.34.0, Clang 19.1.7, pinned Python dependencies.
- Full model: real pinned SmolLM2-135M revision
  `93efa2f097d58c2a74874c7e644dbc9b0cee75a2` only.
- Host: AMD EPYC 9V74 virtual CPUs, affinity 0–8, about 9.7 GiB RAM, no swap or
  exposed GPU. CPU quota, physical topology, power and thermal state are unknown.
- Timed execution: one shared `control/benchmark.lock`; no builds or other lane
  workloads during the window. One thread uses CPU 0, eight use CPUs 0–7.
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
- B, adaptive depth by matrix shape: 364 eligible unique thread/depth/shape
  configurations are planned, not yet counted as measured at this checkpoint.
  A further 28 large-head configurations are deferred for resource calibration.
- C, packed-eight kernel: exploratory Q/O observations only. Exact kernel
  checks and fresh family confirmation precede any possible model integration.

## Ongoing bounded work

1. Extended D correctness is complete: exact 128/512-token prompt parity and
   all 23 local frozen baseline outputs passed. A separate strict raw audit
   checked empty stderr and finite selected logits for the four captured
   long-prompt smoke children, plus raw hashes and clean guarded exits.
   Individual smoke returncodes were checked but not stored per child. The
   23-output checker requires successful child exits, but its child stderr is
   not retained separately; both provenance limits are retained. These are
   correctness calls, not extra selection rounds.
2. Serialized B shape batches, with immutable run specifications and independently
   verified receipts. A passing 28-configuration batch requires 161 raw calls:
   one reference, 40 warmups, 120 measured calls, including repeated controls.
3. Unchanged-baseline long decode: 128/512 requested tokens at one/eight threads,
   an independent local F32/eager oracle, one warmup and three samples per cell.
   Actual generation count/EOS is retained. Cells are interleaved with B;
   a cell's timed window is uninterrupted.
4. C fresh kernel confirmation/family screens, then E attribution diagnostics.

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
