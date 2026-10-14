# Kevin's downstream latency campaign: closeout

No runtime candidate is selected. B's two-thread policy, D's prefill skipping
and E's loader reuse failed their required latency gates. C has favorable
synthetic packed-kernel evidence, but zero integrated model executions.

The campaign is closed at completed evidence. No host execution is pending.
The next packed-fixture triage driver has not run: its final independent source
review was stopped by a platform content-risk flag. That review/action must not
be retried, rephrased or routed through another agent/tool. Earlier completed
audits remain valid.

## Exact evidence categories

| Category | Executed count | Qualification |
|---|---:|---|
| B eligible kernel sweep | 364 unique settings / 13 shapes / 2,093 calls | 13 references + 520 warmups + 1,560 exploratory measurements; 28 largest-head settings deferred |
| B head resource pilots | 30 calls | 2 references + 28 pilot samples, separate from the sweep |
| B final edge correctness | 63 calls | Separately accepted exact-output gate |
| B preliminary history | 63 older edge calls; 52 saved calibration samples | Four further calibration references are inferred from protocol, giving 56 inferred calibration calls; not final sweep evidence |
| C kernel timing | 240 calls | 48 warmups + 72 exploratory measurements + 120 fresh confirmation measurements |
| C later strict parity | 50 calls | Three older parity stdout artifacts remain separate; exact earlier call cardinality is not asserted |
| Native model timed windows | 132 processes | 106 measurements + 26 warmups, including exploratory/final/replication/long-decode characterization in their original scopes |
| Untimed model smokes | 10 processes | Separate from timing samples |
| Other model correctness | 202 full-model + 12 layer-only processes | Baseline 75, D 41, B 98; includes repeated executed correctness checks, not repeated read-only audits |
| Model diagnostics | 12 processes | Initial stage 4 + D instrumented smoke 4 + E attribution 4; not primary timing evidence |
| E old loader fixtures | 22 cases / 2 builds | 12 positive selected-boundary checks + 10 exact expected errors |
| E stage microprobes | 24 calls | 6 warmups + 18 diagnostic measurements; E's 4 model-attribution calls are already in model diagnostics |
| Packed integration compiler work | 14 invocations | 4 failed native builds; 3 verdicts (1 failed / 2 passed); 7 C emissions (2 type errors / 5 successes) |
| Packed runtime/model coverage | 0 | Zero accepted fixture binaries, 0/15 native cases, zero 90-matrix traversals, integrated C builds, model or timing calls |

The documented native model/layer scope totals 368 processes. This is not a count
of all host processes: tokenizer/unit fixtures, Python oracle producers,
compiler phases and read-only audits are excluded. Baseline legacy log counts
have weaker per-child provenance than later raw receipts; D's frozen-bit child
stderr was not retained individually. `final-ledger.json` keeps those limits explicit.

Compiler counts are phase-specific, not an invented exhaustive grand total:
B has 13 build/check phases; C has 37 documented timing/fixture builds plus older
unenumerated Q/O parity builds; E's v2 diagnostic preparation has 1 checker,
1 C emission, 1 instrumentation-generation step and 2 native compiles. These phases
are never added to their nested model-child counts. F's three post-execution
PyTorch producers recreated 82 reference files byte-identically and are separate
validation work.

## Decisions and historical published checkpoint

B's initial five-pair wall savings have median 2.417%, lower bound −0.266%; its
independent batch has median 3.772%, lower bound 2.343%. Both fail 5%. All 28 smoke/warmup/
measured processes preserved 128 token IDs, chosen-logit bits and text. The
roughly 36.22% per-child `wait4` RSS reduction remains secondary. D and E's earlier
negative decisions remain unchanged. No batches or metrics were substituted.

Draft PR #4: https://github.com/kvnloo/llama.bend/pull/4

- Historical two-thread evidence head, verified at 09:53:52 UTC: `fd8808b36b39b7a7d22c72c14ecb136ed770f318`
- Reviewed local counterpart: `300bf0b8ab431df4e0bcd5d4854f421a11fb92df`
- Exact matching tree: `873c96a454938634ddff824a528ab47a0e243208`
- PR #2 base unchanged: `3a5f626dd9f2a6b3bcee51292b10c4e203cb74de`
- Draft/open/unmerged; exact-head CI has no reported runs/checks/statuses as of
 09:53:52 UTC. This is not a CI pass.

This final inventory is additive to the published frozen checkpoints.
No weights, binaries, bulky traces or unreviewed compiler-triage source are
proposed for publication.

## Historical local replay

The closing replay used `/workspace/scratch/df37e4e44cce/llama-campaign` on the
original campaign host:

```sh
source control/env.sh
flock control/benchmark.lock lane-a/reference/.venv/bin/python \
  lane-a/scripts/campaign_summary.py results/two-thread-bands-initial
flock control/benchmark.lock lane-a/reference/.venv/bin/python \
  lane-a/scripts/campaign_summary.py results/two-thread-bands-replication-f
(cd lane-b && ../lane-a/reference/.venv/bin/python bench/adaptive_campaign.py plan)
```

Closing read-only replay rehashed 882 distinct frozen model-trial inputs and
recreated both complete summaries exactly; zero model/compiler children ran.
Receipt: `final-runnable-replay.json`. Original raw results were not changed.

These are historical commands, not a portable replay bundle. During post-closeout
verification on 2026-10-14, the original campaign directory was absent from the
verification workspace. The control/results trees, raw journals/stdout/stderr and
pinned model/binary/oracle inputs are not included in this public checkpoint;
their continued availability on the original host has not been verified. The
receipt records the earlier successful replay, not one performed by this
verification.

The existing runner can resume only the same frozen manifest/output after a
named quiet window; complete revalidated trials are skipped. Do not replace
inputs, overwrite attempts, or start packed-model trials. Public compact
receipts do not replace the local pinned weights/binaries/oracles. The new
packed-trial preparer/tests and C numerical helper are unexecuted/unaccepted
source-only work and are excluded from these commands.

## Next three proposed experiments

1. After legitimate resolution of the blocked review, isolate full-fixture
   emission by the capped 12-call/20-minute reachability plan, preserving all 15
   cases. Goal: a minimal compiler repro or compiling complete split coverage.
2. If all 15 native fixtures, both 90-matrix full-word traversals, numerical,
   23 frozen-output and complete 128-token bit gates pass, run the predeclared one-thread packed
   MLP 128-token wall trial and independent replication.
3. Separately preregister a two-thread head-only policy (layer 0 / head 5 / attention 0)
   to distinguish head-layout benefit from layer scheduling overhead. Require
   the same correctness and independent total-wall gates. Do not retune or pool
   the failed layer-2/head-5 experiment.

These are proposals, not scheduled or running work. Renan Vinícius (nuxyel) and
upstream contributors retain credit for the implementation/proofs; Modular
contributors for scheduling ideas; Kevin for the downstream campaign.
