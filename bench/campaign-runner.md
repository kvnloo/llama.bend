# Resumable downstream CPU trials

Kevin's downstream campaign uses `scripts/campaign_runner.py` to extend PR #2's stdout diagnostic with process
execution, resource guards, strict correctness gates and durable evidence. The
runner does not build binaries. A plan is not a benchmark result.

## Evidence contract

- Freeze `schema: 1`, `seed`, `warmups`, `rounds`, `lock`, `budget`, and `cases` in
  a JSON manifest. Each case has one or more named `variants`, each with an argv
  array, absolute cwd, optional environment additions, and a `validation` rule.
- Add executable, runner, model, config, tokenizer and oracle files to `inputs`
  as `{ "path": "...", "sha256": "..." }` objects. These hashes are checked before
  trials and on resume. Record source revisions, compiler versions/flags and host
  limitations in manifest provenance. A manifest without these inputs is only a
  harness fixture, not reproducible performance evidence.
- `generation` validation requires exact prompt and greedy generated IDs, final
  text and either exact F32 logit bits or a numerical oracle with explicit
  tolerance. Child exit must be zero, stderr empty, all timestamps/logits finite,
  output complete, and no tokens may follow EOS. Native `stdout_sha256` validation
  supports deterministic kernel outputs whose full numerical correctness has
  already been checked separately. Never treat a digest-only check as a new
  numerical oracle.
- Run one untimed smoke first. Use one warmup, three exploratory screening rounds,
  then at least five fresh paired finalist rounds. Seeded random order keeps every
  case's variants together within a round. Duplicate case/variant identities are removed;
  conflicting duplicate identities fail. Deduplicate equivalent shape/thread/depth
  configurations when creating the manifest. The default cap is 420 unique configurations.
- All workers share the same absolute flock path. No compiler or unrelated lane
  workload may run during the timed window. The lock is held throughout one
  manifest including resume checks. It cannot make a shared host globally idle.

Example invocation:

    python3 scripts/campaign_runner.py manifest.json --out /absolute/results/run

`--plan-only` emits `planned_not_executed` without running commands. Reuse exactly
the same manifest and output directory to resume. A failed or interrupted trial
gets a new attempt ID and raw files on retry; passing attempts are skipped only
after their hashes, exit status and correctness are revalidated. A changed
manifest or corrupt journal fails closed; use a new directory and retain the old
one. The JSONL journal is append-only and fsynced. Each start and finish identifies
its case, variant, phase, round and attempt. Raw stdout/stderr are never overwritten.

## Resource and clock semantics

`budget` contains `timeout_s`, `memory_mb`, `min_free_mb`, and `max_output_mb`.
Only the runner's child process group is terminated on timeout, output overflow,
or sampled RSS overflow. RSS is polled every 20 ms. This is a watchdog, not a hard
kernel memory limit; the same watchdog configuration applies to both arms. Bend
reserves a sparse virtual arena, so an address-space ulimit is inappropriate.
`wait4` provides child maximum RSS, separately from sampled aggregate group RSS.

`wall_ms` is process wall time; `process_observed_ttft_upper_bound_ms` is the
parent's arrival time of a complete first ID line. It includes launch, tokenizer
file IO, model load, tokenization and prompt work, plus any stdout buffering and
scheduling delay. It is not a resident TTFT clock or first decoded-text delivery.
Use a documented line-buffering policy identically in all arms and retain the
upper-bound label. The original generation `t=` clock starts after load and
tokenization, has 1 ms resolution, and includes printing overhead; its first
value must never be renamed end-to-end TTFT. Decoded text arrives only at the end
of this demo. Stage instrumentation belongs to a separate opt-in binary and must
not silently change default output or add per-token stage logs to final trials.

This host's virtual CPUs, compiler and memory differ from Renan's reference
machine. Measurements are local observations, not reproductions of his numbers.

Credit: Renan Vinícius (nuxyel) for the implementation, proofs and original
measurements; Modular contributors for scheduling ideas; Kevin for this
independent downstream experiment. No Modular source is copied.

## Comparing fresh finalists

`scripts/campaign_summary.py RESULTS` revalidates raw outputs and their saved
metrics before reporting medians, ranges and within-round paired percentage
savings. The displayed 95% interval is a seeded bootstrap of the paired median;
five pairs still give limited precision. A screening result remains exploratory
regardless of sample count. A confirmatory manifest must explicitly identify its
purpose, a full lowercase 64-hex SHA-256 of the frozen selection manifest and a
prespecified primary metric. The hash format check is not independent proof
that the selection manifest existed or was reviewed. Only
that metric can satisfy the 5% gate after five or more fresh pairs and completion
of every prespecified trial, including warmups. Incomplete journals retain their
missing-trial counts and cannot select a winner. Favorable
secondary metrics do not select a winner. RSS and correctness constraints remain
separate. A confirmation on one small host is a bounded result, not a universal
speed claim or permission for upstream publication.
