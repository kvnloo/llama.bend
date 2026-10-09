# llama.bend

SmolLM2-135M, a Llama-architecture language model, running in Bend 2: token for token against PyTorch,
with the shapes in the types and the structural guarantees (grouped-query head mapping, even RoPE head size,
causal KV cache, weight layout) as laws checked by Bend's kernel. Step 1 of the plans after bend-ml v1.2
("llama.bend" from Victor Taelin's wishlist, https://x.com/VictorTaelin/status/2104949395770712276). The
typed layers will become the BendHub package `bend-ml-llama`. **Private for now.**

## Where things are

- `docs/plan.md`: the milestones, each with its pass criterion, and the laws to prove.
- `NOTES.md`: every finding, measurement and decision, by date. Read it before changing anything.
- `reference/`: the Python side: `prep.py` (download at a pinned revision, bf16 → F32, tokenizer tables, and
  `--check`), and later the PyTorch/`transformers` oracle and the tests.
- `data/` (not committed): what `make prep` downloads and writes.
- bend-ml (`~/Repos/bend2-ml`, https://github.com/nuxyel/bend-ml) is the parent project: its packages are
  imported from BendHub at pinned versions (`bend-ml-tensor-array@0.1.6.0`, `bend-ml-bpe-tokenizer@0.1.2.0`),
  and its GPT-2 demo (`demos/gpt2/`) is the pattern for the loader, the tokenizer and the oracle tests. Its
  `NOTES.md` holds the performance experiments (exp. 1-14) this project builds on.
- Private planning notes live outside this repository.

## Branches and versions

Work on `devel`; `main` only receives releases through a pull request `devel` → `main`. Version numbers follow
bend-ml's `docs/VERSIONING.md` (a major only for a change in what the project is).

## About Renan

Works in AI engineering (agents, NLP) and writes Python and TypeScript; studies ML formally. No background
in type theory: explain each proof in plain language, in a comment and in chat. Machine: Core Ultra 7
155H (22 threads; P-cores are CPUs 0-11, pin one-thread timings with `taskset -c 3`), 32 GB RAM, RTX 4050 6 GB
(for these memory-bound kernels the GPU is slower than the CPU, bend-ml NOTES exp. 10).

## Rules

1. Bend is pinned to **2.0.35** (and Lean 4.34.0 for `--verdict`). Bend 2 changes fast and differs from
   Bend 1: before writing code, read `bend guide`, `bend base <name>` and the Base source
   (`~/.bend/bend2/base.bend`). A version bump goes in `NOTES.md`.
2. Run the checker on every change; it takes a fraction of a second.
3. Every proof is real: no `@unsafe`, no `?TODO`, no axioms. A law that cannot be proved yet is
   recorded in `NOTES.md` as debt and left out of the README.
4. Prove structure (shapes, sizes, composition); test numbers. `F32` is not a real number, so numerics
   are validated against PyTorch / `transformers` / `tokenizers` by the scripts in `reference/`.
5. Every proved law appears in the package README in plain language.
6. Publishing to BendHub is public and permanent: four-number versions (`0.1.0.0`), names of 12+
   characters, a `LICENSE` next to the entry file. Ask Renan before publishing.
7. Benchmarks are honest: hardware and versions written down, numbers measured on an idle machine,
   alternated runs, one thread pinned to a P-core.
8. Everything in the repository is in English. Commits are small, with clear messages and no
   attribution lines (no `Co-Authored-By`, no "Generated with").
9. Public text (READMEs, `NOTES.md`, `docs/`) is written in Renan's first person singular: "I measured",
   never "we"/"our". Where a name is needed: "Renan Vinícius (@nuxyel)". This file stays in the third person.

## Bend 2.0.35 pitfalls (paid for in bend-ml)

- `match` only inspects a parameter or a pattern variable; a computed value (including a destructuring
  `let` of a call) needs its own def or a parameter computed by the caller. Matching several parameters at
  once (`match b small:`) is the way to branch on a computed `Bool` and a value together.
- No mutual recursion. Termination reads arguments left to right: each must be passed unchanged until one
  shrinks. Erased (`-x`) parameters are skipped, so put a changing size after the shrinking argument or
  make it erased.
- Variables are affine; `+x` reuses a `Data` value. A parameter used in a type counts as a use.
- `%e : P` rewrites: `_` marks where the right-hand side of `e` sits in the current goal, and the new goal
  has the left-hand side there; use `Equal.sym` to go the other way. `Nat.sub(1n+p, 1n)` does not reduce to
  `p` in the checker: pass `p` itself.
- Parallel lets (`a b = f(..) g(..)`) silently become sequential when the def's **last parameter is
  erased** (put a runtime parameter after it; fixed upstream by bendlang/bend#1377, not in a release yet),
  and suffer from non-tail recursive helpers in the fork tree (use tail loops with an accumulator). Check
  scaling with `--threads 1` against `--threads 16`, and read the emitted C (`bend X.bend -o x.c`).
- `IO.fork` gives concurrency, not parallelism (by design, bendlang/bend#1415). Use parallel lets in pure
  code for several cores.
- Proofs follow the affine rules too: a lemma about values that hold an `Array` cannot mention them twice at
  run-time positions; pass the expression in an erased argument (`-s`) and the arrays once. A lemma
  quantified over a single-constructor value (`g: G`) can match it, which avoids computing the call that
  produced it.
- `Array` is a plain ADT (`ALeaf`/`ANode`): matching an owned array hands out its halves in O(1) and is
  safe; `Array.clone` is O(n); `Array.fork`/`Array.join` are `@unsafe`. Indexes wrap around, so never write
  past a capacity.
- `File.read_at` returns a `List<U32>` with one cell per byte; producing and walking it is the cost of
  loading (bend-ml exp. 13), and erasing a large unread list costs more than walking it.
- An `import ... as M` does not re-export, so a publishable package is one file. Relative imports
  (`import ./x.bend as X`) work for local copies.

## Agent skills

### Issue tracker

Issues live in GitHub Issues of `nuxyel/llama.bend`, managed with the `gh` CLI. See `docs/agents/issue-tracker.md`.

### Triage labels

The five default labels (`needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`). See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: one `CONTEXT.md` and `docs/adr/` at the repo root, alongside `NOTES.md` and `docs/plan.md`. See `docs/agents/domain.md`.
