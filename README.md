# llama.bend

[![CI](https://github.com/nuxyel/llama.bend/actions/workflows/ci.yml/badge.svg?branch=devel)](https://github.com/nuxyel/llama.bend/actions/workflows/ci.yml)
[![Bend 2.0.36](https://img.shields.io/badge/Bend-2.0.36-7b68ae)](https://github.com/bendlang/bend)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

Llama-architecture language models running in [Bend 2](https://github.com/bendlang/bend), with the shapes in the
types. One typed package, [`llama/`](llama/) (the future BendHub package `bend-ml-llama`), runs SmolLM2-135M,
SmolLM2-360M and SmolLM2-1.7B. Only the config changes between them. All three give the same greedy tokens as
PyTorch (`transformers`).

```console
$ ./gen data/smollm2-1.7b "The capital of France is" 32
weights loaded in 72.6s
prompt ids: 504 3575 282 4649 314
  id 7042  logit 1099424379 16.985586  t=1.4s
  id 30  logit 1100828526 19.663784  t=1.8s
  ...
text: The capital of France is Paris.

The capital of the United States is Washington, D.C.

The capital of the United Kingdom is London.

The capital
```

Those are the same 32 tokens that `transformers` generates, computed in Bend on the CPU (prompt tokenized in Bend
too; the logit of each chosen token is printed as its exact F32 bits and its value).

The structure is proved by Bend's checker and again by its kernel proved in Lean (`--verdict`). The numbers are
tested against PyTorch, because `F32` is not a real number and cannot be proved. It follows
[bend-ml](https://github.com/nuxyel/bend-ml), where GPT-2 small runs in Bend token for token.

## What is proved

The laws are in [`llama/main.bend`](llama/main.bend). Each one is explained in plain language in
[`llama/README.md`](llama/README.md#the-laws-in-plain-language). They look like this:

```python
# the key/value cache of t positions: one flat array of keys and one of values per key/value head;
# the number of positions written is in its type
type KV<-t: Nat> is Type:
  KV{ks: List<&1, Array<F32>>, vs: List<&1, Array<F32>>}

# one layer at position t takes the cache of t positions and returns it with t + 1;
# its weights LW<c> have their shapes from the config c (d is the parallel depth of attention)
def block(+c: Cfg, +d: Nat, lw: LW<c>, +t: Nat, kv: KV<t>, +rt: Rot, +x: List<&2, F32>) -> BR<c, t>

# with g = 1 + i + k query heads per key/value head, query head j·g + i reads key/value head j
law gqa_head:
  for +j: Nat
  for +i: Nat
  for +k: Nat
  {Nat.div(Nat.add(Nat.mul(j, 1n+Nat.add(i, k)), i), 1n+Nat.add(i, k)) == j : Nat}
```

In short:

- **Grouped-query attention reads the right heads.** With g query heads per key/value head, query head
  j·g + i reads key/value head (j·g + i) / g = j, the mapping `transformers` uses. This is proved over `Nat.div`
  itself. The query groups and the key/value heads pair up one to one.
- **The cache is causal by its type.** The key/value cache at position t has the type `KV<t>`: t positions
  written. Attention at position t scores exactly t + 1 keys (positions 0..t), one per position written.
- **The cache has room for its positions.** A cache created for `cap` positions holds every position below `cap`
  inside its arrays, so writing one never wraps around onto position 0. (The type does not stop a write at `cap` or
  beyond; the caller sizes the cache.)
- **A head is even by construction.** The config stores half a head. Splitting a head into the two halves RoPE
  rotates together, then joining them, gives the head back.
- **The weights cover the checkpoint.** Every weight matrix has its shape in its type, computed from the config.
  Those shapes add up to the number of values in each checkpoint (134,515,008, 361,821,120 and 1,711,376,384),
  and the loader stops on any file whose size differs from its type.

## What is tested

All against `transformers` 5.19 in F32 with eager attention, by the scripts in [`reference/`](reference/):

| | SmolLM2-135M | SmolLM2-360M | SmolLM2-1.7B |
|---|---|---|---|
| config | 3 key/value heads × 3 query heads, 30 layers | 5 × 3, 32 layers | 32 × 1, 24 layers, RoPE base 130,000 |
| logits, worst error relative to the largest logit (11 prompts) | 2.1e-5 | 4.8e-5 | 8.3e-5 |
| argmax at each of the 88 positions | the same | the same | the same |
| 32 greedy tokens and decoded text (12 prompts) | identical | identical | identical |

The tokenizer is written in Bend too (SmolLM2's digit split, the GPT-2 regex and 48,900 merges). It gives the
same ids as `tokenizers` on 68 fixed texts and 600 random ones, and decodes like it on 1,500 random id
sequences, most of them invalid UTF-8.

## Running it

Linux (or WSL) and Python 3.14, the version the pinned dependencies were tested with. About 2 GB of disk:
1.2 GB for the Python venv with PyTorch, 0.9 GB for SmolLM2-135M's files.

```bash
make setup   # installs Bend 2.0.36 (SHA256-checked) and Lean 4.34.0 if missing, and the Python venv of reference/
make prep    # downloads SmolLM2-135M at a pinned revision into data/smollm2-135m/ and checks it
make check   # the laws (--verdict), then every test above against tokenizers and transformers (~4 min)
```

Then, to generate text (greedy, 32 tokens):

```bash
bend demo/gen.bend -o gen && ./gen data/smollm2-135m "The capital of France is" 32
```

The other models are one variable away: `make prep check MODEL=smollm2-360m`, or `MODEL=smollm2-1.7b`.
`reference/models.py` pins each one to a revision.

## Limits

- **Not benchmarked yet.** Informally, on my machine (Core Ultra 7 155H, 16 threads), SmolLM2-135M loads in
  ~4 s and runs a forward pass in ~0.12 s. The 1.7B loads in 73-83 s and runs a pass in ~0.8 s, with a peak of
  6.9 GB. A real benchmark against PyTorch is the next milestone.
- CPU only; base models, greedy decoding, no chat template. The vectors between layers are lists whose length is
  not in their type, and the numerics are tested, never proved.

## Where things are

| Path | What |
|---|---|
| [`llama/`](llama/) | the package: config, layers, model, loader and laws |
| [`demo/`](demo/) | `gen.bend` (generation), `forward.bend` and `layer_test.bend` (what the tests read), the tokenizer |
| [`reference/`](reference/) | download and preparation, the `transformers` and `tokenizers` oracles, the tests |
| [`NOTES.md`](NOTES.md) | every finding, measurement and decision, by date |
| [`docs/plan.md`](docs/plan.md) | the milestones and their pass criteria |

| Milestone | Status |
|---|---|
| 1-3. Prep, tokenizer in Bend, one layer | done (2026-10-09) |
| 4-5. The whole forward pass, generation | done (2026-10-10) |
| 6-7. The generic typed package, its laws | done (2026-10-10) |
| 8. Other sizes: SmolLM2-360M, SmolLM2-1.7B | done (2026-10-10) |
| 9. Benchmark against PyTorch; `bend-ml-llama` on BendHub | next |

MIT · Renan Vinícius (@nuxyel)
