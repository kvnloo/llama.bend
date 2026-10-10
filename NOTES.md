# Notes

What I find, measure and decide, by date. The plan is in `docs/plan.md`.

## Pinned versions

- Bend **2.0.35**, Lean 4.34.0 (installed by bend-ml's `scripts/setup.sh`), as in bend-ml v1.2.1.
- bend-ml packages from BendHub: `bend-ml-tensor-array@0.1.6.0`, `bend-ml-bpe-tokenizer@0.1.2.0`.
- Model: `HuggingFaceTB/SmolLM2-135M` at revision `93efa2f097d58c2a74874c7e644dbc9b0cee75a2`.
- Python side: versions in `reference/requirements.txt`.

## 2026-10-09: start, and milestone 1 (prep)

### Decisions

- Its own repository (`nuxyel/llama.bend`), private until it works; the typed layers become the BendHub package
  `bend-ml-llama`. My 2026-10-05 notes put the demo inside bend-ml; a separate repo keeps bend-ml's checks
  fast and lets this one move on its own.
- Products with `Bands` + `Bands.matvec_l` (`bend-ml-tensor-array@0.1.6.0`), not `Mat.matmul_nt` as those
  notes said: since 0.1.6.0 one band costs about what `matmul_nt` costs, and more bands run in parallel.
- Oracle: `transformers` and `tokenizers` on the CPU.

### The model, checked against `config.json` at the pinned revision

| Field | Value |
|---|---|
| architecture | `LlamaForCausalLM` |
| hidden_size | 576 |
| heads / key-value heads | 9 / 3: head size 64, 3 query heads per key-value head, key/value width 192 |
| layers | 30 |
| intermediate_size | 1536 (SwiGLU, `silu`) |
| rms_norm_eps | 1e-5; no bias anywhere |
| rope_theta | 100000, `rope_interleaved = false` (rotate-half) |
| vocab_size | 49152, tied embedding (LM head = embedding) |
| dtype | bfloat16, `model.safetensors` 269,060,552 bytes |

272 tensors, **134,515,008 numbers** (embedding 28,311,552; 3,540,096 per layer; final norm 576).

### Tokenizer facts

- Pre-tokenizer: `Sequence[Digits(individual_digits = true), ByteLevel(use_regex = true)]`. `Digits` uses
  Rust's `char::is_numeric` (`huggingface/tokenizers`, `tk-encode/src/pre_tokenizers/digits.rs`), that is
  Unicode `\p{N}`, which is already class 1 of bend-ml's `demos/gpt2/unicode.bend` table. No new class needed.
- 17 special tokens (ids 0-16), then the byte tokens, then one id per merge.
- **21 bytes have no token**: 0xC0, 0xC1, 0xF5-0xFF (never in valid UTF-8) and the control bytes 4, 6, 19, 20,
  22, 29. So the byte tokens are ids 17-251 (235 of them), and 17 + 235 + 48,900 merges = 49,152. Every merge k
  gives id **252 + k** (all 48,900), no merge rebuilds an earlier token and none is missing from the vocabulary.
- With no `unk_token` and no `byte_fallback`, `tokenizers` drops a byte that has no token, before the merges:
  "a\x04b" gives [a, b]. `reference/prep.py` does the same; the Bend tokenizer will have to as well.
- A special token written as text (`<|endoftext|>`) is matched by `tokenizers` before pre-tokenization; out of
  scope for the demo (prompts are plain text).

### Milestone 1 result

`make prep && make check` (`reference/prep.py`):

- `data/w/<name>.bin`, one little-endian F32 file per tensor, row = output: all 272 equal to the bf16 bits
  shifted left by 16, bit for bit.
- `data/merges_num.txt` (48,900 rules in bend-ml's numbering) and `data/ids.txt` (the model id of each symbol,
  -1 for the 21 bytes): applying the rules in order to the pieces of my pre-tokenizer (each `\p{N}` character
  alone, then the GPT-2 regex) gives the same pieces and the same ids as `tokenizers` on 10 texts: ASCII,
  Portuguese, prices, dates and long numbers, Arabic and Devanagari digits, fractions, Roman numerals,
  superscripts, circled and full-width digits, emoji, CJK, Korean, control bytes, Python code.

## 2026-10-09: milestone 2 (the tokenizer in Bend)

`demo/tok.bend` started as a copy of bend-ml's GPT-2 tokenizer (`demos/gpt2/tok.bend`) and changed in four
steps, each measured with `reference/test_tok.py` (66 fixed texts, 2 more added with the last change, 3
malformed merge tables, 30 random texts):

| Change | Failures |
|---|---|
| bend-ml's GPT-2 tokenizer as is | 89 of 99 |
| merge k is id 252 + k (GPT-2: 256 + k) | 7 |
| every `\p{N}` character is its own piece | 3 |
| the 21 bytes without a token are dropped before the merges | 1 |
| 0x1C-0x1F are not whitespace | **0** |

- **Digits.** SmolLM2 runs `Digits(individual_digits)` before the GPT-2 regex. In the tag scheme a digit is
  class 1, so a piece of class 1 is now exactly one character (its first byte and the continuation bytes,
  tag bit 2048). Two consequences of the split for the regex: a space is never attached to a digit (no
  " ?\p{N}+"), and a whitespace run that ends right before a digit is taken whole, because the regex sees it at
  the end of its segment (`"x   1"` gives "x", "   ", "1"; `"x   y"` gives "x", "  ", " y"). Many ASCII number
  cases passed even before this change: the vocabulary has no multi-digit tokens and no " 1"-style tokens, so
  the merges produced the same ids anyway; the difference showed with non-ASCII digits and spaces before digits.
- **Bytes without a token.** `perm.txt` gives 0 for them and `known_bytes` drops them from each piece before
  `BPE.encode`; the bytes around them can then merge (`"!!\x04!!"` gives the one token "!!!!").
- **0x1C-0x1F.** bend-ml's `class_ascii` counted them as whitespace (Python's `str.isspace` does); `tokenizers`
  does not (they are not Unicode White_Space), so `"...\x1d..."` is one piece there. Whitespace is now the space
  and 9-13 only. bend-ml's GPT-2 tokenizer is right to keep them as whitespace: I checked that tiktoken does
  (`"...\x1d..."`, `"a\x1cb"`, `"x \x1f y"` give the same ids in bend-ml and tiktoken). The two libraries
  differ here, so each tokenizer follows its own oracle.
- **Fuzz.** The random texts now also draw ASCII digits, Arabic-Indic and Devanagari digits, full-width digits,
  number forms, circled numbers and control bytes 0-31. Seeds 7, 11, 23, 47 with 150 random texts each: 0
  failures (221 cases per seed).
- Time per text (one Bend run, compiled, the 48,900 rules applied in order per piece): median 0.15 s, the longest
  0.58 s ("word " × 120), on this machine with a load average of ~1-2.

## 2026-10-09: milestone 3 (one layer in Bend)

`demo/layer.bend` is one SmolLM2 layer: RMSNorm, the q/k/v products, RoPE, grouped-query attention over a
growing key/value cache, the output product, the residual, the second RMSNorm, the SwiGLU MLP and the output.
The products use `Bands` + `matvec_l` from `bend-ml-tensor-array@0.1.6.0`; the weights are loaded with bend-ml's
block reader (`demo/load.bend`). `demo/layer_test.bend` runs it position by position and prints every
intermediate as exact F32 bits; `reference/test_layer.py` compares them with `transformers` 5.19 in F32 with
eager attention (`reference/layer_ref.py`: forward hooks on the layer, plus q and k after RoPE from
transformers' own `apply_rotary_pos_emb`).

I wrote it one step at a time and ran the test after each; every step passed the first time it compiled.
Worst `max|Δ| / max|y|` per position, over the three cases (layer 0 and layer 15 on the hidden states of
"The capital of France is the city of", 8 positions; layer 0 on 40 positions of Gaussian noise):

| Intermediate | Worst | Intermediate | Worst |
|---|---|---|---|
| n1 (RMSNorm) | 8.4e-7 | attn (before Wo) | 1.8e-6 |
| q / k / v | 1.3e-6 / 2.0e-6 / 1.7e-6 | a (Wo) | 1.1e-6 |
| q, k after RoPE | 1.3e-6 / 2.0e-6 | n2 | 1.3e-6 |
| | | m (MLP) | 2.2e-6 |
| | | **y (output)** | **2.2e-6** |

The pass criterion was 1e-5. Notes:

- **Order of operations.** I followed `modeling_llama.py`: RMSNorm multiplies x by `1 / sqrt(mean + eps)` first
  and by the gain after; RoPE is `h1*cos + (-h2)*sin` and `h2*cos + h1*sin`; the attention scale is 0.125 (exact);
  silu is `x / (1 + exp(-x))`, the form of ATen's CPU kernel. The remaining differences are the order of the
  sums inside the products (sequential in Bend, blocked in PyTorch's BLAS).
- **RoPE's `inv_freq`** computed in Bend with `1 / F32.pow(100000, 2i / 64)` is **bit for bit** equal to PyTorch's
  for all 32 values, so the fallback I had planned (reading torch's table from a file) is not needed. `F32.cos`
  and `F32.sin` at positions up to 39 add nothing visible (q and k after RoPE have the same worst error as before).
- **Layer 15 has outliers**: values up to ~19,000 in its input and output (the "massive activations" of Llama
  models), against ~1 in layer 0. Relative to each vector's largest value the error stays at 2e-7 there.
- **Threads.** The output bits are the same with `--threads 1` and `--threads 16` (bands do not change the order
  of any sum). Not a benchmark yet: one run of 40 positions takes ~0.37 s on both, weights load included.
- **Bend notes.** A def must come before the defs that call it, and there is no mutual recursion: the test loop
  carries the previous position's result (as bend-ml's GPT-2 `run` does) instead of two functions calling each
  other. The layer is a chain of small defs (`b1` … `b13`) because each product hands its bands back and `match`
  only reads parameters, the same pattern as bend-ml's `k1` … `k6b`. The plan for milestone 6 (typed heads, a
  cache indexed by position) should shorten it.

## 2026-10-10: the roadmap after milestone 3

### Decisions

- **The goal grows**: llama.bend should be the reference Llama implementation in Bend, generic over the Llama
  family, not a SmolLM2-135M demo. The test of that is a second model with other sizes (SmolLM2-360M) running
  with only the config changed (milestone 8).
- **First make it work, then the API.** Milestones 4 (full forward) and 5 (generation) go on the current code,
  sizes fixed and the `b1` … `b13` chain, to settle the numerical risk of 30 layers first. Then the generic typed
  API (milestone 6), checked bit for bit against milestone 5, and the laws proved on that API (milestone 7), so
  they are not proved twice.
- **A base for a training framework later, not designed now.** I have not decided what the larger framework
  after bend-ml will be. What llama.bend leaves for it: each block returns an activations record (today's `Tr`),
  and `docs/bendtorch.md` (milestone 11) lists what each op's backward pass needs from it.
- The new milestones and their pass criteria are in `docs/plan.md`.

## 2026-10-10: milestone 4 (the whole forward pass)

`demo/forward.bend` runs the embedding row, the 30 layers of `demo/layer.bend` (one key/value cache each), the
final RMSNorm and the tied LM head, position by position, and prints every layer's output, the final norm and
the 49,152 logits as exact F32 bits. `reference/forward_ref.py` takes the same values from `transformers` 5.19
(F32, eager attention; forward hooks on the 30 decoder layers and on the final norm), and
`reference/test_forward.py` compares them on the 11 prompts of bend-ml's GPT-2 test (3 to 13 positions each,
88 in all). The loader for one layer moved from the layer test into `demo/load.bend` (`load_lw`).

- **Logits: worst `max|Δ| / max|y|` per position 2.5e-5** (`"The capital of France is"` and the quick-brown-fox
  prompt), against the pass criterion of 2e-4; **the argmax is the same at all 88 positions.** It passed the
  first time it compiled; the only error the checker gave was the order of a printing helper's arguments
  (termination reads them left to right).
- The worst position is position 0 in every prompt. I have not looked into why.
- **The error does not grow with depth.** Worst relative error of each layer's output over a prompt's positions:
  ~1e-6 for layers 0-8, a peak of 1e-5 to 5e-5 around layers 9-11, back to ~1e-6 to 1e-5 by layer 20, and 1e-5 to
  2e-5 at layers 28-29. The final norm is within 1.5e-5. The 30 layers do not pile up error; whatever happens
  around layer 10 fades afterwards. So the drift risk listed in the plan did not show up.
- **Threads:** the output is the same bit for bit with `--threads 1` and `--threads 16` (11 positions, 8 MB of
  printed bits each). A first comparison said "identical" because both files were empty: I had called
  `/usr/bin/time`, which this machine does not have, so the program never ran. I check sizes before `cmp` now.
- Time per run, not a benchmark (load average ~2): 4.5-5.8 s for 3-13 positions, most of it loading 134.5 M
  numbers and turning 49,152 logits per position into text. One run of 3 positions: 7.7 s with 1 thread, 5.2 s
  with 16.

