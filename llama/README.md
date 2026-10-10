# bend-ml-llama

Llama-architecture language models in Bend 2: RMSNorm, rotary position embeddings, grouped-query attention with a
key/value cache, and SwiGLU. The sizes of a model come from a config (`Cfg`), and **every weight matrix has its
shape in its type**, computed from that config, so a model of other sizes is the same code with another config. The structure is proved by Bend's checker and again by its kernel proved in Lean. The numbers are tested
against PyTorch: they match `transformers` token for token in the
[llama.bend](https://github.com/nuxyel/llama.bend) repository.

- Bend: **2.0.35** · License: MIT · Depends on `bend-ml-tensor@0.1.2.0` and `bend-ml-tensor-array@0.1.6.0`.
- `bend llama/main.bend --verdict` → `ALL PROOFS CHECK` (no `@unsafe`, no `?TODO`, no axioms).

```python
import Base
import bend-ml-llama@0.1.0.0/main.bend as M

# one position: token tok at position t, with the caches of positions 0..t-1 (one per layer)
def step(+c: M.Cfg, +t: Nat, m: M.Model<c>, cs: List<&2, M.Cache(t)>, +tok: Nat) -> M.Out<c, t>:
  M.forward(c, t, m, cs, tok)
```

## The laws, in plain language

| Law | What it says |
|---|---|
| `head_split` | A head is stored as its two halves, the pairs that RoPE rotates together. Cutting a list at `half` and joining the two parts gives the list back, so splitting a head loses nothing and moves nothing. The config stores `half`, not the head size, so a head (`2·half` numbers) is always even. |
| `gqa_head` | With `g` query heads per key/value head, query head `j·g + i` (for `i < g`) reads key/value head `(j·g + i) / g = j`. That division is how `transformers` maps query heads to key/value heads (`repeat_kv`). This package groups the query heads instead (group `j` reads key/value head `j`), and the law says both choose the same key/value head. The proof goes through `Nat.div` itself: `j·g` is `j` whole blocks of `g`, and `i < g` more numbers do not complete another one. |
| `groups_len`, `cols_len` | The query heads form exactly `kv` groups, and the cache gives exactly `kv` key/value columns. Attention walks the two lists together, so every group meets a key/value head; none is dropped, and none reads a missing head. |
| `rows_len`, `causal` | The cache has the type `Cache(t)`: the number of positions it holds is in its type. At position `t`, attention runs over the cache of `t` earlier positions plus the new one: exactly `t + 1` rows, positions `0..t`. A later position cannot be read because the value does not contain it. |
| `layout_135m`, `layout_360m` | The shapes that the types give to the weights (the embedding, every layer's nine tensors and the final norm) add up to 134,515,008 numbers for SmolLM2-135M and 361,821,120 for SmolLM2-360M, the number of values in each checkpoint. The loader stops on any file whose size differs from the shape in its type, so together the files cover the checkpoint exactly. |
| `smollm2_135m_sizes`, `smollm2_360m_sizes` | The two configs give the sizes of their `config.json`: SmolLM2-135M has 9 query heads of 64 numbers, a hidden size of 576 and keys 192 wide; SmolLM2-360M has 15 heads of 64 numbers, a hidden size of 960 and keys 320 wide. |

`add_zero`, `add_succ`, `add_one`, `add_assoc`, `div_block`, `div_blocks` and `div_rest` are the arithmetic lemmas that
the proofs above use.

What is **not** proved: the numbers. `F32` is not a real number. RMSNorm, RoPE, softmax and silu follow the order
of operations of `transformers`' `modeling_llama.py`, and the tests check the results (logits within 2.5e-5 of
PyTorch, the same greedy tokens). The vectors between layers are lists whose length is not in their type.

## How a model is described

```python
type Cfg is Data:
  Cfg{kv: Nat, g: Nat, half: Nat, f: Nat, layers: Nat, vocab: Nat, eps: F32, theta: F32}
```

`kv` key/value heads, `g` query heads per key/value head, heads of `2·half` numbers, MLP width `f`. From these
fields: `Cfg.heads = kv·g`, `Cfg.dim = heads·2·half`, `Cfg.kvw = kv·2·half`. `smollm2_135m()` and `smollm2_360m()`
are given, and `read_cfg(dir)` reads `<dir>/cfg.txt` (written by llama.bend's `reference/prep.py`, which checks
that the `config.json` describes a model this package runs: a Llama with SwiGLU, no biases, plain RoPE and a tied
LM head).

## API

| Name | What |
|---|---|
| `LW<c>`, `Model<c>` | a layer's weights and the whole model; each matrix is a `bend-ml-tensor-array` `Bands<rows, cols>` with its shape from `c` |
| `Cache(t)`, `Cache.empty`, `Cache.push`, `caches(n)` | the key/value cache of `t` positions |
| `block(c, lw, t, kv, rot, x) -> BR<c, t>` | one layer at position `t`: the output, the cache grown to `t + 1` positions, the weights handed back, and the activations (`Act`) |
| `forward(c, t, m, cs, tok) -> Out<c, t>` | one position of the whole model: logits, the final norm, every layer's `Act`, the caches, the model |
| `BR.then`, `Out.then` | continue with the fields of a result; a loop over positions recurses inside them |
| `argmax`, `AM.i`, `AM.v` | the first index of the largest logit, and that logit |
| `rmsnorm`, `inv_freq`, `rot_at`, `rope`, `swiglu`, `attention` | the pieces of a layer |
| `read_cfg`, `load_model`, `load_lw`, `depth_for` | loading a model written by `prep.py` (`<dir>/w/<name>.bin`, little-endian F32, one file per tensor) |

An `Array` has a single owner, so every product hands its matrix back (`block` returns its `LW`, `forward` its
`Model`). `Act` holds every intermediate of a layer (`n1`, `q`, `k`, `v`, `qr`, `kr`, `attn`, `a`, `n2`, `m`,
`y`). The tests compare them with `transformers`, and a backward pass would read them.

MIT · Renan Vinícius (@nuxyel)
