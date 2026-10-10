# llama.bend

Llama-architecture language models (SmolLM2-135M, -360M and -1.7B) running in
[Bend 2](https://github.com/bendlang/bend), with the shapes in the types: one typed package,
[`llama/`](llama/) (the future BendHub package `bend-ml-llama`), and a config per model. Work in progress,
private for now.

It follows [bend-ml](https://github.com/nuxyel/bend-ml), where GPT-2 small runs in Bend and gives the same
tokens as PyTorch. Here the bar is the same (token for token against `transformers`, greedy), and the new
layers come with laws checked by Bend's kernel: every query head of grouped-query attention reads an existing
key/value head, the rotary head size is even by construction, the key/value cache cannot be read past the
current position, and the weight layout covers the file exactly.

| Milestone | Pass criterion | Status |
|---|---|---|
| 1. Prep | weights bit for bit, sizes, tokenizer tables equivalent to `tokenizers` | done (2026-10-09) |
| 2. Tokenizer in Bend | same ids as `tokenizers` | done (2026-10-09): 68 texts + 600 random ones |
| 3. One layer | output within 1e-5 of PyTorch | done (2026-10-09): every intermediate within 2.2e-6 |
| 4. Full forward | logits within ~2e-4, same argmax on 10+ prompts | done (2026-10-10): within 2.5e-5, same argmax at all 88 positions of 11 prompts |
| 5. Generation | the same 32 greedy tokens as `transformers` | done (2026-10-10): the same ids and text on 12 prompts |
| 6. Generic typed API | a `Config` in the types; bit for bit equal to milestone 5 | done (2026-10-10): [`llama/`](llama/), all outputs identical |
| 7. Laws | the laws on that API, `--verdict` | done (2026-10-10): [the laws](llama/README.md#the-laws-in-plain-language) |
| 8. Generality | SmolLM2-360M with only the config changed | done (2026-10-10): 360M and 1.7B, the same tokens |
| 9. Package and benchmark | `bend-ml-llama` on BendHub; time per token against PyTorch | |

The plan is in [`docs/plan.md`](docs/plan.md) and what I find along the way in [`NOTES.md`](NOTES.md).

```bash
make setup   # needs Bend 2.0.35 (bend-ml's scripts/setup.sh installs it) and Python 3.12+
make prep    # downloads SmolLM2-135M at a pinned revision into data/smollm2-135m/
make laws    # the package's proofs: bend, then --verdict (Bend's kernel proved in Lean)
make check   # the laws, prep checks, the Bend tokenizer against `tokenizers`; one layer, the forward and generation against `transformers`
```

Another model of `reference/models.py` is one variable away: `make prep check MODEL=smollm2-360m`.

To generate text (greedy, here 32 tokens):

```bash
bend demo/gen.bend -o gen && ./gen data/smollm2-135m "The capital of France is" 32
```

To tokenize a file with the Bend tokenizer:

```bash
bend demo/tok.bend -o tok && ./tok data/smollm2-135m/merges_num.txt data/smollm2-135m/perm.txt my_text.txt
```

MIT · Renan Vinícius (@nuxyel)
