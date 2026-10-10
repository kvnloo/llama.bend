# llama.bend

SmolLM2-135M, a Llama-architecture language model, running in [Bend 2](https://github.com/bendlang/bend), with
the shapes in the types. Work in progress, private for now.

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
| 6. Generic typed API | a `Config` in the types; bit for bit equal to milestone 5 | |
| 7. Laws | the laws on that API, `--verdict` | |
| 8. Generality | SmolLM2-360M with only the config changed | |
| 9. Package and benchmark | `bend-ml-llama` on BendHub; time per token against PyTorch | |

The plan is in [`docs/plan.md`](docs/plan.md) and what I find along the way in [`NOTES.md`](NOTES.md).

```bash
make setup   # needs Bend 2.0.35 (bend-ml's scripts/setup.sh installs it) and Python 3.12+
make prep    # downloads SmolLM2-135M at a pinned revision into data/
make check   # prep checks, the Bend tokenizer against `tokenizers`; one layer, the forward and generation against `transformers`
```

To generate text (greedy, here 32 tokens):

```bash
bend demo/gen.bend -o gen && ./gen "The capital of France is" 32
```

To tokenize a file with the Bend tokenizer:

```bash
bend demo/tok.bend -o tok && ./tok data/merges_num.txt data/perm.txt my_text.txt
```

MIT · Renan Vinícius (@nuxyel)
