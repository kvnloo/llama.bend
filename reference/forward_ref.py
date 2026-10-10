"""Expected values for the whole SmolLM2-135M forward pass (milestone 4), from `transformers` in F32 with eager
attention.

  reference/.venv/bin/python reference/forward_ref.py

Writes to data/test/, for every prompt i of PROMPTS (the 11 prompts of bend-ml's GPT-2 test), as little-endian
F32 arrays:
  fwd<i>_ids.txt        the prompt's ids (`tokenizers`, no special tokens), one line
  fwd<i>_h.bin          the output of every layer (30 x T x 576), from forward hooks on the decoder layers
  fwd<i>_nf.bin         the final norm (T x 576)
  fwd<i>_logits.bin     the logits (T x 49152)
"""
import os
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HF = os.path.join(ROOT, "data", "hf")
OUT = os.path.join(ROOT, "data", "test")
PROMPTS = [
    "Wait — the café costs €5 😀 and",
    "The capital of France is",
    "Machine learning is",
    "1, 2, 3, 4,",
    "def fibonacci(n):",
    "Once upon a time",
    "The quick brown fox jumps over the lazy dog. The",
    "Q: What is 2 + 2?\nA:",
    "Paris is the capital of",
    "In 1969, humans first",
    "import numpy as np\n\n",
]


def save(a, name):
    a.detach().float().numpy().astype("<f4").tofile(os.path.join(OUT, name))


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    os.makedirs(OUT, exist_ok=True)
    model = AutoModelForCausalLM.from_pretrained(HF, dtype=torch.float32, attn_implementation="eager").eval()
    tok = AutoTokenizer.from_pretrained(HF)
    for i, prompt in enumerate(PROMPTS):
        ids = tok(prompt, add_special_tokens=False)["input_ids"]
        hs, nf = [], []
        hooks = [lay.register_forward_hook(lambda mod, inp, out: hs.append(out[0] if isinstance(out, tuple) else out))
                 for lay in model.model.layers]
        hooks.append(model.model.norm.register_forward_hook(lambda mod, inp, out: nf.append(out)))
        with torch.no_grad():
            logits = model(torch.tensor([ids])).logits[0]
        for h in hooks:
            h.remove()
        with open(os.path.join(OUT, f"fwd{i}_ids.txt"), "w") as f:
            f.write(" ".join(map(str, ids)) + "\n")
        save(torch.stack([h.reshape(len(ids), 576) for h in hs]), f"fwd{i}_h.bin")
        save(nf[0].reshape(len(ids), 576), f"fwd{i}_nf.bin")
        save(logits, f"fwd{i}_logits.bin")
        print(f"fwd{i}: {len(ids)} positions, argmax {logits.argmax(-1).tolist()}  {prompt!r}")


if __name__ == "__main__":
    main()
