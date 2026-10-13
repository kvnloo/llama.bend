#!/usr/bin/env bash
# Runs the commands whose output the README figures and the launch video show, and keeps that output in
# docs/media/outputs/. Nothing there is typed by hand: every file is what one of these commands printed.
#
#   scripts/capture_outputs.sh          the checker's outputs and a generation of the 1.7B (~3 min, needs make prep
#                                       MODEL=smollm2-1.7b)
#   scripts/capture_outputs.sh --full   also the twelve generations of the 1.7B against transformers (~17 min)
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.bend/bin:$PATH" BEND_NO_TELEMETRY=1
O=docs/media/outputs
mkdir -p "$O" .scratch

# what the checker says about the two programs that must not compile (it exits non-zero on them)
bend tests/bad_weight.bend > "$O/bad_weight.txt" 2>&1 || true
bend tests/bad_cap.bend > "$O/bad_cap.txt" 2>&1 || true
# the kernel's verdict on the package, and the statement of one law as it is written in the source
bend llama/main.bend --verdict 2>&1 | tail -1 > "$O/verdict.txt"
grep -c "^law " llama/main.bend > "$O/law_count.txt"
sed -n '/^law gqa_head:/,/^$/p' llama/main.bend | sed '/^$/d' > "$O/law_gqa.txt"
# a generation of the 1.7B: weights loaded, prompt ids, then each token with its logit and a timestamp
bend demo/gen.bend -o .scratch/gen
.scratch/gen data/smollm2-1.7b "The capital of France is" 32 > "$O/gen_17b.txt"
bend version > "$O/bend_version.txt"
if [ "${1:-}" = --full ]; then
  MODEL=smollm2-1.7b reference/.venv/bin/python reference/test_gen.py .scratch/gen > "$O/test_gen_17b.txt"
fi
ls -l "$O"
