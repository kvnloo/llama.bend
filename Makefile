# llama.bend: setup and verification shortcuts. Run `make setup`, `make prep` once, then `make check`.
export BEND_NO_TELEMETRY := 1
export PATH := $(HOME)/.bend/bin:$(HOME)/.elan/bin:$(PATH)
PY := reference/.venv/bin/python

.PHONY: setup prep check

setup:            ## check the pinned Bend, create the Python venv
	scripts/setup.sh

prep:             ## download SmolLM2-135M (pinned revision) and write data/ (about 270 MB down, 540 MB written)
	$(PY) reference/prep.py

check:            ## prep, the Bend tokenizer against `tokenizers`, one layer and the whole forward against transformers
	! grep -n "@unsafe\|?TODO" demo/*.bend
	$(PY) reference/prep.py --check
	mkdir -p .scratch
	bend demo/tok.bend -o .scratch/tok
	$(PY) reference/test_tok.py .scratch/tok
	$(PY) reference/test_detok.py .scratch/tok
	[ -f data/test/inv_freq.bin ] || $(PY) reference/layer_ref.py
	bend demo/layer_test.bend -o .scratch/layer_test
	$(PY) reference/test_layer.py .scratch/layer_test
	[ -f data/test/fwd10_logits.bin ] || $(PY) reference/forward_ref.py
	bend demo/forward.bend -o .scratch/forward
	$(PY) reference/test_forward.py .scratch/forward
