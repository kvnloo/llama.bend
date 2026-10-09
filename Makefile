# llama.bend: setup and verification shortcuts. Run `make setup`, `make prep` once, then `make check`.
export BEND_NO_TELEMETRY := 1
export PATH := $(HOME)/.bend/bin:$(HOME)/.elan/bin:$(PATH)
PY := reference/.venv/bin/python

.PHONY: setup prep check

setup:            ## check the pinned Bend, create the Python venv
	scripts/setup.sh

prep:             ## download SmolLM2-135M (pinned revision) and write data/ (about 270 MB down, 540 MB written)
	$(PY) reference/prep.py

check:            ## weights bit for bit, sizes, tokenizer equivalence with `tokenizers`
	$(PY) reference/prep.py --check
