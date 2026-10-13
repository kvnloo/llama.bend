# Release checklist: Bend 2.0.37 and the first public version

The steps from a new Bend release to llama.bend public and `bend-ml-llama` on BendHub, in order. A step that fails
stops the list. Steps marked **Renan** are his decision or his action: the agent does not make the repository
public, publish to BendHub, tag a release or post.

## 1. The pin

1. Read the release notes (`gh release view v2.0.37 -R bendlang/bend`): breaking changes in Base or the syntax, and
   whether `--verdict` needs a new Lean.
2. `scripts/bump_bend.sh 2.0.37`: downloads the two Linux tarballs, computes their SHA256 and rewrites the pin in
   `scripts/setup.sh`, the CI step, the READMEs and CLAUDE.md. Check its diff; it touches pins only.
3. `make setup` installs it; `bend version` says 2.0.37.

## 2. No regression

1. `make laws`: the checker and `--verdict`, both `ALL PROOFS CHECK`, no `@unsafe`, no `?TODO`.
2. `make check` on the 135M: prep, the tokenizer against `tokenizers`, the layer test, the forward pass and the
   twelve generations against `transformers`, 0 failures.
3. `make bits`: every output of the 135M identical to the saved bits. #1377 and #1479 change how parallel lets are
   compiled and scheduled, not the order of any sum, so the bits should not move; if one does, find out why before
   anything else, and only then `bits.py save`.
4. The same bits check with `--threads 1` (wrappers that pass it), which runs attention's sequential order.
5. `MODEL=smollm2-360m` and `MODEL=smollm2-1.7b`: `reference/test_gen.py` on the twelve prompts, 0 failures. The
   1.7B takes about half an hour; run it alone.
6. The emitted C (`bend demo/gen.bend -o gen.c`) still has `par_att`'s two joins.

## 3. Text

1. CLAUDE.md, "pitfalls": the erased-last-parameter pitfall becomes history (fixed in 2.0.37 by #1377); keep the
   advice to check the joins in the C.
2. NOTES.md: a dated entry for the bump, with what the checks above reported.
3. `BENCHMARK.md` stays dated (2026-10-12, Bend 2.0.36) until the next measurement; say so in the README if the
   release goes out before it.

## 4. bend-ml

The same bump in bend-ml, with its own checklist (bend-ml `NOTES.md`, "Checklist for the bump to the release that
carries #1377"): `make check-full`, the C join check of `bmv`, the workaround comment retired. Its packages are
republished only if their source changes.

## 5. Public (Renan)

1. Read the README, the package README, `BENCHMARK.md` and `docs/` once more as a stranger would. Tracked files
   name no private repository and describe no measurement method, and since 2026-10-12 neither does the history
   (rewritten with git filter-repo; every tree on `devel` unchanged; the old -> new commit map is kept privately).
   GitHub still keeps the pull requests #1 and #2 (the v0.0.1 and v0.0.2 releases) pointing at the old commits, so
   a public repository would show them there. To avoid it: rename this repository (it stays private), create a new
   `nuxyel/llama.bend`, push `devel`, `main` and the tags, and recreate the two releases (a decision for Renan).
2. Make the repository public; bend-ml's README links it.
3. A pull request `devel` → `main`, then the tag `v0.1.0` on `main` (bend-ml's `docs/VERSIONING.md`).

## 6. BendHub (Renan)

1. The API: `cap` is in `KV`'s type since 2026-10-12 (NOTES, "the capacity in the cache's type"), so the first
   version publishes it. Publishing is permanent; a later signature change is a new version that breaks callers.
2. `bend login` (GitHub), then from `llama/`: `bend main.bend --publish bend-ml-llama@0.1.0.0`. The name has 12+
   characters, the version four numbers, `LICENSE` sits next to `main.bend`, and the imports are BendHub packages at
   pinned versions.
3. Check the published package from a scratch directory: `import bend-ml-llama@0.1.0.0/main.bend as M` and a
   forward pass on the 135M.

## 7. After

1. Posts (drafts prepared, Renan posts): X in English, LinkedIn in Portuguese.
2. The next measurement, on an idle machine on mains power: the current commit on 2.0.36 and on 2.0.37 in the same
   session, so the release's effect is measured on its own (one build per Bend version).
