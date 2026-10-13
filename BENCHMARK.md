# Benchmark

Measured on 2026-10-12 by Renan Vinícius (@nuxyel), 08:38 to 12:11, with the machine idle, on mains power in the
`performance` profile: Intel Core Ultra 7 155H (6 performance cores with two threads, 8 efficiency cores, 2 low-power
cores; 22 threads), 32 GB, Linux 7.2.5, clang 22.1.8; Bend 2.0.36 with `bend-ml-tensor-array@0.1.7.0`. "1 thread" is
one process pinned to a performance core (`taskset -c 3`) with every thread setting at 1; "16 threads" is unpinned with
every setting at 16. The cases ran in alternated, shuffled rounds (5 for the 135M and 360M, 2 for the 1.7B and the
512-token runs); the numbers are medians.

The cost per token excludes loading and the prompt. Prompt: "Once upon a time", greedy decoding. Every system produced
the same tokens, with one exception noted below.

Three builds of llama.bend were measured: the key/value cache in lists (before milestone 9), in flat arrays with the
key/value heads one after another, and in flat arrays with the heads in parallel. The current commit runs the heads in
sequence with two threads or fewer and in parallel otherwise; it has not been measured yet.

## Milliseconds per generated token


**SmolLM2-135M, 32 tokens**

| system | 1 thread | 16 threads |
|---|---|---|
| llama.bend, cache in lists (`b212ccd`) | 141 | 100 |
| llama.bend, cache in arrays, heads in sequence (`aa60b5f`) | 122 | 67.4 |
| llama.bend, cache in arrays, heads in parallel (`bd912a7`) | 121 | 74.3 |
| transformers 5.19 (PyTorch 2.14.1) | 26.4 | 14.4 |
| llama.cpp `23b0202` (F32 GGUF) | 23.7 | 13.9 |
| candle `c68b249` (F32) | 26.5 | 27.8 |

**SmolLM2-360M, 32 tokens**

| system | 1 thread | 16 threads |
|---|---|---|
| llama.bend, cache in lists (`b212ccd`) | 334 | 212 |
| llama.bend, cache in arrays, heads in sequence (`aa60b5f`) | 286 | 145 |
| llama.bend, cache in arrays, heads in parallel (`bd912a7`) | 308 | 136 |
| transformers 5.19 (PyTorch 2.14.1) | 59.6 | 33.8 |
| llama.cpp `23b0202` (F32 GGUF) | 63.5 | 28.1 |
| candle `c68b249` (F32) | 72.3 | 81.0 |

**SmolLM2-1.7B, 128 tokens**

| system | 1 thread | 16 threads |
|---|---|---|
| llama.bend, cache in lists (`b212ccd`) | 1985 | 1692 |
| llama.bend, cache in arrays, heads in sequence (`aa60b5f`) | 1230 | 427 |
| llama.bend, cache in arrays, heads in parallel (`bd912a7`) | 1288 | 385 |
| transformers 5.19 (PyTorch 2.14.1) | 228 | 134 |
| llama.cpp `23b0202` (F32 GGUF) | 319 | 98.5 |
| candle `c68b249` (F32) | 288 | 302 |

At 16 threads the 1.7B varies from run to run (367 to 516 ms per token for the parallel build over its four samples,
the processor at 75-90 °C), while each run's own cost per token stays flat; read the 16-thread numbers with that spread.

## Longer generations: 512 tokens

| model | llama.bend, 1 / 16 threads | llama.cpp, 1 / 16 threads |
|---|---|---|
| SmolLM2-135M | 134 / 87.8 | 25.1 / 14.4 |
| SmolLM2-360M | 312 / 154 | 65.3 / 29.0 |
| SmolLM2-1.7B | 1207 / 473 | 334 / 104 |

llama.bend's cost per token no longer grows with the position. From its own per-token timestamps, the growth per
cached position is:

| | cache in lists, 128 tokens | cache in arrays, 512 tokens |
|---|---|---|
| SmolLM2-135M, 1 / 16 threads | not measured | 0.05 / 0.04 ms |
| SmolLM2-360M, 1 / 16 threads | not measured | 0.08 / 0.04 ms |
| SmolLM2-1.7B, 1 / 16 threads | 15.1 / 19.8 ms | 0.18 / 0.00 ms |

On the 360M over 512 tokens, llama.cpp's text leaves the other systems' after about 340 tokens, in every sample; a
separate run of transformers over the same 513 tokens gives llama.bend's tokens exactly, so the difference is
llama.cpp's own order of F32 additions.

## Start-up and memory

Wall time of a run that loads the model and generates one token, and peak resident memory, at 16 threads:

| system | 135M | 360M | 1.7B |
|---|---|---|---|
| llama.bend, cache in lists (`b212ccd`) | 4.3 s, 777 MB | 11.6 s, 2,329 MB | 68.6 s, 7,065 MB |
| llama.bend, cache in arrays, heads in sequence (`aa60b5f`) | 4.5 s, 782 MB | 11.7 s, 2,331 MB | 76.1 s, 6,900 MB |
| llama.bend, cache in arrays, heads in parallel (`bd912a7`) | 4.4 s, 784 MB | 11.7 s, 2,329 MB | 72.1 s, 6,907 MB |
| transformers 5.19 (PyTorch 2.14.1) | 2.7 s, 1,136 MB | 2.9 s, 2,436 MB | 4.5 s, 10,128 MB |
| llama.cpp `23b0202` (F32 GGUF) | 0.2 s, 743 MB | 0.2 s, 1,752 MB | 0.9 s, 8,120 MB |
| candle `c68b249` (F32) | 0.2 s, 789 MB | 0.4 s, 2,107 MB | 2.1 s, 9,844 MB |

Loading is llama.bend's slowest part: reading a file in Bend 2.0.36 gives one list cell per byte, and building and
walking those lists is the cost (bend-ml `NOTES.md`, exp. 13). The weight products are the rest of the distance to
llama.cpp; both are tracked in [`NOTES.md`](NOTES.md).
