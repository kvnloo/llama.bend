# vendor/

`tensor-gpu.bend` is a byte-for-byte copy of bend-ml-tensor-gpu 0.1.0.0 (`tensor-gpu/main.bend` in
[bend-ml](https://github.com/nuxyel/bend-ml), commit `7d8ac2c`, 2026-10-18), kept here until the package is published on
BendHub. Nothing is added to it. Once it is published, `llama/gpu.bend` will import
`bend-ml-tensor-gpu@0.1.0.0/main.bend` instead and this folder goes away.

Only `llama/gpu.bend` (and through it `demo/gen_gpu.bend`) imports it. It relies on Base's unsafe `Array.fork`, so no
file with laws may import it; `make laws` checks that.
