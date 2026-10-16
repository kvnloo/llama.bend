# A bounded Adreno F32 GEMV capability example

I keep this downstream-only example separate from `llama/` and the existing demos. It evaluates **one real
SmolLM2-135M query projection**, using an explicit Bend foreign IO effect and an OpenCL C kernel on an Android
GPU. It is not a GPU inference backend, does not move generation onto Adreno, does not implement Bend's native
`!` evaluator, and makes no speedup or throughput claim. No production model path dispatches to this effect.
The JavaScript target explicitly refuses the operation with error 95; it never substitutes a CPU loop.

The input is the existing prepared tensor `data/smollm2-135m/w/h0.q.bin`: exactly **576 rows × 576 columns**,
row-major little-endian F32, **1,327,104 bytes**. The model is `HuggingFaceTB/SmolLM2-135M` at revision
`93efa2f097d58c2a74874c7e644dbc9b0cee75a2`, as in the repository's prep workflow. The runner records the exact
file SHA256 rather than trusting its name. It uploads that file once and gives both programs the same device
bytes. No weights, generated executables or receipts belong in Git.

## Programs and boundary

- `cpu.bend <model-directory> <pattern>` loads `w/h0.q.bin` through the existing `M.get_bands`, then calls the
  existing `M.mv` from `../../llama/main.bend`. Depth zero gives the production one-thread four-accumulator
  matvec without introducing a handwritten oracle. Only that weight is needed; no configuration or full model
  is loaded.
- `main.bend <tensor-path> <pattern>` calls `Adreno.gemv(path, pattern)` through the documented Bend 2.0.37
  foreign-effect ABI. Blocking OpenCL work runs on `io_work`, not on the Bend evaluator. The effect loads
  `/vendor/lib64/libOpenCL.so`, requires a GPU and an available source compiler, and has no CPU fallback.
  One work item computes each output row. OpenCL failures and nonfinite weights/results are real IO failures.
- `gemv.c` supplies the native effect and kernel; `gemv.js` supplies only explicit refusal.

Both CLIs reject a wrong argument count or unknown mode before loading or dispatching. The only patterns are:

| CLI label | Pattern index | Input of length 576 |
|---|---:|---|
| `ones` | 0 | all +1 |
| `alternating` | 1 | +1, −1, +1, −1, … (index zero is +1) |
| `basis-last` | 2 | zero except index 575, which is +1 |

Successful programs each emit exactly one result line:

```text
CPU_GEMV_BITS pattern=<0|1|2> values=<576 space-separated decimal U32 words>
ADRENO_GEMV_BITS pattern=<0|1|2> values=<576 space-separated decimal U32 words>
```

The words are the exact F32 bit patterns, not rounded decimal float values. The native effect also reports the
actual platform name/vendor/version, device name/vendor/version, driver version and device type. Its status
fields must state `gpu_required=true cpu_fallback=false`, `enqueue_ndrange_status=0 global=576`, and
`kernel_event_execution_status=0 blocking_readback=success`. A result line or completed GPU event alone is not
numerical acceptance.

## Numerical acceptance

`validation.py` requires exactly 576 finite CPU and GPU values with the requested pattern, rejecting missing,
extra, duplicate, corrupt and nonfinite results. It compares **every row in all three patterns**, including the
final row, with the fixed policy

```text
abs(gpu[row] - cpu[row]) <= 2e-5 + 2e-4 * abs(cpu[row])
```

The boundary is inclusive, and signed zeros are numerically equivalent. NaN and either infinity are rejected
on either side. The receipt contains all failed row indices, maximum absolute error and its row, and maximum
relative error and its row. Relative error uses `max(abs(cpu[row]), 1e-30)` only for reporting; it is not a second
acceptance criterion. This policy must not be widened to conceal a failing run. Both native C compilations use
`-ffp-contract=off -fno-fast-math`; the OpenCL kernel disables contraction and does not request relaxed math.

## Build with the installed toolchain

Nothing here downloads or installs a compiler, header, vendor library or tensor. Bend stays pinned to **2.0.37**.
The existing isolated Android/Bionic wrapper supplies the external OpenCL headers, sysroot, linker and runtime
libraries. It can be selected with `ANDROID_CC`; the generated C explicitly includes this source directory via
`-I`, because emitted files live under `.scratch/`.

From the repository root, using the already prepared local assets:

```sh
export BEND=/root/workspace/llama-bend-tools/bend/bin/bend
export ANDROID_CC=/root/workspace/bend-adreno-phone-20261016/android-cc.sh

# Substantive numerical/protocol and receipt-transition regressions, without a GPU or model.
python3 -m unittest discover -s reference -p test_adreno_gemv.py -v

# The shared benchmark lock belongs around the complete build/run, not inside compiler subprocesses.
flock -x /root/workspace/.llama-bend-benchmark.lock effects/adreno/build.sh
```

`build.sh` delegates to the `build` subcommand of `run-device.py`. That recipe records `bend version`, the Android
compiler version, all exact emission/compilation command arrays and their real exit statuses/stdout/stderr. It
emits the CPU C, foreign-effect C and refusal-only JavaScript target, then cross-compiles `cpu-android` and
`gemv-android`. It executes **no generated program**, especially not an Android binary inside glibc/proot.

Build outputs are confined to ignored `.scratch/adreno/build/`. The atomic `.scratch/adreno/build-receipt.json`
records hashes of the sources, pinned package sources, Base, Bend/compiler binaries and generated artifacts.
Every build attempt also retains its own receipt. A failed or interrupted new build replaces earlier current
success before any compiler command; the device runner refuses an unsuccessful build or changed source/artifact.

## Run on one explicit physical device

ADB and Bun must already be installed. Set `ADB_SERIAL` to the exact connected **physical** device serial from
`adb devices -l`; the runner has no default serial and never implicitly selects another device. `--adb` and
`--bun` select explicit executables when they are not on `PATH`.

```sh
flock -x /root/workspace/.llama-bend-benchmark.lock \
  python3 effects/adreno/run-device.py run \
  --serial "$ADB_SERIAL" \
  --tensor /root/workspace/llama.bend/data/smollm2-135m/w/h0.q.bin \
  --timeout 45
```

The runner records ADB/device identity and rejects a reported emulator or a root shell UID. It creates a fresh exclusive
`mktemp -d /data/local/tmp/bend-adreno-gemv.XXXXXXXX` directory with mode 0700. It checks the returned path, directory
ownership, absence of a symlink and initial emptiness before any upload, and rechecks ownership before each
push. It does not use root, credentials, a fixed remote working directory or an existing model directory. It
uploads only its two Android binaries and one tensor into its owned `model/w/h0.q.bin`, then verifies device
SHA256s before and after the cases. It leaves that owned directory intact, including on failure; it does not
remove or overwrite somebody else's device files.

Each CPU and GPU case has both a device-side deadline and a bounded ADB subprocess deadline. Bend runs with
`--threads 1 --gpu off`: the latter disables Bend's native bang backend, **not** this explicit foreign OpenCL
operation. The final local JavaScript invocation must really exit 95 with the target-refusal diagnostic.
Timeouts, launch failures, arbitrary nonzero exits and missing output are failures, never acceptable refusal.

All attempt directories and evidence stay under ignored `.scratch/adreno/`. Before local/device preflight,
`.scratch/adreno/latest.json` is atomically replaced with a new unsuccessful/running attempt. Commands are
persisted before launch and after completion; per-case comparisons are persisted incrementally. Each attempt
also retains `receipt.json`, so a later failure preserves partial stdout/stderr, previously completed cases and
earlier attempts. A host timeout has `exit_code: null` rather than an invented return code; an observed device
timeout exit is retained with `timed_out: true`. Only completion of all comparisons and the actual JavaScript
refusal sets final `acceptance_met: true`. An interrupted or killed attempt cannot expose stale success.

The regression suite exercises numeric tolerance boundaries, signed zero, nonfinite/shape/pattern corruption,
wrong or missing final rows, GPU-only metadata/status requirements, byte-identity failures, exclusive-path/serial
rejection, genuine subprocess nonzero/launch/timeout transitions, partial output and stale-success replacement.
It does not fake GPU execution and is not device evidence. Build/test/run results must be reported from an
actual invocation and its receipts; this source alone makes no claim that a particular driver accepts the kernel.
