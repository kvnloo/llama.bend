// This effect requires the real Android vendor OpenCL GPU and event profiling.
// JavaScript must not substitute CPU computation, simulated events or host time.
function adreno_bench_refused(_path, _source, _params) {
  return {
    $: CID(Fail),
    error: io_tup(95, "Bench.run: JavaScript target refused; build the Android/Bionic native foreign-C effect. A real OpenCL GPU and event profiling are required; no CPU fallback or GPU simulation is provided."),
  };
}
io_eff(CID(Bench.run), adreno_bench_refused);
