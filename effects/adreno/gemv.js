// A JavaScript host cannot execute this Android vendor OpenCL foreign-C effect.
// Never manufacture a GPU result, including by computing GEMV on the CPU.
function adreno_gemv_refused(_path, _pattern) {
  return {
    $: CID(Fail),
    error: io_tup(95, "Adreno.gemv: JavaScript target refused; build the Android/Bionic native foreign-C effect. No CPU fallback or GPU simulation is provided."),
  };
}
io_eff(CID(Adreno.gemv), adreno_gemv_refused);
