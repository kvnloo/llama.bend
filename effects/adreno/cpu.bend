# Capability oracle only: the production loader and matvec, not an inference backend.
import Base
import ../../llama/main.bend as M
import bend-ml-tensor-array@0.1.7.0/main.bend as TA

def bits(f: F32) -> U32:
  match f:
    case F32{w}:
      U32{w}

def show_bits(xs: List<&2, F32>) -> String:
  match xs:
    case Nil{}:
      ""
    case Con{h, t}:
      " " ++ U32.show(bits(h)) ++ show_bits(t)

def show_values(xs: List<&2, F32>) -> String:
  match xs:
    case Nil{}:
      ""
    case Con{h, t}:
      U32.show(bits(h)) ++ show_bits(t)

def finite(xs: List<&2, F32>) -> Bool:
  match xs:
    case Nil{}:
      True{}
    case Con{h, t}:
      Bool.and(U32.is_ne(U32.and(bits(h), 2139095040), 2139095040), finite(t))

def emit(ok: Bool, pattern: U32, xs: List<&2, F32>) -> IO(Unit):
  match ok:
    case True{}:
      IO.print("CPU_GEMV_BITS pattern=" ++ U32.show(pattern) ++ " values=" ++ show_values(xs))
    case False{}:
      IO.die(Unit, 1, "CPU GEMV produced a nonfinite output")

def alternating(n: Nat, +x: F32) -> List<&2, F32>:
  match n:
    case 0n:
      Nil{}
    case 1n+p:
      x <> alternating(p, F32.sub(0.0, x))

def basis_last(n: Nat) -> List<&2, F32>:
  match n:
    case 0n:
      Nil{}
    case 1n++p:
      Bool.pick(F32, Nat.is_eq(p, 0n), 1.0, 0.0) <> basis_last(p)

def run(dir: String, +pattern: U32, x: List<&2, F32>) -> IO(Unit):
  do IO<Unit>:
    w : TA.Bands<576n, 576n> <- M.get_bands(dir, "h0.q", 0n, 576n, 576n)
    M.mv(IO(Unit), 576n, 576n, w, x, matrix => +ys => emit(finite(ys), pattern, ys))

def choose_basis(ok: Bool, dir: String) -> IO(Unit):
  match ok:
    case True{}:
      run(dir, 2, basis_last(576n))
    case False{}:
      IO.die(Unit, 22, "unknown pattern: use ones, alternating, or basis-last")

def choose_alternating(ok: Bool, dir: String, mode: String) -> IO(Unit):
  match ok:
    case True{}:
      run(dir, 1, alternating(576n, 1.0))
    case False{}:
      choose_basis(String.eq(mode, "basis-last"), dir)

def choose_ones(ok: Bool, dir: String, +mode: String) -> IO(Unit):
  match ok:
    case True{}:
      run(dir, 0, List.replicate(F32, 576n, 1.0))
    case False{}:
      choose_alternating(String.eq(mode, "alternating"), dir, mode)

def dispatch(args: List<String>) -> IO(Unit):
  match args:
    case Con{program, Con{dir, Con{+mode, Nil{}}}}:
      choose_ones(String.eq(mode, "ones"), dir, mode)
    case _:
      IO.die(Unit, 22, "usage: cpu <model-directory> <ones|alternating|basis-last>")

def main() -> IO(Unit):
  do IO<Unit>:
    args : List<String> <- IO.args()
    dispatch(args)
