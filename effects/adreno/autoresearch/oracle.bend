# Independent numerical oracle: the unchanged production loader and matvec.
import Base
import ../../../llama/main.bend as M
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
      IO.die(Unit, 1, "CPU GEMV requires exactly 576 finite outputs")

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

# The integer intermediates are small and conversion is exact. Subtraction and
# multiplication by a dyadic scale exactly match bench.c's signed numerators:
# pattern 3: ((i % 29) - 14) / 16
# pattern 4: (((i * 73 + 19) % 257) - 128) / 128
def patterned(n: Nat, +i: U32, +period: U32, +step: U32, +offset: U32, +center: F32, +scale: F32) -> List<&2, F32>:
  match n:
    case 0n:
      Nil{}
    case 1n+p:
      F32.mul(F32.sub(U32.to_f32(U32.mod(U32.add(U32.mul(i, step), offset), period)), center), scale) <> patterned(p, U32.add(i, 1), period, step, offset, center, scale)

def run(dir: String, +pattern: U32, x: List<&2, F32>) -> IO(Unit):
  do IO<Unit>:
    w : TA.Bands<576n, 576n> <- M.get_bands(dir, "h0.q", 0n, 576n, 576n)
    M.mv(IO(Unit), 576n, 576n, w, x, matrix => +ys => emit(Bool.and(Nat.is_eq(List.length(&2, F32, ys), 576n), finite(ys)), pattern, ys))

def choose(pattern: Nat, dir: String) -> IO(Unit):
  match pattern:
    case 0n:
      run(dir, 0, List.replicate(F32, 576n, 1.0))
    case 1n:
      run(dir, 1, alternating(576n, 1.0))
    case 2n:
      run(dir, 2, basis_last(576n))
    case 3n:
      run(dir, 3, patterned(576n, 0, 29, 1, 0, 14.0, 0.0625))
    case 4n:
      run(dir, 4, patterned(576n, 0, 257, 73, 19, 128.0, 0.0078125))
    case _:
      IO.die(Unit, 22, "pattern must be one decimal digit 0..4")

def dispatch(args: List<String>) -> IO(Unit):
  match args:
    case Con{program, Con{dir, Con{+mode, Nil{}}}}:
      choose(Bool.pick(Nat, String.eq(mode, "0"), 0n, Bool.pick(Nat, String.eq(mode, "1"), 1n, Bool.pick(Nat, String.eq(mode, "2"), 2n, Bool.pick(Nat, String.eq(mode, "3"), 3n, Bool.pick(Nat, String.eq(mode, "4"), 4n, 5n))))), dir)
    case _:
      IO.die(Unit, 22, "usage: adreno-oracle <model-directory> <pattern decimal 0..4>")

def main() -> IO(Unit):
  do IO<Unit>:
    args : List<String> <- IO.args()
    dispatch(args)
