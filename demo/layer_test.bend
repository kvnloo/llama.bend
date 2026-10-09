# Runs one SmolLM2 layer on T input rows and prints every intermediate as exact F32 bits, one line per
# intermediate and position: `name t b0 b1 ...`. reference/test_layer.py compares them with transformers.
#
#   layer_test <layer> <input.bin> <T>      (input: T x 576 little-endian F32, from reference/layer_ref.py)

import Base
import ./load.bend as L
import ./layer.bend as Y

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

def line(name: String, +t: Nat, xs: List<&2, F32>) -> IO(Unit):
  IO.print(name ++ " " ++ Nat.show(t) ++ show_bits(xs))

def rows(r: Nat, +c: Nat, +xs: List<&2, F32>) -> List<&2, List<&2, F32>>:
  match r:
    case 0n:
      Nil{}
    case 1n+p:
      List.take(&2, F32, xs, c) <> rows(p, c, List.drop(&2, F32, xs, c))

def step(+t: Nat, +ln1: List<&2, F32>, +x: List<&2, F32>) -> IO(Unit):
  line("n1", t, Y.rmsnorm(x, ln1))

def go(xs: List<&2, List<&2, F32>>, +t: Nat, +ln1: List<&2, F32>) -> IO(Unit):
  match xs:
    case Nil{}:
      IO.pure(Unit, Unit{})
    case Con{+x, rest}:
      do IO<Unit>:
        step(t, ln1, x)
        go(rest, 1n+t, ln1)

def run(+layer: Nat, path: String, +n: Nat) -> IO(Unit):
  do IO<Unit>:
    ln1 : List<&2, F32> <- L.get_list(L.nm(layer, "ln1"), 576n)
    xs : List<&2, F32> <- L.read_floats(path, Nat.mul(n, 576n))
    go(rows(n, 576n, xs), 0n, ln1)

def run2(+layer: Maybe<&2, Nat>, path: String, +n: Maybe<&2, Nat>) -> IO(Unit):
  match layer n:
    case Some{+l} Some{+k}:
      run(l, path, k)
    case _ _:
      IO.print("usage: layer_test <layer> <input.bin> <T>")

def dispatch(args: List<String>) -> IO(Unit):
  match args:
    case Con{prog, Con{l, Con{p, Con{k, Nil{}}}}}:
      run2(Nat.read(l), p, Nat.read(k))
    case _:
      IO.print("usage: layer_test <layer> <input.bin> <T>")

def main() -> IO(Unit):
  do IO<Unit>:
    args : List<String> <- IO.args()
    dispatch(args)
