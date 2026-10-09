# Runs one SmolLM2 layer on T input rows and prints every intermediate as exact F32 bits, one line per
# intermediate and position: `name t b0 b1 ...`. reference/test_layer.py compares them with transformers.
#
#   layer_test <layer> <input.bin> <T>      (input: T x 576 little-endian F32, from reference/layer_ref.py)

import Base
import ./load.bend as L
import ./layer.bend as Y
import bend-ml-tensor-array@0.1.6.0/main.bend as TA

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

# prints a line unless the step is not written yet (empty)
def line_ne(name: String, +t: Nat, xs: List<&2, F32>) -> IO(Unit):
  match xs:
    case Nil{}:
      IO.pure(Unit, Unit{})
    case Con{h, r}:
      line(name, t, Con{h, r})

def show_tr(+t: Nat, tr: Y.Tr) -> IO(Unit):
  match tr:
    case Y.Tr{n1, q, k, v, qr, kr, attn, a, n2, m, y}:
      do IO<Unit>:
        line_ne("n1", t, n1)
        line_ne("q", t, q)
        line_ne("k", t, k)
        line_ne("v", t, v)
        line_ne("qr", t, qr)
        line_ne("kr", t, kr)
        line_ne("attn", t, attn)
        line_ne("a", t, a)
        line_ne("n2", t, n2)
        line_ne("m", t, m)
        line_ne("y", t, y)

# pend is the result of position t - 1 (an empty trace before position 0); its trace is printed, then the
# layer runs on the next input with the cache and the bands that pend hands back
def go(xs: List<&2, List<&2, F32>>, +t: Nat, pend: Y.BR) -> IO(Unit):
  match xs pend:
    case Con{+x, rest} Y.BR{y, kv, lw, tr}:
      do IO<Unit>:
        show_tr(Nat.sub(t, 1n), tr)
        go(rest, 1n+t, Y.block(lw, kv, t, x))
    case Nil{} Y.BR{y, kv, lw, tr}:
      show_tr(Nat.sub(t, 1n), tr)

def load_lw(+layer: Nat, +d: Nat) -> IO(Y.LW):
  do IO<Y.LW>:
    ln1 : List<&2, F32> <- L.get_list(L.nm(layer, "ln1"), 576n)
    q : TA.Bands<576n, 576n> <- L.get_bands(L.nm(layer, "q"), d, 576n, 576n)
    k : TA.Bands<192n, 576n> <- L.get_bands(L.nm(layer, "k"), d, 192n, 576n)
    v : TA.Bands<192n, 576n> <- L.get_bands(L.nm(layer, "v"), d, 192n, 576n)
    o : TA.Bands<576n, 576n> <- L.get_bands(L.nm(layer, "o"), d, 576n, 576n)
    ln2 : List<&2, F32> <- L.get_list(L.nm(layer, "ln2"), 576n)
    gate : TA.Bands<1536n, 576n> <- L.get_bands(L.nm(layer, "gate"), d, 1536n, 576n)
    up : TA.Bands<1536n, 576n> <- L.get_bands(L.nm(layer, "up"), d, 1536n, 576n)
    down : TA.Bands<576n, 1536n> <- L.get_bands(L.nm(layer, "down"), d, 576n, 1536n)
    return Y.LW{ln1, q, k, v, o, ln2, gate, up, down}

def run(+layer: Nat, path: String, +n: Nat) -> IO(Unit):
  do IO<Unit>:
    threads : U32 <- IO.thread_count()
    lw : Y.LW <- load_lw(layer, L.depth_for(threads))
    IO.print("inv_freq" ++ show_bits(Y.inv_freq()))
    xs : List<&2, F32> <- L.read_floats(path, Nat.mul(n, 576n))
    go(rows(n, 576n, xs), 0n, Y.BR{Nil{}, Y.KV{Nil{}, Nil{}}, lw, Y.Tr{Nil{}, Nil{}, Nil{}, Nil{}, Nil{}, Nil{}, Nil{}, Nil{}, Nil{}, Nil{}, Nil{}}})

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
