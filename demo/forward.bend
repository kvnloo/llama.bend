# The whole forward pass of a Llama model (bend-ml-llama, llama/main.bend) over the given ids, position by
# position, printing as exact F32 bits every layer's output, the final norm and the logits:
# `name t b0 b1 ...` (name h<l> for layer l, nf, logits). reference/test_forward.py compares them with
# transformers.
#
#   forward <id> <id> ...        (the model in data/: cfg.txt and w/, written by reference/prep.py)

import Base
import ../llama/main.bend as M

# ---- printing ----

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

# the output of every layer, h0 to h<layers - 1>
def show_acts(acts: List<&2, M.Act>, +l: Nat, +t: Nat) -> IO(Unit):
  match acts:
    case Nil{}:
      IO.pure(Unit, Unit{})
    case Con{M.Act{n1, q, k, v, qr, kr, at, a, n2, m, y}, r}:
      do IO<Unit>:
        line("h" ++ Nat.show(l), t, y)
        show_acts(r, 1n+l, t)

# ids[0] runs at position t with the caches of positions 0..t-1; then the rest
def go(ids: List<&2, Nat>, +c: M.Cfg, +t: Nat, m: M.Model<c>, cs: List<&2, M.Cache(t)>) -> IO(Unit):
  match ids:
    case Nil{}:
      IO.pure(Unit, Unit{})
    case Con{+tok, rest}:
      M.Out.then(IO(Unit), c, t, M.forward(c, t, m, cs, tok), lg => nf => acts => cs2 => m2 =>
        do IO<Unit>:
          show_acts(acts, 0n, t)
          line("nf", t, nf)
          line("logits", t, lg)
          go(rest, c, 1n+t, m2, cs2))

# ---- arguments ----

def read1(m: Maybe<&2, Nat>, +rest: List<&2, Nat>) -> List<&2, Nat>:
  match m:
    case Some{+n}:
      n <> rest
    case None{}:
      rest

def read_ids(xs: List<String>) -> List<&2, Nat>:
  match xs:
    case Nil{}:
      Nil{}
    case Con{s, r}:
      read1(Nat.read(s), read_ids(r))

def run_ids(+ids: List<&2, Nat>) -> IO(Unit):
  do IO<Unit>:
    +c : M.Cfg <- M.read_cfg("data")
    m : M.Model<c> <- M.load_model(c, "data")
    go(ids, c, 0n, m, M.caches(M.Cfg.layers(c)))

def dispatch(args: List<String>) -> IO(Unit):
  match args:
    case Con{prog, Con{a, r}}:
      run_ids(read_ids(Con{a, r}))
    case _:
      IO.print("usage: forward <id> <id> ...")

def main() -> IO(Unit):
  do IO<Unit>:
    args : List<String> <- IO.args()
    dispatch(args)
