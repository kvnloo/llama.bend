# The whole SmolLM2-135M forward pass in Bend: embedding row, the 30 layers with a key/value cache each, the
# final RMSNorm and the tied LM head (the embedding as one more product). Runs the given ids position by
# position and prints, as exact F32 bits, every layer's output, the final norm and the 49,152 logits:
# `name t b0 b1 ...` (name h<l> for layer l, nf, logits). reference/test_forward.py compares them with
# transformers.
#
#   forward <id> <id> ...

import Base
import ./load.bend as L
import ./layer.bend as Y
import bend-ml-tensor-array@0.1.6.0/main.bend as TA

# emb: 49,152 rows of 576 (the input lookup and the LM head); lnf: the final norm's gains
type Model is Type:
  Model{emb: TA.Bands<49152n, 576n>, layers: List<&1, Y.LW>, +lnf: List<&2, F32>}

# ---- the 30 layers at one position ----

# the last position's hidden state, the layers and caches handed back, and every layer's output
type RunOut is Type:
  RunOut{x: List<&2, F32>, layers: List<&1, Y.LW>, cs: List<&2, Y.KV>, hs: List<&2, List<&2, F32>>}

# pend is the result of the previous layer; its output feeds the next one (the GPT-2 demo's loop)
def run(ls: List<&1, Y.LW>, cs: List<&2, Y.KV>, al: List<&1, Y.LW>, ac: List<&2, Y.KV>, hs: List<&2, List<&2, F32>>, +t: Nat, pend: Y.BR) -> RunOut:
  match ls cs pend:
    case Con{l, lt} Con{c, ct} Y.BR{+y, kv, l0, tr}:
      run(lt, ct, l0 <> al, kv <> ac, y <> hs, t, Y.block(l, c, t, y))
    case _ _ Y.BR{+y, kv, l0, tr}:
      RunOut{y, List.reverse(&1, Y.LW, l0 <> al), List.reverse(&2, Y.KV, kv <> ac), List.reverse(&2, List<&2, F32>, y <> hs)}

def start(ls: List<&1, Y.LW>, cs: List<&2, Y.KV>, +t: Nat, +x: List<&2, F32>) -> RunOut:
  match ls cs:
    case Con{l, lt} Con{c, ct}:
      run(lt, ct, Nil{}, Nil{}, Nil{}, t, Y.block(l, c, t, x))
    case _ _:
      RunOut{x, Nil{}, Nil{}, Nil{}}

# ---- one position: embedding row, layers, final norm, logits ----

type Out is Type:
  Out{lg: List<&2, F32>, hs: List<&2, List<&2, F32>>, nf: List<&2, F32>, cs: List<&2, Y.KV>, m: Model}

def f3(layers: List<&1, Y.LW>, +lnf: List<&2, F32>, cs: List<&2, Y.KV>, +hs: List<&2, List<&2, F32>>, +nf: List<&2, F32>, r: TA.BL<49152n, 576n>) -> Out:
  match r:
    case TA.BL{emb, lg}:
      Out{lg, hs, nf, cs, Model{emb, layers, lnf}}

# logits = emb · nf (tie_word_embeddings: the LM head is the embedding matrix)
def f2b(emb: TA.Bands<49152n, 576n>, +lnf: List<&2, F32>, layers: List<&1, Y.LW>, cs: List<&2, Y.KV>, +hs: List<&2, List<&2, F32>>, +nf: List<&2, F32>) -> Out:
  f3(layers, lnf, cs, hs, nf, TA.Bands.matvec_l(49152n, 576n, emb, nf))

def f2(emb: TA.Bands<49152n, 576n>, +lnf: List<&2, F32>, r: RunOut) -> Out:
  match r:
    case RunOut{+x, layers, cs, hs}:
      f2b(emb, lnf, layers, cs, hs, Y.rmsnorm(x, lnf))

def f1(layers: List<&1, Y.LW>, +lnf: List<&2, F32>, cs: List<&2, Y.KV>, +t: Nat, r: TA.BRow<49152n, 576n>) -> Out:
  match r:
    case TA.BRow{emb, row}:
      f2(emb, lnf, start(layers, cs, t, row))

# token tok at position t (0-based), with the caches of positions 0..t-1
def fwd(m: Model, cs: List<&2, Y.KV>, +tok: Nat, +t: Nat) -> Out:
  match m:
    case Model{emb, layers, +lnf}:
      f1(layers, lnf, cs, t, TA.Bands.read_row(49152n, 576n, emb, tok))

# ---- loading ----

def load_layers(k: Nat, +d: Nat, +i: Nat, acc: List<&1, Y.LW>) -> IO(List<&1, Y.LW>):
  match k:
    case 0n:
      IO.pure(List<&1, Y.LW>, List.reverse(&1, Y.LW, acc))
    case 1n+p:
      do IO<List<&1, Y.LW>>:
        l : Y.LW <- L.load_lw(i, d)
        load_layers(p, d, 1n+i, l <> acc)

# the embedding gets one more level of bands than the layers (2^5 with 16 threads), as GPT-2's does
def load_model() -> IO(Model):
  do IO<Model>:
    threads : U32 <- IO.thread_count()
    +d : Nat = L.depth_for(threads)
    emb : TA.Bands<49152n, 576n> <- L.get_bands("emb", Nat.add(d, Nat.min(d, 1n)), 49152n, 576n)
    layers : List<&1, Y.LW> <- load_layers(30n, d, 0n, Nil{})
    lnf : List<&2, F32> <- L.get_list("lnf", 576n)
    return Model{emb, layers, lnf}

def empty_cs(n: Nat) -> List<&2, Y.KV>:
  match n:
    case 0n:
      Nil{}
    case 1n+p:
      Y.KV{Nil{}, Nil{}} <> empty_cs(p)

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

# prints a line unless the list is empty (nothing has run yet)
def line_ne(name: String, +t: Nat, xs: List<&2, F32>) -> IO(Unit):
  match xs:
    case Nil{}:
      IO.pure(Unit, Unit{})
    case Con{h, r}:
      IO.print(name ++ " " ++ Nat.show(t) ++ show_bits(Con{h, r}))

def show_hs(hs: List<&2, List<&2, F32>>, +l: Nat, +t: Nat) -> IO(Unit):
  match hs:
    case Nil{}:
      IO.pure(Unit, Unit{})
    case Con{h, r}:
      do IO<Unit>:
        line_ne("h" ++ Nat.show(l), t, h)
        show_hs(r, 1n+l, t)

# pend is the result of position t - 1 (empty before position 0): it is printed, then the next id runs
def go(ids: List<&2, Nat>, +t: Nat, pend: Out) -> IO(Unit):
  match ids pend:
    case Con{+tok, rest} Out{lg, hs, nf, cs, m}:
      do IO<Unit>:
        show_hs(hs, 0n, Nat.sub(t, 1n))
        line_ne("nf", Nat.sub(t, 1n), nf)
        line_ne("logits", Nat.sub(t, 1n), lg)
        go(rest, 1n+t, fwd(m, cs, tok, t))
    case Nil{} Out{lg, hs, nf, cs, m}:
      do IO<Unit>:
        show_hs(hs, 0n, Nat.sub(t, 1n))
        line_ne("nf", Nat.sub(t, 1n), nf)
        line_ne("logits", Nat.sub(t, 1n), lg)

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
    m : Model <- load_model()
    go(ids, 0n, Out{Nil{}, Nil{}, Nil{}, empty_cs(30n), m})

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
