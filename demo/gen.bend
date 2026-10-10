# Greedy generation with SmolLM2-135M in Bend: the prompt is tokenized in Bend, runs through the forward pass of
# forward.bend position by position (one key/value cache per layer), and then each new token is the argmax of the
# logits, fed back in. Generation stops after n tokens, or after the end-of-text token (id 0), as transformers'
# generate does. reference/test_gen.py compares the ids, the chosen logits and the text with transformers.
#
#   gen "prompt" n
#
# prints `prompt ids: ...`, one line per new token (`id <id> logit <F32 bits> <logit> t=<seconds>`) and
# `text: <prompt and continuation>`.

import Base
import ./tok.bend as K
import ./forward.bend as F
import ./layer.bend as Y
import bend-ml-bpe-tokenizer@0.1.2.0/main.bend as BPE

# ---- argmax: the first index of the largest value (torch.argmax does the same) ----

type AM is Data:
  AM{i: Nat, v: F32}

def gt_head(xs: List<&2, F32>, +v: F32) -> Bool:
  match xs:
    case Nil{}:
      False{}
    case Con{+h, t}:
      F32.is_gt(h, v)

def argmax.go(xs: List<&2, F32>, g: Bool, +i: Nat, +bi: Nat, +bv: F32) -> AM:
  match xs g:
    case Nil{} _:
      AM{bi, bv}
    case Con{+h, +t} True{}:
      argmax.go(t, gt_head(t, h), 1n+i, i, h)
    case Con{h, +t} False{}:
      argmax.go(t, gt_head(t, bv), 1n+i, bi, bv)

def argmax(xs: List<&2, F32>) -> AM:
  match xs:
    case Nil{}:
      AM{0n, 0.0}
    case Con{+h, +t}:
      argmax.go(t, gt_head(t, h), 1n, 0n, h)

# ---- the prompt as UTF-8 bytes ----

def selb(b: Bool, x: List<&2, U32>, y: List<&2, U32>) -> List<&2, U32>:
  match b:
    case True{}:
      x
    case False{}:
      y

def enc1(+cp: U32) -> List<&2, U32>:
  selb(U32.is_lt(cp, 128), cp <> Nil{}, selb(U32.is_lt(cp, 2048), (192 + U32.div(cp, 64) : U32) <> ((128 + U32.mod(cp, 64) : U32) <> Nil{}), selb(U32.is_lt(cp, 65536), (224 + U32.div(cp, 4096) : U32) <> ((128 + U32.mod(U32.div(cp, 64), 64) : U32) <> ((128 + U32.mod(cp, 64) : U32) <> Nil{})), (240 + U32.div(cp, 262144) : U32) <> ((128 + U32.mod(U32.div(cp, 4096), 64) : U32) <> ((128 + U32.mod(U32.div(cp, 64), 64) : U32) <> ((128 + U32.mod(cp, 64) : U32) <> Nil{}))))))

def utf8_of(cs: List<&2, Char>) -> List<&2, U32>:
  match cs:
    case Nil{}:
      Nil{}
    case Con{Chr{cp}, t}:
      List.append(&2, U32, enc1(cp), utf8_of(t))

# ---- the generation loop ----

# cs: the caches; tok: the id to run at position pos; queue: the prompt ids still to feed; left: tokens still to
# generate; gen: the generated ids, newest first
type St is Data:
  St{cs: List<&2, Y.KV>, tok: Nat, pos: Nat, queue: List<&2, Nat>, left: Nat, gen: List<&2, Nat>}

type SM is Type:
  SM{st: St, m: F.Model}

def bits(f: F32) -> U32:
  match f:
    case F32{w}:
      U32{w}

def tsec2(+dt: Nat) -> String:
  Nat.show(Nat.div(dt, 1000n)) ++ "." ++ Nat.show(Nat.div(Nat.mod(dt, 1000n), 100n)) ++ "s"

def tsec(t1: Nat, t0: Nat) -> String:
  tsec2(Nat.sub(t1, t0))

# after the end-of-text token nothing more is generated
def left_after(+id: Nat, +left: Nat) -> Nat:
  K.sel(Nat.is_eq(id, 0n), 0n, Nat.sub(left, 1n))

def adv3(left: Nat, +pos: Nat, gen: List<&2, Nat>, +t0: Nat, cs: List<&2, Y.KV>, m: F.Model, am: AM) -> IO(SM):
  match am:
    case AM{+id, +v}:
      do IO<SM>:
        t1 : Nat <- IO.now()
        IO.print("  id " ++ Nat.show(id) ++ "  logit " ++ U32.show(bits(v)) ++ " " ++ F32.show(v) ++ "  t=" ++ tsec(t1, t0))
        return SM{St{cs, id, 1n+pos, Nil{}, left_after(id, left), id <> gen}, m}

# while the prompt is being fed the logits are not used; after it, the argmax is the next token
def adv2(queue: List<&2, Nat>, left: Nat, +pos: Nat, gen: List<&2, Nat>, +t0: Nat, r: F.Out) -> IO(SM):
  match queue r:
    case Con{q, qt} F.Out{lg, hs, nf, cs, m}:
      IO.pure(SM, SM{St{cs, q, 1n+pos, qt, left, gen}, m})
    case Nil{} F.Out{lg, hs, nf, cs, m}:
      adv3(left, pos, gen, t0, cs, m, argmax(lg))

def adv(m: F.Model, st: St, t0: Nat) -> IO(SM):
  match st:
    case St{cs, +tok, +pos, queue, left, gen}:
      adv2(queue, left, pos, gen, t0, F.fwd(m, cs, tok, pos))

# fuel bounds the number of forward passes; the loop also stops when nothing is left to generate
def loop(fuel: Nat, +t0: Nat, sm: SM) -> IO(St):
  match fuel sm:
    case 0n SM{st, m}:
      IO.pure(St, st)
    case 1n+p SM{St{cs, tok, pos, queue, 0n, gen}, m}:
      IO.pure(St, St{cs, tok, pos, queue, 0n, gen})
    case 1n+p SM{st, m}:
      do IO<St>:
        s2 : SM <- adv(m, st, t0)
        loop(p, t0, s2)

# ---- main program ----

def fin(+table: List<&2, BPE.Rule>, +perm: List<&2, Nat>, ids: List<&2, Nat>, st: St) -> IO(Unit):
  match st:
    case St{cs, tok, pos, queue, left, gen}:
      IO.print("text: " ++ K.decode_text(table, perm, List.append(&2, Nat, ids, List.reverse(&2, Nat, gen))))

def show_nats(xs: List<&2, Nat>) -> String:
  match xs:
    case Nil{}:
      ""
    case Con{h, t}:
      Nat.show(h) ++ " " ++ show_nats(t)

def start2(model: F.Model, +table: List<&2, BPE.Rule>, +perm: List<&2, Nat>, +ids: List<&2, Nat>, +n: Nat) -> IO(Unit):
  match ids:
    case Nil{}:
      IO.print("empty prompt")
    case Con{t0, qt}:
      do IO<Unit>:
        t0ms : Nat <- IO.now()
        IO.print("prompt ids: " ++ show_nats(ids))
        st : St <- loop(Nat.add(List.length(&2, Nat, ids), n), t0ms, SM{St{F.empty_cs(30n), t0, 0n, qt, n, Nil{}}, model})
        fin(table, perm, ids, st)

def generate(model: F.Model, +mb: List<&2, U32>, +pb: List<&2, U32>, prompt: String, n: Nat) -> IO(Unit):
  start2(model, K.rules.go(K.nums(mb), 0n), K.nums(pb), K.encode_text(K.rules.go(K.nums(mb), 0n), K.nums(pb), utf8_of(String.to_list(prompt))), n)

# refuse to run on a malformed merges table (checked before the weights are loaded)
def run3(ok: Bool, +mb: List<&2, U32>, +pb: List<&2, U32>, prompt: String, n: Nat) -> IO(Unit):
  match ok:
    case True{}:
      do IO<Unit>:
        t0 : Nat <- IO.now()
        model : F.Model <- F.load_model()
        t1 : Nat <- IO.now()
        IO.print("weights loaded in " ++ tsec(t1, t0))
        generate(model, mb, pb, prompt, n)
    case False{}:
      IO.print("invalid merges table")

def run2(+mb: List<&2, U32>, +pb: List<&2, U32>, prompt: String, n: Nat) -> IO(Unit):
  run3(K.ids_ok(K.nums(mb), 0n), mb, pb, prompt, n)

def run1(prompt: String, mn: Maybe<&2, Nat>) -> IO(Unit):
  match mn:
    case None{}:
      IO.print("usage: gen <prompt> <n_tokens>")
    case Some{n}:
      do IO<Unit>:
        mb : List<&2, U32> <- K.read_bytes("data/merges_num.txt")
        pb : List<&2, U32> <- K.read_bytes("data/perm.txt")
        run2(mb, pb, prompt, n)

def dispatch(args: List<String>) -> IO(Unit):
  match args:
    case Con{prog, Con{prompt, Con{n, Nil{}}}}:
      run1(prompt, Nat.read(n))
    case _:
      IO.print("usage: gen <prompt> <n_tokens>")

def main() -> IO(Unit):
  do IO<Unit>:
    args : List<String> <- IO.args()
    dispatch(args)
