# One SmolLM2 layer (Llama architecture) in Bend: RMSNorm, RoPE, grouped-query attention with a key/value
# cache, SwiGLU. The order of the F32 operations follows transformers' modeling_llama.py (5.19), so the test
# (reference/test_layer.py) can ask for 1e-5. Heads are list slices here; their types come in milestone 6.

import Base
import bend-ml-tensor@0.1.2.0/main.bend as T
import bend-ml-tensor-array@0.1.6.0/main.bend as TA

# RMSNorm: g * (x * (1 / sqrt(mean(x²) + eps))), in this order (LlamaRMSNorm.forward)
def rmsnorm(+x: List<&2, F32>, +g: List<&2, F32>) -> List<&2, F32>:
  T.zip_with.mul(g, T.scale.l((1.0 / F32.sqrt(((T.sq_sum(x, 0.0) / 576.0 : F32) + 0.00001 : F32)) : F32), x))

# ---- RoPE (rotate-half layout, theta = 100000, head size 64) ----

# inv_freq[i] = 1 / (100000 ^ (2i / 64)), i = 0..31 (LlamaRotaryEmbedding, default rope)
def inv_freq.go(n: Nat, +i: Nat) -> List<&2, F32>:
  match n:
    case 0n:
      Nil{}
    case 1n+p:
      (1.0 / F32.pow(100000.0, (F32.from_nat(Nat.mul(2n, i)) / 64.0 : F32)) : F32) <> inv_freq.go(p, 1n+i)

def inv_freq() -> List<&2, F32>:
  inv_freq.go(32n, 0n)

# angles t * inv_freq[i]
def angles(+t: F32, fs: List<&2, F32>) -> List<&2, F32>:
  match fs:
    case Nil{}:
      Nil{}
    case Con{+f, r}:
      (t * f : F32) <> angles(t, r)

def cos.l(xs: List<&2, F32>) -> List<&2, F32>:
  match xs:
    case Nil{}:
      Nil{}
    case Con{+x, r}:
      F32.cos(x) <> cos.l(r)

def sin.l(xs: List<&2, F32>) -> List<&2, F32>:
  match xs:
    case Nil{}:
      Nil{}
    case Con{+x, r}:
      F32.sin(x) <> sin.l(r)

# first half: h1*cos + (-h2)*sin; second half: h2*cos + h1*sin (q*cos + rotate_half(q)*sin)
def rot_lo(h1: List<&2, F32>, h2: List<&2, F32>, cs: List<&2, F32>, sn: List<&2, F32>) -> List<&2, F32>:
  match h1 h2 cs sn:
    case Con{+a, ta} Con{+b, tb} Con{+c, tc} Con{+s, ts}:
      ((a * c : F32) + (F32.neg(b) * s : F32) : F32) <> rot_lo(ta, tb, tc, ts)
    case _ _ _ _:
      Nil{}

def rot_hi(h1: List<&2, F32>, h2: List<&2, F32>, cs: List<&2, F32>, sn: List<&2, F32>) -> List<&2, F32>:
  match h1 h2 cs sn:
    case Con{+a, ta} Con{+b, tb} Con{+c, tc} Con{+s, ts}:
      ((b * c : F32) + (a * s : F32) : F32) <> rot_hi(ta, tb, tc, ts)
    case _ _ _ _:
      Nil{}

def rope_head(+h: List<&2, F32>, +cs: List<&2, F32>, +sn: List<&2, F32>) -> List<&2, F32>:
  List.append(&2, F32, rot_lo(List.take(&2, F32, h, 32n), List.drop(&2, F32, h, 32n), cs, sn), rot_hi(List.take(&2, F32, h, 32n), List.drop(&2, F32, h, 32n), cs, sn))

# rotates each of the n heads of 64 in xs
def rope(n: Nat, +xs: List<&2, F32>, +cs: List<&2, F32>, +sn: List<&2, F32>) -> List<&2, F32>:
  match n:
    case 0n:
      Nil{}
    case 1n+p:
      List.append(&2, F32, rope_head(List.take(&2, F32, xs, 64n), cs, sn), rope(p, List.drop(&2, F32, xs, 64n), cs, sn))

# ---- grouped-query attention over the cache (9 query heads, 3 key/value heads of 64) ----

def slice64(xs: List<&2, F32>, off: Nat) -> List<&2, F32>:
  List.take(&2, F32, List.drop(&2, F32, xs, off), 64n)

# q·k * 0.125 (1 / sqrt(64)) for every cached key, oldest first
def scores_l(+qh: List<&2, F32>, ks: List<&2, List<&2, F32>>, +off: Nat) -> List<&2, F32>:
  match ks:
    case Nil{}:
      Nil{}
    case Con{k, t}:
      (T.dot.go(qh, slice64(k, off), 0.0) * 0.125 : F32) <> scores_l(qh, t, off)

def weighted(w: List<&2, F32>, vs: List<&2, List<&2, F32>>, +off: Nat, acc: List<&2, F32>) -> List<&2, F32>:
  match w vs:
    case Con{+a, wt} Con{v, vt}:
      weighted(wt, vt, off, T.zip_with.add(acc, T.scale.l(a, slice64(v, off))))
    case _ _:
      acc

def head_out(qh: List<&2, F32>, ks: List<&2, List<&2, F32>>, +vs: List<&2, List<&2, F32>>, +off: Nat) -> List<&2, F32>:
  weighted(T.softmax.l(scores_l(qh, ks, off)), vs, off, T.fill.l(64n, 0.0))

# query heads j..j+n-1; query head j reads key/value head j / 3 (repeat_kv with 3 query heads per group)
def attn(n: Nat, +j: Nat, +q: List<&2, F32>, +ks: List<&2, List<&2, F32>>, +vs: List<&2, List<&2, F32>>) -> List<&2, F32>:
  match n:
    case 0n:
      Nil{}
    case 1n+p:
      List.append(&2, F32, head_out(slice64(q, Nat.mul(64n, j)), ks, vs, Nat.mul(64n, Nat.div(j, 3n))), attn(p, 1n+j, q, ks, vs))

# ---- SwiGLU MLP ----

# silu(x) = x / (1 + exp(-x)), the form ATen's CPU kernel uses
def silu.f(+x: F32) -> F32:
  (x / (1.0 + F32.exp(F32.neg(x)) : F32) : F32)

# silu(gate) * up
def swiglu(g: List<&2, F32>, u: List<&2, F32>) -> List<&2, F32>:
  match g u:
    case Con{+a, ga} Con{+b, ub}:
      (silu.f(a) * b : F32) <> swiglu(ga, ub)
    case _ _:
      Nil{}

# ---- the layer's weights, its key/value cache, and what one position produces ----

# ln1, ln2: 576 gains; products as row bands (row = output), see load.bend
type LW is Type:
  LW{ln1: List<&2, F32>, q: TA.Bands<576n, 576n>, k: TA.Bands<192n, 576n>, v: TA.Bands<192n, 576n>, o: TA.Bands<576n, 576n>, ln2: List<&2, F32>, gate: TA.Bands<1536n, 576n>, up: TA.Bands<1536n, 576n>, down: TA.Bands<576n, 1536n>}

# the keys (after RoPE) and values of the positions so far, oldest first; 192 numbers each (3 heads of 64)
type KV is Data:
  KV{ks: List<&2, List<&2, F32>>, vs: List<&2, List<&2, F32>>}

# every intermediate of one position, for the test (empty while a step is not written yet)
type Tr is Data:
  Tr{n1: List<&2, F32>, q: List<&2, F32>, k: List<&2, F32>, v: List<&2, F32>, qr: List<&2, F32>, kr: List<&2, F32>, attn: List<&2, F32>, a: List<&2, F32>, n2: List<&2, F32>, m: List<&2, F32>, y: List<&2, F32>}

type BR is Type:
  BR{y: List<&2, F32>, kv: KV, lw: LW, tr: Tr}

# ---- one position: each step takes the bands a product hands back (match only reads parameters) ----

# 4. RoPE on q (9 heads) and k (3 heads) at position t
# 11. y = x1 + m: the layer's output, and the trace of the position
def b13(+ln1: List<&2, F32>, wq: TA.Bands<576n, 576n>, wk: TA.Bands<192n, 576n>, wv: TA.Bands<192n, 576n>, wo: TA.Bands<576n, 576n>, +ln2: List<&2, F32>, wg: TA.Bands<1536n, 576n>, wu: TA.Bands<1536n, 576n>, wd: TA.Bands<576n, 1536n>, kv: KV, tr: Tr, +n2: List<&2, F32>, +m: List<&2, F32>, +y: List<&2, F32>) -> BR:
  match tr:
    case Tr{n1, q, k, v, qr, kr, at, a, z2, zm, zy}:
      BR{y, kv, LW{ln1, wq, wk, wv, wo, ln2, wg, wu, wd}, Tr{n1, q, k, v, qr, kr, at, a, n2, m, y}}

# 10. m = Wdown · swiglu
def b12(+ln1: List<&2, F32>, wq: TA.Bands<576n, 576n>, wk: TA.Bands<192n, 576n>, wv: TA.Bands<192n, 576n>, wo: TA.Bands<576n, 576n>, +ln2: List<&2, F32>, wg: TA.Bands<1536n, 576n>, wu: TA.Bands<1536n, 576n>, kv: KV, tr: Tr, +x1: List<&2, F32>, +n2: List<&2, F32>, rd: TA.BL<576n, 1536n>) -> BR:
  match rd:
    case TA.BL{wd, +m}:
      b13(ln1, wq, wk, wv, wo, ln2, wg, wu, wd, kv, tr, n2, m, T.zip_with.add(x1, m))

# 9. u = Wup · n2, then silu(g) * u goes into Wdown
def b11(+ln1: List<&2, F32>, wq: TA.Bands<576n, 576n>, wk: TA.Bands<192n, 576n>, wv: TA.Bands<192n, 576n>, wo: TA.Bands<576n, 576n>, +ln2: List<&2, F32>, wg: TA.Bands<1536n, 576n>, wd: TA.Bands<576n, 1536n>, kv: KV, tr: Tr, +x1: List<&2, F32>, +n2: List<&2, F32>, +g: List<&2, F32>, ru: TA.BL<1536n, 576n>) -> BR:
  match ru:
    case TA.BL{wu, u}:
      b12(ln1, wq, wk, wv, wo, ln2, wg, wu, kv, tr, x1, n2, TA.Bands.matvec_l(576n, 1536n, wd, swiglu(g, u)))

# 8. g = Wgate · n2
def b10(+ln1: List<&2, F32>, wq: TA.Bands<576n, 576n>, wk: TA.Bands<192n, 576n>, wv: TA.Bands<192n, 576n>, wo: TA.Bands<576n, 576n>, +ln2: List<&2, F32>, wu: TA.Bands<1536n, 576n>, wd: TA.Bands<576n, 1536n>, kv: KV, tr: Tr, +x1: List<&2, F32>, +n2: List<&2, F32>, rg: TA.BL<1536n, 576n>) -> BR:
  match rg:
    case TA.BL{wg, g}:
      b11(ln1, wq, wk, wv, wo, ln2, wg, wd, kv, tr, x1, n2, g, TA.Bands.matvec_l(1536n, 576n, wu, n2))

# 7. the residual x1 = x + a and n2 = RMSNorm(x1, ln2)
def b9b(+ln1: List<&2, F32>, wq: TA.Bands<576n, 576n>, wk: TA.Bands<192n, 576n>, wv: TA.Bands<192n, 576n>, wo: TA.Bands<576n, 576n>, +ln2: List<&2, F32>, wg: TA.Bands<1536n, 576n>, wu: TA.Bands<1536n, 576n>, wd: TA.Bands<576n, 1536n>, kv: KV, tr: Tr, +x1: List<&2, F32>, +n2: List<&2, F32>) -> BR:
  b10(ln1, wq, wk, wv, wo, ln2, wu, wd, kv, tr, x1, n2, TA.Bands.matvec_l(1536n, 576n, wg, n2))

def b9(+ln1: List<&2, F32>, wq: TA.Bands<576n, 576n>, wk: TA.Bands<192n, 576n>, wv: TA.Bands<192n, 576n>, wo: TA.Bands<576n, 576n>, +ln2: List<&2, F32>, wg: TA.Bands<1536n, 576n>, wu: TA.Bands<1536n, 576n>, wd: TA.Bands<576n, 1536n>, kv: KV, tr: Tr, +x1: List<&2, F32>) -> BR:
  b9b(ln1, wq, wk, wv, wo, ln2, wg, wu, wd, kv, tr, x1, rmsnorm(x1, ln2))

# 6. a = Wo · attn
def b8(+ln1: List<&2, F32>, wq: TA.Bands<576n, 576n>, wk: TA.Bands<192n, 576n>, wv: TA.Bands<192n, 576n>, +ln2: List<&2, F32>, wg: TA.Bands<1536n, 576n>, wu: TA.Bands<1536n, 576n>, wd: TA.Bands<576n, 1536n>, +x: List<&2, F32>, +n1: List<&2, F32>, +q: List<&2, F32>, +k: List<&2, F32>, +v: List<&2, F32>, +qr: List<&2, F32>, +kr: List<&2, F32>, +nks: List<&2, List<&2, F32>>, +nvs: List<&2, List<&2, F32>>, +at: List<&2, F32>, ro: TA.BL<576n, 576n>) -> BR:
  match ro:
    case TA.BL{wo, +a}:
      b9(ln1, wq, wk, wv, wo, ln2, wg, wu, wd, KV{nks, nvs}, Tr{n1, q, k, v, qr, kr, at, a, Nil{}, Nil{}, Nil{}}, T.zip_with.add(x, a))

def b7(+ln1: List<&2, F32>, wq: TA.Bands<576n, 576n>, wk: TA.Bands<192n, 576n>, wv: TA.Bands<192n, 576n>, wo: TA.Bands<576n, 576n>, +ln2: List<&2, F32>, wg: TA.Bands<1536n, 576n>, wu: TA.Bands<1536n, 576n>, wd: TA.Bands<576n, 1536n>, +x: List<&2, F32>, +n1: List<&2, F32>, +q: List<&2, F32>, +k: List<&2, F32>, +v: List<&2, F32>, +qr: List<&2, F32>, +kr: List<&2, F32>, +nks: List<&2, List<&2, F32>>, +nvs: List<&2, List<&2, F32>>, +at: List<&2, F32>) -> BR:
  b8(ln1, wq, wk, wv, ln2, wg, wu, wd, x, n1, q, k, v, qr, kr, nks, nvs, at, TA.Bands.matvec_l(576n, 576n, wo, at))

# 5b. attention with the grown cache (it is handed on to the next position)
def b6b(+ln1: List<&2, F32>, wq: TA.Bands<576n, 576n>, wk: TA.Bands<192n, 576n>, wv: TA.Bands<192n, 576n>, wo: TA.Bands<576n, 576n>, +ln2: List<&2, F32>, wg: TA.Bands<1536n, 576n>, wu: TA.Bands<1536n, 576n>, wd: TA.Bands<576n, 1536n>, +x: List<&2, F32>, +n1: List<&2, F32>, +q: List<&2, F32>, +k: List<&2, F32>, +v: List<&2, F32>, +qr: List<&2, F32>, +kr: List<&2, F32>, +nks: List<&2, List<&2, F32>>, +nvs: List<&2, List<&2, F32>>) -> BR:
  b7(ln1, wq, wk, wv, wo, ln2, wg, wu, wd, x, n1, q, k, v, qr, kr, nks, nvs, attn(9n, 0n, qr, nks, nvs))

# 5. the rotated key and the value join the cache; attention of the 9 query heads over positions 0..t
def b6(+ln1: List<&2, F32>, wq: TA.Bands<576n, 576n>, wk: TA.Bands<192n, 576n>, wv: TA.Bands<192n, 576n>, wo: TA.Bands<576n, 576n>, +ln2: List<&2, F32>, wg: TA.Bands<1536n, 576n>, wu: TA.Bands<1536n, 576n>, wd: TA.Bands<576n, 1536n>, kv: KV, +x: List<&2, F32>, +n1: List<&2, F32>, +q: List<&2, F32>, +k: List<&2, F32>, +v: List<&2, F32>, +qr: List<&2, F32>, +kr: List<&2, F32>) -> BR:
  match kv:
    case KV{ks, vs}:
      b6b(ln1, wq, wk, wv, wo, ln2, wg, wu, wd, x, n1, q, k, v, qr, kr, List.append(&2, List<&2, F32>, ks, kr <> Nil{}), List.append(&2, List<&2, F32>, vs, v <> Nil{}))

def b5(+ln1: List<&2, F32>, wq: TA.Bands<576n, 576n>, wk: TA.Bands<192n, 576n>, wv: TA.Bands<192n, 576n>, wo: TA.Bands<576n, 576n>, +ln2: List<&2, F32>, wg: TA.Bands<1536n, 576n>, wu: TA.Bands<1536n, 576n>, wd: TA.Bands<576n, 1536n>, kv: KV, +t: Nat, +x: List<&2, F32>, +n1: List<&2, F32>, +q: List<&2, F32>, +k: List<&2, F32>, +v: List<&2, F32>, +cs: List<&2, F32>, +sn: List<&2, F32>) -> BR:
  b6(ln1, wq, wk, wv, wo, ln2, wg, wu, wd, kv, x, n1, q, k, v, rope(9n, q, cs, sn), rope(3n, k, cs, sn))

# 3. v = Wv · n1
def b4(+ln1: List<&2, F32>, wq: TA.Bands<576n, 576n>, wk: TA.Bands<192n, 576n>, wo: TA.Bands<576n, 576n>, +ln2: List<&2, F32>, wg: TA.Bands<1536n, 576n>, wu: TA.Bands<1536n, 576n>, wd: TA.Bands<576n, 1536n>, kv: KV, +t: Nat, +x: List<&2, F32>, +n1: List<&2, F32>, +q: List<&2, F32>, +k: List<&2, F32>, rv: TA.BL<192n, 576n>) -> BR:
  match rv:
    case TA.BL{wv, v}:
      b5(ln1, wq, wk, wv, wo, ln2, wg, wu, wd, kv, t, x, n1, q, k, v, cos.l(angles(F32.from_nat(t), inv_freq())), sin.l(angles(F32.from_nat(t), inv_freq())))

# 2. k = Wk · n1
def b3(+ln1: List<&2, F32>, wq: TA.Bands<576n, 576n>, wv: TA.Bands<192n, 576n>, wo: TA.Bands<576n, 576n>, +ln2: List<&2, F32>, wg: TA.Bands<1536n, 576n>, wu: TA.Bands<1536n, 576n>, wd: TA.Bands<576n, 1536n>, kv: KV, +t: Nat, +x: List<&2, F32>, +n1: List<&2, F32>, +q: List<&2, F32>, rk: TA.BL<192n, 576n>) -> BR:
  match rk:
    case TA.BL{wk, k}:
      b4(ln1, wq, wk, wo, ln2, wg, wu, wd, kv, t, x, n1, q, k, TA.Bands.matvec_l(192n, 576n, wv, n1))

# 1. q = Wq · n1
def b2(+ln1: List<&2, F32>, wk: TA.Bands<192n, 576n>, wv: TA.Bands<192n, 576n>, wo: TA.Bands<576n, 576n>, +ln2: List<&2, F32>, wg: TA.Bands<1536n, 576n>, wu: TA.Bands<1536n, 576n>, wd: TA.Bands<576n, 1536n>, kv: KV, +t: Nat, +x: List<&2, F32>, +n1: List<&2, F32>, rq: TA.BL<576n, 576n>) -> BR:
  match rq:
    case TA.BL{wq, q}:
      b3(ln1, wq, wv, wo, ln2, wg, wu, wd, kv, t, x, n1, q, TA.Bands.matvec_l(192n, 576n, wk, n1))

def b1(+ln1: List<&2, F32>, wq: TA.Bands<576n, 576n>, wk: TA.Bands<192n, 576n>, wv: TA.Bands<192n, 576n>, wo: TA.Bands<576n, 576n>, +ln2: List<&2, F32>, wg: TA.Bands<1536n, 576n>, wu: TA.Bands<1536n, 576n>, wd: TA.Bands<576n, 1536n>, kv: KV, +t: Nat, +x: List<&2, F32>, +n1: List<&2, F32>) -> BR:
  b2(ln1, wk, wv, wo, ln2, wg, wu, wd, kv, t, x, n1, TA.Bands.matvec_l(576n, 576n, wq, n1))

# the layer at position t (0-based) on input x, with the cache of positions 0..t-1
def block(lw: LW, kv: KV, +t: Nat, +x: List<&2, F32>) -> BR:
  match lw:
    case LW{+ln1, wq, wk, wv, wo, +ln2, wg, wu, wd}:
      b1(ln1, wq, wk, wv, wo, ln2, wg, wu, wd, kv, t, x, rmsnorm(x, ln1))
