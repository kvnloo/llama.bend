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
def b5(+ln1: List<&2, F32>, wq: TA.Bands<576n, 576n>, wk: TA.Bands<192n, 576n>, wv: TA.Bands<192n, 576n>, wo: TA.Bands<576n, 576n>, +ln2: List<&2, F32>, wg: TA.Bands<1536n, 576n>, wu: TA.Bands<1536n, 576n>, wd: TA.Bands<576n, 1536n>, kv: KV, +t: Nat, +x: List<&2, F32>, +n1: List<&2, F32>, +q: List<&2, F32>, +k: List<&2, F32>, +v: List<&2, F32>, +cs: List<&2, F32>, +sn: List<&2, F32>) -> BR:
  BR{Nil{}, kv, LW{ln1, wq, wk, wv, wo, ln2, wg, wu, wd}, Tr{n1, q, k, v, rope(9n, q, cs, sn), rope(3n, k, cs, sn), Nil{}, Nil{}, Nil{}, Nil{}, Nil{}}}

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
