# Loading SmolLM2's weights (data/w/<name>.bin, little-endian F32, row = output) into lists and row bands.
# Copied from bend-ml (demos/gpt2/load.bend and demos/gpt2/fast.bend, MIT): the 1 MB block reader that writes
# the bytes straight into each band's Array, the size check at the boundary, and the band depth per thread
# count. Only the paths changed.

import Base
import bend-ml-tensor-array@0.1.6.0/main.bend as TA

# 4 little-endian bytes -> F32 (U32 and F32 hold the same Word(32n): it is a bitcast)
def bitcast(u: U32) -> F32:
  match u:
    case U32{w}:
      F32{w}

def f32_le(+b0: U32, +b1: U32, +b2: U32, +b3: U32) -> F32:
  bitcast((b0 + (b1 * 256 : U32) + (b2 * 65536 : U32) + (b3 * 16777216 : U32) : U32))

def f32s(bs: List<&2, U32>) -> List<&2, F32>:
  match bs:
    case Con{b0, Con{b1, Con{b2, Con{b3, t}}}}:
      f32_le(b0, b1, b2, b3) <> f32s(t)
    case _:
      Nil{}

# reads the file in blocks of `blk` bytes; k blocks remaining; m is the block already read
def rf_loop(k: Nat, +off: U32, +blk: U32, m: File & Result<&1, &1, U32 & String, List<&2, U32>>, acc: List<&2, List<&2, F32>>) -> IO(List<&2, F32>):
  match k m:
    case 0n Tuple{f, r}:
      do IO<List<&2, F32>>:
        File.close(f)
        return List.concat(&2, F32, List.reverse(&2, List<&2, F32>, acc))
    case 1n+p Tuple{f, r}:
      do IO<List<&2, F32>>:
        bs : List<&2, U32> <- IO.pass(List<&2, U32>, r)
        m2 : File & Result<&1, &1, U32 & String, List<&2, U32>> <- File.read_at(f, (off + blk : U32), blk)
        rf_loop(p, (off + blk : U32), blk, m2, f32s(bs) <> acc)

def blocks_of(n: Nat, +per: Nat) -> Nat:
  Nat.div(Nat.add(n, Nat.sub(per, 1n)), per)

# n F32 numbers (little-endian) from a file
def read_floats(path: String, +n: Nat) -> IO(List<&2, F32>):
  do IO<List<&2, F32>>:
    f : File <- IO.try(File, File.open(path, "r"))
    m : File & Result<&1, &1, U32 & String, List<&2, U32>> <- File.read_at(f, 0, 1048576)
    fl : List<&2, F32> <- rf_loop(blocks_of(Nat.mul(n, 4n), 1048576n), 0, 1048576, m, Nil{})
    return List.take(&2, F32, fl, n)

# the weight matrices are kept in 2^d row bands (TA.Bands); each matrix · vector product runs its bands in
# parallel without copying the weights (bend-ml NOTES exp. 11). With 2 threads or fewer the bands cannot help,
# so d = 0: a single band, which computes what the sequential Mat.matmul_nt computes (bend-ml exp. 14).
def depth_sel(few: Bool) -> Nat:
  match few:
    case True{}:
      0n
    case False{}:
      4n

def depth_for(+threads: U32) -> Nat:
  depth_sel(U32.is_le(threads, 2))

def nm(+i: Nat, f: String) -> String:
  "h" ++ Nat.show(i) ++ "." ++ f

# ---- the boundary: every weight file must hold exactly the numbers its shape needs ----

def size_cmp(ok: Bool, +path: String, +got: U32, +want: Nat) -> IO(Unit):
  match ok:
    case True{}:
      IO.pure(Unit, Unit{})
    case False{}:
      IO.die(Unit, 1, path ++ ": expected " ++ Nat.show(want) ++ " bytes, found " ++ U32.show(got) ++ "; run make prep again")

def size_ok(m: File & Result<&1, &1, U32 & String, U32>, +want: Nat, +path: String) -> IO(Unit):
  match m:
    case Tuple{f, r}:
      do IO<Unit>:
        +n : U32 <- IO.pass(U32, r)
        File.close(f)
        size_cmp(U32.is_eq(n, U32.from_nat(want)), path, n, want)

# stops the program unless the file holds exactly n F32 numbers (n * 4 bytes)
def check_size(+path: String, +n: Nat) -> IO(Unit):
  do IO<Unit>:
    f : File <- IO.try(File, File.open(path, "r"))
    m : File & Result<&1, &1, U32 & String, U32> <- File.size(f)
    size_ok(m, Nat.mul(n, 4n), path)

def wpath(name: String) -> String:
  "data/w/" ++ name ++ ".bin"

def get_list(+name: String, +n: Nat) -> IO(List<&2, F32>):
  do IO<List<&2, F32>>:
    check_size(wpath(name), n)
    read_floats(wpath(name), n)

# reads the file in 1 MB blocks and writes each block straight into the matrix, so only one
# block is a list at any time
# ---- matrices kept in bands: each band is read from its own range of the file ----

# writes the little-endian F32 numbers held in bs (4 bytes each) at index i, without building a
# list of F32 first
def fill_bytes(bs: List<&2, U32>, +i: U32, a: Array<F32>) -> Array<F32>:
  match bs:
    case Con{+b0, Con{+b1, Con{+b2, Con{+b3, t}}}}:
      fill_bytes(t, (i + 1 : U32), Array.set(F32, a, i, f32_le(b0, b1, b2, b3)))
    case _:
      a

def fill_bytes_mat(-r: Nat, -c: Nat, +i: U32, bs: List<&2, U32>, m: TA.Mat<r, c>) -> TA.Mat<r, c>:
  match m:
    case TA.Mat{d}:
      TA.Mat{fill_bytes(bs, i, d)}

# reads blocks of at most 1 MB until rem bytes have been read, so a band never takes numbers
# that belong to the next one
def stream_exact(k: Nat, -r: Nat, -c: Nat, +rem: Nat, +off: U32, +idx: U32, m: File & Result<&1, &1, U32 & String, List<&2, U32>>, acc: TA.Mat<r, c>) -> IO(TA.Mat<r, c>):
  match k m:
    case 0n Tuple{f, rs}:
      do IO<TA.Mat<r, c>>:
        File.close(f)
        return acc
    case 1n+p Tuple{f, rs}:
      do IO<TA.Mat<r, c>>:
        bs : List<&2, U32> <- IO.pass(List<&2, U32>, rs)
        m2 : File & Result<&1, &1, U32 & String, List<&2, U32>> <- File.read_at(f, (off + 1048576 : U32), U32.from_nat(Nat.min(1048576n, Nat.sub(rem, 1048576n))))
        stream_exact(p, r, c, Nat.sub(rem, 1048576n), (off + 1048576 : U32), (idx + 262144 : U32), m2, fill_bytes_mat(r, c, idx, bs, acc))

# rr x c numbers starting at number `start` of the file, into a band of r rows (rr == r, see
# load_tree; the Array is sized for rr * c numbers as Mat.zeros would size it)
def get_band(path: String, -r: Nat, +rr: Nat, +c: Nat, +start: Nat) -> IO(TA.Mat<r, c>):
  do IO<TA.Mat<r, c>>:
    f : File <- IO.try(File, File.open(path, "r"))
    m : File & Result<&1, &1, U32 & String, List<&2, U32>> <- File.read_at(f, U32.from_nat(Nat.mul(start, 4n)), U32.from_nat(Nat.min(1048576n, Nat.mul(Nat.mul(rr, c), 4n))))
    stream_exact(blocks_of(Nat.mul(Nat.mul(rr, c), 4n), 1048576n), r, c, Nat.mul(Nat.mul(rr, c), 4n), U32.from_nat(Nat.mul(start, 4n)), 0, m, TA.Mat{Array.new(F32, TA.cap_depth(Nat.mul(rr, c)), 0.0)})

# fills the bands of b (whose shape comes from TA.Bands.zeros) in row order; rr is r at run time
def load_tree(+c: Nat, -r: Nat, b: TA.Bands<r, c>, +rr: Nat, -e: {rr == r : Nat}, +path: String, +start: Nat) -> IO(TA.Bands<r, c>):
  match b:
    case TA.BLeaf{m}:
      do IO<TA.Bands<r, c>>:
        band : TA.Mat<r, c> <- get_band(path, r, rr, c, start)
        return TA.BLeaf{band}
    case TA.BNode{x, y}:
      do IO<TA.Bands<r, c>>:
        x2 : TA.Bands<TA.half(r), c> <- load_tree(c, TA.half(r), x, TA.half(rr), TA.eq_half(rr, r, e), path, start)
        y2 : TA.Bands<Nat.sub(r, TA.half(r)), c> <- load_tree(c, Nat.sub(r, TA.half(r)), y, Nat.sub(rr, TA.half(rr)), TA.eq_rest(rr, r, e), path, Nat.add(start, Nat.mul(TA.half(rr), c)))
        return TA.BNode{x2, y2}

def get_bands(+name: String, +d: Nat, +r: Nat, +c: Nat) -> IO(TA.Bands<r, c>):
  do IO<TA.Bands<r, c>>:
    check_size(wpath(name), Nat.mul(r, c))
    load_tree(c, r, TA.Bands.zeros(d, r, c), r, {==}, wpath(name), 0n)
