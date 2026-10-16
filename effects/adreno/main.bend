import Base

# A fixed foreign-C OpenCL example, not a Bend ! evaluator or a llama backend.
# The C effect prints all 576 read-back F32 words; the CPU oracle is separate.
def Adreno.gemv(path: String, pattern: U32) -> IO(Result<&1, &1, U32 & String, Unit>):
  import "./gemv.c"
  import "./gemv.js"

def run_mode(path: String, ones: Bool, alternating: Bool, basis: Bool) -> IO(Unit):
  match ones alternating basis:
    case True{} _ _:
      IO.try(Unit, Adreno.gemv(path, 0))
    case _ True{} _:
      IO.try(Unit, Adreno.gemv(path, 1))
    case _ _ True{}:
      IO.try(Unit, Adreno.gemv(path, 2))
    case _ _ _:
      IO.die(Unit, 22, "unknown mode: expected ones, alternating or basis-last")

def dispatch(args: List<String>) -> IO(Unit):
  match args:
    case Con{prog, Con{path, Con{+mode, Nil{}}}}:
      run_mode(path, String.eq(mode, "ones"), String.eq(mode, "alternating"), String.eq(mode, "basis-last"))
    case _:
      IO.die(Unit, 22, "usage: adreno-gemv <tensor-path> <ones|alternating|basis-last>")

def main() -> IO(Unit):
  do IO<Unit>:
    args : List<String> <- IO.args()
    dispatch(args)
