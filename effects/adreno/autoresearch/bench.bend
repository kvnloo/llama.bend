import Base

# Foreign-C profiling only: never a native ! evaluator or CPU fallback.
# File access, OpenCL work, validation and evidence emission use io_work's helper.
def Bench.run(path: String, source: String, params: String) -> IO(Result<&1, &1, U32 & String, Unit>):
  import "./bench.c"
  import "./bench.js"

def dispatch(args: List<String>) -> IO(Unit):
  match args:
    case Con{program, Con{path, Con{source, Con{params, Nil{}}}}}:
      IO.try(Unit, Bench.run(path, source, params))
    case _:
      IO.die(Unit, 22, "usage: adreno-autoresearch <tensor-path> <kernel-source-path> <pattern:global:local:samples:warmups:batch>")

def main() -> IO(Unit):
  do IO<Unit>:
    args : List<String> <- IO.args()
    dispatch(args)
