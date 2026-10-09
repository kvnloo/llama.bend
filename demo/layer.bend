# One SmolLM2 layer (Llama architecture) in Bend: RMSNorm, RoPE, grouped-query attention with a key/value
# cache, SwiGLU. The order of the F32 operations follows transformers' modeling_llama.py (5.19), so the test
# (reference/test_layer.py) can ask for 1e-5. Heads are list slices here; their types come in milestone 6.

import Base
import bend-ml-tensor@0.1.2.0/main.bend as T
import bend-ml-tensor-array@0.1.6.0/main.bend as TA

# RMSNorm: g * (x * (1 / sqrt(mean(x²) + eps))), in this order (LlamaRMSNorm.forward)
def rmsnorm(+x: List<&2, F32>, +g: List<&2, F32>) -> List<&2, F32>:
  T.zip_with.mul(g, T.scale.l((1.0 / F32.sqrt(((T.sq_sum(x, 0.0) / 576.0 : F32) + 0.00001 : F32)) : F32), x))
