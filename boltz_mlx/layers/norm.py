"""LayerNorm with the same parameters as mlx.nn.LayerNorm, routed to the custom Metal kernel."""

import mlx.nn as nn

from boltz_mlx.kernels import layer_norm


class LayerNorm(nn.LayerNorm):
    def __call__(self, x):
        return layer_norm(x, self.get("weight"), self.get("bias"), self.eps)

    def f16(self, x, eps_scale: float = 1.0):
        """Normalized output in fp16 (statistics fp32; the cast is fused in the kernel). eps_scale multiplies eps
        (used to undo an exact power-of-two pre-scaling of x: LN(c*x) with eps*c^2 == LN(x) with eps)."""
        import mlx.core as mx

        return layer_norm(x, self.get("weight"), self.get("bias"), self.eps * eps_scale, out_dtype=mx.float16)
