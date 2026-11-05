"""MLX port of the primitives used by boltz triangular attention (inference only).

Upstream `Linear` / `LayerNorm` only add custom initialisation (training concern) and
bf16 handling; both are plain linear / layer-norm (eps=1e-5) at fp32, so MLX's are reused.
"""

import math

import mlx.core as mx
import mlx.nn as nn
from boltz_mlx.layers.norm import LayerNorm as _FastLN

Linear = nn.Linear


def LayerNorm(c_in: int, eps: float = 1e-5) -> nn.LayerNorm:  # noqa: N802
    return _FastLN(c_in, eps=eps)


class Attention(nn.Module):
    """Standard multi-head gated attention allowing multiple additive biases."""

    def __init__(self, c_q: int, c_k: int, c_v: int, c_hidden: int, no_heads: int, gating: bool = True):
        super().__init__()
        self.c_q = c_q
        self.c_k = c_k
        self.c_v = c_v
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.gating = gating

        # c_hidden is the per-head channel dim here (upstream comment notwithstanding)
        self.linear_q = nn.Linear(c_q, c_hidden * no_heads, bias=False)
        self.linear_k = nn.Linear(c_k, c_hidden * no_heads, bias=False)
        self.linear_v = nn.Linear(c_v, c_hidden * no_heads, bias=False)
        self.linear_o = nn.Linear(c_hidden * no_heads, c_q, bias=False)
        if gating:
            self.linear_g = nn.Linear(c_q, c_hidden * no_heads, bias=False)

    def __call__(
        self,
        q_x: mx.array,
        kv_x: mx.array,
        tri_bias: mx.array,
        mask_bias: mx.array,
        mask: mx.array = None,
        use_kernels: bool = False,
    ) -> mx.array:
        """q_x: [*, Q, C_q]; kv_x: [*, K, C_k]; biases broadcastable to [*, H, Q, K]."""
        H = self.no_heads
        q = self.linear_q(q_x)
        k = self.linear_k(kv_x)
        v = self.linear_v(kv_x)

        # [*, H, Q/K, C_hidden]
        q = q.reshape(*q.shape[:-1], H, -1).swapaxes(-2, -3)
        k = k.reshape(*k.shape[:-1], H, -1).swapaxes(-2, -3)
        v = v.reshape(*v.shape[:-1], H, -1).swapaxes(-2, -3)

        q = q / math.sqrt(self.c_hidden)

        a = mx.matmul(q, k.swapaxes(-1, -2))  # [*, H, Q, K]
        for b in (mask_bias, tri_bias):
            a = a + b
        a = mx.softmax(a, axis=-1)
        o = mx.matmul(a, v)  # [*, H, Q, C_hidden]
        o = o.swapaxes(-2, -3)  # [*, Q, H, C_hidden]

        if self.gating:
            g = mx.sigmoid(self.linear_g(q_x))
            g = g.reshape(*g.shape[:-1], H, -1)
            o = o * g

        o = o.reshape(*o.shape[:-2], -1)
        return self.linear_o(o)
