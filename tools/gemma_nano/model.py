"""PyTorch definition of the GemmaNano student (a shrunken Gemma 3 block stack).

Optional fake quantisation (QAT) reproduces the watch runtime's arithmetic:
per-row int8 weights, per-vector int8 activations at every matmul input.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import NanoConfig


def fq_rows(w):
    """Per-row symmetric int8 fake-quant with straight-through gradient."""
    s = w.detach().abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / 127.0
    q = torch.round(w / s).clamp(-127, 127) * s
    return w + (q - w).detach()


def fq_vec(x):
    s = x.detach().abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / 127.0
    q = torch.round(x / s).clamp(-127, 127) * s
    return x + (q - x).detach()


class RMSNorm(nn.Module):
    def __init__(self, n, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(n))
        self.eps = eps

    def forward(self, x):
        xf = x.float()
        y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (y * (1.0 + self.weight.float())).type_as(x)


class QLinear(nn.Linear):
    qat = False

    def forward(self, x):
        if self.qat:
            return F.linear(fq_vec(x), fq_rows(self.weight))
        return F.linear(x, self.weight)


def rope_cache(ctx, hd, theta, device=None):
    inv = theta ** (-torch.arange(0, hd // 2, dtype=torch.float32, device=device) * 2 / hd)
    t = torch.arange(ctx, dtype=torch.float32, device=device)
    ang = torch.outer(t, inv)
    return torch.cos(ang), torch.sin(ang)


def apply_rope(x, cos, sin):
    # x: (B, H, T, hd); HF rotate_half convention
    h = x.shape[-1] // 2
    a, b = x[..., :h], x[..., h:]
    return torch.cat([a * cos - b * sin, b * cos + a * sin], dim=-1)


class Block(nn.Module):
    def __init__(self, c: NanoConfig):
        super().__init__()
        self.c = c
        qd = c.Hq * c.hd
        self.input_layernorm = RMSNorm(c.d, c.eps)
        self.q_proj = QLinear(c.d, qd, bias=False)
        self.k_proj = QLinear(c.d, c.hd, bias=False)
        self.v_proj = QLinear(c.d, c.hd, bias=False)
        self.q_norm = RMSNorm(c.hd, c.eps)
        self.k_norm = RMSNorm(c.hd, c.eps)
        self.o_proj = QLinear(qd, c.d, bias=False)
        self.post_attention_layernorm = RMSNorm(c.d, c.eps)
        self.pre_feedforward_layernorm = RMSNorm(c.d, c.eps)
        self.gate_proj = QLinear(c.d, c.ffn, bias=False)
        self.up_proj = QLinear(c.d, c.ffn, bias=False)
        self.down_proj = QLinear(c.ffn, c.d, bias=False)
        self.post_feedforward_layernorm = RMSNorm(c.d, c.eps)

    def forward(self, x, cos, sin):
        c = self.c
        B, T, _ = x.shape
        h = self.input_layernorm(x)
        q = self.q_proj(h).view(B, T, c.Hq, c.hd).transpose(1, 2)
        k = self.k_proj(h).view(B, T, 1, c.hd).transpose(1, 2)
        v = self.v_proj(h).view(B, T, 1, c.hd).transpose(1, 2)
        q = self.q_norm(q)
        k = self.k_norm(k)
        q = apply_rope(q, cos[:T], sin[:T])
        k = apply_rope(k, cos[:T], sin[:T])
        a = F.scaled_dot_product_attention(q, k.expand(-1, c.Hq, -1, -1), v.expand(-1, c.Hq, -1, -1),
                                           is_causal=True, scale=1.0 / math.sqrt(c.hd))
        a = a.transpose(1, 2).reshape(B, T, c.Hq * c.hd)
        x = x + self.post_attention_layernorm(self.o_proj(a))
        h = self.pre_feedforward_layernorm(x)
        m = F.gelu(self.gate_proj(h), approximate="tanh") * self.up_proj(h)
        x = x + self.post_feedforward_layernorm(self.down_proj(m))
        return x


class GemmaNano(nn.Module):
    def __init__(self, c: NanoConfig):
        super().__init__()
        self.c = c
        self.embed = nn.Embedding(c.V, c.d)
        self.layers = nn.ModuleList([Block(c) for _ in range(c.L)])
        self.norm = RMSNorm(c.d, c.eps)
        cos, sin = rope_cache(max(c.ctx, 256), c.hd, c.theta)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        nn.init.normal_(self.embed.weight, std=0.02)
        for l in self.layers:
            for m in (l.q_proj, l.k_proj, l.v_proj, l.o_proj, l.gate_proj, l.up_proj, l.down_proj):
                nn.init.normal_(m.weight, std=0.02)

    def set_qat(self, on: bool):
        for m in self.modules():
            if isinstance(m, QLinear):
                m.qat = on
        self._qat = on

    def forward(self, ids):
        E = self.embed.weight
        if getattr(self, "_qat", False):
            E = fq_rows(E)
        x = F.embedding(ids, E) * math.sqrt(self.c.d)
        for l in self.layers:
            x = l(x, self.cos, self.sin)
        h = self.norm(x)
        if getattr(self, "_qat", False):
            h = fq_vec(h)
        return h @ E.t()

    @torch.no_grad()
    def generate(self, ids, max_new=40, temp=0.0, eos=2):
        ids = list(ids)
        for _ in range(max_new):
            if len(ids) >= self.c.ctx:
                break
            lg = self(torch.tensor([ids], device=self.embed.weight.device))[0, -1]
            lg[[0, 1, 3]] = -1e9
            nxt = int(torch.argmax(lg)) if temp <= 0 else int(torch.multinomial(torch.softmax(lg / temp, -1), 1))
            ids.append(nxt)
            if nxt == eos:
                break
        return ids
