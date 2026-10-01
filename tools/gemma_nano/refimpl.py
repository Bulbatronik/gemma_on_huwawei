"""Bit-faithful(ish) NumPy model of the watch runtime (gn_engine.js).

Reads the exported blocks and follows the same integer arithmetic, so it can be
used (a) to measure the quality loss of the int8 runtime against the float
model and (b) as the oracle for the JS runtime tests.
"""
import json
import math
import os

import numpy as np

from .config import NanoConfig

f32 = np.float32


def js_round(x):
    return np.floor(np.asarray(x, dtype=np.float64) + 0.5)


def quant(v):
    v = np.asarray(v, dtype=np.float64)
    mx = float(np.max(np.abs(v))) if v.size else 0.0
    if mx == 0:
        return np.zeros(v.shape, np.int64), 0.0
    return js_round(v * (127.0 / mx)).astype(np.int64), mx / 127.0


def rms(v, w, eps):
    v = np.asarray(v, dtype=np.float64)
    r = 1.0 / math.sqrt(float(np.dot(v, v)) / v.size + eps)
    return (v * r * w).astype(f32).astype(np.float64)


def gelu(a):
    z = 1.5957691216057308 * (a + 0.044715 * a * a * a)
    if z > 30:
        return a
    if z < -30:
        return 0.0
    return a / (1 + math.exp(-z))


class RowStream:
    def __init__(self, data, B, start_block):
        self.d, self.B, self.blk, self.off = data, B, start_block, 0

    def rows(self, n_rows, cols):
        need = 2 + cols
        scales = np.empty(n_rows)
        out = np.empty((n_rows, cols), np.int64)
        for r in range(n_rows):
            if self.off + need > self.B:
                self.blk += 1
                self.off = 0
            o = self.blk * self.B + self.off
            scales[r] = float(np.frombuffer(self.d[o:o + 2], np.float16)[0])
            out[r] = np.frombuffer(self.d[o + 2:o + need], np.int8)
            self.off += need
        return out, scales


class RefModel:
    def __init__(self, model_dir):
        with open(os.path.join(model_dir, "meta.json")) as f:
            self.meta = json.load(f)
        self.c = NanoConfig.from_dict(self.meta)
        with open(os.path.join(model_dir, "model.bin"), "rb") as f:
            self.data = f.read()
        c = self.c
        # decode the whole program once (it is the same for every token)
        s = RowStream(self.data, c.B, self.meta["layerStart"])
        self.layers = []
        qd = c.Hq * c.hd
        for _ in range(c.L):
            L = {}
            L["n1"] = self._norm(s, c.d)
            L["q"] = s.rows(qd, c.d)
            L["k"] = s.rows(c.hd, c.d)
            L["v"] = s.rows(c.hd, c.d)
            L["qn"] = self._norm(s, c.hd)
            L["kn"] = self._norm(s, c.hd)
            L["o"] = s.rows(c.d, qd)
            L["n2"] = self._norm(s, c.d)
            L["n3"] = self._norm(s, c.d)
            L["gu"] = s.rows(2 * c.ffn, c.d)
            L["down"] = s.rows(c.d, c.ffn)
            L["n4"] = self._norm(s, c.d)
            self.layers.append(L)
        self.fn = self._norm(s, c.d)
        e = RowStream(self.data, c.B, 0)
        self.emb = e.rows(c.V, c.d)
        self.inv = (c.theta ** (-2.0 * np.arange(c.hd // 2) / c.hd)).astype(f32).astype(np.float64)
        self.reset()

    @staticmethod
    def _norm(s, n):
        q, sc = s.rows(1, n)
        return (q[0] * sc[0]).astype(f32).astype(np.float64)

    def reset(self):
        c = self.c
        self.kc = np.zeros((c.L, c.ctx, c.hd), np.int64)
        self.vc = np.zeros((c.L, c.ctx, c.hd), np.int64)
        self.ks = np.zeros((c.L, c.ctx))
        self.vs = np.zeros((c.L, c.ctx))

    @staticmethod
    def _mm(w, xq, xs):
        q, sc = w
        return (q @ xq) * sc * xs

    def step(self, tok, pos):
        """Run one token; returns the float logits (before repetition penalty)."""
        c = self.c
        eq, es = self.emb
        x = (eq[tok] * (es[tok] * math.sqrt(c.d))).astype(f32).astype(np.float64)
        cs = np.cos(pos * self.inv).astype(f32).astype(np.float64)
        sn = np.sin(pos * self.inv).astype(f32).astype(np.float64)
        half = c.hd // 2

        def rope(v):
            a, b = v[:half].copy(), v[half:].copy()
            return np.concatenate([a * cs - b * sn, b * cs + a * sn]).astype(f32).astype(np.float64)

        for l, L in enumerate(self.layers):
            h = rms(x, L["n1"], c.eps)
            hq, hs = quant(h)
            q = self._mm(L["q"], hq, hs).astype(f32).astype(np.float64)
            k = self._mm(L["k"], hq, hs).astype(f32).astype(np.float64)
            v = self._mm(L["v"], hq, hs).astype(f32).astype(np.float64)
            q = np.concatenate([rms(q[i * c.hd:(i + 1) * c.hd], L["qn"], c.eps) for i in range(c.Hq)])
            k = rms(k, L["kn"], c.eps)
            q = np.concatenate([rope(q[i * c.hd:(i + 1) * c.hd]) for i in range(c.Hq)])
            k = rope(k)
            kq, kss = quant(k)
            self.kc[l, pos], self.ks[l, pos] = kq, f32(kss)
            vq, vss = quant(v)
            self.vc[l, pos], self.vs[l, pos] = vq, f32(vss)
            att = np.zeros(c.Hq * c.hd)
            for hh in range(c.Hq):
                qv = q[hh * c.hd:(hh + 1) * c.hd]
                qq, qs = quant(qv)
                sc = (self.kc[l, :pos + 1] @ qq) * qs * self.ks[l, :pos + 1] * (1 / math.sqrt(c.hd))
                sc = sc.astype(f32).astype(np.float64)
                p = np.exp(sc - sc.max()).astype(f32).astype(np.float64)
                wv = (p / p.sum() * self.vs[l, :pos + 1]).astype(f32).astype(np.float64)
                wmax = wv.max()
                wi = js_round(wv / wmax * 511).astype(np.int64) if wmax > 0 else np.zeros(pos + 1, np.int64)
                att[hh * c.hd:(hh + 1) * c.hd] = (wi @ self.vc[l, :pos + 1]) * (wmax / 511)
            att = att.astype(f32).astype(np.float64)
            aq, as_ = quant(att)
            o = self._mm(L["o"], aq, as_).astype(f32).astype(np.float64)
            x = (x + rms(o, L["n2"], c.eps)).astype(f32).astype(np.float64)
            h = rms(x, L["n3"], c.eps)
            hq, hs = quant(h)
            gu = self._mm(L["gu"], hq, hs)
            m = np.array([gelu(gu[2 * j]) * gu[2 * j + 1] for j in range(c.ffn)]).astype(f32).astype(np.float64)
            mq, ms = quant(m)
            y = self._mm(L["down"], mq, ms).astype(f32).astype(np.float64)
            x = (x + rms(y, L["n4"], c.eps)).astype(f32).astype(np.float64)
        h = rms(x, self.fn, c.eps)
        hq, hs = quant(h)
        return (eq @ hq) * es * hs

    def generate_greedy(self, ids, max_new=40, rep=None):
        c = self.c
        rep = c.rep if rep is None else rep
        self.reset()
        seen = np.zeros(c.V, bool)
        out = []
        pos = 0
        for i, t in enumerate(ids[:-1]):
            self.step(t, pos)
            pos += 1
        cur = ids[-1]
        while pos < c.ctx and len(out) < max_new:
            lg = self.step(cur, pos)
            pos += 1
            lg = np.where(seen, np.where(lg > 0, lg / rep, lg * rep), lg)
            lg[[c.bos, c.sep, c.pad]] = -np.inf
            cur = int(np.argmax(lg))
            out.append(cur)
            seen[cur] = True
            if cur == c.eos:
                break
        return out
