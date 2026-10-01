"""Model / runtime configuration shared by training, export and the watch app."""
from dataclasses import dataclass, asdict, field
import json


@dataclass
class NanoConfig:
    # architecture (Gemma-3 style: RMSNorm(1+w) sandwich norms, q/k norm, MQA, RoPE, GeGLU, tied emb)
    V: int = 1024          # vocabulary size (subset of Gemma's 262k vocab + ASCII chars + specials)
    d: int = 96            # hidden size          (Gemma 3 270M: 640)
    L: int = 4             # layers               (18)
    Hq: int = 3            # query heads          (4)
    hd: int = 32           # head dim             (256)  -- one shared KV head (MQA) like Gemma 3 270M
    ffn: int = 192         # GeGLU hidden         (2048)
    ctx: int = 48          # max tokens on the watch (prompt + answer); KV cache = 2*L*ctx*hd bytes
    theta: float = 10000.0
    eps: float = 1e-6
    # special token ids (fixed by vocab.py)
    pad: int = 0
    bos: int = 1
    eos: int = 2
    sep: int = 3
    # runtime defaults (sampling on the watch)
    topk: int = 16
    temp: float = 0.7
    rep: float = 1.3
    # storage
    B: int = 2048          # bytes per weight block / file on the watch (small = less heap fragmentation)

    def params(self) -> int:
        d, qd = self.d, self.Hq * self.hd
        per_layer = qd * d + 2 * self.hd * d + d * qd + 3 * d * self.ffn + 4 * d + 2 * self.hd
        return self.V * d + self.L * per_layer + d

    def heap_estimate(self) -> int:
        """Approximate JS-heap bytes held by the engine (measured with tests/jerry, +-10%)."""
        kv = 2 * self.L * self.ctx * self.hd + 8 * self.L * self.ctx
        vec = 4 * (5 * self.d + 3 * self.Hq * self.hd + self.ffn + 3 * self.ctx + 2 * self.hd + self.topk) + self.V
        return int((kv + vec) * 1.15) + 2000 + self.B

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=1)

    @staticmethod
    def from_dict(d):
        keys = NanoConfig.__dataclass_fields__.keys()
        return NanoConfig(**{k: v for k, v in d.items() if k in keys})


PRESETS = {
    # ~160K params: fastest, for slow watches / first experiments
    "pico": NanoConfig(V=768, d=64, L=3, Hq=2, hd=32, ffn=128, ctx=48),
    # ~300K params
    "micro": NanoConfig(V=1024, d=80, L=4, Hq=2, hd=32, ffn=160, ctx=48),
    # ~420K params: default if the benchmark allows ~2 s/token
    "nano": NanoConfig(V=1024, d=96, L=4, Hq=3, hd=32, ffn=192, ctx=48),
}


if __name__ == "__main__":
    for k, c in PRESETS.items():
        print(f"{k:6s} params={c.params():,}  heap~{c.heap_estimate():,} B")
