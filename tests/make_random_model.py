"""Export a random-weight model (numpy only) for runtime tests."""
import argparse, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
from gemma_nano.config import PRESETS
from gemma_nano.export import export
from gemma_nano.vocab import build_standalone

ap = argparse.ArgumentParser()
ap.add_argument("--preset", default="pico")
ap.add_argument("--out", required=True)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--B", type=int, default=0)
a = ap.parse_args()
import dataclasses
c = PRESETS[a.preset]
if a.B: c = dataclasses.replace(c, B=a.B)
rng = np.random.default_rng(a.seed)
texts = ["hello how are you today", "the sun is a star", "tell me a joke about cats",
         "what is the capital of france", "i like to read books and drink tea"] * 3
voc = build_standalone(texts, c.V)
# pad the vocab to exactly V with synthetic tokens
toks = voc.tokens + ["#%d" % i for i in range(c.V - len(voc))]
from gemma_nano.vocab import Vocab
voc = Vocab(toks)
sd = {"embed.weight": rng.normal(0, 1, (c.V, c.d)), "norm.weight": rng.normal(0, .1, c.d)}
qd = c.Hq * c.hd
for l in range(c.L):
    p = f"layers.{l}."
    sd[p + "q_proj.weight"] = rng.normal(0, .15, (qd, c.d))
    sd[p + "k_proj.weight"] = rng.normal(0, .15, (c.hd, c.d))
    sd[p + "v_proj.weight"] = rng.normal(0, .15, (c.hd, c.d))
    sd[p + "o_proj.weight"] = rng.normal(0, .15, (c.d, qd))
    sd[p + "gate_proj.weight"] = rng.normal(0, .15, (c.ffn, c.d))
    sd[p + "up_proj.weight"] = rng.normal(0, .15, (c.ffn, c.d))
    sd[p + "down_proj.weight"] = rng.normal(0, .15, (c.d, c.ffn))
    for n in ["input_layernorm", "post_attention_layernorm", "pre_feedforward_layernorm", "post_feedforward_layernorm"]:
        sd[p + n + ".weight"] = rng.normal(0, .1, c.d)
    sd[p + "q_norm.weight"] = rng.normal(0, .1, c.hd)
    sd[p + "k_norm.weight"] = rng.normal(0, .1, c.hd)
meta = export(sd, c, voc, a.out, name="random-" + a.preset)
print(meta)
