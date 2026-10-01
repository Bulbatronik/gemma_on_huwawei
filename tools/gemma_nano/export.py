"""Export a trained GemmaNano into the watch block format (see gn_engine.js header)."""
import hashlib
import json
import os

import numpy as np

from .config import NanoConfig


def _row_record(vec):
    vec = np.asarray(vec, dtype=np.float64)
    mx = float(np.abs(vec).max()) if vec.size else 0.0
    s16 = np.float16(mx / 127.0)
    s = float(s16)
    if s == 0.0:
        q = np.zeros(vec.shape, dtype=np.int8)
    else:
        q = np.clip(np.floor(vec / s + 0.5), -127, 127).astype(np.int8)
    return s16.tobytes() + q.tobytes()


class Packer:
    def __init__(self, B):
        self.B = B
        self.blocks = [bytearray()]

    def row(self, vec):
        rec = _row_record(vec)
        assert len(rec) <= self.B, "row larger than block"
        if len(self.blocks[-1]) + len(rec) > self.B:
            self.new_block()
        self.blocks[-1] += rec

    def rows(self, mat):
        for r in np.asarray(mat):
            self.row(r)

    def new_block(self):
        self.blocks[-1] += bytes(self.B - len(self.blocks[-1]))
        self.blocks.append(bytearray())

    def close(self):
        if len(self.blocks[-1]) == 0:
            self.blocks.pop()
        else:
            self.blocks[-1] += bytes(self.B - len(self.blocks[-1]))


def weights_from_torch(model):
    """Plain numpy dict of the student weights (norm weights converted to 1+w)."""
    sd = {k: v.detach().float().cpu().numpy() for k, v in model.state_dict().items()}
    return sd


def pack(c: NanoConfig, sd):
    p = Packer(c.B)
    p.rows(sd["embed.weight"])
    p.new_block()
    layer_start = len(p.blocks) - 1
    for l in range(c.L):
        pre = f"layers.{l}."
        g = lambda n: sd[pre + n]
        p.row(1.0 + g("input_layernorm.weight"))
        p.rows(g("q_proj.weight"))
        p.rows(g("k_proj.weight"))
        p.rows(g("v_proj.weight"))
        p.row(1.0 + g("q_norm.weight"))
        p.row(1.0 + g("k_norm.weight"))
        p.rows(g("o_proj.weight"))
        p.row(1.0 + g("post_attention_layernorm.weight"))
        p.row(1.0 + g("pre_feedforward_layernorm.weight"))
        gate, up = g("gate_proj.weight"), g("up_proj.weight")
        for j in range(c.ffn):
            p.row(gate[j])
            p.row(up[j])
        p.rows(g("down_proj.weight"))
        p.row(1.0 + g("post_feedforward_layernorm.weight"))
    p.row(1.0 + sd["norm.weight"])
    p.close()
    return [bytes(b) for b in p.blocks], layer_start


def export(model_or_sd, c: NanoConfig, vocab, out_dir, name="gemma-nano"):
    os.makedirs(out_dir, exist_ok=True)
    sd = model_or_sd if isinstance(model_or_sd, dict) else weights_from_torch(model_or_sd)
    blocks, layer_start = pack(c, sd)
    vbin = vocab.to_bin()
    h = hashlib.sha1()
    for b in blocks:
        h.update(b)
    h.update(vbin)
    meta = json.loads(c.to_json())
    meta.update({
        "name": name,
        "layerStart": layer_start,
        "nBlocks": len(blocks),
        "modelId": h.hexdigest()[:12],
        "params": c.params(),
    })
    with open(os.path.join(out_dir, "model.bin"), "wb") as f:
        for b in blocks:
            f.write(b)
    with open(os.path.join(out_dir, "vocab.bin"), "wb") as f:
        f.write(vbin)
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    vocab.save_json(os.path.join(out_dir, "vocab.json"))
    return meta
