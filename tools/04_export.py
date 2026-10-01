"""Step 4: export the student to the watch format and check the int8 runtime.

    python tools/04_export.py --work work/pico --out out/pico

Writes out/<name>/{model.bin, vocab.bin, vocab.json, meta.json} and compares the
float PyTorch model with the NumPy model of the int8 watch runtime.
"""
import argparse
import json
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gemma_nano.config import NanoConfig  # noqa: E402
from gemma_nano.export import export  # noqa: E402
from gemma_nano.model import GemmaNano  # noqa: E402
from gemma_nano.refimpl import RefModel  # noqa: E402
from gemma_nano.vocab import Vocab, normalize_prompt  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="work/pico")
    ap.add_argument("--out", default="out/pico")
    ap.add_argument("--name", default=None)
    ap.add_argument("--check", type=int, default=40, help="validation examples for the int8 check")
    a = ap.parse_args()
    cfg = NanoConfig.from_dict(json.load(open(os.path.join(a.work, "config.json"))))
    vocab = Vocab.load_json(os.path.join(a.work, "vocab.json"))
    model = GemmaNano(cfg)
    model.load_state_dict(torch.load(os.path.join(a.work, "student.pt"), map_location="cpu"))
    model.eval()
    meta = export(model, cfg, vocab, a.out, name=a.name or os.path.basename(a.out.rstrip("/")))
    size = os.path.getsize(os.path.join(a.out, "model.bin"))
    print(f"exported {meta['name']}: {meta['params']:,} params, {meta['nBlocks']} blocks x {cfg.B} B "
          f"= {size / 1024:.0f} KB, id {meta['modelId']}")

    ref = RefModel(a.out)
    data_p = os.path.join(a.work, "dataset.pt")
    if os.path.exists(data_p) and a.check:
        data = torch.load(data_p)[:a.check]
        ce_f, ce_q, n, agree = 0.0, 0.0, 0, 0
        for it in data:
            s = it["s"]
            with torch.no_grad():
                lf = torch.log_softmax(model(torch.tensor([s]))[0].double(), -1).numpy()
            ref.reset()
            for pos in range(len(s) - 1):
                lq = ref.step(s[pos], pos)
                if pos >= it["resp_start"] - 1:
                    lq = lq - lq.max()
                    lq = lq - math.log(np.exp(lq).sum())
                    ce_f -= lf[pos, s[pos + 1]]
                    ce_q -= lq[s[pos + 1]]
                    agree += int(np.argmax(lq) == np.argmax(lf[pos]))
                    n += 1
        print(f"answer-token CE: float {ce_f / n:.3f}  int8-runtime {ce_q / n:.3f}  "
              f"(top-1 agreement {100 * agree / n:.1f}%)")
    for p in ["hello", "what is the sun", "tell me a joke about cats", "how do i sleep better"]:
        ids = [cfg.bos] + vocab.encode_greedy(normalize_prompt(p)) + [cfg.sep]
        out = ref.generate_greedy(ids, max_new=cfg.ctx - len(ids))
        print(f"  [int8] {p!r:32s} -> {vocab.decode([t for t in out if t > 3])!r}")


if __name__ == "__main__":
    main()
