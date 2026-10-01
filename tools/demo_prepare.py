"""Gemma-free demo / smoke-test model data (no Hugging Face access needed).

Builds a dialogue dataset from Tiny Shakespeare (each speech answered by the
next speech), a standalone vocabulary, and writes a work dir that
03_distill.py can train with --kd 0.  The resulting model only exists to
validate the watch pipeline end to end (install, memory, speed, UI); the real
model comes from the Gemma distillation (01 -> 02 -> 03 -> 04).

    curl -o work/shakespeare.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
    python tools/demo_prepare.py --text work/shakespeare.txt --out work/demo --preset pico
    python tools/03_distill.py --work work/demo --kd 0 --steps 3000
    python tools/04_export.py --work work/demo --out out/demo
"""
import argparse
import dataclasses
import os
import random
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gemma_nano.config import PRESETS  # noqa: E402
from gemma_nano.vocab import build_standalone, clean_answer, normalize_prompt  # noqa: E402


def speeches(text):
    out, cur = [], []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            if cur:
                out.append(" ".join(cur))
            cur = []
        elif line.endswith(":") and len(line) < 40 and not cur:
            continue                                   # speaker name
        else:
            cur.append(line)
    if cur:
        out.append(" ".join(cur))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", default="work/shakespeare.txt")
    ap.add_argument("--out", default="work/demo")
    ap.add_argument("--preset", default="pico", choices=list(PRESETS))
    a = ap.parse_args()
    cfg = PRESETS[a.preset]
    text = open(a.text).read()
    sp = speeches(text)
    pairs = []
    for q, r in zip(sp, sp[1:]):                      # speech -> reply
        ans = clean_answer(r)
        p = normalize_prompt(q.split(".")[0])[:60]
        if ans and p:
            pairs.append((p, ans))
    lines = [l.strip() for l in text.split("\n") if l.strip() and not l.strip().endswith(":")]
    for q, r in zip(lines, lines[1:]):                 # line -> next line (more data)
        ans = clean_answer(r, max_chars=120, max_sentences=1)
        p = normalize_prompt(q)[:60]
        if ans and p:
            pairs.append((p, ans))
    print(f"{len(pairs)} dialogue pairs")
    vocab = build_standalone([x for p in pairs for x in p], cfg.V)
    cfg = dataclasses.replace(cfg, V=len(vocab))
    os.makedirs(a.out, exist_ok=True)
    vocab.save_json(os.path.join(a.out, "vocab.json"))
    open(os.path.join(a.out, "config.json"), "w").write(cfg.to_json())
    data = []
    for p, ans in pairs:
        pid = vocab.encode_greedy(p)
        s = [cfg.bos] + pid + [cfg.sep] + vocab.encode_greedy(ans) + [cfg.eos]
        if len(s) <= cfg.ctx:
            data.append({"s": s, "resp_start": len(pid) + 2, "t": [], "pairs": []})
    random.Random(0).shuffle(data)
    torch.save(data, os.path.join(a.out, "dataset.pt"))
    print(f"{len(data)} examples fit ctx={cfg.ctx}; vocab {len(vocab)}")


if __name__ == "__main__":
    main()
