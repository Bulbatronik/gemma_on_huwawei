"""Step 2: build the pruned vocabulary, the aligned distillation dataset and the
compressed-Gemma initialisation of the student.

    python tools/02_prepare.py --data work/teacher.jsonl --preset pico --out work/pico

Outputs in --out:  vocab.json, config.json, dataset.pt, init.pt
"""
import argparse
import dataclasses
import json
import os
import random
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gemma_nano.config import PRESETS  # noqa: E402
from gemma_nano.gemma_init import compress_gemma  # noqa: E402
from gemma_nano.prompts import TEACHER_INSTRUCTION  # noqa: E402
from gemma_nano.vocab import build_from_gemma, gemma_piece_text_fn, normalize_prompt  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default="google/gemma-3-270m-it")
    ap.add_argument("--data", default="work/teacher.jsonl")
    ap.add_argument("--preset", default="pico", choices=list(PRESETS))
    ap.add_argument("--out", default="work/pico")
    ap.add_argument("--init", default="gemma", choices=["gemma", "random"])
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    cfg = PRESETS[a.preset]

    tok = AutoTokenizer.from_pretrained(a.teacher)
    eot = tok.convert_tokens_to_ids("<end_of_turn>")
    rows = [json.loads(l) for l in open(a.data)]
    print(f"{len(rows)} teacher examples")

    # ---- vocabulary --------------------------------------------------------
    texts = [r["answer"] for r in rows] + [normalize_prompt(r["prompt"]) for r in rows]
    vocab = build_from_gemma(tok, texts, cfg.V, eot)
    if len(vocab) < cfg.V:
        print(f"note: corpus only supports {len(vocab)} tokens, shrinking V")
        cfg = dataclasses.replace(cfg, V=len(vocab))
    vocab.save_json(os.path.join(a.out, "vocab.json"))
    with open(os.path.join(a.out, "config.json"), "w") as f:
        f.write(cfg.to_json())
    print(f"vocab: {len(vocab)} tokens, longest '{max(vocab.tokens, key=len)}'")

    # ---- aligned dataset ------------------------------------------------------
    piece = gemma_piece_text_fn(tok)
    data, dropped, n_kd, n_tgt = [], 0, 0, 0
    for r in rows:
        p_ids = vocab.encode_greedy(normalize_prompt(r["prompt"]))
        g_ids = tok(r["answer"], add_special_tokens=False)["input_ids"]
        s_resp, first, ins = vocab.encode_gemma_aligned(g_ids, piece)
        s = [cfg.bos] + p_ids + [cfg.sep] + s_resp + [cfg.eos]
        if len(s) > cfg.ctx:
            dropped += 1
            continue
        chat = tok.apply_chat_template([{"role": "user", "content": TEACHER_INSTRUCTION + r["prompt"]}],
                                       tokenize=False, add_generation_prompt=True)
        t_prefix = tok(chat, add_special_tokens=False)["input_ids"]
        t = t_prefix + g_ids + [eot]
        Ps, Pt = len(p_ids) + 2, len(t_prefix)
        pairs = [(Ps + first[j] - 1, Pt + j - 1) for j in range(len(g_ids)) if ins[j]]
        pairs.append((Ps + len(s_resp) - 1, Pt + len(g_ids) - 1))       # <eos> <-> <end_of_turn>
        data.append({"s": s, "resp_start": Ps, "t": t, "pairs": pairs})
        n_kd += len(pairs)
        n_tgt += len(s_resp) + 1
    random.Random(0).shuffle(data)
    print(f"dataset: {len(data)} kept, {dropped} dropped (longer than ctx={cfg.ctx}); "
          f"{n_tgt} target tokens, {n_kd} ({100 * n_kd / max(1, n_tgt):.0f}%) with teacher logits")
    torch.save(data, os.path.join(a.out, "dataset.pt"))

    # ---- compressed-Gemma initialisation ---------------------------------------
    if a.init == "gemma":
        teacher = AutoModelForCausalLM.from_pretrained(a.teacher, dtype=torch.float32)
        sd = compress_gemma(teacher, vocab, cfg)
        torch.save(sd, os.path.join(a.out, "init.pt"))
        print("wrote init.pt (compressed Gemma weights)")


if __name__ == "__main__":
    main()
