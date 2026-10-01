"""Step 3: distil Gemma 3 270M into the compressed student.

Loss = cross-entropy on Gemma's answers (sequence-level KD)
     + alpha * KL(teacher || student) on positions where Gemma's own token is in
       the pruned vocab (logit KD; teacher distribution renormalised to the subset).
The last part of training turns on fake int8 quantisation (QAT) so the model is
robust to the watch runtime's int8 arithmetic.

    python tools/03_distill.py --work work/pico --steps 6000
"""
import argparse
import math
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gemma_nano.config import NanoConfig  # noqa: E402
from gemma_nano.model import GemmaNano  # noqa: E402
from gemma_nano.vocab import Vocab, normalize_prompt  # noqa: E402

DEMO_PROMPTS = ["hello", "how are you", "what is the sun", "tell me a joke about cats", "how do i sleep better",
                "what is a dog", "why is the sky blue", "give me a fun fact about space"]


def batches(data, bs, ctx, gen):
    idx = torch.randperm(len(data), generator=gen).tolist()
    for i in range(0, len(idx) - bs + 1, bs):
        yield [data[j] for j in idx[i:i + bs]]


def collate(items, ctx, dev):
    B = len(items)
    s = torch.zeros(B, ctx, dtype=torch.long)
    mask = torch.zeros(B, ctx)
    for i, it in enumerate(items):
        n = len(it["s"])
        s[i, :n] = torch.tensor(it["s"])
        mask[i, it["resp_start"] - 1:n - 1] = 1.0      # predict answer tokens + <eos>
    return s.to(dev), mask.to(dev)


def collate_teacher(items, dev, pad_id):
    T = max(len(it["t"]) for it in items)
    t = torch.full((len(items), T), pad_id, dtype=torch.long)
    att = torch.zeros(len(items), T, dtype=torch.long)
    sb, sp, tp = [], [], []
    for i, it in enumerate(items):
        t[i, :len(it["t"])] = torch.tensor(it["t"])
        att[i, :len(it["t"])] = 1
        for a, b in it["pairs"]:
            sb.append(i), sp.append(a), tp.append(b)
    return t.to(dev), att.to(dev), torch.tensor(sb, device=dev), torch.tensor(sp, device=dev), torch.tensor(tp, device=dev)


@torch.no_grad()
def evaluate(model, data, cfg, dev, bs=256):
    model.eval()
    tot, n = 0.0, 0.0
    for i in range(0, len(data), bs):
        s, mask = collate(data[i:i + bs], cfg.ctx, dev)
        lg = model(s)[:, :-1]
        ce = F.cross_entropy(lg.reshape(-1, cfg.V), s[:, 1:].reshape(-1), reduction="none")
        tot += float((ce * mask[:, :-1].reshape(-1)).sum())
        n += float(mask.sum())
    model.train()
    return tot / max(n, 1)


def samples(model, vocab, cfg, dev):
    model.eval()
    for p in DEMO_PROMPTS:
        ids = [cfg.bos] + vocab.encode_greedy(normalize_prompt(p)) + [cfg.sep]
        out = model.generate(ids, max_new=cfg.ctx - len(ids), temp=0.0, eos=cfg.eos)
        print(f"   {p!r:40s} -> {vocab.decode([t for t in out[len(ids):] if t > 3])!r}")
    model.train()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="work/pico")
    ap.add_argument("--teacher", default="google/gemma-3-270m-it")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--kd", type=float, default=1.0, help="logit-KD weight (0 = no teacher at train time)")
    ap.add_argument("--kd-temp", type=float, default=1.0)
    ap.add_argument("--qat-frac", type=float, default=0.25, help="final fraction of steps with fake-int8")
    ap.add_argument("--no-init", action="store_true", help="ignore init.pt (random init, for comparison)")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    cfg = NanoConfig.from_dict(__import__("json").load(open(os.path.join(a.work, "config.json"))))
    vocab = Vocab.load_json(os.path.join(a.work, "vocab.json"))
    data = torch.load(os.path.join(a.work, "dataset.pt"))
    n_val = max(64, len(data) // 50)
    val, train = data[:n_val], data[n_val:]
    print(f"train {len(train)}  val {len(val)}  params {cfg.params():,}  device {dev}")

    model = GemmaNano(cfg).to(dev)
    init = os.path.join(a.work, "init.pt")
    if os.path.exists(init) and not a.no_init:
        missing = model.load_state_dict(torch.load(init), strict=False)
        print("initialised from compressed Gemma", "(missing: %s)" % missing.missing_keys if missing.missing_keys else "")
    print(f"val CE at init: {evaluate(model, val, cfg, dev):.3f}")

    teacher, G, S_idx = None, None, None
    if a.kd > 0:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        ttok = AutoTokenizer.from_pretrained(a.teacher)
        teacher = AutoModelForCausalLM.from_pretrained(
            a.teacher, dtype=torch.bfloat16 if dev == "cuda" else torch.float32).to(dev).eval()
        S_idx = torch.tensor([i for i, g in enumerate(vocab.gemma_ids) if g >= 0], device=dev)
        G = torch.tensor([vocab.gemma_ids[i] for i in S_idx.tolist()], device=dev)
        pad_id = ttok.pad_token_id or 0

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=0.01)
    warm = 200
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, s / a.steps)))))
    gen = torch.Generator().manual_seed(0)
    step, t0, best = 0, time.time(), 1e9
    qat_start = int(a.steps * (1 - a.qat_frac))
    while step < a.steps:
        for items in batches(train, a.bs, cfg.ctx, gen):
            if step == qat_start:
                model.set_qat(True)
                print(f"-- step {step}: fake-int8 QAT on")
            s, mask = collate(items, cfg.ctx, dev)
            lg = model(s)
            ce = F.cross_entropy(lg[:, :-1].reshape(-1, cfg.V), s[:, 1:].reshape(-1), reduction="none")
            ce = (ce * mask[:, :-1].reshape(-1)).sum() / mask.sum()
            loss = ce
            kd = torch.zeros((), device=dev)
            if teacher is not None:
                t, att, sb, sp, tp = collate_teacher(items, dev, pad_id)
                with torch.no_grad():
                    tl = teacher(input_ids=t, attention_mask=att).logits[sb, tp][:, G].float()
                    pt = F.softmax(tl / a.kd_temp, -1)
                ls = F.log_softmax(lg[sb, sp].float() / a.kd_temp, -1)[:, S_idx]
                kd = (pt * (torch.log(pt + 1e-9) - ls)).sum(-1).mean() * a.kd_temp ** 2
                loss = ce + a.kd * kd
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            if step % 100 == 0:
                print(f"step {step:5d}  ce {ce.item():.3f}  kd {kd.item():.3f}  lr {sched.get_last_lr()[0]:.2e}  "
                      f"{(time.time() - t0) / step:.2f}s/step", flush=True)
            if step % 1000 == 0 or step == a.steps:
                v = evaluate(model, val, cfg, dev)
                print(f"== step {step}: val CE {v:.3f} (ppl {math.exp(v):.1f})")
                samples(model, vocab, cfg, dev)
                torch.save(model.state_dict(), os.path.join(a.work, "student.pt"))
            if step >= a.steps:
                break
    print("saved", os.path.join(a.work, "student.pt"))


if __name__ == "__main__":
    main()
