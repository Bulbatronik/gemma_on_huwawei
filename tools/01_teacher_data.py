"""Step 1: let Gemma 3 270M (instruction-tuned) answer the distillation prompts.

    python tools/01_teacher_data.py --out work/teacher.jsonl --samples 6

Needs: `huggingface-cli login` with a token that accepted the Gemma license
(https://huggingface.co/google/gemma-3-270m-it).  ~10-20 min on a GPU.
"""
import argparse
import json
import os
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gemma_nano.prompts import all_prompts, TEACHER_INSTRUCTION  # noqa: E402
from gemma_nano.vocab import clean_answer  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default="google/gemma-3-270m-it")
    ap.add_argument("--out", default="work/teacher.jsonl")
    ap.add_argument("--samples", type=int, default=6, help="answers sampled per prompt")
    ap.add_argument("--batch", type=int, default=96)
    ap.add_argument("--max-prompts", type=int, default=20000)
    ap.add_argument("--max-new", type=int, default=72)
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if dev == "cuda" else torch.float32
    tok = AutoTokenizer.from_pretrained(a.teacher)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(a.teacher, dtype=dtype).to(dev).eval()
    end_ids = [tok.eos_token_id, tok.convert_tokens_to_ids("<end_of_turn>")]

    prompts = all_prompts(max_prompts=a.max_prompts)
    jobs = [p for p in prompts for _ in range(a.samples)]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    kept = 0
    t0 = time.time()
    with open(a.out, "w") as f:
        for i in range(0, len(jobs), a.batch):
            batch = jobs[i:i + a.batch]
            chats = [tok.apply_chat_template([{"role": "user", "content": TEACHER_INSTRUCTION + p}],
                                             tokenize=False, add_generation_prompt=True) for p in batch]
            enc = tok(chats, return_tensors="pt", padding=True, add_special_tokens=False).to(dev)
            with torch.no_grad():
                gen = model.generate(**enc, do_sample=True, temperature=0.8, top_p=0.95, top_k=64,
                                     max_new_tokens=a.max_new, eos_token_id=end_ids,
                                     pad_token_id=tok.pad_token_id)
            outs = tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            for p, o in zip(batch, outs):
                c = clean_answer(o)
                if c:
                    f.write(json.dumps({"prompt": p, "answer": c}) + "\n")
                    kept += 1
            done = min(i + a.batch, len(jobs))
            el = time.time() - t0
            print(f"\r{done}/{len(jobs)} generated, {kept} kept, {el / done * (len(jobs) - done) / 60:.1f} min left",
                  end="", flush=True)
    print(f"\nwrote {kept} examples to {a.out}")


if __name__ == "__main__":
    main()
