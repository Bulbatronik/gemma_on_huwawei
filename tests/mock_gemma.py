"""Build a tiny random Gemma-3-architecture model + Gemma-style tokenizer on disk,
so the Gemma-specific pipeline (02_prepare / 03_distill KD / gemma_init) can be
tested without downloading google/gemma-3-270m-it.

    python tests/mock_gemma.py --text work/shakespeare.txt --out work/mock_gemma
"""
import argparse
import json
import os

from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
from transformers import Gemma3ForCausalLM, Gemma3TextConfig, PreTrainedTokenizerFast

CHAT = ("{{ bos_token }}{% for m in messages %}<start_of_turn>{{ 'model' if m['role'] == 'assistant' else m['role'] }}\n"
        "{{ m['content'] }}<end_of_turn>\n{% endfor %}{% if add_generation_prompt %}<start_of_turn>model\n{% endif %}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    specials = ["<pad>", "<eos>", "<bos>", "<unk>", "<start_of_turn>", "<end_of_turn>"]
    tk = Tokenizer(models.BPE(unk_token="<unk>", byte_fallback=True))
    tk.pre_tokenizer = pre_tokenizers.Metaspace(replacement="▁", prepend_scheme="never")
    tk.decoder = decoders.Metaspace(replacement="▁", prepend_scheme="never")
    alphabet = [chr(c) for c in range(33, 127)] + ["▁"]
    bytes_ = ["<0x%02X>" % i for i in range(256)]
    tr = trainers.BpeTrainer(vocab_size=4000, special_tokens=specials + bytes_, initial_alphabet=alphabet)
    tk.train([a.text], tr)
    fast = PreTrainedTokenizerFast(tokenizer_object=tk, bos_token="<bos>", eos_token="<eos>", pad_token="<pad>",
                                   unk_token="<unk>")
    fast.chat_template = CHAT
    os.makedirs(a.out, exist_ok=True)
    fast.save_pretrained(a.out)
    cfg = Gemma3TextConfig(vocab_size=len(fast), hidden_size=96, intermediate_size=384, num_hidden_layers=6,
                           num_attention_heads=4, num_key_value_heads=1, head_dim=64, max_position_embeddings=512,
                           query_pre_attn_scalar=64, sliding_window=64,
                           pad_token_id=fast.pad_token_id, bos_token_id=fast.bos_token_id, eos_token_id=fast.eos_token_id)
    model = Gemma3ForCausalLM(cfg)
    model.save_pretrained(a.out)
    print("mock gemma saved:", a.out, "vocab", len(fast))
    print(json.dumps({k: getattr(cfg, k) for k in ["hidden_size", "num_hidden_layers", "head_dim"]}))


if __name__ == "__main__":
    main()
