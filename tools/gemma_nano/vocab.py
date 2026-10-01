"""Tiny vocabulary for the watch: a frequency-pruned subset of Gemma's tokenizer.

Token ids:   0 <pad>  1 <bos>  2 <eos>  3 <sep>   then 95 printable ASCII chars,
             then the most frequent multi-character Gemma tokens (ASCII only).

Two encoders are used:
  * encode_greedy(text)   - greedy longest match.  Used for prompts, on the watch
                            AND in training, so both sides see identical ids.
  * encode_gemma_aligned  - Gemma's own segmentation; tokens outside the subset are
                            spelled out as characters.  Used for answers, so the
                            student's targets line up with Gemma's (teacher) positions
                            for logit distillation.
"""
import json
import struct
from collections import Counter

SPECIALS = ["<pad>", "<bos>", "<eos>", "<sep>"]
PRINTABLE = [chr(c) for c in range(32, 127)]

_PROMPT_OK = set("abcdefghijklmnopqrstuvwxyz0123456789?!.,'")


def normalize_prompt(s: str) -> str:
    """Must stay identical to normPrompt() in gn_engine.js."""
    out, sp = [], True
    for ch in s:
        o = ord(ch)
        if 65 <= o <= 90:
            ch = chr(o + 32)
        if ch in _PROMPT_OK:
            out.append(ch)
            sp = False
        elif not sp:
            out.append(" ")
            sp = True
    if out and out[-1] == " ":
        out.pop()
    return "".join(out)


_REPL = {
    "‘": "'", "’": "'", "“": '"', "”": '"', "–": "-", "—": " - ",
    "…": "...", " ": " ", "•": "-", "é": "e", "è": "e", "à": "a",
}


def clean_answer(s: str, max_chars: int = 160, max_sentences: int = 2) -> str | None:
    """ASCII-only, single paragraph, no markdown, a few sentences. None if unusable."""
    for k, v in _REPL.items():
        s = s.replace(k, v)
    s = "".join(ch if 32 <= ord(ch) < 127 or ch in "\n\t" else " " for ch in s)
    for ch in "*#`_|<>[]{}~^\\":
        s = s.replace(ch, " ")
    s = " ".join(s.split())
    if not s:
        return None
    # cut to whole sentences
    out, n = [], 0
    i = 0
    while i < len(s):
        out.append(s[i])
        if s[i] in ".!?" and (i + 1 == len(s) or s[i + 1] == " "):
            n += 1
            if n >= max_sentences:
                break
        i += 1
    s = "".join(out).strip()
    if len(s) > max_chars:
        cut = max(s.rfind(". ", 0, max_chars), s.rfind("! ", 0, max_chars), s.rfind("? ", 0, max_chars))
        if cut < 20:
            return None
        s = s[: cut + 1]
    if len(s) < 2 or s[-1] not in ".!?":
        return None
    return s


class Vocab:
    def __init__(self, tokens, gemma_ids=None):
        self.tokens = list(tokens)                         # id -> text ("" for specials)
        self.gemma_ids = list(gemma_ids) if gemma_ids else [-1] * len(tokens)
        self.index = {t: i for i, t in enumerate(self.tokens) if i >= len(SPECIALS)}
        self.max_len = max(len(t) for t in self.tokens)
        self.g2s = {g: i for i, g in enumerate(self.gemma_ids) if g >= 0}

    def __len__(self):
        return len(self.tokens)

    # ---------- encoders ----------
    def encode_greedy(self, s: str):
        out, p, n_total = [], 0, len(s)
        while p < n_total:
            n = min(self.max_len, n_total - p)
            tid = -1
            while n > 0:
                tid = self.index.get(s[p:p + n], -1)
                if tid >= 0:
                    break
                n -= 1
            if tid < 0:
                p += 1
                continue
            out.append(tid)
            p += n
        return out

    def encode_gemma_aligned(self, gemma_ids, piece_text):
        """Returns (student_ids, first_piece_index_per_gemma_token, in_subset_flags)."""
        out, first, ins = [], [], []
        for g in gemma_ids:
            first.append(len(out))
            sid = self.g2s.get(g)
            if sid is not None:
                out.append(sid)
                ins.append(True)
            else:
                for ch in piece_text(g):
                    cid = self.index.get(ch)
                    if cid is not None:
                        out.append(cid)
                ins.append(False)
        return out, first, ins

    def decode(self, ids):
        return "".join(self.tokens[i] for i in ids)

    # ---------- io ----------
    def save_json(self, path):
        with open(path, "w") as f:
            json.dump({"tokens": self.tokens, "gemma_ids": self.gemma_ids}, f)

    @staticmethod
    def load_json(path):
        with open(path) as f:
            d = json.load(f)
        return Vocab(d["tokens"], d["gemma_ids"])

    def to_bin(self) -> bytes:
        """u16 V | u16 off[V+1] | u16 sorted[V] (0xFFFF padded) | pool  (see Tok in gn_engine.js)"""
        V = len(self.tokens)
        pool = b""
        offs = []
        for t in self.tokens:
            offs.append(len(pool))
            pool += t.encode("ascii")
        offs.append(len(pool))
        assert len(pool) < 65536
        ids = [i for i in range(len(SPECIALS), V)]
        ids.sort(key=lambda i: self.tokens[i].encode("ascii"))
        ids += [0xFFFF] * (V - len(ids))
        return struct.pack("<H", V) + struct.pack(f"<{V + 1}H", *offs) + struct.pack(f"<{V}H", *ids) + pool


def gemma_piece_text_fn(tokenizer):
    """id -> surface text of a single Gemma token (SentencePiece '▁' -> ' ', <0xNN> -> byte)."""
    cache = {}

    def f(g):
        t = cache.get(g)
        if t is None:
            piece = tokenizer.convert_ids_to_tokens(int(g))
            if piece is None:
                t = ""
            elif piece.startswith("<0x") and piece.endswith(">") and len(piece) == 6:
                t = chr(int(piece[3:5], 16))
            else:
                t = piece.replace("▁", " ")
            cache[g] = t
        return t
    return f


def build_from_gemma(tokenizer, texts, V, eos_gemma_id):
    """Pick the V most useful Gemma tokens for this corpus."""
    piece_text = gemma_piece_text_fn(tokenizer)
    counts = Counter()
    for t in texts:
        counts.update(tokenizer(t, add_special_tokens=False)["input_ids"])
    tokens = list(SPECIALS)
    gids = [-1, -1, eos_gemma_id, -1]
    # every printable ASCII char, mapped to its Gemma single-char token where one exists
    for ch in PRINTABLE:
        g = tokenizer.convert_tokens_to_ids(ch if ch != " " else "▁")
        if g is None or g == tokenizer.unk_token_id or piece_text(g) != ch:
            g = -1
        tokens.append(ch)
        gids.append(g)
    seen = set(tokens)
    for g, _ in counts.most_common():
        if len(tokens) >= V:
            break
        t = piece_text(g)
        if len(t) < 2 or t in seen or not all(32 <= ord(c) < 127 for c in t):
            continue
        tokens.append(t)
        gids.append(int(g))
        seen.add(t)
    return Vocab(tokens, gids)


def build_standalone(texts, V):
    """Gemma-free vocab (demo / tests): chars + frequent words and word pieces."""
    counts = Counter()
    for t in texts:
        words = t.split(" ")
        for i, w in enumerate(words):
            if not w:
                continue
            w2 = (" " + w) if i > 0 else w
            counts[w2] += 1
            for n in (2, 3, 4):
                for j in range(0, len(w2) - n + 1):
                    counts[w2[j:j + n]] += 0.2
    tokens = list(SPECIALS) + PRINTABLE
    seen = set(tokens)
    for t, _ in counts.most_common():
        if len(tokens) >= V:
            break
        if len(t) < 2 or t in seen or not all(32 <= ord(c) < 127 for c in t):
            continue
        tokens.append(t)
        seen.add(t)
    return Vocab(tokens)
