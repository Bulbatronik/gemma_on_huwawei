"""Initialise a GemmaNano student by *compressing* Gemma 3 270M weights.

This is structured pruning + low-rank projection, not training from scratch:

  1. vocabulary pruning: keep only the embedding rows of the ~1k tokens the
     student uses (Gemma's 262k-row table is 170M of its 270M parameters);
  2. width reduction: PCA of those embedding rows gives an orthonormal basis
     P (640 x d).  Every weight that reads from / writes to the residual stream
     is projected through P (W @ P, P^T @ W);
  3. depth reduction: keep L of the 18 layers, evenly spaced;
  4. head / neuron pruning: keep the first Hq heads and, inside each head, the
     first hd/2 rotary pairs (dims i and i+128), keep the `ffn` MLP neurons with
     the largest |gate|*|up|*|down| norm product;
  5. RMSNorm (1+w) diagonals are carried into the new basis as
     diag(P^T diag(1+w) P).

The result is a poor model on its own (most of Gemma's capacity is gone), but
it is a much better starting point than random init, and the distillation
step (03_distill.py) then heals it against the full Gemma teacher.
"""
import math

import numpy as np
import torch


def _find(sd, suffix):
    hits = [k for k in sd if k.endswith(suffix)]
    if len(hits) != 1:
        raise KeyError(f"{suffix}: {hits[:4]}")
    return sd[hits[0]].float().cpu().numpy()


def _layer(sd, i, name):
    return _find(sd, f"layers.{i}.{name}")


def compress_gemma(teacher, vocab, cfg, report=print):
    tcfg = teacher.config
    tcfg = getattr(tcfg, "text_config", tcfg)
    sd = teacher.state_dict()
    E = _find(sd, "embed_tokens.weight")                   # (262144, 640)
    Dt = E.shape[1]
    n_layers_t = tcfg.num_hidden_layers
    Ht, Hkv, hdt = tcfg.num_attention_heads, tcfg.num_key_value_heads, tcfg.head_dim
    assert cfg.Hq <= Ht and cfg.hd <= hdt

    # ---- 1+2: vocab subset + PCA basis -------------------------------------
    gids = np.array(vocab.gemma_ids)
    have = gids >= 0
    Esub = E[gids[have]]
    U, S, Vt = np.linalg.svd(Esub.astype(np.float64), full_matrices=False)
    P = Vt[: cfg.d].T                                       # (640, d) orthonormal columns
    kept = float((S[: cfg.d] ** 2).sum() / (S ** 2).sum())
    report(f"[compress] embedding PCA {Dt}->{cfg.d}: keeps {kept * 100:.1f}% of energy "
           f"of the {have.sum()} kept rows (of {E.shape[0]} Gemma rows)")
    emb_scale = math.sqrt(Dt) / math.sqrt(cfg.d)            # keep input magnitude
    Es = np.random.default_rng(0).normal(0, 0.02, (cfg.V, cfg.d))
    Es[have] = (Esub @ P) * emb_scale
    out = {"embed.weight": Es}

    def diag_norm(w):
        g = 1.0 + w
        return np.einsum("ik,i,ik->k", P, g, P) - 1.0

    half_t, half_s = hdt // 2, cfg.hd // 2
    dims = list(range(half_s)) + list(range(half_t, half_t + half_s))   # keep rotary pairs together
    q_rows = [h * hdt + j for h in range(cfg.Hq) for j in dims]
    kv_rows = list(dims)                                                # KV head 0 (Gemma 270M has 1)

    # ---- 3: layer selection --------------------------------------------------
    pick = [round(i * (n_layers_t - 1) / max(1, cfg.L - 1)) for i in range(cfg.L)]
    report(f"[compress] keeping Gemma layers {pick} of {n_layers_t}; heads {cfg.Hq}/{Ht}, "
           f"head dims {cfg.hd}/{hdt}, MLP {cfg.ffn}/{tcfg.intermediate_size}")
    for li, ti in enumerate(pick):
        p = f"layers.{li}."
        Wq, Wk, Wv, Wo = (_layer(sd, ti, f"self_attn.{n}_proj.weight") for n in "qkvo")
        out[p + "q_proj.weight"] = Wq[q_rows] @ P
        out[p + "k_proj.weight"] = Wk[kv_rows] @ P
        out[p + "v_proj.weight"] = Wv[kv_rows] @ P
        out[p + "o_proj.weight"] = P.T @ Wo[:, q_rows]
        out[p + "q_norm.weight"] = _layer(sd, ti, "self_attn.q_norm.weight")[dims]
        out[p + "k_norm.weight"] = _layer(sd, ti, "self_attn.k_norm.weight")[dims]
        for n in ["input_layernorm", "post_attention_layernorm", "pre_feedforward_layernorm",
                  "post_feedforward_layernorm"]:
            out[p + n + ".weight"] = diag_norm(_layer(sd, ti, n + ".weight"))
        G, Up, Dn = (_layer(sd, ti, f"mlp.{n}_proj.weight") for n in ("gate", "up", "down"))
        score = np.linalg.norm(G, axis=1) * np.linalg.norm(Up, axis=1) * np.linalg.norm(Dn, axis=0)
        keep = np.sort(np.argsort(-score)[: cfg.ffn])
        out[p + "gate_proj.weight"] = G[keep] @ P
        out[p + "up_proj.weight"] = Up[keep] @ P
        out[p + "down_proj.weight"] = P.T @ Dn[:, keep]
    out["norm.weight"] = diag_norm(_find(sd, "model.norm.weight") if any(
        k.endswith("model.norm.weight") for k in sd) else _find(sd, ".norm.weight"))
    return {k: torch.tensor(np.asarray(v), dtype=torch.float32) for k, v in out.items()}
