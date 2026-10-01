# Design: from Gemma 3 270M to a 160 KB model on a watch

## 1. Why Gemma cannot run as-is

Gemma 3 270M has 270 M parameters. 170 M of them are a 262 144 × 640 embedding table.
Even at 4 bits that is 135 MB, and one token needs ~100 M multiply-accumulates. The GT 4 gives a
third-party app **a 48 KB JavaScript heap** and an interpreter that does a few hundred thousand int8
MACs per second (to be confirmed by the Benchmark page). The gap is about three orders of magnitude, so
the model has to shrink by roughly 1000× in every dimension. See [PLATFORM.md](PLATFORM.md).

## 2. Compression chain (no training from scratch)

| step | what | Gemma 3 270M | GemmaNano *pico* | *nano* |
|---|---|---|---|---|
| vocabulary pruning | keep the most frequent Gemma tokens in the target domain + all ASCII chars | 262 144 | 768 | 1024 |
| width (PCA of kept embeddings) | residual stream projected onto top-d principal directions | 640 | 64 | 96 |
| depth | evenly spaced layers kept | 18 | 3 | 4 |
| heads | first Hq query heads, first hd/2 rotary pairs of each head, 1 shared KV head (MQA, as in Gemma) | 4 × 256 | 2 × 32 | 3 × 32 |
| MLP | GeGLU neurons with the largest gate·up·down norm product | 2048 | 128 | 192 |
| context | | 32 k | 48 tokens | 48 |
| parameters | | 268 M | **161 K** | 420 K |
| storage | bf16 | 536 MB | 164 KB int8 | 430 KB int8 |

The architecture stays Gemma 3: RMSNorm with (1+w), pre and post "sandwich" norms, q/k-norm,
RoPE, GeGLU (tanh-GELU), tied embeddings scaled by √d.

Compression alone destroys most of the model. **Distillation heals it** (`03_distill.py`):

* **sequence-level KD**: Gemma 3 270M-it answers ~3.6 k everyday prompts several times each (small talk, "what is X",
  facts, tips, jokes). The student is trained on those answers.
* **logit KD**: at every answer position where Gemma's own next token is in the pruned vocabulary,
  the student matches Gemma's distribution, renormalized over that vocabulary. The answers keep
  Gemma's segmentation (tokens outside the vocabulary are spelled out in characters), so teacher and
  student positions line up.
* **QAT**: the final 25% of steps run with fake int8 quantization exactly where the watch runtime quantizes.
  Weights are per-row int8, and activations are per-vector int8 at every matmul input.

What you can expect: short, grammatical, on-topic answers in the narrow domain it was distilled
on. Facts will often be wrong. It has ~0.06% of Gemma's parameters, so treat it as a demo of what's
possible rather than an assistant.

## 3. Runtime (`watch/.../common/gn_engine.js`)

* **Pull-based streaming.** The engine is a state machine. `next()` returns the index of the
  2 KB weight block it needs, the caller reads `internal://app/wNNN`, and `feed(bytes)` consumes it.
  Per token, one embedding block is read, then all layer blocks in order, then the embedding region
  again for the tied output head. Nothing but the current block is held in memory.
* **Block format.** A row is `[fp16 scale][n × int8]`. Rows are packed in program order and never
  straddle a block. Norm weights are stored as rows too, so the stream *is* the program. Gate and up
  rows are interleaved, so the GeGLU product is formed row by row without a second buffer.
* **Integer kernels.** `acc += w[k] * x[k]` over `Int8Array`s stays within JerryScript's unboxed-integer
  range (±2²⁷), so the inner loop never allocates. Attention scores and the value aggregation use the
  same trick: the int8 KV cache and softmax weights are quantized to 9 bits.
* **Sampling while streaming.** Logits are never materialized. A running top-k (k = 16) with a
  repetition penalty is kept while the head rows stream past.
* **Memory (pico):** int8 KV cache 9 KB, vectors ~2 KB, one block 2 KB, top-k and flags ~1 KB.
* **Async safety.** The gen page drives `next/feed` with a trampoline that works whether
  `readArrayBuffer` calls back asynchronously (normal) or synchronously. It yields to the UI once
  per token.
* **Tokenizer.** Greedy longest-match over the pruned vocabulary, the same algorithm as in Python. The vocab
  file is only resident while encoding the prompt. Afterwards each generated token's text is read from
  the file by offset.

## 4. Getting the weights onto the watch

Lite JS cannot read package resources, and a page's code must stay under ~48 KB. `05_make_watch_assets.py`
therefore generates *installer pages*. Each holds 4 blocks (8 KB) as base64 string literals, decodes
them and writes them with `@system.file.writeArrayBuffer`, then routes to the next page. The last page
writes a marker file with the model id. The installer runs once per model.

## 5. Verification done without a watch

* `tests/run_runtime.py`: the JS runtime under Node and under the **watch-profile JerryScript with a 48 KB heap**
  produces token-for-token the same output as the NumPy model of the int8 arithmetic.
* `tests/sim_watch.js`: the real pages (menu → installer pages → keyboard → generation) run against
  a mocked `@system.file`/router, in async and in synchronous-callback mode.
* `tests/check_watch.py`: Lite Wearable rules (tags, events, CSS, page sizes, no regex, ES5 parse).
* `tests/mock_gemma.py`: a tiny random Gemma3ForCausalLM + Gemma-style tokenizer exercising
  `01/02/03` (vocab subset, KD alignment, compression init, logit KD) without downloading Gemma.

## 6. Open questions only the physical watch can answer

1. **int8 MAC/s** of the GT 4's JS engine → model size. Rough plan: <0.3 M/s → pico, 0.3–1 → micro, >1 → nano.
2. **readArrayBuffer latency** → block size. If ~20 ms+ per call dominates, larger blocks (4 KB) and
   batched prefill become worthwhile.
3. **Free heap on the gen page** → `ctx` and preset.
4. Whether `text-overflow: break` wraps on the device. If not, the gen page needs manual line splitting.
