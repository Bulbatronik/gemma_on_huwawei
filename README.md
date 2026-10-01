# GemmaNano: a Gemma-derived language model running fully on a Huawei Watch GT 4

An engineering experiment. Gemma 3 270M is compressed about 1700× and distilled so that it can run on a
Huawei **Lite Wearable** watch (GT 4). The watch gives third-party apps only a cut-down JavaScript
engine and a **48 KB heap**. Prompts are typed on the watch and answers are generated on the watch, with
no phone or cloud inference.

```
Gemma 3 270M-it ──(prune vocab 262k→768, PCA width 640→64, keep 3/18 layers,
                    2/4 heads, 128/2048 MLP neurons)──► compressed init
        │                                                     │
        └── teacher answers + logits (distillation, QAT) ─────┘
                                   │
                    161 K params, int8, 82 × 2 KB blocks
                                   │
      watch: JS runtime streams blocks from flash, int8 integer kernels,
             48-token context, custom keyboard UI
```

| | |
|---|---|
| [docs/GUIDE.md](docs/GUIDE.md) | **Start here.** Step-by-step: DevEco, signing, installing on the watch, training the real model. |
| [docs/PLATFORM.md](docs/PLATFORM.md) | What the GT 4 / Lite Wearable runtime allows (memory, JS engine, files, packaging) and the evidence. |
| [docs/DESIGN.md](docs/DESIGN.md) | Model compression, block format, inference runtime design, verification. |

## Repository layout

```
watch/                     DevEco Studio project (Lite Wearable, JS, FA model)
  entry/src/main/js/MainAbility/
    common/gn_engine.js    the inference runtime (ES5 + typed arrays) and tokenizer
    common/gn_install.js   installer helper (base64 -> internal://app/ files)
    common/gn_meta.js      generated: model config
    pages/index            menu         pages/kb     keyboard
    pages/gen              generation   pages/bench  speed + heap benchmark
    pages/iNNN             generated installer pages carrying the weights
tools/                     Python pipeline (Gemma -> watch)
  01_teacher_data.py       Gemma answers the distillation prompts
  02_prepare.py            pruned vocab, aligned dataset, compressed-Gemma init
  03_distill.py            CE + logit KD + QAT
  04_export.py             int8 block export + float vs int8-runtime check
  05_make_watch_assets.py  writes the model into the watch project
  run_pipeline.sh          all of the above
  demo_prepare.py          Gemma-free demo model (what the repo ships right now)
  gemma_nano/              model, vocab, compression, export, NumPy reference runtime
tests/                     runtime equivalence, watch simulation, Lite rules checker,
                           watch-profile JerryScript build, mock Gemma
```

## Status

* The runtime has been verified against the NumPy int8 reference, under Node and under the watch-profile
  JerryScript with a 48 KB heap.
* The full app flow has been simulated with a mocked file system.
* The Gemma pipeline has been exercised end to end with a mock Gemma.
* **Not yet verified on a physical GT 4**, and the Gemma-distilled weights are not included: Hugging Face is
  not reachable from the environment this was built in. The project ships a small **demo model** trained
  on Tiny Shakespeare dialogue to validate install, memory, speed and UI on the watch. Then run
  `tools/run_pipeline.sh` to produce the real Gemma-derived model.
