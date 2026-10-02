# What the Huawei Watch GT 4 / Lite Wearable platform allows

These findings drove the design. Huawei's own docs (developer.huawei.com) could not be reached from the build
environment. Everything below comes from the open-source OpenHarmony "lite" stack that Huawei's
watch runtime is built on, from real-device reports in open-source GT 4 projects, and from
experiments with that engine compiled locally. The **Benchmark** page in the app measures the numbers
that are still unverified.

| Topic | Finding | Evidence | Consequence for GemmaNano |
|---|---|---|---|
| App model | Lite Wearable apps are **JavaScript only** (FA model, `.hml/.css/.js` pages). ArkTS/Stage, Java and native C/C++ are not available to third-party apps. | DevEco Lite Wearable template, [gt4-ebook-reader](https://github.com/shuhuang-1/gt4-ebook-reader) | The whole inference runtime is written in JS. |
| JS engine | Cut-down **JerryScript**, ES5.1 profile (`JERRY_ES2015 0`), **typed arrays on**, **RegExp off**, **eval off**. | `jerry-core/config.h` in [openharmony/third_party_jerryscript](https://github.com/openharmony/third_party_jerryscript), plus a real-device crash on a regex literal reported by [NexusCheckin](https://github.com/maodou1145/NexusCheckin) | ES5 code, `Int8Array`/`Float32Array` for weights and activations, no regex anywhere. |
| JS heap | **48 KB** global heap (`JERRY_GLOBAL_HEAP_SIZE (48)`). The external-context variant for watches uses 64 KB per task. DevEco's simulator enforces a 48 KB limit. Running out of heap **aborts the app** and cannot be caught. | `config.h`, `jerry-port/config-jupiter.h` | Weights can never be resident. They are streamed from flash in 2 KB blocks. The engine's live state is about 17 KB (pico). |
| Numbers | Integers within ±2^27 are stored unboxed. Every non-integer result allocates a heap cell. | `ecma-globals.h`, `ecma_make_number_value` | All heavy loops use int8×int8 integer arithmetic, with no allocation per multiply-accumulate. Floats are only used for O(d) work. |
| Heap fragmentation | No compaction. Allocating a fresh 8 KB buffer per read hit out-of-memory after a few tokens, even with free space left. | Reproduced with the watch-profile JerryScript at 48 KB, see `tests/` | 2 KB blocks. The engine allocates all its buffers once. |
| Per-page code size | Compiled page JS must stay below about 48 KB (`FILE_CONTENT_LENGTH_MAX = 48K`). Larger pages crash on real watches, and a 275 KB data file crashed a GT 4 during install. | `js_fwk_common.h`, gt4-ebook-reader, NexusCheckin | Weights are not embedded in app JS. Each installer page carries about 11 KB. |
| Package files | JS **cannot read files shipped in the package** (`resources/rawfile`). `@system.file` only sees `internal://app/` (the app's private data directory). | `nativeapi_fs.cpp` (`FILE_PREFIX "internal://app"`), gt4-ebook-reader | One-time install step: installer pages decode base64 and `writeArrayBuffer` the blocks into `internal://app/`. |
| File API | `readArrayBuffer({uri, position, length})` returns a `Uint8Array` allocated **in the JS heap**. Natively it also `malloc`s the *whole file size* on every call. `writeArrayBuffer` needs a `Uint8Array`. `readText` reads at most 4 KB. | `nativeapi_fs.cpp` (`ReadArrayFileInner`) | Many small files (`w000`…), each a single block. |
| Network | `@system.fetch` exists (through the phone, API 10+). Binary responses don't work, large responses exhaust the heap, and parallel requests hang. | NexusCheckin | Not used. Inference is 100% local. |
| UI | Supported tags are `div`, `text`, `list`, `swiper`, `image` and similar. Only **bare `onclick`** registers on devices. Every element needs fixed width and height. **No system keyboard**. Every PNG becomes a w×h×4 bitmap in the package. | NexusCheckin device tests, `lite_component_map` | Custom 28-key keyboard, text-only UI. The DevEco compiler only accepts `text-overflow: clip|ellipsis`, so multi-line text is word-wrapped in JS into fixed single-line slots (`common/gn_wrap.js`). |
| Install | Debug apps are signed with an AppGallery Connect debug certificate and profile bound to the watch UDID. They are installed from an Android phone with the **DevEco Assistant** app over Bluetooth. | Huawei developer forum and Medium guides | See [GUIDE.md](GUIDE.md). |
| Hardware | GT 4: 466×466 AMOLED. The CPU is an unpublished low-power MCU class chip. RAM is in the tens of MB, but the app only gets the JS heap. | — | The **Benchmark** page measures int8 MAC/s, flash read latency and free heap. |

## Measured locally (desktop JerryScript built with the watch profile)

| model | params | flash | engine heap (live) | runs in 40 KB heap | runs in 48 KB heap |
|---|---|---|---|---|---|
| pico (d=64, L=3, V=768) | 161 K | 164 KB | ≈17 KB + 2 KB block | yes | yes |
| nano (d=96, L=4, V=1024) | 420 K | 430 KB | ≈25 KB + 2 KB block | no | yes |

Engine bytecode adds about 9 KB. The JerryScript build used for tests is the OpenHarmony fork with the same
profile as the watch (`tests/jerry/build_jerry.sh`). The JS runtime's outputs match the NumPy model
of the int8 arithmetic token for token (`tests/run_runtime.py`).
