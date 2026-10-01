# Step-by-step guide (assumes you have never used the Huawei tools)

The flow has three parts:

1. **Get the demo app onto the watch.** This proves the toolchain, signing and install work. The repo already contains a small demo model, so no training is needed.
2. **Run the Benchmark page on the watch** and send me the numbers. They decide which model size the GT 4 can afford.
3. **Distil the real Gemma model on your PC** (WSL + GPU), rebuild and reinstall.

Typical time: part 1 takes 1–2 h the first time (mostly accounts and certificates), part 3 takes about 1 h of GPU time.

---

## 0. What you need

| Item | Notes |
|---|---|
| Windows PC with **DevEco Studio 5.x/6.x** | You already have it. Also install the HarmonyOS SDK it offers on first start. |
| **WSL** (Ubuntu) with Python 3.10+ | For the training pipeline. Claude Code runs here too. |
| **NVIDIA GPU** visible in WSL | `nvidia-smi` inside WSL should list it. |
| **Huawei developer account** | <https://developer.huawei.com/consumer/en/>. Free, but needs identity verification, which can take a day, so start this first. |
| **Hugging Face account** | Accept the Gemma license at <https://huggingface.co/google/gemma-3-270m-it> and create a read token. |
| **Android phone** paired with the GT 4 | Needs the **Huawei Health** app and **DevEco Assistant** (Chinese name 应用调测助手). On non-Huawei Android phones, install Huawei's **AppGallery** first (APK from <https://appgallery.huawei.com>), then get DevEco Assistant from AppGallery. |
| GT 4 firmware version | Check *Settings → About* on the watch and tell me. It matters for `compatibleSdkVersion`. |

---

## 1. Open the watch project in DevEco Studio

DevEco on Windows works badly with projects on `\\wsl$\...` network paths. Copy the project to a normal Windows folder:

```bash
# in WSL, from the repo root
rm -rf /mnt/c/GemmaNano && cp -r watch /mnt/c/GemmaNano
```

In DevEco Studio:

1. **File → Open** → select `C:\GemmaNano` → *Trust project*.
2. Wait for "Sync" to finish (bottom status bar). If DevEco offers to **migrate or upgrade** the project or hvigor version, accept.
3. If the sync fails with hvigor or model-version errors, use the fallback in [section 6](#6-fallback-start-from-devecos-own-lite-wearable-template).

### 1a. Try it on the PC first (Previewer and Simulator)

* **Previewer:** open `entry/src/main/js/MainAbility/pages/kb/index.hml` and click **Previewer** on the right edge. You should see the round keyboard. Do the same for `index` and `gen`. The previewer only renders, so tapping and file I/O may not work there.
* **Simulator (optional):** *Tools → Device Manager → Local Emulator*. Pick a Lite Wearable or "Huawei Lite Wearable Simulator" if one is listed, start it, then press **Run ▶**. The simulator enforces the same 48 KB JS memory limit as the watch. The full flow (Install model → Chat) is a good dress rehearsal.

---

## 2. Signing: certificate + profile bound to *your* watch

A physical watch only runs apps signed with a debug certificate and a profile that lists that watch's **UDID**.

### 2a. Choose a bundle name

Edit `watch/entry/src/main/config.json` and change `"bundleName": "com.example.gemmanano"` to something unique, e.g. `com.yourname.gemmanano`. Use the same name in AppGallery Connect below. Re-copy to `C:\GemmaNano`, or edit the copy directly.

### 2b. Get the watch UDID

1. Phone: open **Huawei Health** and make sure the GT 4 is connected.
2. Phone: open **DevEco Assistant** and accept its permissions. It finds the watch through Health and shows **UDID**. Copy it (e-mail or message it to yourself).
   *If it asks you to enable debugging or developer mode on the watch, follow its prompt. On most GT models: Settings → About → tap the version number several times.*

### 2c. AppGallery Connect (web)

Go to <https://developer.huawei.com/consumer/en/service/josp/agc/index.html> → **Certificates, app IDs, and profiles** (sometimes under *Users and permissions*).

1. **Generate key + CSR in DevEco:** *Build → Generate Key and CSR*. Create a new key store (`.p12`, remember both passwords and the alias). This produces a `.csr` file.
2. **Certificates → New certificate** → type **Debug** → upload the `.csr` → download the `.cer`.
3. **App ID / app:** *My projects → Add project*, then *Add app*. Platform **HarmonyOS** (*APP (HarmonyOS)*), device type **Lite wearable** (or *Wearable* if Lite isn't offered), package name = your bundle name.
4. **Devices → Add device** → type *Wearable / Lite wearable*, paste the UDID.
5. **Profiles → Add** → type **Debug**, select your app, the debug certificate and the device → download the `.p7b`.

### 2d. Configure signing in DevEco

*File → Project Structure → Project → Signing Configs*:

* untick *Automatically generate signature*,
* **Store file** = your `.p12`, **Store password**, **Key alias**, **Key password**,
* **Profile file** = the `.p7b`, **Certpath file** = the `.cer`, **Sign alg** = `SHA256withECDSA`,
* OK. DevEco writes this into `build-profile.json5`.

---

## 3. Build the HAP and install it on the watch

1. DevEco: **Build → Build Hap(s)/APP(s) → Build Hap(s)**.
   The output is in `C:\GemmaNano\entry\build\default\outputs\default\` and is named like `entry-default-signed.hap` (the name may contain `lite`). If you only get `...-unsigned.hap`, signing isn't configured, so redo step 2d.
2. Copy the `.hap` to the phone (USB, cloud drive, or mail to yourself).
3. Phone: **DevEco Assistant → Apps → Install watch app** (wording varies) → pick the `.hap` → wait for the Bluetooth transfer. A ~1 MB HAP takes a minute or two. Keep the watch on the charger and near the phone.
4. Watch: open the app list and start **GemmaNano**.

*Alternative:* with the phone connected by USB and DevEco Assistant open, DevEco's **Run ▶** can sometimes install directly. The manual route above is the most reliable.

---

## 4. First run on the watch: install the model, benchmark, chat

1. **Install model** (one time). The app walks through ~22 installer pages (demo/pico) that copy the weights into the watch's private storage. It takes about 1–3 minutes. Keep the screen on and don't leave the app. At the end it returns to the menu saying *Model installed*.
2. **Benchmark → Speed**. Write down the three to five lines it prints (`int8 MAC: … M/s`, `read 1KB/2KB/4KB/8KB: … ms`).
3. **Benchmark → Heap**. This deliberately fills memory in 2 KB steps until the app is killed. Reopen the app, go to Benchmark, and it shows how much heap was free. **Send me both results.**
4. **Chat**: tap **Idea** for a ready-made question, or type one, then **ASK**. You see *Reading prompt 3/7* → *Thinking…* → the answer appears word by word, with *n tokens, x s/token* below it. **Stop** interrupts it, and **New** asks another question.

The demo model was trained on Tiny Shakespeare dialogue. It answers like a confused Elizabethan actor. That is expected, and it proves the whole on-device chain works. The Gemma-distilled model replaces it in part 5.

### How to check it works correctly

* After *Install model* the menu says *Model installed*.
* An answer appears and the status line ends in `- done`.
* Same question with the Gemma model: answers are short and on topic, but not always factually correct. The model is tiny.
* Speed: `s/token` on the status line. A 30-token answer at 2 s/token takes a minute.
* If the app silently closes during *Thinking…*, it ran out of JS heap. Send me the Heap benchmark result and I'll shrink the model or context.

---

## 5. Train the real model (Gemma 3 270M → GemmaNano) in WSL

```bash
cd ~/gemma_on_huwawei               # the repo
python3 -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu124   # CUDA build for WSL
pip install -r tools/requirements.txt
huggingface-cli login                 # paste your HF read token (Gemma license accepted)

PRESET=pico tools/run_pipeline.sh     # or PRESET=micro / nano after the benchmark
```

What the pipeline does:

| step | script | time (GPU) | result |
|---|---|---|---|
| 1 | `01_teacher_data.py` | 10–20 min | Gemma 3 270M-it answers ~3.6k everyday questions 6× each → `work/teacher.jsonl` |
| 2 | `02_prepare.py` | 1–2 min | Pruned vocab (768–1024 Gemma tokens), aligned dataset, **compressed-Gemma init** (PCA width cut, layer, head and neuron pruning) |
| 3 | `03_distill.py` | 20–40 min | CE + logit-KD against the full Gemma teacher, then quantization-aware training |
| 4 | `04_export.py` | 1 min | int8 block format + check: float vs int8-runtime loss, sample answers |
| 5 | `05_make_watch_assets.py` | seconds | Regenerates installer pages and `gn_meta.js` in `watch/` |

Then copy `watch/` to Windows again (step 1). Keep your `build-profile.json5` with the signing settings, or redo 2d. Rebuild (3) and reinstall. On the watch, the menu says *A different model is installed*, so tap **Install model** once more.

Check the printouts of steps 3–4. `val CE` should fall steadily, and the sample answers should become sensible sentences. `int8-runtime` CE should be within ~0.1 of the float CE.

---

## 6. Fallback: start from DevEco's own Lite Wearable template

If DevEco cannot sync the provided project (hvigor or SDK mismatch):

1. *File → New → Create Project* → **Lite Wearable** (under the *Wearable* tab or the API-level list) → *Empty Ability* (JS). Name it GemmaNano.
2. Copy from this repo's `watch/entry/src/main/` into the new project's `entry/src/main/`:
   * the whole `js/MainAbility/` folder (pages, common, app.js),
   * the `"pages"` list and `"bundleName"` from `config.json` (keep the template's other fields),
   * `resources/base/media/icon.png`.
3. Build again.

---

## 7. Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| DevEco: `compatibleSdkVersion` / "SDK not supported" | Set it to a value your DevEco offers for Lite Wearable (*File → Project Structure → Products*). |
| DevEco Assistant install fails: *incompatible version* | The watch firmware is older than `compatibleSdkVersion`. Lower it, rebuild, and tell me the firmware version. |
| Install fails: *signature / profile / certificate* | The bundle name in `config.json` must equal the AGC app's package name. The profile must contain this watch's UDID. Rebuild after any change. |
| App icon tap → black screen | A JS page failed to load. Run `python tests/check_watch.py` in WSL, then try the Simulator (DevEco *Log* tab shows `[ACELite][ERROR]` lines). |
| Install model stops at "write … failed" | Storage full or path error. Note the code shown and send it to me. |
| Watch hangs or Bluetooth drops during HAP install | The HAP is too big for one go. Use a smaller preset, or regenerate with `--per-page 2`. |
| App closes while thinking | Out of JS heap. Use a smaller preset (pico) or a smaller `ctx`, and send me the Heap benchmark. |
| Very slow (>5 s/token) | Use `pico`. Send me the Speed benchmark: if flash reads dominate, a bigger block size helps, otherwise a smaller model does. |

## 8. Developer checks you can run in WSL (no watch needed)

```bash
python tests/check_watch.py                      # Lite Wearable rules (tags, CSS, events, page size, regex)
node tests/sim_watch.js "how are you" --temp0    # runs the real pages with a mocked file system
python tests/run_runtime.py --model out/demo     # JS runtime == NumPy int8 reference
tests/jerry/build_jerry.sh                       # builds the watch-profile JerryScript (48 KB heap)
python tests/run_runtime.py --model out/demo --jerry <path printed above>   # same, inside the 48 KB engine
```
