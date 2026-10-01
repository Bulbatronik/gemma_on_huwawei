#!/usr/bin/env bash
# Full Gemma 3 270M -> GemmaNano -> watch pipeline.
#   PRESET=pico|micro|nano  (default pico)   STEPS=...  SAMPLES=...
# Requires: `huggingface-cli login` with access to google/gemma-3-270m-it.
set -euo pipefail
cd "$(dirname "$0")/.."
PRESET="${PRESET:-pico}"
STEPS="${STEPS:-8000}"
SAMPLES="${SAMPLES:-6}"
mkdir -p work out
if [ ! -s work/teacher.jsonl ]; then
  python tools/01_teacher_data.py --out work/teacher.jsonl --samples "$SAMPLES"
fi
python tools/02_prepare.py --data work/teacher.jsonl --preset "$PRESET" --out "work/$PRESET"
python tools/03_distill.py --work "work/$PRESET" --steps "$STEPS"
python tools/04_export.py --work "work/$PRESET" --out "out/gemma-$PRESET" --name "gemma-$PRESET"
python tools/05_make_watch_assets.py --model "out/gemma-$PRESET"
echo
echo "Done. Copy the watch/ folder to Windows and rebuild the HAP in DevEco (see docs/GUIDE.md)."
