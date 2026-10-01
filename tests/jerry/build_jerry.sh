#!/usr/bin/env bash
# Builds a desktop JerryScript configured like the Huawei Lite Wearable engine
# (OpenHarmony third_party_jerryscript, ES5.1 + typed arrays, no regexp) with a
# 48 KB heap (override with HEAP=<KB>; note: the fork ignores --mem-heap), plus a readBlock() native used by the runtime tests.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
WORK="${WORK:-$HERE/.build}"
HEAP="${HEAP:-48}"
mkdir -p "$WORK"
if [ ! -d "$WORK/third_party_jerryscript" ]; then
  git clone --depth 1 https://github.com/openharmony/third_party_jerryscript.git "$WORK/third_party_jerryscript"
fi
J="$WORK/third_party_jerryscript"
MAIN="$J/jerry-main/main-unix.c"
if ! grep -q gn_read_block "$MAIN"; then
  # insert the handler before register_js_function() and register it next to "print"
  python3 - "$MAIN" "$HERE/readblock.c.inc" <<'PY'
import sys
src = open(sys.argv[1]).read(); inc = open(sys.argv[2]).read()
anchor = "static void\nregister_js_function"
assert anchor in src
src = src.replace(anchor, inc + "\n" + anchor, 1)
reg = 'register_js_function ("print", jerryx_handler_print);'
src = src.replace(reg, reg + '\n  register_js_function ("readBlock", gn_read_block);\n  register_js_function ("heapUsed", gn_heap_used);', 1)
open(sys.argv[1], "w").write(src)
PY
fi
printf "JERRY_ES2015=0\nJERRY_ES2015_BUILTIN_TYPEDARRAY=1\nJERRY_BUILTIN_REGEXP=0\n" > "$WORK/watch.profile"
python3 "$J/tools/build.py" --builddir="$WORK/jerry$HEAP" --profile="$WORK/watch.profile" \
  --compile-flag=-DJERRY_GLOBAL_HEAP_SIZE=$HEAP --error-messages=ON --line-info=ON --compile-flag=-w \
  --cmake-param=-DJERRY_MEM_STATS=ON >"$WORK/build$HEAP.log" 2>&1 || { tail -20 "$WORK/build$HEAP.log"; exit 1; }
echo "$WORK/jerry$HEAP/bin/jerry"
