"""Run gn_engine.js (from the watch app) on an exported model under Node and/or
the desktop JerryScript build, and compare against the NumPy reference.

    python tests/run_runtime.py --model out/demo --jerry path/to/jerry [--prompts ...]
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
ENGINE = os.path.join(ROOT, "watch/entry/src/main/js/MainAbility/common/gn_engine.js")
DRIVER = os.path.join(ROOT, "tests/driver.js")

NODE_PRELUDE = r"""
var fs = require('fs');
function readBlock(p, pos, len) {
  var b = fs.readFileSync(p);
  if (pos !== undefined) b = b.subarray(pos, Math.min(b.length, pos + len));
  return new Uint8Array(b.buffer.slice(b.byteOffset, b.byteOffset + b.length));
}
var print = console.log;
"""


def engine_src():
    src = open(ENGINE).read()
    assert "export default GN;" in src
    return src.replace("export default GN;", "")


def build_script(model_dir, prompts, max_new, node):
    args = "var ARGS = " + json.dumps({"dir": os.path.abspath(model_dir), "prompts": prompts, "maxNew": max_new}) + ";\n"
    return (NODE_PRELUDE if node else "") + args + engine_src() + "\n" + open(DRIVER).read()


def parse(out):
    res, timing = [], None
    for line in out.splitlines():
        if line.startswith("OUT"):
            res.append([int(x) for x in line.split()[1:]])
        elif line.startswith("TIME"):
            timing = line
    return res, timing


def run_js(model_dir, prompts, max_new, jerry=None, mem_stats=False):
    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as f:
        f.write(build_script(model_dir, prompts, max_new, node=jerry is None))
        path = f.name
    cmd = ["node", path] if jerry is None else [jerry] + (["--mem-stats"] if mem_stats else []) + [path]
    p = subprocess.run(cmd, capture_output=True, text=True)
    os.unlink(path)
    if p.returncode != 0:
        raise RuntimeError(f"{cmd[0]} failed:\n{p.stdout}\n{p.stderr}")
    return p.stdout


def run_ref(model_dir, prompts, max_new):
    from gemma_nano.refimpl import RefModel
    from gemma_nano.vocab import Vocab, normalize_prompt
    ref = RefModel(model_dir)
    voc = Vocab.load_json(os.path.join(model_dir, "vocab.json"))
    c = ref.c
    outs = []
    for p in prompts:
        ids = [c.bos] + voc.encode_greedy(normalize_prompt(p)) + [c.sep]
        outs.append(ref.generate_greedy(ids, max_new=max_new))
    return outs, voc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--jerry", help="path to the patched jerry binary (tests/jerry/build_jerry.sh)")
    ap.add_argument("--max-new", type=int, default=24)
    ap.add_argument("--prompts", nargs="*", default=["Hello, how are you?", "what is the sun", "Tell me a joke!"])
    a = ap.parse_args()

    ref, voc = run_ref(a.model, a.prompts, a.max_new)
    node_out = run_js(a.model, a.prompts, a.max_new)
    nres, ntime = parse(node_out)
    ok = True
    for p, r, n in zip(a.prompts, ref, nres):
        same = r == n
        ok &= same
        print(f"[node {'OK ' if same else 'DIFF'}] {p!r} -> {voc.decode([t for t in n if t > 3])!r}")
        if not same:
            print("   ref :", voc.decode([t for t in r if t > 3]))
    print("node", ntime)
    if a.jerry:
        jout = run_js(a.model, a.prompts, a.max_new, jerry=a.jerry, mem_stats=True)
        jres, jtime = parse(jout)
        for p, r, j in zip(a.prompts, ref, jres):
            same = r == j
            ok &= same
            print(f"[jerry {'OK ' if same else 'DIFF'}] {p!r}")
        print("jerry", jtime)
        for line in jout.splitlines():
            if "Heap size" in line or "Peak allocated =" in line:
                print("jerry", line.strip())
    print("ALL MATCH" if ok else "MISMATCH")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
