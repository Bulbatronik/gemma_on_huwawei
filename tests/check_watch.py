"""Static checks for the Lite Wearable project (things DevEco does not catch but
that break on real watches).  Optionally parses the ES5 runtime files with the
watch-configured JerryScript.

    python tests/check_watch.py [--jerry path/to/jerry]
"""
import argparse
import json
import os
import re
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JS_ROOT = os.path.join(ROOT, "watch/entry/src/main/js/MainAbility")
WHITELIST = {"div", "stack", "list", "list-item", "swiper", "tabs", "tab-bar", "tab-content", "image-animator",
             "image", "img", "progress", "text", "marquee", "analog-clock", "clock-hand", "chart", "input",
             "slider", "switch", "picker-view", "qrcode", "canvas"}
PAGE_JS_LIMIT = 40 * 1024
ES5_FILES = ["common/gn_engine.js"]          # must parse as plain ES5 after dropping the export line

problems = []
SIZES = {}


def bad(path, msg):
    problems.append(f"{os.path.relpath(path, ROOT)}: {msg}")


def check_hml(p):
    s = re.sub(r"<!--.*?-->", "", open(p).read(), flags=re.S)
    for m in re.finditer(r"<\s*([a-zA-Z][\w-]*)", s):
        if m.group(1).lower() not in WHITELIST:
            bad(p, f"tag <{m.group(1)}> is not supported on Lite Wearable")
    for m in re.finditer(r"\s(grab:|on:)\w+=", s):
        bad(p, "event prefix '%s' does not register on real devices; use bare onclick" % m.group(1))
    for m in re.finditer(r'class="([^"]*)"', s):
        if " " in m.group(1).strip():
            bad(p, f"multiple classes '{m.group(1)}' - use a single class")
    root = re.search(r"<\s*([a-zA-Z][\w-]*)", s)
    if root and root.group(1) not in ("div", "stack"):
        bad(p, "root element must be div or stack")


def check_css(p):
    s = re.sub(r"/\*.*?\*/", "", open(p).read(), flags=re.S)
    for m in re.finditer(r"([^{}]+)\{([^{}]*)\}", s):
        sel, body = m.group(1).strip(), m.group(2)
        if "," in sel:
            bad(p, f"selector list '{sel}' not supported")
        if re.search(r"\.[\w-]+\.[\w-]+", sel) or re.search(r"[>+~:\[]", sel):
            bad(p, f"selector '{sel}' not supported")
        m2 = re.search(r"text-overflow\s*:\s*([\w-]+)", body)
        if m2 and m2.group(1) not in ("clip", "ellipsis"):
            bad(p, f"text-overflow: {m2.group(1)} in '{sel}' (DevEco accepts only clip | ellipsis)")
        if re.search(r"(width|height)\s*:\s*auto", body):
            bad(p, f"'auto' size in '{sel}'")


def check_js(p):
    s = open(p).read()
    code = re.sub(r"'(?:\\.|[^'\\])*'", "''", s)
    code = re.sub(r"//[^\n]*|/\*.*?\*/", "", code, flags=re.S)
    if re.search(r"(^|[=(,:!&|?{};\s])/(?![/*])[^/\n]+/[gimsuy]*\s*[.;,)]", code, flags=re.M):
        bad(p, "looks like a regex literal (regex is disabled in the watch engine)")
    if re.search(r"\beval\s*\(", code):
        bad(p, "eval is disabled on the watch")
    size = len(s.encode())
    for m in re.finditer(r"^import \w+ from '(\.[^']+)';", s, flags=re.M):   # bundled into the page
        dep = os.path.normpath(os.path.join(os.path.dirname(p), m.group(1)))
        if os.path.exists(dep):
            size += os.path.getsize(dep)
    if "/pages/" in p:
        SIZES[os.path.relpath(p, JS_ROOT)] = size
    if size > PAGE_JS_LIMIT:
        bad(p, f"{size} bytes (with imports) > {PAGE_JS_LIMIT} page limit")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jerry")
    a = ap.parse_args()
    cfg = json.load(open(os.path.join(ROOT, "watch/entry/src/main/config.json")))
    pages = cfg["module"]["js"][0]["pages"]
    for pg in pages:
        base = os.path.join(JS_ROOT, pg)
        for ext in (".hml", ".js"):
            if not os.path.exists(base + ext):
                bad(base + ext, "page listed in config.json is missing")
    n = 0
    for dp, _, fs in os.walk(JS_ROOT):
        for f in fs:
            p = os.path.join(dp, f)
            n += 1
            if f.endswith(".hml"):
                check_hml(p)
            elif f.endswith(".css"):
                check_css(p)
            elif f.endswith(".js"):
                check_js(p)
    if a.jerry:
        for rel in ES5_FILES:
            src = open(os.path.join(JS_ROOT, rel)).read().replace("export default GN;", "")
            with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as t:
                t.write(src)
            r = subprocess.run([a.jerry, "--parse-only", t.name], capture_output=True, text=True)
            os.unlink(t.name)
            if r.returncode != 0:
                bad(os.path.join(JS_ROOT, rel), "JerryScript (ES5.1 profile) parse error: " + (r.stdout + r.stderr)[:300])
    big = sorted(SIZES.items(), key=lambda kv: -kv[1])[:3]
    print(f"checked {n} files, {len(pages)} pages; largest pages incl. imports: " +
          ", ".join(f"{k} {v / 1024:.1f} KB" for k, v in big))
    for x in problems:
        print("FAIL", x)
    print("OK" if not problems else f"{len(problems)} problem(s)")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
