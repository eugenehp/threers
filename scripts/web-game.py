#!/usr/bin/env python3
"""Build a threers-ecs game for the browser, and serve it.

    python3 scripts/web-game.py -p threers-hop --start start_race --title Race --out dist/race
    python3 scripts/web-game.py -p threers-hop --start start_race --serve 8080 --watch \
        --asset src/openscad/DejaVuSans.ttf=sans.ttf

The package is a `cdylib` whose `#[wasm_bindgen]` function `--start` takes a
canvas and runs the game, as `threers-hop`'s `start_race` does. This builds it
fully optimised (LTO, `wasm-opt -O3`, gzip), writes a page with a loading bar,
a click to start (which is also what lets sound play) and fullscreen on F, and
with `--serve` serves it, sending the gzipped files to browsers that take them.
With `--watch` it rebuilds when a source file changes and the open page reloads
itself. `--asset FILE` copies a file beside the page, for `AssetsPlugin::new("")`.
"""

import argparse
import gzip
import http.server
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

PAGE = """<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title>
<style>
html,body{{margin:0;height:100%;background:#0d1117;overflow:hidden;font:15px system-ui,sans-serif;color:#eee}}
canvas{{display:block;width:100vw;height:100vh;outline:none}}
#cover{{position:fixed;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:18px;background:radial-gradient(#1d2a3a,#0d1117)}}
#cover h1{{margin:0;font-size:56px;letter-spacing:6px}}
#bar{{width:320px;height:8px;background:#ffffff22;border-radius:4px;overflow:hidden}}
#fill{{height:100%;width:0;background:#ffcc33;transition:width .1s}}
#go{{display:none;padding:14px 34px;font-size:20px;border:0;border-radius:10px;background:#ffcc33;color:#111;cursor:pointer}}
#note{{opacity:.7;text-align:center;line-height:1.7;max-width:640px}}
#full{{position:fixed;top:12px;right:12px;padding:6px 10px;border:0;border-radius:6px;background:#0008;color:#fff;cursor:pointer;display:none}}
</style>
<canvas id="c" tabindex="0"></canvas>
<button id="full" title="Fullscreen (F)">⛶</button>
<div id="cover"><h1>{title}</h1><div id="bar"><div id="fill"></div></div><div id="status">Loading…</div>
<button id="go">Click to play</button><div id="note">{note}<br>F fullscreen · sound starts with the click</div></div>
<script type="module">
const $ = id => document.getElementById(id);
const canvas = $("c");
const response = await fetch("./pkg/{module}_bg.wasm");
const total = Number(response.headers.get("X-Raw-Length")) || Number(response.headers.get("Content-Length")) || 0;
const reader = response.body.getReader();
const chunks = []; let got = 0;
for (;;) {{
  const {{ done, value }} = await reader.read();
  if (done) break;
  chunks.push(value); got += value.length;
  if (total) {{ $("fill").style.width = Math.min(100, got / total * 100) + "%"; $("status").textContent = `Loading ${{(got / 1e6).toFixed(1)}} of ${{(total / 1e6).toFixed(1)}} MB`; }}
}}
const bytes = new Uint8Array(got); let at = 0;
for (const c of chunks) {{ bytes.set(c, at); at += c.length; }}
$("status").textContent = "Compiling…";
const game = await import("./pkg/{module}.js");
await game.default({{ module_or_path: bytes }});
$("fill").style.width = "100%"; $("status").textContent = "Ready"; $("go").style.display = "block";
const fullscreen = () => (document.fullscreenElement ? document.exitFullscreen() : document.documentElement.requestFullscreen?.())?.catch?.(() => {{}});
$("full").onclick = () => {{ fullscreen(); canvas.focus(); }};
addEventListener("keydown", e => {{ if (e.code === "KeyF" && !e.repeat) fullscreen(); }});
canvas.addEventListener("pointerdown", () => canvas.focus());
const play = async () => {{ $("cover").remove(); $("full").style.display = "block"; canvas.focus(); await game.{start}(canvas); }};
$("go").onclick = play;
if (new URLSearchParams(location.search).has("auto")) setTimeout(play, 0);
// Under `--watch`: reload when a new build is served.
fetch("./__version").then(r => r.ok ? r.text() : null).then(first => {{
  if (first === null) return;
  setInterval(async () => {{ const v = await fetch("./__version").then(r => r.text()).catch(() => first); if (v !== first) location.reload(); }}, 1000);
}});
</script>
"""

OPT_ENV = {
    "CARGO_PROFILE_RELEASE_LTO": "fat",
    "CARGO_PROFILE_RELEASE_CODEGEN_UNITS": "1",
    "CARGO_PROFILE_RELEASE_STRIP": "true",
    "CARGO_PROFILE_RELEASE_PANIC": "abort",
}


def run(cmd, **kw):
    print("$", " ".join(map(str, cmd)), flush=True)
    return subprocess.run(cmd, cwd=ROOT, **kw).returncode == 0


def build(a):
    out = Path(a.out).resolve()
    target = Path(os.environ.get("CARGO_TARGET_DIR", ROOT / "target")) / "web-game"
    env = {**os.environ, **OPT_ENV, "CARGO_TARGET_DIR": str(target)}
    cargo = ["cargo", "build", "--release", "-p", a.package, "--target", "wasm32-unknown-unknown"]
    if a.features:
        cargo += ["--features", a.features]
    if not run(cargo, env=env):
        return False
    module = a.package.replace("-", "_")
    wasm = target / "wasm32-unknown-unknown/release" / f"{module}.wasm"
    pkg = out / "pkg"
    shutil.rmtree(pkg, ignore_errors=True)
    if not run(["wasm-bindgen", wasm, "--target", "web", "--out-dir", pkg, "--no-typescript"]):
        return False
    bg = pkg / f"{module}_bg.wasm"
    if shutil.which("wasm-opt") and not a.no_opt:
        features = ["--enable-bulk-memory", "--enable-nontrapping-float-to-int", "--enable-sign-ext", "--enable-mutable-globals"]
        if not run(["wasm-opt", "-O3", "--strip-debug", *features, bg, "-o", bg]):
            return False
    for f in (bg, pkg / f"{module}.js"):
        (f.parent / (f.name + ".gz")).write_bytes(gzip.compress(f.read_bytes(), 9))
    note = a.note.replace("<", "&lt;")
    (out / "index.html").write_text(PAGE.format(title=a.title, module=module, start=a.start, note=note))
    for asset in a.asset:
        source, _, name = asset.partition("=")
        shutil.copy(ROOT / source, out / (name or Path(source).name))
    size = bg.stat().st_size / 1e6
    packed = (pkg / (bg.name + ".gz")).stat().st_size / 1e6
    print(f"[web-game] {out}: {size:.1f} MB, {packed:.1f} MB gzipped", flush=True)
    return True


class Version:
    value = str(time.time())


def serve(directory, port, verbose=False):
    class Handler(http.server.SimpleHTTPRequestHandler):
        extensions_map = {**http.server.SimpleHTTPRequestHandler.extensions_map,
                          ".wasm": "application/wasm", ".js": "text/javascript"}

        def __init__(self, *args, **kw):
            super().__init__(*args, directory=str(directory), **kw)

        def do_GET(self):
            if self.path.startswith("/__version"):
                body = Version.value.encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)
                return
            path = self.translate_path(self.path.split("?")[0])
            gz = path + ".gz"
            if os.path.isfile(gz) and "gzip" in self.headers.get("Accept-Encoding", ""):
                data = open(gz, "rb").read()
                self.send_response(200)
                self.send_header("Content-Type", self.guess_type(path))
                self.send_header("Content-Encoding", "gzip")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("X-Raw-Length", str(os.path.getsize(path)))
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                self.wfile.write(data)
                return
            super().do_GET()

        def log_message(self, fmt, *args):
            if verbose:
                print(f"[web-game] {self.command} {self.path}", flush=True)

    server = http.server.ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"[web-game] serving http://127.0.0.1:{port}/", flush=True)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def newest():
    stamp = 0.0
    for base in (ROOT / "src", ROOT / "crates"):
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = [d for d in dirnames if d not in ("target", "node_modules", ".venv", "dist") and not d.startswith(".")]
            for name in filenames:
                if name.endswith((".rs", ".wgsl", ".toml")):
                    stamp = max(stamp, os.stat(os.path.join(dirpath, name)).st_mtime)
    return stamp


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-p", "--package", required=True)
    p.add_argument("--start", required=True, help="the #[wasm_bindgen] function that takes the canvas")
    p.add_argument("--features", default="")
    p.add_argument("--out", default="dist/game")
    p.add_argument("--title", default="Game")
    p.add_argument("--note", default="", help="a line under the title: the controls, say")
    p.add_argument("--asset", action="append", default=[], help="a file to copy beside the page, as PATH or PATH=NAME")
    p.add_argument("--verbose", action="store_true", help="print each request served")
    p.add_argument("--serve", type=int, metavar="PORT")
    p.add_argument("--watch", action="store_true", help="rebuild on change; the page reloads")
    p.add_argument("--no-opt", action="store_true", help="skip wasm-opt, for a quicker build")
    a = p.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    if not build(a):
        sys.exit(1)
    if a.serve is None:
        return
    serve(Path(a.out).resolve(), a.serve, a.verbose)
    seen = newest()
    try:
        while True:
            time.sleep(1)
            if a.watch and newest() > seen:
                seen = newest()
                print("[web-game] change seen, building", flush=True)
                if build(a):
                    Version.value = str(time.time())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
