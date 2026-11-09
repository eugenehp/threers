#!/usr/bin/env python3
"""Play `hop` in a real Chrome, on WebGPU and on WebGL2.

Builds crates/threers-hop for wasm, serves it, and drives Chrome over WebDriver:
real key events go in, and the state the game reports and real screenshots come
out. Chrome and a matching chromedriver are fetched the way
scripts/wasm-browser-test.sh fetches them.

    python3 scripts/hop-browser-test.py [webgpu] [webgl2]
"""
import functools
import http.server
import json
import os
import pathlib
import socketserver
import struct
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
import zlib
import base64

ROOT = pathlib.Path(__file__).resolve().parent.parent
TARGET = os.environ.get("CARGO_TARGET_DIR", str(ROOT / "target"))
CACHE = pathlib.Path(os.environ.get("BROWSER_CACHE", pathlib.Path(tempfile.gettempdir()) / "threers-browsers"))
PORT_DRIVER = 9517

PAGE = """<!doctype html><title>hop</title>
<style>html,body{margin:0;background:#222}canvas{display:block;width:640px;height:360px}</style>
<canvas id="c" width="640" height="360"></canvas>
<script type="module">
import * as hop from "./pkg/threers_hop.js";
const { default: init, start, state, request, answer } = hop;
window.hopErrors = [];
const report = m => window.hopErrors.push(String(m));
window.addEventListener("error", e => report(e.message));
window.addEventListener("unhandledrejection", e => report(e.reason));
const log = console.error; console.error = (...a) => { report(a.join(" ")); log(...a); };
await init();
if (hop.initThreadPool) await hop.initThreadPool(4);
await start(document.getElementById("c"));
window.hopState = () => JSON.parse(state() || "null");
window.hopRequest = request;
window.hopAnswer = answer;
</script>"""


def run(*args, **kw):
    print("$", " ".join(map(str, args)), flush=True)
    return subprocess.run(args, check=True, **kw)


THREAD_FLAGS = ("-C target-feature=+atomics,+bulk-memory -C link-arg=--shared-memory "
                "-C link-arg=--max-memory=1073741824 -C link-arg=--import-memory "
                "-C link-arg=--export=__wasm_init_tls -C link-arg=--export=__tls_size "
                "-C link-arg=--export=__tls_align -C link-arg=--export=__tls_base")


def build(out, threads=False):
    target = TARGET
    if threads:
        # Rust's prebuilt wasm std has no atomics, so this rebuilds it, in a target
        # directory of its own since the flags differ.
        target = TARGET + "-threads"
        env = {**os.environ, "RUSTFLAGS": THREAD_FLAGS, "CARGO_TARGET_DIR": target}
        run("cargo", "+nightly-2026-10-30", "build", "--offline", "--release", "-p", "threers-hop",
            "--features", "threads", "--target", "wasm32-unknown-unknown",
            "-Z", "build-std=panic_abort,std", cwd=ROOT, env=env)
    else:
        run("cargo", "build", "--offline", "--release", "-p", "threers-hop",
            "--target", "wasm32-unknown-unknown", cwd=ROOT)
    wasm = pathlib.Path(target) / "wasm32-unknown-unknown/release/threers_hop.wasm"
    run("wasm-bindgen", wasm, "--target", "web", "--out-dir", out / "pkg", "--no-typescript")
    (out / "index.html").write_text(PAGE)
    (out / "tone.ogg").write_bytes((ROOT / "crates/threers-ecs/tests/data/tone.ogg").read_bytes())
    (out / "sans.ttf").write_bytes((ROOT / "src/openscad/DejaVuSans.ttf").read_bytes())


def fetch_browsers():
    CACHE.mkdir(parents=True, exist_ok=True)

    def install(what):
        done = subprocess.run(["npx", "--yes", "@puppeteer/browsers", "install", f"{what}@stable",
                               "--path", str(CACHE)], check=True, capture_output=True, text=True)
        return done.stdout.strip().splitlines()[-1].split(" ", 1)[1]

    driver, chrome = install("chromedriver"), install("chrome")
    # A quarantined chromedriver is killed the moment it runs.
    subprocess.run(["xattr", "-dr", "com.apple.quarantine", str(CACHE)], check=False)
    return driver, chrome


class Quiet(http.server.SimpleHTTPRequestHandler):
    extensions_map = {**http.server.SimpleHTTPRequestHandler.extensions_map,
                      ".wasm": "application/wasm", ".js": "text/javascript"}

    def log_message(self, *args):
        pass

    def end_headers(self):
        # What makes a page cross-origin isolated, and so allowed shared memory.
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        super().end_headers()


class Server(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True


def decode_png(data):
    """RGB(A), 8 bits, not interlaced: what Chrome's screenshots are."""
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    at, idat = 8, b""
    while at < len(data):
        length, kind = struct.unpack(">I4s", data[at:at + 8])
        body = data[at + 8:at + 8 + length]
        at += 12 + length
        if kind == b"IHDR":
            width, height, depth, colour = struct.unpack(">IIBB", body[:10])
            assert depth == 8 and colour in (2, 6), (depth, colour)
        elif kind == b"IDAT":
            idat += body
    bpp = 3 if colour == 2 else 4
    raw, stride, rows, prev = zlib.decompress(idat), width * bpp, [], bytearray(width * bpp)
    for y in range(height):
        start = y * (stride + 1)
        kind, line = raw[start], bytearray(raw[start + 1:start + 1 + stride])
        for i in range(stride):
            a = line[i - bpp] if i >= bpp else 0
            b = prev[i]
            c = prev[i - bpp] if i >= bpp else 0
            if kind == 1:
                line[i] = (line[i] + a) & 255
            elif kind == 2:
                line[i] = (line[i] + b) & 255
            elif kind == 3:
                line[i] = (line[i] + (a + b) // 2) & 255
            elif kind == 4:
                p = a + b - c
                pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
                line[i] = (line[i] + (a if pa <= pb and pa <= pc else b if pb <= pc else c)) & 255
        rows.append(line)
        prev = line
    return width, height, bpp, rows


def census(png):
    """How many pixels of each thing in the game are on screen."""
    width, height, bpp, rows = decode_png(png)
    seen = dict(sky=0, platform=0, player=0, coin=0, text=0, black=0)
    for row in rows:
        for x in range(0, width * bpp, bpp):
            r, g, b = row[x], row[x + 1], row[x + 2]
            if r > 235 and g > 235 and b > 235:
                seen["text"] += 1
            elif r < 12 and g < 12 and b < 12:
                seen["black"] += 1
            elif b > r + 25 and b > 150:
                seen["sky"] += 1
            elif g > r + 15 and g > b + 15:
                seen["platform"] += 1
            elif r > 120 and r > g * 1.8 and r > b * 1.8:
                seen["player"] += 1
            elif r > 180 and g > 130 and b < 110:
                seen["coin"] += 1
    return seen, width * height


class Driver:
    def __init__(self, base):
        self.base, self.session = base, None

    def call(self, method, path, body=None):
        data = None if body is None else json.dumps(body).encode()
        request = urllib.request.Request(self.base + path, data=data, method=method,
                                         headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=120) as response:
            return json.load(response)["value"]

    def start(self, chrome, args):
        capabilities = {"browserName": "chrome",
                        "goog:chromeOptions": {"binary": chrome, "args": args}}
        value = self.call("POST", "/session", {"capabilities": {"alwaysMatch": capabilities}})
        self.session = value["sessionId"]

    def go(self, url):
        self.call("POST", f"/session/{self.session}/url", {"url": url})

    def js(self, script):
        return self.call("POST", f"/session/{self.session}/execute/sync",
                         {"script": script, "args": []})

    def keys(self, *steps):
        actions = [{"type": kind, "value": value} if kind != "pause" else {"type": "pause", "duration": value}
                   for kind, value in steps]
        self.call("POST", f"/session/{self.session}/actions",
                  {"actions": [{"type": "key", "id": "keyboard", "actions": actions}]})

    def touch(self, *steps):
        """A finger: real touch events, through Chrome's input pipeline."""
        self.call("POST", f"/session/{self.session}/actions", {"actions": [
            {"type": "pointer", "id": "finger", "parameters": {"pointerType": "touch"},
             "actions": list(steps)}]})

    def screenshot(self):
        return base64.b64decode(self.call("GET", f"/session/{self.session}/screenshot"))

    def stop(self):
        if self.session:
            self.call("DELETE", f"/session/{self.session}")


def state(driver):
    return driver.js("return window.hopState ? window.hopState() : null")


def wait_for(driver, test, what, seconds=60):
    end = time.time() + seconds
    while time.time() < end:
        errors = driver.js("return window.hopErrors || []")
        assert not errors, f"page errors: {errors[:3]}"
        current = state(driver)
        if current and test(current):
            return current
        time.sleep(0.05)
    raise AssertionError(f"timed out waiting for {what}: {state(driver)}")


def standing(driver):
    """Wait for the player to stop moving, and return where."""
    last = state(driver)
    for _ in range(200):
        time.sleep(0.15)
        now = state(driver)
        if abs(now["y"] - last["y"]) < 1e-3 and now["y"] > 0:
            return now
        last = now
    raise AssertionError(f"never settled: {last}")


def play(driver, url, backend):
    driver.go(url)
    start = wait_for(driver, lambda s: s["frame"] > 5, "frames", 120)
    print("  running:", start)
    assert start["backend"] == backend, f"wanted {backend}, got {start['backend']}"
    settled = wait_for(driver, lambda s: s["steps"] > 40, "the player to land", 120)
    assert 1.2 < settled["y"] < 1.45, f"not standing on the platform: {settled}"

    shot = driver.screenshot()
    seen, total = census(shot)
    print("  pixels:", seen, "of", total)
    assert seen["black"] < total * 0.5, "mostly black: the renderer drew nothing"
    assert seen["sky"] > total * 0.2, "no sky"
    assert seen["platform"] > total * 0.03, "no platforms"
    assert seen["player"] > 100, "no player"
    assert seen["coin"] > 10, "no coins"
    assert seen["text"] > 50, "no HUD text"

    # Real keys, through the page's own events.
    before = state(driver)
    driver.keys(("keyDown", "d"), ("pause", 300), ("keyUp", "d"))
    moved = wait_for(driver, lambda s: s["x"] > before["x"] + 0.3, "walking right")
    print("  walked:", round(moved["x"] - before["x"], 2))

    driver.keys(("keyDown", "a"), ("pause", 300), ("keyUp", "a"))
    wait_for(driver, lambda s: s["x"] < moved["x"] - 0.3, "walking left")

    ground = standing(driver)["y"]
    driver.keys(("keyDown", " "), ("pause", 60), ("keyUp", " "))
    jumped = wait_for(driver, lambda s: s["y"] > ground + 1.0, "jumping")
    print("  jumped to", round(jumped["y"], 2))

    # It came back down, by the game's own clock rather than this script's.
    wait_for(driver, lambda s: abs(s["y"] - ground) < 0.1, "landing", 120)

    # A jump straight up reaches the coin overhead, and the score goes up.
    standing(driver)
    for _ in range(40):
        at = state(driver)
        if abs(at["x"]) < 0.2:
            break
        driver.keys(("keyDown", "a" if at["x"] > 0 else "d"), ("pause", 40),
                    ("keyUp", "a" if at["x"] > 0 else "d"))
    standing(driver)
    driver.keys(("keyDown", " "), ("pause", 60), ("keyUp", " "))
    won = wait_for(driver, lambda s: s["coins"] == 1, "collecting the first coin")
    print("  coins:", won["coins"])

    # The sound the jump and the coin made reached Web Audio's output.
    sound = wait_for(driver, lambda s: s["audio_frames"] > 0 and s["audio_peak"] > 0.05, "sound")
    print("  audio: frames", sound["audio_frames"], "peak", round(sound["audio_peak"], 3))

    # A file came over fetch and decoded, Vorbis and all, inside wasm.
    fetched = wait_for(driver, lambda s: s["fetched_samples"] > 10000, "the fetched sound")
    print("  fetched samples:", fetched["fetched_samples"])

    # A save goes to localStorage, and a reloaded page gets it back.
    def ask(what):
        driver.js(f"hopRequest({json.dumps(what)})")
        end = time.time() + 20
        while time.time() < end:
            reply = driver.js("return hopAnswer()")
            if reply:
                return reply
            time.sleep(0.05)
        raise AssertionError(f"no answer to {what}")

    assert ask("save") == "saved"
    assert "test" in ask("list").split(","), "slot not listed"
    stored = driver.js("return Object.keys(localStorage)")
    assert "hop/save/test" in stored, stored
    driver.go(url)
    fresh = wait_for(driver, lambda s: s["frame"] > 5, "the reloaded page", 120)
    assert fresh["coins"] == 0, fresh
    assert ask("load") == "loaded"
    restored = wait_for(driver, lambda s: s["coins"] == 1, "the restored score")
    print("  restored after reload: coins", restored["coins"])

    # A finger on the screen: the left stick walks, through real touch events.
    standing(driver)
    before = state(driver)
    rect = driver.js("const r = document.getElementById('c').getBoundingClientRect();"
                     "return [r.left, r.top];")
    x, y = int(rect[0] + 120), int(rect[1] + 250)
    driver.touch({"type": "pointerMove", "x": x, "y": y, "duration": 0},
                 {"type": "pointerDown", "button": 0},
                 {"type": "pause", "duration": 80},
                 {"type": "pointerMove", "x": x, "y": y - 60, "duration": 40},
                 {"type": "pause", "duration": 250},
                 {"type": "pointerUp", "button": 0})
    pushed = wait_for(driver, lambda s: s["z"] < before["z"] - 0.2, "the touch stick walking")
    print("  touch walked:", round(before["z"] - pushed["z"], 2))

    # A heavy job on a Web Worker: the right answer, with frames running meanwhile.
    driver.js('hopRequest("job")')
    end = time.time() + 120
    reply = ""
    while time.time() < end and not reply:
        reply = driver.js("return hopAnswer()")
        time.sleep(0.05)
    print("  worker:", reply)
    assert reply.startswith("348513 primes"), reply
    frames = int(reply.split(",")[1].split()[0])
    assert frames > 10, f"the game stalled during the job: {frames} frames"


def play_threaded(driver, url):
    driver.go(url)
    start = wait_for(driver, lambda s: s["frame"] > 5, "frames", 120)
    isolated = driver.js("return crossOriginIsolated")
    assert isolated, "the page is not cross-origin isolated"
    print("  running:", start["backend"], "isolated:", isolated)
    driver.js('hopRequest("parallel")')
    frame = state(driver)["frame"]
    end = time.time() + 120
    reply = ""
    while time.time() < end and not reply:
        reply = driver.js("return hopAnswer()")
        time.sleep(0.05)
    print("  rayon:", reply)
    assert reply.endswith("threads") and int(reply.split()[0]) > 1, reply
    assert state(driver)["frame"] > frame, "the game stopped while rayon ran"


def main():
    modes = sys.argv[1:] or ["webgpu", "webgl2", "threads"]
    out = pathlib.Path(tempfile.mkdtemp(prefix="hop-web-"))
    threaded = pathlib.Path(tempfile.mkdtemp(prefix="hop-web-threads-"))
    if set(modes) - {"threads"}:
        build(out)
    if "threads" in modes:
        build(threaded, threads=True)
    driver_bin, chrome = fetch_browsers()
    handler = functools.partial(Quiet, directory=str(out))
    threaded_server = Server(("127.0.0.1", 0), functools.partial(Quiet, directory=str(threaded)))
    threading.Thread(target=threaded_server.serve_forever, daemon=True).start()
    threaded_url = f"http://127.0.0.1:{threaded_server.server_address[1]}/"
    server = Server(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_address[1]}/"
    process = subprocess.Popen([driver_bin, f"--port={PORT_DRIVER}"], stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL)
    time.sleep(1.5)
    common = ["--headless=new", "--no-sandbox", "--disable-dev-shm-usage", "--window-size=700,500",
              "--autoplay-policy=no-user-gesture-required", "--touch-events=enabled"]
    args = {
        "webgpu": common + ["--enable-unsafe-webgpu", "--enable-features=WebGPU"],
        "webgl2": common + ["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"],
        "threads": common + ["--enable-unsafe-webgpu", "--enable-features=WebGPU"],
    }
    failed = []
    try:
        for mode in modes:
            print(f"== {mode}", flush=True)
            driver = Driver(f"http://127.0.0.1:{PORT_DRIVER}")
            try:
                driver.start(chrome, args[mode])
                if mode == "threads":
                    play_threaded(driver, threaded_url)
                else:
                    play(driver, url, {"webgpu": "BrowserWebGpu", "webgl2": "Gl"}[mode])
                shot = pathlib.Path(tempfile.gettempdir()) / f"hop-{mode}.png"
                shot.write_bytes(driver.screenshot())
                print("  ok, screenshot at", shot)
            except Exception as error:
                print("  FAILED:", error)
                failed.append(mode)
            finally:
                driver.stop()
    finally:
        process.terminate()
    if failed:
        sys.exit(f"failed: {failed}")


if __name__ == "__main__":
    main()
