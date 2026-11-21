#!/usr/bin/env node
// Records docs/images/swipe_typing.gif: web/swipe-demo swiped by a script in
// headless Chrome (real mouse strokes through each word's keys), its
// screencast made a GIF by ffmpeg. Needs Chrome, Node 22+ and ffmpeg, and the
// demo served (`SERVE=1 web/swipe-demo/build.sh`):
//
//   node scripts/swipe-demo-gif.mjs [URL] [WORDS] [OUT_GIF]
//   node scripts/swipe-demo-gif.mjs http://localhost:8787/swipe-demo/ "hello how are you today" docs/images/swipe_typing.gif
import { spawn } from "node:child_process";
import { mkdtempSync, writeFileSync, mkdirSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const URL = process.argv[2] ?? "http://localhost:8787/swipe-demo/";
const WORDS = (process.argv[3] ?? "hello how are you today").split(" ");
const GIF = process.argv[4] ?? "docs/images/swipe_typing.gif";
const OUT = mkdtempSync(join(tmpdir(), "swipe-frames-"));
const [W, H] = [390, 640];
mkdirSync(OUT, { recursive: true });
const PORT = 9335;
// CHROME= another Chrome or Chromium binary.
const chrome = spawn(process.env.CHROME ?? "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome", [
  "--headless=new", `--remote-debugging-port=${PORT}`, `--user-data-dir=${mkdtempSync(join(tmpdir(), "swipe-chrome-"))}`,
  `--window-size=${W},${H}`, "--hide-scrollbars", "about:blank",
], { stdio: "ignore" });
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
let ws, id = 0; const pending = new Map(); const logs = []; const frames = [];
try {
  let t; for (let i = 0; i < 50; i++) { try { t = await (await fetch(`http://127.0.0.1:${PORT}/json`)).json(); break; } catch { await sleep(200); } }
  ws = new WebSocket(t.find((x) => x.type === "page").webSocketDebuggerUrl);
  await new Promise((r) => ws.addEventListener("open", r));
  const send = (method, params = {}) => new Promise((r) => { const i = ++id; pending.set(i, r); ws.send(JSON.stringify({ id: i, method, params })); });
  ws.addEventListener("message", (m) => {
    const d = JSON.parse(m.data);
    if (d.id && pending.has(d.id)) { pending.get(d.id)(d); pending.delete(d.id); }
    if (d.method === "Runtime.consoleAPICalled") logs.push(d.params.args.map((a) => a.value ?? a.description).join(" "));
    if (d.method === "Page.screencastFrame") {
      frames.push({ data: d.params.data, t: d.params.metadata.timestamp });
      send("Page.screencastFrameAck", { sessionId: d.params.sessionId });
    }
  });
  const ev = async (e) => (await send("Runtime.evaluate", { expression: e, returnByValue: true, awaitPromise: true })).result?.result?.value;
  await send("Runtime.enable");
  await send("Emulation.setDeviceMetricsOverride", { width: W, height: H, deviceScaleFactor: 2, mobile: false });
  await send("Page.navigate", { url: URL });
  for (let i = 0; i < 100 && !logs.some((l) => / ms\)$/.test(l)); i++) await sleep(200);
  await sleep(600);
  const c = await ev(`(() => { const r = document.getElementById("view").getBoundingClientRect(); return [r.left, r.top]; })()`);
  const centre = async (ch) => {
    for (const n of [ch, ch.toUpperCase()]) {
      const r = await ev(`Array.from(demo.rect(${JSON.stringify(n)}))`);
      if (r && r.length === 4) return [c[0] + r[0] + r[2] / 2, c[1] + r[1] + r[3] / 2];
    }
    throw new Error(`no key ${ch}`);
  };
  const mouse = (type, x, y) => send("Input.dispatchMouseEvent", { type, x, y, button: "left", buttons: type === "mouseReleased" ? 0 : 1, clickCount: 1 });
  await send("Page.startScreencast", { format: "png", everyNthFrame: 1, maxWidth: W * 2, maxHeight: H * 2 });
  await sleep(900);
  for (const word of WORDS) {
    const pts = []; for (const ch of word) pts.push(await centre(ch));
    await mouse("mouseMoved", ...pts[0]); await mouse("mousePressed", ...pts[0]);
    for (let k = 1; k < pts.length; k++) {
      const [a, b] = [pts[k - 1], pts[k]];
      for (let s = 1; s <= 9; s++) {
        const u = s / 9, e = u * u * (3 - 2 * u), bend = Math.sin(Math.PI * u) * 7;
        await mouse("mouseMoved", a[0] + (b[0] - a[0]) * e, a[1] + (b[1] - a[1]) * e - bend);
        await sleep(22);
      }
    }
    await mouse("mouseReleased", ...pts[pts.length - 1]);
    await sleep(650);
    console.log(word.padEnd(9), "→", JSON.stringify(await ev("demo.text()")), JSON.stringify(await ev("demo.bar()")));
  }
  await sleep(1200);
  await send("Page.stopScreencast");
  await sleep(200);
  // PNGs and a concat list, each frame shown until the next.
  let list = "";
  frames.forEach((f, i) => {
    const name = `f${String(i).padStart(5, "0")}.png`;
    writeFileSync(join(OUT, name), Buffer.from(f.data, "base64"));
    const dur = i + 1 < frames.length ? Math.max(0.01, frames[i + 1].t - f.t) : 1.5;
    list += `file '${name}'\nduration ${dur.toFixed(3)}\n`;
  });
  list += `file 'f${String(frames.length - 1).padStart(5, "0")}.png'\n`;
  writeFileSync(join(OUT, "frames.txt"), list);
  console.log(`${frames.length} frames over ${(frames.at(-1).t - frames[0].t).toFixed(1)} s`);
  // 15 fps, 360 px wide, one palette for the whole clip.
  const ff = spawn("ffmpeg", ["-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", join(OUT, "frames.txt"),
    "-vf", "fps=15,scale=360:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=96:stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle",
    "-loop", "0", GIF], { stdio: "inherit" });
  const code = await new Promise((r) => ff.on("close", r));
  if (code !== 0) throw new Error(`ffmpeg exited ${code}`);
  console.log("wrote", GIF);
} finally { ws?.close(); chrome.kill(); }
