// Drive web/rlx-bench.html in a real Chrome and report what it actually does.
//
// Node answers "does the wasm run". Only a browser answers "does the page
// work": module worker, transferable buffers, whether WebGPU is present, what
// rlx reports as its devices, and — the point of the worker — whether the main
// thread keeps painting while the work happens.
//
//     npm i puppeteer            # or set CHROME=/path/to/chrome
//     RLX=1 RLX_GEO=1 web/build.sh
//     node web/scripts/rlx-bench-validate.mjs
//
// Prints one tab-separated line per measurement, and exits non-zero if the
// page logged an error that was not Chrome asking for a favicon.

import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import puppeteer from 'puppeteer';

const ROOT = path.resolve(import.meta.dirname, '..');
const PORT = 8791;

const MIME = {
    '.html': 'text/html; charset=utf-8',
    '.js': 'text/javascript; charset=utf-8',
    '.mjs': 'text/javascript; charset=utf-8',
    '.wasm': 'application/wasm',
    '.json': 'application/json',
    '.css': 'text/css',
};

const server = http.createServer((req, res) => {
    const url = decodeURIComponent(req.url.split('?')[0]);
    const file = path.join(ROOT, url === '/' ? '/index.html' : url);
    if (!file.startsWith(ROOT) || !fs.existsSync(file) || fs.statSync(file).isDirectory()) {
        res.writeHead(404).end('not found');
        return;
    }
    res.writeHead(200, {
        'Content-Type': MIME[path.extname(file)] ?? 'application/octet-stream',
        // Not required here, but this is what a page needs for threaded wasm;
        // leaving it on proves the harness is not accidentally relying on its
        // absence.
        'Cross-Origin-Opener-Policy': 'same-origin',
        'Cross-Origin-Embedder-Policy': 'require-corp',
    });
    fs.createReadStream(file).pipe(res);
});

await new Promise((r) => server.listen(PORT, r));
console.log(`SERVER\thttp://localhost:${PORT}`);

const browser = await puppeteer.launch({
    executablePath: process.env.CHROME,
    headless: true,
    args: [
        '--enable-unsafe-webgpu',
        '--enable-features=Vulkan',
        '--use-angle=metal',
        '--no-sandbox',
    ],
});
console.log(`BROWSER\t${await browser.version()}`);

const page = await browser.newPage();
const errors = [];
page.on('console', (m) => {
    if (m.type() === 'error') errors.push(m.text());
});
page.on('pageerror', (e) => errors.push(String(e)));
page.on('requestfailed', (r) => errors.push(`requestfailed ${r.url()}`));
page.on('response', (r) => {
    if (r.status() >= 400) console.log(`HTTP-${r.status()}\t${r.url()}`);
});

await page.goto(`http://localhost:${PORT}/rlx-bench.html`, { waitUntil: 'load' });

// Does this Chrome have WebGPU at all? The answer decides what the rlx device
// list is allowed to look like.
const gpu = await page.evaluate(async () => {
    if (!navigator.gpu) return 'absent';
    const adapter = await navigator.gpu.requestAdapter().catch(() => null);
    return adapter ? 'present' : 'no-adapter';
});
console.log(`WEBGPU\t${gpu}`);

await page.click('#run');
await page.waitForFunction(() => window.__bench?.done === true, { timeout: 600_000 });

const bench = await page.evaluate(() => window.__bench);
console.log(`DEVICES\t${bench.devices}`);
for (const row of bench.rows) {
    console.log(`BROWSER-BENCH\t${row.label}\t${row.size}\t${row.first.toFixed(1)}\t${row.best.toFixed(1)}`);
}

// The worker's whole justification, measured.
await page.click('#runMain');
await page.waitForFunction(() => window.__bench?.main !== null, { timeout: 600_000 });
const { main, worker } = await page.evaluate(() => ({
    main: window.__bench.main,
    worker: window.__bench.worker,
}));
console.log(`MAINTHREAD\t${main.ms.toFixed(0)}\t${main.frames}`);
console.log(`INWORKER\t${worker.ms.toFixed(0)}\t${worker.frames}`);

await page.screenshot({ path: 'bench.png', fullPage: true });

// Chrome asks every page for a favicon; a static server that has none is not
// a failure of the thing under test.
const real = errors.filter((e) => !/favicon/i.test(e) && !/status of 404/.test(e));
for (const e of real) console.log(`CONSOLE-ERROR\t${e}`);
console.log(`ERRORS\t${real.length}`);

await browser.close();
server.close();
console.log('VALIDATE DONE');
process.exit(real.length === 0 ? 0 : 1);
