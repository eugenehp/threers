// End-to-end parity suite: for each scene, render the three.js reference and
// the threers (wasm/wgpu) version, screenshot both, pixelmatch. Aggregate.

import puppeteer from 'puppeteer';
import pixelmatch from 'pixelmatch';
import { PNG } from 'pngjs';
import fs from 'fs';
import path from 'path';
import { spawn } from 'child_process';
import os from 'os';
import { fileURLToPath } from 'url';
import { ALL_SCENES, coverageReport, APPROXIMATE_SCENES, APPROXIMATE_THRESHOLD } from './coverage-manifest.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const outDir = path.join(__dirname, 'out');
fs.mkdirSync(outDir, { recursive: true });

// `PORT`: another port, when 8087 is someone else's server (which would be
// used instead of the one spawned here, PKG_DIR and all).
const port = process.env.PORT ? parseInt(process.env.PORT, 10) : 8087;
let server = null;
let spawnedServer = false;
try {
    server = spawn('node', [path.join(__dirname, 'server.js'), String(port)], { stdio: 'ignore' });
    spawnedServer = true;
    server.on('error', () => { spawnedServer = false; });
} catch (_) { /* use existing server on port */ }
await new Promise(r => setTimeout(r, 300));

// `SCENES=a,b,c node run.js` runs just those (any scene with a pair in
// scenes/, listed in the manifest or not).
const SCENES = process.env.SCENES
    ? process.env.SCENES.split(',').map(s => s.trim()).filter(Boolean)
    : ALL_SCENES;
const cov = coverageReport();
// The API inventory imports the shim into node, which fails while
// web/deps/package.json says "commonjs" (Earcut.js is an ES module); the
// report is a nicety, the pixel comparison is not.
let apiCov = { totalExports: '?', exportCoveragePct: '?', sceneTestedApis: '?' };
try {
    apiCov = (await import('./api-inventory.js')).apiCoverageReport();
} catch (e) {
    console.log(`(API inventory unavailable: ${e.message.split('\n')[0]})`);
}
console.log(`Parity suite: ${cov.totalScenes} scenes (${cov.coreScenes} core + ${cov.generatedScenes} generated; ${cov.approximateScenes} approximate postfx/animation)`);
console.log(`API surface: ${apiCov.totalExports} exports, ~${apiCov.exportCoveragePct}% with parity scenes (${apiCov.sceneTestedApis} APIs mapped)`);

async function shoot(browser, url) {
    const page = await browser.newPage();
    await page.setViewport({ width: 800, height: 600 });
    const consoleMsgs = [];
    page.on('console', async m => {
        const t = m.type();
        const parts = [];
        for (const a of m.args()) {
            try {
                const d = await a.executionContext().evaluate(o => {
                    if (typeof o === 'string') return o;
                    try { return o.message || o.toString() || JSON.stringify(o); } catch { return String(o); }
                }, a);
                parts.push(d);
            } catch { parts.push(String(a)); }
        }
        consoleMsgs.push(`[${t}] ${parts.join(' ') || m.text()}`);
    });
    page.on('pageerror', e => consoleMsgs.push(`[ERROR] ${e.message}`));
    let status, errText, png;
    try {
        await page.goto(url, { waitUntil: 'load', timeout: 30000 });
        try {
            await page.waitForFunction(() => document.body.dataset.ready, { timeout: 30000 });
        } catch (e) {
            consoleMsgs.push(`[ERROR] waitForReady timed out — page never set body.dataset.ready`);
        }
        status = await page.evaluate(() => document.body.dataset.ready).catch(() => 'timeout');
        if (status !== 'true') {
            errText = await page.evaluate(() => document.getElementById('err')?.textContent || '(no #err element)').catch(() => '(eval failed)');
        }
        png = await page.screenshot({ type: 'png', clip: { x: 0, y: 0, width: 800, height: 600 } }).catch(() => null);
    } finally {
        await page.close().catch(() => {});
    }
    return { status, errText, consoleMsgs, png };
}

/// The browser to drive: `CHROME` if set; else the newest Chrome for Testing
/// that scripts/wasm-browser-test.sh fetched into its cache
/// (`${BROWSER_CACHE:-$TMPDIR/threers-browsers}`); else Homebrew's chromium.
function findChrome() {
    if (process.env.CHROME) return process.env.CHROME;
    const cache = process.env.BROWSER_CACHE || path.join(process.env.TMPDIR || os.tmpdir(), 'threers-browsers');
    const builds = path.join(cache, 'chrome');
    const found = [];
    if (fs.existsSync(builds)) {
        for (const build of fs.readdirSync(builds)) {
            for (const app of [
                'chrome-mac-arm64/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing',
                'chrome-mac-x64/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing',
                'chrome-linux64/chrome',
            ]) {
                const exe = path.join(builds, build, app);
                if (fs.existsSync(exe)) found.push({ build, exe });
            }
        }
    }
    const version = (b) => (b.split('-')[1] || '').split('.').map(Number);
    found.sort((a, b) => {
        const [x, y] = [version(a.build), version(b.build)];
        for (let i = 0; i < Math.max(x.length, y.length); i++) {
            if ((x[i] || 0) !== (y[i] || 0)) return (x[i] || 0) - (y[i] || 0);
        }
        return 0;
    });
    if (found.length) return found[found.length - 1].exe;
    return '/opt/homebrew/bin/chromium';
}

const chrome = findChrome();
if (!fs.existsSync(chrome)) {
    console.error(`no browser at ${chrome}: set CHROME, or run scripts/wasm-browser-test.sh once to fetch Chrome for Testing`);
    process.exit(2);
}
console.log(`Launching headless ${chrome} (WebGPU enabled)...`);
const browser = await puppeteer.launch({
    executablePath: chrome,
    headless: 'new',
    args: [
        '--enable-unsafe-webgpu',
        '--enable-features=Vulkan,WebGPU',
        '--use-vulkan=swiftshader',
        '--no-sandbox',
        '--disable-dev-shm-usage',
        '--ignore-gpu-blocklist',
        '--enable-webgl',
    ],
});

const results = [];
for (const scene of SCENES) {
    process.stdout.write(`▶ ${scene.padEnd(16)} `);
    const refUrl = `http://localhost:${port}/tests/parity/scenes/threejs-${scene}.html`;
    const cmpUrl = `http://localhost:${port}/tests/parity/scenes/threers-${scene}.html`;
    const ref = await shoot(browser, refUrl);
    const cmp = await shoot(browser, cmpUrl);
    // page.screenshot() returns a Uint8Array under newer puppeteer; pngjs needs a Buffer.
    const refBuf = ref.png ? Buffer.from(ref.png) : Buffer.alloc(0);
    const cmpBuf = cmp.png ? Buffer.from(cmp.png) : Buffer.alloc(0);
    fs.writeFileSync(path.join(outDir, `${scene}-threejs.png`), refBuf);
    fs.writeFileSync(path.join(outDir, `${scene}-threers.png`), cmpBuf);

    if (ref.status !== 'true' || cmp.status !== 'true') {
        console.log('FAIL (page error)');
        if (ref.status !== 'true') console.log(`   three.js: ${ref.errText}`);
        if (cmp.status !== 'true') {
            console.log(`   threers : ${cmp.errText}`);
            cmp.consoleMsgs.filter(m => /error|warn/i.test(m)).slice(0, 8).forEach(m => console.log(`     ${m}`));
        }
        results.push({ scene, ok: false, err: cmp.errText || ref.errText });
        continue;
    }
    const refPng = PNG.sync.read(refBuf);
    const cmpPng = PNG.sync.read(cmpBuf);
    const diff = new PNG({ width: refPng.width, height: refPng.height });
    // threshold 0.2 — strict enough to catch convention/formula bugs (we caught
    // both the missing gamma encode and the ambient/π factor at this level)
    // while ignoring sub-byte precision noise from FP/GGX implementation diffs.
    const diffCount = pixelmatch(refPng.data, cmpPng.data, diff.data, refPng.width, refPng.height, { threshold: 0.2 });
    fs.writeFileSync(path.join(outDir, `${scene}-diff.png`), PNG.sync.write(diff));
    const totalPx = refPng.width * refPng.height;
    const pct = diffCount / totalPx * 100;
    const approx = APPROXIMATE_SCENES.includes(scene);
    const pass = diffCount === 0
        || (approx ? pct < APPROXIMATE_THRESHOLD : pct < 5);
    let tag = diffCount === 0 ? '✅ PERFECT'
              : pct < 0.5      ? '✅ identical'
              : pct < 5        ? '✓ matching'
              : pct < 15       ? '⚠ minor'
              :                  '❌ diverged';
    if (approx && pass && diffCount > 0) tag = '⚠ approximate';
    console.log(`${tag}  ${pct.toFixed(2)}% (${diffCount}/${totalPx})${approx ? ' [approx]' : ''}`);
    // Surface threers console messages on big divergence — helpful for debugging.
    if (pct >= 5) {
        cmp.consoleMsgs.filter(m => !/Failed to load resource|deprecated parameters/.test(m)).slice(0, 12).forEach(m => console.log(`     ${m}`));
    }
    results.push({ scene, ok: diffCount === 0, pass, pct, diffCount, approx });
}

fs.writeFileSync(
    path.join(outDir, 'compare-results.json'),
    JSON.stringify(results.map(r => ({
        scene: r.scene,
        ok: r.ok,
        pass: r.pass,
        pct: r.pct,
        diffCount: r.diffCount,
        approx: r.approx,
        err: r.err,
    })), null, 2),
);

await browser.close();
if (spawnedServer) server?.kill();

console.log('\n─── Summary ───');
console.log(`API exports: ${apiCov.totalExports} total, ${apiCov.exportCoveragePct}% with dedicated parity scenes`);
const passed = results.filter(r => r.ok).length;
console.log(`${passed}/${results.length} pixel-perfect`);
for (const r of results) {
    if (r.ok) console.log(`  ✅ ${r.scene}`);
    else if (r.err) console.log(`  ❌ ${r.scene}  (load error)`);
    else if (r.pass && r.approx) console.log(`  ⚠ ${r.scene}  ${r.pct.toFixed(2)}% (approx)`);
    else if (r.pass) console.log(`  ✓ ${r.scene}  ${r.pct.toFixed(2)}%`);
    else console.log(`  ${r.pct < 5 ? '✓' : '⚠'} ${r.scene}  ${r.pct.toFixed(2)}%`);
}
process.exit(results.every(r => r.ok || r.pass) ? 0 : 1);
