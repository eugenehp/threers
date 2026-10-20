#!/usr/bin/env node
// Parity runner for bvh-csg scenes (requires BVH_CSG=1 wasm build).
import puppeteer from 'puppeteer';
import pixelmatch from 'pixelmatch';
import { PNG } from 'pngjs';
import fs from 'fs';
import path from 'path';
import { spawn } from 'child_process';
import { fileURLToPath } from 'url';
import { viewsForScene } from './parity-camera-views.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const manifest = JSON.parse(
    fs.readFileSync(path.join(__dirname, 'scenes-manifest-bvh-csg.json'), 'utf8'),
);
const SCENES = manifest.scenes;
const outDir = path.join(__dirname, 'out');
const shotDir = path.join(__dirname, 'out-bvh-csg');
fs.mkdirSync(outDir, { recursive: true });
fs.mkdirSync(shotDir, { recursive: true });

const W = 800;
const H = 600;
/** Worst view must beat this to pass (multi-view picks up AA edge noise). */
const PASS_THRESHOLD_PCT = 0.5;
const PERFECT_THRESHOLD_PCT = 0.01;
const port = 8089;
spawn('node', [path.join(__dirname, 'server.js'), String(port)], { stdio: 'ignore' });
await new Promise(r => setTimeout(r, 300));

/// The same triangles as three-bvh-csg: the Rust port cuts every face in the
/// order three-mesh-bvh's tree meets its cutting triangles, so the count is
/// exact, not just close.
function checkMeta(scene, three, threers) {
    return Number(three.triCount) > 0 && Number(threers.triCount) === Number(three.triCount);
}

function toOrbitState(view) {
    const [px, py, pz] = view.pos;
    const [tx, ty, tz] = view.target;
    return {
        position: { x: px, y: py, z: pz },
        target: { x: tx, y: ty, z: tz },
    };
}

async function setCameraView(page, view) {
    await page.evaluate((state) => {
        window.__parityCaptureLock = true;
        const ctx = window.__parityOrbitCtx;
        if (!ctx) throw new Error('__parityOrbitCtx missing');
        if (typeof window.__parityApplyOrbitState !== 'function') {
            throw new Error('parity orbit bridge not installed');
        }
        window.__parityApplyOrbitState(ctx, state);
        if (typeof ctx.render === 'function') ctx.render();
    }, toOrbitState(view));
    await page.evaluate(() => new Promise((r) => requestAnimationFrame(r)));
}

async function shootViews(browser, url, views) {
    const page = await browser.newPage();
    await page.setViewport({ width: W, height: H });
    await page.goto(url, { waitUntil: 'load', timeout: 60000 });
    await page.waitForFunction(() => document.body.dataset.ready === 'true', { timeout: 60000 });
    await page.waitForFunction(
        () => window.__parityOrbitCtx && typeof window.__parityApplyOrbitState === 'function',
        { timeout: 60000 },
    );
    const status = await page.evaluate(() => document.body.dataset.ready);
    const meta = await page.evaluate(() => ({ ...document.body.dataset }));

    const shots = [];
    for (const view of views) {
        await setCameraView(page, view);
        const png = Buffer.from(await page.screenshot({
            type: 'png',
            clip: { x: 0, y: 0, width: W, height: H },
        }));
        shots.push({ view: view.name, png });
    }

    await page.evaluate(() => { window.__parityCaptureLock = false; });
    await page.close();
    return { status, meta, shots };
}

function diffPct(imgA, imgB) {
    const a = PNG.sync.read(imgA);
    const b = PNG.sync.read(imgB);
    const diffPx = pixelmatch(a.data, b.data, null, W, H, { threshold: 0.1 });
    return (diffPx / (W * H)) * 100;
}

const browser = await puppeteer.launch({
    executablePath: process.env.CHROME || '/opt/homebrew/bin/chromium',
    headless: 'new',
    args: ['--enable-unsafe-webgpu', '--enable-features=Vulkan,WebGPU', '--use-vulkan=swiftshader', '--no-sandbox'],
});

const results = [];
let failed = 0;
for (const scene of SCENES) {
    const views = viewsForScene(scene);
    const threeUrl = `http://127.0.0.1:${port}/tests/parity/scenes/threejs-${scene}.html`;
    const threersUrl = `http://127.0.0.1:${port}/tests/parity/scenes/threers-${scene}.html`;
    const three = await shootViews(browser, threeUrl, views);
    const threers = await shootViews(browser, threersUrl, views);

    if (three.status !== 'true' || threers.status !== 'true') {
        console.log(`FAIL ${scene}: ready three=${three.status} threers=${threers.status}`);
        results.push({ scene, pass: false, err: `ready three=${three.status} threers=${threers.status}` });
        failed++;
        continue;
    }

    const viewResults = [];
    let worstPct = 0;
    let worstView = views[0]?.name || 'default';
    let worstThreePng = null;
    let worstThreersPng = null;

    for (let i = 0; i < views.length; i++) {
        const name = views[i].name;
        const tShot = three.shots[i];
        const rShot = threers.shots[i];
        if (!tShot?.png?.length || !rShot?.png?.length) {
            viewResults.push({ name, pct: null, err: 'empty screenshot' });
            worstPct = Infinity;
            worstView = name;
            continue;
        }
        const pct = diffPct(tShot.png, rShot.png);
        viewResults.push({ name, pct: Number(pct.toFixed(4)) });
        if (pct > worstPct) {
            worstPct = pct;
            worstView = name;
            worstThreePng = tShot.png;
            worstThreersPng = rShot.png;
        }
    }

    const metaOk = checkMeta(scene, three.meta, threers.meta);
    const pass = worstPct < PASS_THRESHOLD_PCT && metaOk && Number.isFinite(worstPct);
    const ok = worstPct < PERFECT_THRESHOLD_PCT && pass;

    if (worstThreePng && worstThreersPng) {
        fs.writeFileSync(path.join(shotDir, `${scene}-three.png`), worstThreePng);
        fs.writeFileSync(path.join(shotDir, `${scene}-threers.png`), worstThreersPng);
        fs.writeFileSync(path.join(shotDir, `${scene}-worst-view.txt`), `${worstView}\n`);
    }

    const pctOut = Number(worstPct.toFixed(4));
    const viewSummary = viewResults.map((v) => `${v.name}=${v.pct?.toFixed(2) ?? 'err'}%`).join(' ');
    console.log(
        `${pass ? 'PASS' : 'FAIL'} ${scene}: worst ${pctOut.toFixed(2)}% @${worstView}`
        + ` (${viewSummary})`
        + ` triCount three=${three.meta.triCount} threers=${threers.meta.triCount}`,
    );

    results.push({
        scene,
        ok,
        pass,
        pct: pctOut,
        diffPct: pctOut,
        worstView,
        views: viewResults,
        triCount: threers.meta.triCount,
    });
    if (!pass) failed++;
}

await browser.close();
fs.writeFileSync(
    path.join(outDir, 'compare-results-bvh-csg.json'),
    JSON.stringify(results, null, 2),
);
if (failed) {
    console.error(`${failed} bvh-csg scene(s) failed`);
    process.exit(1);
}
console.log('bvh-csg parity OK');
