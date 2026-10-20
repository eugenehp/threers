#!/usr/bin/env node
// Parity runner for mesh-bvh scenes (requires MESH_BVH=1 wasm build).
import puppeteer from 'puppeteer';
import pixelmatch from 'pixelmatch';
import { PNG } from 'pngjs';
import fs from 'fs';
import path from 'path';
import { spawn } from 'child_process';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const manifest = JSON.parse(
    fs.readFileSync(path.join(__dirname, 'scenes-manifest-mesh-bvh.json'), 'utf8'),
);
const SCENES = manifest.scenes;
const outDir = path.join(__dirname, 'out');
const shotDir = path.join(__dirname, 'out-mesh-bvh');
fs.mkdirSync(outDir, { recursive: true });
fs.mkdirSync(shotDir, { recursive: true });

const port = 8088;
spawn('node', [path.join(__dirname, 'server.js'), String(port)], { stdio: 'ignore' });
await new Promise(r => setTimeout(r, 300));

function checkMeta(scene, three, threers) {
    if (scene.includes('shapecast')) {
        return three.shapecastHits === threers.shapecastHits;
    }
    if (scene.includes('intersects')) {
        return three.boxHit === threers.boxHit && three.sphereHit === threers.sphereHit;
    }
    if (scene.includes('static-gen')) {
        return three.hitCount === threers.hitCount && three.triCount === threers.triCount;
    }
    if (scene.includes('multihit')) {
        return three.hitCount === threers.hitCount
            && Number(three.hitCount) >= 2
            && three.hitDistance === threers.hitDistance;
    }
    return three.hitCount === threers.hitCount
        && three.hitCount !== '0'
        && three.hitDistance === threers.hitDistance;
}

async function shoot(browser, url) {
    const page = await browser.newPage();
    await page.setViewport({ width: 800, height: 600 });
    await page.goto(url, { waitUntil: 'load', timeout: 30000 });
    await page.waitForFunction(() => document.body.dataset.ready, { timeout: 30000 });
    const status = await page.evaluate(() => document.body.dataset.ready);
    const meta = await page.evaluate(() => ({ ...document.body.dataset }));
    const png = await page.screenshot({ type: 'png', clip: { x: 0, y: 0, width: 800, height: 600 } });
    await page.close();
    return { status, meta, png };
}

const browser = await puppeteer.launch({
    executablePath: process.env.CHROME || '/opt/homebrew/bin/chromium',
    headless: 'new',
    args: ['--enable-unsafe-webgpu', '--enable-features=Vulkan,WebGPU', '--use-vulkan=swiftshader', '--no-sandbox'],
});

const results = [];
let failed = 0;
for (const scene of SCENES) {
    const threeUrl = `http://127.0.0.1:${port}/tests/parity/scenes/threejs-${scene}.html`;
    const threersUrl = `http://127.0.0.1:${port}/tests/parity/scenes/threers-${scene}.html`;
    const three = await shoot(browser, threeUrl);
    const threers = await shoot(browser, threersUrl);
    if (three.status !== 'true' || threers.status !== 'true') {
        console.log(`FAIL ${scene}: ready three=${three.status} threers=${threers.status}`);
        results.push({ scene, pass: false, err: `ready three=${three.status} threers=${threers.status}` });
        failed++;
        continue;
    }
    if (!three.png?.length || !threers.png?.length) {
        console.log(`FAIL ${scene}: empty screenshot`);
        results.push({ scene, pass: false, err: 'empty screenshot' });
        failed++;
        continue;
    }
    const a = PNG.sync.read(Buffer.from(three.png));
    const b = PNG.sync.read(Buffer.from(threers.png));
    const diff = new PNG({ width: 800, height: 600 });
    const numDiff = pixelmatch(a.data, b.data, diff.data, 800, 600, { threshold: 0.1 });
    const pctNum = numDiff / (800 * 600) * 100;
    const pct = pctNum.toFixed(2);

    fs.writeFileSync(path.join(shotDir, `${scene}-three.png`), Buffer.from(three.png));
    fs.writeFileSync(path.join(shotDir, `${scene}-threers.png`), Buffer.from(threers.png));
    fs.writeFileSync(path.join(shotDir, `${scene}-diff.png`), PNG.sync.write(diff));

    const metaOk = checkMeta(scene, three.meta, threers.meta);
    const pass = metaOk && pctNum < 5;

    results.push({
        scene,
        ok: pctNum === 0,
        pass,
        pct: pctNum,
        diffCount: numDiff,
        approx: false,
    });

    console.log(`${pass ? 'PASS' : 'WARN'} ${scene}: ${pct}% diff  meta three=${JSON.stringify(three.meta)} threers=${JSON.stringify(threers.meta)}`);
    if (!pass) failed++;
}

await browser.close();

fs.writeFileSync(
    path.join(outDir, 'compare-results-mesh-bvh.json'),
    JSON.stringify(results, null, 2),
);

if (failed) {
    console.error(`${failed} mesh-bvh scene(s) failed`);
    process.exit(1);
}
console.log('mesh-bvh parity OK');
