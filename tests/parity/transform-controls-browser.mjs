// TransformControls in a real browser, three.js r165 beside threers: the
// gizmo has to be drawn (its red / green / blue handles present in the
// canvas, where three.js draws them), hovering a handle has to highlight it
// yellow, and a drag with real DOM pointer events has to move the object to
// the very position three.js moves it to.
//
//   PKG_DIR=/tmp/pkg PORT=8193 node transform-controls-browser.mjs
//
// Uses the scene pair scenes/{threejs,threers}-transform-controls.html, the
// parity server (`PKG_DIR` serves /web/pkg from another build) and the same
// Chrome as run.js (`CHROME`, else Chrome for Testing from
// scripts/wasm-browser-test.sh's cache).

import puppeteer from 'puppeteer';
import { PNG } from 'pngjs';
import fs from 'fs';
import os from 'os';
import path from 'path';
import { spawn } from 'child_process';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const port = process.env.PORT ? parseInt(process.env.PORT, 10) : 8087;
const server = spawn('node', [path.join(__dirname, 'server.js'), String(port)], { stdio: 'ignore', env: process.env });
await new Promise((r) => setTimeout(r, 400));

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
                if (fs.existsSync(exe)) found.push(exe);
            }
        }
    }
    return found.sort().pop() ?? '/opt/homebrew/bin/chromium';
}

const browser = await puppeteer.launch({
    executablePath: findChrome(),
    headless: 'new',
    args: ['--enable-unsafe-webgpu', '--enable-features=Vulkan,WebGPU', '--use-vulkan=swiftshader',
        '--use-angle=swiftshader', '--ignore-gpu-blocklist', '--no-sandbox'],
});

let failures = 0;
const check = (ok, what) => {
    console.log(`${ok ? '✓' : '✗'} ${what}`);
    if (!ok) failures++;
};

/// Pixels per colour class in an 800×600 screenshot.
function census(png) {
    const img = PNG.sync.read(Buffer.from(png));
    const n = { red: 0, green: 0, blue: 0, yellow: 0 };
    for (let i = 0; i < img.data.length; i += 4) {
        const [r, g, b] = [img.data[i], img.data[i + 1], img.data[i + 2]];
        if (r > 180 && g < 70 && b < 70) n.red++;
        else if (g > 180 && r < 70 && b < 70) n.green++;
        else if (b > 180 && r < 70 && g < 90) n.blue++;
        else if (r > 180 && g > 180 && b < 70) n.yellow++;
    }
    return n;
}

async function open(lib) {
    const page = await browser.newPage();
    await page.setViewport({ width: 800, height: 600 });
    page.on('pageerror', (e) => console.log(`[${lib} pageerror] ${e.message}`));
    await page.goto(`http://localhost:${port}/tests/parity/scenes/${lib}-transform-controls.html`, { waitUntil: 'load' });
    await page.waitForFunction(() => document.body.dataset.ready, { timeout: 60000 });
    const status = await page.evaluate(() => document.body.dataset.ready);
    if (status !== 'true') throw new Error(`${lib}: ${await page.evaluate(() => document.getElementById('err')?.textContent)}`);
    // Synthetic pointers have no capture to take.
    await page.evaluate(() => {
        const c = document.getElementById('c');
        c.setPointerCapture = () => {};
        c.releasePointerCapture = () => {};
    });
    return page;
}
const shot = async (page) => {
    // A background tab gets no animation frames.
    await page.bringToFront();
    await page.evaluate(() => new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r))));
    return page.screenshot({ type: 'png', clip: { x: 0, y: 0, width: 800, height: 600 } });
};
/// Fire DOM pointer events at NDC points on the canvas, rendering after each.
const pointer = (page, steps) => page.evaluate(async (steps) => {
    const { scene, cam, r } = window.__tc;
    const c = document.getElementById('c');
    for (const [type, x, y, button] of steps) {
        c.dispatchEvent(new PointerEvent(type, {
            clientX: (x + 1) / 2 * 800, clientY: (1 - y) / 2 * 600, button, buttons: button === 0 ? 1 : 0,
            pointerType: 'mouse', pointerId: 1, bubbles: true,
        }));
        r.render(scene, cam);
    }
}, steps);

try {
    const three = await open('threejs');
    const rs = await open('threers');

    // The gizmo is drawn.
    const [c3, cr] = [census(await shot(three)), census(await shot(rs))];
    console.log('three.js', c3, '\nthreers ', cr);
    // three.js draws the gizmo with `depthTest: false`, over everything;
    // threers' wgpu renderer depth-tests every mesh, so the parts of the
    // handles inside the boxes are hidden (the blue XY square, the back of
    // the rotate rings). Most of each colour must still be there.
    for (const k of ['red', 'green', 'blue']) {
        check(cr[k] > 0.4 * c3[k] && cr[k] < 1.6 * c3[k], `${k} gizmo pixels: ${cr[k]} (three.js ${c3[k]})`);
    }

    // A pointer three.js picks the translate gizmo's Y handle at.
    const at = await three.evaluate(() => {
        const { tA } = window.__tc;
        for (let y = 0.95; y > -0.95; y -= 0.01) {
            for (let x = -0.95; x < 0.95; x += 0.01) {
                tA.pointerHover({ x, y, button: -1 });
                if (tA.axis === 'Y') { tA.pointerHover({ x: 5, y: 5, button: -1 }); return [x, y]; }
            }
        }
        return null;
    });
    check(!!at, `three.js picks Y at ${at}`);
    if (at) {
        const [x, y] = at;
        for (const page of [three, rs]) await pointer(page, [['pointermove', x, y - 0.02, -1]]);
        const [h3, hr] = [census(await shot(three)), census(await shot(rs))];
        check(hr.yellow > 0.5 * h3.yellow && h3.yellow > 0, `hover highlights yellow: ${hr.yellow} (three.js ${h3.yellow})`);
        check(await rs.evaluate(() => window.__tc.tA.axis) === 'Y', 'threers hover picks Y');
        const steps = [['pointerdown', x, y - 0.02, 0], ['pointermove', x + 0.03, y + 0.1, -1],
            ['pointermove', x + 0.05, y + 0.2, -1], ['pointerup', x + 0.05, y + 0.2, 0]];
        await pointer(three, steps);
        await pointer(rs, steps);
        const p3 = await three.evaluate(() => window.__tc.boxA.position.toArray());
        const pr = await rs.evaluate(() => window.__tc.boxA.position.toArray());
        console.log('box after drag: three.js', p3, 'threers', pr);
        check(p3[1] > 0.1 && pr.every((v, i) => v === p3[i]), 'drag moves the box exactly as three.js');
        await shot(rs);
    }
} catch (e) {
    console.error(e);
    failures++;
} finally {
    await browser.close();
    server.kill();
}
console.log(failures ? `${failures} failed` : 'all passed');
process.exit(failures ? 1 : 0);
