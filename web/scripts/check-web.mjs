#!/usr/bin/env node
// Two checks the browser demos need and a compiler cannot give us.
//
//   node web/scripts/check-web.mjs            both
//   node web/scripts/check-web.mjs --syntax   language floor only (no browser)
//   node web/scripts/check-web.mjs --pages    page loads only
//
// **Syntax floor.** A syntax error in a module is not recoverable: the module
// never loads, every export goes with it, and the browser reports one message
// pointing at a line that is perfectly good while the page shows nothing. That
// happened here — `threejs-shim.js` used static class fields, which are ES2022
// and need Safari 16.4, and since every page imports the shim every page died.
// It took two rounds and a parser sweep to find, because Chrome parses it fine.
//
// `tsc --target` will not catch this: it downlevels the syntax rather than
// rejecting it. So this walks the AST instead and names the constructs that
// carry a browser floor higher than the one we mean to support.
//
// **Page loads.** Chrome is past all of these floors, so the check above is the
// only one that can speak for Safari — but a page that parses can still throw
// on the first frame, and that is worth catching too.

import { readdirSync, readFileSync, statSync } from 'node:fs';
import { dirname, join, relative } from 'node:path';
import { fileURLToPath } from 'node:url';
import { createRequire } from 'node:module';

const WEB = dirname(dirname(fileURLToPath(import.meta.url)));
const ROOT = dirname(WEB);
const require = createRequire(join(WEB, 'noop.js'));

const only = process.argv.find((a) => a === '--syntax' || a === '--pages');
let failures = 0;

// ---------------------------------------------------------------- syntax floor

/**
 * Every `.js` under web/ that a browser actually loads.
 *
 * `scripts/` is excluded: it is build tooling that runs in Node, where the
 * floor is whatever Node is, and this file itself uses top-level await.
 */
function sources(dir = WEB, out = []) {
    for (const name of readdirSync(dir)) {
        if (name === 'node_modules' || name === 'scripts' || name.startsWith('.')) continue;
        const path = join(dir, name);
        if (statSync(path).isDirectory()) sources(path, out);
        else if (name.endsWith('.js') || name.endsWith('.mjs')) out.push(path);
    }
    return out;
}

/**
 * Constructs that raise the browser floor above ES2020 (Safari 14).
 *
 * Not a general lint — each entry is here because it is fatal at *parse* time,
 * which is what makes it worth a build-time check rather than a runtime one.
 */
function checkSyntaxFloor(ts) {
    const banned = new Map([
        [ts.SyntaxKind.PropertyDeclaration, 'class field (ES2022, Safari 14.1)'],
        [ts.SyntaxKind.ClassStaticBlockDeclaration, 'static block (ES2022, Safari 16.4)'],
        [ts.SyntaxKind.PrivateIdentifier, 'private name (ES2022, Safari 14.1)'],
    ]);
    let found = 0;
    for (const path of sources()) {
        const text = readFileSync(path, 'utf8');
        const file = ts.createSourceFile(path, text, ts.ScriptTarget.ESNext, true);
        const report = (node, what) => {
            const { line } = file.getLineAndCharacterOfPosition(node.getStart(file));
            console.log(`  ${relative(ROOT, path)}:${line + 1}  ${what}`);
            found++;
        };
        const walk = (node, inFunction) => {
            const why = banned.get(node.kind);
            if (why) report(node, why);
            // Top-level await is ES2022 module syntax and needs Safari 15.
            if (node.kind === ts.SyntaxKind.AwaitExpression && !inFunction) {
                report(node, 'top-level await (ES2022, Safari 15)');
            }
            const opens =
                ts.isFunctionDeclaration(node) ||
                ts.isFunctionExpression(node) ||
                ts.isArrowFunction(node) ||
                ts.isMethodDeclaration(node) ||
                ts.isConstructorDeclaration(node) ||
                ts.isGetAccessor(node) ||
                ts.isSetAccessor(node);
            ts.forEachChild(node, (c) => walk(c, inFunction || opens));
        };
        walk(file, false);
    }
    if (found) {
        console.log(`\nsyntax floor: ${found} construct(s) above ES2020`);
        failures += found;
    } else {
        console.log('syntax floor: everything under web/ parses as ES2020 (Safari 14)');
    }
}

// ------------------------------------------------------------------ page loads

async function checkPages(puppeteer, chrome) {
    const { spawn } = await import('node:child_process');
    const port = 8788;
    const server = spawn('python3', ['-m', 'http.server', String(port)], {
        cwd: ROOT,
        stdio: 'ignore',
    });
    await new Promise((r) => setTimeout(r, 1500));
    const browser = await puppeteer.launch({
        executablePath: chrome,
        args: ['--enable-unsafe-webgpu', '--use-angle=metal', '--no-sandbox'],
    });
    const pages = readdirSync(join(WEB, 'examples')).filter((f) => f.endsWith('.html'));
    for (const name of pages) {
        const page = await browser.newPage();
        await page.setCacheEnabled(false);
        const errs = [];
        page.on('pageerror', (e) => errs.push(e.message.split('\n')[0]));
        try {
            await page.goto(`http://localhost:${port}/web/examples/${name}`, {
                waitUntil: 'networkidle2',
                timeout: 45000,
            });
            await new Promise((r) => setTimeout(r, 4000));
        } catch (e) {
            errs.push(`navigation: ${e.message.split('\n')[0]}`);
        }
        // Demos that need a feature the build lacks say so deliberately; that
        // is a message, not a crash.
        const real = errs.filter((e) => !/rebuild wasm|Rebuild with|disabled/i.test(e));
        console.log(`  ${name.padEnd(26)} ${real.length ? 'FAIL ' + real[0].slice(0, 90) : 'ok'}`);
        failures += real.length;
        await page.close();
    }
    await browser.close();
    server.kill();
}

// ------------------------------------------------------------------------ main

let ts;
try {
    ts = require('typescript');
} catch {
    console.log('syntax floor: skipped (npm install in web/ for typescript)');
}
if (ts && only !== '--pages') checkSyntaxFloor(ts);

if (only !== '--syntax') {
    // Puppeteer is a test dependency, so look where the parity harness keeps
    // it as well as in web/.
    let puppeteer = null;
    for (const from of [join(WEB, 'noop.js'), join(ROOT, 'tests/parity/noop.js')]) {
        try {
            puppeteer = createRequire(from)('puppeteer');
            break;
        } catch { /* try the next */ }
    }
    // Puppeteer's bundled Chrome is often absent; fall back to whatever is in
    // its cache rather than failing the whole check over it.
    let chrome;
    try {
        const base = join(process.env.HOME ?? '', '.cache/puppeteer/chrome');
        const ver = readdirSync(base)[0];
        chrome = join(
            base,
            ver,
            'chrome-mac-arm64/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing',
        );
        statSync(chrome);
    } catch {
        chrome = undefined;
    }
    if (puppeteer) {
        console.log('\npage loads:');
        await checkPages(puppeteer, chrome);
    } else {
        console.log('\npage loads: skipped (no puppeteer)');
    }
}

process.exit(failures ? 1 : 0);
