#!/usr/bin/env node
// Stage web/ as a static site for a CDN: Vercel, GitHub Pages, Netlify,
// Cloudflare Pages, or any host that serves files.
//
//   VARIANT=full ./web/build.sh          # build the wasm first
//   node web/scripts/stage-site.mjs      # → site/
//
// What it does to the copy (web/ itself is left alone):
//
// - **Content-hashed wasm.** Every `*_bg.wasm` becomes `*_bg.<hash>.wasm`, and
//   every reference to it — the wasm-bindgen glue's default URL and the
//   examples' explicit `initThreers('/web/pkg/threers_bg.wasm')` alike — is
//   rewritten. A wasm file can then be cached for a year (`immutable`): a new
//   build is a new name. The JS and HTML that name it stay revalidated on
//   every load, which costs a 304, not a download.
// - **Preload hints.** A page that loads the shim gets `modulepreload` for
//   the shim and the glue, and a `preload` for the wasm, so the 1–3 MB
//   download starts with the HTML instead of after two module round trips.
// - **Minified JS** (esbuild via npx, as `web/build.sh` fetches wasm-opt;
//   `MINIFY=0` skips it). The shim is 0.7 MB as written and well under half
//   that minified; source maps are written beside each file.
// - **Headers** for the hosts that take them: `vercel.json` and `_headers`
//   (Netlify, Cloudflare Pages) with `application/wasm` — without which the
//   browser cannot compile while downloading — and the caching above.
//   `.nojekyll` for GitHub Pages, which takes no headers but serves wasm with
//   the right type and is safe with hashed names.
// - **Size budget.** Fails if any file exceeds `MAX_FILE_MB` (default 25:
//   Cloudflare Pages' per-file limit, the tightest of the four), and prints
//   the largest files and the total.
//
// Left out: node_modules, build scripts, TypeScript sources, `.d.ts`, the
// pre-compressed `.br`/`.gz` copies (every host above compresses itself, with
// brotli where the browser asks for it; `PRECOMPRESS=1` writes fresh ones for
// a host that serves them, such as nginx `brotli_static`), and the 6 GB of
// fetched Earth imagery under `assets/earth/` (its `CREDITS` file is kept).

import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import zlib from 'node:zlib';

const repo = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const web = path.join(repo, 'web');
const out = path.resolve(process.env.OUT || path.join(repo, 'site'));
const siteWeb = path.join(out, 'web');
const minify = process.env.MINIFY !== '0';
const precompress = process.env.PRECOMPRESS === '1';
const maxFileMb = Number(process.env.MAX_FILE_MB || 25);

if (!fs.existsSync(path.join(web, 'pkg', 'threers_bg.wasm'))) {
  console.error('web/pkg/threers_bg.wasm is missing: run `VARIANT=mini ./web/build.sh` (or full) first.');
  process.exit(1);
}

// --- copy ------------------------------------------------------------------

const skipDirs = new Set(['node_modules']);
// TypeScript sources and declarations, build scripts, old source maps and
// compressed copies, and the npm metadata wasm-bindgen writes beside its output.
const skipFile = (rel) =>
  /\.(sh|ts|map|br|gz)$/.test(rel)
  || /^(package(-lock)?\.json|tsconfig\.json|build-id\.txt|variant\.txt)$/.test(path.basename(rel))
  || /^pkg[^/]*\/(README\.md|\.gitignore)$/.test(rel);
function copyTree(src, dst, rel = '') {
  for (const ent of fs.readdirSync(src, { withFileTypes: true })) {
    const r = rel ? `${rel}/${ent.name}` : ent.name;
    if (ent.isDirectory()) {
      if (skipDirs.has(ent.name) && !rel) continue;
      if (r === 'assets/earth') {
        // Fetched imagery, not ours to ship; keep the attribution.
        const credits = path.join(src, ent.name, 'CREDITS');
        if (fs.existsSync(credits)) {
          fs.mkdirSync(path.join(dst, r), { recursive: true });
          fs.copyFileSync(credits, path.join(dst, r, 'CREDITS'));
        }
        continue;
      }
      copyTree(path.join(src, ent.name), dst, r);
    } else if (ent.isFile() && !skipFile(r)) {
      fs.mkdirSync(path.join(dst, rel), { recursive: true });
      fs.copyFileSync(path.join(src, ent.name), path.join(dst, r));
    }
  }
}
fs.rmSync(out, { recursive: true, force: true });
copyTree(web, siteWeb);

const walk = (dir) =>
  fs.readdirSync(dir, { withFileTypes: true }).flatMap((e) =>
    e.isDirectory() ? walk(path.join(dir, e.name)) : [path.join(dir, e.name)]);
const textExt = /\.(js|mjs|html|css|json|svg|txt|md)$/;

// --- hashed wasm -------------------------------------------------------------

const renames = []; // [old basename, new basename, site path]
for (const f of walk(siteWeb).filter((f) => f.endsWith('_bg.wasm'))) {
  const hash = createHash('sha256').update(fs.readFileSync(f)).digest('hex').slice(0, 10);
  const name = path.basename(f);
  const hashed = name.replace(/\.wasm$/, `.${hash}.wasm`);
  fs.renameSync(f, path.join(path.dirname(f), hashed));
  renames.push([name, hashed, path.join(path.dirname(f), hashed)]);
}
// Every textual reference, quoted or in a path: `'threers_bg.wasm'`,
// `/web/pkg/threers_bg.wasm`, `new URL('../pkg/threers_bg.wasm', …)`.
for (const f of walk(siteWeb).filter((f) => textExt.test(f))) {
  let s = fs.readFileSync(f, 'utf8');
  let t = s;
  for (const [name, hashed] of renames) {
    t = t.replace(new RegExp(`(^|[^A-Za-z0-9_.-])${name.replace(/\./g, '\\.')}(?![A-Za-z0-9_.])`, 'g'), `$1${hashed}`);
  }
  if (t !== s) fs.writeFileSync(f, t);
}

// --- preload hints -------------------------------------------------------------

const main = renames.find(([name, , p]) => name === 'threers_bg.wasm' && p.includes(`${path.sep}pkg${path.sep}`));
const usesShim = (html, dir) => {
  if (/threejs-shim\.js|pkg\/threers\.js/.test(html)) return true;
  // A page whose own module script imports the shim.
  for (const m of html.matchAll(/<script[^>]+src=["']([^"']+\.js)["']/g)) {
    const src = m[1].startsWith('/') ? path.join(out, m[1]) : path.join(dir, m[1]);
    if (fs.existsSync(src) && /threejs-shim\.js|pkg\/threers\.js/.test(fs.readFileSync(src, 'utf8'))) return true;
  }
  return false;
};
let preloaded = 0;
if (main) {
  const wasmUrl = '/' + path.relative(out, main[2]).split(path.sep).join('/');
  const hints = [
    '<link rel="modulepreload" href="/web/threejs-shim.js">',
    '<link rel="modulepreload" href="/web/pkg/threers.js">',
    `<link rel="preload" href="${wasmUrl}" as="fetch" type="application/wasm" crossorigin>`,
  ].join('\n    ');
  for (const f of walk(siteWeb).filter((f) => f.endsWith('.html'))) {
    const html = fs.readFileSync(f, 'utf8');
    if (html.includes('rel="preload"') || !usesShim(html, path.dirname(f))) continue;
    // After `<meta charset>` when there is one (it has to come first), else
    // straight after `<head>`.
    const charset = html.search(/<meta[^>]+charset[^>]*>/i);
    const at = charset >= 0 ? charset : html.search(/<head[^>]*>/i);
    if (at < 0) continue;
    const end = html.indexOf('>', at) + 1;
    fs.writeFileSync(f, `${html.slice(0, end)}\n    ${hints}${html.slice(end)}`);
    preloaded++;
  }
}

// --- minify --------------------------------------------------------------------

let minified = 0;
if (minify) {
  const files = walk(siteWeb).filter((f) => /\.m?js$/.test(f));
  try {
    // One esbuild run over every file, in place, each transformed on its own
    // (no bundling): imports, exports and `import.meta.url` are untouched.
    execFileSync('npx', ['--yes', 'esbuild@0.24.2', ...files, '--minify', '--sourcemap',
      '--legal-comments=eof', `--outdir=${siteWeb}`, `--outbase=${siteWeb}`, '--allow-overwrite',
      '--log-level=warning'], { stdio: 'inherit' });
    minified = files.length;
  } catch {
    console.warn('    (esbuild unavailable — JS left unminified; MINIFY=0 silences this)');
  }
}

// --- headers -------------------------------------------------------------------

const immutable = 'public, max-age=31536000, immutable';
const revalidate = 'public, max-age=0, must-revalidate';
fs.writeFileSync(path.join(out, 'vercel.json'), JSON.stringify({
  $schema: 'https://openapi.vercel.sh/vercel.json',
  // Every wasm here is content-hashed, so any of them can be kept forever.
  headers: [
    { source: '/(.*)\\.wasm', headers: [
      { key: 'Content-Type', value: 'application/wasm' },
      { key: 'Cache-Control', value: immutable },
    ] },
    { source: '/(.*)\\.(js|mjs|css|html|json|map)', headers: [
      { key: 'Cache-Control', value: revalidate },
    ] },
    { source: '/(.*)', headers: [{ key: 'X-Content-Type-Options', value: 'nosniff' }] },
  ],
  redirects: [{ source: '/', destination: '/web/examples/', permanent: false }],
}, null, 2) + '\n');
fs.writeFileSync(path.join(out, '_headers'), `# Netlify and Cloudflare Pages. Every wasm here is content-hashed.
/*.wasm
  Content-Type: application/wasm
  Cache-Control: ${immutable}
/*.js
  Cache-Control: ${revalidate}
/*.html
  Cache-Control: ${revalidate}
/*
  X-Content-Type-Options: nosniff
`);
fs.writeFileSync(path.join(out, '_redirects'), '/  /web/examples/  302\n');
fs.writeFileSync(path.join(out, '.nojekyll'), '');
fs.writeFileSync(path.join(out, 'index.html'),
  '<!doctype html><meta charset="utf-8"><title>threers</title>' +
  '<meta http-equiv="refresh" content="0; url=/web/examples/">' +
  '<a href="/web/examples/">threers examples</a>\n');

// --- precompress -----------------------------------------------------------------

if (precompress) {
  for (const f of walk(siteWeb).filter((f) => /\.(wasm|m?js|css|html|json|svg)$/.test(f))) {
    const buf = fs.readFileSync(f);
    if (buf.length < 1024) continue;
    fs.writeFileSync(`${f}.br`, zlib.brotliCompressSync(buf, {
      params: { [zlib.constants.BROTLI_PARAM_QUALITY]: 11, [zlib.constants.BROTLI_PARAM_SIZE_HINT]: buf.length },
    }));
    fs.writeFileSync(`${f}.gz`, zlib.gzipSync(buf, { level: 9 }));
  }
}

// --- report ----------------------------------------------------------------------

const all = walk(out).map((f) => ({ f: path.relative(out, f), n: fs.statSync(f).size }));
const total = all.reduce((a, b) => a + b.n, 0);
const mb = (n) => (n / 1048576).toFixed(2);
const brOf = (f) => zlib.brotliCompressSync(fs.readFileSync(path.join(out, f)), {
  params: { [zlib.constants.BROTLI_PARAM_QUALITY]: 9 },
}).length;
console.log(`==> site staged in ${path.relative(repo, out) || out}`);
console.log(`    ${renames.length} wasm hashed, ${minified} JS minified, ${preloaded} pages preloaded, ${all.length} files, ${mb(total)} MB`);
for (const { f, n } of all.filter((x) => !/\.(br|gz|map)$/.test(x.f)).sort((a, b) => b.n - a.n).slice(0, 6)) {
  console.log(`    ${mb(n).padStart(7)} MB  ${f}  (≈${mb(brOf(f))} MB brotli)`);
}
const over = all.filter(({ n }) => n > maxFileMb * 1048576);
if (over.length) {
  for (const { f, n } of over) console.error(`    ${f}: ${mb(n)} MB, over the ${maxFileMb} MB per-file limit`);
  process.exit(1);
}
