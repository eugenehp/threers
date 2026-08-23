#!/usr/bin/env node
/**
 * Headless Chromium runner for video-export wasm + parallel + worker tests.
 *
 * Usage:
 *   node web/scripts/test-video-export.mjs
 *
 * Env:
 *   PORT=8765
 *   CHROMIUM=/path/to/chromium
 *   TIMEOUT_MS=120000
 */
import http from 'node:http';
import { spawn } from 'node:child_process';
import { createReadStream, existsSync, statSync } from 'node:fs';
import { extname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { once } from 'node:events';

const __dirname = fileURLToPath(new URL('.', import.meta.url));
const WEB_ROOT = resolve(__dirname, '..');
const REPO_ROOT = resolve(WEB_ROOT, '..');
const PORT = Number(process.env.PORT || 8765);
const TIMEOUT_MS = Number(process.env.TIMEOUT_MS || 120_000);
const CHROMIUM = process.env.CHROMIUM
  || (existsSync('/opt/homebrew/bin/chromium') && '/opt/homebrew/bin/chromium')
  || (existsSync('/usr/bin/chromium') && '/usr/bin/chromium')
  || 'chromium';

const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.mjs': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.wasm': 'application/wasm',
  '.json': 'application/json',
  '.map': 'application/json',
  '.png': 'image/png',
  '.gif': 'image/gif',
};

function contentType(path) {
  return MIME[extname(path)] || 'application/octet-stream';
}

function startServer() {
  const server = http.createServer((req, res) => {
    try {
      let urlPath = decodeURIComponent((req.url || '/').split('?')[0]);
      // Tests import `/web/...` from the page; also allow root-relative pkg paths.
      if (urlPath.startsWith('/web/')) urlPath = urlPath.slice(4);
      if (urlPath === '/') urlPath = '/examples/video-export-test.html';
      const filePath = resolve(WEB_ROOT, '.' + urlPath);
      if (!filePath.startsWith(WEB_ROOT) || !existsSync(filePath) || !statSync(filePath).isFile()) {
        res.writeHead(404);
        res.end(`not found: ${urlPath}`);
        return;
      }
      res.writeHead(200, {
        'Content-Type': contentType(filePath),
        'Cross-Origin-Resource-Policy': 'same-origin',
        'Cache-Control': 'no-store',
      });
      createReadStream(filePath).pipe(res);
    } catch (err) {
      res.writeHead(500);
      res.end(String(err));
    }
  });
  return new Promise((resolvePromise) => {
    server.listen(PORT, '127.0.0.1', () => resolvePromise(server));
  });
}

async function runChromium(url) {
  const userData = join(REPO_ROOT, 'out', 'chromium-video-test-profile');
  const args = [
    '--headless=new',
    '--disable-gpu',
    '--enable-unsafe-webgpu',
    '--enable-features=Vulkan,UseSkiaRenderer',
    '--use-angle=swiftshader',
    '--ignore-gpu-blocklist',
    `--user-data-dir=${userData}`,
    '--no-first-run',
    '--no-default-browser-check',
    '--disable-background-networking',
    '--remote-debugging-port=0',
    `--virtual-time-budget=${TIMEOUT_MS}`,
    url,
  ];

  // Prefer CDP evaluate via puppeteer-core if present; else dump title via --dump-dom fallback.
  try {
    const puppeteer = await import('puppeteer-core');
    const browser = await puppeteer.default.launch({
      executablePath: CHROMIUM,
      headless: 'new',
      args: [
        '--enable-unsafe-webgpu',
        '--ignore-gpu-blocklist',
        '--use-angle=swiftshader',
        '--no-first-run',
      ],
    });
    try {
      const page = await browser.newPage();
      page.setDefaultTimeout(TIMEOUT_MS);
      await page.goto(url, { waitUntil: 'domcontentloaded', timeout: TIMEOUT_MS });
      const result = await page.waitForFunction(
        () => {
          const r = window.__THREERS_VIDEO_TEST__;
          if (!r?.finished) return null;
          return r;
        },
        { timeout: TIMEOUT_MS },
      ).then((h) => h.jsonValue());
      return result;
    } finally {
      await browser.close();
    }
  } catch (err) {
    const missing = err?.code === 'ERR_MODULE_NOT_FOUND'
      || /Cannot find package ['"]puppeteer-core['"]/.test(String(err));
    if (!missing) throw err;
  }

  // Fallback: print dump-dom and parse title / log text.
  const child = spawn(CHROMIUM, args, { stdio: ['ignore', 'pipe', 'pipe'] });
  let stdout = '';
  let stderr = '';
  child.stdout.on('data', (d) => { stdout += d; });
  child.stderr.on('data', (d) => { stderr += d; });
  const timed = setTimeout(() => child.kill('SIGKILL'), TIMEOUT_MS + 5_000);
  const [code] = await once(child, 'exit');
  clearTimeout(timed);
  const titleMatch = stdout.match(/<title>([^<]*)<\/title>/i);
  const title = titleMatch ? titleMatch[1] : '';
  const pass = title === 'PASS' || /ALL PASSED/.test(stdout);
  return {
    ok: pass && code === 0,
    title,
    fallback: true,
    exitCode: code,
    stderr: stderr.slice(-2000),
  };
}

async function main() {
  const wasm = join(WEB_ROOT, 'pkg', 'threers_bg.wasm');
  if (!existsSync(wasm)) {
    console.error('missing web/pkg/threers_bg.wasm — run: NATIVE_CODEC=1 web/build.sh');
    process.exit(2);
  }
  // Quick check native-codec symbols exist in glue.
  const glue = await import('node:fs/promises').then((fs) => fs.readFile(join(WEB_ROOT, 'pkg', 'threers.js'), 'utf8'));
  if (!glue.includes('encodeGifRgba')) {
    console.error('web/pkg/threers.js missing encodeGifRgba — rebuild with NATIVE_CODEC=1');
    process.exit(2);
  }
  if (!glue.includes('encodeMp4Rgba')) {
    console.error('web/pkg/threers.js missing encodeMp4Rgba — rebuild with NATIVE_CODEC=1');
    process.exit(2);
  }

  const server = await startServer();
  const url = `http://127.0.0.1:${PORT}/examples/video-export-test.html`;
  console.log(`serving ${WEB_ROOT}`);
  console.log(`chromium: ${CHROMIUM}`);
  console.log(`open ${url}`);

  let result;
  try {
    result = await runChromium(url);
  } finally {
    server.close();
  }

  console.log(JSON.stringify(result, null, 2));
  if (!result?.ok) {
    console.error('video-export tests FAILED');
    process.exit(1);
  }
  console.log('video-export tests PASSED');
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
