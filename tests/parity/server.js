// Tiny static file server for the parity test. Serves the repo root so the
// scene HTMLs can `import` from /web/threejs-shim.js etc.

import http from 'http';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(__dirname, '..', '..');

const port = process.argv[2] ? parseInt(process.argv[2], 10) : 8087;
// `PKG_DIR=/some/dir`: serve /web/pkg/* from there instead, so a run can test
// a wasm build without overwriting the shared web/pkg.
const pkgDir = process.env.PKG_DIR ? path.resolve(process.env.PKG_DIR) : null;

const types = {
    '.html': 'text/html; charset=utf-8',
    '.js':   'application/javascript; charset=utf-8',
    '.mjs':  'application/javascript; charset=utf-8',
    '.wasm': 'application/wasm',
    '.json': 'application/json; charset=utf-8',
    '.css':  'text/css; charset=utf-8',
    '.rs':   'text/plain; charset=utf-8',
    '.png':  'image/png',
    '.svg':  'image/svg+xml',
};

function corsHeaders(extra = {}) {
    return {
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': 'GET, HEAD, OPTIONS',
        ...extra,
    };
}

http.createServer((req, res) => {
    if (req.method === 'OPTIONS') {
        res.writeHead(204, corsHeaders());
        res.end();
        return;
    }
    let url = req.url.split('?')[0];
    if (url === '/tests/parity' || url === '/tests/parity/') url = '/tests/parity/index.html';
    if (url === '/') {
        res.writeHead(302, corsHeaders({ Location: '/tests/parity/index.html' }));
        res.end();
        return;
    }
    const fromPkg = pkgDir && url.startsWith('/web/pkg/');
    const root = fromPkg ? pkgDir : repoRoot;
    const full = path.normalize(fromPkg ? path.join(pkgDir, url.slice('/web/pkg/'.length)) : path.join(repoRoot, url));
    if (!full.startsWith(root)) {
        res.writeHead(403, corsHeaders({ 'Content-Type': 'text/plain' }));
        res.end('forbidden');
        return;
    }
    fs.readFile(full, (err, data) => {
        if (err) {
            if (url === '/tests/parity/out/compare-results.json') {
                res.writeHead(200, corsHeaders({ 'Content-Type': 'application/json; charset=utf-8' }));
                res.end('[]');
                return;
            }
            res.writeHead(404, corsHeaders({ 'Content-Type': 'text/plain' }));
            res.end(`not found: ${url}`);
            return;
        }
        const ext = path.extname(full);
        const headers = corsHeaders({
            'Content-Type': types[ext] || 'application/octet-stream',
        });
        if (ext === '.js' || ext === '.mjs' || ext === '.html' || ext === '.wasm' || ext === '.txt') {
            headers['Cache-Control'] = 'no-cache';
        }
        res.writeHead(200, headers);
        res.end(data);
    });
}).listen(port, () => {
    console.log(`server: http://localhost:${port}`);
    console.log('Open that URL in Safari — do not open index.html via file://');
});
