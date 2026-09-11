// Every neuron in both connectome trees as one node, drawn by threers.
//
// Two facts about the renderer shape this file. threers rasterises Points as
// one camera-facing sprite draw per point, which at 350k points is 350k draw
// calls a frame; LineSegments is a single cached draw for the whole buffer. So
// a node here is a three-axis cross of line segments -- six vertices that read
// as a dot at any camera angle -- and the whole cloud is a handful of big
// LineSegments objects. And vertex colours reach the shader linearly (the
// framebuffer does the sRGB encode), so the palette is decoded on the way in.

import THREE, { initThreers } from '/threers/threejs-shim.js';

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const NM_PER_UM = 1000;
const TREE_GAP_UM = 130;        // clear air between the two animals
const CHUNK_NODES = 40000;      // nodes per LineSegments object
const MAX_SERIES = 8;           // palette slots; everything past this is "other"
const PICK_RADIUS_PX = 7;
const SIDEBAR_PX = 314;         // panel width + margins, kept clear when framing

// The two releases store their axes differently, and neither stores them the
// way a viewer wants them. Both are EM volumes with y increasing downward, so
// y flips. Male CNS additionally runs brain -> VNC along +z (verified against
// the index: optic-lobe neurons sit near z 265k, VNC neurons near z 780k), so
// its z becomes the vertical and the animal hangs head-up like the figures do.
const AXES = {
    'flywire-783': (x, y, z) => [x, -y, z],
    'malecns-v1.0': (x, y, z) => [x, -z, -y],
};

const $ = (id) => document.getElementById(id);
const fmt = (n) => n.toLocaleString('en-US');

// ---------------------------------------------------------------------------
// Colour
// ---------------------------------------------------------------------------

const css = getComputedStyle(document.documentElement);
const hexOf = (name) => css.getPropertyValue(name).trim();

/** sRGB byte triple -> the linear values the shader multiplies with. */
function srgbToLinear(hex) {
    const h = hex.replace('#', '');
    const out = [];
    for (let i = 0; i < 3; i++) {
        const c = parseInt(h.slice(i * 2, i * 2 + 2), 16) / 255;
        out.push(c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4));
    }
    return out;
}

const SERIES_HEX = Array.from({ length: MAX_SERIES }, (_, i) => hexOf(`--series-${i + 1}`));
const OTHER_HEX = hexOf('--series-other');
const SERIES_LIN = SERIES_HEX.map(srgbToLinear);
const OTHER_LIN = srgbToLinear(OTHER_HEX);

// Sequential blue ramp, darkest step first. On a dark surface magnitude has to
// read as *brightness*, so the ramp runs 600 -> 100 rather than light -> dark;
// step 600 is the darkest the ordinal rule allows against this surface.
const RAMP_HEX = ['#184f95', '#1c5cab', '#256abf', '#2a78d6', '#3987e5',
                  '#5598e7', '#6da7ec', '#86b6ef', '#9ec5f4', '#b7d3f6', '#cde2fb'];
const RAMP_LIN = RAMP_HEX.map(srgbToLinear);

const catColor = (i) => (i < MAX_SERIES ? SERIES_LIN[i] : OTHER_LIN);
const catHex = (i) => (i < MAX_SERIES ? SERIES_HEX[i] : OTHER_HEX);

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

const state = {
    meta: null,
    trees: [],
    mode: 'role',
    vocab: {},          // mode -> { names, counts, hidden:Set, solo:number|null }
    minSyn: 0,
    query: '',
    nodeSizeUm: 0.9,
    shown: 0,
    skeletons: new Map(),   // "tree/id" -> { obj, name }
    selected: null,
};

let scene, camera, controls, renderer, canvas;
let cloudMat, cloudObjs = [], highlight = null;
let homeView = null;

// ---------------------------------------------------------------------------
// Loading
// ---------------------------------------------------------------------------

const boot = { steps: 0, done: 0 };
function progress(msg, add = 1) {
    boot.done += add;
    $('bootMsg').textContent = msg;
    $('bootBar').style.width = `${Math.min(100, (boot.done / boot.steps) * 100)}%`;
}

async function bin(url, Type) {
    const r = await fetch(url);
    if (!r.ok) throw new Error(`${url} -> HTTP ${r.status}`);
    return new Type(await r.arrayBuffer());
}

async function loadTree(t) {
    const base = `/data/${t.key}`;
    const [pos, role, nt, cls, syn, ids, labelsRaw] = await Promise.all([
        bin(`${base}/pos.f32`, Float32Array),
        bin(`${base}/role.u8`, Uint8Array),
        bin(`${base}/nt.u8`, Uint8Array),
        bin(`${base}/cls.u16`, Uint16Array),
        bin(`${base}/syn.u32`, Uint32Array),
        bin(`${base}/ids.u64`, BigUint64Array),
        fetch(`${base}/labels.txt`).then((r) => r.text()),
    ]);
    progress(`${t.title}: ${fmt(t.count)} nodes`);
    return {
        ...t, raw: pos, role, nt, cls, syn, ids,
        labels: labelsRaw.split('\n'),
        on: true,
        n: t.count,
        idStrings: null,
        disp: new Float32Array(pos.length),
        vis: new Uint8Array(t.count),
        visIdx: new Int32Array(t.count),
        visCount: 0,
    };
}

/** Remap axes, centre each tree on its own bbox, scale nm -> µm, then lay the
 *  trees out along x so both animals are on screen at once and neither one's
 *  coordinate frame leaks into the other's. */
function placeTrees(trees) {
    for (const t of trees) {
        const remap = AXES[t.key] || ((x, y, z) => [x, y, z]);
        const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
        for (let i = 0; i < t.n; i++) {
            const p = remap(t.raw[i * 3], t.raw[i * 3 + 1], t.raw[i * 3 + 2]);
            for (let d = 0; d < 3; d++) {
                t.disp[i * 3 + d] = p[d];
                if (p[d] < lo[d]) lo[d] = p[d];
                if (p[d] > hi[d]) hi[d] = p[d];
            }
        }
        t.centreNm = lo.map((v, d) => (v + hi[d]) / 2);
        t.sizeUm = hi.map((v, d) => (v - lo[d]) / NM_PER_UM);
        t.remap = remap;
    }
    const totalW = trees.reduce((a, t) => a + t.sizeUm[0], 0) + TREE_GAP_UM * (trees.length - 1);
    let cursor = -totalW / 2;
    for (const t of trees) {
        t.offset = [cursor + t.sizeUm[0] / 2, 0, 0];
        cursor += t.sizeUm[0] + TREE_GAP_UM;
        for (let i = 0; i < t.n; i++) {
            for (let d = 0; d < 3; d++) {
                t.disp[i * 3 + d] = (t.disp[i * 3 + d] - t.centreNm[d]) / NM_PER_UM + t.offset[d];
            }
        }
        // Same map, for one point at a time (skeleton vertices arrive later).
        t.toWorld = (x, y, z) => {
            const p = t.remap(x, y, z);
            return [
                (p[0] - t.centreNm[0]) / NM_PER_UM + t.offset[0],
                (p[1] - t.centreNm[1]) / NM_PER_UM + t.offset[1],
                (p[2] - t.centreNm[2]) / NM_PER_UM + t.offset[2],
            ];
        };
    }
}

// ---------------------------------------------------------------------------
// Vocabularies
// ---------------------------------------------------------------------------

const MODE_FIELD = {
    role: { list: 'roles', counts: 'role_counts' },
    nt: { list: 'nts', counts: 'nt_counts' },
    class: { list: 'classes', counts: 'class_counts' },
};

/** One legend has to span both trees, but the two indexes order their own
 *  vocabularies independently. Merge by name, order by combined count, and
 *  keep a per-tree lookup from the tree's own index into the merged one. */
function buildVocabs(trees) {
    for (const mode of ['role', 'nt', 'class']) {
        const { list, counts } = MODE_FIELD[mode];
        const total = new Map();
        for (const t of trees) {
            t[list].forEach((name, i) => {
                total.set(name, (total.get(name) || 0) + t[counts][i]);
            });
        }
        const names = [...total.keys()].sort((a, b) => total.get(b) - total.get(a) || a.localeCompare(b));
        const ix = new Map(names.map((n, i) => [n, i]));
        state.vocab[mode] = { names, counts: names.map((n) => total.get(n)), hidden: new Set(), solo: null };
        for (const t of trees) {
            const lut = t[list].map((n) => ix.get(n));
            const src = mode === 'role' ? t.role : mode === 'nt' ? t.nt : t.cls;
            const out = new Uint16Array(t.n);
            for (let i = 0; i < t.n; i++) out[i] = lut[src[i]];
            t[`cat_${mode}`] = out;
        }
    }
    // "tree" is its own two-item vocabulary.
    state.vocab.tree = {
        names: trees.map((t) => t.title),
        counts: trees.map((t) => t.n),
        hidden: new Set(), solo: null,
    };
    trees.forEach((t, i) => { t.cat_tree = i; });
}

const catOf = (t, i) => (state.mode === 'tree' ? t.cat_tree : t[`cat_${state.mode}`][i]);

/** Colour for a node, already linear. */
function colorOf(t, i, out) {
    if (state.mode === 'syn') {
        // log scale: synapse counts run over four orders of magnitude.
        const s = t.syn[i];
        const u = s <= 0 ? 0 : Math.min(1, Math.log10(1 + s) / Math.log10(1 + state.synMax));
        const c = RAMP_LIN[Math.min(RAMP_LIN.length - 1, Math.round(u * (RAMP_LIN.length - 1)))];
        out[0] = c[0]; out[1] = c[1]; out[2] = c[2];
        return;
    }
    const c = catColor(catOf(t, i));
    out[0] = c[0]; out[1] = c[1]; out[2] = c[2];
}

// ---------------------------------------------------------------------------
// Filtering
// ---------------------------------------------------------------------------

function idStrings(t) {
    if (!t.idStrings) {
        t.idStrings = new Array(t.n);
        for (let i = 0; i < t.n; i++) t.idStrings[i] = t.ids[i].toString();
    }
    return t.idStrings;
}

function applyFilters() {
    const v = state.vocab[state.mode];
    const q = state.query.trim().toLowerCase();
    state.shown = 0;
    for (const t of state.trees) {
        let k = 0;
        if (t.on) {
            const ids = q ? idStrings(t) : null;
            for (let i = 0; i < t.n; i++) {
                if (t.syn[i] < state.minSyn) continue;
                if (state.mode !== 'syn') {
                    const c = catOf(t, i);
                    if (v.solo !== null ? c !== v.solo : v.hidden.has(c)) continue;
                }
                if (q && !t.labels[i].toLowerCase().includes(q) && !ids[i].includes(q)) continue;
                t.visIdx[k++] = i;
            }
        }
        t.visCount = k;
        state.shown += k;
    }
    $('shown').textContent = fmt(state.shown);
}

// ---------------------------------------------------------------------------
// Cloud geometry
// ---------------------------------------------------------------------------

/** Write one node as three axis-aligned segments centred on the point: six
 *  vertices that stay a dot however the camera turns. */
function writeCross(pos, col, o, x, y, z, s, c) {
    let j = o * 3;
    pos[j] = x - s; pos[j + 1] = y;     pos[j + 2] = z;     j += 3;
    pos[j] = x + s; pos[j + 1] = y;     pos[j + 2] = z;     j += 3;
    pos[j] = x;     pos[j + 1] = y - s; pos[j + 2] = z;     j += 3;
    pos[j] = x;     pos[j + 1] = y + s; pos[j + 2] = z;     j += 3;
    pos[j] = x;     pos[j + 1] = y;     pos[j + 2] = z - s; j += 3;
    pos[j] = x;     pos[j + 1] = y;     pos[j + 2] = z + s;
    const r = c[0], g = c[1], b = c[2];
    for (let k = o * 3, end = (o + 6) * 3; k < end; k += 3) {
        col[k] = r; col[k + 1] = g; col[k + 2] = b;
    }
    return o + 6;
}

let rebuildToken = 0;

async function rebuildCloud() {
    const token = ++rebuildToken;
    applyFilters();

    for (const o of cloudObjs) scene.remove(o);
    cloudObjs = [];

    const s = state.nodeSizeUm;
    const c = [0, 0, 0];
    for (const t of state.trees) {
        for (let start = 0; start < t.visCount; start += CHUNK_NODES) {
            if (token !== rebuildToken) return;             // superseded mid-build
            const n = Math.min(CHUNK_NODES, t.visCount - start);
            const pos = new Float32Array(n * 18);
            const col = new Float32Array(n * 18);
            let o = 0;
            for (let k = 0; k < n; k++) {
                const i = t.visIdx[start + k];
                colorOf(t, i, c);
                o = writeCross(pos, col, o, t.disp[i * 3], t.disp[i * 3 + 1], t.disp[i * 3 + 2], s, c);
            }
            const g = new THREE.BufferGeometry();
            g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
            g.setAttribute('color', new THREE.BufferAttribute(col, 3));
            const obj = new THREE.LineSegments(g, cloudMat);
            scene.add(obj);
            cloudObjs.push(obj);
            // Yield so a 350k-node rebuild never blocks a frame.
            await new Promise((r) => requestAnimationFrame(r));
        }
    }
}

let rebuildTimer = null;
function scheduleRebuild(delay = 0) {
    clearTimeout(rebuildTimer);
    rebuildTimer = setTimeout(() => rebuildCloud(), delay);
}

// ---------------------------------------------------------------------------
// Picking — a ray built from the camera the controls just moved
// ---------------------------------------------------------------------------

function cameraBasis() {
    const p = camera.position, tgt = controls.target;
    let fx = tgt.x - p.x, fy = tgt.y - p.y, fz = tgt.z - p.z;
    const fl = Math.hypot(fx, fy, fz) || 1;
    fx /= fl; fy /= fl; fz /= fl;
    // right = normalize(forward × worldUp) = normalize(-fz, 0, fx); up follows
    // as right × forward, so the basis stays orthonormal at any pitch.
    let rx = -fz, ry = 0, rz = fx;
    let rl = Math.hypot(rx, ry, rz);
    if (rl < 1e-6) { rx = 1; ry = 0; rz = 0; rl = 1; }   // looking straight down
    rx /= rl; ry /= rl; rz /= rl;
    const ux = ry * fz - rz * fy, uy = rz * fx - rx * fz, uz = rx * fy - ry * fx;
    return { p, f: [fx, fy, fz], r: [rx, ry, rz], u: [ux, uy, uz] };
}

function pick(clientX, clientY) {
    const rect = canvas.getBoundingClientRect();
    const ndcX = ((clientX - rect.left) / rect.width) * 2 - 1;
    const ndcY = 1 - ((clientY - rect.top) / rect.height) * 2;
    const { p, f, r, u } = cameraBasis();
    const th = Math.tan((camera.fov * Math.PI) / 360);
    const aspect = rect.width / rect.height;
    const dx = f[0] + r[0] * ndcX * th * aspect + u[0] * ndcY * th;
    const dy = f[1] + r[1] * ndcX * th * aspect + u[1] * ndcY * th;
    const dz = f[2] + r[2] * ndcX * th * aspect + u[2] * ndcY * th;
    const dl = Math.hypot(dx, dy, dz);
    const ox = p.x, oy = p.y, oz = p.z;
    const nx = dx / dl, ny = dy / dl, nz = dz / dl;

    // A pixel is worth `pxScale * t` world units at ray distance t.
    const pxScale = (2 * th) / rect.height;
    const tol = PICK_RADIUS_PX * pxScale;

    let best = null, bestScore = Infinity;
    for (const t of state.trees) {
        const d = t.disp;
        for (let k = 0; k < t.visCount; k++) {
            const i = t.visIdx[k];
            const vx = d[i * 3] - ox, vy = d[i * 3 + 1] - oy, vz = d[i * 3 + 2] - oz;
            const along = vx * nx + vy * ny + vz * nz;
            if (along <= 0) continue;
            const perp2 = vx * vx + vy * vy + vz * vz - along * along;
            const lim = tol * along;
            if (perp2 > lim * lim) continue;
            // Nearest to the ray in screen space, depth as the tie-break.
            const score = Math.sqrt(perp2) / along / pxScale + along * 1e-4;
            if (score < bestScore) { bestScore = score; best = { tree: t, i }; }
        }
    }
    return best;
}

// ---------------------------------------------------------------------------
// Skeletons
// ---------------------------------------------------------------------------

async function loadSkeleton(tree, id, lineColor) {
    const key = `${tree.key}/${id}`;
    if (state.skeletons.has(key)) return state.skeletons.get(key);
    const res = await fetch(`/api/skeleton/${tree.key}/${id}`);
    if (!res.ok) throw new Error(`skeleton ${id}: HTTP ${res.status}`);
    const nodes = Number(res.headers.get('x-nodes') || 0);
    const raw = new Float32Array(await res.arrayBuffer());
    const n = raw.length / 3;
    const pos = new Float32Array(raw.length);
    const col = new Float32Array(raw.length);
    for (let i = 0; i < n; i++) {
        const w = tree.toWorld(raw[i * 3], raw[i * 3 + 1], raw[i * 3 + 2]);
        pos[i * 3] = w[0]; pos[i * 3 + 1] = w[1]; pos[i * 3 + 2] = w[2];
        col[i * 3] = lineColor[0]; col[i * 3 + 1] = lineColor[1]; col[i * 3 + 2] = lineColor[2];
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    g.setAttribute('color', new THREE.BufferAttribute(col, 3));
    const obj = new THREE.LineSegments(g, cloudMat);
    scene.add(obj);
    const entry = {
        obj, tree, id, nodes, segments: n / 2,
        bounds: n ? boundsOf((i, d) => pos[i * 3 + d], n) : null,
    };
    state.skeletons.set(key, entry);
    renderSkelList();
    return entry;
}

function clearSkeletons() {
    for (const { obj } of state.skeletons.values()) scene.remove(obj);
    state.skeletons.clear();
    renderSkelList();
}

function renderSkelList() {
    const el = $('skelList');
    el.innerHTML = '';
    if (!state.skeletons.size) {
        el.innerHTML = '<p class="hint" style="margin:0">none loaded</p>';
        return;
    }
    for (const [key, e] of state.skeletons) {
        const row = document.createElement('div');
        row.className = 'pt';
        row.innerHTML = `<span>${e.id}</span><span class="n">${fmt(e.nodes)} nodes</span>`;
        row.title = `${e.tree.title} · ${fmt(e.segments)} segments — click to remove`;
        row.onclick = () => { scene.remove(e.obj); state.skeletons.delete(key); renderSkelList(); };
        el.appendChild(row);
    }
}

// ---------------------------------------------------------------------------
// Selection & info panel
// ---------------------------------------------------------------------------

function highlightAt(x, y, z, s) {
    if (highlight) scene.remove(highlight);
    const v = [[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],[-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]];
    const e = [[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]];
    const pos = new Float32Array(e.length * 6);
    const col = new Float32Array(e.length * 6).fill(1);
    let o = 0;
    for (const [a, b] of e) {
        for (const k of [a, b]) {
            pos[o++] = x + v[k][0] * s; pos[o++] = y + v[k][1] * s; pos[o++] = z + v[k][2] * s;
        }
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    g.setAttribute('color', new THREE.BufferAttribute(col, 3));
    highlight = new THREE.LineSegments(g, cloudMat);
    scene.add(highlight);
}

async function select(tree, i) {
    const id = tree.ids[i].toString();
    state.selected = { tree, i, id };
    const [type, cls, side] = tree.labels[i].split('\t');
    $('info').classList.add('on');
    $('iName').textContent = type || '(untyped)';
    $('iId').textContent = `${tree.title} · ${id}`;
    $('iBody').innerHTML = '<p class="hint">loading…</p>';

    highlightAt(tree.disp[i * 3], tree.disp[i * 3 + 1], tree.disp[i * 3 + 2],
                Math.max(3, state.nodeSizeUm * 4));

    const c = [0, 0, 0];
    colorOf(tree, i, c);
    loadSkeleton(tree, id, c).catch((e) => console.warn(e));

    let data;
    try {
        const r = await fetch(`/api/neuron/${tree.key}/${id}`);
        data = await r.json();
    } catch (e) {
        $('iBody').innerHTML = `<p class="hint">${e}</p>`;
        return;
    }
    if (state.selected?.id !== id) return;

    const m = data.meta || {};
    const show = ['cell_type', 'super_class', 'cell_class', 'side', 'role', 'sensor_modality',
                  'nerve', 'top_nt', 'skeleton_nodes', 'cable_length_nm', 'in_synapses',
                  'out_synapses', 'in_partners', 'out_partners', 'status'];
    const dl = show.filter((k) => m[k]).map((k) =>
        `<dt>${k.replace(/_/g, ' ')}</dt><dd>${escapeHtml(m[k])}</dd>`).join('');

    const partnerRows = (dir) => ((data.partners || {})[dir] || []).map((p) =>
        `<div class="pt" data-partner="${p.id}"><span>${p.id}</span><span class="n">${fmt(p.syn)}</span></div>`
    ).join('') || '<p class="hint" style="margin:0">none</p>';

    $('iBody').innerHTML =
        `<dl>${dl || `<dt>type</dt><dd>${escapeHtml(type || '—')}</dd><dt>class</dt><dd>${escapeHtml(cls || '—')}</dd><dt>side</dt><dd>${escapeHtml(side || '—')}</dd>`}</dl>` +
        `<h3>Top inputs</h3>${partnerRows('in')}` +
        `<h3>Top outputs</h3>${partnerRows('out')}`;

    for (const el of $('iBody').querySelectorAll('[data-partner]')) {
        el.onclick = () => jumpTo(tree, el.dataset.partner);
    }
}

function escapeHtml(s) {
    return String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
}

/** Follow a partner id into the same tree. */
function jumpTo(tree, id) {
    const strs = idStrings(tree);
    const i = strs.indexOf(id);
    if (i >= 0) return select(tree, i);
    // Positionless neuron (789 Male CNS bodies have neither soma nor skeleton).
    $('iBody').insertAdjacentHTML('afterbegin',
        `<p class="hint">${id} is not in the node cloud — no soma and no skeleton.</p>`);
}

// ---------------------------------------------------------------------------
// UI
// ---------------------------------------------------------------------------

function renderLegend() {
    const wrap = $('legend'), rampWrap = $('rampWrap');
    wrap.innerHTML = '';
    if (state.mode === 'syn') {
        wrap.style.display = 'none';
        rampWrap.style.display = '';
        $('ramp').style.background = `linear-gradient(90deg, ${RAMP_HEX.join(',')})`;
        $('rampLabels').innerHTML = `<span>1 synapse</span><span>${fmt(state.synMax)}</span>`;
        $('legendHint').textContent = 'Log scale; brighter is more connected.';
        return;
    }
    wrap.style.display = '';
    rampWrap.style.display = 'none';
    $('legendHint').textContent = 'Click a row to isolate it · click again to release.';

    const v = state.vocab[state.mode];
    const rows = v.names.map((name, i) => ({ name, i, count: v.counts[i] }));
    const head = rows.slice(0, MAX_SERIES);
    const tail = rows.slice(MAX_SERIES);

    for (const r of head) wrap.appendChild(legendRow(r.name, r.i, r.count, catHex(r.i)));
    if (tail.length) {
        const n = tail.reduce((a, r) => a + r.count, 0);
        const row = legendRow(`other — ${tail.length} more`, -1, n, OTHER_HEX);
        row.title = tail.map((r) => r.name).join(', ');
        wrap.appendChild(row);
    }
}

function legendRow(name, idx, count, hex) {
    const v = state.vocab[state.mode];
    const el = document.createElement('div');
    el.className = 'lg';
    el.innerHTML = `<span class="sw" style="background:${hex}"></span><span>${escapeHtml(name)}</span><span class="n">${fmt(count)}</span>`;
    if (idx >= 0) {
        if (v.solo === idx) el.classList.add('solo');
        else if (v.solo !== null || v.hidden.has(idx)) el.classList.add('off');
        el.onclick = () => {
            v.solo = v.solo === idx ? null : idx;
            renderLegend();
            scheduleRebuild();
        };
    } else {
        el.classList.add('off');
        el.style.cursor = 'default';
    }
    return el;
}

function renderTreeToggles() {
    const el = $('trees');
    el.innerHTML = '';
    for (const t of state.trees) {
        const row = document.createElement('label');
        row.className = 'row';
        row.innerHTML = `<input type="checkbox" ${t.on ? 'checked' : ''}>
            <span>${t.title}<br><span style="color:var(--text-muted);font-size:11px">${t.subtitle}</span></span>
            <span class="n">${fmt(t.n)}</span>`;
        row.querySelector('input').onchange = (e) => { t.on = e.target.checked; scheduleRebuild(); };
        el.appendChild(row);
    }
}

/** Camera placement that frames a world-space box.
 *
 *  Fits both screen axes separately -- a 38° vertical FOV on a wide window has
 *  far more room horizontally -- backs off by the box's own depth so its near
 *  face does not end up behind the camera, and slides the whole view sideways
 *  by exactly the sidebar's share of the frustum, so the content centres in the
 *  glass rather than half-disappearing under the panel. */
function frameBox(lo, hi, margin = 1.1) {
    const c = lo.map((v, d) => (v + hi[d]) / 2);
    const half = lo.map((v, d) => Math.max(1, (hi[d] - v) / 2));
    const usable = Math.max(0.4, (innerWidth - SIDEBAR_PX) / innerWidth);
    const tanY = Math.tan((camera.fov * Math.PI) / 360);
    const tanX = tanY * camera.aspect;
    const dist = Math.max(half[1] / tanY, half[0] / (tanX * usable)) * margin + half[2];
    return { c, dist, dx: dist * tanX * (SIDEBAR_PX / innerWidth) };
}

function applyView(v) {
    const [x, y, z] = [v.c[0] - v.dx, v.c[1], v.c[2]];
    camera.position.set(x, y, z + v.dist);
    camera.lookAt(x, y, z);
    controls.target.set(x, y, z);
    controls.resetFromCamera();
}

function boundsOf(get, n) {
    const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
    for (let i = 0; i < n; i++) {
        for (let d = 0; d < 3; d++) {
            const v = get(i, d);
            if (v < lo[d]) lo[d] = v;
            if (v > hi[d]) hi[d] = v;
        }
    }
    return [lo, hi];
}

function fitView() {
    const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
    for (const t of state.trees) {
        for (let i = 0; i < t.n; i++) {
            for (let d = 0; d < 3; d++) {
                const v = t.disp[i * 3 + d];
                if (v < lo[d]) lo[d] = v;
                if (v > hi[d]) hi[d] = v;
            }
        }
    }
    homeView = frameBox(lo, hi);
    applyView(homeView);
}

function goHome() {
    if (homeView) applyView(homeView);
}

/** Frame the selected neuron -- its arbour if the skeleton arrived, else a
 *  30 µm box around the node itself. */
function focusSelected() {
    const sel = state.selected;
    if (!sel) return;
    const entry = state.skeletons.get(`${sel.tree.key}/${sel.id}`);
    if (entry && entry.bounds) {
        applyView(frameBox(entry.bounds[0], entry.bounds[1], 1.25));
    } else {
        const p = [sel.tree.disp[sel.i * 3], sel.tree.disp[sel.i * 3 + 1], sel.tree.disp[sel.i * 3 + 2]];
        applyView(frameBox(p.map((v) => v - 30), p.map((v) => v + 30), 1.1));
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

async function main() {
    boot.steps = 5;
    progress('starting wasm', 0);
    await initThreers({ module_or_path: '/threers/pkg/threers_bg.wasm' });
    progress('reading the index');

    state.meta = await fetch('/data/meta.json').then((r) => r.json());
    boot.steps = 3 + state.meta.trees.length;
    state.trees = await Promise.all(state.meta.trees.map(loadTree));
    state.synMax = Math.max(...state.trees.map((t) => t.syn.reduce((a, b) => Math.max(a, b), 0)));

    progress('placing nodes');
    placeTrees(state.trees);
    buildVocabs(state.trees);

    progress('starting the renderer');
    canvas = $('c');
    const dpr = Math.min(devicePixelRatio || 1, 2);
    const size = () => {
        canvas.width = Math.floor(innerWidth * dpr);
        canvas.height = Math.floor(innerHeight * dpr);
        return [canvas.width, canvas.height];
    };
    let [w, h] = size();
    renderer = await THREE.WebGLRenderer.create(canvas);
    renderer.setSize(w, h, false);

    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x1a1a19);
    // Unlit line material: the shader multiplies it by the per-vertex colour,
    // so white here means "whatever the vertex says".
    cloudMat = new THREE.LineBasicMaterial({ color: 0xffffff });

    camera = new THREE.PerspectiveCamera(38, w / h, 1, 40000);
    camera.position.set(0, 0, 2500);
    camera.lookAt(0, 0, 0);
    controls = new THREE.OrbitControls(camera, canvas);
    controls.enableDamping = true;
    controls.minDistance = 5;
    controls.maxDistance = 20000;
    fitView();

    addEventListener('resize', () => {
        [w, h] = size();
        renderer.setSize(w, h, false);
        camera.aspect = w / h;
        camera.updateProjectionMatrix?.();
    });

    // --- controls wiring
    renderTreeToggles();
    renderLegend();
    renderSkelList();
    $('total').textContent = fmt(state.meta.total);
    const indexed = state.meta.trees.reduce((a, t) => a + t.total_in_index, 0);
    const missing = state.meta.trees.reduce((a, t) => a + t.unplaceable, 0);
    $('subtitle').textContent =
        `${fmt(state.meta.total)} of ${fmt(indexed)} neurons · ${state.meta.trees.length} trees`;
    // Say what is not on screen rather than letting the count imply "all of it".
    $('coverage').innerHTML = state.meta.trees.map((t) =>
        `<b>${escapeHtml(t.title)}</b>: ${fmt(t.from_index)} placed by ${t.pos_source}`
        + (t.from_skeleton ? `, ${fmt(t.from_skeleton)} by a centroid computed from skeleton.swc` : '')
        + (t.unplaceable ? `, <b>${fmt(t.unplaceable)} with neither</b> — not drawable` : '')
        + '.').join('<br><br>')
        + (missing ? `<br><br>${fmt(missing)} of ${fmt(indexed)} neurons have no 3D position in the release at all; their ids are in <code>data/meta.json</code>.` : '');

    $('colorBy').onchange = (e) => {
        state.mode = e.target.value;
        renderLegend();
        scheduleRebuild();
    };
    $('minSyn').oninput = (e) => {
        // Curve the slider: most neurons live under a few hundred synapses.
        const u = Number(e.target.value) / 100;
        state.minSyn = Math.round(Math.pow(u, 3) * 20000);
        $('minSynVal').textContent = fmt(state.minSyn);
        scheduleRebuild(90);
    };
    $('size').oninput = (e) => {
        state.nodeSizeUm = Number(e.target.value) / 10;
        $('sizeVal').textContent = `${state.nodeSizeUm.toFixed(1)} µm`;
        scheduleRebuild(60);
    };
    $('search').oninput = (e) => {
        state.query = e.target.value;
        clearTimeout(rebuildTimer);
        rebuildTimer = setTimeout(() => {
            rebuildCloud().then(() => {
                $('searchHint').textContent = state.query
                    ? `${fmt(state.shown)} match "${state.query}"` : '';
            });
        }, 220);
    };
    $('spin').onchange = (e) => { controls.autoRotate = e.target.checked; };
    $('clearSkel').onclick = clearSkeletons;
    $('iFocus').onclick = focusSelected;
    $('iClose').onclick = () => {
        $('info').classList.remove('on');
        state.selected = null;
        if (highlight) { scene.remove(highlight); highlight = null; }
    };
    addEventListener('keydown', (e) => {
        if (/^(INPUT|SELECT|TEXTAREA)$/.test(e.target.tagName)) return;
        if (e.key === 'r') goHome();
        if (e.key === 'f') focusSelected();
    });

    // --- hover & click
    let hoverAt = null, dragged = false, down = null;
    canvas.addEventListener('pointerdown', (e) => { down = [e.clientX, e.clientY]; dragged = false; });
    canvas.addEventListener('pointermove', (e) => {
        if (down && Math.hypot(e.clientX - down[0], e.clientY - down[1]) > 4) dragged = true;
        hoverAt = [e.clientX, e.clientY];
    });
    canvas.addEventListener('pointerup', (e) => {
        down = null;
        if (dragged) return;
        const hit = pick(e.clientX, e.clientY);
        if (hit) select(hit.tree, hit.i);
    });
    canvas.addEventListener('pointerleave', () => { hoverAt = null; $('hover').classList.remove('on'); });

    await rebuildCloud();
    $('boot').style.display = 'none';

    // --- loop
    let frames = 0, last = performance.now(), hoverTick = 0;
    function frame() {
        requestAnimationFrame(frame);
        controls.update();
        renderer.render(scene, camera);
        frames++;
        const now = performance.now();
        if (now - last > 500) {
            $('fps').textContent = Math.round((frames * 1000) / (now - last));
            frames = 0; last = now;
        }
        // Hover picking is a linear scan; once every third frame is plenty and
        // keeps the scan off the critical path at 350k nodes.
        if (hoverAt && ++hoverTick % 3 === 0) {
            const hit = pick(hoverAt[0], hoverAt[1]);
            const el = $('hover');
            if (hit) {
                const [type, cls, side] = hit.tree.labels[hit.i].split('\t');
                const cat = state.mode === 'syn'
                    ? `${fmt(hit.tree.syn[hit.i])} synapses`
                    : state.vocab[state.mode].names[catOf(hit.tree, hit.i)];
                el.innerHTML = `<b>${escapeHtml(type || '(untyped)')}</b> · ${escapeHtml(cat)}`
                    + ` · ${escapeHtml(cls || '—')}${side ? ' · ' + escapeHtml(side) : ''}`
                    + ` <span style="color:var(--text-muted)">${hit.tree.ids[hit.i]}</span>`;
                el.classList.add('on');
                canvas.style.cursor = 'pointer';
            } else {
                el.classList.remove('on');
                canvas.style.cursor = '';
            }
        }
    }
    frame();
}

main().catch((e) => {
    console.error(e);
    $('bootMsg').textContent = 'failed';
    $('err').textContent = String(e && e.stack || e);
});
