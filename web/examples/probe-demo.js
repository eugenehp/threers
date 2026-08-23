// Live Cornell: orbit camera → WASM probe/G-buffer → hops+NRC inference each frame.
import initProbe, {
    WebProbeGi,
    expand_planar_rgb,
    probe_in_channels,
    cornell_orbit_camera,
} from '../pkg-probe/threers_probe.js';

const ASSETS = '../assets/probe';
const FOV = 40;
const TARGET = [0, 1, 0];

const statusEl = document.getElementById('status');
const errEl = document.getElementById('err');
const tilesEl = document.getElementById('tiles');
const liveCanvas = document.getElementById('live');
const sideSel = document.getElementById('side');
const sppSel = document.getElementById('spp');
const showProbeEl = document.getElementById('showProbe');

const say = (t) => { statusEl.textContent = t; };
const fail = (e) => {
    console.error(e);
    errEl.textContent = e?.message || String(e);
};

async function fetchBytes(url) {
    const r = await fetch(url);
    if (!r.ok) throw new Error(`${url}: ${r.status}`);
    return new Uint8Array(await r.arrayBuffer());
}

async function fetchF32(url) {
    const bytes = await fetchBytes(url);
    return new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
}

function drawPlanarTo(canvas, planar, side) {
    canvas.width = side;
    canvas.height = side;
    const ctx = canvas.getContext('2d');
    const rgba = expand_planar_rgb(planar, side);
    if (!rgba.length) return;
    const img = ctx.createImageData(side, side);
    for (let i = 0; i < side * side; i++) {
        const o = i * 4;
        img.data[o] = Math.min(255, rgba[o] * 255);
        img.data[o + 1] = Math.min(255, rgba[o + 1] * 255);
        img.data[o + 2] = Math.min(255, rgba[o + 2] * 255);
        img.data[o + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);
}

function addTile(planar, side, label) {
    const fig = document.createElement('figure');
    const canvas = document.createElement('canvas');
    const cap = document.createElement('figcaption');
    cap.textContent = label;
    fig.appendChild(canvas);
    fig.appendChild(cap);
    tilesEl.appendChild(fig);
    drawPlanarTo(canvas, planar, side);
}

function planarSlice(planar, side, ch0 = 0) {
    const n = side * side;
    const out = new Float32Array(n * 3);
    for (let i = 0; i < n; i++) {
        out[i] = planar[ch0 * n + i];
        out[n + i] = planar[(ch0 + 1) * n + i];
        out[2 * n + i] = planar[(ch0 + 2) * n + i];
    }
    return out;
}

function relErr(pred, target, eps = 0.01) {
    let sum = 0;
    const n = target.length;
    for (let i = 0; i < n; i++) {
        const d = pred[i] - target[i];
        const t = target[i];
        sum += (d * d) / (t * t + eps);
    }
    return Math.sqrt(sum / n);
}

class Orbit {
    constructor() {
        this.yaw = 0;
        this.pitch = 0.05;
        this.distance = 3.9;
        this.dragging = false;
        this.lastX = 0;
        this.lastY = 0;
    }

    camera() {
        const v = cornell_orbit_camera(this.yaw, this.pitch, this.distance);
        return {
            pos: [v[0], v[1], v[2]],
            target: [v[3], v[4], v[5]],
        };
    }

    bind(el, onChange) {
        el.addEventListener('pointerdown', (e) => {
            this.dragging = true;
            el.classList.add('dragging');
            el.setPointerCapture(e.pointerId);
            this.lastX = e.clientX;
            this.lastY = e.clientY;
            onChange(true);
        });
        el.addEventListener('pointermove', (e) => {
            if (!this.dragging) return;
            const dx = e.clientX - this.lastX;
            const dy = e.clientY - this.lastY;
            this.lastX = e.clientX;
            this.lastY = e.clientY;
            this.yaw -= dx * 0.005;
            this.pitch += dy * 0.005;
            this.pitch = Math.max(-1.2, Math.min(1.2, this.pitch));
            onChange(true);
        });
        const end = (e) => {
            if (!this.dragging) return;
            this.dragging = false;
            el.classList.remove('dragging');
            try { el.releasePointerCapture(e.pointerId); } catch (_) {}
            onChange(false);
        };
        el.addEventListener('pointerup', end);
        el.addEventListener('pointercancel', end);
        el.addEventListener('wheel', (e) => {
            e.preventDefault();
            this.distance *= e.deltaY > 0 ? 1.08 : 0.92;
            this.distance = Math.max(1.6, Math.min(6.0, this.distance));
            onChange(false);
        }, { passive: false });
    }
}

async function main() {
    try {
        say('initializing WASM…');
        await initProbe();
        const inCh = probe_in_channels();
        if (inCh !== 22) throw new Error(`expected 22 input planes, got ${inCh}`);

        const [probeW, nrcW] = await Promise.all([
            fetchBytes(`${ASSETS}/probe_hops.bin`),
            fetchBytes(`${ASSETS}/nrc_hops.bin`),
        ]);

        // Optional hold-out strip (64² export) if assets exist.
        try {
            const HOLD = 64;
            const [input, reference] = await Promise.all([
                fetchF32(`${ASSETS}/cornell_input.bin`),
                fetchF32(`${ASSETS}/cornell_ref.bin`),
            ]);
            if (input.length === inCh * HOLD * HOLD) {
                const gi64 = new WebProbeGi(probeW, nrcW, HOLD);
                const probePlanar = planarSlice(input, HOLD, 0);
                const hops = gi64.reconstruct_hops(input);
                const fused = gi64.reconstruct(input);
                const eProbe = relErr(probePlanar, reference);
                const eFused = relErr(fused, reference);
                const pct = (100 * eFused / eProbe).toFixed(1);
                addTile(probePlanar, HOLD, `hold-out probe · ${(100 * eProbe).toFixed(0)}% err`);
                addTile(hops, HOLD, 'hold-out hops');
                addTile(fused, HOLD, `hold-out +NRC · ${pct}% of probe`);
                addTile(reference, HOLD, 'hold-out reference');
            }
        } catch (e) {
            console.warn('hold-out strip skipped', e);
        }

        let side = Number(sideSel.value);
        let gi = new WebProbeGi(probeW, nrcW, side);
        const orbit = new Orbit();
        let dirty = true;
        let running = false;
        let lastMs = 0;

        const schedule = (dragging) => {
            dirty = true;
            if (!dragging) {
                // After release, re-infer at the selected (higher) spp.
                idleRefine = true;
            }
            pump();
        };

        let idleRefine = false;

        const rebuildGi = () => {
            side = Number(sideSel.value);
            gi.free();
            gi = new WebProbeGi(probeW, nrcW, side);
            liveCanvas.width = side;
            liveCanvas.height = side;
            dirty = true;
            pump();
        };
        sideSel.addEventListener('change', rebuildGi);
        sppSel.addEventListener('change', () => { dirty = true; pump(); });
        showProbeEl.addEventListener('change', () => { dirty = true; pump(); });

        orbit.bind(liveCanvas, schedule);

        async function pump() {
            if (running || !dirty) return;
            running = true;
            dirty = false;
            const dragging = orbit.dragging;
            let spp = Number(sppSel.value);
            if (dragging) spp = Math.min(spp, 1);
            else if (idleRefine) {
                spp = Math.max(spp, 4);
                idleRefine = false;
            }
            const bounces = dragging ? 2 : 3;
            const { pos } = orbit.camera();
            const t0 = performance.now();
            try {
                const beauty = showProbeEl.checked
                    ? gi.pack_cornell_probe(pos[0], pos[1], pos[2], TARGET[0], TARGET[1], TARGET[2], FOV, spp, bounces)
                    : gi.infer_cornell(pos[0], pos[1], pos[2], TARGET[0], TARGET[1], TARGET[2], FOV, spp, bounces);
                drawPlanarTo(liveCanvas, beauty, side);
                lastMs = performance.now() - t0;
                say(
                    `${side}² · spp ${spp} · ${showProbeEl.checked ? 'probe' : 'hops+NRC'} · ${lastMs.toFixed(0)} ms`
                    + (dragging ? ' · dragging' : ''),
                );
            } catch (e) {
                fail(e);
                say('inference failed');
            }
            running = false;
            if (dirty) pump();
        }

        // First frame.
        liveCanvas.width = side;
        liveCanvas.height = side;
        dirty = true;
        await pump();
    } catch (e) {
        fail(e);
        say('failed');
    }
}

main();
