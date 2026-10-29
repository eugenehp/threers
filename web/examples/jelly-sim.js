// The gummy creature's simulation, apart from the page: it builds the
// creature, takes the hands' input, steps, and hands back a *snapshot* each
// frame — plain typed arrays, so it can live in a worker (jelly-worker.js)
// and post them across without copying, or on the main thread just the same
// (`?thread=main`; see jelly-client.js).
//
//   build()        the creature; what the page needs to draw it, once
//   pick(o, d, a)  a ray from the pointer: takes hold of what it hits
//   input(msg)     the hands, the sliders, reset, startle
//   solver(mode)   'cpu', 'gpu' or 'auto' (see Autopilot)
//   frame(req)     step, and the snapshot to draw
import { SoftBody } from '../threejs-shim.js';
import { CREATURES } from './creatures.js';

// Which solver, decided by measurement. The GPU solver is fastest where the
// GPU has room and slower where the frame is already busy there; neither is
// the better on every machine or at every moment, so in 'auto' this tries
// the GPU when the CPU solver runs over budget, keeps it only if the page's
// frame rate improves, and lets it go when its readbacks fall behind or the
// frame rate sags. Each failed try doubles the wait before the next.
const BUDGET_MS = 12, TRIAL_MS = 2500, WARM_MS = 500;

class Autopilot {
    constructor() { this.state = 'cpu'; this.cooldown = 0; this.backoff = 10000; this.since = 0; this.note = ''; }

    /// What to do this frame: `true` to be on the GPU, `false` the CPU.
    step(now, { cpuMs, fps, lag, gpuAvailable }) {
        const s = this;
        if (s.state === 'cpu') {
            if (!gpuAvailable || now < s.cooldown || cpuMs <= BUDGET_MS) { s.since = 0; return false; }
            s.since ||= now;
            if (now - s.since < 1000) return false;
            Object.assign(s, { state: 'trial', baseline: fps, trialEnd: now + TRIAL_MS, warmEnd: now + WARM_MS, sum: 0, n: 0 });
            s.note = `trying the GPU: the CPU solver takes ${cpuMs.toFixed(1)} ms at ${fps.toFixed(0)} fps`;
            return true;
        }
        if (s.state === 'trial') {
            if (now > s.warmEnd) { s.sum += fps; s.n++; }
            if (now < s.trialEnd) return true;
            const got = s.n ? s.sum / s.n : 0;
            if (got >= s.baseline * 1.05 || got >= 57) {
                Object.assign(s, { state: 'gpu', kept: got, since: 0, backoff: 10000 });
                s.note = `on the GPU: ${got.toFixed(0)} fps against ${s.baseline.toFixed(0)} on the CPU`;
                return true;
            }
            return s.back(now, `back to the CPU: the GPU gave ${got.toFixed(0)} fps against ${s.baseline.toFixed(0)}`);
        }
        // on the GPU: stay while it keeps up
        const slow = lag >= 3 || fps < s.kept * 0.75;
        if (!slow) { s.since = 0; return true; }
        s.since ||= now;
        if (now - s.since < (lag >= 3 ? 1000 : 2000)) return true;
        return s.back(now, lag >= 3 ? `back to the CPU: the GPU is busy (readbacks ${lag} frames behind)` : `back to the CPU: the frame rate fell to ${fps.toFixed(0)}`);
    }
    back(now, note) {
        Object.assign(this, { state: 'cpu', cooldown: now + this.backoff, backoff: Math.min(80000, this.backoff * 2), since: 0, note });
        return false;
    }
}

export class JellySim {
    /// `gpu(on)` turns the GPU solver on or off and says whether it is on;
    /// null where there is no WebGPU.
    constructor(name, gpu = null) {
        this.creature = CREATURES[name];
        this.sugarOn = false;
        this.linesOn = false;
        this.solveMs = 0;
        this.cpuMs = 0;
        this.gpu = gpu;
        this.mode = 'cpu';
        this.autopilot = new Autopilot();
        this.fps = 60;
    }

    /// 'cpu', 'gpu' or 'auto'. Returns whether the GPU solver is on.
    solver(mode) {
        this.mode = mode;
        this.autopilot = new Autopilot();
        if (mode === 'auto') this.autopilot.note = 'auto: on the CPU solver';
        return this.useGpu(mode === 'gpu');
    }
    useGpu(on) {
        if (!this.gpu) return false;
        if (on !== this.body.onGpu) this.gpu(on);
        return this.body.onGpu;
    }

    build() {
        const t0 = performance.now();
        const built = this.creature.build();
        const body = this.body = built.body;
        this.creature.setup(body);
        const topos = 1 + body.parts.length;
        this.skins = Array.from({ length: topos }, (_, t) => body.skin(t));
        this.insides = built.insides.map(({ kind, surf }) => ({ kind, surf, binding: body.bind(0, surf.pos, surf.nrm) }));
        this.segCount = body.segments(true).length / 8;
        // sugar: crystals scattered over each part's rest skin by area, bound
        // to its lattice (the same seed every time, so the same crystals)
        let seed = 7;
        const rand = () => (seed = (seed * 16807) % 2147483647) / 2147483647;
        this.sugar = this.skins.map(({ position: P, normal: N, index: I }, topo) => {
            const pts = [];
            for (let t = 0; t < I.length; t += 3) {
                const a = 3 * I[t], b = 3 * I[t + 1], c = 3 * I[t + 2];
                const ux = P[b] - P[a], uy = P[b + 1] - P[a + 1], uz = P[b + 2] - P[a + 2], vx = P[c] - P[a], vy = P[c + 1] - P[a + 1], vz = P[c + 2] - P[a + 2];
                let k = 0.5 * Math.hypot(uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx) * 900;
                while (k > 0) {
                    if (k < 1 && rand() > k) break;
                    k -= 1;
                    let r1 = rand(), r2 = rand(); if (r1 + r2 > 1) { r1 = 1 - r1; r2 = 1 - r2; }
                    const n = [0, 1, 2].map((q) => N[a + q] + (N[b + q] - N[a + q]) * r1 + (N[c + q] - N[a + q]) * r2);
                    const nl = Math.hypot(...n);
                    for (let q = 0; q < 3; q++) pts.push(P[a + q] + (P[b + q] - P[a + q]) * r1 + (P[c + q] - P[a + q]) * r2 - n[q] / nl * 0.002);
                }
            }
            return { binding: body.bind(topo, pts), n: pts.length / 3 };
        });
        return {
            parts: body.parts,
            limits: body.parts.map((_, i) => body.tearLimit(i)),
            skins: this.skins,
            insides: this.insides.map(({ kind, surf }) => ({ kind, surf })),
            segCount: this.segCount,
            sugarCounts: this.sugar.map((s) => s.n),
            blocks: body.blocks(),
            buildMs: performance.now() - t0,
        };
    }

    /// Take hold of what the ray hits: `{ kind: 'part'|'drop', block|index, point }` or null.
    pick(o, d, axis) {
        const hit = this.body.raycast(o, d);
        if (hit?.kind === 'part') this.body.grab(hit.block, hit.point, axis);
        if (hit?.kind === 'drop') this.body.grabDrop(hit.index, hit.point);
        return hit;
    }

    /// The hands, the sliders and the buttons. Returns a reply when there is one.
    input(m) {
        const b = this.body;
        switch (m.type) {
            case 'move': b.move(m.p); break;
            case 'twist': b.twist(m.angle); break;
            case 'release': b.release(); break;
            case 'moveDrop': b.moveDrop(m.p); break;
            case 'releaseDrop': return { speed: b.releaseDrop() };
            case 'params': b.setParams(m.params); break;
            case 'startle': b.startle(); break;
            case 'reset': b.reset(); break;
            case 'sugar': this.sugarOn = m.on; break;
            case 'lines': this.linesOn = m.on; break;
        }
        return null;
    }

    /// One frame: step by `dt` in `substeps`, inside `walls`, and what to
    /// draw. `hover` is a ray to test for the pointer's highlight.
    frame({ dt, substeps, walls, hover, fps }) {
        const b = this.body;
        if (walls) b.walls = walls;
        if (fps) this.fps = this.fps * 0.8 + fps * 0.2;
        if (this.mode === 'auto' && dt > 0) {
            const now = performance.now();
            const want = this.autopilot.step(now, { cpuMs: this.cpuMs, fps: this.fps, lag: b.gpuLag ?? 0, gpuAvailable: !!this.gpu });
            this.useGpu(want);
        }
        if (dt > 0) {
            b.substeps = substeps;
            const t0 = performance.now();
            b.step(dt);
            const ms = performance.now() - t0;
            this.solveMs = this.solveMs * 0.9 + ms * 0.1;
            // what the CPU solver costs, kept while the GPU one runs
            if (!b.onGpu) this.cpuMs = this.cpuMs * 0.9 + ms * 0.1;
        }
        const blocks = b.blocks();
        const topoOf = (k) => (k.part < 0 ? 0 : k.part + 1);
        // every active block's skin, one after another in block order
        let nv = 0;
        for (const k of blocks) if (k.active) nv += this.skins[topoOf(k)].position.length;
        const skinPos = new Float32Array(nv), skinNrm = new Float32Array(nv);
        let o = 0;
        blocks.forEach((k, bi) => {
            if (!k.active) return;
            const len = this.skins[topoOf(k)].position.length;
            b.carry(topoOf(k), bi, skinPos.subarray(o, o + len), skinNrm.subarray(o, o + len));
            o += len;
        });
        let ni = 0;
        for (const s of this.insides) ni += s.surf.pos.length;
        const insidePos = new Float32Array(ni), insideNrm = new Float32Array(ni);
        o = 0;
        for (const s of this.insides) {
            const len = s.surf.pos.length;
            b.carry(s.binding, 0, insidePos.subarray(o, o + len), insideNrm.subarray(o, o + len));
            o += len;
        }
        let sugar = null;
        if (this.sugarOn) {
            let ns = 0;
            for (const k of blocks) if (k.active) ns += 3 * this.sugar[topoOf(k)].n;
            sugar = new Float32Array(ns);
            o = 0;
            blocks.forEach((k, bi) => {
                if (!k.active) return;
                const S = this.sugar[topoOf(k)];
                b.carry(S.binding, bi, sugar.subarray(o, o + 3 * S.n));
                o += 3 * S.n;
            });
        }
        let hit = null;
        if (hover) {
            const h = b.raycast(hover.o, hover.d);
            hit = h ? { kind: h.kind, block: h.block ?? -1 } : null;
        }
        return {
            blocks, skinPos, skinNrm, insidePos, insideNrm, sugar,
            segments: this.segCount ? b.segments(false) : new Float32Array(0),
            drops: b.drops(),
            lines: this.linesOn ? b.latticeLines() : null,
            measure: b.measure(), events: b.events(), held: b.held, feel: b.feel(),
            regrown: b.regrown, time: b.time, onGpu: b.onGpu, solveMs: this.solveMs, hover: hit,
            solverMode: this.mode, solverNote: this.mode === 'auto' ? this.autopilot.note : '', gpuLag: b.gpuLag ?? 0,
        };
    }
}

/// The snapshot's buffers, to hand over without copying.
export function transfers(snap) {
    return [snap.skinPos, snap.skinNrm, snap.insidePos, snap.insideNrm, snap.sugar, snap.segments, snap.drops, snap.lines]
        .filter((a) => a && a.buffer && a.byteLength > 0)
        .map((a) => a.buffer);
}
