// The page's handle on the simulation, wherever it runs: in a worker (the
// default — the solver, skinning and picking all off the main thread), or on
// the main thread (`?thread=main`). Either way the page asks for a frame and
// draws the latest snapshot it has; it never waits for one.
//
//   const sim = await startSim({ creature, thread, gpu, renderer });
//   sim.info                      what to draw, from the build
//   sim.frame(req)                ask for the next frame (at most one in flight)
//   sim.latest                    the newest snapshot
//   sim.send(msg)                 the hands, sliders and buttons
//   await sim.pick(o, d, axis)    take hold of what a ray hits
//   await sim.setSolver(mode)     'cpu', 'gpu' or 'auto'; whether the GPU is on
import { initThreers } from '../threejs-shim.js';
import { JellySim } from './jelly-sim.js';

const WASM = '/web/pkg/threers_bg.wasm';

export async function startSim({ creature, thread = 'worker', solver = 'auto', renderer = null, onError = () => {} }) {
    if (thread === 'main') return startLocal(creature, solver, renderer);
    const worker = new Worker(new URL('./jelly-worker.js', import.meta.url), { type: 'module' });
    let nextId = 0, inFlight = false;
    const waiting = new Map();
    const client = { latest: null, info: null, onGpu: false, thread: 'worker' };
    const ready = new Promise((resolve, reject) => {
        worker.onmessage = (e) => {
            const m = e.data;
            if (m.type === 'ready') { client.info = m.info; client.onGpu = m.onGpu; client.gpuAvailable = m.gpuAvailable; resolve(); }
            else if (m.type === 'frame') { client.latest = m.snap; client.onGpu = m.snap.onGpu; inFlight = false; }
            else if (m.type === 'reply') { waiting.get(m.id)?.(m.value); waiting.delete(m.id); }
            else if (m.type === 'error') { onError(m.message); reject(new Error(m.message)); inFlight = false; }
        };
        worker.onerror = (e) => { onError(e.message); reject(new Error(e.message)); };
    });
    const ask = (msg) => new Promise((r) => { const id = nextId++; waiting.set(id, r); worker.postMessage({ ...msg, id }); });
    worker.postMessage({ type: 'build', creature, solver, wasm: WASM });
    await ready;
    Object.assign(client, {
        frame(req) { if (inFlight) return false; inFlight = true; worker.postMessage({ type: 'frame', ...req }); return true; },
        send(msg) { worker.postMessage(msg); },
        pick(o, d, axis) { return ask({ type: 'pick', o, d, axis }); },
        releaseDrop() { return ask({ type: 'releaseDrop' }).then((r) => r?.speed ?? 0); },
        async setSolver(mode) { client.onGpu = await ask({ type: 'solver', mode }); return client.onGpu; },
    });
    return client;
}

async function startLocal(creature, solver, renderer) {
    await initThreers({ module_or_path: WASM });
    const webgpu = renderer && String(renderer._w.backend?.() ?? '') === 'webgpu';
    const sim = new JellySim(creature, webgpu ? (on) => sim.body.useGpu(renderer, on) : null);
    const client = { latest: null, info: sim.build(), thread: 'main', gpuAvailable: webgpu };
    client.onGpu = sim.solver(solver);
    Object.assign(client, {
        frame(req) { client.latest = sim.frame(req); return true; },
        send(msg) { sim.input(msg); },
        pick(o, d, axis) { return Promise.resolve(sim.pick(o, d, axis)); },
        releaseDrop() { return Promise.resolve(sim.input({ type: 'releaseDrop' }).speed); },
        setSolver: async (mode) => (client.onGpu = sim.solver(mode)),
    });
    return client;
}
