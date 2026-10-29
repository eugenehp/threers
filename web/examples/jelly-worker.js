// The gummy creature's simulation in a worker: JellySim (jelly-sim.js) behind
// messages. Its snapshots go back without copying; the solver runs on the CPU,
// or with WebGPU compute on a device of the worker's own, or either as it
// measures best ('auto'). See jelly-client.js.
import { initThreers, SoftBody } from '../threejs-shim.js';
import { JellySim, transfers } from './jelly-sim.js';

let sim = null;

// a Rust panic's message, from console.error just before its trap (see jelly-demo.js)
let panicked = '';
const consoleError = console.error.bind(console);
console.error = (...a) => {
    const s = a.map(String).join(' ');
    if (s.includes('panicked at')) panicked = s.split('\n\nStack')[0];
    consoleError(...a);
};

// a WebGPU device of the worker's own, for the GPU solver; null without one
async function computeDevice() {
    if (!('gpu' in navigator)) return null;
    try { return await SoftBody.requestComputeDevice(); } catch (_) { return null; }
}

self.onmessage = async (e) => {
    const m = e.data;
    try {
        if (m.type === 'build') {
            await initThreers({ module_or_path: m.wasm });
            const device = m.solver === 'cpu' ? null : await computeDevice();
            sim = new JellySim(m.creature, device ? (on) => sim.body.useGpuDevice(device, on) : null);
            const info = sim.build();
            const onGpu = sim.solver(m.solver);
            self.postMessage({ type: 'ready', info, onGpu, gpuAvailable: !!device });
        } else if (m.type === 'frame') {
            const snap = sim.frame(m);
            self.postMessage({ type: 'frame', snap }, transfers(snap));
        } else if (m.type === 'pick') {
            self.postMessage({ type: 'reply', id: m.id, value: sim.pick(m.o, m.d, m.axis) });
        } else if (m.type === 'solver') {
            self.postMessage({ type: 'reply', id: m.id, value: sim.solver(m.mode) });
        } else {
            const r = sim.input(m);
            if (m.id !== undefined) self.postMessage({ type: 'reply', id: m.id, value: r });
        }
    } catch (err) {
        self.postMessage({ type: 'error', message: `${String(err?.stack ?? err)}${panicked ? `\n${panicked}` : ''}` });
    }
};
