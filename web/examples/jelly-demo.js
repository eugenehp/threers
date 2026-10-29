// Gummy jelly on threers: gummy creatures you can pull apart — a jellyfish,
// a gecko, a starfish, an octopus, a sea slug, a gummy worm, a gummy bear and a salamander.
//
//   creatures.js     each creature: THREE.SDF fields, muscles, looks, framing
//   creature-kit.js  building one on THREE.SoftBody (threers::softbody, Rust)
//   jelly-sim.js     the simulation: build, hands, step, snapshot
//   jelly-worker.js  …run in a worker — the default — so none of it is on
//                    the main thread; jelly-client.js is the page's handle
//   jelly-scene.js   what is drawn: gel, insides, chains, goo, sugar, table
//   jelly-stage.js   the camera and the hands
//   jelly-sound.js   every noise, synthesized (from Axolotl Jelly, MIT)
//   this file        the page: panel, keys, sounds, readouts, the loop
//
// The soft-body solver is a Rust port of Axolotl Jelly's by Dream Meridian
// (MIT), held to the original bit for bit by tests/softbody_parity.rs.
//
// URL: `?creature=octopus`, `?look=nettle` `?finish=sour` `?table=night`
// start elsewhere; `?thread=main` runs the simulation on the main thread and
// `?solver=cpu|gpu|auto` picks its solver (auto: whichever measures better); `?frames=N` reports to the dev server's /__status after N
// frames (for the verification harness) and carries on. `window.__jelly`
// holds the simulation's handle, the scene and the state for the console.
import THREE, { initThreers } from '../threejs-shim.js';
import { JellySound } from './jelly-sound.js';
import { CREATURES } from './creatures.js';
import { createView, FINISHES, TABLES } from './jelly-scene.js';
import { createStage } from './jelly-stage.js';
import { startSim } from './jelly-client.js';

const $ = (id) => document.getElementById(id);
const params = new URLSearchParams(location.search);
const reportAt = Number(params.get('frames') ?? 0);

function beacon(p) { fetch('/__status?' + new URLSearchParams(p)).catch(() => {}); }
// A Rust panic surfaces as a bare `RuntimeError: unreachable`; its message
// went to console.error just before, so keep that to report with it.
let panicked = '';
const consoleError = console.error.bind(console);
console.error = (...a) => {
    const s = a.map(String).join(' ');
    if (s.includes('panicked at')) panicked = s.split('\n\nStack')[0];
    consoleError(...a);
};
function fail(message) {
    if (panicked && /unreachable/.test(message)) message = `${message}: ${panicked}`;
    $('err').textContent += `${message}\n`;
    $('overlay').classList.add('failed');
    $('overlay-text').textContent = message;
    beacon({ error: message });
}
window.addEventListener('error', (e) => fail(e.message));
window.addEventListener('unhandledrejection', (e) => fail(String(e.reason?.message ?? e.reason)));

const NAMES = Object.keys(CREATURES);
const creatureName = CREATURES[params.get('creature')] ? params.get('creature') : 'jellyfish';
const creature = CREATURES[creatureName];
const LOOKS = creature.looks;
const pick = (key, table, fallback) => (table[params.get(key)] ? params.get(key) : fallback);
const DEFAULTS = { look: creature.look, finish: 'glossy', table: 'paper', firmness: 0.5, damping: 0.4, regrow: 0.5 };
const thread = params.get('thread') === 'main' ? 'main' : 'worker';

async function main() {
    $('overlay-text').textContent = `building the ${creature.title}…`;
    await initThreers({ module_or_path: '/web/pkg/threers_bg.wasm' });
    const canvas = $('c');
    const renderer = await THREE.WebGLRenderer.create(canvas, { antialias: true });
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    const backend = String(renderer._w.backend?.() ?? '');
    // The simulation: in a worker unless `?thread=main`; on whichever solver
    // measures better here ('auto', the default), or `?solver=cpu|gpu`. In a
    // worker the CPU solver already runs beside the drawing; the GPU one is
    // faster where the GPU has room and slower where the frame is busy there,
    // so 'auto' tries it and keeps it only if the frame rate says so.
    let solver = ['auto', 'cpu', 'gpu'].includes(params.get('solver')) ? params.get('solver') : 'auto';
    const sim = await startSim({ creature: creatureName, thread, renderer, solver, onError: fail });
    const info = sim.info;
    const view = createView(renderer, creature, info);
    const camera = new THREE.PerspectiveCamera(30, 16 / 9, 0.1, 60);
    // the solver: 'auto' tries the GPU when the CPU runs over budget and keeps
    // whichever gives the better frame rate here (jelly-sim.js, Autopilot)
    const setSolver = async (mode) => {
        solver = mode;
        document.querySelectorAll('.solver').forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.solver === mode)));
        await sim.setSolver(mode);
    };
    document.querySelectorAll('.solver').forEach((b) => {
        if (b.dataset.solver === 'gpu' && !sim.gpuAvailable) { b.disabled = true; b.title = 'No WebGPU here'; }
        b.addEventListener('click', () => setSolver(b.dataset.solver));
    });
    document.querySelectorAll('.solver').forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.solver === solver)));

    // ── state ──
    const state = {
        look: pick('look', LOOKS, DEFAULTS.look), finish: pick('finish', FINISHES, DEFAULTS.finish), table: pick('table', TABLES, DEFAULTS.table),
        paused: false, slow: false,
    };
    const applyLook = () => {
        view.setLook(LOOKS[state.look], FINISHES[state.finish]);
        sim.send({ type: 'sugar', on: view.sugarVisible() });
        document.querySelectorAll('.swatch').forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.look === state.look)));
        document.querySelectorAll('.finish').forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.finish === state.finish)));
    };
    const applyTable = () => {
        view.setTable(state.table);
        document.documentElement.dataset.table = state.table;
        document.querySelectorAll('.table').forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.table === state.table)));
    };

    // ── words ──
    let sayTimer = 0;
    function say(text) {
        $('say').textContent = text;
        $('say').classList.add('shown');
        clearTimeout(sayTimer);
        sayTimer = setTimeout(() => $('say').classList.remove('shown'), 5000);
    }

    // ── hands ──
    let lastMove = null, moveSpeed = 0;
    const stage = createStage({
        canvas, renderer, camera, sim, framing: creature.camera,
        on: {
            grab(kind, onBody) {
                if (kind === 'part') JellySound.squish(onBody ? 1 : 0.7);
                if (kind === 'drop') JellySound.squish(0.35);
                if (kind !== 'orbit') $('hint').classList.add('gone');
                lastMove = null;
            },
            move(kind, p) {
                const now = performance.now();
                if (lastMove) moveSpeed = Math.hypot(p[0] - lastMove.p[0], p[1] - lastMove.p[1], p[2] - lastMove.p[2]) / Math.max(1e-3, (now - lastMove.t) / 1000);
                lastMove = { p, t: now };
            },
            release(kind, speed) {
                if (kind === 'part') JellySound.release(1);
                if (kind === 'drop') JellySound.fwip(speed);
                lastMove = null; moveSpeed = 0;
            },
        },
    });

    // ── the panel ──
    document.querySelectorAll('.creature').forEach((a) => a.setAttribute('aria-pressed', String(a.dataset.creature === creatureName)));
    $('swatches').innerHTML = Object.entries(LOOKS).map(([k, L], i) =>
        `<button class="pick swatch" data-look="${k}" title="${k} (shift+${i + 1})"><span class="chip" style="background: ${L.chip}"></span>${k[0].toUpperCase() + k.slice(1)}</button>`).join('');
    $('k-parts').textContent = creature.parts;
    $('k-length').textContent = creature.length;
    $('hint').innerHTML = `<b>Tip:</b> ${creature.hint}`;
    document.title = `gummy ${creature.title} — threers`;
    applyTable(); applyLook();
    document.querySelectorAll('.swatch').forEach((b) => b.addEventListener('click', () => { state.look = b.dataset.look; applyLook(); }));
    const setFinish = (f) => {
        state.finish = f; applyLook();
        JellySound.setGrit(f === 'sour' ? 1 : f === 'frosted' ? 0.3 : 0);
        JellySound.squish(0.5);
    };
    document.querySelectorAll('.finish').forEach((b) => b.addEventListener('click', () => setFinish(b.dataset.finish)));
    document.querySelectorAll('.table').forEach((b) => b.addEventListener('click', () => { state.table = b.dataset.table; applyTable(); }));
    const sliders = ['firmness', 'damping', 'regrow'];
    const pushParams = () => sim.send({ type: 'params', params: Object.fromEntries(sliders.map((id) => [id, +$(id).value])) });
    for (const id of sliders) $(id).addEventListener('input', () => { $(id + '-out').textContent = (+$(id).value).toFixed(2); pushParams(); });
    pushParams();
    const startle = () => { sim.send({ type: 'startle' }); JellySound.startle(); };
    const reset = () => {
        sim.send({ type: 'reset' });
        Object.assign(state, { look: DEFAULTS.look, finish: DEFAULTS.finish, table: DEFAULTS.table, slow: false, paused: false });
        applyLook(); applyTable();
        for (const id of sliders) { $(id).value = DEFAULTS[id]; $(id + '-out').textContent = DEFAULTS[id].toFixed(2); }
        pushParams();
        $('slow').checked = false; JellySound.setSlow(false);
        $('mesh').checked = false; view.lines.visible = false; sim.send({ type: 'lines', on: false });
        $('pause').textContent = 'Pause';
        stage.home();
        JellySound.slurp();
        say('Reset. Everything is back the way it started.');
    };
    const togglePause = () => { state.paused = !state.paused; $('pause').textContent = state.paused ? 'Resume' : 'Pause'; };
    const toggle = (id) => { $(id).checked = !$(id).checked; $(id).dispatchEvent(new Event('change')); };
    $('startle').addEventListener('click', startle);
    $('reset').addEventListener('click', reset);
    $('pause').addEventListener('click', togglePause);
    $('slow').addEventListener('change', (e) => { state.slow = e.target.checked; JellySound.setSlow(state.slow); });
    $('mesh').addEventListener('change', (e) => { view.lines.visible = e.target.checked; sim.send({ type: 'lines', on: e.target.checked }); });
    $('sound').checked = JellySound.enabled;
    $('sound').addEventListener('change', (e) => { JellySound.setEnabled(e.target.checked); if (e.target.checked) JellySound.squish(0.6); });
    // audio may only start after a click or a key
    const wake = () => JellySound.unlock();
    document.addEventListener('pointerdown', wake, true);
    document.addEventListener('keydown', wake, true);

    // ── keys ──
    const KEYS = {
        ' ': togglePause, s: startle, r: reset, l: () => toggle('mesh'), q: () => toggle('slow'), m: () => toggle('sound'),
        h: () => stage.home(),
        g: () => { const M = sim.gpuAvailable ? ['auto', 'cpu', 'gpu'] : ['auto', 'cpu']; setSolver(M[(M.indexOf(solver) + 1) % M.length]); },
        f: () => setFinish(Object.keys(FINISHES)[(Object.keys(FINISHES).indexOf(state.finish) + 1) % 3]),
        t: () => { const T = Object.keys(TABLES); state.table = T[(T.indexOf(state.table) + 1) % T.length]; applyTable(); },
        ArrowLeft: () => stage.orbit(-0.1, 0), ArrowRight: () => stage.orbit(0.1, 0), ArrowUp: () => stage.orbit(0, 0.06), ArrowDown: () => stage.orbit(0, -0.06),
        '+': () => stage.zoom(0.9), '=': () => stage.zoom(0.9), '-': () => stage.zoom(1.1),
        '?': () => $('keys').toggleAttribute('open'),
    };
    document.addEventListener('keydown', (e) => {
        if (e.target.closest?.('input, textarea, select') || e.metaKey || e.ctrlKey || e.altKey) return;
        // 1–8 pick a creature; shift+1–5 a look
        if (/^Digit[1-9]$/.test(e.code)) {
            const n = +e.code.slice(5) - 1;
            if (e.shiftKey) { const L = Object.keys(LOOKS)[n]; if (L) { state.look = L; applyLook(); } }
            else if (NAMES[n] && NAMES[n] !== creatureName) { const p = new URLSearchParams(location.search); p.set('creature', NAMES[n]); location.search = p; }
            e.preventDefault();
            return;
        }
        const f = KEYS[e.key] ?? KEYS[e.key.toLowerCase()];
        if (f) { f(); e.preventDefault(); }
    });

    // ── what each new snapshot says: sounds and words ──
    const wriggleCycle = new Map(), tornAt = {};
    let lastAngle = 0;
    function hear(snap, dt) {
        const { held, feel, blocks } = snap;
        for (const e of snap.events) {
            if (e.type === 'plip') JellySound.plip(e.r, e.speed);
            else if (e.type === 'plop') JellySound.plop(e.r, e.speed);
            else if (e.type === 'tick') JellySound.tick(e.speed);
            else if (e.type === 'tear') { tornAt[e.part] = snap.time; JellySound.tear(); say(`The ${info.parts[e.part]} came away. A bud is forming at the stump.`); }
            else if (e.type === 'regrown') { JellySound.regrown(); say(`The ${info.parts[e.part]} grew back in ${(snap.time - (tornAt[e.part] ?? snap.time)).toFixed(1)} s.`); }
        }
        if (dt <= 0) { JellySound.setCreak(0); JellySound.setGrow(-1); JellySound.setWobble(0); JellySound.setSqueak(0); return; }
        if (held) {
            JellySound.squelch(held.block === 0 ? moveSpeed : 0, dt);
            JellySound.setSqueak((held.angle - lastAngle) / dt);
            lastAngle = held.angle;
        } else { JellySound.squelch(0, dt); JellySound.setSqueak(0); }
        moveSpeed *= 0.8;
        JellySound.setWobble(feel.bounce * (held?.block === 0 ? 0.5 : 1));
        const hb = held ? blocks[held.block] : null;
        const pulling = hb && hb.part >= 0 && hb.slot === 0 && hb.grow >= 1;
        JellySound.setCreak(pulling ? Math.max(0.1, hb.strain / info.limits[hb.part]) : 0);
        const growing = blocks.filter((b) => b.part >= 0 && b.slot === 0 && b.grow < 1);
        JellySound.setGrow(growing.length ? Math.max(...growing.map((b) => b.grow)) : -1);
        JellySound.thud(feel.impact);
        const cycle = Math.floor(feel.time * 17 / (2 * Math.PI));
        blocks.forEach((b, i) => {
            if (b.slot === 0 || !b.active || b.twitch <= 0) return;
            if (wriggleCycle.get(i) !== cycle) { wriggleCycle.set(i, cycle); JellySound.wriggle(b.twitch); }
        });
    }

    // ── the loop ──
    // Each frame asks for the next simulated frame and draws the newest one
    // there is; with the simulation in a worker, the main thread only draws.
    let last = performance.now(), statT = 0, frames = 0, mainMs = 0, frameNo = 0, heard = null, pendingDt = 0, fps = 60;
    window.__jelly = { THREE, sim, view, camera, renderer, state, stage, applyLook, applyTable };
    $('status').textContent = `${backend || 'running'} · sim on the ${thread === 'worker' ? 'worker' : 'main thread'}`;
    $('overlay').classList.add('done');
    function frame(now) {
        const t0 = performance.now();
        const realDt = Math.min(0.05, Math.max(0, (now - last) / 1000)); last = Math.max(last, now);
        if (realDt > 0) fps = fps * 0.95 + (1 / realDt) * 0.05;
        stage.frame();
        const dt = state.paused ? 0 : realDt * (state.slow ? 0.25 : 1);
        // time the simulation has not been asked to cover yet (if it is busy)
        pendingDt = Math.min(0.05, pendingDt + dt);
        if (sim.frame({ dt: pendingDt, substeps: Math.max(4, Math.min(16, Math.round(pendingDt * 720))), walls: stage.walls, hover: stage.hoverRay, fps })) {
            const asked = pendingDt;
            pendingDt = 0;
            sim.lastDt = asked;
        }
        const snap = sim.latest;
        if (snap) {
            if (snap !== heard) { hear(snap, sim.lastDt ?? 0); heard = snap; }
            stage.setCursor(snap.hover);
            const held = snap.held, hp = held ? snap.blocks[held.block] : null;
            view.sync(snap, {
                hoverBlock: snap.hover?.kind === 'part' ? snap.hover.block : -1,
                heldBlock: hp && hp.part >= 0 && hp.slot === 0 ? { block: held.block, ratio: held.strain / info.limits[hp.part] } : null,
            });
        }
        renderer.render(view.scene, camera);
        mainMs = mainMs * 0.9 + (performance.now() - t0) * 0.1;
        frames++; statT += realDt; frameNo++;
        if (statT > 0.25 && snap) {
            const m = snap.measure;
            $('m-vol').textContent = (m.volume * 100).toFixed(1);
            $('m-len').textContent = (m.length * 2).toFixed(1);
            $('m-legs').textContent = `${m.attached}/${info.parts.length}`;
            $('m-legs-note').textContent = m.growing !== null ? `${Math.round(m.growing * 100)}% grown` : (m.loose ? `${m.loose} loose` : 'attached');
            $('info').textContent = `${Math.round(frames / statT)} fps · ${snap.solverMode === 'auto' ? 'auto → ' : ''}${snap.onGpu ? 'GPU' : 'CPU'} solver ${snap.solveMs.toFixed(1)} ms on the ${thread === 'worker' ? 'worker' : 'main thread'}${snap.onGpu ? ` (readbacks ${snap.gpuLag} behind)` : ''} · main thread ${mainMs.toFixed(1)} ms · regrown ${snap.regrown}`;
            $('solver-note').textContent = snap.solverNote;
            statT = 0; frames = 0;
        }
        if (reportAt && frameNo === reportAt) {
            beacon({ ok: 1, creature: creatureName, thread, mode: solver, solver: snap?.onGpu ? 'gpu' : 'cpu', frames: frameNo, sim: snap?.solveMs.toFixed(1), main: mainMs.toFixed(1), look: state.look, finish: state.finish, table: state.table, backend });
        }
        requestAnimationFrame(frame);
    }
    requestAnimationFrame(frame);
    $('facts').textContent = `${info.parts.length} parts · ${view.facts} · built in ${Math.round(info.buildMs)} ms`;
}

main().catch((e) => { console.error(e); fail(String(e?.message ?? e)); });
