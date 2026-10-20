/** Orbit sync bridge for parity compare runners (three.js ↔ threers). */

const EPS = 1e-4;

function vec3From(v) {
    return { x: v.x, y: v.y, z: v.z };
}

function vec3Key(v) {
    return `${v.x.toFixed(4)},${v.y.toFixed(4)},${v.z.toFixed(4)}`;
}

export function orbitStateKey(state) {
    return `${vec3Key(state.position)}|${vec3Key(state.target)}`;
}

export function exportOrbitState(ctx) {
    if (!ctx?.cam) return null;
    if (ctx.side === 'threers') {
        if (ctx.orbit?._syncFromWasm) ctx.orbit._syncFromWasm();
        const p = ctx.cam._w.readPosition();
        const t = ctx.cam._w.readTarget();
        return {
            position: { x: p.x, y: p.y, z: p.z },
            target: { x: t.x, y: t.y, z: t.z },
        };
    }
    const cam = ctx.orbit?.object || ctx.cam;
    const target = ctx.orbit?.target;
    if (!cam?.position || !target) return null;
    return {
        position: vec3From(cam.position),
        target: vec3From(target),
    };
}

export function applyOrbitState(ctx, state) {
    if (!ctx?.cam || !state?.position || !state?.target) return false;
    window.__parityOrbitApplying = true;
    try {
        // Both sides through three.js's public camera and OrbitControls API:
        // the shim mirrors the camera's own position and orientation into wasm
        // at each render, so writing the wasm camera directly is overwritten.
        const cam = ctx.orbit?.object || ctx.cam;
        const orbit = ctx.orbit;
        if (!orbit?.target) return false;
        orbit.target.set(state.target.x, state.target.y, state.target.z);
        cam.position.set(state.position.x, state.position.y, state.position.z);
        if (orbit._spherical) {
            const ox = cam.position.x - orbit.target.x;
            const oy = cam.position.y - orbit.target.y;
            const oz = cam.position.z - orbit.target.z;
            const radius = Math.hypot(ox, oy, oz);
            if (radius > 1e-8) {
                orbit._spherical.radius = radius;
                orbit._spherical.theta = Math.atan2(ox, oz);
                orbit._spherical.phi = Math.acos(Math.max(-1, Math.min(1, oy / radius)));
            }
        }
        if (typeof cam.lookAt === 'function') cam.lookAt(orbit.target);
        if (typeof orbit.update === 'function') orbit.update();
        if (typeof ctx.render === 'function') ctx.render();
        return true;
    } finally {
        window.__parityOrbitApplying = false;
    }
}

function flushPendingOrbit(ctx) {
    if (!ctx) return;
    if (window.__parityOrbitPendingSeed) {
        applyOrbitState(ctx, window.__parityOrbitPendingSeed);
        window.__parityOrbitLastKey = orbitStateKey(window.__parityOrbitPendingSeed);
        window.__parityOrbitPendingSeed = null;
    }
    if (window.__parityOrbitPendingSync) {
        const pending = window.__parityOrbitPendingSync;
        window.__parityOrbitPendingSync = null;
        if (pending.from !== ctx.side && applyOrbitState(ctx, pending.state)) {
            window.__parityOrbitLastKey = orbitStateKey(pending.state);
        }
    }
}

export function primeOrbitCtx(ctx) {
    window.__parityOrbitCtx = ctx;
    flushPendingOrbit(ctx);
}

function statesEqual(a, b) {
    if (!a || !b) return false;
    const keys = ['position', 'target'];
    for (const k of keys) {
        for (const axis of ['x', 'y', 'z']) {
            if (Math.abs(a[k][axis] - b[k][axis]) > EPS) return false;
        }
    }
    return true;
}

export function installOrbitSyncBridge() {
    if (window.__parityOrbitBridgeInstalled) return;
    window.__parityOrbitBridgeInstalled = true;
    window.__parityExportOrbitState = exportOrbitState;
    window.__parityApplyOrbitState = applyOrbitState;
    window.__parityOrbitStateKey = orbitStateKey;
    window.__parityOrbitSyncEnabled = false;
    window.__parityOrbitLastKey = '';
    window.__parityOrbitInteracting = false;
    window.__parityOrbitPendingSeed = null;
    window.__parityOrbitPendingSync = null;
    window.__parityPrimeOrbitCtx = primeOrbitCtx;

    let wheelTimer = 0;
    const markInteracting = () => { window.__parityOrbitInteracting = true; };
    const clearInteracting = () => { window.__parityOrbitInteracting = false; };
    document.addEventListener('pointerdown', markInteracting);
    document.addEventListener('pointerup', clearInteracting);
    document.addEventListener('pointercancel', clearInteracting);
    document.addEventListener('wheel', () => {
        markInteracting();
        clearTimeout(wheelTimer);
        wheelTimer = setTimeout(clearInteracting, 120);
    }, { passive: true });

    const relayOrbitState = () => {
        if (!window.__parityOrbitSyncEnabled || window.__parityOrbitApplying) return;
        const ctx = window.__parityOrbitCtx;
        if (!ctx) return;
        const state = exportOrbitState(ctx);
        if (!state) return;
        const key = orbitStateKey(state);
        if (key === window.__parityOrbitLastKey) return;
        window.__parityOrbitLastKey = key;
        window.parent.postMessage({ type: 'parity-orbit-change', source: ctx.side, state }, '*');
    };

    window.__parityOrbitSyncHook = () => {
        try {
            relayOrbitState();
        } catch (e) {
            console.warn('parity orbit sync hook', e);
        }
    };

    // Final camera pose after drag/wheel can differ from the last move event.
    const flushOrbitRelay = () => {
        try {
            relayOrbitState();
        } catch (e) {
            console.warn('parity orbit sync flush', e);
        }
    };
    document.addEventListener('pointerup', flushOrbitRelay);
    document.addEventListener('pointercancel', flushOrbitRelay);

    window.addEventListener('message', (event) => {
        const data = event.data;
        if (!data || typeof data !== 'object') return;
        const ctx = window.__parityOrbitCtx;

        if (data.type === 'parity-orbit-sync-enabled') {
            window.__parityOrbitSyncEnabled = !!data.enabled;
            if (data.enabled && data.seedState) {
                if (ctx) {
                    applyOrbitState(ctx, data.seedState);
                    window.__parityOrbitLastKey = orbitStateKey(data.seedState);
                } else {
                    window.__parityOrbitPendingSeed = data.seedState;
                }
            }
            return;
        }

        if (data.type === 'parity-orbit-sync' && ctx) {
            if (data.from === ctx.side) return;
            if (applyOrbitState(ctx, data.state)) {
                window.__parityOrbitLastKey = orbitStateKey(data.state);
            }
            return;
        }

        if (data.type === 'parity-orbit-sync') {
            window.__parityOrbitPendingSync = { from: data.from, state: data.state };
            return;
        }

        if (data.type === 'parity-orbit-get-state' && ctx) {
            const state = exportOrbitState(ctx);
            if (state) {
                window.parent.postMessage({
                    type: 'parity-orbit-state',
                    source: ctx.side,
                    state,
                }, '*');
            }
        }
    });
}
