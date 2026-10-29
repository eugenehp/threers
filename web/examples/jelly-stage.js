// The stage the creature lies on: the camera (yaw, pitch and zoom round a
// target), the walls where the screen ends, and the hands — pointer, touch
// and wheel — turned into grabs, twists, throws and orbits, sent to the
// simulation (jelly-client.js), which may be in a worker.
//
//   createStage({ canvas, renderer, camera, sim, framing, on })
//
// `on` hears what the hands do, for sound and words: grab(kind, onBody),
// move(kind, point), release(kind, speed), hover(block).

const M4 = {
    mul(a, b) {
        const o = new Float32Array(16);
        for (let c = 0; c < 4; c++) for (let r = 0; r < 4; r++) o[4 * c + r] = a[r] * b[4 * c] + a[4 + r] * b[4 * c + 1] + a[8 + r] * b[4 * c + 2] + a[12 + r] * b[4 * c + 3];
        return o;
    },
    perspective(fovy, aspect, n, f) { const t = 1 / Math.tan(fovy / 2); return new Float32Array([t / aspect, 0, 0, 0, 0, t, 0, 0, 0, 0, f / (n - f), -1, 0, 0, n * f / (n - f), 0]); },
    lookAt(e, c, up) {
        let z = [e[0] - c[0], e[1] - c[1], e[2] - c[2]]; const zl = Math.hypot(...z); z = z.map((v) => v / zl);
        let x = [up[1] * z[2] - up[2] * z[1], up[2] * z[0] - up[0] * z[2], up[0] * z[1] - up[1] * z[0]]; const xl = Math.hypot(...x); x = x.map((v) => v / xl);
        const y = [z[1] * x[2] - z[2] * x[1], z[2] * x[0] - z[0] * x[2], z[0] * x[1] - z[1] * x[0]], d = (a) => -(a[0] * e[0] + a[1] * e[1] + a[2] * e[2]);
        return new Float32Array([x[0], y[0], z[0], 0, x[1], y[1], z[1], 0, x[2], y[2], z[2], 0, d(x), d(y), d(z), 1]);
    },
    invert(m) {
        const [a00, a01, a02, a03, a10, a11, a12, a13, a20, a21, a22, a23, a30, a31, a32, a33] = m, o = new Float32Array(16);
        const b00 = a00 * a11 - a01 * a10, b01 = a00 * a12 - a02 * a10, b02 = a00 * a13 - a03 * a10, b03 = a01 * a12 - a02 * a11, b04 = a01 * a13 - a03 * a11, b05 = a02 * a13 - a03 * a12;
        const b06 = a20 * a31 - a21 * a30, b07 = a20 * a32 - a22 * a30, b08 = a20 * a33 - a23 * a30, b09 = a21 * a32 - a22 * a31, b10 = a21 * a33 - a23 * a31, b11 = a22 * a33 - a23 * a32;
        const det = 1 / (b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06);
        o[0] = (a11 * b11 - a12 * b10 + a13 * b09) * det; o[1] = (a02 * b10 - a01 * b11 - a03 * b09) * det; o[2] = (a31 * b05 - a32 * b04 + a33 * b03) * det; o[3] = (a22 * b04 - a21 * b05 - a23 * b03) * det;
        o[4] = (a12 * b08 - a10 * b11 - a13 * b07) * det; o[5] = (a00 * b11 - a02 * b08 + a03 * b07) * det; o[6] = (a32 * b02 - a30 * b05 - a33 * b01) * det; o[7] = (a20 * b05 - a22 * b02 + a23 * b01) * det;
        o[8] = (a10 * b10 - a11 * b08 + a13 * b06) * det; o[9] = (a01 * b08 - a00 * b10 - a03 * b06) * det; o[10] = (a30 * b04 - a31 * b02 + a33 * b00) * det; o[11] = (a21 * b02 - a20 * b04 - a23 * b00) * det;
        o[12] = (a11 * b07 - a10 * b09 - a12 * b06) * det; o[13] = (a00 * b09 - a01 * b07 + a02 * b06) * det; o[14] = (a31 * b01 - a30 * b03 - a32 * b00) * det; o[15] = (a20 * b03 - a21 * b01 + a22 * b00) * det;
        return o;
    },
    apply(m, p) {
        const [x, y, z] = p, w = p[3] ?? 1;
        return [m[0] * x + m[4] * y + m[8] * z + m[12] * w, m[1] * x + m[5] * y + m[9] * z + m[13] * w, m[2] * x + m[6] * y + m[10] * z + m[14] * w, m[3] * x + m[7] * y + m[11] * z + m[15] * w];
    },
};

// the visible stage as four inward-facing planes, so the jelly bumps off the
// edges of the screen instead of leaving it
function stageWalls(m, left, right, bottom, top, margin) {
    const row = (r) => [m[r], m[4 + r], m[8 + r], m[12 + r]];
    const r0 = row(0), r1 = row(1), r3 = row(3);
    const planes = [r0.map((v, i) => v - left * r3[i]), r0.map((v, i) => right * r3[i] - v), r1.map((v, i) => v - bottom * r3[i]), r1.map((v, i) => top * r3[i] - v)];
    const out = new Float32Array(16);
    planes.forEach((p, k) => { const l = Math.hypot(p[0], p[1], p[2]); out.set([p[0] / l, p[1] / l, p[2] / l, p[3] / l - margin], 4 * k); });
    return out;
}
function keepOnStage(p, W) {
    for (let k = 0; k < W.length; k += 4) {
        const s = W[k] * p[0] + W[k + 1] * p[1] + W[k + 2] * p[2] + W[k + 3];
        if (s < 0) for (let c = 0; c < 3; c++) p[c] -= W[k + c] * s;
    }
    return p;
}

export function createStage({ canvas, renderer, camera, sim, framing, on = {} }) {
    const fov = 30 * Math.PI / 180;
    const home = { yaw: framing.yaw, pitch: framing.pitch, zoom: 1 };
    const cam = { ...home, target: framing.target.slice() };
    let walls = new Float32Array(16), invViewProj = null, forward = [0, 0, -1];

    function frame() {
        const dpr = Math.min(devicePixelRatio || 1, 2);
        const W = canvas.clientWidth, H = canvas.clientHeight;
        const w = Math.max(1, Math.round(W * dpr)), h = Math.max(1, Math.round(H * dpr));
        if (canvas.width !== w || canvas.height !== h) renderer.setSize(w, h, false);
        const aspect = W / H;
        const dist = Math.max(framing.fit / (2 * Math.tan(fov / 2) * aspect), 1.7 / (2 * Math.tan(fov / 2)), 3.6) * cam.zoom;
        const eye = [cam.target[0] + dist * Math.sin(cam.yaw) * Math.cos(cam.pitch), cam.target[1] + dist * Math.sin(cam.pitch), cam.target[2] + dist * Math.cos(cam.yaw) * Math.cos(cam.pitch)];
        camera.aspect = aspect;
        camera.position.set(...eye);
        camera.lookAt(...cam.target);
        camera.updateProjectionMatrix();
        const view = M4.lookAt(eye, cam.target, [0, 1, 0]);
        const viewProj = M4.mul(M4.perspective(fov, aspect, 0.1, 60), view);
        walls = stageWalls(viewProj, -0.97, 0.97, -0.96, 0.95, 0.03);
        invViewProj = M4.invert(viewProj);
        forward = [-view[2], -view[6], -view[10]];
    }
    function rayFrom(ev) {
        const r = canvas.getBoundingClientRect();
        const nx = (ev.clientX - r.left) / r.width * 2 - 1, ny = 1 - (ev.clientY - r.top) / r.height * 2;
        const a = M4.apply(invViewProj, [nx, ny, 0, 1]), b = M4.apply(invViewProj, [nx, ny, 1, 1]);
        const o = [a[0] / a[3], a[1] / a[3], a[2] / a[3]], f = [b[0] / b[3], b[1] / b[3], b[2] / b[3]];
        const d = [f[0] - o[0], f[1] - o[1], f[2] - o[2]], l = Math.hypot(...d);
        return { o, d: d.map((v) => v / l) };
    }
    // where the pointer's ray meets the plane through `p` facing the camera
    function onPlane(ray, p) {
        const n = forward;
        const t = ((p[0] - ray.o[0]) * n[0] + (p[1] - ray.o[1]) * n[1] + (p[2] - ray.o[2]) * n[2]) / (ray.d[0] * n[0] + ray.d[1] * n[1] + ray.d[2] * n[2]);
        return [ray.o[0] + ray.d[0] * t, ray.o[1] + ray.d[1] * t, ray.o[2] + ray.d[2] * t];
    }

    // ── hands ──
    // A press asks the simulation what is under it (it may be in a worker, so
    // the answer comes a moment later); until then the drag is pending, and a
    // release before it lands lets go of whatever it turns out to be.
    const pointers = new Map();
    let drag = null, hoverRay = null, twist = 0;
    canvas.addEventListener('pointerdown', (ev) => {
        try { canvas.setPointerCapture(ev.pointerId); } catch (_) {}
        pointers.set(ev.pointerId, { x: ev.clientX, y: ev.clientY });
        // a second finger on a held part twists it
        if (pointers.size === 2 && drag?.kind === 'part') {
            const [a, b] = [...pointers.values()];
            drag.twist0 = Math.atan2(b.y - a.y, b.x - a.x);
            drag.angle0 = twist;
            return;
        }
        if (pointers.size > 1) return;
        const ray = rayFrom(ev);
        const d = drag = { kind: 'pending', id: ev.pointerId, x: ev.clientX, y: ev.clientY, yaw: cam.yaw, pitch: cam.pitch, last: null };
        hoverRay = null;
        canvas.classList.add('dragging');
        sim.pick(ray.o, ray.d, forward).then((hit) => {
            if (hit?.kind === 'part' || hit?.kind === 'drop') {
                twist = 0;
                Object.assign(d, { kind: hit.kind, plane: hit.point });
                on.grab?.(hit.kind, hit.kind === 'part' && hit.block === 0);
                if (d.released) letGo(d);
                else if (d.last) moveTo(d, d.last);
            } else {
                d.kind = 'orbit';
                on.grab?.('orbit', false);
            }
        });
    });
    function moveTo(d, ev) {
        const p = keepOnStage(onPlane(rayFrom(ev), d.plane), walls);
        p[1] = Math.max(p[1], d.kind === 'drop' ? 0.03 : -0.05);
        sim.send({ type: d.kind === 'part' ? 'move' : 'moveDrop', p });
        on.move?.(d.kind, p);
    }
    function letGo(d) {
        if (d.kind === 'part') { sim.send({ type: 'release' }); on.release?.('part', 0); }
        if (d.kind === 'drop') sim.releaseDrop().then((speed) => on.release?.('drop', speed));
    }
    canvas.addEventListener('pointermove', (ev) => {
        if (pointers.has(ev.pointerId)) pointers.set(ev.pointerId, { x: ev.clientX, y: ev.clientY });
        if (!drag) {
            if (ev.pointerType === 'mouse') hoverRay = rayFrom(ev);
            return;
        }
        if (drag.kind === 'part' && pointers.size === 2 && drag.twist0 !== undefined) {
            const [a, b] = [...pointers.values()];
            twist = drag.angle0 - (Math.atan2(b.y - a.y, b.x - a.x) - drag.twist0);
            sim.send({ type: 'twist', angle: twist });
            return;
        }
        if (ev.pointerId !== drag.id) return;
        // orbit from the press, even while the pick is on its way
        if (drag.kind === 'orbit' || drag.kind === 'pending') {
            if (drag.kind === 'pending') { drag.last = { clientX: ev.clientX, clientY: ev.clientY }; return; }
            cam.yaw = drag.yaw - (ev.clientX - drag.x) * 0.005;
            cam.pitch = Math.min(1.3, Math.max(0.12, drag.pitch + (ev.clientY - drag.y) * 0.004));
            return;
        }
        moveTo(drag, ev);
    });
    const release = (ev) => {
        pointers.delete(ev.pointerId);
        if (!drag) return;
        if (drag.kind === 'part' && pointers.size === 1 && drag.twist0 !== undefined) { drag.twist0 = undefined; return; }
        if (pointers.size > 0) return;
        if (drag.kind === 'pending') drag.released = true;
        else letGo(drag);
        drag = null;
        canvas.classList.remove('dragging');
    };
    canvas.addEventListener('pointerup', release);
    canvas.addEventListener('pointercancel', release);
    canvas.addEventListener('pointerleave', () => { if (!drag) hoverRay = null; });
    canvas.addEventListener('wheel', (ev) => {
        ev.preventDefault();
        // while holding a part, the wheel twists it; otherwise it zooms
        if (drag?.kind === 'part') { twist += ev.deltaY * 0.004; sim.send({ type: 'twist', angle: twist }); }
        else cam.zoom = Math.min(1.5, Math.max(0.55, cam.zoom * Math.exp(ev.deltaY * 0.001)));
    }, { passive: false });
    // double-click the table: back to the home view
    canvas.addEventListener('dblclick', () => { if (!drag || drag.kind === 'orbit') Object.assign(cam, home); });

    return {
        frame,
        get walls() { return walls; },
        /// the pointer's ray, for the simulation to test for the highlight
        get hoverRay() { return drag ? null : hoverRay; },
        setCursor(hit) { canvas.style.cursor = hit ? 'grab' : 'default'; },
        get dragging() { return drag?.kind ?? null; },
        home() { Object.assign(cam, home); },
        orbit(dYaw, dPitch) { cam.yaw += dYaw; cam.pitch = Math.min(1.3, Math.max(0.12, cam.pitch + dPitch)); },
        zoom(f) { cam.zoom = Math.min(1.5, Math.max(0.55, cam.zoom * f)); },
    };
}
