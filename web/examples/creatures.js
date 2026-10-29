// The gummy creatures: each one a set of THREE.SDF fields and point lists for
// creature-kit.js, plus how it moves, how it is coloured, and how it is framed.
//
//   jellyfish   a pulsing bell; eight thick tentacles tear, a fringe flutters
//   gecko       the tail drops and thrashes (as a real gecko's does); legs too
//   starfish    five arms round a disc, any of which comes away and regrows
//   octopus     eight curling arms that tear; the mantle breathes
//   sea slug    rhinophores and oral tentacles tear; rows of cerata flutter
//   gummy worm  pull it in two; a squeeze runs down it as it crawls
//   gummy bear  ears, arms and legs
//   salamander  legs and tail tear and grow back, as a salamander's limbs do
//
// One unit is 2 cm. Everything sits on the table, y >= 0. `build()` returns
// `{ body: THREE.SoftBody, insides }`; `setup(body)` sets its muscles.
import { SDF, Quaternion, Vector3 } from '../threejs-shim.js';
import { buildCreature, smoothstep, mix } from './creature-kit.js';

// ── shared colour helpers (rest space, so patterns ride along) ──────────────
const fract = (x) => x - Math.floor(x);
const hash13 = (x, y, z) => fract(Math.sin(x * 127.1 + y * 311.7 + z * 74.7) * 43758.5453);
export function noise3(x, y, z) {
    const ix = Math.floor(x), iy = Math.floor(y), iz = Math.floor(z);
    const fx = x - ix, fy = y - iy, fz = z - iz;
    const sx = fx * fx * (3 - 2 * fx), sy = fy * fy * (3 - 2 * fy), sz = fz * fz * (3 - 2 * fz);
    const h = (a, b, c) => hash13(ix + a, iy + b, iz + c);
    return mix(mix(mix(h(0, 0, 0), h(1, 0, 0), sx), mix(h(0, 1, 0), h(1, 1, 0), sx), sy),
               mix(mix(h(0, 0, 1), h(1, 0, 1), sx), mix(h(0, 1, 1), h(1, 1, 1), sx), sy), sz);
}
// a hue for u in [0, 1), softened so the gel still reads as candy
export function rainbowTint(u) {
    return [0, 2 / 3, 1 / 3].map((o) => {
        const c = Math.min(1, Math.max(0, Math.abs(fract(u + o) * 6 - 3) - 1));
        return 0.97 + (c - 0.97) * 0.66;
    });
}
const spots = (c, x, y, z, scale, dark) => {
    const sp = smoothstep(0.6, 0.7, noise3(x * scale, y * scale, z * scale));
    return c.map((q) => q * (1 - dark * sp));
};
const WHITE = [1, 1, 1];

const v = (x, y, z) => [x, y, z];
// a chain of tapered capsules through points, radii along it, blended
function tube(pts, radii, k = 0.015) {
    const parts = [];
    for (let i = 0; i < pts.length - 1; i++) parts.push(SDF.capsule(pts[i], pts[i + 1], radii[i], radii[i + 1]));
    return SDF.unionAll(parts, k);
}
const sphereAt = (r, p) => SDF.sphere(r).translate(...p);
const ellipsoidAt = (r, p) => SDF.ellipsoid(...r).translate(...p);
// n + 1 points along a straight line, for a chain
const line = (a, d, len, n) => Array.from({ length: n + 1 }, (_, k) => [a[0] + d[0] * len * k / n, a[1] + d[1] * len * k / n, a[2] + d[2] * len * k / n]);
const norm = (d) => { const l = Math.hypot(...d); return d.map((x) => x / l); };

// ── jellyfish ───────────────────────────────────────────────────────────────
const JR = 0.62, JH = 0.46, JY = 0.04;
function jellyfish() {
    // a half-ellipsoid, the hollow underneath cut out (lip rounded), and a
    // notch between each pair of the sixteen lappets round the margin
    let bell = ellipsoidAt([JR, JH, JR], [0, JY, 0]).intersect(SDF.plane(0, -1, 0, JY))
        .subtract(ellipsoidAt([0.48, 0.25, 0.48], [0, JY, 0]), 0.03);
    for (let k = 0; k < 16; k++) {
        const a = (k + 0.5) / 16 * Math.PI * 2;
        bell = bell.subtract(sphereAt(0.055, [(JR + 0.03) * Math.cos(a), JY, (JR + 0.03) * Math.sin(a)]), 0.02);
    }
    const ORD = ['first', 'second', 'third', 'fourth', 'fifth', 'sixth', 'seventh', 'eighth'];
    const parts = ORD.map((name, i) => {
        const a = (i + 0.5) / 8 * Math.PI * 2, c = Math.cos(a), s = Math.sin(a), curl = i % 2 ? 1 : -1;
        const len = 0.62 + 0.08 * ((i * 5) % 3), pts = [];
        for (let k = 0; k <= 7; k++) {
            const t = k / 7, out = JR - 0.07 + len * t, side = curl * 0.2 * Math.sin(t * 2.4) * t;
            pts.push([c * out - s * side, mix(0.075, 0.026, smoothstep(0, 0.35, t)), s * out + c * side]);
        }
        return { name: `${name} tentacle`, field: tube(pts, pts.map((_, k) => mix(0.042, 0.013, k / 7)), 0.012) };
    });
    const chains = Array.from({ length: 16 }, (_, j) => {
        const a = (j + 0.5 * ((j % 2) ? 0.6 : 1.4)) / 16 * Math.PI * 2 + Math.PI / 8;
        const c = Math.cos(a), s = Math.sin(a), curl = j % 3 - 1, len = 0.42 + 0.12 * ((j * 7) % 4) / 3;
        const points = Array.from({ length: 10 }, (_, k) => {
            const t = k / 9, out = JR - 0.02 + len * t, side = 0.12 * curl * t * t;
            return [c * out - s * side, mix(0.06, 0.008, smoothstep(0, 0.3, t)), s * out + c * side];
        });
        return { points, r0: 0.009, r1: 0.0035, stiff: 0.16 };
    });
    const gonads = SDF.unionAll([0, 1, 2, 3].map((k) => SDF.torus(0.085, 0.02).translate(0.15 * Math.cos(k * Math.PI / 2), 0.34, 0.15 * Math.sin(k * Math.PI / 2))));
    const top = JY + JH - 0.08;
    return buildCreature({
        body: { field: bell, cell: 0.016, h: 0.07 }, parts, chains,
        insides: [{ field: gonads, kind: 'milk', cell: 0.007 }],
        head: [v(0, top, 0), v(0.25, top - 0.06, 0), v(0, top - 0.06, 0.25)],
        span: [v(JR, JY + 0.03, 0), v(-JR, JY + 0.03, 0)],
    });
}
// The bell beats: edges running round it near the margin shorten together.
function jellyfishMuscles(body) {
    body.muscles = {
        weight(m, d) {
            const r = Math.hypot(m[0], m[2]) || 1;
            const tang = Math.abs(d[0] * (-m[2] / r) + d[2] * (m[0] / r));
            return tang * smoothstep(0.3, 0.55, r / JR) * smoothstep(0.32, 0.06, m[1]);
        },
        pulse: { a: 0.09, calm: 1, omegaA: 0.75 * Math.PI * 2, powerA: 3, b: 0.24, omegaB: 2.2 * Math.PI * 2, powerB: 3 },
    };
}

// ── gecko ───────────────────────────────────────────────────────────────────
function gecko() {
    const body = SDF.unionAll([
        ellipsoidAt([0.5, 0.13, 0.19], [0, 0.15, 0]),
        ellipsoidAt([0.2, 0.11, 0.15], [0.6, 0.16, 0]),
        ellipsoidAt([0.1, 0.07, 0.09], [0.77, 0.14, 0]),
    ], 0.1).union(sphereAt(0.05, [0.66, 0.23, 0.11]), 0.03).union(sphereAt(0.05, [0.66, 0.23, -0.11]), 0.03);
    // a fat tail, the way a leopard gecko stores fat, curling off to one side
    const tailPts = [v(-0.36, 0.14, 0), v(-0.75, 0.11, 0.06), v(-1.1, 0.075, 0.18), v(-1.4, 0.05, 0.12), v(-1.6, 0.035, -0.02)];
    const parts = [{ name: 'tail', field: tube(tailPts, [0.1, 0.085, 0.06, 0.04, 0.022], 0.03), h: 0.045 }];
    for (const [front, side, name] of [[1, 1, 'left foreleg'], [1, -1, 'right foreleg'], [0, 1, 'left hind leg'], [0, -1, 'right hind leg']]) {
        const S = front ? v(0.28, 0.13, 0.12 * side) : v(-0.3, 0.13, 0.13 * side);
        const E = front ? v(0.36, 0.09, 0.3 * side) : v(-0.38, 0.09, 0.32 * side);
        const W = front ? v(0.46, 0.035, 0.36 * side) : v(-0.47, 0.035, 0.4 * side);
        let leg = tube([S, E, W], [0.055, 0.042, 0.032], 0.02);
        // five splayed toes ending in sticky pads
        const base = norm([W[0] - E[0], 0, W[2] - E[2]]);
        for (let k = 0; k < 5; k++) {
            const a = mix(-0.9, 0.9, k / 4), c = Math.cos(a), s = Math.sin(a);
            const d = [base[0] * c - base[2] * s, 0, base[0] * s + base[2] * c];
            const tip = [W[0] + d[0] * 0.075, 0.024, W[2] + d[2] * 0.075];
            leg = leg.union(SDF.capsule(W, tip, 0.016, 0.011), 0.012).union(sphereAt(0.021, tip), 0.008);
        }
        parts.push({ name, field: leg, h: 0.04 });
    }
    // the crest: a row of soft spines either side of the spine
    const chains = [];
    const top = (x) => x > 0.42 ? 0.16 + 0.11 * Math.sqrt(Math.max(0, 1 - ((x - 0.6) / 0.2) ** 2)) : 0.15 + 0.13 * Math.sqrt(Math.max(0, 1 - (x / 0.5) ** 2));
    for (let i = 0; i < 12; i++) {
        const x = mix(0.66, -0.36, i / 11);
        for (const s of [1, -1]) chains.push({ points: line([x, top(x) - 0.015, 0.05 * s], norm([-0.3, 1, 0.45 * s]), 0.06 + 0.02 * Math.sin(i), 3), r0: 0.013, r1: 0.004, stiff: 0.3 });
    }
    return buildCreature({
        body: { field: body, cell: 0.014, h: 0.06 }, parts, chains,
        insides: [
            { field: body.round(-0.035).intersect(SDF.plane(0, 1, 0, -0.1)), kind: 'milk', cell: 0.012 },
            { field: sphereAt(0.04, [0.675, 0.25, 0.135]), kind: 'eye', cell: 0.006 },
            { field: sphereAt(0.04, [0.675, 0.25, -0.135]), kind: 'eye', cell: 0.006 },
        ],
        head: [v(0.55, 0.2, 0), v(0.75, 0.18, 0), v(0.55, 0.2, 0.12)],
        span: [v(0.86, 0.14, 0), v(-0.48, 0.15, 0)],
    });
}

// ── salamander ──────────────────────────────────────────────────────────────
function salamander() {
    // a slender body, a broad flat head with bulging eyes and the two glands
    // behind them a fire salamander has, smooth all over
    const body = SDF.unionAll([
        ellipsoidAt([0.48, 0.11, 0.13], [0, 0.13, 0]),
        ellipsoidAt([0.17, 0.085, 0.15], [0.57, 0.13, 0]),
        ellipsoidAt([0.09, 0.06, 0.1], [0.71, 0.11, 0]),
    ], 0.1).union(sphereAt(0.045, [0.64, 0.19, 0.1]), 0.03).union(sphereAt(0.045, [0.64, 0.19, -0.1]), 0.03)
        .union(ellipsoidAt([0.07, 0.035, 0.04], [0.47, 0.2, 0.08]), 0.04).union(ellipsoidAt([0.07, 0.035, 0.04], [0.47, 0.2, -0.08]), 0.04);
    const tailPts = [v(-0.36, 0.13, 0), v(-0.72, 0.1, -0.05), v(-1.05, 0.07, 0.05), v(-1.32, 0.05, 0.12), v(-1.55, 0.035, 0.06)];
    const parts = [{ name: 'tail', field: tube(tailPts, [0.09, 0.07, 0.05, 0.033, 0.016], 0.03), h: 0.045 }];
    // short sprawling legs: four blunt toes in front, five behind
    for (const [front, side, name] of [[1, 1, 'left foreleg'], [1, -1, 'right foreleg'], [0, 1, 'left hind leg'], [0, -1, 'right hind leg']]) {
        const S = front ? v(0.3, 0.12, 0.1 * side) : v(-0.3, 0.12, 0.1 * side);
        const E = front ? v(0.36, 0.08, 0.26 * side) : v(-0.36, 0.08, 0.27 * side);
        const W = front ? v(0.44, 0.03, 0.31 * side) : v(-0.43, 0.03, 0.33 * side);
        let leg = tube([S, E, W], [0.05, 0.038, 0.03], 0.02);
        const toes = front ? 4 : 5, spread = front ? 0.7 : 0.9;
        const base = norm([W[0] - E[0], 0, W[2] - E[2]]);
        for (let k = 0; k < toes; k++) {
            const a = mix(-spread, spread, k / (toes - 1)), c = Math.cos(a), s = Math.sin(a);
            const d = [base[0] * c - base[2] * s, 0, base[0] * s + base[2] * c];
            const tip = [W[0] + d[0] * 0.06, 0.02, W[2] + d[2] * 0.06];
            leg = leg.union(SDF.capsule(W, tip, 0.014, 0.011), 0.012);
        }
        parts.push({ name, field: leg, h: 0.04 });
    }
    return buildCreature({
        body: { field: body, cell: 0.014, h: 0.06 }, parts,
        insides: [
            { field: body.round(-0.035).intersect(SDF.plane(0, 1, 0, -0.09)), kind: 'milk', cell: 0.012 },
            { field: sphereAt(0.036, [0.655, 0.205, 0.12]), kind: 'eye', cell: 0.006 },
            { field: sphereAt(0.036, [0.655, 0.205, -0.12]), kind: 'eye', cell: 0.006 },
        ],
        head: [v(0.55, 0.18, 0), v(0.75, 0.15, 0), v(0.55, 0.18, 0.12)],
        span: [v(0.81, 0.11, 0), v(-0.42, 0.13, 0)],
    });
}

// ── starfish ────────────────────────────────────────────────────────────────
function starfish() {
    const disc = ellipsoidAt([0.27, 0.1, 0.27], [0, 0.1, 0]);
    const parts = ['first', 'second', 'third', 'fourth', 'fifth'].map((name, i) => {
        const a = i / 5 * Math.PI * 2 + Math.PI / 2, c = Math.cos(a), s = Math.sin(a);
        const R = [0.095, 0.08, 0.06, 0.04, 0.022], P = [0.08, 0.32, 0.55, 0.75, 0.9];
        // each arm a little off straight, its tip lifted
        const pts = P.map((r, k) => {
            const bend = 0.06 * Math.sin(k * 1.3 + i) * (k / 4);
            return [c * r - s * bend, R[k] + 0.004 + (k === 4 ? 0.03 : 0), s * r + c * bend];
        });
        return { name: `${name} arm`, field: tube(pts, R, 0.03), h: 0.04 };
    });
    return buildCreature({
        body: { field: disc, cell: 0.012, h: 0.05 }, parts, chains: [],
        insides: [
            { field: SDF.torus(0.12, 0.022).translate(0, 0.1, 0), kind: 'milk', cell: 0.007 },
            { field: sphereAt(0.035, [0.07, 0.13, 0.05]), kind: 'milk', cell: 0.006 },
        ],
        head: [v(0, 0.16, 0), v(0.15, 0.13, 0), v(0, 0.13, 0.15)],
        span: [v(0.25, 0.1, 0), v(-0.25, 0.1, 0)],
    });
}

// ── octopus ─────────────────────────────────────────────────────────────────
function octopus() {
    const body = SDF.unionAll([
        ellipsoidAt([0.27, 0.2, 0.26], [0, 0.2, 0]),
        ellipsoidAt([0.25, 0.26, 0.23], [-0.1, 0.42, 0]),
    ], 0.12).union(sphereAt(0.07, [0.17, 0.3, 0.16]), 0.04).union(sphereAt(0.07, [0.17, 0.3, -0.16]), 0.04);
    const ORD = ['first', 'second', 'third', 'fourth', 'fifth', 'sixth', 'seventh', 'eighth'];
    const parts = ORD.map((name, i) => {
        const a = (i + 0.5) / 8 * Math.PI * 2, curl = (i % 2 ? 1 : -1) * (0.25 + 0.1 * (i % 3));
        const pts = [], R = [];
        for (let k = 0; k <= 9; k++) {
            const t = k / 9, out = 0.13 + 0.85 * t, ang = a + curl * t * t, r = mix(0.075, 0.014, t);
            // the tip curls a little
            const tip = smoothstep(0.7, 1, t) * 0.1 * Math.sign(curl);
            pts.push([Math.cos(ang) * out - Math.sin(ang) * tip, k === 0 ? 0.12 : r + 0.004, Math.sin(ang) * out + Math.cos(ang) * tip]);
            R.push(r);
        }
        return { name: `${name} arm`, field: tube(pts, R, 0.02), h: 0.038 };
    });
    return buildCreature({
        body: { field: body, cell: 0.015, h: 0.065 }, parts, chains: [],
        insides: [
            { field: ellipsoidAt([0.08, 0.1, 0.07], [-0.12, 0.42, 0]), kind: 'milk', cell: 0.01 },
            { field: sphereAt(0.05, [0.215, 0.32, 0.2]), kind: 'eye', cell: 0.007 },
            { field: sphereAt(0.05, [0.215, 0.32, -0.2]), kind: 'eye', cell: 0.007 },
        ],
        head: [v(0, 0.35, 0), v(0.2, 0.3, 0), v(0, 0.3, 0.2)],
        span: [v(0.26, 0.2, 0), v(-0.34, 0.42, 0)],
    });
}
// The mantle breathes; startled, it pumps.
function octopusMuscles(body) {
    body.muscles = { weight: (m) => smoothstep(0.3, 0.5, m[1]), pulse: { a: 0.05, omegaA: 2.2, b: 0.2, omegaB: 9, powerB: 1 } };
}

// ── sea slug (a nudibranch) ─────────────────────────────────────────────────
function seaSlug() {
    const body = ellipsoidAt([0.62, 0.075, 0.2], [0, 0.08, 0]).union(ellipsoidAt([0.5, 0.12, 0.14], [-0.02, 0.14, 0]), 0.08);
    const parts = [];
    for (const [s, side] of [[1, 'left'], [-1, 'right']]) {
        parts.push({ name: `${side} rhinophore`, field: SDF.capsule([0.4, 0.18, 0.06 * s], [0.5, 0.42, 0.1 * s], 0.036, 0.018).union(sphereAt(0.026, [0.5, 0.42, 0.1 * s]), 0.01), h: 0.032 });
        parts.push({ name: `${side} oral tentacle`, field: SDF.capsule([0.54, 0.06, 0.07 * s], [0.8, 0.03, 0.22 * s], 0.03, 0.013), h: 0.032 });
    }
    // the cerata: two rows of clubs along the back
    const chains = [];
    for (let i = 0; i < 11; i++) {
        const x = mix(0.3, -0.48, i / 10), top = 0.14 + 0.12 * Math.sqrt(Math.max(0, 1 - ((x + 0.02) / 0.5) ** 2)) - 0.03;
        for (const s of [1, -1]) chains.push({ points: line([x, top, 0.08 * s], norm([-0.2, 0.85, 0.5 * s]), 0.17 + 0.04 * Math.cos(i * 1.7), 4), r0: 0.032, r1: 0.016, stiff: 0.24 });
    }
    return buildCreature({
        body: { field: body, cell: 0.012, h: 0.05 }, parts, chains,
        insides: [{ field: ellipsoidAt([0.4, 0.05, 0.08], [0, 0.12, 0]), kind: 'milk', cell: 0.01 }],
        head: [v(0.3, 0.2, 0), v(0.5, 0.15, 0), v(0.3, 0.2, 0.1)],
        span: [v(0.62, 0.08, 0), v(-0.62, 0.08, 0)],
    });
}

// ── gummy worm ──────────────────────────────────────────────────────────────
// an S along x; s in [0, 1] from tail to head
const wormAt = (s) => [mix(-1.15, 1.15, s), 0, 0.33 * Math.sin(s * Math.PI * 2.2)];
const wormR = (s) => 0.052 + 0.024 * Math.sin(Math.PI * Math.min(1, Math.max(0, s)));
const WORM = Array.from({ length: 241 }, (_, i) => wormAt(i / 240));
function wormS(x, z) {
    let best = 0, bd = Infinity;
    for (let i = 0; i < WORM.length; i++) {
        const d = (WORM[i][0] - x) ** 2 + (WORM[i][2] - z) ** 2;
        if (d < bd) { bd = d; best = i; }
    }
    return best / 240;
}
function wormTube(s0, s1, n) {
    const pts = [], R = [];
    for (let k = 0; k <= n; k++) {
        const s = mix(s0, s1, k / n), p = wormAt(s), r = wormR(s);
        pts.push([p[0], r + 0.003, p[2]]); R.push(r);
    }
    return tube(pts, R, 0.02);
}
function gummyWorm() {
    const body = wormTube(0.5, 1, 10);
    // the clitellum, a band round the body just under the skin
    const s = 0.78, p = wormAt(s), q = wormAt(s + 0.01), t = norm([q[0] - p[0], 0, q[2] - p[2]]);
    const band = SDF.torus(0.05, 0.02).rotate(new Quaternion().setFromUnitVectors(new Vector3(0, 1, 0), new Vector3(...t))).translate(p[0], wormR(s) + 0.003, p[2]);
    return buildCreature({
        body: { field: body, cell: 0.012, h: 0.045 },
        parts: [{ name: 'back half', field: wormTube(0, 0.56, 12), h: 0.04 }],
        chains: [],
        insides: [{ field: band, kind: 'milk', cell: 0.006 }],
        head: [wormAt(0.9).map((x, i) => (i === 1 ? 0.07 : x)), wormAt(0.97).map((x, i) => (i === 1 ? 0.07 : x)), [wormAt(0.9)[0], 0.12, wormAt(0.9)[2]]],
        span: [[...wormAt(1)].map((x, i) => (i === 1 ? 0.05 : x)), [...wormAt(0.5)].map((x, i) => (i === 1 ? 0.07 : x))],
    });
}
// Peristalsis: a squeeze runs from head to tail along the edges that run
// along the body.
function wormMuscles(body) {
    body.muscles = {
        weight(m, d) {
            const s = wormS(m[0], m[2]), a = wormAt(s), b = wormAt(Math.min(1, s + 0.01)), t = norm([b[0] - a[0], 0.0001, b[2] - a[2]]);
            return Math.abs(d[0] * t[0] + d[2] * t[2]);
        },
        phase: (m) => wormS(m[0], m[2]),
        pulse: { a: 0.1, omegaA: 4, phaseA: 16, b: 0.22, omegaB: 12, phaseB: 24 },
    };
}

// ── gummy bear ──────────────────────────────────────────────────────────────
// Lying on its back, the way a gummy bear comes out of the bag: the face
// and tummy up, head towards +z.
function gummyBear() {
    const body = SDF.unionAll([
        ellipsoidAt([0.27, 0.13, 0.31], [0, 0.135, 0]),
        ellipsoidAt([0.25, 0.13, 0.21], [0, 0.14, 0.5]),
    ], 0.08).union(ellipsoidAt([0.1, 0.07, 0.085], [0, 0.22, 0.6]), 0.05);
    const parts = [];
    for (const [s, side] of [[1, 'left'], [-1, 'right']]) {
        parts.push({ name: `${side} ear`, field: sphereAt(0.085, [0.19 * s, 0.13, 0.68]), h: 0.04 });
        parts.push({ name: `${side} arm`, field: SDF.capsule([0.2 * s, 0.14, 0.2], [0.4 * s, 0.1, 0.06], 0.09, 0.08), h: 0.045 });
        parts.push({ name: `${side} leg`, field: SDF.capsule([0.14 * s, 0.13, -0.2], [0.24 * s, 0.105, -0.43], 0.11, 0.1), h: 0.045 });
    }
    return buildCreature({
        // the face is moulded into the gel, as on the real thing: no eyes inside
        body: { field: body, cell: 0.013, h: 0.06 }, parts, chains: [], insides: [],
        head: [v(0, 0.2, 0.5), v(0.15, 0.2, 0.5), v(0, 0.2, 0.65)],
        span: [v(0, 0.14, 0.7), v(0, 0.13, -0.3)],
    });
}

// ── the table of creatures ──────────────────────────────────────────────────
// Looks: tint, the light one unit of gel lets through, and depth, how many
// units deep that colour is set; milk, what is inside; gill and gillDepth,
// the chains' tint and density; deep, the shadow's colour; chip, the swatch.
const look = (tint, depth, milk, gill, gillDepth, deep, chip, extra = {}) => ({ tint, depth, milk, gill, gillDepth, deep, chip, ...extra });
const RAINBOW_CHIP = 'linear-gradient(90deg, #ff5a5a, #ffb23d, #ffe04a, #6fdc6a, #4fb4ff)';

export const CREATURES = {
    jellyfish: {
        title: 'jellyfish', build: jellyfish, setup: jellyfishMuscles,
        parts: 'Tentacles', length: 'Width', hint: 'grab a thick tentacle near its middle and pull',
        camera: { yaw: 0.6, pitch: 0.78, target: [0, 0.12, 0], fit: 2.9 },
        look: 'moon',
        looks: {
            moon: look([0.8, 0.88, 0.98], 2.2, [0.86, 0.42, 0.82], [0.86, 0.9, 1.0], 10, [0.7, 0.76, 0.88], '#d8e4f6'),
            nettle: look([1.0, 0.6, 0.22], 3.4, [0.6, 0.16, 0.06], [0.8, 0.2, 0.08], 22, [0.86, 0.6, 0.4], '#f39a3c', { stripes: 1 }),
            blubber: look([0.32, 0.46, 0.96], 3.4, [0.92, 0.95, 1.0], [0.18, 0.22, 0.75], 20, [0.42, 0.5, 0.82], '#5b78e8'),
            lion: look([0.95, 0.3, 0.2], 3.8, [1.0, 0.86, 0.55], [0.9, 0.24, 0.1], 22, [0.82, 0.48, 0.42], '#e2523a'),
            rainbow: look([0.95, 0.75, 0.8], 2.6, [1.0, 0.9, 0.95], [0.9, 0.5, 0.8], 14, [0.66, 0.6, 0.72], 'conic-gradient(#ff5a5a, #ffb23d, #ffe04a, #6fdc6a, #4fb4ff, #b36bff, #ff5a5a)', { rainbow: 1 }),
        },
        // the hue goes round the bell; the nettle's stripes run down it
        paint(L, x, y, z, part) {
            let c = L.rainbow ? rainbowTint(Math.atan2(z, x) / (Math.PI * 2) + 0.5) : WHITE;
            if (L.stripes && part === 0) {
                const st = smoothstep(0.55, 0.9, Math.cos(16 * Math.atan2(z, x) + 8 * y)) * smoothstep(0.05, 0.2, y);
                c = c.map((q) => q * (1 - 0.55 * st));
            }
            return c;
        },
    },
    gecko: {
        // the solver's own tail sway; and the tail lets go at half the pull a
        // leg needs, as a gecko's does (autotomy)
        title: 'gecko', build: gecko, setup: (body) => { body.muscles = 'sway'; body.setTearLimit(0, 0.035); },
        parts: 'Tail & legs', length: 'Length', hint: 'grab the tail and pull: it drops off and thrashes',
        camera: { yaw: 0.55, pitch: 0.62, target: [-0.35, 0.12, 0], fit: 3.2 },
        look: 'leopard',
        looks: {
            leopard: look([1.0, 0.8, 0.35], 3.8, [1.0, 0.95, 0.85], [1.0, 0.7, 0.3], 14, [0.85, 0.7, 0.45], '#f2c35a', { spots: 1 }),
            flame: look([0.98, 0.5, 0.2], 3.8, [1.0, 0.9, 0.7], [1.0, 0.75, 0.4], 14, [0.85, 0.55, 0.4], '#ee7a33'),
            lime: look([0.55, 0.88, 0.3], 3.6, [0.95, 1.0, 0.85], [0.75, 0.95, 0.4], 14, [0.6, 0.75, 0.45], '#86d64b'),
            azure: look([0.3, 0.6, 0.98], 3.6, [0.9, 0.95, 1.0], [0.4, 0.7, 1.0], 14, [0.45, 0.6, 0.85], '#4f9bf0'),
            rainbow: look([0.95, 0.75, 0.7], 3.6, [1.0, 0.96, 0.9], [1.0, 0.8, 0.7], 14, [0.66, 0.6, 0.66], RAINBOW_CHIP, { rainbow: 1 }),
        },
        paint(L, x, y, z, part) {
            let c = L.rainbow ? rainbowTint((1.0 - x) * 0.3) : WHITE;
            if (L.spots) c = spots(c, x, y, z, 9, 0.7);
            return c;
        },
    },
    starfish: {
        title: 'starfish', build: starfish, setup: (body) => { body.muscles = { weight: () => 1, pulse: { a: 0.02, omegaA: 1.2, b: 0.12, omegaB: 9 } }; },
        parts: 'Arms', length: 'Disc', hint: 'grab an arm and pull until it comes away',
        camera: { yaw: 0.4, pitch: 0.95, target: [0, 0.05, 0], fit: 2.5 },
        look: 'orange',
        looks: {
            orange: look([1.0, 0.5, 0.2], 3.4, [1.0, 0.85, 0.6], WHITE, 10, [0.85, 0.55, 0.4], '#f27a33', { dots: 1 }),
            ochre: look([0.68, 0.3, 0.75], 3.4, [1.0, 0.8, 0.95], WHITE, 10, [0.6, 0.45, 0.66], '#a65bbf', { dots: 1 }),
            red: look([0.95, 0.25, 0.25], 3.4, [1.0, 0.8, 0.7], WHITE, 10, [0.8, 0.45, 0.45], '#e04848', { dots: 1 }),
            blue: look([0.3, 0.5, 0.95], 3.4, [0.85, 0.9, 1.0], WHITE, 10, [0.45, 0.55, 0.82], '#4f7fe8'),
            sand: look([0.95, 0.8, 0.55], 3.0, [1.0, 0.95, 0.85], WHITE, 10, [0.8, 0.72, 0.6], '#efcf94', { dots: 1 }),
        },
        paint(L, x, y, z) { return L.dots ? spots(WHITE, x, y, z, 28, 0.35) : WHITE; },
    },
    octopus: {
        title: 'octopus', build: octopus, setup: octopusMuscles,
        parts: 'Arms', length: 'Size', hint: 'grab an arm and pull until it comes away',
        camera: { yaw: 0.55, pitch: 0.72, target: [0, 0.15, 0], fit: 3.0 },
        look: 'common',
        looks: {
            common: look([0.95, 0.38, 0.28], 3.8, [0.3, 0.15, 0.25], WHITE, 10, [0.8, 0.5, 0.45], '#e2654d', { spots: 1 }),
            ringed: look([0.98, 0.82, 0.35], 3.6, [0.3, 0.2, 0.1], WHITE, 10, [0.85, 0.75, 0.5], '#f0cc55', { rings: 1 }),
            violet: look([0.6, 0.32, 0.85], 3.6, [0.25, 0.15, 0.3], WHITE, 10, [0.6, 0.48, 0.75], '#9a55d6'),
            coral: look([1.0, 0.58, 0.48], 3.4, [0.35, 0.2, 0.2], WHITE, 10, [0.85, 0.62, 0.56], '#fb917a'),
            ghost: look([0.9, 0.93, 0.98], 1.8, [0.5, 0.5, 0.6], WHITE, 10, [0.75, 0.78, 0.85], '#e6ecf6'),
        },
        paint(L, x, y, z) {
            let c = L.spots ? spots(WHITE, x, y, z, 14, 0.4) : WHITE;
            if (L.rings) {
                const n = noise3(x * 7, y * 7, z * 7), ring = 1 - smoothstep(0.02, 0.06, Math.abs(n - 0.55));
                c = c.map((q, i) => q * mix(1, [0.25, 0.4, 1.0][i], ring));
            }
            return c;
        },
    },
    seaslug: {
        title: 'sea slug', build: seaSlug, setup: (body) => { body.muscles = 'sway'; },
        parts: 'Feelers', length: 'Length', hint: 'grab a horn or a front feeler and pull',
        camera: { yaw: 0.55, pitch: 0.62, target: [0, 0.12, 0], fit: 2.3 },
        look: 'shawl',
        looks: {
            shawl: look([0.62, 0.38, 0.95], 3.6, [0.95, 0.85, 1.0], [1.0, 0.45, 0.1], 18, [0.6, 0.5, 0.8], '#9c63f0'),
            opal: look([0.9, 0.93, 1.0], 2.4, [1.0, 0.95, 0.85], [1.0, 0.55, 0.2], 18, [0.75, 0.78, 0.85], '#e7ecfa'),
            lemon: look([1.0, 0.9, 0.3], 3.4, [1.0, 1.0, 0.85], [1.0, 0.8, 0.15], 18, [0.85, 0.8, 0.5], '#f5df4f'),
            dragon: look([0.3, 0.52, 1.0], 3.4, [0.85, 0.9, 1.0], [0.18, 0.32, 0.98], 18, [0.45, 0.55, 0.85], '#4f84f5'),
            rainbow: look([0.95, 0.8, 0.85], 3.0, [1.0, 0.95, 0.95], [1.0, 0.5, 0.6], 16, [0.66, 0.6, 0.7], RAINBOW_CHIP, { rainbow: 1 }),
        },
        paint(L, x) { return L.rainbow ? rainbowTint((0.7 - x) * 0.6) : WHITE; },
    },
    worm: {
        title: 'gummy worm', build: gummyWorm, setup: wormMuscles,
        parts: 'Halves', length: 'Length', hint: 'grab the back half and pull it away',
        camera: { yaw: 0.35, pitch: 0.85, target: [0, 0.05, 0], fit: 2.8 },
        look: 'classic',
        looks: {
            classic: look([0.97, 0.94, 0.9], 2.0, [1.0, 0.95, 0.9], WHITE, 10, [0.75, 0.7, 0.66], 'linear-gradient(90deg, #ff4d4d, #ffb23d 50%, #5fd65a)', { ends: [[1, 0.3, 0.28], [0.4, 0.92, 0.32]] }),
            earth: look([0.88, 0.5, 0.48], 3.6, [0.95, 0.7, 0.65], WHITE, 10, [0.75, 0.55, 0.52], '#d8807a', { bands: 1 }),
            sour: look([0.97, 0.94, 0.9], 2.0, [1.0, 0.95, 0.9], WHITE, 10, [0.75, 0.72, 0.7], 'linear-gradient(90deg, #ffb233, #4f8fff)', { ends: [[1, 0.68, 0.18], [0.35, 0.58, 1.0]] }),
            glow: look([0.42, 0.72, 1.0], 3.2, [0.9, 0.95, 1.0], WHITE, 10, [0.5, 0.65, 0.85], '#6fb2ff', { bands: 1 }),
            rainbow: look([0.97, 0.94, 0.9], 2.0, [1.0, 0.95, 0.9], WHITE, 10, [0.7, 0.66, 0.7], RAINBOW_CHIP, { rainbow: 1 }),
        },
        // two-tone along the worm, rings round it
        paint(L, x, y, z) {
            const s = wormS(x, z);
            let c = WHITE;
            if (L.ends) c = [0, 1, 2].map((i) => mix(L.ends[0][i], L.ends[1][i], smoothstep(0.35, 0.65, s)));
            if (L.rainbow) c = rainbowTint(s * 0.8);
            if (L.bands) c = c.map((q) => q * (1 - 0.2 * smoothstep(0.75, 1, Math.cos(s * 2.6 / 0.045 * Math.PI * 2))));
            return c;
        },
    },
    bear: {
        title: 'gummy bear', build: gummyBear, setup: (body) => { body.muscles = 'still'; },
        parts: 'Limbs', length: 'Height', hint: 'grab an ear, an arm or a leg and pull',
        camera: { yaw: Math.PI + 0.25, pitch: 0.95, target: [0, 0.1, 0.12], fit: 2.0 },
        look: 'red',
        looks: {
            red: look([1.0, 0.22, 0.22], 4.0, WHITE, WHITE, 10, [0.85, 0.45, 0.45], '#ee3b3b'),
            orange: look([1.0, 0.55, 0.15], 4.0, WHITE, WHITE, 10, [0.88, 0.62, 0.42], '#f78a2a'),
            yellow: look([1.0, 0.9, 0.3], 3.6, WHITE, WHITE, 10, [0.88, 0.82, 0.5], '#f7dd4f'),
            green: look([0.45, 0.95, 0.3], 3.6, WHITE, WHITE, 10, [0.55, 0.78, 0.45], '#6fdc4a'),
            clear: look([0.97, 0.96, 0.92], 1.4, WHITE, WHITE, 10, [0.8, 0.8, 0.78], '#f4f1e8'),
        },
        paint() { return WHITE; },
    },
    salamander: {
        // limbs are what a salamander is known to regrow: its legs tear at
        // the usual pull and come back from the stump; the body sways
        title: 'salamander', build: salamander, setup: (body) => { body.muscles = 'sway'; },
        parts: 'Legs & tail', length: 'Length', hint: 'grab a leg and pull it off: watch it grow back',
        camera: { yaw: 0.55, pitch: 0.62, target: [-0.3, 0.12, 0], fit: 3.1 },
        look: 'fire',
        looks: {
            fire: look([1.0, 0.8, 0.15], 3.8, [1.0, 0.92, 0.7], WHITE, 10, [0.6, 0.5, 0.3], 'linear-gradient(90deg, #1d1a16 0 30%, #f5c82a 30% 60%, #1d1a16 60%)', { fire: 1 }),
            eft: look([1.0, 0.42, 0.15], 3.6, [1.0, 0.85, 0.7], WHITE, 10, [0.85, 0.55, 0.4], '#f56b2a', { dots: 1 }),
            tiger: look([0.88, 0.8, 0.3], 3.6, [1.0, 0.96, 0.8], WHITE, 10, [0.72, 0.68, 0.42], '#cdbf4a', { bars: 1 }),
            cave: look([0.97, 0.86, 0.86], 1.8, [1.0, 0.7, 0.72], WHITE, 10, [0.8, 0.72, 0.74], '#f5dede'),
            rainbow: look([0.95, 0.75, 0.7], 3.6, [1.0, 0.96, 0.9], WHITE, 10, [0.66, 0.6, 0.66], RAINBOW_CHIP, { rainbow: 1 }),
        },
        paint(L, x, y, z) {
            let c = L.rainbow ? rainbowTint((1.0 - x) * 0.3) : WHITE;
            // fire: near black, with the yellow left showing in blotches
            if (L.fire) { const sp = smoothstep(0.5, 0.6, noise3(x * 7, y * 7, z * 7)); c = c.map((q) => q * mix(0.12, 1, sp)); }
            if (L.dots) c = spots(c, x, y, z, 26, 0.6);
            // tiger: dark bars across the back, broken up by noise
            if (L.bars) { const bar = smoothstep(0.35, 0.75, Math.sin(x * 14 + 3 * noise3(x * 5, y * 5, z * 5))); c = c.map((q) => q * (1 - 0.7 * bar)); }
            return c;
        },
    },
};
