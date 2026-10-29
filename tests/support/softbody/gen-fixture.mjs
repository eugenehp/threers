// Writes fixture.txt for tests/softbody_parity.rs: the JavaScript solver
// (soft-sim.js, soft-lattice.js — Axolotl Jelly's, kept here as the
// reference) building and running a small creature, so the Rust port can be
// held to what it computes.
//
//   node tests/support/softbody/gen-fixture.mjs
//
// The fields are plain arithmetic, written identically in the Rust test.
// Math.random is replaced by the LCG js_math::Lcg is, seeded alike.
import { writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { SoftLattice } from './soft-lattice.js';
import { AxoSim } from './soft-sim.js';

let seed = 7;
Math.random = () => { seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0; return seed / 4294967296; };

// ── the creature: an ellipsoid body, two capsule arms, two chains ──
const body = (x, y, z) => {
  const X = x / 0.35, Y = (y - 0.17) / 0.16, Z = z / 0.3;
  return (Math.sqrt(X * X + Y * Y + Z * Z) - 1) * 0.16;
};
const capsule = (x, y, z, a, b, r) => {
  const bx = b[0] - a[0], by = b[1] - a[1], bz = b[2] - a[2];
  const qx = x - a[0], qy = y - a[1], qz = z - a[2];
  let t = (qx * bx + qy * by + qz * bz) / (bx * bx + by * by + bz * bz);
  t = t < 0 ? 0 : t > 1 ? 1 : t;
  const dx = qx - bx * t, dy = qy - by * t, dz = qz - bz * t;
  return Math.sqrt(dx * dx + dy * dy + dz * dz) - r;
};
const ARMS = [[[0.2, 0.12, 0.05], [0.72, 0.07, 0.16], 0.065], [[-0.2, 0.12, -0.05], [-0.7, 0.07, -0.2], 0.06]];
const arm = (k) => (x, y, z) => Math.max(capsule(x, y, z, ...ARMS[k]), -(body(x, y, z) + 0.03));

// skins: points on a rough shell round each field (only their binding matters)
function skinOf(f, B, step) {
  const pts = [];
  for (let x = B.min[0]; x <= B.max[0]; x += step) for (let y = 0.01; y <= B.max[1]; y += step) for (let z = B.min[2]; z <= B.max[2]; z += step)
    if (Math.abs(f(x, y, z)) < step * 0.5) pts.push(x, y, z);
  return { pos: new Float32Array(pts), mat: new Float32Array(pts.length / 3 * 4) };
}
const BB = { min: [-0.42, -0.03, -0.37], max: [0.42, 0.4, 0.37] };
const AB = [{ min: [0.0, -0.03, -0.1], max: [0.85, 0.2, 0.3] }, { min: [-0.85, -0.03, -0.35], max: [0.0, 0.2, 0.1] }];
const bodyPart = SoftLattice.buildPart(body, BB, 0.08, 0, skinOf(body, BB, 0.03));

const nearest = (P, p) => {
  let best = -1, bd = Infinity;
  for (let i = 0; i < P.length / 3; i++) {
    const d = (P[3 * i] - p[0]) ** 2 + (P[3 * i + 1] - p[1]) ** 2 + (P[3 * i + 2] - p[2]) ** 2;
    if (d < bd) { bd = d; best = i; }
  }
  return best;
};
// glue, pivot and anchor, as creature-kit.js builds them
const legs = [0, 1].map((k) => {
  const f = arm(k), part = SoftLattice.buildPart(f, AB[k], 0.045, 1 + k, skinOf(f, AB[k], 0.02));
  const legIdx = [], bodyIdx = [], N = part.nodes, BN = bodyPart.nodes;
  for (let a = 0; a < N.length / 3; a++) {
    if (body(N[3 * a], N[3 * a + 1], N[3 * a + 2]) > 0.025) continue;
    const near = [];
    for (let j = 0; j < BN.length / 3; j++) {
      const d = Math.hypot(N[3 * a] - BN[3 * j], N[3 * a + 1] - BN[3 * j + 1], N[3 * a + 2] - BN[3 * j + 2]);
      if (d < 0.12) near.push([d, j]);
    }
    near.sort((x, y) => x[0] - y[0]);
    for (const [, j] of near.slice(0, 3)) { legIdx.push(a); bodyIdx.push(j); }
  }
  const pivot = [0, 0, 0], uniq = [...new Set(legIdx)];
  for (const a of uniq) for (let c = 0; c < 3; c++) pivot[c] += N[3 * a + c] / uniq.length;
  return { ...part, def: { i: k, name: `arm ${k}` }, glue: { leg: Uint32Array.from(legIdx), body: Uint32Array.from(bodyIdx) }, pivot, anchor: nearest(BN, pivot) };
});
// chains, as creature-kit.js builds them
const CHAINS = [
  { points: [[0.1, 0.3, 0.1], [0.12, 0.38, 0.14], [0.14, 0.45, 0.18], [0.16, 0.5, 0.22]], r0: 0.02, r1: 0.008, stiff: 0.2 },
  { points: [[-0.1, 0.3, -0.1], [-0.12, 0.38, -0.14], [-0.14, 0.45, -0.18]], r0: 0.02, r1: 0.008, stiff: 0.16 },
];
const pos = [], parent = [], anchor = [], stiff = [], rad = [], segR = [], roots = [];
for (const c of CHAINS) {
  const n = c.points.length - 1; let root = -1;
  c.points.forEach((p, k) => {
    const t = k / n, id = pos.length / 3;
    pos.push(...p); rad.push(c.r0 + (c.r1 - c.r0) * t);
    parent.push(k === 0 ? -1 : id - 1); anchor.push(k === 0 ? -1 : root);
    stiff.push(k === 0 ? 0 : c.stiff * (1 - 0.5 * t)); segR.push(k === 0 ? 0 : rad[id - 1]);
    if (k === 0) { root = id; roots.push(id); }
  });
}
const GP = new Float32Array(pos);
const gills = {
  rest: GP, parent: Int32Array.from(parent), anchor: Int32Array.from(anchor), stiff: Float32Array.from(stiff),
  radius: Float32Array.from(rad), segRadius: Float32Array.from(segR), roots: Int32Array.from(roots),
  rootEmb: SoftLattice.embedPoints(new Float32Array(roots.flatMap((r) => [GP[3 * r], GP[3 * r + 1], GP[3 * r + 2]])), bodyPart.lat, bodyPart.DmInv),
};
const BN = bodyPart.nodes;
const MARKS = { head: [[0.0, 0.3, 0.0], [0.2, 0.25, 0.0], [0.0, 0.25, 0.2]], span: [[0.33, 0.17, 0], [-0.33, 0.17, 0]] };
const landmarks = { head: MARKS.head.map((p) => nearest(BN, p)), snout: nearest(BN, MARKS.span[0]), tail: nearest(BN, MARKS.span[1]), eyes: [] };
const sim = new AxoSim({ body: bodyPart, legs, gills, landmarks });
sim.walls = Float32Array.from([1, 0, 0, 0.9, -1, 0, 0, 0.9, 0, 0, 1, 0.8, 0, 0, -1, 0.8]);

// ── the record ──
const out = [];
// Float32Arrays as the hex of their bits (exact, and half the size of the
// decimal); everything else as JavaScript prints a number.
const put = (name, arr) => {
  if (arr instanceof Float32Array) {
    const u = new Uint32Array(arr.buffer, arr.byteOffset, arr.length);
    out.push(`${name} ${arr.length} ${Array.from(u, (b) => b.toString(16)).join(' ')} f32`);
  } else out.push(`${name} ${arr.length} ${Array.from(arr).join(' ')}`);
};
const part = (tag, p) => {
  put(`${tag}.nodes`, p.nodes); put(`${tag}.tets`, p.tets); put(`${tag}.edges`, p.edges); put(`${tag}.restVol`, p.restVol);
  put(`${tag}.dmInv`, p.DmInv); put(`${tag}.boundary`, p.boundary); put(`${tag}.radius`, p.radius);
  put(`${tag}.skin`, p.surf.pos); put(`${tag}.tetOf`, p.emb.tetOf); put(`${tag}.bary`, p.emb.bary);
  put(`${tag}.bounds`, [...(tag === 'body' ? BB : AB[+tag.slice(3)]).min, ...(tag === 'body' ? BB : AB[+tag.slice(3)]).max]);
};
part('body', bodyPart);
legs.forEach((l, k) => { part(`arm${k}`, l); put(`arm${k}.glueLeg`, l.glue.leg); put(`arm${k}.glueBody`, l.glue.body); put(`arm${k}.pivot`, l.pivot); put(`arm${k}.anchor`, [l.anchor]); });
put('chains.rest', gills.rest); put('chains.stiff', gills.stiff); put('chains.radius', gills.radius); put('chains.segRadius', gills.segRadius);
put('chains.tetOf', gills.rootEmb.tetOf); put('chains.bary', gills.rootEmb.bary);
put('landmarks', [...landmarks.head, landmarks.snout, landmarks.tail]);

// idle, then a hand on arm 0's tip pulling outward until it tears, then let go
const SNAP = new Set([1, 10, 20, 30, 60, 90]), SNAP_F = new Set([20, 90]);
const tip = ARMS[0][1];
for (let frame = 1; frame <= 90; frame++) {
  if (frame === 11) { sim.startGrab([tip[0], tip[1], tip[2]], sim.legSlots[0][0]); sim.grab.axis = [0.3, -0.5, -0.8]; }
  if (sim.grab && frame > 11) sim.grab.target = [tip[0] + 0.03 * (frame - 11), tip[1] + 0.01 * (frame - 11), tip[2] + 0.015 * (frame - 11)];
  if (sim.grab && frame > 11) sim.grab.angle = 0.02 * (frame - 11);
  if (frame === 45) sim.grab = null;
  if (frame === 50) sim.flick();
  sim.step(1 / 60);
  for (const e of sim.events) out.push(`event ${frame} ${e.type} ${e.leg ?? ''} ${e.r ?? ''} ${e.speed ?? ''}`.trim());
  sim.events.length = 0;
  if (SNAP.has(frame)) {
    put(`f${frame}.x`, sim.x); put(`f${frame}.gx`, sim.gx); put(`f${frame}.dp`, sim.dp.subarray(0, 3 * sim.dCount));
    if (SNAP_F.has(frame)) put(`f${frame}.nodeF`, sim.nodeF);
    put(`f${frame}.blocks`, sim.blocks.flatMap((b) => [b.active ? 1 : 0, b.scale, b.grow, b.twitch, b.strain ?? 0]));
    const m = sim.measure();
    put(`f${frame}.measure`, [m.volume, m.length, m.attached, m.loose]);
  }
}
const hit = sim.raycast([0, 1.5, 0.3], (() => { const d = [0.05, -1, -0.2], l = Math.hypot(...d); return d.map((v) => v / l); })());
put('raycast', hit.kind ? [hit.kind === 'part' ? 0 : 1, hit.kind === 'part' ? sim.blocks.indexOf(hit.block) : hit.index, hit.t] : [-1]);
writeFileSync(join(dirname(fileURLToPath(import.meta.url)), 'fixture.txt'), out.join('\n') + '\n');
console.log(`wrote ${out.length} lines; events: ${out.filter((l) => l.startsWith('event')).map((l) => l.split(' ').slice(1, 3).join(':')).join(', ')}`);
