// What a gummy creature is made of, on THREE.SoftBody (threers::softbody):
// a soft body, parts glued on that tear away and regrow, particle chains
// rooted in it, and opaque surfaces inside the gel. A creature module
// declares those as THREE.SDF fields and point lists; Rust builds the skins
// (surface nets), the lattices and the glue, and runs it.
//
//   buildCreature({
//     body:   { field, cell, h },             the main soft body
//     parts:  [{ field, name, cell, h }],     tearable; stopped just inside the body
//     chains: [{ points, r0, r1, stiff }],    rooted in the body at points[0]
//     insides:[{ field, kind, cell }],        'milk' | 'eye' | 'mouth'
//     head:   [p0, p1, p2],                   a frame the chains turn with
//     span:   [a, b],                         two body points for the size readout
//   })  →  { body: THREE.SoftBody, insides: [{ kind, surf }] }
//
// Everything must sit on or above the table, y >= 0.
import { SoftBody } from '../threejs-shim.js';

export function buildCreature(spec) {
    const t0 = performance.now();
    const body = SoftBody.build(spec);
    const insides = (spec.insides ?? []).map(({ field, kind, cell }) => {
        const m = field.surfaceNets(cell ?? 0.008);
        return { kind, surf: { pos: m.position, nrm: m.normal, idx: m.index } };
    });
    return { body, insides, buildMs: performance.now() - t0 };
}

export const smoothstep = (a, b, x) => { const t = Math.min(1, Math.max(0, (x - a) / (b - a))); return t * t * (3 - 2 * t); };
export const mix = (a, b, t) => a + (b - a) * t;
