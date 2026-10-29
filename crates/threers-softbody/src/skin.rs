//! Carrying things along with the lattice: a skin, or any other points bound
//! to a part (an [`Embedding`]), moved each frame with the tetrahedron each
//! lies in, and its normal by the cofactor of the deformation gradient there.

use crate::lattice::Embedding;
use crate::sim::SoftBody;

/// Where the points of `emb` — bound to topology `topo` (0 the body, `1 + i`
/// part `i`) — are now on `block`, into `pos` (xyz); and, when `rest_normal`
/// and `nrm` are not empty, their normals, into `nrm`.
pub fn carry(sim: &SoftBody, topo: usize, emb: &Embedding, rest_normal: &[f32], block: usize, pos: &mut [f32], nrm: &mut [f32]) {
    let t = &sim.topos[topo];
    let off = sim.blocks[block].off;
    let (x, f) = (&sim.x, &sim.node_f);
    let with_n = !nrm.is_empty() && !rest_normal.is_empty();
    for v in 0..emb.tet_of.len().min(pos.len() / 3) {
        let tet = emb.tet_of[v] as usize;
        let (b1, b2, b3) = (emb.bary[3 * v], emb.bary[3 * v + 1], emb.bary[3 * v + 2]);
        let w = [1.0 - b1 - b2 - b3, b1, b2, b3];
        let (mut p, mut a, mut bb, mut c) = ([0.0f32; 3], [0.0f32; 3], [0.0f32; 3], [0.0f32; 3]);
        for k in 0..4 {
            let i = off + t.tets[4 * tet + k] as usize;
            for q in 0..3 {
                p[q] += x[3 * i + q] * w[k];
            }
            if with_n {
                let o = 12 * i;
                for q in 0..3 {
                    a[q] += f[o + q] * w[k];
                    bb[q] += f[o + 4 + q] * w[k];
                    c[q] += f[o + 8 + q] * w[k];
                }
            }
        }
        pos[3 * v..3 * v + 3].copy_from_slice(&p);
        if with_n {
            let rn = &rest_normal[3 * v..3 * v + 3];
            let cross = |u: [f32; 3], v: [f32; 3]| [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]];
            let (bc, ca, ab) = (cross(bb, c), cross(c, a), cross(a, bb));
            let n = [0, 1, 2].map(|q| bc[q] * rn[0] + ca[q] * rn[1] + ab[q] * rn[2]);
            let l = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
            let l = if l > 0.0 { l } else { 1.0 };
            nrm[3 * v..3 * v + 3].copy_from_slice(&n.map(|q| q / l));
        }
    }
}
