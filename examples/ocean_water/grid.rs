//! The water mesh: a radial disc, built once and never rebuilt.
//!
//! Rings grow geometrically outward from the viewer, so density follows the
//! perspective — tight where a pixel covers centimetres, sparse where it covers
//! tens of metres. It is the cheap stand-in for the nested LOD grid a production
//! ocean would use.
//!
//! The disc is anchored to the *camera*, not the world. Pinned to the origin it
//! would put the finest rings wherever the origin happens to be and leave the
//! water directly under the viewer on 30-metre triangles. The wave field stays
//! world-anchored regardless: the mesh slides beneath it, and the compute pass
//! is handed the disc's current centre to add back.

use std::f32::consts::PI;

use threers::{BufferAttribute, BufferGeometry};

pub struct Disc {
    pub geometry: BufferGeometry,
    /// Per vertex: `[x, z, spacing, 0]` in disc-local metres. Uploaded once as
    /// the compute pass's input and never touched again.
    pub base: Vec<[f32; 4]>,
}

/// Build the disc. `r0` is the innermost ring radius, `r_max` the outermost.
pub fn build(rings: usize, sectors: usize, r0: f32, r_max: f32) -> Disc {
    let growth = (r_max / r0).powf(1.0 / rings as f32);
    let n_verts = 1 + rings * sectors;

    let mut base = Vec::with_capacity(n_verts);
    base.push([0.0, 0.0, r0, 0.0]);

    let mut prev_r = 0.0f32;
    for ring in 1..=rings {
        let r = r0 * growth.powi(ring as i32 - 1);
        // Aliasing is set by the coarser of the two directions, not the finer.
        let spacing = (r - prev_r).max(2.0 * PI * r / sectors as f32);
        prev_r = r;
        for s in 0..sectors {
            let a = 2.0 * PI * s as f32 / sectors as f32;
            base.push([r * a.cos(), r * a.sin(), spacing, 0.0]);
        }
    }

    // vertex_color is the one varying that reaches the fragment untouched by any
    // material transform, so the sample coordinate rides in it. Positions and
    // normals are left at zero: the compute pass writes them before the first
    // draw, and nothing ever reads the CPU-side copies again.
    let mut colors = Vec::with_capacity(n_verts * 4);
    let mut uvs = Vec::with_capacity(n_verts * 2);
    for b in &base {
        colors.extend_from_slice(&[b[0], b[1], b[2], 1.0]);
        uvs.extend_from_slice(&[b[0] * 0.01, b[1] * 0.01]);
    }

    let mut index: Vec<u32> = Vec::with_capacity(rings * sectors * 6);
    for s in 0..sectors {
        // Centre fan.
        index.extend_from_slice(&[0, (1 + (s + 1) % sectors) as u32, (1 + s) as u32]);
    }
    for ring in 1..rings {
        let inner = 1 + (ring - 1) * sectors;
        let outer = 1 + ring * sectors;
        for s in 0..sectors {
            let s1 = (s + 1) % sectors;
            let (i0, i1) = ((inner + s) as u32, (inner + s1) as u32);
            let (o0, o1) = ((outer + s) as u32, (outer + s1) as u32);
            index.extend_from_slice(&[i0, o1, o0, i0, i1, o1]);
        }
    }

    let mut geometry = BufferGeometry::new();
    geometry.set_attribute("position", BufferAttribute::new(vec![0.0; n_verts * 3], 3));
    geometry.set_attribute("normal", BufferAttribute::new(vec![0.0; n_verts * 3], 3));
    geometry.set_attribute("uv", BufferAttribute::new(uvs, 2));
    geometry.set_attribute("color", BufferAttribute::new(colors, 4));
    geometry.set_index(index);
    // The compute pass writes this buffer directly; see `crate::waves_gpu`.
    geometry.gpu_writable = true;

    Disc { geometry, base }
}

/// Reverse triangle winding, so a sphere reads as front-facing from the inside.
/// [`ShaderMaterial`](threers::ShaderMaterial) pipelines cull back faces
/// unconditionally, which is the one thing standing between a sphere and a sky
/// dome.
pub fn invert_winding(mut geom: BufferGeometry) -> BufferGeometry {
    if let Some(mut idx) = geom.index.clone() {
        for tri in idx.chunks_mut(3) {
            tri.swap(1, 2);
        }
        geom.set_index(idx);
    }
    geom
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disc_spacing_grows_outward_and_indexes_every_vertex() {
        let d = build(16, 32, 2.0, 1000.0);
        assert_eq!(d.base.len(), 1 + 16 * 32);

        // Spacing must ascend, or the band limit would keep detail the mesh
        // cannot carry out at the horizon.
        let outer = d.base[d.base.len() - 1][2];
        let inner = d.base[1][2];
        assert!(outer > inner * 10.0, "{inner} -> {outer}");

        let idx = d.geometry.index.as_ref().unwrap();
        assert_eq!(
            idx.iter().copied().max().unwrap() as usize,
            d.base.len() - 1
        );
        assert!(idx.len().is_multiple_of(3));
    }
}
