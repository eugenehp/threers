//! CSG booleans via `manifold-rust`, a pure-Rust port of the Manifold kernel.
//!
//! # Why this exists
//!
//! The in-house arrangement kernel in this module's parent is exact where it
//! succeeds and honest where it does not — it verifies its own output and
//! declines rather than return a mesh it cannot vouch for. The trouble is what
//! happens next: the caller falls back to the float evaluator, and when *that*
//! is not watertight either the geometry is returned anyway. On the reference
//! model 4 of 17 booleans take that path, and the resulting solid has cracks.
//!
//! The root cause is architectural rather than a bug to be found. That kernel
//! computes intersection points in floating point and then reconciles them with
//! roughly sixty absolute tolerances scattered across the pipeline, of which
//! three are scale-relative. Exact predicates over inexact constructions give no
//! guarantees, and the tolerances have an implicit ordering nothing enforces —
//! moving one of them from `1e-7` to `1e-6` collapses the reference model from
//! 40,173 triangles to 2,937.
//!
//! Manifold takes the other road: exact rational arithmetic (via `dashu`), and
//! manifoldness guaranteed by construction rather than checked afterwards.
//! OpenSCAD — the front end this crate implements — moved its own geometry
//! backend to Manifold for these reasons.
//!
//! The port is pure Rust, which is what makes it usable here: it builds for
//! `wasm32-unknown-unknown` with no C++ toolchain, so the browser target is
//! unaffected.
//!
//! # Status
//!
//! Behind the `manifold` feature and off by default, because the port is young.
//! The parent kernel remains the default. Enable it, run
//! `tests/watertight_gate.rs`, and check
//! [`unverified_booleans`](crate::exact_csg::unverified_booleans) reaches zero for your
//! models before relying on it.

use manifold_rust::manifold::Manifold;
use manifold_rust::types::{BooleanEngine, MeshGL64, OpType};

use super::Op;
#[cfg(test)]
use crate::core::BufferAttribute;
use crate::core::BufferGeometry;

/// Convert a geometry to Manifold's mesh form.
///
/// Positions are widened to `f64` and handed over as a triangle soup; Manifold
/// welds and orients it. Nothing else transfers — normals and UVs are rebuilt
/// downstream from the result, since a boolean invalidates them anyway.
fn to_mesh(g: &BufferGeometry) -> Option<MeshGL64> {
    let pos = g.get_attribute("position")?;
    let verts: Vec<f64> = pos.array.iter().map(|v| *v as f64).collect();
    let tri_verts: Vec<u64> = match &g.index {
        Some(idx) => idx.iter().map(|i| *i as u64).collect(),
        None => (0..(verts.len() / 3) as u64).collect(),
    };
    if tri_verts.len() < 3 {
        return None;
    }
    Some(MeshGL64 {
        num_prop: 3,
        vert_properties: verts,
        tri_verts,
        ..Default::default()
    })
}

/// Convert a Manifold result back to a geometry.
///
/// Expanded to an unindexed triangle soup, matching what the arrangement kernel
/// emits. That is not a detail: Manifold returns an indexed mesh with vertices
/// shared between adjacent faces, and `compute_vertex_normals` then averages a
/// normal across every sharp edge — which renders a machined part as if it had
/// been left in the sun.
fn from_manifold(m: &Manifold) -> BufferGeometry {
    let mesh = m.get_mesh_gl64(-1);
    let stride = mesh.num_prop.max(3) as usize;
    let vert = |i: u64| -> super::V3 {
        let o = i as usize * stride;
        [
            mesh.vert_properties[o],
            mesh.vert_properties[o + 1],
            mesh.vert_properties[o + 2],
        ]
    };
    let tris: Vec<[super::V3; 3]> = mesh
        .tri_verts
        .chunks_exact(3)
        .map(|t| [vert(t[0]), vert(t[1]), vert(t[2])])
        .collect();
    super::build_geometry(&tris)
}

/// `a op b`, or `None` if either input could not be read as a mesh or the kernel
/// reported an error.
///
/// Returning `None` rather than a best effort keeps the caller's existing
/// contract: it falls back exactly as it would if the in-house arrangement had
/// declined.
pub fn boolean(a: &BufferGeometry, b: &BufferGeometry, op: Op) -> Option<BufferGeometry> {
    let (ma, mb) = (to_mesh(a)?, to_mesh(b)?);
    // Strict import first, soup only if that fails.
    //
    // What arrives here is whatever the caller built — an OpenSCAD primitive, or
    // the output of an earlier boolean — and it is often an unwelded triangle
    // soup, which the strict import rejects. But `AllowSoup` marks the operand
    // soup-backed, and `BooleanEngine::Auto` then has to take the Robust engine,
    // which is the slower of the two. Importing everything as soup means paying
    // for that on every boolean including the ones that did not need it.
    let import = |m: &MeshGL64| {
        let strict = Manifold::from_mesh_gl64(m);
        if strict.status() == manifold_rust::types::Error::NoError {
            strict
        } else {
            Manifold::from_mesh_gl64_robust(m)
        }
    };
    let (ma, mb) = (import(&ma), import(&mb));

    // Verify the *operands*, not just the result.
    //
    // A mesh that fails to import is not reported as an error on the boolean:
    // it behaves as an empty solid. So a subtraction whose second operand did
    // not survive the trip quietly returns its first operand unchanged — and
    // since that operand was closed to begin with, every check below passes and
    // the caller gets a confident, wrong answer. This is how a `sphere($fn=32)`
    // bitten out of a cube came back as the whole cube: the sphere imported
    // `NotClosed` under both the strict and the robust reader, and the gate
    // downstream had nothing to object to.
    if ma.status() != manifold_rust::types::Error::NoError
        || mb.status() != manifold_rust::types::Error::NoError
    {
        REJECTED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        return None;
    }
    let out = ma.boolean_with_engine(
        &mb,
        match op {
            Op::Union => OpType::Add,
            Op::Difference => OpType::Subtract,
            Op::Intersection => OpType::Intersect,
        },
        BooleanEngine::Auto,
    );
    if out.status() != manifold_rust::types::Error::NoError {
        return None;
    }
    let g = from_manifold(&out);

    // Verify the result rather than trust it. Manifold guarantees manifoldness
    // by construction and this gate has never fired on the corpus — but a
    // guarantee that holds is cheap to check and a guarantee that stops holding
    // is expensive to discover downstream, and this is a five-week-old crate. A
    // failure here falls through to the arrangement kernel, so the backend is
    // never worse than not having it.
    //
    // The arrangement kernel checks itself the same way and declines rather than
    // return a mesh it cannot vouch for; that self-verification is the reason its
    // own defect was diagnosable at all. It belongs around every backend.
    if !super::is_closed_manifold(&super::triangles(&g)) {
        REJECTED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        return None;
    }
    Some(g)
}

/// How many Manifold results failed the watertightness check and fell through to
/// the arrangement kernel, since process start.
///
/// Expected to be zero. If it is not, the backend is doing something this crate
/// did not anticipate and the corpus comparison in `examples/kernel_compare.rs`
/// is the place to start.
pub fn rejected_results() -> usize {
    REJECTED.load(std::sync::atomic::Ordering::Relaxed)
}

static REJECTED: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

#[cfg(test)]
mod tests {
    use super::*;

    fn cube(size: f32, at: [f32; 3]) -> BufferGeometry {
        let src: BufferGeometry = crate::geometries::BoxGeometry::new(size, size, size);
        let pos = src.get_attribute("position").expect("position");
        let moved: Vec<f32> = pos
            .array
            .chunks(3)
            .flat_map(|v| [v[0] + at[0], v[1] + at[1], v[2] + at[2]])
            .collect();
        let mut g = BufferGeometry::new();
        g.set_attribute("position", BufferAttribute::new(moved, 3));
        if let Some(i) = &src.index {
            g.set_index(i.clone());
        }
        g
    }

    /// Every edge used by exactly two triangles — the property the in-house
    /// kernel cannot guarantee and this backend exists to provide.
    fn watertight(g: &BufferGeometry) -> bool {
        use std::collections::HashMap;
        let tris = super::super::triangles(g);
        let key = |p: [f64; 3]| {
            (
                (p[0] * 1e5).round() as i64,
                (p[1] * 1e5).round() as i64,
                (p[2] * 1e5).round() as i64,
            )
        };
        let mut e: HashMap<_, u32> = HashMap::new();
        for t in &tris {
            for k in 0..3 {
                let (mut u, mut v) = (key(t[k]), key(t[(k + 1) % 3]));
                if u > v {
                    std::mem::swap(&mut u, &mut v);
                }
                *e.entry((u, v)).or_insert(0) += 1;
            }
        }
        !e.is_empty() && e.values().all(|&c| c == 2)
    }

    #[test]
    fn overlapping_cubes_are_watertight_for_every_op() {
        let a = cube(10.0, [0.0; 3]);
        let b = cube(10.0, [5.0, 5.0, 5.0]);
        for op in [Op::Union, Op::Difference, Op::Intersection] {
            let g = boolean(&a, &b, op).expect("manifold boolean");
            assert!(watertight(&g), "{op:?} produced a mesh with cracks");
        }
    }

    /// Two cubes meeting exactly on a face. This is the configuration that makes
    /// centroid ray-parity a coin flip in the in-house kernel — the point being
    /// classified lies exactly on the surface it is tested against.
    #[test]
    fn a_shared_face_is_resolved_not_guessed() {
        let a = cube(10.0, [0.0; 3]);
        let b = cube(10.0, [10.0, 0.0, 0.0]);
        let g = boolean(&a, &b, Op::Union).expect("manifold boolean");
        assert!(watertight(&g), "coincident faces left cracks");
        let v = super::super::metrics::volume(&super::super::triangles(&g));
        assert!(
            (v.abs() - 2000.0).abs() < 1e-6,
            "union of two 10³ cubes = 2000, got {v}"
        );
    }

    #[test]
    fn disjoint_difference_is_the_original() {
        let a = cube(10.0, [0.0; 3]);
        let b = cube(2.0, [100.0, 0.0, 0.0]);
        let g = boolean(&a, &b, Op::Difference).expect("manifold boolean");
        let v = super::super::metrics::volume(&super::super::triangles(&g));
        assert!(
            (v.abs() - 1000.0).abs() < 1e-6,
            "cutting nothing away, got {v}"
        );
    }
}
