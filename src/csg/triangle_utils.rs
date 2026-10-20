//! Shared triangle helpers (hash grid + degeneracy) used by CSG ops.

use crate::math::{Triangle, Vector3};

use super::js_topology::{is_tri_degenerate as js_is_tri_degenerate, JsTriangle};

/// three-bvh-csg's `isTriDegenerate`, for an f32 triangle.
pub fn is_tri_degenerate(tri: &Triangle, eps: f32) -> bool {
    js_is_tri_degenerate(JsTriangle::from_triangle(*tri), eps as f64)
}

/// Hash a vertex the same way as JS `hashVertex3` (`(v * 1e6 + 0.5) | 0`).
pub fn hash_vertex3(v: Vector3) -> (i32, i32, i32) {
    (
        hash_coord(v.x as f64),
        hash_coord(v.y as f64),
        hash_coord(v.z as f64),
    )
}

pub(crate) fn hash_coord(v: f64) -> i32 {
    const HASH_MULTIPLIER: f64 = 1_000_000.0;
    const HASH_ADDITION: f64 = 0.5;
    (v * HASH_MULTIPLIER + HASH_ADDITION) as i32
}
