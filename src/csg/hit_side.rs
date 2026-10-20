use crate::math::Matrix4;
#[cfg(test)]
use crate::math::Triangle;
use crate::mesh_bvh::MeshBvh;

use super::constants::*;
use super::js_topology::JsTriangle;

#[cfg(test)]
pub fn get_hit_side(tri: &Triangle, bvh: &MeshBvh) -> i8 {
    get_hit_side_js(JsTriangle::from_triangle(*tri), bvh)
}

/// three-bvh-csg's `getHitSide`: cast from the triangle's midpoint along its
/// normal into `bvh` (both faces); a first hit whose face normal points
/// along the ray means the triangle is inside. In f64 through
/// [`MeshBvh::raycast_first_js`], as JavaScript runs it — a midpoint on a
/// shared edge is decided by which triangle three-mesh-bvh returns first.
pub fn get_hit_side_js(tri: JsTriangle, bvh: &MeshBvh) -> i8 {
    let origin = tri.get_midpoint();
    let n = tri.get_normal();
    let dir = [n.x, n.y, n.z];
    let back = bvh
        .raycast_first_js(
            [origin.x, origin.y, origin.z],
            dir,
            DOUBLE_SIDE,
            0.0,
            f64::INFINITY,
        )
        .is_some_and(|h| dot(dir, h.face_normal) > 0.0);
    if back {
        BACK_SIDE
    } else {
        FRONT_SIDE
    }
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// three-bvh-csg's `getHitSideWithCoplanarCheck`, used once a coplanar
/// triangle took part in the split: up to three casts, the direction nudged
/// and flipped each time so a surface lying on the triangle is seen from both
/// sides; a hit at no distance is coplanar, aligned or opposite. Upstream
/// nudges by `Math.random()`; this takes a fixed sequence in the same range,
/// so the answer is deterministic and agrees with upstream's wherever the
/// nudge does not decide it.
pub fn get_hit_side_with_coplanar_check_js(tri: JsTriangle, bvh: &MeshBvh) -> i8 {
    const JITTER_EPSILON: f64 = 1e-8;
    const OFFSET_EPSILON: f64 = 1e-15;
    const NUDGES: [[f64; 3]; 3] = [[0.25, -0.5, 0.5], [-0.5, 0.25, -0.25], [0.5, 0.5, -0.5]];
    let n = tri.get_normal();
    let normal = [n.x, n.y, n.z];
    let m = tri.get_midpoint();
    let origin = [m.x, m.y, m.z];
    let mut dir = normal;
    let total = 3;
    let mut count = 0;
    let mut min_distance = f64::INFINITY;
    for (i, nudge) in NUDGES.iter().enumerate() {
        for k in 0..3 {
            dir[k] += nudge[k] * JITTER_EPSILON;
        }
        dir = dir.map(|v| -v);
        let hit = bvh.raycast_first_js(origin, dir, DOUBLE_SIDE, 0.0, f64::INFINITY);
        if hit.is_some_and(|h| dot(dir, h.face_normal) > 0.0) {
            count += 1;
        }
        if let Some(h) = hit {
            min_distance = min_distance.min(h.distance);
        }
        if min_distance <= OFFSET_EPSILON {
            // As upstream: the latest hit's face decides.
            let face = hit.map_or([0.0; 3], |h| h.face_normal);
            return if dot(face, normal) > 0.0 {
                COPLANAR_ALIGNED
            } else {
                COPLANAR_OPPOSITE
            };
        }
        if count as f64 / total as f64 > 0.5 || (i as f64 - count as f64 + 1.0) / total as f64 > 0.5
        {
            break;
        }
    }
    if count as f64 / total as f64 > 0.5 {
        BACK_SIDE
    } else {
        FRONT_SIDE
    }
}

pub fn get_operation_action(operation: u8, hit_side: i8, invert: bool) -> u8 {
    use super::constants::*;
    match operation {
        ADDITION => {
            if hit_side == FRONT_SIDE || (hit_side == COPLANAR_ALIGNED && !invert) {
                ADD_TRI
            } else {
                SKIP_TRI
            }
        }
        SUBTRACTION => {
            if invert {
                if hit_side == BACK_SIDE {
                    INVERT_TRI
                } else {
                    SKIP_TRI
                }
            } else if hit_side == FRONT_SIDE || hit_side == COPLANAR_OPPOSITE {
                ADD_TRI
            } else {
                SKIP_TRI
            }
        }
        REVERSE_SUBTRACTION => {
            if invert {
                if hit_side == FRONT_SIDE || hit_side == COPLANAR_OPPOSITE {
                    ADD_TRI
                } else {
                    SKIP_TRI
                }
            } else if hit_side == BACK_SIDE {
                INVERT_TRI
            } else {
                SKIP_TRI
            }
        }
        DIFFERENCE => match hit_side {
            BACK_SIDE => INVERT_TRI,
            FRONT_SIDE => ADD_TRI,
            _ => SKIP_TRI,
        },
        INTERSECTION => {
            if hit_side == BACK_SIDE || (hit_side == COPLANAR_ALIGNED && !invert) {
                ADD_TRI
            } else {
                SKIP_TRI
            }
        }
        HOLLOW_SUBTRACTION => {
            if !invert && (hit_side == FRONT_SIDE || hit_side == COPLANAR_OPPOSITE) {
                ADD_TRI
            } else {
                SKIP_TRI
            }
        }
        HOLLOW_INTERSECTION => {
            if !invert && (hit_side == BACK_SIDE || hit_side == COPLANAR_ALIGNED) {
                ADD_TRI
            } else {
                SKIP_TRI
            }
        }
        _ => SKIP_TRI,
    }
}

pub fn matrix_b_into_a(a: &Matrix4, b: &Matrix4) -> Matrix4 {
    a.invert().multiply(b)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::csg::brush::CsgBrush;
    use crate::csg::constants::SUBTRACTION;
    use crate::geometries::BoxGeometry;
    use crate::math::Triangle;

    #[test]
    fn inner_box_triangles_are_back_side_in_outer_bvh() {
        let mut outer = CsgBrush::new(BoxGeometry::new(4.0, 2.5, 2.5));
        let mut inner = CsgBrush::new(BoxGeometry::new(3.6, 2.1, 2.1));
        outer.prepare_geometry();
        inner.prepare_geometry();
        let pos = inner.geometry.get_attribute("position").unwrap();
        let tri = Triangle::new(
            crate::csg::geometry_prep::read_position(pos, 0),
            crate::csg::geometry_prep::read_position(pos, 1),
            crate::csg::geometry_prep::read_position(pos, 2),
        );
        let hit_side = get_hit_side(&tri, &outer.bounds_tree());
        let action = get_operation_action(SUBTRACTION, hit_side, true);
        assert_eq!(hit_side, BACK_SIDE);
        assert_eq!(action, INVERT_TRI);
    }
}
