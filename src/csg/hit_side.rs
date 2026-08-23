#[cfg(test)]
use crate::math::Triangle;
use crate::math::{Matrix4, Ray, Vector3};
use crate::mesh_bvh::MeshBvh;

use super::constants::*;
use super::js_topology::JsTriangle;

const JITTER_EPSILON: f32 = 1e-8;
const OFFSET_EPSILON: f32 = 1e-15;

#[cfg(test)]
pub fn get_hit_side(tri: &Triangle, bvh: &MeshBvh) -> i8 {
    get_hit_side_ray(tri.midpoint(), tri.normal(), bvh)
}

/// Hit-side from JS-topology splitter output (f64 pool triangle).
pub fn get_hit_side_js(tri: JsTriangle, bvh: &MeshBvh) -> i8 {
    get_hit_side_ray(tri.get_midpoint().to_v(), tri.get_normal().to_v(), bvh)
}

fn get_hit_side_ray(origin: Vector3, direction: Vector3, bvh: &MeshBvh) -> i8 {
    let ray = Ray::new(origin, direction);
    let hit = bvh.raycast_first_with_side(&ray, 0.0, f32::INFINITY, DOUBLE_SIDE);
    let hit_back = hit
        .map(|h| {
            let face = bvh.triangle_at_geometry_index(h.face_index);
            ray.direction.dot(face.normal()) > 0.0
        })
        .unwrap_or(false);
    if hit_back {
        BACK_SIDE
    } else {
        FRONT_SIDE
    }
}

/// Deterministic coplanar hit-side (JS `getHitSideWithCoplanarCheck`).
pub fn get_hit_side_with_coplanar_check_js(tri: JsTriangle, bvh: &MeshBvh) -> i8 {
    get_hit_side_with_coplanar_check_ray(tri.get_midpoint().to_v(), tri.get_normal().to_v(), bvh)
}

fn get_hit_side_with_coplanar_check_ray(origin: Vector3, normal: Vector3, bvh: &MeshBvh) -> i8 {
    let jitters = [
        Vector3::new(0.25, -0.5, 0.5),
        Vector3::new(-0.5, 0.25, -0.25),
        Vector3::new(0.5, 0.5, -0.5),
    ];
    let total = 3;
    let mut count = 0;
    let mut min_distance = f32::INFINITY;
    for (i, &jitter) in jitters.iter().take(total).enumerate() {
        let mut direction = normal + jitter * JITTER_EPSILON;
        direction = -direction;
        let ray = Ray::new(origin, direction);
        let hit = bvh.raycast_first_with_side(&ray, 0.0, f32::INFINITY, DOUBLE_SIDE);
        if let Some(h) = hit {
            let face = bvh.triangle_at_geometry_index(h.face_index);
            if ray.direction.dot(face.normal()) > 0.0 {
                count += 1;
            }
            min_distance = min_distance.min(h.distance);
            if min_distance <= OFFSET_EPSILON {
                return if face.normal().dot(normal) > 0.0 {
                    COPLANAR_ALIGNED
                } else {
                    COPLANAR_OPPOSITE
                };
            }
        }
        if count as f32 / total as f32 > 0.5 || (i as i32 - count + 1) as f32 / total as f32 > 0.5 {
            break;
        }
    }
    if count as f32 / total as f32 > 0.5 {
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
