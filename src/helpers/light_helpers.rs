use super::three_lines::{self, HelperGeometry, IDENTITY};
use crate::core::{BufferAttribute, BufferGeometry, LineSegments, Object3D};
use crate::lights::{DirectionalLight, HemisphereLight, PointLight, SpotLight};
use crate::materials::LineBasicMaterial;
use crate::math::{Color, Vector3};

/// A line-segments object drawing `g`'s lines, in `color` (times the
/// per-point colours when it has them).
pub(crate) fn line_object(g: &HelperGeometry, color: Color) -> Object3D {
    let mut geometry = BufferGeometry::new();
    geometry.set_attribute("position", BufferAttribute::new(g.lines.clone(), 3));
    if !g.colors.is_empty() {
        geometry.set_attribute("color", BufferAttribute::new(g.colors.clone(), 3));
    }
    // three.js helpers are `toneMapped: false`.
    let material = LineBasicMaterial { tone_mapped: false, ..LineBasicMaterial::new(color) };
    Object3D::line_segments(LineSegments::new(geometry, material.into()))
}

fn dir3(v: Vector3) -> [f64; 3] {
    [v.x as f64, v.y as f64, v.z as f64]
}

/// three.js's `DirectionalLightHelper`: a square facing the target and a line
/// to it, drawn in the light's frame (the light at the origin, the target
/// along `light.direction`).
pub struct DirectionalLightHelper;

impl DirectionalLightHelper {
    pub fn new(light: &DirectionalLight, size: f32) -> Object3D {
        let g = three_lines::directional_light(&IDENTITY, dir3(light.direction), size as f64);
        line_object(&g, light.color)
    }
}

/// three.js's `PointLightHelper`: a 4×2 sphere wireframe of radius `size`
/// at the light.
pub struct PointLightHelper;

impl PointLightHelper {
    pub fn new(light: &PointLight, size: f32) -> Object3D {
        line_object(&three_lines::point_light(&IDENTITY, size as f64), light.color)
    }
}

/// three.js's `SpotLightHelper`: five spokes and a rim, as long as the
/// light's `distance` (`length` when that is 0 — three.js uses 1000) and as
/// wide as its `angle`, pointing along `light.direction`.
pub struct SpotLightHelper;

impl SpotLightHelper {
    pub fn new(light: &SpotLight, length: f32) -> Object3D {
        let distance = if light.distance != 0.0 { light.distance } else { length };
        let g = three_lines::spot_light(
            &IDENTITY,
            dir3(light.direction),
            distance as f64,
            light.angle as f64,
        );
        line_object(&g, light.color)
    }
}

/// three.js's `HemisphereLightHelper`: an octahedron wireframe, sky colour
/// on its upper vertices and ground colour on the lower.
pub struct HemisphereLightHelper;

impl HemisphereLightHelper {
    pub fn new(light: &HemisphereLight, size: f32) -> Object3D {
        let c = |c: Color| [c.r, c.g, c.b];
        let g = three_lines::hemisphere_light(
            &IDENTITY,
            size as f64,
            c(light.sky_color),
            c(light.ground_color),
        );
        line_object(&g, Color::WHITE)
    }
}

/// three.js's `SkeletonHelper`: a blue-to-green line from each bone to its
/// parent bone.
pub struct SkeletonHelper;

impl SkeletonHelper {
    /// Bones as world positions with the index of their parent bone, if any.
    pub fn from_world_positions(bones_world: &[(Vector3, Option<usize>)]) -> Object3D {
        let matrices: Vec<three_lines::Mat4> = bones_world
            .iter()
            .map(|(p, _)| {
                let mut m = IDENTITY;
                m[12] = p.x as f64;
                m[13] = p.y as f64;
                m[14] = p.z as f64;
                m
            })
            .collect();
        let parents: Vec<i32> = bones_world
            .iter()
            .map(|(_, parent)| parent.map_or(-1, |p| p as i32))
            .collect();
        line_object(&three_lines::skeleton(&IDENTITY, &matrices, &parents), Color::WHITE)
    }
}
