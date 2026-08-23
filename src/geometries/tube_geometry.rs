use crate::core::{BufferAttribute, BufferGeometry};
use crate::curves::Curve3;
use crate::math::Vector3;
use std::f32::consts::PI;

pub struct TubeGeometry;

impl TubeGeometry {
    /// Extrude a circular cross-section of `radius` along `path`. Uses
    /// parallel-transport frames (approximation of three.js's Frenet frames).
    pub fn new(
        path: &dyn Curve3,
        tubular_segments: usize,
        radius: f32,
        radial_segments: usize,
        closed: bool,
    ) -> BufferGeometry {
        let ts = tubular_segments.max(2);
        let rs = radial_segments.max(3);

        let centers = path.get_points(ts);
        let frames = compute_frames(path, ts);

        let mut positions = Vec::with_capacity((ts + 1) * (rs + 1) * 3);
        let mut normals = Vec::with_capacity((ts + 1) * (rs + 1) * 3);
        let mut uvs = Vec::with_capacity((ts + 1) * (rs + 1) * 2);
        let mut indices = Vec::new();

        for i in 0..=ts {
            let p = centers[i];
            let (normal_axis, binormal_axis) = (frames[i].0, frames[i].1);
            for j in 0..=rs {
                let v = j as f32 / rs as f32 * PI * 2.0;
                let (sv, cv) = v.sin_cos();
                let dir = normal_axis * (-radius * cv) + binormal_axis * (-radius * sv);
                let vtx = p + dir;
                positions.extend_from_slice(&[vtx.x, vtx.y, vtx.z]);
                let n = dir.normalize();
                normals.extend_from_slice(&[n.x, n.y, n.z]);
                uvs.extend_from_slice(&[i as f32 / ts as f32, j as f32 / rs as f32]);
            }
        }

        let row = (rs + 1) as u32;
        let last_i = if closed { ts } else { ts - 1 };
        for i in 0..last_i {
            let i = i as u32;
            for j in 0..rs as u32 {
                let a = i * row + j;
                let b = (i + 1) * row + j;
                let c = (i + 1) * row + j + 1;
                let d = i * row + j + 1;
                indices.extend_from_slice(&[a, b, d]);
                indices.extend_from_slice(&[b, c, d]);
            }
        }

        let mut g = BufferGeometry::new();
        g.set_attribute("position", BufferAttribute::new(positions, 3));
        g.set_attribute("normal", BufferAttribute::new(normals, 3));
        g.set_attribute("uv", BufferAttribute::new(uvs, 2));
        g.set_index(indices);
        #[cfg(feature = "brep")]
        crate::brep::primitives::tag_tube(&mut g, &centers, &frames, radius);
        g
    }
}

/// Parallel-transport frames: returns (normal, binormal) per sample. Tangent
/// recomputed from `Curve3::get_tangent`.
fn compute_frames(path: &dyn Curve3, ts: usize) -> Vec<(Vector3, Vector3)> {
    let mut tangents: Vec<Vector3> = Vec::with_capacity(ts + 1);
    for i in 0..=ts {
        tangents.push(path.get_tangent(i as f32 / ts as f32));
    }
    // Initial normal: pick any vector not parallel to tangent.
    let t0 = tangents[0];
    let helper = if t0.y.abs() < 0.999 {
        Vector3::new(0.0, 1.0, 0.0)
    } else {
        Vector3::new(1.0, 0.0, 0.0)
    };
    let mut normal = t0.cross(helper).normalize();
    let mut binormal = t0.cross(normal).normalize();
    let mut frames = Vec::with_capacity(ts + 1);
    frames.push((normal, binormal));
    for i in 1..=ts {
        let t = tangents[i];
        let prev_t = tangents[i - 1];
        let axis = prev_t.cross(t);
        let axis_len = axis.length();
        if axis_len > 1e-6 {
            let axis_n = axis * (1.0 / axis_len);
            let dot = prev_t.dot(t).clamp(-1.0, 1.0);
            let angle = dot.acos();
            normal = rotate_axis_angle(normal, axis_n, angle);
        }
        binormal = t.cross(normal).normalize();
        frames.push((normal, binormal));
    }
    frames
}

fn rotate_axis_angle(v: Vector3, axis: Vector3, angle: f32) -> Vector3 {
    let (s, c) = angle.sin_cos();
    v * c + axis.cross(v) * s + axis * (axis.dot(v) * (1.0 - c))
}
