use crate::core::{BufferAttribute, BufferGeometry};
use std::f32::consts::PI;

pub struct CylinderGeometry;

impl CylinderGeometry {
    /// Open/closed truncated cone along the Y axis. Matches three.js's
    /// `CylinderGeometry(radiusTop, radiusBottom, height, radialSegments,
    /// heightSegments, openEnded, thetaStart, thetaLength)`.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        radius_top: f32,
        radius_bottom: f32,
        height: f32,
        radial_segments: usize,
        height_segments: usize,
        open_ended: bool,
        theta_start: f32,
        theta_length: f32,
    ) -> BufferGeometry {
        let rs = radial_segments.max(3);
        let hs = height_segments.max(1);
        let half_h = height * 0.5;

        let mut positions: Vec<f32> = Vec::new();
        let mut normals: Vec<f32> = Vec::new();
        let mut uvs: Vec<f32> = Vec::new();
        let mut indices: Vec<u32> = Vec::new();

        // Slope for side normal in the YZ projection.
        let slope = (radius_bottom - radius_top) / height;

        // -- side --
        let mut index_array: Vec<Vec<u32>> = Vec::with_capacity(hs + 1);
        let mut idx = 0u32;
        for y in 0..=hs {
            let mut row = Vec::with_capacity(rs + 1);
            let v = y as f32 / hs as f32;
            let radius = v * (radius_bottom - radius_top) + radius_top;
            for x in 0..=rs {
                let u = x as f32 / rs as f32;
                let theta = u * theta_length + theta_start;
                let (s, c) = theta.sin_cos();
                positions.extend_from_slice(&[radius * s, -v * height + half_h, radius * c]);
                let n_len = (s * s + slope * slope + c * c).sqrt();
                normals.extend_from_slice(&[s / n_len, slope / n_len, c / n_len]);
                uvs.extend_from_slice(&[u, 1.0 - v]);
                row.push(idx);
                idx += 1;
            }
            index_array.push(row);
        }
        for x in 0..rs {
            for y in 0..hs {
                let a = index_array[y][x];
                let b = index_array[y + 1][x];
                let c = index_array[y + 1][x + 1];
                let d = index_array[y][x + 1];
                indices.extend_from_slice(&[a, b, d]);
                indices.extend_from_slice(&[b, c, d]);
            }
        }

        // Recorded here rather than recomputed later: only this function knows
        // how many triangles each part emitted, and the cap rules (skipped when
        // a radius is zero, or when open-ended) would otherwise need a second
        // copy in the tagging code.
        let side_triangles = indices.len() / 3;
        let mut top_cap_triangles = 0usize;
        // Read only by the `brep` tagging below, same as the two above.
        #[cfg_attr(not(feature = "brep"), allow(unused_variables, unused_assignments))]
        let mut bottom_cap_triangles = 0usize;

        // -- caps --
        if !open_ended {
            if radius_top > 0.0 {
                Self::cap(
                    true,
                    radius_top,
                    half_h,
                    rs,
                    theta_start,
                    theta_length,
                    &mut positions,
                    &mut normals,
                    &mut uvs,
                    &mut indices,
                    &mut idx,
                );
                top_cap_triangles = indices.len() / 3 - side_triangles;
            }
            if radius_bottom > 0.0 {
                Self::cap(
                    false,
                    radius_bottom,
                    -half_h,
                    rs,
                    theta_start,
                    theta_length,
                    &mut positions,
                    &mut normals,
                    &mut uvs,
                    &mut indices,
                    &mut idx,
                );
                #[cfg_attr(not(feature = "brep"), allow(unused_assignments))]
                {
                    bottom_cap_triangles = indices.len() / 3 - side_triangles - top_cap_triangles;
                }
            }
        }

        let mut geom = BufferGeometry::new();
        geom.set_attribute("position", BufferAttribute::new(positions, 3));
        geom.set_attribute("normal", BufferAttribute::new(normals, 3));
        geom.set_attribute("uv", BufferAttribute::new(uvs, 2));
        geom.set_index(indices);
        #[cfg(feature = "brep")]
        crate::brep::primitives::tag_cylinder(
            &mut geom,
            radius_top,
            radius_bottom,
            height,
            side_triangles,
            top_cap_triangles,
            bottom_cap_triangles,
        );
        geom
    }

    #[allow(clippy::too_many_arguments)]
    fn cap(
        top: bool,
        radius: f32,
        y: f32,
        radial_segments: usize,
        theta_start: f32,
        theta_length: f32,
        positions: &mut Vec<f32>,
        normals: &mut Vec<f32>,
        uvs: &mut Vec<f32>,
        indices: &mut Vec<u32>,
        idx: &mut u32,
    ) {
        let sign = if top { 1.0 } else { -1.0 };
        let center = *idx;
        positions.extend_from_slice(&[0.0, y, 0.0]);
        normals.extend_from_slice(&[0.0, sign, 0.0]);
        uvs.extend_from_slice(&[0.5, 0.5]);
        *idx += 1;

        let first_rim = *idx;
        for x in 0..=radial_segments {
            let u = x as f32 / radial_segments as f32;
            let theta = u * theta_length + theta_start;
            let (s, c) = theta.sin_cos();
            positions.extend_from_slice(&[radius * s, y, radius * c]);
            normals.extend_from_slice(&[0.0, sign, 0.0]);
            uvs.extend_from_slice(&[s * 0.5 + 0.5, c * 0.5 * sign + 0.5]);
            *idx += 1;
        }
        for x in 0..radial_segments as u32 {
            let a = first_rim + x;
            let b = a + 1;
            if top {
                indices.extend_from_slice(&[a, b, center]);
            } else {
                indices.extend_from_slice(&[b, a, center]);
            }
        }
    }

    pub fn default_(radius: f32, height: f32) -> BufferGeometry {
        Self::new(radius, radius, height, 32, 1, false, 0.0, PI * 2.0)
    }
}
