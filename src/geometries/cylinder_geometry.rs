use crate::core::BufferGeometry;
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
        use super::{js_angle, js_number};
        Self::from_f64(
            js_number(radius_top),
            js_number(radius_bottom),
            js_number(height),
            radial_segments,
            height_segments,
            open_ended,
            js_angle(theta_start),
            js_angle(theta_length),
        )
    }

    /// [`new`](Self::new) with JavaScript's numbers, as three.js takes them.
    #[allow(clippy::too_many_arguments)]
    pub fn from_f64(
        radius_top: f64,
        radius_bottom: f64,
        height: f64,
        radial_segments: usize,
        height_segments: usize,
        open_ended: bool,
        theta_start: f64,
        theta_length: f64,
    ) -> BufferGeometry {
        Self::build(
            radius_top,
            radius_bottom,
            height,
            radial_segments,
            height_segments,
            open_ended,
            theta_start,
            theta_length,
            false,
        )
    }

    /// The same mesh with its positions computed in `f32`, as threers did
    /// before [`new`](Self::new) followed three.js's `f64` arithmetic.
    ///
    /// For the exact CSG kernel (OpenSCAD's `cylinder`): which booleans it can
    /// resolve exactly depends on the operands' last bits, and its suite was
    /// built on these. Two identical cylinders offset along their axis resolve
    /// exactly from these positions and defer to the float kernel from the
    /// `f64`-rounded ones. Normals and uvs are `new`'s.
    #[allow(clippy::too_many_arguments)]
    pub fn single_precision(
        radius_top: f32,
        radius_bottom: f32,
        height: f32,
        radial_segments: usize,
        height_segments: usize,
        open_ended: bool,
        theta_start: f32,
        theta_length: f32,
    ) -> BufferGeometry {
        Self::build(
            radius_top as f64,
            radius_bottom as f64,
            height as f64,
            radial_segments,
            height_segments,
            open_ended,
            theta_start as f64,
            theta_length as f64,
            true,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn build(
        radius_top: f64,
        radius_bottom: f64,
        height: f64,
        radial_segments: usize,
        height_segments: usize,
        open_ended: bool,
        theta_start: f64,
        theta_length: f64,
        single: bool,
    ) -> BufferGeometry {
        use super::JsBuffers;
        // A position on the side or a rim, at fraction `u` around and radius
        // `radius`, height `y` — in f32 for `single_precision`.
        let at = |u: f64, radius: f64, y: f64| -> [f64; 3] {
            if single {
                let (s, c) = ((u as f32) * (theta_length as f32) + (theta_start as f32)).sin_cos();
                let r = radius as f32;
                [(r * s) as f64, (y as f32) as f64, (r * c) as f64]
            } else {
                let theta = u * theta_length + theta_start;
                [radius * theta.sin(), y, radius * theta.cos()]
            }
        };
        let rs = radial_segments.max(1);
        let hs = height_segments.max(1);
        // In f64 from the caller's numbers, as three.js computes them.
        let (rt, rb, h) = (radius_top, radius_bottom, height);
        let (ts, tl) = (theta_start, theta_length);
        let half_h = h / 2.0;
        let mut out = JsBuffers::default();
        let mut polygons = super::polygons::CellPolygons::new();

        // `generateTorso`.
        let slope = (rb - rt) / h;
        let mut rows: Vec<Vec<u32>> = Vec::with_capacity(hs + 1);
        for y in 0..=hs {
            let v = y as f64 / hs as f64;
            let radius = v * (rb - rt) + rt;
            let mut row = Vec::with_capacity(rs + 1);
            for x in 0..=rs {
                let u = x as f64 / rs as f64;
                let theta = u * tl + ts;
                let (sin, cos) = (theta.sin(), theta.cos());
                row.push(out.vertex_count());
                let p = if single {
                    // As the old generator: the radius and height in f32 too.
                    let (vf, rtf, rbf, hf) = (v as f32, rt as f32, rb as f32, h as f32);
                    at(
                        u,
                        (vf * (rbf - rtf) + rtf) as f64,
                        (-vf * hf + hf * 0.5) as f64,
                    )
                } else {
                    [radius * sin, -v * h + half_h, radius * cos]
                };
                out.vertex(p[0], p[1], p[2]);
                out.normal_normalized(sin, slope, cos);
                out.uv(u, 1.0 - v);
            }
            rows.push(row);
        }
        for x in 0..rs {
            for y in 0..hs {
                let (a, b, c, d) = (
                    rows[y][x],
                    rows[y + 1][x],
                    rows[y + 1][x + 1],
                    rows[y][x + 1],
                );
                out.indices.extend_from_slice(&[a, b, d, b, c, d]);
                polygons.quad(&out.positions, [a, b, c, d]);
            }
        }
        let side_triangles = out.indices.len() / 3;
        out.groups.push((0, out.indices.len() as u32, 0));

        // `generateCap`: a centre vertex per segment, then the rim.
        let cap = |out: &mut JsBuffers, top: bool| {
            let start = out.indices.len() as u32;
            let radius = if top { rt } else { rb };
            let sign = if top { 1.0 } else { -1.0 };
            let centre_start = out.vertex_count();
            for _ in 0..rs {
                out.vertex(0.0, half_h * sign, 0.0);
                out.normal(0.0, sign, 0.0);
                out.uv(0.5, 0.5);
            }
            let centre_end = out.vertex_count();
            for x in 0..=rs {
                let u = x as f64 / rs as f64;
                let theta = u * tl + ts;
                let (cos, sin) = (theta.cos(), theta.sin());
                let p = if single {
                    at(u, radius, ((h as f32) * 0.5 * sign as f32) as f64)
                } else {
                    [radius * sin, half_h * sign, radius * cos]
                };
                out.vertex(p[0], p[1], p[2]);
                out.normal(0.0, sign, 0.0);
                out.uv(cos * 0.5 + 0.5, sin * 0.5 * sign + 0.5);
            }
            for x in 0..rs as u32 {
                let (c, i) = (centre_start + x, centre_end + x);
                if top {
                    out.indices.extend_from_slice(&[i, i + 1, c]);
                } else {
                    out.indices.extend_from_slice(&[i + 1, i, c]);
                }
            }
            let count = out.indices.len() as u32 - start;
            out.groups.push((start, count, if top { 1 } else { 2 }));
            (count / 3) as usize
        };
        let (mut top_cap_triangles, mut bottom_cap_triangles) = (0, 0);
        if !open_ended {
            if rt > 0.0 {
                top_cap_triangles = cap(&mut out, true);
            }
            if rb > 0.0 {
                bottom_cap_triangles = cap(&mut out, false);
            }
        }
        #[cfg(not(feature = "brep"))]
        let _ = (top_cap_triangles, bottom_cap_triangles);

        let mut geom = out.into_geometry();
        // Side cells are quads (a cone's tip row collapses to triangles). The
        // caps are fanned from centre vertices, so stay triangles.
        polygons.triangles(geom.triangle_count() - side_triangles);
        polygons.apply(&mut geom);
        #[cfg(feature = "brep")]
        crate::brep::primitives::tag_cylinder(
            &mut geom,
            radius_top as f32,
            radius_bottom as f32,
            height as f32,
            side_triangles,
            top_cap_triangles,
            bottom_cap_triangles,
        );
        geom
    }

    pub fn default_(radius: f32, height: f32) -> BufferGeometry {
        Self::new(radius, radius, height, 32, 1, false, 0.0, PI * 2.0)
    }
}
