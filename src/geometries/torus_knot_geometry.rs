use crate::core::{BufferAttribute, BufferGeometry};

pub struct TorusKnotGeometry;

impl TorusKnotGeometry {
    /// (p, q)-torus knot. p windings around the symmetry axis; q around the tube.
    pub fn new(
        radius: f32,
        tube: f32,
        tubular_segments: usize,
        radial_segments: usize,
        p: u32,
        q: u32,
    ) -> BufferGeometry {
        Self::from_f64(
            super::js_number(radius),
            super::js_number(tube),
            tubular_segments,
            radial_segments,
            p,
            q,
        )
    }

    /// [`new`](Self::new) with JavaScript's numbers, as three.js takes them.
    pub fn from_f64(
        radius: f64,
        tube: f64,
        tubular_segments: usize,
        radial_segments: usize,
        p: u32,
        q: u32,
    ) -> BufferGeometry {
        // three.js floors the counts and clamps neither.
        let ts = tubular_segments.max(1);
        let rs = radial_segments.max(1);

        let mut positions = Vec::new();
        let mut normals = Vec::new();
        let mut uvs = Vec::new();
        let mut indices = Vec::new();

        // In f64 from the caller's numbers, rounded once, as three.js does.
        type V = [f64; 3];
        let r = radius;
        let (pf, qf) = (p as f64, q as f64);
        // `calculatePositionOnCurve`.
        let pos_on_curve = |u: f64| -> V {
            let (cu, su) = (u.cos(), u.sin());
            let qu_over_p = qf / pf * u;
            let cs = qu_over_p.cos();
            [
                r * (2.0 + cs) * 0.5 * cu,
                r * (2.0 + cs) * su * 0.5,
                r * qu_over_p.sin() * 0.5,
            ]
        };
        let sub = |a: V, b: V| -> V { [a[0] - b[0], a[1] - b[1], a[2] - b[2]] };
        let add = |a: V, b: V| -> V { [a[0] + b[0], a[1] + b[1], a[2] + b[2]] };
        let cross = |a: V, b: V| -> V {
            [
                a[1] * b[2] - a[2] * b[1],
                a[2] * b[0] - a[0] * b[2],
                a[0] * b[1] - a[1] * b[0],
            ]
        };
        let normalize = |a: V| -> V {
            let len = (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt();
            let inv = 1.0 / if len == 0.0 { 1.0 } else { len };
            [a[0] * inv, a[1] * inv, a[2] * inv]
        };
        for i in 0..=ts {
            let u = i as f64 / ts as f64 * pf * std::f64::consts::PI * 2.0;
            let p1 = pos_on_curve(u);
            let p2 = pos_on_curve(u + 0.01);
            // three.js's basis: T unnormalised, B = T × (P1 + P2), N = B × T.
            let t = sub(p2, p1);
            let n = add(p2, p1);
            let b = cross(t, n);
            let n = cross(b, t);
            let (b, n) = (normalize(b), normalize(n));
            for j in 0..=rs {
                let v = j as f64 / rs as f64 * std::f64::consts::PI * 2.0;
                let cx = -tube * v.cos();
                let cy = tube * v.sin();
                let pos = [
                    p1[0] + (cx * n[0] + cy * b[0]),
                    p1[1] + (cx * n[1] + cy * b[1]),
                    p1[2] + (cx * n[2] + cy * b[2]),
                ];
                positions.extend_from_slice(&pos.map(|c| c as f32));
                normals.extend_from_slice(&normalize(sub(pos, p1)).map(|c| c as f32));
                uvs.extend_from_slice(&[
                    (i as f64 / ts as f64) as f32,
                    (j as f64 / rs as f64) as f32,
                ]);
            }
        }

        let row = (rs + 1) as u32;
        let mut polygons = super::polygons::CellPolygons::new();
        for i in 1..=ts as u32 {
            for j in 1..=rs as u32 {
                let a = (row * (i - 1)) + j - 1;
                let b = (row * i) + j - 1;
                let c = (row * i) + j;
                let d = (row * (i - 1)) + j;
                indices.extend_from_slice(&[a, b, d]);
                indices.extend_from_slice(&[b, c, d]);
                polygons.quad(&positions, [a, b, c, d]);
            }
        }

        let mut geom = BufferGeometry::new();
        geom.set_attribute("position", BufferAttribute::new(positions, 3));
        geom.set_attribute("normal", BufferAttribute::new(normals, 3));
        geom.set_attribute("uv", BufferAttribute::new(uvs, 2));
        geom.set_index(indices);
        polygons.apply(&mut geom);
        geom
    }

    pub fn default_(radius: f32, tube: f32) -> BufferGeometry {
        Self::new(radius, tube, 64, 8, 2, 3)
    }
}
