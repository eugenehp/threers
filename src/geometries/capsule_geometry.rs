use crate::core::BufferGeometry;
use crate::geometries::LatheGeometry;
use std::f64::consts::PI;

pub struct CapsuleGeometry;

impl CapsuleGeometry {
    /// Cylinder along Y with hemispherical end caps. Matches three.js r165's
    /// `CapsuleGeometry(radius, length, capSegments, radialSegments)`. `length`
    /// is the cylinder length between the cap centers.
    ///
    /// Built the way three.js builds it: a [`LatheGeometry`] of the path
    /// `absarc(0, -length/2, radius, 3π/2, 0)` + `absarc(0, length/2, radius, 0, π/2)`
    /// sampled by `Path.getPoints(capSegments)`. That samples each arc with
    /// `2 * capSegments` divisions (`CurvePath.getPoints` doubles the
    /// resolution of ellipse curves), so the profile runs bottom pole to top
    /// pole through `4 * capSegments + 2` points and the buffers come out in
    /// three.js's order.
    pub fn new(
        radius: f32,
        length: f32,
        cap_segments: usize,
        radial_segments: usize,
    ) -> BufferGeometry {
        Self::with_f64(
            super::js_number(radius),
            super::js_number(length),
            cap_segments,
            radial_segments,
        )
    }

    /// [`new`](Self::new) with JavaScript's `f64` parameters, for bit-exact
    /// r165 buffers from the browser bindings.
    pub fn with_f64(
        radius: f64,
        length: f64,
        cap_segments: usize,
        radial_segments: usize,
    ) -> BufferGeometry {
        let cs = cap_segments.max(1);
        let rs = radial_segments.max(3);
        LatheGeometry::with_f64(&profile(radius, length, cs), rs, 0.0, PI * 2.0)
    }

    pub fn default_(radius: f32, length: f32) -> BufferGeometry {
        Self::new(radius, length, 4, 16)
    }
}

/// three.js `EllipseCurve.getPoint(t)` for a counter-clockwise, unrotated
/// arc of radius `r` about `(ax, ay)` from `start` to `end`: the sweep
/// derived as three.js derives it (wrapped into `[0, 2π]`), not taken as a
/// constant — the two differ in the last bit.
fn ellipse_point(ax: f64, ay: f64, r: f64, start: f64, end: f64, t: f64) -> (f64, f64) {
    let two_pi = PI * 2.0;
    let mut delta = end - start;
    let same = delta.abs() < f64::EPSILON;
    while delta < 0.0 {
        delta += two_pi;
    }
    while delta > two_pi {
        delta -= two_pi;
    }
    if delta < f64::EPSILON {
        delta = if same { 0.0 } else { two_pi };
    }
    let angle = start + t * delta;
    (ax + r * angle.cos(), ay + r * angle.sin())
}

/// The lathe profile three.js's `CapsuleGeometry` builds:
/// `absarc(0, −length/2, radius, 1.5π, 0)`, `absarc(0, length/2, radius, 0,
/// 0.5π)` and `Path.getPoints(capSegments)` — each arc sampled at
/// `2·capSegments` divisions, the joining `lineTo`, and consecutive exact
/// duplicates dropped, as `CurvePath` does.
fn profile(radius: f64, length: f64, cs: usize) -> Vec<[f64; 2]> {
    let divisions = 2 * cs;
    let (y0, y1) = (-length / 2.0, length / 2.0);
    let mut points: Vec<(f64, f64)> = Vec::with_capacity(2 * divisions + 2);
    let mut push = |p: (f64, f64)| {
        if points.last() != Some(&p) {
            points.push(p);
        }
    };
    let arc1 =
        |d: usize| ellipse_point(0.0, y0, radius, PI * 1.5, 0.0, d as f64 / divisions as f64);
    let arc2 =
        |d: usize| ellipse_point(0.0, y1, radius, 0.0, PI * 0.5, d as f64 / divisions as f64);
    (0..=divisions).for_each(|d| push(arc1(d)));
    // `absellipse` joins the second arc with a `lineTo` from the first's end
    // (a duplicate, dropped) to the second's start.
    push(arc2(0));
    (0..=divisions).for_each(|d| push(arc2(d)));
    points.into_iter().map(|(x, y)| [x, y]).collect()
}

#[cfg(test)]
#[allow(clippy::excessive_precision)] // r165's buffers, digit for digit as dumped
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn attr(g: &BufferGeometry, name: &str) -> Vec<f32> {
        g.get_attribute(name).expect(name).array.clone()
    }

    /// `new THREE.CapsuleGeometry(0.5, 1, 1, 4)` from three.js r165, dumped by
    /// node from `three@0.165.0/build/three.module.js`.
    #[rustfmt::skip]
    const R165_POS: [f32; 90] = [0.00000000, -1.00000000, -9.18485073e-17, 0.00000000, -0.853553414, 0.353553385, 0.00000000, -0.500000000, 0.500000000, 0.00000000, 0.500000000, 0.500000000, 0.00000000, 0.853553414, 0.353553385, 0.00000000, 1.00000000, 3.06161713e-17, -9.18485073e-17, -1.00000000, -5.62409909e-33, 0.353553385, -0.853553414, 2.16489015e-17, 0.500000000, -0.500000000, 3.06161713e-17, 0.500000000, 0.500000000, 3.06161713e-17, 0.353553385, 0.853553414, 2.16489015e-17, 3.06161713e-17, 1.00000000, 1.87469970e-33, -1.12481982e-32, -1.00000000, 9.18485073e-17, 4.32978030e-17, -0.853553414, -0.353553385, 6.12323426e-17, -0.500000000, -0.500000000, 6.12323426e-17, 0.500000000, -0.500000000, 4.32978030e-17, 0.853553414, -0.353553385, 3.74939939e-33, 1.00000000, -3.06161713e-17, 9.18485073e-17, -1.00000000, 1.68722980e-32, -0.353553385, -0.853553414, -6.49467012e-17, -0.500000000, -0.500000000, -9.18485073e-17, -0.500000000, 0.500000000, -9.18485073e-17, -0.353553385, 0.853553414, -6.49467012e-17, -3.06161713e-17, 1.00000000, -5.62409909e-33, 2.24963964e-32, -1.00000000, -9.18485073e-17, -8.65956060e-17, -0.853553414, 0.353553385, -1.22464685e-16, -0.500000000, 0.500000000, -1.22464685e-16, 0.500000000, 0.500000000, -8.65956060e-17, 0.853553414, 0.353553385, -7.49879879e-33, 1.00000000, 3.06161713e-17];
    #[rustfmt::skip]
    const R165_NRM: [f32; 90] = [0.00000000, -0.923879504, 0.382683426, 0.00000000, -0.707106769, 0.707106769, 0.00000000, -0.107566431, 0.994197905, 0.00000000, 0.107566431, 0.994197905, 0.00000000, 0.707106769, 0.707106769, 0.00000000, 0.353553385, 0.146446615, 0.382683426, -0.923879504, 2.34326023e-17, 0.707106769, -0.707106769, 4.32978030e-17, 0.994197905, -0.107566431, 6.08770652e-17, 0.994197905, 0.107566431, 6.08770652e-17, 0.707106769, 0.707106769, 4.32978030e-17, 0.146446615, 0.353553385, 8.96726898e-18, 4.68652047e-17, -0.923879504, -0.382683426, 8.65956060e-17, -0.707106769, -0.707106769, 1.21754130e-16, -0.107566431, -0.994197905, 1.21754130e-16, 0.107566431, -0.994197905, 8.65956060e-17, 0.707106769, -0.707106769, 1.79345380e-17, 0.353553385, -0.146446615, -0.382683426, -0.923879504, -7.02978054e-17, -0.707106769, -0.707106769, -1.29893402e-16, -0.994197905, -0.107566431, -1.82631196e-16, -0.994197905, 0.107566431, -1.82631196e-16, -0.707106769, 0.707106769, -1.29893402e-16, -0.146446615, 0.353553385, -2.69018061e-17, -9.37304094e-17, -0.923879504, 0.382683426, -1.73191212e-16, -0.707106769, 0.707106769, -2.43508261e-16, -0.107566431, 0.994197905, -2.43508261e-16, 0.107566431, 0.994197905, -1.73191212e-16, 0.707106769, 0.707106769, -3.58690759e-17, 0.353553385, 0.146446615];
    #[rustfmt::skip]
    const R165_UV: [f32; 60] = [0.00000000, 0.00000000, 0.00000000, 0.200000003, 0.00000000, 0.400000006, 0.00000000, 0.600000024, 0.00000000, 0.800000012, 0.00000000, 1.00000000, 0.250000000, 0.00000000, 0.250000000, 0.200000003, 0.250000000, 0.400000006, 0.250000000, 0.600000024, 0.250000000, 0.800000012, 0.250000000, 1.00000000, 0.500000000, 0.00000000, 0.500000000, 0.200000003, 0.500000000, 0.400000006, 0.500000000, 0.600000024, 0.500000000, 0.800000012, 0.500000000, 1.00000000, 0.750000000, 0.00000000, 0.750000000, 0.200000003, 0.750000000, 0.400000006, 0.750000000, 0.600000024, 0.750000000, 0.800000012, 0.750000000, 1.00000000, 1.00000000, 0.00000000, 1.00000000, 0.200000003, 1.00000000, 0.400000006, 1.00000000, 0.600000024, 1.00000000, 0.800000012, 1.00000000, 1.00000000];
    #[rustfmt::skip]
    const R165_IDX: [u32; 120] = [0,6,1,7,1,6,1,7,2,8,2,7,2,8,3,9,3,8,3,9,4,10,4,9,4,10,5,11,5,10,6,12,7,13,7,12,7,13,8,14,8,13,8,14,9,15,9,14,9,15,10,16,10,15,10,16,11,17,11,16,12,18,13,19,13,18,13,19,14,20,14,19,14,20,15,21,15,20,15,21,16,22,16,21,16,22,17,23,17,22,18,24,19,25,19,24,19,25,20,26,20,25,20,26,21,27,21,26,21,27,22,28,22,27,22,28,23,29,23,28];

    fn assert_close(name: &str, got: &[f32], want: &[f32], tol: f32) {
        assert_eq!(got.len(), want.len(), "{name} length");
        for (i, (g, w)) in got.iter().zip(want).enumerate() {
            assert!(
                (g - w).abs() <= tol,
                "{name}[{i}]: got {g}, three.js r165 {w}"
            );
        }
    }

    #[test]
    fn matches_three_r165_buffers() {
        let g = CapsuleGeometry::new(0.5, 1.0, 1, 4);
        assert_close("position", &attr(&g, "position"), &R165_POS, 1e-6);
        assert_close("normal", &attr(&g, "normal"), &R165_NRM, 1e-6);
        assert_close("uv", &attr(&g, "uv"), &R165_UV, 1e-6);
        assert_eq!(g.index.as_deref().expect("indexed"), &R165_IDX[..]);
    }

    #[test]
    fn shape_and_counts_match_three_r165() {
        // three.js r165 samples each cap arc with `capSegments * 2` divisions
        // (`CurvePath.getPoints` doubles the resolution of ellipse curves), so
        // the lathe profile has 4*cs + 2 points.
        for (radius, length, cs, rs) in [(0.5, 1.0, 4, 12), (0.35, 0.9, 8, 16), (1.0, 0.0, 3, 5)] {
            let g = CapsuleGeometry::new(radius, length, cs, rs);
            let pos = attr(&g, "position");
            let profile = 4 * cs + 2;
            assert_eq!(pos.len() / 3, (rs + 1) * profile, "vertex count");
            assert_eq!(
                g.index.as_deref().unwrap().len(),
                rs * (profile - 1) * 6,
                "index count"
            );
            let half = length * 0.5;
            let (mut min_y, mut max_y) = (f32::MAX, f32::MIN);
            for p in pos.chunks_exact(3) {
                min_y = min_y.min(p[1]);
                max_y = max_y.max(p[1]);
                // Every vertex lies on the capsule: distance `radius` from the
                // axis segment between the cap centres.
                let cy = p[1].clamp(-half, half);
                let d = (p[0] * p[0] + (p[1] - cy).powi(2) + p[2] * p[2]).sqrt();
                assert!(
                    (d - radius).abs() < 1e-5,
                    "vertex {p:?} off the surface by {}",
                    d - radius
                );
            }
            assert!(
                (min_y + half + radius).abs() < 1e-6,
                "bottom pole at {min_y}"
            );
            assert!((max_y - half - radius).abs() < 1e-6, "top pole at {max_y}");
        }
    }

    #[test]
    fn closed_after_welding() {
        let g = CapsuleGeometry::new(0.5, 1.0, 4, 12);
        let pos = attr(&g, "position");
        let key = |v: u32| {
            let i = v as usize * 3;
            [0, 1, 2].map(|k| (pos[i + k] * 1e5).round() as i64)
        };
        let mut ids: HashMap<[i64; 3], usize> = HashMap::new();
        let welded: Vec<usize> = (0..(pos.len() / 3) as u32)
            .map(|v| {
                let n = ids.len();
                *ids.entry(key(v)).or_insert(n)
            })
            .collect();
        let mut edges: HashMap<(usize, usize), usize> = HashMap::new();
        for t in g.index.as_deref().unwrap().chunks_exact(3) {
            let w = [0, 1, 2].map(|k| welded[t[k] as usize]);
            if w[0] == w[1] || w[1] == w[2] || w[0] == w[2] {
                continue; // a pole cell's zero-area triangle
            }
            for k in 0..3 {
                let (a, b) = (w[k], w[(k + 1) % 3]);
                *edges.entry((a.min(b), a.max(b))).or_default() += 1;
            }
        }
        let open: Vec<_> = edges.iter().filter(|(_, &n)| n != 2).collect();
        assert!(
            open.is_empty(),
            "{} edges not shared by exactly two triangles",
            open.len()
        );
    }
}
