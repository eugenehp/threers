//! Sampled scalar fields — where a [`grade`](super::Lattice::grade) comes from.
//!
//! [`grade`](super::Lattice::grade) takes a closure, which is the most general
//! thing it could take and the least useful thing to be handed: a part is
//! graded because a solver said where the stress was, or a scan said where the
//! bone was, and none of that arrives as a closure over world space. A [`Field`]
//! is the adapter. Give it a grid, a function, or a bag of scattered points with
//! values, and it interpolates, rescales and hands back the closure.
//!
//! ```
//! use threers::{Field, Lattice, LatticeKind, Strut, Vector3};
//!
//! // Wherever a solver put its stress — element centroids and a scalar.
//! let stress = [
//!     (Vector3::new(-5.0, 0.0, 0.0), 120.0),
//!     (Vector3::new(5.0, 0.0, 0.0), 10.0),
//! ];
//! let field = Field::scattered(
//!     threers::Box3::from_center_and_size(Vector3::ZERO, Vector3::new(20.0, 20.0, 20.0)),
//!     [12, 12, 12],
//!     &stress,
//! );
//!
//! // Half thickness where it is idle, double where it is worked.
//! let geom = Lattice::new(LatticeKind::Strut(Strut::Octet))
//!     .size(Vector3::new(20.0, 20.0, 20.0))
//!     .cells([4, 4, 4])
//!     .thickness(0.6)
//!     .grade(field.into_grade(0.5, 2.0))
//!     .build();
//! # assert!(geom.index.is_some());
//! ```
//!
//! # Rescaling is the whole job
//!
//! A grade is a *multiplier*: 1 leaves the thickness alone. Stress is in
//! megapascals, a scan in Hounsfield units, a temperature in kelvin, and none
//! of them is near 1. [`into_grade`](Field::into_grade) is what closes that gap
//! — it reads the field's own range once, up front, and maps it onto the two
//! multipliers asked for, so the thinnest place in the part is the field's
//! minimum and the thickest its maximum. A field that is constant grades to the
//! midpoint rather than dividing by zero.
//!
//! Reach for [`map`](Field::map) first if the relationship is not linear. Beam
//! stiffness goes as the cube of the thickness, so grading on the cube root of
//! stress spreads the material where a solver would put it rather than
//! over-weighting the one hot element.

use crate::math::{Box3, Vector3};

/// A scalar sampled on a regular grid, interpolated between the samples.
#[derive(Clone, Debug)]
pub struct Field {
    bounds: Box3,
    dims: [usize; 3],
    values: Vec<f32>,
}

impl Field {
    /// A field from values already on a grid, ordered x fastest then y then z.
    ///
    /// `None` if the length does not match the dimensions, or any dimension is
    /// zero.
    pub fn grid(bounds: Box3, dims: [usize; 3], values: Vec<f32>) -> Option<Self> {
        if dims.contains(&0) || values.len() != dims[0] * dims[1] * dims[2] {
            return None;
        }
        Some(Self {
            bounds,
            dims,
            values,
        })
    }

    /// A field by evaluating a function on a grid — for a closure that is too
    /// slow to call once per lattice sample.
    ///
    /// A lattice samples its field millions of times; a field baked at 32³ is
    /// 32 768 evaluations, read back by interpolation. Worth it for anything
    /// that touches a BVH, a solver or a file.
    pub fn from_fn(
        bounds: Box3,
        dims: [usize; 3],
        f: impl Fn(Vector3) -> f32 + Sync + Send,
    ) -> Self {
        let dims = [dims[0].max(1), dims[1].max(1), dims[2].max(1)];
        let total = dims[0] * dims[1] * dims[2];
        let values = crate::utils::parallel::par_map_range(total, |n| {
            let i = n % dims[0];
            let j = (n / dims[0]) % dims[1];
            let k = n / (dims[0] * dims[1]);
            f(node(bounds, dims, i, j, k))
        });
        Self {
            bounds,
            dims,
            values,
        }
    }

    /// A field from scattered samples — solver output, sensor readings, a point
    /// cloud — resampled onto a grid by inverse-distance weighting.
    ///
    /// Each node averages the samples in its own and the neighbouring cells,
    /// weighted by one over distance squared, searching further out only where
    /// that finds nothing. Samples do not have to be evenly spread, and no
    /// sample is ever ignored: a node with nothing near it takes the nearest
    /// thing there is, so the field is defined over the whole of `bounds`
    /// however sparse the input.
    ///
    /// Empty samples give a field of zeros.
    pub fn scattered(bounds: Box3, dims: [usize; 3], samples: &[(Vector3, f32)]) -> Self {
        let dims = [dims[0].max(1), dims[1].max(1), dims[2].max(1)];
        if samples.is_empty() {
            return Self {
                bounds,
                dims,
                values: vec![0.0; dims[0] * dims[1] * dims[2]],
            };
        }
        let buckets = Buckets::build(bounds, dims, samples);
        let total = dims[0] * dims[1] * dims[2];
        let values = crate::utils::parallel::par_map_range(total, |n| {
            let i = n % dims[0];
            let j = (n / dims[0]) % dims[1];
            let k = n / (dims[0] * dims[1]);
            buckets.interpolate(node(bounds, dims, i, j, k), samples)
        });
        Self {
            bounds,
            dims,
            values,
        }
    }

    /// A field from a value per vertex of a mesh — a simulation result read
    /// back onto the geometry it was run on.
    ///
    /// `None` if the mesh has no positions or the counts disagree. The grid is
    /// sized to the mesh's own bounds, expanded by a cell so that a lattice
    /// filling the mesh does not sample past the last node.
    pub fn from_mesh(
        geometry: &crate::core::BufferGeometry,
        values: &[f32],
        dims: [usize; 3],
    ) -> Option<Self> {
        let positions: Vec<Vector3> = geometry.positions()?.collect();
        if positions.len() != values.len() || positions.is_empty() {
            return None;
        }
        let samples: Vec<(Vector3, f32)> = positions
            .iter()
            .copied()
            .zip(values.iter().copied())
            .collect();
        let mut bounds = Box3::from_points(&positions);
        let pad = bounds.size().length() * 0.01 + 1e-6;
        bounds.expand_by_scalar(pad);
        Some(Self::scattered(bounds, dims, &samples))
    }

    /// The box the samples span.
    pub fn bounds(&self) -> Box3 {
        self.bounds
    }

    /// The sample counts per axis.
    pub fn dims(&self) -> [usize; 3] {
        self.dims
    }

    /// The value at a world point, trilinearly interpolated.
    ///
    /// Outside the bounds the edge value is held rather than extrapolated: a
    /// lattice's sample grid overhangs its bounds by a few samples, and
    /// extrapolating a stress field past the part it was solved on produces
    /// numbers no solver would stand behind.
    pub fn value(&self, p: Vector3) -> f32 {
        let size = self.bounds.size();
        let axis = |v: f32, lo: f32, extent: f32, n: usize| -> (usize, usize, f32) {
            if n < 2 {
                return (0, 0, 0.0);
            }
            let t = if extent > 1e-12 {
                (v - lo) / extent * (n - 1) as f32
            } else {
                0.0
            };
            let t = t.clamp(0.0, (n - 1) as f32);
            let i = (t.floor() as usize).min(n - 2);
            (i, i + 1, t - i as f32)
        };
        let (i0, i1, fx) = axis(p.x, self.bounds.min.x, size.x, self.dims[0]);
        let (j0, j1, fy) = axis(p.y, self.bounds.min.y, size.y, self.dims[1]);
        let (k0, k1, fz) = axis(p.z, self.bounds.min.z, size.z, self.dims[2]);
        let at = |i: usize, j: usize, k: usize| -> f32 {
            self.values[(k * self.dims[1] + j) * self.dims[0] + i]
        };
        let lerp = |a: f32, b: f32, t: f32| a + (b - a) * t;
        let y0 = lerp(
            lerp(at(i0, j0, k0), at(i1, j0, k0), fx),
            lerp(at(i0, j1, k0), at(i1, j1, k0), fx),
            fy,
        );
        let y1 = lerp(
            lerp(at(i0, j0, k1), at(i1, j0, k1), fx),
            lerp(at(i0, j1, k1), at(i1, j1, k1), fx),
            fy,
        );
        lerp(y0, y1, fz)
    }

    /// The smallest and largest sample.
    pub fn range(&self) -> (f32, f32) {
        self.values.iter().fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), &v| {
            (lo.min(v), hi.max(v))
        })
    }

    /// Put every sample through a function — a square root, a logarithm, a
    /// threshold.
    pub fn map(mut self, f: impl Fn(f32) -> f32) -> Self {
        for v in &mut self.values {
            *v = f(*v);
        }
        self
    }

    /// Rescale so the samples run from 0 to 1. A constant field becomes 0.5.
    pub fn normalized(self) -> Self {
        let (lo, hi) = self.range();
        let span = hi - lo;
        if !span.is_finite() || span.abs() < 1e-12 {
            return self.map(|_| 0.5);
        }
        self.map(move |v| (v - lo) / span)
    }

    /// Hold every sample between two values.
    pub fn clamped(self, lo: f32, hi: f32) -> Self {
        self.map(move |v| v.clamp(lo, hi))
    }

    /// Average each sample with its six neighbours, `passes` times.
    ///
    /// Solver output is noisiest exactly where it is largest, and a lattice
    /// graded straight off it gets a thickness that jumps between neighbouring
    /// cells. Two or three passes cost nothing next to the build and leave
    /// something a printer can follow.
    pub fn smoothed(mut self, passes: usize) -> Self {
        let [nx, ny, nz] = self.dims;
        for _ in 0..passes {
            let src = self.values.clone();
            for k in 0..nz {
                for j in 0..ny {
                    for i in 0..nx {
                        let at = |i: usize, j: usize, k: usize| src[(k * ny + j) * nx + i];
                        let mut sum = at(i, j, k);
                        let mut count = 1.0f32;
                        let mut add = |i: usize, j: usize, k: usize| {
                            sum += at(i, j, k);
                            count += 1.0;
                        };
                        if i > 0 {
                            add(i - 1, j, k);
                        }
                        if i + 1 < nx {
                            add(i + 1, j, k);
                        }
                        if j > 0 {
                            add(i, j - 1, k);
                        }
                        if j + 1 < ny {
                            add(i, j + 1, k);
                        }
                        if k > 0 {
                            add(i, j, k - 1);
                        }
                        if k + 1 < nz {
                            add(i, j, k + 1);
                        }
                        self.values[(k * ny + j) * nx + i] = sum / count;
                    }
                }
            }
        }
        self
    }

    /// The closure [`grade`](super::Lattice::grade) wants: the field's own
    /// range mapped onto two thickness multipliers.
    ///
    /// The range is read once, here, so the mapping does not shift as the field
    /// is sampled and repeated builds agree. `at_min` may exceed `at_max` — that
    /// is how a field where *low* means *loaded*, such as a distance to a
    /// contact patch, is graded.
    pub fn into_grade(self, at_min: f32, at_max: f32) -> impl Fn(Vector3) -> f32 + Sync + Send {
        let (lo, hi) = self.range();
        let span = hi - lo;
        move |p| {
            if !span.is_finite() || span.abs() < 1e-12 {
                return 0.5 * (at_min + at_max);
            }
            let t = ((self.value(p) - lo) / span).clamp(0.0, 1.0);
            at_min + (at_max - at_min) * t
        }
    }
}

/// The world position of a grid node.
fn node(bounds: Box3, dims: [usize; 3], i: usize, j: usize, k: usize) -> Vector3 {
    let size = bounds.size();
    let along = |n: usize, index: usize, lo: f32, extent: f32| -> f32 {
        if n < 2 {
            lo + extent * 0.5
        } else {
            lo + extent * index as f32 / (n - 1) as f32
        }
    };
    Vector3::new(
        along(dims[0], i, bounds.min.x, size.x),
        along(dims[1], j, bounds.min.y, size.y),
        along(dims[2], k, bounds.min.z, size.z),
    )
}

/// Which cell of the grid a point falls in.
fn cell_of(bounds: Box3, dims: [usize; 3], p: Vector3) -> [usize; 3] {
    let size = bounds.size();
    let axis = |v: f32, lo: f32, extent: f32, n: usize| -> usize {
        if extent <= 1e-12 {
            return 0;
        }
        (((v - lo) / extent * n as f32).floor() as isize).clamp(0, n as isize - 1) as usize
    };
    [
        axis(p.x, bounds.min.x, size.x, dims[0]),
        axis(p.y, bounds.min.y, size.y, dims[1]),
        axis(p.z, bounds.min.z, size.z, dims[2]),
    ]
}

/// Scattered samples sorted into the grid's own cells, so a node only looks at
/// the samples that could be near it.
struct Buckets {
    bounds: Box3,
    dims: [usize; 3],
    /// Sample indices per bucket.
    cells: Vec<Vec<u32>>,
}

impl Buckets {
    fn build(bounds: Box3, dims: [usize; 3], samples: &[(Vector3, f32)]) -> Self {
        let mut cells = vec![Vec::new(); dims[0] * dims[1] * dims[2]];
        for (n, (p, _)) in samples.iter().enumerate() {
            let [i, j, k] = cell_of(bounds, dims, *p);
            cells[(k * dims[1] + j) * dims[0] + i].push(n as u32);
        }
        Self {
            bounds,
            dims,
            cells,
        }
    }

    /// Inverse-distance average of the samples near `p`.
    ///
    /// The search starts on the point's own bucket and grows a ring at a time
    /// until it finds something. A ring further out cannot beat one already
    /// found by enough to matter — the weights fall off as the square — so the
    /// first non-empty ring, plus the one after it, is where it stops.
    fn interpolate(&self, p: Vector3, samples: &[(Vector3, f32)]) -> f32 {
        let [ci, cj, ck] = cell_of(self.bounds, self.dims, p);
        let reach = self.dims[0].max(self.dims[1]).max(self.dims[2]) as isize;
        let mut sum = 0.0f64;
        let mut weight = 0.0f64;
        let mut found_at: Option<isize> = None;
        for ring in 0..=reach {
            if let Some(first) = found_at {
                if ring > first + 1 {
                    break;
                }
            }
            let mut any = false;
            for dk in -ring..=ring {
                for dj in -ring..=ring {
                    for di in -ring..=ring {
                        // Only the shell of the ring; the inside is done.
                        if ring > 0 && di.abs() != ring && dj.abs() != ring && dk.abs() != ring {
                            continue;
                        }
                        let (i, j, k) = (ci as isize + di, cj as isize + dj, ck as isize + dk);
                        if i < 0
                            || j < 0
                            || k < 0
                            || i >= self.dims[0] as isize
                            || j >= self.dims[1] as isize
                            || k >= self.dims[2] as isize
                        {
                            continue;
                        }
                        let bucket = &self.cells
                            [(k as usize * self.dims[1] + j as usize) * self.dims[0] + i as usize];
                        for &n in bucket {
                            let (q, v) = samples[n as usize];
                            let w = 1.0 / ((q - p).length_sq().max(1e-12) as f64);
                            sum += w * v as f64;
                            weight += w;
                            any = true;
                        }
                    }
                }
            }
            if any && found_at.is_none() {
                found_at = Some(ring);
            }
        }
        if weight > 0.0 {
            (sum / weight) as f32
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit_box() -> Box3 {
        Box3::from_center_and_size(Vector3::ZERO, Vector3::new(2.0, 2.0, 2.0))
    }

    #[test]
    fn a_baked_function_reads_back() {
        let f = Field::from_fn(unit_box(), [17, 17, 17], |p| p.x + 2.0 * p.y);
        for (p, want) in [
            (Vector3::ZERO, 0.0),
            (Vector3::new(0.5, 0.25, 0.0), 1.0),
            (Vector3::new(-1.0, 1.0, 0.7), 1.0),
        ] {
            assert!((f.value(p) - want).abs() < 1e-3, "{p:?}: {}", f.value(p));
        }
    }

    #[test]
    fn outside_the_bounds_holds_the_edge() {
        let f = Field::from_fn(unit_box(), [9, 9, 9], |p| p.x);
        assert!((f.value(Vector3::new(50.0, 0.0, 0.0)) - 1.0).abs() < 1e-4);
        assert!((f.value(Vector3::new(-50.0, 0.0, 0.0)) + 1.0).abs() < 1e-4);
    }

    #[test]
    fn scattered_samples_land_on_their_own_values() {
        let samples = [
            (Vector3::new(-0.8, 0.0, 0.0), 10.0),
            (Vector3::new(0.8, 0.0, 0.0), 30.0),
        ];
        let f = Field::scattered(unit_box(), [16, 16, 16], &samples);
        assert!(f.value(Vector3::new(-0.8, 0.0, 0.0)) < 15.0);
        assert!(f.value(Vector3::new(0.8, 0.0, 0.0)) > 25.0);
        // And in between it is in between.
        let mid = f.value(Vector3::ZERO);
        assert!((10.0..=30.0).contains(&mid), "{mid}");
    }

    #[test]
    fn a_sparse_field_is_still_defined_everywhere() {
        // One sample in a corner: every node has to find it.
        let f = Field::scattered(unit_box(), [8, 8, 8], &[(Vector3::new(-1.0, -1.0, -1.0), 7.0)]);
        for p in [
            Vector3::new(1.0, 1.0, 1.0),
            Vector3::ZERO,
            Vector3::new(-1.0, 1.0, 0.0),
        ] {
            assert!((f.value(p) - 7.0).abs() < 1e-3, "{p:?}: {}", f.value(p));
        }
    }

    #[test]
    fn grade_maps_the_range_onto_the_multipliers() {
        let f = Field::from_fn(unit_box(), [9, 9, 9], |p| p.x);
        let grade = f.into_grade(0.5, 2.0);
        assert!((grade(Vector3::new(-1.0, 0.0, 0.0)) - 0.5).abs() < 1e-4);
        assert!((grade(Vector3::new(1.0, 0.0, 0.0)) - 2.0).abs() < 1e-4);
        assert!((grade(Vector3::ZERO) - 1.25).abs() < 1e-4);
        // Past the field, the multiplier holds rather than running away.
        assert!((grade(Vector3::new(99.0, 0.0, 0.0)) - 2.0).abs() < 1e-4);
    }

    #[test]
    fn a_constant_field_grades_to_the_middle() {
        let f = Field::from_fn(unit_box(), [4, 4, 4], |_| 3.0);
        let grade = f.into_grade(0.5, 2.5);
        assert!((grade(Vector3::ZERO) - 1.5).abs() < 1e-6);
    }

    #[test]
    fn normalizing_and_smoothing_keep_the_shape() {
        let f = Field::from_fn(unit_box(), [9, 9, 9], |p| p.x * 100.0).normalized();
        let (lo, hi) = f.range();
        assert!((lo - 0.0).abs() < 1e-5 && (hi - 1.0).abs() < 1e-5);
        let smooth = f.smoothed(3);
        // Still monotonic in x, just gentler.
        assert!(smooth.value(Vector3::new(-0.5, 0.0, 0.0)) < smooth.value(Vector3::new(0.5, 0.0, 0.0)));
    }

    #[test]
    fn grid_checks_its_own_length() {
        assert!(Field::grid(unit_box(), [2, 2, 2], vec![0.0; 8]).is_some());
        assert!(Field::grid(unit_box(), [2, 2, 2], vec![0.0; 7]).is_none());
        assert!(Field::grid(unit_box(), [0, 2, 2], vec![]).is_none());
    }

    #[test]
    fn empty_samples_are_not_a_panic() {
        let f = Field::scattered(unit_box(), [4, 4, 4], &[]);
        assert_eq!(f.value(Vector3::ZERO), 0.0);
    }
}
