//! Stochastic lattices — foams, not tilings.
//!
//! The other three families repeat one cell on a lattice of integers. These
//! two do not repeat at all: they are built from a seeded random field, so
//! every cell is a different shape and no direction is special. That is the
//! point of them. A beam lattice is stiff along its struts and soft between
//! them, and however the part is loaded some of those directions are wasted;
//! a foam approaches isotropy at a few cells across, which is why real foams,
//! bone and every energy-absorbing pad are built this way.
//!
//! | Family | Field | What it is |
//! |--------|-------|------------|
//! | [`Stochastic::Voronoi`] | distance to the Voronoi edges of a jittered point set | open-cell foam: struts, fully connected void |
//! | [`Stochastic::VoronoiWall`] | distance to the Voronoi faces | closed-cell foam: sealed bubbles |
//! | [`Stochastic::Spinodal`] and friends | a Gaussian random field of plane waves | spinodal decomposition — what a quenched alloy or a phase-separating polymer freezes into |
//!
//! # Randomness that does not move
//!
//! Nothing here holds a point set or an RNG. The seeds are a hash of their
//! cell index and the wave directions a hash of their own index, so the field
//! is a pure function of the point and the [`seed`](super::Lattice::seed): the
//! same lattice samples the same everywhere, at any resolution, on any thread,
//! in any process. A stored point set would have to be sorted, spatially
//! indexed and serialised to make the same promise.
//!
//! # Spinodoids
//!
//! [`Stochastic::Spinodal`] draws its wave vectors uniformly over the sphere,
//! which gives an isotropic solid. Restricting them to cones instead gives the
//! anisotropic classes of Kumar et al. (2020) — lamellar, columnar and cubic —
//! whose stiffness can be tuned by direction without changing the density.
//! Waves *along* an axis stack plates across it; waves *around* the equator
//! leave columns along it.

use crate::math::Vector3;
use std::f32::consts::TAU;

/// How many plane waves make a spinodal field.
///
/// The central limit theorem is what makes the sum Gaussian, and it arrives
/// quickly: past about sixteen waves the one-point statistics stop changing
/// and only the cost keeps going up. Sixteen is also enough that the cubic
/// class gets a fair share on each of its three axes.
const WAVES: usize = 24;

/// Half-angle of the cones the anisotropic classes draw their waves from.
const CONE: f32 = 0.261_799_4; // 15°

/// A random-field cell.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Stochastic {
    /// Open-cell foam: round struts along the edges of a Voronoi diagram.
    ///
    /// Every void connects to every other, which is what a filter, a scaffold
    /// or anything that has to drain powder needs.
    Voronoi,
    /// Closed-cell foam: the Voronoi faces themselves, as walls.
    ///
    /// Sealed bubbles — buoyant and good at absorbing a blast, impossible to
    /// clear un-fused powder out of.
    VoronoiWall,
    /// Isotropic spinodal decomposition: waves drawn over the whole sphere.
    Spinodal,
    /// Waves within 15° of z: plates stacked across the axis. Stiff in plane,
    /// soft across it.
    SpinodalLamellar,
    /// Waves within 15° of the xy plane: columns running along z. Stiff along
    /// the axis, soft across it.
    SpinodalColumnar,
    /// Waves within 15° of each of the three axes: cubic symmetry, stiffer on
    /// the axes than on the diagonals.
    SpinodalCubic,
}

impl Stochastic {
    /// Every stochastic cell, in declaration order.
    pub const ALL: [Stochastic; 6] = [
        Stochastic::Voronoi,
        Stochastic::VoronoiWall,
        Stochastic::Spinodal,
        Stochastic::SpinodalLamellar,
        Stochastic::SpinodalColumnar,
        Stochastic::SpinodalCubic,
    ];

    /// Lower-case identifier, unique across every lattice family.
    pub fn name(self) -> &'static str {
        match self {
            Stochastic::Voronoi => "voronoi",
            Stochastic::VoronoiWall => "voronoi-wall",
            Stochastic::Spinodal => "spinodal",
            Stochastic::SpinodalLamellar => "spinodal-lamellar",
            Stochastic::SpinodalColumnar => "spinodal-columnar",
            Stochastic::SpinodalCubic => "spinodal-cubic",
        }
    }

    /// The cell with this [`name`](Self::name).
    pub fn from_name(name: &str) -> Option<Stochastic> {
        Stochastic::ALL.into_iter().find(|s| s.name() == name)
    }

    /// Whether this cell is a level set of a random field rather than a
    /// distance to a skeleton.
    ///
    /// The field cells behave like a [`Tpms`](super::Tpms): they answer to
    /// [`LatticeStyle`](super::LatticeStyle), so the same generator gives
    /// either a sheet with two open void networks or a solid with one. The
    /// Voronoi cells are distances to a skeleton, like a
    /// [`Strut`](super::Strut), and ignore the style.
    pub fn is_field(self) -> bool {
        !matches!(self, Stochastic::Voronoi | Stochastic::VoronoiWall)
    }
}

/// Three values in `[0, 1)` from an integer key — a hash, not a sequence, so
/// the same cell always seeds the same point however the samples are ordered.
fn hash_unit(i: i32, j: i32, k: i32, seed: u32) -> [f32; 3] {
    let mut h = (i as i64 as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15)
        ^ (j as i64 as u64).wrapping_mul(0xbf58_476d_1ce4_e5b9)
        ^ (k as i64 as u64).wrapping_mul(0x94d0_49bb_1331_11eb)
        ^ (seed as u64).wrapping_mul(0xd6e8_feb8_6659_fd93);
    let mut out = [0.0f32; 3];
    for axis in &mut out {
        h ^= h >> 33;
        h = h.wrapping_mul(0xff51_afd7_ed55_8ccd);
        h ^= h >> 33;
        *axis = (h >> 40) as f32 / (1u32 << 24) as f32;
    }
    out
}

/// Distance from `p` to the nearest Voronoi edge, or face when `wall`.
///
/// One seed per cell of the integer grid, displaced from the cell centre by
/// `jitter` of a cell — 0 for a regular grid, 1 for a seed that can land
/// anywhere in its own cell. Keeping each seed inside its own cell is what
/// makes the 3×3×3 neighbourhood enough to find the three nearest.
///
/// A Voronoi face is equidistant from its two nearest seeds and an edge from
/// its three, so `(d2 - d1) / 2` and `(d3 - d1) / 2` are zero exactly on them
/// and grow away — the standard implicit reading, and the reason no diagram
/// has to be built to contour one.
pub(crate) fn voronoi_distance(
    p: Vector3,
    origin: Vector3,
    cell: Vector3,
    wall: bool,
    jitter: f32,
    seed: u32,
) -> f32 {
    let u = [
        (p.x - origin.x) / cell.x,
        (p.y - origin.y) / cell.y,
        (p.z - origin.z) / cell.z,
    ];
    let base = [
        u[0].floor() as i32,
        u[1].floor() as i32,
        u[2].floor() as i32,
    ];
    let jitter = jitter.clamp(0.0, 1.0);
    // The three nearest seed distances, smallest first.
    let mut best = [f32::INFINITY; 3];
    for oz in -1..=1 {
        for oy in -1..=1 {
            for ox in -1..=1 {
                let c = [base[0] + ox, base[1] + oy, base[2] + oz];
                let h = hash_unit(c[0], c[1], c[2], seed);
                // Measured in world units, so an anisotropic cell stretches
                // the foam rather than distorting which seed is nearest.
                let d = Vector3::new(
                    (u[0] - (c[0] as f32 + 0.5 + jitter * (h[0] - 0.5))) * cell.x,
                    (u[1] - (c[1] as f32 + 0.5 + jitter * (h[1] - 0.5))) * cell.y,
                    (u[2] - (c[2] as f32 + 0.5 + jitter * (h[2] - 0.5))) * cell.z,
                )
                .length();
                if d < best[0] {
                    best = [d, best[0], best[1]];
                } else if d < best[1] {
                    best = [best[0], d, best[1]];
                } else if d < best[2] {
                    best[2] = d;
                }
            }
        }
    }
    let far = if wall { best[1] } else { best[2] };
    if !far.is_finite() {
        return f32::INFINITY;
    }
    (far - best[0]) * 0.5
}

/// One plane wave of a spinodal field: a world-space wave vector and a phase.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Wave {
    k: Vector3,
    phase: f32,
}

/// The wave set for a cell type, in world units.
///
/// Built once per lattice and reused for every sample — a few hundred bytes
/// against `WAVES` trigonometric calls a point, which is the whole cost of the
/// field.
pub(crate) fn waves(kind: Stochastic, cell: Vector3, seed: u32) -> Vec<Wave> {
    (0..WAVES)
        .map(|i| {
            let h = hash_unit(i as i32, 0x5eed, 0, seed);
            let n = direction(kind, i, h);
            Wave {
                // Per axis, so an anisotropic cell stretches the field the
                // same way it stretches every other family.
                k: Vector3::new(
                    TAU * n.x / cell.x,
                    TAU * n.y / cell.y,
                    TAU * n.z / cell.z,
                ),
                phase: h[2] * TAU,
            }
        })
        .collect()
}

/// A wave direction for the class: uniform over the sphere, or inside a cone.
fn direction(kind: Stochastic, i: usize, h: [f32; 3]) -> Vector3 {
    let azimuth = h[1] * TAU;
    match kind {
        // Uniform in cos θ, not in θ — otherwise the poles get crowded and the
        // "isotropic" field is stiffer along z than across it.
        Stochastic::Spinodal => from_polar(Vector3::new(0.0, 0.0, 1.0), 2.0 * h[0] - 1.0, azimuth),
        Stochastic::SpinodalLamellar => from_polar(
            Vector3::new(0.0, 0.0, 1.0),
            1.0 - h[0] * (1.0 - CONE.cos()),
            azimuth,
        ),
        // A band about the equator rather than a cap about the pole.
        Stochastic::SpinodalColumnar => from_polar(
            Vector3::new(0.0, 0.0, 1.0),
            (2.0 * h[0] - 1.0) * CONE.sin(),
            azimuth,
        ),
        Stochastic::SpinodalCubic => {
            let axis = match i % 3 {
                0 => Vector3::new(1.0, 0.0, 0.0),
                1 => Vector3::new(0.0, 1.0, 0.0),
                _ => Vector3::new(0.0, 0.0, 1.0),
            };
            from_polar(axis, 1.0 - h[0] * (1.0 - CONE.cos()), azimuth)
        }
        // Not a field cell; any direction will do and none is ever asked for.
        _ => Vector3::new(0.0, 0.0, 1.0),
    }
}

/// A unit vector at `cos_polar` from `axis`, turned `azimuth` about it.
fn from_polar(axis: Vector3, cos_polar: f32, azimuth: f32) -> Vector3 {
    let cos_polar = cos_polar.clamp(-1.0, 1.0);
    let sin_polar = (1.0 - cos_polar * cos_polar).max(0.0).sqrt();
    // Any perpendicular will do — the azimuth is random about it.
    let seed = if axis.x.abs() < 0.9 {
        Vector3::new(1.0, 0.0, 0.0)
    } else {
        Vector3::new(0.0, 1.0, 0.0)
    };
    let u = axis.cross(seed).normalize();
    let v = axis.cross(u);
    let (s, c) = azimuth.sin_cos();
    axis * cos_polar + u * (sin_polar * c) + v * (sin_polar * s)
}

/// The field and its gradient at a world point.
///
/// Scaled to unit variance so that a level of `l` cuts the same fraction of
/// space whatever the wave count is: the sum of `n` unit cosines with random
/// phases has variance `n / 2`.
pub(crate) fn value_and_gradient(waves: &[Wave], p: Vector3) -> (f32, Vector3) {
    let mut value = 0.0f32;
    let mut grad = Vector3::ZERO;
    for w in waves {
        let (s, c) = (w.k.dot(p) + w.phase).sin_cos();
        value += c;
        grad = grad - w.k * s;
    }
    let scale = (2.0 / waves.len().max(1) as f32).sqrt();
    (value * scale, grad * scale)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_round_trip() {
        for s in Stochastic::ALL {
            assert_eq!(Stochastic::from_name(s.name()), Some(s));
        }
        assert_eq!(Stochastic::from_name("nope"), None);
    }

    #[test]
    fn voronoi_edges_are_further_apart_than_faces() {
        // Every point is at least as far from an edge as from a face: the
        // edges are a subset of the faces' boundaries.
        let cell = Vector3::new(1.0, 1.0, 1.0);
        for i in 0..200 {
            let h = hash_unit(i, 7, 11, 3);
            let p = Vector3::new(h[0] * 3.0, h[1] * 3.0, h[2] * 3.0);
            let face = voronoi_distance(p, Vector3::ZERO, cell, true, 0.8, 1);
            let edge = voronoi_distance(p, Vector3::ZERO, cell, false, 0.8, 1);
            assert!(edge >= face - 1e-5, "edge {edge} < face {face}");
        }
    }

    #[test]
    fn voronoi_is_deterministic() {
        let cell = Vector3::new(2.0, 3.0, 1.5);
        let p = Vector3::new(0.37, -1.2, 4.8);
        let a = voronoi_distance(p, Vector3::ZERO, cell, false, 1.0, 42);
        let b = voronoi_distance(p, Vector3::ZERO, cell, false, 1.0, 42);
        assert_eq!(a, b);
        let c = voronoi_distance(p, Vector3::ZERO, cell, false, 1.0, 43);
        assert!((a - c).abs() > 1e-6, "the seed did nothing");
    }

    #[test]
    fn spinodal_gradient_matches_differences() {
        let w = waves(Stochastic::Spinodal, Vector3::new(1.0, 1.0, 1.0), 5);
        let p = Vector3::new(0.31, -0.22, 0.77);
        let (_, g) = value_and_gradient(&w, p);
        let h = 1e-3;
        for axis in 0..3 {
            let mut d = Vector3::ZERO;
            match axis {
                0 => d.x = h,
                1 => d.y = h,
                _ => d.z = h,
            }
            let (up, _) = value_and_gradient(&w, p + d);
            let (dn, _) = value_and_gradient(&w, p - d);
            let numeric = (up - dn) / (2.0 * h);
            let analytic = match axis {
                0 => g.x,
                1 => g.y,
                _ => g.z,
            };
            assert!(
                (numeric - analytic).abs() < 1e-2,
                "axis {axis}: {numeric} vs {analytic}"
            );
        }
    }

    #[test]
    fn spinodal_has_unit_variance() {
        let w = waves(Stochastic::Spinodal, Vector3::new(1.0, 1.0, 1.0), 9);
        let mut sum = 0.0f64;
        let mut sum_sq = 0.0f64;
        let n = 20;
        for k in 0..n {
            for j in 0..n {
                for i in 0..n {
                    let p = Vector3::new(i as f32 * 0.31, j as f32 * 0.29, k as f32 * 0.37);
                    let (v, _) = value_and_gradient(&w, p);
                    sum += v as f64;
                    sum_sq += (v * v) as f64;
                }
            }
        }
        let count = (n * n * n) as f64;
        let mean = sum / count;
        let var = sum_sq / count - mean * mean;
        assert!(mean.abs() < 0.15, "mean {mean}");
        assert!((var - 1.0).abs() < 0.25, "variance {var}");
    }

    #[test]
    fn lamellar_waves_point_along_z() {
        let w = waves(Stochastic::SpinodalLamellar, Vector3::new(1.0, 1.0, 1.0), 2);
        for wave in &w {
            let n = wave.k.normalize();
            assert!(n.z > CONE.cos() - 1e-3, "wave off axis: {:?}", n);
        }
    }

    #[test]
    fn columnar_waves_lie_near_the_equator() {
        let w = waves(Stochastic::SpinodalColumnar, Vector3::new(1.0, 1.0, 1.0), 2);
        for wave in &w {
            let n = wave.k.normalize();
            assert!(n.z.abs() < CONE.sin() + 1e-3, "wave off equator: {:?}", n);
        }
    }

    #[test]
    fn cubic_waves_sit_on_the_axes() {
        let w = waves(Stochastic::SpinodalCubic, Vector3::new(1.0, 1.0, 1.0), 2);
        for wave in &w {
            let n = wave.k.normalize();
            let best = n.x.abs().max(n.y.abs()).max(n.z.abs());
            assert!(best > CONE.cos() - 1e-3, "wave off the axes: {:?}", n);
        }
    }
}
