//! Triply periodic minimal surfaces, as nodal (trigonometric) approximations.
//!
//! Each function is periodic with period `2π` on every axis, so one unit cell
//! spans `2π` and the level set `F = 0` divides space into two interpenetrating
//! labyrinths of equal volume. That is what makes them useful as infill: a
//! shell around `F = 0` is a self-supporting wall of near-uniform thickness with
//! no closed voids, and either labyrinth on its own is a connected solid.
//!
//! These are the classical nodal approximations, not the exact minimal
//! surfaces — the Weierstrass representations have no closed form in `x, y, z`.
//! Their level sets are within a percent or two of the true surfaces and share
//! their symmetry group, which is what matters for a lattice.

/// A triply periodic surface family.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Tpms {
    /// Schoen's gyroid — no straight lines, no mirror symmetries, and the two
    /// labyrinths are chiral. The usual default: it prints without support in
    /// any orientation and is close to isotropic.
    Gyroid,
    /// Schwarz P ("primitive") — cubic symmetry with round channels along the
    /// three axes. Stiffest along those axes, weakest on the diagonals.
    SchwarzP,
    /// Schwarz D ("diamond") — tetrahedral channels; the two labyrinths are
    /// each a diamond network. Stiffer than the gyroid at equal density.
    Diamond,
    /// Neovius — like Schwarz P with thicker nodes and narrower necks.
    Neovius,
    /// Schoen I-WP ("wrapped package") — one labyrinth is a set of near-spheres
    /// joined at the cube corners, the other surrounds it. Strongly asymmetric,
    /// so the two solid styles differ a lot.
    IWP,
    /// Fischer–Koch S — hexagonal-looking channels, low relative density at a
    /// given wall thickness.
    FischerKochS,
    /// Lidinoid — a gyroid relative from the same associate family, with
    /// hexagonal rather than cubic symmetry.
    Lidinoid,
    /// Split P — the gyroid's channels split into pairs; more surface area per
    /// unit volume than any other entry here.
    SplitP,
    /// Helicoid — a double screw up the middle of every cell column, one turn
    /// a cell, cut off at 0.45 of a cell from its axis. Not a nodal approximation: the
    /// helicoid is itself a minimal surface, the only ruled one besides the
    /// plane, and an array of them is a lattice of Archimedean screws. Under
    /// [`LatticeStyle::Sheet`](super::LatticeStyle::Sheet) the threads are
    /// separate blades on a shared spine; under `Solid` they are a threaded
    /// rod.
    Helicoid,
}

impl Tpms {
    /// Every family, in declaration order.
    pub const ALL: [Tpms; 9] = [
        Tpms::Gyroid,
        Tpms::SchwarzP,
        Tpms::Diamond,
        Tpms::Neovius,
        Tpms::IWP,
        Tpms::FischerKochS,
        Tpms::Lidinoid,
        Tpms::SplitP,
        Tpms::Helicoid,
    ];

    /// The nodal function at a point given in radians — one cell per `2π`.
    ///
    /// Positive on one labyrinth, negative on the other, zero on the surface.
    pub fn value(self, x: f32, y: f32, z: f32) -> f32 {
        let (sx, cx) = x.sin_cos();
        let (sy, cy) = y.sin_cos();
        let (sz, cz) = z.sin_cos();
        match self {
            Tpms::Helicoid => helicoid([(sx, cx), (sy, cy), (sz, cz)]).0,
            Tpms::Gyroid => sx * cy + sy * cz + sz * cx,
            Tpms::SchwarzP => cx + cy + cz,
            Tpms::Diamond => sx * sy * sz + sx * cy * cz + cx * sy * cz + cx * cy * sz,
            Tpms::Neovius => 3.0 * (cx + cy + cz) + 4.0 * cx * cy * cz,
            Tpms::IWP => {
                let (c2x, c2y, c2z) = (cos2(cx), cos2(cy), cos2(cz));
                2.0 * (cx * cy + cy * cz + cz * cx) - (c2x + c2y + c2z)
            }
            Tpms::FischerKochS => {
                let (c2x, c2y, c2z) = (cos2(cx), cos2(cy), cos2(cz));
                c2x * sy * cz + c2y * sz * cx + c2z * sx * cy
            }
            Tpms::Lidinoid => {
                let (s2x, s2y, s2z) = (sin2(sx, cx), sin2(sy, cy), sin2(sz, cz));
                let (c2x, c2y, c2z) = (cos2(cx), cos2(cy), cos2(cz));
                0.5 * (s2x * cy * sz + s2y * cz * sx + s2z * cx * sy)
                    - 0.5 * (c2x * c2y + c2y * c2z + c2z * c2x)
                    + 0.15
            }
            Tpms::SplitP => {
                let (s2x, s2y, s2z) = (sin2(sx, cx), sin2(sy, cy), sin2(sz, cz));
                let (c2x, c2y, c2z) = (cos2(cx), cos2(cy), cos2(cz));
                1.1 * (s2x * sz * cy + s2y * sx * cz + s2z * sy * cx)
                    - 0.2 * (c2x * c2y + c2y * c2z + c2z * c2x)
                    - 0.4 * (c2x + c2y + c2z)
            }
        }
    }

    /// The nodal function *and* its gradient, from one set of sines and
    /// cosines.
    ///
    /// The gradient is what turns the function into a distance — see
    /// [`Lattice::thickness`](super::Lattice::thickness) — so every sample of
    /// every TPMS lattice needs both. Differencing the function for it costs
    /// six extra evaluations and picks a step size that is either too coarse to
    /// track the surface or fine enough to be lost in `f32`; the closed form
    /// costs a handful of multiplies and has neither problem.
    ///
    /// The gradient is with respect to the radian coordinates. Multiply each
    /// component by that axis' radians per unit length to get a world gradient.
    pub fn value_and_gradient(self, x: f32, y: f32, z: f32) -> (f32, [f32; 3]) {
        self.value_and_gradient_trig([x.sin_cos(), y.sin_cos(), z.sin_cos()])
    }

    /// [`value_and_gradient`](Self::value_and_gradient) from the sines and
    /// cosines of the three coordinates, `(sin, cos)` per axis.
    ///
    /// Each depends on one coordinate alone, so a caller sampling a grid can
    /// work them out once per row of the grid rather than once per point —
    /// they are most of the cost of a sample — and get exactly the same bits.
    pub fn value_and_gradient_trig(self, trig: [(f32, f32); 3]) -> (f32, [f32; 3]) {
        let [(sx, cx), (sy, cy), (sz, cz)] = trig;
        match self {
            Tpms::Helicoid => helicoid(trig),
            Tpms::Gyroid => (
                sx * cy + sy * cz + sz * cx,
                [cx * cy - sx * sz, cy * cz - sx * sy, cx * cz - sy * sz],
            ),
            Tpms::SchwarzP => (cx + cy + cz, [-sx, -sy, -sz]),
            Tpms::Diamond => (
                sx * sy * sz + sx * cy * cz + cx * sy * cz + cx * cy * sz,
                [
                    cx * sy * sz + cx * cy * cz - sx * sy * cz - sx * cy * sz,
                    sx * cy * sz - sx * sy * cz + cx * cy * cz - cx * sy * sz,
                    sx * sy * cz - sx * cy * sz - cx * sy * sz + cx * cy * cz,
                ],
            ),
            Tpms::Neovius => (
                3.0 * (cx + cy + cz) + 4.0 * cx * cy * cz,
                [
                    -3.0 * sx - 4.0 * sx * cy * cz,
                    -3.0 * sy - 4.0 * cx * sy * cz,
                    -3.0 * sz - 4.0 * cx * cy * sz,
                ],
            ),
            Tpms::IWP => {
                let (c2x, c2y, c2z) = (cos2(cx), cos2(cy), cos2(cz));
                let (s2x, s2y, s2z) = (sin2(sx, cx), sin2(sy, cy), sin2(sz, cz));
                (
                    2.0 * (cx * cy + cy * cz + cz * cx) - (c2x + c2y + c2z),
                    [
                        -2.0 * sx * (cy + cz) + 2.0 * s2x,
                        -2.0 * sy * (cx + cz) + 2.0 * s2y,
                        -2.0 * sz * (cx + cy) + 2.0 * s2z,
                    ],
                )
            }
            Tpms::FischerKochS => {
                let (c2x, c2y, c2z) = (cos2(cx), cos2(cy), cos2(cz));
                let (s2x, s2y, s2z) = (sin2(sx, cx), sin2(sy, cy), sin2(sz, cz));
                (
                    c2x * sy * cz + c2y * sz * cx + c2z * sx * cy,
                    [
                        -2.0 * s2x * sy * cz - c2y * sz * sx + c2z * cx * cy,
                        c2x * cy * cz - 2.0 * s2y * sz * cx - c2z * sx * sy,
                        -c2x * sy * sz + c2y * cz * cx - 2.0 * s2z * sx * cy,
                    ],
                )
            }
            Tpms::Lidinoid => {
                let (s2x, s2y, s2z) = (sin2(sx, cx), sin2(sy, cy), sin2(sz, cz));
                let (c2x, c2y, c2z) = (cos2(cx), cos2(cy), cos2(cz));
                (
                    0.5 * (s2x * cy * sz + s2y * cz * sx + s2z * cx * sy)
                        - 0.5 * (c2x * c2y + c2y * c2z + c2z * c2x)
                        + 0.15,
                    [
                        0.5 * (2.0 * c2x * cy * sz + s2y * cz * cx - s2z * sx * sy)
                            + s2x * (c2y + c2z),
                        0.5 * (-s2x * sy * sz + 2.0 * c2y * cz * sx + s2z * cx * cy)
                            + s2y * (c2x + c2z),
                        0.5 * (s2x * cy * cz - s2y * sz * sx + 2.0 * c2z * cx * sy)
                            + s2z * (c2x + c2y),
                    ],
                )
            }
            Tpms::SplitP => {
                let (s2x, s2y, s2z) = (sin2(sx, cx), sin2(sy, cy), sin2(sz, cz));
                let (c2x, c2y, c2z) = (cos2(cx), cos2(cy), cos2(cz));
                (
                    1.1 * (s2x * sz * cy + s2y * sx * cz + s2z * sy * cx)
                        - 0.2 * (c2x * c2y + c2y * c2z + c2z * c2x)
                        - 0.4 * (c2x + c2y + c2z),
                    [
                        1.1 * (2.0 * c2x * sz * cy + s2y * cx * cz - s2z * sy * sx)
                            + 0.4 * s2x * (c2y + c2z)
                            + 0.8 * s2x,
                        1.1 * (-s2x * sz * sy + 2.0 * c2y * sx * cz + s2z * cy * cx)
                            + 0.4 * s2y * (c2x + c2z)
                            + 0.8 * s2y,
                        1.1 * (s2x * cz * cy - s2y * sx * sz + 2.0 * c2z * sy * cx)
                            + 0.4 * s2z * (c2x + c2y)
                            + 0.8 * s2z,
                    ],
                )
            }
        }
    }

    /// How far from its column's axis the surface is cut off, in radians of
    /// the cell's `2π`: only the [`Helicoid`](Tpms::Helicoid) is, the others
    /// fill the cell.
    pub fn clip_radius(self) -> Option<f32> {
        match self {
            Tpms::Helicoid => Some(0.45 * std::f32::consts::TAU),
            _ => None,
        }
    }

    /// Lower-case identifier, for logs and CLI arguments.
    pub fn name(self) -> &'static str {
        match self {
            Tpms::Gyroid => "gyroid",
            Tpms::SchwarzP => "schwarz-p",
            Tpms::Diamond => "diamond",
            Tpms::Neovius => "neovius",
            Tpms::IWP => "iwp",
            Tpms::FischerKochS => "fischer-koch-s",
            Tpms::Lidinoid => "lidinoid",
            Tpms::SplitP => "split-p",
            Tpms::Helicoid => "helicoid",
        }
    }

    /// The family with this [`name`](Self::name), if any.
    pub fn from_name(name: &str) -> Option<Tpms> {
        Tpms::ALL.into_iter().find(|t| t.name() == name)
    }
}

/// `cos 2θ` from `cos θ`, without a second trig call.
#[inline]
fn cos2(c: f32) -> f32 {
    2.0 * c * c - 1.0
}

/// `sin 2θ` from `sin θ` and `cos θ`.
#[inline]
fn sin2(s: f32, c: f32) -> f32 {
    2.0 * s * c
}

/// Threads a helicoid has: two, so a cross-section is a bar through the axis.
const HELICOID_STARTS: f32 = 2.0;

/// `r·sin(n(θ − z))` about the column axis through the middle of each cell,
/// `x ≡ y ≡ π (mod 2π)`, and its gradient. The angles are recovered from their
/// sines and cosines — `atan2(−sin, −cos)` is the angle less π, exactly — so
/// the helicoid takes the same per-axis tables as every other surface here.
/// Through the middle rather than the corners, so a part's box cuts no screw.
fn helicoid(trig: [(f32, f32); 3]) -> (f32, [f32; 3]) {
    let [(sx, cx), (sy, cy), (sz, cz)] = trig;
    let (x, y, z) = ((-sx).atan2(-cx), (-sy).atan2(-cy), sz.atan2(cz));
    let r = x.hypot(y);
    let n = HELICOID_STARTS;
    let phase = n * (y.atan2(x) - z);
    let (s, c) = phase.sin_cos();
    if r < 1e-6 {
        // On the axis every thread meets; the gradient across it is the
        // thread's direction there, which any one will do for.
        return (0.0, [s, -c, 0.0]);
    }
    let (ux, uy) = (x / r, y / r);
    (
        r * s,
        [ux * s - n * uy * c, uy * s + n * ux * c, -n * r * c],
    )
}

/// Distance from the helicoid's column axis, in radians of the cell's `2π`.
pub(crate) fn axis_radius(trig: [(f32, f32); 3]) -> f32 {
    (-trig[0].0).atan2(-trig[0].1).hypot((-trig[1].0).atan2(-trig[1].1))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_helicoid_is_a_screw() {
        use std::f32::consts::PI;
        // On the screw: the thread angle tracks the height.
        for i in 0..50 {
            let z = -PI + i as f32 * 0.12;
            let r = 0.3 + (i % 7) as f32 * 0.3;
            for thread in 0..2 {
                let theta = z + thread as f32 * PI;
                // About the axis through the middle of the cell, at π.
                let v = Tpms::Helicoid.value(PI + r * theta.cos(), PI + r * theta.sin(), z);
                assert!(v.abs() < 1e-4, "off the surface by {v} at r {r}, z {z}");
            }
        }
        // And its gradient is its slope.
        for i in 0..40 {
            let p = [
                PI + 0.3 + i as f32 * 0.05,
                PI - 0.7 + i as f32 * 0.03,
                0.2 * i as f32 - 2.0,
            ];
            let (_, g) = Tpms::Helicoid.value_and_gradient(p[0], p[1], p[2]);
            let e = 1e-3;
            for axis in 0..3 {
                let (mut up, mut dn) = (p, p);
                up[axis] += e;
                dn[axis] -= e;
                let numeric = (Tpms::Helicoid.value(up[0], up[1], up[2])
                    - Tpms::Helicoid.value(dn[0], dn[1], dn[2]))
                    / (2.0 * e);
                assert!(
                    (numeric - g[axis]).abs() < 2e-2 * g[axis].abs().max(1.0),
                    "axis {axis} at {p:?}: {numeric} vs {}",
                    g[axis]
                );
            }
        }
    }
    use std::f32::consts::TAU;

    #[test]
    fn every_family_is_periodic() {
        for f in Tpms::ALL {
            for &(x, y, z) in &[(0.3, 1.1, 2.7), (-1.4, 0.2, 5.0), (2.0, 2.0, 2.0)] {
                let base = f.value(x, y, z);
                for shift in [
                    (TAU, 0.0, 0.0),
                    (0.0, TAU, 0.0),
                    (0.0, 0.0, TAU),
                    (-TAU, TAU, TAU),
                ] {
                    let moved = f.value(x + shift.0, y + shift.1, z + shift.2);
                    assert!(
                        (base - moved).abs() < 1e-3,
                        "{} is not periodic: {base} vs {moved}",
                        f.name()
                    );
                }
            }
        }
    }

    #[test]
    fn every_family_changes_sign() {
        // A surface only exists where the function crosses zero; a family that
        // never changes sign over a cell would silently produce nothing.
        for f in Tpms::ALL {
            let (mut lo, mut hi) = (f32::MAX, f32::MIN);
            let n = 24;
            for k in 0..n {
                for j in 0..n {
                    for i in 0..n {
                        let s = |a: usize| a as f32 / n as f32 * TAU;
                        let v = f.value(s(i), s(j), s(k));
                        lo = lo.min(v);
                        hi = hi.max(v);
                    }
                }
            }
            assert!(lo < 0.0 && hi > 0.0, "{} never crosses zero", f.name());
        }
    }

    #[test]
    fn balanced_families_split_space_evenly() {
        // The classical minimal surfaces divide space in half at F = 0. I-WP,
        // Lidinoid and Split P are the deliberately unbalanced ones.
        for f in [
            Tpms::Gyroid,
            Tpms::SchwarzP,
            Tpms::Diamond,
            Tpms::Neovius,
            Tpms::FischerKochS,
        ] {
            let n = 40;
            let mut positive = 0;
            for k in 0..n {
                for j in 0..n {
                    for i in 0..n {
                        let s = |a: usize| (a as f32 + 0.5) / n as f32 * TAU;
                        if f.value(s(i), s(j), s(k)) > 0.0 {
                            positive += 1;
                        }
                    }
                }
            }
            let fraction = positive as f32 / (n * n * n) as f32;
            assert!(
                (fraction - 0.5).abs() < 0.02,
                "{} splits {fraction:.3}, not half",
                f.name()
            );
        }
    }

    #[test]
    fn the_closed_form_gradient_matches_differencing_it() {
        // These are derived by hand, one line per family per axis, and a slip
        // in any of them would show up as a wall of the wrong thickness rather
        // than as anything obviously broken. Differencing in f64 is the check:
        // it is slow and it is what the closed form replaced, but it is
        // independent of it.
        for f in Tpms::ALL {
            let mut worst = 0.0f64;
            let n = 11;
            for k in 0..n {
                for j in 0..n {
                    for i in 0..n {
                        // Deliberately off the symmetry planes, where a wrong
                        // term can cancel itself out.
                        let at = |a: usize| (a as f64 + 0.37) / n as f64 * TAU as f64;
                        let (x, y, z) = (at(i), at(j), at(k));
                        let value =
                            |x: f64, y: f64, z: f64| f.value(x as f32, y as f32, z as f32) as f64;
                        let h = 1e-3;
                        let numeric = [
                            (value(x + h, y, z) - value(x - h, y, z)) / (2.0 * h),
                            (value(x, y + h, z) - value(x, y - h, z)) / (2.0 * h),
                            (value(x, y, z + h) - value(x, y, z - h)) / (2.0 * h),
                        ];
                        let (_, exact) = f.value_and_gradient(x as f32, y as f32, z as f32);
                        for a in 0..3 {
                            worst = worst.max((numeric[a] - exact[a] as f64).abs());
                        }
                    }
                }
            }
            // f32 evaluation of the difference quotient is the floor here, not
            // the derivative.
            assert!(worst < 2e-2, "{}: gradient off by {worst}", f.name());
        }
    }

    #[test]
    fn the_two_value_paths_agree() {
        // `value` and `value_and_gradient` each spell the function out, so they
        // can drift apart.
        for f in Tpms::ALL {
            for i in 0..97 {
                let at = |m: u32| (i as f32 * m as f32 * 0.137) % TAU;
                let (x, y, z) = (at(1), at(3), at(7));
                let (both, _) = f.value_and_gradient(x, y, z);
                assert!(
                    (f.value(x, y, z) - both).abs() < 1e-5,
                    "{} disagrees with itself",
                    f.name()
                );
            }
        }
    }

    #[test]
    fn the_gradient_never_vanishes_on_the_surface() {
        // Dividing by |∇F| is what makes thickness a length; a family whose
        // gradient died on its own zero set would give an infinite wall there.
        for f in Tpms::ALL {
            let mut weakest = f32::MAX;
            let n = 60;
            for k in 0..n {
                for j in 0..n {
                    for i in 0..n {
                        let at = |a: usize| (a as f32 + 0.5) / n as f32 * TAU;
                        let (x, y, z) = (at(i), at(j), at(k));
                        let (v, g) = f.value_and_gradient(x, y, z);
                        if v.abs() < 0.05 {
                            let len = (g[0] * g[0] + g[1] * g[1] + g[2] * g[2]).sqrt();
                            weakest = weakest.min(len);
                        }
                    }
                }
            }
            assert!(weakest > 0.05, "{} flattens out at {weakest}", f.name());
        }
    }

    #[test]
    fn names_round_trip() {
        for f in Tpms::ALL {
            assert_eq!(Tpms::from_name(f.name()), Some(f));
        }
        assert_eq!(Tpms::from_name("not-a-surface"), None);
    }

    #[test]
    fn double_angle_helpers_match_trig() {
        for i in 0..64 {
            let t = i as f32 / 64.0 * TAU - 3.0;
            let (s, c) = t.sin_cos();
            assert!((cos2(c) - (2.0 * t).cos()).abs() < 1e-5);
            assert!((sin2(s, c) - (2.0 * t).sin()).abs() < 1e-5);
        }
    }
}
