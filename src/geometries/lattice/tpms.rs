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
}

impl Tpms {
    /// Every family, in declaration order.
    pub const ALL: [Tpms; 8] = [
        Tpms::Gyroid,
        Tpms::SchwarzP,
        Tpms::Diamond,
        Tpms::Neovius,
        Tpms::IWP,
        Tpms::FischerKochS,
        Tpms::Lidinoid,
        Tpms::SplitP,
    ];

    /// The nodal function at a point given in radians — one cell per `2π`.
    ///
    /// Positive on one labyrinth, negative on the other, zero on the surface.
    pub fn value(self, x: f32, y: f32, z: f32) -> f32 {
        let (sx, cx) = x.sin_cos();
        let (sy, cy) = y.sin_cos();
        let (sz, cz) = z.sin_cos();
        match self {
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
        let (sx, cx) = x.sin_cos();
        let (sy, cy) = y.sin_cos();
        let (sz, cz) = z.sin_cos();
        match self {
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

#[cfg(test)]
mod tests {
    use super::*;
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
