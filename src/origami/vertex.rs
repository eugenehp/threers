//! Degree-4 flat-foldable vertex kinematics (Akitaya et al., arXiv:1812.01160).
//!
//! A single vertex of this type is a 1-DOF mechanism with two modes that meet
//! only at the flat state. That branching is why a global mountain–valley
//! assignment is a combinatorial choice, not a continuous one.

use super::{dot2, sub2, V2};

/// Theorem 6 case: which pair of opposite creases carries the driving fold.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VertexMode {
    /// Opposite creases 0 and 2 share `t`; 1 and 3 are `±p_a t`.
    A,
    /// Opposite creases 1 and 3 share `t`; 0 and 2 are `±p_b t`.
    B,
}

/// Local geometry of a flat-foldable degree-4 vertex, creases ordered CCW.
///
/// Sector angles are `(α, β, π−α, π−β)` — Kawasaki's condition, which is
/// necessary for a rigid folding through the flat state.
#[derive(Debug, Clone, Copy)]
pub struct Degree4 {
    pub alpha: f64,
    pub beta: f64,
}

impl Degree4 {
    /// `p_a(α, β)` — fold-angle multiplier for mode A.
    pub fn p_a(self) -> f64 {
        let ta = (self.alpha * 0.5).tan();
        let tb = (self.beta * 0.5).tan();
        (1.0 - ta * tb) / (1.0 + ta * tb)
    }

    /// `p_b(α, β)` — fold-angle multiplier for mode B.
    pub fn p_b(self) -> f64 {
        let ta = (self.alpha * 0.5).tan();
        let tb = (self.beta * 0.5).tan();
        (tb - ta) / (tb + ta)
    }

    /// Multiplier when `β = π/2`: `p_a = p_b = (1 − tan(α/2)) / (1 + tan(α/2))`.
    pub fn p_right(self) -> f64 {
        let ta = (self.alpha * 0.5).tan();
        (1.0 - ta) / (1.0 + ta)
    }

    /// Half-angle tangents `(t0, t1, t2, t3)` for a mode and a driving `t`.
    ///
    /// `t_i = tan(ρ_i / 2)`. The sign of `t` flips the mountain–valley assignment
    /// without leaving the mode.
    pub fn tangents(self, mode: VertexMode, t: f64) -> [f64; 4] {
        match mode {
            VertexMode::A => {
                let p = self.p_a();
                [t, -p * t, t, p * t]
            }
            VertexMode::B => {
                let p = self.p_b();
                [-p * t, t, p * t, t]
            }
        }
    }

    /// Fold angles `ρ_i = 2 atan(t_i)`.
    pub fn fold_angles(self, mode: VertexMode, t: f64) -> [f64; 4] {
        self.tangents(mode, t).map(|ti| 2.0 * ti.atan())
    }

    /// Recover the driving `t` from one known crease tangent, if the mode allows it.
    pub fn drive_from(self, mode: VertexMode, crease: usize, ti: f64) -> Option<f64> {
        let k = crease & 3;
        match mode {
            VertexMode::A => {
                let p = self.p_a();
                match k {
                    0 | 2 => Some(ti),
                    1 if p.abs() > 1e-14 => Some(ti / -p),
                    3 if p.abs() > 1e-14 => Some(ti / p),
                    1 | 3 if ti.abs() <= 1e-12 => None, // unused crease; t is free
                    _ => None,
                }
            }
            VertexMode::B => {
                let p = self.p_b();
                match k {
                    1 | 3 => Some(ti),
                    0 if p.abs() > 1e-14 => Some(ti / -p),
                    2 if p.abs() > 1e-14 => Some(ti / p),
                    0 | 2 if ti.abs() <= 1e-12 => None,
                    _ => None,
                }
            }
        }
    }

    /// Sequential rotation product from the paper's proof of Theorem 6.
    ///
    /// A rigid folded state at this vertex must give the identity. Used as a
    /// check, not as a solver.
    pub fn rotation_product(self, rho: [f64; 4]) -> [[f64; 3]; 3] {
        let a = self.alpha;
        let b = self.beta;
        let mut m = mat_ident();
        m = matmul(m, rz(std::f64::consts::PI - b));
        m = matmul(m, rx(rho[0]));
        m = matmul(m, rz(a));
        m = matmul(m, rx(rho[1]));
        m = matmul(m, rz(b));
        m = matmul(m, rx(rho[2]));
        m = matmul(m, rz(std::f64::consts::PI - a));
        m = matmul(m, rx(rho[3]));
        m
    }
}

/// Kawasaki at a degree-4 vertex: opposite sector angles sum to π.
///
/// `dirs` are the four crease directions from the vertex, already CCW. The test
/// is the one in Theorem 13: reflecting the first direction through the others
/// returns it.
pub fn kawasaki(dirs: [V2; 4]) -> bool {
    let mut p = dirs[0];
    for q in dirs.iter().copied().skip(1) {
        p = reflect_through(p, q);
    }
    let err = sub2(p, dirs[0]);
    dot2(err, err).sqrt() <= 1e-9 * (1.0 + dot2(dirs[0], dirs[0]).sqrt())
}

/// Degree-4 description from four CCW directions, starting at `dirs[0]`.
pub fn degree4_from_dirs(dirs: [V2; 4]) -> Option<Degree4> {
    if !kawasaki(dirs) {
        return None;
    }
    let th = sectors_in_order(dirs);
    Some(Degree4 {
        alpha: th[0],
        beta: th[1],
    })
}

fn sectors_in_order(dirs: [V2; 4]) -> [f64; 4] {
    let mut out = [0.0; 4];
    for i in 0..4 {
        let a = dirs[i];
        let b = dirs[(i + 1) & 3];
        // Signed angle from a to b, expected CCW in (0, π].
        let cross = a[0] * b[1] - a[1] * b[0];
        let d = (dot2(a, b) / (dot2(a, a) * dot2(b, b)).sqrt()).clamp(-1.0, 1.0);
        let mut ang = d.acos();
        if cross < 0.0 {
            ang = std::f64::consts::TAU - ang;
        }
        out[i] = ang;
    }
    out
}

/// Reflection of `p` through `q` in 2D (Theorem 13, eq. 7).
pub fn reflect_through(p: V2, q: V2) -> V2 {
    let q_perp = [-q[1], q[0]];
    let qq = dot2(q, q);
    if qq < 1e-30 {
        return p;
    }
    let s = 2.0 * dot2(p, q_perp) / qq;
    [p[0] - s * q_perp[0], p[1] - s * q_perp[1]]
}

pub(crate) fn mat_ident() -> [[f64; 3]; 3] {
    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
}

fn rx(th: f64) -> [[f64; 3]; 3] {
    let (s, c) = th.sin_cos();
    [[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]]
}

fn rz(th: f64) -> [[f64; 3]; 3] {
    let (s, c) = th.sin_cos();
    [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]
}

pub(crate) fn matmul(a: [[f64; 3]; 3], b: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut c = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            c[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
        }
    }
    c
}

#[cfg(test)]
fn mat_err_ident(m: [[f64; 3]; 3]) -> f64 {
    let i = mat_ident();
    let mut e = 0.0_f64;
    for r in 0..3 {
        for c in 0..3 {
            e = e.max((m[r][c] - i[r][c]).abs());
        }
    }
    e
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn perpendicular_cross_has_zero_multipliers() {
        let d = Degree4 {
            alpha: std::f64::consts::FRAC_PI_2,
            beta: std::f64::consts::FRAC_PI_2,
        };
        assert!(d.p_a().abs() < 1e-12);
        assert!(d.p_b().abs() < 1e-12);
        // Figure 1: mode A folds creases 0 and 2 only.
        let ta = d.tangents(VertexMode::A, 0.4);
        assert!((ta[0] - 0.4).abs() < 1e-12);
        assert!(ta[1].abs() < 1e-12);
        assert!((ta[2] - 0.4).abs() < 1e-12);
        assert!(ta[3].abs() < 1e-12);
        let tb = d.tangents(VertexMode::B, 0.4);
        assert!(tb[0].abs() < 1e-12);
        assert!((tb[1] - 0.4).abs() < 1e-12);
        assert!(tb[2].abs() < 1e-12);
        assert!((tb[3] - 0.4).abs() < 1e-12);
    }

    #[test]
    fn theorem6_rotation_product_is_identity() {
        let d = Degree4 {
            alpha: 0.7,
            beta: 1.1,
        };
        for mode in [VertexMode::A, VertexMode::B] {
            for t in [-0.8, -0.2, 0.15, 0.9] {
                let rho = d.fold_angles(mode, t);
                let err = mat_err_ident(d.rotation_product(rho));
                assert!(err < 1e-8, "mode {mode:?} t={t} rotation product err {err}");
            }
        }
    }

    #[test]
    fn p_right_matches_pa_pb_when_beta_is_90() {
        // α = arctan(3/4) ⇒ sin α = 3/5, cos α = 4/5, tan(α/2) = 1/3, p = 1/2.
        let paper = Degree4 {
            alpha: (0.75f64).atan(),
            beta: std::f64::consts::FRAC_PI_2,
        };
        let p = paper.p_right();
        assert!((p - 0.5).abs() < 1e-12, "p={p}");
        assert!((paper.p_a() - p).abs() < 1e-12);
        assert!((paper.p_b() - p).abs() < 1e-12);
    }

    #[test]
    fn kawasaki_holds_for_supplementary_opposites() {
        let a = 0.6_f64;
        let b = 1.2_f64;
        let dirs = [
            [1.0, 0.0],
            [a.cos(), a.sin()],
            [(a + b).cos(), (a + b).sin()],
            [
                (a + b + std::f64::consts::PI - a).cos(),
                (a + b + std::f64::consts::PI - a).sin(),
            ],
        ];
        assert!(kawasaki(dirs));
        let d = degree4_from_dirs(dirs).unwrap();
        assert!((d.alpha - a).abs() < 1e-12);
        assert!((d.beta - b).abs() < 1e-12);
    }
}
