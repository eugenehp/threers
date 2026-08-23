//! Serial revolute chains: frames, geometric Jacobian, Newton–Euler dynamics.
//!
//! Each [`RevoluteJoint`] is a proper hinge — an origin, a unit axis, and a
//! rigid link to the next origin — not a free ball. Forward kinematics is the
//! product of those hinges. The geometric Jacobian is `ω × r` at the tool, so
//! IK does not have to finite-difference the chain. Inverse dynamics is
//! recursive Newton–Euler, which is what the coupled plant integrates.

use super::{dot, norm, unit, V3};

pub(crate) type M3 = [[f64; 3]; 3];

pub(crate) fn add(a: V3, b: V3) -> V3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

pub(crate) fn sub(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

pub(crate) fn scale(v: V3, s: f64) -> V3 {
    [v[0] * s, v[1] * s, v[2] * s]
}

pub(crate) fn cross(a: V3, b: V3) -> V3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

pub(crate) fn rot(axis: V3, deg: f64) -> M3 {
    let a = unit(axis);
    let (s, c) = deg.to_radians().sin_cos();
    let t = 1.0 - c;
    [
        [
            t * a[0] * a[0] + c,
            t * a[0] * a[1] - s * a[2],
            t * a[0] * a[2] + s * a[1],
        ],
        [
            t * a[0] * a[1] + s * a[2],
            t * a[1] * a[1] + c,
            t * a[1] * a[2] - s * a[0],
        ],
        [
            t * a[0] * a[2] - s * a[1],
            t * a[1] * a[2] + s * a[0],
            t * a[2] * a[2] + c,
        ],
    ]
}

pub(crate) fn mul(a: M3, b: M3) -> M3 {
    let mut o = [[0.0; 3]; 3];
    for (r, row) in o.iter_mut().enumerate() {
        for (c, v) in row.iter_mut().enumerate() {
            *v = (0..3).map(|k| a[r][k] * b[k][c]).sum();
        }
    }
    o
}

pub(crate) fn apply(m: M3, v: V3) -> V3 {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
}

fn ident() -> M3 {
    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
}

/// One hinge in a serial chain: axis in the local frame, then a rigid link.
#[derive(Debug, Clone, Copy)]
pub struct RevoluteJoint {
    /// Unit axis in this joint's local frame (before its own rotation).
    pub axis: V3,
    /// Vector from this hinge to the next, in the frame *after* the rotation.
    pub link: V3,
    /// Mass of the outboard link, kg.
    pub mass: f64,
    /// Centre of mass from the hinge, same frame as [`Self::link`].
    pub com: V3,
    /// Travel limits, degrees.
    pub limits: (f64, f64),
}

impl RevoluteJoint {
    pub fn along_z(length: f64, mass: f64) -> Self {
        Self {
            axis: [0.0, 1.0, 0.0],
            link: [0.0, 0.0, length],
            mass,
            com: [0.0, 0.0, 0.5 * length],
            limits: (-179.0, 179.0),
        }
    }
}

/// World pose of one hinge after FK.
#[derive(Debug, Clone, Copy)]
pub struct JointPose {
    /// Hinge origin, model units.
    pub origin: V3,
    /// World-space unit axis.
    pub axis: V3,
    /// Distal end of this link (next hinge, or the tool).
    pub distal: V3,
    /// World orientation of the frame after this joint's rotation (columns).
    pub rotation: M3,
}

/// Serial product of revolute hinges.
#[derive(Debug, Clone)]
pub struct SerialChain {
    pub joints: Vec<RevoluteJoint>,
    /// Gravity in m/s², world.
    pub gravity: V3,
    /// Metres per model length unit (0.001 if the model is millimetres).
    pub metres: f64,
}

impl SerialChain {
    /// 400 + 300 + 200 mm planar 3R, joints about +Y, +Z up. Masses scale with
    /// length at 2 kg/m.
    pub fn planar_3r() -> Self {
        let mm = [400.0, 300.0, 200.0];
        let kg_per_m = 2.0;
        Self {
            joints: mm
                .into_iter()
                .map(|l| RevoluteJoint::along_z(l, kg_per_m * l * 0.001))
                .collect(),
            gravity: [0.0, 0.0, -9.81],
            metres: 0.001,
        }
    }

    pub fn n(&self) -> usize {
        self.joints.len()
    }

    pub fn tool(&self, q: &[f64]) -> (V3, V3) {
        let poses = self.poses(q);
        let tip = poses.last().map(|p| p.distal).unwrap_or([0.0; 3]);
        let axis = poses
            .last()
            .map(|p| unit(sub(p.distal, p.origin)))
            .unwrap_or([0.0, 0.0, 1.0]);
        (tip, axis)
    }

    pub fn poses(&self, q: &[f64]) -> Vec<JointPose> {
        assert!(q.len() >= self.n());
        let mut r = ident();
        let mut p = [0.0; 3];
        let mut out = Vec::with_capacity(self.n());
        for (j, &angle) in self.joints.iter().zip(q.iter()) {
            let origin = p;
            let axis_w = unit(apply(r, j.axis));
            r = mul(r, rot(j.axis, angle));
            let distal = add(p, apply(r, j.link));
            out.push(JointPose {
                origin,
                axis: axis_w,
                distal,
                rotation: r,
            });
            p = distal;
        }
        out
    }

    /// Base + each distal, so a 3R arm yields 4 points.
    pub fn skeleton(&self, q: &[f64]) -> Vec<V3> {
        let poses = self.poses(q);
        let mut pts = vec![[0.0; 3]];
        for p in &poses {
            pts.push(p.distal);
        }
        pts
    }

    /// Geometric Jacobian columns: `Jv = axis × (tip − origin)`, `Jw = axis`.
    /// Length unit matches the model (mm), so `Jv · dq_rad` is in mm.
    pub fn jacobian(&self, q: &[f64]) -> (Vec<V3>, Vec<V3>) {
        let poses = self.poses(q);
        let tip = poses.last().map(|p| p.distal).unwrap_or([0.0; 3]);
        let mut jv = Vec::with_capacity(self.n());
        let mut jw = Vec::with_capacity(self.n());
        for p in &poses {
            jv.push(cross(p.axis, sub(tip, p.origin)));
            jw.push(p.axis);
        }
        (jv, jw)
    }

    /// Inverse dynamics τ(q, q̇, q̈) in N·m. `q` is degrees; rates are SI.
    pub fn inverse_dynamics(&self, q: &[f64], qd: &[f64], qdd: &[f64], gravity: bool) -> Vec<f64> {
        let n = self.n();
        let poses = self.poses(q);
        let s = self.metres;
        let g = if gravity {
            self.gravity
        } else {
            [0.0; 3]
        };

        let mut omega = vec![[0.0; 3]; n];
        let mut alpha = vec![[0.0; 3]; n];
        let mut a_com = vec![[0.0; 3]; n];
        let mut r_link = vec![[0.0; 3]; n];
        let mut r_com = vec![[0.0; 3]; n];
        let mut u_rod = vec![[0.0; 3]; n];
        let mut ixx = vec![0.0; n];

        let mut w_prev = [0.0; 3];
        let mut al_prev = [0.0; 3];
        let mut a_origin = scale(g, -1.0);

        for i in 0..n {
            let axis = poses[i].axis;
            let qd_i = qd.get(i).copied().unwrap_or(0.0);
            let qdd_i = qdd.get(i).copied().unwrap_or(0.0);
            omega[i] = add(w_prev, scale(axis, qd_i));
            alpha[i] = add(
                add(al_prev, scale(axis, qdd_i)),
                cross(w_prev, scale(axis, qd_i)),
            );
            r_link[i] = scale(sub(poses[i].distal, poses[i].origin), s);
            r_com[i] = scale(apply(poses[i].rotation, self.joints[i].com), s);
            let len = norm(r_link[i]);
            u_rod[i] = if len > 1e-12 {
                scale(r_link[i], 1.0 / len)
            } else {
                [0.0, 0.0, 1.0]
            };
            let m = self.joints[i].mass;
            ixx[i] = m * len * len / 12.0;
            a_com[i] = add(
                add(a_origin, cross(alpha[i], r_com[i])),
                cross(omega[i], cross(omega[i], r_com[i])),
            );
            a_origin = add(
                add(a_origin, cross(alpha[i], r_link[i])),
                cross(omega[i], cross(omega[i], r_link[i])),
            );
            w_prev = omega[i];
            al_prev = alpha[i];
        }

        let mut tau = vec![0.0; n];
        let mut f_next = [0.0; 3];
        let mut n_next = [0.0; 3];
        for i in (0..n).rev() {
            let m = self.joints[i].mass;
            let f = add(f_next, scale(a_com[i], m));
            // Thin-rod I about CoM: (m L²/12) (I − uuᵀ).
            let u = u_rod[i];
            let i_alpha = {
                let w = alpha[i];
                let proj = dot(w, u);
                scale(sub(w, scale(u, proj)), ixx[i])
            };
            let i_omega = {
                let w = omega[i];
                let proj = dot(w, u);
                scale(sub(w, scale(u, proj)), ixx[i])
            };
            let spin = add(i_alpha, cross(omega[i], i_omega));
            let n_here = add(
                add(
                    add(n_next, cross(r_com[i], scale(a_com[i], m))),
                    cross(r_link[i], f_next),
                ),
                spin,
            );
            tau[i] = dot(n_here, poses[i].axis);
            f_next = f;
            n_next = n_here;
        }
        tau
    }

    /// Joint-space mass matrix, kg·m², row-major n×n.
    pub fn mass_matrix(&self, q: &[f64]) -> Vec<f64> {
        let n = self.n();
        let qd = vec![0.0; n];
        let mut m = vec![0.0; n * n];
        for j in 0..n {
            let mut qdd = vec![0.0; n];
            qdd[j] = 1.0;
            let col = self.inverse_dynamics(q, &qd, &qdd, false);
            for i in 0..n {
                m[i * n + j] = col[i];
            }
        }
        m
    }

    /// Coriolis + gravity bias h(q, q̇), N·m.
    pub fn bias(&self, q: &[f64], qd: &[f64]) -> Vec<f64> {
        let qdd = vec![0.0; self.n()];
        self.inverse_dynamics(q, qd, &qdd, true)
    }

    pub fn gravity_torque(&self, q: &[f64]) -> Vec<f64> {
        let z = vec![0.0; self.n()];
        self.inverse_dynamics(q, &z, &z, true)
    }

    /// q̈ = M⁻¹ (τ − h). `tau` is N·m; extra diagonal inertia (reflected rotors)
    /// is added before the solve.
    pub fn forward_dynamics(&self, q: &[f64], qd: &[f64], tau: &[f64], extra_diag: &[f64]) -> Vec<f64> {
        let n = self.n();
        let mut m = self.mass_matrix(q);
        for i in 0..n {
            m[i * n + i] += extra_diag.get(i).copied().unwrap_or(0.0);
        }
        let h = self.bias(q, qd);
        let mut b: Vec<f64> = (0..n).map(|i| tau[i] - h[i]).collect();
        if !linsolve(&mut m, &mut b, n) {
            return vec![0.0; n];
        }
        b
    }
}

fn linsolve(m: &mut [f64], y: &mut [f64], n: usize) -> bool {
    for i in 0..n {
        let mut piv = i;
        for r in i + 1..n {
            if m[r * n + i].abs() > m[piv * n + i].abs() {
                piv = r;
            }
        }
        if m[piv * n + i].abs() < 1e-14 {
            return false;
        }
        if piv != i {
            for c in 0..n {
                m.swap(i * n + c, piv * n + c);
            }
            y.swap(i, piv);
        }
        let d = m[i * n + i];
        for c in i..n {
            m[i * n + c] /= d;
        }
        y[i] /= d;
        for r in 0..n {
            if r != i && m[r * n + i] != 0.0 {
                let f = m[r * n + i];
                for c in i..n {
                    m[r * n + c] -= f * m[i * n + c];
                }
                y[r] -= f * y[i];
            }
        }
    }
    true
}

/// Planar 3R convenience wrapper used by demos and older call sites.
#[derive(Debug, Clone)]
pub struct PlanarArm {
    pub chain: SerialChain,
}

impl Default for PlanarArm {
    fn default() -> Self {
        Self::three_link()
    }
}

impl PlanarArm {
    pub fn three_link() -> Self {
        Self {
            chain: SerialChain::planar_3r(),
        }
    }

    pub fn tool(&self, q: &[f64]) -> (V3, V3) {
        self.chain.tool(q)
    }

    /// `[base, j1, j2, j3, tip]` — `tip` duplicates the last distal.
    pub fn joint_positions(&self, q: &[f64]) -> [V3; 5] {
        let sk = self.chain.skeleton(q);
        let mut out = [[0.0; 3]; 5];
        for (i, p) in sk.iter().take(4).enumerate() {
            out[i] = *p;
        }
        out[4] = out[3];
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_pose_stands_straight_up() {
        let arm = PlanarArm::three_link();
        let (p, ax) = arm.tool(&[0.0, 0.0, 0.0]);
        assert!((p[2] - 900.0).abs() < 1e-9);
        assert!((ax[2] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn hinge_axis_stays_along_y() {
        let c = SerialChain::planar_3r();
        let poses = c.poses(&[30.0, -40.0, 20.0]);
        for p in poses {
            assert!((p.axis[1] - 1.0).abs() < 1e-9, "{:?}", p.axis);
            assert!(p.axis[0].abs() < 1e-9 && p.axis[2].abs() < 1e-9);
        }
    }

    #[test]
    fn geometric_jacobian_matches_numeric() {
        let c = SerialChain::planar_3r();
        let q = [20.0, -35.0, 50.0];
        let (jv, _) = c.jacobian(&q);
        // Central difference in radians — forward difference at 1e-5 truncates
        // by ~½ H L ≈ 4e-3 mm on a 900 mm arm.
        const H: f64 = 1e-6;
        for i in 0..3 {
            let mut qp = q;
            let mut qm = q;
            qp[i] += H.to_degrees();
            qm[i] -= H.to_degrees();
            let (pp, _) = c.tool(&qp);
            let (pm, _) = c.tool(&qm);
            let num = scale(sub(pp, pm), 0.5 / H);
            for k in 0..3 {
                assert!(
                    (jv[i][k] - num[k]).abs() < 1e-5,
                    "col {i} {k}: {:?} vs {:?}",
                    jv[i],
                    num
                );
            }
        }
    }

    #[test]
    fn mass_matrix_is_spd() {
        let c = SerialChain::planar_3r();
        let m = c.mass_matrix(&[10.0, 20.0, 30.0]);
        for i in 0..3 {
            assert!(m[i * 3 + i] > 0.0);
            for j in 0..3 {
                assert!((m[i * 3 + j] - m[j * 3 + i]).abs() < 1e-8);
            }
        }
    }

    #[test]
    fn gravity_is_zero_when_stacked_on_z() {
        let c = SerialChain::planar_3r();
        let g = c.gravity_torque(&[0.0, 0.0, 0.0]);
        for t in g {
            assert!(t.abs() < 1e-8, "{t}");
        }
    }

    #[test]
    fn gravity_pulls_a_horizontal_arm() {
        let c = SerialChain::planar_3r();
        // 90° at the shoulder: links lie along +X, gravity −Z makes a restoring
        // (negative) shoulder torque.
        let g = c.gravity_torque(&[90.0, 0.0, 0.0]);
        assert!(g[0] < -0.1, "expected sag torque, got {:?}", g);
    }
}
