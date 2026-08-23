//! Coupled arm plant: drives a serial chain through Newton–Euler dynamics.
//!
//! Independent 1-DOF [`ServoJoint::step`](crate::kinematics::ServoJoint::step) is
//! still the right test for a single gearbox / valve / tendon. A real arm is
//! coupled: this plant integrates `M(q) q̈ + h = τ` with each joint's
//! [`crate::kinematics::Drive`] reflected inertia on the diagonal.

use super::actuator::Drive;
use super::chain::SerialChain;
use super::servo::{ServoJoint, ServoMode};
use super::transmission::GearTrain;
use super::wrap_deg;

/// Serial chain + one actuator per hinge.
#[derive(Debug, Clone)]
pub struct ArmPlant {
    pub chain: SerialChain,
    pub servos: Vec<ServoJoint>,
    /// Add `G(q)` to the PD command so the arm holds a pose against gravity.
    pub gravity_comp: bool,
}

impl ArmPlant {
    pub fn new(chain: SerialChain, servos: Vec<ServoJoint>) -> Self {
        assert_eq!(chain.n(), servos.len());
        Self {
            chain,
            servos,
            gravity_comp: true,
        }
    }

    pub fn from_train(chain: SerialChain, train: GearTrain) -> Self {
        Self::from_drive(chain, Drive::Gear(train))
    }

    pub fn from_drive(chain: SerialChain, drive: Drive) -> Self {
        let n = chain.n();
        let servos = (0..n)
            .map(|_| ServoJoint::from_drive(drive.clone()))
            .collect();
        Self::new(chain, servos)
    }

    pub fn with_gravity_comp(mut self, on: bool) -> Self {
        self.gravity_comp = on;
        self
    }

    /// Set operating temperature (°C) on every joint drive (oil / grease / cable).
    pub fn set_temp(&mut self, temp_c: f64) {
        for s in &mut self.servos {
            s.drive.set_temp(temp_c);
        }
    }

    pub fn temp_c(&self) -> f64 {
        self.servos
            .first()
            .map(|s| s.drive.env().temp_c)
            .unwrap_or(25.0)
    }

    pub fn seed(&mut self, q_deg: &[f64]) {
        for (s, &q) in self.servos.iter_mut().zip(q_deg) {
            s.theta_deg = q;
            s.omega = 0.0;
            s.theta_cmd_deg = q;
        }
    }

    pub fn q_deg(&self) -> Vec<f64> {
        self.servos.iter().map(|s| s.theta_deg).collect()
    }

    pub fn qd(&self) -> Vec<f64> {
        self.servos.iter().map(|s| s.omega).collect()
    }

    /// Advance toward `q_cmd_deg` with diagonal PD + optional gravity feedforward,
    /// then each joint's [`crate::kinematics::plant::Drive`].
    pub fn step(&mut self, q_cmd_deg: &[f64], dt: f64) {
        let n = self.servos.len();
        let q = self.q_deg();
        let qd = self.qd();
        let m = self.chain.mass_matrix(&q);
        let g = if self.gravity_comp {
            self.chain.gravity_torque(&q)
        } else {
            vec![0.0; n]
        };

        let mut tau = vec![0.0; n];
        let mut extra = vec![0.0; n];
        for i in 0..n {
            extra[i] = self.servos[i].drive.reflected_inertia();
            let servo = &mut self.servos[i];
            servo.theta_cmd_deg = q_cmd_deg[i];
            let cmd = servo.backlash_command(q_cmd_deg[i]);
            let j_gain = m[i * n + i].max(1e-6);
            let tau_pd = match servo.mode {
                ServoMode::Position {
                    stiffness_hz,
                    damping_ratio,
                } => {
                    let wn = 2.0 * std::f64::consts::PI * stiffness_hz;
                    let e = wrap_deg(cmd - servo.theta_deg).to_radians();
                    j_gain * wn * wn * e - 2.0 * damping_ratio * wn * j_gain * servo.omega
                }
                ServoMode::Torque => servo.tau_motor_cmd,
            };
            let tau_des = tau_pd + g[i];
            let out = servo.drive.apply(tau_des, servo.omega, dt);
            servo.tau_motor_cmd = out.tau_cmd_report;
            servo.tau_joint = out.tau_joint;
            servo.tau_ext_est = out.tau_ext_est;
            tau[i] = out.tau_joint;
        }

        let qdd = self.chain.forward_dynamics(&q, &qd, &tau, &extra);
        for ((servo, &acceleration), joint) in self.servos[..n]
            .iter_mut()
            .zip(qdd.iter())
            .zip(self.chain.joints.iter())
        {
            servo.omega += acceleration * dt;
            let mut th = servo.theta_deg + servo.omega.to_degrees() * dt;
            let (lo, hi) = joint.limits;
            if th < lo {
                th = lo;
                if servo.omega < 0.0 {
                    servo.omega = 0.0;
                }
            } else if th > hi {
                th = hi;
                if servo.omega > 0.0 {
                    servo.omega = 0.0;
                }
            }
            servo.theta_deg = th;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kinematics::actuator::Drive;
    use crate::kinematics::SerialChain;

    #[test]
    fn gravity_comp_holds_a_horizontal_pose() {
        let chain = SerialChain::planar_3r();
        let mut plant = ArmPlant::from_train(chain, GearTrain::direct_drive()).with_gravity_comp(true);
        let q = [90.0, 0.0, 0.0];
        plant.seed(&q);
        let dt = 1.0 / 240.0;
        for _ in 0..240 {
            plant.step(&q, dt);
        }
        let got = plant.q_deg();
        assert!((got[0] - 90.0).abs() < 3.0, "shoulder sagged: {:?}", got);
    }

    #[test]
    fn without_comp_the_arm_sags() {
        let chain = SerialChain::planar_3r();
        let mut plant =
            ArmPlant::from_train(chain, GearTrain::direct_drive()).with_gravity_comp(false);
        let q = [90.0, 0.0, 0.0];
        plant.seed(&q);
        for s in &mut plant.servos {
            if let ServoMode::Position { stiffness_hz, .. } = &mut s.mode {
                *stiffness_hz = 2.0;
            }
        }
        let dt = 1.0 / 240.0;
        for _ in 0..480 {
            plant.step(&q, dt);
        }
        assert!(
            plant.q_deg()[0] > 95.0,
            "expected sag toward hang, got {:?}",
            plant.q_deg()
        );
    }

    #[test]
    fn hydraulic_and_tendon_plants_step() {
        let chain = SerialChain::planar_3r();
        for drive in [Drive::hydraulic(), Drive::tendon()] {
            let mut plant = ArmPlant::from_drive(chain.clone(), drive);
            plant.seed(&[-2.8, 88.9, 93.9]);
            for _ in 0..50 {
                plant.step(&[-2.8, 88.9, 93.9], 1.0 / 240.0);
            }
        }
    }
}
