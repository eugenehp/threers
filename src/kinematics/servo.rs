//! Per-joint actuator: drive modality + PD state.
//!
//! Position mode issues a joint-space PD torque, then routes it through
//! [`crate::kinematics::servo::Drive`] (gears, hydraulics, or tendons). Torque mode bypasses the PD.

use super::actuator::{backlash_for, Drive, DriveOutput};
use super::transmission::{BacklashState, GearTrain};

/// How the outer loop is driven.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ServoMode {
    /// Joint-space PD → [`Drive::apply`].
    Position {
        stiffness_hz: f64,
        damping_ratio: f64,
    },
    /// Raw desired joint torque → [`Drive::apply`].
    Torque,
}

impl Default for ServoMode {
    fn default() -> Self {
        Self::Position {
            stiffness_hz: 15.0,
            damping_ratio: 1.0,
        }
    }
}

/// One joint: pose state + drive + outer-loop mode.
#[derive(Debug, Clone)]
pub struct ServoJoint {
    pub drive: Drive,
    pub mode: ServoMode,
    /// Load inertia for independent 1-DOF tests (plant uses M(q)).
    pub load_inertia: f64,
    pub theta_deg: f64,
    pub omega: f64,
    /// Last actuator-side command (motor N·m / Pa / tension N).
    pub tau_motor_cmd: f64,
    /// Last joint torque delivered by the drive, N·m.
    pub tau_joint: f64,
    pub tau_ext_est: f64,
    pub tau_max: f64,
    backlash: BacklashState,
    pub theta_cmd_deg: f64,
}

impl ServoJoint {
    pub fn new(train: GearTrain) -> Self {
        Self::from_drive(Drive::Gear(train))
    }

    pub fn from_drive(drive: Drive) -> Self {
        let backlash = backlash_for(&drive);
        let tau_max = drive.torque_limit();
        Self {
            drive,
            mode: ServoMode::default(),
            load_inertia: 0.05,
            theta_deg: 0.0,
            omega: 0.0,
            tau_motor_cmd: 0.0,
            tau_joint: 0.0,
            tau_ext_est: 0.0,
            tau_max,
            backlash,
            theta_cmd_deg: 0.0,
        }
    }

    pub fn with_mode(mut self, mode: ServoMode) -> Self {
        self.mode = mode;
        self
    }

    pub fn with_load_inertia(mut self, j: f64) -> Self {
        self.load_inertia = j;
        self
    }

    /// Back-compat: gear train when the drive is geared.
    pub fn train(&self) -> GearTrain {
        match &self.drive {
            Drive::Gear(g) => *g,
            _ => GearTrain::direct_drive(),
        }
    }

    fn gain_inertia(&self) -> f64 {
        self.load_inertia.max(1e-12)
    }

    fn plant_inertia(&self) -> f64 {
        (self.load_inertia + self.drive.reflected_inertia()).max(1e-12)
    }

    pub fn pd_joint_torque(&self, theta_cmd_deg: f64) -> f64 {
        let ServoMode::Position {
            stiffness_hz,
            damping_ratio,
        } = self.mode
        else {
            return 0.0;
        };
        let j = self.gain_inertia();
        let wn = 2.0 * std::f64::consts::PI * stiffness_hz;
        let e = super::wrap_deg(theta_cmd_deg - self.theta_deg).to_radians();
        j * wn * wn * e - 2.0 * damping_ratio * wn * j * self.omega
    }

    pub(crate) fn backlash_command(&mut self, theta_cmd_deg: f64) -> f64 {
        self.backlash
            .map_command(theta_cmd_deg.to_radians())
            .to_degrees()
    }

    /// Independent 1-DOF step (gearbox / drive unit test).
    pub fn step(&mut self, theta_cmd_deg: f64, tau_ext: f64, dt: f64) {
        self.theta_cmd_deg = theta_cmd_deg;
        let cmd = self.backlash_command(theta_cmd_deg);
        let tau_des = match self.mode {
            ServoMode::Position { .. } => self.pd_joint_torque(cmd) + tau_ext,
            ServoMode::Torque => self.tau_motor_cmd + tau_ext,
        };
        let DriveOutput {
            tau_joint,
            tau_cmd_report,
            tau_ext_est,
        } = self.drive.apply(tau_des - tau_ext, self.omega, dt);
        // Re-add external after the drive so contact still accelerates the load.
        let tau_net = tau_joint + tau_ext;
        self.tau_motor_cmd = tau_cmd_report;
        self.tau_joint = tau_joint;
        self.tau_ext_est = tau_ext_est + tau_ext;

        let j = self.plant_inertia();
        let alpha = tau_net / j;
        self.omega += alpha * dt;
        self.theta_deg += (self.omega * dt).to_degrees();
    }

    pub fn tracking_error_deg(&self) -> f64 {
        super::wrap_deg(self.theta_cmd_deg - self.theta_deg).abs()
    }

    pub fn force_snr(&self) -> f64 {
        self.drive.force_snr(self.tau_ext_est)
    }
}

pub fn chain(count: usize, train: GearTrain) -> Vec<ServoJoint> {
    (0..count).map(|_| ServoJoint::new(train)).collect()
}

pub fn chain_drive(count: usize, drive: Drive) -> Vec<ServoJoint> {
    (0..count)
        .map(|_| ServoJoint::from_drive(drive.clone()))
        .collect()
}

pub mod presets {
    use super::*;
    use crate::kinematics::actuator::Drive;
    use crate::kinematics::transmission::GearTrain;

    pub fn direct_drive(count: usize) -> Vec<ServoJoint> {
        chain(count, GearTrain::direct_drive())
    }

    pub fn qdd(count: usize) -> Vec<ServoJoint> {
        chain(count, GearTrain::qdd())
    }

    pub fn high_ratio(count: usize) -> Vec<ServoJoint> {
        chain(count, GearTrain::high_ratio_servo())
    }

    pub fn hydraulic(count: usize) -> Vec<ServoJoint> {
        chain_drive(count, Drive::hydraulic())
    }

    pub fn tendon(count: usize) -> Vec<ServoJoint> {
        chain_drive(count, Drive::tendon())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kinematics::actuator::Drive;
    use crate::kinematics::transmission::GearTrain;

    #[test]
    fn high_ratio_lags_low_ratio_on_step() {
        let mut ideal = ServoJoint::new(GearTrain::direct_drive());
        let mut high = ServoJoint::new(GearTrain::high_ratio_servo());
        high.mode = ServoMode::Position {
            stiffness_hz: 15.0,
            damping_ratio: 1.0,
        };

        let dt = 1.0 / 240.0;
        for _ in 0..2400 {
            ideal.step(30.0, 0.0, dt);
            high.step(30.0, 0.0, dt);
        }
        assert!(ideal.tracking_error_deg() < high.tracking_error_deg());
    }

    #[test]
    fn gentle_contact_visible_on_qdd_not_high_ratio() {
        let qdd = GearTrain::qdd();
        let high = GearTrain::high_ratio_servo();
        let gentle = 0.01;
        assert!(qdd.force_snr(gentle) > high.force_snr(gentle));

        let mut jq = ServoJoint::new(qdd);
        let mut jh = ServoJoint::new(high);
        jq.mode = ServoMode::Torque;
        jh.mode = ServoMode::Torque;
        let contact = 0.12;
        let dt = 1.0 / 240.0;
        for _ in 0..240 {
            jq.step(0.0, contact, dt);
            jh.step(0.0, contact, dt);
        }
        let eq = (jq.tau_ext_est - contact).abs();
        let eh = (jh.tau_ext_est - contact).abs();
        assert!(eq < eh || jq.force_snr() > jh.force_snr());
    }

    #[test]
    fn tendon_drive_steps_without_panic() {
        let mut j = ServoJoint::from_drive(Drive::tendon());
        for _ in 0..100 {
            j.step(10.0, 0.0, 1.0 / 240.0);
        }
    }
}
