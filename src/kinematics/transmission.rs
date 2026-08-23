//! Gear-train physics for joint actuators.
//!
//! Models the non-ideal output of a real servo:
//!
//! ```text
//! τ_out = η·N·τ_in − f(ω) − backlash(θ) − …
//! J_reflected = N² · J_rotor
//! ```
//!
//! Friction follows a Stribeck curve with a discontinuity at zero velocity
//! (stiction), scaled by operating temperature (grease viscosity). Backlash is
//! a joint-side dead zone that must be traversed on direction reversal. These
//! are the terms that make high-ratio gearboxes poison sim-to-real transfer.

use super::materials::DriveEnv;

/// Stribeck friction parameters, all in joint coordinates (after the gear).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StribeckFriction {
    /// Coulomb friction torque, N·m (joint side).
    pub tau_c: f64,
    /// Static (stiction) peak above Coulomb at ω = 0, N·m.
    pub tau_s: f64,
    /// Stribeck velocity scale, rad/s at the joint.
    pub v_s: f64,
    /// Viscous coefficient, N·m·s/rad.
    pub viscous: f64,
}

impl Default for StribeckFriction {
    fn default() -> Self {
        Self {
            tau_c: 0.0,
            tau_s: 0.0,
            v_s: 0.01,
            viscous: 0.0,
        }
    }
}

impl StribeckFriction {
    /// Smooth Stribeck curve (no stiction hold). Opposes motion.
    pub fn torque_smooth(&self, omega: f64) -> f64 {
        let sign = if omega >= 0.0 { 1.0 } else { -1.0 };
        let abs_w = omega.abs();
        let stribeck = self.tau_c
            + (self.tau_s - self.tau_c) * (-((abs_w / self.v_s).powi(2))).exp();
        -sign * stribeck - self.viscous * omega
    }

    /// Friction opposing motion, with stiction hold when |ω| ≈ 0.
    pub fn torque(&self, omega: f64, tau_applied: f64) -> f64 {
        if omega.abs() < 1e-8 {
            let abs_tau = tau_applied.abs();
            if abs_tau <= self.tau_s {
                return -tau_applied.clamp(-self.tau_s, self.tau_s);
            }
            let sign = if tau_applied >= 0.0 { 1.0 } else { -1.0 };
            return -sign * self.tau_c;
        }
        self.torque_smooth(omega)
    }

    /// Scale a reference (25 °C) friction pack with temperature.
    ///
    /// Viscous term tracks an Arrhenius-like oil/grease viscosity; Coulomb and
    /// stiction drop slightly when warm.
    pub fn at_temp(&self, temp_c: f64) -> Self {
        let dt = temp_c - 25.0;
        let visc_scale = (-0.04 * dt).exp();
        let dry_scale = 1.0 - 0.003 * dt;
        Self {
            tau_c: (self.tau_c * dry_scale.clamp(0.5, 1.5)).max(0.0),
            tau_s: (self.tau_s * dry_scale.clamp(0.5, 1.5)).max(0.0),
            v_s: self.v_s * (1.0 + 0.01 * dt).clamp(0.5, 2.0),
            viscous: (self.viscous * visc_scale).max(0.0),
        }
    }
}

/// A single-stage gear reduction between motor and joint.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GearTrain {
    /// Reduction ratio N = θ_motor / θ_joint (motor revolutions per joint rev).
    pub ratio: f64,
    /// Motor rotor inertia, kg·m².
    pub rotor_inertia: f64,
    /// Mesh efficiency (0..1).
    pub efficiency: f64,
    /// Total joint-side backlash width, degrees.
    pub backlash_deg: f64,
    /// Joint-side friction at the reference temperature in [`Self::env`].
    pub friction: StribeckFriction,
    /// Operating temperature for grease viscosity scaling.
    pub env: DriveEnv,
}

impl Default for GearTrain {
    fn default() -> Self {
        Self::direct_drive()
    }
}

impl GearTrain {
    pub fn direct_drive() -> Self {
        Self {
            ratio: 1.0,
            rotor_inertia: 0.0,
            efficiency: 1.0,
            backlash_deg: 0.0,
            friction: StribeckFriction::default(),
            env: DriveEnv::default(),
        }
    }

    pub fn qdd() -> Self {
        Self {
            ratio: 15.0,
            rotor_inertia: 1.2e-5,
            efficiency: 0.92,
            backlash_deg: 0.15,
            friction: StribeckFriction {
                tau_c: 0.008,
                tau_s: 0.012,
                v_s: 0.02,
                viscous: 0.0004,
            },
            env: DriveEnv::default(),
        }
    }

    pub fn high_ratio_servo() -> Self {
        Self {
            ratio: 288.0,
            // Sized so N² J_rotor dominates the distal links of the planar 3R
            // (~0.4 kg·m² reflected) and the tracking penalty is obvious.
            rotor_inertia: 5.0e-6,
            efficiency: 0.75,
            backlash_deg: 1.2,
            friction: StribeckFriction {
                tau_c: 0.045,
                tau_s: 0.085,
                v_s: 0.005,
                viscous: 0.002,
            },
            env: DriveEnv::default(),
        }
    }

    pub fn reflected_inertia(&self) -> f64 {
        self.ratio * self.ratio * self.rotor_inertia
    }

    #[inline]
    pub fn motor_angle(&self, theta_joint: f64) -> f64 {
        self.ratio * theta_joint
    }

    #[inline]
    pub fn motor_velocity(&self, omega_joint: f64) -> f64 {
        self.ratio * omega_joint
    }

    pub fn joint_torque_from_motor(&self, tau_motor: f64) -> f64 {
        self.efficiency * self.ratio * tau_motor
    }

    pub fn force_snr(&self, tau_ext: f64) -> f64 {
        let f = self.friction.at_temp(self.env.temp_c);
        let noise = f.tau_s.max(f.tau_c) * self.ratio;
        if noise < 1e-15 {
            return f64::INFINITY;
        }
        tau_ext.abs() / noise
    }

    /// Motor-side torque limit that delivers roughly the same joint torque
    /// across ratios (~15 N·m at the joint). That is enough to hold the
    /// planar 3R against gravity so comparisons isolate N² inertia and
    /// friction rather than amp rating.
    pub fn motor_torque_limit(&self) -> f64 {
        let joint_tau = 15.0;
        let n = self.ratio.max(1e-6);
        let eta = self.efficiency.max(1e-3);
        joint_tau / (eta * n)
    }
}

/// Joint-side backlash dead zone.
#[derive(Debug, Clone, Copy)]
pub struct BacklashState {
    half_width_rad: f64,
    out_rad: f64,
}

impl BacklashState {
    pub fn new(backlash_deg: f64) -> Self {
        Self {
            half_width_rad: backlash_deg.to_radians() / 2.0,
            out_rad: 0.0,
        }
    }

    pub fn map_command(&mut self, theta_cmd: f64) -> f64 {
        if self.half_width_rad <= 0.0 {
            self.out_rad = theta_cmd;
            return theta_cmd;
        }
        let gap = theta_cmd - self.out_rad;
        if gap > self.half_width_rad {
            self.out_rad = theta_cmd - self.half_width_rad;
        } else if gap < -self.half_width_rad {
            self.out_rad = theta_cmd + self.half_width_rad;
        }
        self.out_rad
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reflected_inertia_scales_with_n_squared() {
        let base = GearTrain {
            ratio: 1.0,
            rotor_inertia: 1e-5,
            ..GearTrain::direct_drive()
        };
        let scaled = GearTrain {
            ratio: 10.0,
            rotor_inertia: 1e-5,
            ..GearTrain::direct_drive()
        };
        assert!((scaled.reflected_inertia() / base.reflected_inertia() - 100.0).abs() < 1e-6);
    }

    #[test]
    fn stiction_holds_until_breakaway() {
        let f = StribeckFriction {
            tau_c: 0.02,
            tau_s: 0.05,
            v_s: 0.01,
            viscous: 0.0,
        };
        assert!(f.torque(0.0, 0.03).abs() > 0.029);
        assert!(f.torque(0.0, 0.08).abs() < 0.021);
    }

    #[test]
    fn high_ratio_blinds_gentle_contact() {
        let qdd = GearTrain::qdd();
        let high = GearTrain::high_ratio_servo();
        assert!(qdd.force_snr(0.01) > high.force_snr(0.01));
    }

    #[test]
    fn gear_viscous_drops_when_hot() {
        let f = StribeckFriction {
            tau_c: 0.05,
            tau_s: 0.08,
            v_s: 0.02,
            viscous: 0.01,
        };
        assert!(f.at_temp(60.0).viscous < f.at_temp(25.0).viscous);
    }

    #[test]
    fn backlash_dead_zone_on_reversal() {
        let mut b = BacklashState::new(2.0);
        let a0 = b.map_command(0.0);
        let a1 = b.map_command(0.5_f64.to_radians());
        assert!((a1 - a0).abs() < 1e-12);
        let forward = b.map_command(5.0_f64.to_radians());
        assert!(forward > a0 + 1.0_f64.to_radians());
        let back = b.map_command(-5.0_f64.to_radians());
        assert!(back < forward - 2.0_f64.to_radians());
    }
}
