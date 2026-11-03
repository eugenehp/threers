//! Joint drive modalities: gears, hydraulics, and tendons.
//!
//! A revolute joint is always the same constraint. What differs is how torque
//! is produced — and which **fluid / cable / grease** and **temperature** sit
//! between command and joint:
//!
//! | Drive | Dominant non-ideality |
//! |-------|------------------------|
//! | [`crate::kinematics::actuator::GearTrain`] | \(N^2 J_\mathrm{rotor}\), backlash, temp-scaled Stribeck |
//! | [`HydraulicDrive`] | fluid ν(T), valve lag, leak, bulk modulus |
//! | [`TendonDrive`] | material stretch, CTE pretension, routing μ(T) |
//!
//! [`crate::kinematics::actuator::Drive`] is the enum the plant steps through.

use super::materials::{DriveEnv, HydraulicFluid, TendonMaterial};
use super::transmission::{BacklashState, GearTrain, StribeckFriction};

/// How a joint produces torque.
#[derive(Debug, Clone)]
pub enum Drive {
    /// Electric motor through a gear reduction (includes direct-drive as N=1).
    Gear(GearTrain),
    /// Rotary hydraulic motor / vane at the joint.
    Hydraulic(HydraulicDrive),
    /// Antagonistic cable pair; motors treated as proximal (no joint rotor).
    Tendon(TendonDrive),
}

impl Drive {
    pub fn direct() -> Self {
        Self::Gear(GearTrain::direct_drive())
    }
    pub fn qdd() -> Self {
        Self::Gear(GearTrain::qdd())
    }
    pub fn high_ratio() -> Self {
        Self::Gear(GearTrain::high_ratio_servo())
    }
    pub fn hydraulic() -> Self {
        Self::Hydraulic(HydraulicDrive::default())
    }
    pub fn hydraulic_fluid(fluid: HydraulicFluid) -> Self {
        Self::Hydraulic(HydraulicDrive::with_fluid(fluid))
    }
    pub fn tendon() -> Self {
        Self::Tendon(TendonDrive::default())
    }
    pub fn tendon_material(mat: TendonMaterial) -> Self {
        Self::Tendon(TendonDrive::with_material(mat))
    }

    pub fn label(&self) -> &'static str {
        match self {
            Self::Gear(g) if g.ratio <= 1.0 + 1e-9 => "direct",
            Self::Gear(g) if (g.ratio - 15.0).abs() < 1e-6 => "qdd-15:1",
            Self::Gear(_) => "servo-geared",
            Self::Hydraulic(_) => "hydraulic",
            Self::Tendon(_) => "tendon",
        }
    }

    pub fn env(&self) -> DriveEnv {
        match self {
            Self::Gear(g) => g.env,
            Self::Hydraulic(h) => h.env,
            Self::Tendon(t) => t.env,
        }
    }

    pub fn set_temp(&mut self, temp_c: f64) {
        match self {
            Self::Gear(g) => g.env.temp_c = temp_c,
            Self::Hydraulic(h) => h.env.temp_c = temp_c,
            Self::Tendon(t) => t.env.temp_c = temp_c,
        }
    }

    pub fn material_name(&self) -> &'static str {
        match self {
            Self::Gear(_) => "grease",
            Self::Hydraulic(h) => h.fluid.name,
            Self::Tendon(t) => t.material.name,
        }
    }

    /// Inertia added on the joint diagonal, kg·m².
    pub fn reflected_inertia(&self) -> f64 {
        match self {
            Self::Gear(g) => g.reflected_inertia(),
            Self::Hydraulic(h) => h.rotor_inertia,
            Self::Tendon(_) => 0.0,
        }
    }

    pub fn backlash_deg(&self) -> f64 {
        match self {
            Self::Gear(g) => g.backlash_deg,
            Self::Hydraulic(_) | Self::Tendon(_) => 0.0,
        }
    }

    /// Peak joint torque the drive can sustain, N·m.
    pub fn torque_limit(&self) -> f64 {
        match self {
            Self::Gear(g) => g.joint_torque_from_motor(g.motor_torque_limit()),
            Self::Hydraulic(h) => h.pressure_max * h.displacement,
            Self::Tendon(t) => {
                let t_max = t.material.break_n.min(t.tension_max);
                let t0 = t.live_pretension();
                (t_max - t0).max(0.0) * t.radius * 2.0
            }
        }
    }

    pub fn force_snr(&self, tau_ext: f64) -> f64 {
        match self {
            Self::Gear(g) => {
                let f = g.friction.at_temp(g.env.temp_c);
                let noise = f.tau_s.max(f.tau_c) * g.ratio;
                if noise < 1e-15 {
                    return f64::INFINITY;
                }
                tau_ext.abs() / noise
            }
            Self::Hydraulic(h) => {
                let noise = h.fluid.coulomb(h.env.temp_c).max(1e-6);
                tau_ext.abs() / noise
            }
            Self::Tendon(t) => {
                let fr = t
                    .material
                    .routing_friction(t.env.temp_c, t.live_pretension());
                let noise = fr * t.radius * 2.0;
                if noise < 1e-15 {
                    return f64::INFINITY;
                }
                tau_ext.abs() / noise.max(1e-9)
            }
        }
    }

    /// Pass a desired joint torque through the drive; returns applied τ and
    /// updates internal actuator state (`dt` seconds).
    pub fn apply(&mut self, tau_des: f64, omega: f64, dt: f64) -> DriveOutput {
        match self {
            Self::Gear(g) => {
                let n = g.ratio.max(1e-6);
                let eta = g.efficiency.max(1e-3);
                let tau_m =
                    (tau_des / (eta * n)).clamp(-g.motor_torque_limit(), g.motor_torque_limit());
                let tau_j = g.joint_torque_from_motor(tau_m);
                let fric_pack = g.friction.at_temp(g.env.temp_c);
                let fric = fric_pack.torque(omega, tau_j);
                let smooth = fric_pack.torque_smooth(omega);
                DriveOutput {
                    tau_joint: tau_j + fric,
                    tau_cmd_report: tau_m,
                    tau_ext_est: (tau_j + fric) - (tau_j + smooth),
                }
            }
            Self::Hydraulic(h) => h.apply(tau_des, omega, dt),
            Self::Tendon(t) => t.apply(tau_des, omega, dt),
        }
    }
}

/// Result of one drive step.
#[derive(Debug, Clone, Copy)]
pub struct DriveOutput {
    /// Net torque into the plant, N·m.
    pub tau_joint: f64,
    /// Actuator-side command (motor N·m, pressure Pa, or tension N) for HUD.
    pub tau_cmd_report: f64,
    /// Proprioceptive residual, N·m.
    pub tau_ext_est: f64,
}

/// Rotary hydraulic actuator at the joint.
///
/// ```text
/// τ = P·D − b(ν)·ω − coulomb(T)
/// Ṗ = 2π f_valve(ν)·(P* − P) − leak(ν)·P / D_eff
/// ```
#[derive(Debug, Clone, Copy)]
pub struct HydraulicDrive {
    pub fluid: HydraulicFluid,
    pub env: DriveEnv,
    /// Displacement, m³/rad.
    pub displacement: f64,
    /// Max differential pressure, Pa.
    pub pressure_max: f64,
    /// Valve bandwidth at reference fluid (ISO 46 @ 40 °C), Hz.
    pub valve_hz: f64,
    /// Hydraulic motor / fluid inertia at the joint, kg·m².
    pub rotor_inertia: f64,
    /// Current differential pressure, Pa.
    pub pressure: f64,
}

impl Default for HydraulicDrive {
    fn default() -> Self {
        Self::with_fluid(HydraulicFluid::iso_vg_46())
    }
}

impl HydraulicDrive {
    pub fn with_fluid(fluid: HydraulicFluid) -> Self {
        Self {
            fluid,
            env: DriveEnv::default(),
            displacement: 2.0e-6,
            pressure_max: 1.0e7,
            valve_hz: 8.0,
            rotor_inertia: 0.002,
            pressure: 0.0,
        }
    }

    pub fn apply(&mut self, tau_des: f64, omega: f64, dt: f64) -> DriveOutput {
        let t = self.env.temp_c;
        let d = self.displacement.max(1e-12);
        let p_star = (tau_des / d).clamp(-self.pressure_max, self.pressure_max);

        // Valve bandwidth derated by fluid viscosity; soft bulk modulus slows further.
        let bulk_ref = 1.5e9;
        let bulk_scale = (self.fluid.bulk_modulus / bulk_ref).sqrt().clamp(0.5, 1.4);
        let f_valve = self.valve_hz * self.fluid.valve_scale(t) * bulk_scale;
        let wn = 2.0 * std::f64::consts::PI * f_valve.max(0.5);
        self.pressure += (p_star - self.pressure) * (1.0 - (-wn * dt).exp());

        // Internal leakage bleeds differential pressure (hot thin oil → more leak).
        let leak = self.fluid.leak(t);
        // Convert volumetric leak to pressure drop: ΔP ≈ −(leak·P / (D²)) · dt  (lumped)
        let p_leak = leak * self.pressure / (d * d).max(1e-24);
        self.pressure -= p_leak * dt;
        self.pressure = self.pressure.clamp(-self.pressure_max, self.pressure_max);

        let tau_hyd = self.pressure * d;
        let viscous = self.fluid.viscous_coeff(t);
        let coulomb = self.fluid.coulomb(t);
        let sign = friction_sign(omega, tau_hyd);
        let tau_j = tau_hyd - viscous * omega - coulomb * sign;
        DriveOutput {
            tau_joint: tau_j,
            tau_cmd_report: self.pressure,
            tau_ext_est: tau_des - tau_j,
        }
    }
}

/// Antagonistic tendon pair on a joint pulley.
///
/// ```text
/// T0(T) = pretension + CTE·k·L·ΔT
/// τ* → stretch LPF(k(T), c(T)) → τ − μ(T)·r
/// ```
#[derive(Debug, Clone, Copy)]
pub struct TendonDrive {
    pub material: TendonMaterial,
    pub env: DriveEnv,
    /// Pulley radius, m.
    pub radius: f64,
    /// Install pretension at 25 °C, N.
    pub pretension: f64,
    /// Soft tension limit (≤ material break), N.
    pub tension_max: f64,
    /// Stretch-state bandwidth at reference, Hz.
    pub stretch_hz: f64,
    /// Filtered joint torque state, N·m.
    pub tau_state: f64,
    /// Last live tensions (agonist / antagonist), N — for HUD.
    pub tension_plus: f64,
    pub tension_minus: f64,
}

impl Default for TendonDrive {
    fn default() -> Self {
        Self::with_material(TendonMaterial::nylon())
    }
}

impl TendonDrive {
    pub fn with_material(material: TendonMaterial) -> Self {
        let pretension = 80.0;
        Self {
            material,
            env: DriveEnv::default(),
            radius: 0.03,
            pretension,
            tension_max: material.break_n * 0.45,
            stretch_hz: 6.0,
            tau_state: 0.0,
            tension_plus: pretension,
            tension_minus: pretension,
        }
    }

    pub fn live_pretension(&self) -> f64 {
        self.material
            .thermal_pretension_delta(self.env.temp_c, self.pretension)
    }

    pub fn apply(&mut self, tau_des: f64, omega: f64, dt: f64) -> DriveOutput {
        let r = self.radius.max(1e-6);
        let temp = self.env.temp_c;
        let t0 = self.live_pretension();
        let t_max = self.tension_max.min(self.material.break_n);
        let tau_cap = (t_max - t0).max(0.0) * 2.0 * r;
        let tau_star = tau_des.clamp(-tau_cap, tau_cap);

        let k = self.material.stiffness_at(temp);
        let c = self.material.damping_at(temp);
        let k_joint = 2.0 * k * r * r;
        let c_joint = 2.0 * c * r * r;
        // Absolute stiffness sets how fast tension tracks (vs nylon baseline).
        const K_REF: f64 = 1.5e4;
        let hz = self.stretch_hz * (k / K_REF).sqrt().clamp(0.35, 2.5);
        let wn = 2.0 * std::f64::consts::PI * hz;
        let tau_target = tau_star - c_joint * omega * 0.05;
        self.tau_state += (tau_target - self.tau_state) * (1.0 - (-wn * dt).exp());

        // Antagonistic split around pretension.
        let half = self.tau_state / (2.0 * r);
        self.tension_plus = (t0 + half).clamp(0.0, t_max);
        self.tension_minus = (t0 - half).clamp(0.0, t_max);

        let fric_f = self.material.routing_friction(temp, t0);
        let fric = fric_f * r * 2.0;
        let visc = self.material.viscous * r * r * omega;
        let sign = friction_sign(omega, self.tau_state);
        let tau_j = self.tau_state - fric * sign - visc;
        let _ = k_joint;
        DriveOutput {
            tau_joint: tau_j,
            tau_cmd_report: self.tension_plus - self.tension_minus,
            tau_ext_est: tau_des - tau_j,
        }
    }
}

fn friction_sign(omega: f64, tau_hint: f64) -> f64 {
    if omega.abs() < 1e-5 {
        if tau_hint >= 0.0 {
            1.0
        } else {
            -1.0
        }
    } else if omega > 0.0 {
        1.0
    } else {
        -1.0
    }
}

/// Build a backlash mapper for whatever drive is installed.
pub fn backlash_for(drive: &Drive) -> BacklashState {
    BacklashState::new(drive.backlash_deg())
}

/// Shared friction helper for callers that still want a GearTrain handle.
pub fn gear_friction(train: &GearTrain) -> StribeckFriction {
    train.friction.at_temp(train.env.temp_c)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hydraulic_lags_a_step_command() {
        let mut h = HydraulicDrive::default();
        let mut tau = 0.0;
        for _ in 0..5 {
            let o = h.apply(10.0, 0.0, 1.0 / 240.0);
            tau = o.tau_joint;
        }
        assert!(tau > 0.5 && tau < 10.0, "valve should be mid-ramp, got {tau}");
        for _ in 0..500 {
            let o = h.apply(10.0, 0.0, 1.0 / 240.0);
            tau = o.tau_joint;
        }
        assert!((tau - 10.0).abs() < 2.0, "should settle near 10 N·m, got {tau}");
    }

    #[test]
    fn cold_vg68_more_drag_than_hot_vg32() {
        let mut cold = HydraulicDrive::with_fluid(HydraulicFluid::iso_vg_68());
        cold.env.temp_c = 5.0;
        let mut hot = HydraulicDrive::with_fluid(HydraulicFluid::iso_vg_32());
        hot.env.temp_c = 60.0;
        // Steady spin: compare viscous loss magnitude.
        let oc = cold.apply(0.0, 5.0, 1.0 / 240.0).tau_joint.abs();
        let oh = hot.apply(0.0, 5.0, 1.0 / 240.0).tau_joint.abs();
        assert!(oc > oh, "cold heavy oil should drag more: cold={oc} hot={oh}");
    }

    #[test]
    fn tendon_has_no_joint_rotor() {
        assert_eq!(Drive::tendon().reflected_inertia(), 0.0);
        assert!(Drive::high_ratio().reflected_inertia() > 0.1);
    }

    #[test]
    fn tendon_stretch_softens_a_step() {
        let mut t = TendonDrive::default();
        let o0 = t.apply(5.0, 0.0, 1.0 / 240.0);
        assert!(o0.tau_joint.abs() < 5.0);
        let mut last = o0.tau_joint;
        for _ in 0..400 {
            last = t.apply(5.0, 0.0, 1.0 / 240.0).tau_joint;
        }
        assert!(last > o0.tau_joint, "stretch should wind up");
    }

    #[test]
    fn steel_winds_faster_than_nylon() {
        let mut steel = TendonDrive::with_material(TendonMaterial::steel());
        let mut nylon = TendonDrive::with_material(TendonMaterial::nylon());
        let dt = 1.0 / 240.0;
        steel.apply(8.0, 0.0, dt);
        nylon.apply(8.0, 0.0, dt);
        assert!(
            steel.tau_state > nylon.tau_state,
            "steel stretch bandwidth should lead: steel={} nylon={}",
            steel.tau_state,
            nylon.tau_state
        );
    }

    #[test]
    fn set_temp_propagates() {
        let mut d = Drive::hydraulic();
        d.set_temp(-5.0);
        assert!((d.env().temp_c - (-5.0)).abs() < 1e-9);
    }
}
