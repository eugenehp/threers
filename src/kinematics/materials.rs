//! Working fluids, cable materials, and temperature for joint drives.
//!
//! These are *engineering* models sized for the planar 3R plant — enough to
//! make oil grade, cable type, and °C change tracking — not CFD or FEA.

/// Ambient / oil / cable temperature shared by a drive step.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DriveEnv {
    /// Operating temperature, °C.
    pub temp_c: f64,
}

impl Default for DriveEnv {
    fn default() -> Self {
        Self { temp_c: 25.0 }
    }
}

impl DriveEnv {
    pub fn new(temp_c: f64) -> Self {
        Self { temp_c }
    }

    pub fn kelvin(self) -> f64 {
        self.temp_c + 273.15
    }
}

// ---------------------------------------------------------------------------
// Hydraulic fluid
// ---------------------------------------------------------------------------

/// Hydraulic working fluid (ISO VG oils, water-glycol, silicone).
///
/// Viscosity uses the ASTM D341 / Walther form between the 40 °C and 100 °C
/// kinematic viscosities. Bulk modulus softens the valve response; leakage
/// scales as pressure / viscosity.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HydraulicFluid {
    pub name: &'static str,
    /// Density at ~15 °C, kg/m³.
    pub density: f64,
    /// Kinematic viscosity at 40 °C, cSt (mm²/s).
    pub nu_40_cst: f64,
    /// Kinematic viscosity at 100 °C, cSt.
    pub nu_100_cst: f64,
    /// Effective bulk modulus of the fluid+hose, Pa.
    pub bulk_modulus: f64,
    /// Internal leakage conductance at reference viscosity, m³/(s·Pa).
    pub leak_ref: f64,
    /// Seal Coulomb scale at 25 °C, N·m (before fluid scaling).
    pub seal_coulomb: f64,
    /// Viscous drag scale: τ_v = scale · μ(T) · ω, with μ in Pa·s.
    pub viscous_scale: f64,
}

impl HydraulicFluid {
    /// ISO VG 32 — light machine oil, quick valve, more leak when hot.
    pub fn iso_vg_32() -> Self {
        Self {
            name: "ISO VG 32",
            density: 870.0,
            nu_40_cst: 32.0,
            nu_100_cst: 5.4,
            bulk_modulus: 1.4e9,
            leak_ref: 2.5e-13,
            seal_coulomb: 0.28,
            viscous_scale: 4.0e-3,
        }
    }

    /// ISO VG 46 — common industrial hydraulic oil (default).
    pub fn iso_vg_46() -> Self {
        Self {
            name: "ISO VG 46",
            density: 875.0,
            nu_40_cst: 46.0,
            nu_100_cst: 6.8,
            bulk_modulus: 1.5e9,
            leak_ref: 1.8e-13,
            seal_coulomb: 0.35,
            viscous_scale: 4.5e-3,
        }
    }

    /// ISO VG 68 — heavier; sluggish when cold.
    pub fn iso_vg_68() -> Self {
        Self {
            name: "ISO VG 68",
            density: 880.0,
            nu_40_cst: 68.0,
            nu_100_cst: 8.7,
            bulk_modulus: 1.55e9,
            leak_ref: 1.2e-13,
            seal_coulomb: 0.42,
            viscous_scale: 5.5e-3,
        }
    }

    /// Water-glycol fire-resistant fluid — lower viscosity index, more leak.
    pub fn water_glycol() -> Self {
        Self {
            name: "water-glycol",
            density: 1060.0,
            nu_40_cst: 36.0,
            nu_100_cst: 6.0,
            bulk_modulus: 2.2e9, // less air affinity
            leak_ref: 4.0e-13,
            seal_coulomb: 0.22,
            viscous_scale: 3.5e-3,
        }
    }

    /// Silicone damping fluid — high viscosity, soft bulk modulus.
    pub fn silicone() -> Self {
        Self {
            name: "silicone",
            density: 970.0,
            nu_40_cst: 100.0,
            nu_100_cst: 35.0,
            bulk_modulus: 0.9e9,
            leak_ref: 0.6e-13,
            seal_coulomb: 0.5,
            viscous_scale: 8.0e-3,
        }
    }

    pub fn from_name(name: &str) -> Option<Self> {
        match name.to_ascii_lowercase().as_str() {
            "iso32" | "vg32" | "iso vg 32" => Some(Self::iso_vg_32()),
            "iso46" | "vg46" | "iso vg 46" | "oil" | "default" => Some(Self::iso_vg_46()),
            "iso68" | "vg68" | "iso vg 68" => Some(Self::iso_vg_68()),
            "water-glycol" | "water_glycol" | "glycol" | "hfc" => Some(Self::water_glycol()),
            "silicone" | "si" => Some(Self::silicone()),
            _ => None,
        }
    }

    /// Kinematic viscosity (cSt) at `temp_c` via Walther interpolation.
    pub fn nu_cst(&self, temp_c: f64) -> f64 {
        walther_nu(self.nu_40_cst, self.nu_100_cst, temp_c)
    }

    /// Dynamic viscosity μ = ν·ρ, Pa·s.
    pub fn mu(&self, temp_c: f64) -> f64 {
        let nu_m2_s = self.nu_cst(temp_c) * 1e-6;
        nu_m2_s * self.density
    }

    /// Viscous joint coefficient b(T), N·m·s/rad.
    pub fn viscous_coeff(&self, temp_c: f64) -> f64 {
        self.viscous_scale * self.mu(temp_c)
    }

    /// Seal Coulomb at temperature (warms soften slightly).
    pub fn coulomb(&self, temp_c: f64) -> f64 {
        let f = 1.0 + 0.004 * (25.0 - temp_c); // ~0.4%/°C
        (self.seal_coulomb * f).max(0.05)
    }

    /// Leakage conductance at temperature, m³/(s·Pa).
    pub fn leak(&self, temp_c: f64) -> f64 {
        let mu_ref = self.mu(40.0).max(1e-6);
        self.leak_ref * (mu_ref / self.mu(temp_c).max(1e-9))
    }

    /// Valve bandwidth derate when cold (high ν slows spool).
    pub fn valve_scale(&self, temp_c: f64) -> f64 {
        let nu = self.nu_cst(temp_c).max(1.0);
        (46.0 / nu).sqrt().clamp(0.25, 1.6)
    }
}

/// ASTM D341-style Walther: log10(log10(ν+0.7)) = A + B·log10(T_K).
fn walther_nu(nu40: f64, nu100: f64, temp_c: f64) -> f64 {
    let t = (temp_c + 273.15).clamp(250.0, 450.0);
    let y = |nu: f64| (nu + 0.7).max(1.01).log10().log10();
    let x = |tk: f64| tk.log10();
    let y40 = y(nu40);
    let y100 = y(nu100);
    let x40 = x(313.15);
    let x100 = x(373.15);
    let b = (y100 - y40) / (x100 - x40);
    let a = y40 - b * x40;
    let yt = a + b * x(t);
    let log_nu = 10f64.powf(yt);
    (10f64.powf(log_nu) - 0.7).max(0.5)
}

// ---------------------------------------------------------------------------
// Tendon / cable material
// ---------------------------------------------------------------------------

/// Cable / tendon material for antagonistic drives.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TendonMaterial {
    pub name: &'static str,
    /// Axial stiffness of one free length, N/m (at 25 °C).
    pub stiffness: f64,
    /// Series damping, N·s/m.
    pub damping: f64,
    /// Linear thermal expansion, 1/K.
    pub cte: f64,
    /// Sheath / pulley Coulomb friction (force), N at 25 °C and unit pretension scale.
    pub mu_force: f64,
    /// Extra viscous drag along the cable, N·s/m → joint via r².
    pub viscous: f64,
    /// Softening: stiffness × (1 − soften·ΔT), ΔT = T−25.
    pub soften_per_c: f64,
    /// Breaking tension, N.
    pub break_n: f64,
    /// Free length used for thermal pretension shift, m.
    pub free_length: f64,
}

impl TendonMaterial {
    /// 7×19 stainless wire rope — stiff, high routing friction.
    pub fn steel() -> Self {
        Self {
            name: "steel",
            stiffness: 8.0e4,
            damping: 120.0,
            cte: 1.2e-5,
            mu_force: 6.0,
            viscous: 15.0,
            soften_per_c: 0.0002,
            break_n: 1200.0,
            free_length: 0.45,
        }
    }

    /// UHMWPE (Dyneema / Spectra) — light, low stretch, low friction.
    pub fn uhmwpe() -> Self {
        Self {
            name: "uhmwpe",
            stiffness: 5.0e4,
            damping: 60.0,
            cte: -1.2e-5, // slight contraction when warm
            mu_force: 2.5,
            viscous: 6.0,
            soften_per_c: 0.0015,
            break_n: 900.0,
            free_length: 0.45,
        }
    }

    /// Nylon / polymer — stretchy, strong temp softening (default soft hand).
    pub fn nylon() -> Self {
        Self {
            name: "nylon",
            stiffness: 1.5e4,
            damping: 90.0,
            cte: 8.0e-5,
            mu_force: 5.0,
            viscous: 20.0,
            soften_per_c: 0.008,
            break_n: 500.0,
            free_length: 0.45,
        }
    }

    /// Kevlar / aramid — stiff, low CTE.
    pub fn aramid() -> Self {
        Self {
            name: "aramid",
            stiffness: 6.5e4,
            damping: 70.0,
            cte: -2.0e-6,
            mu_force: 4.0,
            viscous: 10.0,
            soften_per_c: 0.0008,
            break_n: 1000.0,
            free_length: 0.45,
        }
    }

    pub fn from_name(name: &str) -> Option<Self> {
        match name.to_ascii_lowercase().as_str() {
            "steel" | "wire" | "cable" => Some(Self::steel()),
            "uhmwpe" | "dyneema" | "spectra" | "hmpe" => Some(Self::uhmwpe()),
            "nylon" | "polymer" | "default" => Some(Self::nylon()),
            "aramid" | "kevlar" => Some(Self::aramid()),
            _ => None,
        }
    }

    pub fn stiffness_at(&self, temp_c: f64) -> f64 {
        let s = 1.0 - self.soften_per_c * (temp_c - 25.0);
        (self.stiffness * s.clamp(0.2, 1.5)).max(1e3)
    }

    pub fn damping_at(&self, temp_c: f64) -> f64 {
        // Polymers damp more when warm; steel almost flat.
        let s = 1.0 + 0.5 * self.soften_per_c * (temp_c - 25.0);
        (self.damping * s).max(1.0)
    }

    /// Routing friction force (N) at pretension `t0`.
    pub fn routing_friction(&self, temp_c: f64, pretension: f64) -> f64 {
        let warm = 1.0 - 0.003 * (temp_c - 25.0);
        let load = (pretension / 80.0).sqrt().clamp(0.4, 2.5);
        (self.mu_force * warm.max(0.4) * load).max(0.2)
    }

    /// Thermal pretension shift from free expansion against fixed anchors, N.
    pub fn thermal_pretension_delta(&self, temp_c: f64, t0_install: f64) -> f64 {
        let dt = temp_c - 25.0;
        // ΔL = α L ΔT; ΔT_force ≈ k ΔL, but expansion into slack lowers tension.
        let d_t = -self.cte * self.free_length * dt * self.stiffness_at(temp_c);
        (t0_install + d_t).max(5.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cold_oil_is_thicker() {
        let f = HydraulicFluid::iso_vg_46();
        assert!(f.nu_cst(0.0) > f.nu_cst(40.0));
        assert!(f.nu_cst(40.0) > f.nu_cst(80.0));
        assert!((f.nu_cst(40.0) - 46.0).abs() < 3.0);
    }

    #[test]
    fn silicone_more_viscous_than_vg32() {
        let a = HydraulicFluid::iso_vg_32().viscous_coeff(25.0);
        let b = HydraulicFluid::silicone().viscous_coeff(25.0);
        assert!(b > a * 2.0);
    }

    #[test]
    fn nylon_softens_when_hot() {
        let n = TendonMaterial::nylon();
        assert!(n.stiffness_at(60.0) < n.stiffness_at(25.0) * 0.85);
    }

    #[test]
    fn steel_thermal_pretension_rises_when_cold() {
        let s = TendonMaterial::steel();
        let t_cold = s.thermal_pretension_delta(0.0, 80.0);
        let t_hot = s.thermal_pretension_delta(50.0, 80.0);
        assert!(t_cold > t_hot);
    }
}
