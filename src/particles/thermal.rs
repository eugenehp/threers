//! Glowing particles that heat and cool by physics: embers and sparks as
//! burning grains of char.
//!
//! Each particle carries a temperature, and its glow is a black body's at it
//! (see [`blackbody`](super::render::blackbody)). It loses heat by radiation,
//! `ε σ (T⁴ − T_air⁴)`, and by convection to the air rushing past it, `h (T
//! − T_air)` with `h` from the Ranz–Marshall correlation `Nu = 2 + 0.6 Re^½
//! Pr^⅓` — so an ember in flight cools faster than one at rest. If it is
//! burning it gains the heat of its char's oxidation, the oxygen reaching its
//! surface limited by diffusion through the air round it (the same
//! correlation, as a Sherwood number) and by the reaction's own rate, an
//! Arrhenius `A e^(−E/RT)`, in series: hot, diffusion rules and blowing on an
//! ember makes it glow brighter; cold, the reaction stalls and it goes out.
//! Burning eats it: it shrinks as its carbon goes, and is gone when it is
//! ash. Come to rest on something cold (snow), it is quenched by contact.
//!
//! The numbers are charcoal's: density 400 kg/m³, heat capacity 1000 J/kg K,
//! emissivity 0.9, an activation energy of 130 kJ/mol. A 2 mm ember at rest
//! in still air settles near 1290 K, bright orange; blown at 3 m/s it climbs
//! to 1700 K, yellow, and burns three times faster. One merely warm (600 K)
//! does not catch; one landing on snow is dark in half a second.

use super::{ParticleSystem, Range};

/// The Stefan–Boltzmann constant, W/m² K⁴.
const SIGMA: f32 = 5.670_374e-8;
/// Air's thermal conductivity and kinematic viscosity in the hot film round
/// an ember (at about 800 K): W/m K, m²/s.
const K_AIR: f32 = 0.057;
const NU_AIR: f32 = 1.0e-4;
/// Oxygen's diffusivity in that film, m²/s, and its density in the air,
/// kg/m³.
const D_O2: f32 = 1.0e-4;
const RHO_O2: f32 = 0.27;
/// Heat released per kilogram of oxygen burning carbon to carbon monoxide,
/// J/kg; the carbon used per kilogram of oxygen.
const HEAT_PER_O2: f32 = 6.9e6;
const CARBON_PER_O2: f32 = 0.75;
/// Arrhenius: activation energy over the gas constant, K, and the
/// pre-exponential rate, m/s.
const E_OVER_R: f32 = 130_000.0 / 8.314;
const RATE: f32 = 5.0e6;
/// Pr^⅓ for air.
const PR_THIRD: f32 = 0.888;

/// How a system's particles hold and lose heat.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Thermal {
    /// Temperature at birth, kelvin.
    pub initial: Range,
    /// The air's, kelvin.
    pub ambient: f32,
    /// A particle's physical diameter as a share of its drawn size: what
    /// glows is a grain; what is drawn is the grain and its glow.
    pub diameter: f32,
    /// kg/m³, J/kg K, and 0–1.
    pub density: f32,
    pub heat_capacity: f32,
    pub emissivity: f32,
    /// Burning char: gaining the heat of its oxidation, and eaten by it.
    pub burning: bool,
    /// Heat transfer to what it rests on, W/m² K (0: none; snow about
    /// 3000), and that surface's temperature. Resting on it, a particle
    /// stops burning.
    pub contact: f32,
    pub surface_temperature: f32,
}

impl Thermal {
    /// Burning embers of charcoal in air at `ambient` kelvin.
    pub fn ember(ambient: f32) -> Self {
        Self {
            initial: Range::new(1100.0, 1350.0),
            ambient,
            diameter: 0.05,
            density: 400.0,
            heat_capacity: 1000.0,
            emissivity: 0.9,
            burning: true,
            contact: 0.0,
            surface_temperature: ambient,
        }
    }

    /// The rate of change of a grain's temperature, K/s, and of its
    /// diameter, m/s: at `t` kelvin, `d` metres across, the air going past
    /// at `speed`, burning or not, resting on a surface or not.
    pub fn rates(&self, t: f32, d: f32, speed: f32, burning: bool, resting: bool) -> (f32, f32) {
        let d = d.max(1e-5);
        let area = std::f32::consts::PI * d * d;
        let mass = self.density * std::f32::consts::PI * d * d * d / 6.0;
        let re = speed * d / NU_AIR;
        let nu = 2.0 + 0.6 * re.sqrt() * PR_THIRD;
        let h = nu * K_AIR / d;
        let ta = self.ambient;
        let mut loss = area * (self.emissivity * SIGMA * (t.powi(4) - ta.powi(4)) + h * (t - ta));
        if resting && self.contact > 0.0 {
            loss += area * self.contact * (t - self.surface_temperature);
        }
        let mut gain = 0.0;
        let mut shrink = 0.0;
        if burning && !(resting && self.contact > 0.0) {
            // Oxygen to the surface: diffusion and reaction in series.
            let diffusion = nu * D_O2 * RHO_O2 / d;
            let reaction = RHO_O2 * RATE * (-E_OVER_R / t.max(1.0)).exp();
            let flux = diffusion * reaction / (diffusion + reaction);
            gain = area * HEAT_PER_O2 * flux;
            // dm/dt = −A · carbon flux; for a sphere, dd/dt = −2 · flux / ρ.
            shrink = 2.0 * CARBON_PER_O2 * flux / self.density;
        }
        ((gain - loss) / (mass * self.heat_capacity), -shrink)
    }
}

impl ParticleSystem {
    /// Heat and cool every particle `dt` seconds (with [`thermal`](ParticleSystem::thermal)
    /// set); those burnt to ash are retired.
    pub(crate) fn heat(&mut self, dt: f32) {
        let Some(th) = self.thermal else { return };
        let n = self.particles.len();
        let ps = &mut self.particles;
        // The newly born: their temperature, and all of their size.
        while ps.temperature.len() < n {
            ps.temperature.push(th.initial.sample(&mut self.rng));
        }
        while ps.remaining.len() < n {
            ps.remaining.push(1.0);
        }
        for i in 0..n {
            let d0 = ps.size[i] * th.diameter;
            let air = self
                .air
                .get(i)
                .copied()
                .filter(|a| !a.x.is_nan())
                .unwrap_or(self.forces.wind);
            let speed = (ps.velocity[i] - air).length();
            let mut t = ps.temperature[i];
            let mut r = ps.remaining[i];
            // Explicit steps no longer than half the grain's thermal time
            // constant: a small hot grain changes fast.
            let tau = th.density * d0.max(1e-5) * r * th.heat_capacity
                / (6.0
                    * (4.0 * th.emissivity * SIGMA * t.max(th.ambient).powi(3)
                        + 3000.0f32.min(th.contact + 60.0)));
            let steps = ((dt / (0.5 * tau.max(1e-5))).ceil() as usize).clamp(1, 64);
            let h = dt / steps as f32;
            for _ in 0..steps {
                let (dt_dt, dd_dt) = th.rates(t, d0 * r, speed, th.burning, ps.resting[i]);
                t = (t + dt_dt * h).max(th.ambient.min(th.surface_temperature));
                r = (r + dd_dt * h / d0.max(1e-6)).max(0.0);
            }
            ps.temperature[i] = t;
            ps.remaining[i] = r;
            if r < 0.2 {
                // Ash.
                ps.age[i] = f32::INFINITY;
            }
        }
    }
}
