//! The flames as one volume of burning gas: threers' 3D fluid on Metal, fed
//! at every flame's foot in its colour (see `threers_candles_set_fire`).

use std::collections::BTreeMap;

use threers::fluid::FluidConfig3d;
use threers::metal::{MetalDevice, MetalFluid3d};
use threers::prelude::*;

use crate::table::{Placed, CALM};

/// The fire's volume: the table's square, half as high as it is wide.
const FIRE_MIN: [f32; 3] = [-0.62, 0.0, -0.62];
const FIRE_MAX: [f32; 3] = [0.62, 0.62, 0.62];
/// Flames nearer than this, in metres, lean into each other and burn as one.
const MIX_REACH: f32 = 0.09;
/// A flame's own colour, toward which a colour's `amount` < 1 leans.
const FLAME_COLOR: [f32; 3] = [1.0, 0.55, 0.22];

pub(crate) struct Fire {
    pub(crate) sim: MetalFluid3d,
    pub(crate) device: MetalDevice,
}

impl Fire {
    pub(crate) fn new(device: &MetalDevice) -> Option<Self> {
        let config = FluidConfig3d {
            resolution: 64,
            dye_resolution: 128,
            // Hot gas rises hard and curls, and is gone a hand's height up:
            // a flame, not a column of smoke.
            buoyancy: 150.0,
            curl: 4.0,
            density_dissipation: 0.7,
            velocity_dissipation: 0.6,
            splat_radius: 0.012,
            absorption: 140.0,
            emission: 2.2,
            march_steps: 64,
            transparent: true,
            bounds_min: FIRE_MIN,
            bounds_max: FIRE_MAX,
            ..FluidConfig3d::default()
        };
        let sim = MetalFluid3d::new(device, config).ok()?;
        Some(Fire {
            sim,
            device: device.clone(),
        })
    }

    /// Where `x`, `y`, `z` (metres) is in the volume's unit cube.
    fn unit(x: f32, y: f32, z: f32) -> [f32; 3] {
        let p = [x, y, z];
        std::array::from_fn(|i| (p[i] - FIRE_MIN[i]) / (FIRE_MAX[i] - FIRE_MIN[i]))
    }

    /// Feed every flame into the gas and move it on by `dt`. Flames near
    /// each other feed the same gas, so their plumes run together and their
    /// colours mix.
    /// `amount` (0–1) is how much fire: how bright the gas burns and how
    /// high it rises.
    pub(crate) fn burn(
        &mut self,
        candles: &BTreeMap<u32, Placed>,
        time: f32,
        dt: f32,
        amount: f32,
    ) -> bool {
        let dt = dt.clamp(1.0 / 240.0, 1.0 / 20.0);
        // Every frame, in proportion to it: a steady stream, not puffs.
        let k = dt * 30.0;
        let (bright, rise) = (0.3 + 0.7 * amount, 0.55 + 0.6 * amount);
        let feed: Vec<_> = candles
            .iter()
            .map(|(id, p)| {
                // A wandering base, each its own, so the plume breathes.
                let s = *id as f32 * 1.618;
                let wx = 0.0015 * (time * 2.1 + s).sin();
                let wz = 0.0015 * (time * 1.7 + s * 2.0).cos();
                // A calm burn: a third of the flicker, at half its pace. A
                // flare throws the flame up, brighter, for a moment.
                let flicker = threers::extras::candle::flicker(time * CALM, s);
                let f = (1.0 + (flicker - 1.0) * 0.35) * (1.0 + 0.7 * p.flare);
                // Toward the flames near it, harder the nearer: two that
                // stand close draw the same air and burn into one.
                let (mut lx, mut lz) = (0.0, 0.0);
                for (other, q) in candles {
                    let (dx, dz) = (q.at.0 - p.at.0, q.at.1 - p.at.1);
                    let d = (dx * dx + dz * dz).sqrt();
                    if other != id && d < MIX_REACH && d > 1e-4 {
                        let pull = 1.0 - d / MIX_REACH;
                        (lx, lz) = (lx + dx / d * pull, lz + dz / d * pull);
                    }
                }
                (
                    Self::unit(p.at.0 + wx, 0.03, p.at.1 + wz),
                    [
                        (wx * 4000.0 + lx * 500.0) * k,
                        300.0 * rise * f * k,
                        (wz * 4000.0 + lz * 500.0) * k,
                    ],
                    fire_color(p.color).map(|c| c * bright * f * k),
                )
            })
            .collect();
        if self.sim.splat_batch(&self.device, &feed).is_err() {
            return false;
        }
        self.sim.update(&self.device, dt).is_ok()
    }
}

/// The colour a flame feeds the fire: its own lerped `amount` of the way to
/// the one asked for, at full brightness either way — a dark end of a
/// gradient still burns, in its hue.
fn fire_color(color: Option<(Color, f32)>) -> [f32; 3] {
    let Some((c, amount)) = color else {
        return FLAME_COLOR;
    };
    let peak = c.r.max(c.g).max(c.b).max(1e-4);
    let a = amount.clamp(0.0, 1.0);
    let to = [c.r / peak, c.g / peak, c.b / peak];
    std::array::from_fn(|i| FLAME_COLOR[i] + (to[i] - FLAME_COLOR[i]) * a)
}
