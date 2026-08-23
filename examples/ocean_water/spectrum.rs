//! The sea state a preset implies: how big the waves are, and how big the
//! biggest of them is.
//!
//! This used to be the wave field itself — a JONSWAP spectrum sampled into a few
//! dozen directional Gerstner components, evaluated per vertex and per pixel.
//! [`crate::ocean_fft`] replaced all of that with a spectrum on a full lattice,
//! so what survives here is the pair of bulk numbers the cascades are calibrated
//! against and the report prints.

use crate::preset::Preset;

/// Gravitational acceleration. Real units: the dispersion relation is `w^2 = gk`,
/// so getting this wrong makes waves of the right size move at the wrong speed.
pub const G: f32 = 9.81;

/// The two numbers that describe a sea, both in metres.
#[derive(Clone, Copy, Debug)]
pub struct SeaState {
    /// Significant wave height — the mean of the highest third, and the number a
    /// mariner would quote. The cascades are scaled so their variance matches it.
    pub significant_height: f32,
    /// Wavelength of the spectral peak: the size of the dominant wave.
    pub peak_wavelength: f32,
}

impl SeaState {
    pub fn from_preset(p: &Preset) -> SeaState {
        let u = p.wind_speed.max(0.5);
        // Fully developed sea (Pierson-Moskowitz): w_p = 0.877 g / U, and
        // H_s ~ 0.22 U^2/g. An explicit `peak_wavelength` overrides the size
        // without touching the energy — that independence is the whole point of
        // having both knobs.
        let peak_wavelength = if p.peak_wavelength > 0.0 {
            p.peak_wavelength
        } else {
            let w_p = 0.877 * G / u;
            2.0 * std::f32::consts::PI * G / (w_p * w_p)
        };
        SeaState {
            significant_height: 0.22 * u * u / G * p.amplitude,
            peak_wavelength,
        }
    }
}

/// A rigid body floating on the surface, resolved by sampling under several
/// points of its hull rather than one.
///
/// One sample can only tell you where the water is, so a single-point float
/// tracks the surface exactly and never rolls. Sampling a ring and fitting
/// through it gives heave *and* attitude, and makes the response depend on the
/// object's size: a hull longer than the wave rides over it instead of following
/// it into every trough.
pub struct Float {
    /// Where it sits in the horizontal plane. It bobs; it does not drift.
    pub anchor: [f32; 2],
    /// Half-width of the hull, metres — the radius the samples are taken over.
    pub radius: f32,
    /// Fraction of the surface slope the attitude follows. A real hull is
    /// ballasted and lags the water it rides.
    pub stiffness: f32,
}

/// Resolved pose: world position, and the up axis the hull settles onto.
pub struct Pose {
    pub position: [f32; 3],
    pub up: [f32; 3],
}

/// Taps around the hull: enough to average out a wavelength comparable to it
/// without the cost growing into something worth moving to the GPU.
const RING: usize = 6;

impl Float {
    /// Surface samples one hull needs: its centre plus [`RING`].
    pub const TAPS: usize = 1 + RING;

    /// Where the hull has to know the surface: its centre, then a ring of six.
    ///
    /// Split out from [`Float::resolve`] so that the GPU probe and the CPU
    /// approximation ask about the *same* points in the same order — the probe
    /// answers by index, and an index that meant something else would put the
    /// hull's bow where its stern is.
    pub fn query_points(&self) -> [[f32; 2]; Self::TAPS] {
        let mut pts = [[0.0f32; 2]; Self::TAPS];
        pts[0] = self.anchor;
        for i in 0..RING {
            let a = std::f32::consts::TAU * i as f32 / RING as f32;
            pts[1 + i] = [
                self.anchor[0] + self.radius * a.cos(),
                self.anchor[1] + self.radius * a.sin(),
            ];
        }
        pts
    }

    /// `sample` returns the surface height and slope at a world point. Taking it
    /// as a closure keeps this independent of *how* the sea is generated.
    pub fn resolve(&self, sample: impl Fn(f32, f32) -> (f32, [f32; 2])) -> Pose {
        let pts = self.query_points();
        let mut taps = [(0.0f32, [0.0f32; 2]); Self::TAPS];
        for (t, p) in taps.iter_mut().zip(pts) {
            *t = sample(p[0], p[1]);
        }
        self.resolve_from(&taps)
    }

    /// The same answer from taps already in hand — the probe's path.
    pub fn resolve_from(&self, taps: &[(f32, [f32; 2])]) -> Pose {
        if taps.len() < Self::TAPS {
            return Pose {
                position: [self.anchor[0], 0.0, self.anchor[1]],
                up: [0.0, 1.0, 0.0],
            };
        }
        let centre_y = taps[0].0;
        let mut mean_y = 0.0f32;
        let mut slope = [0.0f32; 2];
        for (y, g) in taps[1..=RING].iter() {
            mean_y += y;
            slope[0] += g[0];
            slope[1] += g[1];
        }
        let inv = 1.0 / RING as f32;
        // Heave is the mean surface height under the hull, not the height at its
        // centre: that is what stops a long hull snapping into every trough.
        let y = mean_y * inv * 0.65 + centre_y * 0.35;

        // The surface normal of a height field is (-dy/dx, 1, -dy/dz).
        let n = [
            -slope[0] * inv * self.stiffness,
            1.0,
            -slope[1] * inv * self.stiffness,
        ];
        let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt().max(1e-9);
        Pose {
            position: [self.anchor[0], y, self.anchor[1]],
            up: [n[0] / len, n[1] / len, n[2] / len],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn significant_height_tracks_wind() {
        let p = crate::preset::all()[1];
        let s = SeaState::from_preset(&p);
        assert!(
            (s.significant_height - 0.22 * p.wind_speed * p.wind_speed / G * p.amplitude).abs()
                < 1e-4
        );
        // A moderate wind is metre-scale, not centimetre or ten-metre.
        assert!(s.significant_height > 0.5 && s.significant_height < 3.0);
    }

    #[test]
    fn an_overridden_peak_does_not_change_the_energy() {
        // Size and energy are independent knobs; that is what lets a storm be
        // both violent and readable from a human vantage.
        let mut a = crate::preset::all()[3];
        a.peak_wavelength = 0.0;
        let mut b = a;
        b.peak_wavelength = 130.0;
        let (sa, sb) = (SeaState::from_preset(&a), SeaState::from_preset(&b));
        assert!((sa.significant_height - sb.significant_height).abs() < 1e-6);
        assert!((sb.peak_wavelength - 130.0).abs() < 1e-6);
    }

    #[test]
    fn float_follows_a_slope_without_capsizing() {
        let f = Float {
            anchor: [40.0, -25.0],
            radius: 3.0,
            stiffness: 0.7,
        };
        // A plane tilted in x: height and slope are exact and known.
        let pose = f.resolve(|x, _z| (0.2 * x, [0.2, 0.0]));
        assert!((pose.position[1] - 0.2 * 40.0).abs() < 0.2);
        assert!(pose.up[1] > 0.5, "capsized: {:?}", pose.up);
        assert!(pose.up[0] < 0.0, "should lean into the up-slope");
        let n = (pose.up[0].powi(2) + pose.up[1].powi(2) + pose.up[2].powi(2)).sqrt();
        assert!((n - 1.0).abs() < 1e-4);
    }
}
