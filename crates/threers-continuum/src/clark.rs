//! Clark coordinates: two numbers for a segment's bend, whatever the cable
//! count.
//!
//! A three-cable segment has three commands and two degrees of freedom, so one
//! of the three is redundant — and a controller that drives them individually
//! spends its time discovering that, badly. The Clark transform (the same one
//! that turns three motor phases into two) collapses them:
//!
//! ```text
//! c = 2/n · Σ lᵢ (cos θᵢ, sin θᵢ)        lᵢ = c₀ cos θᵢ + c₁ sin θᵢ
//! ```
//!
//! where `θᵢ` is where cable `i` sits around the rod. `c` is a vector: its
//! **direction** is the plane the segment bends in and its **magnitude** is the
//! total bend angle times the cable standoff, so a segment is steered by
//! pointing a two-dimensional stick.
//!
//! ```
//! use threers_continuum::prelude::*;
//!
//! // Three cables at 120°, 4 mm off the backbone.
//! let clark = Clark::new(3, 0.004, 0.0);
//!
//! // Bend by 30° in the plane the first cable sits in: pull that one, pay the
//! // other two out.
//! let pulls = clark.to_pulls([0.004 * 30f32.to_radians(), 0.0]);
//! assert!(pulls[0] > 0.0 && pulls[1] < 0.0 && pulls[2] < 0.0);
//!
//! // ...and it reads back.
//! let back = clark.from_pulls(&pulls);
//! assert!((back[0] - 0.004 * 30f32.to_radians()).abs() < 1e-6);
//! ```

/// The transform for one segment's ring of cables.
#[derive(Debug, Clone, PartialEq)]
pub struct Clark {
    angles: Vec<f32>,
    /// Distance from the backbone to the cables.
    pub standoff: f32,
}

impl Clark {
    /// `n` cables evenly spaced, the first at `phase` radians.
    pub fn new(n: usize, standoff: f32, phase: f32) -> Self {
        let n = n.max(1);
        Self {
            angles: (0..n)
                .map(|i| phase + std::f32::consts::TAU * i as f32 / n as f32)
                .collect(),
            standoff,
        }
    }

    /// Cables at arbitrary angles — an uneven ring, or one calibrated against
    /// hardware that turned out not to be even.
    pub fn from_angles(angles: Vec<f32>, standoff: f32) -> Self {
        Self { angles, standoff }
    }

    pub fn len(&self) -> usize {
        self.angles.len()
    }

    pub fn is_empty(&self) -> bool {
        self.angles.is_empty()
    }

    pub fn angles(&self) -> &[f32] {
        &self.angles
    }

    /// Clark coordinates to the pull each cable needs, in length units.
    ///
    /// Positive is reeled in. The pulls sum to zero for an even ring, which is
    /// what makes this a *bend* and not a squeeze: nothing here shortens the
    /// backbone.
    pub fn to_pulls(&self, clark: [f32; 2]) -> Vec<f32> {
        self.angles
            .iter()
            .map(|a| clark[0] * a.cos() + clark[1] * a.sin())
            .collect()
    }

    /// And back: the pulls a segment is actually holding, as one bend vector.
    ///
    /// The inverse of an over-determined system, so it is a projection: the
    /// component of the pulls that no bend can explain — all three cables
    /// reeled in together — is dropped, because no bend produced it.
    pub fn from_pulls(&self, pulls: &[f32]) -> [f32; 2] {
        let n = self.angles.len().max(1) as f32;
        let mut c = [0.0f32; 2];
        for (a, l) in self.angles.iter().zip(pulls) {
            c[0] += l * a.cos();
            c[1] += l * a.sin();
        }
        [c[0] * 2.0 / n, c[1] * 2.0 / n]
    }

    /// The bend a Clark coordinate asks for: total angle in radians, and the
    /// direction of the plane it bends in.
    ///
    /// The vector points **along** the bend, not against it. A cable that is
    /// reeled in is shorter than the backbone it runs beside, so it is on the
    /// *inside* of the curve and the rod bends toward it — which makes
    /// `to_pulls` and `from_bend` agree without a sign anywhere between them.
    pub fn bend(&self, clark: [f32; 2]) -> (f32, f32) {
        let magnitude = (clark[0] * clark[0] + clark[1] * clark[1]).sqrt();
        let angle = if self.standoff > 0.0 {
            magnitude / self.standoff
        } else {
            0.0
        };
        (angle, clark[1].atan2(clark[0]))
    }

    /// The Clark coordinate for a given bend — the inverse of [`Self::bend`].
    pub fn from_bend(&self, angle: f32, direction: f32) -> [f32; 2] {
        let magnitude = angle * self.standoff;
        [magnitude * direction.cos(), magnitude * direction.sin()]
    }

    /// Clamp a Clark coordinate to a maximum bend angle, keeping its direction.
    ///
    /// A controller that lets the magnitude run wants this: past a certain
    /// curvature the cables on the outside have to *lengthen* more than there
    /// is slack for, and the segment stops tracking the command.
    pub fn clamp(&self, clark: [f32; 2], max_angle: f32) -> [f32; 2] {
        let limit = max_angle.abs() * self.standoff;
        let magnitude = (clark[0] * clark[0] + clark[1] * clark[1]).sqrt();
        if magnitude <= limit || magnitude <= 0.0 {
            return clark;
        }
        let k = limit / magnitude;
        [clark[0] * k, clark[1] * k]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_pull_round_trips_through_the_transform() {
        let clark = Clark::new(3, 0.004, 0.0);
        for c in [[0.001, 0.0], [0.0, -0.002], [0.0015, 0.0007]] {
            let back = clark.from_pulls(&clark.to_pulls(c));
            assert!(
                (back[0] - c[0]).abs() < 1e-7 && (back[1] - c[1]).abs() < 1e-7,
                "{c:?} came back as {back:?}"
            );
        }
    }

    #[test]
    fn four_cables_round_trip_too() {
        let clark = Clark::new(4, 0.01, 0.3);
        let c = [0.002, -0.001];
        let back = clark.from_pulls(&clark.to_pulls(c));
        assert!((back[0] - c[0]).abs() < 1e-7 && (back[1] - c[1]).abs() < 1e-7);
    }

    #[test]
    fn a_bend_pulls_one_side_and_pays_out_the_other() {
        let clark = Clark::new(3, 0.004, 0.0);
        let pulls = clark.to_pulls([0.001, 0.0]);
        assert!(pulls[0] > 0.0, "the cable in the bending plane reels in");
        assert!(pulls[1] < 0.0 && pulls[2] < 0.0, "the other two pay out");
        assert!(
            pulls.iter().sum::<f32>().abs() < 1e-7,
            "an even ring's pulls sum to zero — a bend, not a squeeze"
        );
    }

    #[test]
    fn reeling_every_cable_in_together_is_not_a_bend() {
        // The redundant direction of a three-cable segment: it shortens the
        // whole thing, and no bend can account for it, so it projects out.
        let clark = Clark::new(3, 0.004, 0.0);
        let c = clark.from_pulls(&[0.002, 0.002, 0.002]);
        assert!(c[0].abs() < 1e-7 && c[1].abs() < 1e-7, "got {c:?}");
    }

    #[test]
    fn magnitude_is_the_bend_angle_times_the_standoff() {
        let clark = Clark::new(3, 0.004, 0.0);
        let ninety = std::f32::consts::FRAC_PI_2;
        let c = clark.from_bend(ninety, 0.0);
        let (angle, _) = clark.bend(c);
        assert!((angle - ninety).abs() < 1e-5, "got {angle}");
        assert!(
            (c[0] - 0.004 * ninety).abs() < 1e-6,
            "bending toward +x reels the +x cable in: {c:?}"
        );
        assert!(
            clark.to_pulls(c)[0] > 0.0,
            "...which is what the pulls should say too"
        );
    }

    #[test]
    fn clamping_keeps_the_direction_and_caps_the_angle() {
        let clark = Clark::new(3, 0.004, 0.0);
        let c = clark.from_bend(2.0, 0.7);
        let capped = clark.clamp(c, 0.5);
        let (angle, direction) = clark.bend(capped);
        assert!((angle - 0.5).abs() < 1e-5, "angle {angle}");
        assert!((direction - 0.7).abs() < 1e-4, "direction {direction}");
        // Under the cap it is left alone.
        assert_eq!(clark.clamp(capped, 1.0), capped);
    }
}
