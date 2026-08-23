//! What a flexible rod is, and what beam theory says its links should be.

/// A uniform flexible rod, described the way a data sheet describes one.
///
/// Nothing here is a simulation parameter except [`links`](Self::links), which
/// is the one honest knob: how finely to chop a continuum that has no joints in
/// it into rigid pieces that do.
///
/// Units are the model's own. Everything is consistent as long as `youngs` and
/// `density` are quoted in the same length unit as `length` and `radius` — SI
/// throughout, or millimetre-based throughout, but not half of each.
#[derive(Debug, Clone, PartialEq)]
pub struct Rod {
    /// Arc length of the whole rod.
    pub length: f32,
    /// How many rigid links stand in for it. The rod is built from `links + 1`
    /// bodies — the two ends are half-links — with `links` elastic stations
    /// between them.
    pub links: usize,
    /// What the rod is drawn and collided as, and what its mass comes from.
    pub radius: f32,
    /// The load-bearing core, which is what the stiffness comes from. Equal to
    /// `radius` for a bare rod, much smaller for a spine inside a sheath.
    pub core_radius: f32,
    /// Inner radius, for a tube. Applies to the core.
    pub bore: f32,
    /// Young's modulus.
    pub youngs: f32,
    /// Poisson's ratio, which turns `youngs` into the shear modulus torsion
    /// needs.
    pub poisson: f32,
    /// Mass per unit volume of the rod **as drawn** — see
    /// [`link_properties`](Self::link_properties) for why that matters.
    pub density: f32,
    /// Damping as a fraction of the station's own stiffness. A rod with none
    /// rings forever; 0.01 to 0.1 covers most real materials.
    pub damping_ratio: f32,
    /// Give each station a third hinge, about the rod's own axis.
    ///
    /// Off by default. Torsion costs a constraint row per link and a rod bent
    /// in one plane never uses it — but a rod pulled by tendons at three
    /// different angles does, and a segment above another segment does most of
    /// all.
    pub twist: bool,
    /// How far one station may bend, in radians. `None` is unlimited.
    ///
    /// Per *station*: `links` of them make up the rod, so a limit of ±0.1 rad
    /// on a 30-link rod still allows three radians of total curl.
    pub station_limit: Option<[f32; 2]>,
    /// How many independently actuated sections the rod is divided into, root
    /// to tip.
    pub segments: usize,
    /// Whether the rod's own links collide with each other.
    ///
    /// Off by default, and rarely worth turning on: a rod that can touch itself
    /// costs `n²` narrow-phase pairs to discover that it usually does not.
    pub self_collide: bool,
}

impl Default for Rod {
    /// A 300 mm silicone-ish rod at 30 links. Not a real material — override
    /// what you know and leave the rest.
    fn default() -> Self {
        Self {
            length: 0.3,
            links: 30,
            radius: 0.006,
            core_radius: 0.006,
            bore: 0.0,
            youngs: 1.0e6,
            poisson: 0.45,
            density: 1200.0,
            damping_ratio: 0.02,
            twist: false,
            station_limit: None,
            segments: 1,
            self_collide: false,
        }
    }
}

/// What one link of a discretised rod weighs, and how hard the station beside
/// it resists.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LinkProperties {
    /// Length of a full link. The two end links are half this.
    pub length: f32,
    /// Mass of a full link. The two end links are half this.
    pub mass: f32,
    /// Bending stiffness at one station, in torque per radian.
    pub bend_stiffness: f32,
    /// Torsional stiffness at one station.
    pub torsion_stiffness: f32,
    pub bend_damping: f32,
    pub torsion_damping: f32,
}

impl Rod {
    /// A rod of the given length and link count, everything else default.
    pub fn new(length: f32, links: usize) -> Self {
        Self {
            length,
            links: links.max(1),
            ..Self::default()
        }
    }

    /// Set the material: Young's modulus, Poisson's ratio and density.
    pub fn material(mut self, youngs: f32, poisson: f32, density: f32) -> Self {
        self.youngs = youngs;
        self.poisson = poisson;
        self.density = density;
        self
    }

    /// Set the drawn radius, and the core with it.
    pub fn radius(mut self, radius: f32) -> Self {
        self.radius = radius;
        self.core_radius = radius;
        self
    }

    /// Set the load-bearing core separately from the drawn radius.
    pub fn core(mut self, core_radius: f32) -> Self {
        self.core_radius = core_radius;
        self
    }

    /// Divide the rod into independently actuated sections.
    pub fn segments(mut self, segments: usize) -> Self {
        self.segments = segments.max(1);
        self
    }

    /// Add torsion at every station.
    pub fn twisting(mut self) -> Self {
        self.twist = true;
        self
    }

    /// Damping as a fraction of stiffness.
    pub fn damped(mut self, ratio: f32) -> Self {
        self.damping_ratio = ratio.max(0.0);
        self
    }

    /// Second moment of area of the load-bearing core.
    pub fn second_moment(&self) -> f32 {
        let (ro, ri) = (self.core_radius.max(0.0), self.bore.max(0.0));
        (std::f32::consts::PI / 4.0) * (ro.powi(4) - ri.powi(4)).max(0.0)
    }

    /// Cross-sectional area of the rod as drawn — what its mass comes from.
    pub fn area(&self) -> f32 {
        let (ro, ri) = (self.radius.max(0.0), self.bore.max(0.0));
        std::f32::consts::PI * (ro * ro - ri * ri).max(0.0)
    }

    /// Shear modulus, `E / 2(1+ν)`.
    pub fn shear_modulus(&self) -> f32 {
        self.youngs / (2.0 * (1.0 + self.poisson))
    }

    /// Total mass of the rod.
    pub fn mass(&self) -> f32 {
        self.density * self.area() * self.length
    }

    /// Beam theory, per link and per station:
    ///
    /// ```text
    /// link length = L/n         link mass = ρ·A·L/n
    /// K_bend      = n·E·Iₓ / L  Iₓ = π/4 (rₒ⁴ − rᵢ⁴)
    /// K_twist     = n·G·I_z / L  I_z = 2Iₓ,  G = E / 2(1+ν)
    /// ```
    ///
    /// The `n/L` is the whole trick: a station stands in for `L/n` of rod, and
    /// a shorter piece of the same beam is a stiffer hinge. Double the links
    /// and each station doubles its stiffness, so the rod as a whole bends the
    /// same and merely does it more smoothly. That invariance is what makes
    /// `links` a fidelity knob rather than a physics one, and
    /// `convergence_is_in_the_shape_not_the_stiffness` in the tests is what
    /// checks it.
    ///
    /// # Two radii
    ///
    /// Stiffness comes from [`core_radius`](Self::core_radius) and mass from
    /// [`radius`](Self::radius), because a spine inside a sheath has almost all
    /// of one and almost none of the other. Getting this backwards on a 6 mm
    /// finger over a 0.5 mm spine overstates the stiffness by `(6/0.5)⁴` —
    /// twenty thousand times, and the rod does not visibly bend at all.
    ///
    /// MuJoCo models the same robot the other way round, taking mass from the
    /// backbone too and making up the difference with an invented density.
    /// Set `radius == core_radius` to match a model written that way.
    pub fn link_properties(&self) -> LinkProperties {
        let n = self.links.max(1) as f32;
        let l = self.length.max(f32::MIN_POSITIVE);
        let i_x = self.second_moment();
        let bend = n * self.youngs * i_x / l;
        let torsion = n * self.shear_modulus() * (2.0 * i_x) / l;
        LinkProperties {
            length: l / n,
            mass: self.density * self.area() * l / n,
            bend_stiffness: bend,
            torsion_stiffness: torsion,
            bend_damping: self.damping_ratio * bend,
            torsion_damping: self.damping_ratio * torsion,
        }
    }

    /// Which links belong to `segment` (1-based from the root), as a half-open
    /// range over station indices.
    ///
    /// Any remainder goes to the segments nearest the root, which is the end
    /// carrying every tendon above it.
    pub fn segment_stations(&self, segment: usize) -> std::ops::Range<usize> {
        let n = self.segments.max(1);
        if segment == 0 || segment > n {
            return 0..0;
        }
        let base = self.links / n;
        let extra = self.links % n;
        let start: usize = (0..segment - 1)
            .map(|i| base + usize::from(i < extra))
            .sum();
        let len = base + usize::from(segment - 1 < extra);
        start..start + len
    }

    /// Arc length from the root to the tip of `segment`.
    pub fn segment_end(&self, segment: usize) -> f32 {
        let stations = self.segment_stations(segment);
        stations.end as f32 * self.length / self.links.max(1) as f32
    }

    /// The tip-side body index of `segment` — where a tendon terminating there
    /// is anchored.
    pub fn segment_tip_link(&self, segment: usize) -> usize {
        self.segment_stations(segment).end.min(self.links)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Spring steel: 200 GPa, 0.5 mm radius, 300 mm long.
    fn steel() -> Rod {
        Rod::new(0.3, 30)
            .radius(0.0005)
            .material(200.0e9, 0.3, 7800.0)
    }

    #[test]
    fn stiffness_is_the_beam_formula() {
        let rod = steel();
        let i_x = std::f32::consts::PI / 4.0 * 0.0005f32.powi(4);
        let expected = 30.0 * 200.0e9 * i_x / 0.3;
        let got = rod.link_properties().bend_stiffness;
        assert!(
            (got - expected).abs() < 1e-4 * expected,
            "station stiffness {got}, beam theory {expected}"
        );
    }

    #[test]
    fn a_station_stiffens_as_the_rod_is_chopped_finer() {
        // `n·EI/L`: the rod is the same rod, so the station standing in for
        // half as much of it is twice as stiff.
        let coarse = Rod { links: 10, ..steel() }.link_properties();
        let fine = Rod { links: 20, ..steel() }.link_properties();
        assert!((fine.bend_stiffness / coarse.bend_stiffness - 2.0).abs() < 1e-4);
        assert!((fine.mass / coarse.mass - 0.5).abs() < 1e-4);
        assert!((fine.length / coarse.length - 0.5).abs() < 1e-4);
    }

    #[test]
    fn torsion_uses_the_shear_modulus_and_twice_the_moment() {
        let rod = steel();
        let p = rod.link_properties();
        let ratio = p.torsion_stiffness / p.bend_stiffness;
        // G·2I / E·I = 2G/E = 1/(1+ν)
        assert!((ratio - 1.0 / (1.0 + 0.3)).abs() < 1e-5, "got {ratio}");
    }

    #[test]
    fn mass_comes_from_the_sheath_and_stiffness_from_the_spine() {
        let bare = Rod::new(0.3, 30).radius(0.006).material(1.0e6, 0.45, 1200.0);
        let sheathed = bare.clone().core(0.0005);
        assert_eq!(bare.mass(), sheathed.mass(), "same rod, same mass");
        let ratio = bare.link_properties().bend_stiffness
            / sheathed.link_properties().bend_stiffness;
        assert!(
            (ratio - (0.006f32 / 0.0005).powi(4)).abs() < 1.0,
            "the fourth power of the radius ratio, got {ratio}"
        );
    }

    #[test]
    fn a_tube_is_softer_and_lighter_than_the_bar_it_came_from() {
        let bar = Rod::new(0.3, 30).radius(0.01);
        let tube = Rod { bore: 0.008, ..bar.clone() };
        assert!(tube.mass() < bar.mass());
        assert!(tube.link_properties().bend_stiffness < bar.link_properties().bend_stiffness);
        // ...but far softer than it is lighter, which is the point of a tube.
        let mass_ratio = tube.mass() / bar.mass();
        let stiffness_ratio =
            tube.link_properties().bend_stiffness / bar.link_properties().bend_stiffness;
        assert!(
            stiffness_ratio > mass_ratio,
            "a tube keeps more stiffness than mass: {stiffness_ratio} vs {mass_ratio}"
        );
    }

    #[test]
    fn segments_divide_the_stations_with_the_remainder_at_the_root() {
        let rod = Rod::new(1.0, 10).segments(3);
        assert_eq!(rod.segment_stations(1), 0..4);
        assert_eq!(rod.segment_stations(2), 4..7);
        assert_eq!(rod.segment_stations(3), 7..10);
        assert_eq!(rod.segment_stations(9), 0..0);
        assert!((rod.segment_end(1) - 0.4).abs() < 1e-6);
        assert!((rod.segment_end(3) - 1.0).abs() < 1e-6);
    }
}
