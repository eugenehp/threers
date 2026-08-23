//! Building a rod straight out of a `.scad` model's own declarations.
//!
//! The model says what the rod *is*; this is the beam theory it did not want to
//! know about. Requires the `openscad` feature.
//!
//! ```no_run
//! use threers_continuum::prelude::*;
//! use threers_physics::prelude::*;
//! use threers::parse_scad_mechanism_file;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let spec = parse_scad_mechanism_file("finger.scad")?;
//! let mut world = World::new();
//! let rods = build_scad_continua(&mut world, &spec, &Default::default());
//! # Ok(())
//! # }
//! ```

use crate::build::{Continuum, TendonRoute};
use crate::rod::Rod;
use std::collections::HashMap;
use threers::openscad::mechanism::{ContinuumSpec, TendonSpec};
use threers::openscad::MechanismSpec;
use threers_physics::prelude::*;

/// A rod built from a `continuum()` declaration, and the cables on it.
#[derive(Debug, Clone)]
pub struct ScadContinuum {
    pub name: String,
    pub continuum: Continuum,
    /// Cable names, parallel to `continuum.tendons`.
    pub tendon_names: Vec<String>,
}

impl ScadContinuum {
    /// Index of a cable by the name the model gave it.
    pub fn tendon(&self, name: &str) -> Option<usize> {
        self.tendon_names.iter().position(|n| n == name)
    }
}

/// Turn one `continuum()` declaration into a [`Rod`].
///
/// Degrees become radians here — the model's boundary — and nothing else
/// changes: the model's own length units are carried through, so a rod declared
/// in millimetres stays in millimetres and its Young's modulus had better be in
/// the same units.
pub fn rod_from_spec(spec: &ContinuumSpec) -> Rod {
    Rod {
        length: spec.length,
        links: spec.links.max(1),
        radius: spec.radius,
        core_radius: spec.structural_radius(),
        bore: spec.bore,
        youngs: spec.youngs,
        poisson: spec.poisson,
        density: spec.density,
        damping_ratio: spec.damping_ratio,
        twist: spec.twist,
        station_limit: spec
            .range
            .map(|[a, b]| [a.to_radians(), b.to_radians()]),
        segments: spec.segments.max(1),
        self_collide: false,
    }
}

/// Turn one `tendon()` declaration into a route.
pub fn route_from_spec(spec: &TendonSpec) -> TendonRoute {
    TendonRoute {
        offset: spec.offset,
        phase: spec.phase.to_radians(),
        segment: spec.segment,
        pretension: spec.pretension,
    }
}

/// Build every rod a model declared, with its cables.
///
/// `bases` maps a part name to the body already in the world that it was built
/// as — a rod declared `on = "base"` is welded to `bases["base"]`. A rod naming
/// a part that is not there is mounted in the world instead, which is what an
/// unmounted rod wants and is the more useful failure for a typo than not
/// building at all.
pub fn build_scad_continua(
    world: &mut World,
    spec: &MechanismSpec,
    bases: &HashMap<String, BodyId>,
) -> Vec<ScadContinuum> {
    let mut out = Vec::with_capacity(spec.continua.len());
    for declaration in &spec.continua {
        let rod = rod_from_spec(declaration);
        let origin = Vector3::new(
            declaration.at[0],
            declaration.at[1],
            declaration.at[2],
        );
        let axis = Vector3::new(
            declaration.axis[0],
            declaration.axis[1],
            declaration.axis[2],
        );
        let base = bases.get(&declaration.base).copied();
        let mut continuum = Continuum::build(world, rod, base, origin, axis);

        let mut tendon_names = Vec::new();
        for cable in spec
            .tendons
            .iter()
            .filter(|t| t.along == declaration.name)
        {
            continuum.add_tendon(world, route_from_spec(cable));
            tendon_names.push(cable.name.clone());
            // A declared pull is applied straight away: the model is describing
            // a pose, not just a robot.
            if cable.pull != 0.0 || cable.stiffness.is_some() {
                let index = continuum.tendons.len() - 1;
                let rest = continuum.rest_lengths[index];
                let id = continuum.tendons[index];
                if let Some(tendon) = world.tendon_mut(id) {
                    tendon.kind = match cable.stiffness {
                        Some(k) => TendonKind::Spring {
                            rest_length: (rest - cable.pull).max(0.0),
                            stiffness: k,
                            damping: cable.damping,
                        },
                        None => TendonKind::winch(
                            (rest - cable.pull).max(0.0),
                            cable.max_force.unwrap_or(f32::MAX),
                        ),
                    };
                }
            }
        }

        out.push(ScadContinuum {
            name: declaration.name.clone(),
            continuum,
            tendon_names,
        });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use threers::parse_scad_mechanism;

    const MODEL: &str = r#"
        part("base", fixed = true) cylinder(h = 0.01, r = 0.012);
        continuum("backbone", on = "base", at = [0, 0, 0], axis = [0, 1, 0],
                  length = 0.3, links = 12, radius = 0.006, segments = 2,
                  youngs = 200e9, poisson = 0.3, density = 1200,
                  backbone_radius = 0.0005, damping = 0.05, range = [-20, 20]);
        tendon("t0", along = "backbone", offset = 0.004, phase = 0,   pretension = 0.5);
        tendon("t1", along = "backbone", offset = 0.004, phase = 120, pretension = 0.5);
        tendon("t2", along = "backbone", offset = 0.004, phase = 240, pretension = 0.5);
    "#;

    #[test]
    fn a_declaration_becomes_a_rod_with_the_right_beam_numbers() {
        let spec = parse_scad_mechanism(MODEL).unwrap();
        let rod = rod_from_spec(spec.continuum("backbone").unwrap());
        assert_eq!(rod.links, 12);
        assert_eq!(rod.core_radius, 0.0005);
        assert_eq!(rod.radius, 0.006);
        // Degrees in the model, radians in the rod.
        let limit = rod.station_limit.unwrap();
        assert!((limit[1] - 20f32.to_radians()).abs() < 1e-6);

        let expected = 12.0 * 200.0e9 * rod.second_moment() / 0.3;
        assert!((rod.link_properties().bend_stiffness - expected).abs() < 1e-3 * expected);
    }

    #[test]
    fn the_model_builds_into_a_world_with_its_cables_attached() {
        let spec = parse_scad_mechanism(MODEL).unwrap();
        let mut world = World::new();
        let base = world.add_body(RigidBody::fixed().shape(Shape::cylinder(0.005, 0.012)));
        let mut bases = HashMap::new();
        bases.insert("base".to_string(), base);

        let rods = build_scad_continua(&mut world, &spec, &bases);
        assert_eq!(rods.len(), 1);
        let built = &rods[0];
        assert_eq!(built.name, "backbone");
        assert_eq!(built.continuum.bodies.len(), 13);
        assert_eq!(built.continuum.stations.len(), 12);
        assert_eq!(built.continuum.tendons.len(), 3);
        assert_eq!(built.tendon("t1"), Some(1));
        assert!(built.continuum.mount.is_some(), "it was told to sit on base");

        // The rod grows along +Y as declared, so the tip is 0.3 up.
        let tip = built.continuum.tip(&world);
        assert!((tip.y - 0.3).abs() < 1e-3, "tip at {tip:?}");
    }

    #[test]
    fn a_rod_on_a_part_that_is_not_there_still_builds_in_the_world() {
        let spec = parse_scad_mechanism(MODEL).unwrap();
        let mut world = World::new();
        let rods = build_scad_continua(&mut world, &spec, &HashMap::new());
        assert_eq!(rods.len(), 1);
        assert!(rods[0].continuum.mount.is_none());
        assert!(world.body(rods[0].continuum.bodies[0]).unwrap().is_fixed());
    }
}
