//! Mechanism declarations read out of a `.scad` model.
//!
//! A model can say what its parts *are* and how they are joined, alongside the
//! geometry that draws them:
//!
//! ```text
//! part("box", fixed = true)  box_body();
//! part("lid")                lid_body();
//!
//! hinge("lid_pivot", parts = ["lid", "box"],
//!       at = [0, 40, 8], axis = [1, 0, 0], range = [0, 105]);
//!
//! drive("lid_pivot", to = 105, over = 1.5);
//! ```
//!
//! A drive either goes somewhere or just goes: `to = …` moves to a position,
//! and `speed = …` turns at that rate for as long as the simulation runs. The
//! second is the only one that works for a shaft making full turns — a hinge's
//! angle wraps at ±360°, so a position past that is not expressible.
//!
//! What comes back is plain data — names, points, axes and angles. There is no
//! physics in this module and no dependency on any; it is the *declaration*, and
//! `threers-physics` is what turns one into a simulated mechanism. That split is
//! deliberate: the model should be readable, diffable and testable without a
//! solver anywhere near it.
//!
//! Read one with [`parse_scad_mechanism`](crate::openscad::parse_scad_mechanism)
//! and its `_at` / `_file` siblings.
//!
//! # Angles are degrees
//!
//! Everything angular here is in degrees, because that is what the rest of
//! OpenSCAD uses — `rotate([0, 0, 45])` is 45 degrees, and a hinge range beside
//! it that meant radians would be a trap. The conversion happens at the physics
//! boundary, once.
//!
//! # Rods, and the cables that bend them
//!
//! A part is rigid, which most things are and a flexible rod is not. A rod is
//! declared as itself and expanded into a chain of stiff hinges by whatever
//! builds the mechanism:
//!
//! ```text
//! continuum("backbone", on = "base", at = [0, 0, 10], axis = [0, 0, 1],
//!           length = 300, links = 30, radius = 6, segments = 1,
//!           youngs = 200e9, density = 1200, backbone_radius = 0.5);
//!
//! tendon("t0", along = "backbone", offset = 4, phase = 0,   pretension = 1);
//! tendon("t1", along = "backbone", offset = 4, phase = 120, pretension = 1);
//! tendon("t2", along = "backbone", offset = 4, phase = 240, pretension = 1);
//! ```
//!
//! See [`ContinuumSpec`] and [`TendonSpec`]. Both are declarations like the
//! rest: no beam theory happens here, only the numbers it needs.
//!
//! # Keeping the file openable in OpenSCAD
//!
//! These modules are threers' own. OpenSCAD does not have them, and it drops an
//! unknown module *along with its children* — so a bare `part("lid") { … }`
//! would render as nothing there. Declare the shims at the top of the file and
//! it opens in both:
//!
//! ```text
//! module part(name, fixed = false, density = undef) { children(); }
//! module hinge(name, parts, at, axis, range) { }
//! module drive(name, to, over, at, torque, speed) { }
//! module continuum(name, on, at, axis, length, links, radius) { }
//! module tendon(name, along, offset, phase) { }
//! ```
//!
//! threers overrides them with the real thing, so the shims cost nothing here
//! and make the file behave in OpenSCAD.

use super::Solid;

/// Everything a `.scad` model said about its own mechanism.
#[derive(Debug, Clone)]
pub struct MechanismSpec {
    /// The whole model, exactly as [`parse_scad`](crate::openscad::scad::parse_scad)
    /// would return it — parts, scenery and all.
    pub model: Solid,
    /// The bodies, in declaration order.
    pub parts: Vec<PartSpec>,
    /// The joints between them.
    pub mates: Vec<MateSpec>,
    /// The motion the model asked for.
    pub drives: Vec<DriveSpec>,
    /// The flexible rods, which stand for many parts and many joints each.
    pub continua: Vec<ContinuumSpec>,
    /// The cables routed along them.
    pub tendons: Vec<TendonSpec>,
}

impl MechanismSpec {
    /// Whether the model declared a mechanism at all.
    pub fn is_empty(&self) -> bool {
        self.parts.is_empty() && self.mates.is_empty() && self.continua.is_empty()
    }

    pub fn part(&self, name: &str) -> Option<&PartSpec> {
        self.parts.iter().find(|p| p.name == name)
    }

    pub fn continuum(&self, name: &str) -> Option<&ContinuumSpec> {
        self.continua.iter().find(|c| c.name == name)
    }

    /// Names a tendon runs along that no `continuum()` declared.
    pub fn dangling_tendons(&self) -> Vec<&str> {
        let mut out = Vec::new();
        for tendon in &self.tendons {
            if self.continuum(&tendon.along).is_none() && !out.contains(&tendon.along.as_str()) {
                out.push(tendon.along.as_str());
            }
        }
        out
    }

    pub fn mate(&self, name: &str) -> Option<&MateSpec> {
        self.mates.iter().find(|m| m.name == name)
    }

    /// Names a mate refers to that no `part()` declared.
    ///
    /// The most common way to get a mechanism wrong is a typo in a part name,
    /// which otherwise shows up much later as a mate that does nothing.
    pub fn dangling_parts(&self) -> Vec<&str> {
        let mut out = Vec::new();
        for mate in &self.mates {
            for name in mate
                .parts
                .iter()
                .chain(mate.carrier.iter())
            {
                if self.part(name).is_none() && !out.contains(&name.as_str()) {
                    out.push(name.as_str());
                }
            }
        }
        out
    }

    /// Names a drive refers to that no mate declared.
    pub fn dangling_drives(&self) -> Vec<&str> {
        let mut out = Vec::new();
        for drive in &self.drives {
            if self.mate(&drive.mate).is_none() && !out.contains(&drive.mate.as_str()) {
                out.push(drive.mate.as_str());
            }
        }
        out
    }
}

/// One rigid body, and the geometry that draws it.
#[derive(Debug, Clone)]
pub struct PartSpec {
    pub name: String,
    /// The subtree `part()` wrapped, unevaluated. Booleans inside it have not
    /// been run yet, so collecting a mechanism costs a parse and not a CSG.
    pub solid: Solid,
    /// `fixed = true` — the part grounds the mechanism and never moves.
    pub fixed: bool,
    /// `density = …`, in mass per cubic model unit.
    pub density: Option<f32>,
    /// `mass = …` — the total, overriding whatever the density would give.
    pub mass: Option<f32>,
    /// `collider = "…"` — what the part touches things with. `None` leaves the
    /// default: every triangle for a fixed part, a convex hull for a moving one.
    pub fit: Option<PartFit>,
    /// `friction = …`, unitless. `None` leaves the material default.
    pub friction: Option<f32>,
    /// `bounce = …` (or `restitution`), 0 to 1.
    pub restitution: Option<f32>,
    /// `damping = [linear, angular]` — resistance proportional to *speed*,
    /// which is a different thing from friction and cannot do friction's job:
    /// damping slows a fall and cannot stop one. Air, oil, a dashpot.
    pub damping: Option<[f32; 2]>,
}

/// How a part's drawn mesh becomes the thing it collides with.
///
/// A mate says how two parts are *joined*; this says what happens where they
/// merely *touch*. It matters as soon as a mechanism has a part whose job is to
/// run into another one — a latch, a pawl, a stop, a cam — because a convex
/// hull quietly fills in the hook that does the catching.
///
/// It is also what [`interference checks`](../../../threers_physics/assembly/struct.Assembly.html#method.check)
/// test against, so it is two decisions at once: a hull on a part with a pocket
/// reports the pocket as solid, and anything sitting in that pocket reads as
/// fouling when it does not.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PartFit {
    /// `"hull"` — convex hull of the vertices. The default for a moving part:
    /// exact for a convex shape, and safe for anything the solver has to push.
    #[default]
    Hull,
    /// `"mesh"` — every triangle. Exact, free for a part that never moves, and
    /// the wrong answer for one that does: a triangle mesh is a hollow surface,
    /// so a moving body can be pushed out through the wrong face.
    Mesh,
    /// `"decompose"` — split into convex pieces, so a concave part keeps its
    /// shape *and* can still move. What a hook, a fork or a pocket needs. Costs
    /// a decomposition per part, once.
    Decompose,
    /// `"box"` — the axis-aligned box around it.
    Box,
    /// `"ball"` — the bounding sphere. Cheapest of all, and rolls.
    Ball,
    /// `"capsule"` — a capsule down the longest axis.
    Capsule,
    /// `"cylinder"` — a cylinder down the longest axis. Wheels, rollers, pins.
    Cylinder,
}

impl PartFit {
    /// The name a `.scad` model writes, e.g. `collider = "decompose"`.
    pub fn from_name(name: &str) -> Option<PartFit> {
        Some(match name.trim().to_ascii_lowercase().as_str() {
            "hull" | "convex" | "convex_hull" | "convexhull" => PartFit::Hull,
            "mesh" | "trimesh" | "exact" => PartFit::Mesh,
            "decompose" | "decomposed" | "concave" => PartFit::Decompose,
            "box" | "aabb" | "cuboid" => PartFit::Box,
            "ball" | "sphere" => PartFit::Ball,
            "capsule" => PartFit::Capsule,
            "cylinder" => PartFit::Cylinder,
            _ => return None,
        })
    }

    /// Every name a model may write, for the error message when it writes
    /// something else.
    pub const NAMES: &'static [&'static str] = &[
        "hull",
        "mesh",
        "decompose",
        "box",
        "ball",
        "capsule",
        "cylinder",
    ];
}

/// What a mate holds. Angular values are **degrees**.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MateSpecKind {
    /// `hinge()` — one rotation about the axis.
    Hinge,
    /// `slider()` — one translation along it.
    Slider,
    /// `cylindrical()` — both.
    Cylindrical,
    /// `ball()` — three rotations about the point.
    Ball,
    /// `weld()` — nothing; two parts that move as one.
    Weld,
    /// `planar()` — sliding and spinning in the plane the axis is normal to.
    Planar,
    /// `screw(pitch = …)` — travel per full turn, in model units.
    Screw { pitch: f32 },
    /// `gear(ratio = …)` — turns of the first part per turn of the second.
    /// Negative for meshing external gears, which counter-rotate.
    Gear { ratio: f32 },
    /// `rack(radius = …)` — pitch radius, in model units.
    Rack { radius: f32 },
}

impl MateSpecKind {
    /// Whether this mate's coordinate is an angle. Ranges and drive targets on
    /// an angular mate are in degrees; on the others they are model units.
    pub fn is_angular(self) -> bool {
        matches!(self, MateSpecKind::Hinge)
    }

    /// Whether a continuous `speed = …` on this mate turns something.
    ///
    /// Not the same question as [`Self::is_angular`], and a screw is why. A
    /// screw's *coordinate* is its travel — its range and its drive targets are
    /// millimetres — but the thing a motor on one actually drives is the shaft,
    /// so its speed is degrees a second like any other rotation. Answering with
    /// `is_angular` gives a screw a speed in radians a second, which is a unit
    /// no part of the model uses and which reads as a mechanism running five
    /// times too slow.
    pub fn motor_is_angular(self) -> bool {
        !matches!(self, MateSpecKind::Slider | MateSpecKind::Planar)
    }
}

/// One declared joint.
#[derive(Debug, Clone)]
pub struct MateSpec {
    pub name: String,
    pub kind: MateSpecKind,
    /// `parts = [moving, base]` — the part that moves, then what it moves
    /// against. Everything the mate reports is in that sense.
    pub parts: [String; 2],
    /// `at = [x, y, z]` — where the joint sits, in **model** coordinates.
    ///
    /// The same point in both parts, which is the point: a model draws the pin
    /// and the bore at the same place, so there is one coordinate to write and
    /// nothing to keep in step.
    pub at: [f32; 3],
    /// `axis = [x, y, z]` — the axle, the slide direction, or the plane normal.
    pub axis: [f32; 3],
    /// `range = [min, max]` — travel limits. Degrees for a hinge, model units
    /// for a slider or screw.
    pub range: Option<[f32; 2]>,
    /// `axis_b = [x, y, z]` — the *second*-named part's own axis, for the mates
    /// where the two parts do not share one.
    ///
    /// Spelled `rack_axis` on a `rack()`, where it is the rack's travel
    /// direction against the pinion's spin; either name sets it. A `gear()`
    /// needs it whenever the two shafts are not parallel — a worm and its
    /// wheel, or a bevel pair — and without it the only gear trains that can be
    /// declared are the ones on parallel shafts.
    pub axis_b: Option<[f32; 3]>,
    /// `bearing = [mu, radius]` — friction in the joint itself, scaling with
    /// what it carries: the torque opposing motion is `mu · radius · reaction`.
    ///
    /// This is the difference between a bearing and an ideal joint, and it is
    /// not a small one. A real journal is not frictionless and a real screw is
    /// not reversible; without it a worm drive back-drives, a gate does not stay
    /// where it was left, and anything that coasts, coasts forever.
    pub bearing: Option<[f32; 2]>,
    /// `friction = <torque>` — friction that is there under no load at all: a
    /// preloaded race, a gland, a seal, a gib done up tight. Torque units for a
    /// hinge, force units for a slider.
    pub friction: Option<f32>,
    /// `spring = [frequency, damping]` — give in the joint's own constraint,
    /// as an oscillation rate and a fraction of critical damping rather than a
    /// stiffness, so the number means the same thing whatever it is holding.
    pub spring: Option<[f32; 2]>,
    /// `elastic = [stiffness, rest, damping]` — a spring on the mate's own
    /// coordinate: a return spring, a torsion bar, a rubber bush.
    ///
    /// Not the same thing as `spring` beside it, and the difference is worth
    /// the two words it takes: `spring` is how hard the joint holds the
    /// constraint it has, and this is where the freedom it *leaves* would
    /// rather be. A hinge can have both.
    ///
    /// `stiffness` is torque per degree for a hinge and force per model unit
    /// for a slider; `rest` is in the mate's own coordinate; `damping` opposes
    /// speed in it.
    pub elastic: Option<[f32; 3]>,
    /// `carrier = "part"` — what a `gear()` is mounted on, when that is not the
    /// world.
    ///
    /// A mesh relates the two wheels' rates *to the case they run in*. Bolt the
    /// case to the airframe and the case is the world, so it need not be said.
    /// Bolt it to something that itself turns — an epicyclic train, a slew drive
    /// carried on the segment upstream of it — and it very much does: the ratio
    /// is between `(a − carrier)` and `(b − carrier)`, and leaving it out states
    /// a different mechanism rather than an approximate one.
    ///
    /// Ignored on the mates that are not couplings.
    pub carrier: Option<String>,
    /// `collide = true` — the two parts also *touch* each other, on top of
    /// being joined.
    ///
    /// Off by default, and for a good reason: a hinge lives inside both the
    /// door and the frame, so leaving that contact on makes the contact and the
    /// constraint fight each other over the same overlap.
    ///
    /// It is worth knowing when to turn it on, because CAD draws things this
    /// way constantly: a carriage on a rail is usually mated to the *same part*
    /// that carries its end stops, and with the pair excluded it slides
    /// straight through them — silently, and in the sweep and the interference
    /// check as well as in the simulation. Either put the stops on a separate
    /// part or say `collide = true` here.
    pub collide: bool,
}

/// A flexible rod, declared as one thing and simulated as many.
///
/// ```text
/// continuum("backbone", on = "base", at = [0, 0, 10], axis = [0, 0, 1],
///           length = 300, links = 30, radius = 6,
///           youngs = 200e9, density = 1200, backbone_radius = 0.5);
/// ```
///
/// A continuum backbone has no joints in it — that is what makes it continuous
/// — so simulating one means choosing not to. This says what the rod *is*, in
/// the terms a data sheet uses, and leaves the choice of how finely to chop it
/// to `links`.
///
/// # What a builder does with it
///
/// The rod becomes `links + 1` rigid bodies with half-length ones at each end,
/// joined by hinge pairs (or triples, with [`twist`](Self::twist)) whose
/// stiffness comes from beam theory:
///
/// ```text
/// K_bend    = n·E·Iₓ / L      Iₓ = π/4 (rₒ⁴ − rᵢ⁴)
/// K_twist   = n·G·I_z / L     I_z = 2Iₓ,  G = E / 2(1+ν)
/// ```
///
/// which is exactly the stiffness a a joint spring wants, and the reason it
/// takes a stiffness rather than a frequency.
///
/// # Two radii, and they are usually not the same
///
/// [`radius`](Self::radius) is what the rod *looks like* and collides as;
/// [`backbone_radius`](Self::backbone_radius) is the load-bearing core inside
/// it. A 6 mm silicone finger over a 0.5 mm spring-steel spine has both, and
/// using the visible one for stiffness overstates it by `(6/0.5)⁴` — a factor
/// of twenty thousand. When only one is given the visible one is used, which is
/// right for a bare rod and wrong the moment there is a sheath.
///
/// Angles are degrees, like everything else here.
#[derive(Debug, Clone, PartialEq)]
pub struct ContinuumSpec {
    pub name: String,
    /// `on = "part"` — what the rod grows out of. Empty mounts it in the world.
    pub base: String,
    /// `at = [x, y, z]` — where the rod's root sits, in model coordinates.
    pub at: [f32; 3],
    /// `axis = [x, y, z]` — which way it grows when undeflected.
    pub axis: [f32; 3],
    /// `length = …` — arc length of the whole rod, in model units.
    pub length: f32,
    /// `links = …` — how many rigid segments stand in for it. More is smoother
    /// and slower; the error falls off roughly as `1/n²`.
    pub links: usize,
    /// `radius = …` — what it is drawn and collided as.
    pub radius: f32,
    /// `backbone_radius = …` — the load-bearing core. Falls back to `radius`.
    pub backbone_radius: Option<f32>,
    /// `bore = …` — inner radius, for a tube.
    pub bore: f32,
    /// `youngs = …` — Young's modulus, in force per square model unit.
    pub youngs: f32,
    /// `poisson = …` — Poisson's ratio, which turns `youngs` into a shear
    /// modulus. Most materials sit between 0.2 and 0.5.
    pub poisson: f32,
    /// `density = …` — mass per cubic model unit.
    pub density: f32,
    /// `damping = …` — as a fraction of the computed stiffness. A rod with none
    /// rings forever.
    pub damping_ratio: f32,
    /// `twist = true` — give each station a third hinge about the rod's own
    /// axis. Off by default: torsion costs a constraint row per link and a
    /// planar bend never uses it.
    pub twist: bool,
    /// `range = [min, max]` — how far one station may bend, in degrees.
    ///
    /// Per *station*, not for the rod: thirty stations of ±5° is a rod that
    /// curls up on itself. Leave it off unless the real thing has a hard stop.
    pub range: Option<[f32; 2]>,
    /// `segments = …` — how many independently actuated sections the rod is
    /// divided into, root to tip. Tendons name one; the default is a single
    /// segment spanning the whole rod.
    pub segments: usize,
}

impl ContinuumSpec {
    /// The core radius the stiffness is computed from.
    pub fn structural_radius(&self) -> f32 {
        self.backbone_radius.unwrap_or(self.radius)
    }

    /// Which links belong to `segment` (1-based), as a half-open range.
    ///
    /// Segments divide the rod evenly, with any remainder going to the ones
    /// nearest the root — a 10-link rod in 3 segments is 4/3/3, not 3/3/4,
    /// because the root end is the one carrying every tendon above it.
    pub fn segment_links(&self, segment: usize) -> std::ops::Range<usize> {
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
}

/// A cable routed along a [`ContinuumSpec`], and what pulls it.
///
/// ```text
/// tendon("t0", along = "backbone", offset = 4, phase = 0,   pretension = 1);
/// tendon("t1", along = "backbone", offset = 4, phase = 120, pretension = 1);
/// tendon("t2", along = "backbone", offset = 4, phase = 240, pretension = 1);
/// ```
///
/// Three of those at 120° apart is a segment that can be bent in any direction
/// — which is the whole mechanism of a tendon-driven continuum robot, and the
/// reason the routing is declared by an offset and an angle rather than by
/// listing every guide: there are `links + 1` of them and they are all the same
/// two numbers.
#[derive(Debug, Clone, PartialEq)]
pub struct TendonSpec {
    pub name: String,
    /// `along = "…"` — the continuum it is routed down.
    pub along: String,
    /// `offset = …` — how far the cable runs from the backbone, in model units.
    /// Should be under the rod's radius, or it is outside the rod.
    pub offset: f32,
    /// `phase = …` — where around the rod, in degrees from the model's first
    /// cross-axis, counter-clockwise looking down the rod.
    pub phase: f32,
    /// `segment = …` — which segment it terminates at, 1-based from the root.
    /// It is anchored at that segment's tip and runs through everything below,
    /// which is what makes a multi-segment robot's lower tendons pass through
    /// its upper ones. Zero or absent runs the whole rod.
    pub segment: usize,
    /// `pretension = …` — constant pull, in the model's force units. What the
    /// robot is assembled with, before anything drives it.
    pub pretension: f32,
    /// `stiffness = …` — force per unit of extension, making the cable a length
    /// actuator with this gain rather than an inextensible one.
    pub stiffness: Option<f32>,
    /// `damping = …` — force per unit of closing speed, with `stiffness`.
    pub damping: f32,
    /// `pull = …` — how far to reel it in from its built length, in model
    /// units. Positive shortens. This is the *command*: one number per tendon
    /// is what a real robot's controller sends.
    pub pull: f32,
    /// `force = …` — ceiling on what the drive may apply. `None` leaves it to
    /// the builder.
    pub max_force: Option<f32>,
}

/// One authored move — see the module docs.
///
/// Nothing here is a pose. `to` is where the mate should end up and `over` is
/// how long it may take, both of which the solver treats as a *request*: the
/// mechanism arrives when it arrives, or stalls against whatever is in the way.
///
/// # Going somewhere, or just going
///
/// `drive("lid", to = 105, over = 1.5)` moves to a position. `drive("axle",
/// speed = 120)` turns at 120°/s and never stops — which is the right tool for
/// anything that makes full turns, and the only one that works for them: a
/// hinge's angle is derived from a quaternion, so it wraps at ±360° and a
/// position target beyond that is not expressible. A motor has no such limit
/// because it never asks where the joint is.
#[derive(Debug, Clone, PartialEq)]
pub struct DriveSpec {
    /// The mate this drives, by name.
    pub mate: String,
    /// Target, in the mate's own coordinate — degrees for a hinge.
    ///
    /// `None` when the model gave a `speed` instead: that is a motor, turning
    /// for as long as the simulation runs.
    pub to: Option<f32>,
    /// `over = …` seconds. `None` moves as fast as the drive can.
    pub over: Option<f32>,
    /// `at = …` seconds on the timeline. Zero starts immediately, and several
    /// drives with different starts are a choreography.
    pub start: f32,
    /// `torque = …`, in the model's own force units. `None` lets the assembly
    /// size one from the mass it has to move.
    pub torque: Option<f32>,
    /// `speed = …` — a slew ceiling on a move to a target, or *the* speed when
    /// there is no target.
    pub max_speed: Option<f32>,
}

impl DriveSpec {
    /// Whether this turns forever rather than going somewhere.
    pub fn is_continuous(&self) -> bool {
        self.to.is_none()
    }
}

#[cfg(test)]
mod tests {
    use crate::openscad::geometry_bounds;
    use crate::{parse_scad, parse_scad_mechanism};

    #[test]
    fn a_model_declares_its_own_parts_and_joints() {
        let spec = parse_scad_mechanism(
            r#"
            part("box", fixed = true)  cube([40, 20, 8]);
            part("lid", density = 1.2) translate([0, 0, 8]) cube([40, 20, 2]);

            hinge("lid_pivot", parts = ["lid", "box"],
                  at = [0, -10, 8], axis = [1, 0, 0], range = [0, 105]);

            drive("lid_pivot", to = 105, over = 1.5, torque = 4.5);
            "#,
        )
        .unwrap();

        assert_eq!(spec.parts.len(), 2);
        assert!(spec.part("box").unwrap().fixed);
        assert!(!spec.part("lid").unwrap().fixed);
        assert_eq!(spec.part("lid").unwrap().density, Some(1.2));

        let mate = spec.mate("lid_pivot").unwrap();
        assert_eq!(mate.parts, ["lid".to_string(), "box".to_string()]);
        assert_eq!(mate.at, [0.0, -10.0, 8.0]);
        // Degrees, as the rest of OpenSCAD is.
        assert_eq!(mate.range, Some([0.0, 105.0]));

        let drive = &spec.drives[0];
        assert_eq!(drive.mate, "lid_pivot");
        assert_eq!(
            (drive.to, drive.over, drive.torque),
            (Some(105.0), Some(1.5), Some(4.5))
        );
        assert_eq!(drive.start, 0.0);
    }

    #[test]
    fn wrapping_a_subtree_in_part_changes_no_geometry() {
        // `part()` names a body. It must not also *move* one, or a model would
        // render differently for having been described.
        let plain = "cube([40, 20, 8]); translate([0, 0, 8]) cube([40, 20, 2]);";
        let named = r#"
            part("box") cube([40, 20, 8]);
            part("lid") translate([0, 0, 8]) cube([40, 20, 2]);
        "#;
        let a = geometry_bounds(&parse_scad(plain).unwrap().to_geometry());
        let b = geometry_bounds(&parse_scad_mechanism(named).unwrap().model.to_geometry());
        for i in 0..3 {
            assert!((a.0[i] - b.0[i]).abs() < 1e-4, "min {i}: {a:?} vs {b:?}");
            assert!((a.1[i] - b.1[i]).abs() < 1e-4, "max {i}: {a:?} vs {b:?}");
        }
    }

    #[test]
    fn the_mechanism_modules_are_inert_in_an_ordinary_parse() {
        // A model with its own `module part(...)` must keep its own meaning
        // everywhere except where a mechanism was actually asked for —
        // builtins beat user modules here, so the gate is what makes this safe.
        let src = r#"
            module part(name) { translate([100, 0, 0]) children(); }
            part("shifted") cube([10, 10, 10]);
        "#;
        let bounds = geometry_bounds(&parse_scad(src).unwrap().to_geometry());
        assert!(
            bounds.0[0] > 90.0,
            "the user's own part() was overridden: {bounds:?}"
        );

        // And under a mechanism parse the builtin wins, so the cube stays put
        // and is recorded as a body.
        let spec = parse_scad_mechanism(src).unwrap();
        let bounds = geometry_bounds(&spec.model.to_geometry());
        assert!(
            bounds.0[0] < 1.0,
            "the builtin did not take over: {bounds:?}"
        );
        assert_eq!(spec.parts.len(), 1);
        assert_eq!(spec.parts[0].name, "shifted");
    }

    #[test]
    fn every_mate_kind_parses() {
        let spec = parse_scad_mechanism(
            r#"
            part("a") cube(10);
            part("b") translate([20, 0, 0]) cube(10);
            hinge("h",       parts = ["a", "b"], at = [0,0,0], axis = [0,0,1]);
            slider("s",      parts = ["a", "b"], at = [0,0,0], axis = [1,0,0], range = [0, 25]);
            cylindrical("c", parts = ["a", "b"], at = [0,0,0], axis = [0,1,0]);
            ball("k",        parts = ["a", "b"], at = [0,0,0]);
            weld("w",        parts = ["a", "b"], at = [0,0,0], axis = [0,0,1]);
            planar("p",      parts = ["a", "b"], at = [0,0,0], axis = [0,0,1]);
            screw("t",       parts = ["a", "b"], at = [0,0,0], axis = [0,0,1], pitch = 1.25);
            gear("g",        parts = ["a", "b"], at = [0,0,0], axis = [0,1,0], ratio = -3);
            rack("r",        parts = ["a", "b"], at = [0,0,0], axis = [0,1,0], radius = 6,
                             rack_axis = [1,0,0]);
            "#,
        )
        .unwrap();
        assert_eq!(spec.mates.len(), 9);
        assert!(spec.dangling_parts().is_empty());
        assert_eq!(
            spec.mate("t").unwrap().kind,
            super::MateSpecKind::Screw { pitch: 1.25 }
        );
        assert_eq!(spec.mate("r").unwrap().axis_b, Some([1.0, 0.0, 0.0]));
        // A ball socket is the one mate with no axis to give.
        assert!(spec.mate("k").is_some());
    }

    #[test]
    fn a_drive_either_goes_somewhere_or_just_goes() {
        let spec = parse_scad_mechanism(
            r#"
            part("a") cube(10);
            part("b") cube(10);
            hinge("axle", parts = ["a", "b"], at = [0,0,0], axis = [0,0,1]);
            drive("axle", speed = 120, torque = 5);
            "#,
        )
        .unwrap();
        let drive = &spec.drives[0];
        assert!(drive.is_continuous(), "a speed with no target is a motor");
        assert_eq!(drive.to, None);
        assert_eq!(drive.max_speed, Some(120.0));

        // And one that says neither is a mistake, not a drive that does
        // nothing.
        let err = parse_scad_mechanism(
            r#"
            part("a") cube(10);
            part("b") cube(10);
            hinge("axle", parts = ["a", "b"], at = [0,0,0], axis = [0,0,1]);
            drive("axle", over = 1.0);
            "#,
        )
        .unwrap_err();
        assert!(err.contains("to ="), "unhelpful error: {err}");
        assert!(err.contains("speed ="), "unhelpful error: {err}");
    }

    #[test]
    fn a_typo_in_a_part_name_is_reported_rather_than_ignored() {
        let spec = parse_scad_mechanism(
            r#"
            part("lid") cube(10);
            hinge("h", parts = ["lid", "bxo"], at = [0,0,0], axis = [1,0,0]);
            drive("nope", to = 10);
            "#,
        )
        .unwrap();
        assert_eq!(spec.dangling_parts(), vec!["bxo"]);
        assert_eq!(spec.dangling_drives(), vec!["nope"]);
    }

    #[test]
    fn a_mate_missing_its_geometry_says_which_argument() {
        let err = parse_scad_mechanism(
            r#"part("a") cube(10); hinge("h", parts = ["a", "a"], axis = [1,0,0]);"#,
        )
        .unwrap_err();
        assert!(err.contains("at ="), "unhelpful error: {err}");

        let err =
            parse_scad_mechanism(r#"part("a") cube(10); hinge("h", at = [0,0,0]);"#).unwrap_err();
        assert!(err.contains("parts ="), "unhelpful error: {err}");
    }

    #[test]
    fn declarations_can_come_from_loops_and_modules() {
        // The collector is a side channel precisely so this works: the
        // declarations are nowhere near the top level.
        let spec = parse_scad_mechanism(
            r#"
            module finger(i) {
                part(str("finger", i)) translate([i * 12, 0, 0]) cube([10, 10, 40]);
                hinge(str("knuckle", i), parts = [str("finger", i), "palm"],
                      at = [i * 12, 0, 0], axis = [1, 0, 0], range = [0, 90]);
            }
            part("palm", fixed = true) cube([40, 10, 10]);
            for (i = [0 : 2]) finger(i);
            "#,
        )
        .unwrap();
        assert_eq!(
            spec.parts.len(),
            4,
            "{:?}",
            spec.parts.iter().map(|p| &p.name).collect::<Vec<_>>()
        );
        assert_eq!(spec.mates.len(), 3);
        assert!(spec.part("finger2").is_some());
        assert!(spec.dangling_parts().is_empty());
    }

    #[test]
    fn a_failed_parse_leaves_no_mechanism_behind_for_the_next_one() {
        // The collector is thread-local and the eval thread is reused, so a
        // half-collected mechanism from a failed run must not leak forward.
        let _ = parse_scad_mechanism(r#"part("a") cube(10); hinge("h", at = [0,0,0]);"#);
        let spec = parse_scad_mechanism(r#"part("b") cube(10);"#).unwrap();
        assert_eq!(spec.parts.len(), 1);
        assert_eq!(spec.parts[0].name, "b");
        assert!(spec.mates.is_empty());
    }

    #[test]
    fn a_part_can_say_what_it_collides_with() {
        use crate::openscad::mechanism::PartFit;
        let spec = parse_scad_mechanism(
            r#"
            part("plain") cube(10);
            part("hooked", collider = "decompose", friction = 0.4, bounce = 0.1) cube(10);
            part("rail", fixed = true, collider = "mesh", mass = 12) cube(10);
            part("roller", collider = "cylinder", restitution = 0.5) cube(10);
            "#,
        )
        .unwrap();
        // Unstated means unstated: a fixed part still defaults to its mesh and a
        // moving one to a hull, and that decision stays in the assembly rather
        // than being baked in here.
        assert_eq!(spec.parts[0].fit, None);
        assert_eq!(spec.parts[0].friction, None);

        assert_eq!(spec.parts[1].fit, Some(PartFit::Decompose));
        assert_eq!(spec.parts[1].friction, Some(0.4));
        assert_eq!(spec.parts[1].restitution, Some(0.1));

        assert_eq!(spec.parts[2].fit, Some(PartFit::Mesh));
        assert_eq!(spec.parts[2].mass, Some(12.0));

        assert_eq!(spec.parts[3].fit, Some(PartFit::Cylinder));
        // `restitution` is the solver's word for it and `bounce` is CAD's.
        assert_eq!(spec.parts[3].restitution, Some(0.5));
    }

    #[test]
    fn a_mate_can_say_its_parts_also_touch() {
        let spec = parse_scad_mechanism(
            r#"
            part("a", fixed = true) cube(10);
            part("b") cube(10);
            slider("plain", parts = ["b", "a"], at = [0,0,0], axis = [1,0,0]);
            slider("touching", parts = ["b", "a"], at = [0,0,0], axis = [1,0,0], collide = true);
            "#,
        )
        .unwrap();
        assert!(!spec.mate("plain").unwrap().collide);
        assert!(spec.mate("touching").unwrap().collide);
    }

    #[test]
    fn an_unknown_collider_is_an_error_naming_the_alternatives() {
        let err = parse_scad_mechanism(r#"part("a", collider = "squishy") cube(10);"#)
            .expect_err("should not accept it");
        assert!(err.contains("squishy"), "{err}");
        assert!(
            err.contains("decompose"),
            "the message should list what works: {err}"
        );
    }

    #[test]
    fn a_model_declares_a_flexible_rod_and_the_cables_on_it() {
        let spec = parse_scad_mechanism(
            r#"
            part("base", fixed = true) cylinder(h = 10, r = 12);

            continuum("backbone", on = "base", at = [0, 0, 10], axis = [0, 0, 1],
                      length = 300, links = 30, radius = 6, segments = 3,
                      youngs = 200e9, poisson = 0.3, density = 1200,
                      backbone_radius = 0.5, damping = 0.05);

            tendon("t0", along = "backbone", offset = 4, phase = 0,   pretension = 1);
            tendon("t1", along = "backbone", offset = 4, phase = 120, pretension = 1, segment = 2);
            tendon("t2", along = "backbone", offset = 4, phase = 240, pull = 3, force = 40);
            "#,
        )
        .unwrap();

        let rod = spec.continuum("backbone").unwrap();
        assert_eq!(rod.base, "base");
        assert_eq!(rod.at, [0.0, 0.0, 10.0]);
        assert_eq!((rod.links, rod.segments), (30, 3));
        assert_eq!(rod.structural_radius(), 0.5, "the core, not the sheath");
        assert_eq!(rod.damping_ratio, 0.05);
        assert!(!rod.twist, "torsion is opt-in");

        assert_eq!(spec.tendons.len(), 3);
        assert_eq!(spec.tendons[1].phase, 120.0);
        assert_eq!(spec.tendons[1].segment, 2);
        assert_eq!(spec.tendons[2].pull, 3.0);
        assert_eq!(spec.tendons[2].max_force, Some(40.0));
        assert!(spec.dangling_tendons().is_empty());
    }

    #[test]
    fn segments_divide_the_links_and_the_remainder_goes_to_the_root() {
        let spec = parse_scad_mechanism(
            r#"
            part("base", fixed = true) cube(1);
            continuum("rod", on = "base", length = 100, links = 10, radius = 2,
                      segments = 3, youngs = 1e6, density = 1000);
            "#,
        )
        .unwrap();
        let rod = spec.continuum("rod").unwrap();
        assert_eq!(rod.segment_links(1), 0..4);
        assert_eq!(rod.segment_links(2), 4..7);
        assert_eq!(rod.segment_links(3), 7..10);
        assert_eq!(rod.segment_links(4), 0..0, "there is no fourth segment");
    }

    #[test]
    fn a_cable_along_a_rod_that_does_not_exist_is_reported() {
        let spec = parse_scad_mechanism(
            r#"
            part("base", fixed = true) cube(1);
            tendon("t0", along = "spine", offset = 4);
            "#,
        )
        .unwrap();
        assert_eq!(spec.dangling_tendons(), vec!["spine"]);
    }

    #[test]
    fn a_rod_needs_the_numbers_beam_theory_needs() {
        let err = parse_scad_mechanism(r#"continuum("rod", length = 100, links = 10);"#)
            .expect_err("should not accept it");
        assert!(err.contains("radius"), "{err}");

        let err = parse_scad_mechanism(
            r#"continuum("rod", length = 100, links = 2, radius = 1, segments = 5,
                         youngs = 1e6, density = 1000);"#,
        )
        .expect_err("more segments than links");
        assert!(err.contains("empty"), "{err}");
    }

    #[test]
    fn a_mate_can_be_elastic_as_well_as_soft() {
        let spec = parse_scad_mechanism(
            r#"
            part("a") cube(10);
            part("b") cube(10);
            hinge("h", parts = ["a", "b"], at = [0,0,0], axis = [1,0,0],
                  spring = [8, 0.7], elastic = [2.5, 15, 0.1]);
            "#,
        )
        .unwrap();
        let mate = spec.mate("h").unwrap();
        assert_eq!(mate.spring, Some([8.0, 0.7]), "give in the constraint");
        assert_eq!(
            mate.elastic,
            Some([2.5, 15.0, 0.1]),
            "and somewhere it would rather be"
        );
    }

    #[test]
    fn declarations_can_read_the_animation_variable() {
        use crate::openscad::parse_scad_mechanism_at;
        let src = r#"
            part("a") cube(10);
            part("b") cube(10);
            hinge("h", parts = ["a", "b"], at = [0,0,0], axis = [1,0,0], range = [0, 90]);
            drive("h", to = 90 * $t);
        "#;
        assert_eq!(
            parse_scad_mechanism_at(src, 0.0).unwrap().drives[0].to,
            Some(0.0)
        );
        assert_eq!(
            parse_scad_mechanism_at(src, 0.5).unwrap().drives[0].to,
            Some(45.0)
        );
    }
    #[test]
    fn a_gear_can_name_the_part_it_is_carried_on() {
        let spec = parse_scad_mechanism(
            r#"
            part("case", fixed = true) cube([10, 10, 10]);
            part("arm")                cube([20, 4, 4]);
            part("sun")                cylinder(4, 6);
            part("planet")             cylinder(4, 3);

            hinge("arm_pivot", parts = ["arm", "case"], at = [0, 0, 0], axis = [0, 0, 1]);
            hinge("sun_pivot", parts = ["sun", "case"], at = [0, 0, 0], axis = [0, 0, 1]);
            hinge("planet_pivot", parts = ["planet", "arm"], at = [9, 0, 0], axis = [0, 0, 1]);
            gear("mesh", parts = ["planet", "sun"], at = [9, 0, 0], axis = [0, 0, 1],
                 ratio = -2, carrier = "arm");
            "#,
        )
        .unwrap();
        let mesh = spec.mate("mesh").unwrap();
        assert_eq!(mesh.carrier.as_deref(), Some("arm"));
        assert!(spec.dangling_parts().is_empty(), "{:?}", spec.dangling_parts());

        // A mate with no carrier says so, and is the ordinary world-referenced
        // coupling it always was.
        assert!(spec.mate("planet_pivot").unwrap().carrier.is_none());
    }

    #[test]
    fn a_carrier_naming_nothing_is_reported_like_any_other_typo() {
        let spec = parse_scad_mechanism(
            r#"
            part("case", fixed = true) cube([10, 10, 10]);
            part("sun")                cylinder(4, 6);
            part("planet")             cylinder(4, 3);
            gear("mesh", parts = ["planet", "sun"], at = [9, 0, 0], axis = [0, 0, 1],
                 ratio = -2, carrier = "arn");
            "#,
        )
        .unwrap();
        assert_eq!(spec.dangling_parts(), vec!["arn"]);
    }

}
