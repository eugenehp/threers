//! Checking a declared mechanism against the parts that have to perform it.
//!
//! A model states where its hinge is. The hardware drawn around that hinge
//! states the same thing a second time, in the shape of a bore and a pin. Both
//! statements look right on their own, and nothing in either one notices when
//! they stop agreeing — the simulation happily turns the lid about the axis it
//! was *told* about, and the bore it was drawn with goes along for the ride.
//!
//! This runs the mechanism and then asks the geometry what it did. The axis
//! comes back out of the moving triangles by
//! [`recover_screw`](threers::assembly::recover_screw), with nothing consulted
//! but the vertices, and is compared against the declaration. Two accounts of
//! one fact, checked against each other.
//!
//! ```no_run
//! use threers_physics::mechanism::ScadMechanism;
//! use threers_physics::verify::Verify;
//!
//! let mut mech = ScadMechanism::from_file("box.scad")?;
//! let report = Verify::new().run(&mut mech);
//! for finding in &report.findings {
//!     println!("{finding}");
//! }
//! # Ok::<(), threers_physics::mechanism::MechanismError>(())
//! ```
//!
//! # What it can and cannot conclude
//!
//! The recovery needs the same vertices in the same order across poses, and gets
//! them: [`crate::mechanism::ScadMechanism`] evaluates each part once and only ever moves it. That
//! is the one condition `recover_screw` cannot check for itself, and here it
//! holds by construction rather than by luck.
//!
//! What it cannot do is find a fault in a joint that never moves. A mate with no
//! travel in the run produces no motion to recover an axis from, and is reported
//! as unchecked rather than as correct.

use crate::assembly::MateId;
use crate::mechanism::ScadMechanism;
use threers::assembly::{
    engagement_with, floating_with, from_geometry, from_geometry_at, recover_screw, Tri,
};
use threers::math::{Matrix4, Vector3};

/// Something the geometry says that the declaration does not.
#[derive(Debug, Clone, PartialEq)]
pub enum Finding {
    /// The parts turn about a line parallel to the declared one, but not the
    /// same line. The distance is between the two lines, in model units.
    AxisOffset {
        mate: String,
        declared: [f32; 3],
        recovered: [f32; 3],
        distance: f32,
    },
    /// The parts turn about a line pointing somewhere else.
    AxisTilt {
        mate: String,
        declared: [f32; 3],
        recovered: [f32; 3],
        degrees: f32,
    },
    /// The motion is not the kind the mate says it is — a hinge that slides, or
    /// a slider that turns.
    WrongKind {
        mate: String,
        declared: &'static str,
        observed: &'static str,
    },
    /// The two parts did not move rigidly with respect to each other. Something
    /// is bending, or the two "parts" are not one part each.
    NotRigid { mate: String, residual: f32 },
    /// Nothing could be checked here. Not a fault in the model — a gap in what
    /// this run was able to say about it.
    Unchecked { mate: String, reason: &'static str },
    /// Two parts that touch at the start and have separated by the end.
    CameApart {
        a: String,
        b: String,
        widest: f32,
    },
    /// A part that touches nothing, in any pose.
    Floating { part: String },
}

impl std::fmt::Display for Finding {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::AxisOffset {
                mate,
                declared,
                recovered,
                distance,
            } => write!(
                f,
                "{mate}: declared at {declared:?}, turns about {recovered:?} — {distance:.4} away"
            ),
            Self::AxisTilt {
                mate,
                declared,
                recovered,
                degrees,
            } => write!(
                f,
                "{mate}: declared along {declared:?}, turns about {recovered:?} — {degrees:.2}° off"
            ),
            Self::WrongKind {
                mate,
                declared,
                observed,
            } => write!(f, "{mate}: declared a {declared}, behaves like a {observed}"),
            Self::NotRigid { mate, residual } => write!(
                f,
                "{mate}: the parts do not move rigidly together — the fit missed by {residual:.4}"
            ),
            Self::Unchecked { mate, reason } => {
                write!(f, "{mate}: not checked — {reason}")
            }
            Self::CameApart { a, b, widest } => write!(
                f,
                "{a} and {b} start in contact and end {widest:.4} apart"
            ),
            Self::Floating { part } => write!(f, "{part}: touches nothing, in any pose"),
        }
    }
}

impl Finding {
    /// Whether this is a disagreement rather than a gap in coverage.
    ///
    /// [`Finding::Unchecked`] is the only one that is not: it says the run could
    /// not reach a conclusion, which is worth printing and is not a fault.
    pub fn is_fault(&self) -> bool {
        !matches!(self, Self::Unchecked { .. })
    }
}

/// What the run found.
#[derive(Debug, Clone, Default)]
pub struct Report {
    pub findings: Vec<Finding>,
    /// Poses sampled.
    pub poses: usize,
    /// Mates whose axis was recovered and agreed with the declaration.
    pub confirmed: usize,
}

impl Report {
    /// Whether the geometry agrees with everything the model declared.
    pub fn agrees(&self) -> bool {
        !self.findings.iter().any(Finding::is_fault)
    }

    pub fn faults(&self) -> impl Iterator<Item = &Finding> {
        self.findings.iter().filter(|f| f.is_fault())
    }
}

/// How hard to look, and how much disagreement to allow.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Verify {
    /// Poses to sample across the run. More is slower and finds the same
    /// things; the axis recovery wants two poses far apart, not many close ones.
    pub poses: usize,
    /// How far the recovered axis may be from the declared line, in model units.
    pub axis_tolerance: f32,
    /// How far it may point away, in degrees.
    pub tilt_tolerance: f32,
    /// How close counts as touching, for the engagement checks.
    pub contact_tolerance: f32,
    /// Travel below which a mate counts as not having moved, in radians or
    /// model units. An axis recovered from a body that barely turned is an axis
    /// recovered from rounding.
    pub minimum_travel: f32,
}

impl Default for Verify {
    fn default() -> Self {
        Self {
            poses: 8,
            axis_tolerance: 1e-3,
            tilt_tolerance: 0.5,
            contact_tolerance: 1e-3,
            minimum_travel: 0.05,
        }
    }
}

impl Verify {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn poses(mut self, poses: usize) -> Self {
        self.poses = poses.max(2);
        self
    }

    /// How far the recovered axis may be from where it was declared.
    pub fn axis_tolerance(mut self, units: f32) -> Self {
        self.axis_tolerance = units.max(0.0);
        self
    }

    /// How far it may point away, in degrees.
    pub fn tilt_tolerance(mut self, degrees: f32) -> Self {
        self.tilt_tolerance = degrees.max(0.0);
        self
    }

    /// How close two parts must be to count as touching.
    pub fn contact_tolerance(mut self, units: f32) -> Self {
        self.contact_tolerance = units.max(0.0);
        self
    }

    /// Check the declarations against the model's **own** `$t` animation.
    ///
    /// This is the one that finds drift. A model usually says where its hinge is
    /// twice — once in the `rotate()` that swings the lid, and once in the
    /// `hinge()` that declares it — and the two are edited on different days.
    /// Evaluating the model at two values of `$t` and reading the axis back out
    /// of the triangles asks the *drawing* where the lid turns, with the
    /// declaration consulted only at the end, to disagree with.
    ///
    /// [`Self::run`] cannot do this and should not be expected to: it drives the
    /// simulation, and the simulation builds its joints *from* the declaration,
    /// so the parts necessarily turn where they were told to. What `run` checks
    /// is that the pipeline from declaration to solved motion preserves the axis
    /// — worth knowing, and a different question.
    ///
    /// Needs a base part that holds still between the two poses, which is what
    /// "the part it moves against" almost always means. A mate whose base also
    /// moves is reported as unchecked rather than measured against a moving
    /// datum.
    ///
    /// ```no_run
    /// # use threers_physics::verify::Verify;
    /// let source = std::fs::read_to_string("box.scad")?;
    /// let report = Verify::new().against_model(&source, 0.0, 1.0)?;
    /// assert!(report.agrees(), "{:?}", report.faults().collect::<Vec<_>>());
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn against_model(&self, source: &str, from: f64, to: f64) -> Result<Report, String> {
        let a = threers::parse_scad_mechanism_at(source, from)?;
        let b = threers::parse_scad_mechanism_at(source, to)?;

        let triangles = |spec: &threers::openscad::MechanismSpec| -> Vec<(String, Vec<Tri>)> {
            spec.parts
                .iter()
                .map(|p| (p.name.clone(), from_geometry(&p.solid.clone().to_geometry())))
                .collect()
        };
        let before = triangles(&a);
        let after = triangles(&b);
        let find = |set: &[(String, Vec<Tri>)], name: &str| -> Option<Vec<Tri>> {
            set.iter().find(|(n, _)| n == name).map(|(_, t)| t.clone())
        };

        let mut report = Report {
            poses: 2,
            ..Default::default()
        };
        for mate in &a.mates {
            let name = mate.name.clone();
            if !mate.kind.is_angular() {
                continue; // only a turn locates an axis
            }
            let (Some(moving_a), Some(moving_b)) = (
                find(&before, &mate.parts[0]),
                find(&after, &mate.parts[0]),
            ) else {
                report.findings.push(Finding::Unchecked {
                    mate: name,
                    reason: "its moving part is not declared by a part()",
                });
                continue;
            };
            // The datum has to hold still, or "where it turns" has no meaning.
            match (find(&before, &mate.parts[1]), find(&after, &mate.parts[1])) {
                (Some(base_a), Some(base_b)) if !still(&base_a, &base_b) => {
                    report.findings.push(Finding::Unchecked {
                        mate: name,
                        reason: "the part it moves against also moves",
                    });
                    continue;
                }
                (None, _) | (_, None) => {
                    report.findings.push(Finding::Unchecked {
                        mate: name,
                        reason: "its base part is not declared by a part()",
                    });
                    continue;
                }
                _ => {}
            }

            let Some(screw) = recover_screw(&moving_a, &moving_b) else {
                report.findings.push(Finding::Unchecked {
                    mate: name,
                    reason: "the part did not move rigidly between the two poses",
                });
                continue;
            };
            let scale = model_scale(&moving_a);
            if screw.residual > (scale * 1e-3).max(1e-9) as f64 {
                report.findings.push(Finding::NotRigid {
                    mate: name,
                    residual: screw.residual as f32,
                });
                continue;
            }
            if screw.angle.abs() < self.minimum_travel as f64 {
                report.findings.push(Finding::Unchecked {
                    mate: name,
                    reason: "it did not turn far enough between the two poses",
                });
                continue;
            }

            self.compare(
                &name,
                Vector3::new(mate.at[0], mate.at[1], mate.at[2]),
                Vector3::new(mate.axis[0], mate.axis[1], mate.axis[2]).normalize(),
                &screw,
                &mut report,
            );
        }
        Ok(report)
    }

    /// Run the mechanism and check what it did against what it said.
    ///
    /// Advances the simulation — this is not an inspection of a mechanism at
    /// rest, and cannot be: an axis is only visible in something that moved.
    ///
    /// The axis comparison here confirms the *pipeline*, not the model: the
    /// joints were built from the declarations, so they hold the parts on the
    /// declared axis by construction. [`Self::against_model`] is the one that
    /// asks the drawing. What is not tautological here is everything about
    /// contact — a joint can come apart under load, and a part mated to nothing
    /// stays where it was put and touches nobody.
    pub fn run(&self, mech: &mut ScadMechanism) -> Report {
        let bodies = mech.assembly.parts().len();
        if bodies == 0 {
            return Report::default();
        }

        // The base triangles, converted once. Every pose is these, moved.
        let base: Vec<Vec<Tri>> = (0..bodies)
            .map(|i| {
                mech.assembly
                    .part_of(crate::assembly::PartId::from_index(i))
                    .and_then(|p| p.geometry.as_ref())
                    .map(from_geometry)
                    .unwrap_or_default()
            })
            .collect();

        // Sample the run, recording where each body ended up.
        let mut placements: Vec<Vec<Matrix4>> = Vec::with_capacity(self.poses);
        let mut coordinates: Vec<Vec<Option<f32>>> = Vec::with_capacity(self.poses);
        let between = (mech.frame_count().max(1) / self.poses.max(1)).max(1);
        for pose in 0..self.poses {
            if pose > 0 {
                for _ in 0..between {
                    mech.step();
                }
            }
            placements.push(self.snapshot_placements(mech, bodies));
            coordinates.push(self.snapshot_coordinates(mech));
        }

        let mut report = Report {
            poses: placements.len(),
            ..Default::default()
        };
        self.check_axes(mech, &base, &placements, &coordinates, &mut report);
        self.check_contacts(mech, &base, &placements, &mut report);
        report
    }

    fn snapshot_placements(&self, mech: &ScadMechanism, bodies: usize) -> Vec<Matrix4> {
        (0..bodies)
            .map(|i| {
                mech.assembly
                    .body_of(crate::assembly::PartId::from_index(i))
                    .and_then(|b| mech.world.body(b))
                    .map(|b| b.position.to_matrix4())
                    .unwrap_or_default()
            })
            .collect()
    }

    fn snapshot_coordinates(&self, mech: &ScadMechanism) -> Vec<Option<f32>> {
        (0..mech.assembly.mates().len())
            .map(|i| mech.assembly.coordinate(MateId::from_index(i), &mech.world))
            .collect()
    }

    /// Compare each mate's declared axis against the one its parts turned about.
    fn check_axes(
        &self,
        mech: &ScadMechanism,
        base: &[Vec<Tri>],
        placements: &[Vec<Matrix4>],
        coordinates: &[Vec<Option<f32>>],
        report: &mut Report,
    ) {
        for (index, mate) in mech.assembly.mates().iter().enumerate() {
            let name = if mate.name.is_empty() {
                format!("mate {index}")
            } else {
                mate.name.clone()
            };
            let (Some(declared_axis), Some(_)) = (mate.b.direction(), mate.kind.axial()) else {
                continue; // no axis to check, or no coordinate to move along it
            };
            let declared_point = mate.b.origin();

            // The two poses furthest apart in this mate's own coordinate. An
            // axis is located by a turn, and a small turn locates it badly.
            let Some((first, last, travel)) = extreme_poses(coordinates, index) else {
                report.findings.push(Finding::Unchecked {
                    mate: name,
                    reason: "no two sampled poses are far enough apart to locate an axis",
                });
                continue;
            };
            if travel < self.minimum_travel {
                report.findings.push(Finding::Unchecked {
                    mate: name,
                    reason: "the mate travelled less than this check needs to say anything",
                });
                continue;
            }

            let moving = mate.a.part.index();
            let holding = mate.b.part.index();
            // Measured in the frame of the part being moved against, so a base
            // that is itself moving does not smear the answer.
            let (Some(a), Some(b)) = (
                relative_pose(placements, first, moving, holding),
                relative_pose(placements, last, moving, holding),
            ) else {
                report.findings.push(Finding::Unchecked {
                    mate: name,
                    reason: "a part had no pose in the sampled run",
                });
                continue;
            };
            let before = from_geometry_at(geometry_of(mech, moving), &a);
            let after = from_geometry_at(geometry_of(mech, moving), &b);
            let _ = base;

            let Some(screw) = recover_screw(&before, &after) else {
                report.findings.push(Finding::Unchecked {
                    mate: name,
                    reason: "the motion did not reduce to a screw, so no axis could be recovered",
                });
                continue;
            };

            let scale = model_scale(&before);
            if screw.residual > (scale * 1e-3).max(1e-9) as f64 {
                report.findings.push(Finding::NotRigid {
                    mate: name,
                    residual: screw.residual as f32,
                });
                continue;
            }

            let recovered_dir = Vector3::new(
                screw.direction[0] as f32,
                screw.direction[1] as f32,
                screw.direction[2] as f32,
            );
            let recovered_point = Vector3::new(
                screw.point[0] as f32,
                screw.point[1] as f32,
                screw.point[2] as f32,
            );

            // A recovered direction may come back either way round — an axis is
            // a line, not an arrow — so the tilt is measured to the nearer end.
            let cos = declared_axis.dot(recovered_dir).abs().clamp(0.0, 1.0);
            let degrees = cos.acos().to_degrees();
            if degrees > self.tilt_tolerance {
                report.findings.push(Finding::AxisTilt {
                    mate: name,
                    declared: declared_axis.to_array(),
                    recovered: recovered_dir.to_array(),
                    degrees,
                });
                continue;
            }

            let distance = point_to_line(recovered_point, declared_point, declared_axis);
            if distance > self.axis_tolerance {
                report.findings.push(Finding::AxisOffset {
                    mate: name,
                    declared: declared_point.to_array(),
                    recovered: recovered_point.to_array(),
                    distance,
                });
                continue;
            }

            // A hinge that slides, or a slider that turns.
            let observed = describe(&screw, scale);
            let declared = match mate.kind {
                crate::assembly::MateKind::Hinge => "hinge",
                crate::assembly::MateKind::Slider => "slider",
                _ => "screw",
            };
            if observed != declared && !matches!(mate.kind, crate::assembly::MateKind::Screw { .. })
            {
                report.findings.push(Finding::WrongKind {
                    mate: name,
                    declared,
                    observed,
                });
                continue;
            }

            report.confirmed += 1;
        }
    }

    /// Joints that come apart, and parts attached to nothing.
    fn check_contacts(
        &self,
        mech: &ScadMechanism,
        base: &[Vec<Tri>],
        placements: &[Vec<Matrix4>],
        report: &mut Report,
    ) {
        let bodies = base.len();
        let poses: Vec<Vec<Vec<Tri>>> = placements
            .iter()
            .map(|pose| {
                (0..bodies)
                    .map(|i| {
                        from_geometry_at(geometry_of(mech, i), &pose[i])
                    })
                    .collect()
            })
            .collect();
        // The correspondence is known — these bodies were moved, not rebuilt —
        // so it is stated rather than guessed at.
        let alignment: Vec<Vec<Option<usize>>> =
            vec![(0..bodies).map(Some).collect(); poses.len()];

        let name = |i: usize| {
            mech.assembly
                .part_of(crate::assembly::PartId::from_index(i))
                .map(|p| p.name.clone())
                .unwrap_or_else(|| format!("part {i}"))
        };

        for e in engagement_with(&poses, &alignment, self.contact_tolerance as f64) {
            if !e.persistent() {
                report.findings.push(Finding::CameApart {
                    a: name(e.a),
                    b: name(e.b),
                    widest: e.widest as f32,
                });
            }
        }
        for i in floating_with(&poses, &alignment, self.contact_tolerance as f64) {
            report.findings.push(Finding::Floating { part: name(i) });
        }
    }
}

impl Verify {
    /// Compare a recovered motion against a declared axis, recording whatever
    /// they disagree about.
    fn compare(
        &self,
        name: &str,
        declared_point: Vector3,
        declared_axis: Vector3,
        screw: &threers::assembly::Screw,
        report: &mut Report,
    ) {
        let recovered_dir = Vector3::new(
            screw.direction[0] as f32,
            screw.direction[1] as f32,
            screw.direction[2] as f32,
        );
        let recovered_point = Vector3::new(
            screw.point[0] as f32,
            screw.point[1] as f32,
            screw.point[2] as f32,
        );

        // A recovered direction may come back either way round — an axis is a
        // line, not an arrow — so the tilt is measured to the nearer end.
        let cos = declared_axis.dot(recovered_dir).abs().clamp(0.0, 1.0);
        let degrees = cos.acos().to_degrees();
        if degrees > self.tilt_tolerance {
            report.findings.push(Finding::AxisTilt {
                mate: name.to_string(),
                declared: declared_axis.to_array(),
                recovered: recovered_dir.to_array(),
                degrees,
            });
            return;
        }

        // Distance between two lines, not between two points: an axis point is
        // only defined up to sliding along the axis, so comparing the points
        // directly would report a difference that is not one.
        let distance = point_to_line(recovered_point, declared_point, declared_axis);
        if distance > self.axis_tolerance {
            report.findings.push(Finding::AxisOffset {
                mate: name.to_string(),
                declared: declared_point.to_array(),
                recovered: recovered_point.to_array(),
                distance,
            });
            return;
        }
        report.confirmed += 1;
    }
}

/// Whether a body is in the same place in both poses.
fn still(a: &[Tri], b: &[Tri]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let tol = (model_scale(a) as f64 * 1e-6).max(1e-12);
    a.iter().zip(b.iter()).all(|(x, y)| {
        x.iter()
            .zip(y.iter())
            .all(|(p, q)| (0..3).all(|k| (p[k] - q[k]).abs() <= tol))
    })
}

fn geometry_of(mech: &ScadMechanism, part: usize) -> &threers::core::BufferGeometry {
    static EMPTY: std::sync::OnceLock<threers::core::BufferGeometry> = std::sync::OnceLock::new();
    mech.assembly
        .part_of(crate::assembly::PartId::from_index(part))
        .and_then(|p| p.geometry.as_ref())
        .unwrap_or_else(|| EMPTY.get_or_init(threers::core::BufferGeometry::new))
}

/// The two sampled poses in which a mate's coordinate is furthest apart.
fn extreme_poses(coordinates: &[Vec<Option<f32>>], mate: usize) -> Option<(usize, usize, f32)> {
    let mut lo: Option<(usize, f32)> = None;
    let mut hi: Option<(usize, f32)> = None;
    for (p, row) in coordinates.iter().enumerate() {
        let Some(Some(v)) = row.get(mate) else {
            continue;
        };
        if lo.is_none_or(|(_, x)| *v < x) {
            lo = Some((p, *v));
        }
        if hi.is_none_or(|(_, x)| *v > x) {
            hi = Some((p, *v));
        }
    }
    let (a, av) = lo?;
    let (b, bv) = hi?;
    (a != b).then_some((a, b, (bv - av).abs()))
}

/// Where `moving` sits in `holding`'s frame, at one pose.
fn relative_pose(
    placements: &[Vec<Matrix4>],
    pose: usize,
    moving: usize,
    holding: usize,
) -> Option<Matrix4> {
    let row = placements.get(pose)?;
    let m = row.get(moving)?;
    let h = row.get(holding)?;
    Some(h.invert().multiply(m))
}

/// Largest extent of a body, for judging what counts as a small residual.
fn model_scale(tris: &[Tri]) -> f32 {
    let (lo, hi) = threers::assembly::aabb(tris);
    let mut d: f64 = 0.0;
    for k in 0..3 {
        d = d.max(hi[k] - lo[k]);
    }
    d as f32
}

/// Distance from a point to a line.
fn point_to_line(p: Vector3, origin: Vector3, direction: Vector3) -> f32 {
    let d = p - origin;
    let along = d.dot(direction);
    (d - direction * along).length()
}

/// What the motion looks like, as distinct from what it was called.
fn describe(screw: &threers::assembly::Screw, scale: f32) -> &'static str {
    let slide = screw.slide.abs() as f32;
    let turn = screw.angle.abs() as f32;
    // A slide is meaningful against the body's size; a turn against a radian.
    let sliding = slide > scale * 1e-3;
    let turning = turn > 1e-3;
    match (turning, sliding) {
        (true, false) => "hinge",
        (false, true) => "slider",
        (true, true) => "screw",
        (false, false) => "nothing",
    }
}
