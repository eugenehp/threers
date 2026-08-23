//! Cosserat-rod reference solutions to check a discretised rod against.
//!
//! [`Rod`](crate::rod::Rod) approximates a continuum by chopping it up. Whether
//! that approximation is any good is not a question the approximation can
//! answer, and `wL⁴/8EI` only answers it for small deflections of an
//! unloaded cantilever — which is the easy half. This reads the reference
//! solutions the [SoRoSim](https://github.com/SoRoSim/SoRoSim) Cosserat-rod
//! solver produced for the same rods under loads large enough to bend them
//! through tens of degrees.
//!
//! # Where the data comes from
//!
//! It ships with [opencr-mujoco](https://github.com/ContinuumRoboticsLab/opencr-mujoco)
//! (MIT) under `data/reference/sorosim/`, and is **not** vendored here — 4.3 MB
//! of someone else's validation set does not belong in this repository. Point
//! [`data_dir`] at a clone:
//!
//! ```bash
//! git clone https://github.com/ContinuumRoboticsLab/opencr-mujoco ../opencr-mujoco
//! # ...which is also the default location, so usually nothing to set
//! export THREERS_SOROSIM_DATA=/somewhere/else/sorosim   # if it is not
//! ```
//!
//! Everything here returns `None` rather than panicking when the data is
//! absent, so a build without it degrades to skipped validation rather than a
//! failing one.
//!
//! # Two frames
//!
//! SoRoSim's files are in SoRoSim's frame and the rod is simulated in
//! threers'. Every 3-vector is rotated by [`FILE_TO_SIM`] on the way in — once,
//! at load, so nothing downstream has to remember. In the converted frame the
//! rod is clamped at the origin and runs along **+Z**.
//!
//! # What the sets contain
//!
//! - [`Statics`] — 500 equilibrium shapes per material, each under its own
//!   randomly oriented gravity, a wrench at mid-span and a wrench at the tip,
//!   measured at 14 non-uniformly spaced arc stations.
//! - [`TipRelease`] — a rod held bent by two wrenches, let go, and tracked at
//!   mid-span and tip for a couple of seconds at 200 Hz.
//!
//! # What it says so far
//!
//! Mean over six shapes, as a percentage of the rod's own length. These are
//! large deflections — tens of degrees, tip wrenches comparable to the rod's
//! whole weight — not the small-deflection regime `wL⁴/8EI` covers.
//!
//! Mean shape error at 32 substeps of 128 iterations, no shape diverging:
//!
//! | links | TPU, 5 mm, `E` = 70 MPa | spring steel, 0.8 mm, `E` = 200 GPa |
//! |---|---|---|
//! | 5 | 1.56% | **1.48%** |
//! | 10 | 0.63% | 2.31% |
//! | 20 | 0.32% | 6.95% |
//! | 30 | **0.23%** | 12.23% |
//!
//! Three things fall out of that, and none was obvious beforehand.
//!
//! **Torsion is the largest single term.** With the stations' twist locked —
//! [`Rod`]'s default, because a planar bend never uses it — the same comparison
//! reads 5.7% for the TPU rod and 42% for the steel one, with half the steel
//! shapes diverging outright. The reference wrenches carry moment on all three
//! axes; a rod that cannot take the component along its own simply settles
//! somewhere else. [`Material::rod`] therefore turns it on.
//!
//! **The discretisation converges, and this is the proof.** The soft rod halves
//! its error at every doubling of `links`, from 1.56% down to 0.23%, and is
//! completely indifferent to the solver budget and to how long it is left to
//! settle. `n·EI/L` does what it claims.
//!
//! **What limits the stiff rod is the solver, and it shows up as accuracy
//! rather than as a crash.** Steel is best at *five* links and gets steadily
//! worse as it is chopped finer — the opposite of convergence, and the
//! signature of a chain the solver cannot keep up with: more links is a longer
//! chain of stiffer constraints on lighter bodies, and sequential impulses
//! carry information about one constraint per iteration. Below 32/128 it does
//! crash, diverging on one shape in six at 16/64, and given *longer* to settle
//! at that budget it gets worse rather than better.
//!
//! So the budget has to scale with both the chain and the stiffness, and for a
//! stiff light rod there is a point past which more links buys nothing. That
//! point is a property of the solver — improving how it conditions chains would
//! move it, and would show up here immediately.

use crate::build::Continuum;
use crate::rod::Rod;
use std::path::{Path, PathBuf};
use threers::math::Vector3;

/// Rotation taking a SoRoSim file vector into the simulation frame.
///
/// Rows, so `v' = FILE_TO_SIM · v`. It is a permutation with a flip, and the
/// only reason it is written out rather than folded into the parser is that a
/// frame conversion nobody can see is the one that gets applied twice.
pub const FILE_TO_SIM: [[f32; 3]; 3] = [[0.0, 0.0, 1.0], [0.0, -1.0, 0.0], [1.0, 0.0, 0.0]];

fn to_sim(v: Vector3) -> Vector3 {
    let m = FILE_TO_SIM;
    Vector3::new(
        m[0][0] * v.x + m[0][1] * v.y + m[0][2] * v.z,
        m[1][0] * v.x + m[1][1] * v.y + m[1][2] * v.z,
        m[2][0] * v.x + m[2][1] * v.y + m[2][2] * v.z,
    )
}

/// Where the reference data lives.
///
/// `$THREERS_SOROSIM_DATA` if set, else `../opencr-mujoco/data/reference/sorosim`
/// beside this workspace. `None` when neither exists.
pub fn data_dir() -> Option<PathBuf> {
    if let Ok(dir) = std::env::var("THREERS_SOROSIM_DATA") {
        let path = PathBuf::from(dir);
        return path.is_dir().then_some(path);
    }
    let beside = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../../opencr-mujoco/data/reference/sorosim");
    beside.is_dir().then(|| beside)
}

/// A force and a moment applied at one point.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Wrench {
    pub force: Vector3,
    pub moment: Vector3,
}

impl Wrench {
    pub const ZERO: Self = Self {
        force: Vector3::ZERO,
        moment: Vector3::ZERO,
    };

    pub fn is_zero(&self) -> bool {
        *self == Self::ZERO
    }
}

impl Default for Wrench {
    fn default() -> Self {
        Self::ZERO
    }
}

/// One of the two rods the reference set covers.
///
/// The numbers are from opencr-mujoco's `configs/evaluation/*.json`, and the
/// pair is chosen to bracket the interesting range: a steel wire that is stiff
/// and nearly massless against its own stiffness, and a thick soft polymer that
/// is the other way round.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Material {
    /// 0.8 mm spring steel, 600 mm long. `E = 200 GPa`.
    SpringSteel,
    /// 5 mm TPU, 400 mm long. `E = 70 MPa`.
    Tpu,
}

impl Material {
    /// The file stem the reference set uses.
    pub fn test_type(self) -> &'static str {
        match self {
            Material::SpringSteel => "SpringSteelRodMuJoCo",
            Material::Tpu => "TPURodMuJoCo",
        }
    }

    /// The rod itself, at the requested discretisation.
    ///
    /// `radius` and the structural core are the same here — these are bare
    /// rods with no sheath, so both conventions agree and the comparison does
    /// not hinge on which one this crate uses.
    ///
    /// **Twisting**, which is not the default elsewhere and has to be here: the
    /// reference wrenches carry moments in all three axes, and a rod whose
    /// stations cannot twist simply refuses the component along its own — it
    /// then reaches a different equilibrium and the comparison measures the
    /// missing degree of freedom rather than the discretisation.
    pub fn rod(self, links: usize) -> Rod {
        match self {
            Material::SpringSteel => Rod::new(0.6, links)
                .radius(0.0008)
                .material(200.0e9, 0.33, 7870.0)
                .damped(0.1)
                .twisting(),
            Material::Tpu => Rod::new(0.4, links)
                .radius(0.005)
                .material(70.0e6, 0.4, 1200.0)
                .damped(0.1)
                .twisting(),
        }
    }

    /// How long the reference protocol ramps the wrenches on over, in seconds.
    ///
    /// Long enough that the rod arrives at equilibrium rather than swinging
    /// into it — a statics comparison against a rod still ringing is a
    /// dynamics comparison nobody asked for.
    pub fn ramp_seconds(self) -> f32 {
        match self {
            Material::SpringSteel => 5.0,
            Material::Tpu => 2.0,
        }
    }
}

/// One equilibrium shape, with the loads that produced it.
#[derive(Debug, Clone, PartialEq)]
pub struct StaticsShape {
    /// Gravity for this shape, in the simulation frame. Randomly oriented, and
    /// 9.81 in magnitude.
    pub gravity: Vector3,
    /// Applied at arc 0.5.
    pub mid: Wrench,
    /// Applied at arc 1.
    pub tip: Wrench,
    /// Backbone positions at [`Statics::arc`], in the simulation frame.
    pub stations: Vec<Vector3>,
}

impl StaticsShape {
    /// Straight-line distance from the root to the tip.
    pub fn chord(&self) -> f32 {
        match (self.stations.first(), self.stations.last()) {
            (Some(a), Some(b)) => (*b - *a).length(),
            _ => 0.0,
        }
    }
}

/// The static-equilibrium set for one material.
#[derive(Debug, Clone, PartialEq)]
pub struct Statics {
    pub material: Material,
    /// Normalised arc position of each measured station, 0 at the root to 1 at
    /// the tip.
    ///
    /// **Not uniform, and with a repeat.** SoRoSim models the rod as two
    /// segments and reports Gauss–Lobatto stations within each, so the
    /// positions bunch at the ends of both halves and the segment boundary is
    /// measured twice.
    pub arc: Vec<f32>,
    pub shapes: Vec<StaticsShape>,
}

impl Statics {
    /// Load one material's shapes, or `None` when the data is not present.
    pub fn load(material: Material) -> Option<Self> {
        let dir = data_dir()?;
        let path = dir
            .join("sorosim_statics")
            .join(format!("{}_dataStatics.csv", material.test_type()));
        let text = std::fs::read_to_string(path).ok()?;
        Self::parse(material, &text)
    }

    /// The reference shape resampled onto arbitrary arc positions.
    ///
    /// Linear in arc, component by component — the same thing the reference
    /// pipeline does, and the reason two solutions sampled differently can be
    /// compared at all. `targets` outside `[0, 1]` clamp to the ends.
    pub fn interpolate(&self, shape: &StaticsShape, targets: &[f32]) -> Vec<Vector3> {
        targets
            .iter()
            .map(|t| interpolate_at(&self.arc, &shape.stations, *t))
            .collect()
    }

    fn parse(material: Material, text: &str) -> Option<Self> {
        let mut lines = text.lines().filter(|l| !l.trim().is_empty());
        let header: Vec<&str> = lines.next()?.split(',').map(str::trim).collect();
        let arc_row: Vec<&str> = lines.next()?.split(',').map(str::trim).collect();

        // `gravity`, `mid_wrench` and `tip_wrench` each hold one component per
        // row of the six-row group; the rest are the measurement stations.
        let column = |name: &str| header.iter().position(|c| *c == name);
        let (gravity_col, mid_col, tip_col) = (
            column("gravity")?,
            column("mid_wrench")?,
            column("tip_wrench")?,
        );
        let stations: Vec<usize> = header
            .iter()
            .enumerate()
            .filter(|(_, c)| c.starts_with("seg"))
            .map(|(i, _)| i)
            .collect();
        if stations.is_empty() {
            return None;
        }

        // Segment-local arc positions become global ones: SoRoSim reports
        // `seg2_s01` at its own 0, which is halfway along the rod.
        let segments = stations
            .iter()
            .filter_map(|i| segment_number(header[*i]))
            .max()?
            .max(1);
        let mut arc = Vec::with_capacity(stations.len());
        for i in &stations {
            let seg = segment_number(header[*i])? as f32;
            let local: f32 = arc_row.get(*i)?.parse().ok()?;
            arc.push((seg - 1.0 + local) / segments as f32);
        }

        let rows: Vec<Vec<&str>> = lines.map(|l| l.split(',').map(str::trim).collect()).collect();
        let mut shapes = Vec::with_capacity(rows.len() / 6);
        for group in rows.chunks_exact(6) {
            // Six rows per shape: the three Euler rows carry the *moment* of a
            // wrench and the three position rows carry its *force*, alongside
            // the angles and positions they are named for.
            let (ex, ey, ez, px, py, pz) = (
                &group[0], &group[1], &group[2], &group[3], &group[4], &group[5],
            );
            let vector = |row_x: &[&str], row_y: &[&str], row_z: &[&str], col: usize| {
                Some(Vector3::new(
                    row_x.get(col)?.parse().ok()?,
                    row_y.get(col)?.parse().ok()?,
                    row_z.get(col)?.parse().ok()?,
                ))
            };
            let mut points = Vec::with_capacity(stations.len());
            for i in &stations {
                points.push(to_sim(vector(px, py, pz, *i)?));
            }
            shapes.push(StaticsShape {
                gravity: to_sim(vector(px, py, pz, gravity_col)?),
                mid: Wrench {
                    force: to_sim(vector(px, py, pz, mid_col)?),
                    moment: to_sim(vector(ex, ey, ez, mid_col)?),
                },
                tip: Wrench {
                    force: to_sim(vector(px, py, pz, tip_col)?),
                    moment: to_sim(vector(ex, ey, ez, tip_col)?),
                },
                stations: points,
            });
        }
        Some(Self {
            material,
            arc,
            shapes,
        })
    }
}

/// One tip-release test: a rod held bent, let go, and tracked as it springs
/// back.
#[derive(Debug, Clone, PartialEq)]
pub struct TipRelease {
    pub material: Material,
    /// The damping SoRoSim ran with, and the gravity it ran under.
    pub gravity: Vector3,
    /// What held the rod bent before the release, at arc 0.5 and arc 1.
    pub mid: Wrench,
    pub tip: Wrench,
    /// Seconds since release.
    pub time: Vec<f32>,
    /// Mid-span position at each sample, in the simulation frame.
    pub mid_track: Vec<Vector3>,
    /// Tip position at each sample.
    pub tip_track: Vec<Vector3>,
}

impl TipRelease {
    /// Load test `index` (1-based, 1..=10) for a material.
    pub fn load(material: Material, index: usize) -> Option<Self> {
        let dir = data_dir()?;
        let path = dir
            .join("sorosim_dynamics")
            .join(format!("{}_{}.txt", material.test_type(), index));
        let text = std::fs::read_to_string(path).ok()?;
        Self::parse(material, &text)
    }

    fn parse(material: Material, text: &str) -> Option<Self> {
        let mut lines = text.lines().filter(|l| !l.trim().is_empty());
        let numbers = |line: &str| -> Vec<f32> {
            line.split_whitespace()
                .filter_map(|t| t.parse::<f32>().ok())
                .collect()
        };
        // Line 1 is the damping and the gravity vector; line 2 the two holding
        // wrenches; the rest is `time` and two poses of six numbers each.
        let first = numbers(lines.next()?);
        let second = numbers(lines.next()?);
        let gravity = to_sim(Vector3::new(
            *first.get(first.len().checked_sub(3)?)?,
            *first.get(first.len().checked_sub(2)?)?,
            *first.get(first.len().checked_sub(1)?)?,
        ));
        let wrench = |at: usize| -> Option<Wrench> {
            Some(Wrench {
                force: to_sim(Vector3::new(
                    *second.get(at)?,
                    *second.get(at + 1)?,
                    *second.get(at + 2)?,
                )),
                moment: to_sim(Vector3::new(
                    *second.get(at + 3)?,
                    *second.get(at + 4)?,
                    *second.get(at + 5)?,
                )),
            })
        };
        let (mid, tip) = (wrench(0)?, wrench(6)?);

        let (mut time, mut mid_track, mut tip_track) = (Vec::new(), Vec::new(), Vec::new());
        for line in lines {
            let v = numbers(line);
            if v.len() < 13 {
                continue;
            }
            time.push(v[0]);
            // Each pose is three angles then three positions.
            mid_track.push(to_sim(Vector3::new(v[4], v[5], v[6])));
            tip_track.push(to_sim(Vector3::new(v[10], v[11], v[12])));
        }
        (!time.is_empty()).then_some(Self {
            material,
            gravity,
            mid,
            tip,
            time,
            mid_track,
            tip_track,
        })
    }
}

/// How to run the statics comparison.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StaticsRun {
    /// How finely to chop the rod.
    pub links: usize,
    pub substeps: usize,
    pub velocity_iterations: usize,
    /// Seconds spent ramping the wrenches on and then holding them. The
    /// material's own [`ramp_seconds`](Material::ramp_seconds) is used for the
    /// ramp; this is the total.
    pub seconds: f32,
    /// How far the backbone may stretch before the shape is called diverged,
    /// as a fraction of the rod's length. A rod is inextensible, so any
    /// measurable stretch means the solver lost the chain rather than that the
    /// answer is merely inaccurate.
    pub stretch_tolerance: f32,
    /// Whether the rod's stations may twist. On, and it should stay on — see
    /// [`Material::rod`]. Exposed so the test suite can measure what turning it
    /// off costs, which is most of the accuracy.
    pub twist: bool,
}

impl Default for StaticsRun {
    fn default() -> Self {
        Self {
            links: 10,
            twist: true,
            substeps: 16,
            velocity_iterations: 64,
            seconds: 10.0,
            stretch_tolerance: 0.02,
        }
    }
}

impl StaticsRun {
    pub fn links(mut self, links: usize) -> Self {
        self.links = links.max(1);
        self
    }

    pub fn budget(mut self, substeps: usize, velocity_iterations: usize) -> Self {
        self.substeps = substeps.max(1);
        self.velocity_iterations = velocity_iterations.max(1);
        self
    }
}

/// What one shape came out at.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ShapeReport {
    pub index: usize,
    /// Mean distance between the simulated backbone and the reference one,
    /// paired by arc.
    pub shape_error: f32,
    pub tip_error: f32,
    /// How far the reference tip is from the root — the scale the errors above
    /// should be read against.
    pub reference_chord: f32,
    /// What the backbone measured. A rod is inextensible, so this should be
    /// the rod's length.
    pub arc_length: f32,
    /// The solver lost the chain: the rod stretched, or went non-finite.
    pub diverged: bool,
}

/// The whole comparison, one material at one discretisation.
#[derive(Debug, Clone, PartialEq)]
pub struct StaticsReport {
    pub material: Material,
    pub run: StaticsRun,
    pub shapes: Vec<ShapeReport>,
}

impl StaticsReport {
    /// Shapes the solver actually held.
    pub fn converged(&self) -> impl Iterator<Item = &ShapeReport> {
        self.shapes.iter().filter(|s| !s.diverged)
    }

    pub fn diverged_count(&self) -> usize {
        self.shapes.iter().filter(|s| s.diverged).count()
    }

    /// Mean shape error over the shapes that converged, as a fraction of the
    /// rod's length.
    ///
    /// Relative because the two materials are different lengths and the
    /// absolute millimetres are not comparable between them.
    pub fn mean_relative_shape_error(&self, rod_length: f32) -> f32 {
        let (sum, n) = self
            .converged()
            .fold((0.0, 0usize), |(s, n), r| (s + r.shape_error, n + 1));
        if n == 0 || rod_length <= 0.0 {
            f32::NAN
        } else {
            sum / n as f32 / rod_length
        }
    }

    /// The same for the tip alone, which is what a controller cares about.
    pub fn mean_relative_tip_error(&self, rod_length: f32) -> f32 {
        let (sum, n) = self
            .converged()
            .fold((0.0, 0usize), |(s, n), r| (s + r.tip_error, n + 1));
        if n == 0 || rod_length <= 0.0 {
            f32::NAN
        } else {
            sum / n as f32 / rod_length
        }
    }

    /// The worst shape that did not diverge, as a fraction of the rod's length.
    pub fn worst_relative_shape_error(&self, rod_length: f32) -> f32 {
        self.converged()
            .map(|r| r.shape_error / rod_length)
            .fold(0.0f32, f32::max)
    }
}

/// Settle one reference shape and measure how close the rod gets.
///
/// The protocol is the reference pipeline's: set that shape's gravity, ramp its
/// two wrenches on over the material's ramp time, hold, then compare the
/// backbone against the reference **paired by arc length** rather than by
/// index. See [`Statics::interpolate`] for why that distinction is not a
/// detail.
pub fn evaluate_shape(set: &Statics, index: usize, run: StaticsRun) -> Option<ShapeReport> {
    use threers_physics::prelude::*;

    let shape = set.shapes.get(index)?;
    let rod = set.material.rod(run.links);
    let length = rod.length;

    let mut world = World::new();
    world.substeps = run.substeps;
    world.solver_config.velocity_iterations = run.velocity_iterations;
    world.gravity = shape.gravity;

    // The reference rod is clamped at the origin and runs along +Z.
    let arm = Continuum::build(
        &mut world,
        rod,
        None,
        Vector3::ZERO,
        Vector3::new(0.0, 0.0, 1.0),
    );

    let steps = (run.seconds / world.timestep).max(1.0) as usize;
    let ramp_steps = (set.material.ramp_seconds() / world.timestep).max(1.0);
    for step in 0..steps {
        // Ramped rather than applied at once, so the rod arrives at its
        // equilibrium instead of swinging into it and being measured mid-ring.
        let scale = ((step as f32) / ramp_steps).min(1.0);
        arm.apply_wrench_at_arc(
            &mut world,
            0.5,
            shape.mid.force * scale,
            shape.mid.moment * scale,
        );
        arm.apply_wrench_at_arc(
            &mut world,
            1.0,
            shape.tip.force * scale,
            shape.tip.moment * scale,
        );
        world.step_fixed();
    }

    let simulated = arm.backbone(&world);
    let arc = arm.backbone_arc();
    let reference = set.interpolate(shape, &arc);
    let arc_length = arm.arc_length(&world);

    let finite = simulated
        .iter()
        .all(|p| p.x.is_finite() && p.y.is_finite() && p.z.is_finite());
    let diverged =
        !finite || (arc_length - length).abs() > run.stretch_tolerance * length;

    let n = simulated.len().min(reference.len()).max(1);
    let shape_error = simulated
        .iter()
        .zip(&reference)
        .map(|(a, b)| (*a - *b).length())
        .sum::<f32>()
        / n as f32;
    let tip_error = match (simulated.last(), reference.last()) {
        (Some(a), Some(b)) => (*a - *b).length(),
        _ => f32::NAN,
    };

    Some(ShapeReport {
        index,
        shape_error,
        tip_error,
        reference_chord: shape.chord(),
        arc_length,
        diverged,
    })
}

/// Run the first `count` shapes of a set. `None` for all 500.
pub fn evaluate_statics(set: &Statics, run: StaticsRun, count: Option<usize>) -> StaticsReport {
    let total = count.unwrap_or(set.shapes.len()).min(set.shapes.len());
    StaticsReport {
        material: set.material,
        run,
        shapes: (0..total)
            .filter_map(|i| evaluate_shape(set, i, run))
            .collect(),
    }
}

/// Linear interpolation of a polyline sampled at `arc`, evaluated at `target`.
///
/// `arc` must be non-decreasing. Repeats are tolerated — the reference set has
/// one at its segment boundary — and resolve to the first of the pair, which is
/// the same point.
fn interpolate_at(arc: &[f32], points: &[Vector3], target: f32) -> Vector3 {
    if arc.is_empty() || points.is_empty() {
        return Vector3::ZERO;
    }
    let t = target.clamp(arc[0], arc[arc.len() - 1]);
    for i in 1..arc.len().min(points.len()) {
        if t <= arc[i] {
            let span = arc[i] - arc[i - 1];
            if span <= 0.0 {
                return points[i - 1];
            }
            let f = (t - arc[i - 1]) / span;
            return points[i - 1] + (points[i] - points[i - 1]) * f;
        }
    }
    points[points.len() - 1]
}

fn segment_number(column: &str) -> Option<usize> {
    column
        .split('_')
        .next()?
        .strip_prefix("seg")?
        .parse()
        .ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = "\
rowLabel,gravity,mid_wrench,tip_wrench,seg1_s01,seg1_s02,seg1_s03,seg2_s01,seg2_s02,seg2_s03
arclength,0,0,0,0,0.5,1,0,0.5,1
EulX_s001,0,1,4,0,0,0,0,0,0
EulY_s001,0,2,5,0,0,0,0,0,0
EulZ_s001,0,3,6,0,0,0,0,0,0
Px_s001,0.1,7,10,0,0,0,0,0,0
Py_s001,0.2,8,11,0,0,0,0,0,0
Pz_s001,9.81,9,12,0,0.15,0.3,0.3,0.45,0.6
";

    #[test]
    fn the_frame_conversion_is_a_permutation_with_a_flip() {
        // Whatever else it does it must not change a length, or every error in
        // the comparison is scaled by however much it does.
        let v = Vector3::new(1.0, -2.0, 3.0);
        let c = to_sim(v);
        assert!((c.length() - v.length()).abs() < 1e-6);
        assert_eq!(c, Vector3::new(3.0, 2.0, 1.0));
    }

    #[test]
    fn segment_local_arc_becomes_global_arc() {
        let set = Statics::parse(Material::SpringSteel, SAMPLE).unwrap();
        assert_eq!(set.arc, vec![0.0, 0.25, 0.5, 0.5, 0.75, 1.0]);
    }

    #[test]
    fn the_six_rows_split_into_forces_moments_and_positions() {
        let set = Statics::parse(Material::SpringSteel, SAMPLE).unwrap();
        assert_eq!(set.shapes.len(), 1);
        let shape = &set.shapes[0];
        // Gravity is (0.1, 0.2, 9.81) in the file, so (9.81, -0.2, 0.1) here.
        assert_eq!(shape.gravity, Vector3::new(9.81, -0.2, 0.1));
        // Mid force from the P rows, mid moment from the Eul rows.
        assert_eq!(shape.mid.force, Vector3::new(9.0, -8.0, 7.0));
        assert_eq!(shape.mid.moment, Vector3::new(3.0, -2.0, 1.0));
        assert_eq!(shape.tip.force, Vector3::new(12.0, -11.0, 10.0));
        assert_eq!(shape.tip.moment, Vector3::new(6.0, -5.0, 4.0));
        // The rod runs along +Z in the file, so along +X once converted.
        assert_eq!(shape.stations.last().unwrap(), &Vector3::new(0.6, 0.0, 0.0));
        assert!((shape.chord() - 0.6).abs() < 1e-6);
    }

    #[test]
    fn interpolation_pairs_by_arc_and_survives_the_repeated_station() {
        let set = Statics::parse(Material::SpringSteel, SAMPLE).unwrap();
        let got = set.interpolate(&set.shapes[0], &[0.0, 0.125, 0.5, 0.875, 1.0]);
        let x: Vec<f32> = got.iter().map(|p| p.x).collect();
        for (got, want) in x.iter().zip([0.0, 0.075, 0.3, 0.525, 0.6]) {
            assert!((got - want).abs() < 1e-5, "{x:?}");
        }
    }

    #[test]
    fn interpolation_clamps_rather_than_extrapolating() {
        let set = Statics::parse(Material::SpringSteel, SAMPLE).unwrap();
        let got = set.interpolate(&set.shapes[0], &[-1.0, 2.0]);
        assert_eq!(got[0], Vector3::ZERO);
        assert_eq!(got[1], Vector3::new(0.6, 0.0, 0.0));
    }

    #[test]
    fn the_materials_match_the_rods_the_reference_used() {
        let steel = Material::SpringSteel.rod(30);
        assert_eq!((steel.length, steel.radius, steel.youngs), (0.6, 0.0008, 200.0e9));
        let tpu = Material::Tpu.rod(30);
        assert_eq!((tpu.length, tpu.radius, tpu.youngs), (0.4, 0.005, 70.0e6));
        // Bare rods: no sheath, so the two radii agree and the comparison does
        // not depend on which one the stiffness is taken from.
        assert_eq!(steel.radius, steel.core_radius);
        assert_eq!(tpu.radius, tpu.core_radius);
    }
}
