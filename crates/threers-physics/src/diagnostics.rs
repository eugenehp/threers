//! What the engine noticed while it was running.
//!
//! A physics engine cannot stop when something is wrong with the scene it was
//! handed. A body whose velocity has gone to `NaN` has to be repaired and
//! stepped anyway, because the alternative — refusing to run — is worse for
//! everyone downstream. The repair is the right behaviour and it is also
//! *invisible*, which is how an afternoon disappears into a scene that behaves
//! oddly for no reason a print statement can find.
//!
//! So the repairs are counted. Nothing here changes what the engine does; it
//! records what it already did, and each record names a decision the engine made
//! on your behalf:
//!
//! ```
//! use threers_physics::prelude::*;
//! use threers_physics::diagnostics::Warning;
//!
//! let mut world = World::new();
//! let body = world.add_body(RigidBody::dynamic().shape(Shape::ball(0.5)));
//!
//! // A velocity written directly, past every guard at the door.
//! world.bodies_mut().get_mut(body).unwrap().linear_velocity = Vector3::new(f32::NAN, 0.0, 0.0);
//! world.step(1.0 / 60.0);
//!
//! assert_eq!(world.diagnostics().count(Warning::NonFiniteVelocity), 1);
//! ```
//!
//! # Timings
//!
//! [`World::set_profiling`](crate::world::World::set_profiling) turns on
//! per-stage timing, which answers the question the total step time cannot: a
//! scene at 8 ms is spending it *somewhere*, and collision detection and the
//! solver want opposite fixes. Off by default, and free when off — the clock is
//! not read at all.
//!
//! Timing needs a clock, and `wasm32` has none in `std`. There the timings stay
//! zero and [`crate::diagnostics::Timings::supported`] returns `false`, so a profiler UI can say so
//! rather than draw a graph of nothing.

use std::time::Duration;

/// Something the engine repaired, dropped or ignored, and carried on through.
///
/// Every one of these is a decision made on your behalf. None of them is fatal
/// and none of them stops the step.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Warning {
    /// A body's velocity was not finite at the start of a step, and was zeroed.
    ///
    /// One `NaN` does not stay in one body: the solver spreads it down every
    /// contact and joint that body touches, and a moment later the scene is
    /// gone. Zeroing is the only recovery that contains it — but the value came
    /// from somewhere, and this is the only sign of where.
    NonFiniteVelocity,
    /// A body was moving faster than
    /// [`World::max_velocity`](crate::world::World::max_velocity) and was slowed
    /// to it.
    ///
    /// The limit exists to catch the values that pass the finiteness check and
    /// still destroy a scene. Hitting it is not itself a bug — a spawn inside
    /// geometry legitimately produces one — but a body that hits it *every*
    /// step is being driven by something that has already gone wrong.
    VelocityClamped,
    /// [`World::step`](crate::world::World::step) was called with a timestep
    /// that is not a positive, finite number. Nothing was simulated.
    InvalidTimestep,
    /// Real time arrived faster than
    /// [`World::max_substeps`](crate::world::World::max_substeps) could consume
    /// it, and the excess was discarded.
    ///
    /// The simulation is running slower than real time and has stopped trying to
    /// catch up — which is deliberate, because the alternative is an accumulator
    /// that grows without bound and makes every later frame worse than the last.
    /// Persistent occurrences mean the scene is too expensive for the budget.
    TimeDiscarded,
    /// A tendon's path point is inside the obstacle it was supposed to wrap
    /// around, so that wrap was ignored and the path left straight.
    ///
    /// Usually a radius larger than the model intended, or an anchor placed at
    /// the centre of the pulley rather than on its rim.
    TendonInsideObstacle,
    /// A tendon path could not be read as written: a wrap that is not bracketed
    /// by two via points, or a pulley with a non-positive divisor. The offending
    /// element was skipped.
    MalformedTendonPath,
    /// A body's *position* was not finite at the start of a step, and it was put
    /// back where the engine last had it.
    ///
    /// Velocity is guarded on the way through the integrator, but a transform
    /// can be written straight into a public field, and a non-finite one is
    /// worse than a non-finite velocity: it is read by the broad phase and the
    /// narrow phase before any of them is asked to be sensible, so it reaches
    /// every body compared against it in the same step. There is no way to
    /// integrate out of it, so the last position the engine produced itself is
    /// restored and the velocities are dropped.
    NonFinitePosition,
}

impl Warning {
    /// Every warning, in declaration order.
    pub const ALL: [Warning; 6] = [
        Warning::NonFiniteVelocity,
        Warning::VelocityClamped,
        Warning::InvalidTimestep,
        Warning::TimeDiscarded,
        Warning::TendonInsideObstacle,
        Warning::MalformedTendonPath,
    ];

    fn index(self) -> usize {
        match self {
            Warning::NonFiniteVelocity => 0,
            Warning::VelocityClamped => 1,
            Warning::InvalidTimestep => 2,
            Warning::TimeDiscarded => 3,
            Warning::TendonInsideObstacle => 4,
            Warning::MalformedTendonPath => 5,
            Warning::NonFinitePosition => 6,
        }
    }

    /// A one-line description, for logging.
    pub fn message(self) -> &'static str {
        match self {
            Warning::NonFiniteVelocity => "a body's velocity was not finite and was zeroed",
            Warning::VelocityClamped => "a body was slowed to max_velocity",
            Warning::InvalidTimestep => "step called with a non-positive or non-finite timestep",
            Warning::TimeDiscarded => "real time arrived faster than max_substeps could consume",
            Warning::TendonInsideObstacle => "a tendon point is inside the obstacle it wraps",
            Warning::MalformedTendonPath => "a tendon path element could not be read and was skipped",
            Warning::NonFinitePosition => "a body's position was not finite and was restored",
        }
    }
}

impl std::fmt::Display for Warning {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.message())
    }
}

/// How often one warning fired, and when it last did.
///
/// The count is of *occurrences*, not of steps: the stages that can raise one
/// run per body and per substep, so a single body clamped through a four-substep
/// step counts four. That is the number worth having — it separates one body
/// misbehaving from the whole scene doing it — and [`WarningRecord::last_step`]
/// is there for "is this still happening".
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct WarningRecord {
    /// Times it has fired since the last [`Diagnostics::clear_warnings`].
    pub count: u64,
    /// The step number it first fired on.
    pub first_step: u64,
    /// The step number it last fired on. Equal to the current step means it is
    /// still happening, which is the difference between a transient and a
    /// scene that is broken right now.
    pub last_step: u64,
}

impl WarningRecord {
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }
}

/// Where a step's time went.
///
/// All zero unless [`World::set_profiling`](crate::world::World::set_profiling)
/// is on, and all zero on `wasm32`, which has no clock in `std` — see
/// [`crate::diagnostics::Timings::supported`].
///
/// The stages sum to slightly less than [`Timings::step`]: what is left is the
/// bookkeeping between them, which is not worth its own clock read.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Timings {
    /// One whole call to [`World::step_fixed`](crate::world::World::step_fixed).
    pub step: Duration,
    /// Finding candidate pairs from the spatial index.
    pub broad_phase: Duration,
    /// Turning those pairs into contact manifolds.
    pub narrow_phase: Duration,
    /// Splitting the constraint set into independent islands.
    pub islands: Duration,
    /// Every substep of the constraint solve. Normally the largest.
    pub solver: Duration,
    /// Integrating velocities and positions, over every substep.
    pub integration: Duration,
    /// Continuous collision: sweeping the bodies that outran their own size.
    pub continuous: Duration,
    /// Deciding what falls asleep.
    pub sleeping: Duration,
    /// Rebuilding the spatial index so queries see where things ended up.
    pub index: Duration,
}

impl Timings {
    /// Whether this platform has a clock the engine can read.
    ///
    /// `false` on `wasm32`, where every field stays zero however the profiling
    /// flag is set.
    pub const fn supported() -> bool {
        !cfg!(target_arch = "wasm32")
    }

    /// The stages, in pipeline order, with their names — for a profiler bar.
    pub fn stages(&self) -> [(&'static str, Duration); 8] {
        [
            ("broad phase", self.broad_phase),
            ("narrow phase", self.narrow_phase),
            ("islands", self.islands),
            ("solver", self.solver),
            ("integration", self.integration),
            ("continuous", self.continuous),
            ("sleeping", self.sleeping),
            ("index", self.index),
        ]
    }

    /// What the named stages add up to. Less than [`Self::step`].
    pub fn accounted(&self) -> Duration {
        self.stages().iter().map(|(_, d)| *d).sum()
    }
}

/// How much of everything the last step had.
///
/// Free — these are counts the step already knows. Useful mostly as the x-axis
/// of a performance problem: 8 ms of solver is expected with 4000 contacts and
/// suspicious with 40.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Counters {
    /// Every body, asleep or awake.
    pub bodies: usize,
    /// Bodies that are dynamic, enabled and not asleep — what the step pays for.
    pub awake_bodies: usize,
    /// Contact manifolds after the narrow phase.
    pub manifolds: usize,
    /// Individual contact points across those manifolds — the solver's real
    /// workload, since each is a row.
    pub contact_points: usize,
    /// Islands the constraint set split into. One large island is the case
    /// threading cannot help.
    pub islands: usize,
    /// Substeps the last step ran.
    pub substeps: u32,
}

/// The engine's own account of the last step.
///
/// Read it from [`World::diagnostics`](crate::world::World::diagnostics).
#[derive(Debug, Clone, Default)]
pub struct Diagnostics {
    records: [WarningRecord; 7],
    timings: Timings,
    counters: Counters,
    profiling: bool,
    steps: u64,
}

impl Diagnostics {
    pub fn new() -> Self {
        Self::default()
    }

    /// How many steps have run since the world was made.
    ///
    /// This is the number [`WarningRecord::first_step`] and
    /// [`WarningRecord::last_step`] are measured in.
    pub fn steps(&self) -> u64 {
        self.steps
    }

    /// The full record for one warning.
    pub fn warning(&self, warning: Warning) -> WarningRecord {
        self.records[warning.index()]
    }

    /// How many times a warning has fired.
    pub fn count(&self, warning: Warning) -> u64 {
        self.records[warning.index()].count
    }

    /// Whether anything at all has been recorded.
    pub fn is_clean(&self) -> bool {
        self.records.iter().all(WarningRecord::is_empty)
    }

    /// Every warning that has fired at least once, with its record.
    pub fn warnings(&self) -> impl Iterator<Item = (Warning, WarningRecord)> + '_ {
        Warning::ALL
            .into_iter()
            .map(|w| (w, self.warning(w)))
            .filter(|(_, r)| !r.is_empty())
    }

    /// Warnings that fired during the most recent step, rather than at any point
    /// in the run.
    pub fn current(&self) -> impl Iterator<Item = (Warning, WarningRecord)> + '_ {
        let now = self.steps;
        self.warnings()
            .filter(move |(_, r)| now > 0 && r.last_step == now - 1)
    }

    /// Forget every warning. The step counter keeps running.
    pub fn clear_warnings(&mut self) {
        self.records = Default::default();
    }

    /// Where the last step's time went. All zero unless profiling is on.
    pub fn timings(&self) -> Timings {
        self.timings
    }

    /// How much of everything the last step had.
    pub fn counters(&self) -> Counters {
        self.counters
    }

    /// Whether per-stage timing is being collected.
    pub fn profiling(&self) -> bool {
        self.profiling
    }

    // ---- recording, from the world ----------------------------------------

    pub(crate) fn set_profiling(&mut self, on: bool) {
        self.profiling = on && Timings::supported();
        if !self.profiling {
            self.timings = Timings::default();
        }
    }

    /// Record `n` occurrences of a warning against the current step.
    pub(crate) fn warn(&mut self, warning: Warning, n: u64) {
        if n == 0 {
            return;
        }
        let step = self.steps;
        let record = &mut self.records[warning.index()];
        if record.count == 0 {
            record.first_step = step;
        }
        record.count += n;
        record.last_step = step;
    }

    pub(crate) fn end_step(&mut self, timings: Timings, counters: Counters) {
        if self.profiling {
            self.timings = timings;
        }
        self.counters = counters;
        self.steps += 1;
    }

    pub(crate) fn stopwatch(&self) -> Stopwatch {
        Stopwatch::start(self.profiling)
    }
}

/// A clock read that costs nothing when profiling is off, and does not exist on
/// platforms without a clock.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct Stopwatch {
    #[cfg(not(target_arch = "wasm32"))]
    start: Option<std::time::Instant>,
}

impl Stopwatch {
    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn start(on: bool) -> Self {
        Self {
            start: on.then(std::time::Instant::now),
        }
    }

    #[cfg(target_arch = "wasm32")]
    pub(crate) fn start(_on: bool) -> Self {
        Self {}
    }

    /// Time since [`Self::start`], or zero if it was never started.
    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn read(self) -> Duration {
        self.start.map_or(Duration::ZERO, |t| t.elapsed())
    }

    #[cfg(target_arch = "wasm32")]
    pub(crate) fn read(self) -> Duration {
        Duration::ZERO
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_warning_records_when_it_first_and_last_fired() {
        let mut d = Diagnostics::new();
        d.end_step(Timings::default(), Counters::default()); // step 0 done
        d.warn(Warning::VelocityClamped, 3);
        d.end_step(Timings::default(), Counters::default());
        d.end_step(Timings::default(), Counters::default());
        d.warn(Warning::VelocityClamped, 1);

        let r = d.warning(Warning::VelocityClamped);
        assert_eq!((r.count, r.first_step, r.last_step), (4, 1, 3));
        assert!(!d.is_clean());
        assert_eq!(d.warnings().count(), 1);
    }

    #[test]
    fn nothing_is_recorded_for_a_count_of_zero() {
        let mut d = Diagnostics::new();
        d.warn(Warning::NonFiniteVelocity, 0);
        assert!(d.is_clean());
        assert_eq!(d.warnings().count(), 0);
    }

    #[test]
    fn current_reports_only_the_step_that_just_ran() {
        let mut d = Diagnostics::new();
        d.warn(Warning::TimeDiscarded, 1);
        d.end_step(Timings::default(), Counters::default());
        assert_eq!(d.current().count(), 1, "fired on the step that just ended");

        d.end_step(Timings::default(), Counters::default());
        assert_eq!(d.current().count(), 0, "a step later it is history");
        assert_eq!(d.count(Warning::TimeDiscarded), 1, "but still counted");
    }

    #[test]
    fn profiling_is_refused_where_there_is_no_clock() {
        let mut d = Diagnostics::new();
        d.set_profiling(true);
        assert_eq!(d.profiling(), Timings::supported());
    }
}
