//! Animating and rendering OpenSCAD models.
//!
//! OpenSCAD animates by re-evaluating the whole program once per frame with
//! `$t` stepped from 0 to 1 — geometry is a function of time. This module turns
//! that into frames you can look at: it drives the evaluation, keeps the parts
//! [`color()`](crate::openscad::Solid::color) gave them, frames a camera on the
//! model, and hands the result to the renderer or to video export.
//!
//! ```no_run
//! use threers::openscad::animate::{ScadAnimation, ScadCamera, ScadRender};
//! // `export_video` needs the `video` feature; hidden behind a `cfg` so this
//! // compiles as documentation whether or not the feature is on.
//! # #[cfg(feature = "video")]
//! # fn demo() {
//! // A model that opens over one loop of $t.
//! let mut animation = ScadAnimation::from_file("hinge.scad").frames(120).fps(30);
//! ScadRender::new(1280, 720)
//!     .camera(ScadCamera::turntable())
//!     .export_video(&mut animation, "hinge.mp4")
//!     .unwrap();
//! # }
//! ```
//!
//! | Piece | Role |
//! |-------|------|
//! | [`crate::openscad::animate::ScadAnimation`] | what to evaluate, how many frames, at what rate |
//! | [`crate::openscad::animate::ScadFrame`] | one evaluated frame: colored parts plus its bounds |
//! | [`crate::openscad::animate::ScadCamera`] | how to frame the model — auto fit, turntable, `$vp*`, or fixed |
//! | [`crate::openscad::animate::ScadRender`] | size, look, and the render/export entry points |
//!
//! **Cost.** Every frame is a fresh evaluation of the whole program, including
//! the CSG. That is what makes SCAD animation expensive, so this module spends
//! its effort avoiding the work:
//!
//! - A model that never reads `$t` or a seeded variable is evaluated **once**
//!   and shared by every frame ([`crate::openscad::animate::ScadAnimation::is_animated`]).
//! - Frames that do differ are evaluated several at a time
//!   ([`crate::openscad::animate::ScadAnimation::concurrency`]) — about 3× on the reference model.
//! - `color()` splits the model into parts, and a cutter that cannot reach a
//!   part is dropped from it, so a boolean is only paid for where it applies.

use std::path::PathBuf;
use std::sync::Arc;

use crate::cameras::PerspectiveCamera;
use crate::core::{Mesh, Object3D};
use crate::lights::{AmbientLight, DirectionalLight, HemisphereLight};
use crate::materials::{Material, StandardMaterial};
use crate::math::{Color, Vector3};
use crate::openscad::{scad, ScadPart, Solid};
use crate::scene::Scene;

// ---------------------------------------------------------------------------
// Source
// ---------------------------------------------------------------------------

/// Where an animation's geometry comes from.
enum Source {
    /// OpenSCAD source text, re-evaluated per frame.
    Text(String),
    /// A `.scad` file, re-read and re-evaluated per frame.
    File(PathBuf),
    /// A Rust closure of time — the `Solid`/`scad!` DSL equivalent of `$t`.
    Builder(Arc<dyn Fn(f64) -> Solid + Send + Sync>),
}

// A frame is only data, so it lives where wasm32 can reach it — the renderer
// below is what cannot go there. Re-exported so `animate::ScadFrame` still
// names the thing every caller already imports.
pub use crate::openscad::frame::ScadFrame;

/// The speed profile a loop follows.
///
/// This shapes how `$t` advances, so it changes how fast the *model* moves at
/// each point in the loop — as distinct from [`ScadAnimation::speed`], which
/// changes how fast the whole thing plays back.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum ScadEasing {
    /// Constant rate — OpenSCAD's own behaviour.
    #[default]
    Linear,
    /// Starts slow, ends fast.
    EaseIn,
    /// Starts fast, ends slow.
    EaseOut,
    /// Slow at both ends, quick through the middle. Reads as a mechanism
    /// accelerating away from rest and settling at the other end.
    EaseInOut,
    /// A full cosine ease that starts *and* finishes at rest at the same place
    /// — one smooth out-and-back per loop.
    Sine,
}

impl ScadEasing {
    /// Remap raw progress `p` (`0..=1`) through the curve.
    ///
    /// ```
    /// use threers::openscad::animate::ScadEasing;
    /// // Every curve pins the ends and stays inside the range.
    /// for e in [ScadEasing::Linear, ScadEasing::EaseIn, ScadEasing::EaseOut,
    ///           ScadEasing::EaseInOut] {
    ///     assert_eq!(e.apply(0.0), 0.0);
    ///     assert!((e.apply(1.0) - 1.0).abs() < 1e-9);
    /// }
    /// // Ease-in lags a linear ramp; ease-out leads it.
    /// assert!(ScadEasing::EaseIn.apply(0.5) < 0.5);
    /// assert!(ScadEasing::EaseOut.apply(0.5) > 0.5);
    /// ```
    pub fn apply(self, p: f64) -> f64 {
        let p = p.clamp(0.0, 1.0);
        match self {
            ScadEasing::Linear => p,
            ScadEasing::EaseIn => p * p,
            ScadEasing::EaseOut => 1.0 - (1.0 - p) * (1.0 - p),
            ScadEasing::EaseInOut => {
                if p < 0.5 {
                    2.0 * p * p
                } else {
                    1.0 - 2.0 * (1.0 - p) * (1.0 - p)
                }
            }
            // Out and back in one curve, at rest at both ends.
            ScadEasing::Sine => (1.0 - (p * std::f64::consts::TAU).cos()) * 0.5,
        }
    }
}

/// Which boolean kernel evaluates each frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum ScadKernel {
    /// The robust arrangement kernel with a float fallback — matches
    /// [`Solid::to_geometry_exact`]. Correct, and the slower of the two.
    #[default]
    Exact,
    /// The float `CsgEvaluator` — matches [`Solid::to_geometry`]. Faster, and
    /// usually indistinguishable once rendered, which is what a preview wants.
    ///
    /// The float kernel can fail on hard geometry and return nothing at all;
    /// when a frame comes back empty it is re-evaluated with
    /// [`Exact`](Self::Exact) rather than rendered blank.
    Float,
}

/// Progress while evaluating an animation's frames.
#[derive(Clone, Debug)]
pub struct ScadProgress {
    /// Frames finished so far.
    pub done: usize,
    /// Frames in total.
    pub total: usize,
    /// Ready-to-print status line.
    pub message: String,
}

type ProgressFn = Box<dyn FnMut(&ScadProgress) + Send>;

/// Build the progress record for `done` of `total` frames.
fn progress_tick(done: usize, total: usize) -> ScadProgress {
    let pct = (done * 100).checked_div(total).unwrap_or(100);
    ScadProgress {
        done,
        total,
        message: format!("Evaluating {done}/{total} ({pct}%)"),
    }
}

/// A SCAD model over time.
///
/// ```no_run
/// use threers::openscad::animate::ScadAnimation;
/// let animation = ScadAnimation::from_source("rotate([0, 0, 360 * $t]) cube(10);")
///     .frames(60)
///     .fps(30);
/// assert_eq!(animation.duration(), 2.0);
/// let frame = animation.frame(15).unwrap();   // $t = 0.25
/// assert!(frame.triangle_count() > 0);
/// ```
pub struct ScadAnimation {
    source: Source,
    frames: usize,
    fps: u32,
    kernel: ScadKernel,
    /// Extra root-scope variables, each a function of `t`.
    #[allow(clippy::type_complexity)]
    vars: Vec<(String, Arc<dyn Fn(f64) -> f64 + Send + Sync>)>,
    /// `true` → `$t` sweeps `[0, 1)` so the loop is seamless (OpenSCAD's rule).
    looping: bool,
    /// Root-scope values that do NOT vary with the frame, so they play no part
    /// in deciding whether the model is animated.
    constants: Vec<(String, f64)>,
    /// Playback multiplier applied to the output frame rate.
    speed: f64,
    /// When set, the loop is retimed to last this many seconds.
    target_seconds: Option<f64>,
    /// `$t` runs forward then back over the loop.
    ping_pong: bool,
    /// Speed profile through the loop.
    easing: ScadEasing,
    /// Frames evaluated at once; `None` picks a default from the core count.
    concurrency: Option<usize>,
    on_progress: Option<ProgressFn>,
}

impl std::fmt::Debug for ScadAnimation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ScadAnimation")
            .field(
                "source",
                &match &self.source {
                    Source::Text(_) => "text",
                    Source::File(p) => return f.write_fmt(format_args!("ScadAnimation({p:?})")),
                    Source::Builder(_) => "builder",
                },
            )
            .field("frames", &self.frames)
            .field("fps", &self.fps)
            .field("kernel", &self.kernel)
            .finish()
    }
}

impl ScadAnimation {
    fn with_source(source: Source) -> Self {
        Self {
            source,
            frames: 60,
            fps: 30,
            kernel: ScadKernel::default(),
            vars: Vec::new(),
            looping: true,
            constants: Vec::new(),
            speed: 1.0,
            target_seconds: None,
            ping_pong: false,
            easing: ScadEasing::Linear,
            concurrency: None,
            on_progress: None,
        }
    }

    /// Animate OpenSCAD source text.
    pub fn from_source(src: impl Into<String>) -> Self {
        Self::with_source(Source::Text(src.into()))
    }

    /// Animate a `.scad` file. `include`/`use`/`import` resolve against its
    /// folder, and the file is re-read each frame so edits show up.
    pub fn from_file(path: impl Into<PathBuf>) -> Self {
        Self::with_source(Source::File(path.into()))
    }

    /// Animate a Rust closure — the [`Solid`] DSL's answer to `$t`.
    ///
    /// ```
    /// use threers::openscad::animate::ScadAnimation;
    /// use threers::{cube, sphere};
    /// let animation = ScadAnimation::from_fn(|t| {
    ///     cube([10.0, 10.0, 10.0]).difference(sphere(4.0 + 3.0 * t as f32))
    /// })
    /// .frames(4);
    /// assert_eq!(animation.frame_count(), 4);
    /// ```
    pub fn from_fn(f: impl Fn(f64) -> Solid + Send + Sync + 'static) -> Self {
        Self::with_source(Source::Builder(Arc::new(f)))
    }

    /// Number of frames in one loop (minimum 1).
    pub fn frames(mut self, frames: usize) -> Self {
        self.frames = frames.max(1);
        self
    }

    /// Playback rate (minimum 1).
    pub fn fps(mut self, fps: u32) -> Self {
        self.fps = fps.max(1);
        self
    }

    /// Which boolean kernel to evaluate with.
    pub fn kernel(mut self, kernel: ScadKernel) -> Self {
        self.kernel = kernel;
        self
    }

    /// Seed an extra root-scope variable that varies with the frame — the way
    /// to drive a model parameterised on `DEPLOY` or `ANGLE` rather than `$t`.
    ///
    /// ```
    /// use threers::openscad::animate::ScadAnimation;
    /// let animation = ScadAnimation::from_source("cube([10, 10, HEIGHT]);")
    ///     .frames(10)
    ///     .var("HEIGHT", |t| 1.0 + 20.0 * t);
    /// assert!(animation.frame(9).unwrap().bounds().unwrap().1[2] > 10.0);
    /// ```
    pub fn var(
        mut self,
        name: impl Into<String>,
        value: impl Fn(f64) -> f64 + Send + Sync + 'static,
    ) -> Self {
        self.vars.push((name.into(), Arc::new(value)));
        self
    }

    /// Seed a root-scope value that does not change over the animation.
    ///
    /// Unlike [`var`](Self::var) this does not make the model count as animated
    /// — a constant is just a parameter, so the model is still evaluated once
    /// and shared if nothing else varies.
    ///
    /// ```
    /// use threers::openscad::animate::ScadAnimation;
    /// let a = ScadAnimation::from_source("cube([10, 10, H]);")
    ///     .frames(5)
    ///     .constant("H", 3.0);
    /// assert!(!a.is_animated());
    /// assert_eq!(a.frame(0).unwrap().bounds().unwrap().1[2], 3.0);
    /// ```
    pub fn constant(mut self, name: impl Into<String>, value: f64) -> Self {
        self.constants.push((name.into(), value));
        self
    }

    /// Whether `$t` sweeps `[0, 1)` (default, seamless loop) or `[0, 1]`.
    ///
    /// OpenSCAD uses `$t = frame / frames`, so the last frame stops just short
    /// of 1 and the animation joins back to frame 0 cleanly. Turn this off for
    /// a one-shot sweep that should actually reach `t = 1`.
    pub fn looping(mut self, looping: bool) -> Self {
        self.looping = looping;
        self
    }

    /// Play the loop faster or slower without re-evaluating anything.
    ///
    /// A multiplier on the output frame rate: `2.0` runs the same frames twice
    /// as fast, `0.5` at half speed. The motion is unchanged — only how long it
    /// takes. To make the model itself move differently, use
    /// [`easing`](Self::easing) or [`ping_pong`](Self::ping_pong).
    ///
    /// ```
    /// use threers::openscad::animate::ScadAnimation;
    /// let a = ScadAnimation::from_source("cube(1);").frames(60).fps(30);
    /// assert_eq!(a.duration(), 2.0);
    /// let fast = ScadAnimation::from_source("cube(1);").frames(60).fps(30).speed(2.0);
    /// assert_eq!(fast.frame_rate(), 60);
    /// assert_eq!(fast.duration(), 1.0);
    /// ```
    pub fn speed(mut self, multiplier: f64) -> Self {
        self.speed = if multiplier.is_finite() && multiplier > 0.0 {
            multiplier
        } else {
            1.0
        };
        self
    }

    /// Retime the loop to last `seconds`, whatever the frame count.
    ///
    /// Order-independent: the frame rate is derived when it is asked for, so
    /// this and [`frames`](Self::frames) can be called in either order.
    /// [`speed`](Self::speed) still applies on top.
    ///
    /// ```
    /// use threers::openscad::animate::ScadAnimation;
    /// let a = ScadAnimation::from_source("cube(1);").seconds(4.0).frames(120);
    /// assert_eq!(a.frame_rate(), 30);
    /// assert_eq!(a.duration(), 4.0);
    /// ```
    pub fn seconds(mut self, seconds: f64) -> Self {
        self.target_seconds = (seconds.is_finite() && seconds > 0.0).then_some(seconds);
        self
    }

    /// Run `$t` forward then back over the loop, instead of wrapping.
    ///
    /// What an open-and-close motion usually wants: the model returns the way
    /// it came rather than snapping back at the loop point.
    ///
    /// ```
    /// use threers::openscad::animate::ScadAnimation;
    /// let a = ScadAnimation::from_source("cube(1);").frames(4).ping_pong(true);
    /// assert_eq!(a.t_at(0), 0.0);
    /// assert_eq!(a.t_at(2), 1.0);   // halfway through the loop is the far end
    /// assert_eq!(a.t_at(3), 0.5);
    /// ```
    pub fn ping_pong(mut self, ping_pong: bool) -> Self {
        self.ping_pong = ping_pong;
        self
    }

    /// Speed profile through the loop — see [`ScadEasing`].
    ///
    /// This changes how fast the *model* moves at each point, unlike
    /// [`speed`](Self::speed), which changes how fast the whole thing plays.
    pub fn easing(mut self, easing: ScadEasing) -> Self {
        self.easing = easing;
        self
    }

    /// Trade evaluation time for fidelity — see [`ScadQuality`]. Sets the
    /// boolean kernel, and the curve resolution for models that do not pin
    /// their own `$fn`/`$fa`/`$fs`.
    ///
    /// ```
    /// use threers::openscad::animate::{ScadAnimation, ScadQuality};
    /// // A model with no `$fn` of its own gets coarser curves in Draft.
    /// let draft = ScadAnimation::from_source("sphere(20);").frames(1).quality(ScadQuality::Draft);
    /// let fine = ScadAnimation::from_source("sphere(20);").frames(1).quality(ScadQuality::Fine);
    /// assert!(draft.frame(0).unwrap().triangle_count() < fine.frame(0).unwrap().triangle_count());
    /// ```
    pub fn quality(mut self, quality: ScadQuality) -> Self {
        self.kernel = quality.kernel();
        // Seeded as defaults, so a model that sets `$fn`/`$fa`/`$fs` itself
        // still wins — the same precedence OpenSCAD applies.
        self.constants.retain(|(n, _)| n != "$fa" && n != "$fs");
        if let Some((fa, fs)) = quality.facets() {
            self.constants.push(("$fa".into(), fa));
            self.constants.push(("$fs".into(), fs));
        }
        self
    }

    /// Report evaluation progress.
    pub fn on_progress(mut self, cb: impl FnMut(&ScadProgress) + Send + 'static) -> Self {
        self.on_progress = Some(Box::new(cb));
        self
    }

    /// How many frames to evaluate at once. `1` evaluates them one by one.
    ///
    /// Frames are independent, so this is close to a linear speedup on the part
    /// of the job that dominates — evaluation, which is about 99% of it; the
    /// GPU render is under two milliseconds a frame. The default is one per
    /// core. Drop it for models where the transient working set of a single
    /// CSG is large enough to matter.
    ///
    /// The `parallel` feature does not help here and costs about a quarter:
    /// measured on the reference model, four frames at once went from 188 ms a
    /// frame to 235 ms with it on, because a frame's own CSG is too
    /// fine-grained to pay for the split and the two levels of threading then
    /// fight for the same cores. Frame-level concurrency is the one that pays.
    ///
    /// ```
    /// use threers::openscad::animate::ScadAnimation;
    /// let a = ScadAnimation::from_source("cube(1);").frames(8).concurrency(1);
    /// assert_eq!(a.frame_count(), 8);
    /// ```
    pub fn concurrency(mut self, frames_at_once: usize) -> Self {
        self.concurrency = Some(frames_at_once.max(1));
        self
    }

    /// Frame count.
    pub fn frame_count(&self) -> usize {
        self.frames
    }

    /// Playback rate, after [`seconds`](Self::seconds) and
    /// [`speed`](Self::speed) are applied.
    pub fn frame_rate(&self) -> u32 {
        let base = match self.target_seconds {
            Some(seconds) => self.frames as f64 / seconds,
            None => self.fps as f64,
        };
        (base * self.speed).round().clamp(1.0, u32::MAX as f64) as u32
    }

    /// Wall-clock length of one loop, in seconds.
    pub fn duration(&self) -> f64 {
        self.frames as f64 / self.frame_rate() as f64
    }

    /// The animation variable for frame `index`.
    ///
    /// ```
    /// use threers::openscad::animate::ScadAnimation;
    /// let a = ScadAnimation::from_source("cube(1);").frames(4);
    /// assert_eq!(a.t_at(0), 0.0);
    /// assert_eq!(a.t_at(3), 0.75);          // [0, 1) — frame 4 would be the loop point
    /// let once = a.looping(false);
    /// assert_eq!(once.t_at(3), 1.0);        // [0, 1]
    /// ```
    pub fn t_at(&self, index: usize) -> f64 {
        self.eval().t_at(index)
    }

    /// Whether the model's geometry can change over the animation.
    ///
    /// A source that never mentions `$t` or a seeded variable is evaluated once
    /// and shared by every frame. A closure source is always treated as
    /// animated, since there is nothing to inspect.
    pub fn is_animated(&self) -> bool {
        let text = match &self.source {
            Source::Builder(_) => return true,
            Source::Text(s) => std::borrow::Cow::Borrowed(s.as_str()),
            Source::File(p) => match std::fs::read_to_string(p) {
                Ok(s) => std::borrow::Cow::Owned(s),
                // Unreadable: assume animated and let evaluation report the error.
                Err(_) => return true,
            },
        };
        // `include`d files could reference `$t` too, so a hit anywhere in the
        // top-level text is treated as animated and a miss checks the includes.
        if mentions_any(&text, &self.var_names()) {
            return true;
        }
        text.contains("include") || text.contains("use <")
    }

    fn var_names(&self) -> Vec<String> {
        let mut names = vec!["$t".to_string()];
        names.extend(self.vars.iter().map(|(n, _)| n.clone()));
        names
    }

    /// Frames evaluated at once, resolving the default when unset.
    ///
    /// One per core. This used to cap at four on the grounds that each
    /// concurrent frame holds a full evaluation's worth of geometry — but
    /// `evaluate` returns *every* frame, so all of that geometry is live at the
    /// end whatever the concurrency; what scales with it is the transient
    /// working set of a CSG, not the results. The cap was leaving most of a
    /// modern machine idle: on a fourteen-core box the reference model went
    /// from 188 ms a frame at four to 118 ms at twelve.
    fn resolved_concurrency(&self) -> usize {
        self.concurrency.unwrap_or_else(|| {
            std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1)
        })
    }

    /// The parts of `self` that evaluation actually needs — all `Sync`, unlike
    /// the progress callback, so worker threads can share them.
    fn eval(&self) -> Eval<'_> {
        Eval {
            source: &self.source,
            vars: &self.vars,
            constants: &self.constants,
            kernel: self.kernel,
            frames: self.frames,
            looping: self.looping,
            ping_pong: self.ping_pong,
            easing: self.easing,
        }
    }

    /// Evaluate one frame.
    ///
    /// Each call is a full evaluation; use [`evaluate`](Self::evaluate) to do
    /// the whole animation at once (and in parallel where available).
    pub fn frame(&self, index: usize) -> Result<ScadFrame, String> {
        self.eval().frame(index)
    }

    /// The viewport the model asks for at time `t`.
    pub fn viewport_at(&self, t: f64) -> scad::Viewport {
        self.eval().viewport_at(t)
    }

    /// Evaluate every frame.
    ///
    /// A static model is evaluated once and the result shared, so the returned
    /// frames alias one allocation. With the `parallel` feature the frames of
    /// an animated model are evaluated across all cores.
    pub fn evaluate(&mut self) -> Result<Vec<ScadFrame>, String> {
        let total = self.frames;
        if !self.is_animated() {
            let eval = self.eval();
            let parts = Arc::new(eval.parts_at(eval.t_at(0))?);
            let frames: Vec<ScadFrame> = (0..total)
                .map(|index| {
                    let t = eval.t_at(index);
                    ScadFrame {
                        index,
                        t,
                        parts: parts.clone(),
                        viewport: eval.viewport_at(t),
                    }
                })
                .collect();
            self.report(total, total);
            return Ok(frames);
        }

        // `evaluate_all_frames` reports each frame as it lands, including the
        // last, so there is nothing left to announce here.
        self.evaluate_all_frames()
    }

    /// Evaluate the frames across [`concurrency`](Self::concurrency) threads.
    fn evaluate_all_frames(&mut self) -> Result<Vec<ScadFrame>, String> {
        let total = self.frames;
        let workers = self.resolved_concurrency().min(total).max(1);
        // Lift the callback out so the workers can borrow `self`'s evaluation
        // state immutably while progress is still reported.
        let mut progress = self.on_progress.take();
        let outcome = evaluate_frames(self.eval(), total, workers, &mut progress);
        self.on_progress = progress;
        outcome
    }

    fn report(&mut self, done: usize, total: usize) {
        if let Some(cb) = self.on_progress.as_mut() {
            cb(&progress_tick(done, total));
        }
    }
}

/// A borrowed, thread-shareable view of everything frame evaluation needs.
struct Eval<'a> {
    source: &'a Source,
    #[allow(clippy::type_complexity)]
    vars: &'a [(String, Arc<dyn Fn(f64) -> f64 + Send + Sync>)],
    constants: &'a [(String, f64)],
    kernel: ScadKernel,
    frames: usize,
    looping: bool,
    ping_pong: bool,
    easing: ScadEasing,
}

impl Eval<'_> {
    fn t_at(&self, index: usize) -> f64 {
        let index = index.min(self.frames.saturating_sub(1));
        // Raw progress through the loop.
        let p = if self.looping {
            index as f64 / self.frames as f64
        } else if self.frames <= 1 {
            0.0
        } else {
            index as f64 / (self.frames - 1) as f64
        };
        // Out and back, so the motion reverses instead of snapping.
        let p = if self.ping_pong {
            1.0 - (2.0 * p - 1.0).abs()
        } else {
            p
        };
        self.easing.apply(p)
    }

    fn seeds(&self, t: f64) -> Vec<(String, f64)> {
        let mut seeds = vec![("$t".to_string(), t)];
        // Constants first: an animated `var` of the same name should win.
        seeds.extend(self.constants.iter().cloned());
        seeds.extend(self.vars.iter().map(|(n, f)| (n.clone(), f(t))));
        seeds
    }

    fn viewport_at(&self, t: f64) -> scad::Viewport {
        match self.source {
            Source::Text(s) => scad::scad_viewport(s, t),
            Source::File(p) => scad::scad_viewport_file(p, t),
            Source::Builder(_) => scad::Viewport::default(),
        }
    }

    fn parts_at(&self, t: f64) -> Result<Vec<ScadPart>, String> {
        let solid = match self.source {
            Source::Builder(f) => f(t),
            Source::Text(src) => {
                let seeds = self.seeds(t);
                let refs: Vec<(&str, f64)> = seeds.iter().map(|(k, v)| (k.as_str(), *v)).collect();
                scad::parse_scad_with(src, &refs)?
            }
            Source::File(path) => {
                let seeds = self.seeds(t);
                let refs: Vec<(&str, f64)> = seeds.iter().map(|(k, v)| (k.as_str(), *v)).collect();
                scad::parse_scad_file_with(path, &refs)?
            }
        };
        Ok(match self.kernel {
            ScadKernel::Exact => solid.parts(),
            ScadKernel::Float => {
                // The float kernel can fail outright on hard geometry and hand
                // back nothing. Rendering an empty frame would be a silent
                // wrong answer, so fall back to the robust kernel instead —
                // slower, but it produces the model that was asked for.
                let parts = solid.clone().parts_float();
                if parts.is_empty() {
                    solid.parts()
                } else {
                    parts
                }
            }
        })
    }

    fn frame(&self, index: usize) -> Result<ScadFrame, String> {
        let t = self.t_at(index);
        Ok(ScadFrame {
            index,
            t,
            parts: Arc::new(self.parts_at(t)?),
            viewport: self.viewport_at(t),
        })
    }
}

/// Evaluate `total` frames using `workers` threads, reporting as each lands.
///
/// Deliberately plain `std` threads rather than rayon: a frame's evaluation
/// already runs on its own worker thread and, with the `parallel` feature, uses
/// rayon inside that. Driving the frames from rayon too would block every
/// worker in the global pool on a thread that then needs the same pool — which
/// deadlocks as soon as the frame count passes the core count.
fn evaluate_frames(
    eval: Eval<'_>,
    total: usize,
    workers: usize,
    progress: &mut Option<ProgressFn>,
) -> Result<Vec<ScadFrame>, String> {
    use std::sync::atomic::{AtomicUsize, Ordering};

    let mut report = |done: usize| {
        if let Some(cb) = progress.as_mut() {
            cb(&progress_tick(done, total));
        }
    };

    if workers <= 1 {
        let mut out = Vec::with_capacity(total);
        for index in 0..total {
            out.push(eval.frame(index)?);
            report(index + 1);
        }
        return Ok(out);
    }

    let next = AtomicUsize::new(0);
    let (tx, rx) = std::sync::mpsc::channel::<(usize, Result<ScadFrame, String>)>();
    let mut slots: Vec<Option<Result<ScadFrame, String>>> = (0..total).map(|_| None).collect();

    std::thread::scope(|scope| {
        for _ in 0..workers {
            let tx = tx.clone();
            let next = &next;
            let eval = &eval;
            scope.spawn(move || loop {
                let index = next.fetch_add(1, Ordering::Relaxed);
                if index >= total {
                    return;
                }
                if tx.send((index, eval.frame(index))).is_err() {
                    return;
                }
            });
        }
        // Dropping this end lets `rx` finish once every worker has exited.
        drop(tx);
        // The main thread owns the progress callback, so it does the draining.
        let mut done = 0usize;
        while let Ok((index, result)) = rx.recv() {
            slots[index] = Some(result);
            done += 1;
            report(done);
        }
    });

    slots
        .into_iter()
        .map(|slot| slot.unwrap_or_else(|| Err("frame was never evaluated".into())))
        .collect()
}

/// Whether `text` uses any of `names` as an identifier (not inside a longer one).
fn mentions_any(text: &str, names: &[String]) -> bool {
    names.iter().any(|name| mentions(text, name))
}

/// Whole-identifier search: `HEIGHT` must not match `MAX_HEIGHTS`.
fn mentions(text: &str, name: &str) -> bool {
    let is_ident = |c: char| c.is_alphanumeric() || c == '_' || c == '$';
    let mut from = 0;
    while let Some(hit) = text[from..].find(name) {
        let at = from + hit;
        let before_ok = text[..at].chars().next_back().is_none_or(|c| !is_ident(c));
        let after_ok = text[at + name.len()..]
            .chars()
            .next()
            .is_none_or(|c| !is_ident(c));
        if before_ok && after_ok {
            return true;
        }
        from = at + name.len().max(1);
    }
    false
}

// ---------------------------------------------------------------------------
// Camera
// ---------------------------------------------------------------------------

/// How to frame the model.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ScadCamera {
    /// Fit the model in view from a fixed direction, as a CAD "home" view does.
    ///
    /// `yaw` and `pitch` are degrees; `zoom` scales the fitted distance
    /// (below 1 moves closer).
    Auto { yaw: f32, pitch: f32, zoom: f32 },
    /// Fit the model and orbit it over the animation — a turntable.
    ///
    /// `turns` is how many full revolutions one loop makes.
    Turntable {
        pitch: f32,
        turns: f32,
        zoom: f32,
        /// Yaw at `t = 0`, in degrees.
        start_yaw: f32,
    },
    /// Obey the model's own `$vpr` / `$vpt` / `$vpd` / `$vpf`, the way the
    /// OpenSCAD GUI does — including when they are written as functions of
    /// `$t`. Falls back to [`Auto`](Self::Auto) for a model that sets none.
    Viewport,
    /// An explicit eye and target in model space.
    Fixed {
        eye: [f32; 3],
        target: [f32; 3],
        fov: f32,
    },
}

impl Default for ScadCamera {
    fn default() -> Self {
        Self::auto()
    }
}

impl ScadCamera {
    /// The default three-quarter view, fitted to the model.
    pub fn auto() -> Self {
        ScadCamera::Auto {
            yaw: 35.0,
            pitch: 25.0,
            zoom: 1.0,
        }
    }

    /// One full revolution over the animation, fitted to the model.
    pub fn turntable() -> Self {
        ScadCamera::Turntable {
            pitch: 25.0,
            turns: 1.0,
            zoom: 1.0,
            start_yaw: 35.0,
        }
    }

    /// A copy with the distance scaled — below 1 moves closer.
    pub fn zoomed(self, zoom: f32) -> Self {
        match self {
            ScadCamera::Auto { yaw, pitch, .. } => ScadCamera::Auto { yaw, pitch, zoom },
            ScadCamera::Turntable {
                pitch,
                turns,
                start_yaw,
                ..
            } => ScadCamera::Turntable {
                pitch,
                turns,
                zoom,
                start_yaw,
            },
            other => other,
        }
    }

    /// Whether the framing changes from frame to frame.
    pub fn is_animated(self) -> bool {
        matches!(self, ScadCamera::Turntable { .. } | ScadCamera::Viewport)
    }
}

/// Place a camera for one frame.
///
/// `fit` is the box the camera should frame — normally the union over every
/// frame, so the model neither jumps nor clips as it animates.
fn place_camera(
    camera: ScadCamera,
    frame: &ScadFrame,
    fit: ([f32; 3], [f32; 3]),
    aspect: f32,
) -> PerspectiveCamera {
    // OpenSCAD's world is z-up, and so is every model written for it.
    let up = Vector3::new(0.0, 0.0, 1.0);
    let center = Vector3::new(
        (fit.0[0] + fit.1[0]) * 0.5,
        (fit.0[1] + fit.1[1]) * 0.5,
        (fit.0[2] + fit.1[2]) * 0.5,
    );
    let corners = box_corners(fit);
    let radius = corners
        .iter()
        .map(|c| (*c - center).length())
        .fold(0.0f32, f32::max)
        .max(1e-3);

    let default_fov = 45.0f32;
    let (eye, target, fov) = match camera {
        ScadCamera::Fixed { eye, target, fov } => (
            Vector3::new(eye[0], eye[1], eye[2]),
            Vector3::new(target[0], target[1], target[2]),
            fov,
        ),
        ScadCamera::Viewport if frame.viewport.explicit => {
            let vp = frame.viewport;
            let target = Vector3::new(vp.target[0], vp.target[1], vp.target[2]);
            (
                orbit_eye_from_vpr(vp.rotation, vp.distance, target),
                target,
                vp.fov,
            )
        }
        // No viewport in the model — frame it the default way.
        ScadCamera::Viewport => {
            let (yaw, pitch, zoom) = (35.0, 25.0, 1.0);
            let dir = orbit_direction(yaw, pitch);
            let d = fit_distance(&corners, center, dir, up, default_fov, aspect) * zoom;
            (center + dir * d, center, default_fov)
        }
        ScadCamera::Auto { yaw, pitch, zoom } => {
            let dir = orbit_direction(yaw, pitch);
            let d = fit_distance(&corners, center, dir, up, default_fov, aspect) * zoom.max(0.01);
            (center + dir * d, center, default_fov)
        }
        ScadCamera::Turntable {
            pitch,
            turns,
            zoom,
            start_yaw,
        } => {
            let yaw = start_yaw + 360.0 * turns * frame.t as f32;
            let dir = orbit_direction(yaw, pitch);
            // One distance for the whole orbit, or the model would breathe in
            // and out as its silhouette widened and narrowed. Taking the worst
            // case over the yaws actually visited keeps that fixed distance far
            // tighter than a bounding sphere would.
            let d = orbit_fit_distance(&corners, center, pitch, up, default_fov, aspect)
                * zoom.max(0.01);
            (center + dir * d, center, default_fov)
        }
    };

    let span = (eye - target).length().max(1e-3);
    // A tight depth range keeps precision where CAD parts need it, but the near
    // plane must clear the closest geometry with room to spare: `span - radius`
    // sits exactly on the nearest point of the bounding sphere, so using it
    // verbatim slices the front off the model.
    let near = ((span - radius) * 0.5).max(span * 0.002).max(1e-4);
    let far = span + radius * 4.0;
    let mut cam = PerspectiveCamera::new(fov, aspect, near, far);
    cam.up = up;
    cam.position = eye;
    cam.look_at(target);
    cam
}

/// The eight corners of an axis-aligned box.
fn box_corners(bounds: ([f32; 3], [f32; 3])) -> [Vector3; 8] {
    let (min, max) = bounds;
    std::array::from_fn(|i| {
        Vector3::new(
            if i & 1 == 0 { min[0] } else { max[0] },
            if i & 2 == 0 { min[1] } else { max[1] },
            if i & 4 == 0 { min[2] } else { max[2] },
        )
    })
}

/// Unit vector from the model center toward the eye, for a yaw/pitch orbit in
/// a z-up world.
fn orbit_direction(yaw_deg: f32, pitch_deg: f32) -> Vector3 {
    let yaw = yaw_deg.to_radians();
    // Just shy of straight down/up, so the view direction never becomes
    // parallel to `up` and the camera basis stays well defined.
    let pitch = pitch_deg.to_radians().clamp(-1.5533, 1.5533);
    Vector3::new(
        pitch.cos() * yaw.sin(),
        -pitch.cos() * yaw.cos(),
        pitch.sin(),
    )
    .normalize()
}

/// Smallest distance at which the whole box fits both the vertical and the
/// horizontal field of view, from the given direction.
///
/// Exact for a box: for every corner, the depth needed to keep its off-axis
/// offset inside the frustum at its own depth.
fn fit_distance(
    corners: &[Vector3; 8],
    center: Vector3,
    dir_to_eye: Vector3,
    up: Vector3,
    fov_deg: f32,
    aspect: f32,
) -> f32 {
    let forward = dir_to_eye * -1.0;
    let right = forward.cross(up).normalize();
    let cam_up = right.cross(forward).normalize();
    let tan_v = (fov_deg.to_radians() * 0.5).tan().max(1e-4);
    let tan_h = (tan_v * aspect.max(0.05)).max(1e-4);

    let mut distance = 0.0f32;
    for corner in corners {
        let v = *corner - center;
        // How much further from the eye this corner sits than the center.
        let along = v.dot(forward);
        let x = v.dot(right).abs();
        let y = v.dot(cam_up).abs();
        distance = distance.max(x / tan_h - along).max(y / tan_v - along);
    }
    // A little air so the silhouette does not graze the frame edge.
    (distance.max(1e-3)) * 1.06
}

/// The largest [`fit_distance`] over a full turn at `pitch` — one distance that
/// frames the model from every yaw a turntable passes through.
///
/// Sampling every 5° is exact enough: between two samples the required distance
/// varies by well under the 6% margin `fit_distance` already adds.
fn orbit_fit_distance(
    corners: &[Vector3; 8],
    center: Vector3,
    pitch: f32,
    up: Vector3,
    fov_deg: f32,
    aspect: f32,
) -> f32 {
    (0..72)
        .map(|i| {
            let dir = orbit_direction(i as f32 * 5.0, pitch);
            fit_distance(corners, center, dir, up, fov_deg, aspect)
        })
        .fold(0.0f32, f32::max)
        .max(1e-3)
}

/// Eye position from OpenSCAD's `$vpr` rotation and `$vpd` distance.
///
/// `$vpr` rotates the *scene* about x then z; the camera looks down -y of the
/// rotated frame from `$vpd` away. Applying that rotation to the camera offset
/// gives the eye in model space.
fn orbit_eye_from_vpr(rotation: [f32; 3], distance: f32, target: Vector3) -> Vector3 {
    let (rx, rz) = (rotation[0].to_radians(), rotation[2].to_radians());
    let (sx, cx) = rx.sin_cos();
    let (sz, cz) = rz.sin_cos();
    // Camera offset before rotation: straight back along -y.
    let v = Vector3::new(0.0, -distance, 0.0);
    let after_x = Vector3::new(v.x, v.y * cx - v.z * sx, v.y * sx + v.z * cx);
    let after_z = Vector3::new(
        after_x.x * cz - after_x.y * sz,
        after_x.x * sz + after_x.y * cz,
        after_x.z,
    );
    target - after_z
}

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------

/// Surface look applied to parts the model did not `color()`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScadMaterial {
    pub color: [f32; 4],
    pub metalness: f32,
    pub roughness: f32,
}

impl Default for ScadMaterial {
    /// OpenSCAD's familiar yellow, as a physically-plausible plastic.
    fn default() -> Self {
        Self {
            color: [0.96, 0.76, 0.20, 1.0],
            metalness: 0.05,
            roughness: 0.45,
        }
    }
}

impl ScadMaterial {
    /// Base color for parts with no `color()` of their own.
    pub fn color(mut self, rgba: [f32; 4]) -> Self {
        self.color = rgba;
        self
    }

    /// How metallic the surface reads (`0` dielectric, `1` bare metal).
    pub fn metalness(mut self, metalness: f32) -> Self {
        self.metalness = metalness.clamp(0.0, 1.0);
        self
    }

    /// How rough the surface reads (`0` mirror, `1` fully diffuse).
    pub fn roughness(mut self, roughness: f32) -> Self {
        self.roughness = roughness.clamp(0.0, 1.0);
        self
    }
}

/// A color scheme: a background plus the colors parts take when the model does
/// not choose for itself.
///
/// ```
/// use threers::openscad::animate::ScadPalette;
/// let p = ScadPalette::cornfield();
/// assert_eq!(p.name, "cornfield");
/// // The cycle repeats, so any part index resolves to a color.
/// assert_eq!(p.color_at(0), p.color_at(p.cycle.len()));
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct ScadPalette {
    /// Lower-case identifier, as [`ScadPalette::by_name`] takes.
    pub name: String,
    /// Background behind the model.
    pub background: [f32; 4],
    /// Fallback part color when [`cycle`](Self::cycle) is empty.
    pub default_part: [f32; 4],
    /// Colors handed to successive parts, repeating as needed.
    pub cycle: Vec<[f32; 4]>,
    /// Light arriving from above — the sky half of the ambient hemisphere.
    pub sky: [f32; 3],
    /// Light bouncing up from below — the ground half.
    pub ground: [f32; 3],
}

/// Build a palette from hex literals, which read far better than float arrays.
fn palette(name: &str, background: u32, parts: &[u32], sky: u32, ground: u32) -> ScadPalette {
    let rgba = |hex: u32| {
        [
            ((hex >> 16) & 0xFF) as f32 / 255.0,
            ((hex >> 8) & 0xFF) as f32 / 255.0,
            (hex & 0xFF) as f32 / 255.0,
            1.0,
        ]
    };
    let rgb = |hex: u32| {
        let c = rgba(hex);
        [c[0], c[1], c[2]]
    };
    ScadPalette {
        name: name.to_string(),
        background: rgba(background),
        default_part: rgba(parts.first().copied().unwrap_or(0xF9D72C)),
        cycle: parts.iter().map(|h| rgba(*h)).collect(),
        sky: rgb(sky),
        ground: rgb(ground),
    }
}

impl Default for ScadPalette {
    fn default() -> Self {
        Self::studio()
    }
}

impl ScadPalette {
    /// Every built-in palette, in the order [`by_name`](Self::by_name) matches.
    pub fn all() -> Vec<ScadPalette> {
        vec![
            Self::studio(),
            Self::cornfield(),
            Self::metallic(),
            Self::sunset(),
            Self::midnight(),
            Self::blueprint(),
            Self::nature(),
            Self::monochrome(),
        ]
    }

    /// Look up a built-in palette by name, case-insensitively.
    ///
    /// ```
    /// use threers::openscad::animate::ScadPalette;
    /// assert_eq!(ScadPalette::by_name("Cornfield").unwrap().name, "cornfield");
    /// assert!(ScadPalette::by_name("chartreuse").is_none());
    /// ```
    pub fn by_name(name: &str) -> Option<ScadPalette> {
        let wanted = name.trim().to_ascii_lowercase();
        Self::all().into_iter().find(|p| p.name == wanted)
    }

    /// Neutral grey studio — the default. Nothing competes with the model.
    pub fn studio() -> Self {
        palette(
            "studio",
            0x12141B,
            &[0xC8CDD6, 0x7FA6C9, 0xD9A05B, 0x8FBF7F, 0xC98F9E, 0x9C8FBF],
            0xB9C6DA,
            0x2A2E38,
        )
    }

    /// Yellow on cream, after OpenSCAD's default viewport scheme.
    pub fn cornfield() -> Self {
        palette(
            "cornfield",
            0xFFFFE5,
            &[0xF9D72C, 0xC5A800, 0xE8C33A, 0xA8912B, 0xFFE97F, 0x8F7B1E],
            0xFFF6D8,
            0x6B6046,
        )
    }

    /// Cool metals on a pale blue ground.
    pub fn metallic() -> Self {
        palette(
            "metallic",
            0xAAAAFF,
            &[0xDDDDFF, 0x9FA8B8, 0xC0C6D0, 0x7C8796, 0xE6E9F0, 0x5E6875],
            0xDDE4FF,
            0x3C4250,
        )
    }

    /// Warm oranges and reds against dusk.
    pub fn sunset() -> Self {
        palette(
            "sunset",
            0x2B1B2E,
            &[0xFF9E4A, 0xE5603D, 0xFFD08A, 0xB8425A, 0xF7C948, 0x7A3B5C],
            0xFFD2A8,
            0x40202E,
        )
    }

    /// Near-black with saturated accents — good for a dark page.
    pub fn midnight() -> Self {
        palette(
            "midnight",
            0x05060A,
            &[0x4FC3F7, 0xFFB74D, 0x81C784, 0xE57373, 0xBA68C8, 0xFFF176],
            0x8FB4D9,
            0x0A0F1A,
        )
    }

    /// White line-work blues on drafting paper.
    pub fn blueprint() -> Self {
        palette(
            "blueprint",
            0x0D3B66,
            &[0xE8F1F8, 0xA9C8E8, 0x6FA3D2, 0xCFE2F3, 0x4A7FB5, 0x8FB8DC],
            0xDCEBFA,
            0x0A2C50,
        )
    }

    /// Greens and woods.
    pub fn nature() -> Self {
        palette(
            "nature",
            0x1A2418,
            &[0x8FBF6F, 0xC9A96A, 0x5F8F4F, 0xE0D5A8, 0x3F6B3A, 0xA88B5A],
            0xD8E8C8,
            0x2A2418,
        )
    }

    /// One hue, several values — for figures that must survive greyscale print.
    pub fn monochrome() -> Self {
        palette(
            "monochrome",
            0x101010,
            &[0xF0F0F0, 0xBDBDBD, 0x8A8A8A, 0x5C5C5C, 0xD6D6D6, 0x757575],
            0xFFFFFF,
            0x1A1A1A,
        )
    }

    /// A palette from your own colors. The first becomes the background.
    ///
    /// ```
    /// use threers::openscad::animate::ScadPalette;
    /// let p = ScadPalette::from_hex("brand", 0x101820, &[0xF2A900, 0x0072CE]);
    /// assert_eq!(p.cycle.len(), 2);
    /// ```
    pub fn from_hex(name: &str, background: u32, parts: &[u32]) -> Self {
        // Neutral hemisphere; override with `lit_by` for a tinted scheme.
        palette(name, background, parts, 0xC8D2E0, 0x24282F)
    }

    /// Replace the ambient hemisphere — light from above and bounce from below.
    pub fn lit_by(mut self, sky: [f32; 3], ground: [f32; 3]) -> Self {
        self.sky = sky;
        self.ground = ground;
        self
    }

    /// Replace the part colors, keeping the background.
    pub fn colors(mut self, cycle: Vec<[f32; 4]>) -> Self {
        if let Some(first) = cycle.first() {
            self.default_part = *first;
        }
        self.cycle = cycle;
        self
    }

    /// Replace the background.
    pub fn background(mut self, rgba: [f32; 4]) -> Self {
        self.background = rgba;
        self
    }

    /// The color for part `index`, wrapping around the cycle.
    pub fn color_at(&self, index: usize) -> [f32; 4] {
        if self.cycle.is_empty() {
            self.default_part
        } else {
            self.cycle[index % self.cycle.len()]
        }
    }
}

/// Where a part's color comes from.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum ScadColoring {
    /// Honor `color()`; anything the model left untagged takes the material
    /// color. What OpenSCAD itself shows.
    #[default]
    Model,
    /// Honor `color()`, but give untagged parts successive palette colors — so
    /// an assembly that was never colored still reads as separate pieces.
    ModelThenPalette,
    /// Ignore `color()` entirely and walk the palette. Useful for telling parts
    /// apart, or for re-skinning a model you did not write.
    ///
    /// Colors are assigned by part order, so a model whose part *count* changes
    /// mid-animation (geometry behind an `if`) will shift colors when it does.
    Palette,
    /// One color for every part, from the material.
    Uniform,
}

/// How much work to spend on quality — the speed/fidelity dial.
///
/// Applies to two different pipelines, so both objects take it:
/// [`ScadAnimation::quality`] sets the boolean kernel and the curve resolution
/// used for models that do not pin their own `$fn`/`$fa`/`$fs`, while
/// [`ScadRender::quality`] sets supersampling.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum ScadQuality {
    /// Fastest. Float kernel, coarse curves, no supersampling — for finding
    /// your framing before committing to a long render.
    Draft,
    /// The default: robust kernel, OpenSCAD's own curve defaults, 2× supersampling.
    #[default]
    Balanced,
    /// Slowest. Robust kernel, fine curves, 3× supersampling.
    Fine,
}

impl ScadQuality {
    /// Supersample factor [`ScadRender::quality`] applies.
    pub fn supersample(self) -> u32 {
        match self {
            ScadQuality::Draft => 1,
            ScadQuality::Balanced => 2,
            ScadQuality::Fine => 3,
        }
    }

    /// Boolean kernel [`ScadAnimation::quality`] applies.
    pub fn kernel(self) -> ScadKernel {
        match self {
            ScadQuality::Draft => ScadKernel::Float,
            _ => ScadKernel::Exact,
        }
    }

    /// `($fa, $fs)` — the minimum facet angle and size for curved primitives.
    /// `None` leaves OpenSCAD's defaults (12°, 2 mm) alone.
    pub fn facets(self) -> Option<(f64, f64)> {
        match self {
            ScadQuality::Draft => Some((24.0, 4.0)),
            ScadQuality::Balanced => None,
            ScadQuality::Fine => Some((6.0, 0.8)),
        }
    }
}

/// One light in a [`Custom`](ScadLighting::custom) rig.
///
/// Directions are in the model's own z-up world, and name the direction the
/// light *travels* — `[0, 0, -1]` shines straight down.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ScadLight {
    /// Uniform light from every direction. Cheap, and flattens everything;
    /// prefer [`Hemisphere`](Self::Hemisphere) for fill.
    Ambient { color: [f32; 3], intensity: f32 },
    /// Sky above, bounce below — fill that still shows which way is up.
    Hemisphere {
        sky: [f32; 3],
        ground: [f32; 3],
        intensity: f32,
    },
    /// Parallel light from infinitely far away, like the sun.
    Directional {
        color: [f32; 3],
        intensity: f32,
        direction: [f32; 3],
    },
    /// A directional light placed relative to the *camera*, so it holds still
    /// against the view while the model or the turntable turns.
    ///
    /// `azimuth` is degrees to the right of the view axis, `elevation` degrees
    /// above it — `(0, 0)` is a lamp on the lens.
    Key {
        color: [f32; 3],
        intensity: f32,
        azimuth: f32,
        elevation: f32,
    },
    /// A local light at a fixed point in the model's world.
    Point {
        color: [f32; 3],
        intensity: f32,
        position: [f32; 3],
    },
}

/// Which arrangement of lights to build.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum ScadRig {
    /// Key, fill and rim placed relative to the camera, over a sky/ground
    /// hemisphere. Because the lights travel with the view, a turntable never
    /// swings the model into its own shadow — the usual failure of a
    /// world-fixed rig.
    #[default]
    Studio,
    /// One hard sun from a fixed compass bearing plus sky fill. World-fixed on
    /// purpose: the shading tells you which way the part is facing, which is
    /// what an assembly drawing wants.
    Sun,
    /// Shadowless and low-contrast, close to OpenSCAD's own preview.
    Flat,
    /// Only the lights in [`ScadLighting::extra`].
    Custom,
}

/// How a model is lit.
///
/// The default is a camera-relative three-point [`Studio`](ScadRig::Studio) rig
/// with soft self-shadowing, which is what makes a CAD part read as a solid
/// object rather than a flat silhouette.
///
/// ```
/// use threers::openscad::animate::{ScadLighting, ScadRig};
/// let bright = ScadLighting::studio().intensity(1.4).shadows(false);
/// assert_eq!(bright.rig, ScadRig::Studio);
///
/// // A fixed sun from the south-west, 40° up.
/// let sun = ScadLighting::sun(225.0, 40.0);
/// assert_eq!(sun.rig, ScadRig::Sun);
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct ScadLighting {
    /// Which arrangement to build.
    pub rig: ScadRig,
    /// Multiplier over every light in the rig.
    pub intensity: f32,
    /// Strength of the ambient hemisphere. Higher opens up the shadows and
    /// flattens the form; lower is more dramatic and can go black.
    pub ambient: f32,
    /// Sky / ground colors for the hemisphere. `None` takes them from the
    /// palette, so the light matches the scheme.
    pub hemisphere: Option<([f32; 3], [f32; 3])>,
    /// Tints the key warm and the fill cool. `0` is neutral, `1` is a strong
    /// warm-key/cool-fill split — the oldest trick for making a grey object
    /// look like it is somewhere.
    pub warmth: f32,
    /// Key placement: degrees right of, and above, the view axis
    /// ([`Studio`](ScadRig::Studio)) or the compass bearing and elevation of
    /// the sun ([`Sun`](ScadRig::Sun)).
    pub key_angles: [f32; 2],
    /// Cast shadows from the key light. Ignored by [`Flat`](ScadRig::Flat).
    pub shadows: bool,
    /// Extra lights added on top of the rig.
    pub extra: Vec<ScadLight>,
}

impl Default for ScadLighting {
    fn default() -> Self {
        Self::studio()
    }
}

impl ScadLighting {
    /// Camera-relative three-point rig — the default.
    pub fn studio() -> Self {
        Self {
            rig: ScadRig::Studio,
            intensity: 1.0,
            ambient: 0.35,
            hemisphere: None,
            warmth: 0.35,
            key_angles: [35.0, 30.0],
            shadows: true,
            extra: Vec::new(),
        }
    }

    /// A single fixed sun at `azimuth` degrees (compass bearing, 0 = from the
    /// −Y side) and `elevation` degrees above the horizon, plus sky fill.
    pub fn sun(azimuth: f32, elevation: f32) -> Self {
        Self {
            rig: ScadRig::Sun,
            intensity: 1.0,
            ambient: 0.32,
            hemisphere: None,
            warmth: 0.5,
            key_angles: [azimuth, elevation],
            shadows: true,
            extra: Vec::new(),
        }
    }

    /// Flat, shadowless light — closest to what the OpenSCAD GUI shows.
    pub fn flat() -> Self {
        Self {
            rig: ScadRig::Flat,
            intensity: 1.0,
            ambient: 0.82,
            hemisphere: None,
            warmth: 0.0,
            key_angles: [20.0, 20.0],
            shadows: false,
            extra: Vec::new(),
        }
    }

    /// Exactly the lights you give it, and nothing else.
    pub fn custom(lights: Vec<ScadLight>) -> Self {
        Self {
            rig: ScadRig::Custom,
            intensity: 1.0,
            ambient: 0.0,
            hemisphere: None,
            warmth: 0.0,
            key_angles: [0.0, 0.0],
            shadows: false,
            extra: lights,
        }
    }

    /// Scale every light in the rig.
    pub fn intensity(mut self, intensity: f32) -> Self {
        self.intensity = intensity.max(0.0);
        self
    }

    /// Strength of the ambient hemisphere — how far into shadow you can see.
    pub fn ambient(mut self, ambient: f32) -> Self {
        self.ambient = ambient.max(0.0);
        self
    }

    /// Pin the hemisphere colors instead of taking them from the palette.
    pub fn hemisphere(mut self, sky: [f32; 3], ground: [f32; 3]) -> Self {
        self.hemisphere = Some((sky, ground));
        self
    }

    /// Warm key against cool fill. `0` neutral, `1` strong.
    pub fn warmth(mut self, warmth: f32) -> Self {
        self.warmth = warmth.clamp(0.0, 1.0);
        self
    }

    /// Move the key: degrees right of and above the view axis (or the sun's
    /// bearing and elevation).
    pub fn key_angles(mut self, azimuth: f32, elevation: f32) -> Self {
        self.key_angles = [azimuth, elevation];
        self
    }

    /// Cast shadows from the key light.
    pub fn shadows(mut self, shadows: bool) -> Self {
        self.shadows = shadows;
        self
    }

    /// Add a light on top of the rig.
    pub fn add_light(mut self, light: ScadLight) -> Self {
        self.extra.push(light);
        self
    }

    /// Warm and cool tints for the key and fill at the current `warmth`.
    fn tints(&self) -> ([f32; 3], [f32; 3]) {
        let w = self.warmth.clamp(0.0, 1.0);
        // Nudge toward amber and toward sky-blue by equal amounts, so overall
        // brightness barely moves as `warmth` rises.
        let key = [1.0, 1.0 - 0.06 * w, 1.0 - 0.16 * w];
        let fill = [1.0 - 0.14 * w, 1.0 - 0.06 * w, 1.0];
        (key, fill)
    }
}

/// Size, look, and framing for rendering a [`crate::openscad::animate::ScadAnimation`].
///
/// ```no_run
/// use threers::openscad::animate::{ScadAnimation, ScadCamera, ScadRender};
/// // `export_video` needs the `video` feature; hidden behind a `cfg` so this
/// // still compiles as documentation when it is off.
/// # #[cfg(feature = "video")]
/// # fn demo() {
/// let mut animation = ScadAnimation::from_file("part.scad").frames(90);
/// ScadRender::new(1280, 720)
///     .supersample(2)
///     .camera(ScadCamera::turntable())
///     .export_video(&mut animation, "part.mp4")
///     .unwrap();
/// # }
/// ```
pub struct ScadRender {
    width: u32,
    height: u32,
    supersample: u32,
    background: [f32; 4],
    camera: ScadCamera,
    material: ScadMaterial,
    palette: ScadPalette,
    coloring: ScadColoring,
    lighting: ScadLighting,
    /// Recomputed per frame instead of fitted once over the whole animation.
    fit_per_frame: bool,
}

impl ScadRender {
    /// A renderer for `width × height` output.
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width: width.max(1),
            height: height.max(1),
            supersample: 1,
            background: [0.07, 0.08, 0.11, 1.0],
            camera: ScadCamera::default(),
            material: ScadMaterial::default(),
            palette: ScadPalette::default(),
            coloring: ScadColoring::default(),
            lighting: ScadLighting::default(),
            fit_per_frame: false,
        }
    }

    /// Render at `factor`× and box-filter down — cheap, effective anti-aliasing
    /// for the hard edges CAD geometry is made of.
    pub fn supersample(mut self, factor: u32) -> Self {
        self.supersample = factor.clamp(1, 4);
        self
    }

    /// Background color behind the model.
    pub fn background(mut self, rgba: [f32; 4]) -> Self {
        self.background = rgba;
        self
    }

    /// How to frame the model.
    pub fn camera(mut self, camera: ScadCamera) -> Self {
        self.camera = camera;
        self
    }

    /// Look of parts the model did not `color()`.
    pub fn material(mut self, material: ScadMaterial) -> Self {
        self.material = material;
        self
    }

    /// Choose a color scheme. Also sets the background, so call
    /// [`background`](Self::background) *after* this to override it.
    ///
    /// On its own a palette only affects parts the model left untagged; pair it
    /// with [`coloring`](Self::coloring) to override `color()` as well.
    ///
    /// ```
    /// use threers::openscad::animate::{ScadColoring, ScadPalette, ScadRender};
    /// let render = ScadRender::new(640, 360)
    ///     .palette(ScadPalette::blueprint())
    ///     .coloring(ScadColoring::Palette);
    /// assert_eq!(render.palette_ref().name, "blueprint");
    /// ```
    pub fn palette(mut self, palette: ScadPalette) -> Self {
        self.background = palette.background;
        self.palette = palette;
        self
    }

    /// Choose a built-in palette by name (`"cornfield"`, `"blueprint"`, …).
    /// An unknown name leaves the current palette alone.
    ///
    /// See [`ScadPalette::all`] for the list.
    pub fn palette_named(self, name: &str) -> Self {
        match ScadPalette::by_name(name) {
            Some(p) => self.palette(p),
            None => self,
        }
    }

    /// Where each part's color comes from.
    pub fn coloring(mut self, coloring: ScadColoring) -> Self {
        self.coloring = coloring;
        self
    }

    /// Give parts these colors in order, repeating as needed, ignoring any
    /// `color()` the model set.
    ///
    /// ```
    /// use threers::openscad::animate::ScadRender;
    /// let render = ScadRender::new(320, 240)
    ///     .part_colors(vec![[1.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0]]);
    /// assert_eq!(render.palette_ref().color_at(2), [1.0, 0.0, 0.0, 1.0]);
    /// ```
    pub fn part_colors(mut self, colors: Vec<[f32; 4]>) -> Self {
        self.palette = self.palette.clone().colors(colors);
        self.coloring = ScadColoring::Palette;
        self
    }

    /// How the model is lit — see [`ScadLighting`].
    ///
    /// ```
    /// use threers::openscad::animate::{ScadLighting, ScadRender};
    /// // A hard, fixed sun with deep shadows.
    /// let render = ScadRender::new(640, 360)
    ///     .lighting(ScadLighting::sun(215.0, 35.0).ambient(0.25));
    /// assert!(render.lighting_ref().shadows);
    /// ```
    pub fn lighting(mut self, lighting: ScadLighting) -> Self {
        self.lighting = lighting;
        self
    }

    /// The active lighting rig.
    pub fn lighting_ref(&self) -> &ScadLighting {
        &self.lighting
    }

    /// Trade render time for fidelity — see [`ScadQuality`]. Sets supersampling.
    pub fn quality(mut self, quality: ScadQuality) -> Self {
        self.supersample = quality.supersample();
        self
    }

    /// The active palette.
    pub fn palette_ref(&self) -> &ScadPalette {
        &self.palette
    }

    /// Where part colors currently come from.
    pub fn coloring_mode(&self) -> ScadColoring {
        self.coloring
    }

    /// Fit the camera to each frame separately instead of to the whole
    /// animation.
    ///
    /// Off by default, and usually should stay off: fitting per frame makes a
    /// growing model appear to sit still while the world shrinks around it.
    /// Turn it on when the model changes size so much that one fit would leave
    /// it tiny.
    pub fn fit_per_frame(mut self, per_frame: bool) -> Self {
        self.fit_per_frame = per_frame;
        self
    }

    /// Output size in pixels.
    pub fn size(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    /// Build the [`Scene`] for one frame — lights, materials, and one mesh per
    /// colored part. Useful when you want to add to it before rendering.
    ///
    /// The camera is needed because the default rig places its lights relative
    /// to the view; see [`ScadLighting`].
    pub fn build_scene(&self, frame: &ScadFrame, camera: &PerspectiveCamera) -> Scene {
        let mut scene = Scene::new();
        scene.background = Color::new(self.background[0], self.background[1], self.background[2]);
        scene.background_alpha = self.background[3];

        let shadows = self.add_lights(&mut scene, frame, camera);

        for (index, part) in frame.parts.iter().enumerate() {
            let rgba = self.color_for(part, index);
            let mut material = StandardMaterial::new(Color::new(rgba[0], rgba[1], rgba[2]));
            material.metalness = self.material.metalness;
            material.roughness = self.material.roughness;
            let transparent = rgba[3] < 0.999;
            if transparent {
                // `opacity < 1` is all the renderer needs to route this through
                // the alpha pipeline; double-siding lets you see the inside of
                // a see-through part, which is the point of making it one.
                material.opacity = rgba[3];
                material.side = 2; // DoubleSide
            }
            let mut object = Object3D::mesh(Mesh::new(
                part.geometry.clone(),
                Material::Standard(material),
            ));
            // A see-through part casting a hard shadow reads as a bug, and the
            // shadow pass skips alpha topologies anyway.
            object.cast_shadow = shadows && !transparent;
            object.receive_shadow = shadows;
            scene.add(object);
        }
        scene
    }

    /// Populate `scene` with the configured rig. Returns whether shadows are on.
    fn add_lights(&self, scene: &mut Scene, frame: &ScadFrame, camera: &PerspectiveCamera) -> bool {
        let light = &self.lighting;
        let gain = light.intensity;
        let (key_tint, fill_tint) = light.tints();
        let (sky, ground) = light
            .hemisphere
            .unwrap_or((self.palette.sky, self.palette.ground));

        // Camera basis, in the model's z-up world.
        let world_up = Vector3::new(0.0, 0.0, 1.0);
        let forward = (camera.target - camera.position).normalize();
        let right = forward.cross(world_up);
        // Looking straight down leaves `right` degenerate; fall back to +X.
        let right = if right.length() < 1e-4 {
            Vector3::new(1.0, 0.0, 0.0)
        } else {
            right.normalize()
        };
        let up = right.cross(forward).normalize();

        // Unit vector from the subject toward a light at (azimuth, elevation)
        // relative to the view. `-forward` is straight back at the camera.
        let toward_view_light = |azimuth: f32, elevation: f32| {
            let (az, el) = (azimuth.to_radians(), elevation.to_radians());
            (right * (az.sin() * el.cos()) + up * el.sin() - forward * (az.cos() * el.cos()))
                .normalize()
        };

        // The hemisphere's up axis comes from the light object's own rotation,
        // and its local up is +Y — so rotate +Y onto the world's +Z.
        let mut hemi = Object3D::light(HemisphereLight::new(
            Color::new(sky[0], sky[1], sky[2]),
            Color::new(ground[0], ground[1], ground[2]),
            light.ambient * gain,
        ));
        hemi.quaternion =
            crate::math::Quaternion::from_euler_xyz(std::f32::consts::FRAC_PI_2, 0.0, 0.0);

        // The renderer takes the *first* shadow-casting directional light as the
        // caster, so the key has to be the first one added — and has to carry
        // the shadow settings itself rather than trailing a second light.
        let shadow_settings = light
            .shadows
            .then(|| frame.bounds().map(shadow_settings_for))
            .flatten();
        let shadows = shadow_settings.is_some();

        let key_light = |toward: Vector3, tint: [f32; 3], strength: f32| {
            let mut key = directional(tint, strength, toward);
            if let Some(settings) = shadow_settings {
                key.cast_shadow = true;
                key.shadow = settings;
            }
            key
        };

        match light.rig {
            ScadRig::Studio => {
                scene.add(hemi);
                let [az, el] = light.key_angles;
                // Key first: it is the shadow caster.
                scene.add_light(key_light(toward_view_light(az, el), key_tint, 1.9 * gain));
                // Fill: opposite side, near eye level, soft.
                scene.add_light(directional(
                    fill_tint,
                    0.55 * gain,
                    toward_view_light(-(az + 25.0), el * 0.2),
                ));
                // Rim: behind and above, to lift the silhouette off the
                // background — the light that stops a dark part disappearing.
                scene.add_light(directional(
                    [0.90, 0.94, 1.0],
                    0.85 * gain,
                    toward_view_light(180.0 - az * 0.5, el + 20.0),
                ));
            }
            ScadRig::Sun => {
                scene.add(hemi);
                let [az, el] = light.key_angles;
                scene.add_light(key_light(orbit_direction(az, el), key_tint, 2.1 * gain));
                // A weak bounce from the opposite side so the shaded faces are
                // not pure hemisphere.
                scene.add_light(directional(
                    fill_tint,
                    0.40 * gain,
                    orbit_direction(az + 150.0, 12.0),
                ));
            }
            ScadRig::Flat => {
                scene.add(hemi);
                // One soft lamp on the view axis: enough to shade curvature
                // without casting anything that reads as a shadow.
                scene.add_light(directional(
                    [1.0, 1.0, 1.0],
                    1.05 * gain,
                    toward_view_light(12.0, 14.0),
                ));
            }
            ScadRig::Custom => {}
        }

        for extra in &light.extra {
            match *extra {
                ScadLight::Ambient { color, intensity } => {
                    scene.add_light(AmbientLight::new(
                        Color::new(color[0], color[1], color[2]),
                        intensity * gain,
                    ));
                }
                ScadLight::Hemisphere {
                    sky,
                    ground,
                    intensity,
                } => {
                    let mut object = Object3D::light(HemisphereLight::new(
                        Color::new(sky[0], sky[1], sky[2]),
                        Color::new(ground[0], ground[1], ground[2]),
                        intensity * gain,
                    ));
                    object.quaternion = crate::math::Quaternion::from_euler_xyz(
                        std::f32::consts::FRAC_PI_2,
                        0.0,
                        0.0,
                    );
                    scene.add(object);
                }
                ScadLight::Directional {
                    color,
                    intensity,
                    direction,
                } => {
                    let d = Vector3::new(direction[0], direction[1], direction[2]);
                    if d.length() > 1e-6 {
                        scene.add_light(directional(color, intensity * gain, d.normalize() * -1.0));
                    }
                }
                ScadLight::Key {
                    color,
                    intensity,
                    azimuth,
                    elevation,
                } => {
                    scene.add_light(directional(
                        color,
                        intensity * gain,
                        toward_view_light(azimuth, elevation),
                    ));
                }
                ScadLight::Point {
                    color,
                    intensity,
                    position,
                } => {
                    let mut object = Object3D::light(crate::lights::PointLight::new(
                        Color::new(color[0], color[1], color[2]),
                        intensity * gain,
                    ));
                    object.position = Vector3::new(position[0], position[1], position[2]);
                    scene.add(object);
                }
            }
        }

        shadows
    }

    /// The color part `index` will be drawn in, under the current
    /// [`coloring`](Self::coloring).
    pub fn color_for(&self, part: &ScadPart, index: usize) -> [f32; 4] {
        match self.coloring {
            ScadColoring::Model => part.rgba_or(self.material.color),
            ScadColoring::ModelThenPalette => {
                part.color.unwrap_or_else(|| self.palette.color_at(index))
            }
            ScadColoring::Palette => self.palette.color_at(index),
            ScadColoring::Uniform => self.material.color,
        }
    }

    /// The camera for one frame, framed on `fit`.
    pub fn build_camera(&self, frame: &ScadFrame, fit: ([f32; 3], [f32; 3])) -> PerspectiveCamera {
        place_camera(
            self.camera,
            frame,
            fit,
            self.width as f32 / self.height as f32,
        )
    }
}

// ---------------------------------------------------------------------------
// Render / export
// ---------------------------------------------------------------------------

impl ScadRender {
    /// Evaluate `animation` and render every frame to tightly-packed RGBA8.
    ///
    /// Frames come back at the configured [`size`](Self::size), already
    /// downsampled from any [`supersample`](Self::supersample) factor.
    pub fn render_frames(&self, animation: &mut ScadAnimation) -> Result<Vec<Vec<u8>>, String> {
        self.render_evaluated(&animation.evaluate()?)
    }

    /// Render frames that have already been evaluated.
    ///
    /// [`render_frames`](Self::render_frames) evaluates every time it is called,
    /// because [`ScadAnimation::evaluate`] does not cache. Use this when you
    /// want more than one render out of one evaluation — several sizes, a
    /// contact sheet, a quality comparison — or to time the render path without
    /// the CSG in front of it.
    ///
    /// Framing spans all of `frames`, exactly as `render_frames` does, so
    /// passing a subset reframes the shot.
    pub fn render_evaluated(&self, frames: &[ScadFrame]) -> Result<Vec<Vec<u8>>, String> {
        let mut renderer = self.headless()?;
        let fit = animation_bounds(frames);
        let mut out = Vec::with_capacity(frames.len());
        for frame in frames {
            out.push(self.render_one(&mut renderer, frame, fit));
        }
        Ok(out)
    }

    /// Render a single frame of `animation` to RGBA8, framed on that frame alone.
    pub fn render_frame(&self, animation: &ScadAnimation, index: usize) -> Result<Vec<u8>, String> {
        let frame = animation.frame(index)?;
        let fit = frame.bounds().unwrap_or(([-1.0; 3], [1.0; 3]));
        let mut renderer = self.headless()?;
        Ok(self.render_one(&mut renderer, &frame, fit))
    }

    /// Render one frame of `animation` and write it as a PNG.
    pub fn render_png(
        &self,
        animation: &ScadAnimation,
        index: usize,
        path: impl AsRef<std::path::Path>,
    ) -> Result<(), String> {
        let rgba = self.render_frame(animation, index)?;
        let bytes = crate::utils::png::encode_png(self.width, self.height, &rgba);
        std::fs::write(path.as_ref(), bytes)
            .map_err(|e| format!("cannot write {}: {e}", path.as_ref().display()))
    }

    /// Render every frame into `dir` as `{prefix}0000.png`, `{prefix}0001.png`, …
    ///
    /// The numbered-sequence form every video tool takes as input, for when you
    /// want to hand the frames to something else.
    pub fn export_png_sequence(
        &self,
        animation: &mut ScadAnimation,
        dir: impl AsRef<std::path::Path>,
        prefix: &str,
    ) -> Result<Vec<std::path::PathBuf>, String> {
        let dir = dir.as_ref();
        std::fs::create_dir_all(dir)
            .map_err(|e| format!("cannot create {}: {e}", dir.display()))?;
        let frames = self.render_frames(animation)?;
        let mut paths = Vec::with_capacity(frames.len());
        for (i, rgba) in frames.iter().enumerate() {
            let path = dir.join(format!("{prefix}{i:04}.png"));
            let bytes = crate::utils::png::encode_png(self.width, self.height, rgba);
            std::fs::write(&path, bytes)
                .map_err(|e| format!("cannot write {}: {e}", path.display()))?;
            paths.push(path);
        }
        Ok(paths)
    }

    /// Render `animation` and encode it to `output`, picking the codec from the
    /// file extension (`.mp4` → H.264, `.webm` → VP9, `.gif`, `.apng`).
    ///
    /// Needs the `video` feature. For control over quality, captions, or the
    /// codec, build your own [`VideoOptions`](crate::video::VideoOptions) and
    /// use [`export_video_with`](Self::export_video_with).
    #[cfg(all(feature = "video", not(target_arch = "wasm32")))]
    pub fn export_video(
        &self,
        animation: &mut ScadAnimation,
        output: impl AsRef<std::path::Path>,
    ) -> Result<(), String> {
        let path = output.as_ref();
        let codec = codec_for_path(path);
        let options = crate::video::VideoOptions::new(path.to_string_lossy().into_owned())
            .fps(animation.frame_rate())
            .codec(codec);
        self.export_video_with(animation, &options)
    }

    /// Render `animation` and encode it with explicit
    /// [`VideoOptions`](crate::video::VideoOptions) — the way to set quality,
    /// a codec that does not match the extension, or subtitles.
    ///
    /// The options' `fps` is used as given; set it from
    /// [`ScadAnimation::frame_rate`] to keep playback at the authored rate.
    #[cfg(all(feature = "video", not(target_arch = "wasm32")))]
    pub fn export_video_with(
        &self,
        animation: &mut ScadAnimation,
        options: &crate::video::VideoOptions,
    ) -> Result<(), String> {
        let frames = self.render_frames(animation)?;
        if frames.is_empty() {
            return Err("animation produced no frames".into());
        }
        crate::video::export_video(self.width, self.height, frames.len(), options, |i| {
            frames[i].clone()
        })
        .map_err(|e| e.to_string())
    }

    /// A headless renderer sized for this configuration.
    fn headless(&self) -> Result<crate::renderer::HeadlessRenderer, String> {
        crate::renderer::HeadlessRenderer::builder()
            .size(self.width, self.height)
            .supersample(self.supersample)
            // Rgba8Unorm, not the sRGB default: the mesh shader already encodes
            // sRGB, and an sRGB target would encode a second time.
            .color_format(wgpu::TextureFormat::Rgba8Unorm)
            .high_resolution(true)
            .build()
    }

    /// Render one already-evaluated frame, downsampling any supersample.
    fn render_one(
        &self,
        renderer: &mut crate::renderer::HeadlessRenderer,
        frame: &ScadFrame,
        animation_fit: ([f32; 3], [f32; 3]),
    ) -> Vec<u8> {
        let fit = if self.fit_per_frame {
            frame.bounds().unwrap_or(animation_fit)
        } else {
            animation_fit
        };
        let camera = self.build_camera(frame, fit);
        let mut scene = self.build_scene(frame, &camera);
        // Resolved, so any supersample is averaged down in the render pass the
        // GPU is already running — on the CPU that averaging cost several times
        // more than drawing the frame did.
        let rgba = renderer.render_to_rgba_resolved(&mut scene, &camera);
        // Each frame builds fresh geometry, so the pointer-keyed GPU cache has
        // to be dropped or it would grow for the whole animation.
        renderer.renderer().clear_geometry_cache();
        rgba
    }
}

/// Pick a codec from an output path's extension.
#[cfg(all(feature = "video", not(target_arch = "wasm32")))]
fn codec_for_path(path: &std::path::Path) -> crate::video::VideoCodec {
    use crate::video::VideoCodec;
    match path
        .extension()
        .map(|e| e.to_string_lossy().to_ascii_lowercase())
        .as_deref()
    {
        Some("webm") => VideoCodec::Vp9,
        Some("gif") => VideoCodec::Gif,
        Some("apng" | "png") => VideoCodec::Apng,
        Some("mov") => VideoCodec::Hevc,
        // `.mp4` and anything unrecognized: the most widely playable option.
        _ => VideoCodec::H264,
    }
}

/// Shadow settings that actually cover a model with these bounds.
///
/// The renderer's shadow box is an orthographic cube centred on the world
/// origin, so it has to reach the model's furthest corner — spanning merely the
/// model's *size* would clip a part that sits off-origin, which most SCAD
/// models do (they stand on z = 0).
fn shadow_settings_for(bounds: ([f32; 3], [f32; 3])) -> crate::lights::ShadowSettings {
    let reach = box_corners(bounds)
        .iter()
        .map(|c| c.length())
        .fold(0.0f32, f32::max)
        .max(1e-3);
    crate::lights::ShadowSettings {
        map_size: 2048,
        // Scaled to the model, so a 5 mm part and a 5 m one both avoid acne
        // without their shadows detaching.
        bias: (reach * 3.0e-4).max(1e-5),
        // Sized to the model like the depth bias, and doing the work the depth
        // bias cannot: a face nearly edge-on to the light needs the lookup moved
        // off the surface, not pushed deeper. Without it a sun low enough to
        // graze a face makes that face shadow itself, which reads as the rig
        // simply not lighting the model.
        normal_bias: (reach * 4.0e-3).max(1e-4),
        camera_size: reach * 1.15,
        // Fitted to the model, not merely large enough to contain it.
        //
        // `bias` is subtracted in NDC, so what a given bias is *worth* depends
        // on how much world depth the near–far range is spread over. A range of
        // 0.01..8·reach around a model 2·reach across spends most of its
        // precision on empty space, and the quantisation left over at the
        // surface is bigger than the bias — which is self-shadowing, and it
        // reads as the light simply not arriving. The renderer puts the shadow
        // eye at 1.725·reach, so the model occupies 0.725·reach..2.725·reach.
        camera_near: (reach * 0.7).max(1e-3),
        camera_far: reach * 2.8,
    }
}

/// A directional light of `color` at `intensity`, travelling *from* `toward`.
///
/// `toward` points from the subject toward the light, which is how the rig
/// reasons about placement; `DirectionalLight::direction` wants the opposite.
fn directional(color: [f32; 3], intensity: f32, toward: Vector3) -> DirectionalLight {
    DirectionalLight::new(Color::new(color[0], color[1], color[2]), intensity)
        .with_direction(toward * -1.0)
}

/// Union of every frame's bounds, so one camera fit covers the whole animation.
pub fn animation_bounds(frames: &[ScadFrame]) -> ([f32; 3], [f32; 3]) {
    let mut lo = [f32::MAX; 3];
    let mut hi = [f32::MIN; 3];
    let mut any = false;
    for frame in frames {
        if let Some((min, max)) = frame.bounds() {
            any = true;
            for i in 0..3 {
                lo[i] = lo[i].min(min[i]);
                hi[i] = hi[i].max(max[i]);
            }
        }
    }
    if any {
        (lo, hi)
    } else {
        ([-1.0; 3], [1.0; 3])
    }
}
