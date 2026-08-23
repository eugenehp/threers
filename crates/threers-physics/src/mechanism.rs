//! A `.scad` mechanism, simulated.
//!
//! OpenSCAD animates by re-evaluating the whole program once per frame with
//! `$t` stepped from 0 to 1 — the geometry *is* a function of time. That is a
//! fine way to draw a lid opening and a bad way to know whether it can: a model
//! told to open 105° opens 105° whether or not the shelf above it is in the
//! way, and a hinge told to go past its stop goes past it.
//!
//! This replaces the loop rather than the renderer. The parts are evaluated
//! **once**, the joints the model declared become real constraints, and each
//! frame is one step of the solver.
//!
//! ```no_run
//! use threers_physics::mechanism::ScadMechanism;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let mut mech = ScadMechanism::from_file("box.scad")?.frames(180).fps(60);
//! let track = mech.record();
//! # Ok(())
//! # }
//! ```
//!
//! # A frame is a pose, not a copy of the model
//!
//! `$t` animation has no choice about this: its geometry really is a function of
//! time, so every frame is a new mesh. A mechanism's is not. The solver moves
//! rigid parts and never reshapes them, so what changes between frames is seven
//! numbers per part — and baking that into the vertices re-derives every frame
//! what a matrix already said.
//!
//! Measured on a two-part assembly of 39,786 triangles over 120 frames:
//!
//! | | Time | Size |
//! |---|---|---|
//! | [`record`](crate::mechanism::ScadMechanism::record) | 1.4 ms | 6.7 kB |
//! | [`crate::mechanism::PoseTrack::baked`] | 128 ms | 172 MB |
//! | [`crate::mechanism::PoseTrack::spawn`] once, then [`apply`](crate::mechanism::PoseTrack::apply) × 120 | 0.3 ms | — |
//! | [`crate::mechanism::PoseTrack::bounds`], for framing a camera | 83 µs | — |
//!
//! So [`record`](crate::mechanism::ScadMechanism::record) is the one to reach for, and
//! [`crate::mechanism::PoseTrack`] will hand the result to a scene graph, bake a single frame on
//! demand, or bake the lot if something really needs the whole sequence in hand.
//! [`threers::openscad::frame::ScadFrame`] remains the currency of
//! [`ScadRender`](threers::openscad::animate::ScadRender), so that path still
//! works — it is just no longer the only one, and no longer the default.
//!
//! # And then it stops needing a simulation
//!
//! [`crate::mechanism::PoseTrack::to_clip_simplified`] turns the run into an ordinary
//! [`threers::animation::AnimationClip`], which
//! [`AnimationMixer`](threers::animation::AnimationMixer) plays like any other —
//! at any rate, blended, retimed — with the solver, the colliders and the whole
//! physics world gone. A mechanism becomes an animation asset.
//!
//! Most of the keys go with it. A mechanism's motion is smooth by construction,
//! being a solver following a drive, so frames that sit on the line between
//! their neighbours store the sampling rate rather than the motion. On the run
//! above: 960 keys down to 76, with the fixed part collapsing to the two the
//! format needs and the moving one keeping 36. A part that stalls against
//! something keeps two for the stall.
//!
//! ```no_run
//! # use threers_physics::mechanism::ScadMechanism;
//! use threers::openscad::animate::{ScadCamera, ScadRender};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! # let mut mech = ScadMechanism::from_file("box.scad")?;
//! let frames = mech.record().baked();
//! ScadRender::new(1280, 720)
//!     .camera(ScadCamera::turntable())
//!     .render_evaluated(&frames)?;
//! # Ok(())
//! # }
//! ```
//!
//! # What the model has to say
//!
//! ```text
//! part("box", fixed = true)  box_body();
//! part("lid")                lid_body();
//!
//! hinge("lid_pivot", parts = ["lid", "box"],
//!       at = [0, 0, 40], axis = [1, 0, 0], range = [0, 105]);
//!
//! drive("lid_pivot", to = 105, over = 1.5);
//! ```
//!
//! See [`threers::openscad::mechanism`] for the full vocabulary.
//!
//! # Two rules worth knowing
//!
//! **Only `part()` is drawn.** Geometry outside a part is not simulated, so it
//! is not rendered either — drawing it would mean drawing something the
//! mechanism does not know about, sitting still while everything around it
//! moves. Wrap scenery in `part("floor", fixed = true)` and it both draws and
//! collides.
//!
//! **Part geometry is evaluated once.** A part is a rigid body; its *pose*
//! changes over the animation and its shape does not. A model whose `part()`
//! bodies read `$t` is describing something this cannot simulate, and the shape
//! it gets is the one at `$t = 0`. That restriction is the feature: motion comes
//! from the solver, so it cannot pass through anything.

use crate::assembly::{Assembly, AssemblyError, ScadMechanismError, SCAD_GRAVITY};
use crate::math::Isometry;
use crate::world::World;
use std::sync::Arc;
use threers::animation::{AnimationClip, KeyframeTrack, TrackTarget};
use threers::core::{BufferAttribute, BufferGeometry, Mesh, Object3D, ObjectArena, ObjectId};
use threers::materials::Material;
use threers::math::Vector3;
use threers::openscad::frame::ScadFrame;
use threers::openscad::{scad::Viewport, MechanismSpec, ScadPart};

/// Why a `.scad` file could not be turned into a running mechanism.
#[derive(Debug, Clone, PartialEq)]
pub enum MechanismError {
    /// The file could not be read or parsed.
    Scad(String),
    /// The model declared no `part()` at all, so there is nothing to simulate.
    NoParts,
    /// The declarations do not line up with each other.
    Declaration(ScadMechanismError),
    /// The parts could not be put together.
    Assembly(AssemblyError),
}

impl std::fmt::Display for MechanismError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Scad(e) => write!(f, "{e}"),
            Self::NoParts => write!(
                f,
                "the model declares no part() — there is nothing to simulate"
            ),
            Self::Declaration(e) => write!(f, "{e}"),
            Self::Assembly(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for MechanismError {}

impl From<ScadMechanismError> for MechanismError {
    fn from(e: ScadMechanismError) -> Self {
        Self::Declaration(e)
    }
}

impl From<AssemblyError> for MechanismError {
    fn from(e: AssemblyError) -> Self {
        Self::Assembly(e)
    }
}

/// A model, its mechanism, and the world it runs in.
///
/// See the [module docs](self).
pub struct ScadMechanism {
    /// The parts and mates, for checking and for driving by hand.
    pub assembly: Assembly,
    /// The simulation. Gravity is [`SCAD_GRAVITY`] — z down, as the model was
    /// drawn — rather than the y-down default.
    pub world: World,
    /// Each part's drawable pieces, in the part's own frame, evaluated once.
    ///
    /// A part split by `color()` keeps its pieces, so a red lid renders red.
    display: Vec<Vec<ScadPart>>,
    frames: usize,
    fps: u32,
    settle: f32,
    viewport: Viewport,
}

impl std::fmt::Debug for ScadMechanism {
    /// A summary rather than a dump: the world alone is thousands of lines, and
    /// what anyone wants from `{:?}` here is how big the mechanism is.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ScadMechanism")
            .field("parts", &self.assembly.parts().len())
            .field("mates", &self.assembly.mates().len())
            .field("frames", &self.frames)
            .field("fps", &self.fps)
            .finish()
    }
}

impl ScadMechanism {
    /// Read a mechanism from OpenSCAD source and get it ready to run.
    pub fn from_source(src: &str) -> Result<Self, MechanismError> {
        let spec = threers::parse_scad_mechanism(src).map_err(MechanismError::Scad)?;
        Self::from_spec(&spec)
    }

    /// Read one from a file, resolving `include`/`use` beside it.
    pub fn from_file(path: impl AsRef<std::path::Path>) -> Result<Self, MechanismError> {
        let spec = threers::parse_scad_mechanism_file(path).map_err(MechanismError::Scad)?;
        Self::from_spec(&spec)
    }

    /// Build from a spec that has already been read.
    ///
    /// The parts are evaluated here, once: each is split into its `color()`
    /// pieces for drawing and the pieces are concatenated back into the one
    /// mesh the collider fits to. They are the same triangles either way, so
    /// the thing you see and the thing you hit cannot disagree.
    ///
    /// # Which boolean kernel
    ///
    /// The watertight one, and not as a matter of taste. Measured on a block
    /// with a bore through it, `difference()` on the float kernel produced
    /// **119,334** vertices in 113 ms where the exact kernel produced **1,608**
    /// in 10 ms — and on a cylinder unioned with a box it produced nothing at
    /// all, or 42,210 vertices, depending on which operand came first. A
    /// mechanism's geometry becomes its colliders and its interference checks;
    /// a mesh that is wrong or seventy times larger than it needs to be is not
    /// a saving.
    pub fn from_spec(spec: &MechanismSpec) -> Result<Self, MechanismError> {
        if spec.parts.is_empty() {
            return Err(MechanismError::NoParts);
        }

        let mut display: Vec<Vec<ScadPart>> = Vec::with_capacity(spec.parts.len());
        let assembly = Assembly::from_scad_parts(spec, |part| {
            let pieces = part.solid.clone().parts();
            let merged = merge_pieces(&pieces);
            display.push(pieces);
            merged
        })?;

        let mut world = World::new();
        world.gravity = SCAD_GRAVITY;

        let mut mech = Self {
            assembly,
            world,
            display,
            frames: 120,
            fps: 60,
            settle: 0.0,
            viewport: Viewport::default(),
        };
        // The model drew its parts in place, so this normally moves nothing —
        // and says so loudly when the declarations and the drawing disagree.
        mech.assembly.solve()?;
        mech.assembly.build(&mut mech.world)?;
        Ok(mech)
    }

    /// How many frames to produce. Defaults to 120.
    pub fn frames(mut self, frames: usize) -> Self {
        self.frames = frames.max(1);
        self
    }

    /// Frame rate, which is also the simulation's step rate. Defaults to 60.
    pub fn fps(mut self, fps: u32) -> Self {
        self.fps = fps.max(1);
        self
    }

    /// Let the mechanism settle for this long before the first frame.
    ///
    /// A mechanism built exactly on its mates starts with nothing touching
    /// anything, and the first few frames are it finding its own weight. Half a
    /// second of settling puts that before the camera rather than in shot.
    pub fn settle(mut self, seconds: f32) -> Self {
        self.settle = seconds.max(0.0);
        self
    }

    /// The camera the model asked for, if it set `$vp*`.
    pub fn viewport(mut self, viewport: Viewport) -> Self {
        self.viewport = viewport;
        self
    }

    /// Say the model is drawn in millimetres, as most CAD models are.
    ///
    /// Densities follow: in millimetres, `density = 0.0012` is the 1.2 g/cm³ of
    /// printed plastic, and the masses that come out are grams.
    pub fn millimetres(self) -> Self {
        self.units(1000.0)
    }

    /// Say how many of the model's units make a metre.
    ///
    /// Gravity is the obvious thing this changes, and not the only one — which
    /// is the whole reason this exists rather than a line setting `gravity`.
    ///
    /// The solver carries half a dozen constants that are *lengths*: how much
    /// overlap to leave uncorrected, how fast overlap may be pushed apart, the
    /// impact speed below which nothing bounces, the speed below which a body
    /// counts as still. Every one is tuned for a scene measured in metres. Drawn
    /// in millimetres, the same numbers mean a five-micron slop and a
    /// three-millimetre-per-second recovery — a thousand times tighter and
    /// slower than intended — and the mechanism reads as sluggish or stuck for
    /// reasons that look nothing like a units mistake.
    ///
    /// Measured: a crank motored at 120°/s in a millimetre model reached
    /// 1.1°/s with the metre-scale defaults, and its own 120 once they were
    /// scaled.
    ///
    /// Counts and fractions — iteration budgets, the Baumgarte factor, the
    /// angular thresholds, times — are left alone, because those are not
    /// lengths.
    pub fn units(mut self, per_metre: f32) -> Self {
        let s = if per_metre > 0.0 { per_metre } else { 1.0 };
        self.world.gravity = SCAD_GRAVITY * s;
        self.world.solver_config.penetration_slop *= s;
        self.world.solver_config.max_correction_velocity *= s;
        self.world.solver_config.restitution_threshold *= s;
        self.world.sleep.linear_threshold *= s;
        self
    }

    /// How long the animation lasts, in seconds.
    pub fn duration(&self) -> f32 {
        self.frames as f32 / self.fps as f32
    }

    /// How many frames the run produces.
    pub fn frame_count(&self) -> usize {
        self.frames
    }

    /// Steps per second, which is also the frame rate.
    pub fn rate(&self) -> u32 {
        self.fps
    }

    /// Advance drives and simulation by one frame's worth of time.
    ///
    /// Exposed for callers driving their own loop — a live viewer, or a test
    /// that wants to poke the world between steps. [`Self::simulate`] is this
    /// in a loop with a snapshot after each one.
    pub fn step(&mut self) {
        let dt = 1.0 / self.fps as f32;
        self.assembly.update(dt, &mut self.world);
        self.world.step(dt);
    }

    /// The model as it stands right now, ready to render.
    pub fn snapshot(&self, index: usize, t: f64) -> ScadFrame {
        let mut parts = Vec::new();
        for (i, pieces) in self.display.iter().enumerate() {
            let Some(body) = self
                .assembly
                .body_of(crate::assembly::PartId::from_index(i))
                .and_then(|id| self.world.body(id))
            else {
                continue;
            };
            let pose = body.position;
            for piece in pieces {
                parts.push(ScadPart {
                    geometry: placed(&piece.geometry, &pose),
                    color: piece.color,
                });
            }
        }
        ScadFrame {
            index,
            t,
            parts: Arc::new(parts),
            viewport: self.viewport,
        }
    }

    /// Run the whole animation, recording where each part went.
    ///
    /// The cheap way, and the one to reach for. A part is rigid, so a frame is
    /// seven numbers rather than a copy of its mesh — see [`crate::mechanism::PoseTrack`], which
    /// can hand the result to a scene graph without touching a vertex, or bake
    /// frames on demand for the renderer that wants geometry.
    ///
    /// ```no_run
    /// # use threers_physics::mechanism::ScadMechanism;
    /// # let mut mech = ScadMechanism::from_source("part(\"a\") cube(1);")?;
    /// let track = mech.record();
    /// println!("{} frames, {:.1} MB if baked", track.frames(),
    ///          track.baked_bytes() as f64 / 1e6);
    /// # Ok::<(), threers_physics::mechanism::MechanismError>(())
    /// ```
    pub fn record(&mut self) -> PoseTrack {
        self.settle_in();
        let mut parts: Vec<TrackedPart> = self
            .display
            .iter()
            .enumerate()
            .map(|(i, pieces)| TrackedPart {
                name: self
                    .assembly
                    .part_of(crate::assembly::PartId::from_index(i))
                    .map(|p| p.name.clone())
                    .unwrap_or_else(|| format!("part {i}")),
                pieces: pieces.clone(),
                poses: Vec::with_capacity(self.frames),
            })
            .collect();

        for _ in 0..self.frames {
            for (i, part) in parts.iter_mut().enumerate() {
                let pose = self
                    .assembly
                    .body_of(crate::assembly::PartId::from_index(i))
                    .and_then(|b| self.world.body(b))
                    .map(|b| b.position)
                    .unwrap_or(Isometry::IDENTITY);
                part.poses.push(pose);
            }
            self.step();
        }

        PoseTrack {
            parts,
            fps: self.fps,
            viewport: self.viewport,
        }
    }

    /// Run the whole animation and collect every frame, geometry and all.
    ///
    /// [`Self::record`] then [`crate::mechanism::PoseTrack::baked`]. Kept because the render path
    /// takes `ScadFrame`s — but on anything but a small model, read
    /// [`PoseTrack::baked_bytes`] before calling this: 120 frames of a
    /// 40,000-triangle assembly is 172 MB of geometry describing a motion that
    /// is 7 kB of poses.
    pub fn simulate(&mut self) -> Vec<ScadFrame> {
        self.record().baked()
    }

    fn settle_in(&mut self) {
        let dt = 1.0 / self.fps as f32;
        let mut settled = 0.0;
        while settled < self.settle {
            self.world.step(dt);
            settled += dt;
        }
    }

    /// Put the parts in a live scene, driven by the bodies.
    ///
    /// For a viewer rather than a recording: each part gets a node, the body
    /// that moves it is bound to that node, and one
    /// [`World::sync_to_scene`](crate::world::World::sync_to_scene) per frame
    /// moves them. No geometry is touched after this call.
    ///
    /// Returns one node per part, in part order.
    pub fn spawn(
        &mut self,
        arena: &mut ObjectArena,
        parent: Option<ObjectId>,
        mut material: impl FnMut(&ScadPart) -> Material,
    ) -> Vec<ObjectId> {
        let mut nodes = Vec::with_capacity(self.display.len());
        for (i, pieces) in self.display.iter().enumerate() {
            let node = arena.insert(Object3D::group());
            if let Some(parent) = parent {
                arena.add_child(parent, node);
            }
            for piece in pieces {
                let mesh = arena.insert(Object3D::mesh(Mesh::new(
                    piece.geometry.clone(),
                    material(piece),
                )));
                arena.add_child(node, mesh);
            }
            // Bind the body to the node, so `sync_to_scene` drives it.
            if let Some(body) = self
                .assembly
                .body_of(crate::assembly::PartId::from_index(i))
                .and_then(|b| self.world.body_mut(b))
            {
                body.scene_object = Some(node);
            }
            nodes.push(node);
        }
        nodes
    }
}

/// A recorded run: the geometry once, and where each part was in each frame.
///
/// The representation an assembly's animation actually wants. A part is rigid —
/// the solver moves it and never reshapes it — so a frame is a *pose*, seven
/// numbers, and not a copy of the mesh. Baking the pose into the vertices turns
/// a few kilobytes into hundreds of megabytes and re-derives every frame what a
/// matrix already said.
///
/// Measured on a two-part assembly of 39,786 triangles, 120 frames:
///
/// | | |
/// |---|---|
/// | baked into [`threers::openscad::frame::ScadFrame`]s | 172 MB |
/// | as poses | 6.7 kB |
///
/// Three ways to use one:
///
/// * [`spawn`](Self::spawn) then [`apply`](Self::apply) — put the meshes in a
///   scene once and move the nodes per frame. Nothing is copied and no vertex is
///   touched, which is what a scene graph is for.
/// * [`frame`](Self::frame) — bake one frame on demand, for the
///   [`ScadRender`](threers::openscad::animate::ScadRender) path. Render it,
///   drop it, bake the next.
/// * [`baked`](Self::baked) — all of them at once, when something needs the
///   whole sequence in hand.
#[derive(Debug, Clone)]
pub struct PoseTrack {
    parts: Vec<TrackedPart>,
    fps: u32,
    viewport: Viewport,
}

/// One part of a [`crate::mechanism::PoseTrack`]: its drawable pieces, and where it went.
#[derive(Debug, Clone)]
pub struct TrackedPart {
    pub name: String,
    /// The drawable pieces in the part's own frame, evaluated once. A part split
    /// by `color()` keeps its pieces, and they all move together.
    pub pieces: Vec<ScadPart>,
    /// Where the part was, one entry per frame.
    pub poses: Vec<Isometry>,
}

impl TrackedPart {
    /// Bounds of the part in its own frame.
    fn local_bounds(&self) -> Option<([f32; 3], [f32; 3])> {
        let mut lo = [f32::MAX; 3];
        let mut hi = [f32::MIN; 3];
        let mut any = false;
        for piece in &self.pieces {
            let (a, b) = threers::openscad::geometry_bounds(&piece.geometry);
            if a[0] > b[0] {
                continue;
            }
            any = true;
            for k in 0..3 {
                lo[k] = lo[k].min(a[k]);
                hi[k] = hi[k].max(b[k]);
            }
        }
        any.then_some((lo, hi))
    }
}

impl PoseTrack {
    pub fn frames(&self) -> usize {
        self.parts.first().map_or(0, |p| p.poses.len())
    }

    pub fn fps(&self) -> u32 {
        self.fps
    }

    pub fn duration(&self) -> f32 {
        self.frames() as f32 / self.fps as f32
    }

    pub fn parts(&self) -> &[TrackedPart] {
        &self.parts
    }

    /// Where a part was in a frame.
    pub fn pose(&self, part: usize, frame: usize) -> Option<Isometry> {
        self.parts.get(part)?.poses.get(frame).copied()
    }

    /// What the whole run sweeps through, over every frame.
    ///
    /// Computed from the poses and each part's own bounds, so framing a camera
    /// on the animation costs nothing — the alternative is baking every frame to
    /// ask it how big it is.
    pub fn bounds(&self) -> Option<([f32; 3], [f32; 3])> {
        let mut lo = [f32::MAX; 3];
        let mut hi = [f32::MIN; 3];
        let mut any = false;
        for part in &self.parts {
            let Some((plo, phi)) = part.local_bounds() else {
                continue;
            };
            for pose in &part.poses {
                for i in 0..8 {
                    let corner = Vector3::new(
                        if i & 1 == 0 { plo[0] } else { phi[0] },
                        if i & 2 == 0 { plo[1] } else { phi[1] },
                        if i & 4 == 0 { plo[2] } else { phi[2] },
                    );
                    let p = pose.transform_point(corner);
                    any = true;
                    for (k, v) in [p.x, p.y, p.z].iter().enumerate() {
                        lo[k] = lo[k].min(*v);
                        hi[k] = hi[k].max(*v);
                    }
                }
            }
        }
        any.then_some((lo, hi))
    }

    /// Bake one frame into the geometry a [`threers::openscad::frame::ScadFrame`] carries.
    ///
    /// The expensive direction, and the one to reach for a frame at a time
    /// rather than all at once — see [`Self::baked`].
    pub fn frame(&self, index: usize) -> ScadFrame {
        let mut parts = Vec::new();
        for part in &self.parts {
            let Some(pose) = part.poses.get(index) else {
                continue;
            };
            for piece in &part.pieces {
                parts.push(ScadPart {
                    geometry: placed(&piece.geometry, pose),
                    color: piece.color,
                });
            }
        }
        ScadFrame {
            index,
            t: if self.frames() > 0 {
                index as f64 / self.frames() as f64
            } else {
                0.0
            },
            parts: Arc::new(parts),
            viewport: self.viewport,
        }
    }

    /// Bake every frame.
    ///
    /// Costs [`Self::baked_bytes`], which for anything but a small model is a
    /// number worth reading first.
    pub fn baked(&self) -> Vec<ScadFrame> {
        (0..self.frames()).map(|i| self.frame(i)).collect()
    }

    /// What [`Self::baked`] would allocate, in bytes.
    ///
    /// Positions only, so the truth is somewhat higher. Enough to answer "should
    /// I bake this".
    pub fn baked_bytes(&self) -> usize {
        let per_frame: usize = self
            .parts
            .iter()
            .flat_map(|p| p.pieces.iter())
            .filter_map(|piece| piece.geometry.get_attribute("position"))
            .map(|a| a.array.len() * std::mem::size_of::<f32>())
            .sum();
        per_frame * self.frames()
    }

    /// Put the parts in a scene, once. One node per part; move it with
    /// [`Self::apply`].
    ///
    /// Every piece of a part hangs off one node, so a part split by `color()`
    /// still moves as one thing.
    pub fn spawn(
        &self,
        arena: &mut ObjectArena,
        parent: Option<ObjectId>,
        material: Material,
    ) -> Vec<ObjectId> {
        self.spawn_with(arena, parent, |_| material.clone())
    }

    /// The same, choosing a material per drawable piece — which is how the
    /// model's own `color()` reaches the picture.
    pub fn spawn_with(
        &self,
        arena: &mut ObjectArena,
        parent: Option<ObjectId>,
        mut material: impl FnMut(&ScadPart) -> Material,
    ) -> Vec<ObjectId> {
        let mut nodes = Vec::with_capacity(self.parts.len());
        for part in &self.parts {
            let node = arena.insert(Object3D::group());
            if let Some(parent) = parent {
                arena.add_child(parent, node);
            }
            for piece in &part.pieces {
                let mesh = arena.insert(Object3D::mesh(Mesh::new(
                    piece.geometry.clone(),
                    material(piece),
                )));
                arena.add_child(node, mesh);
            }
            nodes.push(node);
        }
        self.apply(arena, &nodes, 0);
        nodes
    }

    /// Turn the run into an [`threers::animation::AnimationClip`] driving `nodes`.
    ///
    /// The point at which a simulated mechanism stops needing a simulation. A
    /// clip plays through [`AnimationMixer`](threers::animation::AnimationMixer)
    /// like any other — at any rate, looped, blended, retimed — and the solver,
    /// the colliders and the whole physics world can go away. What was a
    /// mechanism becomes an animation asset.
    ///
    /// One position track and one rotation track per part, keyed at every frame.
    /// [`to_clip_simplified`](Self::to_clip_simplified) is usually the better
    /// call: most of those keys say nothing the interpolation would not have
    /// worked out.
    pub fn to_clip(&self, nodes: &[ObjectId], name: impl Into<String>) -> AnimationClip {
        self.build_clip(nodes, name, None)
    }

    /// The same, with the keys the interpolation would have reproduced dropped.
    ///
    /// A mechanism's motion is smooth by construction — it is a solver following
    /// a drive, not a hand-animated curve — so most frames sit on the line
    /// between their neighbours. Keeping them stores the sampling rate rather
    /// than the motion.
    ///
    /// Every key that survives is one the interpolation could not have guessed
    /// within [`Tolerance`]; every key dropped is one it can. Sampling the
    /// result at the original frame times therefore reproduces them to within
    /// that tolerance, which is a promise the reduction keeps by construction
    /// rather than by luck.
    pub fn to_clip_simplified(
        &self,
        nodes: &[ObjectId],
        name: impl Into<String>,
        tolerance: Tolerance,
    ) -> AnimationClip {
        self.build_clip(nodes, name, Some(tolerance))
    }

    fn build_clip(
        &self,
        nodes: &[ObjectId],
        name: impl Into<String>,
        tolerance: Option<Tolerance>,
    ) -> AnimationClip {
        let frames = self.frames();
        let step = 1.0 / self.fps as f32;
        // A position tolerance in model units, from one given as a fraction of
        // the run's own size — so the same number means the same thing whether
        // the model is drawn in metres or millimetres.
        let scale = self
            .bounds()
            .map(|(lo, hi)| {
                ((hi[0] - lo[0]).powi(2) + (hi[1] - lo[1]).powi(2) + (hi[2] - lo[2]).powi(2)).sqrt()
            })
            .unwrap_or(1.0);

        let mut tracks = Vec::with_capacity(self.parts.len() * 2);
        for (part, &node) in self.parts.iter().zip(nodes.iter()) {
            if part.poses.is_empty() {
                continue;
            }
            let keys = match tolerance {
                Some(t) => reduce(&part.poses, t.position * scale, t.rotation.to_radians()),
                None => (0..part.poses.len()).collect(),
            };
            let times: Vec<f32> = keys.iter().map(|&k| k as f32 * step).collect();
            tracks.push(KeyframeTrack::vector(
                node,
                TrackTarget::Position,
                times.clone(),
                keys.iter().map(|&k| part.poses[k].translation).collect(),
            ));
            tracks.push(KeyframeTrack::quaternion(
                node,
                TrackTarget::Quaternion,
                times,
                keys.iter().map(|&k| part.poses[k].rotation).collect(),
            ));
        }

        AnimationClip::new(name, frames.saturating_sub(1) as f32 * step, tracks)
    }

    /// Move the spawned nodes to a frame.
    ///
    /// Seven numbers per part. No geometry is read, copied or transformed —
    /// which is the whole point, and the difference between an animation that
    /// fits in a cache line per part and one that does not fit in memory.
    pub fn apply(&self, arena: &mut ObjectArena, nodes: &[ObjectId], frame: usize) {
        for (part, &node) in self.parts.iter().zip(nodes.iter()) {
            let Some(pose) = part.poses.get(frame) else {
                continue;
            };
            let Some(object) = arena.get_mut(node) else {
                continue;
            };
            object.position = pose.translation;
            object.quaternion = pose.rotation;
            object.update_matrix();
        }
    }
}

/// How far a dropped keyframe may pull the interpolated result.
///
/// The position figure is a **fraction of the run's own size**, not a distance.
/// A tolerance in millimetres is a statement about what the model is drawn in
/// rather than about how accurate the animation is, and the same number would
/// throw away everything on a model in metres and nothing on one in microns.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Tolerance {
    /// As a fraction of the diagonal of everything the run sweeps through.
    pub position: f32,
    /// In degrees.
    pub rotation: f32,
}

impl Default for Tolerance {
    /// A thousandth of the model and a quarter of a degree — below what a
    /// render at any sane resolution can show.
    fn default() -> Self {
        Self {
            position: 1e-3,
            rotation: 0.25,
        }
    }
}

impl Tolerance {
    pub fn new(position: f32, rotation: f32) -> Self {
        Self {
            position: position.max(0.0),
            rotation: rotation.max(0.0),
        }
    }
}

/// Which frames have to be kept for interpolation to reproduce the rest.
///
/// Greedy: hold an anchor, extend the span as far as the straight line from the
/// anchor still passes through every pose in between, and plant a key where it
/// stops doing so. Linear in the keys it keeps rather than in the frames it is
/// given, which for a mechanism — smooth by construction, being a solver
/// following a drive — is a very different number.
fn reduce(poses: &[Isometry], position: f32, rotation: f32) -> Vec<usize> {
    if poses.len() <= 2 {
        return (0..poses.len()).collect();
    }
    let fits = |a: usize, b: usize| -> bool {
        let span = (b - a) as f32;
        for i in (a + 1)..b {
            let u = (i - a) as f32 / span;
            let guessed = poses[a].translation.lerp(poses[b].translation, u);
            if (guessed - poses[i].translation).length() > position {
                return false;
            }
            let turned = poses[a].rotation.slerp(poses[b].rotation, u);
            // Angle between two rotations, taking the short way round: a
            // quaternion and its negation are the same rotation.
            let angle = 2.0 * turned.dot(poses[i].rotation).abs().clamp(0.0, 1.0).acos();
            if angle > rotation {
                return false;
            }
        }
        true
    };

    let mut keys = vec![0usize];
    let mut anchor = 0usize;
    for end in 2..poses.len() {
        if !fits(anchor, end) {
            keys.push(end - 1);
            anchor = end - 1;
        }
    }
    let last = poses.len() - 1;
    if *keys.last().unwrap_or(&0) != last {
        keys.push(last);
    }
    keys
}

/// Concatenate a part's colour pieces back into one mesh.
///
/// They came from one model split by colour, so they are disjoint by
/// construction and no boolean is needed to put them back.
fn merge_pieces(pieces: &[ScadPart]) -> BufferGeometry {
    let mut out: Option<BufferGeometry> = None;
    for piece in pieces {
        out = Some(match out {
            None => piece.geometry.clone(),
            Some(acc) => threers::openscad::concat_geometry(&acc, &piece.geometry),
        });
    }
    out.unwrap_or_default()
}

/// The same mesh, moved to where its body is.
///
/// Positions transform and normals rotate; nothing is recomputed. Recomputing
/// them would re-average the model's own shading every frame, which shows up as
/// a mesh that changes appearance as it moves.
fn placed(geometry: &BufferGeometry, pose: &Isometry) -> BufferGeometry {
    let mut out = geometry.clone();
    if let Some(attr) = geometry.get_attribute("position") {
        let mut array = attr.array.clone();
        for v in array.chunks_exact_mut(3) {
            let p = pose.transform_point(Vector3::new(v[0], v[1], v[2]));
            v[0] = p.x;
            v[1] = p.y;
            v[2] = p.z;
        }
        // Through `set_attribute` so the cached bounds and any acceleration
        // sidecar are invalidated with it.
        out.set_attribute("position", BufferAttribute::new(array, attr.item_size));
    }
    if let Some(attr) = geometry.get_attribute("normal") {
        let mut array = attr.array.clone();
        for v in array.chunks_exact_mut(3) {
            let n = pose.transform_vector(Vector3::new(v[0], v[1], v[2]));
            v[0] = n.x;
            v[1] = n.y;
            v[2] = n.z;
        }
        out.set_attribute("normal", BufferAttribute::new(array, attr.item_size));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const LID_BOX: &str = r#"
        part("box", fixed = true) cube([0.30, 0.20, 0.16]);
        part("lid") translate([0, 0, 0.16]) cube([0.30, 0.20, 0.012]);
        hinge("lid_pivot", parts = ["lid", "box"],
              at = [0.15, 0, 0.16], axis = [1, 0, 0], range = [0, 105]);
        drive("lid_pivot", to = 105, over = 1.0);
    "#;

    #[test]
    fn a_model_simulates_into_frames() {
        let mut mech = ScadMechanism::from_source(LID_BOX).unwrap().frames(90).fps(60);
        assert_eq!(mech.duration(), 1.5);

        let frames = mech.simulate();
        assert_eq!(frames.len(), 90);
        // Two parts, neither coloured, so one drawable piece each.
        assert_eq!(frames[0].parts.len(), 2);
        assert!(frames[0].parts.iter().all(|p| p
            .geometry
            .get_attribute("position")
            .is_some_and(|a| a.count() > 0)));
    }

    #[test]
    fn the_lid_actually_moves_between_frames() {
        let mut mech = ScadMechanism::from_source(LID_BOX).unwrap().frames(90).fps(60);
        let frames = mech.simulate();

        let height = |f: &ScadFrame| {
            f.bounds().map(|(_, max)| max[2]).unwrap_or(0.0)
        };
        let first = height(&frames[0]);
        let last = height(&frames[frames.len() - 1]);
        assert!(
            last > first + 0.1,
            "the lid never opened: {first} then {last}"
        );
    }

    #[test]
    fn the_pose_comes_from_the_solver_and_stops_at_the_limit() {
        let mut mech = ScadMechanism::from_source(
            // Asks for 200°, which the hinge does not have.
            &LID_BOX.replace("to = 105", "to = 200"),
        )
        .unwrap()
        .frames(180)
        .fps(60);
        mech.simulate();

        let hinge = mech.assembly.mate_named("lid_pivot").unwrap();
        let angle = mech.assembly.coordinate(hinge, &mech.world).unwrap();
        assert!(
            (angle - 105f32.to_radians()).abs() < 5f32.to_radians(),
            "the limit should have held it at 105°, not {}°",
            angle.to_degrees()
        );
    }

    /// A gear train declared entirely in the model: two wheels, each on its own
    /// hinge, coupled by a ratio. This is the whole stack — SCAD declaration to
    /// mate to solver constraint — on the one mate kind that constrains a
    /// *rate* rather than a pose.
    #[test]
    fn a_geared_pair_declared_in_scad_turns_at_its_ratio() {
        let mut mech = ScadMechanism::from_source(
            r#"
            part("frame", fixed = true) translate([-0.02, -0.02, -0.02]) cube([0.04, 0.04, 0.04]);

            part("pinion") translate([0, 0, 0]) cylinder(h = 0.02, r = 0.02, center = true);
            part("wheel")  translate([0.06, 0, 0]) cylinder(h = 0.02, r = 0.04, center = true);

            hinge("pinion_axle", parts = ["pinion", "frame"],
                  at = [0, 0, 0], axis = [0, 0, 1]);
            hinge("wheel_axle",  parts = ["wheel", "frame"],
                  at = [0.06, 0, 0], axis = [0, 0, 1]);

            // A 2:1 reduction between meshing gears, so they counter-rotate.
            gear("mesh", parts = ["pinion", "wheel"], at = [0.03, 0, 0],
                 axis = [0, 0, 1], ratio = -2);

            // A speed and no target: a motor. A shaft that makes full turns
            // cannot be driven to a position, because the angle it would be
            // measured against wraps at ±360°.
            drive("pinion_axle", speed = 120, torque = 5);
            "#,
        )
        .unwrap()
        .frames(120)
        .fps(60);

        // The gear couples rates, so it neither places anything nor takes a
        // range — and the check says so instead of the mate quietly not being
        // there.
        let report = mech.assembly.check();
        assert!(report.unsatisfied.is_empty(), "{report:?}");
        assert!(report.ignored_limits.is_empty(), "{report:?}");

        mech.simulate();

        // Rates, not angles: the angles have wrapped by now, and the rate is
        // what a gear constrains anyway.
        let pinion = mech.assembly.mate_named("pinion_axle").unwrap();
        let wheel = mech.assembly.mate_named("wheel_axle").unwrap();
        let fast = mech.assembly.speed(pinion, &mech.world).unwrap();
        let slow = mech.assembly.speed(wheel, &mech.world).unwrap();

        assert!(
            (fast - 120f32.to_radians()).abs() < 0.2,
            "the motor asked for 120°/s, the pinion ran at {}°/s",
            fast.to_degrees()
        );
        // Half the speed, the other way.
        assert!(
            (slow + fast / 2.0).abs() < 0.2,
            "a -2:1 pair should give {}°/s against the pinion's {}°/s, got {}°/s",
            (-fast / 2.0).to_degrees(),
            fast.to_degrees(),
            slow.to_degrees()
        );
    }

    #[test]
    fn a_range_on_a_mate_that_cannot_use_one_is_reported() {
        let mech = ScadMechanism::from_source(
            r#"
            part("a", fixed = true) cube([0.10, 0.10, 0.02]);
            part("b") translate([0, 0, 0.02]) cube([0.10, 0.10, 0.02]);
            ball("socket", parts = ["b", "a"], at = [0.05, 0.05, 0.02], range = [0, 45]);
            "#,
        )
        .unwrap();
        let report = mech.assembly.check();
        assert_eq!(report.ignored_limits.len(), 1, "{report:?}");
        assert!(!report.clear(), "a dropped limit must not read as a pass");
    }

    #[test]
    fn a_model_with_no_parts_says_so() {
        let err = ScadMechanism::from_source("cube(10);").unwrap_err();
        assert_eq!(err, MechanismError::NoParts);
        assert!(err.to_string().contains("part()"));
    }

    #[test]
    fn colours_survive_into_the_frames() {
        let mut mech = ScadMechanism::from_source(
            r#"
            part("base", fixed = true) color("red") cube([0.20, 0.20, 0.02]);
            part("top") translate([0, 0, 0.02]) color("blue") cube([0.20, 0.20, 0.02]);
            weld("w", parts = ["top", "base"], at = [0, 0, 0.02], axis = [0, 0, 1]);
            "#,
        )
        .unwrap()
        .frames(2);
        let frames = mech.simulate();
        let colours: Vec<_> = frames[0].parts.iter().filter_map(|p| p.color).collect();
        assert_eq!(colours.len(), 2, "the model's colours were dropped");
        assert!(colours.iter().any(|c| c[0] > 0.5 && c[2] < 0.5), "no red");
        assert!(colours.iter().any(|c| c[2] > 0.5 && c[0] < 0.5), "no blue");
    }

    #[test]
    fn geometry_is_evaluated_once_and_only_moved() {
        // The triangle count must not change between frames: a mechanism moves
        // its parts, it does not rebuild them.
        let mut mech = ScadMechanism::from_source(LID_BOX).unwrap().frames(30);
        let frames = mech.simulate();
        let count = |f: &ScadFrame| f.triangle_count();
        assert_eq!(count(&frames[0]), count(&frames[29]));
        assert!(count(&frames[0]) > 0);
    }
}
