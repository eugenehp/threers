//! Particles: emitters, a physical simulation, and the things built on it —
//! falling and settling snow, pouring and drifting sand, and presets for fire,
//! smoke, sparks, rain, dust and fountains.
//!
//! The simulation runs on the CPU, so it is the same on every backend (native,
//! WebGPU, WebGL2) and testable without a GPU; drawing is one instanced draw of
//! camera-facing quads ([`render`]).
//!
//! # The physics
//!
//! A particle moves under gravity and through air. Air pulls on it by
//! quadratic drag, `k |u| u` with `u` the air's velocity relative to the
//! particle, and `k` set by the particle's **terminal speed** — `k = g / v_t²`,
//! the speed at which drag balances gravity. That one number is what makes a
//! snowflake (about 1 m/s) drift with every eddy and a grain of sand (a few
//! m/s) or a raindrop (about 9 m/s) plough straight through them. A particle
//! with no terminal speed feels no air at all: a spark, a fountain's water.
//!
//! The air is the wind plus **turbulence**: a sum of divergence-free modes (the
//! curl of a few plane waves, so the air neither piles up nor empties anywhere)
//! about `turbulence.scale` across, carried downwind at the wind's speed. Drag
//! is integrated implicitly, so a light particle in a strong gust stays stable
//! at any step and settles at exactly its terminal speed.
//!
//! # Settling
//!
//! A particle that lands can bounce, stick, die, or **deposit**: hand its
//! volume to a [`Heightfield`] — a grid of material over the ground — which
//! slumps wherever it is steeper than the material's angle of repose. That is
//! how [`snow::SnowCover`] accumulates on roofs and ground and
//! [`sand::SandBed`] builds a pile at sand's 34°, and the bed also drifts into
//! dunes under wind (Werner's slab model).
//!
//! ```
//! use threers::particles::{presets, ParticleSystem};
//! use threers::math::Vector3;
//!
//! let mut sparks = presets::sparks(Vector3::new(0.0, 1.0, 0.0));
//! for _ in 0..60 {
//!     sparks.step(1.0 / 60.0, None);
//! }
//! assert!(sparks.len() > 0);
//! ```

pub mod animator;
pub mod dunes;
pub mod heightfield;
pub mod presets;
pub mod render;
pub mod sand;
pub mod snow;
pub mod surface;
pub mod thermal;
pub mod volume;
pub mod weather;

pub use animator::{Layer, ParticleAnimator};
pub use heightfield::Heightfield;
pub use render::{particles_object, ParticleBlend, ParticleEffect, ParticleKind, ParticleLook};
pub use thermal::Thermal;
pub use weather::Weather;

use crate::core::BufferGeometry;
use crate::math::{Color, Vector3};

/// A value drawn uniformly from `min..=max` for each new particle.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Range {
    pub min: f32,
    pub max: f32,
}

impl Range {
    pub const fn new(min: f32, max: f32) -> Self {
        Self { min, max }
    }
    /// The same value every time.
    pub const fn fixed(v: f32) -> Self {
        Self { min: v, max: v }
    }
    fn sample(&self, rng: &mut Rng) -> f32 {
        // Drawn either way, so a fixed range uses the generator as a varying
        // one does; but returned as given — `INFINITY - INFINITY` is NaN.
        let r = rng.next();
        if self.min == self.max {
            self.min
        } else {
            self.min + (self.max - self.min) * r
        }
    }
}

/// Where an emitter puts new particles, in its own frame (centred on
/// [`Emitter::position`]).
#[derive(Debug, Clone)]
pub enum EmitterShape {
    /// All from one point.
    Point,
    /// Anywhere inside a ball (`surface`: on its surface only).
    Sphere { radius: f32, surface: bool },
    /// Anywhere inside a box of these half-extents.
    Box { half: Vector3 },
    /// Anywhere on a disc of this radius facing `normal`.
    Disc { radius: f32, normal: Vector3 },
    /// On a mesh's surface, by area: every triangle gets its share.
    Surface(SurfaceSampler),
    /// At one of these points, picked at random, give or take `jitter`
    /// metres: along a dune's crests, say.
    Points {
        points: std::sync::Arc<[Vector3]>,
        jitter: f32,
    },
}

/// Which way a new particle goes, at a speed drawn from [`Emitter::speed`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Direction {
    /// Away from the shape: outward from a sphere's centre, along a disc's or
    /// a triangle's normal, uniformly for a point or a box.
    Outward,
    /// Within `angle` radians of `axis`, uniformly over the cap.
    Cone { axis: Vector3, angle: f32 },
    /// Every direction alike.
    Any,
}

/// Particles in, at a rate and in bursts.
#[derive(Debug, Clone)]
pub struct Emitter {
    pub shape: EmitterShape,
    /// The shape's centre, in the world.
    pub position: Vector3,
    /// Particles a second, spread evenly over time.
    pub rate: f32,
    /// One-off bursts: at a time (seconds since the system began), a count.
    pub bursts: Vec<(f32, u32)>,
    pub direction: Direction,
    /// Speed along [`direction`](Self::direction), m/s.
    pub speed: Range,
    /// Added to every new particle's velocity: the emitter's own motion, or a
    /// wind it is born into.
    pub velocity: Vector3,
    /// Seconds a particle lives. `f32::INFINITY` lives until it lands or
    /// leaves.
    pub lifetime: Range,
    /// Drawn size, metres across.
    pub size: Range,
    /// Terminal speed in still air, m/s (see the module docs). 0: no drag.
    pub terminal_speed: Range,
    /// Spin, radians a second (drawn quads turn by it).
    pub spin: Range,
    /// Material volume one particle carries, m³, handed to a heightfield
    /// when it [deposits](Response::Deposit).
    pub volume: f32,
    pub enabled: bool,
    accumulated: f32,
    fired: usize,
}

impl Emitter {
    pub fn new(shape: EmitterShape, position: Vector3) -> Self {
        Self {
            shape,
            position,
            rate: 0.0,
            bursts: Vec::new(),
            direction: Direction::Outward,
            speed: Range::fixed(0.0),
            velocity: Vector3::ZERO,
            lifetime: Range::fixed(2.0),
            size: Range::fixed(0.05),
            terminal_speed: Range::fixed(0.0),
            spin: Range::fixed(0.0),
            volume: 0.0,
            enabled: true,
            accumulated: 0.0,
            fired: 0,
        }
    }

    /// Particles due between `t0` and `t0 + dt`: the rate's share (carried
    /// across steps, so a rate below one a frame still emits) and any bursts.
    fn due(&mut self, t0: f32, dt: f32) -> usize {
        if !self.enabled {
            return 0;
        }
        self.accumulated += self.rate.max(0.0) * dt;
        let n = self.accumulated.floor();
        self.accumulated -= n;
        let mut count = n as usize;
        while self.fired < self.bursts.len() && self.bursts[self.fired].0 < t0 + dt {
            count += self.bursts[self.fired].1 as usize;
            self.fired += 1;
        }
        count
    }

    /// A new particle's position and velocity.
    fn spawn(&self, rng: &mut Rng) -> (Vector3, Vector3) {
        let (offset, outward) = match &self.shape {
            EmitterShape::Point => (Vector3::ZERO, rng.unit_vector()),
            EmitterShape::Sphere { radius, surface } => {
                let d = rng.unit_vector();
                let r = if *surface {
                    *radius
                } else {
                    radius * rng.next().cbrt()
                };
                (d * r, d)
            }
            EmitterShape::Box { half } => (
                Vector3::new(
                    (rng.next() * 2.0 - 1.0) * half.x,
                    (rng.next() * 2.0 - 1.0) * half.y,
                    (rng.next() * 2.0 - 1.0) * half.z,
                ),
                rng.unit_vector(),
            ),
            EmitterShape::Disc { radius, normal } => {
                let n = normal.normalize();
                let (u, v) = basis(n);
                let r = radius * rng.next().sqrt();
                let a = std::f32::consts::TAU * rng.next();
                (u * (r * a.cos()) + v * (r * a.sin()), n)
            }
            EmitterShape::Surface(s) => s.sample(rng),
            EmitterShape::Points { points, jitter } => {
                if points.is_empty() {
                    (Vector3::ZERO, Vector3::new(0.0, 1.0, 0.0))
                } else {
                    let p =
                        points[((rng.next() * points.len() as f32) as usize).min(points.len() - 1)];
                    let j = Vector3::new(
                        rng.next() * 2.0 - 1.0,
                        rng.next() * 2.0 - 1.0,
                        rng.next() * 2.0 - 1.0,
                    );
                    (p + j * *jitter, Vector3::new(0.0, 1.0, 0.0))
                }
            }
        };
        let dir = match self.direction {
            Direction::Outward => outward,
            Direction::Any => rng.unit_vector(),
            Direction::Cone { axis, angle } => rng.in_cone(axis.normalize(), angle),
        };
        (
            self.position + offset,
            dir * self.speed.sample(rng) + self.velocity,
        )
    }
}

/// Uniform sampling over a mesh's surface by area, from its triangles.
#[derive(Debug, Clone)]
pub struct SurfaceSampler {
    triangles: Vec<[Vector3; 3]>,
    /// Running area, for picking a triangle by a uniform number.
    cumulative: Vec<f32>,
}

impl SurfaceSampler {
    /// The surface of `geometry`, in its own coordinates.
    pub fn new(geometry: &BufferGeometry) -> Self {
        let mut triangles = Vec::new();
        if let Some(p) = geometry.get_attribute("position") {
            let v = |i: usize| Vector3::new(p.array[i * 3], p.array[i * 3 + 1], p.array[i * 3 + 2]);
            let count = p.count();
            let index: Vec<usize> = match &geometry.index {
                Some(ix) => ix.iter().map(|&i| i as usize).collect(),
                None => (0..count).collect(),
            };
            for t in index.chunks_exact(3) {
                triangles.push([v(t[0]), v(t[1]), v(t[2])]);
            }
        }
        let mut cumulative = Vec::with_capacity(triangles.len());
        let mut total = 0.0;
        for [a, b, c] in &triangles {
            total += (*b - *a).cross(*c - *a).length() * 0.5;
            cumulative.push(total);
        }
        Self {
            triangles,
            cumulative,
        }
    }

    fn sample(&self, rng: &mut Rng) -> (Vector3, Vector3) {
        let Some(&total) = self.cumulative.last() else {
            return (Vector3::ZERO, Vector3::new(0.0, 1.0, 0.0));
        };
        let x = rng.next() * total;
        let i = self
            .cumulative
            .partition_point(|&c| c < x)
            .min(self.triangles.len() - 1);
        let [a, b, c] = self.triangles[i];
        let (mut u, mut v) = (rng.next(), rng.next());
        if u + v > 1.0 {
            u = 1.0 - u;
            v = 1.0 - v;
        }
        let n = (b - a).cross(c - a);
        let n = if n.length() > 0.0 {
            n.normalize()
        } else {
            Vector3::new(0.0, 1.0, 0.0)
        };
        (a + (b - a) * u + (c - a) * v, n)
    }
}

/// Turbulence: divergence-free air motion carried on the wind.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Turbulence {
    /// Typical speed of the eddies, m/s.
    pub strength: f32,
    /// Typical eddy size, metres.
    pub scale: f32,
    /// How fast the eddies change as they go by: 0 is frozen turbulence
    /// carried on the wind, 1 is a new eddy every `scale / strength` seconds.
    pub churn: f32,
    /// 0: eddies alike every way. 1: as in the surface layer near the ground
    /// in a wind — drawn out downwind and flattened by the ground, so gusts
    /// are strongest along the wind and weakest up and down, the speeds'
    /// spreads about 2.5 : 1.9 : 1.25 (Panofsky & Dutton's neutral surface
    /// layer). Still divergence-free: each eddy's wave vector is stretched
    /// and its flow kept square to it.
    pub anisotropy: f32,
}

impl Default for Turbulence {
    fn default() -> Self {
        Self {
            strength: 0.0,
            scale: 1.0,
            churn: 0.3,
            anisotropy: 0.0,
        }
    }
}

/// Something that pulls particles toward a point (negative `strength`
/// pushes) or swirls them round an axis.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Field {
    /// An acceleration of `strength` m/s² toward `position`, faded to
    /// nothing at `radius`.
    Attractor {
        position: Vector3,
        strength: f32,
        radius: f32,
    },
    /// Air swirling round the line through `position` along `axis` at
    /// `speed` m/s, faded to nothing at `radius` — a dust devil, a vortex.
    Vortex {
        position: Vector3,
        axis: Vector3,
        speed: f32,
        radius: f32,
    },
}

/// The wind's slowing toward the ground: the log law of the atmosphere's
/// surface layer, `u(z) = u_ref · ln(z / z₀) / ln(z_ref / z₀)` at height `z`
/// above `ground`, `z₀` the surface's roughness length. The wind given in
/// [`Forces::wind`] is the wind at `reference` metres.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BoundaryLayer {
    pub ground: f32,
    /// Metres: about 0.0005 over snow, 0.001 over sand, 0.03 over grass,
    /// 1 over a town.
    pub roughness: f32,
    /// Where the wind is as given; 10 m is the meteorologists'.
    pub reference: f32,
}

impl BoundaryLayer {
    /// The wind's share at height `y`: 0 at the ground, 1 at the reference.
    pub fn factor(&self, y: f32) -> f32 {
        let z0 = self.roughness.max(1e-5);
        let z = (y - self.ground).max(z0);
        ((z / z0).ln() / (self.reference.max(z0 * 2.0) / z0).ln()).max(0.0)
    }
}

/// Hot gas cooling as it rises and mixes: its buoyancy (the upward
/// [`Forces::gravity`] it is born with) fades with time constant `time`
/// toward `cold_gravity` — so a plume climbs fast, slows and spreads.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Cooling {
    pub time: f32,
    pub cold_gravity: Vector3,
}

/// What acts on every particle.
#[derive(Debug, Clone, PartialEq)]
pub struct Forces {
    /// m/s². Smoke and flame take an upward one: buoyancy.
    pub gravity: Vector3,
    /// The air's mean velocity, m/s (at the boundary layer's reference
    /// height, if there is one).
    pub wind: Vector3,
    pub turbulence: Turbulence,
    pub fields: Vec<Field>,
    /// Wind slowing toward the ground. `None`: the same at every height.
    pub boundary_layer: Option<BoundaryLayer>,
    /// Buoyancy fading as hot gas cools. `None`: gravity is what it is.
    pub cooling: Option<Cooling>,
    /// Flutter, m/s²: a falling flake or leaf sheds vortices from alternate
    /// edges and zig-zags as it falls, each particle at its own rate. 0: none.
    pub flutter: f32,
}

impl Default for Forces {
    fn default() -> Self {
        Self {
            gravity: Vector3::new(0.0, -9.81, 0.0),
            wind: Vector3::ZERO,
            turbulence: Turbulence::default(),
            fields: Vec::new(),
            boundary_layer: None,
            cooling: None,
            flutter: 0.0,
        }
    }
}

/// What happens when a particle reaches a surface.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Response {
    /// Bounce: the normal speed times `restitution` back out, the tangential
    /// speed times `1 - friction`.
    Bounce { restitution: f32, friction: f32 },
    /// Stop where it touched and stay until its life runs out.
    Stick,
    /// Gone at once.
    Die,
    /// Gone, its [`Emitter::volume`] handed over as a
    /// [`Landing`](ParticleSystem::landed) — for a heightfield to take up.
    Deposit,
}

/// Particles thrown off where one hits a surface: a raindrop's splash, a
/// spark's spray. They are drawn and simulated as the system's own, but
/// never spray in turn, and die where they next land.
#[derive(Debug, Clone, PartialEq)]
pub struct Spray {
    /// How many per hit.
    pub count: Range,
    /// Speed off the surface, m/s, within `angle` radians of its normal.
    pub speed: Range,
    pub angle: f32,
    /// How much of the hitting particle's velocity along the surface they
    /// keep.
    pub carry: f32,
    pub lifetime: Range,
    /// Size, as a multiple of the hitting particle's.
    pub size: Range,
    pub terminal_speed: Range,
}

/// A surface particles collide with.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Collider {
    /// The plane `y = height`, from above.
    Ground { height: f32 },
    /// A ball, from outside.
    Sphere { center: Vector3, radius: f32 },
    /// An axis-aligned box, from outside.
    Box { min: Vector3, max: Vector3 },
}

/// How much of a particle's life has gone, `0..=1`, mapped to a value by
/// straight lines between keys.
#[derive(Debug, Clone, PartialEq)]
pub struct Curve(pub Vec<(f32, f32)>);

impl Curve {
    pub fn constant(v: f32) -> Self {
        Self(vec![(0.0, v)])
    }
    pub fn at(&self, t: f32) -> f32 {
        lerp_keys(&self.0, t, |a, b, s| a + (b - a) * s).unwrap_or(1.0)
    }
}

/// Colour and opacity over a particle's life, by keys like [`Curve`]'s.
/// Colours are linear.
#[derive(Debug, Clone, PartialEq)]
pub struct Gradient(pub Vec<(f32, [f32; 4])>);

impl Gradient {
    pub fn constant(color: Color, alpha: f32) -> Self {
        Self(vec![(0.0, [color.r, color.g, color.b, alpha])])
    }
    pub fn at(&self, t: f32) -> [f32; 4] {
        lerp_keys(&self.0, t, |a, b, s| {
            [0, 1, 2, 3].map(|i| a[i] + (b[i] - a[i]) * s)
        })
        .unwrap_or([1.0; 4])
    }
}

fn lerp_keys<T: Copy>(keys: &[(f32, T)], t: f32, mix: impl Fn(T, T, f32) -> T) -> Option<T> {
    let first = keys.first()?;
    if t <= first.0 {
        return Some(first.1);
    }
    for w in keys.windows(2) {
        let ((t0, a), (t1, b)) = (w[0], w[1]);
        if t <= t1 {
            let s = if t1 > t0 { (t - t0) / (t1 - t0) } else { 1.0 };
            return Some(mix(a, b, s));
        }
    }
    keys.last().map(|k| k.1)
}

/// Where a depositing particle came down, and the material it carried.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Landing {
    pub position: Vector3,
    pub volume: f32,
}

/// The particles alive, one column per property.
#[derive(Debug, Clone, Default)]
pub struct Particles {
    pub position: Vec<Vector3>,
    pub velocity: Vec<Vector3>,
    pub age: Vec<f32>,
    pub lifetime: Vec<f32>,
    pub size: Vec<f32>,
    /// Drag coefficient `g / v_t²` (0: none).
    pub drag: Vec<f32>,
    pub rotation: Vec<f32>,
    pub spin: Vec<f32>,
    /// Material volume, for a deposit.
    pub volume: Vec<f32>,
    /// Stuck where it landed.
    pub resting: Vec<bool>,
    /// A per-particle random number in `0..1`, for variety in the shader.
    pub seed: Vec<f32>,
    /// 0 for an emitted particle, 1 for one thrown off by a [`Spray`].
    pub generation: Vec<u8>,
    /// Kelvin — with [`ParticleSystem::thermal`] set, else empty.
    pub temperature: Vec<f32>,
    /// The share of its diameter not yet burnt — likewise.
    pub remaining: Vec<f32>,
}

impl Particles {
    pub fn len(&self) -> usize {
        self.position.len()
    }
    pub fn is_empty(&self) -> bool {
        self.position.is_empty()
    }
    fn swap_remove(&mut self, i: usize) {
        self.position.swap_remove(i);
        self.velocity.swap_remove(i);
        self.age.swap_remove(i);
        self.lifetime.swap_remove(i);
        self.size.swap_remove(i);
        self.drag.swap_remove(i);
        self.rotation.swap_remove(i);
        self.spin.swap_remove(i);
        self.volume.swap_remove(i);
        self.resting.swap_remove(i);
        self.seed.swap_remove(i);
        self.generation.swap_remove(i);
        if i < self.temperature.len() {
            self.temperature.swap_remove(i);
        }
        if i < self.remaining.len() {
            self.remaining.swap_remove(i);
        }
    }
}

/// Emitters, forces, colliders and the particles they make.
#[derive(Debug, Clone)]
pub struct ParticleSystem {
    pub emitters: Vec<Emitter>,
    pub forces: Forces,
    pub colliders: Vec<Collider>,
    /// What a particle does on reaching a collider or a heightfield.
    pub response: Response,
    /// What it throws off as it hits, if anything.
    pub spray: Option<Spray>,
    /// Drawn size over life, as a multiple of the particle's own.
    pub size_over_life: Curve,
    /// Colour (linear) and opacity over life.
    pub color_over_life: Gradient,
    /// Temperature over life, kelvin: if set, each particle glows as a
    /// black body at it — its colour along the Planckian locus and its
    /// brightness going as T⁴ — times the colour over life (fire, embers).
    pub temperature_over_life: Option<Curve>,
    /// Heat held and lost by physics, each particle its own temperature
    /// (embers, sparks): see [`Thermal`]. Overrides `temperature_over_life`.
    pub thermal: Option<Thermal>,
    /// How they are drawn.
    pub look: ParticleLook,
    /// No more than this many at once; emission waits for room.
    pub max_particles: usize,
    /// Particles further than this below the lowest collider (or from the
    /// origin, with none) are dropped: they have left the scene.
    pub bounds: f32,
    pub particles: Particles,
    landed: Vec<Landing>,
    time: f32,
    rng: Rng,
    modes: [Mode; MODES],
    modes_norm: f32,
    /// Each particle's air velocity, refreshed every other step (half of
    /// them each step): eddies change over seconds, not frames, and the
    /// air is the dearest thing in a step. NaN: not yet worked out.
    air: Vec<Vector3>,
    steps: u64,
    /// The modes stretched for anisotropic turbulence, and the wind and
    /// anisotropy they were stretched for.
    stretched: Option<(Vector3, f32, [Mode; MODES])>,
}

impl ParticleSystem {
    pub fn new(look: ParticleLook) -> Self {
        Self {
            emitters: Vec::new(),
            forces: Forces::default(),
            colliders: Vec::new(),
            response: Response::Die,
            spray: None,
            size_over_life: Curve::constant(1.0),
            color_over_life: Gradient::constant(Color::WHITE, 1.0),
            temperature_over_life: None,
            thermal: None,
            look,
            max_particles: 20_000,
            bounds: 1_000.0,
            particles: Particles::default(),
            landed: Vec::new(),
            time: 0.0,
            rng: Rng::new(0x5eed),
            modes: Mode::set(0x5eed),
            modes_norm: Mode::norm(),
            air: Vec::new(),
            steps: 0,
            stretched: None,
        }
    }

    /// A different random sequence: emission positions, speeds and the
    /// turbulence's eddies.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.rng = Rng::new(seed);
        self.modes = Mode::set(seed);
        self
    }

    pub fn with_emitter(mut self, emitter: Emitter) -> Self {
        self.emitters.push(emitter);
        self
    }

    pub fn len(&self) -> usize {
        self.particles.len()
    }
    pub fn is_empty(&self) -> bool {
        self.particles.is_empty()
    }
    /// Seconds simulated.
    pub fn time(&self) -> f32 {
        self.time
    }

    /// Landings since the last call, taken: for a heightfield to absorb.
    pub fn take_landed(&mut self) -> Vec<Landing> {
        std::mem::take(&mut self.landed)
    }
    /// Landings waiting to be taken.
    pub fn landed(&self) -> &[Landing] {
        &self.landed
    }

    /// The air's velocity at `p`: the wind, its turbulence, and the fields'
    /// swirls.
    pub fn air_at(&self, p: Vector3) -> Vector3 {
        let shelter = self.forces.boundary_layer.map_or(1.0, |b| b.factor(p.y));
        let mut u = (self.forces.wind + self.turbulence_at(p, self.time)) * shelter;
        for f in &self.forces.fields {
            if let Field::Vortex {
                position,
                axis,
                speed,
                radius,
            } = *f
            {
                let a = axis.normalize();
                let r = p - position;
                let radial = r - a * r.dot(a);
                let d = radial.length();
                if d > 1e-4 && d < radius {
                    let fade = 1.0 - d / radius;
                    u = u + a.cross(radial * (1.0 / d)) * (speed * fade);
                }
            }
        }
        u
    }

    /// The turbulence alone: a sum of the curls of plane waves, each mode
    /// `(k × e) cos(k · x + φ)` and so divergence-free, in coordinates carried
    /// downwind.
    pub fn turbulence_at(&self, p: Vector3, t: f32) -> Vector3 {
        let turb = &self.forces.turbulence;
        if turb.strength <= 0.0 {
            return Vector3::ZERO;
        }
        let q = (p - self.forces.wind * t) * (1.0 / turb.scale.max(1e-3));
        let churn = turb.churn * turb.strength / turb.scale.max(1e-3) * t;
        let mut u = Vector3::ZERO;
        let fresh;
        let modes = if turb.anisotropy > 0.0 {
            match &self.stretched {
                Some((w, a, m)) if *w == self.forces.wind && *a == turb.anisotropy => m,
                _ => {
                    fresh = self.stretch_modes();
                    &fresh
                }
            }
        } else {
            &self.modes
        };
        for m in modes {
            // Smaller eddies turn over faster: a rate going as k^(2/3).
            u = u + m.curl * (m.amplitude * fast_cos(q.dot(m.k) + m.phase + churn * m.rate));
        }
        u * (turb.strength * self.modes_norm)
    }

    /// The turbulence's modes stretched in the wind's frame: wave vectors
    /// `k' = S⁻¹ k` (eddies longer downwind and across it, shorter up and
    /// down), each mode's flow projected square to its `k'` so the sum stays
    /// divergence-free.
    fn stretch_modes(&self) -> [Mode; MODES] {
        let a = self.forces.turbulence.anisotropy.clamp(0.0, 1.0);
        let w = Vector3::new(self.forces.wind.x, 0.0, self.forces.wind.z);
        let along = if w.length() > 1e-4 {
            w.normalize()
        } else {
            Vector3::new(1.0, 0.0, 0.0)
        };
        let up = Vector3::new(0.0, 1.0, 0.0);
        let across = up.cross(along);
        // Eddy lengths relative to the isotropic ones.
        let (lx, ly, lz) = (1.0 + 2.0 * a, 1.0 + 0.6 * a, 1.0 - 0.45 * a);
        self.modes.map(|m| {
            let k = along * (m.k.dot(along) / lx)
                + across * (m.k.dot(across) / ly)
                + up * (m.k.dot(up) / lz);
            let kn = k.normalize();
            let c = m.curl - kn * m.curl.dot(kn);
            let curl = if c.length() > 1e-4 {
                c.normalize()
            } else {
                kn.cross(up).normalize()
            };
            Mode { k, curl, ..m }
        })
    }

    /// Advance `dt` seconds: emit, move, collide (with the colliders and, if
    /// given, a heightfield's surface), age and retire.
    pub fn step(&mut self, dt: f32, ground: Option<&Heightfield>) {
        if dt <= 0.0 {
            return;
        }
        let turb = self.forces.turbulence;
        if turb.anisotropy > 0.0
            && !matches!(&self.stretched, Some((w, a, _)) if *w == self.forces.wind && *a == turb.anisotropy)
        {
            self.stretched = Some((self.forces.wind, turb.anisotropy, self.stretch_modes()));
        }
        self.emit(dt);
        // Substeps keep collisions honest for fast particles: none moves more
        // than about one size per substep... within reason.
        let fastest = self
            .particles
            .velocity
            .iter()
            .map(|v| v.length())
            .fold(0.0f32, f32::max);
        let substeps = ((fastest * dt / 0.25).ceil() as usize).clamp(1, 8);
        let h = dt / substeps as f32;
        for _ in 0..substeps {
            self.integrate(h, ground);
            self.time += h;
        }
        self.heat(dt);
        self.retire();
    }

    fn emit(&mut self, dt: f32) {
        let t0 = self.time;
        let mut spawns = Vec::new();
        for (e, emitter) in self.emitters.iter_mut().enumerate() {
            let n = emitter.due(t0, dt);
            spawns.push((e, n));
        }
        for (e, n) in spawns {
            for k in 0..n {
                if self.particles.len() >= self.max_particles {
                    return;
                }
                let emitter = &self.emitters[e];
                let (p, v) = emitter.spawn(&mut self.rng);
                // Born at some moment within the step, as a continuous source
                // would be — not all at its start, which would draw a stream
                // as a string of beads, one per frame. Its age starts that
                // far below zero, and it moves only for the share of the
                // step it was alive.
                let late = dt * (k as f32 + self.rng.next()) / n.max(1) as f32;
                let vt = emitter.terminal_speed.sample(&mut self.rng);
                let g = self.forces.gravity.length().max(1e-3);
                let ps = &mut self.particles;
                ps.position.push(p);
                ps.velocity.push(v);
                ps.age.push(-late);
                ps.lifetime
                    .push(emitter.lifetime.sample(&mut self.rng).max(1e-3));
                ps.size.push(emitter.size.sample(&mut self.rng));
                ps.drag.push(if vt > 0.0 { g / (vt * vt) } else { 0.0 });
                ps.rotation.push(self.rng.next() * std::f32::consts::TAU);
                ps.spin.push(emitter.spin.sample(&mut self.rng));
                ps.volume.push(emitter.volume);
                ps.resting.push(false);
                ps.seed.push(self.rng.next());
                ps.generation.push(0);
            }
        }
    }

    fn integrate(&mut self, h: f32, ground: Option<&Heightfield>) {
        let g = self.forces.gravity;
        let n = self.particles.len();
        let mut sprays: Vec<(Vector3, Vector3, Vector3, f32)> = Vec::new();
        self.air.resize(n, Vector3::new(f32::NAN, 0.0, 0.0));
        self.steps += 1;
        let parity = (self.steps & 1) as usize;
        for i in 0..n {
            let born = self.particles.age[i];
            self.particles.age[i] += h;
            if self.particles.resting[i] {
                continue;
            }
            // Born during this substep: alive for only part of it.
            let h = if born < 0.0 { (born + h).max(0.0) } else { h };
            if h <= 0.0 {
                continue;
            }
            let p = self.particles.position[i];
            let mut v = self.particles.velocity[i];
            let age = self.particles.age[i].max(0.0);
            let mut a = match self.forces.cooling {
                Some(c) => c.cold_gravity + (g - c.cold_gravity) * (-age / c.time.max(1e-3)).exp(),
                None => g,
            };
            if self.forces.flutter > 0.0 {
                // Each particle's own rate (about 1–2 Hz, as a falling flake
                // or leaf swings) and phase, from its seed.
                let s = self.particles.seed[i];
                let w = std::f32::consts::TAU * (0.9 + 0.9 * s);
                a = a + Vector3::new(
                    fast_cos(w * age + s * 17.0 - std::f32::consts::FRAC_PI_2),
                    0.0,
                    fast_cos(w * 0.83 * age + s * 31.0),
                ) * self.forces.flutter;
            }
            for f in &self.forces.fields {
                if let Field::Attractor {
                    position,
                    strength,
                    radius,
                } = *f
                {
                    let d = position - p;
                    let r = d.length();
                    if r > 1e-4 && r < radius {
                        a = a + d * (strength * (1.0 - r / radius) / r);
                    }
                }
            }
            let k = self.particles.drag[i];
            if k > 0.0 {
                // Implicit in the drag: relative velocity after the step is
                // (u_rel + a h) / (1 + k |u_rel| h), stable at any h and
                // exactly at terminal speed when it settles.
                if self.air[i].x.is_nan() || i & 1 == parity {
                    self.air[i] = self.air_at(p);
                }
                let air = self.air[i];
                let rel = v - air + a * h;
                let speed = (v - air).length();
                v = air + rel * (1.0 / (1.0 + k * speed * h));
            } else {
                v = v + a * h;
            }
            let mut np = p + v * h;
            self.particles.rotation[i] += self.particles.spin[i] * h;

            // Collisions: the first surface crossed.
            let mut hit: Option<(Vector3, Vector3)> = None; // (contact point, normal)
            if let Some(field) = ground {
                if let Some(top) = field.height_at(np.x, np.z) {
                    if np.y < top {
                        hit = Some((Vector3::new(np.x, top, np.z), field.normal_at(np.x, np.z)));
                    }
                }
            }
            if hit.is_none() {
                for c in &self.colliders {
                    if let Some(contact) = c.contact(np) {
                        hit = Some(contact);
                        break;
                    }
                }
            }
            if let Some((point, normal)) = hit {
                let child = self.particles.generation[i] > 0;
                if !child && self.spray.is_some() {
                    sprays.push((point, normal, v, self.particles.size[i]));
                }
                // Sprayed particles never spray, and do not pile up: they die
                // where they next land unless the system bounces.
                let response = match self.response {
                    Response::Deposit | Response::Stick if child => Response::Die,
                    r => r,
                };
                match response {
                    Response::Bounce {
                        restitution,
                        friction,
                    } => {
                        let vn = normal * v.dot(normal);
                        let vt = v - vn;
                        if v.dot(normal) < 0.0 {
                            v = vt * (1.0 - friction).max(0.0) - vn * restitution;
                        }
                        np = point + normal * 1e-4;
                    }
                    Response::Stick => {
                        v = Vector3::ZERO;
                        np = point;
                        self.particles.resting[i] = true;
                    }
                    Response::Die | Response::Deposit => {
                        np = point;
                        // Retired below; a deposit is recorded there.
                        self.particles.age[i] = f32::INFINITY;
                        self.particles.resting[i] = true;
                        if response == Response::Deposit {
                            self.landed.push(Landing {
                                position: point,
                                volume: self.particles.volume[i],
                            });
                        }
                    }
                }
            }
            self.particles.position[i] = np;
            self.particles.velocity[i] = v;
        }
        for (point, normal, v, size) in sprays {
            self.spray_at(point, normal, v, size);
        }
    }

    /// Particles thrown off a hit at `point` on a surface facing `normal`, by
    /// a particle of `size` arriving at `v`.
    fn spray_at(&mut self, point: Vector3, normal: Vector3, v: Vector3, size: f32) {
        let Some(spray) = self.spray.clone() else {
            return;
        };
        let g = self.forces.gravity.length().max(1e-3);
        let along = v - normal * v.dot(normal);
        let n = spray.count.sample(&mut self.rng).round().max(0.0) as usize;
        for _ in 0..n {
            if self.particles.len() >= self.max_particles {
                return;
            }
            let dir = self.rng.in_cone(normal, spray.angle);
            let vel = dir * spray.speed.sample(&mut self.rng) + along * spray.carry;
            let vt = spray.terminal_speed.sample(&mut self.rng);
            let ps = &mut self.particles;
            ps.position.push(point + normal * 1e-3);
            ps.velocity.push(vel);
            ps.age.push(0.0);
            ps.lifetime
                .push(spray.lifetime.sample(&mut self.rng).max(1e-3));
            ps.size.push(size * spray.size.sample(&mut self.rng));
            ps.drag.push(if vt > 0.0 { g / (vt * vt) } else { 0.0 });
            ps.rotation.push(self.rng.next() * std::f32::consts::TAU);
            ps.spin.push(0.0);
            ps.volume.push(0.0);
            ps.resting.push(false);
            ps.seed.push(self.rng.next());
            ps.generation.push(1);
        }
    }

    fn retire(&mut self) {
        let floor = self
            .colliders
            .iter()
            .filter_map(|c| match c {
                Collider::Ground { height } => Some(*height),
                _ => None,
            })
            .fold(f32::INFINITY, f32::min);
        let mut i = 0;
        while i < self.particles.len() {
            let p = self.particles.position[i];
            let gone = self.particles.age[i] >= self.particles.lifetime[i]
                || !p.x.is_finite()
                || p.length() > self.bounds
                || (floor.is_finite() && p.y < floor - 1.0);
            if gone {
                self.particles.swap_remove(i);
                if i < self.air.len() {
                    self.air.swap_remove(i);
                }
            } else {
                i += 1;
            }
        }
    }

    /// How far through its life particle `i` is, `0..=1`.
    pub fn life_fraction(&self, i: usize) -> f32 {
        let l = self.particles.lifetime[i];
        if l.is_finite() {
            (self.particles.age[i] / l).clamp(0.0, 1.0)
        } else {
            0.0
        }
    }
}

impl Collider {
    /// If `p` is inside: the nearest surface point and its outward normal.
    fn contact(&self, p: Vector3) -> Option<(Vector3, Vector3)> {
        match *self {
            Collider::Ground { height } => (p.y < height)
                .then(|| (Vector3::new(p.x, height, p.z), Vector3::new(0.0, 1.0, 0.0))),
            Collider::Sphere { center, radius } => {
                let d = p - center;
                let r = d.length();
                (r < radius).then(|| {
                    let n = if r > 1e-6 {
                        d * (1.0 / r)
                    } else {
                        Vector3::new(0.0, 1.0, 0.0)
                    };
                    (center + n * radius, n)
                })
            }
            Collider::Box { min, max } => {
                let inside = p.x > min.x
                    && p.x < max.x
                    && p.y > min.y
                    && p.y < max.y
                    && p.z > min.z
                    && p.z < max.z;
                if !inside {
                    return None;
                }
                // Out through the nearest face.
                let faces = [
                    (p.x - min.x, Vector3::new(-1.0, 0.0, 0.0)),
                    (max.x - p.x, Vector3::new(1.0, 0.0, 0.0)),
                    (p.y - min.y, Vector3::new(0.0, -1.0, 0.0)),
                    (max.y - p.y, Vector3::new(0.0, 1.0, 0.0)),
                    (p.z - min.z, Vector3::new(0.0, 0.0, -1.0)),
                    (max.z - p.z, Vector3::new(0.0, 0.0, 1.0)),
                ];
                let (d, n) = faces
                    .into_iter()
                    .fold((f32::INFINITY, Vector3::ZERO), |a, b| {
                        if b.0 < a.0 {
                            b
                        } else {
                            a
                        }
                    });
                Some((p + n * d, n))
            }
        }
    }
}

/// Turbulence modes: four in each of three octaves of eddy size.
const MODES: usize = 12;

/// One turbulence mode: a wave vector, the curl it carries, its share of the
/// energy, its phase and how fast it churns.
#[derive(Debug, Clone, Copy)]
struct Mode {
    k: Vector3,
    curl: Vector3,
    amplitude: f32,
    phase: f32,
    rate: f32,
}

impl Mode {
    /// Three octaves, each eddy half the last's size: Kolmogorov's inertial
    /// range, where energy `E(k) ∝ k^(-5/3)` makes an eddy's speed go as
    /// `k^(-1/3)` — the small ones weaker but faster-turning, which is what
    /// makes a flake's path wander rather than just sway.
    fn set(seed: u64) -> [Mode; MODES] {
        let mut rng = Rng::new(seed ^ 0x7ab1e);
        let mut n = 0;
        [(); MODES].map(|_| {
            let octave = n / 4;
            n += 1;
            let scale = 2f32.powi(octave);
            let k = rng.unit_vector() * (std::f32::consts::TAU * (0.7 + 0.6 * rng.next()) * scale);
            let e = rng.unit_vector();
            let c = k.cross(e);
            let curl = if c.length() > 1e-4 {
                c.normalize()
            } else {
                Vector3::new(1.0, 0.0, 0.0)
            };
            Mode {
                k,
                curl,
                amplitude: scale.powf(-1.0 / 3.0),
                phase: rng.next() * std::f32::consts::TAU,
                rate: (0.5 + rng.next()) * scale.powf(2.0 / 3.0),
            }
        })
    }

    /// Scales the sum of modes to an rms speed of 1: each mode's mean square
    /// is half its amplitude squared.
    fn norm() -> f32 {
        let total: f32 = (0..MODES)
            .map(|n| 2f32.powi((n / 4) as i32).powf(-2.0 / 3.0) * 0.5)
            .sum();
        1.0 / total.sqrt()
    }
}

/// Cosine to about 0.001 — turbulence and flutter evaluate a dozen per
/// particle per step, and the library's is most of a frame's cost at fifty
/// thousand particles. (The turbulence stays exactly divergence-free: each
/// mode's direction `k × e` is what makes it so, whatever its profile.)
#[inline]
pub(crate) fn fast_cos(x: f32) -> f32 {
    use std::f32::consts::{FRAC_PI_2, PI, TAU};
    // cos x = sin(x + π/2), reduced to [-π, π].
    let mut t = x + FRAC_PI_2;
    t -= TAU * (t * (1.0 / TAU)).round();
    // A parabola through sin's zeros and peaks, then a correction toward the
    // curve.
    let y = (4.0 / PI) * t - (4.0 / (PI * PI)) * t * t.abs();
    0.225 * (y * y.abs() - y) + y
}

/// Two unit vectors perpendicular to `n` and each other.
pub(crate) fn basis(n: Vector3) -> (Vector3, Vector3) {
    let t = if n.x.abs() < 0.9 {
        Vector3::new(1.0, 0.0, 0.0)
    } else {
        Vector3::new(0.0, 1.0, 0.0)
    };
    let u = n.cross(t).normalize();
    (u, n.cross(u))
}

/// A small, fast, seeded generator (xorshift64*): the same particles from
/// the same seed, on every platform.
#[derive(Debug, Clone)]
pub(crate) struct Rng(u64);

impl Rng {
    pub(crate) fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1)
    }
    /// Uniform in `0..1`.
    pub(crate) fn next(&mut self) -> f32 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        let x = self.0.wrapping_mul(0x2545_f491_4f6c_dd1d);
        (x >> 40) as f32 / (1u64 << 24) as f32
    }
    pub(crate) fn unit_vector(&mut self) -> Vector3 {
        let z = self.next() * 2.0 - 1.0;
        let a = self.next() * std::f32::consts::TAU;
        let r = (1.0 - z * z).max(0.0).sqrt();
        Vector3::new(r * a.cos(), r * a.sin(), z)
    }
    /// Uniform over the cap within `angle` of `axis`.
    pub(crate) fn in_cone(&mut self, axis: Vector3, angle: f32) -> Vector3 {
        let cos_max = angle.cos();
        let z = 1.0 - self.next() * (1.0 - cos_max);
        let a = self.next() * std::f32::consts::TAU;
        let r = (1.0 - z * z).max(0.0).sqrt();
        let (u, v) = basis(axis);
        u * (r * a.cos()) + v * (r * a.sin()) + axis * z
    }
}

#[cfg(test)]
mod tests;
