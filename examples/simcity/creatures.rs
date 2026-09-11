//! Part of the `simcity` example; see `mod.rs`.
//!
//! The things in the city that are alive and are not people or traffic.
//!
//! Vehicles and pedestrians are lane-bound: a `Mover` is a scalar position on
//! a one-dimensional run, which is exactly what makes queueing, give-way and
//! junction logic tractable. Animals are not on rails, so they get their own
//! much simpler model: each one walks a closed loop of its own around a home
//! point, the loop being a circle bent by two harmonics keyed on the animal's
//! seed. No neighbour queries, no collisions, no state to advance — position
//! and heading are both a pure function of the clock, which means a swarm
//! costs one transform write per animal per frame and nothing else.
#![allow(dead_code)]

use super::*;

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum CritterKind {
    /// In the air, banking into its turns, wings beating.
    Bird,
    /// The same, in gull colours and half again the size, over water.
    Gull,
    /// On the ground, pottering and pecking.
    Pigeon,
    /// On the water.
    Duck,
    /// The same, larger and white, with a neck that carries itself.
    Swan,
    /// Ground, quick, tail over its back. Parks and groves.
    Squirrel,
    /// Ground, after dark. Bins, back streets and the edges of parks.
    Fox,
    /// Air, after dark, and nothing like a bird about how it flies.
    Bat,
    /// Air, low, over long grass in daylight.
    Butterfly,
    /// Air, over water, in daylight. Darts rather than flies.
    Dragonfly,
    /// Air, over the refuse. A knot of them, going nowhere.
    Fly,
    /// Ground, after dark, against the bins. Small, quick and low.
    Rat,
    /// An airliner crossing high above, on a loop so wide it reads straight.
    Plane,
    /// A helicopter orbiting the city at working height.
    Helicopter,
    /// A quadcopter, low and quick.
    Drone,
    /// Off the lead, inside a dog run. The one place in the city a dog is not
    /// attached to a pedestrian.
    ParkDog,
}

impl CritterKind {
    /// Whether this one is in the air. Flyers are placed at an altitude and
    /// allowed to rise and fall; everything else is pinned to whatever surface
    /// it was put on, and drifting off it is a bug.
    pub(crate) fn flies(self) -> bool {
        matches!(
            self,
            CritterKind::Bird
                | CritterKind::Gull
                | CritterKind::Bat
                | CritterKind::Butterfly
                | CritterKind::Dragonfly
                | CritterKind::Fly
                | CritterKind::Plane
                | CritterKind::Helicopter
                | CritterKind::Drone
        )
    }
}

/// When an animal is about.
///
/// A city has two populations and they barely overlap: pigeons and squirrels
/// work daylight, foxes and bats work the dark. Drawing both at once is the
/// single most obvious way to make a night render look wrong, and swapping
/// them is nearly free — the swarms are already separate objects.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Shift {
    Day,
    Night,
    Always,
}

impl Shift {
    /// `lights` is the street-lighting term: 0 in daylight, 1 after dark.
    pub(crate) fn awake(self, lights: f32) -> bool {
        match self {
            Shift::Day => lights < 0.55,
            Shift::Night => lights > 0.35,
            Shift::Always => true,
        }
    }
}

#[derive(Clone)]
pub(crate) struct Critter {
    pub(crate) hx: f32,
    pub(crate) hz: f32,
    /// Mean radius of the loop it walks, swims or flies.
    pub(crate) r: f32,
    pub(crate) y: f32,
    /// Where it starts on the loop, in turns.
    pub(crate) phase: f32,
    /// Turns per second.
    pub(crate) rate: f32,
    /// How far the loop departs from a circle, as a fraction of `r`.
    pub(crate) wander: f32,
    pub(crate) seed: u32,
    pub(crate) scale: f32,
    /// Amplitude of the vertical oscillation: a bird rising and falling, a
    /// duck riding a wave, a pigeon hopping.
    pub(crate) climb: f32,
}

impl Critter {
    /// Radius of the loop at angle `a`. Two harmonics is the fewest that
    /// reads as a wandering path rather than as an ellipse.
    fn radius(&self, a: f32) -> f32 {
        let s = self.seed as f32 * 0.001;
        self.r
            * (1.0
                + self.wander * ((a * 2.0 + s).sin() * 0.6 + (a * 3.0 + s * 2.7).sin() * 0.4))
    }

    /// `(x, y, z, yaw, roll)` at time `t`.
    pub(crate) fn at(&self, t: f32) -> (f32, f32, f32, f32, f32) {
        let a = (self.phase + t * self.rate) * TAU;
        let ra = self.radius(a);
        let (x, z) = (self.hx + ra * a.cos(), self.hz + ra * a.sin());
        // Heading from a finite difference along the loop. Differentiating the
        // harmonics analytically saves nothing here and is easy to get wrong.
        let da = 0.02;
        let rb = self.radius(a + da);
        let (x2, z2) = (
            self.hx + rb * (a + da).cos(),
            self.hz + rb * (a + da).sin(),
        );
        let yaw = (x2 - x).atan2(z2 - z);
        let y = self.y + self.climb * (a * 1.37 + self.seed as f32 * 0.01).sin();
        // Bank into the turn. Curvature is roughly constant round the loop, so
        // the roll follows the loop rate and the radius rather than being
        // recomputed from the path.
        // Signed, so a flock circling the other way banks the other way. The
        // clamp used to run 0..0.7, which silently flattened every
        // anticlockwise flock in the city.
        let roll = -(self.rate * self.r * 0.09).clamp(-0.7, 0.7);
        (x, y, z, yaw, roll)
    }
}

/// One population of one kind of animal, drawn from a handful of pose meshes.
pub(crate) struct Swarm {
    pub(crate) critters: Vec<Critter>,
    /// `(mesh, pose)`. One instanced mesh per pose: one draw each.
    pub(crate) groups: Vec<(ObjectId, usize)>,
    pub(crate) poses: usize,
    /// Pose cycles per second.
    pub(crate) beat: f32,
    /// Metres past which the animal is not drawn at all. Nothing here is more
    /// than a foot long, so there is no impostor worth having — past this it
    /// is a subpixel speck and dropping it outright is both cheaper and, at
    /// this size, invisible.
    pub(crate) lod: f32,
    pub(crate) kind: CritterKind,
    pub(crate) shift: Shift,
    /// Set by `apply_sky` from the clock. A sleeping swarm is neither drawn
    /// nor walked.
    pub(crate) awake: bool,
}

/// Place every animal in a swarm for this frame. Returns how many were drawn.
pub(crate) fn write_swarm(scene: &mut Scene, swarm: &Swarm, eye: Vector3, t: f32) -> usize {
    if !swarm.awake {
        return 0;
    }
    let lod2 = swarm.lod * swarm.lod;
    let mut drawn = 0usize;
    for (mesh, pose) in &swarm.groups {
        let Some(obj) = scene.get_mut(*mesh) else {
            continue;
        };
        let ObjectKind::InstancedMesh(im) = &mut obj.kind else {
            continue;
        };
        im.transforms.clear();
        for (i, c) in swarm.critters.iter().enumerate() {
            let (x, y, z, yaw, roll) = c.at(t);
            if (x - eye.x).powi(2) + (z - eye.z).powi(2) > lod2 {
                continue;
            }
            // Each animal is offset in the cycle by its index, so a flock does
            // not beat its wings in unison — which is the single thing that
            // most gives away a crowd of clones.
            let step = ((t * swarm.beat + i as f32 * 0.37) as i64).rem_euclid(swarm.poses as i64);
            if step as usize != *pose {
                continue;
            }
            let rot = Quaternion::from_axis_angle(Vector3::UP, yaw)
                * Quaternion::from_axis_angle(Vector3::new(0.0, 0.0, 1.0), roll);
            im.transforms.push(Matrix4::compose(
                Vector3::new(x, y, z),
                rot,
                Vector3::ONE * c.scale,
            ));
            drawn += 1;
        }
    }
    drawn
}

// ---------------------------------------------------------------------------
// Geometry. Everything here is tiny on screen, so the budget is a few dozen
// triangles each and the shapes have to work as silhouettes.
// ---------------------------------------------------------------------------

/// A bird in flight. `beat` runs -1 (wings down) through 0 (level) to 1 (up).
pub(crate) fn bird_geometry(beat: f32) -> BufferGeometry {
    bird_shape(beat, false)
}

/// A gull: the same armature, white-bodied with grey wings, and heavier.
pub(crate) fn gull_geometry(beat: f32) -> BufferGeometry {
    bird_shape(beat, true)
}

fn bird_shape(beat: f32, gull: bool) -> BufferGeometry {
    let mut m = MeshBuilder::default();
    let (body, pale) = if gull {
        (Color::from_hex(0xf0eee8), Color::from_hex(0x8d99a4))
    } else {
        (Color::from_hex(0x2e3238), Color::from_hex(0x6d7278))
    };
    // Body along +Z, which is the heading `write_swarm` rotates to.
    m.add_ellipsoid(Vector3::new(0.0, 0.0, 0.0), 0.075, 0.075, 0.24, 4, 7, body);
    m.add_ellipsoid(Vector3::new(0.0, 0.03, 0.20), 0.055, 0.055, 0.07, 3, 6, body);
    // Tail.
    m.add_yaw_box(
        Vector3::new(0.0, 0.0, -0.30),
        Vector3::new(0.075, 0.010, 0.10),
        0.0,
        pale,
        Uv::Unit,
    );
    // Wings: a swept pair, hinged at the shoulder and lifted by `beat`. The
    // outer panel lags the inner one, which is what makes a beat read as a
    // beat rather than as a pair of scissors.
    for s in [-1.0f32, 1.0] {
        let inner_y = 0.10 * beat;
        let outer_y = 0.26 * beat;
        m.tri(
            [
                [0.0, 0.02, 0.02],
                [s * 0.22, inner_y, -0.02],
                [s * 0.20, inner_y, 0.10],
            ],
            [0.0, 1.0, 0.0],
            [[0.5, 0.5]; 3],
            [body; 3],
        );
        m.tri(
            [
                [s * 0.22, inner_y, -0.02],
                [s * 0.52, outer_y, -0.10],
                [s * 0.20, inner_y, 0.10],
            ],
            [0.0, 1.0, 0.0],
            [[0.5, 0.5]; 3],
            [pale; 3],
        );
        // Backs of the same two panels, so a wing is not invisible from below.
        m.tri(
            [
                [s * 0.20, inner_y, 0.10],
                [s * 0.22, inner_y, -0.02],
                [0.0, 0.02, 0.02],
            ],
            [0.0, -1.0, 0.0],
            [[0.5, 0.5]; 3],
            [body; 3],
        );
        m.tri(
            [
                [s * 0.20, inner_y, 0.10],
                [s * 0.52, outer_y, -0.10],
                [s * 0.22, inner_y, -0.02],
            ],
            [0.0, -1.0, 0.0],
            [[0.5, 0.5]; 3],
            [pale; 3],
        );
    }
    m.build()
}

/// A pigeon. Pose 0 stands, pose 1 has its head down feeding.
pub(crate) fn pigeon_geometry(pecking: bool) -> BufferGeometry {
    let mut m = MeshBuilder::default();
    let body = Color::from_hex(0x555b64);
    let neck = Color::from_hex(0x3f6b64);
    let beak = Color::from_hex(0x2a2c30);
    let foot = Color::from_hex(0xa8564a);
    let lean = if pecking { 0.55f32 } else { 0.0 };
    m.add_ellipsoid(
        Vector3::new(0.0, 0.115, -0.01),
        0.055,
        0.058,
        0.095,
        4,
        7,
        body,
    );
    // Head, swung forward and down when feeding.
    let hz = 0.075 + lean * 0.055;
    let hy = 0.185 - lean * 0.125;
    m.add_ellipsoid(Vector3::new(0.0, hy, hz), 0.032, 0.034, 0.036, 3, 6, neck);
    m.add_limb(
        Vector3::new(0.0, hy, hz + 0.02),
        Vector3::new(0.0, hy - lean * 0.03, hz + 0.065),
        0.012,
        0.004,
        4,
        beak,
        true,
    );
    // Tail, cocked up as the head goes down.
    m.add_yaw_box(
        Vector3::new(0.0, 0.125 + lean * 0.03, -0.135),
        Vector3::new(0.030, 0.007, 0.055),
        0.0,
        Color::from_hex(0x40454c),
        Uv::Unit,
    );
    for s in [-1.0f32, 1.0] {
        m.add_limb(
            Vector3::new(s * 0.020, 0.065, -0.01),
            Vector3::new(s * 0.020, 0.0, -0.01),
            0.007,
            0.006,
            4,
            foot,
            false,
        );
        // Folded wing, just proud of the flank.
        m.add_ellipsoid(
            Vector3::new(s * 0.050, 0.120, -0.015),
            0.014,
            0.040,
            0.080,
            3,
            5,
            Color::from_hex(0x4a5058),
        );
    }
    m.build()
}

/// A duck on the water. Only the part above the waterline is built.
pub(crate) fn duck_geometry(drake: bool) -> BufferGeometry {
    let mut m = MeshBuilder::default();
    let body = if drake {
        Color::from_hex(0x6b6152)
    } else {
        Color::from_hex(0x7a6b55)
    };
    let head = if drake {
        Color::from_hex(0x1f4f3a)
    } else {
        Color::from_hex(0x5e5140)
    };
    let bill = Color::from_hex(0xc2a13c);
    m.add_ellipsoid(Vector3::new(0.0, 0.05, 0.0), 0.10, 0.075, 0.20, 4, 8, body);
    // Neck and head, carried forward of the body.
    m.add_limb(
        Vector3::new(0.0, 0.08, 0.10),
        Vector3::new(0.0, 0.20, 0.15),
        0.035,
        0.030,
        5,
        head,
        false,
    );
    m.add_ellipsoid(Vector3::new(0.0, 0.22, 0.16), 0.042, 0.042, 0.052, 3, 6, head);
    m.add_limb(
        Vector3::new(0.0, 0.215, 0.19),
        Vector3::new(0.0, 0.205, 0.255),
        0.020,
        0.014,
        4,
        bill,
        true,
    );
    // Stern, tipped up the way a dabbling duck carries it.
    m.add_yaw_box(
        Vector3::new(0.0, 0.085, -0.20),
        Vector3::new(0.045, 0.022, 0.055),
        0.0,
        body,
        Uv::Unit,
    );
    m.build()
}

/// A dog, trotting. `swing` runs -1 to 1 across the gait cycle.
pub(crate) fn dog_geometry(swing: f32, coat: Color) -> BufferGeometry {
    let mut m = MeshBuilder::default();
    let dark = scale_color(coat, 0.72);
    // Body along +Z. A dog is about 0.6 m nose to tail at this scale.
    m.add_limb_flat(
        Vector3::new(0.0, 0.34, -0.20),
        Vector3::new(0.0, 0.36, 0.16),
        0.095,
        0.085,
        0.85,
        7,
        coat,
        true,
    );
    m.add_limb(
        Vector3::new(0.0, 0.36, 0.14),
        Vector3::new(0.0, 0.44, 0.24),
        0.055,
        0.048,
        5,
        coat,
        false,
    );
    m.add_ellipsoid(Vector3::new(0.0, 0.46, 0.28), 0.055, 0.055, 0.075, 3, 6, coat);
    m.add_limb(
        Vector3::new(0.0, 0.45, 0.33),
        Vector3::new(0.0, 0.42, 0.40),
        0.028,
        0.022,
        4,
        dark,
        true,
    );
    // Ears.
    for s in [-1.0f32, 1.0] {
        m.tri(
            [
                [s * 0.030, 0.50, 0.27],
                [s * 0.075, 0.58, 0.24],
                [s * 0.030, 0.49, 0.31],
            ],
            [s, 0.2, 0.0],
            [[0.5, 0.5]; 3],
            [dark; 3],
        );
    }
    // Legs: diagonal pairs in antiphase, which is what a trot is.
    for (sx, sz, ph) in [
        (-1.0f32, 1.0f32, 1.0f32),
        (1.0, 1.0, -1.0),
        (-1.0, -1.0, -1.0),
        (1.0, -1.0, 1.0),
    ] {
        let reach = swing * ph * 0.10;
        m.add_limb(
            Vector3::new(sx * 0.058, 0.32, sz * 0.13),
            Vector3::new(sx * 0.058, 0.0, sz * 0.13 + reach),
            0.028,
            0.020,
            5,
            coat,
            false,
        );
    }
    // Tail, up and curled over the back.
    m.add_limb(
        Vector3::new(0.0, 0.37, -0.20),
        Vector3::new(0.0, 0.54, -0.30 - swing * 0.04),
        0.024,
        0.012,
        4,
        coat,
        false,
    );
    m.build()
}

/// A squirrel: body, head, and the tail that is most of what you see.
///
/// The tail is the whole silhouette. A squirrel drawn without one is a rat,
/// and at this size the difference between the two animals is entirely that
/// arc over the back.
pub(crate) fn squirrel_geometry(sitting: bool) -> BufferGeometry {
    let mut m = MeshBuilder::default();
    let coat = Color::from_hex(0x7a5a3e);
    let pale = Color::from_hex(0xc8b79c);
    let lean = if sitting { 0.0f32 } else { 0.62 };
    // Body pitches forward as it runs and sits up when it stops.
    // Lower and longer than the first attempt, which stood permanently upright
    // and read as a meerkat.
    let (by, bz) = (mix(0.115, 0.075, lean), mix(0.0, 0.02, lean));
    m.add_limb_flat(
        Vector3::new(0.0, by - 0.05, bz - 0.075),
        Vector3::new(0.0, by + 0.05 - lean * 0.045, bz + 0.075),
        0.045,
        0.040,
        0.82,
        6,
        coat,
        true,
    );
    let hy = by + mix(0.10, 0.045, lean);
    let hz = bz + mix(0.055, 0.115, lean);
    m.add_ellipsoid(Vector3::new(0.0, hy, hz), 0.032, 0.032, 0.038, 3, 6, coat);
    for s in [-1.0f32, 1.0] {
        // Ears: small, upright, and set well back.
        m.tri(
            [
                [s * 0.014, hy + 0.026, hz - 0.010],
                [s * 0.030, hy + 0.070, hz - 0.020],
                [s * 0.014, hy + 0.026, hz + 0.014],
            ],
            [s, 0.3, 0.0],
            [[0.5, 0.5]; 3],
            [coat; 3],
        );
        // Forelegs, tucked when sitting.
        m.add_limb(
            Vector3::new(s * 0.026, by, bz + 0.055),
            Vector3::new(s * 0.026, by - mix(0.055, 0.085, lean), bz + mix(0.045, 0.095, lean)),
            0.013,
            0.010,
            4,
            coat,
            false,
        );
        m.add_limb(
            Vector3::new(s * 0.030, by - 0.02, bz - 0.055),
            Vector3::new(s * 0.030, 0.0, bz - 0.070),
            0.016,
            0.011,
            4,
            coat,
            false,
        );
    }
    // The tail: a chain sweeping up and over, thickening as it goes.
    let mut prev = Vector3::new(0.0, by - 0.02, bz - 0.085);
    for i in 1..=5 {
        let t = i as f32 / 5.0;
        let a = t * 2.1;
        let p = Vector3::new(
            0.0,
            by - 0.02 + a.sin() * 0.19,
            bz - 0.085 - (1.0 - a.cos()) * 0.085,
        );
        // Thicken hard toward the tip: a squirrel's tail is a plume, and a
        // tapered rod reads as a rat's.
        m.add_limb_flat(
            prev,
            p,
            0.020 + t * 0.048,
            0.024 + t * 0.052,
            0.55,
            6,
            mix_color(coat, pale, t),
            false,
        );
        prev = p;
    }
    m.build()
}

/// A swan. Bigger than a duck and carried differently: the neck is the point.
pub(crate) fn swan_geometry(curled: bool) -> BufferGeometry {
    let mut m = MeshBuilder::default();
    let white = Color::from_hex(0xf2f0ea);
    let bill = Color::from_hex(0xd8622e);
    m.add_ellipsoid(Vector3::new(0.0, 0.10, 0.0), 0.19, 0.14, 0.36, 5, 9, white);
    // The neck as an S: forward and up, then back over, then the head down.
    let bend = if curled { 0.55f32 } else { 0.0 };
    let pts = [
        Vector3::new(0.0, 0.16, 0.20),
        Vector3::new(0.0, 0.36, 0.30 - bend * 0.10),
        Vector3::new(0.0, 0.56, 0.28 - bend * 0.18),
        Vector3::new(0.0, 0.66, 0.36 - bend * 0.22),
    ];
    for w in pts.windows(2) {
        m.add_limb(w[0], w[1], 0.055, 0.042, 6, white, false);
    }
    let head = pts[3] + Vector3::new(0.0, 0.02, 0.03);
    m.add_ellipsoid(head, 0.050, 0.050, 0.065, 3, 6, white);
    m.add_limb(
        head + Vector3::new(0.0, -0.005, 0.04),
        head + Vector3::new(0.0, -0.03, 0.13),
        0.026,
        0.018,
        4,
        bill,
        true,
    );
    // Wings held slightly raised over the back, which is how one sits at rest.
    for s in [-1.0f32, 1.0] {
        m.add_ellipsoid(
            Vector3::new(s * 0.13, 0.20, -0.02),
            0.055,
            0.10,
            0.26,
            3,
            6,
            white,
        );
    }
    m.add_yaw_box(
        Vector3::new(0.0, 0.15, -0.36),
        Vector3::new(0.07, 0.03, 0.10),
        0.0,
        white,
        Uv::Unit,
    );
    m.build()
}

/// A fox. A dog's proportions pulled long and low, with the tail and the ears
/// doing the identifying.
pub(crate) fn fox_geometry(swing: f32) -> BufferGeometry {
    let mut m = MeshBuilder::default();
    let coat = Color::from_hex(0xb35a25);
    let pale = Color::from_hex(0xe8e2d6);
    let dark = Color::from_hex(0x2b2724);
    m.add_limb_flat(
        Vector3::new(0.0, 0.28, -0.22),
        Vector3::new(0.0, 0.30, 0.18),
        0.082,
        0.070,
        0.80,
        7,
        coat,
        true,
    );
    m.add_limb(
        Vector3::new(0.0, 0.30, 0.16),
        Vector3::new(0.0, 0.36, 0.27),
        0.048,
        0.040,
        5,
        coat,
        false,
    );
    m.add_ellipsoid(Vector3::new(0.0, 0.38, 0.31), 0.046, 0.044, 0.060, 3, 6, coat);
    // A long straight muzzle, which is most of the difference from a dog.
    m.add_limb(
        Vector3::new(0.0, 0.375, 0.35),
        Vector3::new(0.0, 0.355, 0.46),
        0.026,
        0.014,
        4,
        pale,
        true,
    );
    for s in [-1.0f32, 1.0] {
        // Ears: tall, pointed, black-backed.
        m.tri(
            [
                [s * 0.020, 0.42, 0.29],
                [s * 0.055, 0.53, 0.27],
                [s * 0.020, 0.41, 0.34],
            ],
            [s, 0.2, 0.0],
            [[0.5, 0.5]; 3],
            [dark; 3],
        );
        m.tri(
            [
                [s * 0.020, 0.41, 0.34],
                [s * 0.055, 0.53, 0.27],
                [s * 0.020, 0.42, 0.29],
            ],
            [-s, 0.2, 0.0],
            [[0.5, 0.5]; 3],
            [coat; 3],
        );
    }
    for (sx, sz, ph) in [
        (-1.0f32, 1.0f32, 1.0f32),
        (1.0, 1.0, -1.0),
        (-1.0, -1.0, -1.0),
        (1.0, -1.0, 1.0),
    ] {
        let reach = swing * ph * 0.09;
        m.add_limb(
            Vector3::new(sx * 0.050, 0.26, sz * 0.14),
            Vector3::new(sx * 0.050, 0.0, sz * 0.14 + reach),
            0.022,
            0.015,
            5,
            dark,
            false,
        );
    }
    // Brush: thick, straight out behind, white-tipped.
    m.add_limb(
        Vector3::new(0.0, 0.29, -0.22),
        Vector3::new(0.0, 0.22, -0.44),
        0.045,
        0.055,
        6,
        coat,
        false,
    );
    m.add_limb(
        Vector3::new(0.0, 0.22, -0.44),
        Vector3::new(0.0, 0.19, -0.55),
        0.050,
        0.022,
        6,
        pale,
        true,
    );
    m.build()
}

/// A bat. Two membranes and almost no body.
pub(crate) fn bat_geometry(beat: f32) -> BufferGeometry {
    let mut m = MeshBuilder::default();
    let skin = Color::from_hex(0x241f26);
    let wing = Color::from_hex(0x3a2f3a);
    m.add_ellipsoid(Vector3::new(0.0, 0.0, 0.0), 0.030, 0.032, 0.055, 3, 5, skin);
    for s in [-1.0f32, 1.0] {
        // A wing is a triangle fan off two spars, which is what gives the
        // scalloped trailing edge a bird's does not have.
        let elbow = [s * 0.09, 0.05 * beat, 0.02];
        let tip = [s * 0.20, 0.16 * beat, -0.03];
        let heel = [s * 0.05, 0.02 * beat - 0.02, -0.07];
        for (a, b, c) in [
            ([0.0, 0.01, 0.02], elbow, tip),
            ([0.0, 0.01, 0.02], tip, heel),
        ] {
            m.tri(a_b_c(a, b, c), [0.0, 1.0, 0.0], [[0.5, 0.5]; 3], [wing; 3]);
            m.tri(a_b_c(a, c, b), [0.0, -1.0, 0.0], [[0.5, 0.5]; 3], [wing; 3]);
        }
        m.add_limb(
            Vector3::new(0.0, 0.01, 0.02),
            Vector3::new(elbow[0], elbow[1], elbow[2]),
            0.008,
            0.006,
            3,
            skin,
            false,
        );
    }
    m.build()
}

fn a_b_c(a: [f32; 3], b: [f32; 3], c: [f32; 3]) -> [[f32; 3]; 3] {
    [a, b, c]
}

/// A butterfly: two wings and a thread of body.
pub(crate) fn butterfly_geometry(open: f32, tint: Color) -> BufferGeometry {
    let mut m = MeshBuilder::default();
    let body = Color::from_hex(0x2a2622);
    m.add_limb(
        Vector3::new(0.0, 0.0, -0.030),
        Vector3::new(0.0, 0.0, 0.030),
        0.008,
        0.005,
        4,
        body,
        true,
    );
    for s in [-1.0f32, 1.0] {
        // `open` swings the wings from folded overhead to flat out.
        let (wx, wy) = (0.075 * open, 0.075 * (1.0 - open * 0.85));
        for (a, b, c) in [
            ([0.0, 0.0, 0.02], [s * wx, wy, 0.055], [s * wx * 0.9, wy * 0.9, -0.01]),
            ([0.0, 0.0, -0.01], [s * wx * 0.9, wy * 0.9, -0.01], [s * wx * 0.6, wy * 0.7, -0.05]),
        ] {
            m.tri(a_b_c(a, b, c), [0.0, 1.0, 0.0], [[0.5, 0.5]; 3], [tint; 3]);
            m.tri(a_b_c(a, c, b), [0.0, -1.0, 0.0], [[0.5, 0.5]; 3], [tint; 3]);
        }
    }
    m.build()
}

/// Somebody sitting down.
///
/// Every bench, step and kerb in this city was empty, and every pedestrian was
/// walking somewhere at a constant speed — which reads less as a city than as
/// a treadmill. A seated figure needs no simulation at all: it is placed once
/// and never moves, so it costs one transform and nothing per frame.
///
/// `lean` runs 0 (upright, hands on knees) to 1 (slumped back, legs out).
pub(crate) fn seated_geometry(lean: f32, coat: Color, skin: Color, trouser: Color) -> BufferGeometry {
    let mut m = MeshBuilder::default();
    let seat = 0.46f32;
    let back = -0.10 - lean * 0.13;
    // Torso, tipped back as it slumps.
    let hip = Vector3::new(0.0, seat + 0.06, back);
    let shoulder = Vector3::new(0.0, seat + 0.58 - lean * 0.06, back - lean * 0.10);
    m.add_limb_flat(hip, shoulder, 0.155, 0.170, 0.62, 8, coat, true);
    // Head and neck.
    let head = Vector3::new(0.0, shoulder.y + 0.20, shoulder.z + 0.02 + lean * 0.02);
    m.add_limb(
        shoulder,
        Vector3::new(0.0, shoulder.y + 0.09, shoulder.z),
        0.055,
        0.050,
        5,
        skin,
        false,
    );
    m.add_ellipsoid(head, 0.105, 0.125, 0.105, 4, 7, skin);
    // Hair, which is what stops a head reading as a bald peg.
    m.add_ellipsoid(
        Vector3::new(head.x, head.y + 0.045, head.z - 0.015),
        0.108,
        0.095,
        0.108,
        3,
        7,
        scale_color(coat, 0.45),
    );
    for s in [-1.0f32, 1.0] {
        // Thigh forward along the seat, shin down to the ground.
        let knee = Vector3::new(s * 0.10, seat - 0.02, back + 0.42 + lean * 0.16);
        m.add_limb_flat(
            Vector3::new(s * 0.09, seat + 0.02, back + 0.04),
            knee,
            0.090,
            0.078,
            0.85,
            6,
            trouser,
            false,
        );
        let foot = Vector3::new(s * 0.10, 0.0, knee.z - lean * 0.10);
        m.add_limb_flat(knee, foot, 0.070, 0.055, 0.85, 6, trouser, false);
        m.add_yaw_box(
            Vector3::new(foot.x, 0.035, foot.z + 0.05),
            Vector3::new(0.055, 0.035, 0.105),
            0.0,
            Color::from_hex(0x25211d),
            Uv::Unit,
        );
        // Arm: down to the knee when upright, along the backrest when slumped.
        let hand = Vector3::new(
            s * (0.17 + lean * 0.14),
            mix(seat + 0.02, seat + 0.34, lean),
            mix(back + 0.30, back - 0.06, lean),
        );
        m.add_limb(
            Vector3::new(s * 0.155, shoulder.y - 0.03, shoulder.z + 0.02),
            hand,
            0.052,
            0.042,
            5,
            coat,
            false,
        );
        m.add_ellipsoid(hand, 0.045, 0.045, 0.050, 3, 5, skin);
    }
    m.build()
}

/// Somebody standing still.
///
/// Not a walker with the animation paused: weight on one leg, the other
/// relaxed, and the hands somewhere. A figure with both feet square and arms
/// hanging reads as a mannequin, which is worse than a walker.
pub(crate) fn standing_geometry(pose: usize, coat: Color, skin: Color, trouser: Color) -> BufferGeometry {
    let mut m = MeshBuilder::default();
    let lean = if pose == 0 { 0.05f32 } else { -0.05 };
    let hip = Vector3::new(lean * 0.10, 0.92, 0.0);
    let shoulder = Vector3::new(lean * 0.16, 1.44, 0.0);
    m.add_limb_flat(hip, shoulder, 0.16, 0.185, 0.60, 8, coat, true);
    let head = Vector3::new(shoulder.x, 1.66, 0.02);
    m.add_limb(
        shoulder,
        Vector3::new(shoulder.x, 1.53, 0.0),
        0.058,
        0.052,
        5,
        skin,
        false,
    );
    m.add_ellipsoid(head, 0.105, 0.128, 0.105, 4, 7, skin);
    m.add_ellipsoid(
        Vector3::new(head.x, head.y + 0.048, head.z - 0.018),
        0.108,
        0.098,
        0.108,
        3,
        7,
        scale_color(coat, 0.45),
    );
    for (i, s) in [-1.0f32, 1.0].into_iter().enumerate() {
        // One leg straight and carrying the weight, the other slack.
        let slack = if (i == 0) == (pose == 0) { 1.0f32 } else { 0.0 };
        let knee = Vector3::new(s * 0.095 + slack * 0.03, 0.50, slack * 0.06);
        m.add_limb_flat(
            Vector3::new(s * 0.09, 0.92, 0.0),
            knee,
            0.095,
            0.080,
            0.85,
            6,
            trouser,
            false,
        );
        let foot = Vector3::new(knee.x, 0.0, knee.z + slack * 0.06);
        m.add_limb_flat(knee, foot, 0.072, 0.056, 0.85, 6, trouser, false);
        m.add_yaw_box(
            Vector3::new(foot.x, 0.035, foot.z + 0.05),
            Vector3::new(0.055, 0.035, 0.105),
            slack * 0.35 * s,
            Color::from_hex(0x25211d),
            Uv::Unit,
        );
        // Hands: one pose has them in pockets, the other holds something up.
        let hand = if pose == 0 {
            Vector3::new(s * 0.19, 0.98, 0.06)
        } else {
            Vector3::new(s * 0.14, 1.24, 0.20)
        };
        m.add_limb(
            Vector3::new(s * 0.165, 1.40, 0.0),
            hand,
            0.054,
            0.044,
            5,
            coat,
            false,
        );
        m.add_ellipsoid(hand, 0.046, 0.046, 0.050, 3, 5, skin);
    }
    m.build()
}

/// A rat. A squirrel's frame with the tail taken off it — long, bare and
/// dragging — and the whole animal dropped closer to the ground.
pub(crate) fn rat_geometry(scurrying: bool) -> BufferGeometry {
    let mut m = MeshBuilder::default();
    let coat = Color::from_hex(0x4a4239);
    let bare = Color::from_hex(0x8a7266);
    let lean = if scurrying { 1.0f32 } else { 0.0 };
    let by = 0.055;
    m.add_limb_flat(
        Vector3::new(0.0, by, -0.075),
        Vector3::new(0.0, by + 0.004, 0.075),
        0.036,
        0.030,
        0.85,
        6,
        coat,
        true,
    );
    let head = Vector3::new(0.0, by - 0.004, 0.115);
    m.add_ellipsoid(head, 0.024, 0.024, 0.034, 3, 6, coat);
    // Snout and ears: small, round and set high, which is the whole of the
    // difference between this and a shrew at ten metres.
    m.add_limb(
        head,
        Vector3::new(0.0, by - 0.012, 0.165),
        0.014,
        0.006,
        4,
        bare,
        true,
    );
    for s in [-1.0f32, 1.0] {
        m.add_ellipsoid(
            Vector3::new(s * 0.020, by + 0.026, 0.098),
            0.016,
            0.016,
            0.005,
            3,
            5,
            bare,
        );
        for z in [-0.05f32, 0.05] {
            m.add_limb(
                Vector3::new(s * 0.028, by - 0.012, z),
                Vector3::new(s * 0.028, 0.0, z + lean * 0.02),
                0.010,
                0.008,
                3,
                coat,
                false,
            );
        }
    }
    // The tail: as long as the body and it stays on the ground.
    let mut prev = Vector3::new(0.0, by - 0.010, -0.080);
    for i in 1..=4 {
        let t = i as f32 / 4.0;
        let p = Vector3::new(
            (t * t) * 0.05 * if scurrying { 1.0 } else { -1.0 },
            0.014,
            -0.080 - t * 0.150,
        );
        m.add_limb(prev, p, 0.011 - t * 0.005, 0.010 - t * 0.005, 4, bare, false);
        prev = p;
    }
    m.build()
}

/// A dragonfly: a long body and four wings held out flat.
///
/// Nothing else here holds its wings *out* at rest, and at this size that flat
/// cross is the entire recognition cue — the body could be a matchstick.
pub(crate) fn dragonfly_geometry(beat: f32, tint: Color) -> BufferGeometry {
    let mut m = MeshBuilder::default();
    let wing = Color::from_hex(0xcfe0ea);
    // Thorax, then a long tapering abdomen behind it.
    m.add_ellipsoid(Vector3::new(0.0, 0.0, 0.02), 0.018, 0.018, 0.030, 3, 5, tint);
    m.add_limb(
        Vector3::new(0.0, 0.0, -0.005),
        Vector3::new(0.0, 0.004, -0.115),
        0.011,
        0.004,
        4,
        scale_color(tint, 0.85),
        true,
    );
    m.add_ellipsoid(
        Vector3::new(0.0, 0.004, 0.052),
        0.019,
        0.019,
        0.019,
        3,
        5,
        Color::from_hex(0x2f4a3a),
    );
    // Four wings, the hind pair swept back a little further than the fore.
    for s in [-1.0f32, 1.0] {
        for (z0, len, lift) in [(0.03f32, 0.10f32, 1.0f32), (-0.01, 0.095, -1.0)] {
            let y = 0.012 * beat * lift;
            {
                let (a, b2, c) = ([0.0, 0.004, z0], [s * len, y, z0 + 0.028], [s * len * 0.9, y, z0 - 0.030]);
                m.tri([a, b2, c], [0.0, 1.0, 0.0], [[0.5, 0.5]; 3], [wing; 3]);
                m.tri([a, c, b2], [0.0, -1.0, 0.0], [[0.5, 0.5]; 3], [wing; 3]);
            }
        }
    }
    m.build()
}

/// A fly. Barely more than a dark speck with a hint of wing, which is all one
/// is — and a knot of them over a bin is the point, not the individual.
pub(crate) fn fly_geometry(beat: f32) -> BufferGeometry {
    let mut m = MeshBuilder::default();
    let body = Color::from_hex(0x1c1a18);
    m.add_ellipsoid(Vector3::new(0.0, 0.0, 0.0), 0.010, 0.009, 0.016, 3, 4, body);
    for s in [-1.0f32, 1.0] {
        m.tri(
            [
                [0.0, 0.004, 0.002],
                [s * 0.017, 0.006 * beat, 0.012],
                [s * 0.015, 0.006 * beat, -0.010],
            ],
            [0.0, 1.0, 0.0],
            [[0.5, 0.5]; 3],
            [Color::from_hex(0x9aa4ac); 3],
        );
    }
    m.build()
}

/// An airliner. Twin-engined, swept wings, and about thirty metres of it.
///
/// It flies a loop like everything else here, but at a radius of a couple of
/// kilometres — over a city three hundred metres across that reads as a
/// straight line, which is what an airliner does, and it costs nothing extra.
pub(crate) fn plane_geometry(strobe: bool) -> BufferGeometry {
    let mut m = MeshBuilder::default();
    let hull = Color::from_hex(0xe8eaec);
    let dark = Color::from_hex(0x2f3438);
    let belly = Color::from_hex(0x2f5f8a);
    // Fuselage along +Z, with a tapered nose and tail.
    m.add_limb_flat(
        Vector3::new(0.0, 0.0, -13.0),
        Vector3::new(0.0, 0.0, 12.0),
        1.55,
        1.55,
        0.92,
        9,
        hull,
        false,
    );
    m.add_limb_flat(
        Vector3::new(0.0, 0.0, 12.0),
        Vector3::new(0.0, -0.25, 15.5),
        1.55,
        0.35,
        0.92,
        8,
        hull,
        true,
    );
    m.add_limb_flat(
        Vector3::new(0.0, 0.0, -13.0),
        Vector3::new(0.0, 0.85, -17.5),
        1.5,
        0.30,
        0.92,
        8,
        hull,
        true,
    );
    // A cheatline down the flanks, which is most of what makes a white tube
    // read as an airliner.
    for sx in [-1.0f32, 1.0] {
        m.add_box(
            Vector3::new(sx * 1.56 - 0.04, -0.35, -12.0),
            Vector3::new(sx * 1.56 + 0.04, 0.15, 13.0),
            belly,
            Uv::Unit,
        );
    }
    // Wings: swept back, with a little dihedral.
    for sx in [-1.0f32, 1.0] {
        for (a, b, c) in [
            (
                [sx * 1.4, -0.4, 3.0],
                [sx * 16.0, 1.1, -6.0],
                [sx * 1.4, -0.4, -3.5],
            ),
            (
                [sx * 16.0, 1.1, -6.0],
                [sx * 15.2, 1.1, -8.2],
                [sx * 1.4, -0.4, -3.5],
            ),
        ] {
            m.tri([a, b, c], [0.0, 1.0, 0.0], [[0.5, 0.5]; 3], [hull; 3]);
            m.tri([a, c, b], [0.0, -1.0, 0.0], [[0.5, 0.5]; 3], [hull; 3]);
        }
        // Engine slung under and ahead of the wing.
        m.add_limb_flat(
            Vector3::new(sx * 7.0, -1.4, 1.6),
            Vector3::new(sx * 7.0, -1.1, -2.6),
            1.05,
            1.0,
            1.0,
            8,
            hull,
            true,
        );
        m.add_limb(
            Vector3::new(sx * 7.0, -1.4, 1.65),
            Vector3::new(sx * 7.0, -1.4, 1.5),
            0.95,
            0.95,
            8,
            dark,
            true,
        );
        // Tailplane.
        {
            let (a, b, c) = (
            [sx * 1.2, 0.4, -12.2],
            [sx * 6.4, 0.9, -16.2],
            [sx * 1.2, 0.4, -15.8],
        );
            m.tri([a, b, c], [0.0, 1.0, 0.0], [[0.5, 0.5]; 3], [hull; 3]);
            m.tri([a, c, b], [0.0, -1.0, 0.0], [[0.5, 0.5]; 3], [hull; 3]);
        }
        // Navigation light: red to port, green to starboard.
        m.add_ellipsoid(
            Vector3::new(sx * 15.8, 1.1, -7.0),
            0.42,
            0.42,
            0.42,
            3,
            5,
            if sx < 0.0 {
                Color::from_hex(0xff2a1a)
            } else {
                Color::from_hex(0x2aff5a)
            },
        );
    }
    // Fin.
    {
        let (a, b, c) = (
        [0.0, 0.7, -11.5],
        [0.0, 6.4, -16.6],
        [0.0, 0.7, -16.2],
    );
        m.tri([a, b, c], [1.0, 0.0, 0.0], [[0.5, 0.5]; 3], [belly; 3]);
        m.tri([a, c, b], [-1.0, 0.0, 0.0], [[0.5, 0.5]; 3], [belly; 3]);
    }
    // The anti-collision strobe on the belly, on half the cycle.
    if strobe {
        m.add_ellipsoid(
            Vector3::new(0.0, -1.7, 0.0),
            0.5,
            0.5,
            0.5,
            3,
            5,
            Color::from_hex(0xfff4e0),
        );
    }
    m.build()
}

/// A helicopter. The rotor disc is the whole silhouette from below and from a
/// distance, so it is drawn as a disc rather than as blades.
pub(crate) fn heli_geometry(blade: f32) -> BufferGeometry {
    let mut m = MeshBuilder::default();
    let body = Color::from_hex(0x2f4a6b);
    let glass = Color::from_hex(0x2a3038);
    let dark = Color::from_hex(0x2b2d31);
    m.add_limb_flat(
        Vector3::new(0.0, 0.0, -2.2),
        Vector3::new(0.0, 0.1, 2.6),
        1.35,
        1.05,
        0.86,
        8,
        body,
        true,
    );
    // Nose glazing, and a boom running back to the tail.
    m.add_ellipsoid(Vector3::new(0.0, 0.0, 3.0), 1.0, 0.95, 1.2, 4, 7, glass);
    m.add_limb_flat(
        Vector3::new(0.0, 0.35, -2.2),
        Vector3::new(0.0, 0.75, -8.4),
        0.55,
        0.24,
        0.9,
        7,
        body,
        false,
    );
    // Fin and tail rotor.
    {
        let (a, b, c) = (
        [0.0, 0.75, -8.6],
        [0.0, 3.0, -9.4],
        [0.0, 0.75, -9.6],
    );
        m.tri([a, b, c], [1.0, 0.0, 0.0], [[0.5, 0.5]; 3], [body; 3]);
        m.tri([a, c, b], [-1.0, 0.0, 0.0], [[0.5, 0.5]; 3], [body; 3]);
    }
    for k in 0..2 {
        let a = blade * PI + k as f32 * PI;
        m.add_limb(
            Vector3::new(0.35, 1.9 - a.sin() * 1.2, -9.1 - a.cos() * 1.2),
            Vector3::new(0.35, 1.9 + a.sin() * 1.2, -9.1 + a.cos() * 1.2),
            0.10,
            0.06,
            4,
            dark,
            false,
        );
    }
    // Mast, then the main rotor: a thin disc with two blades over it, so it
    // reads as spinning rather than as a propeller stopped for a photograph.
    m.add_cylinder(
        Vector3::new(0.0, 1.35, 0.2),
        0.22,
        0.18,
        0.75,
        6,
        dark,
        true,
        Uv::Unit,
    );
    m.add_ground_disc(0.0, 0.2, 7.2, 20, 0.0, 2.12, Color::new(0.30, 0.32, 0.35));
    for k in 0..2 {
        let a = blade * PI * 0.5 + k as f32 * PI;
        m.add_limb(
            Vector3::new(-a.sin() * 7.0, 2.16, 0.2 - a.cos() * 7.0),
            Vector3::new(a.sin() * 7.0, 2.16, 0.2 + a.cos() * 7.0),
            0.16,
            0.16,
            4,
            dark,
            false,
        );
    }
    // Skids.
    for sx in [-1.0f32, 1.0] {
        m.add_limb(
            Vector3::new(sx * 0.95, -1.35, -1.6),
            Vector3::new(sx * 0.95, -1.35, 2.0),
            0.10,
            0.10,
            5,
            dark,
            true,
        );
        m.add_limb(
            Vector3::new(sx * 0.75, -0.55, 0.2),
            Vector3::new(sx * 0.95, -1.35, 0.2),
            0.08,
            0.08,
            4,
            dark,
            false,
        );
        m.add_ellipsoid(
            Vector3::new(sx * 1.35, 0.1, 0.4),
            0.26,
            0.26,
            0.26,
            3,
            5,
            if sx < 0.0 {
                Color::from_hex(0xff2a1a)
            } else {
                Color::from_hex(0x2aff5a)
            },
        );
    }
    m.build()
}

/// A quadcopter. Small, and defined entirely by the four rotor discs.
pub(crate) fn drone_geometry(spin: f32) -> BufferGeometry {
    let mut m = MeshBuilder::default();
    let shell = Color::from_hex(0x2b2d31);
    let trim = Color::from_hex(0xd8d3c4);
    m.add_ellipsoid(Vector3::new(0.0, 0.0, 0.0), 0.22, 0.11, 0.30, 3, 6, shell);
    // Gimballed camera under the nose.
    m.add_ellipsoid(Vector3::new(0.0, -0.16, 0.16), 0.10, 0.10, 0.10, 3, 5, trim);
    for (sx, sz) in [(-1.0f32, 1.0f32), (1.0, 1.0), (1.0, -1.0), (-1.0, -1.0)] {
        let (ax, az) = (sx * 0.38, sz * 0.38);
        m.add_limb(
            Vector3::new(sx * 0.12, 0.0, sz * 0.14),
            Vector3::new(ax, 0.04, az),
            0.035,
            0.03,
            4,
            shell,
            false,
        );
        m.add_cylinder(
            Vector3::new(ax, 0.04, az),
            0.05,
            0.05,
            0.06,
            5,
            shell,
            true,
            Uv::Unit,
        );
        // The disc, plus two blades so the spin reads at close range.
        m.add_ground_disc(ax, az, 0.20, 10, 0.0, 0.115, Color::new(0.26, 0.27, 0.30));
        for k in 0..2 {
            let a = spin * PI + k as f32 * PI + (sx * sz) * 0.6;
            m.add_limb(
                Vector3::new(ax - a.sin() * 0.19, 0.12, az - a.cos() * 0.19),
                Vector3::new(ax + a.sin() * 0.19, 0.12, az + a.cos() * 0.19),
                0.02,
                0.02,
                3,
                shell,
                false,
            );
        }
        // A navigation LED under each arm: green forward, red aft.
        m.add_ellipsoid(
            Vector3::new(ax, -0.02, az),
            0.045,
            0.045,
            0.045,
            3,
            4,
            if sz > 0.0 {
                Color::from_hex(0x2aff5a)
            } else {
                Color::from_hex(0xff2a1a)
            },
        );
    }
    m.build()
}
