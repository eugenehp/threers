//! In-browser benchmark and demo for `threers-physics`.
//!
//! Builds to wasm and exposes a handful of scenes plus the numbers needed to
//! judge them: step time, body count, how many bodies have gone to sleep.
//!
//! The point is to make the cost of each setting *visible*. Substeps, solver
//! iterations and body count all trade against frame time, and the trade is very
//! hard to reason about from documentation alone.
//!
//! ```text
//! crates/threers-physics-bench/build.sh   # then serve web/physics-bench/
//! ```
//!
//! It is also a normal Rust library, so the scenes can be run and timed
//! natively — see `examples/headless_bench.rs`.

use threers_physics::prelude::*;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

/// The demo scenes.
///
/// Each stresses a different part of the engine, because they fail in different
/// ways: stacks expose solver convergence, joints expose constraint
/// propagation, and a spray of mixed shapes exposes the narrow phase.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Scene {
    /// A pyramid of boxes. Tests stacking, friction and sleeping.
    Pyramid,
    /// Bouncing spheres in a box. Tests restitution and the broad phase.
    BouncingBalls,
    /// Hanging chains. Tests joint convergence — the case substepping fixes.
    Chains,
    /// Satellites orbiting a point mass. Tests zero-gravity integration.
    Orbits,
    /// Assorted shapes tipped into a heap. Tests every narrow-phase pair.
    MixedShapes,
    /// Fast bodies against thin walls. Tests continuous collision detection.
    Bullets,
}

impl Scene {
    pub fn from_index(i: u32) -> Self {
        match i {
            1 => Self::BouncingBalls,
            2 => Self::Chains,
            3 => Self::Orbits,
            4 => Self::MixedShapes,
            5 => Self::Bullets,
            _ => Self::Pyramid,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Pyramid => "Pyramid (stacking, friction, sleeping)",
            Self::BouncingBalls => "Bouncing balls (restitution, broad phase)",
            Self::Chains => "Chains (joints, substepping)",
            Self::Orbits => "Orbits (zero gravity, point gravity)",
            Self::MixedShapes => "Mixed shapes (every narrow-phase pair)",
            Self::Bullets => "Bullets (continuous collision detection)",
        }
    }
}

/// Deterministic noise, so a given scene and count always build the same world.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed | 1)
    }
    fn next(&mut self) -> f32 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        ((self.0.wrapping_mul(0x2545F4914F6CDD1D) >> 40) as f32) / ((1u32 << 24) as f32)
    }
    fn range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + self.next() * (hi - lo)
    }
}

/// One benchmark run.
pub struct Bench {
    world: World,
    scene: Scene,
    /// Half-extent of the drawn shape, for the visualiser.
    sizes: Vec<f32>,
    /// A stable colour index per body.
    tints: Vec<f32>,
    ids: Vec<BodyId>,
    last_step_ms: f32,
    stepped: u32,
    total_ms: f32,
}

impl Bench {
    /// Build a scene with roughly `count` dynamic bodies at the given quality.
    pub fn new(scene: Scene, count: u32, quality: SimulationQuality) -> Self {
        let mut world = World::new().with_quality(quality);
        let mut sizes = Vec::new();
        let mut tints = Vec::new();
        let mut ids = Vec::new();
        let mut rng = Rng::new(0x5EED);

        match scene {
            Scene::Pyramid => {
                world.add_body(RigidBody::fixed().shape(Shape::ground()).friction(0.9));
                // Rows shrink by one, so a target count sets the base width.
                let mut rows = 1usize;
                while rows * (rows + 1) / 2 < count as usize {
                    rows += 1;
                }
                let half = 0.5;
                for row in 0..rows {
                    for i in 0..(rows - row) {
                        let n = rows - row;
                        let x = (i as f32 - (n - 1) as f32 * 0.5) * (half * 2.05);
                        ids.push(world.add_body(
                            RigidBody::dynamic()
                                .shape(Shape::cuboid(half, half, half))
                                .translation(Vector3::new(x, half + row as f32 * half * 2.0, 0.0))
                                .friction(0.9),
                        ));
                        sizes.push(half);
                        tints.push(row as f32 / rows as f32);
                    }
                }
            }

            Scene::BouncingBalls => {
                sealed_box(&mut world, 8.0);
                for _ in 0..count {
                    let r = rng.range(0.2, 0.5);
                    ids.push(world.add_body(
                        RigidBody::dynamic()
                            .shape(Shape::ball(r))
                            .translation(Vector3::new(
                                rng.range(-6.0, 6.0),
                                rng.range(-6.0, 6.0),
                                rng.range(-6.0, 6.0),
                            ))
                            .linear_velocity(Vector3::new(
                                rng.range(-8.0, 8.0),
                                rng.range(-8.0, 8.0),
                                rng.range(-8.0, 8.0),
                            ))
                            .restitution(0.9)
                            .gravity_scale(0.0)
                            .can_sleep(false),
                    ));
                    sizes.push(r);
                    tints.push(rng.next());
                }
            }

            Scene::Chains => {
                let per_chain = 16;
                let chains = (count as usize / per_chain).max(1);
                for c in 0..chains {
                    let x = (c as f32 - (chains - 1) as f32 * 0.5) * 1.5;
                    let anchor =
                        world.add_body(RigidBody::fixed().translation(Vector3::new(x, 9.0, 0.0)));
                    let mut previous = anchor;
                    for i in 0..per_chain {
                        let link = world.add_body(
                            RigidBody::dynamic()
                                .shape(Shape::ball(0.18))
                                .translation(Vector3::new(x, 9.0 - (i + 1) as f32 * 0.5, 0.0))
                                .can_sleep(false),
                        );
                        world.add_joint(Joint::distance(
                            previous,
                            link,
                            Vector3::ZERO,
                            Vector3::ZERO,
                            0.5,
                        ));
                        previous = link;
                        ids.push(link);
                        sizes.push(0.18);
                        tints.push(c as f32 / chains as f32);
                    }
                }
                // Something for them to swing into.
                world.add_body(RigidBody::fixed().shape(Shape::ground()).friction(0.5));
            }

            Scene::Orbits => {
                world.gravity = Vector3::ZERO;
                world.gravity_model = GravityModel::Point {
                    centre: Vector3::ZERO,
                    mu: 600.0,
                    min_distance: 1.5,
                };
                for _ in 0..count {
                    let radius = rng.range(4.0, 14.0);
                    let angle = rng.range(0.0, std::f32::consts::TAU);
                    let tilt = rng.range(-0.4, 0.4);
                    let speed = (600.0f32 / radius).sqrt();
                    let position = Vector3::new(
                        angle.cos() * radius,
                        tilt * radius * 0.3,
                        angle.sin() * radius,
                    );
                    // Perpendicular to the radius: a circular orbit.
                    let velocity = Vector3::new(-angle.sin(), 0.0, angle.cos()) * speed;
                    ids.push(world.add_body(
                        RigidBody::dynamic()
                            .shape(Shape::ball(0.25))
                            .translation(position)
                            .linear_velocity(velocity)
                            .linear_damping(0.0)
                            .can_sleep(false),
                    ));
                    sizes.push(0.25);
                    tints.push(radius / 14.0);
                }
            }

            Scene::MixedShapes => {
                world.add_body(RigidBody::fixed().shape(Shape::ground()).friction(0.7));
                for i in 0..count {
                    let (shape, size) = match i % 5 {
                        0 => (Shape::ball(0.4), 0.4),
                        1 => (Shape::cuboid(0.35, 0.35, 0.35), 0.35),
                        2 => (Shape::capsule(0.25, 0.25), 0.4),
                        3 => (Shape::cylinder(0.3, 0.3), 0.35),
                        _ => (Shape::cone(0.35, 0.35), 0.35),
                    };
                    ids.push(world.add_body(
                        RigidBody::dynamic()
                            .shape(shape)
                            .translation(Vector3::new(
                                rng.range(-2.5, 2.5),
                                1.0 + i as f32 * 0.55,
                                rng.range(-2.5, 2.5),
                            ))
                            .friction(0.6)
                            .restitution(0.2),
                    ));
                    sizes.push(size);
                    tints.push((i % 5) as f32 / 5.0);
                }
            }

            Scene::Bullets => {
                world.add_body(RigidBody::fixed().shape(Shape::ground()).friction(0.5));
                // A row of thin walls, exactly what a fast body tunnels through.
                for i in 0..4 {
                    world.add_body(
                        RigidBody::fixed()
                            .shape(Shape::cuboid(0.05, 4.0, 6.0))
                            .translation(Vector3::new(i as f32 * 4.0, 4.0, 0.0)),
                    );
                }
                for i in 0..count {
                    ids.push(world.add_body(
                        RigidBody::dynamic()
                            .shape(Shape::ball(0.15))
                            .translation(Vector3::new(
                                -12.0 - rng.range(0.0, 6.0),
                                1.0 + (i % 12) as f32 * 0.6,
                                rng.range(-2.5, 2.5),
                            ))
                            .linear_velocity(Vector3::new(rng.range(60.0, 140.0), 0.0, 0.0))
                            .ccd(true)
                            .can_sleep(false),
                    ));
                    sizes.push(0.15);
                    tints.push(rng.next());
                }
            }
        }

        Self {
            world,
            scene,
            sizes,
            tints,
            ids,
            last_step_ms: 0.0,
            stepped: 0,
            total_ms: 0.0,
        }
    }

    /// Advance one fixed step. Returns the time it took, in milliseconds.
    ///
    /// Timing is supplied by the caller rather than measured here: `std::time`
    /// panics on wasm, and the browser's own clock is what the page is judged
    /// by anyway.
    pub fn step_fixed(&mut self) {
        self.world.step_fixed();
        self.stepped += 1;
    }

    pub fn record_step_ms(&mut self, ms: f32) {
        self.last_step_ms = ms;
        self.total_ms += ms;
    }

    pub fn scene(&self) -> Scene {
        self.scene
    }

    pub fn body_count(&self) -> usize {
        self.world.body_count()
    }

    pub fn dynamic_count(&self) -> usize {
        self.ids.len()
    }

    pub fn sleeping_count(&self) -> usize {
        self.world.sleeping_count()
    }

    pub fn contact_count(&self) -> usize {
        self.world.contacts().len()
    }

    pub fn last_step_ms(&self) -> f32 {
        self.last_step_ms
    }

    pub fn average_step_ms(&self) -> f32 {
        if self.stepped == 0 {
            0.0
        } else {
            self.total_ms / self.stepped as f32
        }
    }

    pub fn world(&self) -> &World {
        &self.world
    }

    pub fn world_mut(&mut self) -> &mut World {
        &mut self.world
    }

    /// Flat `[x, y, z, radius, tint, sleeping]` per dynamic body, for drawing.
    ///
    /// One packed array rather than a struct per body: crossing the wasm
    /// boundary once per frame instead of once per object is the difference
    /// between the visualiser costing nothing and costing more than the physics.
    pub fn render_data(&self) -> Vec<f32> {
        let mut out = Vec::with_capacity(self.ids.len() * 6);
        for (i, id) in self.ids.iter().enumerate() {
            let Some(body) = self.world.body(*id) else {
                continue;
            };
            let p = body.translation();
            out.push(p.x);
            out.push(p.y);
            out.push(p.z);
            out.push(self.sizes.get(i).copied().unwrap_or(0.3));
            out.push(self.tints.get(i).copied().unwrap_or(0.5));
            out.push(if body.is_sleeping() { 1.0 } else { 0.0 });
        }
        out
    }
}

// ---- wasm bindings --------------------------------------------------------

/// Browser-facing handle. Mirrors [`Bench`], with types JavaScript can hold.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub struct PhysicsBench {
    inner: Bench,
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
impl PhysicsBench {
    /// `scene` is a [`Scene`] index, `quality` is 0 fast / 1 balanced / 2 high.
    #[wasm_bindgen(constructor)]
    pub fn new(scene: u32, count: u32, quality: u32) -> PhysicsBench {
        console_error_panic_hook::set_once();
        let quality = match quality {
            0 => SimulationQuality::Fast,
            2 => SimulationQuality::High,
            _ => SimulationQuality::Balanced,
        };
        PhysicsBench {
            inner: Bench::new(Scene::from_index(scene), count.clamp(1, 4000), quality),
        }
    }

    /// Run one fixed step.
    pub fn step(&mut self) {
        self.inner.step_fixed();
    }

    /// Tell the bench how long the last `step` took, as measured by
    /// `performance.now()` on the JS side.
    pub fn record(&mut self, ms: f32) {
        self.inner.record_step_ms(ms);
    }

    #[wasm_bindgen(js_name = renderData)]
    pub fn render_data(&self) -> Vec<f32> {
        self.inner.render_data()
    }

    #[wasm_bindgen(js_name = bodyCount)]
    pub fn body_count(&self) -> usize {
        self.inner.body_count()
    }

    #[wasm_bindgen(js_name = sleepingCount)]
    pub fn sleeping_count(&self) -> usize {
        self.inner.sleeping_count()
    }

    #[wasm_bindgen(js_name = contactCount)]
    pub fn contact_count(&self) -> usize {
        self.inner.contact_count()
    }

    #[wasm_bindgen(js_name = averageStepMs)]
    pub fn average_step_ms(&self) -> f32 {
        self.inner.average_step_ms()
    }

    #[wasm_bindgen(js_name = sceneName)]
    pub fn scene_name(&self) -> String {
        self.inner.scene().name().to_string()
    }

    /// Fire an impulse into the scene, so the demo can be poked.
    pub fn disturb(&mut self, strength: f32) {
        let ids: Vec<BodyId> = self.inner.ids.clone();
        for (i, id) in ids.iter().enumerate() {
            if let Some(body) = self.inner.world_mut().body_mut(*id) {
                let angle = i as f32 * 0.7;
                body.apply_impulse(
                    Vector3::new(angle.cos(), 1.0, angle.sin()) * (strength * body.mass()),
                );
            }
        }
    }
}

fn sealed_box(world: &mut World, half: f32) {
    let t = 0.5;
    for (centre, extents) in [
        (Vector3::new(0.0, -half - t, 0.0), Vector3::new(half + t, t, half + t)),
        (Vector3::new(0.0, half + t, 0.0), Vector3::new(half + t, t, half + t)),
        (Vector3::new(-half - t, 0.0, 0.0), Vector3::new(t, half + t, half + t)),
        (Vector3::new(half + t, 0.0, 0.0), Vector3::new(t, half + t, half + t)),
        (Vector3::new(0.0, 0.0, -half - t), Vector3::new(half + t, half + t, t)),
        (Vector3::new(0.0, 0.0, half + t), Vector3::new(half + t, half + t, t)),
    ] {
        world.add_body(
            RigidBody::fixed()
                .shape(Shape::cuboid(extents.x, extents.y, extents.z))
                .translation(centre)
                .restitution(0.9),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SCENES: [Scene; 6] = [
        Scene::Pyramid,
        Scene::BouncingBalls,
        Scene::Chains,
        Scene::Orbits,
        Scene::MixedShapes,
        Scene::Bullets,
    ];

    #[test]
    fn every_scene_builds_and_runs_without_blowing_up() {
        for scene in SCENES {
            let mut bench = Bench::new(scene, 60, SimulationQuality::Balanced);
            assert!(bench.dynamic_count() > 0, "{scene:?} produced no bodies");

            for _ in 0..180 {
                bench.step_fixed();
            }

            for chunk in bench.render_data().chunks(6) {
                assert!(
                    chunk[0].is_finite() && chunk[1].is_finite() && chunk[2].is_finite(),
                    "{scene:?} produced a non-finite position"
                );
                assert!(
                    chunk[0].abs() < 1e4 && chunk[1].abs() < 1e4 && chunk[2].abs() < 1e4,
                    "{scene:?} flung a body to {:?}",
                    &chunk[..3]
                );
            }
        }
    }

    #[test]
    fn render_data_is_packed_six_floats_per_body() {
        let bench = Bench::new(Scene::Pyramid, 20, SimulationQuality::Fast);
        let data = bench.render_data();
        assert_eq!(data.len(), bench.dynamic_count() * 6);
        assert!(!data.is_empty());
        // The sleeping flag is a flag.
        for chunk in data.chunks(6) {
            assert!(chunk[5] == 0.0 || chunk[5] == 1.0);
        }
    }

    #[test]
    fn every_quality_setting_produces_a_standing_pyramid() {
        for quality in [
            SimulationQuality::Fast,
            SimulationQuality::Balanced,
            SimulationQuality::High,
        ] {
            let mut bench = Bench::new(Scene::Pyramid, 15, quality);
            for _ in 0..300 {
                bench.step_fixed();
            }
            let highest = bench
                .render_data()
                .chunks(6)
                .map(|c| c[1])
                .fold(0.0f32, f32::max);
            assert!(highest > 2.0, "{quality:?} collapsed the pyramid to {highest}");
        }
    }

    #[test]
    fn the_pyramid_settles_and_sleeps() {
        let mut bench = Bench::new(Scene::Pyramid, 15, SimulationQuality::Balanced);
        for _ in 0..400 {
            bench.step_fixed();
        }
        assert!(
            bench.sleeping_count() >= bench.dynamic_count(),
            "only {} of {} asleep",
            bench.sleeping_count(),
            bench.dynamic_count()
        );
    }

    #[test]
    fn bullets_do_not_pass_through_the_walls() {
        let mut bench = Bench::new(Scene::Bullets, 24, SimulationQuality::Balanced);
        for _ in 0..240 {
            bench.step_fixed();
        }
        // The last wall is at x = 12; nothing should be far past it.
        let furthest = bench
            .render_data()
            .chunks(6)
            .map(|c| c[0])
            .fold(f32::MIN, f32::max);
        assert!(furthest < 13.0, "a bullet tunnelled through to x = {furthest}");
    }

    #[test]
    fn orbits_stay_in_orbit() {
        let mut bench = Bench::new(Scene::Orbits, 40, SimulationQuality::Balanced);
        for _ in 0..900 {
            bench.step_fixed();
        }
        for chunk in bench.render_data().chunks(6) {
            let r = (chunk[0] * chunk[0] + chunk[1] * chunk[1] + chunk[2] * chunk[2]).sqrt();
            assert!(r > 1.0 && r < 30.0, "a satellite left its orbit, r = {r}");
        }
    }

    #[test]
    fn scene_indices_round_trip() {
        for (i, scene) in SCENES.iter().enumerate() {
            assert_eq!(Scene::from_index(i as u32), *scene);
            assert!(!scene.name().is_empty());
        }
        // Out of range falls back rather than panicking.
        assert_eq!(Scene::from_index(999), Scene::Pyramid);
    }
}
