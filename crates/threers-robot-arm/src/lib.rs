//! A robot cell for `threers-physics`, and a recording of it you can scrub.
//!
//! ```text
//! cargo run -p threers-robot-arm --example console      # prints a report
//! crates/threers-robot-arm/build.sh                     # builds the web player
//! ```
//!
//! # Two arms, one programme
//!
//! The same pick-and-place runs on two different machines:
//!
//! - [`Mode::Kinematic`] — links placed exactly where the solver says. Immune to
//!   its payload, exact to a millimetre, and the right model for an arm that is
//!   scenery.
//! - [`Mode::Servo`] — the same geometry out of dynamic bodies, welded together
//!   by joints that give. It sags under load, lags behind fast commands, and can
//!   be shoved. The right model for an arm that is part of the game.
//!
//! Running both from one solver output is the point: what you see is the
//! difference between the models, not between two demos.
//!
//! # Why it records
//!
//! A simulation you can only watch forwards is hard to inspect. The whole
//! programme is a few hundred deterministic steps, so it is run once up front
//! and every body's pose kept. Scrubbing is then a array index rather than a
//! re-simulation, which is what makes dragging a slider feel like dragging a
//! video and not like waiting.

pub mod arm;
#[cfg(target_arch = "wasm32")]
mod web;

pub use arm::{ArmDriver, Chain, KinematicArm, ServoArm, COLUMN, ELBOW, LINKS};

use threers_physics::prelude::*;

/// Which machine is running the programme.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    /// Links teleported to the solved pose. Exact, and unmoved by anything.
    Kinematic,
    /// Links with mass, held by servos that give under load.
    Servo,
}

impl Mode {
    pub fn from_index(i: u32) -> Self {
        match i {
            1 => Self::Servo,
            _ => Self::Kinematic,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Kinematic => "kinematic",
            Self::Servo => "servo-driven",
        }
    }
}

/// A step of the programme the arm is running.
#[derive(Debug, Clone, Copy)]
pub struct Move {
    pub name: &'static str,
    pub target: Vector3,
    /// Frames to spend easing into it. Real arms ramp; so does this, because
    /// snapping between poses would give the payload an impulse it should never
    /// see.
    pub frames: usize,
    pub grip: Option<bool>,
}

/// The pick-and-place programme, relative to an arm's own base.
pub fn programme_for(base: Vector3) -> Vec<Move> {
    let bin = bin_for(base);
    let home = base + Vector3::new(0.75, COLUMN + 0.55, 0.0);
    vec![
        Move { name: "home",         target: home,                                  frames: 60,  grip: None },
        Move { name: "approach",     target: base + Vector3::new(0.95, 0.36, 0.0),  frames: 90,  grip: None },
        // Down until the tool rests on the part's top face, not through it: the
        // ball is 0.05 and the cube's top is at 0.10, so 0.155 is contact. Any
        // lower shoves the part away before the gripper can take hold — which is
        // exactly what a real cell would do.
        Move { name: "descend",      target: base + Vector3::new(0.95, 0.155, 0.0), frames: 60,  grip: None },
        Move { name: "grip",         target: base + Vector3::new(0.95, 0.155, 0.0), frames: 25,  grip: Some(true) },
        Move { name: "lift",         target: base + Vector3::new(0.85, 0.55, 0.0),  frames: 70,  grip: None },
        Move { name: "traverse",     target: base + Vector3::new(0.20, 0.60, 0.60), frames: 110, grip: None },
        Move { name: "over the bin", target: bin + Vector3::new(0.0, 0.42, 0.0),    frames: 90,  grip: None },
        Move { name: "release",      target: bin + Vector3::new(0.0, 0.42, 0.0),    frames: 25,  grip: Some(false) },
        Move { name: "retreat",      target: base + Vector3::new(0.1, 0.85, 0.5),   frames: 70,  grip: None },
        Move { name: "home",         target: home,                                  frames: 90,  grip: None },
    ]
}

pub fn bin_for(base: Vector3) -> Vector3 {
    base + Vector3::new(-0.15, 0.0, 0.85)
}

/// How a body should be drawn. Kept separate from the per-frame poses because it
/// never changes, and sending it once per frame would be most of the data.
#[derive(Debug, Clone, Copy)]
pub struct Drawable {
    /// 0 capsule, 1 box, 2 sphere, 3 cylinder.
    pub kind: u32,
    /// Half-extents for a box; `(radius, half_height, _)` otherwise.
    pub size: [f32; 3],
    /// 0 arm, 1 payload, 2 bin, 3 base, 4 floor.
    pub role: u32,
}

/// Everything needed to draw one instant.
pub struct Frame {
    /// Position and rotation of every drawable, in order.
    pub poses: Vec<f32>,
    /// Index into [`Recording::moves`].
    pub move_index: u32,
    /// Distance from the tool to where it was commanded, in metres.
    pub tool_error: f32,
    pub gripping: bool,
}

/// One complete run of the programme, kept so it can be replayed at any speed,
/// in any direction, from any point.
pub struct Recording {
    pub mode: Mode,
    pub drawables: Vec<Drawable>,
    pub frames: Vec<Frame>,
    /// Move names in programme order, for chapter markers.
    pub moves: Vec<&'static str>,
    /// First frame of each move.
    pub move_starts: Vec<u32>,
    /// Worst tool error over the run, in metres.
    pub worst_error: f32,
    /// Where the part finished, and whether that was in the bin.
    pub landed: Vector3,
    pub placed: bool,
}

/// Build the cell and run the programme, keeping every frame.
///
/// `arms` is how many machines to put side by side; each gets its own part and
/// its own bin. `stiffness` only matters in [`Mode::Servo`].
pub fn record(mode: Mode, arms: usize, stiffness: f32) -> Recording {
    const PITCH: f32 = 2.6;
    const DT: f32 = 1.0 / 60.0;
    let arms = arms.clamp(1, 8);

    let mut world = World::new();
    let mut drawables = Vec::new();
    let mut tracked: Vec<BodyId> = Vec::new();

    let floor_depth = PITCH * arms as f32;
    let floor = world.add_body(
        RigidBody::fixed()
            .shape(Shape::cuboid(2.0, 0.05, floor_depth * 0.5 + 0.6))
            .translation(Vector3::new(
                0.0,
                -0.05,
                PITCH * (arms as f32 - 1.0) * 0.5,
            ))
            .friction(0.8),
    );
    tracked.push(floor);
    drawables.push(Drawable {
        kind: 1,
        size: [2.0, 0.05, floor_depth * 0.5 + 0.6],
        role: 4,
    });

    // Bases, drawn but not driven.
    for i in 0..arms {
        let base = Vector3::new(0.0, 0.0, i as f32 * PITCH);
        let column = world.add_body(
            RigidBody::fixed()
                .shape(Shape::cylinder(0.06, 0.22))
                .translation(base + Vector3::new(0.0, 0.06, 0.0)),
        );
        tracked.push(column);
        drawables.push(Drawable {
            kind: 3,
            size: [0.22, 0.06, 0.0],
            role: 3,
        });
    }

    let mut drivers: Vec<Box<dyn ArmDriver>> = Vec::new();
    let mut payloads = Vec::new();
    let mut bins = Vec::new();

    for i in 0..arms {
        let base = Vector3::new(0.0, 0.0, i as f32 * PITCH);
        let driver: Box<dyn ArmDriver> = match mode {
            Mode::Kinematic => Box::new(KinematicArm::build(&mut world, base)),
            Mode::Servo => Box::new(ServoArm::build(&mut world, base, stiffness)),
        };
        for (k, length) in LINKS.iter().enumerate() {
            let radius = arm::link_radius(k);
            drawables.push(Drawable {
                kind: 0,
                size: [radius, (length * 0.5 - radius).max(0.01), 0.0],
                role: 0,
            });
        }
        tracked.extend_from_slice(driver.bodies());
        drivers.push(driver);

        let payload = world.add_body(
            RigidBody::dynamic()
                .shape(Shape::cuboid(0.05, 0.05, 0.05))
                .mass(0.6)
                .translation(base + Vector3::new(0.95, 0.05, 0.0))
                .friction(0.8)
                .can_sleep(false),
        );
        payloads.push(payload);
        tracked.push(payload);
        drawables.push(Drawable {
            kind: 1,
            size: [0.05, 0.05, 0.05],
            role: 1,
        });

        let bin = bin_for(base);
        bins.push(bin);
        for (dx, dz, hx, hz) in [
            (0.0f32, -0.2f32, 0.2f32, 0.02f32),
            (0.0, 0.2, 0.2, 0.02),
            (-0.2, 0.0, 0.02, 0.2),
            (0.2, 0.0, 0.02, 0.2),
        ] {
            let wall = world.add_body(
                RigidBody::fixed()
                    .shape(Shape::cuboid(hx, 0.09, hz))
                    .translation(bin + Vector3::new(dx, 0.09, dz)),
            );
            tracked.push(wall);
            drawables.push(Drawable {
                kind: 1,
                size: [hx, 0.09, hz],
                role: 2,
            });
        }
    }

    let programme = programme_for(Vector3::ZERO);
    let mut frames = Vec::new();
    let mut moves = Vec::new();
    let mut move_starts = Vec::new();
    let mut worst_error: f32 = 0.0;

    let capture = |world: &World, tracked: &[BodyId]| -> Vec<f32> {
        let mut out = Vec::with_capacity(tracked.len() * 7);
        for id in tracked {
            let (p, q) = world
                .body(*id)
                .map(|b| (b.translation(), b.rotation()))
                .unwrap_or((Vector3::ZERO, Quaternion::identity()));
            out.extend_from_slice(&[p.x, p.y, p.z, q.x, q.y, q.z, q.w]);
        }
        out
    };

    for (index, step) in programme.iter().enumerate() {
        moves.push(step.name);
        move_starts.push(frames.len() as u32);

        let starts: Vec<Vector3> = drivers.iter().map(|d| d.chain().tip()).collect();
        for frame in 0..step.frames {
            // Ease in and out, so the payload is never jerked.
            let t = (frame + 1) as f32 / step.frames as f32;
            let smooth = t * t * (3.0 - 2.0 * t);

            let mut commanded = Vec::with_capacity(drivers.len());
            for (i, driver) in drivers.iter_mut().enumerate() {
                let base = Vector3::new(0.0, 0.0, i as f32 * PITCH);
                let target = step.target + base;
                let waypoint = starts[i] + (target - starts[i]) * smooth;
                driver.aim(&mut world, waypoint);
                commanded.push(waypoint);
            }
            world.step(DT);

            let mut error: f32 = 0.0;
            for (i, driver) in drivers.iter().enumerate() {
                error = error.max((driver.tool_position(&world) - commanded[i]).length());
            }
            worst_error = worst_error.max(error);
            frames.push(Frame {
                poses: capture(&world, &tracked),
                move_index: index as u32,
                tool_error: error,
                gripping: drivers[0].is_gripping(),
            });
        }

        for (i, driver) in drivers.iter_mut().enumerate() {
            match step.grip {
                Some(true) => {
                    driver.close_gripper(&mut world, payloads[i]);
                }
                Some(false) => driver.open_gripper(&mut world),
                None => {}
            }
        }
    }

    // Let the part settle after being let go, so the recording ends at rest.
    moves.push("settling");
    move_starts.push(frames.len() as u32);
    let last = programme.len() as u32;
    for _ in 0..150 {
        world.step(DT);
        frames.push(Frame {
            poses: capture(&world, &tracked),
            move_index: last,
            tool_error: 0.0,
            gripping: false,
        });
    }

    let landed = world
        .body(payloads[0])
        .map(|b| b.translation())
        .unwrap_or(Vector3::ZERO);
    let placed = (0..arms).all(|i| {
        world
            .body(payloads[i])
            .map(|b| {
                let p = b.translation();
                (p.x - bins[i].x).abs() < 0.2 && (p.z - bins[i].z).abs() < 0.2 && p.y < 0.2
            })
            .unwrap_or(false)
    });

    Recording {
        mode,
        drawables,
        frames,
        moves,
        move_starts,
        worst_error,
        landed,
        placed,
    }
}
