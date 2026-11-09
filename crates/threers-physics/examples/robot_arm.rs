//! A four-axis arm doing a pick-and-place, driven kinematically.
//!
//! ```text
//! cargo run -p threers-physics --example robot_arm
//! ```
//!
//! # Why kinematic
//!
//! A real robot arm is not pushed around by the world. Its controller commands
//! joint angles and the servos hold them against whatever load turns up — so a
//! payload that is heavier than expected makes the arm *slower*, not lower.
//! Modelling that with motors and torques means tuning a controller until it
//! happens to behave, and the tuning is wrong again the moment the payload
//! changes.
//!
//! Kinematic bodies are the honest match. The simulation puts them exactly where
//! they are told, they are immovable by anything they touch, and — crucially —
//! they still push what *is* movable, with the velocity implied by how far they
//! travelled. That is precisely a servo with enough torque.
//!
//! The pieces this exercises together:
//!
//! - [`IkChain::solve_ccd`] to turn a target into a chain pose. CCD rather
//!   than FABRIK because it rotates about one hinge at a time, which is what a
//!   machine with real hinges actually does.
//! - A decoupled wrist, as industrial arms have: the tool hangs straight down
//!   from the wrist, so the solver places the *wrist* above the target and
//!   the tool's attitude never depends on how the solve happened to go.
//!   Solving for the tip alone lets the chain reach it from any side, and the
//!   nearest such pose was the elbow under the table and the forearm through
//!   the payload.
//! - [`IkConstraint::Hinge`] limits, so the elbow bends one way only (up).
//!   The hinge axis turns with the arm's heading, since the arm swings about
//!   its base and a world-fixed axis would only be right facing one way.
//! - Kinematic [`RigidBody`]s driven by [`RigidBody::set_kinematic_target`], one
//!   per link, posed from the solved chain.
//! - A [`Joint::fixed_at_point`] created and destroyed at runtime, which is how
//!   a magnetic or vacuum gripper behaves: it grabs what it is touching, carries
//!   it, and lets go.
//!
//! # What it checks
//!
//! Everything printed is also asserted, so this doubles as an integration test.
//! An arm that is "working properly" has to satisfy all of:
//!
//! 1. The tool reaches every waypoint inside its own tolerance.
//! 2. Segment lengths never change — a stretching arm means the solver gave up.
//! 3. The elbow stays inside its limits at every step, not just at waypoints.
//! 4. Sending it somewhere unreachable is reported, not silently approximated.
//! 5. The payload is actually carried, and actually lands in the bin.
//! 6. Going home twice gives the same pose both times.

use std::f32::consts::PI;
use threers_physics::prelude::*;

const DT: f32 = 1.0 / 60.0;

/// Where the arm is bolted down.
const BASE: Vector3 = Vector3::new(0.0, 0.0, 0.0);
/// Shoulder height above the base plate.
const COLUMN: f32 = 0.55;
/// Upper arm, forearm, and the tool sticking out of the wrist.
const LINKS: [f32; 3] = [0.80, 0.70, 0.22];
/// How far the elbow may bend, from straight, in radians: never backwards, and
/// never folded flat onto the upper arm.
const ELBOW: (f32, f32) = (0.15, 2.6);

/// The arm's pose: the shoulder–elbow–wrist chain the solver moves, and a tool
/// hanging straight down from the wrist.
struct Arm {
    chain: IkChain,
    /// Kinematic bodies, one per segment.
    links: Vec<BodyId>,
    /// The tool flange — what the gripper attaches to.
    tool: BodyId,
    /// Set while carrying something.
    grip: Option<JointId>,
}

impl Arm {
    fn build(world: &mut World) -> Self {
        // The base plate is fixed: it is bolted to the table and nothing should
        // move it, including a collision with the arm's own payload.
        world.add_body(
            RigidBody::fixed()
                .shape(Shape::cylinder(0.06, 0.22))
                .translation(BASE + Vector3::new(0.0, 0.06, 0.0)),
        );

        // Joints run up the column and then out along +X, which puts the arm in
        // a sensible starting pose rather than folded through itself. The
        // elbow starts slightly bent, so its hinge has a side to bend to.
        let shoulder = BASE + Vector3::new(0.0, COLUMN, 0.0);
        let elbow = shoulder + Vector3::new(LINKS[0] * 0.8, LINKS[0] * 0.6, 0.0);
        let wrist = elbow + Vector3::new(LINKS[1], 0.0, 0.0);

        let mut chain = IkChain::from_points(&[shoulder, elbow, wrist]);
        chain.max_iterations = 60;
        chain.tolerance = 0.002;
        // The shoulder swings within a cone about the column.
        chain.base_direction = Some(Vector3::new(0.0, 1.0, 0.0));
        chain.set_constraint(0, IkConstraint::cone(2.2));

        // One kinematic body per segment. Capsules, because a robot link is a
        // rounded bar and a capsule is the cheapest shape that behaves like one.
        let mut links = Vec::new();
        for (i, length) in LINKS.iter().enumerate() {
            let radius = if i == 2 { 0.045 } else { 0.07 };
            links.push(
                world.add_body(
                    RigidBody::kinematic()
                        .shape(Shape::capsule((length * 0.5 - radius).max(0.01), radius))
                        .friction(0.9),
                ),
            );
        }
        // The flange: a small body at the very tip, which is what grabs things.
        let tool = world.add_body(
            RigidBody::kinematic()
                .shape(Shape::ball(0.05))
                .friction(1.0),
        );

        let mut arm = Self {
            chain,
            links,
            tool,
            grip: None,
        };
        arm.drive(world);
        arm
    }

    /// Point the tool at `target`, and report what the solver managed.
    fn aim(&mut self, target: Vector3) -> IkResult {
        let wrist = target + Vector3::new(0.0, LINKS[2], 0.0);
        // The elbow's hinge is horizontal and across the arm's heading, so it
        // bends in the vertical plane the arm points along.
        let heading = Vector3::new(wrist.x, 0.0, wrist.z);
        if heading.length() > 1e-4 {
            let axis = Vector3::new(0.0, 1.0, 0.0).cross(heading.normalize());
            self.chain
                .set_constraint(1, IkConstraint::hinge_limited(axis, ELBOW.0, ELBOW.1));
        }
        let mut result = self.chain.solve_ccd(wrist);
        result.error = (self.tip() - target).length();
        result
    }

    /// Every joint, tool tip last.
    fn points(&self) -> [Vector3; 4] {
        let j = &self.chain.joints;
        [j[0].position, j[1].position, j[2].position, self.tip()]
    }

    /// The tool tip, straight below the wrist.
    fn tip(&self) -> Vector3 {
        self.chain.tip() - Vector3::new(0.0, LINKS[2], 0.0)
    }

    /// Push the current chain pose into the kinematic bodies.
    ///
    /// This is forward kinematics: the solver gives joint *positions*, and a
    /// link is the capsule spanning two of them. Setting a target rather than a
    /// position is what makes the arm push things — the simulation derives the
    /// velocity it must have travelled at, and that velocity is what a contact
    /// sees.
    fn drive(&mut self, world: &mut World) {
        let points = self.points();
        for (i, body) in self.links.iter().enumerate() {
            let (a, b) = (points[i], points[i + 1]);
            let centre = (a + b) * 0.5;
            // Capsules are Y-aligned, so the link's rotation is whatever takes
            // +Y onto the segment.
            let rotation = rotation_from_to(Vector3::new(0.0, 1.0, 0.0), b - a);
            if let Some(link) = world.body_mut(*body) {
                link.set_kinematic_target(Isometry::new(centre, rotation));
            }
        }
        let tip = self.tip();
        if let Some(tool) = world.body_mut(self.tool) {
            tool.set_kinematic_target(Isometry::new(tip, Quaternion::identity()));
        }
    }

    /// Teleport into place, for the initial pose only. Using this mid-run would
    /// move the arm without any velocity, so it would pass through things.
    fn snap(&mut self, world: &mut World) {
        let points = self.points();
        for (i, body) in self.links.iter().enumerate() {
            let (a, b) = (points[i], points[i + 1]);
            if let Some(link) = world.body_mut(*body) {
                link.set_translation((a + b) * 0.5);
                link.set_rotation(rotation_from_to(Vector3::new(0.0, 1.0, 0.0), b - a));
            }
        }
        let tip = self.tip();
        if let Some(tool) = world.body_mut(self.tool) {
            tool.set_translation(tip);
        }
    }

    /// Grab whatever the tool is touching, if anything is close enough.
    ///
    /// A fixed joint, made at the moment of contact. That is what a vacuum cup
    /// or an electromagnet is: it does not squeeze, it just holds whatever it
    /// met until told otherwise — and the payload's mass and inertia carry on
    /// mattering, so a heavy box still swings the arm's tip on the way round.
    fn close_gripper(&mut self, world: &mut World, payload: BodyId) -> bool {
        if self.grip.is_some() {
            return true;
        }
        let tip = self.tip();
        let Some(target) = world.body(payload) else {
            return false;
        };
        if (target.translation() - tip).length() > 0.16 {
            return false;
        }
        let joint = Joint::fixed_at_point(world.bodies(), self.tool, payload, tip);
        self.grip = joint.map(|j| world.add_joint(j));
        self.grip.is_some()
    }

    fn open_gripper(&mut self, world: &mut World) {
        if let Some(id) = self.grip.take() {
            world.remove_joint(id);
        }
    }

    /// Joint angles in degrees, the way a robot controller would report them.
    fn angles(&self) -> Vec<f32> {
        let mut out = Vec::new();
        let mut previous = Vector3::new(0.0, 1.0, 0.0); // up the column
        let points = self.points();
        for i in 0..points.len() - 1 {
            let segment = points[i + 1] - points[i];
            let (a, b) = (previous.normalize(), segment.normalize());
            out.push(a.dot(b).clamp(-1.0, 1.0).acos().to_degrees());
            previous = segment;
        }
        out
    }

    /// Longest deviation of any segment from the length it was built with.
    ///
    /// The one number that says whether the solve is trustworthy: FABRIK and CCD
    /// both hold lengths by construction, so a non-zero value here means
    /// something has gone wrong that a reach error would not reveal.
    fn stretch(&self) -> f32 {
        let mut worst: f32 = 0.0;
        let points = self.points();
        for (i, expected) in LINKS.iter().enumerate() {
            let actual = (points[i + 1] - points[i]).length();
            worst = worst.max((actual - expected).abs());
        }
        worst
    }
}

/// Shortest rotation taking `from` onto `to`.
fn rotation_from_to(from: Vector3, to: Vector3) -> Quaternion {
    let (a, b) = (from.normalize(), to.normalize());
    let dot = a.dot(b).clamp(-1.0, 1.0);
    if dot > 0.999_999 {
        return Quaternion::identity();
    }
    if dot < -0.999_999 {
        // Opposed: any perpendicular axis will do.
        let axis = if a.x.abs() < 0.9 {
            a.cross(Vector3::new(1.0, 0.0, 0.0))
        } else {
            a.cross(Vector3::new(0.0, 1.0, 0.0))
        };
        return Quaternion::from_axis_angle(axis.normalize(), PI);
    }
    Quaternion::from_axis_angle(a.cross(b).normalize(), dot.acos())
}

/// A step of the programme the arm is running.
struct Move {
    name: &'static str,
    target: Vector3,
    /// Frames to spend easing into it. Real arms ramp; so does this, because
    /// snapping between poses would give the payload an impulse it should
    /// never see.
    frames: usize,
    grip: Option<bool>,
}

fn main() {
    let mut world = World::new();
    world.add_body(
        RigidBody::fixed()
            .shape(Shape::cuboid(1.5, 0.05, 1.5))
            .translation(Vector3::new(0.0, -0.05, 0.0))
            .friction(0.8),
    );

    let mut arm = Arm::build(&mut world);
    arm.snap(&mut world);

    // The payload, and a bin with three walls to drop it into.
    let payload = world.add_body(
        RigidBody::dynamic()
            .shape(Shape::cuboid(0.05, 0.05, 0.05))
            .mass(0.6)
            .translation(Vector3::new(0.95, 0.05, 0.0))
            .friction(0.8)
            .can_sleep(false),
    );
    let bin = Vector3::new(-0.15, 0.0, 0.85);
    for (dx, dz, hx, hz) in [
        (0.0f32, -0.2f32, 0.2f32, 0.02f32),
        (0.0, 0.2, 0.2, 0.02),
        (-0.2, 0.0, 0.02, 0.2),
        (0.2, 0.0, 0.02, 0.2),
    ] {
        world.add_body(
            RigidBody::fixed()
                .shape(Shape::cuboid(hx, 0.09, hz))
                .translation(bin + Vector3::new(dx, 0.09, dz)),
        );
    }

    let home = Vector3::new(0.75, COLUMN + 0.55, 0.0);
    let programme = [
        Move {
            name: "home",
            target: home,
            frames: 60,
            grip: None,
        },
        Move {
            name: "approach",
            target: Vector3::new(0.95, 0.34, 0.0),
            frames: 90,
            grip: None,
        },
        Move {
            name: "descend",
            // The tool's 5 cm ball resting on the payload's top (y 0.10).
            target: Vector3::new(0.95, 0.155, 0.0),
            frames: 60,
            grip: None,
        },
        Move {
            name: "grip",
            target: Vector3::new(0.95, 0.155, 0.0),
            frames: 20,
            grip: Some(true),
        },
        Move {
            name: "lift",
            target: Vector3::new(0.85, 0.55, 0.0),
            frames: 70,
            grip: None,
        },
        Move {
            name: "traverse",
            target: Vector3::new(0.20, 0.60, 0.60),
            frames: 110,
            grip: None,
        },
        Move {
            name: "over the bin",
            target: bin + Vector3::new(0.0, 0.42, 0.0),
            frames: 90,
            grip: None,
        },
        Move {
            name: "release",
            target: bin + Vector3::new(0.0, 0.42, 0.0),
            frames: 20,
            grip: Some(false),
        },
        Move {
            name: "retreat",
            target: Vector3::new(0.1, 0.85, 0.5),
            frames: 70,
            grip: None,
        },
        Move {
            name: "home",
            target: home,
            frames: 90,
            grip: None,
        },
    ];

    println!(
        "4-axis arm — reach {:.2} m from a shoulder at {:.2} m",
        LINKS.iter().sum::<f32>(),
        COLUMN
    );
    println!();
    println!(
        "{:<14} {:>7} {:>8} {:>9}  {:<22} gripper",
        "move", "err mm", "stretch", "payload y", "joint angles (deg)"
    );
    println!("{}", "-".repeat(78));

    let mut worst_error: f32 = 0.0;
    let mut worst_stretch: f32 = 0.0;
    let mut elbow_min = f32::MAX;
    let mut elbow_max = f32::MIN;
    let mut home_poses: Vec<Vec<f32>> = Vec::new();

    for step in &programme {
        let from = arm.tip();
        for frame in 0..step.frames {
            // Ease in and out, so the payload is never jerked.
            let t = (frame + 1) as f32 / step.frames as f32;
            let smooth = t * t * (3.0 - 2.0 * t);
            let waypoint = from + (step.target - from) * smooth;

            let result = arm.aim(waypoint);
            arm.drive(&mut world);
            world.step(DT);

            worst_stretch = worst_stretch.max(arm.stretch());
            let elbow = arm.angles()[1];
            elbow_min = elbow_min.min(elbow);
            elbow_max = elbow_max.max(elbow);

            assert!(
                !result.out_of_reach,
                "'{}' asked for {waypoint:?}, which is outside the workspace",
                step.name
            );
        }

        match step.grip {
            Some(true) => {
                let got = arm.close_gripper(&mut world, payload);
                assert!(got, "the gripper closed on nothing at {:?}", arm.tip());
            }
            Some(false) => arm.open_gripper(&mut world),
            None => {}
        }

        let error = (arm.tip() - step.target).length();
        worst_error = worst_error.max(error);
        let angles = arm.angles();
        if step.name == "home" {
            home_poses.push(angles.clone());
        }

        println!(
            "{:<14} {:>7.2} {:>8.4} {:>9.3}  {:<22} {}",
            step.name,
            error * 1000.0,
            arm.stretch(),
            world.body(payload).unwrap().translation().y,
            angles
                .iter()
                .map(|a| format!("{a:.0}"))
                .collect::<Vec<_>>()
                .join(", "),
            if arm.grip.is_some() { "closed" } else { "open" },
        );
    }

    // Let the payload settle after being let go.
    for _ in 0..150 {
        world.step(DT);
    }

    let landed = world.body(payload).unwrap().translation();
    println!();
    println!(
        "Worst reach error over the programme: {:.2} mm",
        worst_error * 1000.0
    );
    println!(
        "Worst segment stretch:                {:.5} m",
        worst_stretch
    );
    println!("Elbow travelled:                      {elbow_min:.0}° to {elbow_max:.0}°");
    println!(
        "Payload came to rest at:              ({:.2}, {:.2}, {:.2})",
        landed.x, landed.y, landed.z
    );

    // ---- the checks ---------------------------------------------------------
    //
    // Printing all this is only worth anything if something objects when it is
    // wrong, so every line above has an assertion behind it.

    assert!(
        worst_error < 0.01,
        "the tool missed a waypoint by {:.1} mm",
        worst_error * 1000.0
    );
    assert!(
        worst_stretch < 1e-3,
        "the arm stretched by {worst_stretch} m — the solver is not holding link lengths"
    );
    assert!(
        // Half a degree of slack: the angle is measured from joint positions.
        elbow_min > ELBOW.0.to_degrees() - 0.5 && elbow_max < ELBOW.1.to_degrees() + 0.5,
        "the elbow reached {elbow_min:.0}°..{elbow_max:.0}°, outside its hinge limits"
    );
    let in_bin = (landed.x - bin.x).abs() < 0.2 && (landed.z - bin.z).abs() < 0.2;
    assert!(
        in_bin,
        "the payload ended at {landed:?}, not in the bin at {bin:?}"
    );
    assert!(
        landed.y > 0.0 && landed.y < 0.2,
        "the payload is at y = {:.2}, so it is either buried or perched on a wall",
        landed.y
    );
    assert_eq!(home_poses.len(), 2);
    let drift: f32 = home_poses[0]
        .iter()
        .zip(&home_poses[1])
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max);
    assert!(
        drift < 1.0,
        "returning home gave a different pose the second time, off by {drift:.1}°"
    );
    println!("Repeatability returning home:         {drift:.2}°");

    // ---- and the failure the workspace check exists for ---------------------
    let far = Vector3::new(4.0, 3.0, 0.0);
    let refused = arm.aim(far);
    assert!(
        refused.out_of_reach && !refused.reached,
        "a target 5 m away was not reported as out of reach"
    );
    println!(
        "Target at 5 m correctly refused:      out_of_reach={}, short by {:.2} m",
        refused.out_of_reach, refused.error
    );
}
