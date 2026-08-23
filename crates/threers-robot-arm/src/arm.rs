//! A four-axis arm, and two ways of driving it.
//!
//! # Kinematic, and servo-driven
//!
//! [`KinematicArm`] puts its links exactly where the solver says. That is the
//! honest model of a machine whose controller commands joint angles and whose
//! servos hold them: a heavier payload makes such an arm *slower*, never lower.
//! It is what you want when the arm is scenery — reliable, exact, and immune to
//! everything it touches.
//!
//! [`ServoArm`] is the same geometry built out of dynamic bodies, held together
//! by joints that are *soft* rather than rigid. Its links have mass, gravity
//! pulls on them, and each joint holds its commanded angle only as hard as its
//! stiffness allows. So it sags under a load, lags behind a fast command, and
//! can be pushed out of the way. That is what you want when the arm is part of
//! the game rather than part of the set.
//!
//! Both take the same programme and the same solver output, which is the point:
//! the difference on screen is the difference between the two models, not
//! between two demos.

use threers_physics::prelude::*;

/// Shoulder height above the base plate.
pub const COLUMN: f32 = 0.55;
/// Upper arm, forearm, and the tool sticking out of the wrist.
pub const LINKS: [f32; 3] = [0.80, 0.70, 0.22];
/// How far the elbow may bend, in radians. It never straightens fully and it
/// never folds back on itself.
pub const ELBOW: (f32, f32) = (0.25, 2.4);

/// Radius of link `i`.
pub fn link_radius(i: usize) -> f32 {
    if i == 2 {
        0.045
    } else {
        0.07
    }
}

/// Where the chain sits when the controller says "home" — a configuration, not
/// a point. See the note on repeatability in [`crate::Recording`].
pub fn ready_pose(base: Vector3) -> Vec<Vector3> {
    // Bent, not straight. A straight arm has a zero-degree elbow, which is
    // outside the elbow's own limit, so it would begin by climbing out of a pose
    // it would never have chosen. Real arms park bent for the same reason.
    let mut point = base + Vector3::new(0.0, COLUMN, 0.0);
    let mut points = vec![point];
    for (length, dir) in LINKS.iter().zip([
        Vector3::new(1.0, 1.0, 0.0),
        Vector3::new(1.0, -1.0, 0.0),
        Vector3::new(0.4, -1.0, 0.0),
    ]) {
        point = point + dir.normalize() * *length;
        points.push(point);
    }
    points
}

/// The solver, and the pose it has arrived at.
///
/// Shared by both drivers: the kinematics are the same machine either way, and
/// only what is done with the answer differs.
pub struct Chain {
    pub chain: IkChain,
    pub base: Vector3,
}

impl Chain {
    pub fn new(base: Vector3) -> Self {
        let mut chain = IkChain::from_points(&ready_pose(base));
        chain.max_iterations = 40;
        chain.tolerance = 0.002;
        // A constraint at joint `i` limits segment `i` against segment `i - 1`,
        // so index 1 is the elbow and index 2 is the wrist. Getting these the
        // wrong way round is not a subtle bug: with a cone on the elbow and a
        // wide hinge on the wrist, the solver folds the tool link back on itself
        // and reaches its target by scything the tool sideways across the table,
        // knocking the part it came to collect out of the cell.
        chain.set_constraint(
            1,
            IkConstraint::hinge_limited(Vector3::new(0.0, 0.0, 1.0), ELBOW.0, ELBOW.1),
        );
        // The wrist stays roughly in line with the forearm. A real cell commands
        // tool *orientation* as well as position; this chain solves for position
        // only, so keeping the last link from flapping stands in for it.
        chain.set_constraint(2, IkConstraint::cone(0.55));
        Self { chain, base }
    }

    /// Point the tool at `target`, and report what the solver managed.
    pub fn aim(&mut self, target: Vector3) -> IkResult {
        // The elbow axle turns with the base. `IkConstraint::Hinge` takes a
        // *world* axis, so a fixed one is only right while the arm stays in the
        // plane it was set up in — swing the base ninety degrees and the hinge
        // is suddenly asking the elbow to bend sideways, which it cannot, and
        // the solve stalls a hundred millimetres short with nothing to say about
        // why.
        //
        // For an articulated arm the answer is simple: the shoulder and elbow
        // work in the vertical plane through the target, so the elbow's axle is
        // horizontal and across that plane.
        let shoulder = self.chain.joints[0].position;
        let flat = Vector3::new(target.x - shoulder.x, 0.0, target.z - shoulder.z);
        if flat.length() > 1e-4 {
            let axis = Vector3::new(0.0, 1.0, 0.0).cross(flat.normalize());
            self.chain
                .set_constraint(1, IkConstraint::hinge_limited(axis, ELBOW.0, ELBOW.1));
        }
        self.chain.solve_ccd(target)
    }

    pub fn tip(&self) -> Vector3 {
        self.chain.tip()
    }

    pub fn pose(&self) -> Vec<Vector3> {
        self.chain.joints.iter().map(|j| j.position).collect()
    }

    pub fn set_pose(&mut self, pose: &[Vector3]) {
        for (joint, p) in self.chain.joints.iter_mut().zip(pose) {
            joint.position = *p;
        }
    }

    /// World transform of link `i`: a capsule spanning two joints.
    pub fn link_transform(&self, i: usize) -> Isometry {
        let a = self.chain.joints[i].position;
        let b = self.chain.joints[i + 1].position;
        Isometry::new(
            (a + b) * 0.5,
            rotation_from_to(Vector3::new(0.0, 1.0, 0.0), b - a),
        )
    }

    /// Joint angles in degrees, the way a controller would report them.
    pub fn angles(&self) -> Vec<f32> {
        let mut out = Vec::new();
        let mut previous = Vector3::new(0.0, 1.0, 0.0); // up the column
        for i in 0..self.chain.joints.len() - 1 {
            let segment = self.chain.joints[i + 1].position - self.chain.joints[i].position;
            let (a, b) = (previous.normalize(), segment.normalize());
            out.push(a.dot(b).clamp(-1.0, 1.0).acos().to_degrees());
            previous = segment;
        }
        out
    }

    /// Longest deviation of any segment from the length it was built with.
    ///
    /// The one number that says whether a solve is trustworthy: both solvers
    /// hold lengths by construction, so anything non-zero here means something
    /// has gone wrong that a reach error would not reveal.
    pub fn stretch(&self) -> f32 {
        let mut worst: f32 = 0.0;
        for (i, expected) in LINKS.iter().enumerate() {
            let actual =
                (self.chain.joints[i + 1].position - self.chain.joints[i].position).length();
            worst = worst.max((actual - expected).abs());
        }
        worst
    }
}

/// Shortest rotation taking `from` onto `to`.
pub fn rotation_from_to(from: Vector3, to: Vector3) -> Quaternion {
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
        return Quaternion::from_axis_angle(axis.normalize(), std::f32::consts::PI);
    }
    Quaternion::from_axis_angle(a.cross(b).normalize(), dot.acos())
}

/// What both drivers have to be able to do, so the programme runs on either.
pub trait ArmDriver {
    /// Command the tool to a point. Returns what the solver managed.
    fn aim(&mut self, world: &mut World, target: Vector3) -> IkResult;
    /// Grab whatever the tool is touching. `false` if nothing is close enough.
    fn close_gripper(&mut self, world: &mut World, payload: BodyId) -> bool;
    fn open_gripper(&mut self, world: &mut World);
    fn is_gripping(&self) -> bool;
    /// The solved chain, for reporting.
    fn chain(&self) -> &Chain;
    /// Bodies to draw, in the order they were added.
    fn bodies(&self) -> &[BodyId];
    /// Where the tool actually is, which for a servo arm is not where it was
    /// told to be.
    fn tool_position(&self, world: &World) -> Vector3;
}

// ---- the kinematic driver -------------------------------------------------

/// Links placed exactly where the solver says, every step.
pub struct KinematicArm {
    pub solver: Chain,
    links: Vec<BodyId>,
    tool: BodyId,
    grip: Option<JointId>,
}

impl KinematicArm {
    pub fn build(world: &mut World, base: Vector3) -> Self {
        world.add_body(
            RigidBody::fixed()
                .shape(Shape::cylinder(0.06, 0.22))
                .translation(base + Vector3::new(0.0, 0.06, 0.0)),
        );

        let solver = Chain::new(base);
        let mut links = Vec::new();
        for (i, length) in LINKS.iter().enumerate() {
            let radius = link_radius(i);
            links.push(
                world.add_body(
                    RigidBody::kinematic()
                        .shape(Shape::capsule((length * 0.5 - radius).max(0.01), radius))
                        .friction(0.9),
                ),
            );
        }
        let tool = world.add_body(RigidBody::kinematic().shape(Shape::ball(0.05)).friction(1.0));

        let mut arm = Self {
            solver,
            links,
            tool,
            grip: None,
        };
        arm.snap(world);
        arm
    }

    /// Teleport into place. For the initial pose only — used mid-run it would
    /// move the arm with no velocity, so it would pass through things.
    pub fn snap(&mut self, world: &mut World) {
        for (i, body) in self.links.iter().enumerate() {
            let iso = self.solver.link_transform(i);
            if let Some(link) = world.body_mut(*body) {
                link.set_translation(iso.translation);
                link.set_rotation(iso.rotation);
            }
        }
        let tip = self.solver.tip();
        if let Some(tool) = world.body_mut(self.tool) {
            tool.set_translation(tip);
        }
    }

    /// Push the solved pose into the kinematic bodies.
    ///
    /// Setting a *target* rather than a position is what makes the arm push
    /// things: the simulation derives the velocity the body must have travelled
    /// at, and that velocity is what a contact sees.
    fn drive(&mut self, world: &mut World) {
        for (i, body) in self.links.iter().enumerate() {
            let iso = self.solver.link_transform(i);
            if let Some(link) = world.body_mut(*body) {
                link.set_kinematic_target(iso);
            }
        }
        let tip = self.solver.tip();
        if let Some(tool) = world.body_mut(self.tool) {
            tool.set_kinematic_target(Isometry::new(tip, Quaternion::identity()));
        }
    }
}

impl ArmDriver for KinematicArm {
    fn aim(&mut self, world: &mut World, target: Vector3) -> IkResult {
        let result = self.solver.aim(target);
        self.drive(world);
        result
    }

    fn close_gripper(&mut self, world: &mut World, payload: BodyId) -> bool {
        close_gripper(&mut self.grip, world, self.tool, payload, self.solver.tip())
    }

    fn open_gripper(&mut self, world: &mut World) {
        if let Some(id) = self.grip.take() {
            world.remove_joint(id);
        }
    }

    fn is_gripping(&self) -> bool {
        self.grip.is_some()
    }

    fn chain(&self) -> &Chain {
        &self.solver
    }

    fn bodies(&self) -> &[BodyId] {
        &self.links
    }

    fn tool_position(&self, world: &World) -> Vector3 {
        world
            .body(self.tool)
            .map(|b| b.translation())
            .unwrap_or(self.solver.tip())
    }
}

// ---- the servo driver -----------------------------------------------------

/// Links with mass, held by joints that give.
///
/// Each link is a dynamic body welded to the one before it, and the weld's rest
/// orientation is rewritten every step to whatever the solver asked for. That is
/// a servo: it holds the commanded angle as hard as its stiffness lets it, and
/// no harder. Gravity, the payload and the arm's own inertia all get a say,
/// which is the entire difference from [`KinematicArm`].
pub struct ServoArm {
    pub solver: Chain,
    links: Vec<BodyId>,
    /// One weld per link, the first anchoring the arm to its base.
    servos: Vec<JointId>,
    tool: BodyId,
    grip: Option<JointId>,
}

impl ServoArm {
    /// `stiffness` is the servo frequency in hertz — see [`Softness`]. Around 8
    /// gives a stiff industrial arm; 3 gives something visibly springy.
    pub fn build(world: &mut World, base: Vector3, stiffness: f32) -> Self {
        let column = world.add_body(
            RigidBody::fixed()
                .shape(Shape::cylinder(0.06, 0.22))
                .translation(base + Vector3::new(0.0, 0.06, 0.0)),
        );

        let solver = Chain::new(base);
        let mut links = Vec::new();
        let mut servos = Vec::new();
        let mut previous = column;
        for (i, length) in LINKS.iter().enumerate() {
            let radius = link_radius(i);
            let iso = solver.link_transform(i);
            let link = world.add_body(
                RigidBody::dynamic()
                    .shape(Shape::capsule((length * 0.5 - radius).max(0.01), radius))
                    // Light links: a real arm's links are mostly air and the
                    // motors are at the joints. Heavy ones would need a servo
                    // stiff enough to be indistinguishable from kinematic.
                    .density(220.0)
                    .position(iso)
                    .friction(0.9)
                    .can_sleep(false),
            );
            // The weld sits at the joint the two links share.
            let anchor = solver.chain.joints[i].position;
            let joint = Joint::fixed_at_point(world.bodies(), previous, link, anchor)
                .expect("both bodies exist")
                .soft(stiffness, 1.0);
            servos.push(world.add_joint(joint));
            links.push(link);
            previous = link;
        }

        let tip = solver.tip();
        let tool = world.add_body(
            RigidBody::dynamic()
                .shape(Shape::ball(0.05))
                .density(400.0)
                .translation(tip)
                .friction(1.0)
                .can_sleep(false),
        );
        world.add_joint(
            Joint::fixed_at_point(world.bodies(), previous, tool, tip).expect("both bodies exist"),
        );

        Self {
            solver,
            links,
            servos,
            tool,
            grip: None,
        }
    }

    /// Rewrite every servo's commanded angle from the solved pose.
    fn drive(&mut self, world: &mut World) {
        for i in 0..self.links.len() {
            let commanded = self.solver.link_transform(i).rotation;
            // The weld holds `b` at a fixed orientation relative to `a`, so the
            // command is expressed relative to whatever the parent link is
            // *actually* doing — not where it was told to be. That is what makes
            // the error accumulate down the arm the way a real one's does.
            let parent = if i == 0 {
                Quaternion::identity()
            } else {
                world
                    .body(self.links[i - 1])
                    .map(|b| b.rotation())
                    .unwrap_or(Quaternion::identity())
            };
            let relative = parent.conjugate().multiply(commanded);
            if let Some(joint) = world.joint_mut(self.servos[i]) {
                if let JointKind::Fixed { rest_rotation } = &mut joint.kind {
                    *rest_rotation = relative;
                }
            }
        }
    }
}

impl ArmDriver for ServoArm {
    fn aim(&mut self, world: &mut World, target: Vector3) -> IkResult {
        let result = self.solver.aim(target);
        self.drive(world);
        result
    }

    fn close_gripper(&mut self, world: &mut World, payload: BodyId) -> bool {
        let tip = self.tool_position(world);
        close_gripper(&mut self.grip, world, self.tool, payload, tip)
    }

    fn open_gripper(&mut self, world: &mut World) {
        if let Some(id) = self.grip.take() {
            world.remove_joint(id);
        }
    }

    fn is_gripping(&self) -> bool {
        self.grip.is_some()
    }

    fn chain(&self) -> &Chain {
        &self.solver
    }

    fn bodies(&self) -> &[BodyId] {
        &self.links
    }

    fn tool_position(&self, world: &World) -> Vector3 {
        world
            .body(self.tool)
            .map(|b| b.translation())
            .unwrap_or(self.solver.tip())
    }
}

/// Grab whatever the tool is touching, if anything is close enough.
///
/// A fixed joint made at the moment of contact. That is what a vacuum cup or an
/// electromagnet is: it does not squeeze, it holds whatever it met until told
/// otherwise — and the payload's mass and inertia carry on mattering, so a heavy
/// box still swings the tip on the way round.
fn close_gripper(
    grip: &mut Option<JointId>,
    world: &mut World,
    tool: BodyId,
    payload: BodyId,
    tip: Vector3,
) -> bool {
    if grip.is_some() {
        return true;
    }
    let Some(target) = world.body(payload) else {
        return false;
    };
    if (target.translation() - tip).length() > 0.2 {
        return false;
    }
    let joint = Joint::fixed_at_point(world.bodies(), tool, payload, tip);
    *grip = joint.map(|j| world.add_joint(j));
    grip.is_some()
}
