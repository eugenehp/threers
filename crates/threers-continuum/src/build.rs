//! Turning a [`Rod`] into bodies, elastic stations and cables in a `World`.

use crate::rod::Rod;
use threers_physics::prelude::*;
use threers_physics::collider::Collider;

/// The collision group rod links are put in when they may not touch each other.
///
/// Membership in this group and a filter that excludes it: links ignore each
/// other and collide with everything else, which is what
/// [`Rod::self_collide`] being false means.
pub const ROD_GROUP: u32 = 1 << 31;

/// How a cable runs down a rod.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TendonRoute {
    /// Distance from the backbone. Should be under the rod's radius or the
    /// cable runs outside it.
    pub offset: f32,
    /// Where around the rod, in radians, counter-clockwise looking along the
    /// rod's own axis from the root.
    pub phase: f32,
    /// Which segment it terminates at, 1-based from the root. It is anchored
    /// there and passes through every station below, which is what makes a
    /// multi-segment robot's lower cables run through its upper ones. Zero runs
    /// the whole rod.
    pub segment: usize,
    /// Constant pull, before anything drives it.
    pub pretension: f32,
}

impl TendonRoute {
    /// A cable `offset` from the backbone at `phase` radians, down the whole
    /// rod.
    pub fn new(offset: f32, phase: f32) -> Self {
        Self {
            offset,
            phase,
            segment: 0,
            pretension: 0.0,
        }
    }

    /// `n` cables evenly spaced around the rod, the first at `phase`.
    ///
    /// Three is the fewest that can bend a segment in any direction, which is
    /// why three is the usual number.
    pub fn ring(n: usize, offset: f32, phase: f32) -> Vec<Self> {
        (0..n.max(1))
            .map(|i| {
                Self::new(
                    offset,
                    phase + std::f32::consts::TAU * i as f32 / n.max(1) as f32,
                )
            })
            .collect()
    }

    pub fn terminating_at(mut self, segment: usize) -> Self {
        self.segment = segment;
        self
    }

    pub fn pretensioned(mut self, tension: f32) -> Self {
        self.pretension = tension;
        self
    }
}

/// A built rod: the bodies, the stations between them, and the cables on it.
///
/// The rod runs along each body's **local +Y**, which is how the physics
/// crate's cylinders and capsules are aligned. The whole chain is turned by the
/// base rotation, so a rod told to grow along world `+Z` has every link rotated
/// to match and every hinge axis rotated with it — there is no separate axis
/// convention to keep in step.
#[derive(Debug, Clone)]
pub struct Continuum {
    pub rod: Rod,
    /// `links + 1` bodies, root first. The first and last are half-length.
    pub bodies: Vec<BodyId>,
    /// `links` elastic stations, root first. Station `i` joins body `i` to
    /// body `i + 1`.
    pub stations: Vec<JointId>,
    /// The cables, in the order they were added.
    pub tendons: Vec<TendonId>,
    /// The route each cable took, parallel to [`tendons`](Self::tendons).
    pub routes: Vec<TendonRoute>,
    /// Length of each cable as built, which is what a pull is measured from.
    pub rest_lengths: Vec<f32>,
    /// The weld to whatever the rod grows out of, if anything.
    pub mount: Option<JointId>,
}

impl Continuum {
    /// Build a rod into `world`, growing from `origin` along `axis`.
    ///
    /// With `base` given, the rod's root is welded to that body and the rod
    /// moves with it. Without, the root link is fixed in the world.
    ///
    /// ```
    /// use threers_continuum::prelude::*;
    /// use threers_physics::prelude::*;
    ///
    /// let mut world = World::new();
    /// let rod = Rod::new(0.3, 20).radius(0.006).core(0.0005)
    ///     .material(200.0e9, 0.3, 1200.0);
    /// let arm = Continuum::build(&mut world, rod, None, Vector3::ZERO, Vector3::UP);
    ///
    /// assert_eq!(arm.bodies.len(), 21);
    /// assert_eq!(arm.stations.len(), 20);
    /// assert!((arm.tip(&world).y - 0.3).abs() < 1e-3, "it starts straight");
    /// ```
    pub fn build(
        world: &mut World,
        rod: Rod,
        base: Option<BodyId>,
        origin: Vector3,
        axis: Vector3,
    ) -> Self {
        let rod = Rod {
            links: rod.links.max(1),
            segments: rod.segments.max(1).min(rod.links.max(1)),
            ..rod
        };
        let props = rod.link_properties();
        let n = rod.links;
        let up = normalize_or_up(axis);
        // Local +Y is the rod; turn the whole chain so that lands on `axis`.
        let rotation = rotation_from_up(up);

        let groups = if rod.self_collide {
            InteractionGroups::ALL
        } else {
            InteractionGroups::new(ROD_GROUP, !ROD_GROUP)
        };

        let mut bodies = Vec::with_capacity(n + 1);
        for i in 0..=n {
            let half = i == 0 || i == n;
            let (len, mass) = if half {
                (props.length * 0.5, props.mass * 0.5)
            } else {
                (props.length, props.mass)
            };
            // Centre of body i along the rod. The end half-links sit a quarter
            // of a link in from each end; everything between is on the whole
            // links.
            let s = if i == 0 {
                props.length * 0.25
            } else if i == n {
                rod.length - props.length * 0.25
            } else {
                i as f32 * props.length
            };
            let fixed = base.is_none() && i == 0;
            let builder = if fixed {
                RigidBody::fixed()
            } else {
                RigidBody::dynamic()
            };
            let mut collider = Collider::new(Shape::cylinder(len * 0.5, rod.radius));
            collider.groups = groups;
            bodies.push(
                world.add_body(
                    builder
                        .collider(collider)
                        .mass(mass.max(f32::MIN_POSITIVE))
                        .translation(origin + up * s)
                        .rotation(rotation),
                ),
            );
        }

        // The links carry local +Y along the rod, so a generic joint's own axes
        // — which are body A's, the frames being identity — come out as: 0 and
        // 2 across the rod, 1 along it.
        let along = Vector3::new(0.0, 1.0, 0.0);
        let half_at = |i: usize| {
            if i == 0 || i == n {
                props.length * 0.25
            } else {
                props.length * 0.5
            }
        };

        let mut stations = Vec::with_capacity(n);
        for i in 0..n {
            // Station i sits between body i and body i + 1, at the far face of
            // one and the near face of the other.
            let anchor_a = along * half_at(i);
            let anchor_b = along * -half_at(i + 1);

            // Two hinges in series would need a massless body between them, so
            // a station is one generic joint with its freedoms set directly:
            // translations locked, the two cross-axes sprung at `n·EI/L`, and
            // the rod's own axis either sprung at `n·GJ/L` or locked.
            let bend = bend_dof(&rod, props.bend_stiffness, props.bend_damping);
            let twist = if rod.twist {
                bend_dof(&rod, props.torsion_stiffness, props.torsion_damping)
            } else {
                Dof::LOCKED
            };
            let joint =
                Joint::generic(world.bodies(), bodies[i], bodies[i + 1], anchor_a, anchor_b)
                    .expect("both bodies were just inserted")
                    .with_angular_dof(0, bend)
                    .with_angular_dof(1, twist)
                    .with_angular_dof(2, bend);
            stations.push(world.add_joint(joint));
        }

        let mount = base.map(|b| {
            let root = world.body(bodies[0]).map(|x| x.translation()).unwrap_or(origin);
            let anchor_root = root - up * (props.length * 0.25);
            let joint = Joint::fixed_at_point(world.bodies(), b, bodies[0], anchor_root)
                .expect("the base and the root link both exist");
            world.add_joint(joint)
        });

        Self {
            rod,
            bodies,
            stations,
            tendons: Vec::new(),
            routes: Vec::new(),
            rest_lengths: Vec::new(),
            mount,
        }
    }

    /// Route a cable down the rod and add it to the world.
    ///
    /// One guide per link, on the ring of radius `offset` at angle `phase`, from
    /// the root to the tip of the route's segment. The cable is anchored on the
    /// body it ends at, which is what makes pulling it bend everything below.
    pub fn add_tendon(&mut self, world: &mut World, route: TendonRoute) -> TendonId {
        let last = if route.segment == 0 {
            self.rod.links
        } else {
            self.rod.segment_tip_link(route.segment)
        };
        let props = self.rod.link_properties();
        let n = self.rod.links;
        let radial = Vector3::new(route.phase.cos(), 0.0, -route.phase.sin()) * route.offset;
        let half_at = |i: usize| {
            if i == 0 || i == n {
                props.length * 0.25
            } else {
                props.length * 0.5
            }
        };

        // One guide at the *middle* of each link, plus an anchor at each end.
        //
        // The middle is not an aesthetic choice. Put the guides on the station
        // planes instead — one on each side, where a real eyelet would sit —
        // and the two coincide exactly while the rod is straight, so every
        // segment of the path is either zero-length or rigidly inside one link.
        // The routed length then has a *zero* first derivative with respect to
        // bending: the cable can be pulled infinitely hard and the rod will not
        // move, because to first order pulling it does not shorten anything.
        // A straight rod is exactly the state everything starts in.
        //
        // Guides at the middles put a station between every consecutive pair,
        // so bending station `j` by `θ` changes that segment by `offset·θ` —
        // first order, which is the moment arm the cable is supposed to have.
        let mut points = Vec::with_capacity(last + 3);
        points.push(TendonPoint::new(
            self.bodies[0],
            radial - Vector3::new(0.0, half_at(0), 0.0),
        ));
        for i in 0..=last {
            points.push(TendonPoint::new(self.bodies[i], radial));
        }
        points.push(TendonPoint::new(
            self.bodies[last],
            radial + Vector3::new(0.0, half_at(last), 0.0),
        ));

        let mut tendon = Tendon::new(points);
        let rest = tendon.length(world.bodies());
        tendon.kind = if route.pretension > 0.0 {
            TendonKind::Force {
                tension: route.pretension,
            }
        } else {
            TendonKind::rope(rest)
        };
        let id = world.add_tendon(tendon);
        self.tendons.push(id);
        self.routes.push(route);
        self.rest_lengths.push(rest);
        id
    }

    /// Route a ring of `n` cables, evenly spaced, all terminating at `segment`.
    pub fn add_tendon_ring(
        &mut self,
        world: &mut World,
        n: usize,
        offset: f32,
        phase: f32,
        segment: usize,
        pretension: f32,
    ) -> Vec<TendonId> {
        TendonRoute::ring(n, offset, phase)
            .into_iter()
            .map(|r| {
                self.add_tendon(
                    world,
                    r.terminating_at(segment).pretensioned(pretension),
                )
            })
            .collect()
    }

    /// Reel cable `index` in by `pull` from the length it was built at.
    ///
    /// Positive shortens. This is the command a real robot's controller sends —
    /// one number per cable — and it is applied as a winch rather than a force,
    /// so it holds where it is told within `max_force`.
    pub fn set_pull(&self, world: &mut World, index: usize, pull: f32, max_force: f32) {
        let (Some(&id), Some(&rest)) = (self.tendons.get(index), self.rest_lengths.get(index))
        else {
            return;
        };
        if let Some(tendon) = world.tendon_mut(id) {
            tendon.kind = TendonKind::winch((rest - pull).max(0.0), max_force);
        }
    }

    /// Reel every cable at once, in the order they were added.
    pub fn set_pulls(&self, world: &mut World, pulls: &[f32], max_force: f32) {
        for (i, pull) in pulls.iter().enumerate() {
            self.set_pull(world, i, *pull, max_force);
        }
    }

    /// How much each cable has been reeled in from its built length.
    pub fn pulls(&self, world: &World) -> Vec<f32> {
        self.tendons
            .iter()
            .zip(&self.rest_lengths)
            .map(|(id, rest)| rest - world.tendon_length(*id).unwrap_or(*rest))
            .collect()
    }

    /// Where the rod's tip is.
    pub fn tip(&self, world: &World) -> Vector3 {
        let props = self.rod.link_properties();
        let last = *self.bodies.last().expect("a rod has at least two bodies");
        match world.body(last) {
            Some(body) => body
                .position
                .transform_point(Vector3::new(0.0, props.length * 0.25, 0.0)),
            None => Vector3::ZERO,
        }
    }

    /// The backbone as a polyline: the root, then the far end of every link.
    ///
    /// `links + 2` points, which is what a renderer wants and what an
    /// arc-length comparison against a reference solution measures.
    pub fn backbone(&self, world: &World) -> Vec<Vector3> {
        let props = self.rod.link_properties();
        let n = self.rod.links;
        let mut out = Vec::with_capacity(n + 2);
        for (i, id) in self.bodies.iter().enumerate() {
            let Some(body) = world.body(*id) else {
                continue;
            };
            let half = if i == 0 || i == n {
                props.length * 0.25
            } else {
                props.length * 0.5
            };
            if i == 0 {
                out.push(
                    body.position
                        .transform_point(Vector3::new(0.0, -half, 0.0)),
                );
            }
            out.push(body.position.transform_point(Vector3::new(0.0, half, 0.0)));
        }
        out
    }

    /// Normalised arc position of each point [`backbone`](Self::backbone)
    /// returns, root first.
    ///
    /// Needed to compare a shape against anything measured somewhere else. Two
    /// solutions of the same rod sampled at different stations are not
    /// comparable point by point — pairing them by index compares places up to
    /// a tenth of the rod apart and puts a floor under the error that no amount
    /// of accuracy removes. Pair them by *arc*.
    pub fn backbone_arc(&self) -> Vec<f32> {
        let n = self.rod.links;
        let mut out = Vec::with_capacity(n + 2);
        out.push(0.0);
        for k in 1..=n {
            out.push((k as f32 - 0.5) / n as f32);
        }
        out.push(1.0);
        out
    }

    /// The body carrying a given arc position, and where on it, in that body's
    /// own frame.
    ///
    /// `arc` runs 0 at the root to 1 at the tip and is clamped to that range.
    pub fn point_at_arc(&self, arc: f32) -> (BodyId, Vector3) {
        let n = self.rod.links;
        let link_length = self.rod.length / n as f32;
        let x = arc.clamp(0.0, 1.0) * self.rod.length;
        let (index, centre) = if x <= link_length * 0.5 {
            (0, link_length * 0.25)
        } else if x >= self.rod.length - link_length * 0.5 {
            (n, self.rod.length - link_length * 0.25)
        } else {
            let i = (x / link_length).round().clamp(1.0, n as f32 - 1.0) as usize;
            (i, i as f32 * link_length)
        };
        (
            self.bodies[index],
            Vector3::new(0.0, x - centre, 0.0),
        )
    }

    /// Where a given arc position currently is, in the world.
    pub fn world_point_at_arc(&self, world: &World, arc: f32) -> Vector3 {
        let (body, local) = self.point_at_arc(arc);
        match world.body(body) {
            Some(b) => b.position.transform_point(local),
            None => Vector3::ZERO,
        }
    }

    /// Push and twist the rod at one station along its length.
    ///
    /// The force acts at that point rather than at the carrying link's centre
    /// of mass, so the moment arm is the one the load really has — half a link
    /// of arm is a large fraction of the moment on a finely chopped rod, and
    /// getting it wrong quietly bends the answer.
    ///
    /// Forces are cleared at the end of every step, so a sustained load has to
    /// be re-applied each step, like any other.
    pub fn apply_wrench_at_arc(
        &self,
        world: &mut World,
        arc: f32,
        force: Vector3,
        moment: Vector3,
    ) {
        let (id, local) = self.point_at_arc(arc);
        let Some(body) = world.body_mut(id) else {
            return;
        };
        let point = body.position.transform_point(local);
        body.add_force_at_point(force, point);
        body.add_torque(moment);
    }

    /// Where the tip of `segment` is — the point a cable terminating there
    /// pulls on, and what a multi-point controller steers.
    pub fn segment_tip(&self, world: &World, segment: usize) -> Vector3 {
        let link = self.rod.segment_tip_link(segment);
        let props = self.rod.link_properties();
        let half = if link == 0 || link == self.rod.links {
            props.length * 0.25
        } else {
            props.length * 0.5
        };
        match self.bodies.get(link).and_then(|id| world.body(*id)) {
            Some(body) => body.position.transform_point(Vector3::new(0.0, half, 0.0)),
            None => Vector3::ZERO,
        }
    }

    /// Arc length actually traced by the backbone. A rod that stretches is a
    /// rod whose stations are being pulled apart, so this drifting from
    /// [`Rod::length`] is a sign the tension is too much for the solver.
    pub fn arc_length(&self, world: &World) -> f32 {
        self.backbone(world)
            .windows(2)
            .map(|w| (w[1] - w[0]).length())
            .sum()
    }
}

fn bend_dof(rod: &Rod, stiffness: f32, damping: f32) -> Dof {
    let dof = match rod.station_limit {
        Some([min, max]) => Dof::limited(min, max),
        None => Dof::FREE,
    };
    dof.sprung(stiffness, 0.0, damping)
}

fn normalize_or_up(v: Vector3) -> Vector3 {
    let len = v.length();
    if len > 1e-9 {
        v * (1.0 / len)
    } else {
        Vector3::UP
    }
}

/// The rotation taking local +Y onto `up`.
fn rotation_from_up(up: Vector3) -> Quaternion {
    let y = Vector3::UP;
    let dot = y.dot(up).clamp(-1.0, 1.0);
    if dot > 1.0 - 1e-6 {
        return Quaternion::identity();
    }
    if dot < -1.0 + 1e-6 {
        // Antiparallel: any axis across Y will do, and X is across Y.
        return Quaternion::from_axis_angle(Vector3::new(1.0, 0.0, 0.0), std::f32::consts::PI);
    }
    let axis = y.cross(up);
    Quaternion::from_axis_angle(axis * (1.0 / axis.length()), dot.acos())
}
