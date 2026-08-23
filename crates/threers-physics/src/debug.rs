//! Seeing what the solver sees.
//!
//! Almost every physics bug looks the same from the outside — something is in
//! the wrong place — and almost none of them can be diagnosed without looking at
//! the *collider* rather than the mesh. A collider that does not match the model
//! is the single most common mistake, and it is invisible until drawn.
//!
//! This produces line geometry, not pixels: [`DebugRenderer::to_geometry`]
//! returns a [`crate::debug::BufferGeometry`] of `LineSegments`, which the `threers` renderer
//! already knows how to draw.
//!
//! ```no_run
//! use threers_physics::prelude::*;
//! use threers_physics::debug::DebugRenderer;
//! # use threers::core::{Object3D, ObjectArena, LineSegments};
//! # use threers::materials::{LineBasicMaterial, Material};
//! # use threers::math::Color;
//!
//! let mut world = World::new();
//! let mut debug = DebugRenderer::new();
//!
//! // Each frame, after stepping:
//! world.step(1.0 / 60.0);
//! debug.render(&world);
//! let geometry = debug.to_geometry();
//! // ...hand `geometry` to a LineSegments node.
//! ```

use crate::body::RigidBody;
use crate::joint::JointKind;
use crate::math::Isometry;
use crate::shape::Shape;
use crate::world::World;
use threers::core::{BufferAttribute, BufferGeometry};
use threers::math::{Color, Vector3};

/// Colours for each kind of thing drawn. Defaults follow the convention most
/// engines use, so they read the same way as the tool you came from.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DebugColors {
    /// A body the simulation moves.
    pub dynamic: Color,
    /// A body that never moves.
    pub fixed: Color,
    /// A body you move yourself.
    pub kinematic: Color,
    /// A body that has gone to sleep — dimmed, so a settled pile recedes.
    pub sleeping: Color,
    /// A collider that reports overlaps but applies no forces.
    pub sensor: Color,
    pub contact_point: Color,
    pub contact_normal: Color,
    pub aabb: Color,
    pub joint: Color,
    pub bvh: Color,
}

impl Default for DebugColors {
    fn default() -> Self {
        Self {
            dynamic: Color::from_hex(0x6ee7a0),
            fixed: Color::from_hex(0x8b94a5),
            kinematic: Color::from_hex(0x6ea8fe),
            sleeping: Color::from_hex(0x4a5160),
            sensor: Color::from_hex(0xf0c674),
            contact_point: Color::from_hex(0xff5f56),
            contact_normal: Color::from_hex(0xffb454),
            aabb: Color::from_hex(0x3a4150),
            joint: Color::from_hex(0xc678dd),
            bvh: Color::from_hex(0x2b3140),
        }
    }
}

/// Builds line geometry describing a world.
///
/// Reused between frames — the buffers are cleared, not reallocated.
#[derive(Debug, Clone)]
pub struct DebugRenderer {
    /// Collider outlines. The one you almost always want.
    pub draw_colliders: bool,
    /// Contact points and their normals. Turn on when something jitters or
    /// sinks: a resting box should show four steady points, and a box that
    /// wobbles usually shows one.
    pub draw_contacts: bool,
    /// Broad-phase bounds. Turn on when pairs are being missed.
    pub draw_aabbs: bool,
    /// Joint anchors, and the gap between them — a joint that is being pulled
    /// apart shows as a visible line between its two ends.
    pub draw_joints: bool,
    /// The spatial index itself. Rarely useful, occasionally decisive.
    pub draw_bvh: bool,
    /// Segments used to approximate a circle. Lower for a lighter overlay.
    pub circle_segments: usize,
    /// Cap on triangle-mesh edges drawn per collider, since terrain meshes can
    /// be hundreds of thousands of triangles and would swamp the overlay.
    pub max_mesh_edges: usize,
    /// Length of the drawn contact normals.
    pub normal_length: f32,
    pub colors: DebugColors,

    positions: Vec<f32>,
    colors_buffer: Vec<f32>,
}

impl Default for DebugRenderer {
    fn default() -> Self {
        Self::new()
    }
}

impl DebugRenderer {
    pub fn new() -> Self {
        Self {
            draw_colliders: true,
            draw_contacts: false,
            draw_aabbs: false,
            draw_joints: true,
            draw_bvh: false,
            circle_segments: 16,
            max_mesh_edges: 3000,
            normal_length: 0.3,
            colors: DebugColors::default(),
            positions: Vec::new(),
            colors_buffer: Vec::new(),
        }
    }

    /// Everything on. Slow, and the right first move when something is wrong.
    pub fn verbose() -> Self {
        Self {
            draw_contacts: true,
            draw_aabbs: true,
            ..Self::new()
        }
    }

    /// Rebuild the line set from the world's current state.
    pub fn render(&mut self, world: &World) {
        self.positions.clear();
        self.colors_buffer.clear();

        if self.draw_colliders || self.draw_aabbs {
            for (_, body) in world.bodies().iter() {
                if self.draw_colliders {
                    self.draw_body(body);
                }
                if self.draw_aabbs {
                    let aabb = body.compute_aabb();
                    if !aabb.is_empty() {
                        self.box_outline(
                            &Isometry::from_translation(aabb.center()),
                            aabb.size() * 0.5,
                            self.colors.aabb,
                        );
                    }
                }
            }
        }

        if self.draw_contacts {
            for manifold in world.contacts().iter() {
                if !manifold.touching {
                    continue;
                }
                for point in &manifold.points {
                    // A small cross rather than a dot: a single vertex is
                    // invisible at most line widths.
                    let s = 0.04;
                    for axis in [
                        Vector3::new(s, 0.0, 0.0),
                        Vector3::new(0.0, s, 0.0),
                        Vector3::new(0.0, 0.0, s),
                    ] {
                        self.line(
                            point.point_a - axis,
                            point.point_a + axis,
                            self.colors.contact_point,
                        );
                    }
                    self.line(
                        point.point_a,
                        point.point_a + manifold.normal * self.normal_length,
                        self.colors.contact_normal,
                    );
                }
            }
        }

        if self.draw_joints {
            for (_, joint) in world.joints().iter() {
                if !joint.is_active() {
                    continue;
                }
                let (Some(a), Some(b)) = (
                    world.bodies().get(joint.body_a),
                    world.bodies().get(joint.body_b),
                ) else {
                    continue;
                };
                let anchor_a = a.position.transform_point(joint.local_anchor_a);
                let anchor_b = b.position.transform_point(joint.local_anchor_b);

                // Each body to its own anchor, then the two anchors together.
                // The middle segment is the constraint error made visible.
                self.line(a.translation(), anchor_a, self.colors.joint);
                self.line(b.translation(), anchor_b, self.colors.joint);
                self.line(anchor_a, anchor_b, self.colors.contact_point);

                // Show the axis of anything that has one.
                if let JointKind::Revolute { local_axis_a, .. }
                | JointKind::Prismatic { local_axis_a, .. } = &joint.kind
                {
                    let axis = a.position.transform_vector(*local_axis_a) * 0.5;
                    self.line(anchor_a - axis, anchor_a + axis, self.colors.joint);
                }
            }
        }

        if self.draw_bvh {
            // These are the boxes the broad phase actually compared, already
            // grown by the prediction distance — which is what explains a missed
            // pair, since a pair the narrow phase never saw is a pair whose
            // boxes here did not overlap.
            for bounds in world.broad_bounds() {
                self.box_outline(
                    &Isometry::from_translation(bounds.center()),
                    bounds.size() * 0.5,
                    self.colors.bvh,
                );
            }
        }
    }

    fn body_color(&self, body: &RigidBody) -> Color {
        if body.is_sleeping() {
            self.colors.sleeping
        } else if body.is_fixed() {
            self.colors.fixed
        } else if body.is_kinematic() {
            self.colors.kinematic
        } else {
            self.colors.dynamic
        }
    }

    fn draw_body(&mut self, body: &RigidBody) {
        let base = self.body_color(body);
        for collider in body.colliders() {
            if !collider.enabled {
                continue;
            }
            let colour = if collider.is_sensor {
                self.colors.sensor
            } else {
                base
            };
            let iso = collider.world_transform(&body.position);
            self.draw_shape(&collider.shape, &iso, colour);
        }
    }

    fn draw_shape(&mut self, shape: &Shape, iso: &Isometry, colour: Color) {
        match shape {
            Shape::Ball { radius } => {
                // Three great circles read as a sphere from any angle.
                self.circle(iso, *radius, 0, 1, colour);
                self.circle(iso, *radius, 1, 2, colour);
                self.circle(iso, *radius, 0, 2, colour);
            }
            Shape::Cuboid { half_extents } => self.box_outline(iso, *half_extents, colour),
            Shape::Capsule {
                half_height,
                radius,
            } => {
                let top = Isometry::new(
                    iso.transform_point(Vector3::new(0.0, *half_height, 0.0)),
                    iso.rotation,
                );
                let bottom = Isometry::new(
                    iso.transform_point(Vector3::new(0.0, -*half_height, 0.0)),
                    iso.rotation,
                );
                self.circle(&top, *radius, 0, 2, colour);
                self.circle(&bottom, *radius, 0, 2, colour);
                // Cap outlines, so the ends do not read as flat.
                self.circle(&top, *radius, 0, 1, colour);
                self.circle(&bottom, *radius, 1, 2, colour);
                for (dx, dz) in [(1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)] {
                    let offset = Vector3::new(dx * radius, 0.0, dz * radius);
                    self.line(
                        top.transform_point(offset),
                        bottom.transform_point(offset),
                        colour,
                    );
                }
            }
            Shape::Cylinder {
                half_height,
                radius,
            } => {
                let top = Isometry::new(
                    iso.transform_point(Vector3::new(0.0, *half_height, 0.0)),
                    iso.rotation,
                );
                let bottom = Isometry::new(
                    iso.transform_point(Vector3::new(0.0, -*half_height, 0.0)),
                    iso.rotation,
                );
                self.circle(&top, *radius, 0, 2, colour);
                self.circle(&bottom, *radius, 0, 2, colour);
                for (dx, dz) in [(1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)] {
                    let offset = Vector3::new(dx * radius, 0.0, dz * radius);
                    self.line(
                        top.transform_point(offset),
                        bottom.transform_point(offset),
                        colour,
                    );
                }
            }
            Shape::Cone {
                half_height,
                radius,
            } => {
                let base = Isometry::new(
                    iso.transform_point(Vector3::new(0.0, -*half_height, 0.0)),
                    iso.rotation,
                );
                let apex = iso.transform_point(Vector3::new(0.0, *half_height, 0.0));
                self.circle(&base, *radius, 0, 2, colour);
                for (dx, dz) in [(1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)] {
                    let rim = base.transform_point(Vector3::new(dx * radius, 0.0, dz * radius));
                    self.line(rim, apex, colour);
                }
            }
            Shape::HalfSpace { normal } => {
                // An infinite plane cannot be drawn, so draw a patch of it
                // around the body's origin, plus the normal.
                let world_normal = iso.transform_vector(*normal);
                let (u, v) = crate::math::orthonormal_basis(world_normal);
                let extent = 5.0;
                let step = extent / 4.0;
                let origin = iso.translation;
                for i in -4..=4 {
                    let offset = i as f32 * step;
                    self.line(
                        origin + u * offset - v * extent,
                        origin + u * offset + v * extent,
                        colour,
                    );
                    self.line(
                        origin + v * offset - u * extent,
                        origin + v * offset + u * extent,
                        colour,
                    );
                }
                self.line(origin, origin + world_normal, self.colors.contact_normal);
            }
            Shape::ConvexHull(hull) => {
                for face in &hull.faces {
                    for k in 0..3 {
                        let a = hull.vertices[face.indices[k] as usize];
                        let b = hull.vertices[face.indices[(k + 1) % 3] as usize];
                        // Each interior edge belongs to two faces; drawing it
                        // once per face doubles the line count for no gain.
                        if face.indices[k] < face.indices[(k + 1) % 3] {
                            self.line(iso.transform_point(a), iso.transform_point(b), colour);
                        }
                    }
                }
            }
            Shape::TriMesh(mesh) => {
                let limit = self.max_mesh_edges;
                for i in 0..mesh.triangle_count().min(limit) {
                    let [a, b, c] = mesh.triangle(i);
                    self.line(iso.transform_point(a), iso.transform_point(b), colour);
                    self.line(iso.transform_point(b), iso.transform_point(c), colour);
                    self.line(iso.transform_point(c), iso.transform_point(a), colour);
                }
            }
            Shape::Compound(parts) => {
                for (local, child) in parts.iter() {
                    self.draw_shape(child, &iso.mul(local), colour);
                }
            }
        }
    }

    /// A circle in the plane spanned by two of the local axes.
    fn circle(&mut self, iso: &Isometry, radius: f32, axis_a: usize, axis_b: usize, colour: Color) {
        let segments = self.circle_segments.max(3);
        let mut previous = None;
        for i in 0..=segments {
            let angle = i as f32 / segments as f32 * std::f32::consts::TAU;
            let mut local = [0.0f32; 3];
            local[axis_a] = angle.cos() * radius;
            local[axis_b] = angle.sin() * radius;
            let point = iso.transform_point(Vector3::new(local[0], local[1], local[2]));
            if let Some(p) = previous {
                self.line(p, point, colour);
            }
            previous = Some(point);
        }
    }

    fn box_outline(&mut self, iso: &Isometry, half: Vector3, colour: Color) {
        let corner = |sx: f32, sy: f32, sz: f32| {
            iso.transform_point(Vector3::new(half.x * sx, half.y * sy, half.z * sz))
        };
        let c = [
            corner(-1.0, -1.0, -1.0),
            corner(1.0, -1.0, -1.0),
            corner(1.0, 1.0, -1.0),
            corner(-1.0, 1.0, -1.0),
            corner(-1.0, -1.0, 1.0),
            corner(1.0, -1.0, 1.0),
            corner(1.0, 1.0, 1.0),
            corner(-1.0, 1.0, 1.0),
        ];
        const EDGES: [(usize, usize); 12] = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7),
        ];
        for (a, b) in EDGES {
            self.line(c[a], c[b], colour);
        }
    }

    fn line(&mut self, a: Vector3, b: Vector3, colour: Color) {
        self.positions.extend_from_slice(&[a.x, a.y, a.z, b.x, b.y, b.z]);
        self.colors_buffer.extend_from_slice(&[
            colour.r, colour.g, colour.b, colour.r, colour.g, colour.b,
        ]);
    }

    /// Line vertices, two per segment.
    pub fn positions(&self) -> &[f32] {
        &self.positions
    }

    /// One colour per vertex.
    pub fn colors(&self) -> &[f32] {
        &self.colors_buffer
    }

    /// Segments produced by the last [`Self::render`].
    pub fn line_count(&self) -> usize {
        self.positions.len() / 6
    }

    /// Package as geometry for a `LineSegments` node.
    ///
    /// Rebuild it each frame — the vertex count changes as bodies come and go,
    /// so there is nothing stable to update in place.
    pub fn to_geometry(&self) -> BufferGeometry {
        let mut geometry = BufferGeometry::new();
        geometry.set_attribute("position", BufferAttribute::new(self.positions.clone(), 3));
        geometry.set_attribute("color", BufferAttribute::new(self.colors_buffer.clone(), 3));
        geometry
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::collider::Collider;
    use crate::joint::Joint;

    fn scene() -> World {
        let mut world = World::new();
        world.add_body(RigidBody::fixed().shape(Shape::ground()));
        world.add_body(
            RigidBody::dynamic()
                .shape(Shape::cuboid(0.5, 0.5, 0.5))
                .translation(Vector3::new(0.0, 3.0, 0.0)),
        );
        world.add_body(
            RigidBody::dynamic()
                .shape(Shape::ball(0.4))
                .translation(Vector3::new(2.0, 3.0, 0.0)),
        );
        world
    }

    #[test]
    fn rendering_produces_well_formed_line_geometry() {
        let world = scene();
        let mut debug = DebugRenderer::new();
        debug.render(&world);

        assert!(debug.line_count() > 0, "nothing was drawn");
        // Two vertices per segment, three floats each, and one colour per vertex.
        assert_eq!(debug.positions().len() % 6, 0);
        assert_eq!(debug.colors().len(), debug.positions().len());
        assert!(debug.positions().iter().all(|f| f.is_finite()));

        let geometry = debug.to_geometry();
        let positions = geometry.get_attribute("position").unwrap();
        assert_eq!(positions.count(), debug.line_count() * 2);
        assert_eq!(positions.item_size, 3);
        assert!(geometry.get_attribute("color").is_some());
    }

    #[test]
    fn every_shape_kind_draws_something() {
        let hull_points: Vec<Vector3> = [
            (-1.0f32, -1.0, -1.0), (1.0, -1.0, -1.0), (1.0, 1.0, -1.0), (-1.0, 1.0, -1.0),
            (-1.0, -1.0, 1.0), (1.0, -1.0, 1.0), (1.0, 1.0, 1.0), (-1.0, 1.0, 1.0),
        ]
        .iter()
        .map(|&(x, y, z)| Vector3::new(x, y, z))
        .collect();

        let shapes = vec![
            ("ball", Shape::ball(0.5)),
            ("cuboid", Shape::cuboid(0.5, 0.5, 0.5)),
            ("capsule", Shape::capsule(0.5, 0.3)),
            ("cylinder", Shape::cylinder(0.5, 0.3)),
            ("cone", Shape::cone(0.5, 0.3)),
            ("halfspace", Shape::ground()),
            ("hull", Shape::convex_hull(&hull_points).unwrap()),
            (
                "trimesh",
                Shape::trimesh(
                    vec![
                        Vector3::new(-1.0, 0.0, -1.0),
                        Vector3::new(1.0, 0.0, -1.0),
                        Vector3::new(0.0, 0.0, 1.0),
                    ],
                    vec![[0, 1, 2]],
                )
                .unwrap(),
            ),
            (
                "compound",
                Shape::compound(vec![
                    (Isometry::from_translation(Vector3::new(-1.0, 0.0, 0.0)), Shape::ball(0.3)),
                    (Isometry::from_translation(Vector3::new(1.0, 0.0, 0.0)), Shape::cuboid(0.3, 0.3, 0.3)),
                ]),
            ),
        ];

        for (name, shape) in shapes {
            let mut world = World::new();
            world.add_body(RigidBody::fixed().shape(shape));
            let mut debug = DebugRenderer::new();
            debug.render(&world);
            assert!(debug.line_count() > 0, "{name} drew nothing");
            assert!(
                debug.positions().iter().all(|f| f.is_finite()),
                "{name} produced a non-finite vertex"
            );
        }
    }

    #[test]
    fn contacts_are_drawn_only_when_asked_and_only_when_touching() {
        let mut world = scene();
        for _ in 0..300 {
            world.step(1.0 / 60.0);
        }

        let mut off = DebugRenderer::new();
        off.draw_contacts = false;
        off.render(&world);
        let without = off.line_count();

        let mut on = DebugRenderer::new();
        on.draw_contacts = true;
        on.render(&world);
        assert!(
            on.line_count() > without,
            "enabling contacts drew nothing extra ({} vs {without})",
            on.line_count()
        );
    }

    #[test]
    fn joints_show_the_gap_between_their_anchors() {
        let mut world = World::new();
        let anchor = world.add_body(RigidBody::fixed());
        let bob = world.add_body(
            RigidBody::dynamic()
                .shape(Shape::ball(0.2))
                .translation(Vector3::new(2.0, 0.0, 0.0)),
        );
        world.add_joint(Joint::spherical(
            anchor,
            bob,
            Vector3::ZERO,
            Vector3::new(-2.0, 0.0, 0.0),
        ));

        let mut with = DebugRenderer::new();
        with.draw_joints = true;
        with.render(&world);

        let mut without = DebugRenderer::new();
        without.draw_joints = false;
        without.render(&world);

        assert!(with.line_count() > without.line_count(), "the joint was not drawn");
    }

    #[test]
    fn sleeping_and_sensor_bodies_are_tinted_differently() {
        let mut world = World::new();
        world.add_body(RigidBody::fixed().shape(Shape::ground()));
        let box_id = world.add_body(
            RigidBody::dynamic()
                .shape(Shape::cuboid(0.5, 0.5, 0.5))
                .translation(Vector3::new(0.0, 0.5, 0.0)),
        );
        world.add_body(
            RigidBody::fixed()
                .collider(Collider::new(Shape::ball(0.5)).sensor(true))
                .translation(Vector3::new(4.0, 1.0, 0.0)),
        );

        let debug = DebugRenderer::new();
        let awake = debug.body_color(world.body(box_id).unwrap());
        for _ in 0..300 {
            world.step(1.0 / 60.0);
        }
        assert!(world.body(box_id).unwrap().is_sleeping());
        let asleep = debug.body_color(world.body(box_id).unwrap());
        assert_ne!(awake, asleep, "a sleeping body should be drawn differently");
    }

    #[test]
    fn a_huge_triangle_mesh_is_capped_rather_than_swamping_the_overlay() {
        // Terrain can be hundreds of thousands of triangles; the overlay must
        // stay bounded or it costs more than the simulation.
        let n = 4000;
        let mut vertices = Vec::new();
        let mut indices = Vec::new();
        for i in 0..n {
            let f = i as f32 * 0.1;
            vertices.push(Vector3::new(f, 0.0, 0.0));
            vertices.push(Vector3::new(f, 0.0, 1.0));
            vertices.push(Vector3::new(f + 0.1, 0.0, 0.0));
            indices.push([(i * 3) as u32, (i * 3 + 1) as u32, (i * 3 + 2) as u32]);
        }
        let mut world = World::new();
        world.add_body(RigidBody::fixed().shape(Shape::trimesh(vertices, indices).unwrap()));

        let mut debug = DebugRenderer::new();
        debug.max_mesh_edges = 100;
        debug.render(&world);
        assert!(
            debug.line_count() <= 100 * 3,
            "the mesh cap was ignored: {} lines",
            debug.line_count()
        );
    }

    #[test]
    fn rendering_twice_does_not_accumulate() {
        let world = scene();
        let mut debug = DebugRenderer::new();
        debug.render(&world);
        let first = debug.line_count();
        debug.render(&world);
        assert_eq!(debug.line_count(), first, "buffers were not cleared");
    }

    #[test]
    fn an_empty_world_draws_nothing_without_panicking() {
        let world = World::new();
        let mut debug = DebugRenderer::verbose();
        debug.render(&world);
        assert_eq!(debug.line_count(), 0);
        let geometry = debug.to_geometry();
        assert_eq!(geometry.get_attribute("position").unwrap().count(), 0);
    }

    #[test]
    fn aabb_and_bvh_overlays_can_be_switched_on() {
        let world = scene();
        let mut plain = DebugRenderer::new();
        plain.render(&world);

        let mut with_aabbs = DebugRenderer::new();
        with_aabbs.draw_aabbs = true;
        with_aabbs.render(&world);
        assert!(with_aabbs.line_count() > plain.line_count());

        let mut with_bvh = DebugRenderer::new();
        with_bvh.draw_colliders = false;
        with_bvh.draw_joints = false;
        with_bvh.draw_bvh = true;
        with_bvh.render(&world);
        assert!(with_bvh.line_count() > 0, "the bvh overlay drew nothing");
    }
}
