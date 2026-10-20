//! three.js r165 `DragControls`, on an [`ObjectArena`].
//!
//! The control is a small state machine fed one pointer event at a time. What
//! it picks comes from a raycast the caller makes — three.js casts with
//! `raycaster.intersectObjects(objects, recursive)`, which consults each
//! object's own `raycast`, so a mirror of a three.js graph casts there and
//! hands the nearest hit in — or, for a graph that lives in Rust, from
//! [`DragControls::pick`]. The control then moves the picked object exactly as
//! three.js's does: in `Translate` mode along the plane through the object
//! facing the camera, with the parent's transform divided out so a parented
//! object follows the pointer in world space; in `Rotate` mode about the
//! camera's up and right axes.

use crate::cameras::Camera;
use crate::core::{ObjectArena, ObjectId, Raycaster};
use crate::math::{Matrix4, Plane, Quaternion, Ray, Vector2, Vector3};

/// three.js `DragControls.mode`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DragMode {
    /// Move the object in the plane through it facing the camera.
    #[default]
    Translate,
    /// Turn the object about the camera's up and right axes.
    Rotate,
}

/// The events three.js's `DragControls` dispatches, with the object each is
/// about.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DragEvent {
    HoverOn(ObjectId),
    HoverOff(ObjectId),
    DragStart(ObjectId),
    Drag(ObjectId),
    DragEnd(ObjectId),
}

impl DragEvent {
    /// three.js's event `type`.
    pub fn kind(&self) -> &'static str {
        match self {
            DragEvent::HoverOn(_) => "hoveron",
            DragEvent::HoverOff(_) => "hoveroff",
            DragEvent::DragStart(_) => "dragstart",
            DragEvent::Drag(_) => "drag",
            DragEvent::DragEnd(_) => "dragend",
        }
    }

    /// The event's `object`.
    pub fn object(&self) -> ObjectId {
        match *self {
            DragEvent::HoverOn(o)
            | DragEvent::HoverOff(o)
            | DragEvent::DragStart(o)
            | DragEvent::Drag(o)
            | DragEvent::DragEnd(o) => o,
        }
    }
}

/// What one pointer event did: the events, in three.js's order, and the
/// cursor three.js would set on the element (`None`: left alone).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct DragOutcome {
    pub events: Vec<DragEvent>,
    pub cursor: Option<&'static str>,
}

/// The camera as `DragControls` reads it: `matrixWorld` (for its world
/// direction) and `quaternion` (the rotate mode's axes — three.js reads the
/// camera's own quaternion, not its world one).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DragCamera {
    pub matrix_world: Matrix4,
    pub quaternion: Quaternion,
}

impl DragCamera {
    /// From a Rust camera, which has no parent: its view's inverse.
    pub fn from_camera(camera: &dyn Camera) -> Self {
        let matrix_world = camera.view_matrix().invert();
        let (_, quaternion, _) = matrix_world.decompose();
        Self {
            matrix_world,
            quaternion,
        }
    }

    /// three.js `getWorldDirection` for a camera: its world -Z.
    fn world_direction(&self) -> Vector3 {
        let e = &self.matrix_world.elements;
        Vector3::new(-e[8], -e[9], -e[10]).normalize()
    }
}

/// three.js r165 `DragControls`. See the module docs.
#[derive(Debug, Clone)]
pub struct DragControls {
    /// The draggable objects (three.js `getObjects()` / `setObjects()`).
    pub objects: Vec<ObjectId>,
    /// Off: every pointer event is ignored.
    pub enabled: bool,
    /// Cast against the objects' descendants too (three.js default `true`).
    pub recursive: bool,
    /// Drag the outermost `Group` ([`is_group`](crate::core::Object3D::is_group))
    /// above the object picked instead of the object itself.
    pub transform_group: bool,
    pub mode: DragMode,
    pub rotate_speed: f32,
    selected: Option<ObjectId>,
    hovered: Option<ObjectId>,
    plane: Plane,
    offset: Vector3,
    inverse_matrix: Matrix4,
    previous_pointer: Vector2,
    up: Vector3,
    right: Vector3,
}

impl DragControls {
    pub fn new(objects: Vec<ObjectId>) -> Self {
        Self {
            objects,
            enabled: true,
            recursive: true,
            transform_group: false,
            mode: DragMode::Translate,
            rotate_speed: 1.0,
            selected: None,
            hovered: None,
            plane: Plane::default(),
            offset: Vector3::ZERO,
            inverse_matrix: Matrix4::identity(),
            previous_pointer: Vector2::ZERO,
            up: Vector3::ZERO,
            right: Vector3::ZERO,
        }
    }

    /// The object being dragged.
    pub fn selected(&self) -> Option<ObjectId> {
        self.selected
    }

    /// The object under the pointer, as the last hover saw it.
    pub fn hovered(&self) -> Option<ObjectId> {
        self.hovered
    }

    /// The nearest of [`objects`](Self::objects) (and, when
    /// [`recursive`](Self::recursive), their descendants) under `raycaster`'s
    /// ray, for a graph that lives in Rust. World matrices must be current.
    pub fn pick(&self, arena: &ObjectArena, raycaster: &Raycaster) -> Option<ObjectId> {
        raycaster
            .intersect_object_list(arena, &self.objects, self.recursive)
            .first()
            .map(|h| h.object)
    }

    /// three.js `findGroup`: the outermost `Group` on the way up from `id`.
    fn find_group(arena: &ObjectArena, id: ObjectId) -> Option<ObjectId> {
        let mut group = None;
        let mut cur = Some(id);
        while let Some(c) = cur {
            let node = arena.get(c)?;
            if node.is_group {
                group = Some(c);
            }
            cur = node.parent;
        }
        group
    }

    fn world_position(arena: &ObjectArena, id: ObjectId) -> Vector3 {
        arena
            .get(id)
            .map(|o| o.world_position())
            .unwrap_or(Vector3::ZERO)
    }

    /// `pointerdown`: `pointer` in NDC, `ray` the raycaster's ray through it
    /// and `hit` the nearest object it hit among the draggables. World
    /// matrices must be current.
    pub fn pointer_down(
        &mut self,
        arena: &mut ObjectArena,
        pointer: Vector2,
        ray: &Ray,
        hit: Option<ObjectId>,
        camera: &DragCamera,
    ) -> DragOutcome {
        let mut out = DragOutcome::default();
        if !self.enabled {
            return out;
        }
        if let Some(hit) = hit {
            let selected = if self.transform_group {
                Self::find_group(arena, hit)
            } else {
                Some(hit)
            };
            if let Some(sel) = selected {
                self.selected = Some(sel);
                let world_position = Self::world_position(arena, sel);
                self.plane =
                    Plane::from_normal_and_coplanar_point(camera.world_direction(), world_position);
                if let Some(t) = ray.intersect_plane(&self.plane) {
                    let intersection = ray.at(t);
                    match self.mode {
                        DragMode::Translate => {
                            self.inverse_matrix = arena.parent_world_matrix(sel).invert();
                            self.offset = intersection - world_position;
                        }
                        DragMode::Rotate => {
                            // three.js: "the controls only support Y+ up".
                            self.up = Vector3::new(0.0, 1.0, 0.0)
                                .apply_quaternion(camera.quaternion)
                                .normalize();
                            self.right = Vector3::new(1.0, 0.0, 0.0)
                                .apply_quaternion(camera.quaternion)
                                .normalize();
                        }
                    }
                }
                out.cursor = Some("move");
                out.events.push(DragEvent::DragStart(sel));
            }
        }
        self.previous_pointer = pointer;
        out
    }

    /// `pointermove`: drags the selected object, or — with nothing selected
    /// and `hover_hit` given (three.js hovers for mouse and pen pointers
    /// only; pass `None` for touch) — tracks the hovered one. `hover_hit` is
    /// `Some(None)` when the hover cast hit nothing.
    pub fn pointer_move(
        &mut self,
        arena: &mut ObjectArena,
        pointer: Vector2,
        ray: &Ray,
        hover_hit: Option<Option<ObjectId>>,
        camera: &DragCamera,
    ) -> DragOutcome {
        let mut out = DragOutcome::default();
        if !self.enabled {
            return out;
        }
        if let Some(sel) = self.selected {
            match self.mode {
                DragMode::Translate => {
                    if let Some(t) = ray.intersect_plane(&self.plane) {
                        let p = (ray.at(t) - self.offset).apply_matrix4(&self.inverse_matrix);
                        if let Some(node) = arena.get_mut(sel) {
                            node.position = p;
                        }
                    }
                }
                DragMode::Rotate => {
                    let diff = (pointer - self.previous_pointer) * self.rotate_speed;
                    if let Some(node) = arena.get_mut(sel) {
                        node.rotate_on_world_axis(self.up, diff.x);
                        node.rotate_on_world_axis(self.right.normalize(), -diff.y);
                    }
                }
            }
            out.events.push(DragEvent::Drag(sel));
        } else if let Some(hit) = hover_hit {
            match hit {
                Some(object) => {
                    self.plane = Plane::from_normal_and_coplanar_point(
                        camera.world_direction(),
                        Self::world_position(arena, object),
                    );
                    if let Some(h) = self.hovered.filter(|h| *h != object) {
                        out.events.push(DragEvent::HoverOff(h));
                        out.cursor = Some("auto");
                        self.hovered = None;
                    }
                    if self.hovered != Some(object) {
                        out.events.push(DragEvent::HoverOn(object));
                        out.cursor = Some("pointer");
                        self.hovered = Some(object);
                    }
                }
                None => {
                    if let Some(h) = self.hovered.take() {
                        out.events.push(DragEvent::HoverOff(h));
                        out.cursor = Some("auto");
                    }
                }
            }
        }
        self.previous_pointer = pointer;
        out
    }

    /// `pointerup` / `pointerleave`: ends a drag.
    pub fn pointer_cancel(&mut self) -> DragOutcome {
        let mut out = DragOutcome::default();
        if !self.enabled {
            return out;
        }
        if let Some(sel) = self.selected.take() {
            out.events.push(DragEvent::DragEnd(sel));
        }
        out.cursor = Some(if self.hovered.is_some() {
            "pointer"
        } else {
            "auto"
        });
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cameras::PerspectiveCamera;
    use crate::core::{BufferAttribute, BufferGeometry, Mesh, Object3D};
    use crate::materials::{BasicMaterial, Material};
    use crate::math::Color;

    fn quad() -> Object3D {
        let mut g = BufferGeometry::new();
        g.set_attribute(
            "position",
            BufferAttribute::new(
                vec![
                    -0.5, -0.5, 0.0, 0.5, -0.5, 0.0, 0.5, 0.5, 0.0, -0.5, 0.5, 0.0,
                ],
                3,
            ),
        );
        g.set_index(vec![0, 1, 2, 0, 2, 3]);
        Object3D::mesh(Mesh::new(
            g,
            Material::Basic(BasicMaterial::new(Color::WHITE)),
        ))
    }

    /// A quad inside a scaled, offset group, the camera on +z looking down -z.
    fn setup() -> (ObjectArena, ObjectId, ObjectId, ObjectId, PerspectiveCamera) {
        let mut arena = ObjectArena::new();
        let root = arena.insert(Object3D::group());
        let mut g = Object3D::group();
        g.is_group = true;
        g.position = Vector3::new(1.0, 0.0, 0.0);
        g.scale = Vector3::new(2.0, 2.0, 2.0);
        let group = arena.insert(g);
        arena.add_child(root, group);
        let mesh = arena.insert(quad());
        arena.add_child(group, mesh);
        arena.update_world_matrices(root, Matrix4::identity());
        let mut camera = PerspectiveCamera::new(50.0, 1.0, 0.1, 100.0);
        camera.position = Vector3::new(1.0, 0.0, 5.0);
        camera.target = Vector3::new(1.0, 0.0, 0.0);
        (arena, root, group, mesh, camera)
    }

    fn cast(camera: &PerspectiveCamera, ndc: Vector2) -> Raycaster {
        let mut rc = Raycaster::default();
        rc.set_from_camera(ndc, camera);
        rc
    }

    #[test]
    fn a_parented_object_follows_the_pointer_in_world_space() {
        let (mut arena, root, _group, mesh, camera) = setup();
        let cam = DragCamera::from_camera(&camera);
        let mut dc = DragControls::new(vec![mesh]);
        let rc = cast(&camera, Vector2::ZERO);
        let hit = dc.pick(&arena, &rc);
        assert_eq!(hit, Some(mesh));
        let down = dc.pointer_down(&mut arena, Vector2::ZERO, &rc.ray, hit, &cam);
        assert_eq!(down.events, [DragEvent::DragStart(mesh)]);
        assert_eq!(down.cursor, Some("move"));
        // Move the pointer: the point under it on the drag plane is where the
        // grabbed point goes, whatever the group's scale.
        let to = Vector2::new(0.2, 0.1);
        let rc = cast(&camera, to);
        let moved = dc.pointer_move(&mut arena, to, &rc.ray, None, &cam);
        assert_eq!(moved.events, [DragEvent::Drag(mesh)]);
        arena.update_world_matrices(root, Matrix4::identity());
        let want = rc.ray.at(rc.ray.intersect_plane(&dc.plane).unwrap());
        let got = arena.get(mesh).unwrap().world_position();
        assert!((got - want).length() < 1e-5, "{got:?} vs {want:?}");
        // In the group's space: halved by its scale.
        let local = arena.get(mesh).unwrap().position;
        assert!((local - (want - Vector3::new(1.0, 0.0, 0.0)) * 0.5).length() < 1e-5);
        assert_eq!(dc.pointer_cancel().events, [DragEvent::DragEnd(mesh)]);
    }

    #[test]
    fn transform_group_drags_the_outermost_group() {
        let (mut arena, _root, group, mesh, camera) = setup();
        let cam = DragCamera::from_camera(&camera);
        let mut dc = DragControls::new(vec![mesh]);
        dc.transform_group = true;
        let rc = cast(&camera, Vector2::ZERO);
        let down = dc.pointer_down(&mut arena, Vector2::ZERO, &rc.ray, Some(mesh), &cam);
        assert_eq!(down.events, [DragEvent::DragStart(group)]);
        assert_eq!(dc.selected(), Some(group));
    }

    #[test]
    fn hover_reports_on_and_off() {
        let (mut arena, _root, _group, mesh, camera) = setup();
        let cam = DragCamera::from_camera(&camera);
        let mut dc = DragControls::new(vec![mesh]);
        let rc = cast(&camera, Vector2::ZERO);
        let on = dc.pointer_move(&mut arena, Vector2::ZERO, &rc.ray, Some(Some(mesh)), &cam);
        assert_eq!(on.events, [DragEvent::HoverOn(mesh)]);
        assert_eq!(on.cursor, Some("pointer"));
        let again = dc.pointer_move(&mut arena, Vector2::ZERO, &rc.ray, Some(Some(mesh)), &cam);
        assert!(again.events.is_empty());
        let off = dc.pointer_move(&mut arena, Vector2::ZERO, &rc.ray, Some(None), &cam);
        assert_eq!(off.events, [DragEvent::HoverOff(mesh)]);
        assert_eq!(dc.pointer_cancel().cursor, Some("auto"));
    }

    #[test]
    fn disabled_ignores_everything() {
        let (mut arena, _root, _group, mesh, camera) = setup();
        let cam = DragCamera::from_camera(&camera);
        let mut dc = DragControls::new(vec![mesh]);
        dc.enabled = false;
        let rc = cast(&camera, Vector2::ZERO);
        assert!(dc
            .pointer_down(&mut arena, Vector2::ZERO, &rc.ray, Some(mesh), &cam)
            .events
            .is_empty());
        assert_eq!(dc.selected(), None);
    }

    #[test]
    fn rotate_mode_turns_about_the_camera_axes() {
        let (mut arena, _root, _group, mesh, camera) = setup();
        let cam = DragCamera::from_camera(&camera);
        let mut dc = DragControls::new(vec![mesh]);
        dc.mode = DragMode::Rotate;
        let rc = cast(&camera, Vector2::ZERO);
        dc.pointer_down(&mut arena, Vector2::ZERO, &rc.ray, Some(mesh), &cam);
        dc.pointer_move(&mut arena, Vector2::new(0.5, 0.0), &rc.ray, None, &cam);
        // Half a radian about world +y (the camera's up).
        let q = arena.get(mesh).unwrap().quaternion;
        let want = Quaternion::from_axis_angle(Vector3::new(0.0, 1.0, 0.0), 0.5);
        assert!(
            (q.x - want.x).abs() < 1e-6
                && (q.y - want.y).abs() < 1e-6
                && (q.w - want.w).abs() < 1e-6
        );
    }
}
