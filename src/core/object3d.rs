use super::{InstancedMesh, Layers, LineSegments, Mesh, Points, SkinnedMesh, Sprite};
use crate::lights::Light;
use crate::math::{Euler, EulerOrder, Matrix4, Quaternion, Vector3};
use slotmap::{new_key_type, SlotMap};

new_key_type! {
    /// Stable handle to an Object3D in a Scene. Cloneable, `Copy`.
    pub struct ObjectId;
}

/// What kind of node this is. Groups have no geometry; meshes do.
/// Cameras live in their own types — this enum is for the renderable graph.
#[derive(Debug, Clone)]
pub enum ObjectKind {
    Group,
    Mesh(Mesh),
    Light(Light),
    LineSegments(LineSegments),
    Points(Points),
    Sprite(Sprite),
    InstancedMesh(InstancedMesh),
    SkinnedMesh(SkinnedMesh),
}

/// A node in the scene graph. Matches three.js's `Object3D`: position,
/// rotation, scale, plus parent/child links and a world matrix updated by the
/// renderer.
///
/// # Rotation
///
/// The quaternion is the rotation. three.js keeps an `Euler` (`rotation`) and
/// a quaternion and syncs each from the other on every write; here the Euler
/// view is derived on demand — [`rotation`](Self::rotation) reads it in
/// [`rotation_order`](Self::rotation_order), [`set_rotation`](Self::set_rotation)
/// writes it — so the two can never disagree. All six of three.js's orders
/// are supported.
///
/// # Matrices
///
/// With [`matrix_auto_update`](Self::matrix_auto_update) on (the default, as
/// three.js's `Object3D.DEFAULT_MATRIX_AUTO_UPDATE`) the world update
/// recomposes [`matrix`](Self::matrix) from position, quaternion and scale.
/// Off, `matrix` is left as written — any affine matrix, shear included — and
/// position/quaternion/scale no longer drive it.
/// [`matrix_world_auto_update`](Self::matrix_world_auto_update) is honoured
/// where a three.js render honours it: on the scene's root, where it turns
/// the whole per-frame world update off (`WebGLRenderer.render` checks
/// `scene.matrixWorldAutoUpdate`). Below the root three.js's update runs
/// forced — the root recomposes, which marks it dirty, which forces every
/// descendant — so a descendant's flag changes nothing there, and here.
#[derive(Debug, Clone)]
pub struct Object3D {
    pub name: String,
    pub position: Vector3,
    pub quaternion: Quaternion,
    pub scale: Vector3,
    pub up: Vector3,
    pub visible: bool,
    pub layers: Layers,
    /// three.js `Object3D.castShadow` — included in shadow-map depth passes.
    pub cast_shadow: bool,
    /// three.js `Object3D.receiveShadow` — directional/spot/point shadows darken this mesh.
    pub receive_shadow: bool,
    /// three.js `Object3D.renderOrder` — lower values draw first within the same transparency class.
    pub render_order: i32,
    /// Whether this object is captured into the screen-space glass buffer and so
    /// refracted/reflected by `TransparencyMode::Refract` surfaces. `false` draws
    /// it *over* the glass instead (an overlay) — e.g. annotations or connectivity
    /// that should stay crisp and not be bent by the glass. Default `true`.
    pub refract_capture: bool,
    /// A world-space clipping plane: fragments on its negative side — where
    /// `dot(normal, p) + constant < 0` — are discarded, as three.js discards
    /// them for a plane in `material.clippingPlanes` (with
    /// `renderer.localClippingEnabled`). Per object rather than per material
    /// because it is uniform data, and an object is where a draw's uniforms
    /// come from; a three.js layer maps a material's first plane onto each
    /// object drawn with it. `None` (the default) clips nothing.
    ///
    /// Clipped in every colour pass, opaque and transparent. Not in the shadow
    /// maps or the refraction back-face prepass, which have no fragment stage
    /// to discard in: a clipped caster still casts its whole shadow.
    pub clip_plane: Option<crate::math::Plane>,
    pub kind: ObjectKind,
    /// three.js `isGroup`: this node is a `Group`, as opposed to a bare
    /// `Object3D` — both are [`ObjectKind::Group`] here. Only what reads
    /// three.js's flag cares, such as `DragControls.transformGroup`, which
    /// drags the outermost `Group` above the object picked.
    pub is_group: bool,

    pub matrix: Matrix4,
    pub matrix_world: Matrix4,
    /// three.js `matrixAutoUpdate`: recompose [`matrix`](Self::matrix) from
    /// position/quaternion/scale on every world update. Default `true`.
    pub matrix_auto_update: bool,
    /// three.js `matrixWorldAutoUpdate`. Default `true`. Off on a scene's
    /// root, [`Scene::update_world`](crate::Scene::update_world) leaves every
    /// world matrix as it is; see the type docs for why it does nothing
    /// lower down.
    pub matrix_world_auto_update: bool,
    /// three.js `rotation.order`: the axis order [`rotation`](Self::rotation)
    /// reports the quaternion in. Default `XYZ`.
    pub rotation_order: EulerOrder,

    pub parent: Option<ObjectId>,
    pub children: Vec<ObjectId>,
}

impl Object3D {
    pub fn group() -> Self {
        Self::with_kind(ObjectKind::Group)
    }

    pub fn mesh(mesh: Mesh) -> Self {
        Self::with_kind(ObjectKind::Mesh(mesh))
    }

    pub fn light(light: impl Into<Light>) -> Self {
        Self::with_kind(ObjectKind::Light(light.into()))
    }

    pub fn line_segments(ls: LineSegments) -> Self {
        Self::with_kind(ObjectKind::LineSegments(ls))
    }

    pub fn points(p: Points) -> Self {
        Self::with_kind(ObjectKind::Points(p))
    }

    pub fn sprite(s: Sprite) -> Self {
        Self::with_kind(ObjectKind::Sprite(s))
    }

    pub fn instanced_mesh(im: InstancedMesh) -> Self {
        Self::with_kind(ObjectKind::InstancedMesh(im))
    }

    pub fn skinned_mesh(sm: SkinnedMesh) -> Self {
        Self::with_kind(ObjectKind::SkinnedMesh(sm))
    }

    fn with_kind(kind: ObjectKind) -> Self {
        Self {
            name: String::new(),
            position: Vector3::ZERO,
            quaternion: Quaternion::identity(),
            scale: Vector3::ONE,
            up: Vector3::UP,
            visible: true,
            layers: Layers::default(),
            cast_shadow: false,
            receive_shadow: false,
            render_order: 0,
            clip_plane: None,
            refract_capture: true,
            kind,
            is_group: false,
            matrix: Matrix4::identity(),
            matrix_world: Matrix4::identity(),
            matrix_auto_update: true,
            matrix_world_auto_update: true,
            rotation_order: EulerOrder::XYZ,
            parent: None,
            children: Vec::new(),
        }
    }

    /// Recompute local matrix from position/quaternion/scale — three.js
    /// `updateMatrix()`, which composes whatever
    /// [`matrix_auto_update`](Self::matrix_auto_update) says.
    pub fn update_matrix(&mut self) {
        self.matrix = Matrix4::compose(self.position, self.quaternion, self.scale);
    }

    /// Set [`matrix`](Self::matrix) directly and turn
    /// [`matrix_auto_update`](Self::matrix_auto_update) off, so the next world
    /// update keeps it — three.js's `object.matrix.copy(m);
    /// object.matrixAutoUpdate = false`. Position, quaternion and scale are
    /// decomposed from it for reading, as far as a TRS can describe it.
    pub fn set_matrix(&mut self, m: Matrix4) -> &mut Self {
        self.matrix = m;
        self.matrix_auto_update = false;
        let (p, q, s) = m.decompose();
        self.position = p;
        self.quaternion = q;
        self.scale = s;
        self
    }

    /// three.js `rotation`: the quaternion as Euler angles in
    /// [`rotation_order`](Self::rotation_order).
    pub fn rotation(&self) -> Euler {
        Euler::from_quaternion(self.quaternion, self.rotation_order)
    }

    /// three.js `rotation.copy(e)` / `rotation.set(x, y, z, order)`: set the
    /// quaternion from `e` and adopt its order.
    pub fn set_rotation(&mut self, e: Euler) -> &mut Self {
        self.quaternion = Quaternion::from_euler(e);
        self.rotation_order = e.order;
        self
    }

    /// three.js `setRotationFromAxisAngle(axis, angle)` (unit axis).
    pub fn set_rotation_from_axis_angle(&mut self, axis: Vector3, angle: f32) -> &mut Self {
        self.quaternion = Quaternion::from_axis_angle(axis, angle);
        self
    }

    /// three.js `setRotationFromMatrix(m)`: the upper 3x3 must be a pure
    /// rotation.
    pub fn set_rotation_from_matrix(&mut self, m: &Matrix4) -> &mut Self {
        self.quaternion = Quaternion::from_rotation_matrix(m);
        self
    }

    /// three.js `setRotationFromQuaternion(q)`.
    pub fn set_rotation_from_quaternion(&mut self, q: Quaternion) -> &mut Self {
        self.quaternion = q;
        self
    }

    /// Whether three.js's `lookAt` points this object's -Z at the target
    /// (cameras and lights) rather than +Z (everything else). Cameras are not
    /// scene nodes here, so of the node kinds only lights.
    fn looks_down_negative_z(&self) -> bool {
        matches!(self.kind, ObjectKind::Light(_))
    }

    /// The rotation [`look_at`](Self::look_at) gives an object at world
    /// position `eye`, before any parent correction.
    fn look_rotation(&self, eye: Vector3, target: Vector3) -> Quaternion {
        let m = if self.looks_down_negative_z() {
            Matrix4::target_look_at(eye, target, self.up)
        } else {
            Matrix4::target_look_at(target, eye, self.up)
        };
        Quaternion::from_rotation_matrix(&m)
    }

    // -- transformation helpers (matches three.js `Object3D`) --

    /// Translate by `distance` along an axis given in **local** space.
    pub fn translate_on_axis(&mut self, axis: Vector3, distance: f32) -> &mut Self {
        let world_axis = axis.apply_quaternion(self.quaternion);
        self.position = self.position + world_axis * distance;
        self
    }

    pub fn translate_x(&mut self, d: f32) -> &mut Self {
        self.translate_on_axis(Vector3::new(1.0, 0.0, 0.0), d)
    }

    pub fn translate_y(&mut self, d: f32) -> &mut Self {
        self.translate_on_axis(Vector3::new(0.0, 1.0, 0.0), d)
    }

    pub fn translate_z(&mut self, d: f32) -> &mut Self {
        self.translate_on_axis(Vector3::new(0.0, 0.0, 1.0), d)
    }

    /// Rotate about a unit axis given in **local** space.
    pub fn rotate_on_axis(&mut self, axis: Vector3, angle: f32) -> &mut Self {
        let q = Quaternion::from_axis_angle(axis, angle);
        self.quaternion = self.quaternion.multiply(q);
        self
    }

    /// Rotate about a unit axis given in **world** space.
    pub fn rotate_on_world_axis(&mut self, axis: Vector3, angle: f32) -> &mut Self {
        let q = Quaternion::from_axis_angle(axis, angle);
        self.quaternion = self.quaternion.premultiply(q);
        self
    }

    pub fn rotate_x(&mut self, angle: f32) -> &mut Self {
        self.rotate_on_axis(Vector3::new(1.0, 0.0, 0.0), angle)
    }

    pub fn rotate_y(&mut self, angle: f32) -> &mut Self {
        self.rotate_on_axis(Vector3::new(0.0, 1.0, 0.0), angle)
    }

    pub fn rotate_z(&mut self, angle: f32) -> &mut Self {
        self.rotate_on_axis(Vector3::new(0.0, 0.0, 1.0), angle)
    }

    /// three.js `applyMatrix4(m)`: pre-multiply the local matrix by `m`
    /// (recomposing it first when [`matrix_auto_update`](Self::matrix_auto_update)
    /// is on) and decompose the result into position/quaternion/scale. With
    /// auto-update off the product stays in [`matrix`](Self::matrix) as is,
    /// shear and all.
    pub fn apply_matrix4(&mut self, m: &Matrix4) -> &mut Self {
        if self.matrix_auto_update {
            self.update_matrix();
        }
        self.matrix = m.multiply(&self.matrix);
        let (p, q, s) = self.matrix.decompose();
        self.position = p;
        self.quaternion = q;
        self.scale = s;
        self
    }

    /// three.js `applyQuaternion(q)`: rotate by `q` in the parent's space.
    pub fn apply_quaternion(&mut self, q: Quaternion) -> &mut Self {
        self.quaternion = q.multiply(self.quaternion);
        self
    }

    /// three.js `lookAt(target)` for an object without a parent (its
    /// `position` is its world position): lights turn their -Z towards
    /// `target`, every other object its +Z. For a parented object use
    /// [`ObjectArena::look_at`], which reads the world position and corrects
    /// for the parent's rotation as three.js does.
    pub fn look_at(&mut self, target: Vector3) -> &mut Self {
        self.quaternion = self.look_rotation(self.position, target);
        self
    }

    pub fn world_position(&self) -> Vector3 {
        let e = &self.matrix_world.elements;
        Vector3::new(e[12], e[13], e[14])
    }

    pub fn world_quaternion(&self) -> Quaternion {
        let (_, q, _) = self.matrix_world.decompose();
        q
    }

    pub fn world_scale(&self) -> Vector3 {
        let (_, _, s) = self.matrix_world.decompose();
        s
    }
}

/// Arena that owns all `Object3D`s in a graph and exposes three.js-style
/// `add`/`remove`/`traverse`. Roots are tracked separately so the renderer
/// can walk from any root.
#[derive(Debug, Default)]
pub struct ObjectArena {
    pub nodes: SlotMap<ObjectId, Object3D>,
}

impl ObjectArena {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&mut self, obj: Object3D) -> ObjectId {
        self.nodes.insert(obj)
    }

    pub fn get(&self, id: ObjectId) -> Option<&Object3D> {
        self.nodes.get(id)
    }

    pub fn get_mut(&mut self, id: ObjectId) -> Option<&mut Object3D> {
        self.nodes.get_mut(id)
    }

    /// Attach `child` under `parent`. If `child` already has a parent, it is
    /// detached first — matches three.js semantics.
    pub fn add_child(&mut self, parent: ObjectId, child: ObjectId) {
        if let Some(old_parent) = self.nodes.get(child).and_then(|c| c.parent) {
            if let Some(p) = self.nodes.get_mut(old_parent) {
                p.children.retain(|c| *c != child);
            }
        }
        if let Some(c) = self.nodes.get_mut(child) {
            c.parent = Some(parent);
        }
        if let Some(p) = self.nodes.get_mut(parent) {
            p.children.push(child);
        }
    }

    /// Detach `child` from its parent.
    pub fn remove_child(&mut self, child: ObjectId) {
        let parent = self.nodes.get(child).and_then(|c| c.parent);
        if let Some(p) = parent {
            if let Some(parent_obj) = self.nodes.get_mut(p) {
                parent_obj.children.retain(|c| *c != child);
            }
        }
        if let Some(c) = self.nodes.get_mut(child) {
            c.parent = None;
        }
    }

    /// Remove `id` from the arena for good: detached from its parent, its
    /// children left without one (they stay in the arena, as three.js's
    /// children of a dropped object stay alive while referenced). The id is
    /// dead afterwards.
    pub fn delete(&mut self, id: ObjectId) {
        self.remove_child(id);
        if let Some(node) = self.nodes.remove(id) {
            for c in node.children {
                if let Some(child) = self.nodes.get_mut(c) {
                    if child.parent == Some(id) {
                        child.parent = None;
                    }
                }
            }
        }
    }

    /// Make `children`, in that order, exactly `parent`'s children: each is
    /// moved under `parent` (out of wherever it was), and any current child
    /// not in the list is detached. three.js keeps `children` in insertion
    /// order, and the order is what a traversal — so draw order among equals
    /// — follows; a mirror of a three.js graph sets it whole. Ids that are
    /// not in the arena, `parent` itself, or an ancestor of `parent` (which
    /// would make a cycle) are skipped.
    pub fn set_children(&mut self, parent: ObjectId, children: &[ObjectId]) {
        if !self.nodes.contains_key(parent) {
            return;
        }
        let mut ancestors = vec![parent];
        let mut cur = self.nodes[parent].parent;
        while let Some(p) = cur {
            ancestors.push(p);
            cur = self.nodes.get(p).and_then(|n| n.parent);
        }
        let keep: Vec<ObjectId> = children
            .iter()
            .copied()
            .filter(|c| self.nodes.contains_key(*c) && !ancestors.contains(c))
            .collect();
        let old = std::mem::take(&mut self.nodes[parent].children);
        for c in old {
            if !keep.contains(&c) {
                if let Some(n) = self.nodes.get_mut(c) {
                    n.parent = None;
                }
            }
        }
        for &c in &keep {
            if let Some(prev) = self.nodes[c].parent.filter(|p| *p != parent) {
                if let Some(p) = self.nodes.get_mut(prev) {
                    p.children.retain(|x| *x != c);
                }
            }
            self.nodes[c].parent = Some(parent);
        }
        self.nodes[parent].children = keep;
    }

    /// Walk the subtree rooted at `id`, updating each node's `matrix_world` —
    /// three.js `updateMatrixWorld(true)`: a node with
    /// [`matrix_auto_update`](Object3D::matrix_auto_update) recomposes its
    /// local matrix first, one without keeps the matrix it was given, and
    /// every node's world matrix is its parent's times its own (forced, so
    /// [`matrix_world_auto_update`](Object3D::matrix_world_auto_update) does
    /// not stop it, as it does not in three.js).
    pub fn update_world_matrices(&mut self, id: ObjectId, parent_world: Matrix4) {
        let world = {
            let Some(node) = self.nodes.get_mut(id) else {
                return;
            };
            if node.matrix_auto_update {
                node.update_matrix();
            }
            node.matrix_world = parent_world.multiply(&node.matrix);
            node.matrix_world
        };
        let children = self.nodes[id].children.clone();
        for c in children {
            self.update_world_matrices(c, world);
        }
    }

    /// The world matrix of `id`'s parent, freshly computed from the roots
    /// down (three.js `updateWorldMatrix(true, false)` on the parent), or
    /// identity for a root.
    pub fn parent_world_matrix(&mut self, id: ObjectId) -> Matrix4 {
        let mut chain = Vec::new();
        let mut cur = self.nodes.get(id).and_then(|n| n.parent);
        while let Some(p) = cur {
            chain.push(p);
            cur = self.nodes.get(p).and_then(|n| n.parent);
        }
        let mut world = Matrix4::identity();
        for &n in chain.iter().rev() {
            let node = &mut self.nodes[n];
            if node.matrix_auto_update {
                node.update_matrix();
            }
            node.matrix_world = world.multiply(&node.matrix);
            world = node.matrix_world;
        }
        world
    }

    /// three.js `object.lookAt(target)`, `target` in world space: lights turn
    /// their -Z to it, everything else its +Z, and a parent's world rotation
    /// is divided out so the result holds in world space. Like three.js, not
    /// for objects under a non-uniformly scaled parent.
    pub fn look_at(&mut self, id: ObjectId, target: Vector3) {
        let parent_world = self.parent_world_matrix(id);
        let Some(node) = self.nodes.get_mut(id) else {
            return;
        };
        if node.matrix_auto_update {
            node.update_matrix();
        }
        let world = parent_world.multiply(&node.matrix);
        let eye = Vector3::new(world.elements[12], world.elements[13], world.elements[14]);
        let mut q = node.look_rotation(eye, target);
        if node.parent.is_some() {
            let pq = Quaternion::from_rotation_matrix(&parent_world.extract_rotation());
            q = pq.invert().multiply(q);
        }
        node.quaternion = q;
    }

    /// three.js `attach(child)`: reparent `child` under `parent` keeping its
    /// world transform. With `child`'s `matrix_auto_update` off the new local
    /// matrix is stored as is (shear survives); otherwise it is decomposed.
    pub fn attach(&mut self, parent: ObjectId, child: ObjectId) {
        let parent_world = {
            let pw = self.parent_world_matrix(parent);
            let Some(p) = self.nodes.get_mut(parent) else {
                return;
            };
            if p.matrix_auto_update {
                p.update_matrix();
            }
            pw.multiply(&p.matrix)
        };
        let child_world = {
            let cw = self.parent_world_matrix(child);
            let Some(c) = self.nodes.get_mut(child) else {
                return;
            };
            if c.matrix_auto_update {
                c.update_matrix();
            }
            cw.multiply(&c.matrix)
        };
        let local = parent_world.invert().multiply(&child_world);
        self.add_child(parent, child);
        if let Some(c) = self.nodes.get_mut(child) {
            c.matrix = local;
            let (p, q, s) = local.decompose();
            c.position = p;
            c.quaternion = q;
            c.scale = s;
        }
    }

    /// Pre-order traverse. Calls `f(id)` for `id` and every descendant.
    pub fn traverse(&self, id: ObjectId, f: &mut impl FnMut(ObjectId, &Object3D)) {
        if let Some(node) = self.nodes.get(id) {
            f(id, node);
            for c in node.children.clone() {
                self.traverse(c, f);
            }
        }
    }

    /// Pre-order traverse, skipping subtrees whose root is not visible.
    pub fn traverse_visible(&self, id: ObjectId, f: &mut impl FnMut(ObjectId, &Object3D)) {
        if let Some(node) = self.nodes.get(id) {
            if !node.visible {
                return;
            }
            f(id, node);
            for c in node.children.clone() {
                self.traverse_visible(c, f);
            }
        }
    }

    /// Walk up the chain from `id` through its ancestors.
    pub fn traverse_ancestors(&self, id: ObjectId, f: &mut impl FnMut(ObjectId, &Object3D)) {
        let mut cur = self.nodes.get(id).and_then(|n| n.parent);
        while let Some(p) = cur {
            if let Some(n) = self.nodes.get(p) {
                f(p, n);
                cur = n.parent;
            } else {
                break;
            }
        }
    }

    /// First object in `root`'s subtree whose name matches.
    pub fn get_object_by_name(&self, root: ObjectId, name: &str) -> Option<ObjectId> {
        let mut found = None;
        self.traverse(root, &mut |id, obj| {
            if found.is_none() && obj.name == name {
                found = Some(id);
            }
        });
        found
    }

    /// All objects in `root`'s subtree whose name matches.
    pub fn get_objects_by_name(&self, root: ObjectId, name: &str) -> Vec<ObjectId> {
        let mut out = Vec::new();
        self.traverse(root, &mut |id, obj| {
            if obj.name == name {
                out.push(id);
            }
        });
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ids(arena: &ObjectArena, id: ObjectId) -> Vec<ObjectId> {
        arena.get(id).unwrap().children.clone()
    }

    /// `set_children` makes exactly the given list, in order, moving nodes in
    /// from other parents and detaching the ones left out; it refuses cycles.
    #[test]
    fn set_children_mirrors_a_child_list() {
        let mut arena = ObjectArena::new();
        let [a, b, c, d] = [(); 4].map(|_| arena.insert(Object3D::group()));
        arena.add_child(a, b);
        arena.add_child(a, c);
        arena.add_child(d, c); // c moves under d
        arena.set_children(a, &[c, b]);
        assert_eq!(ids(&arena, a), vec![c, b], "the given order");
        assert!(ids(&arena, d).is_empty(), "moved out of its old parent");
        assert_eq!(arena.get(c).unwrap().parent, Some(a));
        arena.set_children(a, &[b]);
        assert_eq!(arena.get(c).unwrap().parent, None, "left out: detached");
        arena.set_children(b, &[a]);
        assert!(
            ids(&arena, b).is_empty(),
            "an ancestor is never made a child"
        );
    }

    /// `delete` drops a node and leaves its children parentless.
    #[test]
    fn delete_orphans_children() {
        let mut arena = ObjectArena::new();
        let [a, b, c] = [(); 3].map(|_| arena.insert(Object3D::group()));
        arena.add_child(a, b);
        arena.add_child(b, c);
        arena.delete(b);
        assert!(arena.get(b).is_none());
        assert!(ids(&arena, a).is_empty());
        assert_eq!(arena.get(c).unwrap().parent, None);
    }
}
