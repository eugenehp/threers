use super::derived::Derived;
use super::{
    mesh_draws, posed_positions, BufferGeometry, InstancedMesh, Layers, LineSegments, Mesh,
    MeshDraw, Object3D, ObjectArena, ObjectId, ObjectKind, Points, SkinnedMesh, Sprite,
};
use crate::cameras::Camera;
use crate::materials::Material;
use crate::math::{Box3, Matrix4, Ray, Sphere, Triangle, Vector2, Vector3};

/// three.js's `FrontSide` / `BackSide` / `DoubleSide`, as
/// [`Material::side`] reports them.
const FRONT_SIDE: u32 = 0;
const BACK_SIDE: u32 = 1;

/// The triangle a mesh hit landed on. Mirrors three.js's `intersection.face`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Face {
    /// Vertex indices of the triangle's corners.
    pub a: usize,
    pub b: usize,
    pub c: usize,
    /// Geometric normal of the triangle as hit — after skinning — in the
    /// object's **local** space, as three.js reports it. Not flipped toward
    /// the ray.
    pub normal: Vector3,
    /// Which entry of a material array the triangle's group draws with
    /// (three.js `face.materialIndex`); `0` for a single-material mesh.
    pub material_index: usize,
}

/// A ray hit. Mirrors three.js's intersection record; the fields three.js
/// leaves `undefined` for a given object kind are `None` here.
///
/// | kind | `face` | `uv` | `normal` | `index` | `instance_id` |
/// |---|---|---|---|---|---|
/// | `Mesh`, `SkinnedMesh` | yes | with a `uv` attribute | with a `normal` attribute | — | — |
/// | `InstancedMesh` | yes | as mesh | as mesh | — | yes |
/// | `LineSegments` | — | — | — | first vertex of the segment | — |
/// | `Points` | — | — | — | the point | — |
/// | `Sprite` | — | yes | — | — | — |
///
/// Construct one with [`Intersection::new`] and struct-update syntax; new
/// fields may follow.
#[derive(Debug, Clone, Copy)]
pub struct Intersection {
    /// World-space distance from the ray origin.
    pub distance: f32,
    /// World-space hit point. For a line, the closest point on the *segment*
    /// (three.js r165); for points, the closest point on the *ray*.
    pub point: Vector3,
    /// Triangle index within the geometry. For indexed geometries this is the
    /// triple-index (i.e. positions of the three vertices are at index*3, +1, +2);
    /// for non-indexed geometries it's the triangle position in vertex order.
    ///
    /// three.js has no face for the other kinds and reports `null`; here it is
    /// the segment number for a line, the point's index for points, and `0`
    /// for a sprite. Check [`face`](Self::face) to tell a triangle hit apart.
    pub face_index: usize,
    /// threers extension: which polygon of the geometry's grouping
    /// ([`BufferGeometry::polygons`](crate::core::BufferGeometry::polygons))
    /// the hit triangle belongs to — the quad that was clicked, not half of
    /// it. `None` when the geometry has no grouping.
    pub polygon_index: Option<usize>,
    pub object: ObjectId,
    /// The triangle hit, for mesh kinds.
    pub face: Option<Face>,
    /// The geometry's `normal` attribute interpolated at the hit, in the
    /// object's **local** space and flipped to face the ray — three.js r165
    /// exactly. Not renormalised. `None` without a `normal` attribute.
    pub normal: Option<Vector3>,
    /// The `uv` attribute interpolated at the hit (sprites: the quad's own uv).
    pub uv: Option<Vector2>,
    /// The `uv1` attribute interpolated at the hit.
    pub uv1: Option<Vector2>,
    /// Barycentric weights of the hit on `face`'s corners `a`, `b`, `c`. Not
    /// in three.js r165 (later releases add it); a threers extension until then.
    pub barycoord: Option<Vector3>,
    /// The instance hit, for an [`InstancedMesh`].
    pub instance_id: Option<usize>,
    /// Vertex index: the first vertex of the segment for a line, the point
    /// for points.
    pub index: Option<usize>,
    /// For points: how far the point is from the ray, in the object's local
    /// space (three.js `distanceToRay`).
    pub distance_to_ray: Option<f32>,
}

impl Intersection {
    /// A hit with only the fields every kind has; the rest are empty.
    pub fn new(distance: f32, point: Vector3, object: ObjectId) -> Self {
        Self {
            distance,
            point,
            face_index: 0,
            polygon_index: None,
            object,
            face: None,
            normal: None,
            uv: None,
            uv1: None,
            barycoord: None,
            instance_id: None,
            index: None,
            distance_to_ray: None,
        }
    }
}

/// A pick threshold: three.js's `raycaster.params.Line` / `.Points`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ThresholdParams {
    /// World-space distance from the ray within which a line or point counts
    /// as hit. Converted to local space by the object's *own* average scale,
    /// as three.js does.
    pub threshold: f32,
}

impl Default for ThresholdParams {
    fn default() -> Self {
        Self { threshold: 1.0 }
    }
}

/// three.js's `raycaster.params`.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct RaycasterParams {
    pub line: ThresholdParams,
    pub points: ThresholdParams,
}

/// What a raycast needs to know about the camera: sprites face it, so their
/// quad depends on it. three.js keeps the camera itself on `raycaster.camera`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RaycastCamera {
    /// Camera-to-world (three.js `camera.matrixWorld`).
    pub matrix_world: Matrix4,
    /// World-to-camera — the view matrix (three.js `matrixWorldInverse`).
    pub matrix_world_inverse: Matrix4,
    /// Perspective rather than orthographic.
    pub perspective: bool,
}

impl RaycastCamera {
    pub fn from_camera(camera: &dyn Camera) -> Self {
        let view = camera.view_matrix();
        Self {
            matrix_world: view.invert(),
            matrix_world_inverse: view,
            perspective: is_perspective(&camera.projection_matrix()),
        }
    }
}

/// A perspective projection's last row is `(0, 0, -1, 0)`; an orthographic
/// one's is `(0, 0, 0, 1)`.
fn is_perspective(projection: &Matrix4) -> bool {
    projection.elements[15] == 0.0
}

/// Mirrors three.js's `Raycaster`.
///
/// Two deliberate differences from three.js: invisible objects (and their
/// subtrees) are skipped, as the renderer skips them — unless
/// [`include_invisible`](Self::include_invisible), which is three.js's rule —
/// and [`first_hit_only`](Self::first_hit_only) is built in (three-mesh-bvh's
/// `firstHitOnly`).
#[derive(Debug, Clone, Copy)]
pub struct Raycaster {
    /// The direction must be unit length, as in three.js.
    pub ray: Ray,
    pub near: f32,
    pub far: f32,
    pub layers: Layers,
    pub params: RaycasterParams,
    /// Needed to hit sprites; set by the `set_from_camera*` methods. Sprites
    /// are skipped while it is `None` (three.js logs an error).
    pub camera: Option<RaycastCamera>,
    /// Return at most one hit, the nearest — and do less work to find it:
    /// objects whose bounds start beyond the nearest hit so far are skipped,
    /// and a mesh with a BVH runs its closest-hit traversal. The hit is the
    /// one the full, sorted result would put first.
    pub first_hit_only: bool,
    /// Test invisible objects too, as three.js's `Raycaster` does (its
    /// `intersect` consults only the layers). Off by default: a hit on what is
    /// not drawn is rarely what a pick wants.
    pub include_invisible: bool,
}

impl Default for Raycaster {
    fn default() -> Self {
        Self {
            ray: Ray::default(),
            near: 0.0,
            far: f32::INFINITY,
            // three.js: `this.layers = new Layers()` — channel 0 only.
            layers: Layers::default(),
            params: RaycasterParams::default(),
            camera: None,
            first_hit_only: false,
            include_invisible: false,
        }
    }
}

/// Hits collected over one raycast. In first-hit mode only the nearest is
/// kept, and the distance it sets prunes everything after it.
struct Hits {
    list: Vec<Intersection>,
    first_only: bool,
}

impl Hits {
    /// The farthest distance still worth reporting.
    fn limit(&self, far: f32) -> f32 {
        match (self.first_only, self.list.first()) {
            (true, Some(h)) => h.distance.min(far),
            _ => far,
        }
    }

    fn push(&mut self, hit: Intersection) {
        if !self.first_only || self.list.is_empty() {
            self.list.push(hit);
        } else if hit.distance < self.list[0].distance {
            // Strictly nearer only: on a tie the earlier hit stays, which is
            // what a stable sort of the full list puts first.
            self.list[0] = hit;
        }
    }
}

/// One mesh-like thing to cast against: a mesh, one instance of an
/// instanced mesh, or a skinned mesh in its current pose.
struct MeshCast<'a> {
    id: ObjectId,
    geometry: &'a BufferGeometry,
    /// The spans to test and the material each is tested with — the mesh's
    /// draws, three.js's group loop (see [`mesh_draws`]).
    draws: &'a [MeshDraw<'a>],
    /// Local-to-world for the triangles in `positions`.
    matrix_world: Matrix4,
    /// xyz per vertex.
    positions: &'a [f32],
    /// Local bounds of `positions`.
    sphere: Sphere,
    bounds: Box3,
    derived: &'a Derived,
    instance_id: Option<usize>,
    /// The geometry's BVH describes `positions` (not true once skinned).
    use_bvh: bool,
}

impl Raycaster {
    pub fn new(origin: Vector3, direction: Vector3, near: f32, far: f32) -> Self {
        Self {
            ray: Ray::new(origin, direction),
            near,
            far,
            ..Self::default()
        }
    }

    /// three.js `set`: a new origin and (unit) direction.
    pub fn set(&mut self, origin: Vector3, direction: Vector3) -> &mut Self {
        self.ray = Ray::new(origin, direction);
        self
    }

    /// Build a ray from normalized device coordinates (x, y ∈ [-1, 1]) and a
    /// camera, perspective or orthographic — three.js's `setFromCamera`. The
    /// projection matrix tells the two apart; see
    /// [`set_from_camera_perspective`](Self::set_from_camera_perspective) and
    /// [`set_from_camera_ortho`](Self::set_from_camera_ortho) for each.
    pub fn set_from_camera(&mut self, ndc: Vector2, camera: &dyn Camera) {
        if is_perspective(&camera.projection_matrix()) {
            self.set_from_camera_perspective(ndc, camera);
        } else {
            self.set_from_camera_ortho(ndc, camera);
        }
    }

    /// Build a ray from normalized device coordinates (x, y ∈ [-1, 1]) and a
    /// perspective camera: from the eye through the point. Mirrors three.js's
    /// `setFromCamera` for `PerspectiveCamera`: sets the ray and
    /// [`camera`](Self::camera), nothing else — [`layers`](Self::layers) stay
    /// the caller's.
    pub fn set_from_camera_perspective(&mut self, ndc: Vector2, camera: &dyn Camera) {
        let view = camera.view_matrix();
        let proj = camera.projection_matrix();
        let near = Vector3::new(ndc.x, ndc.y, 0.5).unproject(&view, &proj);
        let origin = camera.position();
        self.ray = Ray::new(origin, (near - origin).normalize());
        self.camera = Some(RaycastCamera::from_camera(camera));
    }

    /// Orthographic ray: three.js's `setFromCamera` for `OrthographicCamera`.
    /// The origin is the NDC point (x, y) *in the plane of the camera* — the
    /// plane through the eye, perpendicular to the view — and the direction
    /// is the camera's forward axis. Distances therefore measure from the
    /// camera, as three.js's do. Sets the ray and [`camera`](Self::camera),
    /// nothing else.
    ///
    /// The NDC depth of that plane depends on the depth range. three.js
    /// writes it as `(near + far) / (near - far)`, which is where view-space
    /// `z = 0` lands under OpenGL's `[-1, 1]`. This crate's projections use
    /// the wgpu/D3D range `[0, 1]` — `Matrix4::orthographic` maps view depth
    /// `z` to `(-z - near) / (far - near)` — so the same plane is at
    /// `-near / (far - near)` here, and three.js's constant (or OpenGL's near
    /// plane, `-1`) would land the origin most of a frustum behind the camera,
    /// every hit past `far`. Rather than hard-code either, the plane's depth is
    /// read off the projection: view-space `(0, 0, 0, 1)` projects to clip
    /// `(…, m[14], m[15])`.
    pub fn set_from_camera_ortho(&mut self, ndc: Vector2, camera: &dyn Camera) {
        let view = camera.view_matrix();
        let proj = camera.projection_matrix();
        let e = &proj.elements;
        let camera_plane = e[14] / e[15];
        let origin = Vector3::new(ndc.x, ndc.y, camera_plane).unproject(&view, &proj);
        // three.js: `direction.set(0, 0, -1).transformDirection(matrixWorld)`.
        let forward = linear(&view.invert(), Vector3::new(0.0, 0.0, -1.0)).normalize();
        self.ray = Ray::new(origin, forward);
        self.camera = Some(RaycastCamera::from_camera(camera));
    }

    /// three.js `intersectObject`: test `object` and, when `recursive`, its
    /// visible descendants. Sorted by distance ascending.
    pub fn intersect_object(
        &self,
        arena: &ObjectArena,
        object: ObjectId,
        recursive: bool,
    ) -> Vec<Intersection> {
        self.intersect_object_list(arena, &[object], recursive)
    }

    /// Same as [`intersect_object`](Self::intersect_object) — the historical
    /// name, which takes one root. three.js's `intersectObjects` (a list) is
    /// [`intersect_object_list`](Self::intersect_object_list).
    pub fn intersect_objects(
        &self,
        arena: &ObjectArena,
        root: ObjectId,
        recursive: bool,
    ) -> Vec<Intersection> {
        self.intersect_object_list(arena, &[root], recursive)
    }

    /// three.js `intersectObjects`: every object in `objects` (and, when
    /// `recursive`, their visible descendants), all hits sorted together by
    /// distance ascending.
    pub fn intersect_object_list(
        &self,
        arena: &ObjectArena,
        objects: &[ObjectId],
        recursive: bool,
    ) -> Vec<Intersection> {
        let mut hits = Hits {
            list: Vec::new(),
            first_only: self.first_hit_only,
        };
        for &id in objects {
            self.walk(arena, id, recursive, &mut hits);
        }
        let mut out = hits.list;
        out.sort_by(|a, b| {
            a.distance
                .partial_cmp(&b.distance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        out
    }

    /// Each object of `objects` on its own — no descent, no layer test (the
    /// caller walked the graph and chose these, as three.js's `intersect`
    /// does before it calls each object's `raycast`) — with its own
    /// first-hit flag: `true` keeps only that object's nearest hit,
    /// three-mesh-bvh's `firstHitOnly` for a mesh with a `boundsTree`. Hits
    /// come back object by object, each object's sorted by distance; an
    /// invisible object is skipped unless
    /// [`include_invisible`](Self::include_invisible).
    pub fn intersect_each(
        &self,
        arena: &ObjectArena,
        objects: &[(ObjectId, bool)],
    ) -> Vec<Intersection> {
        let mut out = Vec::new();
        for &(id, first_only) in objects {
            let Some(obj) = arena.get(id) else {
                continue;
            };
            if !obj.visible && !self.include_invisible {
                continue;
            }
            let mut hits = Hits {
                list: Vec::new(),
                first_only,
            };
            self.raycast_object(id, obj, &mut hits);
            hits.list.sort_by(|a, b| {
                a.distance
                    .partial_cmp(&b.distance)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            out.extend(hits.list);
        }
        out
    }

    /// three.js r165 `setFromCamera` from the camera's matrices as three.js
    /// keeps them: `matrix_world` (camera to world) and
    /// `projection_inverse` (`projectionMatrixInverse`, OpenGL's `[-1, 1]`
    /// depth — three.js's convention, whatever this crate's renderers use).
    /// Perspective (`ortho` `None`): from the camera's position through the
    /// NDC point unprojected at depth 0.5. Orthographic (`ortho` the camera's
    /// `(near, far)`): from the NDC point in the plane three.js puts it in,
    /// `(near + far) / (near - far)`, along the camera's -Z. Also sets
    /// [`camera`](Self::camera), for sprites.
    pub fn set_from_matrices(
        &mut self,
        ndc: Vector2,
        matrix_world: &Matrix4,
        projection_inverse: &Matrix4,
        ortho: Option<(f32, f32)>,
    ) {
        let unproject = |p: Vector3| -> Vector3 {
            let e = &projection_inverse.elements;
            let w = e[3] * p.x + e[7] * p.y + e[11] * p.z + e[15];
            let w = if w != 0.0 { 1.0 / w } else { 1.0 };
            let v = Vector3::new(
                (e[0] * p.x + e[4] * p.y + e[8] * p.z + e[12]) * w,
                (e[1] * p.x + e[5] * p.y + e[9] * p.z + e[13]) * w,
                (e[2] * p.x + e[6] * p.y + e[10] * p.z + e[14]) * w,
            );
            v.apply_matrix4(matrix_world)
        };
        let m = &matrix_world.elements;
        let perspective = ortho.is_none();
        match ortho {
            None => {
                let origin = Vector3::new(m[12], m[13], m[14]);
                let toward = unproject(Vector3::new(ndc.x, ndc.y, 0.5));
                self.ray = Ray::new(origin, (toward - origin).normalize());
            }
            Some((near, far)) => {
                let origin = unproject(Vector3::new(ndc.x, ndc.y, (near + far) / (near - far)));
                let forward = linear(matrix_world, Vector3::new(0.0, 0.0, -1.0)).normalize();
                self.ray = Ray::new(origin, forward);
            }
        }
        self.camera = Some(RaycastCamera {
            matrix_world: *matrix_world,
            matrix_world_inverse: matrix_world.invert(),
            perspective,
        });
    }

    fn walk(&self, arena: &ObjectArena, id: ObjectId, recursive: bool, hits: &mut Hits) {
        let Some(obj) = arena.get(id) else {
            return;
        };
        if !obj.visible && !self.include_invisible {
            return;
        }
        if self.layers.test(&obj.layers) {
            self.raycast_object(id, obj, hits);
        }
        if recursive {
            for &child in &obj.children {
                self.walk(arena, child, true, hits);
            }
        }
    }

    fn raycast_object(&self, id: ObjectId, obj: &Object3D, hits: &mut Hits) {
        match &obj.kind {
            ObjectKind::Mesh(mesh) => self.raycast_mesh(id, obj, mesh, hits),
            ObjectKind::InstancedMesh(im) => self.raycast_instanced(id, obj, im, hits),
            ObjectKind::SkinnedMesh(sm) => self.raycast_skinned(id, obj, sm, hits),
            ObjectKind::LineSegments(ls) => self.raycast_line_segments(id, obj, ls, hits),
            ObjectKind::Points(p) => self.raycast_points(id, obj, p, hits),
            ObjectKind::Sprite(s) => self.raycast_sprite(id, obj, s, hits),
            ObjectKind::Group | ObjectKind::Light(_) => {}
        }
    }

    // ------------------------------------------------------------------
    // Meshes
    // ------------------------------------------------------------------

    /// three.js `Mesh.raycast`, morph targets included: the vertices are
    /// three.js's `getVertexPosition`, the pose every renderer draws (see
    /// [`Morph`](super::Morph)). Groups and the draw range decide which
    /// triangles are tested and with which material, as in three.js's
    /// `_computeIntersections`.
    fn raycast_mesh(&self, id: ObjectId, obj: &Object3D, mesh: &Mesh, hits: &mut Hits) {
        let draws = mesh.draws();
        self.cast_posed(
            id,
            obj,
            &mesh.geometry,
            &draws,
            &mesh.morph_influences,
            None,
            hits,
        );
    }

    /// The mesh test for a geometry in its drawn pose: morphed by
    /// `influences`, then skinned when `skin` is given. The rest pose keeps
    /// the geometry's stored bounds and its BVH; a posed one is bounded on the
    /// spot and walked without the BVH, which describes the rest pose.
    #[allow(clippy::too_many_arguments)]
    fn cast_posed(
        &self,
        id: ObjectId,
        obj: &Object3D,
        geometry: &BufferGeometry,
        draws: &[MeshDraw<'_>],
        influences: &[f32],
        skin: Option<&super::Skeleton>,
        hits: &mut Hits,
    ) {
        let Some(rest) = position_array(geometry) else {
            return;
        };
        if draws.is_empty() {
            return;
        }
        let derived = geometry.derived();
        let posed = posed_positions(geometry, influences, skin.map(|s| (s, &obj.matrix_world)));
        let Some(posed) = posed else {
            self.cast(
                MeshCast {
                    id,
                    geometry,
                    draws,
                    matrix_world: obj.matrix_world,
                    positions: rest,
                    sphere: geometry.bounding_sphere.unwrap_or(derived.bounding_sphere),
                    bounds: geometry.bounding_box.unwrap_or(derived.bounding_box),
                    derived: &derived,
                    instance_id: None,
                    use_bvh: true,
                },
                hits,
            );
            return;
        };
        let mut bounds = Box3::empty();
        for p in posed.chunks_exact(3) {
            bounds.expand_by_point(Vector3::new(p[0], p[1], p[2]));
        }
        if bounds.is_empty() {
            return;
        }
        let center = bounds.center();
        let r2 = posed
            .chunks_exact(3)
            .map(|p| (Vector3::new(p[0], p[1], p[2]) - center).length_sq())
            .fold(0.0f32, f32::max);
        self.cast(
            MeshCast {
                id,
                geometry,
                draws,
                matrix_world: obj.matrix_world,
                positions: &posed,
                sphere: Sphere::new(center, r2.sqrt()),
                bounds,
                derived: &derived,
                instance_id: None,
                use_bvh: false,
            },
            hits,
        );
    }

    /// three.js `InstancedMesh.raycast`: the mesh test once per instance,
    /// under `matrix_world × instance matrix`, tagging each hit with its
    /// instance. A BVH on the geometry serves every instance.
    fn raycast_instanced(&self, id: ObjectId, obj: &Object3D, im: &InstancedMesh, hits: &mut Hits) {
        let geometry = &*im.geometry;
        let Some(positions) = position_array(geometry) else {
            return;
        };
        // three.js's `InstancedMesh.raycast` hands each instance to a `Mesh`
        // carrying the same material (array), so groups pick the hit's
        // `materialIndex` as for a mesh. That `Mesh` has no
        // `morphTargetInfluences`, so the rest pose is what is tested.
        let draws = mesh_draws(geometry, &im.material, &im.materials);
        let derived = geometry.derived();
        let sphere = geometry.bounding_sphere.unwrap_or(derived.bounding_sphere);
        let bounds = geometry.bounding_box.unwrap_or(derived.bounding_box);
        for (instance, local) in im.transforms.iter().enumerate() {
            self.cast(
                MeshCast {
                    id,
                    geometry,
                    draws: &draws,
                    matrix_world: obj.matrix_world.multiply(local),
                    positions,
                    sphere,
                    bounds,
                    derived: &derived,
                    instance_id: Some(instance),
                    use_bvh: true,
                },
                hits,
            );
        }
    }

    /// three.js `SkinnedMesh.raycast`: the triangles in their current pose —
    /// morphed, then skinned — cast in the mesh's local space under its
    /// `matrix_world`.
    ///
    /// The pose is the one every renderer draws — three.js's skinning shader,
    /// documented on [`SkinnedMesh`] — and lands in local space, three.js's
    /// `transformed` (after `bindMatrixInverse`). So, as in three.js, the
    /// face normal is local and the interpolated `normal` (the unskinned
    /// attribute, as three.js reads it) is flipped against the *local* ray.
    /// Without `joint`/`weight` attributes the renderers draw the mesh at
    /// rest (morphed) under its `matrix_world`, and so does this.
    ///
    /// The bounds are taken from the pose on every cast, where three.js
    /// computes `SkinnedMesh.boundingSphere` once, on the first raycast, and
    /// keeps it however the bones move. The posed positions are computed once
    /// per raycast and the BVH, which describes the rest pose, is not used.
    fn raycast_skinned(&self, id: ObjectId, obj: &Object3D, sm: &SkinnedMesh, hits: &mut Hits) {
        let draws = sm.draws();
        self.cast_posed(
            id,
            obj,
            &sm.geometry,
            &draws,
            &sm.morph_influences,
            Some(&sm.skeleton),
            hits,
        );
    }

    /// The body of three.js's `Mesh.raycast` + `_computeIntersections`.
    fn cast(&self, m: MeshCast, hits: &mut Hits) {
        if m.sphere.is_empty() {
            return;
        }
        let far = hits.limit(self.far);
        // Bounding sphere in world space, measured from the near plane: a
        // sphere the ray enters only beyond `far` holds nothing in range.
        //
        // Grown by a few f32 ulps first. The sphere is exact in three.js's
        // f64 and only nearly so here: a vertex on the sphere — every corner
        // of a symmetric shape — sits a rounding error outside it, and a ray
        // through that vertex would be culled before its triangle is tested.
        // This is a cull, so erring outward costs nothing but a test.
        let mut sphere = m.sphere.apply_matrix4(&m.matrix_world);
        sphere.radius += (sphere.radius + (sphere.center - self.ray.origin).length()) * 1e-6;
        let from_near = self.ray.recast(self.near);
        if !sphere.contains_point(from_near.origin) {
            let Some(t) = from_near.intersect_sphere(&sphere) else {
                return;
            };
            if t > far - self.near {
                return;
            }
        }

        let inverse = m.matrix_world.invert();
        let local = self.ray.apply_matrix4(&inverse);
        if !local.intersects_box(&m.bounds) {
            return;
        }

        // The BVH answers "which triangles", with one side rule for all of
        // them; it serves when every span starts on a triangle boundary (so
        // its triangle numbers are the spans' triangles) and every span's
        // material agrees about the side.
        #[cfg(feature = "mesh-bvh")]
        if m.use_bvh {
            let side = m.draws[0].material.side();
            let uniform = m
                .draws
                .iter()
                .all(|d| d.start % 3 == 0 && d.material.side() == side);
            if let (Some(bvh), true) = (&m.geometry.bounds_tree, uniform) {
                self.cast_bvh(&m, bvh, &local, side, hits);
                return;
            }
        }
        #[cfg(not(feature = "mesh-bvh"))]
        let _ = m.use_bvh;

        // Local ray parameter ↔ world distance: a local step of `t` along the
        // (unit) local direction is a world step of `t · scale` along the
        // world ray, exactly, for any affine matrix — non-uniform scale and
        // shear included. The range is widened by a hair so rounding can
        // never drop a hit; the world-space check in `mesh_hit` is the one
        // that decides.
        let scale = linear(&m.matrix_world, local.direction).length();
        let (mut t_min, mut t_max) = local_range(self.near, far, scale);

        let index = m.geometry.index.as_deref();
        let vertex = |v: usize| -> Option<Vector3> {
            let p = m.positions.get(v * 3..v * 3 + 3)?;
            Some(Vector3::new(p[0], p[1], p[2]))
        };
        // three.js's loops: `for (j = start; j < end; j += 3)` over each span,
        // reading `j`, `j + 1`, `j + 2` — the last triangle of a span whose
        // length is not a multiple of three reads past its end, as there.
        for draw in m.draws {
            let side = draw.material.side();
            let mut j = draw.start as usize;
            while j < draw.end as usize {
                let face = j / 3;
                let corners = match index {
                    Some(idx) => match idx.get(j..j + 3) {
                        Some(t) => [t[0] as usize, t[1] as usize, t[2] as usize],
                        None => break,
                    },
                    None => [j, j + 1, j + 2],
                };
                j += 3;
                let (Some(pa), Some(pb), Some(pc)) =
                    (vertex(corners[0]), vertex(corners[1]), vertex(corners[2]))
                else {
                    continue;
                };
                let Some((t, bary)) = intersect_sided(&local, pa, pb, pc, side) else {
                    continue;
                };
                if t < t_min || t > t_max {
                    continue;
                }
                if let Some(mut hit) =
                    self.mesh_hit(&m, &local, face, corners, [pa, pb, pc], bary, hits)
                {
                    if let Some(f) = hit.face.as_mut() {
                        f.material_index = draw.material_index as usize;
                    }
                    hits.push(hit);
                    if hits.first_only {
                        (t_min, t_max) = local_range(self.near, hits.limit(self.far), scale);
                    }
                }
            }
        }
    }

    /// The mesh test through the geometry's BVH (three-mesh-bvh's
    /// `acceleratedRaycast`), with the same side rule and the same hit record
    /// as the brute-force walk.
    #[cfg(feature = "mesh-bvh")]
    fn cast_bvh(
        &self,
        m: &MeshCast,
        bvh: &crate::mesh_bvh::MeshBvh,
        local: &Ray,
        side: u32,
        hits: &mut Hits,
    ) {
        let far = hits.limit(self.far);
        let scale = linear(&m.matrix_world, local.direction).length();
        let (t_min, t_max) = local_range(self.near, far, scale);
        let cull = side == FRONT_SIDE;
        let record = |h: &crate::mesh_bvh::BvhHit, hits: &mut Hits| -> bool {
            let Some(corners) = triangle_corners(m.geometry, h.face_index, m.positions.len() / 3)
            else {
                return false;
            };
            let [pa, pb, pc] = corners.map(|v| {
                Vector3::new(
                    m.positions[v * 3],
                    m.positions[v * 3 + 1],
                    m.positions[v * 3 + 2],
                )
            });
            if side == BACK_SIDE && Triangle::new(pa, pb, pc).is_front_facing(local.direction) {
                return false;
            }
            let (u, v) = (h.uv.x, h.uv.y);
            let bary = Vector3::new(1.0 - u - v, u, v);
            // One hit per span that contains the triangle — overlapping
            // groups report it once each, as three.js's group loop does.
            let j = (h.face_index * 3) as u32;
            let mut pushed = false;
            for d in m.draws.iter().filter(|d| d.start <= j && j < d.end) {
                if let Some(mut hit) =
                    self.mesh_hit(m, local, h.face_index, corners, [pa, pb, pc], bary, hits)
                {
                    if let Some(f) = hit.face.as_mut() {
                        f.material_index = d.material_index as usize;
                    }
                    hits.push(hit);
                    pushed = true;
                }
            }
            pushed
        };
        if hits.first_only && side != BACK_SIDE {
            match bvh.raycast_first(local, t_min, t_max, cull) {
                None => return,
                Some(h) => {
                    if record(&h, hits) {
                        return;
                    }
                    // The closest hit fell in the rounding margin of the
                    // range; look at the rest.
                }
            }
        }
        for h in bvh.raycast(local, t_min, t_max, cull) {
            let pushed = record(&h, hits);
            if pushed && hits.first_only {
                return;
            }
        }
    }

    /// A triangle hit as three.js's `checkGeometryIntersection` reports it,
    /// or `None` outside `[near, far]`.
    #[allow(clippy::too_many_arguments)]
    fn mesh_hit(
        &self,
        m: &MeshCast,
        local: &Ray,
        face: usize,
        corners: [usize; 3],
        p: [Vector3; 3],
        bary: Vector3,
        hits: &Hits,
    ) -> Option<Intersection> {
        let local_point = p[0] * bary.x + p[1] * bary.y + p[2] * bary.z;
        let point = local_point.apply_matrix4(&m.matrix_world);
        let distance = (point - self.ray.origin).length();
        if distance < self.near || distance > hits.limit(self.far) {
            return None;
        }
        let g = m.geometry;
        let normal = interpolate3(g.get_attribute("normal"), corners, bary).map(|n| {
            if n.dot(local.direction) > 0.0 {
                n * -1.0
            } else {
                n
            }
        });
        Some(Intersection {
            face_index: face,
            polygon_index: m.derived.polygon_of(face),
            face: Some(Face {
                a: corners[0],
                b: corners[1],
                c: corners[2],
                normal: Triangle::new(p[0], p[1], p[2]).normal(),
                material_index: 0,
            }),
            normal,
            uv: interpolate2(g.get_attribute("uv"), corners, bary),
            uv1: interpolate2(g.get_attribute("uv1"), corners, bary),
            barycoord: Some(bary),
            instance_id: m.instance_id,
            ..Intersection::new(distance, point, m.id)
        })
    }

    // ------------------------------------------------------------------
    // Lines, points, sprites
    // ------------------------------------------------------------------

    /// three.js `Line.raycast`: for `LineSegments` (step 2) the segments,
    /// for a `Line` / `LineLoop` ([`LineSegments::strip`], step 1) the strip's
    /// consecutive vertices, a loop's closing segment last — those within
    /// `params.line.threshold` of the ray, over the geometry's draw range.
    /// The threshold is world space, carried into local space by the
    /// object's own average scale. A hit's `index` is the segment's first
    /// vertex (of the strip, for a strip).
    fn raycast_line_segments(
        &self,
        id: ObjectId,
        obj: &Object3D,
        ls: &LineSegments,
        hits: &mut Hits,
    ) {
        let (geometry, step, closed) = match &ls.strip {
            Some(strip) => (&*strip.geometry, 1, strip.closed),
            None => (&*ls.geometry, 2, false),
        };
        let Some(positions) = position_array(geometry) else {
            return;
        };
        let threshold = self.params.line.threshold;
        let Some((local, local_threshold_sq)) =
            self.threshold_setup(geometry, obj, threshold, hits)
        else {
            return;
        };
        let vertex = |v: usize| -> Option<Vector3> {
            let p = positions.get(v * 3..v * 3 + 3)?;
            Some(Vector3::new(p[0], p[1], p[2]))
        };
        let count = match &geometry.index {
            Some(idx) => idx.len(),
            None => positions.len() / 3,
        };
        let at = |i: usize| match &geometry.index {
            Some(idx) => idx[i] as usize,
            None => i,
        };
        let start = (geometry.draw_range.start as usize).min(count);
        let end = (geometry.draw_range.end() as usize).min(count);
        let check = |a: usize, b: usize, segment: usize, hits: &mut Hits| {
            let (Some(va), Some(vb)) = (vertex(a), vertex(b)) else {
                return;
            };
            let (dist_sq, on_ray, on_segment) = local.distance_sq_to_segment(va, vb);
            if dist_sq > local_threshold_sq {
                return;
            }
            let distance = (on_ray.apply_matrix4(&obj.matrix_world) - self.ray.origin).length();
            if distance < self.near || distance > hits.limit(self.far) {
                return;
            }
            hits.push(Intersection {
                face_index: segment,
                index: Some(a),
                ..Intersection::new(distance, on_segment.apply_matrix4(&obj.matrix_world), id)
            });
        };
        let mut i = start;
        while i + 1 < end {
            check(at(i), at(i + 1), (i - start) / step, hits);
            i += step;
        }
        if closed && end > start {
            check(at(end - 1), at(start), end - start - 1, hits);
        }
    }

    /// three.js `Points.raycast`: the points strictly within
    /// `params.points.threshold` of the ray, reported at the closest point on
    /// the ray.
    fn raycast_points(&self, id: ObjectId, obj: &Object3D, pts: &Points, hits: &mut Hits) {
        let geometry = &*pts.geometry;
        let Some(positions) = position_array(geometry) else {
            return;
        };
        let threshold = self.params.points.threshold;
        let Some((local, local_threshold_sq)) =
            self.threshold_setup(geometry, obj, threshold, hits)
        else {
            return;
        };
        let test = |v: usize, hits: &mut Hits| {
            let Some(p) = positions.get(v * 3..v * 3 + 3) else {
                return;
            };
            let p = Vector3::new(p[0], p[1], p[2]);
            let dist_sq = local.distance_sq_to_point(p);
            if dist_sq >= local_threshold_sq {
                return;
            }
            let point = local
                .closest_point_to_point(p)
                .apply_matrix4(&obj.matrix_world);
            let distance = (point - self.ray.origin).length();
            if distance < self.near || distance > hits.limit(self.far) {
                return;
            }
            hits.push(Intersection {
                face_index: v,
                index: Some(v),
                distance_to_ray: Some(dist_sq.sqrt()),
                ..Intersection::new(distance, point, id)
            });
        };
        // three.js reads the draw range of the index (or of the vertices).
        let count = match &geometry.index {
            Some(idx) => idx.len(),
            None => positions.len() / 3,
        };
        let start = (geometry.draw_range.start as usize).min(count);
        let end = (geometry.draw_range.end() as usize).min(count);
        match &geometry.index {
            Some(idx) => idx[start..end].iter().for_each(|&v| test(v as usize, hits)),
            None => (start..end).for_each(|v| test(v, hits)),
        }
    }

    /// The shared start of `Line.raycast` and `Points.raycast`: the bounding
    /// sphere grown by the threshold, then the ray and the squared threshold
    /// in local space.
    fn threshold_setup(
        &self,
        geometry: &BufferGeometry,
        obj: &Object3D,
        threshold: f32,
        hits: &Hits,
    ) -> Option<(Ray, f32)> {
        let mut sphere = geometry.bounding_sphere_cached();
        if sphere.is_empty() {
            return None;
        }
        sphere = sphere.apply_matrix4(&obj.matrix_world);
        sphere.radius += threshold;
        if !self.ray.intersects_sphere(&sphere) {
            return None;
        }
        // Not three.js, but safe: nothing in the grown sphere is nearer than
        // where the ray enters it.
        if let Some(t) = self.ray.intersect_sphere(&sphere) {
            if !sphere.contains_point(self.ray.origin) && t > hits.limit(self.far) {
                return None;
            }
        }
        let local = self.ray.apply_matrix4(&obj.matrix_world.invert());
        let s = obj.scale;
        let local_threshold = threshold / ((s.x + s.y + s.z) / 3.0);
        Some((local, local_threshold * local_threshold))
    }

    /// three.js `Sprite.raycast`: the camera-facing quad, honouring the
    /// sprite's `center` and the material's `rotation` and
    /// `size_attenuation`. Needs [`camera`](Self::camera). The renderers draw
    /// the same quad (three.js's `sprite_vert`).
    fn raycast_sprite(&self, id: ObjectId, obj: &Object3D, sprite: &Sprite, hits: &mut Hits) {
        let Some(camera) = &self.camera else {
            return;
        };
        let e = &obj.matrix_world.elements;
        let mut world_scale = Vector3::new(
            Vector3::new(e[0], e[1], e[2]).length(),
            Vector3::new(e[4], e[5], e[6]).length(),
            Vector3::new(e[8], e[9], e[10]).length(),
        );
        let model_view = camera.matrix_world_inverse.multiply(&obj.matrix_world);
        let mv = &model_view.elements;
        let mv_position = Vector3::new(mv[12], mv[13], mv[14]);
        let (rotation, size_attenuation) = match &*sprite.material {
            Material::Sprite(m) => (m.rotation, m.size_attenuation),
            _ => (0.0, true),
        };
        if camera.perspective && !size_attenuation {
            world_scale = world_scale * -mv_position.z;
        }
        let (sin, cos) = rotation.sin_cos();
        let center = sprite.center;
        let corner = |x: f32, y: f32| -> Vector3 {
            let ax = (x - center.x + 0.5) * world_scale.x;
            let ay = (y - center.y + 0.5) * world_scale.y;
            let (rx, ry) = if rotation != 0.0 {
                (cos * ax - sin * ay, sin * ax + cos * ay)
            } else {
                (ax, ay)
            };
            Vector3::new(mv_position.x + rx, mv_position.y + ry, mv_position.z)
                .apply_matrix4(&camera.matrix_world)
        };
        let va = corner(-0.5, -0.5);
        let mut vb = corner(0.5, -0.5);
        let vc = corner(0.5, 0.5);
        let uv_a = Vector2::new(0.0, 0.0);
        let mut uv_b = Vector2::new(1.0, 0.0);
        let uv_c = Vector2::new(1.0, 1.0);
        let t = match intersect_triangle(&self.ray, va, vb, vc, false) {
            Some((t, _, _)) => t,
            None => {
                vb = corner(-0.5, 0.5);
                uv_b = Vector2::new(0.0, 1.0);
                match intersect_triangle(&self.ray, va, vc, vb, false) {
                    Some((t, _, _)) => t,
                    None => return,
                }
            }
        };
        let point = self.ray.at(t);
        let distance = (point - self.ray.origin).length();
        if distance < self.near || distance > hits.limit(self.far) {
            return;
        }
        let w = Triangle::new(va, vb, vc).barycoord(point);
        let uv = Vector2::new(
            uv_a.x * w.x + uv_b.x * w.y + uv_c.x * w.z,
            uv_a.y * w.x + uv_b.y * w.y + uv_c.y * w.z,
        );
        hits.push(Intersection {
            uv: Some(uv),
            ..Intersection::new(distance, point, id)
        });
    }

    /// Convenience: intersect a single AABB in world space (no scene walk).
    pub fn intersect_box(&self, b: &Box3) -> Option<f32> {
        self.ray
            .intersect_box(b)
            .filter(|t| *t >= self.near && *t <= self.far)
    }
}

/// The `position` array, when it is xyz.
fn position_array(g: &BufferGeometry) -> Option<&[f32]> {
    g.get_attribute("position")
        .filter(|a| a.item_size == 3)
        .map(|a| a.array.as_slice())
}

/// Vertex indices of triangle `face`, checked against the vertex count.
#[cfg(feature = "mesh-bvh")]
fn triangle_corners(g: &BufferGeometry, face: usize, vertices: usize) -> Option<[usize; 3]> {
    let corners = match &g.index {
        Some(idx) => {
            let t = idx.get(face * 3..face * 3 + 3)?;
            [t[0] as usize, t[1] as usize, t[2] as usize]
        }
        None => [face * 3, face * 3 + 1, face * 3 + 2],
    };
    corners.iter().all(|&v| v < vertices).then_some(corners)
}

/// `m`'s linear part applied to `v` — a direction, no translation.
fn linear(m: &Matrix4, v: Vector3) -> Vector3 {
    let e = &m.elements;
    Vector3::new(
        e[0] * v.x + e[4] * v.y + e[8] * v.z,
        e[1] * v.x + e[5] * v.y + e[9] * v.z,
        e[2] * v.x + e[6] * v.y + e[10] * v.z,
    )
}

/// The local ray parameters that can map to world distances in
/// `[near, far]`, given that one local unit is `scale` world units, widened
/// slightly for rounding.
fn local_range(near: f32, far: f32, scale: f32) -> (f32, f32) {
    if !(scale.is_finite() && scale > 0.0) {
        return (0.0, f32::INFINITY);
    }
    const SLACK: f32 = 1e-4;
    let t_min = (near / scale) * (1.0 - SLACK);
    let t_max = if far.is_finite() {
        (far / scale) * (1.0 + SLACK)
    } else {
        f32::INFINITY
    };
    (t_min, t_max)
}

/// three.js `Ray.intersectTriangle`: the ray parameter and the weights of `b`
/// and `c` — see [`Ray::intersect_triangle_weights`].
fn intersect_triangle(
    ray: &Ray,
    a: Vector3,
    b: Vector3,
    c: Vector3,
    cull: bool,
) -> Option<(f32, f32, f32)> {
    ray.intersect_triangle_weights(a, b, c, cull)
}

/// three.js `checkIntersection`'s side rule: `FrontSide` tests `(a, b, c)`
/// culling back faces, `BackSide` tests `(c, b, a)` culling — so only back
/// faces hit — and `DoubleSide` tests `(a, b, c)` unculled. Returns the ray
/// parameter and the barycentric weights of `a`, `b`, `c`.
fn intersect_sided(
    ray: &Ray,
    a: Vector3,
    b: Vector3,
    c: Vector3,
    side: u32,
) -> Option<(f32, Vector3)> {
    if side == BACK_SIDE {
        let (t, wb, wa) = intersect_triangle(ray, c, b, a, true)?;
        Some((t, Vector3::new(wa, wb, 1.0 - wa - wb)))
    } else {
        let (t, wb, wc) = intersect_triangle(ray, a, b, c, side == FRONT_SIDE)?;
        Some((t, Vector3::new(1.0 - wb - wc, wb, wc)))
    }
}

/// Barycentric interpolation of a 2-component (or wider: the first two are
/// read, as three.js's `fromBufferAttribute` does) attribute.
fn interpolate2(
    attr: Option<&super::BufferAttribute>,
    [a, b, c]: [usize; 3],
    w: Vector3,
) -> Option<Vector2> {
    let attr = attr.filter(|a| a.item_size >= 2)?;
    let n = attr.item_size;
    let read = |v: usize| attr.array.get(v * n..v * n + 2);
    let (pa, pb, pc) = (read(a)?, read(b)?, read(c)?);
    Some(Vector2::new(
        pa[0] * w.x + pb[0] * w.y + pc[0] * w.z,
        pa[1] * w.x + pb[1] * w.y + pc[1] * w.z,
    ))
}

/// Barycentric interpolation of a 3-component attribute.
fn interpolate3(
    attr: Option<&super::BufferAttribute>,
    [a, b, c]: [usize; 3],
    w: Vector3,
) -> Option<Vector3> {
    let attr = attr.filter(|a| a.item_size >= 3)?;
    let n = attr.item_size;
    let read = |v: usize| {
        attr.array
            .get(v * n..v * n + 3)
            .map(|p| Vector3::new(p[0], p[1], p[2]))
    };
    Some(read(a)? * w.x + read(b)? * w.y + read(c)? * w.z)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{BufferAttribute, BufferGeometry};
    use crate::materials::{BasicMaterial, Material};
    use crate::math::Color;

    fn unit_triangle_mesh() -> Mesh {
        let mut g = BufferGeometry::new();
        g.set_attribute(
            "position",
            BufferAttribute::new(vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0], 3),
        );
        g.compute_bounding_sphere();
        let mat = Material::Basic(BasicMaterial::new(Color::from_hex(0xffffff)));
        Mesh::new(g, mat)
    }

    #[test]
    fn ray_hits_triangle_through_origin() {
        let mut arena = ObjectArena::new();
        let root = arena.insert(Object3D::group());
        let mesh_id = arena.insert(Object3D::mesh(unit_triangle_mesh()));
        arena.add_child(root, mesh_id);
        arena.update_world_matrices(root, crate::math::Matrix4::identity());

        // From the triangle's front (+z): the material is FrontSide, so the
        // same ray from behind is culled, as in three.js.
        let rc = Raycaster {
            ray: Ray::new(Vector3::new(0.25, 0.25, 1.0), Vector3::new(0.0, 0.0, -1.0)),
            ..Raycaster::default()
        };
        let hits = rc.intersect_objects(&arena, root, true);
        assert_eq!(hits.len(), 1, "expected one hit, got {:?}", hits);
        assert!((hits[0].distance - 1.0).abs() < 1e-4);
        assert_eq!(hits[0].object, mesh_id);
    }

    #[test]
    fn ray_misses_when_layer_mismatched() {
        let mut arena = ObjectArena::new();
        let root = arena.insert(Object3D::group());
        let mut mesh_obj = Object3D::mesh(unit_triangle_mesh());
        mesh_obj.layers.set(5);
        let mesh_id = arena.insert(mesh_obj);
        arena.add_child(root, mesh_id);
        arena.update_world_matrices(root, crate::math::Matrix4::identity());

        let mut rc = Raycaster::default();
        rc.layers.set(0); // explicit channel 0 only
                          // From the front, so only the layer can be what misses.
        rc.ray = Ray::new(Vector3::new(0.25, 0.25, 1.0), Vector3::new(0.0, 0.0, -1.0));
        let hits = rc.intersect_objects(&arena, root, true);
        assert!(hits.is_empty(), "expected no hits, got {:?}", hits);
        rc.layers.enable(5);
        assert_eq!(rc.intersect_objects(&arena, root, true).len(), 1);
    }

    /// A 3×1 strip of unit quads in z = 0, centred, two triangles each.
    fn quad_strip(grouped: bool) -> BufferGeometry {
        let mut g = crate::geometries::PlaneGeometry::with_segments(3.0, 1.0, 3, 1);
        if !grouped {
            g.polygons = None;
        }
        g.compute_bounding_sphere();
        g
    }

    fn hits_on(g: BufferGeometry, x: f32, y: f32) -> Vec<Intersection> {
        let mut arena = ObjectArena::new();
        let root = arena.insert(Object3D::group());
        let mat = Material::Basic(BasicMaterial::new(Color::from_hex(0xffffff)));
        let mesh_id = arena.insert(Object3D::mesh(Mesh::new(g, mat)));
        arena.add_child(root, mesh_id);
        arena.update_world_matrices(root, crate::math::Matrix4::identity());
        let rc = Raycaster {
            ray: Ray::new(Vector3::new(x, y, 1.0), Vector3::new(0.0, 0.0, -1.0)),
            ..Raycaster::default()
        };
        rc.intersect_objects(&arena, root, true)
    }

    /// Both triangles of a quad report the quad, and the quads count in
    /// order along the strip.
    #[test]
    fn a_hit_reports_its_polygon() {
        // PlaneGeometry is centred: x in [-1.5, 1.5], y in [-0.5, 0.5].
        for (x, quad) in [(-1.2, 0), (-0.3, 1), (0.3, 1), (1.2, 2)] {
            for y in [-0.4, 0.4] {
                let hits = hits_on(quad_strip(true), x, y);
                assert_eq!(hits.len(), 1, "{x},{y}: {hits:?}");
                assert_eq!(hits[0].polygon_index, Some(quad), "{x},{y}");
                assert_eq!(hits[0].face_index / 2, quad, "{x},{y}: triangles pair up");
            }
        }
        // The two halves of one quad are different triangles, one polygon.
        let a = hits_on(quad_strip(true), -0.45, 0.45)[0];
        let b = hits_on(quad_strip(true), -0.05, -0.45)[0];
        assert_ne!(a.face_index, b.face_index);
        assert_eq!(a.polygon_index, b.polygon_index);
    }

    #[test]
    fn no_grouping_no_polygon() {
        let hits = hits_on(quad_strip(false), 0.3, 0.2);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].polygon_index, None);
    }

    /// The BVH path reports the same polygon as the brute-force walk.
    #[cfg(feature = "mesh-bvh")]
    #[test]
    fn the_bvh_path_reports_the_polygon_too() {
        let sphere = || {
            let mut g = crate::geometries::SphereGeometry::new(1.0, 16, 12);
            g.compute_bounding_sphere();
            g
        };
        for (x, y) in [(0.1, 0.2), (-0.5, 0.3), (0.0, 0.97), (0.6, -0.6)] {
            let plain = hits_on(sphere(), x, y);
            let mut g = sphere();
            g.compute_bounds_tree(Default::default());
            let fast = hits_on(g, x, y);
            assert!(!plain.is_empty());
            assert_eq!(plain.len(), fast.len());
            for (p, f) in plain.iter().zip(&fast) {
                assert_eq!(p.face_index, f.face_index);
                assert!(p.polygon_index.is_some());
                assert_eq!(p.polygon_index, f.polygon_index);
            }
        }
    }

    /// Orthographic picking starts in the camera's own plane, as three.js's
    /// `setFromCamera` does, so distances measure from the camera.
    ///
    /// The depth range matters here: this crate's projections map view depth
    /// to `[0, 1]`, not OpenGL's `[-1, 1]`, so three.js's NDC
    /// `(near + far) / (near - far)` is not the camera plane here. Using it
    /// (or OpenGL's near plane, `-1`) would put the origin a long way behind
    /// the camera and every hit beyond `far`.
    #[test]
    fn ortho_camera_ray_starts_in_the_camera_plane() {
        use crate::cameras::OrthographicCamera;

        let mut arena = ObjectArena::new();
        let root = arena.insert(Object3D::group());
        let mut mesh_obj = Object3D::mesh(unit_triangle_mesh());
        // Straddle the origin so the camera axis passes through the triangle.
        mesh_obj.position = Vector3::new(-0.25, -0.25, 0.0);
        let mesh_id = arena.insert(mesh_obj);
        arena.add_child(root, mesh_id);
        arena.update_world_matrices(root, crate::math::Matrix4::identity());

        let mut camera = OrthographicCamera::new(-2.0, 2.0, 2.0, -2.0, 0.1, 100.0);
        camera.position = Vector3::new(0.0, 0.0, 5.0);
        camera.target = Vector3::ZERO;

        let mut rc = Raycaster {
            near: 0.1,
            far: 100.0,
            ..Default::default()
        };
        rc.set_from_camera_ortho(Vector2::new(0.0, 0.0), &camera);

        // three.js r165: origin (0, 0, 5), the camera's position.
        assert!(
            (rc.ray.origin - Vector3::new(0.0, 0.0, 5.0)).length() < 1e-4,
            "ray should start in the camera plane, got {:?}",
            rc.ray.origin
        );
        assert!((rc.ray.direction - Vector3::new(0.0, 0.0, -1.0)).length() < 1e-6);
        let hits = rc.intersect_objects(&arena, root, true);
        assert_eq!(hits.len(), 1, "expected the triangle, got {:?}", hits);
        assert_eq!(hits[0].object, mesh_id);
        assert!(
            (hits[0].distance - 5.0).abs() < 1e-4,
            "distance from the camera plane, got {}",
            hits[0].distance
        );

        // The review's repro: PlaneGeometry(1, 1) at the origin. three.js
        // reports two hits at distance 5 (the ray runs down the diagonal).
        let (arena, root, _) = scene_of(Object3D::mesh(Mesh::new(
            crate::geometries::PlaneGeometry::new(1.0, 1.0),
            basic(0),
        )));
        let hits = rc.intersect_objects(&arena, root, true);
        assert_eq!(hits.len(), 2, "{hits:?}");
        assert!(
            hits.iter().all(|h| close(h.distance, 5.0, 1e-4)),
            "{hits:?}"
        );

        // Off-axis, and a camera that is not on an axis: the origin is the
        // NDC point in the plane through the camera, perpendicular to its view.
        let mut cam = OrthographicCamera::new(-3.0, 1.0, 2.0, -1.0, 0.5, 50.0);
        cam.position = Vector3::new(4.0, 3.0, 2.0);
        cam.target = Vector3::new(0.5, -1.0, 0.0);
        rc.set_from_camera_ortho(Vector2::new(0.5, -0.25), &cam);
        let forward = (cam.target - cam.position).normalize();
        assert!((rc.ray.direction - forward).length() < 1e-5);
        assert!(
            (rc.ray.origin - cam.position).dot(forward).abs() < 1e-4,
            "origin {:?} is off the camera plane",
            rc.ray.origin
        );
        // NDC (0.5, -0.25) of [-3, 1] × [-1, 2]: (0, 0.125) in the camera frame.
        let view = cam.view_matrix();
        let in_view = rc.ray.origin.apply_matrix4(&view);
        assert!(
            close(in_view.x, 0.0, 1e-4) && close(in_view.y, 0.125, 1e-4),
            "{in_view:?}"
        );
    }

    /// three.js's `setFromCamera` sets the ray and the camera, and nothing
    /// else: the raycaster's own layer mask stays the caller's.
    #[test]
    fn set_from_camera_leaves_the_layers_alone() {
        use crate::cameras::{OrthographicCamera, PerspectiveCamera};
        let mut persp = PerspectiveCamera::new(60.0, 1.0, 0.1, 100.0);
        persp.layers.set(4);
        let ortho = OrthographicCamera::new(-1.0, 1.0, 1.0, -1.0, 0.1, 10.0);
        let mut rc = Raycaster::default();
        rc.layers.set(3);
        let mask = rc.layers.mask;
        rc.set_from_camera(Vector2::new(0.1, 0.2), &persp);
        assert_eq!(rc.layers.mask, mask);
        rc.set_from_camera_perspective(Vector2::new(0.1, 0.2), &persp);
        assert_eq!(rc.layers.mask, mask);
        rc.set_from_camera(Vector2::new(0.1, 0.2), &ortho);
        assert_eq!(rc.layers.mask, mask);
        rc.set_from_camera_ortho(Vector2::new(0.1, 0.2), &ortho);
        assert_eq!(rc.layers.mask, mask);
    }

    /// A ray running along a face of a flat geometry's (zero-thickness)
    /// bounding box: the review's repro, against three.js r165.
    ///
    /// `PlaneGeometry(2, 2)`: three.js reports one hit at distance 5 on face 1
    /// for x = 1, face 0 for x = -1 and y = 1, and both triangles at the
    /// corner. The unit triangle at (0, 0.25) — on its x = 0 edge — is hit at
    /// distance 1.
    #[test]
    fn grazing_rays_on_the_bounding_box_face_hit() {
        let plane = || crate::geometries::PlaneGeometry::new(2.0, 2.0);
        let cases: [([f32; 3], &[usize]); 4] = [
            ([1.0, 0.3, 5.0], &[1]),
            ([-1.0, 0.3, 5.0], &[0]),
            ([0.3, 1.0, 5.0], &[0]),
            ([1.0, 1.0, 5.0], &[0, 1]),
        ];
        let check = |g: BufferGeometry, label: &str| {
            let (arena, root, _) = scene_of(Object3D::mesh(Mesh::new(g, basic(0))));
            for (origin, faces) in cases {
                let rc = ray(origin, [0.0, 0.0, -1.0]);
                let hits = rc.intersect_objects(&arena, root, true);
                let mut got: Vec<usize> = hits.iter().map(|h| h.face_index).collect();
                got.sort();
                assert_eq!(got, faces, "{label} {origin:?}: {hits:?}");
                assert!(hits.iter().all(|h| close(h.distance, 5.0, 1e-5)));
                let first = Raycaster {
                    first_hit_only: true,
                    ..rc
                }
                .intersect_objects(&arena, root, true);
                assert_eq!(first.len(), 1, "{label} first-hit {origin:?}");
                assert!(close(first[0].distance, 5.0, 1e-5));
            }
        };
        check(plane(), "plane");
        #[cfg(feature = "mesh-bvh")]
        {
            let mut g = plane();
            g.compute_bounds_tree(Default::default());
            check(g, "plane + bvh");
        }
        let (arena, root, _) = scene_of(Object3D::mesh(unit_triangle_mesh()));
        let hits = ray([0.0, 0.25, 1.0], [0.0, 0.0, -1.0]).intersect_objects(&arena, root, true);
        assert_eq!(hits.len(), 1, "{hits:?}");
        assert!(close(hits[0].distance, 1.0, 1e-6));
        assert!(close(hits[0].point.y, 0.25, 1e-6));
    }

    /// three.js reports a skinned hit's face normal — and flips the
    /// interpolated normal — in the mesh's local space, after
    /// `bindMatrixInverse`, not in world space.
    ///
    /// Identity bones, the mesh turned a quarter about y: the triangle stays
    /// where it is in the world (the bones place it, not the mesh), and its
    /// local normal is the world +z seen from the turned frame, (-1, 0, 0).
    /// three.js r165: distance 10, face.normal (-1, 0, 0), normal (0, 0, 1).
    #[test]
    fn skinned_hits_report_local_space_normals() {
        let mut g = BufferGeometry::new();
        g.set_attribute(
            "position",
            BufferAttribute::new(vec![-1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 0.0, 1.0, 0.0], 3),
        );
        g.set_attribute("normal", BufferAttribute::new([0.0, 0.0, 1.0].repeat(3), 3));
        let mut arena = ObjectArena::new();
        let root = arena.insert(Object3D::group());
        let bone = arena.insert(Object3D::group());
        arena.add_child(root, bone);
        let skeleton = crate::core::Skeleton::new(vec![crate::core::Bone {
            node: bone,
            inverse_bind: Matrix4::identity(),
        }]);
        let mut obj = Object3D::skinned_mesh(SkinnedMesh::from_unweighted(g, basic(2), skeleton));
        obj.rotate_y(std::f32::consts::FRAC_PI_2);
        let mesh = arena.insert(obj);
        arena.add_child(root, mesh);
        arena.update_world_matrices(root, Matrix4::identity());
        let hits = ray([0.2, 0.0, 10.0], [0.0, 0.0, -1.0]).intersect_objects(&arena, root, true);
        assert_eq!(hits.len(), 1, "{hits:?}");
        let h = hits[0];
        assert!(close(h.distance, 10.0, 1e-5));
        assert!(
            (h.point - Vector3::new(0.2, 0.0, 0.0)).length() < 1e-5,
            "{:?}",
            h.point
        );
        let n = h.face.unwrap().normal;
        assert!(
            (n - Vector3::new(-1.0, 0.0, 0.0)).length() < 1e-5,
            "face normal {n:?}"
        );
        let nn = h.normal.unwrap();
        assert!(
            (nn - Vector3::new(0.0, 0.0, 1.0)).length() < 1e-5,
            "normal {nn:?}"
        );
    }

    // ------------------------------------------------------------------
    // three.js behaviours, against geometric truth
    // ------------------------------------------------------------------

    use crate::core::{InstancedMesh, LineSegments, Points, SkinnedMesh, Sprite};
    use crate::math::Matrix4;

    fn basic(side: u32) -> Material {
        let mut m = BasicMaterial::new(Color::from_hex(0xffffff));
        m.side = side;
        Material::Basic(m)
    }

    /// One object under a root, world matrices updated.
    fn scene_of(obj: Object3D) -> (ObjectArena, ObjectId, ObjectId) {
        let mut arena = ObjectArena::new();
        let root = arena.insert(Object3D::group());
        let id = arena.insert(obj);
        arena.add_child(root, id);
        arena.update_world_matrices(root, Matrix4::identity());
        (arena, root, id)
    }

    fn ray(origin: [f32; 3], dir: [f32; 3]) -> Raycaster {
        Raycaster {
            ray: Ray::new(
                Vector3::new(origin[0], origin[1], origin[2]),
                Vector3::new(dir[0], dir[1], dir[2]).normalize(),
            ),
            ..Raycaster::default()
        }
    }

    fn close(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() <= tol
    }

    /// FrontSide hits only the face whose winding faces the ray, BackSide only
    /// the other, DoubleSide both — three.js's `checkIntersection`.
    #[test]
    fn material_side_decides_which_faces_hit() {
        // Counter-clockwise seen from +z: the front faces +z.
        for (side, from_front, from_back) in [(0, true, false), (1, false, true), (2, true, true)] {
            let mut g = BufferGeometry::new();
            g.set_attribute(
                "position",
                BufferAttribute::new(vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0], 3),
            );
            let (arena, root, _) = scene_of(Object3D::mesh(Mesh::new(g, basic(side))));
            let front =
                ray([0.25, 0.25, 2.0], [0.0, 0.0, -1.0]).intersect_objects(&arena, root, true);
            let back =
                ray([0.25, 0.25, -3.0], [0.0, 0.0, 1.0]).intersect_objects(&arena, root, true);
            assert_eq!(!front.is_empty(), from_front, "side {side} from the front");
            assert_eq!(!back.is_empty(), from_back, "side {side} from behind");
            for h in front.iter().chain(&back) {
                // Barycentrics of (0.25, 0.25) on a=(0,0) b=(1,0) c=(0,1).
                let w = h.barycoord.unwrap();
                assert!(close(w.x, 0.5, 1e-5) && close(w.y, 0.25, 1e-5) && close(w.z, 0.25, 1e-5));
                assert!(close(h.point.x, 0.25, 1e-5) && close(h.point.z, 0.0, 1e-5));
                // The face normal is geometric and never flipped.
                assert!(close(h.face.unwrap().normal.z, 1.0, 1e-5));
            }
            if let Some(h) = back.first() {
                assert!(close(h.distance, 3.0, 1e-5));
            }
        }
    }

    /// The uv at a hit is the plane's analytic uv there, and the normal is the
    /// attribute's, turned to face the ray.
    #[test]
    fn plane_uv_and_normal_at_the_hit() {
        let g = crate::geometries::PlaneGeometry::with_segments(2.0, 4.0, 3, 5);
        let (arena, root, id) = scene_of(Object3D::mesh(Mesh::new(g, basic(2))));
        // Not (0, 0): that is the middle cell's centre, on its diagonal, and
        // a ray through an edge two triangles share hits both — as three.js's
        // `Ray.intersectTriangle`, whose edge tests are inclusive, does too.
        for (x, y) in [(0.1, 0.2), (-0.93, 1.71), (0.66, -1.9), (0.05, 0.03)] {
            let hits = ray([x, y, 3.0], [0.0, 0.0, -1.0]).intersect_objects(&arena, root, true);
            assert_eq!(hits.len(), 1, "{x},{y}");
            let h = hits[0];
            assert_eq!(h.object, id);
            let uv = h.uv.unwrap();
            assert!(close(uv.x, x / 2.0 + 0.5, 1e-5), "u at {x},{y}: {uv:?}");
            assert!(close(uv.y, y / 4.0 + 0.5, 1e-5), "v at {x},{y}: {uv:?}");
            assert!(h.uv1.is_none());
            assert!(close(h.normal.unwrap().z, 1.0, 1e-5));
            let f = h.face.unwrap();
            assert_eq!(f.material_index, 0);
            // The face's corners are the hit triangle's.
            let tri = h.face_index;
            let idx = arena.get(id).map(|o| match &o.kind {
                ObjectKind::Mesh(m) => m.geometry.triangle(tri),
                _ => unreachable!(),
            });
            assert_eq!(idx, Some([f.a as u32, f.b as u32, f.c as u32]));
            // From below a DoubleSide plane, the normal turns to face the ray.
            let below = ray([x, y, -3.0], [0.0, 0.0, 1.0]).intersect_objects(&arena, root, true);
            assert!(close(below[0].normal.unwrap().z, -1.0, 1e-5));
            assert!(close(below[0].face.unwrap().normal.z, 1.0, 1e-5));
        }
    }

    /// A sphere hit lands where the analytic ray–sphere intersection does, to
    /// within the tessellation's sag.
    #[test]
    fn sphere_hits_match_the_analytic_sphere() {
        let (segments, rings) = (64, 32);
        let g = crate::geometries::SphereGeometry::new(1.0, segments, rings);
        // Largest gap between a chord and the arc it spans.
        let sag = 1.0 - (std::f32::consts::PI / rings as f32).cos();
        let (arena, root, _) = scene_of(Object3D::mesh(Mesh::new(g, basic(0))));
        // Off the vertices: a ray through one hits every triangle sharing it.
        for (x, y) in [(0.013, 0.021), (0.3, -0.2), (-0.7, 0.5), (0.1, 0.9)] {
            let hits = ray([x, y, 5.0], [0.0, 0.0, -1.0]).intersect_objects(&arena, root, true);
            // FrontSide: the near face only.
            assert_eq!(hits.len(), 1, "{x},{y}: {hits:?}");
            let z = (1.0f32 - x * x - y * y).sqrt();
            assert!(
                close(hits[0].distance, 5.0 - z, sag * 2.0),
                "{x},{y}: {}",
                hits[0].distance
            );
            let n = hits[0].normal.unwrap().normalize();
            assert!(n.dot(Vector3::new(x, y, z)) > 0.99, "{x},{y}: normal {n:?}");
        }
    }

    /// `far` is a world distance however the object is scaled. Under a
    /// non-uniform scale the old averaged-scale reject dropped hits inside it.
    #[test]
    fn far_is_exact_under_non_uniform_scale() {
        // A triangle facing −x at local x = 50, scaled by 0.1 in x: world x = 5.
        let mut g = BufferGeometry::new();
        g.set_attribute(
            "position",
            BufferAttribute::new(vec![50.0, -1.0, -1.0, 50.0, 1.0, -1.0, 50.0, 0.0, 1.0], 3),
        );
        let mut obj = Object3D::mesh(Mesh::new(g, basic(2)));
        obj.scale = Vector3::new(0.1, 1.0, 1.0);
        let (arena, root, _) = scene_of(obj);
        for (far, hit) in [(5.001, true), (4.999, false), (f32::INFINITY, true)] {
            let mut rc = ray([0.0, 0.0, 0.0], [1.0, 0.0, 0.0]);
            rc.far = far;
            let hits = rc.intersect_objects(&arena, root, true);
            assert_eq!(!hits.is_empty(), hit, "far {far}: {hits:?}");
            if hit {
                assert!(close(hits[0].distance, 5.0, 1e-4));
            }
        }
        for (near, hit) in [(4.999, true), (5.001, false)] {
            let mut rc = ray([0.0, 0.0, 0.0], [1.0, 0.0, 0.0]);
            rc.near = near;
            assert_eq!(
                !rc.intersect_objects(&arena, root, true).is_empty(),
                hit,
                "near {near}"
            );
        }
        // A sheared, rotated, non-uniformly scaled frame, at an angle: the
        // distance still equals the world-space hit's.
        let mut obj = Object3D::mesh(Mesh::new(
            crate::geometries::PlaneGeometry::new(4.0, 4.0),
            basic(2),
        ));
        obj.scale = Vector3::new(3.0, 0.5, 7.0);
        obj.rotate_x(0.4).rotate_y(-0.3);
        obj.position = Vector3::new(0.5, 0.2, -1.0);
        let (arena, root, _) = scene_of(obj);
        let rc = ray([0.3, 0.1, 6.0], [0.05, -0.02, -1.0]);
        let h = rc.intersect_objects(&arena, root, true)[0];
        assert!(close(h.distance, (h.point - rc.ray.origin).length(), 1e-4));
        for (far, hit) in [(h.distance + 1e-3, true), (h.distance - 1e-3, false)] {
            let rc = Raycaster { far, ..rc };
            assert_eq!(!rc.intersect_objects(&arena, root, true).is_empty(), hit);
        }
    }

    #[test]
    fn line_threshold_is_world_space_by_the_objects_own_scale() {
        let line = |scale: f32| {
            let mut g = BufferGeometry::new();
            g.set_attribute(
                "position",
                BufferAttribute::new(
                    vec![-1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 5.0, 5.0, 0.0, 6.0, 5.0, 0.0],
                    3,
                ),
            );
            let mat = Material::Line(crate::materials::LineBasicMaterial::new(Color::from_hex(
                0xffffff,
            )));
            let mut obj = Object3D::line_segments(LineSegments::new(g, mat));
            obj.scale = Vector3::new(scale, scale, scale);
            scene_of(obj)
        };
        // 0.3 above the first segment's middle, looking down −z.
        let (arena, root, id) = line(1.0);
        for (threshold, hit) in [(0.31, true), (0.29, false)] {
            let mut rc = ray([0.2, 0.3, 4.0], [0.0, 0.0, -1.0]);
            rc.params.line.threshold = threshold;
            let hits = rc.intersect_objects(&arena, root, true);
            assert_eq!(hits.len(), hit as usize, "threshold {threshold}");
            if hit {
                let h = hits[0];
                assert_eq!(h.object, id);
                assert_eq!(h.index, Some(0));
                assert_eq!(h.face_index, 0);
                assert!(h.face.is_none());
                // The point is on the segment; the distance to the ray's
                // closest point.
                assert!(close(h.point.x, 0.2, 1e-5) && close(h.point.y, 0.0, 1e-5));
                assert!(close(h.distance, 4.0, 1e-5));
            }
        }
        // The second segment is segment 1, starting at vertex 2.
        let mut rc = ray([5.5, 5.1, 4.0], [0.0, 0.0, -1.0]);
        rc.params.line.threshold = 0.2;
        let hits = rc.intersect_objects(&arena, root, true);
        assert_eq!((hits[0].index, hits[0].face_index), (Some(2), 1));
        // Scaled by 2 the line sits at world y = 0, the ray 0.6 above it, and
        // the threshold divides by the object's scale: 0.61 / 2 local against
        // 0.3 local.
        let (arena, root, _) = line(2.0);
        for (threshold, hit) in [(0.61, true), (0.59, false)] {
            let mut rc = ray([0.4, 0.6, 4.0], [0.0, 0.0, -1.0]);
            rc.params.line.threshold = threshold;
            assert_eq!(rc.intersect_objects(&arena, root, true).len(), hit as usize);
        }
    }

    #[test]
    fn points_threshold_and_index() {
        let mut g = BufferGeometry::new();
        g.set_attribute(
            "position",
            BufferAttribute::new(vec![0.0, 0.0, 0.0, 1.0, 0.0, -2.0, 3.0, 3.0, 0.0], 3),
        );
        let mat = Material::Points(crate::materials::PointsMaterial::new(
            Color::from_hex(0xffffff),
            1.0,
        ));
        let (arena, root, id) = scene_of(Object3D::points(Points::new(g, mat)));
        let mut rc = ray([0.5, 0.1, 5.0], [0.0, 0.0, -1.0]);
        rc.params.points.threshold = 0.6;
        let hits = rc.intersect_objects(&arena, root, true);
        // Points 0 and 1 are both 0.51 from the ray; point 1 is 2 further.
        assert_eq!(hits.len(), 2, "{hits:?}");
        assert_eq!((hits[0].index, hits[1].index), (Some(0), Some(1)));
        assert_eq!(hits[0].object, id);
        let d = (0.5f32 * 0.5 + 0.1 * 0.1).sqrt();
        assert!(close(hits[0].distance_to_ray.unwrap(), d, 1e-5));
        assert!(close(hits[0].distance, 5.0, 1e-5) && close(hits[1].distance, 7.0, 1e-5));
        // The point reported is on the ray.
        assert!(close(hits[1].point.x, 0.5, 1e-5) && close(hits[1].point.z, -2.0, 1e-5));
        rc.params.points.threshold = 0.5;
        assert!(rc.intersect_objects(&arena, root, true).is_empty());
    }

    /// A sprite is the camera-facing quad of its world scale, offset by its
    /// `center` and turned by the material's `rotation`.
    #[test]
    fn sprite_quad_follows_center_and_rotation() {
        use crate::cameras::PerspectiveCamera;
        use crate::materials::SpriteMaterial;
        let camera = PerspectiveCamera::new(50.0, 1.0, 0.1, 100.0); // at z = 5
        let sprite = |center: Vector2, rotation: f32| {
            let mut s = Sprite::new(Material::Sprite(SpriteMaterial {
                rotation,
                ..SpriteMaterial::default()
            }));
            s.center = center;
            let mut obj = Object3D::sprite(s);
            obj.scale = Vector3::new(2.0, 1.0, 1.0);
            scene_of(obj)
        };
        let cast = |arena: &ObjectArena, root, x: f32, y: f32| {
            let mut rc = ray([x, y, 5.0], [0.0, 0.0, -1.0]);
            rc.camera = Some(RaycastCamera::from_camera(&camera));
            rc.intersect_objects(arena, root, true)
        };
        // Centred: x in [-1, 1], y in [-0.5, 0.5].
        let (arena, root, id) = sprite(Vector2::new(0.5, 0.5), 0.0);
        let h = cast(&arena, root, 0.5, 0.25);
        assert_eq!(h.len(), 1);
        assert_eq!(h[0].object, id);
        assert!(close(h[0].distance, 5.0, 1e-4));
        let uv = h[0].uv.unwrap();
        assert!(close(uv.x, 0.75, 1e-4) && close(uv.y, 0.75, 1e-4), "{uv:?}");
        assert!(!cast(&arena, root, 0.9, 0.0).is_empty());
        assert!(cast(&arena, root, 1.1, 0.0).is_empty());
        assert!(cast(&arena, root, 0.0, 0.6).is_empty());
        // No camera, no sprite hits (three.js requires `raycaster.camera`).
        assert!(ray([0.0, 0.0, 5.0], [0.0, 0.0, -1.0])
            .intersect_objects(&arena, root, true)
            .is_empty());
        // Anchored at its bottom-left corner: x in [0, 2], y in [0, 1].
        let (arena, root, _) = sprite(Vector2::new(0.0, 0.0), 0.0);
        assert!(!cast(&arena, root, 1.9, 0.9).is_empty());
        assert!(cast(&arena, root, -0.1, 0.5).is_empty());
        // A quarter turn: the 2-wide quad now spans y, the 1-high one x.
        let (arena, root, _) = sprite(Vector2::new(0.5, 0.5), std::f32::consts::FRAC_PI_2);
        assert!(!cast(&arena, root, 0.0, 0.9).is_empty());
        assert!(cast(&arena, root, 0.9, 0.0).is_empty());
        assert!(!cast(&arena, root, 0.4, 0.0).is_empty());
    }

    #[test]
    fn instanced_hits_report_their_instance() {
        let g = crate::geometries::BoxGeometry::new(1.0, 1.0, 1.0);
        let mut im = InstancedMesh::new(g, basic(0), 3);
        for (i, x) in [-3.0, 0.0, 3.0].into_iter().enumerate() {
            im.set_matrix_at(i, Matrix4::translation(Vector3::new(x, 0.0, -(i as f32))));
        }
        let (arena, root, id) = scene_of(Object3D::instanced_mesh(im));
        for (i, x) in [-3.0f32, 0.0, 3.0].into_iter().enumerate() {
            let hits =
                ray([x + 0.1, 0.2, 10.0], [0.0, 0.0, -1.0]).intersect_objects(&arena, root, true);
            assert_eq!(hits.len(), 1, "{hits:?}");
            assert_eq!(hits[0].object, id);
            assert_eq!(hits[0].instance_id, Some(i));
            // Front face of a unit box pushed back by i.
            assert!(close(hits[0].distance, 9.5 + i as f32, 1e-5));
            assert!(hits[0].polygon_index.is_some());
        }
        assert!(ray([1.5, 0.0, 10.0], [0.0, 0.0, -1.0])
            .intersect_objects(&arena, root, true)
            .is_empty());
    }

    /// The hit follows the posed bone, as the renderer's skinning places it.
    #[test]
    fn skinned_hit_follows_the_posed_bone() {
        let mut g = BufferGeometry::new();
        g.set_attribute(
            "position",
            BufferAttribute::new(vec![-1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 0.0, 1.0, 0.0], 3),
        );
        let mut arena = ObjectArena::new();
        let root = arena.insert(Object3D::group());
        let bone_node = arena.insert(Object3D::group());
        arena.add_child(root, bone_node);
        let skeleton = crate::core::Skeleton::new(vec![crate::core::Bone {
            node: bone_node,
            inverse_bind: Matrix4::identity(),
        }]);
        let sm = SkinnedMesh::from_unweighted(g, basic(2), skeleton);
        let mesh = arena.insert(Object3D::skinned_mesh(sm));
        arena.add_child(root, mesh);
        let pose = |arena: &mut ObjectArena, at: Vector3| {
            arena.get_mut(bone_node).unwrap().position = at;
            arena.update_world_matrices(root, Matrix4::identity());
            let mut sk = match &arena.get(mesh).unwrap().kind {
                ObjectKind::SkinnedMesh(sm) => (*sm.skeleton).clone(),
                _ => unreachable!(),
            };
            sk.update(arena);
            if let ObjectKind::SkinnedMesh(sm) = &mut arena.get_mut(mesh).unwrap().kind {
                sm.skeleton = std::sync::Arc::new(sk);
            }
        };
        let down = |x: f32| ray([x, 0.0, 10.0], [0.0, 0.0, -1.0]);
        pose(&mut arena, Vector3::ZERO);
        let rest = down(0.0).intersect_objects(&arena, root, true);
        assert!(close(rest[0].distance, 10.0, 1e-5));
        // Moved 5 along x and 3 back: the rest spot is empty, the posed one hit.
        pose(&mut arena, Vector3::new(5.0, 0.0, -3.0));
        assert!(down(0.0).intersect_objects(&arena, root, true).is_empty());
        let posed = down(5.0).intersect_objects(&arena, root, true);
        assert_eq!(posed.len(), 1);
        assert_eq!(posed[0].object, mesh);
        assert!(close(posed[0].distance, 13.0, 1e-5));
        assert!(close(posed[0].point.x, 5.0, 1e-5));
    }

    /// Weights that do not sum to one: three.js's shader leaves
    /// `Σ w·B·p + (1 − Σ w)·translation(matrix_world)`, and that is where
    /// every renderer draws the mesh, so that is where it is hit. Identity
    /// bones, weight 0.5, the mesh moved to x = 2: the triangle is drawn
    /// halved and centred on x = 1.
    #[test]
    fn skinned_hits_follow_unnormalised_weights_as_drawn() {
        let mut g = BufferGeometry::new();
        g.set_attribute(
            "position",
            BufferAttribute::new(vec![-1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 0.0, 1.0, 0.0], 3),
        );
        g.set_attribute("joint", BufferAttribute::new(vec![0.0; 12], 4));
        g.set_attribute(
            "weight",
            BufferAttribute::new([0.5, 0.0, 0.0, 0.0].repeat(3), 4),
        );
        let skeleton = crate::core::Skeleton::new(vec![crate::core::Bone {
            node: ObjectId::default(),
            inverse_bind: Matrix4::identity(),
        }]);
        let mut obj = Object3D::skinned_mesh(SkinnedMesh::new(g, basic(2), skeleton));
        obj.position = Vector3::new(2.0, 0.0, 0.0);
        let (arena, root, id) = scene_of(obj);
        // Drawn corners: (0.5, -0.5), (1.5, -0.5), (1, 0.5).
        for (x, y, hit) in [
            (1.0, -0.4, true),
            (1.4, -0.45, true),
            (1.0, 0.45, true),
            (1.6, -0.4, false),
            (0.4, -0.4, false),
            (1.0, 0.6, false),
            (2.0, 0.0, false),
        ] {
            let hits = ray([x, y, 5.0], [0.0, 0.0, -1.0]).intersect_objects(&arena, root, true);
            assert_eq!(!hits.is_empty(), hit, "({x}, {y}): {hits:?}");
            if hit {
                assert_eq!(hits[0].object, id);
                assert!(close(hits[0].distance, 5.0, 1e-5));
            }
        }
    }

    /// Raycasting reads the polygon table through the geometry's cache, and a
    /// geometry changed in place — by a mutator, or by a direct field write
    /// followed by `bump_version` — reports its new grouping.
    #[test]
    fn polygon_index_follows_geometry_changes() {
        let mut g = crate::geometries::PlaneGeometry::with_segments(4.0, 1.0, 4, 1);
        g.set_polygons(vec![2, 2, 2, 2]);
        let mat = Material::Basic(BasicMaterial::new(Color::from_hex(0xffffff)));
        let (mut arena, root, id) = scene_of(Object3D::mesh(Mesh::new(g, mat)));
        let at = |arena: &ObjectArena, x: f32| {
            ray([x, 0.1, 1.0], [0.0, 0.0, -1.0]).intersect_objects(arena, root, true)[0]
                .polygon_index
        };
        assert_eq!(at(&arena, 1.5), Some(3));
        let edit = |arena: &mut ObjectArena, f: &dyn Fn(&mut BufferGeometry)| {
            if let ObjectKind::Mesh(m) = &mut arena.get_mut(id).unwrap().kind {
                f(std::sync::Arc::get_mut(&mut m.geometry).expect("unshared"));
            }
        };
        edit(&mut arena, &|g| {
            g.set_polygons(vec![4, 4]);
        });
        assert_eq!(at(&arena, 1.5), Some(1));
        // A direct write of a same-length table is seen after the bump the
        // contract asks for (see `DerivedCache`).
        edit(&mut arena, &|g| {
            g.polygons = Some(vec![6, 2]);
            g.bump_version();
        });
        assert_eq!(at(&arena, 0.5), Some(0));
        assert_eq!(at(&arena, 1.5), Some(1));
        edit(&mut arena, &|g| {
            let index = g.index.clone().unwrap();
            g.set_index(index);
        });
        assert_eq!(at(&arena, 1.5), None);
    }

    /// `first_hit_only` returns exactly the first entry of the full, sorted
    /// result, across meshes, a BVH mesh, instances and lines.
    #[test]
    fn first_hit_only_is_the_first_of_the_full_result() {
        let mut arena = ObjectArena::new();
        let root = arena.insert(Object3D::group());
        for i in 0..27 {
            let mut o = Object3D::mesh(Mesh::new(
                crate::geometries::BoxGeometry::new(0.7, 0.7, 0.7),
                basic(i % 3),
            ));
            o.position = Vector3::new(
                (i % 3) as f32 - 1.0,
                ((i / 3) % 3) as f32 - 1.0,
                (i / 9) as f32 - 1.0,
            ) * 1.3;
            o.rotate_y(i as f32 * 0.37);
            let id = arena.insert(o);
            arena.add_child(root, id);
        }
        #[allow(unused_mut)]
        let mut sphere = crate::geometries::SphereGeometry::new(0.8, 24, 16);
        #[cfg(feature = "mesh-bvh")]
        sphere.compute_bounds_tree(Default::default());
        let mut s = Object3D::mesh(Mesh::new(sphere, basic(2)));
        s.position = Vector3::new(0.4, -0.3, 0.6);
        let sid = arena.insert(s);
        arena.add_child(root, sid);
        let mut im = InstancedMesh::new(
            crate::geometries::BoxGeometry::new(0.3, 0.3, 0.3),
            basic(0),
            20,
        );
        for i in 0..20 {
            let t = i as f32 * 0.31;
            im.set_matrix_at(
                i,
                Matrix4::translation(Vector3::new(t.sin() * 2.0, t.cos() * 2.0, t - 3.0)),
            );
        }
        let iid = arena.insert(Object3D::instanced_mesh(im));
        arena.add_child(root, iid);
        arena.update_world_matrices(root, Matrix4::identity());

        let mut state = 12345u32;
        let mut rnd = || {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            (state >> 8) as f32 / (1u32 << 24) as f32 * 2.0 - 1.0
        };
        let mut compared = 0;
        for _ in 0..400 {
            let o = Vector3::new(rnd(), rnd(), rnd()).normalize() * 8.0;
            let target = Vector3::new(rnd(), rnd(), rnd()) * 1.5;
            let mut rc = Raycaster::new(o, (target - o).normalize(), 0.0, f32::INFINITY);
            rc.near = rnd().abs() * 4.0;
            let all = rc.intersect_objects(&arena, root, true);
            rc.first_hit_only = true;
            let first = rc.intersect_objects(&arena, root, true);
            assert_eq!(first.len(), all.len().min(1));
            if let (Some(a), Some(f)) = (all.first(), first.first()) {
                compared += 1;
                assert_eq!(a.object, f.object);
                assert_eq!(a.instance_id, f.instance_id);
                assert!(close(a.distance, f.distance, 1e-6), "{a:?} vs {f:?}");
                // A tie on a shared edge may name either triangle.
                if (all.len() < 2) || (all[1].distance - a.distance).abs() > 1e-5 {
                    assert_eq!(a.face_index, f.face_index);
                }
            }
        }
        assert!(compared > 200, "only {compared} rays hit anything");
    }

    #[test]
    fn set_from_camera_dispatches_on_the_projection() {
        use crate::cameras::{OrthographicCamera, PerspectiveCamera};
        let mut persp = PerspectiveCamera::new(60.0, 1.0, 0.1, 100.0);
        persp.position = Vector3::new(1.0, 2.0, 5.0);
        let mut rc = Raycaster::default();
        rc.set_from_camera(Vector2::new(0.3, -0.2), &persp);
        // From the eye.
        assert!((rc.ray.origin - persp.position).length() < 1e-5);
        assert!(rc.camera.unwrap().perspective);
        let mut ortho = OrthographicCamera::new(-2.0, 2.0, 2.0, -2.0, 0.1, 100.0);
        ortho.position = Vector3::new(0.0, 0.0, 5.0);
        ortho.target = Vector3::ZERO;
        rc.set_from_camera(Vector2::new(0.5, 0.5), &ortho);
        // From the camera plane, straight down the view axis (three.js r165:
        // origin (1, 1, 5)).
        assert!(close(rc.ray.origin.z, 5.0, 1e-4) && close(rc.ray.origin.x, 1.0, 1e-4));
        assert!(close(rc.ray.direction.z, -1.0, 1e-6));
        assert!(!rc.camera.unwrap().perspective);
    }

    #[test]
    fn intersect_object_list_sorts_across_objects() {
        let mut arena = ObjectArena::new();
        let mut ids = Vec::new();
        for z in [-4.0, -1.0, -2.0] {
            let mut o = Object3D::mesh(unit_triangle_mesh());
            o.position = Vector3::new(0.0, 0.0, z);
            ids.push(arena.insert(o));
        }
        for &id in &ids {
            arena.update_world_matrices(id, Matrix4::identity());
        }
        let rc = ray([0.2, 0.2, 1.0], [0.0, 0.0, -1.0]);
        let hits = rc.intersect_object_list(&arena, &ids, false);
        let order: Vec<ObjectId> = hits.iter().map(|h| h.object).collect();
        assert_eq!(order, vec![ids[1], ids[2], ids[0]]);
        assert_eq!(rc.intersect_object(&arena, ids[2], false)[0].object, ids[2]);
    }
    /// three.js tests only the layers: with `include_invisible` an invisible
    /// object (and anything under an invisible parent) is hit.
    #[test]
    fn include_invisible_is_three_js_rule() {
        let mut arena = ObjectArena::new();
        let root = arena.insert(Object3D::group());
        let mut hidden = Object3D::group();
        hidden.visible = false;
        let hidden = arena.insert(hidden);
        arena.add_child(root, hidden);
        let mesh = arena.insert(Object3D::mesh(unit_triangle_mesh()));
        arena.add_child(hidden, mesh);
        arena.update_world_matrices(root, Matrix4::identity());
        let mut rc = ray([0.2, 0.2, 1.0], [0.0, 0.0, -1.0]);
        assert!(
            rc.intersect_object(&arena, root, true).is_empty(),
            "skipped by default"
        );
        rc.include_invisible = true;
        assert_eq!(rc.intersect_object(&arena, root, true).len(), 1);
        assert_eq!(rc.intersect_each(&arena, &[(mesh, false)]).len(), 1);
    }

    /// `intersect_each`: one object at a time, no descent, a first-hit flag
    /// each — three-mesh-bvh's per-mesh `firstHitOnly`.
    #[test]
    fn intersect_each_casts_objects_alone() {
        let mut arena = ObjectArena::new();
        let mut g = BufferGeometry::new();
        // Two triangles at z = 0 and z = -1.
        g.set_attribute(
            "position",
            BufferAttribute::new(
                vec![
                    0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., -1., 1., 0., -1., 0., 1., -1.,
                ],
                3,
            ),
        );
        let two = arena.insert(Object3D::mesh(Mesh::new(g, basic(2))));
        let child = arena.insert(Object3D::mesh(unit_triangle_mesh()));
        arena.add_child(two, child);
        arena.update_world_matrices(two, Matrix4::identity());
        let rc = ray([0.2, 0.2, 1.0], [0.0, 0.0, -1.0]);
        let all = rc.intersect_each(&arena, &[(two, false)]);
        assert_eq!(all.len(), 2, "no descent into the child");
        assert!(all[0].distance < all[1].distance);
        let first = rc.intersect_each(&arena, &[(two, true), (child, false)]);
        assert_eq!(first.len(), 2);
        assert_eq!(first[0].object, two);
        assert!(close(first[0].distance, 1.0, 1e-6));
        assert_eq!(first[1].object, child);
    }

    /// `set_from_matrices` (three.js's `setFromCamera` from three.js's own
    /// OpenGL-convention matrices) and `set_from_camera` agree for a camera
    /// both can describe.
    #[test]
    fn set_from_matrices_matches_set_from_camera() {
        use crate::cameras::{Camera, PerspectiveCamera};
        let mut cam = PerspectiveCamera::new(55.0, 1.6, 0.5, 40.0);
        cam.position = Vector3::new(1.5, 1.1, 5.0);
        cam.target = Vector3::new(0.1, -0.2, 0.0);
        let world = cam.view_matrix().invert();
        // three.js's perspective projection, OpenGL depth.
        let (near, far) = (0.5f32, 40.0f32);
        let top = near * (55.0f32.to_radians() * 0.5).tan();
        let (h, w) = (2.0 * top, 1.6 * 2.0 * top);
        let mut p = [0.0f32; 16];
        p[0] = 2.0 * near / w;
        p[5] = 2.0 * near / h;
        p[10] = -(far + near) / (far - near);
        p[11] = -1.0;
        p[14] = -2.0 * far * near / (far - near);
        let proj = Matrix4 { elements: p };
        for ndc in [
            Vector2::new(0.0, 0.0),
            Vector2::new(0.7, -0.4),
            Vector2::new(-0.9, 0.9),
        ] {
            let mut a = Raycaster::default();
            a.set_from_camera(ndc, &cam);
            let mut b = Raycaster::default();
            b.set_from_matrices(ndc, &world, &proj.invert(), None);
            assert!((a.ray.origin - b.ray.origin).length() < 1e-4);
            assert!(
                (a.ray.direction - b.ray.direction).length() < 1e-4,
                "{ndc:?}"
            );
            assert!(b.camera.unwrap().perspective);
        }
    }

    /// A strip (`Line`) reports strip vertices and honours the draw range; a
    /// loop closes; points honour the draw range too — three.js r165.
    #[test]
    fn line_strips_loops_and_draw_ranges() {
        let mut g = BufferGeometry::new();
        // A zigzag at z = 0: (0,0) (1,1) (2,0) (3,1).
        g.set_attribute(
            "position",
            BufferAttribute::new(vec![0., 0., 0., 1., 1., 0., 2., 0., 0., 3., 1., 0.], 3),
        );
        let g = std::sync::Arc::new(g);
        let mat = std::sync::Arc::new(basic(0));
        let mut arena = ObjectArena::new();
        let strip = arena.insert(Object3D::line_segments(LineSegments::strip(
            g.clone(),
            mat.clone(),
            false,
        )));
        let lp = arena.insert(Object3D::line_segments(LineSegments::strip(
            g.clone(),
            mat.clone(),
            true,
        )));
        for id in [strip, lp] {
            arena.update_world_matrices(id, Matrix4::identity());
        }
        let mut rc = ray([2.5, 0.5, 5.0], [0.0, 0.0, -1.0]);
        rc.params.line.threshold = 0.1;
        // Through the middle of segment 2 → 3.
        let hits = rc.intersect_object(&arena, strip, false);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].index, Some(2), "strip vertex index, as three.js");
        // The loop's closing segment 3 → 0 passes (2.7, 0.9), 0.14 from the strip.
        let mut rc2 = ray([2.7, 0.9, 5.0], [0.0, 0.0, -1.0]);
        rc2.params.line.threshold = 0.1;
        assert!(rc2.intersect_object(&arena, strip, false).is_empty());
        let closing = rc2.intersect_object(&arena, lp, false);
        assert_eq!(closing.len(), 1);
        assert_eq!(closing[0].index, Some(3));
        // Draw range [1, 3): only segment 1 → 2.
        let mut ranged = (*g).clone();
        ranged.set_draw_range(1, 2);
        let ranged = arena.insert(Object3D::line_segments(LineSegments::strip(
            std::sync::Arc::new(ranged),
            mat.clone(),
            false,
        )));
        arena.update_world_matrices(ranged, Matrix4::identity());
        assert!(rc.intersect_object(&arena, ranged, false).is_empty());
        let mut rc3 = ray([1.5, 0.5, 5.0], [0.0, 0.0, -1.0]);
        rc3.params.line.threshold = 0.1;
        assert_eq!(
            rc3.intersect_object(&arena, ranged, false)[0].index,
            Some(1)
        );
        // Points: the draw range picks which are tested.
        let mut pg = (*g).clone();
        pg.set_draw_range(2, 2);
        let pts = arena.insert(Object3D::points(Points {
            geometry: std::sync::Arc::new(pg),
            material: mat,
        }));
        arena.update_world_matrices(pts, Matrix4::identity());
        let mut rp = ray([1.0, 1.0, 5.0], [0.0, 0.0, -1.0]);
        rp.params.points.threshold = 0.2;
        assert!(
            rp.intersect_object(&arena, pts, false).is_empty(),
            "point 1 is outside the range"
        );
        rp.set(Vector3::new(2.0, 0.0, 5.0), Vector3::new(0.0, 0.0, -1.0));
        assert_eq!(rp.intersect_object(&arena, pts, false)[0].index, Some(2));
    }
}
