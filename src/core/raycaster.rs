use super::{Layers, Mesh, Object3D, ObjectArena, ObjectId, ObjectKind};
use crate::cameras::Camera;
use crate::math::{Box3, Ray, Sphere, Triangle, Vector2, Vector3};

/// A ray-hit result on a single mesh face. Mirrors three.js's intersection record.
#[derive(Debug, Clone, Copy)]
pub struct Intersection {
    pub distance: f32,
    pub point: Vector3,
    /// Triangle index within the geometry. For indexed geometries this is the
    /// triple-index (i.e. positions of the three vertices are at index*3, +1, +2);
    /// for non-indexed geometries it's the triangle position in vertex order.
    pub face_index: usize,
    pub object: ObjectId,
}

/// Mirrors three.js's `Raycaster`.
#[derive(Debug, Clone, Copy)]
pub struct Raycaster {
    pub ray: Ray,
    pub near: f32,
    pub far: f32,
    pub layers: Layers,
}

impl Default for Raycaster {
    fn default() -> Self {
        Self {
            ray: Ray::default(),
            near: 0.0,
            far: f32::INFINITY,
            layers: {
                let mut l = Layers::default();
                l.enable_all();
                l
            },
        }
    }
}

impl Raycaster {
    pub fn new(origin: Vector3, direction: Vector3, near: f32, far: f32) -> Self {
        Self {
            ray: Ray::new(origin, direction),
            near,
            far,
            layers: Self::default().layers,
        }
    }

    /// Build a ray from normalized device coordinates (x, y ∈ [-1, 1]) and a camera.
    /// Mirrors three.js's `setFromCamera` for `PerspectiveCamera`. Orthographic
    /// support requires camera-trait inspection — see `set_from_camera_ortho`.
    pub fn set_from_camera_perspective(&mut self, ndc: Vector2, camera: &dyn Camera) {
        let view = camera.view_matrix();
        let proj = camera.projection_matrix();
        let near = Vector3::new(ndc.x, ndc.y, 0.5).unproject(&view, &proj);
        let origin = camera.position();
        self.ray = Ray::new(origin, (near - origin).normalize());
        self.layers = camera.layers();
    }

    /// Orthographic ray: origin lies on the near plane at NDC (x, y); direction
    /// is the camera's forward axis (negative-Z of view matrix's inverse rotation).
    ///
    /// The near plane is NDC `z = 0`, not `z = -1`. This crate's projection
    /// matrices use the wgpu/D3D depth range `[0, 1]`; under OpenGL's `[-1, 1]`
    /// the near plane is at `-1`, and unprojecting that here lands the ray
    /// origin most of a frustum *behind* the camera. Every hit then measures
    /// past `far` and picking returns nothing at all.
    pub fn set_from_camera_ortho(&mut self, ndc: Vector2, camera: &dyn Camera) {
        let view = camera.view_matrix();
        let proj = camera.projection_matrix();
        let near = Vector3::new(ndc.x, ndc.y, 0.0).unproject(&view, &proj);
        let far = Vector3::new(ndc.x, ndc.y, 1.0).unproject(&view, &proj);
        self.ray = Ray::new(near, (far - near).normalize());
        self.layers = camera.layers();
    }

    /// Test all visible meshes in `root`'s subtree. Sorted by distance ascending.
    pub fn intersect_objects(
        &self,
        arena: &ObjectArena,
        root: ObjectId,
        recursive: bool,
    ) -> Vec<Intersection> {
        let mut out = Vec::new();
        if recursive {
            arena.traverse_visible(root, &mut |id, obj| {
                if self.layers.test(&obj.layers) {
                    if let ObjectKind::Mesh(mesh) = &obj.kind {
                        self.intersect_mesh(id, obj, mesh, &mut out);
                    }
                    // Lines/Points/Sprites/InstancedMesh — raycasting not implemented yet.
                }
            });
        } else if let Some(obj) = arena.get(root) {
            if obj.visible && self.layers.test(&obj.layers) {
                if let ObjectKind::Mesh(mesh) = &obj.kind {
                    self.intersect_mesh(root, obj, mesh, &mut out);
                }
            }
        }
        out.sort_by(|a, b| {
            a.distance
                .partial_cmp(&b.distance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        out
    }

    fn intersect_mesh(
        &self,
        id: ObjectId,
        obj: &Object3D,
        mesh: &Mesh,
        out: &mut Vec<Intersection>,
    ) {
        // Bounding-sphere reject (in world space — coarse but cheap).
        let world_sphere = match mesh.geometry.bounding_sphere {
            Some(s) => s.apply_matrix4(&obj.matrix_world),
            None => {
                // Fallback: compute a bounding sphere on the fly from positions.
                let Some(iter) = mesh.geometry.positions() else {
                    return;
                };
                let pts: Vec<Vector3> = iter.collect();
                if pts.is_empty() {
                    return;
                }
                Sphere::from_points(&pts).apply_matrix4(&obj.matrix_world)
            }
        };
        if !self.ray.intersects_sphere(&world_sphere) {
            return;
        }

        // Transform ray into local space for triangle intersection.
        let inv_world = obj.matrix_world.invert();
        let local_ray = self.ray.apply_matrix4(&inv_world);
        let max_local_t = if self.far.is_finite() {
            self.far * inv_world_scale(&obj.matrix_world)
        } else {
            f32::INFINITY
        };

        #[cfg(feature = "mesh-bvh")]
        if let Some(bvh) = &mesh.geometry.bounds_tree {
            let near_local = self.near * inv_world_scale(&obj.matrix_world);
            let hits = bvh.raycast(&local_ray, near_local, max_local_t, false);
            for hit in hits {
                let world_point = hit.point.apply_matrix4(&obj.matrix_world);
                let world_distance = (world_point - self.ray.origin).length();
                if world_distance < self.near || world_distance > self.far {
                    continue;
                }
                out.push(Intersection {
                    distance: world_distance,
                    point: world_point,
                    face_index: hit.face_index,
                    object: id,
                });
            }
            return;
        }

        let positions: Vec<Vector3> = match mesh.geometry.positions() {
            Some(it) => it.collect(),
            None => return,
        };

        let push_hit = |face_index: usize, t: f32, tri: &Triangle, out: &mut Vec<Intersection>| {
            if t < 0.0 {
                return;
            }
            let local_point = local_ray.at(t);
            let world_point = local_point.apply_matrix4(&obj.matrix_world);
            let world_distance = (world_point - self.ray.origin).length();
            if world_distance < self.near || world_distance > self.far {
                return;
            }
            let _ = tri;
            out.push(Intersection {
                distance: world_distance,
                point: world_point,
                face_index,
                object: id,
            });
        };

        if let Some(idx) = &mesh.geometry.index {
            let n_tris = idx.len() / 3;
            for i in 0..n_tris {
                let a = idx[i * 3] as usize;
                let b = idx[i * 3 + 1] as usize;
                let c = idx[i * 3 + 2] as usize;
                if a >= positions.len() || b >= positions.len() || c >= positions.len() {
                    continue;
                }
                let tri = Triangle::new(positions[a], positions[b], positions[c]);
                if let Some(t) = local_ray.intersect_triangle(&tri, false) {
                    if t <= max_local_t {
                        push_hit(i, t, &tri, out);
                    }
                }
            }
        } else {
            let n_tris = positions.len() / 3;
            for i in 0..n_tris {
                let tri =
                    Triangle::new(positions[i * 3], positions[i * 3 + 1], positions[i * 3 + 2]);
                if let Some(t) = local_ray.intersect_triangle(&tri, false) {
                    if t <= max_local_t {
                        push_hit(i, t, &tri, out);
                    }
                }
            }
        }
    }

    /// Convenience: intersect a single AABB in world space (no scene walk).
    pub fn intersect_box(&self, b: &Box3) -> Option<f32> {
        self.ray
            .intersect_box(b)
            .filter(|t| *t >= self.near && *t <= self.far)
    }
}

/// Approximate inverse uniform world-scale (used to convert world-space `far`
/// into the equivalent local-space ray parameter). Not exact under non-uniform
/// scale, but tight enough for the early reject.
fn inv_world_scale(m: &crate::math::Matrix4) -> f32 {
    let e = &m.elements;
    let sx = Vector3::new(e[0], e[1], e[2]).length();
    let sy = Vector3::new(e[4], e[5], e[6]).length();
    let sz = Vector3::new(e[8], e[9], e[10]).length();
    let s = (sx + sy + sz) / 3.0;
    if s == 0.0 {
        1.0
    } else {
        1.0 / s
    }
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

        let rc = Raycaster {
            ray: Ray::new(Vector3::new(0.25, 0.25, -1.0), Vector3::new(0.0, 0.0, 1.0)),
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
        rc.ray = Ray::new(Vector3::new(0.25, 0.25, -1.0), Vector3::new(0.0, 0.0, 1.0));
        let hits = rc.intersect_objects(&arena, root, true);
        assert!(hits.is_empty(), "expected no hits, got {:?}", hits);
    }

    /// Orthographic picking against the crate's `[0, 1]` depth range.
    ///
    /// Unprojecting NDC `z = -1` for the near plane — the OpenGL convention —
    /// puts the ray origin nearly a whole frustum behind the camera, so every
    /// hit measures past `far` and gets discarded. The symptom is picking that
    /// silently returns nothing for every orthographic camera.
    #[test]
    fn ortho_camera_ray_starts_at_the_near_plane() {
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

        // The origin sits on the near plane, 0.1 in front of the camera.
        assert!(
            (rc.ray.origin.z - 4.9).abs() < 1e-3,
            "ray should start on the near plane, got z = {}",
            rc.ray.origin.z
        );
        let hits = rc.intersect_objects(&arena, root, true);
        assert_eq!(hits.len(), 1, "expected the triangle, got {:?}", hits);
        assert_eq!(hits[0].object, mesh_id);
        assert!(
            (hits[0].distance - 4.9).abs() < 1e-3,
            "distance from the near plane, got {}",
            hits[0].distance
        );
    }
}
