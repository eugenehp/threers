use crate::core::{Object3D, ObjectArena, ObjectId};
use crate::lights::Light;
use crate::math::{Color, Matrix4};
use crate::textures::CubeTexture;
use std::sync::Arc;

/// Fog parameters mirroring three.js [`Fog`](https://threejs.org/docs/#api/en/scene/Fog)
/// (linear) and `FogExp2`.
///
/// `mode`: `0` = disabled, `1` = linear, `2` = exponential squared.
#[derive(Debug, Clone, Copy)]
pub struct FogParams {
    pub color: Color,
    pub near: f32,
    pub far: f32,
    pub density: f32,
    pub mode: u32,
}
impl Default for FogParams {
    fn default() -> Self {
        Self {
            color: Color::WHITE,
            near: 1.0,
            far: 1000.0,
            density: 0.0,
            mode: 0,
        }
    }
}

/// Root scene container: object arena, background, environment, fog.
///
/// Mirrors three.js [`Scene`](https://threejs.org/docs/#api/en/scenes/Scene) —
/// the root is an `Object3D` group named `"Scene"`.
pub struct Scene {
    pub arena: ObjectArena,
    pub root: ObjectId,
    pub background: Color,
    /// Clear alpha for the background (0.0 = fully transparent framebuffer,
    /// 1.0 = opaque). Lets offscreen renders produce transparent PNGs.
    pub background_alpha: f32,
    /// Optional environment map used for image-based lighting (PBR ambient).
    /// The current renderer uploads it as a cube texture but does not yet
    /// prefilter for roughness — sampling falls back to the default white
    /// 1x1 fallback if `None`.
    pub environment: Option<Arc<CubeTexture>>,
    /// Alternate environment input: a CubeRenderTarget's cube view. When set,
    /// the renderer samples from the live RT contents (typically produced by
    /// a CubeCamera) instead of the CPU-backed `environment` cube texture.
    pub environment_cube_rt: Option<u32>,
    pub fog: FogParams,
}

impl Scene {
    /// Empty scene with dark gray background and no fog.
    pub fn new() -> Self {
        let mut arena = ObjectArena::new();
        let mut root_obj = Object3D::group();
        root_obj.name = "Scene".into();
        let root = arena.insert(root_obj);
        Self {
            arena,
            root,
            background: Color::from_hex(0x111111),
            background_alpha: 1.0,
            environment: None,
            environment_cube_rt: None,
            fog: FogParams::default(),
        }
    }

    /// `scene.add(obj)` — insert and parent under the scene root.
    pub fn add(&mut self, obj: Object3D) -> ObjectId {
        let id = self.arena.insert(obj);
        self.arena.add_child(self.root, id);
        id
    }

    /// Convenience: build a light Object3D and add it to the scene.
    /// Geometry plus material into the scene, in one call.
    ///
    /// `scene.add(Object3D::mesh(Mesh::new(geometry, material)))` is three
    /// constructors deep to say one thing, and it is the single most repeated
    /// line in every example in this crate.
    pub fn add_mesh(
        &mut self,
        geometry: crate::core::BufferGeometry,
        material: crate::materials::Material,
    ) -> ObjectId {
        self.add(Object3D::mesh(crate::core::Mesh::new(geometry, material)))
    }

    pub fn add_light(&mut self, light: impl Into<Light>) -> ObjectId {
        self.add(Object3D::light(light))
    }

    /// Add under a specific parent (not the scene root).
    pub fn add_to(&mut self, parent: ObjectId, obj: Object3D) -> ObjectId {
        let id = self.arena.insert(obj);
        self.arena.add_child(parent, id);
        id
    }

    pub fn get(&self, id: ObjectId) -> Option<&Object3D> {
        self.arena.get(id)
    }

    pub fn get_mut(&mut self, id: ObjectId) -> Option<&mut Object3D> {
        self.arena.get_mut(id)
    }

    /// Refresh all `matrix_world`s. The renderer calls this each frame.
    pub fn update_world(&mut self) {
        self.arena
            .update_world_matrices(self.root, Matrix4::identity());
    }
}

impl Default for Scene {
    fn default() -> Self {
        Self::new()
    }
}
