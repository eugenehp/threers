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
    /// three.js `scene.environment`: the image-based light of every Standard
    /// and Physical material without an `envMap` of its own. Sampled through
    /// its PMREM atlas when it carries one, else through its mip chain.
    pub environment: Option<Arc<CubeTexture>>,
    /// three.js `scene.environmentIntensity` (default 1).
    pub environment_intensity: f32,
    /// three.js `scene.environmentRotation` (default zero).
    pub environment_rotation: crate::math::Euler,
    /// Alternate environment input: a CubeRenderTarget's cube view. When set,
    /// the renderer samples from the live RT contents (typically produced by
    /// a CubeCamera) instead of the CPU-backed `environment` cube texture.
    pub environment_cube_rt: Option<u32>,
    pub fog: FogParams,
}

impl Scene {
    /// Every visible [`PositionalAudio`](crate::audio::PositionalAudio) as
    /// `listener` hears it, for a playback backend to apply each frame. Uses
    /// the world matrices of the last update.
    #[cfg(feature = "audio")]
    pub fn audio_mix(
        &self,
        listener: &crate::audio::AudioListener,
    ) -> Vec<(ObjectId, crate::audio::SpatialMix)> {
        let mut mix = Vec::new();
        self.arena.traverse_visible(self.root, &mut |id, obj| {
            if let crate::core::ObjectKind::Audio(audio) = &obj.kind {
                mix.push((id, audio.mix(&obj.matrix_world, listener)));
            }
        });
        mix
    }

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
            environment_intensity: 1.0,
            environment_rotation: crate::math::Euler::default(),
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

    /// Give the scene root a fixed transform that every object inherits — the
    /// single orienting/centring group a host app (e.g. a viewer that rotates
    /// its whole scene) would otherwise keep on the CPU side. `update_world`
    /// composes it into each child's world matrix, so cameras, picking and
    /// rendering all see the objects in that frame. Takes the matrix as given
    /// (auto-recompose from position/quaternion/scale is turned off on the root).
    pub fn set_root_transform(&mut self, matrix: Matrix4) {
        if let Some(root) = self.arena.get_mut(self.root) {
            root.matrix = matrix;
            root.matrix_auto_update = false;
            root.matrix_world_auto_update = true;
        }
    }

    pub fn get(&self, id: ObjectId) -> Option<&Object3D> {
        self.arena.get(id)
    }

    pub fn get_mut(&mut self, id: ObjectId) -> Option<&mut Object3D> {
        self.arena.get_mut(id)
    }

    /// Refresh all `matrix_world`s. The renderer calls this each frame.
    /// Bring every world matrix up to date — what three.js's renderer does
    /// with `scene.updateMatrixWorld()` before drawing, and skips when the
    /// root's `matrix_world_auto_update` is off.
    pub fn update_world(&mut self) {
        if self
            .arena
            .get(self.root)
            .is_some_and(|r| !r.matrix_world_auto_update)
        {
            return;
        }
        self.arena
            .update_world_matrices(self.root, Matrix4::identity());
        // three.js's renderer runs `skeleton.update()` for each skinned mesh
        // it draws, from the bones' fresh world matrices.
        crate::core::update_skeletons(&mut self.arena);
    }

    /// Writes each object's [`flicker`](Object3D::flicker) at `time`
    /// seconds into the scene, for the frame about to be drawn: an object
    /// at level 0 is hidden, one partly shown has its materials' opacity and
    /// its light's intensity scaled. A flicker reaches what is under its
    /// object, multiplied by any flicker of their own. Call it once a frame,
    /// before rendering, with the time the frame will show at; nothing
    /// flickers until it is called.
    ///
    /// ```
    /// use threers::math::flicker::Flicker;
    /// use threers::{Object3D, Scene};
    ///
    /// let mut scene = Scene::new();
    /// let marker = scene.add(Object3D::group());
    /// scene.get_mut(marker).unwrap().flicker = Some(Flicker::square(15.0));
    /// scene.apply_flicker(0.05); // 0.75 of a cycle in
    /// assert!(!scene.get(marker).unwrap().visible);
    /// ```
    pub fn apply_flicker(&mut self, time: f64) {
        let mut stack = vec![(self.root, 1.0f32)];
        let mut children = Vec::new();
        while let Some((id, above)) = stack.pop() {
            let Some(obj) = self.arena.get_mut(id) else {
                continue;
            };
            // Most scenes flicker nothing: nothing to write, and nothing to give back.
            let level = above * obj.flicker.map_or(1.0, |f| f.level(time));
            if level < 1.0 || obj.flicker_saved.is_some() {
                obj.show_at(level);
            }
            // Hidden takes what is under it out of the frame too; it is
            // brought up to date when it shows again.
            if level > 0.0 {
                children.clone_from(&obj.children);
                stack.extend(children.iter().map(|&c| (c, level)));
            }
        }
    }
}

impl Default for Scene {
    fn default() -> Self {
        Self::new()
    }
}
