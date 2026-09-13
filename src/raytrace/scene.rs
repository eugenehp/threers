//! The tracer's view of a [`Scene`]: a flat, world-space, backend-agnostic
//! intermediate form.
//!
//! Nothing downstream of here knows about [`Object3D`](crate::core::Object3D),
//! the arena, or `Arc<Material>`. A triangle is three world-space points plus
//! an index into a material table; a light is a small enum; a texture is a
//! decoded byte buffer. That is the boundary a GPU backend needs — the same
//! [`RaytraceScene`] can be packed into storage buffers without walking the
//! scene graph in a shader — and it is also what makes the integrator testable
//! without a `Scene` at all.
//!
//! The conversion is lossy in one direction on purpose. three.js has fourteen
//! material types, most of which describe a *shading trick* rather than a
//! surface: `MeshNormalMaterial` paints the normal, `MeshToonMaterial` quantises
//! a ramp, `MeshMatcapMaterial` looks up a lit sphere. A path tracer has no
//! place to put those, so each is mapped to the closest physical surface and
//! the choice is recorded in [`crate::raytrace::scene::BuildReport::approximated`].

use std::collections::HashMap;
use std::sync::Arc;

use crate::core::{Layers, ObjectKind};
use crate::lights::Light;
use crate::materials::Material;
use crate::math::{Color, Matrix3, Matrix4, Vector2, Vector3};
use crate::scene::Scene;
use crate::textures::CubeTexture;

use super::bvh::RtBvh;
use super::distribution::EnvDistribution;
use super::primitives::{line_segment_quads, line_width_for, point_radius_for, point_spheres, sprite_quads};
use super::sampler::{cosine_hemisphere, Onb};
use super::settings::{BackgroundMode, RaytraceSettings};
use super::texture::{CpuTexture, TextureCache};

/// A surface, reduced to the parameters a principled BSDF actually integrates.
#[derive(Debug, Clone)]
pub struct RtMaterial {
    /// Diffuse/metal base colour, linear.
    pub base_color: Vector3,
    pub base_color_map: Option<Arc<CpuTexture>>,
    /// Coverage. Below 1 the surface is partly *not there* — a ray passes
    /// straight through with probability `1 - opacity`, which is how a cutout
    /// leaf casts a leaf-shaped shadow rather than a rectangular one.
    pub opacity: f32,
    /// Alpha below this is treated as a hole outright, matching three.js's
    /// `alphaTest`. 0 disables.
    pub alpha_test: f32,
    pub roughness: f32,
    pub roughness_map: Option<Arc<CpuTexture>>,
    pub metallic: f32,
    pub metallic_map: Option<Arc<CpuTexture>>,
    /// Emitted radiance, already multiplied by `emissiveIntensity`.
    pub emission: Vector3,
    pub emissive_map: Option<Arc<CpuTexture>>,
    pub normal_map: Option<Arc<CpuTexture>>,
    pub normal_scale: Vector2,
    /// Fraction of the non-metallic response that refracts rather than
    /// scattering diffusely.
    pub transmission: f32,
    pub ior: f32,
    /// Beer–Lambert tint of the medium behind a transmissive surface.
    pub attenuation_color: Vector3,
    /// Distance over which `attenuation_color` is reached. 0 disables
    /// absorption, leaving clear glass.
    pub attenuation_distance: f32,
    pub clearcoat: f32,
    pub clearcoat_roughness: f32,
    /// Dielectric F0 tint. Physical dielectrics are colourless at normal
    /// incidence; this exists to carry `MeshPhongMaterial::specular`.
    pub specular_tint: Vector3,
    /// -1..1. Stretches the specular lobe along the surface tangent.
    pub anisotropy: f32,
    /// Rotation of the anisotropy direction within the tangent plane, radians.
    pub anisotropy_rotation: f32,
    /// Charlie sheen lobe strength (fabric grazing highlight).
    pub sheen: f32,
    pub sheen_color: Vector3,
    pub sheen_roughness: f32,
    pub iridescence: f32,
    pub iridescence_ior: f32,
    /// Thin-film thickness in nanometres.
    pub iridescence_thickness: f32,
    pub iridescence_thickness_map: Option<Arc<CpuTexture>>,
    /// Chromatic dispersion strength for transmission (Abbe-like RGB split).
    pub dispersion: f32,
    /// Simple subsurface diffuse boost at grazing angles.
    pub subsurface: f32,
    pub subsurface_radius: Vector3,
    pub displacement_map: Option<Arc<CpuTexture>>,
    pub displacement_scale: f32,
    pub displacement_bias: f32,
    /// The surface emits exactly `emission` and scatters nothing — the tracer's
    /// reading of `MeshBasicMaterial` and friends. It looks like its own colour
    /// from every angle, which is what "unlit" means, and it lights the scene,
    /// which is what a surface of that colour would do.
    pub unlit: bool,
}

impl Default for RtMaterial {
    fn default() -> Self {
        Self {
            base_color: Vector3::new(1.0, 1.0, 1.0),
            base_color_map: None,
            opacity: 1.0,
            alpha_test: 0.0,
            roughness: 1.0,
            roughness_map: None,
            metallic: 0.0,
            metallic_map: None,
            emission: Vector3::ZERO,
            emissive_map: None,
            normal_map: None,
            normal_scale: Vector2::new(1.0, 1.0),
            transmission: 0.0,
            ior: 1.5,
            attenuation_color: Vector3::new(1.0, 1.0, 1.0),
            attenuation_distance: 0.0,
            clearcoat: 0.0,
            clearcoat_roughness: 0.0,
            specular_tint: Vector3::new(1.0, 1.0, 1.0),
            anisotropy: 0.0,
            anisotropy_rotation: 0.0,
            sheen: 0.0,
            sheen_color: Vector3::ZERO,
            sheen_roughness: 1.0,
            iridescence: 0.0,
            iridescence_ior: 1.3,
            iridescence_thickness: 400.0,
            iridescence_thickness_map: None,
            dispersion: 0.0,
            subsurface: 0.0,
            subsurface_radius: Vector3::new(1.0, 0.2, 0.1),
            displacement_map: None,
            displacement_scale: 1.0,
            displacement_bias: 0.0,
            unlit: false,
        }
    }
}

impl RtMaterial {
    /// True when a ray can pass through without being scattered — the test that
    /// decides whether shadow rays need the slow attenuating walk instead of a
    /// first-hit test.
    ///
    /// The map is consulted for whether it *has* transparency, not merely for
    /// whether it exists: most base-colour maps are fully opaque, and treating
    /// every textured surface as a possible cutout would put every shadow ray
    /// in a textured scene on the slow path for nothing.
    pub fn is_shadow_transparent(&self) -> bool {
        self.opacity < 1.0
            || self.alpha_test > 0.0
            || self.transmission > 0.0
            || self
                .base_color_map
                .as_ref()
                .is_some_and(|m| m.has_transparency())
    }

    /// Whether this surface emits at all.
    ///
    /// The constant decides, not the map. three.js multiplies `emissiveMap` by
    /// `emissive`, which defaults to black — so a material carrying an emissive
    /// map but no emissive colour emits nothing, and counting it as a light
    /// would spend next-event samples on a surface that returns zero every
    /// time, diluting the probability of picking the emitters that do emit.
    pub fn is_emissive(&self) -> bool {
        self.emission.x > 0.0 || self.emission.y > 0.0 || self.emission.z > 0.0
    }
}

/// A light the tracer samples directly.
///
/// Analytic lights are not geometry, so a path can never *hit* one; they are
/// reached only by next-event estimation and never need a MIS weight against
/// BSDF sampling. Emissive surfaces are the opposite case and live in
/// [`RaytraceScene::emissive`].
#[derive(Debug, Clone, Copy)]
pub enum RtLight {
    /// Parallel light. `direction` points *from* the light *toward* the scene,
    /// matching three.js's `DirectionalLight` after it resolves its target.
    Directional {
        direction: Vector3,
        radiance: Vector3,
        /// Angular radius of the disc, in radians. 0 is a delta light.
        angular_radius: f32,
    },
    Point {
        position: Vector3,
        intensity: Vector3,
        /// Sphere radius. 0 is a delta light.
        radius: f32,
        /// Cutoff distance; 0 means unbounded. Matches `PointLight::distance`.
        distance: f32,
        decay: f32,
    },
    Spot {
        position: Vector3,
        /// Axis, pointing away from the light.
        direction: Vector3,
        intensity: Vector3,
        radius: f32,
        distance: f32,
        decay: f32,
        cos_outer: f32,
        cos_inner: f32,
    },
    /// A rectangle emitting from one face, oriented like three.js's
    /// `RectAreaLight`: `right` and `up` are half-extents and the emitting side
    /// faces `right × up`.
    Rect {
        position: Vector3,
        right: Vector3,
        up: Vector3,
        radiance: Vector3,
    },
}

/// A hemisphere light, as a two-tone sky.
#[derive(Debug, Clone, Copy)]
pub struct RtHemisphere {
    pub sky: Vector3,
    pub ground: Vector3,
    pub up: Vector3,
}

/// Everything a ray sees when it hits nothing.
#[derive(Debug, Clone)]
pub struct World {
    /// `scene.background`, linear.
    pub background: Vector3,
    pub background_alpha: f32,
    pub environment: Option<Arc<CubeTexture>>,
    pub environment_intensity: f32,
    /// Sum of every `AmbientLight`, as uniform radiance from all directions.
    /// A Lambertian of albedo `a` under uniform radiance `L` reflects `a·L`,
    /// which is exactly what the raster path computes for an ambient light —
    /// so the two agree without a fudge factor.
    pub ambient: Vector3,
    pub hemispheres: Vec<RtHemisphere>,
    pub mode: BackgroundMode,
    /// Decoded environment faces, in [`CubeTexture`] order.
    env_faces: Option<Box<[CpuTexture; 6]>>,
    /// A density over the environment's own brightness, for importance
    /// sampling. `None` when there is no environment, or it is black.
    env_distribution: Option<EnvDistribution>,
    /// Participating medium from `scene.fog`. Mode `0` = off, `1` = linear,
    /// `2` = exponential squared.
    pub fog_color: Vector3,
    pub fog_mode: u32,
    pub fog_near: f32,
    pub fog_far: f32,
    pub fog_density: f32,
}

impl Default for World {
    fn default() -> Self {
        Self {
            background: Vector3::ZERO,
            background_alpha: 1.0,
            environment: None,
            environment_intensity: 1.0,
            ambient: Vector3::ZERO,
            hemispheres: Vec::new(),
            mode: BackgroundMode::Color,
            env_faces: None,
            env_distribution: None,
            fog_color: Vector3::ZERO,
            fog_mode: 0,
            fog_near: 1.0,
            fog_far: 1000.0,
            fog_density: 0.0,
        }
    }
}

impl World {
    /// Radiance arriving from `dir` for *lighting* purposes. Independent of
    /// [`BackgroundMode`]: changing what the camera sees behind the subject
    /// must not change how the subject is lit.
    pub fn lighting_radiance(&self, dir: Vector3) -> Vector3 {
        let mut l = self.ambient;
        for h in &self.hemispheres {
            // three.js's hemisphere term: full sky straight up, full ground
            // straight down, linear in the cosine between.
            let t = 0.5 * dir.dot(h.up) + 0.5;
            l = l + h.ground.lerp(h.sky, t.clamp(0.0, 1.0));
        }
        if let Some(faces) = &self.env_faces {
            l = l + sample_cube(faces, dir) * self.environment_intensity;
        }
        l
    }

    /// Radiance and alpha for a *camera* ray that escaped.
    pub fn camera_radiance(&self, dir: Vector3) -> (Vector3, f32) {
        match self.mode {
            BackgroundMode::Transparent => (Vector3::ZERO, 0.0),
            BackgroundMode::Environment => match &self.env_faces {
                Some(faces) => (
                    sample_cube(faces, dir) * self.environment_intensity,
                    self.background_alpha,
                ),
                None => (self.background, self.background_alpha),
            },
            BackgroundMode::Color => (self.background, self.background_alpha),
        }
    }

    /// How often the environment's own distribution is used instead of a
    /// cosine-weighted guess.
    ///
    /// Half and half, because neither strategy is good at what the other is for.
    /// The environment distribution finds a sun that cosine sampling would miss
    /// entirely; cosine sampling covers the smooth part of the sky, the ambient
    /// and hemisphere lights the distribution knows nothing about, and the
    /// hemisphere the surface can actually see — which for a surface facing away
    /// from the sun is most of what reaches it. Sampling from the mixture, and
    /// then reporting the *mixture's* density, keeps the estimator unbiased
    /// whichever one produced the direction.
    const ENV_STRATEGY: f32 = 0.5;

    /// Draw a direction to sample the world along, with its density.
    ///
    /// `pick` chooses the strategy and `u1`/`u2` drive it.
    pub fn sample_direction(
        &self,
        normal: Vector3,
        pick: f32,
        u1: f32,
        u2: f32,
    ) -> Option<(Vector3, f32)> {
        let dir = match &self.env_distribution {
            Some(d) if pick < Self::ENV_STRATEGY => d.sample(u1, u2).0,
            _ => Onb::new(normal).to_world(cosine_hemisphere(u1, u2)),
        };
        let pdf = self.direction_pdf(normal, dir);
        (pdf > 1e-9).then_some((dir, pdf))
    }

    /// The density [`Self::sample_direction`] has for `dir` — the mixture, not
    /// whichever half happened to produce it.
    pub fn direction_pdf(&self, normal: Vector3, dir: Vector3) -> f32 {
        let cosine = (normal.dot(dir) / std::f32::consts::PI).max(0.0);
        match &self.env_distribution {
            Some(d) => Self::ENV_STRATEGY * d.pdf(dir) + (1.0 - Self::ENV_STRATEGY) * cosine,
            None => cosine,
        }
    }

    /// The environment's sampling distribution, if one was built.
    pub fn env_distribution(&self) -> Option<&EnvDistribution> {
        self.env_distribution.as_ref()
    }

    /// Whether an environment map was supplied *and* decoded.
    pub fn has_environment(&self) -> bool {
        self.env_faces.is_some()
    }

    /// The decoded environment faces, in [`CubeTexture`] order. A device
    /// backend packs these into its own atlas.
    pub fn env_faces(&self) -> Option<&[CpuTexture; 6]> {
        self.env_faces.as_deref()
    }

    /// Whether the world emits anything at all. When it does not, escaped rays
    /// contribute nothing and environment sampling can be skipped outright.
    pub fn is_black(&self) -> bool {
        self.env_faces.is_none()
            && self.hemispheres.is_empty()
            && self.ambient.x <= 0.0
            && self.ambient.y <= 0.0
            && self.ambient.z <= 0.0
    }

    /// Beer–Lambert visibility and in-scattered fog radiance at `distance`.
    pub fn fog_at(&self, distance: f32) -> (f32, Vector3) {
        if self.fog_mode == 0 || distance <= 0.0 {
            return (1.0, Vector3::ZERO);
        }
        let vis = match self.fog_mode {
            1 => {
                let span = (self.fog_far - self.fog_near).max(1e-6);
                ((self.fog_far - distance) / span).clamp(0.0, 1.0)
            }
            2 => (-self.fog_density * self.fog_density * distance).exp(),
            _ => 1.0,
        };
        (vis, self.fog_color * (1.0 - vis))
    }

    /// Legacy helper — per-channel transmittance only.
    pub fn fog_transmittance(&self, distance: f32) -> Vector3 {
        let (vis, _) = self.fog_at(distance);
        Vector3::new(vis, vis, vis)
    }
}

/// Per-triangle shading data. Positions live in [`RaytraceScene::tris`]; this
/// is everything else, in the same order.
#[derive(Debug, Clone, Copy)]
pub struct TriShading {
    /// World-space vertex normals. A geometry without a `normal` attribute gets
    /// the face normal in all three slots, which shades flat — the correct
    /// reading of "no normals were supplied".
    pub normals: [Vector3; 3],
    pub uvs: [Vector2; 3],
    /// Per-vertex colours, white when the geometry carries none.
    ///
    /// The raster path multiplies material colour by vertex colour; without
    /// these the traced image loses that channel entirely and a line set with a
    /// colour per segment comes out one flat shade. Costs 36 bytes a triangle,
    /// which is the honest price of not silently dropping an attribute the
    /// geometry supplied.
    pub colors: [Vector3; 3],
    pub material: u32,
}

/// One emitting triangle, with its slot in the sampling distribution.
#[derive(Debug, Clone, Copy)]
pub struct EmissiveTri {
    pub triangle: u32,
    pub area: f32,
    /// Cumulative selection weight, normalised so the last entry is 1.
    pub cdf: f32,
}

/// The set of emitting triangles, and the distribution used to pick one.
///
/// Weighted by `area × luminance`, so a small bright panel and a large dim one
/// are each sampled about as often as they matter. A textured emitter is
/// weighted by its constant term only — the estimator stays unbiased either
/// way, it just importance-samples a textured emitter less well.
#[derive(Debug, Clone, Default)]
pub struct EmissiveSet {
    pub tris: Vec<EmissiveTri>,
    /// `slot_of[triangle] = index into tris`, or `u32::MAX`. Needed on the
    /// other side of MIS: when a BSDF ray lands on an emitter by chance, the
    /// weight needs the probability light sampling *would* have chosen it.
    slot_of: Vec<u32>,
    total_weight: f32,
}

impl EmissiveSet {
    pub fn is_empty(&self) -> bool {
        self.tris.is_empty()
    }

    pub fn len(&self) -> usize {
        self.tris.len()
    }

    /// Pick an emitter by the area×luminance distribution. Returns the entry
    /// and the probability of having chosen it.
    pub fn sample(&self, u: f32) -> Option<(&EmissiveTri, f32)> {
        if self.tris.is_empty() {
            return None;
        }
        // Binary search the CDF; `partition_point` gives the first entry whose
        // cumulative weight exceeds u.
        let i = self
            .tris
            .partition_point(|e| e.cdf <= u)
            .min(self.tris.len() - 1);
        let prev = if i == 0 { 0.0 } else { self.tris[i - 1].cdf };
        let p = (self.tris[i].cdf - prev).max(1e-9);
        Some((&self.tris[i], p))
    }

    /// Probability that [`Self::sample`] would return triangle `tri`, or 0 if
    /// it does not emit.
    pub fn probability_of(&self, tri: u32) -> f32 {
        let slot = match self.slot_of.get(tri as usize) {
            Some(&s) if s != u32::MAX => s as usize,
            _ => return 0.0,
        };
        let prev = if slot == 0 {
            0.0
        } else {
            self.tris[slot - 1].cdf
        };
        (self.tris[slot].cdf - prev).max(0.0)
    }

    /// Total area×luminance across every emitter.
    pub fn total_weight(&self) -> f32 {
        self.total_weight
    }

    /// Which entry a triangle occupies, or `None` if it does not emit. This is
    /// the table [`Self::probability_of`] reads, exposed so a device backend can
    /// upload it instead of rebuilding the mapping by search.
    pub fn slot_of(&self, triangle: u32) -> Option<u32> {
        match self.slot_of.get(triangle as usize) {
            Some(&s) if s != u32::MAX => Some(s),
            _ => None,
        }
    }

    /// The whole slot table, one entry per scene triangle, `u32::MAX` where the
    /// triangle does not emit.
    pub fn slots(&self) -> &[u32] {
        &self.slot_of
    }
}

/// What the conversion did and could not do. Worth printing once after a build:
/// a black render is usually a missing light, and a flat one is usually a
/// material that had nowhere to map.
#[derive(Debug, Clone, Default)]
pub struct BuildReport {
    pub triangles: usize,
    pub meshes: usize,
    pub materials: usize,
    pub lights: usize,
    pub emissive_triangles: usize,
    pub textures_decoded: usize,
    /// Materials mapped to something the tracer can integrate, with a note on
    /// what was assumed. One entry per distinct material.
    pub approximated: Vec<String>,
    /// Objects left out entirely, with the reason.
    pub skipped: Vec<String>,
}

impl BuildReport {
    /// One-line summary, for a log or a progress line.
    pub fn summary(&self) -> String {
        format!(
            "{} triangles from {} meshes, {} materials, {} lights, {} emissive triangles, {} textures",
            self.triangles,
            self.meshes,
            self.materials,
            self.lights,
            self.emissive_triangles,
            self.textures_decoded,
        )
    }
}

/// A scene flattened for tracing.
#[derive(Debug, Clone, Default)]
pub struct RaytraceScene {
    /// World-space triangle vertices, denormalised.
    pub tris: Vec<[Vector3; 3]>,
    /// Shading attributes, parallel to `tris`.
    pub shading: Vec<TriShading>,
    pub materials: Vec<RtMaterial>,
    pub lights: Vec<RtLight>,
    pub emissive: EmissiveSet,
    pub world: World,
    pub bvh: RtBvh,
    pub report: BuildReport,
    /// True when no material can let a shadow ray through, which lets the
    /// occlusion test stop at the first hit instead of walking every layer.
    pub(crate) shadows_all_opaque: bool,
}


/// One mesh as the path tracer collects it: geometry, its world transform, and
/// the material it was drawn with.
type CollectedMesh = (Arc<crate::core::BufferGeometry>, Matrix4, Arc<Material>);

/// Triangles and their shading, keyed by mesh and by the material slot it was
/// drawn with — the same geometry under two materials is two entries.
type MeshCache = HashMap<(usize, u32), (Vec<[Vector3; 3]>, Vec<TriShading>)>;

impl RaytraceScene {
    /// Flatten `scene` for tracing.
    ///
    /// Takes `&mut Scene` for the same reason
    /// [`Renderer::render`](crate::renderer::Renderer::render) does: world
    /// matrices are lazily maintained, and tracing a graph whose transforms
    /// have not been resolved would silently render the bind pose.
    pub fn build(scene: &mut Scene, settings: &RaytraceSettings) -> Self {
        Self::build_for_layers(scene, settings, {
            let mut l = Layers::default();
            l.enable_all();
            l
        })
    }

    /// As [`Self::build`], but only objects whose layers the mask accepts —
    /// pass `camera.layers()` to match what that camera would rasterise.
    pub fn build_for_layers(
        scene: &mut Scene,
        settings: &RaytraceSettings,
        layers: Layers,
    ) -> Self {
        scene
            .arena
            .update_world_matrices(scene.root, Matrix4::identity());

        let mut out = RaytraceScene {
            world: World {
                background: v3(scene.background),
                background_alpha: scene.background_alpha,
                environment: scene.environment.clone(),
                environment_intensity: settings.environment_intensity,
                ambient: Vector3::ZERO,
                hemispheres: Vec::new(),
                mode: settings.background,
                env_faces: None,
                env_distribution: None,
                fog_color: v3(scene.fog.color),
                fog_mode: scene.fog.mode,
                fog_near: scene.fog.near,
                fog_far: scene.fog.far,
                fog_density: scene.fog.density,
            },
            ..Default::default()
        };

        let mut cache = TextureCache::new();
        // Materials are deduplicated by `Arc` identity, the same rule the GPU
        // renderer caches pipelines by — a hundred meshes sharing one material
        // produce one table entry and decode its textures once.
        let mut material_slots: HashMap<usize, u32> = HashMap::new();
        let mut collected: Vec<CollectedMesh> = Vec::new();
        let mut local_mesh_cache: MeshCache = HashMap::new();
        let light_layers = settings.light_layers;

        let root = scene.root;
        scene.arena.traverse_visible(root, &mut |_id, obj| {
            if !layers.test(&obj.layers) {
                return;
            }
            match &obj.kind {
                ObjectKind::Mesh(mesh) => {
                    collected.push((
                        mesh.geometry.clone(),
                        obj.matrix_world,
                        mesh.material.clone(),
                    ));
                }
                ObjectKind::SkinnedMesh(sm) => {
                    // Skinning is a vertex-shader stage in the raster path;
                    // here it has to happen on the CPU, once, before the BVH is
                    // built over the deformed positions.
                    match skin_geometry(&sm.geometry, &sm.skeleton) {
                        Some(geom) => {
                            collected.push((Arc::new(geom), obj.matrix_world, sm.material.clone()))
                        }
                        None => out.report.skipped.push(format!(
                            "skinned mesh '{}': missing joint/weight attributes",
                            obj.name
                        )),
                    }
                }
                ObjectKind::InstancedMesh(im) => {
                    for t in &im.transforms {
                        collected.push((
                            im.geometry.clone(),
                            obj.matrix_world.multiply(t),
                            im.material.clone(),
                        ));
                    }
                }
                ObjectKind::Light(light) => {
                    if !light_layers.test(&obj.layers) {
                        return;
                    }
                    match light {
                    Light::Ambient(l) => {
                        out.world.ambient = out.world.ambient + v3(l.color) * l.intensity;
                    }
                    Light::Hemisphere(l) => {
                        out.world.hemispheres.push(RtHemisphere {
                            sky: v3(l.sky_color) * l.intensity,
                            ground: v3(l.ground_color) * l.intensity,
                            up: transform_direction(&obj.matrix_world, Vector3::UP).normalize(),
                        });
                    }
                    Light::Directional(l) => {
                        // three.js resolves a directional light's direction as
                        // `normalize(target - position)` with the target at the
                        // origin by default; only when the light sits *at* the
                        // origin does its own `direction` field take over.
                        let pos = obj.world_position();
                        let d = Vector3::new(-pos.x, -pos.y, -pos.z);
                        let direction = if d.length_sq() > 1e-8 {
                            d.normalize()
                        } else {
                            transform_direction(&obj.matrix_world, l.direction).normalize()
                        };
                        out.lights.push(RtLight::Directional {
                            direction,
                            radiance: v3(l.color) * l.intensity,
                            angular_radius: settings.sun_angular_radius,
                        });
                    }
                    Light::Point(l) => out.lights.push(RtLight::Point {
                        position: obj.world_position(),
                        intensity: v3(l.color) * l.intensity,
                        radius: settings.light_radius,
                        distance: l.distance,
                        decay: l.decay,
                    }),
                    Light::Spot(l) => out.lights.push(RtLight::Spot {
                        position: obj.world_position(),
                        direction: transform_direction(&obj.matrix_world, l.direction).normalize(),
                        intensity: v3(l.color) * l.intensity,
                        radius: settings.light_radius,
                        distance: l.distance,
                        decay: l.decay,
                        cos_outer: l.angle.cos(),
                        cos_inner: (l.angle * (1.0 - l.penumbra)).cos(),
                    }),
                    Light::RectArea(l) => out.lights.push(RtLight::Rect {
                        position: obj.world_position(),
                        right: transform_direction(
                            &obj.matrix_world,
                            Vector3::new(l.width * 0.5, 0.0, 0.0),
                        ),
                        up: transform_direction(
                            &obj.matrix_world,
                            Vector3::new(0.0, l.height * 0.5, 0.0),
                        ),
                        radiance: v3(l.color) * l.intensity,
                    }),
                }
                }
                ObjectKind::LineSegments(ls) => {
                    let width = line_width_for(&ls.material);
                    if let Some(geom) = line_segment_quads(&ls.geometry, &obj.matrix_world, width)
                    {
                        collected.push((geom, Matrix4::identity(), ls.material.clone()));
                    } else {
                        out.report.skipped.push(format!(
                            "'{}': line segments could not be expanded",
                            obj.name
                        ));
                    }
                }
                ObjectKind::Points(p) => {
                    let radius = point_radius_for(&p.material);
                    if let Some(geom) = point_spheres(&p.geometry, &obj.matrix_world, radius) {
                        collected.push((geom, Matrix4::identity(), p.material.clone()));
                    } else {
                        out.report.skipped.push(format!("'{}': points could not be expanded", obj.name));
                    }
                }
                ObjectKind::Sprite(s) => {
                    let size = obj.world_scale().x.max(obj.world_scale().y).max(1.0);
                    let geom = sprite_quads(&obj.matrix_world, size);
                    collected.push((geom, Matrix4::identity(), s.material.clone()));
                }
                ObjectKind::Group => {}
            }
        });

        for (geometry, world, material) in collected {
            let slot = *material_slots
                .entry(Arc::as_ptr(&material) as usize)
                .or_insert_with(|| {
                    let (m, note) = convert_material(&material, &mut cache);
                    if let Some(note) = note {
                        out.report.approximated.push(note);
                    }
                    out.materials.push(m);
                    (out.materials.len() - 1) as u32
                });
            let before = out.tris.len();
            let cache_key = (Arc::as_ptr(&geometry) as usize, slot);
            let mat = out.materials[slot as usize].clone();
            if world == Matrix4::identity() {
                if let Some((tris, shading)) = local_mesh_cache.get(&cache_key) {
                    out.tris.extend_from_slice(tris);
                    out.shading.extend_from_slice(shading);
                } else {
                    out.append_geometry(&geometry, &world, slot, &mat);
                    local_mesh_cache.insert(
                        cache_key,
                        (
                            out.tris[before..].to_vec(),
                            out.shading[before..].to_vec(),
                        ),
                    );
                }
            } else {
                out.append_geometry(&geometry, &world, slot, &mat);
            }
            if out.tris.len() > before {
                out.report.meshes += 1;
            }
        }

        out.world.env_faces = out
            .world
            .environment
            .as_ref()
            .and_then(|c| decode_cube(c))
            .map(Box::new);
        // Built from the decoded faces, so it costs one pass over a 256x128
        // grid rather than a second decode. Only the *shape* matters, so
        // `environment_intensity` is deliberately left out: a constant factor
        // cannot change which direction is more likely.
        out.world.env_distribution = out
            .world
            .env_faces
            .as_ref()
            .and_then(|faces| EnvDistribution::build(|d| sample_cube(faces, d)));

        out.shadows_all_opaque = !out.materials.iter().any(|m| m.is_shadow_transparent());
        out.build_emissive_set();
        out.bvh = RtBvh::build(&out.tris);

        out.report.triangles = out.tris.len();
        out.report.materials = out.materials.len();
        out.report.lights = out.lights.len();
        out.report.emissive_triangles = out.emissive.len();
        out.report.textures_decoded = cache.len();
        out
    }

    /// Total world-space extent, used to pick sensible ray epsilons and to
    /// normalise the depth AOV.
    pub fn bounds(&self) -> (Vector3, Vector3) {
        self.bvh.bounds()
    }

    /// Length of the scene's diagonal, or 1 for an empty scene.
    pub fn scale(&self) -> f32 {
        let (min, max) = self.bounds();
        let d = (max - min).length();
        if d.is_finite() && d > 0.0 {
            d
        } else {
            1.0
        }
    }

    /// True when at least one thing can put light into the scene. A scene that
    /// fails this renders black however many samples you throw at it.
    pub fn has_light(&self) -> bool {
        !self.lights.is_empty() || !self.emissive.is_empty() || !self.world.is_black()
    }

    fn append_geometry(
        &mut self,
        geometry: &crate::core::BufferGeometry,
        world: &Matrix4,
        material: u32,
        mat: &RtMaterial,
    ) {
        let Some(pos) = geometry.get_attribute("position") else {
            return;
        };
        if pos.item_size < 3 {
            return;
        }
        let vertex_count = pos.count();
        let normal_attr = geometry
            .get_attribute("normal")
            .filter(|a| a.item_size >= 3);
        let uv_attr = geometry.get_attribute("uv").filter(|a| a.item_size >= 2);
        let normal_matrix = Matrix3::normal_matrix(world);

        let read_pos = |i: usize, uv: Vector2, n: Vector3| -> Vector3 {
            let b = i * pos.item_size;
            let mut p = Vector3::new(pos.array[b], pos.array[b + 1], pos.array[b + 2]);
            if let Some(map) = &mat.displacement_map {
                let h = map.sample(uv)[0] * mat.displacement_scale + mat.displacement_bias;
                p = p + n * h;
            }
            p.apply_matrix4(world)
        };
        let read_normal = |i: usize| -> Option<Vector3> {
            let a = normal_attr?;
            let b = i * a.item_size;
            let n = Vector3::new(a.array[b], a.array[b + 1], a.array[b + 2]);
            Some(apply_matrix3(&normal_matrix, n))
        };
        // Vertex colours, if the geometry supplied them. White otherwise, so a
        // geometry without the attribute multiplies through unchanged.
        let color_attr = geometry
            .attributes
            .get("color")
            .filter(|a| a.item_size >= 3);
        let read_color = |i: usize| -> Vector3 {
            match color_attr {
                Some(a) => {
                    let b = i * a.item_size;
                    Vector3::new(a.array[b], a.array[b + 1], a.array[b + 2])
                }
                None => Vector3::new(1.0, 1.0, 1.0),
            }
        };
        let read_uv = |i: usize| -> Vector2 {
            match uv_attr {
                Some(a) => {
                    let b = i * a.item_size;
                    Vector2::new(a.array[b], a.array[b + 1])
                }
                None => Vector2::new(0.0, 0.0),
            }
        };

        let push = |a: usize, b: usize, c: usize, s: &mut Self| {
            if a >= vertex_count || b >= vertex_count || c >= vertex_count {
                return;
            }
            let uvs = [read_uv(a), read_uv(b), read_uv(c)];
            let colors = [read_color(a), read_color(b), read_color(c)];
            let n = match (read_normal(a), read_normal(b), read_normal(c)) {
                (Some(na), Some(nb), Some(nc)) => [na, nb, nc],
                _ => {
                    let p0 = Vector3::new(
                        pos.array[a * pos.item_size],
                        pos.array[a * pos.item_size + 1],
                        pos.array[a * pos.item_size + 2],
                    )
                    .apply_matrix4(world);
                    let p1 = Vector3::new(
                        pos.array[b * pos.item_size],
                        pos.array[b * pos.item_size + 1],
                        pos.array[b * pos.item_size + 2],
                    )
                    .apply_matrix4(world);
                    let p2 = Vector3::new(
                        pos.array[c * pos.item_size],
                        pos.array[c * pos.item_size + 1],
                        pos.array[c * pos.item_size + 2],
                    )
                    .apply_matrix4(world);
                    let face = (p1 - p0).cross(p2 - p0);
                    if face.length_sq() <= 0.0 {
                        return;
                    }
                    let face = face.normalize();
                    [face; 3]
                }
            };
            let p = [
                read_pos(a, uvs[0], n[0]),
                read_pos(b, uvs[1], n[1]),
                read_pos(c, uvs[2], n[2]),
            ];
            let face = (p[1] - p[0]).cross(p[2] - p[0]);
            if face.length_sq() <= 0.0 || !face.x.is_finite() {
                return;
            }
            let face = face.normalize();
            let nn = [
                normalize_or(n[0], face),
                normalize_or(n[1], face),
                normalize_or(n[2], face),
            ];
            s.tris.push(p);
            s.shading.push(TriShading {
                normals: nn,
                uvs,
                colors,
                material,
            });
        };

        match &geometry.index {
            Some(idx) => {
                for t in idx.chunks_exact(3) {
                    push(t[0] as usize, t[1] as usize, t[2] as usize, self);
                }
            }
            None => {
                for i in 0..vertex_count / 3 {
                    push(i * 3, i * 3 + 1, i * 3 + 2, self);
                }
            }
        }
    }

    fn build_emissive_set(&mut self) {
        let mut slot_of = vec![u32::MAX; self.tris.len()];
        let mut tris: Vec<EmissiveTri> = Vec::new();
        let mut total = 0.0f32;
        for (i, sh) in self.shading.iter().enumerate() {
            let m = &self.materials[sh.material as usize];
            if !m.is_emissive() {
                continue;
            }
            let t = &self.tris[i];
            let area = 0.5 * (t[1] - t[0]).cross(t[2] - t[0]).length();
            if area <= 0.0 {
                continue;
            }
            // A textured emitter is weighted by its constant term alone. The
            // map only attenuates it, so this over-weights a mostly-black map —
            // which costs a little importance-sampling quality and nothing in
            // correctness, since the estimator divides by whatever density it
            // used.
            let w = area * luminance(m.emission);
            if w <= 0.0 {
                continue;
            }
            total += w;
            slot_of[i] = tris.len() as u32;
            tris.push(EmissiveTri {
                triangle: i as u32,
                area,
                cdf: total,
            });
        }
        if total > 0.0 {
            for e in &mut tris {
                e.cdf /= total;
            }
            // Guard the search: floating-point division can leave the last
            // entry a hair under 1, and a u of 0.9999999 would then fall off
            // the end.
            if let Some(last) = tris.last_mut() {
                last.cdf = 1.0;
            }
        }
        self.emissive = EmissiveSet {
            tris,
            slot_of,
            total_weight: total,
        };
    }
}

/// Map a three.js material onto the principled surface the tracer integrates.
/// The second return value is a note for [`crate::raytrace::scene::BuildReport::approximated`] when the
/// mapping involved a judgement call.
fn convert_material(material: &Material, cache: &mut TextureCache) -> (RtMaterial, Option<String>) {
    let mut tex = |t: &Option<Arc<crate::textures::Texture>>| t.as_ref().and_then(|t| cache.get(t));
    match material {
        Material::Standard(m) => (
            RtMaterial {
                base_color: v3(m.color),
                base_color_map: tex(&m.map),
                opacity: m.opacity,
                roughness: m.roughness,
                roughness_map: tex(&m.roughness_map),
                metallic: m.metalness,
                metallic_map: tex(&m.metalness_map),
                emission: v3(m.emissive) * m.emissive_intensity,
                emissive_map: tex(&m.emissive_map),
                normal_map: tex(&m.normal_map),
                normal_scale: m.normal_scale,
                ..Default::default()
            },
            None,
        ),
        Material::Physical(m) => (
            RtMaterial {
                base_color: v3(m.color),
                base_color_map: tex(&m.map),
                opacity: m.opacity,
                roughness: m.roughness,
                roughness_map: tex(&m.roughness_map),
                metallic: m.metalness,
                metallic_map: tex(&m.metalness_map),
                emission: v3(m.emissive) * m.emissive_intensity,
                emissive_map: tex(&m.emissive_map),
                normal_map: tex(&m.normal_map),
                normal_scale: m.normal_scale,
                transmission: m.transmission,
                ior: m.ior,
                attenuation_color: v3(m.attenuation_color),
                attenuation_distance: m.attenuation_distance,
                clearcoat: m.clearcoat,
                clearcoat_roughness: m.clearcoat_roughness,
                anisotropy: m.anisotropy,
                anisotropy_rotation: m.anisotropy_rotation,
                sheen: m.sheen,
                sheen_color: v3(m.sheen_color),
                sheen_roughness: m.sheen_roughness,
                iridescence: m.iridescence,
                iridescence_ior: m.iridescence_ior,
                iridescence_thickness: m.iridescence_thickness,
                iridescence_thickness_map: tex(&m.iridescence_thickness_map),
                dispersion: m.dispersion,
                displacement_map: tex(&m.displacement_map),
                displacement_scale: m.displacement_scale,
                displacement_bias: m.displacement_bias,
                ..Default::default()
            },
            unsupported_physical(m),
        ),
        Material::Lambert(m) => (
            RtMaterial {
                base_color: v3(m.color),
                opacity: m.opacity,
                roughness: 1.0,
                metallic: 0.0,
                emission: v3(m.emissive),
                ..Default::default()
            },
            None,
        ),
        Material::Phong(m) => {
            // Blinn-Phong's exponent and GGX's roughness describe the same lobe
            // width differently. Equating the two at their half-widths gives
            // α = sqrt(2 / (shininess + 2)), and roughness is sqrt(α).
            let alpha = (2.0 / (m.shininess.max(0.0) + 2.0)).sqrt();
            (
                RtMaterial {
                    base_color: v3(m.color),
                    opacity: m.opacity,
                    roughness: alpha.sqrt().clamp(0.02, 1.0),
                    metallic: 0.0,
                    specular_tint: v3(m.specular),
                    emission: v3(m.emissive),
                    ..Default::default()
                },
                Some(format!(
                    "MeshPhongMaterial: shininess {} mapped to GGX roughness {:.3}, and `specular` \
                     becomes the dielectric F0 tint",
                    m.shininess,
                    alpha.sqrt()
                )),
            )
        }
        Material::Toon(m) => (
            RtMaterial {
                base_color: v3(m.color),
                opacity: m.opacity,
                roughness: 1.0,
                ..Default::default()
            },
            Some("MeshToonMaterial: the ramp is a raster effect; traced as a matte diffuse surface"
                .into()),
        ),
        Material::Matcap(m) => (
            RtMaterial {
                base_color: v3(m.color),
                opacity: m.opacity,
                roughness: 0.5,
                ..Default::default()
            },
            Some(
                "MeshMatcapMaterial: the capture bakes in a light rig; traced as a mid-gloss dielectric"
                    .into(),
            ),
        ),
        Material::Basic(m) => (
            RtMaterial {
                base_color: Vector3::ZERO,
                // The map lands in both slots: it colours the emission, and its
                // alpha is what the cutout test reads.
                base_color_map: tex(&m.map),
                emission: v3(m.color),
                emissive_map: tex(&m.map),
                opacity: m.opacity,
                alpha_test: m.alpha_test,
                unlit: true,
                ..Default::default()
            },
            Some(
                "MeshBasicMaterial: unlit, so traced as a pure emitter of its own colour — it lights the scene"
                    .into(),
            ),
        ),
        Material::Mirror(m) => (
            RtMaterial {
                base_color: v3(m.color),
                roughness: 0.0,
                metallic: 1.0,
                ..Default::default()
            },
            Some("MirrorMaterial: traced as a smooth metal instead of a projected render target".into()),
        ),
        Material::Normal(_) => (
            RtMaterial {
                base_color: Vector3::new(0.5, 0.5, 0.5),
                roughness: 1.0,
                ..Default::default()
            },
            Some("MeshNormalMaterial: a debug view with no BSDF; traced as 50% grey".into()),
        ),
        Material::Depth(_) => (
            RtMaterial {
                base_color: Vector3::new(0.5, 0.5, 0.5),
                roughness: 1.0,
                ..Default::default()
            },
            Some("MeshDepthMaterial: a debug view with no BSDF; traced as 50% grey".into()),
        ),
        // A line material is unlit by definition — that is what
        // `LineBasicMaterial` means, and what the raster path does with it. It
        // was traced as a plain diffuse surface, which in a scene with no
        // lights resolves to black: the geometry is present, gets hit, and
        // returns nothing. Emitting its own colour instead matches the raster
        // result and needs no light to be visible.
        Material::Line(m) => (
            RtMaterial {
                base_color: Vector3::ZERO,
                emission: Vector3::new(
                    m.color.r * m.opacity.max(0.0),
                    m.color.g * m.opacity.max(0.0),
                    m.color.b * m.opacity.max(0.0),
                ),
                unlit: true,
                ..Default::default()
            },
            Some("line material: traced as unlit emission, as the raster path draws it".into()),
        ),
        // Points and sprites are unlit too, for the same reason lines are: the
        // raster path draws them as flat colour, and tracing them as diffuse
        // makes them black in any scene without lights.
        Material::Points(m) => (
            RtMaterial {
                base_color: Vector3::ZERO,
                emission: Vector3::new(
                    m.color.r * m.opacity.max(0.0),
                    m.color.g * m.opacity.max(0.0),
                    m.color.b * m.opacity.max(0.0),
                ),
                unlit: true,
                ..Default::default()
            },
            Some("points material: traced as unlit emission, as the raster path draws it".into()),
        ),
        Material::Sprite(m) => (
            RtMaterial {
                base_color: Vector3::ZERO,
                emission: Vector3::new(
                    m.color.r * m.opacity.max(0.0),
                    m.color.g * m.opacity.max(0.0),
                    m.color.b * m.opacity.max(0.0),
                ),
                unlit: true,
                ..Default::default()
            },
            Some("sprite material: traced as unlit emission, as the raster path draws it".into()),
        ),
        // A distance material is a depth debug view, not a surface.
        Material::Distance(_) => (
            RtMaterial {
                base_color: Vector3::new(0.8, 0.8, 0.8),
                ..Default::default()
            },
            Some("distance material on a mesh: traced as a plain diffuse surface".into()),
        ),
        Material::Sky(_) => (
            RtMaterial {
                base_color: Vector3::ZERO,
                emission: Vector3::new(0.4, 0.6, 1.0),
                unlit: true,
                ..Default::default()
            },
            Some(
                "SkyMaterial: the Preetham model is not evaluated here; traced as a flat blue emitter"
                    .into(),
            ),
        ),
        Material::Atmosphere(_) => (
            RtMaterial {
                base_color: Vector3::new(0.6, 0.75, 1.0),
                transmission: 1.0,
                ior: 1.0003,
                roughness: 0.0,
                ..Default::default()
            },
            Some(
                "AtmosphereMaterial: analytic scattering is not evaluated here; traced as a clear tinted shell"
                    .into(),
            ),
        ),
        Material::Shader(_) => (
            RtMaterial {
                base_color: Vector3::new(0.8, 0.8, 0.8),
                roughness: 0.8,
                ..Default::default()
            },
            Some(
                "ShaderMaterial: custom WGSL cannot run on the CPU tracer; traced as a rough dielectric"
                    .into(),
            ),
        ),
    }
}

/// Name the `MeshPhysicalMaterial` features this BSDF does not model, or `None`
/// if the material uses none of them.
fn unsupported_physical(_m: &crate::materials::PhysicalMaterial) -> Option<String> {
    None
}

/// Apply the skeleton on the CPU, producing a deformed copy of the geometry in
/// its own local space (the object's world matrix is applied afterwards, as for
/// any other mesh).
fn skin_geometry(
    geometry: &crate::core::BufferGeometry,
    skeleton: &crate::core::Skeleton,
) -> Option<crate::core::BufferGeometry> {
    let pos = geometry.get_attribute("position")?;
    let joints = geometry.get_attribute("joint")?;
    let weights = geometry.get_attribute("weight")?;
    if joints.item_size < 4 || weights.item_size < 4 || pos.item_size < 3 {
        return None;
    }
    let n = pos.count();
    let normals = geometry
        .get_attribute("normal")
        .filter(|a| a.item_size >= 3);
    let mut out_pos = vec![0.0f32; n * 3];
    let mut out_nrm = normals.map(|_| vec![0.0f32; n * 3]);

    for i in 0..n {
        let p = Vector3::new(
            pos.array[i * pos.item_size],
            pos.array[i * pos.item_size + 1],
            pos.array[i * pos.item_size + 2],
        );
        let nrm = normals.map(|a| {
            Vector3::new(
                a.array[i * a.item_size],
                a.array[i * a.item_size + 1],
                a.array[i * a.item_size + 2],
            )
        });
        let mut skinned = Vector3::ZERO;
        let mut skinned_n = Vector3::ZERO;
        let mut total = 0.0f32;
        for k in 0..4 {
            let w = weights.array[i * weights.item_size + k];
            if w == 0.0 {
                continue;
            }
            let j = joints.array[i * joints.item_size + k] as usize;
            let Some(m) = skeleton.bone_matrices.get(j) else {
                continue;
            };
            skinned = skinned + p.apply_matrix4(m) * w;
            if let Some(nrm) = nrm {
                skinned_n = skinned_n + apply_matrix3(&Matrix3::normal_matrix(m), nrm) * w;
            }
            total += w;
        }
        // Unweighted vertices stay where they were rather than collapsing to
        // the origin.
        let (p_out, n_out) = if total > 0.0 {
            (skinned * (1.0 / total), skinned_n * (1.0 / total))
        } else {
            (p, nrm.unwrap_or(Vector3::UP))
        };
        out_pos[i * 3] = p_out.x;
        out_pos[i * 3 + 1] = p_out.y;
        out_pos[i * 3 + 2] = p_out.z;
        if let Some(o) = &mut out_nrm {
            let n_out = normalize_or(n_out, Vector3::UP);
            o[i * 3] = n_out.x;
            o[i * 3 + 1] = n_out.y;
            o[i * 3 + 2] = n_out.z;
        }
    }

    let mut g = crate::core::BufferGeometry::new();
    g.set_attribute("position", crate::core::BufferAttribute::new(out_pos, 3));
    if let Some(o) = out_nrm {
        g.set_attribute("normal", crate::core::BufferAttribute::new(o, 3));
    }
    if let Some(uv) = geometry.get_attribute("uv") {
        g.set_attribute("uv", uv.clone());
    }
    if let Some(idx) = &geometry.index {
        g.set_index(idx.clone());
    }
    Some(g)
}

/// Decode all six faces of an environment cube.
///
/// `faces_f32` first when it is there. That field is the HDR input — the 8-bit
/// `faces` beside it is a Reinhard-compressed, gamma-encoded copy kept so that
/// display paths keep working, and integrating *that* would throw away the
/// several thousand to one range between an HDRI's sun and its sky, which is
/// the whole reason to light a scene with one.
///
/// Returns `None` if any face cannot be decoded: a cube missing a face would
/// leak black into one sixth of the sky.
fn decode_cube(cube: &CubeTexture) -> Option<[CpuTexture; 6]> {
    let bilinear = cube.mag_filter == crate::textures::TextureFilter::Linear;
    let mut faces = Vec::with_capacity(6);
    for i in 0..6 {
        let decoded = match &cube.faces_f32 {
            Some(hdr) => {
                CpuTexture::from_linear_f32(cube.size, cube.size, hdr[i].as_ref().clone(), bilinear)
            }
            None => {
                let mut tex = crate::textures::Texture::new(
                    cube.size,
                    cube.size,
                    cube.format,
                    cube.faces[i].as_ref().clone(),
                );
                tex.mag_filter = cube.mag_filter;
                tex.min_filter = cube.min_filter;
                tex.wrap_s = crate::textures::TextureWrap::ClampToEdge;
                tex.wrap_t = crate::textures::TextureWrap::ClampToEdge;
                // Face data is stored top-row-first and the direction mapping
                // below assumes that, so no flip.
                tex.flip_y = false;
                CpuTexture::from_texture(&tex)
            }
        };
        faces.push(decoded?);
    }
    let mut it = faces.into_iter();
    Some([
        it.next()?,
        it.next()?,
        it.next()?,
        it.next()?,
        it.next()?,
        it.next()?,
    ])
}

/// Sample a decoded cube along `dir`.
///
/// The mapping is the inverse of the one
/// [`PmremGenerator::from_equirect_f32`](crate::extras::PmremGenerator::from_equirect_f32)
/// writes with, which is the standard cube-map convention — the same one the
/// hardware sampler uses when the raster renderer binds a plain
/// `CubeTexture`, and therefore the convention the bytes in
/// [`CubeTexture::faces`](crate::textures::CubeTexture::faces) are actually
/// stored in.
///
/// It is deliberately *not* three.js's CubeUV convention. That one applies to
/// the PMREM atlas, which is a different layout reached through a blit that
/// flips rows and swaps ±X — so reading `faces` as if it were CubeUV rotates
/// every face by half a turn, which is invisible on any test that only checks
/// which face a direction lands on.
fn sample_cube(faces: &[CpuTexture; 6], dir: Vector3) -> Vector3 {
    let (ax, ay, az) = (dir.x.abs(), dir.y.abs(), dir.z.abs());
    let (face, sc, tc, ma) = if ax >= ay && ax >= az {
        if dir.x > 0.0 {
            (0usize, -dir.z, -dir.y, ax)
        } else {
            (1, dir.z, -dir.y, ax)
        }
    } else if ay >= az {
        if dir.y > 0.0 {
            (2, dir.x, dir.z, ay)
        } else {
            (3, dir.x, -dir.z, ay)
        }
    } else if dir.z > 0.0 {
        (4, dir.x, -dir.y, az)
    } else {
        (5, -dir.x, -dir.y, az)
    };
    if ma <= 0.0 {
        return Vector3::ZERO;
    }
    let uv = Vector2::new((sc / ma + 1.0) * 0.5, (tc / ma + 1.0) * 0.5);
    if !uv.x.is_finite() || !uv.y.is_finite() {
        return Vector3::ZERO;
    }
    faces[face].sample_rgb(uv)
}

fn v3(c: Color) -> Vector3 {
    Vector3::new(c.r, c.g, c.b)
}

/// Rec. 709 luminance — the weighting used to turn an emitter's colour into a
/// single sampling weight.
pub fn luminance(c: Vector3) -> f32 {
    0.2126 * c.x + 0.7152 * c.y + 0.0722 * c.z
}

fn normalize_or(v: Vector3, fallback: Vector3) -> Vector3 {
    if v.length_sq() > 1e-20 {
        v.normalize()
    } else {
        fallback
    }
}

/// Rotate a direction by a matrix's upper 3×3, ignoring translation.
fn transform_direction(m: &Matrix4, v: Vector3) -> Vector3 {
    let e = &m.elements;
    Vector3::new(
        e[0] * v.x + e[4] * v.y + e[8] * v.z,
        e[1] * v.x + e[5] * v.y + e[9] * v.z,
        e[2] * v.x + e[6] * v.y + e[10] * v.z,
    )
}

fn apply_matrix3(m: &Matrix3, v: Vector3) -> Vector3 {
    let e = &m.elements;
    Vector3::new(
        e[0] * v.x + e[3] * v.y + e[6] * v.z,
        e[1] * v.x + e[4] * v.y + e[7] * v.z,
        e[2] * v.x + e[5] * v.y + e[8] * v.z,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::Object3D;
    use crate::geometries::BoxGeometry;
    use crate::lights::{AmbientLight, DirectionalLight, PointLight};
    use crate::materials::StandardMaterial;

    fn box_mesh(color: Color) -> crate::core::Mesh {
        crate::core::Mesh::new(
            BoxGeometry::new(1.0, 1.0, 1.0),
            Material::Standard(StandardMaterial::new(color)),
        )
    }

    #[test]
    fn a_box_flattens_to_twelve_triangles() {
        let mut scene = Scene::new();
        scene.add(Object3D::mesh(box_mesh(Color::WHITE)));
        let rt = RaytraceScene::build(&mut scene, &RaytraceSettings::default());
        assert_eq!(rt.tris.len(), 12);
        assert_eq!(rt.shading.len(), 12);
        assert_eq!(rt.materials.len(), 1);
        assert_eq!(rt.report.meshes, 1);
    }

    #[test]
    fn world_matrices_are_applied() {
        let mut scene = Scene::new();
        let mut obj = Object3D::mesh(box_mesh(Color::WHITE));
        obj.position = Vector3::new(10.0, 0.0, 0.0);
        scene.add(obj);
        let rt = RaytraceScene::build(&mut scene, &RaytraceSettings::default());
        let (min, max) = rt.bounds();
        assert!(min.x > 9.0, "min {min:?}");
        assert!(max.x < 11.0, "max {max:?}");
    }

    #[test]
    fn shared_materials_collapse_to_one_slot() {
        let mut scene = Scene::new();
        let geom = Arc::new(BoxGeometry::new(1.0, 1.0, 1.0));
        let mat = Arc::new(Material::Standard(StandardMaterial::new(Color::RED)));
        for i in 0..5 {
            let mut obj = Object3D::mesh(crate::core::Mesh::from_arc(geom.clone(), mat.clone()));
            obj.position = Vector3::new(i as f32 * 3.0, 0.0, 0.0);
            scene.add(obj);
        }
        let rt = RaytraceScene::build(&mut scene, &RaytraceSettings::default());
        assert_eq!(rt.materials.len(), 1, "one Arc, one material slot");
        assert_eq!(rt.tris.len(), 60);
    }

    #[test]
    fn instanced_meshes_expand() {
        let mut scene = Scene::new();
        let mut im = crate::core::InstancedMesh::new(
            BoxGeometry::new(1.0, 1.0, 1.0),
            Material::Standard(StandardMaterial::new(Color::WHITE)),
            4,
        );
        for i in 0..4 {
            im.set_matrix_at(
                i,
                Matrix4::translation(Vector3::new(i as f32 * 2.0, 0.0, 0.0)),
            );
        }
        scene.add(Object3D::instanced_mesh(im));
        let rt = RaytraceScene::build(&mut scene, &RaytraceSettings::default());
        assert_eq!(rt.tris.len(), 48);
        let (min, max) = rt.bounds();
        assert!(
            max.x - min.x > 6.0,
            "instances did not spread: {min:?}..{max:?}"
        );
    }

    #[test]
    fn lights_are_collected_with_three_js_conventions() {
        let mut scene = Scene::new();
        let mut dir = Object3D::light(DirectionalLight::new(Color::WHITE, 1.0));
        dir.position = Vector3::new(0.0, 10.0, 0.0);
        scene.add(dir);
        let mut pt = Object3D::light(PointLight::new(Color::RED, 2.0));
        pt.position = Vector3::new(3.0, 0.0, 0.0);
        scene.add(pt);
        scene.add_light(AmbientLight::new(Color::WHITE, 0.25));

        let rt = RaytraceScene::build(&mut scene, &RaytraceSettings::default());
        assert_eq!(rt.lights.len(), 2, "ambient is world lighting, not a light");
        assert!((rt.world.ambient.x - 0.25).abs() < 1e-6);

        match rt.lights[0] {
            RtLight::Directional { direction, .. } => {
                // Light above the origin shines straight down.
                assert!((direction.y + 1.0).abs() < 1e-5, "got {direction:?}");
            }
            other => panic!("expected a directional light, got {other:?}"),
        }
        match rt.lights[1] {
            RtLight::Point {
                position,
                intensity,
                ..
            } => {
                assert!((position.x - 3.0).abs() < 1e-5);
                assert!((intensity.x - 2.0).abs() < 1e-5);
            }
            other => panic!("expected a point light, got {other:?}"),
        }
    }

    #[test]
    fn emissive_surfaces_enter_the_sampling_distribution() {
        let mut scene = Scene::new();
        let mut mat = StandardMaterial::new(Color::BLACK);
        mat.emissive = Color::WHITE;
        mat.emissive_intensity = 5.0;
        scene.add(Object3D::mesh(crate::core::Mesh::new(
            BoxGeometry::new(1.0, 1.0, 1.0),
            Material::Standard(mat),
        )));
        scene.add(Object3D::mesh(box_mesh(Color::WHITE)));

        let rt = RaytraceScene::build(&mut scene, &RaytraceSettings::default());
        assert_eq!(rt.emissive.len(), 12, "only the emitting box's faces");
        assert!(rt.emissive.total_weight() > 0.0);
        // The CDF must be sorted, end at 1, and every entry must be findable.
        let mut prev = 0.0;
        for e in &rt.emissive.tris {
            assert!(e.cdf >= prev, "cdf not monotonic");
            prev = e.cdf;
        }
        assert!((prev - 1.0).abs() < 1e-6);
        for e in &rt.emissive.tris {
            assert!(rt.emissive.probability_of(e.triangle) > 0.0);
        }
    }

    #[test]
    fn emissive_sampling_hits_every_slot() {
        let mut scene = Scene::new();
        let mut mat = StandardMaterial::new(Color::BLACK);
        mat.emissive = Color::WHITE;
        scene.add(Object3D::mesh(crate::core::Mesh::new(
            BoxGeometry::new(1.0, 1.0, 1.0),
            Material::Standard(mat),
        )));
        let rt = RaytraceScene::build(&mut scene, &RaytraceSettings::default());
        let mut seen = vec![false; rt.emissive.len()];
        for i in 0..1000 {
            let u = i as f32 / 1000.0;
            let (e, p) = rt.emissive.sample(u).unwrap();
            assert!(p > 0.0);
            let slot = rt
                .emissive
                .tris
                .iter()
                .position(|t| t.triangle == e.triangle)
                .unwrap();
            seen[slot] = true;
        }
        assert!(seen.iter().all(|&s| s), "some emitter is unreachable");
    }

    #[test]
    fn displacement_is_baked_at_build_time() {
        let mut scene = Scene::new();
        let mut m = crate::materials::PhysicalMaterial::new(Color::WHITE);
        m.displacement_map = Some(Arc::new(crate::textures::Texture::new(
            2,
            2,
            crate::textures::TextureFormat::R8Unorm,
            vec![255u8; 4],
        )));
        m.displacement_scale = 0.5;
        scene.add(Object3D::mesh(crate::core::Mesh::new(
            BoxGeometry::new(1.0, 1.0, 1.0),
            Material::Physical(m),
        )));
        let rt = RaytraceScene::build(&mut scene, &RaytraceSettings::default());
        assert!(
            rt.report.approximated.is_empty(),
            "displacement is baked into geometry: {:?}",
            rt.report.approximated
        );
        assert!(rt.materials[0].displacement_map.is_some());
    }

    #[test]
    fn physical_material_features_are_simulated_not_approximated() {
        // Sheen and iridescence are simulated — not approximated.
        let mut scene = Scene::new();
        let mut m = crate::materials::PhysicalMaterial::new(Color::WHITE);
        m.sheen = 0.8;
        m.iridescence = 0.4;
        scene.add(Object3D::mesh(crate::core::Mesh::new(
            BoxGeometry::new(1.0, 1.0, 1.0),
            Material::Physical(m),
        )));
        let rt = RaytraceScene::build(&mut scene, &RaytraceSettings::default());
        assert!(
            rt.report.approximated.is_empty(),
            "simulated features must not be approximated: {:?}",
            rt.report.approximated
        );
        assert!((rt.materials[0].sheen - 0.8).abs() < 1e-6);
        assert!((rt.materials[0].iridescence - 0.4).abs() < 1e-6);

        // A plain physical material has nothing to report.
        let mut scene = Scene::new();
        scene.add(Object3D::mesh(crate::core::Mesh::new(
            BoxGeometry::new(1.0, 1.0, 1.0),
            Material::Physical(crate::materials::PhysicalMaterial::new(Color::WHITE)),
        )));
        let rt = RaytraceScene::build(&mut scene, &RaytraceSettings::default());
        assert!(
            rt.report.approximated.is_empty(),
            "{:?}",
            rt.report.approximated
        );
    }

    #[test]
    fn basic_material_becomes_an_emitter() {
        let mut scene = Scene::new();
        scene.add(Object3D::mesh(crate::core::Mesh::new(
            BoxGeometry::new(1.0, 1.0, 1.0),
            Material::Basic(crate::materials::BasicMaterial::new(Color::GREEN)),
        )));
        let rt = RaytraceScene::build(&mut scene, &RaytraceSettings::default());
        assert!(rt.materials[0].unlit);
        assert!(rt.materials[0].emission.y > 0.9);
        assert!(
            !rt.report.approximated.is_empty(),
            "the choice must be reported"
        );
    }

    #[test]
    fn empty_scene_is_valid_and_unlit() {
        let mut scene = Scene::new();
        let rt = RaytraceScene::build(&mut scene, &RaytraceSettings::default());
        assert_eq!(rt.tris.len(), 0);
        assert!(!rt.has_light());
        assert_eq!(rt.scale(), 1.0);
    }

    #[test]
    fn degenerate_triangles_are_dropped() {
        let mut g = crate::core::BufferGeometry::new();
        g.set_attribute(
            "position",
            crate::core::BufferAttribute::new(
                vec![
                    // A real triangle.
                    0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, //
                    // Three collinear points.
                    0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 2.0, 0.0, 0.0,
                ],
                3,
            ),
        );
        let mut scene = Scene::new();
        scene.add(Object3D::mesh(crate::core::Mesh::new(
            g,
            Material::Standard(StandardMaterial::new(Color::WHITE)),
        )));
        let rt = RaytraceScene::build(&mut scene, &RaytraceSettings::default());
        assert_eq!(rt.tris.len(), 1);
    }

    #[test]
    fn geometry_without_normals_shades_flat() {
        let mut g = crate::core::BufferGeometry::new();
        g.set_attribute(
            "position",
            crate::core::BufferAttribute::new(vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0], 3),
        );
        let mut scene = Scene::new();
        scene.add(Object3D::mesh(crate::core::Mesh::new(
            g,
            Material::Standard(StandardMaterial::new(Color::WHITE)),
        )));
        let rt = RaytraceScene::build(&mut scene, &RaytraceSettings::default());
        let n = rt.shading[0].normals;
        assert!((n[0] - n[1]).length() < 1e-6 && (n[1] - n[2]).length() < 1e-6);
        assert!(
            (n[0].z.abs() - 1.0).abs() < 1e-5,
            "expected +/-Z, got {:?}",
            n[0]
        );
    }

    #[test]
    fn ambient_only_world_still_counts_as_lit() {
        let mut scene = Scene::new();
        scene.add(Object3D::mesh(box_mesh(Color::WHITE)));
        scene.add_light(AmbientLight::new(Color::WHITE, 0.5));
        let rt = RaytraceScene::build(&mut scene, &RaytraceSettings::default());
        assert!(rt.has_light());
        let l = rt.world.lighting_radiance(Vector3::UP);
        assert!((l.x - 0.5).abs() < 1e-6);
    }

    #[test]
    fn background_mode_does_not_change_lighting() {
        let mut scene = Scene::new();
        scene.background = Color::from_hex(0xff0000);
        scene.add_light(AmbientLight::new(Color::WHITE, 0.25));
        let s = RaytraceSettings {
            background: BackgroundMode::Transparent,
            ..Default::default()
        };
        let rt = RaytraceScene::build(&mut scene, &s);
        let (bg, alpha) = rt.world.camera_radiance(Vector3::UP);
        assert_eq!(alpha, 0.0);
        assert_eq!(bg, Vector3::ZERO);
        // Lighting is unaffected.
        assert!((rt.world.lighting_radiance(Vector3::UP).x - 0.25).abs() < 1e-6);
    }

    #[test]
    fn cube_sampling_recovers_the_face_a_direction_points_at() {
        // Six faces, each a distinct constant, in CubeTexture order.
        let size = 4u32;
        let mut faces: [Vec<u8>; 6] = Default::default();
        for (i, f) in faces.iter_mut().enumerate() {
            let mut px = vec![0u8; (size * size * 4) as usize];
            for p in px.chunks_exact_mut(4) {
                p[0] = (i * 40) as u8;
                p[1] = (i * 40) as u8;
                p[2] = (i * 40) as u8;
                p[3] = 255;
            }
            *f = px;
        }
        let cube = CubeTexture::new(size, crate::textures::TextureFormat::Rgba8Unorm, faces);
        let decoded = decode_cube(&cube).unwrap();
        // CubeTexture order: +X, -X, +Y, -Y, +Z, -Z.
        let expect = [
            (Vector3::new(1.0, 0.0, 0.0), 0),
            (Vector3::new(-1.0, 0.0, 0.0), 1),
            (Vector3::new(0.0, 1.0, 0.0), 2),
            (Vector3::new(0.0, -1.0, 0.0), 3),
            (Vector3::new(0.0, 0.0, 1.0), 4),
            (Vector3::new(0.0, 0.0, -1.0), 5),
        ];
        for (dir, face) in expect {
            let got = sample_cube(&decoded, dir);
            let want = (face * 40) as f32 / 255.0;
            assert!(
                (got.x - want).abs() < 1e-3,
                "direction {dir:?} should read face {face} ({want}), got {got:?}"
            );
        }
    }

    /// An opaque texture must not put every shadow ray in the scene on the slow
    /// attenuating walk. Only a map that actually contains transparency should.
    #[test]
    fn an_opaque_texture_keeps_the_fast_shadow_path() {
        let opaque = Arc::new(crate::textures::Texture::new(
            2,
            2,
            crate::textures::TextureFormat::Rgba8Unorm,
            vec![255u8; 16],
        ));
        let mut cutout_data = vec![255u8; 16];
        cutout_data[3] = 0;
        let cutout = Arc::new(crate::textures::Texture::new(
            2,
            2,
            crate::textures::TextureFormat::Rgba8Unorm,
            cutout_data,
        ));

        for (map, expect_opaque) in [(opaque, true), (cutout, false)] {
            let mut scene = Scene::new();
            let mut m = StandardMaterial::new(Color::WHITE);
            m.map = Some(map);
            scene.add(Object3D::mesh(crate::core::Mesh::new(
                BoxGeometry::new(1.0, 1.0, 1.0),
                Material::Standard(m),
            )));
            let rt = RaytraceScene::build(&mut scene, &RaytraceSettings::default());
            assert_eq!(
                rt.shadows_all_opaque, expect_opaque,
                "textured material classified wrongly"
            );
        }
    }

    /// three.js multiplies `emissiveMap` by `emissive`, which defaults to black.
    /// Counting such a material as a light spends next-event samples on a
    /// surface that returns zero every time.
    #[test]
    fn an_emissive_map_without_an_emissive_colour_is_not_a_light() {
        let map = Arc::new(crate::textures::Texture::new(
            2,
            2,
            crate::textures::TextureFormat::Rgba8Unorm,
            vec![255u8; 16],
        ));
        let mut scene = Scene::new();
        let mut m = StandardMaterial::new(Color::WHITE);
        m.emissive_map = Some(map.clone());
        // emissive stays at its default, black.
        scene.add(Object3D::mesh(crate::core::Mesh::new(
            BoxGeometry::new(1.0, 1.0, 1.0),
            Material::Standard(m),
        )));
        let rt = RaytraceScene::build(&mut scene, &RaytraceSettings::default());
        assert!(rt.emissive.is_empty(), "black emissive should emit nothing");

        // Give it a colour and it becomes one.
        let mut scene = Scene::new();
        let mut m = StandardMaterial::new(Color::WHITE);
        m.emissive_map = Some(map);
        m.emissive = Color::WHITE;
        scene.add(Object3D::mesh(crate::core::Mesh::new(
            BoxGeometry::new(1.0, 1.0, 1.0),
            Material::Standard(m),
        )));
        let rt = RaytraceScene::build(&mut scene, &RaytraceSettings::default());
        assert_eq!(rt.emissive.len(), 12);
    }

    /// A round trip through the equirect -> cube conversion the HDR
    /// environment path uses.
    ///
    /// The map encodes its own coordinates, so a sampled direction can be
    /// checked against the direction it should have come from. This catches
    /// three things at once that a face-identity test cannot: the within-face
    /// orientation (a half-turn per face leaves every face centre where it
    /// was), the latitude mapping, and whether the HDR faces are read at all.
    #[test]
    fn an_hdr_environment_round_trips_through_the_cube() {
        let (w, h) = (256u32, 128u32);
        let mut src = vec![0.0f32; (w * h * 4) as usize];
        for y in 0..h {
            for x in 0..w {
                let i = ((y * w + x) * 4) as usize;
                src[i] = (x as f32 + 0.5) / w as f32; // red  = longitude
                src[i + 1] = (y as f32 + 0.5) / h as f32; // green = latitude
                src[i + 3] = 1.0;
            }
        }
        let cube = crate::extras::PmremGenerator::from_equirect_f32(&src, w, h, 64);
        let mut scene = Scene::new();
        scene.environment = Some(Arc::new(cube));
        let rt = RaytraceScene::build(&mut scene, &RaytraceSettings::default());

        // Every face, and off-centre on each — a half-turn per face leaves the
        // centres alone. Nothing sits on the longitude seam at ±pi: the red
        // channel encodes longitude and is discontinuous there, so a bilinear
        // tap across it averages 0.99 and 0.01 into a meaningless 0.5.
        for d in [
            Vector3::new(1.0, 0.2, -0.3).normalize(),
            Vector3::new(-1.0, -0.25, 0.4).normalize(),
            Vector3::new(0.15, 0.2, 1.0).normalize(),
            Vector3::new(0.3, -0.2, -1.0).normalize(),
            Vector3::new(0.89, 0.38, 0.25).normalize(),
            Vector3::new(-0.40, 0.80, 0.45).normalize(),
            Vector3::new(0.3, -0.9, -0.31).normalize(),
        ] {
            let lon = d.z.atan2(d.x);
            let lat = d.y.clamp(-1.0, 1.0).asin();
            let want_u = (lon / (2.0 * std::f32::consts::PI) + 0.5).rem_euclid(1.0);
            let want_v = (0.5 - lat / std::f32::consts::PI).clamp(0.0, 1.0);
            let got = rt.world.lighting_radiance(d);
            assert!(
                (got.x - want_u).abs() < 0.02,
                "direction {d:?}: longitude {want_u} read back as {}",
                got.x
            );
            assert!(
                (got.y - want_v).abs() < 0.02,
                "direction {d:?}: latitude {want_v} read back as {}",
                got.y
            );
        }
    }

    /// The dynamic range has to survive. `CubeTexture::new_f32` keeps the HDR
    /// data in `faces_f32` and puts a Reinhard-compressed, gamma-encoded copy
    /// in `faces` for display; integrating the display copy would cap an
    /// environment at about 1.0 and lose the sun entirely.
    #[test]
    fn an_hdr_environment_keeps_values_above_one() {
        let size = 8u32;
        let bright = 500.0f32;
        let faces: [Vec<f32>; 6] = std::array::from_fn(|_| {
            let mut f = vec![0.0f32; (size * size * 4) as usize];
            for p in f.chunks_exact_mut(4) {
                p[0] = bright;
                p[1] = bright;
                p[2] = bright;
                p[3] = 1.0;
            }
            f
        });
        let mut scene = Scene::new();
        scene.environment = Some(Arc::new(crate::textures::CubeTexture::new_f32(size, faces)));
        let rt = RaytraceScene::build(&mut scene, &RaytraceSettings::default());
        let l = rt.world.lighting_radiance(Vector3::new(0.0, 1.0, 0.0));
        assert!(
            (l.x - bright).abs() < 0.01 * bright,
            "expected {bright}, read {} — the 8-bit fallback was used",
            l.x
        );
    }

    #[test]
    fn shadow_opacity_is_detected() {
        let mut scene = Scene::new();
        scene.add(Object3D::mesh(box_mesh(Color::WHITE)));
        let rt = RaytraceScene::build(&mut scene, &RaytraceSettings::default());
        assert!(rt.shadows_all_opaque);

        let mut scene = Scene::new();
        let mut m = StandardMaterial::new(Color::WHITE);
        m.opacity = 0.4;
        scene.add(Object3D::mesh(crate::core::Mesh::new(
            BoxGeometry::new(1.0, 1.0, 1.0),
            Material::Standard(m),
        )));
        let rt = RaytraceScene::build(&mut scene, &RaytraceSettings::default());
        assert!(!rt.shadows_all_opaque);
    }

    #[test]
    fn fog_modes_match_three_js_curves() {
        let mut w = World {
            fog_color: Vector3::new(1.0, 0.0, 0.0),
            fog_mode: 1,
            fog_near: 0.0,
            fog_far: 10.0,
            ..Default::default()
        };
        let (vis, fog) = w.fog_at(5.0);
        assert!((vis - 0.5).abs() < 1e-5);
        assert!((fog.x - 0.5).abs() < 1e-5);

        w.fog_mode = 2;
        w.fog_density = 0.1;
        let (vis, _) = w.fog_at(10.0);
        let expect = (-0.1f32 * 0.1 * 10.0).exp();
        assert!((vis - expect).abs() < 1e-5);
    }
}
