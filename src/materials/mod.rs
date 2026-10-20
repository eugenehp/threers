//! Materials. Mirrors three.js's `Material` family, including
//! [`ShaderMaterial`] for custom WGSL fragments.

mod atmosphere;
mod basic;
mod depth;
mod lambert;
mod line;
mod matcap;
mod normal_mat;
mod phong;
mod physical;
mod points_mat;
pub mod presets;
mod shader_material;
mod sprite_mat;
mod standard;
mod toon;

/// The rows of three.js's `envMapRotation · diag(flipEnvMap, 1, 1)` for a
/// cube texture (`WebGLMaterials.refreshUniformsCommon`): the Euler negated to
/// suit the cube's left-handed frame — all three axes, then y and z back, as
/// for every cube that is not a render target — and x mirrored
/// (`flipEnvMap = −1`).
pub(crate) fn env_lookup_rows(rotation: crate::math::Euler) -> [[f32; 4]; 3] {
    let e = crate::math::Euler { x: -rotation.x, ..rotation };
    let m = crate::math::Matrix4::from_quaternion(crate::math::Quaternion::from_euler(e)).elements;
    // Column-major: row r, column c is m[c * 4 + r]; column 0 takes the flip.
    std::array::from_fn(|r| [-m[r], m[4 + r], m[8 + r], 0.0])
}

/// The rows of three.js's `envMapRotation` for a Standard or Physical
/// material's image-based light, whose map is a PMREM render target: the
/// Euler with every axis negated, for the cube's left-handed frame, and no
/// mirror (`WebGLMaterials.refreshUniformsCommon`).
pub(crate) fn pmrem_lookup_rows(rotation: crate::math::Euler) -> [[f32; 4]; 3] {
    let e = crate::math::Euler { x: -rotation.x, y: -rotation.y, z: -rotation.z, ..rotation };
    let m = crate::math::Matrix4::from_quaternion(crate::math::Quaternion::from_euler(e)).elements;
    std::array::from_fn(|r| [m[r], m[4 + r], m[8 + r], 0.0])
}

/// The lookup rows for an image-based light at `rotation`. three.js
/// prefilters every cube texture through PMREM, which reads it mirrored in x
/// (`flipEnvMap = −1`): sampled directly, as a cube without a PMREM atlas is
/// by these renderers, it is mirrored the same way (`raw_cube`). A PMREM atlas
/// and a cube render target are already in the lookup's frame.
pub fn environment_lookup_rows(rotation: crate::math::Euler, raw_cube: bool) -> [[f32; 4]; 3] {
    let mut rows = pmrem_lookup_rows(rotation);
    if raw_cube {
        rows[0] = rows[0].map(|v| -v);
    }
    rows
}

/// three.js `material.combine`: how a Basic, Lambert or Phong material's
/// `envMap` meets its shaded colour (`envmap_fragment`), weighted by
/// `specularStrength × reflectivity`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Combine {
    /// `MultiplyOperation` (three.js's default): the colour tinted by the
    /// environment.
    #[default]
    Multiply = 0,
    /// `MixOperation`: blended toward the environment.
    Mix = 1,
    /// `AddOperation`: the environment added.
    Add = 2,
}

pub use atmosphere::AtmosphereMaterial;
pub use basic::BasicMaterial;
pub use depth::DepthMaterial;
pub use lambert::LambertMaterial;
pub use line::LineBasicMaterial;
pub use matcap::MatcapMaterial;
pub use normal_mat::NormalMaterial;
pub use phong::PhongMaterial;
pub use physical::PhysicalMaterial;
pub use points_mat::PointsMaterial;
pub use shader_material::ShaderMaterial;
pub use sprite_mat::SpriteMaterial;
pub use standard::StandardMaterial;
pub use toon::ToonMaterial;

use crate::math::{Color, Vector2};
use crate::textures::Texture;
use std::sync::Arc;

/// three.js's `Material.polygonOffset` with its `polygonOffsetFactor` and
/// `polygonOffsetUnits`: `glPolygonOffset(factor, units)`, which adds
/// `factor · m + units · r` to every fragment's depth — `m` the triangle's
/// largest window-space depth slope, `r` the smallest resolvable depth step.
/// Negative values pull the surface toward the camera (a decal over the
/// surface it was cut from). WebGPU's and Metal's depth bias are the same
/// formula — `slope_scale = factor`, `constant = units` — so the backends
/// pass the values straight through (negated under reverse-Z).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct PolygonOffset {
    pub factor: f32,
    pub units: f32,
}

impl PolygonOffset {
    pub const fn new(factor: f32, units: f32) -> Self {
        Self { factor, units }
    }
}

/// three.js's `Material.depthFunc`, with three.js's constant values
/// (`NeverDepth` = 0 … `NotEqualDepth` = 7).
///
/// The comparison is `fragment depth <op> stored depth` with depth growing
/// away from the camera, as in WebGL. A backend that stores reversed depth
/// (Metal's reverse-Z passes) flips the operator, so `Less` still means
/// "nearer than what is there".
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(u8)]
pub enum DepthFunc {
    Never = 0,
    Always = 1,
    Less = 2,
    /// three.js's default.
    #[default]
    LessEqual = 3,
    Equal = 4,
    GreaterEqual = 5,
    Greater = 6,
    NotEqual = 7,
}

impl DepthFunc {
    /// From a three.js constant (`THREE.LessEqualDepth` …); `None` for
    /// anything else.
    pub fn from_three(value: u32) -> Option<Self> {
        Some(match value {
            0 => Self::Never,
            1 => Self::Always,
            2 => Self::Less,
            3 => Self::LessEqual,
            4 => Self::Equal,
            5 => Self::GreaterEqual,
            6 => Self::Greater,
            7 => Self::NotEqual,
            _ => return None,
        })
    }

    /// The same test on a reversed depth buffer (near = 1, far = 0).
    pub fn reversed(self) -> Self {
        match self {
            Self::Less => Self::Greater,
            Self::LessEqual => Self::GreaterEqual,
            Self::Greater => Self::Less,
            Self::GreaterEqual => Self::LessEqual,
            other => other,
        }
    }
}

/// three.js `blending`, for the modes the wgpu backend applies.
///
/// `Additive` is three's `AdditiveBlending` (SrcAlpha, One): overlapping
/// fragments add up, so dense point clouds and stacked translucent slices glow
/// instead of occluding one another. It only matters on a material that blends
/// (transparent, or opacity below 1). The Metal backend does not apply it yet.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Blending {
    /// three.js `NormalBlending`: source-over alpha compositing.
    #[default]
    Normal,
    /// three.js `AdditiveBlending`.
    Additive,
}

impl Blending {
    /// From three.js's numeric constant (`THREE.NormalBlending` = 1,
    /// `THREE.AdditiveBlending` = 2); anything else is `None`.
    pub fn from_three(value: u32) -> Option<Self> {
        match value {
            1 => Some(Self::Normal),
            2 => Some(Self::Additive),
            _ => None,
        }
    }
}

/// The per-material fixed-function state three.js keeps on every `Material`:
/// `depthTest`, `depthWrite`, `depthFunc`, `colorWrite` and `blending`.
///
/// The backends turn it into pipeline state (wgpu's `DepthStencilState` and
/// colour write mask, Metal's `MTLDepthStencilState` and write mask) for every
/// draw path a material can take — plain, skinned, instanced, lines, points,
/// sprites and shader materials, opaque or blended. The order-independent and
/// glass transparency modes of [`PhysicalMaterial`] own their depth state (a
/// prepass, an accumulation target) and ignore this.
///
/// As in WebGL, turning the depth test off also turns depth writes off: a
/// `depthTest: false` overlay never occludes what is drawn after it. The
/// material still casts shadows — three.js's shadow pass draws with its own
/// depth material and reads none of this.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RenderState {
    /// three.js `depthTest` (default `true`).
    pub depth_test: bool,
    /// three.js `depthWrite`. `None` (the Rust default) follows the material's
    /// transparency — opaque writes, blended does not — which is what most
    /// native scenes want; three.js's own default is `Some(true)` whether or
    /// not the material is transparent, and the JS layer sends that.
    pub depth_write: Option<bool>,
    /// three.js `depthFunc` (default [`DepthFunc::LessEqual`]).
    pub depth_func: DepthFunc,
    /// three.js `colorWrite` (default `true`). `false` with depth writes on
    /// makes an invisible occluder.
    pub color_write: bool,
    /// three.js `blending` (default [`Blending::Normal`]).
    pub blending: Blending,
}

impl Default for RenderState {
    fn default() -> Self {
        Self::DEFAULT
    }
}

impl RenderState {
    /// three.js's defaults, less `depthWrite` (see
    /// [`depth_write`](Self::depth_write)).
    pub const DEFAULT: Self = Self {
        depth_test: true,
        depth_write: None,
        depth_func: DepthFunc::LessEqual,
        color_write: true,
        blending: Blending::Normal,
    };

    /// Whether this is the state every pipeline is built with, so a draw can
    /// use the stock pipeline without looking anything up.
    pub fn is_default(&self) -> bool {
        *self == Self::default()
    }

    /// The depth comparison actually applied (`Always` when the test is off).
    pub fn effective_func(&self) -> DepthFunc {
        if self.depth_test {
            self.depth_func
        } else {
            DepthFunc::Always
        }
    }

    /// Whether depth is written, given what the pipeline would do by default
    /// (`default_write`: opaque true, blended false). Off whenever the test is.
    pub fn effective_write(&self, default_write: bool) -> bool {
        self.depth_test && self.depth_write.unwrap_or(default_write)
    }
}

/// How a transparent [`PhysicalMaterial`] is composited. Ordinary alpha blending
/// re-sorts overlapping layers per-object and shimmers on rotation for complex
/// surfaces; the other two modes fix that.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TransparencyMode {
    /// Back-to-front alpha blending (default; order-dependent).
    #[default]
    Blend,
    /// **Single-layer glass** via a depth prepass: only the nearest surface
    /// blends, so a bumpy translucent mesh is flicker-free
    /// while rotating. You see the front shell + whatever is behind it.
    Glass,
    /// **Weighted-blended order-independent transparency** (McGuire): all layers
    /// accumulate order-independently, so even the far wall shows through the
    /// near wall with no flicker.
    Oit,
    /// **Screen-space refraction glass** (Blender/EEVEE-style): the opaque scene
    /// is captured to a mipped color buffer + depth, then the glass surface
    /// refracts it via Snell's law (with dispersion, total internal reflection,
    /// roughness→mip blur) and reflects it via screen-space ray-marching with an
    /// environment-probe fallback. The most physically-convincing mode; requires
    /// an environment for the reflection/refraction fallback to read well.
    Refract,
}

// The variants differ in size by design: boxing the large ones would turn
// every `Material::Physical(m)` a caller writes into a `Box` pattern, for a
// value that lives behind an `Arc` in every mesh anyway.
#[allow(clippy::large_enum_variant)]
#[derive(Debug, Clone)]
pub enum Material {
    Basic(BasicMaterial),
    Lambert(LambertMaterial),
    Phong(PhongMaterial),
    Standard(StandardMaterial),
    Physical(PhysicalMaterial),
    Normal(NormalMaterial),
    Depth(DepthMaterial),
    Toon(ToonMaterial),
    Matcap(MatcapMaterial),
    Line(LineBasicMaterial),
    Points(PointsMaterial),
    Sprite(SpriteMaterial),
    Distance(DistanceMaterial),
    Sky(SkyMaterial),
    Mirror(MirrorMaterial),
    /// User-defined custom-shader material (extensibility escape hatch).
    Shader(ShaderMaterial),
    /// Analytic planetary atmosphere — see [`AtmosphereMaterial`].
    Atmosphere(AtmosphereMaterial),
}

/// Mirror material — drives the Reflector / Refractor / Water shader path
/// which projectively samples a render-target texture using a per-frame
/// texture matrix supplied from JS. The texture matrix encodes
/// `scaleBias * virtualCamProj * virtualCamView` so that fragment-side
/// `texture_matrix * world_pos` lands in `[0,1]` UV space.
#[derive(Debug, Clone)]
pub struct MirrorMaterial {
    pub color: Color,
    pub map: Option<Arc<Texture>>,
    pub texture_matrix: [f32; 16],
    pub side: u32,
    /// three.js `depthTest` / `depthWrite` / `depthFunc` / `colorWrite`.
    pub render_state: RenderState,
}
impl Default for MirrorMaterial {
    fn default() -> Self {
        Self {
            color: Color::WHITE,
            map: None,
            texture_matrix: [
                1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
            ],
            side: 0,
            render_state: RenderState::default(),
        }
    }
}

/// Sky shader material. Drives the Preetham atmospheric scattering shader.
/// Mirrors three.js's `Sky` from `examples/jsm/objects/Sky.js`.
#[derive(Debug, Clone, Copy)]
pub struct SkyMaterial {
    pub sun_position: crate::math::Vector3,
    pub turbidity: f32,
    pub rayleigh: f32,
    pub mie_coefficient: f32,
    pub mie_directional_g: f32,
    /// three.js `depthTest` / `depthWrite` / `depthFunc` / `colorWrite`.
    /// The sky pipeline never writes depth already, as three.js's `Sky`
    /// (`depthWrite: false`).
    pub render_state: RenderState,
}
impl Default for SkyMaterial {
    fn default() -> Self {
        Self {
            sun_position: crate::math::Vector3::new(0.0, 1.0, 0.0),
            turbidity: 10.0,
            rayleigh: 3.0,
            mie_coefficient: 0.005,
            mie_directional_g: 0.7,
            render_state: RenderState::default(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum MaterialKind {
    Basic = 0,
    Lambert = 1,
    Phong = 2,
    Standard = 3,
    Physical = 4,
    Normal = 5,
    Depth = 6,
    Toon = 7,
    Matcap = 8,
    Line = 9,
    Points = 10,
    Sprite = 11,
    Distance = 12,
    Sky = 13,
    Mirror = 14,
    Shader = 15,
    Atmosphere = 16,
}

#[derive(Debug, Clone)]
pub struct DistanceMaterial {
    pub reference_position: crate::math::Vector3,
    pub near_distance: f32,
    pub far_distance: f32,
    /// three.js `depthTest` / `depthWrite` / `depthFunc` / `colorWrite`.
    pub render_state: RenderState,
}
impl Default for DistanceMaterial {
    fn default() -> Self {
        Self {
            reference_position: crate::math::Vector3::ZERO,
            near_distance: 1.0,
            far_distance: 1000.0,
            render_state: RenderState::default(),
        }
    }
}
impl DistanceMaterial {
    pub fn new(reference_position: crate::math::Vector3, near: f32, far: f32) -> Self {
        Self {
            reference_position,
            near_distance: near.max(0.0),
            far_distance: far.max(near + 0.001),
            render_state: RenderState::default(),
        }
    }
}

/// The maps Lambert, Phong and Toon share with three.js's versions of them.
fn lit_slots(
    map: &Option<Arc<Texture>>,
    emissive: &Option<Arc<Texture>>,
    normal: &Option<Arc<Texture>>,
    ao: &Option<Arc<Texture>>,
    displacement: &Option<Arc<Texture>>,
    alpha: &Option<Arc<Texture>>,
) -> MaterialTextureSlots {
    MaterialTextureSlots {
        map: map.clone(),
        alpha_map: alpha.clone(),
        emissive_map: emissive.clone(),
        normal_map: normal.clone(),
        ao_map: ao.clone(),
        displacement_map: displacement.clone(),
        ..Default::default()
    }
}

#[derive(Default, Clone)]
pub struct MaterialTextureSlots {
    pub map: Option<Arc<Texture>>,
    pub normal_map: Option<Arc<Texture>>,
    pub roughness_map: Option<Arc<Texture>>,
    pub metalness_map: Option<Arc<Texture>>,
    pub ao_map: Option<Arc<Texture>>,
    pub emissive_map: Option<Arc<Texture>>,
    pub matcap_map: Option<Arc<Texture>>,
    /// three.js `gradientMap` on MeshToonMaterial.
    pub gradient_map: Option<Arc<Texture>>,
    /// Height map, sampled in the *vertex* stage to move geometry.
    pub displacement_map: Option<Arc<Texture>>,
    /// Cloud cover that casts a shadow onto this surface.
    pub cloud_shadow_map: Option<Arc<Texture>>,
    /// Thin-film thickness map (three.js `iridescenceThicknessMap`). Sampled
    /// from the green channel and remapped into the material's thickness range.
    pub iridescence_thickness_map: Option<Arc<Texture>>,
    /// three.js `alphaMap` (green channel), multiplying the opacity.
    pub alpha_map: Option<Arc<Texture>>,
    /// three.js `lightMap`, baked irradiance added to the indirect light.
    pub light_map: Option<Arc<Texture>>,
    /// three.js `bumpMap`: a height field that perturbs the shading normal.
    pub bump_map: Option<Arc<Texture>>,
    /// three.js `specularMap` (red channel) on Basic, Lambert and Phong.
    pub specular_map: Option<Arc<Texture>>,
    /// three.js `envMap` on Basic, Lambert and Phong: a cube map reflected
    /// (or refracted) and combined with the shaded colour.
    pub env_map: Option<Arc<crate::textures::CubeTexture>>,
    /// three.js `envMap` on Standard and Physical: the material's own
    /// image-based light, bound as its environment (group 3) in place of the
    /// scene's.
    pub pbr_env_map: Option<Arc<crate::textures::CubeTexture>>,
    /// MeshPhysicalMaterial's own maps, as three.js names them:
    /// `specularIntensityMap` (alpha), `specularColorMap` (rgb),
    /// `clearcoatMap` (red), `clearcoatRoughnessMap` (green),
    /// `clearcoatNormalMap`, `iridescenceMap` (red), `sheenColorMap` (rgb),
    /// `sheenRoughnessMap` (alpha), `transmissionMap` (red), `thicknessMap`
    /// (green) and `anisotropyMap` (direction in red/green, strength in blue).
    pub specular_intensity_map: Option<Arc<Texture>>,
    pub specular_color_map: Option<Arc<Texture>>,
    pub clearcoat_map: Option<Arc<Texture>>,
    pub clearcoat_roughness_map: Option<Arc<Texture>>,
    pub clearcoat_normal_map: Option<Arc<Texture>>,
    pub iridescence_map: Option<Arc<Texture>>,
    pub sheen_color_map: Option<Arc<Texture>>,
    pub sheen_roughness_map: Option<Arc<Texture>>,
    pub transmission_map: Option<Arc<Texture>>,
    pub thickness_map: Option<Arc<Texture>>,
    pub anisotropy_map: Option<Arc<Texture>>,
}

impl MaterialTextureSlots {
    /// The texture whose filter and wrap the material's shared sampler takes:
    /// the colour map (or matcap), else the first map present. Both raster
    /// backends bind one sampler per material, so maps of one material that
    /// ask for different filtering share the first one's — three.js keeps a
    /// sampler per texture. Taking the first present map, rather than only the
    /// colour map, keeps a material without a colour map (an `aoMap` alone, a
    /// nearest-filtered `emissiveMap`) on its own map's settings.
    pub fn sampler_source(&self) -> Option<&Arc<Texture>> {
        self.map
            .as_ref()
            .or(self.matcap_map.as_ref())
            .or(self.emissive_map.as_ref())
            .or(self.normal_map.as_ref())
            .or(self.ao_map.as_ref())
            .or(self.roughness_map.as_ref())
            .or(self.metalness_map.as_ref())
            .or(self.displacement_map.as_ref())
    }
}

impl Material {
    pub fn color(&self) -> Color {
        match self {
            Material::Basic(m) => m.color,
            Material::Lambert(m) => m.color,
            Material::Phong(m) => m.color,
            Material::Standard(m) => m.color,
            Material::Physical(m) => m.color,
            Material::Normal(_) => Color::WHITE,
            Material::Depth(_) => Color::WHITE,
            Material::Toon(m) => m.color,
            Material::Matcap(m) => m.color,
            Material::Line(m) => m.color,
            Material::Points(m) => m.color,
            Material::Sprite(m) => m.color,
            Material::Distance(_) => Color::WHITE,
            Material::Sky(_) => Color::WHITE,
            Material::Mirror(m) => m.color,
            Material::Atmosphere(m) => m.color,
            Material::Shader(_) => Color::WHITE,
        }
    }

    pub fn wireframe(&self) -> bool {
        match self {
            Material::Basic(m) => m.wireframe,
            Material::Lambert(m) => m.wireframe,
            Material::Phong(m) => m.wireframe,
            Material::Standard(m) => m.wireframe,
            Material::Physical(m) => m.wireframe,
            Material::Normal(m) => m.wireframe,
            Material::Depth(m) => m.wireframe,
            Material::Toon(m) => m.wireframe,
            Material::Matcap(m) => m.wireframe,
            _ => false,
        }
    }

    pub fn transparent(&self) -> bool {
        match self {
            Material::Basic(m) => m.transparent,
            // three.js draws a transmissive material opaque — it sees the
            // scene behind through the transmission capture, not by blending
            // (see `transmissive`). The opt-in glass modes still blend it.
            Material::Physical(m) => {
                (m.transmission > 0.0 && m.transparency != TransparencyMode::Blend)
                    || m.transparent.unwrap_or(m.opacity < 1.0)
            }
            Material::Shader(m) => m.transparent || m.opacity < 1.0,
            // An atmosphere is nothing but coverage — it is never opaque, even
            // at `opacity: 1.0`, where that value only scales the optical depth.
            Material::Atmosphere(_) => true,
            // Other materials: three.js's flag when it was given; otherwise
            // blend when opacity < 1, this crate's rule before the flag.
            _ => self
                .explicit_transparent()
                .unwrap_or_else(|| self.opacity() < 1.0),
        }
    }

    fn explicit_transparent(&self) -> Option<bool> {
        match self {
            Material::Lambert(m) => m.transparent,
            Material::Phong(m) => m.transparent,
            Material::Standard(m) => m.transparent,
            Material::Toon(m) => m.transparent,
            Material::Matcap(m) => m.transparent,
            Material::Normal(m) => m.transparent,
            Material::Depth(m) => m.transparent,
            Material::Line(m) => m.transparent,
            Material::Points(m) => m.transparent,
            // three.js's SpriteMaterial is `transparent: true` by default.
            Material::Sprite(m) => Some(m.transparent.unwrap_or(true)),
            _ => None,
        }
    }

    /// The material's polygon offset, when enabled (three.js
    /// `polygonOffset: true`). Materials without the field report `None`.
    pub fn polygon_offset(&self) -> Option<PolygonOffset> {
        match self {
            Material::Basic(m) => m.polygon_offset,
            Material::Lambert(m) => m.polygon_offset,
            Material::Phong(m) => m.polygon_offset,
            Material::Standard(m) => m.polygon_offset,
            Material::Physical(m) => m.polygon_offset,
            Material::Toon(m) => m.polygon_offset,
            Material::Normal(m) => m.polygon_offset,
            Material::Matcap(m) => m.polygon_offset,
            Material::Depth(m) => m.polygon_offset,
            _ => None,
        }
        .filter(|o| o.factor != 0.0 || o.units != 0.0)
    }

    /// three.js `toneMapped`: whether the renderer's tone mapping applies to
    /// this material. Data materials (normal, depth, distance) are never tone
    /// mapped — three.js's shaders for them have no `tonemapping_fragment` —
    /// and the rest default to `true`, as `Material.toneMapped` does.
    pub fn tone_mapped(&self) -> bool {
        match self {
            Material::Basic(m) => m.tone_mapped,
            Material::Lambert(m) => m.tone_mapped,
            Material::Phong(m) => m.tone_mapped,
            Material::Standard(m) => m.tone_mapped,
            Material::Physical(m) => m.tone_mapped,
            Material::Toon(m) => m.tone_mapped,
            Material::Matcap(m) => m.tone_mapped,
            Material::Line(m) => m.tone_mapped,
            Material::Points(m) => m.tone_mapped,
            Material::Sprite(m) => m.tone_mapped,
            Material::Shader(m) => m.tone_mapped,
            Material::Normal(_) | Material::Depth(_) | Material::Distance(_) => false,
            _ => true,
        }
    }

    /// Set three.js `toneMapped` on the materials that carry it; a no-op on
    /// the rest.
    pub fn set_tone_mapped(&mut self, on: bool) {
        match self {
            Material::Basic(m) => m.tone_mapped = on,
            Material::Lambert(m) => m.tone_mapped = on,
            Material::Phong(m) => m.tone_mapped = on,
            Material::Standard(m) => m.tone_mapped = on,
            Material::Physical(m) => m.tone_mapped = on,
            Material::Toon(m) => m.tone_mapped = on,
            Material::Matcap(m) => m.tone_mapped = on,
            Material::Line(m) => m.tone_mapped = on,
            Material::Points(m) => m.tone_mapped = on,
            Material::Sprite(m) => m.tone_mapped = on,
            Material::Shader(m) => m.tone_mapped = on,
            _ => {}
        }
    }

    /// three.js `depthTest`, `depthWrite`, `depthFunc` and `colorWrite` —
    /// see [`RenderState`].
    pub fn render_state(&self) -> RenderState {
        match self {
            Material::Basic(m) => m.render_state,
            Material::Lambert(m) => m.render_state,
            Material::Phong(m) => m.render_state,
            Material::Standard(m) => m.render_state,
            Material::Physical(m) => m.render_state,
            Material::Normal(m) => m.render_state,
            Material::Depth(m) => m.render_state,
            Material::Toon(m) => m.render_state,
            Material::Matcap(m) => m.render_state,
            Material::Line(m) => m.render_state,
            Material::Points(m) => m.render_state,
            Material::Sprite(m) => m.render_state,
            Material::Distance(m) => m.render_state,
            Material::Sky(m) => m.render_state,
            Material::Mirror(m) => m.render_state,
            Material::Shader(m) => m.render_state,
            Material::Atmosphere(m) => m.render_state,
        }
    }

    /// The material's [`RenderState`], to change it.
    pub fn render_state_mut(&mut self) -> &mut RenderState {
        match self {
            Material::Basic(m) => &mut m.render_state,
            Material::Lambert(m) => &mut m.render_state,
            Material::Phong(m) => &mut m.render_state,
            Material::Standard(m) => &mut m.render_state,
            Material::Physical(m) => &mut m.render_state,
            Material::Normal(m) => &mut m.render_state,
            Material::Depth(m) => &mut m.render_state,
            Material::Toon(m) => &mut m.render_state,
            Material::Matcap(m) => &mut m.render_state,
            Material::Line(m) => &mut m.render_state,
            Material::Points(m) => &mut m.render_state,
            Material::Sprite(m) => &mut m.render_state,
            Material::Distance(m) => &mut m.render_state,
            Material::Sky(m) => &mut m.render_state,
            Material::Mirror(m) => &mut m.render_state,
            Material::Shader(m) => &mut m.render_state,
            Material::Atmosphere(m) => &mut m.render_state,
        }
    }

    /// three.js `depthTest`: whether the material tests the depth buffer.
    /// `false` draws over whatever is already there (an x-ray overlay, a
    /// gizmo) — in draw order, so give it a later `render_order` or make it
    /// transparent to have it drawn after the scene.
    pub fn depth_test(&self) -> bool {
        self.render_state().depth_test
    }

    /// three.js's transmissive material (`transmission > 0`), drawn as three.js
    /// draws it: after the opaque objects, reading them from the transmission
    /// capture. Not in the opt-in glass modes, which have their own.
    pub fn transmissive(&self) -> bool {
        matches!(self, Material::Physical(m) if m.transmission > 0.0 && m.transparency == TransparencyMode::Blend)
    }

    /// Which transparency technique a *transparent* material uses. Only
    /// `PhysicalMaterial` can opt out of plain alpha blending (into single-layer
    /// glass or order-independent transparency); everything else is `Blend`.
    pub fn transparency_mode(&self) -> TransparencyMode {
        match self {
            Material::Physical(m) => m.transparency,
            _ => TransparencyMode::Blend,
        }
    }

    /// three.js `material.opacity`, on every material that has one.
    pub fn set_opacity(&mut self, v: f32) {
        match self {
            Material::Basic(m) => m.opacity = v,
            Material::Lambert(m) => m.opacity = v,
            Material::Phong(m) => m.opacity = v,
            Material::Standard(m) => m.opacity = v,
            Material::Physical(m) => m.opacity = v,
            Material::Normal(m) => m.opacity = v,
            Material::Depth(m) => m.opacity = v,
            Material::Toon(m) => m.opacity = v,
            Material::Matcap(m) => m.opacity = v,
            Material::Line(m) => m.opacity = v,
            Material::Points(m) => m.opacity = v,
            Material::Sprite(m) => m.opacity = v,
            Material::Atmosphere(m) => m.opacity = v,
            Material::Shader(m) => m.opacity = v,
            Material::Distance(_) | Material::Sky(_) | Material::Mirror(_) => {}
        }
    }

    pub fn opacity(&self) -> f32 {
        match self {
            Material::Basic(m) => m.opacity,
            Material::Lambert(m) => m.opacity,
            Material::Phong(m) => m.opacity,
            Material::Standard(m) => m.opacity,
            Material::Physical(m) => m.opacity,
            Material::Normal(m) => m.opacity,
            Material::Depth(m) => m.opacity,
            Material::Toon(m) => m.opacity,
            Material::Matcap(m) => m.opacity,
            Material::Line(m) => m.opacity,
            Material::Points(m) => m.opacity,
            Material::Sprite(m) => m.opacity,
            Material::Distance(_) => 1.0,
            Material::Sky(_) => 1.0,
            Material::Mirror(_) => 1.0,
            Material::Atmosphere(m) => m.opacity,
            Material::Shader(m) => m.opacity,
        }
    }

    /// three.js `alphaTest` (0 when the material has none).
    pub fn alpha_test(&self) -> f32 {
        match self {
            Material::Basic(m) => m.alpha_test,
            Material::Lambert(m) => m.alpha_test,
            Material::Phong(m) => m.alpha_test,
            Material::Standard(m) => m.alpha_test,
            Material::Physical(m) => m.alpha_test,
            Material::Toon(m) => m.alpha_test,
            Material::Matcap(m) => m.alpha_test,
            Material::Normal(m) => m.alpha_test,
            Material::Depth(m) => m.alpha_test,
            Material::Line(m) => m.alpha_test,
            Material::Points(m) => m.alpha_test,
            Material::Sprite(m) => m.alpha_test,
            _ => 0.0,
        }
    }

    /// Set three.js `alphaTest` on any material that has it; a no-op on the
    /// rest. Negative values clamp to 0.
    pub fn set_alpha_test(&mut self, value: f32) {
        let v = value.max(0.0);
        match self {
            Material::Basic(m) => m.alpha_test = v,
            Material::Lambert(m) => m.alpha_test = v,
            Material::Phong(m) => m.alpha_test = v,
            Material::Standard(m) => m.alpha_test = v,
            Material::Physical(m) => m.alpha_test = v,
            Material::Toon(m) => m.alpha_test = v,
            Material::Matcap(m) => m.alpha_test = v,
            Material::Normal(m) => m.alpha_test = v,
            Material::Depth(m) => m.alpha_test = v,
            Material::Line(m) => m.alpha_test = v,
            Material::Points(m) => m.alpha_test = v,
            Material::Sprite(m) => m.alpha_test = v,
            _ => {}
        }
    }

    /// Set three.js `transparent` on any material that has it. Basic and
    /// shader materials store the flag as given; the others store `Some`.
    pub fn set_transparent(&mut self, transparent: bool) {
        let t = Some(transparent);
        match self {
            Material::Basic(m) => m.transparent = transparent,
            Material::Shader(m) => m.transparent = transparent,
            Material::Lambert(m) => m.transparent = t,
            Material::Phong(m) => m.transparent = t,
            Material::Standard(m) => m.transparent = t,
            Material::Physical(m) => m.transparent = t,
            Material::Toon(m) => m.transparent = t,
            Material::Matcap(m) => m.transparent = t,
            Material::Normal(m) => m.transparent = t,
            Material::Depth(m) => m.transparent = t,
            Material::Line(m) => m.transparent = t,
            Material::Points(m) => m.transparent = t,
            Material::Sprite(m) => m.transparent = t,
            _ => {}
        }
    }

    pub fn shadow_only(&self) -> bool {
        match self {
            Material::Basic(m) => m.shadow_only,
            _ => false,
        }
    }

    pub fn emissive(&self) -> Color {
        match self {
            Material::Lambert(m) => m.emissive,
            Material::Phong(m) => m.emissive,
            Material::Standard(m) => Color::new(
                m.emissive.r * m.emissive_intensity,
                m.emissive.g * m.emissive_intensity,
                m.emissive.b * m.emissive_intensity,
            ),
            Material::Physical(m) => Color::new(
                m.emissive.r * m.emissive_intensity,
                m.emissive.g * m.emissive_intensity,
                m.emissive.b * m.emissive_intensity,
            ),
            Material::Toon(m) => m.emissive,
            _ => Color::BLACK,
        }
    }

    pub fn specular(&self) -> Color {
        match self {
            Material::Phong(m) => m.specular,
            _ => Color::BLACK,
        }
    }

    /// Material's render side. `0` = FrontSide (default), `1` = BackSide,
    /// `2` = DoubleSide. Threaded from JS via `WebMaterial.setSide`.
    pub fn side(&self) -> u32 {
        match self {
            Material::Basic(m) => m.side,
            Material::Lambert(m) => m.side,
            Material::Phong(m) => m.side,
            Material::Standard(m) => m.side,
            Material::Physical(m) => m.side,
            Material::Toon(m) => m.side,
            Material::Matcap(_) => 0,
            Material::Normal(_) => 0,
            Material::Depth(_) => 0,
            Material::Distance(_) => 0,
            Material::Line(_) => 0,
            Material::Points(_) => 0,
            Material::Sprite(_) => 0,
            Material::Sky(_) => 1, // Sky uses BackSide → no-cull pipeline.
            Material::Mirror(m) => m.side,
            // Front faces only: the shader measures the ray's path through the
            // shell analytically, so it needs exactly one entry point.
            Material::Atmosphere(_) => 0,
            Material::Shader(m) => m.side,
        }
    }

    /// Set the render side, where the material has one to set.
    ///
    /// The kinds that answer a fixed number from [`side`](Self::side) — a sky
    /// is back faces because of how it is drawn, an atmosphere front faces
    /// because the shader needs one entry point — keep it, since changing it
    /// would break the shader rather than change the look.
    pub fn set_side(&mut self, side: u32) {
        match self {
            Material::Basic(m) => m.side = side,
            Material::Lambert(m) => m.side = side,
            Material::Phong(m) => m.side = side,
            Material::Standard(m) => m.side = side,
            Material::Physical(m) => m.side = side,
            Material::Toon(m) => m.side = side,
            Material::Mirror(m) => m.side = side,
            Material::Shader(m) => m.side = side,
            _ => {}
        }
    }

    pub fn shininess(&self) -> f32 {
        match self {
            Material::Phong(m) => m.shininess,
            _ => 0.0,
        }
    }

    pub fn roughness(&self) -> f32 {
        match self {
            Material::Standard(m) => m.roughness,
            Material::Physical(m) => m.roughness,
            _ => 1.0,
        }
    }

    pub fn metalness(&self) -> f32 {
        match self {
            Material::Standard(m) => m.metalness,
            Material::Physical(m) => m.metalness,
            _ => 0.0,
        }
    }

    pub fn ao_intensity(&self) -> f32 {
        match self {
            Material::Standard(m) => m.ao_intensity,
            Material::Physical(m) => m.ao_intensity,
            Material::Lambert(m) => m.ao_intensity,
            Material::Phong(m) => m.ao_intensity,
            Material::Toon(m) => m.ao_intensity,
            _ => 1.0,
        }
    }

    pub fn normal_scale(&self) -> Vector2 {
        match self {
            Material::Standard(m) => m.normal_scale,
            Material::Physical(m) => m.normal_scale,
            Material::Lambert(m) => m.normal_scale,
            Material::Phong(m) => m.normal_scale,
            Material::Toon(m) => m.normal_scale,
            _ => Vector2::ONE,
        }
    }

    /// `[cloud_height, cloud_shadow, cloud_rotation, twilight]`.
    pub fn atmosphere_params(&self) -> [f32; 4] {
        match self {
            Material::Standard(m) => [m.cloud_height, m.cloud_shadow, m.cloud_rotation, m.twilight],
            Material::Physical(m) => [m.cloud_height, m.cloud_shadow, m.cloud_rotation, m.twilight],
            _ => [0.0; 4],
        }
    }

    /// A sphere that can eclipse the sun for this surface, and how big the sun
    /// is: `[x, y, z, radius]` and the sun's angular radius. Radius 0 = none.
    pub fn eclipse(&self) -> ([f32; 4], f32) {
        match self {
            Material::Standard(m) => (m.eclipse_occluder.unwrap_or([0.0; 4]), m.sun_angular_radius),
            Material::Physical(m) => (m.eclipse_occluder.unwrap_or([0.0; 4]), m.sun_angular_radius),
            _ => ([0.0; 4], 0.0),
        }
    }

    /// Volume scattering: `[strength, anisotropy, ozone, aurora]`.
    ///
    /// The first two belong to a cloud deck and the last two to an atmosphere
    /// shell; no material uses both, so they share a slot.
    pub fn scatter_params(&self) -> [f32; 4] {
        match self {
            Material::Standard(m) => [m.cloud_scatter, m.cloud_anisotropy, 0.0, 0.0],
            Material::Physical(m) => [m.cloud_scatter, m.cloud_anisotropy, 0.0, 0.0],
            Material::Atmosphere(m) => [0.0, 0.0, m.ozone, m.aurora],
            _ => [0.0; 4],
        }
    }

    /// Aurora colour, and the geomagnetic colatitude of the oval in degrees.
    pub fn aurora_params(&self) -> [f32; 4] {
        match self {
            Material::Atmosphere(m) => {
                let c = m.aurora_color;
                [c.r, c.g, c.b, m.aurora_colatitude]
            }
            _ => [0.0; 4],
        }
    }

    /// The colour scattered light takes on near the terminator.
    pub fn twilight_color(&self) -> Color {
        match self {
            Material::Standard(m) => m.twilight_color,
            Material::Physical(m) => m.twilight_color,
            _ => Color::BLACK,
        }
    }

    /// Height-map `(scale, bias)`. Zero scale for materials that have none,
    /// which is also what disables the vertex-stage sample.
    /// How strongly the emissive map is confined to the night side, 0..1.
    pub fn emissive_night_side(&self) -> f32 {
        match self {
            Material::Standard(m) => m.emissive_night_side,
            _ => 0.0,
        }
    }

    pub fn displacement(&self) -> (f32, f32) {
        match self {
            Material::Standard(m) => (m.displacement_scale, m.displacement_bias),
            Material::Physical(m) => (m.displacement_scale, m.displacement_bias),
            Material::Lambert(m) => (m.displacement_scale, m.displacement_bias),
            Material::Phong(m) => (m.displacement_scale, m.displacement_bias),
            Material::Toon(m) => (m.displacement_scale, m.displacement_bias),
            _ => (0.0, 0.0),
        }
    }

    pub fn depth_range(&self) -> (f32, f32) {
        match self {
            Material::Depth(m) => (m.near, m.far),
            _ => (0.0, 0.0),
        }
    }

    pub fn point_size(&self) -> f32 {
        match self {
            Material::Points(m) => m.size,
            _ => 1.0,
        }
    }

    pub fn point_size_attenuation(&self) -> bool {
        match self {
            Material::Points(m) => m.size_attenuation,
            _ => false,
        }
    }

    pub fn kind(&self) -> MaterialKind {
        match self {
            Material::Basic(_) => MaterialKind::Basic,
            Material::Lambert(_) => MaterialKind::Lambert,
            Material::Phong(_) => MaterialKind::Phong,
            Material::Standard(_) => MaterialKind::Standard,
            Material::Physical(_) => MaterialKind::Physical,
            Material::Normal(_) => MaterialKind::Normal,
            Material::Depth(_) => MaterialKind::Depth,
            Material::Toon(_) => MaterialKind::Toon,
            Material::Matcap(_) => MaterialKind::Matcap,
            Material::Line(_) => MaterialKind::Line,
            Material::Points(_) => MaterialKind::Points,
            Material::Sprite(_) => MaterialKind::Sprite,
            Material::Distance(_) => MaterialKind::Distance,
            Material::Sky(_) => MaterialKind::Sky,
            Material::Mirror(_) => MaterialKind::Mirror,
            Material::Atmosphere(_) => MaterialKind::Atmosphere,
            Material::Shader(_) => MaterialKind::Shader,
        }
    }

    /// A Standard or Physical material's own `envMap`, its
    /// `envMapIntensity` and `envMapRotation`.
    pub fn pbr_env(&self) -> Option<(&Arc<crate::textures::CubeTexture>, f32, crate::math::Euler)> {
        match self {
            Material::Standard(m) => m.env_map.as_ref().map(|e| (e, m.env_map_intensity, m.env_map_rotation)),
            Material::Physical(m) => m.env_map.as_ref().map(|e| (e, m.env_map_intensity, m.env_map_rotation)),
            _ => None,
        }
    }

    /// The uniform block of a material shader variant, six `vec4`s:
    ///
    /// 0. `[lightMapIntensity, bumpScale, iridescenceThicknessRange[0], 0]`;
    /// 1. `[combine, reflectivity, refractionRatio, refraction mapping]` for
    ///    the `envMap`;
    /// 2. (with 3 and 4) the rows of three.js's
    ///    `envMapRotation · diag(flipEnvMap, 1, 1)` — the direction an
    ///    `envMap` is looked up along;
    /// 5. `[clearcoatNormalScale.x, clearcoatNormalScale.y, 0, 0]`.
    ///
    /// three.js's defaults where the material has none of these.
    pub fn map_uniforms(&self) -> [[f32; 4]; 6] {
        let (light, bump) = match self {
            Material::Basic(m) => (m.light_map_intensity, 1.0),
            Material::Lambert(m) => (m.light_map_intensity, m.bump_scale),
            Material::Phong(m) => (m.light_map_intensity, m.bump_scale),
            Material::Toon(m) => (m.light_map_intensity, m.bump_scale),
            Material::Standard(m) => (m.light_map_intensity, m.bump_scale),
            Material::Physical(m) => (m.light_map_intensity, m.bump_scale),
            _ => (1.0, 1.0),
        };
        let env = match self {
            Material::Basic(m) => Some((&m.env_map, m.combine, m.reflectivity, m.refraction_ratio, m.env_map_rotation)),
            Material::Lambert(m) => Some((&m.env_map, m.combine, m.reflectivity, m.refraction_ratio, m.env_map_rotation)),
            Material::Phong(m) => Some((&m.env_map, m.combine, m.reflectivity, m.refraction_ratio, m.env_map_rotation)),
            _ => None,
        };
        let (params, rows) = match env {
            Some((map, combine, reflectivity, ratio, rotation)) => {
                let refraction = map
                    .as_ref()
                    .is_some_and(|c| c.mapping == crate::textures::CubeMapping::Refraction);
                (
                    [combine as u32 as f32, reflectivity, ratio, refraction as u32 as f32],
                    env_lookup_rows(rotation),
                )
            }
            None => ([0.0, 1.0, 0.98, 0.0], env_lookup_rows(crate::math::Euler::default())),
        };
        let (thickness_min, cc_scale) = match self {
            Material::Physical(m) => (m.iridescence_thickness_range[0], m.clearcoat_normal_scale),
            _ => (0.0, crate::math::Vector2::new(1.0, 1.0)),
        };
        [
            [light, bump, thickness_min, 0.0],
            params,
            rows[0],
            rows[1],
            rows[2],
            [cc_scale.x, cc_scale.y, 0.0, 0.0],
        ]
    }

    /// The map slot three.js calls `name` (`"map"`, `"alphaMap"`,
    /// `"aoMap"`, `"emissiveMap"`, `"normalMap"`, `"roughnessMap"`,
    /// `"metalnessMap"`, `"displacementMap"`, `"lightMap"`, `"bumpMap"`,
    /// `"specularMap"`, `"matcap"`, and threers's `"cloudShadowMap"`), when
    /// this material has one.
    pub fn map_slot_mut(&mut self, name: &str) -> Option<&mut Option<Arc<Texture>>> {
        macro_rules! lit {
            ($m:expr) => {
                match name {
                    "map" => Some(&mut $m.map),
                    "alphaMap" => Some(&mut $m.alpha_map),
                    "aoMap" => Some(&mut $m.ao_map),
                    "emissiveMap" => Some(&mut $m.emissive_map),
                    "normalMap" => Some(&mut $m.normal_map),
                    "displacementMap" => Some(&mut $m.displacement_map),
                    "lightMap" => Some(&mut $m.light_map),
                    "bumpMap" => Some(&mut $m.bump_map),
                    _ => None,
                }
            };
        }
        macro_rules! pbr {
            ($m:expr) => {
                match name {
                    "roughnessMap" => Some(&mut $m.roughness_map),
                    "metalnessMap" => Some(&mut $m.metalness_map),
                    "cloudShadowMap" => Some(&mut $m.cloud_shadow_map),
                    _ => lit!($m),
                }
            };
        }
        match self {
            Material::Lambert(m) => match name {
                "specularMap" => Some(&mut m.specular_map),
                _ => lit!(m),
            },
            Material::Phong(m) => match name {
                "specularMap" => Some(&mut m.specular_map),
                _ => lit!(m),
            },
            Material::Toon(m) => match name {
                "gradientMap" => Some(&mut m.gradient_map),
                _ => lit!(m),
            },
            Material::Standard(m) => pbr!(m),
            Material::Physical(m) => match name {
                "specularIntensityMap" => Some(&mut m.specular_intensity_map),
                "specularColorMap" => Some(&mut m.specular_color_map),
                "clearcoatMap" => Some(&mut m.clearcoat_map),
                "clearcoatRoughnessMap" => Some(&mut m.clearcoat_roughness_map),
                "clearcoatNormalMap" => Some(&mut m.clearcoat_normal_map),
                "iridescenceMap" => Some(&mut m.iridescence_map),
                "iridescenceThicknessMap" => Some(&mut m.iridescence_thickness_map),
                "sheenColorMap" => Some(&mut m.sheen_color_map),
                "sheenRoughnessMap" => Some(&mut m.sheen_roughness_map),
                "transmissionMap" => Some(&mut m.transmission_map),
                "thicknessMap" => Some(&mut m.thickness_map),
                "anisotropyMap" => Some(&mut m.anisotropy_map),
                _ => pbr!(m),
            },
            Material::Basic(m) => match name {
                "map" => Some(&mut m.map),
                "alphaMap" => Some(&mut m.alpha_map),
                "lightMap" => Some(&mut m.light_map),
                "specularMap" => Some(&mut m.specular_map),
                _ => None,
            },
            Material::Points(m) => match name {
                "map" => Some(&mut m.map),
                "alphaMap" => Some(&mut m.alpha_map),
                _ => None,
            },
            Material::Sprite(m) => match name {
                "map" => Some(&mut m.map),
                "alphaMap" => Some(&mut m.alpha_map),
                _ => None,
            },
            Material::Matcap(m) => match name {
                "matcap" => Some(&mut m.matcap),
                "map" => Some(&mut m.map),
                "alphaMap" => Some(&mut m.alpha_map),
                _ => None,
            },
            _ => None,
        }
    }

    pub fn texture_slots(&self) -> MaterialTextureSlots {
        match self {
            Material::Standard(m) => MaterialTextureSlots {
                cloud_shadow_map: m.cloud_shadow_map.clone(),
                iridescence_thickness_map: None,
                map: m.map.clone(),
                normal_map: m.normal_map.clone(),
                roughness_map: m.roughness_map.clone(),
                metalness_map: m.metalness_map.clone(),
                ao_map: m.ao_map.clone(),
                emissive_map: m.emissive_map.clone(),
                displacement_map: m.displacement_map.clone(),
                alpha_map: m.alpha_map.clone(),
                light_map: m.light_map.clone(),
                bump_map: m.bump_map.clone(),
                pbr_env_map: m.env_map.clone(),
                matcap_map: None,
                ..Default::default()
            },
            Material::Physical(m) => MaterialTextureSlots {
                cloud_shadow_map: m.cloud_shadow_map.clone(),
                iridescence_thickness_map: m.iridescence_thickness_map.clone(),
                map: m.map.clone(),
                normal_map: m.normal_map.clone(),
                roughness_map: m.roughness_map.clone(),
                metalness_map: m.metalness_map.clone(),
                ao_map: m.ao_map.clone(),
                emissive_map: m.emissive_map.clone(),
                displacement_map: m.displacement_map.clone(),
                alpha_map: m.alpha_map.clone(),
                light_map: m.light_map.clone(),
                bump_map: m.bump_map.clone(),
                pbr_env_map: m.env_map.clone(),
                matcap_map: None,
                specular_intensity_map: m.specular_intensity_map.clone(),
                specular_color_map: m.specular_color_map.clone(),
                clearcoat_map: m.clearcoat_map.clone(),
                clearcoat_roughness_map: m.clearcoat_roughness_map.clone(),
                clearcoat_normal_map: m.clearcoat_normal_map.clone(),
                iridescence_map: m.iridescence_map.clone(),
                sheen_color_map: m.sheen_color_map.clone(),
                sheen_roughness_map: m.sheen_roughness_map.clone(),
                transmission_map: m.transmission_map.clone(),
                thickness_map: m.thickness_map.clone(),
                anisotropy_map: m.anisotropy_map.clone(),
                ..Default::default()
            },
            Material::Matcap(m) => MaterialTextureSlots {
                matcap_map: m.matcap.clone(),
                map: m.map.clone(),
                alpha_map: m.alpha_map.clone(),
                ..Default::default()
            },
            Material::Lambert(m) => MaterialTextureSlots {
                env_map: m.env_map.clone(),
                light_map: m.light_map.clone(),
                bump_map: m.bump_map.clone(),
                specular_map: m.specular_map.clone(),
                ..lit_slots(&m.map, &m.emissive_map, &m.normal_map, &m.ao_map, &m.displacement_map, &m.alpha_map)
            },
            Material::Phong(m) => MaterialTextureSlots {
                env_map: m.env_map.clone(),
                light_map: m.light_map.clone(),
                bump_map: m.bump_map.clone(),
                specular_map: m.specular_map.clone(),
                ..lit_slots(&m.map, &m.emissive_map, &m.normal_map, &m.ao_map, &m.displacement_map, &m.alpha_map)
            },
            Material::Toon(m) => MaterialTextureSlots {
                gradient_map: m.gradient_map.clone(),
                light_map: m.light_map.clone(),
                bump_map: m.bump_map.clone(),
                ..lit_slots(&m.map, &m.emissive_map, &m.normal_map, &m.ao_map, &m.displacement_map, &m.alpha_map)
            },
            Material::Points(m) => MaterialTextureSlots {
                map: m.map.clone(),
                alpha_map: m.alpha_map.clone(),
                ..Default::default()
            },
            Material::Sprite(m) => MaterialTextureSlots {
                map: m.map.clone(),
                alpha_map: m.alpha_map.clone(),
                ..Default::default()
            },
            Material::Basic(m) => MaterialTextureSlots {
                env_map: m.env_map.clone(),
                map: m.map.clone(),
                alpha_map: m.alpha_map.clone(),
                light_map: m.light_map.clone(),
                specular_map: m.specular_map.clone(),
                ..Default::default()
            },
            Material::Mirror(m) => MaterialTextureSlots {
                map: m.map.clone(),
                ..Default::default()
            },
            _ => MaterialTextureSlots::default(),
        }
    }
}

impl From<BasicMaterial> for Material {
    fn from(m: BasicMaterial) -> Self {
        Material::Basic(m)
    }
}
impl From<LambertMaterial> for Material {
    fn from(m: LambertMaterial) -> Self {
        Material::Lambert(m)
    }
}
impl From<PhongMaterial> for Material {
    fn from(m: PhongMaterial) -> Self {
        Material::Phong(m)
    }
}
impl From<StandardMaterial> for Material {
    fn from(m: StandardMaterial) -> Self {
        Material::Standard(m)
    }
}
impl From<PhysicalMaterial> for Material {
    fn from(m: PhysicalMaterial) -> Self {
        Material::Physical(m)
    }
}
impl From<NormalMaterial> for Material {
    fn from(m: NormalMaterial) -> Self {
        Material::Normal(m)
    }
}
impl From<DepthMaterial> for Material {
    fn from(m: DepthMaterial) -> Self {
        Material::Depth(m)
    }
}
impl From<ToonMaterial> for Material {
    fn from(m: ToonMaterial) -> Self {
        Material::Toon(m)
    }
}
impl From<MatcapMaterial> for Material {
    fn from(m: MatcapMaterial) -> Self {
        Material::Matcap(m)
    }
}
impl From<LineBasicMaterial> for Material {
    fn from(m: LineBasicMaterial) -> Self {
        Material::Line(m)
    }
}
impl From<PointsMaterial> for Material {
    fn from(m: PointsMaterial) -> Self {
        Material::Points(m)
    }
}
impl From<SpriteMaterial> for Material {
    fn from(m: SpriteMaterial) -> Self {
        Material::Sprite(m)
    }
}
impl From<DistanceMaterial> for Material {
    fn from(m: DistanceMaterial) -> Self {
        Material::Distance(m)
    }
}
impl From<SkyMaterial> for Material {
    fn from(m: SkyMaterial) -> Self {
        Material::Sky(m)
    }
}
impl From<MirrorMaterial> for Material {
    fn from(m: MirrorMaterial) -> Self {
        Material::Mirror(m)
    }
}
impl From<ShaderMaterial> for Material {
    fn from(m: ShaderMaterial) -> Self {
        Material::Shader(m)
    }
}
