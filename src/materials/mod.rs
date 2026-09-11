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
}
impl Default for SkyMaterial {
    fn default() -> Self {
        Self {
            sun_position: crate::math::Vector3::new(0.0, 1.0, 0.0),
            turbidity: 10.0,
            rayleigh: 3.0,
            mie_coefficient: 0.005,
            mie_directional_g: 0.7,
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
}
impl Default for DistanceMaterial {
    fn default() -> Self {
        Self {
            reference_position: crate::math::Vector3::ZERO,
            near_distance: 1.0,
            far_distance: 1000.0,
        }
    }
}
impl DistanceMaterial {
    pub fn new(reference_position: crate::math::Vector3, near: f32, far: f32) -> Self {
        Self {
            reference_position,
            near_distance: near.max(0.0),
            far_distance: far.max(near + 0.001),
        }
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
    /// Height map, sampled in the *vertex* stage to move geometry.
    pub displacement_map: Option<Arc<Texture>>,
    /// Cloud cover that casts a shadow onto this surface.
    pub cloud_shadow_map: Option<Arc<Texture>>,
    /// Thin-film thickness map (three.js `iridescenceThicknessMap`). Sampled
    /// from the green channel and remapped into the material's thickness range.
    pub iridescence_thickness_map: Option<Arc<Texture>>,
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
            // Transmissive glass needs the alpha pipeline even at opacity 1.0.
            Material::Physical(m) => m.opacity < 1.0 || m.transmission > 0.0,
            Material::Shader(m) => m.transparent || m.opacity < 1.0,
            // An atmosphere is nothing but coverage — it is never opaque, even
            // at `opacity: 1.0`, where that value only scales the optical depth.
            Material::Atmosphere(_) => true,
            // Other materials: treat opacity < 1 as transparent by default,
            // matching three.js's behavior when only opacity is set.
            _ => self.opacity() < 1.0,
        }
    }

    /// Whether the material should test the depth buffer. Overlay strokes
    /// (`LineBasicMaterial::depth_test == false`) skip the test so they stay
    /// visible through dense occluders.
    pub fn depth_test(&self) -> bool {
        match self {
            Material::Line(m) => m.depth_test,
            _ => true,
        }
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

    pub fn alpha_test(&self) -> f32 {
        match self {
            Material::Basic(m) => m.alpha_test,
            _ => 0.0,
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
            _ => 1.0,
        }
    }

    pub fn normal_scale(&self) -> Vector2 {
        match self {
            Material::Standard(m) => m.normal_scale,
            Material::Physical(m) => m.normal_scale,
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
            _ => (0.0, 0.0),
        }
    }

    pub fn depth_range(&self) -> (f32, f32) {
        match self {
            Material::Depth(m) => (m.near, m.far),
            _ => (0.0, 0.0),
        }
    }

    pub fn toon_steps(&self) -> u32 {
        match self {
            Material::Toon(m) => m.steps,
            _ => 0,
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
                matcap_map: None,
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
                matcap_map: None,
            },
            Material::Matcap(m) => MaterialTextureSlots {
                matcap_map: m.matcap.clone(),
                ..Default::default()
            },
            Material::Sprite(m) => MaterialTextureSlots {
                map: m.map.clone(),
                ..Default::default()
            },
            Material::Basic(m) => MaterialTextureSlots {
                map: m.map.clone(),
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
