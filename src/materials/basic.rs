use crate::math::Color;
use crate::textures::Texture;
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct BasicMaterial {
    pub color: Color,
    pub wireframe: bool,
    pub map: Option<Arc<Texture>>,
    pub transparent: bool,
    pub opacity: f32,
    /// 0=FrontSide (default cull-back), 1=BackSide (cull-front),
    /// 2=DoubleSide (no cull). Both 1 and 2 currently route to the
    /// no-cull pipeline.
    pub side: u32,
    /// When true, only shadow darkness is visible (three.js ShadowMaterial).
    pub shadow_only: bool,
    pub alpha_test: f32,
    /// three.js `alphaMap`: its green channel multiplies the opacity, before
    /// `alphaTest` (`alphamap_fragment`). Shows only with `transparent` or
    /// `alpha_test`, as in three.js.
    pub alpha_map: Option<Arc<Texture>>,
    /// three.js `envMap`: a cube map reflected (or, under
    /// [`CubeMapping::Refraction`](crate::textures::CubeMapping), refracted)
    /// and combined with the shaded colour per [`combine`](Self::combine).
    pub env_map: Option<Arc<crate::textures::CubeTexture>>,
    /// three.js `combine` (default `Multiply`).
    pub combine: super::Combine,
    /// three.js `reflectivity` (default 1): the environment's weight.
    pub reflectivity: f32,
    /// three.js `refractionRatio` (default 0.98), for a refraction-mapped
    /// `env_map`.
    pub refraction_ratio: f32,
    /// three.js `envMapRotation` (default zero).
    pub env_map_rotation: crate::math::Euler,
    /// three.js `lightMap`: baked irradiance added to the indirect light, times
    /// [`light_map_intensity`](Self::light_map_intensity).
    pub light_map: Option<Arc<Texture>>,
    /// three.js `lightMapIntensity` (default 1).
    pub light_map_intensity: f32,
    /// three.js `specularMap`: its red channel scales the specular term.
    pub specular_map: Option<Arc<Texture>>,
    /// three.js `polygonOffset` / `polygonOffsetFactor` / `polygonOffsetUnits`:
    /// `Some` enables the depth offset (see [`PolygonOffset`](super::PolygonOffset)).
    pub polygon_offset: Option<super::PolygonOffset>,
    /// three.js `toneMapped` (default `true`): whether the renderer's tone
    /// mapping applies to this material's output.
    pub tone_mapped: bool,
    /// three.js `depthTest` / `depthWrite` / `depthFunc` / `colorWrite`.
    pub render_state: super::RenderState,
}

impl Default for BasicMaterial {
    fn default() -> Self {
        Self {
            color: Color::WHITE,
            wireframe: false,
            map: None,
            transparent: false,
            opacity: 1.0,
            side: 0,
            shadow_only: false,
            alpha_test: 0.0,
            alpha_map: None,
            env_map: None,
            combine: super::Combine::Multiply,
            reflectivity: 1.0,
            refraction_ratio: 0.98,
            env_map_rotation: crate::math::Euler::default(),
            light_map: None,
            light_map_intensity: 1.0,
            specular_map: None,
            polygon_offset: None,
            tone_mapped: true,
            render_state: super::RenderState::default(),
        }
    }
}

impl BasicMaterial {
    pub fn new(color: Color) -> Self {
        Self {
            color,
            wireframe: false,
            map: None,
            transparent: false,
            opacity: 1.0,
            side: 0,
            shadow_only: false,
            alpha_test: 0.0,
            alpha_map: None,
            env_map: None,
            combine: super::Combine::Multiply,
            reflectivity: 1.0,
            refraction_ratio: 0.98,
            env_map_rotation: crate::math::Euler::default(),
            light_map: None,
            light_map_intensity: 1.0,
            specular_map: None,
            polygon_offset: None,
            tone_mapped: true,
            render_state: super::RenderState::default(),
        }
    }

    pub fn wireframe(mut self, w: bool) -> Self {
        self.wireframe = w;
        self
    }

    pub fn with_opacity(mut self, opacity: f32) -> Self {
        self.opacity = opacity;
        self
    }

    pub fn with_transparent(mut self, transparent: bool) -> Self {
        self.transparent = transparent;
        self
    }
}
