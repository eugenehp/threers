use crate::math::Vector2;
use crate::textures::Texture;
use std::sync::Arc;
use crate::math::Color;

/// Diffuse + Blinn-Phong specular + emissive. Matches three.js's `MeshPhongMaterial`.
#[derive(Debug, Clone)]
pub struct PhongMaterial {
    pub color: Color,
    pub emissive: Color,
    pub specular: Color,
    pub shininess: f32,
    pub opacity: f32,
    /// three.js `alphaTest`: fragments whose alpha (opacity × map alpha) is
    /// below this are discarded. 0 (the default) keeps every fragment.
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
    /// three.js `bumpMap`: a height field (red channel) perturbing the normal
    /// through its screen-space slope; ignored under a `normal_map`, as in
    /// three.js.
    pub bump_map: Option<Arc<Texture>>,
    /// three.js `bumpScale` (default 1).
    pub bump_scale: f32,
    /// three.js `specularMap`: its red channel scales the specular term.
    pub specular_map: Option<Arc<Texture>>,
    /// three.js `transparent`. `Some` is three.js's flag as given — `false`
    /// draws opaquely whatever the opacity. `None` (the default) blends when
    /// `opacity < 1`, the rule this crate used before it had the flag.
    pub transparent: Option<bool>,
    /// three.js `map`: the diffuse colour map (its alpha multiplies opacity).
    pub map: Option<Arc<Texture>>,
    /// three.js `emissiveMap`, multiplying `emissive`.
    pub emissive_map: Option<Arc<Texture>>,
    /// three.js `normalMap` (tangent space), scaled by `normal_scale`.
    pub normal_map: Option<Arc<Texture>>,
    pub normal_scale: Vector2,
    /// three.js `aoMap` (red channel) and `aoMapIntensity`: occludes the
    /// indirect (ambient and hemisphere) light.
    pub ao_map: Option<Arc<Texture>>,
    pub ao_intensity: f32,
    /// three.js `displacementMap`, `displacementScale`, `displacementBias`.
    pub displacement_map: Option<Arc<Texture>>,
    pub displacement_scale: f32,
    pub displacement_bias: f32,
    pub wireframe: bool,
    pub side: u32,
    /// three.js `polygonOffset` / `polygonOffsetFactor` / `polygonOffsetUnits`:
    /// `Some` enables the depth offset (see [`PolygonOffset`](super::PolygonOffset)).
    pub polygon_offset: Option<super::PolygonOffset>,
    /// three.js `toneMapped` (default `true`): whether the renderer's tone
    /// mapping applies to this material's output.
    pub tone_mapped: bool,
    /// three.js `depthTest` / `depthWrite` / `depthFunc` / `colorWrite`.
    pub render_state: super::RenderState,
}

impl Default for PhongMaterial {
    fn default() -> Self {
        Self {
            color: Color::WHITE,
            emissive: Color::BLACK,
            // three.js's `0x111111`, which is sRGB: 0.0056 in linear.
            specular: Color::from_hex(0x111111),
            shininess: 30.0,
            opacity: 1.0,
            alpha_test: 0.0,
            alpha_map: None,
            env_map: None,
            combine: super::Combine::Multiply,
            reflectivity: 1.0,
            refraction_ratio: 0.98,
            env_map_rotation: crate::math::Euler::default(),
            light_map: None,
            light_map_intensity: 1.0,
            bump_map: None,
            bump_scale: 1.0,
            specular_map: None,
            transparent: None,
            map: None,
            emissive_map: None,
            normal_map: None,
            normal_scale: Vector2::ONE,
            ao_map: None,
            ao_intensity: 1.0,
            displacement_map: None,
            displacement_scale: 1.0,
            displacement_bias: 0.0,
            wireframe: false,
            side: 0,
            polygon_offset: None,
            tone_mapped: true,
            render_state: super::RenderState::default(),
        }
    }
}

impl PhongMaterial {
    pub fn new(color: Color) -> Self {
        Self {
            color,
            ..Default::default()
        }
    }

    pub fn with_specular(mut self, specular: Color, shininess: f32) -> Self {
        self.specular = specular;
        self.shininess = shininess;
        self
    }

    pub fn with_emissive(mut self, c: Color) -> Self {
        self.emissive = c;
        self
    }
}
