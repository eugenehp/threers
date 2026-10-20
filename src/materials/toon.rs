use crate::math::Color;
use crate::math::Vector2;
use crate::textures::Texture;
use std::sync::Arc;

/// Stepped/cel-shaded NPR material. Matches three.js's `MeshToonMaterial`:
/// every light's irradiance goes through `getGradientIrradiance` and then
/// `BRDF_Lambert` (÷π), ambient and hemisphere light through `BRDF_Lambert`.
#[derive(Debug, Clone)]
pub struct ToonMaterial {
    pub color: Color,
    pub emissive: Color,
    pub opacity: f32,
    /// three.js `alphaTest`: fragments whose alpha (opacity × map alpha) is
    /// below this are discarded. 0 (the default) keeps every fragment.
    pub alpha_test: f32,
    /// three.js `alphaMap`: its green channel multiplies the opacity, before
    /// `alphaTest` (`alphamap_fragment`). Shows only with `transparent` or
    /// `alpha_test`, as in three.js.
    pub alpha_map: Option<Arc<Texture>>,
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
    /// three.js `gradientMap`: the ramp each light's irradiance is read
    /// through, at `(n·l/2 + ½, 0)`, its rgb multiplying the light
    /// (`getGradientIrradiance`). Sampled with the texture's own filter —
    /// three.js's examples set `NearestFilter` for hard bands. `None` (the
    /// default) is three.js without one: two tones, 0.7 of a light below
    /// `n·l = 0.4` and all of it above.
    pub gradient_map: Option<Arc<Texture>>,
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

impl Default for ToonMaterial {
    fn default() -> Self {
        Self {
            color: Color::WHITE,
            emissive: Color::BLACK,
            opacity: 1.0,
            alpha_test: 0.0,
            alpha_map: None,
            light_map: None,
            light_map_intensity: 1.0,
            bump_map: None,
            bump_scale: 1.0,
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
            gradient_map: None,
            wireframe: false,
            side: 0,
            polygon_offset: None,
            tone_mapped: true,
            render_state: super::RenderState::default(),
        }
    }
}

impl ToonMaterial {
    pub fn new(color: Color) -> Self {
        Self {
            color,
            ..Default::default()
        }
    }

    /// An `n`-texel grey ramp, texel `i` holding `i/(n − 1)`, nearest
    /// filtered — the `threeTone.jpg` / `fiveTone.jpg` gradient maps three.js's
    /// toon example loads, as data. `n` below 2 is taken as 2.
    pub fn gradient_steps(n: u32) -> Arc<Texture> {
        let n = n.max(2);
        let data = (0..n)
            .flat_map(|i| {
                let v = (i as f32 / (n - 1) as f32 * 255.0).round() as u8;
                [v, v, v, 255]
            })
            .collect();
        let mut t = Texture::new(n, 1, crate::textures::TextureFormat::Rgba8Unorm, data);
        t.mag_filter = crate::textures::TextureFilter::Nearest;
        t.min_filter = crate::textures::TextureFilter::Nearest;
        t.mipmap_filter = None;
        Arc::new(t)
    }

    /// With a gradient map.
    pub fn with_gradient_map(mut self, map: Arc<Texture>) -> Self {
        self.gradient_map = Some(map);
        self
    }
}
