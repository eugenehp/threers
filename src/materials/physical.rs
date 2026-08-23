use crate::math::{Color, Vector2};
use crate::textures::Texture;
use std::sync::Arc;

/// Extended PBR with clearcoat, IOR, transmission, sheen, iridescence and
/// anisotropy layers. Matches three.js's `MeshPhysicalMaterial`.
///
/// All layers below are honored by the renderer. Two notes on fidelity:
///
/// - **Metals need an environment.** At `metalness = 1.0` the BRDF has no
///   diffuse term, so a metal with no `Scene::environment` renders near-black.
///   Set one (see [`crate::extras::PmremGenerator`]) before reaching for gold,
///   silver or foil looks.
/// - **Anisotropy needs UVs.** The tangent frame is derived from UV screen-space
///   derivatives so `anisotropy_rotation` is meaningful in texture space. On a
///   mesh with no UVs it falls back to a position-derived frame, where the
///   streak direction is view-dependent and the rotation is arbitrary.
#[derive(Debug, Clone)]
pub struct PhysicalMaterial {
    pub color: Color,
    pub emissive: Color,
    pub emissive_intensity: f32,
    pub roughness: f32,
    pub metalness: f32,
    pub ao_intensity: f32,
    pub normal_scale: Vector2,
    /// How far the displacement map pushes, in world units per unit of red.
    pub displacement_scale: f32,
    /// Constant offset added after scaling — use it to centre a height map
    /// whose midpoint is sea level rather than zero.
    pub displacement_bias: f32,
    pub opacity: f32,
    pub wireframe: bool,

    pub clearcoat: f32,
    pub clearcoat_roughness: f32,
    pub ior: f32,
    pub transmission: f32,
    pub thickness: f32,
    /// Chromatic dispersion (Abbe-like). 0 = none. Splits the refraction IOR per
    /// RGB channel so the screen-space-glass path shows a rainbow edge fringe.
    pub dispersion: f32,
    /// Per-vertex emission: adds `vertexColor.rgb × vertexColor.a` as emitted
    /// light, scaled by this factor (0 = off). A general mechanism for data-baked
    /// or decal glow — the alpha channel gates *where* it emits and the rgb sets
    /// the color, both authored into the mesh's vertex colors by the app.
    pub vertex_emissive: f32,
    pub sheen: f32,
    pub sheen_color: Color,
    pub sheen_roughness: f32,
    pub iridescence: f32,
    pub iridescence_ior: f32,
    /// Thin-film thickness in **nanometres**. This is what sets the hue of the
    /// interference: ~100nm reads gold/magenta, ~400nm cyan/blue, and sweeping
    /// it walks the whole rainbow. Only used when `iridescence > 0`.
    pub iridescence_thickness: f32,
    /// Anisotropy strength (0 = isotropic). Stretches the specular highlight
    /// along the surface tangent for a brushed/streaked look.
    pub anisotropy: f32,
    /// Anisotropy direction as a rotation (radians) of the tangent in the
    /// surface plane.
    pub anisotropy_rotation: f32,
    /// Distance (world units) over which transmitted light is absorbed down to
    /// `attenuation_color`. `INFINITY` (the default) disables volume absorption,
    /// in which case transmission falls back to tinting by the base `color` —
    /// the historical threers behaviour. Set this finite to opt into physical
    /// Beer-Lambert absorption instead.
    pub attenuation_distance: f32,
    /// The color transmitted light trends toward at `attenuation_distance`.
    /// Only used when `attenuation_distance` is finite and positive.
    pub attenuation_color: Color,

    pub map: Option<Arc<Texture>>,
    pub normal_map: Option<Arc<Texture>>,
    pub roughness_map: Option<Arc<Texture>>,
    pub metalness_map: Option<Arc<Texture>>,
    pub ao_map: Option<Arc<Texture>>,
    pub emissive_map: Option<Arc<Texture>>,
    /// A sphere that can eclipse the sun for this surface: world-space
    /// `[x, y, z, radius]`, or `None`.
    ///
    /// Both the sun and the occluder are circles on the sky from any shaded
    /// point, so the light lost is the area their discs share — which gives a
    /// penumbra for free, and an annular ring when the occluder is smaller than
    /// the sun and centred on it. Shadow maps cannot do either at this scale:
    /// a map fitted to a planet would not reach a moon sixty radii away.
    pub eclipse_occluder: Option<[f32; 4]>,
    /// Angular *radius* of the sun in radians, as seen from this body. 0.00465
    /// is the Sun from Earth — half a degree across.
    pub sun_angular_radius: f32,
    /// Cloud map whose alpha casts a shadow onto this surface.
    ///
    /// The shader intersects the sun ray from each shaded point with a shell
    /// `cloud_height` above the body and samples the map where it crosses, so
    /// the shadow lands offset from the cloud that casts it — further as the
    /// sun gets lower, which is what makes a cloud deck read as floating above
    /// the ground rather than painted onto it. Assumes the mesh is a sphere
    /// centred on its own origin, equirectangularly mapped.
    pub cloud_shadow_map: Option<Arc<Texture>>,
    /// Height of the cloud shell, as a fraction of the body's radius.
    pub cloud_height: f32,
    /// How dark the cloud shadow gets, 0 (off) to 1.
    pub cloud_shadow: f32,
    /// Longitude offset of the cloud shell relative to this surface, in turns.
    /// Clouds drift; without this the shadows would be pinned to the ground.
    pub cloud_rotation: f32,
    /// Forward scattering through a cloud deck. 0 disables it.
    ///
    /// A cloud is not a surface. Sunlight enters the top, bounces between
    /// droplets and leaves in a direction strongly biased *forward*, which is
    /// why a deck with the sun behind it is far brighter than the same deck lit
    /// from behind the camera, and why its edge lights up before its middle
    /// does. A Lambert shell cannot do either: it is at its brightest facing
    /// the sun and goes flat black past the terminator.
    pub cloud_scatter: f32,
    /// Henyey-Greenstein asymmetry for that scattering, -1 to 1.
    ///
    /// Cloud droplets are large compared with visible wavelengths, so they
    /// scatter hard forward: around 0.8 for water, which is the default.
    pub cloud_anisotropy: f32,
    /// Width of the twilight wrap, in cosine units. 0 disables it.
    ///
    /// A planet with air is not lit by a hard Lambert terminator: light
    /// scatters round the limb and reaches a little way onto the night side,
    /// reddened by the length of the path it took.
    ///
    /// Keep it small — around 0.1 for Earth. The wide orange band you see round
    /// a night limb from space is the *atmosphere*, which
    /// [`AtmosphereMaterial`](crate::AtmosphereMaterial) draws; light reaching
    /// the *ground* is down to a four-hundredth of daylight six degrees past
    /// the terminator, and gone by eighteen.
    pub twilight: f32,
    /// Colour of that scattered light — the long path through air reddens it.
    pub twilight_color: Color,
    /// Height map. Offsets each vertex along its normal by
    /// `red * displacement_scale + displacement_bias`, so it moves real
    /// geometry rather than faking relief in the shading — which means it needs
    /// a mesh dense enough to have vertices where the detail is.
    pub displacement_map: Option<Arc<Texture>>,
    /// Per-texel thin-film thickness (three.js `iridescenceThicknessMap`).
    /// The green channel scales between 0 and `iridescence_thickness` nm, so a
    /// noise or gradient map turns a single hue into an oil-slick sweep.
    pub iridescence_thickness_map: Option<Arc<Texture>>,

    pub side: u32,

    /// Transparency compositing technique (blend / single-layer glass / OIT).
    pub transparency: super::TransparencyMode,
}

impl Default for PhysicalMaterial {
    fn default() -> Self {
        Self {
            transparency: super::TransparencyMode::default(),
            color: Color::WHITE,
            emissive: Color::BLACK,
            emissive_intensity: 1.0,
            roughness: 1.0,
            metalness: 0.0,
            ao_intensity: 1.0,
            normal_scale: Vector2::new(1.0, 1.0),
            displacement_scale: 1.0,
            displacement_bias: 0.0,
            opacity: 1.0,
            wireframe: false,
            clearcoat: 0.0,
            clearcoat_roughness: 0.0,
            ior: 1.5,
            transmission: 0.0,
            thickness: 0.01,
            dispersion: 0.0,
            vertex_emissive: 0.0,
            sheen: 0.0,
            sheen_color: Color::BLACK,
            sheen_roughness: 1.0,
            iridescence: 0.0,
            iridescence_ior: 1.3,
            iridescence_thickness: 400.0,
            anisotropy: 0.0,
            anisotropy_rotation: 0.0,
            attenuation_distance: f32::INFINITY,
            attenuation_color: Color::WHITE,
            map: None,
            normal_map: None,
            roughness_map: None,
            metalness_map: None,
            ao_map: None,
            emissive_map: None,
            displacement_map: None,
            eclipse_occluder: None,
            sun_angular_radius: 0.00465,
            cloud_shadow_map: None,
            cloud_height: 0.012,
            cloud_shadow: 0.0,
            cloud_scatter: 0.0,
            cloud_anisotropy: 0.8,
            cloud_rotation: 0.0,
            twilight: 0.0,
            twilight_color: Color::new(1.0, 0.55, 0.32),
            iridescence_thickness_map: None,
            side: 0,
        }
    }
}

impl PhysicalMaterial {
    pub fn new(color: Color) -> Self {
        Self {
            color,
            ..Default::default()
        }
    }

    pub fn with_roughness(mut self, r: f32) -> Self {
        self.roughness = r;
        self
    }

    pub fn with_metalness(mut self, m: f32) -> Self {
        self.metalness = m;
        self
    }

    pub fn with_emissive(mut self, c: Color, intensity: f32) -> Self {
        self.emissive = c;
        self.emissive_intensity = intensity;
        self
    }

    /// Clear lacquer / cover-glass layer over the base: `strength` in `[0,1]`.
    pub fn with_clearcoat(mut self, strength: f32, roughness: f32) -> Self {
        self.clearcoat = strength;
        self.clearcoat_roughness = roughness;
        self
    }

    /// Brushed/streaked specular. `strength` in [-1,1] (sign flips the streak
    /// axis), `rotation` in radians within the UV tangent plane.
    pub fn with_anisotropy(mut self, strength: f32, rotation: f32) -> Self {
        self.anisotropy = strength;
        self.anisotropy_rotation = rotation;
        self
    }

    /// Retroreflective fabric-style grazing lobe (Charlie distribution).
    pub fn with_sheen(mut self, strength: f32, color: Color, roughness: f32) -> Self {
        self.sheen = strength;
        self.sheen_color = color;
        self.sheen_roughness = roughness;
        self
    }

    /// Thin-film interference. `thickness_nm` sets the hue (see the field docs).
    pub fn with_iridescence(mut self, strength: f32, ior: f32, thickness_nm: f32) -> Self {
        self.iridescence = strength;
        self.iridescence_ior = ior;
        self.iridescence_thickness = thickness_nm;
        self
    }

    /// See-through refraction. Pair with [`Self::with_attenuation`] for tinted
    /// glass and `TransparencyMode::Refract` for the screen-space path.
    pub fn with_transmission(mut self, transmission: f32, ior: f32, thickness: f32) -> Self {
        self.transmission = transmission;
        self.ior = ior;
        self.thickness = thickness;
        self
    }

    /// Beer-Lambert volume absorption: transmitted light trends toward `color`
    /// over `distance` world units.
    pub fn with_attenuation(mut self, color: Color, distance: f32) -> Self {
        self.attenuation_color = color;
        self.attenuation_distance = distance;
        self
    }

    pub fn with_map(mut self, t: Arc<Texture>) -> Self {
        self.map = Some(t);
        self
    }

    pub fn with_normal_map(mut self, t: Arc<Texture>) -> Self {
        self.normal_map = Some(t);
        self
    }

    pub fn with_roughness_map(mut self, t: Arc<Texture>) -> Self {
        self.roughness_map = Some(t);
        self
    }

    pub fn with_metalness_map(mut self, t: Arc<Texture>) -> Self {
        self.metalness_map = Some(t);
        self
    }

    /// Attach a thin-film thickness map; see the field docs.
    pub fn with_iridescence_thickness_map(mut self, t: Arc<Texture>) -> Self {
        self.iridescence_thickness_map = Some(t);
        self
    }

    pub fn with_transparency(mut self, mode: super::TransparencyMode) -> Self {
        self.transparency = mode;
        self
    }
}
