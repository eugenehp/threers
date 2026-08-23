use crate::math::{Color, Vector2};
use crate::textures::Texture;
use std::sync::Arc;

/// PBR roughness/metalness workflow. Matches three.js's `MeshStandardMaterial`.
#[derive(Debug, Clone)]
pub struct StandardMaterial {
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

    // Textures (all optional). When set, the corresponding GPU texture slot is
    // bound and the WGSL flag enables sampling.
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
    /// Confine the emissive map to the hemisphere facing AWAY from the sun.
    ///
    /// 0 (default) is the plain three.js behaviour: emissive is added to the
    /// shaded colour unconditionally. That is right for a lamp, and wrong for a
    /// city-lights map -- an emissive map masks WHERE the cities are, not
    /// WHETHER it is night there, so at 0 every city on the planet burns
    /// through broad daylight. Set to 1 and the term is multiplied by how far
    /// the point is past the terminator, which is what a night map means.
    pub emissive_night_side: f32,
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

    pub side: u32,
}

impl Default for StandardMaterial {
    fn default() -> Self {
        Self {
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
            map: None,
            normal_map: None,
            roughness_map: None,
            metalness_map: None,
            ao_map: None,
            emissive_map: None,
            displacement_map: None,
            eclipse_occluder: None,
            sun_angular_radius: 0.00465,
            emissive_night_side: 0.0,
            cloud_shadow_map: None,
            cloud_height: 0.012,
            cloud_shadow: 0.0,
            cloud_scatter: 0.0,
            cloud_anisotropy: 0.8,
            cloud_rotation: 0.0,
            twilight: 0.0,
            twilight_color: Color::new(1.0, 0.55, 0.32),
            side: 0,
        }
    }
}

impl StandardMaterial {
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
    pub fn with_ao_map(mut self, t: Arc<Texture>) -> Self {
        self.ao_map = Some(t);
        self
    }
    pub fn with_emissive_map(mut self, t: Arc<Texture>) -> Self {
        self.emissive_map = Some(t);
        self
    }
}
