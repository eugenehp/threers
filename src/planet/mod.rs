//! Planets, moons and starfields — behind the `planet` feature.
//!
//! Assembling a believable planet is a handful of objects that all have to
//! agree with each other: a textured surface, a cloud shell a little above it,
//! an atmosphere shell above that, and a sky box around everything. This module
//! is those objects, plus the map generation and derivation to feed them.
//!
//! ```
//! use threers::planet::{generate_earth_maps, Planet, Starfield};
//! use threers::Scene;
//!
//! let mut scene = Scene::new();
//! let earth = Planet::earth().maps(generate_earth_maps(512));
//! let handles = earth.add_to(&mut scene);
//! assert!(handles.clouds.is_some() && handles.atmosphere.is_some());
//!
//! Starfield::procedural(1024).add_to(&mut scene);
//! ```
//!
//! Real imagery works the same way — hand [`PlanetMaps`] textures you decoded
//! yourself instead of generated ones. `scripts/fetch-earth-textures.sh` pulls
//! NASA's public-domain Blue Marble, Black Marble, MODIS cloud, GEBCO
//! elevation, Deep Star Maps and CGI Moon Kit products for exactly that.
//!
//! Nothing here is required to render a sphere with a texture on it; it exists
//! so you do not have to rediscover the details — that colour maps are sRGB and
//! data maps are not, that longitude has to wrap and latitude must not, that a
//! cloud layer needs its shape in the alpha channel, and that an atmosphere is
//! not a surface.

mod maps;
#[cfg(not(target_arch = "wasm32"))]
mod nasa;

pub use maps::{
    blackbody_rgb, blend_polar_cap, blend_polar_cap_in, direction, ease_poles,
    equirect_to_cube_faces, fbm, generate_earth_maps, generate_starfield, ridged, value_noise,
    HeightField, MapBuffer, PlanetMaps, CUBE_FACES,
};
#[cfg(not(target_arch = "wasm32"))]
pub use nasa::{clouds_from_grey, EarthTextures, MapSources};

use std::sync::Arc;

use crate::core::{Mesh, Object3D, ObjectId};
use crate::geometries::SphereGeometry;
use crate::materials::{AtmosphereMaterial, BasicMaterial, Material, StandardMaterial};
use crate::math::{Color, Quaternion, Vector3};
use crate::scene::Scene;
use crate::textures::Texture;

/// What [`Planet::add_to`] put in the scene, so you can animate it afterwards.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlanetHandles {
    /// The solid body.
    pub surface: ObjectId,
    /// The cloud shell, when the planet has one.
    pub clouds: Option<ObjectId>,
    /// The atmosphere shell, when the planet has one.
    pub atmosphere: Option<ObjectId>,
}

/// How thick and what colour a planet's air is.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Atmosphere {
    /// Shell height as a fraction of the planet's radius.
    pub height: f32,
    /// Scattering tint — what the air does to light passing through it.
    pub color: Color,
    /// Colour along the terminator, where the light has grazed furthest.
    pub sunset_color: Color,
    /// Overall strength.
    pub intensity: f32,
    /// How sharply density falls off with altitude.
    pub falloff: f32,
    /// Ozone absorption. See [`AtmosphereMaterial::ozone`].
    pub ozone: f32,
    /// Aurora brightness. See [`AtmosphereMaterial::aurora`].
    pub aurora: f32,
    /// Colour of the aurora's green base.
    pub aurora_color: Color,
    /// Angular radius of the auroral oval, in degrees from the pole.
    pub aurora_colatitude: f32,
}

impl Default for Atmosphere {
    /// Earth's air.
    fn default() -> Self {
        Self {
            // Earth's visible haze is nearer 1.6% of the radius, but at that
            // depth with a sharp density falloff the glow lands inside ~10px on
            // a 700px globe and reads as an outline, not as air. 4.5% with a
            // gentler falloff spreads it into something you can see.
            height: 0.045,
            color: Color::new(0.30, 0.55, 1.0),
            sunset_color: Color::new(1.0, 0.48, 0.20),
            intensity: 2.6,
            falloff: 2.3,
            // Enough to keep the limb blue where the path is longest, which is
            // what separates sky from haze.
            ozone: 0.9,
            aurora: 0.0,
            aurora_color: Color::new(0.25, 1.0, 0.45),
            aurora_colatitude: 23.0,
        }
    }
}

impl Atmosphere {
    /// A thin, cold, blue atmosphere — Earth's.
    pub fn earth() -> Self {
        Self::default()
    }

    /// A thin dusty one, tinted butterscotch.
    pub fn mars() -> Self {
        Self {
            height: 0.012,
            color: Color::new(0.82, 0.55, 0.36),
            sunset_color: Color::new(0.45, 0.55, 0.85), // Mars' sunsets are blue
            intensity: 1.2,
            falloff: 2.4,
            // No ozone layer to speak of, and no global field to funnel the
            // solar wind into an oval.
            ozone: 0.0,
            ..Self::default()
        }
    }

    /// Scattering tint.
    pub fn color(mut self, color: Color) -> Self {
        self.color = color;
        self
    }
    /// Terminator colour.
    pub fn sunset_color(mut self, color: Color) -> Self {
        self.sunset_color = color;
        self
    }
    /// Shell height as a fraction of the planet radius.
    pub fn height(mut self, height: f32) -> Self {
        self.height = height.clamp(1e-4, 1.0);
        self
    }
    /// Overall strength.
    pub fn intensity(mut self, intensity: f32) -> Self {
        self.intensity = intensity.max(0.0);
        self
    }
    /// Density falloff with altitude.
    pub fn falloff(mut self, falloff: f32) -> Self {
        self.falloff = falloff.clamp(0.1, 32.0);
        self
    }

    /// The material for a planet of `radius`.
    pub fn material(&self, radius: f32) -> AtmosphereMaterial {
        AtmosphereMaterial::new(radius, radius * (1.0 + self.height))
            .color(self.color)
            .sunset_color(self.sunset_color)
            .intensity(self.intensity)
            .falloff(self.falloff)
            .ozone(self.ozone)
            .aurora(self.aurora, self.aurora_color, self.aurora_colatitude)
    }
}

/// A planet: a textured body, optionally wrapped in cloud and air.
///
/// ```
/// use threers::planet::{Atmosphere, Planet};
/// use threers::Vector3;
///
/// let mars = Planet::new(0.53)
///     .atmosphere(Some(Atmosphere::mars()))
///     .position(Vector3::new(8.0, 0.0, 0.0))
///     .tilt_degrees(25.2);
/// assert_eq!(mars.radius, 0.53);
/// ```
#[derive(Debug, Clone)]
pub struct Planet {
    /// Body radius in world units.
    pub radius: f32,
    /// Centre position.
    pub position: Vector3,
    /// Surface, cloud and night maps.
    pub maps: PlanetMaps,
    /// Sphere tessellation, `(longitude, latitude)` segments.
    pub segments: (usize, usize),
    /// Cloud shell height as a fraction of the radius. Ignored without a cloud
    /// map.
    pub cloud_height: f32,
    /// Cloud opacity. Below 1 so the shell reaches the alpha pipeline, where
    /// the map's own alpha shapes it.
    pub cloud_opacity: f32,
    /// Forward scattering through the cloud deck. See
    /// [`StandardMaterial::cloud_scatter`](crate::StandardMaterial::cloud_scatter).
    pub cloud_scatter: f32,
    /// Angular radius of this system's sun, in radians. Softens the terminator
    /// and sizes the penumbra of an eclipse; 0.00465 is the Sun from Earth.
    pub sun_angular_radius: f32,
    /// Air, if the planet has any.
    pub atmosphere: Option<Atmosphere>,
    /// Brightness of the emissive night map.
    pub night_intensity: f32,
    /// Base colour where there is no albedo map.
    pub color: Color,
    /// Roughness, scaled by the roughness map where one is bound.
    pub roughness: f32,
    /// How far the height map pushes the surface, in world units.
    ///
    /// Zero by default, and deliberately so: at true scale Everest is 0.14% of
    /// Earth's radius, which is under a pixel on any view that shows the whole
    /// planet. Displacement only reads when it is exaggerated, and by how much
    /// is a judgement about the picture, not about the planet.
    pub displacement: f32,
    /// Constant offset applied after the height map is scaled. Set it to
    /// `-displacement / 2` to keep a centred height map from inflating the
    /// body — the map's midpoint then lands exactly on the sphere.
    pub displacement_bias: f32,
    /// Metalness. Rock and water are both dielectric, so this is normally 0.
    pub metalness: f32,
    /// Axial tilt in radians, about +X.
    pub tilt: f32,
    /// Rotation about the (tilted) polar axis, in radians.
    pub spin: f32,
}

impl Default for Planet {
    fn default() -> Self {
        Self::new(1.0)
    }
}

impl Planet {
    /// A bare body of `radius` — no maps, no clouds, no air.
    pub fn new(radius: f32) -> Self {
        Self {
            radius: radius.max(1e-4),
            position: Vector3::ZERO,
            maps: PlanetMaps::new(),
            segments: (192, 96),
            cloud_height: 0.012,
            cloud_opacity: 0.96,
            cloud_scatter: 0.35,
            sun_angular_radius: 0.00465,
            atmosphere: None,
            night_intensity: 2.3,
            color: Color::WHITE,
            roughness: 1.0,
            displacement: 0.0,
            displacement_bias: 0.0,
            metalness: 0.0,
            tilt: 0.0,
            spin: 0.0,
        }
    }

    /// Earth: unit radius, 23.44° of tilt, and an atmosphere. Add maps.
    ///
    /// ```
    /// use threers::planet::Planet;
    /// let earth = Planet::earth();
    /// assert!(earth.atmosphere.is_some());
    /// assert!((earth.tilt.to_degrees() - 23.44).abs() < 1e-3);
    /// ```
    pub fn earth() -> Self {
        Self {
            atmosphere: Some(Atmosphere::earth()),
            tilt: 23.44f32.to_radians(),
            ..Self::new(1.0)
        }
    }

    /// The Moon: airless, cloudless, and rough. Radius is Earth-relative.
    pub fn moon() -> Self {
        Self {
            radius: 0.2727,
            roughness: 0.95,
            night_intensity: 0.0,
            // Craters are the Moon's entire character, and displacement can
            // only be as detailed as the mesh carrying it. 384x192 is 74k
            // vertices — trivial for one body, and enough that crater rims
            // break the silhouette instead of being painted onto a circle.
            segments: (384, 192),
            atmosphere: None,
            ..Self::new(0.2727)
        }
    }

    /// Surface, cloud and night maps.
    pub fn maps(mut self, maps: PlanetMaps) -> Self {
        self.maps = maps;
        self
    }
    /// Centre position.
    pub fn position(mut self, position: Vector3) -> Self {
        self.position = position;
        self
    }
    /// Sphere tessellation.
    pub fn segments(mut self, longitude: usize, latitude: usize) -> Self {
        self.segments = (longitude.max(8), latitude.max(4));
        self
    }
    /// Air, or `None` for an airless body.
    pub fn atmosphere(mut self, atmosphere: Option<Atmosphere>) -> Self {
        self.atmosphere = atmosphere;
        self
    }
    /// Axial tilt.
    pub fn tilt_degrees(mut self, degrees: f32) -> Self {
        self.tilt = degrees.to_radians();
        self
    }
    /// Rotation about the polar axis.
    pub fn spin_degrees(mut self, degrees: f32) -> Self {
        self.spin = degrees.to_radians();
        self
    }
    /// Brightness of the night map.
    pub fn night_intensity(mut self, intensity: f32) -> Self {
        self.night_intensity = intensity.max(0.0);
        self
    }
    /// Base colour where no albedo map is bound.
    pub fn color(mut self, color: Color) -> Self {
        self.color = color;
        self
    }
    /// Surface roughness, scaled by the roughness map.
    pub fn roughness(mut self, roughness: f32) -> Self {
        self.roughness = roughness.clamp(0.0, 1.0);
        self
    }

    /// Height-map amplitude in world units, measured up from the sphere.
    ///
    /// For a map whose zero means "the surface" — Earth's, where
    /// [`HeightField::to_height_map`] puts sea level at 0. Needs a displacement
    /// map and a mesh dense enough to carry it; see
    /// [`segments`](Self::segments).
    pub fn displacement(mut self, world_units: f32) -> Self {
        self.displacement = world_units.max(0.0);
        self.displacement_bias = 0.0;
        self
    }

    /// Height-map amplitude in world units, centred on the sphere.
    ///
    /// For a map whose *midpoint* means "the surface" — anything from
    /// [`HeightField::to_height_map_normalized`], where the range is stretched
    /// and there is no natural zero. Half the amplitude falls below the
    /// original radius and half above, so the body keeps its mean size instead
    /// of swelling by the map's average.
    pub fn displacement_centered(mut self, world_units: f32) -> Self {
        self.displacement = world_units.max(0.0);
        self.displacement_bias = -self.displacement * 0.5;
        self
    }

    /// Constant offset applied after scaling the height map.
    pub fn displacement_bias(mut self, bias: f32) -> Self {
        self.displacement_bias = bias;
        self
    }

    /// Orientation from the tilt and spin.
    fn orientation(&self) -> Quaternion {
        Quaternion::from_euler_xyz(self.tilt, self.spin, 0.0)
    }

    /// The surface material.
    fn surface_material(&self) -> StandardMaterial {
        let mut m = StandardMaterial::new(self.color);
        m.map = self.maps.albedo.clone();
        m.normal_map = self.maps.normal.clone();
        m.roughness_map = self.maps.roughness.clone();
        m.roughness = self.roughness;
        m.metalness = self.metalness;
        if self.displacement > 0.0 {
            m.displacement_map = self.maps.displacement.clone();
            m.displacement_scale = self.displacement;
            m.displacement_bias = self.displacement_bias;
        }
        if let Some(night) = self.maps.night.clone() {
            // The night side is lit only by its own cities.
            //
            // The map alone does NOT achieve that, though this said it did: it
            // masks where the cities are, not whether it is night there, and
            // emissive is added to the shaded colour whatever the sun is doing.
            // Every city on the planet therefore glowed through full daylight,
            // which is invisible from far away and ruins any close pass over
            // land. `emissive_night_side` is what actually confines it.
            m.emissive_map = Some(night);
            m.emissive = Color::WHITE;
            m.emissive_intensity = self.night_intensity;
            m.emissive_night_side = 1.0;
        }
        m
    }

    /// Add the body — and its cloud and atmosphere shells — to `scene`.
    pub fn add_to(&self, scene: &mut Scene) -> PlanetHandles {
        let orientation = self.orientation();

        let mut body = Object3D::mesh(Mesh::new(
            SphereGeometry::new(self.radius, self.segments.0, self.segments.1),
            Material::Standard(self.surface_material()),
        ));
        body.position = self.position;
        body.quaternion = orientation;
        let surface = scene.add(body);

        let clouds = self.maps.clouds.clone().map(|map| {
            let mut m = StandardMaterial::new(Color::WHITE);
            m.map = Some(map);
            m.roughness = 0.95;
            m.metalness = 0.0;
            // A cloud is a volume, not a surface — see `cloud_scatter`.
            m.cloud_scatter = self.cloud_scatter;
            m.cloud_anisotropy = 0.8;
            m.sun_angular_radius = self.sun_angular_radius;
            // Below 1 so the shell lands on the alpha pipeline; the map's alpha
            // does the shaping from there.
            m.opacity = self.cloud_opacity.min(0.999);
            let mut shell = Object3D::mesh(Mesh::new(
                SphereGeometry::new(
                    self.radius * (1.0 + self.cloud_height),
                    self.segments.0 * 3 / 4,
                    self.segments.1 * 3 / 4,
                ),
                Material::Standard(m),
            ));
            shell.name = "clouds".into();
            shell.position = self.position;
            shell.quaternion = orientation;
            scene.add(shell)
        });

        let atmosphere = self.atmosphere.map(|air| {
            let mut shell = Object3D::mesh(Mesh::new(
                SphereGeometry::new(self.radius * (1.0 + air.height), 128, 64),
                Material::Atmosphere(air.material(self.radius)),
            ));
            shell.name = "atmosphere".into();
            shell.position = self.position;
            scene.add(shell)
        });

        PlanetHandles {
            surface,
            clouds,
            atmosphere,
        }
    }
}

/// The sky: an unlit inverted sphere far enough out to sit behind everything.
///
/// ```
/// use threers::planet::Starfield;
/// use threers::Scene;
/// let mut scene = Scene::new();
/// let sky = Starfield::procedural(256).radius(500.0);
/// let id = sky.add_to(&mut scene);
/// assert!(scene.get(id).is_some());
/// ```
#[derive(Clone)]
pub struct Starfield {
    /// Equirectangular sky map.
    pub texture: Arc<Texture>,
    /// How far out the sphere sits. Keep it inside the camera's far plane.
    pub radius: f32,
    /// Brightness multiplier.
    pub intensity: f32,
    /// Cube faces, when the sky is drawn as quads. See [`Starfield::faceted`].
    pub faces: Option<Vec<Arc<Texture>>>,
}

impl std::fmt::Debug for Starfield {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Starfield")
            .field(
                "texture",
                &format!("{}x{}", self.texture.width, self.texture.height),
            )
            .field("radius", &self.radius)
            .finish()
    }
}

/// Light reflected off one body onto the night side of another.
///
/// Returns the direction the light travels and how bright it is, ready for a
/// [`DirectionalLight`](crate::DirectionalLight). The Moon's dark limb is not
/// black: it is lit by a nearly full Earth, which from there is about sixty
/// times the area of our full Moon and four times as reflective, so the ashen
/// glow is bright enough to read surface detail by. Leaving it black is the
/// single most obvious thing wrong with a rendered crescent.
///
/// The brightness follows the *reflector's* phase, and that phase is the
/// complement of the lit body's: a full Moon happens when the Earth seen from
/// the Moon is new, so earthshine is at its faintest exactly when the Moon is
/// at its brightest, and strongest on a thin crescent. Which is when you can
/// actually see it — the old moon in the new moon's arms.
///
/// `albedo` is the reflector's bond albedo — 0.30 for Earth, against 0.12 for
/// the Moon.
///
/// ```
/// use threers::planet::earthshine;
/// use threers::Vector3;
///
/// let sun = Vector3::new(1.0, 0.0, 0.0);
/// let earth = Vector3::ZERO;
/// // Moon on the far side of the Earth from the sun: full Moon, new Earth.
/// let (_, full) = earthshine(sun, earth, Vector3::new(-60.0, 0.0, 0.0), 1.0, 0.3);
/// // Moon off to the side: half Earth.
/// let (_, half) = earthshine(sun, earth, Vector3::new(0.0, 0.0, 60.0), 1.0, 0.3);
/// assert!(half > full, "earthshine is faintest at full moon");
/// ```
pub fn earthshine(
    sun_direction: Vector3,
    reflector: Vector3,
    lit_body: Vector3,
    reflector_radius: f32,
    albedo: f32,
) -> (Vector3, f32) {
    let to_sun = sun_direction.normalize();
    let offset = lit_body - reflector;
    let distance = offset.length().max(1e-6);
    let to_body = offset * (1.0 / distance);
    // Illuminated fraction of the reflector as seen from the lit body. The
    // phase angle is between the sun and the body as seen from the reflector.
    // The phase angle at the reflector is between the sun and the observer, and
    // the illuminated fraction of a sphere at phase angle α is (1 + cos α)/2.
    // Directly away from the sun the observer sees a *new* reflector, not a
    // full one — which is why a full moon is the darkest earthshine there is.
    let phase = to_sun.dot(to_body).clamp(-1.0, 1.0);
    let illuminated = 0.5 * (1.0 + phase);
    // Inverse square on the reflector's angular size, and Lambert's factor for
    // a diffuse sphere seen at that phase.
    let solid_angle = (reflector_radius / distance).powi(2);
    let intensity = albedo.max(0.0) * illuminated * solid_angle;
    // The light travels *from* the reflector toward the body.
    (to_body, intensity)
}

/// The yaw that keeps a satellite's near side turned toward what it orbits.
///
/// A tidally locked body *does* rotate — once per orbit, exactly — and that is
/// precisely why it shows one face. Leaving it unrotated is wrong in the
/// obvious way, since the far side swings into view over a month; rotating it
/// at the orbital rate but with the wrong phase is wrong in a way that is easy
/// to miss, because the face stays put and simply happens to be the wrong one.
///
/// `SphereGeometry` puts `u = 0.5` — the middle of the near side on the CGI
/// Moon Kit's map — at local +X, and a yaw of `a` sends +X to
/// `(cos a, 0, -sin a)`. Pointing that back at the primary needs
/// `a = pi - atan2(z, x)`.
///
/// ```
/// use threers::planet::tidal_lock_yaw;
/// use threers::{Quaternion, Vector3};
///
/// // Wherever the satellite is, the same face looks back at the origin.
/// for angle in [0.0f32, 1.0, 2.5, -2.0] {
///     let pos = Vector3::new(angle.cos() * 60.0, 0.0, angle.sin() * 60.0);
///     let q = Quaternion::from_euler_xyz(0.0, tidal_lock_yaw(pos), 0.0);
///     // Local +X after the yaw.
///     let near = Vector3::new(1.0, 0.0, 0.0).apply_quaternion(q);
///     let to_primary = (pos * -1.0).normalize();
///     assert!(near.dot(to_primary) > 0.999, "near side points away at {angle}");
/// }
/// ```
pub fn tidal_lock_yaw(position: Vector3) -> f32 {
    std::f32::consts::PI - position.z.atan2(position.x)
}

impl Starfield {
    /// A sky from your own map — NASA's Deep Star Maps, say.
    pub fn new(texture: Texture) -> Self {
        Self {
            texture: Arc::new(texture),
            radius: 400.0,
            intensity: 1.0,
            faces: None,
        }
    }

    /// A generated sky of `width × width/2`, with a star count scaled to it.
    pub fn procedural(width: u32) -> Self {
        let count = (width as u64 * width as u64 / 320).clamp(500, 200_000) as u32;
        Self::new(generate_starfield(width, count))
    }

    /// Distance of the sky sphere.
    pub fn radius(mut self, radius: f32) -> Self {
        self.radius = radius.max(1.0);
        self
    }

    /// Brightness multiplier.
    pub fn intensity(mut self, intensity: f32) -> Self {
        self.intensity = intensity.max(0.0);
        self
    }

    /// Draw the sky as six quads rather than a sphere, resampling the map onto
    /// cube faces at `face_size`.
    ///
    /// An equirectangular map on a UV sphere converges every longitude onto one
    /// texel at each pole, and anything with any width at all near there gets
    /// fanned out radially when it is wrapped. A cube has no such point: its
    /// faces are planes, their texels are near enough uniform in size and
    /// direction everywhere, and there is nothing for a fan to converge on.
    ///
    /// Six quads is also less geometry than the sphere it replaces — twelve
    /// triangles against four thousand — and it removes the date-line seam
    /// along with the poles, since no face has a wrap in it.
    ///
    /// `face_size` wants to be about a quarter of the map's width: four faces
    /// span the 360 degrees the map spends its full width on. Zero picks that.
    pub fn faceted(mut self, face_size: u32) -> Self {
        let n = if face_size == 0 {
            (self.texture.width / 4).clamp(64, 4096)
        } else {
            face_size
        };
        self.faces = Some(
            equirect_to_cube_faces(&self.texture, n)
                .into_iter()
                .map(Arc::new)
                .collect(),
        );
        self
    }

    /// Add the sky to `scene`.
    pub fn add_to(&self, scene: &mut Scene) -> ObjectId {
        if let Some(faces) = &self.faces {
            if faces.len() == 6 {
                return self.add_cube(scene, faces);
            }
        }
        let mut m = BasicMaterial::new(Color::new(self.intensity, self.intensity, self.intensity));
        m.map = Some(self.texture.clone());
        // BackSide: we are inside it.
        m.side = 1;
        let mut sky = Object3D::mesh(Mesh::new(
            SphereGeometry::new(self.radius, 64, 32),
            Material::Basic(m),
        ));
        sky.name = "starfield".into();
        scene.add(sky)
    }

    /// Six quads at the cube's faces, each carrying one face texture.
    ///
    /// The corners are placed from the same basis that generated the texture,
    /// so a face texel and the direction it was sampled from line up without
    /// any rotation bookkeeping: the quad's `u` runs along the face's right and
    /// `v` along its up, exactly as the resampler wrote them.
    fn add_cube(&self, scene: &mut Scene, faces: &[Arc<Texture>]) -> ObjectId {
        let r = self.radius;
        let mut group = Object3D::group();
        group.name = "starfield".into();
        let root = scene.add(group);
        for (face, (f, right, up)) in faces.iter().zip(CUBE_FACES.iter()) {
            let corner = |x: f32, y: f32| {
                [
                    (f[0] + x * right[0] + y * up[0]) * r,
                    (f[1] + x * right[1] + y * up[1]) * r,
                    (f[2] + x * right[2] + y * up[2]) * r,
                ]
            };
            let mut geom = crate::core::BufferGeometry::new();
            let mut positions = Vec::with_capacity(12);
            for (x, y) in [(-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0), (1.0, 1.0)] {
                positions.extend_from_slice(&corner(x, y));
            }
            geom.set_attribute("position", crate::core::BufferAttribute::new(positions, 3));
            geom.set_attribute(
                "normal",
                crate::core::BufferAttribute::new([-f[0], -f[1], -f[2]].repeat(4).to_vec(), 3),
            );
            geom.set_attribute(
                "uv",
                crate::core::BufferAttribute::new(vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0], 2),
            );
            geom.set_index(vec![0, 1, 2, 2, 1, 3]);

            let mut m =
                BasicMaterial::new(Color::new(self.intensity, self.intensity, self.intensity));
            m.map = Some(face.clone());
            // Double-sided: which way a face's winding comes out depends on the
            // handedness of its basis, and being inside the cube means the
            // answer has to be "visible" either way.
            m.side = 2;
            let mut quad = Object3D::mesh(Mesh::new(geom, Material::Basic(m)));
            quad.name = "starfield face".into();
            scene.add_to(root, quad);
        }
        root
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn earth_defaults_are_earth_like() {
        let earth = Planet::earth();
        assert_eq!(earth.radius, 1.0);
        assert!((earth.tilt.to_degrees() - 23.44).abs() < 1e-3);
        let air = earth.atmosphere.expect("earth has air");
        assert!(air.height > 0.0 && air.height < 0.1, "a thin shell");
        // Rock and water are dielectric.
        assert_eq!(earth.metalness, 0.0);
    }

    #[test]
    fn displacement_centres_only_when_asked() {
        // Two height-map conventions, two builders. Earth's map puts sea level
        // at zero, so biasing it would sink the oceans below the sphere; the
        // Moon's is a stretched range with no natural zero, so leaving the bias
        // at zero would inflate the whole body by the map's mean.
        let up = Planet::earth().displacement(0.05);
        assert_eq!(up.displacement, 0.05);
        assert_eq!(
            up.displacement_bias, 0.0,
            "a sea-level map must not be shifted"
        );

        let mid = Planet::moon().displacement_centered(0.02);
        assert_eq!(mid.displacement, 0.02);
        assert!(
            (mid.displacement_bias + 0.01).abs() < 1e-6,
            "a centred map should sit half below the surface: {}",
            mid.displacement_bias
        );
    }

    #[test]
    fn the_moon_is_airless_and_dark_at_night() {
        let moon = Planet::moon();
        assert!(moon.atmosphere.is_none());
        assert_eq!(moon.night_intensity, 0.0, "no cities on the Moon");
        assert!(moon.radius < Planet::earth().radius);
    }

    #[test]
    fn adding_a_planet_creates_the_shells_it_has_maps_for() {
        let mut scene = Scene::new();
        let bare = Planet::earth().atmosphere(None).add_to(&mut scene);
        assert!(bare.clouds.is_none(), "no cloud map, no cloud shell");
        assert!(bare.atmosphere.is_none());

        let full = Planet::earth()
            .maps(generate_earth_maps(64))
            .add_to(&mut scene);
        assert!(full.clouds.is_some());
        assert!(full.atmosphere.is_some());
        assert!(scene.get(full.surface).is_some());
        assert!(scene.get(full.clouds.unwrap()).is_some());
        assert!(scene.get(full.atmosphere.unwrap()).is_some());
    }

    #[test]
    fn the_shells_are_stacked_in_the_right_order() {
        let earth = Planet::earth();
        let air = earth.atmosphere.unwrap();
        let cloud_r = earth.radius * (1.0 + earth.cloud_height);
        let air_r = earth.radius * (1.0 + air.height);
        assert!(
            earth.radius < cloud_r && cloud_r < air_r,
            "surface {} < clouds {cloud_r} < air {air_r}",
            earth.radius
        );
    }

    #[test]
    fn the_cloud_shell_stays_on_the_alpha_pipeline() {
        // Opacity 1.0 would route the shell to the opaque pipeline, where the
        // map's alpha is never read and the planet vanishes behind white.
        let mut scene = Scene::new();
        let handles = Planet::earth()
            .maps(generate_earth_maps(64))
            .add_to(&mut scene);
        let clouds = scene.get(handles.clouds.unwrap()).unwrap();
        if let crate::ObjectKind::Mesh(mesh) = &clouds.kind {
            assert!(mesh.material.transparent(), "clouds must be transparent");
            assert!(mesh.material.opacity() < 1.0);
        } else {
            panic!("expected a mesh");
        }
    }

    #[test]
    fn the_atmosphere_material_spans_the_two_radii() {
        let air = Atmosphere::earth().height(0.05);
        let m = air.material(2.0);
        assert_eq!(m.planet_radius, 2.0);
        assert!((m.atmosphere_radius - 2.1).abs() < 1e-5);
        assert!(m.thickness() > 0.0);
    }

    #[test]
    fn a_starfield_is_unlit_and_inside_out() {
        let mut scene = Scene::new();
        let sky = Starfield::procedural(128).radius(250.0);
        let id = sky.add_to(&mut scene);
        let obj = scene.get(id).unwrap();
        if let crate::ObjectKind::Mesh(mesh) = &obj.kind {
            // Unlit, so the sky does not respond to the sun.
            assert!(matches!(&*mesh.material, Material::Basic(_)));
            // BackSide, because the camera is inside the sphere.
            assert_eq!(mesh.material.side(), 1);
        } else {
            panic!("expected a mesh");
        }
    }

    #[test]
    fn maps_are_optional_and_independent() {
        let maps = PlanetMaps::new().albedo(generate_starfield(32, 10));
        assert!(maps.albedo.is_some());
        assert!(maps.clouds.is_none());
        assert!(!maps.is_empty());
        assert!(PlanetMaps::new().is_empty());

        // A planet with no maps still builds.
        let mut scene = Scene::new();
        let handles = Planet::new(1.0).add_to(&mut scene);
        assert!(scene.get(handles.surface).is_some());
    }

    /// A night map must not glow in daylight.
    ///
    /// `emissive` is added to the shaded colour unconditionally, so an emissive
    /// city-lights map lights every city on the planet at noon. The map masks
    /// WHERE the cities are; only `emissive_night_side` masks WHETHER it is
    /// night there. This asserts the planet asks for that, and that materials
    /// which did not ask are unchanged -- the gate is opt-in, so a lamp still
    /// behaves like a lamp.
    #[test]
    fn night_map_is_confined_to_the_night_side() {
        use crate::textures::{Texture, TextureFormat};
        use std::sync::Arc;

        let night = Arc::new(Texture::new(
            1,
            1,
            TextureFormat::Rgba8UnormSrgb,
            vec![255, 255, 255, 255],
        ));
        let maps = PlanetMaps {
            night: Some(night),
            ..Default::default()
        };
        let m = Planet::new(1.0).maps(maps).surface_material();
        assert!(m.emissive_map.is_some(), "the night map should be bound");
        assert_eq!(
            m.emissive_night_side, 1.0,
            "a planet's night map must be gated to the unlit hemisphere"
        );

        // Opt-in: anything that did not ask keeps three.js behaviour.
        assert_eq!(
            StandardMaterial::new(Color::WHITE).emissive_night_side,
            0.0,
            "the gate must be off unless a material asks for it"
        );
        // And a planet with no night map has nothing to gate.
        assert_eq!(Planet::new(1.0).surface_material().emissive_night_side, 0.0);
    }
}
