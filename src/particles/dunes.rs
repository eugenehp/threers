//! Dune fields: a desert of sand dunes, its sky, and the sand the wind
//! blows off its crests.
//!
//! The shape is grown in two steps. First a procedural field, in the layers
//! dunes show from the air: a network of sharp, curving crests (the cell
//! edges of a Voronoi pattern whose coordinates are pushed about by noise,
//! so the crests bend), broken up by a broad flowing noise and carrying
//! smaller dunes on the big ones. Then the wind works it, by the slab model
//! of [`SandBed::blow`](super::sand::SandBed::blow) with avalanches between:
//! sand is lifted from the windward slopes and dropped in the lee, so every
//! dune ends up asymmetric — a long gentle stoss slope up to a sharp brink
//! and a slip face at the angle of repose, 34°, beyond. The field is
//! periodic, so the wind can carry sand off one edge and onto the other.
//!
//! [`DesertTime`] is a time of day: the sun (or moon), the sky's colours,
//! the haze and the sand's. [`sky_dome`] draws its sky; the sand's surface
//! shader brightens its haze toward the sun by the same law, so the dunes
//! fade into the sky's horizon without a seam. [`crests`] finds the brinks,
//! for [`presets::spindrift`](super::presets::spindrift) to blow sand off.

use crate::core::{Mesh, Object3D};
use crate::geometries::SphereGeometry;
use crate::materials::{Material, RenderState};
use crate::math::{Color, Vector3};
use crate::ShaderMaterial;

use super::heightfield::Heightfield;
use super::sand::{Downwind, SandBed};

/// How a dune field is grown.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DuneShape {
    /// Metres from crest to crest, about.
    pub wavelength: f32,
    /// Metres from trough to crest, about.
    pub height: f32,
    /// How much the crests curve, 0 (straight-edged cells) to 1.
    pub warp: f32,
    /// The broad flowing breakup, 0 to 1.
    pub flow: f32,
    /// Smaller dunes on the big ones, 0 to 1.
    pub secondary: f32,
    /// How far the crests run across the wind, as a multiple of the spacing
    /// along it: 1 for a network of star-like dunes, 2–3 for the long
    /// sinuous transverse ridges a steady wind builds.
    pub stretch: f32,
    /// The prevailing wind, for the shaping.
    pub wind: Downwind,
    /// How long the wind works it: slab moves per cell. 0: the procedural
    /// field as it is, symmetric.
    pub shaping: f32,
    pub seed: u64,
}

impl Default for DuneShape {
    fn default() -> Self {
        Self {
            wavelength: 110.0,
            height: 14.0,
            warp: 0.4,
            flow: 0.35,
            secondary: 0.3,
            stretch: 2.2,
            wind: Downwind::PlusX,
            shaping: 24.0,
            seed: 7,
        }
    }
}

fn hash(x: i64, y: i64, seed: u64) -> f32 {
    let mut h = (x as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ (y as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F)
        ^ seed.wrapping_mul(0x1656_67B1_9E37_79F9);
    h ^= h >> 31;
    h = h.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    h ^= h >> 29;
    (h >> 40) as f32 / (1u64 << 24) as f32
}

/// Value noise with period `period` lattice cells along each axis.
fn noise(x: f32, y: f32, period: [i64; 2], seed: u64) -> f32 {
    let (ix, iy) = (x.floor() as i64, y.floor() as i64);
    let (fx, fy) = (x - ix as f32, y - iy as f32);
    let (ux, uy) = (fx * fx * (3.0 - 2.0 * fx), fy * fy * (3.0 - 2.0 * fy));
    let h = |i: i64, j: i64| hash(i.rem_euclid(period[0]), j.rem_euclid(period[1]), seed);
    let a = h(ix, iy) + (h(ix + 1, iy) - h(ix, iy)) * ux;
    let b = h(ix, iy + 1) + (h(ix + 1, iy + 1) - h(ix, iy + 1)) * ux;
    a + (b - a) * uy
}

/// Three octaves of periodic noise, 0..1.
fn fbm(x: f32, y: f32, period: [i64; 2], seed: u64) -> f32 {
    let mut s = 0.0;
    let mut a = 0.5;
    let mut f = 1.0;
    for o in 0..3 {
        s += a * noise(
            x * f,
            y * f,
            [period[0] * f as i64, period[1] * f as i64],
            seed + o,
        );
        a *= 0.5;
        f *= 2.0;
    }
    s / 0.875
}

/// The distance to the nearest of one jittered point per lattice cell
/// (Worley's F1), periodic in `period` cells along each axis.
fn voronoi(x: f32, y: f32, period: [i64; 2], seed: u64) -> f32 {
    let (ix, iy) = (x.floor() as i64, y.floor() as i64);
    let mut best = f32::INFINITY;
    for j in -1..=1 {
        for i in -1..=1 {
            let (cx, cy) = (ix + i, iy + j);
            let (px, py) = (cx.rem_euclid(period[0]), cy.rem_euclid(period[1]));
            let fx = cx as f32 + 0.1 + 0.8 * hash(px, py, seed);
            let fy = cy as f32 + 0.1 + 0.8 * hash(px, py, seed ^ 0xabc);
            best = best.min((fx - x).hypot(fy - y));
        }
    }
    best
}

impl SandBed {
    /// Fill the bed with a dune field (replacing what lies on it) and let
    /// the wind work it: see the [module](self) notes. The field repeats
    /// across the bed's width and depth.
    pub fn dunes(&mut self, shape: DuneShape) {
        let f = &mut self.field;
        // Lattice cells a wavelength along the wind and `stretch` of them
        // across it, a whole number of each across the bed so it repeats.
        let across_x = matches!(shape.wind, Downwind::PlusZ | Downwind::MinusZ);
        let length = |cells: usize, along: bool| {
            let extent = cells as f32 * f.cell;
            let l = shape.wavelength.max(1.0) * if along { 1.0 } else { shape.stretch.max(0.2) };
            ((extent / l).round().max(1.0) as i64, extent)
        };
        let (px, ex) = length(f.nx, !across_x);
        let (pz, ez) = length(f.nz, across_x);
        let period = [px, pz];
        let seed = shape.seed;
        for j in 0..f.nz {
            for i in 0..f.nx {
                let x = (i as f32 + 0.5) * f.cell * px as f32 / ex;
                let y = (j as f32 + 0.5) * f.cell * pz as f32 / ez;
                // Crests bent by noise.
                let wx = (fbm(x, y, period, seed + 11) - 0.5) * 2.0 * shape.warp;
                let wy = (fbm(x + 17.3, y + 5.1, period, seed + 13) - 0.5) * 2.0 * shape.warp;
                let (qx, qy) = (x + wx, y + wy);
                // The crest network: the cells' edges are their high ground.
                let f1 = (voronoi(qx, qy, period, seed) / 0.7).min(1.0);
                let mut h = f1.powf(1.3);
                // Broad flowing breakup.
                h += (fbm(x * 2.0, y * 2.0, [period[0] * 2, period[1] * 2], seed + 21) - 0.5)
                    * shape.flow;
                // Smaller dunes on the big ones.
                let f2 = (voronoi(
                    qx * 3.0,
                    qy * 3.0,
                    [period[0] * 3, period[1] * 3],
                    seed + 31,
                ) / 0.7)
                    .min(1.0);
                h += f2.powf(1.3) * shape.secondary * 0.35;
                let k = i + f.nx * j;
                f.depth[k] = (h.max(0.0) + 0.05) * shape.height;
            }
        }
        f.touch_all();
        f.version += 1;
        if shape.shaping > 0.0 {
            // The wind, in rounds, with avalanches between: a slab is a
            // hundredth of the dunes' height, so the shaping scales with them.
            let cells = f.nx * f.nz;
            let saved = self.saltation;
            self.saltation.slab = shape.height / 100.0;
            self.saltation.hop = 2;
            let total = (shape.shaping * cells as f32) as usize;
            let rounds = 24;
            for _ in 0..rounds {
                self.carry(shape.wind, total / rounds);
                self.settle_within(4 * cells);
            }
            self.saltation = saved;
        }
        // Settled to rest here, once: left slumping, a field this size would
        // go on avalanching a little every frame it is drawn.
        for _ in 0..200 {
            if self.field.relax(self.angle_of_repose, 64) <= 0.0 {
                break;
            }
        }
    }
}

/// The brinks of `field`'s dunes, downwind toward `downwind`: cells at the
/// top of a slip face — the ground falling away beyond them more steeply
/// than `steep` (radians), rising or level before. Their surface points,
/// lifted `lift` metres; at most `limit` of them, evenly picked.
pub fn crests(
    field: &Heightfield,
    downwind: Downwind,
    steep: f32,
    lift: f32,
    limit: usize,
) -> Vec<Vector3> {
    let (di, dj) = downwind.step();
    let tan = steep.tan() * field.cell;
    let mut out = Vec::new();
    for j in 1..field.nz.saturating_sub(1) {
        for i in 1..field.nx.saturating_sub(1) {
            let at =
                |a: i64, b: i64| field.surface((i as i64 + a) as usize, (j as i64 + b) as usize);
            let h = at(0, 0);
            if h - at(di, dj) > tan && h >= at(-di, -dj) - 0.02 * tan {
                let [x, z] = field.cell_center(i, j);
                out.push(Vector3::new(x, h + lift, z));
            }
        }
    }
    if out.len() > limit && limit > 0 {
        let step = out.len() as f32 / limit as f32;
        out = (0..limit)
            .map(|k| out[(k as f32 * step) as usize])
            .collect();
    }
    out
}

/// A time of day over the desert: its light, sky, haze and sand.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DesertTime {
    /// Toward the sun (or the moon, at night).
    pub sun: Vector3,
    /// Its light (linear) and intensity, and how big its disc is (angular
    /// radius, radians).
    pub sun_color: Color,
    pub sun_intensity: f32,
    pub sun_radius: f32,
    /// The sky straight up and at the horizon (linear radiance); the haze
    /// over the sand is the horizon's.
    pub zenith: Color,
    pub horizon: Color,
    /// The haze's density (exp², per metre).
    pub haze: f32,
    /// How much brighter the haze is toward the sun (forward scattering).
    pub glow: f32,
    /// The skylight: from above, from the sand, and how strong.
    pub sky_light: Color,
    pub ground_light: Color,
    pub sky_intensity: f32,
    /// The sand's colour and its shadows' (linear).
    pub sand: Color,
    pub shadow: Color,
    /// Stars, 0 to 1.
    pub stars: f32,
}

fn dir(azimuth_deg: f32, elevation_deg: f32) -> Vector3 {
    let (a, e) = (azimuth_deg.to_radians(), elevation_deg.to_radians());
    Vector3::new(a.cos() * e.cos(), e.sin(), a.sin() * e.cos())
}

impl DesertTime {
    /// Mid-morning to noon: a high white sun, a deep blue sky, pale haze.
    pub fn day() -> Self {
        Self {
            sun: dir(-30.0, 52.0),
            sun_color: Color::new(1.0, 0.95, 0.88),
            sun_intensity: 3.4,
            sun_radius: 0.0047,
            zenith: Color::new(0.10, 0.24, 0.62),
            horizon: Color::new(0.62, 0.70, 0.80),
            haze: 0.0016,
            glow: 0.35,
            sky_light: Color::new(0.45, 0.6, 0.9),
            ground_light: Color::new(0.55, 0.38, 0.22),
            sky_intensity: 0.9,
            sand: Color::new(0.72, 0.42, 0.18),
            shadow: Color::new(0.30, 0.20, 0.16),
            stars: 0.0,
        }
    }

    /// The sun just up behind a ridge: long shadows, a cool sky, warm light.
    pub fn dawn() -> Self {
        Self {
            sun: dir(160.0, 9.0),
            sun_color: Color::new(1.0, 0.72, 0.5),
            sun_intensity: 2.6,
            sun_radius: 0.0055,
            zenith: Color::new(0.10, 0.16, 0.36),
            horizon: Color::new(0.80, 0.62, 0.58),
            haze: 0.0026,
            glow: 0.7,
            sky_light: Color::new(0.45, 0.48, 0.7),
            ground_light: Color::new(0.5, 0.32, 0.22),
            sky_intensity: 0.55,
            sand: Color::new(0.70, 0.40, 0.20),
            shadow: Color::new(0.20, 0.18, 0.30),
            stars: 0.05,
        }
    }

    /// Low in the west: the sand orange, its shadows blue, the haze aglow.
    pub fn sunset() -> Self {
        Self {
            sun: dir(15.0, 5.0),
            sun_color: Color::new(1.0, 0.52, 0.22),
            sun_intensity: 3.0,
            sun_radius: 0.006,
            zenith: Color::new(0.06, 0.09, 0.24),
            horizon: Color::new(0.95, 0.48, 0.24),
            haze: 0.0022,
            glow: 0.9,
            sky_light: Color::new(0.35, 0.32, 0.55),
            ground_light: Color::new(0.45, 0.22, 0.10),
            sky_intensity: 0.5,
            sand: Color::new(0.74, 0.40, 0.16),
            shadow: Color::new(0.16, 0.12, 0.26),
            stars: 0.0,
        }
    }

    /// Moonlight: blue-grey dunes under stars.
    pub fn night() -> Self {
        Self {
            sun: dir(-60.0, 38.0),
            sun_color: Color::new(0.62, 0.72, 1.0),
            sun_intensity: 0.32,
            sun_radius: 0.0045,
            zenith: Color::new(0.004, 0.007, 0.02),
            horizon: Color::new(0.025, 0.035, 0.06),
            haze: 0.0013,
            glow: 0.5,
            sky_light: Color::new(0.08, 0.11, 0.2),
            ground_light: Color::new(0.03, 0.03, 0.04),
            sky_intensity: 0.6,
            sand: Color::new(0.55, 0.40, 0.26),
            shadow: Color::new(0.03, 0.04, 0.08),
            stars: 1.0,
        }
    }

    /// The four, in order through the day: dawn, day, sunset, night.
    pub fn all() -> [(&'static str, Self); 4] {
        [
            ("dawn", Self::dawn()),
            ("day", Self::day()),
            ("sunset", Self::sunset()),
            ("night", Self::night()),
        ]
    }

    /// The fog for this time: the horizon's colour, exp².
    pub fn fog(&self) -> crate::scene::FogParams {
        crate::scene::FogParams {
            color: self.horizon,
            density: self.haze,
            mode: 2,
            ..crate::scene::FogParams::default()
        }
    }

    /// The sky dome's material.
    pub fn sky_material(&self) -> Material {
        let v = |c: Color| format!("vec3<f32>({:?}, {:?}, {:?})", c.r, c.g, c.b);
        let s = self.sun.normalize();
        let consts = format!(
            "const SKY_SUN : vec3<f32> = vec3<f32>({:?}, {:?}, {:?});\n\
             const SKY_SUN_COLOR : vec3<f32> = {};\n\
             const SKY_ZENITH : vec3<f32> = {};\n\
             const SKY_HORIZON : vec3<f32> = {};\n\
             const SKY_GLOW : f32 = {:?};\n\
             const SKY_DISC : f32 = {:?};\n\
             const SKY_STARS : f32 = {:?};\n",
            s.x,
            s.y,
            s.z,
            v(Color::new(
                self.sun_color.r * self.sun_intensity,
                self.sun_color.g * self.sun_intensity,
                self.sun_color.b * self.sun_intensity
            )),
            v(self.zenith),
            v(self.horizon),
            self.glow,
            self.sun_radius,
            self.stars,
        );
        Material::Shader(ShaderMaterial {
            fragment: consts + SKY_WGSL,
            side: 2,
            render_state: RenderState {
                depth_write: Some(false),
                ..RenderState::DEFAULT
            },
            fallback_color: self.horizon,
            ..Default::default()
        })
    }
}

/// A sky dome `radius` metres round the origin, for `time`: keep it centred
/// on the camera (its position), and the camera's far plane beyond it.
pub fn sky_dome(time: &DesertTime, radius: f32) -> Object3D {
    let mut o = Object3D::mesh(Mesh::new(
        SphereGeometry::new(radius, 48, 24),
        time.sky_material(),
    ));
    o.cast_shadow = false;
    o.receive_shadow = false;
    o.render_order = -1000;
    o.name = "sky".into();
    o
}

/// The sky: the horizon's haze shading up to the zenith; that haze brighter
/// toward the sun (`(1 + glow · max(cos θ, 0)⁸)`, the law the sand's fog
/// follows, so the two meet at the horizon); the sun's aureole and disc; and
/// stars. Not fogged: it is the fog's far end.
pub(crate) const SKY_WGSL: &str = r#"
fn sky_hash(p : vec2<f32>) -> f32 {
    var q = fract(p * vec2<f32>(127.31, 311.71));
    q = q + vec2<f32>(dot(q, q + 19.19));
    return fract(q.x * q.y);
}

@fragment
fn fs_main(in : VsOut) -> @location(0) vec4<f32> {
    let d = normalize(in.world_pos - frame.camera_position.xyz);
    let up = max(d.y, 0.0);
    var col = mix(SKY_HORIZON, SKY_ZENITH, pow(up, 0.45));
    let cs = dot(d, SKY_SUN);
    // The haze toward the sun, strongest along the horizon.
    col = col * (1.0 + SKY_GLOW * pow(max(cs, 0.0), 8.0) * (1.0 - up));
    // The aureole: light scattered a little way off the sun by the haze's
    // larger particles.
    col = col + SKY_SUN_COLOR * (0.01 * pow(max(cs, 0.0), 400.0) + 0.04 * pow(max(cs, 0.0), 4000.0));
    // The disc, gone below the horizon.
    let disc = smoothstep(cos(SKY_DISC * 1.2), cos(SKY_DISC), cs);
    col = col + SKY_SUN_COLOR * disc * 12.0 * smoothstep(-0.01, 0.01, d.y);
    if (SKY_STARS > 0.0 && d.y > 0.0) {
        // A star in one cell in a few hundred, by direction.
        let a = vec2<f32>(atan2(d.z, d.x) * 180.0, asin(d.y) * 180.0);
        let c = floor(a);
        let h = sky_hash(c);
        let f = fract(a) - vec2<f32>(sky_hash(c + 3.1), sky_hash(c + 7.7)) * 0.6 - 0.2;
        let star = step(0.985, h) * smoothstep(0.12, 0.0, length(f)) * (h - 0.985) * 60.0;
        col = col + vec3<f32>(0.8, 0.85, 1.0) * star * SKY_STARS * smoothstep(0.0, 0.15, d.y);
    }
    return vec4<f32>(framebuffer_encode(apply_tone_mapping(col)), 1.0);
}
"#;
