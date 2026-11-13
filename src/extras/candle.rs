//! Candles: a wax body, a wick, a flame and the light it gives, added to any
//! scene in one call and animated with one call a frame. Two flames:
//!
//! - [`FlameStyle::Simple`] — a teardrop mesh with a glow shader (hot
//!   white-yellow where it faces you, orange at the rim, a touch of blue at
//!   the base), flickered by moving the mesh. A few dozen shader
//!   instructions a pixel: for many candles, phones and headsets.
//! - [`FlameStyle::Realistic`] — a volumetric flame, ray-marched through a
//!   box: a teardrop of hot gas pushed about by rising 3D noise, coloured by
//!   its temperature as a black body (yellow-white in the sooty core, deep
//!   orange at the edge and the tip), with the dark non-luminous zone above
//!   the wick and the thin blue sheet of burning vapour around the base.
//!   The wax glows from inside near the flame, as light scattered through
//!   it does; the wick's tip is an ember; the light casts shadows.
//!
//! The flame's flicker and the light's are the same signal, so the room
//! breathes with the flame.
//!
//! ```
//! use threers::extras::candle::{Candle, FlameStyle};
//! use threers::{Scene, Vector3};
//!
//! let mut scene = Scene::new();
//! let candle = Candle::new(FlameStyle::Realistic).add_to(&mut scene, Vector3::new(0.0, 0.75, 0.0));
//! // Each frame:
//! candle.update(&mut scene, 1.5);
//! ```

use crate::core::{Mesh, Object3D, ObjectId, ObjectKind};
use crate::geometries::{BoxGeometry, CylinderGeometry, LatheGeometry, SphereGeometry};
use crate::lights::{Light, PointLight};
use crate::materials::{Blending, Material, ShaderMaterial, StandardMaterial};
use crate::math::{Color, Euler, Vector2, Vector3};
use crate::scene::Scene;

/// How a candle's flame is drawn. See the module docs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FlameStyle {
    /// A glowing teardrop mesh: cheap.
    #[default]
    Simple,
    /// A ray-marched volumetric flame, a glowing wax top and a shadowing light.
    Realistic,
}

/// A candle to add to a scene: its size in metres, its wax, its light.
#[derive(Debug, Clone)]
pub struct Candle {
    pub style: FlameStyle,
    /// The wax body's height and radius.
    pub height: f32,
    pub radius: f32,
    pub wax: Color,
    /// The flame's height (its width follows).
    pub flame_height: f32,
    /// The light's intensity and colour.
    pub light_intensity: f32,
    pub light_color: Color,
    /// Steps through the volume a pixel, for [`FlameStyle::Realistic`]:
    /// more is smoother and slower.
    pub steps: u32,
    /// Sets one candle's flicker apart from another's.
    pub seed: f32,
    /// Whether the light casts shadows (by default, the realistic style's
    /// does). Off where shadow casters are costly — fur, hair, crowds: a
    /// point light draws every caster into six shadow views.
    pub cast_shadow: bool,
}

impl Candle {
    /// A pillar candle 12 cm tall, ivory, with a 4.5 cm flame.
    pub fn new(style: FlameStyle) -> Self {
        Candle {
            style,
            height: 0.12,
            radius: 0.024,
            wax: Color::from_hex(0xf1e6cf),
            flame_height: 0.045,
            // About a candela, for a room exposed for candlelight.
            light_intensity: 0.06,
            light_color: Color::from_hex(0xffa451),
            steps: 64,
            seed: 0.0,
            cast_shadow: style == FlameStyle::Realistic,
        }
    }

    pub fn with_seed(mut self, seed: f32) -> Self {
        self.seed = seed;
        self
    }

    pub fn with_size(mut self, height: f32, radius: f32) -> Self {
        (self.height, self.radius) = (height, radius);
        self
    }

    pub fn with_wax(mut self, wax: Color) -> Self {
        self.wax = wax;
        self
    }

    /// Adds the candle to `scene`, standing on `position`. The handle
    /// animates it.
    pub fn add_to(&self, scene: &mut Scene, position: Vector3) -> CandleHandle {
        let realistic = self.style == FlameStyle::Realistic;
        let mut root = Object3D::group();
        root.position = position;
        let root = scene.add(root);
        let (h, r) = (self.height, self.radius);

        // The body: a cylinder with a soft shoulder and a shallow melt pool.
        let body = LatheGeometry::new(&wax_profile(h, r), 48, 0.0, std::f32::consts::TAU);
        let mut wax = StandardMaterial::new(self.wax).with_roughness(if realistic { 0.45 } else { 0.6 });
        wax.emissive = self.wax;
        wax.emissive_intensity = 0.02;
        let mut body = Object3D::mesh(Mesh::new(body, wax.into()));
        body.cast_shadow = true;
        body.receive_shadow = true;
        scene.add_to(root, body);

        // The wick, leaning a little, and (realistic) its glowing tip.
        let wick_h = self.flame_height * 0.24;
        let mut wick = Object3D::mesh(Mesh::new(
            CylinderGeometry::new(r * 0.045, r * 0.05, wick_h, 8, 1, false, 0.0, std::f32::consts::TAU),
            StandardMaterial::new(Color::from_hex(0x1c1610)).with_roughness(0.9).into(),
        ));
        wick.position = Vector3::new(0.0, h - 0.004 + wick_h / 2.0, 0.0);
        wick.quaternion = Euler::new(0.0, 0.0, 0.12).to_quaternion();
        scene.add_to(root, wick);
        let top = Vector3::new(-wick_h * 0.12, h - 0.004 + wick_h, 0.0);
        if realistic {
            let mut ember = StandardMaterial::new(Color::from_hex(0x2a1a10));
            ember.emissive = Color::from_hex(0xff5a1a);
            ember.emissive_intensity = 2.5;
            let mut tip = Object3D::mesh(Mesh::new(SphereGeometry::new(r * 0.06, 10, 6), ember.into()));
            tip.position = top;
            scene.add_to(root, tip);
        }

        // The flame, its base just above the wick's top.
        let fh = self.flame_height;
        let fw = fh * 0.25;
        let base = h - 0.004 + wick_h * 0.55;
        let mut data = vec![[0.0f32; 4]; 16];
        data[0] = [0.0, 1.0, self.seed, 0.0];
        data[1] = [fw, fh, self.steps as f32, 0.0];
        let (geometry, fragment, at) = match self.style {
            FlameStyle::Simple => (LatheGeometry::new(&flame_profile(fh, fw * 0.55), 24, 0.0, std::f32::consts::TAU), SIMPLE_FLAME.to_owned(), base),
            // A box around the volume, its middle at half the flame's height.
            FlameStyle::Realistic => (BoxGeometry::new(fw * 2.2, fh * 1.25, fw * 2.2), format!("{}{}", threers_math::flicker::FLAME_WGSL, REALISTIC_FLAME), base + fh * 1.25 / 2.0),
        };
        let mut material = ShaderMaterial::new(fragment).with_data(data).with_transparent(true).with_depth_write(false);
        material.render_state.blending = Blending::Additive;
        let mut flame = Object3D::mesh(Mesh::new(geometry, Material::Shader(material.clone())));
        flame.position = Vector3::new(0.0, at, 0.0);
        let flame = scene.add_to(root, flame);

        // The wax lit from inside near the top (realistic): a shell over its
        // upper part, adding a warm glow that fades down the side.
        let glow = realistic.then(|| {
            let mut data = vec![[0.0f32; 4]; 16];
            data[0] = [0.0, 1.0, self.seed, 0.0];
            data[1] = [h, r, 0.0, 0.0];
            let mut m = ShaderMaterial::new(WAX_GLOW).with_data(data).with_transparent(true).with_depth_write(false);
            m.render_state.blending = Blending::Additive;
            let shell = LatheGeometry::new(&wax_profile(h, r * 1.004).iter().map(|p| Vector2::new(p.x, p.y + 0.0003)).collect::<Vec<_>>(), 48, 0.0, std::f32::consts::TAU);
            scene.add_to(root, Object3D::mesh(Mesh::new(shell, Material::Shader(m.clone()))));
            m
        });

        // The light, in the flame's brightest part.
        let mut light = PointLight::new(self.light_color, self.light_intensity);
        light.decay = 2.0;
        light.cast_shadow = self.cast_shadow;
        light.shadow.normal_bias = 0.01;
        let mut light = Object3D::light(Light::Point(light));
        let light_at = Vector3::new(0.0, base + fh * 0.4, 0.0);
        light.position = light_at;
        let light = scene.add_to(root, light);

        CandleHandle { root, flame, light, material, glow, style: self.style, seed: self.seed, intensity: self.light_intensity, light_at, flame_at: at }
    }
}

/// A candle in a scene: [`update`](Self::update) it each frame.
#[derive(Debug, Clone)]
pub struct CandleHandle {
    /// The candle's group (move it to move the candle).
    pub root: ObjectId,
    pub flame: ObjectId,
    pub light: ObjectId,
    material: ShaderMaterial,
    glow: Option<ShaderMaterial>,
    style: FlameStyle,
    seed: f32,
    intensity: f32,
    light_at: Vector3,
    flame_at: f32,
}

impl CandleHandle {
    /// The flame and its light at `time` seconds.
    pub fn update(&self, scene: &mut Scene, time: f32) {
        let f = flicker(time, self.seed);
        let (sx, sz) = sway(time, self.seed);
        self.material.set_uniform(0, [time, f, self.seed, 0.0]);
        self.material.set_uniform(2, [sx, sz, 0.0, 0.0]);
        if let Some(g) = &self.glow {
            g.set_uniform(0, [time, f, self.seed, 0.0]);
        }
        if self.style == FlameStyle::Simple
            && let Some(o) = scene.get_mut(self.flame)
        {
            // The mesh itself flickers: taller and narrower as it flares,
            // and leaning with the draught.
            o.scale = Vector3::new(1.0 / f.sqrt(), f, 1.0 / f.sqrt());
            o.quaternion = Euler::new(sz * 0.35, 0.0, -sx * 0.35).to_quaternion();
            o.position.y = self.flame_at;
        }
        if let Some(o) = scene.get_mut(self.light) {
            o.position = self.light_at + Vector3::new(sx * 0.004, (f - 1.0) * 0.006, sz * 0.004);
            if let ObjectKind::Light(Light::Point(p)) = &mut o.kind {
                p.intensity = self.intensity * (0.55 + 0.45 * f * f);
            }
        }
    }
}

/// The flame's brightness and stretch at `time` (see
/// [`threers_math::flicker::flame`], shared with the UI's candles).
pub fn flicker(time: f32, seed: f32) -> f32 {
    threers_math::flicker::flame(time, seed)
}

fn sway(time: f32, seed: f32) -> (f32, f32) {
    threers_math::flicker::sway(time, seed)
}

/// The wax body's outline (radius, height), base to top: straight sides, a
/// rounded shoulder, a pool dipping to the wick.
fn wax_profile(h: f32, r: f32) -> Vec<Vector2> {
    let mut p = vec![Vector2::new(0.0, 0.0), Vector2::new(r, 0.0), Vector2::new(r, h - r * 0.25)];
    for i in 1..=6 {
        let a = i as f32 / 6.0 * std::f32::consts::FRAC_PI_2;
        p.push(Vector2::new(r - r * 0.25 * (1.0 - a.cos()), h - r * 0.25 + r * 0.25 * a.sin()));
    }
    for i in 1..=8 {
        let s = i as f32 / 8.0;
        let x = (r * 0.75) * (1.0 - s);
        p.push(Vector2::new(x, h - r * 0.16 * (1.0 - (1.0 - s) * (1.0 - s))));
    }
    p
}

/// The simple flame's teardrop (radius, height), from its base to its tip:
/// round at the bottom, drawn out to a point.
fn flame_profile(h: f32, w: f32) -> Vec<Vector2> {
    (0..=24)
        .map(|i| {
            let y = i as f32 / 24.0;
            let r = w * 2.1 * y.powf(0.45) * (1.0 - y).powf(0.75);
            Vector2::new(r, y * h)
        })
        .collect()
}

/// The simple flame: a glow by how directly each point faces the eye.
/// data[0]: time, flicker, seed; data[1]: width, height.
const SIMPLE_FLAME: &str = r#"
@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let flicker = u_data.data[0].y;
    let y = clamp(in.local_pos.y / u_data.data[1].y, 0.0, 1.0);
    let n = normalize(in.world_normal);
    let v = normalize(frame.camera_position.xyz - in.world_pos);
    let facing = abs(dot(n, v));
    let core = pow(facing, 2.0);
    // White-yellow in the middle, orange at the rim and tip.
    var c = mix(vec3<f32>(1.0, 0.36, 0.06), vec3<f32>(1.0, 0.86, 0.58), core * (1.0 - 0.6 * y));
    // The dim zone above the wick, and the blue at the base's edge.
    c = c * mix(0.35, 1.0, smoothstep(0.04, 0.28, y));
    c = c + vec3<f32>(0.18, 0.32, 1.0) * (1.0 - smoothstep(0.0, 0.3, y)) * (1.0 - core) * 0.8;
    let a = smoothstep(0.0, 0.55, facing) * (1.0 - 0.7 * smoothstep(0.75, 1.0, y));
    return vec4<f32>(framebuffer_encode(apply_tone_mapping(c * 1.5 * flicker)), a);
}
"#;

/// The realistic flame: a ray through the box, gathering what the gas
/// ([`FLAME_WGSL`](threers_math::flicker::FLAME_WGSL)) gives off. data[0]:
/// time, flicker, seed; data[1]: width, height, steps; data[2]: the lean.
const REALISTIC_FLAME: &str = r#"
@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let t = u_data.data[0].x;
    let flicker = u_data.data[0].y;
    let seed = u_data.data[0].z;
    let lean = u_data.data[2].xy;
    let steps = u32(u_data.data[1].z);
    // The ray in the box's own space (the box may be moved, turned, scaled).
    let dw = normalize(in.world_pos - frame.camera_position.xyz);
    let c0 = mesh.model[0].xyz;
    let c1 = mesh.model[1].xyz;
    let c2 = mesh.model[2].xyz;
    let dir = normalize(vec3<f32>(dot(c0, dw) / dot(c0, c0), dot(c1, dw) / dot(c1, c1), dot(c2, dw) / dot(c2, c2)));
    let half = vec3<f32>(u_data.data[1].x * 1.1, u_data.data[1].y * 0.625, u_data.data[1].x * 1.1);
    let o = in.local_pos;
    // Out the far side: the nearest of the three exits.
    let exit = (sign(dir) * half - o) / select(dir, vec3<f32>(1e-6), abs(dir) < vec3<f32>(1e-6));
    let far = max(min(min(exit.x, exit.y), exit.z), 0.0);
    let dt = far / f32(steps);
    // A different start in each pixel, so steps show as grain, not bands.
    var s = dt * flame_hash(vec3<f32>(in.clip_pos.xy, t));
    var c = vec3<f32>(0.0);
    for (var i = 0u; i < steps; i++) {
        let p = o + dir * s;
        let q = vec3<f32>(p.x / half.x, (p.y + half.y) / (2.0 * half.y), p.z / half.z);
        // The box is a quarter taller than the flame.
        c = c + flame_gas(vec3<f32>(q.x, q.y * 1.25, q.z), t, flicker, lean, seed) * (dt / half.x);
        s = s + dt;
    }
    // The gas's light is many times white in the core; a scene exposed for
    // candlelight takes about half of it before its tone mapping.
    // As bright, so opaque: over a camera's picture (AR) the box round the
    // flame leaves it as it is.
    let lit = apply_tone_mapping(c * flicker * 0.45);
    return vec4<f32>(framebuffer_encode(lit), clamp(max(lit.r, max(lit.g, lit.b)), 0.0, 1.0));
}
"#;

/// Light from the flame scattered through the wax near the top: brightest
/// at the rim of the pool, fading down the side, stronger at grazing angles
/// where more wax is between the eye and the light. data[0]: time, flicker;
/// data[1]: the body's height and radius.
const WAX_GLOW: &str = r#"
@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let flicker = u_data.data[0].y;
    let h = u_data.data[1].x;
    let r = u_data.data[1].y;
    let below = max(h - in.local_pos.y, 0.0);
    let n = normalize(in.world_normal);
    let v = normalize(frame.camera_position.xyz - in.world_pos);
    let grazing = 1.0 - abs(dot(n, v));
    let glow = exp(-below / (r * 0.55)) * (0.45 + 0.55 * grazing) * flicker * flicker;
    let c = vec3<f32>(1.0, 0.48, 0.16) * glow * 0.3;
    return vec4<f32>(framebuffer_encode(apply_tone_mapping(c)), 1.0);
}
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_candle_adds_its_parts_and_flickers_its_light() {
        for style in [FlameStyle::Simple, FlameStyle::Realistic] {
            let mut scene = Scene::new();
            let c = Candle::new(style).add_to(&mut scene, Vector3::new(1.0, 0.75, 0.0));
            let intensity = |scene: &Scene| match &scene.get(c.light).unwrap().kind {
                ObjectKind::Light(Light::Point(p)) => p.intensity,
                _ => panic!("a point light"),
            };
            let mut seen = vec![];
            for i in 0..50 {
                c.update(&mut scene, i as f32 * 0.07);
                seen.push(intensity(&scene));
            }
            let spread = seen.iter().cloned().fold(f32::MIN, f32::max) - seen.iter().cloned().fold(f32::MAX, f32::min);
            let base = Candle::new(style).light_intensity;
            assert!(spread / base > 0.1, "{style:?}: the light flickers ({spread} of {base})");
            assert_eq!(scene.get(c.root).unwrap().position, Vector3::new(1.0, 0.75, 0.0));
        }
    }

    #[test]
    fn the_profiles_close_at_the_axis() {
        let w = wax_profile(0.12, 0.024);
        assert_eq!(w.first().unwrap().x, 0.0);
        assert!(w.last().unwrap().x.abs() < 1e-6, "the pool meets the wick");
        let f = flame_profile(0.045, 0.007);
        assert!(f.first().unwrap().x.abs() < 1e-6 && f.last().unwrap().x.abs() < 1e-6);
    }
}
