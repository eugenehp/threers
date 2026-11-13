//! Progress indicators as objects in a scene: a glass tube or a glass ring
//! with glowing liquid in it, floating wherever it is put. Each is one box
//! mesh whose shader ray-marches the shapes inside it: the glass reflecting
//! a studio (more at a glancing angle), the liquid seen through it bent by
//! refraction, its colour running through three stops along the length,
//! and its light spilling round it as a glow. Added light, so it shows
//! over anything behind it and feeds a bloom pass.
//!
//! The same looks as the UI's indicators (`threers::ui::indicators`), in
//! metres and in the world: turned and moved with the object, seen in the
//! scene's perspective, from any side, in a headset.
//!
//! ```
//! use threers::extras::progress::{Indicator3d, Palette3d};
//! use threers::{Scene, Vector3};
//!
//! let mut scene = Scene::new();
//! let ring = Indicator3d::ring(0.3, 0.05).palette(Palette3d::EMBER).add_to(&mut scene, Vector3::new(0.0, 1.5, -1.0));
//! // Each frame:
//! ring.set(Some(0.4));
//! ring.update(&mut scene, 1.25);
//! ```

use crate::core::{Mesh, Object3D, ObjectId};
use crate::geometries::BoxGeometry;
use crate::materials::{Blending, Material, ShaderMaterial};
use crate::math::{Color, Euler, Vector3};
use crate::scene::Scene;

/// A tube or a ring.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Shape3d {
    /// Along x: `length` end to end, the glass `radius` thick.
    Bar { length: f32, radius: f32 },
    /// In the xy plane, facing +z: `radius` to the tube's middle, the tube
    /// `tube` thick (its radius).
    Ring { radius: f32, tube: f32 },
}

/// Three colours from the start to the end (linear, as [`Color`] is).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Palette3d(pub [Color; 3]);

impl Palette3d {
    pub const AURORA: Palette3d = Palette3d([hex(0x22ff9a), hex(0x22d3ee), hex(0xa855f7)]);
    pub const EMBER: Palette3d = Palette3d([hex(0xff2d55), hex(0xff8a00), hex(0xffe14d)]);
    pub const OCEAN: Palette3d = Palette3d([hex(0x2b5cff), hex(0x00a8ff), hex(0x5ef2ff)]);
    pub const NEON: Palette3d = Palette3d([hex(0xff2bd6), hex(0x8b5cf6), hex(0x00e5ff)]);
    pub const CITRUS: Palette3d = Palette3d([hex(0x9dff3a), hex(0xffd21f), hex(0xff6b6b)]);

    /// From three `0xRRGGBB` sRGB colours.
    pub fn hex(from: u32, mid: u32, to: u32) -> Palette3d {
        Palette3d([Color::from_hex(from), Color::from_hex(mid), Color::from_hex(to)])
    }

    /// From one colour to another, through their mix.
    pub fn two(from: Color, to: Color) -> Palette3d {
        let mid = Color { r: (from.r + to.r) * 0.5, g: (from.g + to.g) * 0.5, b: (from.b + to.b) * 0.5 };
        Palette3d([from, mid, to])
    }
}

/// `0xRRGGBB` decoded to linear, at compile time (the sRGB curve as
/// [`Color::from_hex`] has it, to within a part in a thousand).
const fn hex(c: u32) -> Color {
    Color { r: lin(((c >> 16) & 255) as f32 / 255.0), g: lin(((c >> 8) & 255) as f32 / 255.0), b: lin((c & 255) as f32 / 255.0) }
}

const fn lin(v: f32) -> f32 {
    if v <= 0.04045 {
        v / 12.92
    } else {
        // ((v + 0.055) / 1.055)^2.4 by x^2 * x^0.4, x^0.4 ~ a few Newton
        // steps of the fifth root of x^2 (const fns have no powf).
        let x = (v + 0.055) / 1.055;
        let x2 = x * x;
        let mut y = x;
        let mut i = 0;
        while i < 12 {
            y = y - (y * y * y * y * y - x2) / (5.0 * y * y * y * y);
            i += 1;
        }
        x2 * y
    }
}

/// An indicator to add to a scene. Start from [`Indicator3d::bar`] or
/// [`Indicator3d::ring`].
#[derive(Debug, Clone, PartialEq)]
pub struct Indicator3d {
    pub shape: Shape3d,
    pub palette: Palette3d,
    /// How far its light reaches past the glass, as a share of the glass's
    /// thickness (0 for none).
    pub glow: f32,
    /// How bright the liquid is (1 as designed; more for bloom to catch).
    pub intensity: f32,
    /// A band of light running along the liquid.
    pub shimmer: bool,
    /// Turning round its own vertical axis, in turns a second.
    pub spin: f32,
    /// Bobbing up and down this far, in metres.
    pub bob: f32,
    /// Most steps through the box a pixel, near (fewer as it gets smaller
    /// on screen: a few pixels thick, 16 and no refraction).
    pub steps: u32,
}

impl Indicator3d {
    /// A tube `length` long, the glass `radius` thick.
    pub fn bar(length: f32, radius: f32) -> Self {
        Self::new(Shape3d::Bar { length, radius })
    }

    /// A ring of `radius`, its tube `tube` thick.
    pub fn ring(radius: f32, tube: f32) -> Self {
        Self::new(Shape3d::Ring { radius, tube })
    }

    fn new(shape: Shape3d) -> Self {
        Self { shape, palette: Palette3d::AURORA, glow: 2.5, intensity: 1.0, shimmer: false, spin: 0.0, bob: 0.0, steps: 64 }
    }

    pub fn palette(mut self, palette: Palette3d) -> Self {
        self.palette = palette;
        self
    }

    pub fn glow(mut self, glow: f32) -> Self {
        self.glow = glow.max(0.0);
        self
    }

    pub fn intensity(mut self, intensity: f32) -> Self {
        self.intensity = intensity.max(0.0);
        self
    }

    pub fn shimmer(mut self, shimmer: bool) -> Self {
        self.shimmer = shimmer;
        self
    }

    pub fn spin(mut self, turns_per_second: f32) -> Self {
        self.spin = turns_per_second;
        self
    }

    pub fn bob(mut self, metres: f32) -> Self {
        self.bob = metres.max(0.0);
        self
    }

    /// The glass's thickness: what the glow is measured in.
    fn thickness(&self) -> f32 {
        match self.shape {
            Shape3d::Bar { radius, .. } => radius,
            Shape3d::Ring { tube, .. } => tube,
        }
    }

    /// Adds it at `position`, at 0%, and hands back what moves it on.
    pub fn add_to(&self, scene: &mut Scene, position: Vector3) -> IndicatorHandle3d {
        let reach = self.thickness() * self.glow;
        let (kind, a, r, size) = match self.shape {
            Shape3d::Bar { length, radius } => {
                let half = (length * 0.5 - radius).max(0.0);
                let e = radius + reach;
                (0.0, half, radius, Vector3::new((half + e) * 2.0, e * 2.0, e * 2.0))
            }
            Shape3d::Ring { radius, tube } => {
                let e = radius + tube + reach;
                (1.0, radius, tube, Vector3::new(e * 2.0, e * 2.0, (tube + reach) * 2.0))
            }
        };
        let [c0, c1, c2] = self.palette.0;
        let mut data = vec![[0.0f32; 4]; 16];
        data[0] = [0.0, 0.0, 0.0, if self.shimmer { 1.0 } else { 0.0 }];
        data[1] = [kind, a, r, reach];
        data[2] = [c0.r, c0.g, c0.b, 0.0];
        data[3] = [c1.r, c1.g, c1.b, 0.0];
        data[4] = [c2.r, c2.g, c2.b, 0.0];
        data[5] = [size.x * 0.5, size.y * 0.5, size.z * 0.5, self.steps as f32];
        data[6] = [self.intensity, self.glow.clamp(0.0, 1.0), 0.0, 0.0];
        let mut material = ShaderMaterial::new(FRAGMENT).with_data(data).with_transparent(true).with_depth_write(false);
        material.render_state.blending = Blending::Additive;
        let mut object = Object3D::mesh(Mesh::new(BoxGeometry::new(size.x, size.y, size.z), Material::Shader(material.clone())));
        object.position = position;
        let id = scene.add(object);
        IndicatorHandle3d { id, material, at: position, spin: self.spin, bob: self.bob, shimmer: self.shimmer, value: std::cell::Cell::new(Some(0.0)) }
    }
}

/// An indicator in a scene: [`set`](Self::set) its value, and
/// [`update`](Self::update) it each frame.
#[derive(Debug, Clone)]
pub struct IndicatorHandle3d {
    /// Its object (turn it, scale it, move it).
    pub id: ObjectId,
    material: ShaderMaterial,
    at: Vector3,
    spin: f32,
    bob: f32,
    shimmer: bool,
    value: std::cell::Cell<Option<f32>>,
}

impl IndicatorHandle3d {
    /// How far along, 0 to 1; `None` while it is not known (busy: a comet
    /// runs along the tube, an arc chases round the ring).
    pub fn set(&self, value: Option<f32>) {
        self.value.set(value.map(|v| v.clamp(0.0, 1.0)));
    }

    pub fn value(&self) -> Option<f32> {
        self.value.get()
    }

    /// The liquid, the shimmer, and the spin and bob, at `time` seconds.
    pub fn update(&self, scene: &mut Scene, time: f32) {
        let v = self.value.get();
        self.material.set_uniform(0, [time, v.unwrap_or(0.0), if v.is_none() { 1.0 } else { 0.0 }, if self.shimmer { 1.0 } else { 0.0 }]);
        if (self.spin != 0.0 || self.bob != 0.0)
            && let Some(o) = scene.get_mut(self.id)
        {
            if self.spin != 0.0 {
                o.quaternion = Euler::new(0.0, time * self.spin * std::f32::consts::TAU, 0.0).to_quaternion();
            }
            o.position = self.at + Vector3::new(0.0, (time * 1.7).sin() * self.bob, 0.0);
        }
    }
}

/// data[0]: time, value, busy, shimmer. data[1]: kind (0 tube, 1 ring), the
/// tube's half length or the ring's radius, the glass's radius, the glow's
/// reach. data[2..5]: the colours. data[5]: the box's half size, steps.
/// data[6]: intensity, glow strength.
const FRAGMENT: &str = r#"
const PI_TAU: f32 = 6.2831853;

fn pi_ramp(t: f32) -> vec3<f32> {
    let u = clamp(t, 0.0, 1.0);
    let a = mix(u_data.data[2].rgb, u_data.data[3].rgb, clamp(u * 2.0, 0.0, 1.0));
    let b = mix(u_data.data[3].rgb, u_data.data[4].rgb, clamp(u * 2.0 - 1.0, 0.0, 1.0));
    return select(b, a, u < 0.5);
}

// Round a ring, closed: no seam at the top when it is full.
fn pi_ring_ramp(t: f32) -> vec3<f32> {
    let u = fract(t);
    return mix(pi_ramp(u / 0.9), u_data.data[2].rgb, smoothstep(0.9, 1.0, u));
}

// Where the liquid runs: along the tube (0 to 1), or round the ring (start
// and length in turns).
fn pi_span(now: f32, value: f32, busy: f32, ring: bool) -> vec2<f32> {
    if (busy < 0.5) {
        return vec2<f32>(0.0, clamp(value, 0.0, 1.0));
    }
    if (ring) {
        let breathe = 0.5 + 0.5 * sin(now * 3.1);
        return vec2<f32>(fract(now * 0.9 + 0.08 * sin(now * 3.1)), 0.1 + 0.55 * breathe);
    }
    let c = fract(now * 0.6);
    let e = c * c * (3.0 - 2.0 * c);
    let head = e * 1.45 - 0.05;
    return vec2<f32>(clamp(head - 0.4, 0.0, 1.0), clamp(head, 0.0, 1.0) - clamp(head - 0.4, 0.0, 1.0));
}

fn pi_segment(q: vec3<f32>, a: vec3<f32>, b: vec3<f32>) -> vec3<f32> {
    let ab = b - a;
    return a + ab * clamp(dot(q - a, ab) / max(dot(ab, ab), 0.0000001), 0.0, 1.0);
}

fn pi_core(q: vec3<f32>, rr: f32) -> vec3<f32> {
    let l = length(q.xy);
    return vec3<f32>(select(vec2<f32>(0.0, rr), q.xy / l * rr, l > 0.0000001), 0.0);
}

// The nearest point of the glass's core (a segment, or a circle).
fn pi_glass_core(q: vec3<f32>) -> vec3<f32> {
    let a = u_data.data[1].y;
    if (u_data.data[1].x < 0.5) {
        return pi_segment(q, vec3<f32>(-a, 0.0, 0.0), vec3<f32>(a, 0.0, 0.0));
    }
    return pi_core(q, a);
}

// The nearest point of the liquid's core: a part of the segment, or an
// arc of the circle with round ends.
fn pi_liquid_core(q: vec3<f32>, span: vec2<f32>) -> vec3<f32> {
    let a = u_data.data[1].y;
    if (u_data.data[1].x < 0.5) {
        let x0 = -a + 2.0 * a * span.x;
        let x1 = x0 + 2.0 * a * span.y;
        return pi_segment(q, vec3<f32>(x0, 0.0, 0.0), vec3<f32>(x1, 0.0, 0.0));
    }
    if (span.y >= 0.9995) {
        return pi_core(q, a);
    }
    let ang = fract(atan2(q.x, q.y + 0.00001) / PI_TAU);
    if (fract(ang - span.x) <= span.y) {
        return pi_core(q, a);
    }
    let a0 = span.x * PI_TAU;
    let a1 = (span.x + span.y) * PI_TAU;
    let e0 = vec3<f32>(sin(a0), cos(a0), 0.0) * a;
    let e1 = vec3<f32>(sin(a1), cos(a1), 0.0) * a;
    return select(e1, e0, length(q - e0) < length(q - e1));
}

// How far along the liquid `q` is, 0 to 1 (busy: from its tail to its head).
fn pi_along(q: vec3<f32>, span: vec2<f32>, busy: bool) -> f32 {
    let a = u_data.data[1].y;
    if (u_data.data[1].x < 0.5) {
        let u = (q.x + a) / max(2.0 * a, 0.000001);
        return select(clamp(u, 0.0, 1.0), clamp((u - span.x) / max(span.y, 0.0001), 0.0, 1.0), busy);
    }
    let ang = fract(atan2(q.x, q.y + 0.00001) / PI_TAU);
    let rel = fract(ang - span.x);
    let kk = clamp(rel / max(span.y, 0.0001), 0.0, 1.0);
    let k = select(kk, select(1.0, 0.0, rel > span.y + (1.0 - span.y) * 0.5), rel > span.y);
    return select(select(k * span.y, ang, span.y >= 0.9995), k, busy);
}

fn pi_colour(t: f32, busy: bool) -> vec3<f32> {
    if (u_data.data[1].x > 0.5 && !busy) {
        return pi_ring_ramp(t);
    }
    return pi_ramp(t);
}

// A studio for the glass to reflect, in the world (y up).
fn pi_env(r: vec3<f32>) -> vec3<f32> {
    let sky = mix(vec3<f32>(0.03, 0.035, 0.05), vec3<f32>(0.16, 0.18, 0.24), clamp(r.y, 0.0, 1.0));
    var c = select(vec3<f32>(0.01), sky, r.y > -0.05);
    c = c + vec3<f32>(2.2) * smoothstep(0.8, 0.9, r.y) * smoothstep(0.6, 0.35, abs(r.x));
    let side = (atan2(r.x, r.z + 0.00001) - 1.9) / 0.12;
    c = c + vec3<f32>(1.4) * exp(-side * side) * smoothstep(-0.25, 0.25, r.y);
    let rim = (atan2(r.x, r.z + 0.00001) + 1.7) / 0.08;
    c = c + vec3<f32>(0.45, 0.62, 1.0) * 0.8 * exp(-rim * rim) * smoothstep(-0.4, 0.2, r.y);
    return c;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let now = u_data.data[0].x;
    let busy = u_data.data[0].z > 0.5;
    let ring = u_data.data[1].x > 0.5;
    let rg = u_data.data[1].z;
    let ri = rg * 0.6;
    let reach = max(u_data.data[1].w, 0.000001);
    let span = pi_span(now, u_data.data[0].y, u_data.data[0].z, ring);
    let has = span.y > 0.0005;
    let half = u_data.data[5].xyz;
    // How big a pixel is here, in the box's own units: what the steps,
    // the refraction and the softened outline go by.
    // (Where two faces of the box meet in a pixel the derivative jumps:
    // held to a fraction of the tube.)
    let px = clamp(length(fwidth(in.local_pos)), 0.0000001, u_data.data[1].z * 0.2);
    // The ray in the box's own space (the box may be moved, turned, scaled).
    let dw = normalize(in.world_pos - frame.camera_position.xyz);
    let c0 = mesh.model[0].xyz;
    let c1 = mesh.model[1].xyz;
    let c2 = mesh.model[2].xyz;
    let rd = normalize(vec3<f32>(dot(c0, dw) / dot(c0, c0), dot(c1, dw) / dot(c1, c1), dot(c2, dw) / dot(c2, c2)));
    let o = in.local_pos;
    let exit = (sign(rd) * half - o) / select(rd, vec3<f32>(0.000001), abs(rd) < vec3<f32>(0.000001));
    let far = max(min(min(exit.x, exit.y), exit.z), 0.0);
    // Level of detail: a tube only a few pixels thick gets fewer steps and
    // no refraction (the liquid seen straight through).
    let pixels = rg / px;
    let steps = clamp(i32(pixels * 4.0), 16, i32(u_data.data[5].w));
    let refracted = pixels > 3.0;
    let eps = max(rg * 0.004, px * 0.25);
    // To the glass, noting how near the liquid comes (the glow) and the
    // glass (its outline, softened over a pixel).
    var t = 0.0;
    var hit = false;
    var near = 100000.0;
    var near_at = o;
    var near_g = 100000.0;
    var near_gt = 0.0;
    for (var i = 0; i < steps; i = i + 1) {
        let q = o + rd * t;
        let dg = length(q - pi_glass_core(q)) - rg;
        let dl = select(100000.0, length(q - pi_liquid_core(q, span)) - ri, has);
        if (dl < near) {
            near = dl;
            near_at = q;
        }
        if (dg < near_g) {
            near_g = dg;
            near_gt = t;
        }
        if (dg < eps) {
            hit = true;
            break;
        }
        t = t + max(min(dg, dl + rg * 0.5), eps);
        if (t > far) {
            break;
        }
    }
    var light = vec3<f32>(0.0);
    // The glow, coloured by the nearest of the liquid.
    let hk = pi_along(near_at, span, busy);
    let hfade = select(1.0, smoothstep(0.0, 0.85, hk), busy);
    let w = clamp(max(near, 0.0) / reach, 0.0, 1.0);
    let glow = exp(-4.0 * w) * (1.0 - w) * (1.0 - w) * u_data.data[6].y;
    light = light + select(vec3<f32>(0.0), pi_colour(hk, busy) * glow * 0.6 * hfade, has);
    // The glass, hit or passed within a pixel of: its outline smoothed
    // (MSAA smooths the box's edges, not what is ray-marched inside it).
    let gcov = select(clamp(1.0 - near_g / (px * 1.5), 0.0, 1.0), 1.0, hit);
    if (gcov > 0.0) {
        let q = o + rd * select(near_gt, t, hit);
        let n = normalize(q - pi_glass_core(q));
        let nw = normalize((mesh.normal_matrix * vec4<f32>(n, 0.0)).xyz);
        // The liquid through the glass, along the bent ray.
        let rr = refract(rd, n, 1.0 / 1.5);
        var tl = 0.0;
        var near_l = 100000.0;
        var at_l = q;
        if (has && refracted) {
            for (var i = 0; i < 32; i = i + 1) {
                let p = q + rr * tl;
                let d = length(p - pi_liquid_core(p, span)) - ri;
                if (d < near_l) {
                    near_l = d;
                    at_l = p;
                }
                if (d < eps) {
                    break;
                }
                tl = tl + max(d, eps);
                if (tl > rg * 4.0) {
                    break;
                }
            }
        }
        // Too small to refract: the liquid straight through, by how near
        // the ray came.
        if (has && !refracted) {
            near_l = near;
            at_l = near_at;
        }
        let k = pi_along(at_l, span, busy);
        let fade = select(1.0, smoothstep(0.0, 0.85, k), busy);
        let col = pi_colour(k, busy);
        let ln = normalize(at_l - pi_liquid_core(at_l, span));
        let facing = max(dot(ln, -rr), 0.0);
        let sh = u_data.data[0].w * exp(-pow(fract(k * 0.8 - now * 0.55) - 0.5, 2.0) * 90.0);
        let liquid = (col * (0.45 + 0.8 * facing) * (1.0 + sh * 0.8) + vec3<f32>(pow(facing, 8.0) * 0.45 + sh * 0.2)) * fade * u_data.data[6].x;
        let cover = select(0.0, clamp(1.0 - near_l / max(rg * 0.08, px), 0.0, 1.0), has);
        // The liquid's light scattered in the glass round it.
        let dl = length(q - pi_liquid_core(q, span)) - ri;
        let scatter = select(vec3<f32>(0.0), col * 0.3 * exp(-max(dl, 0.0) / (rg * 1.5)) * fade, has);
        // The glass: Fresnel, a studio reflected, two speculars.
        let ndv = clamp(-dot(nw, dw), 0.0, 1.0);
        let f = 0.04 + 0.96 * pow(1.0 - ndv, 5.0);
        let sun = normalize(vec3<f32>(-0.4, 0.8, 0.45));
        let nh = max(dot(nw, normalize(sun - dw)), 0.0);
        let spec = pow(nh, 220.0) * 3.0 + pow(nh, 28.0) * 0.1;
        let tint = mix(vec3<f32>(1.0), vec3<f32>(0.85, 0.93, 1.0), 0.25 + 0.5 * pow(1.0 - ndv, 2.0));
        let through = (liquid * cover + scatter * (1.0 - cover)) * tint * (1.0 - f);
        let glass = through + pi_env(reflect(dw, nw)) * f + vec3<f32>(spec) + vec3<f32>(0.03, 0.035, 0.045) * (0.3 + pow(1.0 - ndv, 2.0));
        light = mix(light, light * (1.0 - cover) + glass, gcov);
    }
    let out = apply_tone_mapping(light);
    return vec4<f32>(framebuffer_encode(out), 1.0);
}
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_palette_constants_match_color_from_hex() {
        for (c, h) in [(Palette3d::EMBER.0[1], 0xff8a00), (Palette3d::AURORA.0[2], 0xa855f7)] {
            let want = Color::from_hex(h);
            for (a, b) in [(c.r, want.r), (c.g, want.g), (c.b, want.b)] {
                assert!((a - b).abs() < 1e-3, "{a} vs {b}");
            }
        }
    }

    #[test]
    fn it_is_added_as_one_box_and_takes_its_value() {
        let mut scene = Scene::new();
        let h = Indicator3d::bar(1.0, 0.05).spin(0.25).add_to(&mut scene, Vector3::new(0.0, 1.0, 0.0));
        h.set(Some(1.5));
        assert_eq!(h.value(), Some(1.0));
        h.set(None);
        h.update(&mut scene, 1.0);
        assert!(scene.get(h.id).is_some());
    }
}
