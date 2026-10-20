//! Every shadow-casting light casts, not just the first of each kind.
//!
//! Three lights of one kind light a floor around a box. Where each light's
//! shadow falls comes from geometry, not from another renderer: a floor point
//! is in light `k`'s shadow when the segment from it to the light crosses the
//! box. Each pixel is measured against the same frame with the box casting
//! nothing, so falloff and cones cancel, in the regions that classification
//! predicts: blocked from light `k` alone must lose light, and reached by all
//! three must not, for every `k`. Before shadow maps became per-light arrays only light
//! 0 of a kind cast, and lights 1 and 2 failed this.
//!
//! Both renderers: wgpu, and Metal with the `metal` feature.
//!
//! Skipped when no GPU adapter is available.

use threers::lights::{DirectionalLight, Light, PointLight, SpotLight};
use threers::prelude::*;

const N: u32 = 256;
const EYE: Vector3 = Vector3::new(0.0, 14.0, 12.0);
const FOV_DEG: f32 = 45.0;
/// The occluder: an axis-aligned box standing on the floor.
const BOX_MIN: Vector3 = Vector3::new(-1.0, 0.0, -1.0);
const BOX_MAX: Vector3 = Vector3::new(1.0, 3.0, 1.0);
/// Off every axis of symmetry, so no two shadows coincide.
const LIGHTS: [Vector3; 3] = [
    Vector3::new(-6.0, 7.0, 3.0),
    Vector3::new(6.5, 7.5, 2.0),
    Vector3::new(1.0, 8.0, -6.0),
];

#[derive(Clone, Copy, Debug)]
enum Kind {
    Directional,
    Spot,
    Point,
}

fn scene(kind: Kind, casts: bool) -> Scene {
    let mut s = Scene::new();
    s.background = Color::from_hex(0x000000);
    for &at in &LIGHTS {
        let to_origin = (Vector3::ZERO - at).normalize();
        let light = match kind {
            Kind::Directional => {
                let mut l = DirectionalLight::new(Color::WHITE, 1.2);
                l.direction = to_origin;
                l.cast_shadow = true;
                l.shadow.camera_size = 16.0;
                l.shadow.camera_near = 0.5;
                l.shadow.camera_far = 60.0;
                l.shadow.normal_bias = 0.05;
                Light::Directional(l)
            }
            Kind::Spot => {
                let mut l = SpotLight::new(Color::WHITE, 120.0);
                l.angle = 0.9;
                l.penumbra = 0.1;
                l.direction = to_origin;
                l.cast_shadow = true;
                l.shadow.camera_near = 0.5;
                l.shadow.camera_far = 60.0;
                l.shadow.normal_bias = 0.05;
                Light::Spot(l)
            }
            Kind::Point => {
                let mut l = PointLight::new(Color::WHITE, 120.0);
                l.distance = 60.0;
                l.cast_shadow = true;
                l.shadow.normal_bias = 0.05;
                Light::Point(l)
            }
        };
        let mut obj = Object3D::light(light);
        obj.position = at;
        s.add(obj);
    }
    let mut m = StandardMaterial::new(Color::WHITE);
    m.metalness = 0.0;
    m.roughness = 1.0;
    let mut floor = Object3D::mesh(Mesh::new(PlaneGeometry::new(60.0, 60.0), m.clone().into()));
    floor.rotate_x(-std::f32::consts::FRAC_PI_2);
    floor.receive_shadow = true;
    s.add(floor);
    let size = BOX_MAX - BOX_MIN;
    let mut b = Object3D::mesh(Mesh::new(
        BoxGeometry::new(size.x, size.y, size.z),
        m.into(),
    ));
    b.position = (BOX_MIN + BOX_MAX) * 0.5;
    b.cast_shadow = casts;
    s.add(b);
    s
}

fn camera() -> PerspectiveCamera {
    let mut c = PerspectiveCamera::new(FOV_DEG, 1.0, 0.1, 100.0);
    c.position = EYE;
    c.look_at(Vector3::ZERO);
    c
}

/// Where along `o + t·d` (t in `(0, t_max)`) the ray enters the box, if it does.
fn hits_box(o: Vector3, d: Vector3, t_max: f32) -> Option<f32> {
    let (mut t0, mut t1) = (1e-4f32, t_max);
    for (o, d, lo, hi) in [
        (o.x, d.x, BOX_MIN.x, BOX_MAX.x),
        (o.y, d.y, BOX_MIN.y, BOX_MAX.y),
        (o.z, d.z, BOX_MIN.z, BOX_MAX.z),
    ] {
        if d.abs() < 1e-8 {
            if o < lo || o > hi {
                return None;
            }
            continue;
        }
        let (a, b) = ((lo - o) / d, (hi - o) / d);
        t0 = t0.max(a.min(b));
        t1 = t1.min(a.max(b));
        if t0 > t1 {
            return None;
        }
    }
    Some(t0)
}

/// The floor point under pixel `(x, y)` — `None` where the box hides it or
/// the ray misses the floor.
fn floor_point(x: f32, y: f32) -> Option<Vector3> {
    let f = (Vector3::ZERO - EYE).normalize();
    let r = f.cross(Vector3::UP).normalize();
    let u = r.cross(f);
    let h = (FOV_DEG.to_radians() * 0.5).tan();
    let (nx, ny) = (x / N as f32 * 2.0 - 1.0, 1.0 - y / N as f32 * 2.0);
    let d = (f + r * (nx * h) + u * (ny * h)).normalize();
    if d.y >= 0.0 {
        return None;
    }
    let t = -EYE.y / d.y;
    if hits_box(EYE, d, t).is_some() {
        return None;
    }
    Some(EYE + d * t)
}

/// Which of the three lights the box blocks from `p`, as bits.
fn blocked(kind: Kind, p: Vector3) -> u8 {
    let mut bits = 0;
    for (k, &at) in LIGHTS.iter().enumerate() {
        let (d, t) = match kind {
            // A directional light is the direction alone.
            Kind::Directional => (at.normalize(), 100.0),
            Kind::Spot | Kind::Point => ((at - p).normalize(), (at - p).length()),
        };
        if hits_box(p, d, t).is_some() {
            bits |= 1 << k;
        }
    }
    bits
}

/// The pixel's class, or `None` near a shadow edge (a penumbra is neither) or
/// off the visible floor.
fn class(kind: Kind, x: u32, y: u32) -> Option<u8> {
    let (cx, cy) = (x as f32 + 0.5, y as f32 + 0.5);
    let c = blocked(kind, floor_point(cx, cy)?);
    for (dx, dy) in [(-2.0, 0.0), (2.0, 0.0), (0.0, -2.0), (0.0, 2.0)] {
        if blocked(kind, floor_point(cx + dx, cy + dy)?) != c {
            return None;
        }
    }
    Some(c)
}

fn linear(b: u8) -> f64 {
    let c = b as f64 / 255.0;
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

#[test]
fn every_shadow_casting_light_casts() {
    let Ok(mut hr) = HeadlessRenderer::builder().size(N, N).build() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    check("wgpu", |kind, casts| {
        hr.render_to_rgba(&mut scene(kind, casts), &camera())
    });
}

#[cfg(feature = "metal")]
#[test]
fn every_shadow_casting_light_casts_on_metal() {
    use threers::metal::{MetalError, MetalHeadlessRenderer};
    let mut r = match MetalHeadlessRenderer::builder().size(N, N).build() {
        Ok(r) => r,
        Err(MetalError::NoDevice) => return,
        Err(e) => panic!("metal renderer: {e:?}"),
    };
    check("metal", |kind, casts| {
        r.render_to_rgba(&mut scene(kind, casts), &camera())
            .unwrap()
    });
}

/// Each light's shadow, measured in `render`'s frames.
fn check(backend: &str, mut render: impl FnMut(Kind, bool) -> Vec<u8>) {
    for kind in [Kind::Directional, Kind::Spot, Kind::Point] {
        let img = render(kind, true);
        let bare = render(kind, false);
        // Mean fraction of the light kept, and pixel count, per class.
        let mut sum = [0.0f64; 8];
        let mut count = [0usize; 8];
        for y in 0..N {
            for x in 0..N {
                let Some(c) = class(kind, x, y) else { continue };
                let i = ((y * N + x) * 4) as usize;
                let lum = |img: &[u8]| (0..3).map(|j| linear(img[i + j])).sum::<f64>();
                if lum(&bare) < 1e-3 {
                    continue;
                }
                sum[c as usize] += lum(&img) / lum(&bare);
                count[c as usize] += 1;
            }
        }
        let mean = |c: usize| sum[c] / count[c].max(1) as f64;
        assert!(
            count[0] > 1000,
            "{kind:?}: too little fully lit floor in view ({})",
            count[0]
        );
        assert!(
            mean(0) > 0.97,
            "{kind:?}: floor no light is blocked from keeps only {:.3}",
            mean(0)
        );
        for k in 0..3 {
            let only = 1usize << k;
            assert!(
                count[only] >= 40,
                "{kind:?}: light {k}'s shadow is barely in view ({} px) — the fixture is off",
                count[only]
            );
            let ratio = mean(only);
            eprintln!("{backend} {kind:?} light {k}: {} px in its shadow alone, keeping {ratio:.3} of the light", count[only]);
            assert!(
                ratio < 0.9,
                "{backend} {kind:?}: where only light {k} is blocked the floor keeps {ratio:.3} of its \
                 light — that light casts no shadow"
            );
            assert!(
                ratio > 0.2,
                "{kind:?}: light {k}'s shadow also blacks out the other lights ({ratio:.3})"
            );
        }
    }
}
