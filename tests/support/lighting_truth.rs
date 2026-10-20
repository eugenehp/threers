//! three.js r165's material models, evaluated analytically, for holding the
//! raster backends to three.js rather than to each other.
//!
//! One unit sphere at the origin, seen by a perspective camera on +Z, under an
//! ambient, a hemisphere, a directional, a point and a spot light. For any
//! pixel the ray through its centre gives the exact surface point, normal and
//! view direction, and the functions here return the colour three.js's
//! `WebGLRenderer` would write there — the formulas transcribed from r165's
//! `lights_pars_begin`, `lights_lambert_pars_fragment`,
//! `lights_phong_pars_fragment` (`BRDF_BlinnPhong`), `lights_toon_pars_fragment`
//! (`getGradientIrradiance` without a gradient map),
//! `lights_physical_pars_fragment` (`BRDF_GGX`, `V_GGX_SmithCorrelated`,
//! `D_GGX`), `common`'s `F_Schlick`, and the `meshnormal` / `meshmatcap` /
//! `meshbasic` fragment shaders. Physically-correct light units (r155+): no
//! factor of π on any light; `BRDF_Lambert = diffuse / π` on direct *and*
//! indirect (ambient, hemisphere) irradiance.
//!
//! Shared by `tests/material_truth.rs` (wgpu) and `tests/backend_parity.rs`
//! (Metal) through `#[path]`.
#![allow(dead_code)]

use threers::cameras::PerspectiveCamera;
use threers::core::{Mesh, Object3D};
use threers::geometries::SphereGeometry;
use threers::lights::{
    AmbientLight, DirectionalLight, HemisphereLight, Light, PointLight, SpotLight,
};
use threers::materials::Material;
use threers::math::{Color, Vector3};
use threers::scene::Scene;

pub const N: u32 = 160;
pub const CAM_Z: f64 = 6.0;
pub const FOV_DEG: f64 = 30.0;

type V3 = [f64; 3];

fn sub(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn add(a: V3, b: V3) -> V3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}
fn mul(a: V3, s: f64) -> V3 {
    [a[0] * s, a[1] * s, a[2] * s]
}
fn hadamard(a: V3, b: V3) -> V3 {
    [a[0] * b[0], a[1] * b[1], a[2] * b[2]]
}
fn dot(a: V3, b: V3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn cross(a: V3, b: V3) -> V3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
fn norm(a: V3) -> V3 {
    mul(a, 1.0 / dot(a, a).sqrt())
}
fn saturate(x: f64) -> f64 {
    x.clamp(0.0, 1.0)
}
fn smoothstep(e0: f64, e1: f64, x: f64) -> f64 {
    let t = saturate((x - e0) / (e1 - e0));
    t * t * (3.0 - 2.0 * t)
}

// The lights. Linear colours, intensities folded in where three.js folds them
// (`WebGLLights`: `color * intensity`).
pub const AMBIENT: V3 = [0.10, 0.08, 0.06];
pub const HEMI_SKY: V3 = [0.10, 0.14, 0.20];
pub const HEMI_GROUND: V3 = [0.05, 0.03, 0.0];
pub const DIR_COLOR: V3 = [1.4, 1.4, 1.4];
/// Direction the directional light travels.
pub const DIR_TRAVEL: V3 = [-1.0, -0.5, -1.0];
pub const POINT_POS: V3 = [-2.5, 1.0, 2.5];
pub const POINT_COLOR: V3 = [6.0, 4.8, 3.6];
pub const SPOT_POS: V3 = [1.5, -2.5, 3.0];
pub const SPOT_TARGET: V3 = [0.3, -0.3, 0.0];
pub const SPOT_COLOR: V3 = [6.0, 8.0, 10.0];
pub const SPOT_ANGLE: f64 = 0.4;
pub const SPOT_PENUMBRA: f64 = 0.5;

fn c(v: V3) -> Color {
    Color::new(v[0] as f32, v[1] as f32, v[2] as f32)
}
fn v3(v: V3) -> Vector3 {
    Vector3::new(v[0] as f32, v[1] as f32, v[2] as f32)
}

pub fn camera() -> PerspectiveCamera {
    let mut cam = PerspectiveCamera::new(FOV_DEG as f32, 1.0, 0.1, 100.0);
    cam.position = Vector3::new(0.0, 0.0, CAM_Z as f32);
    cam.look_at(Vector3::ZERO);
    cam
}

/// Every light the analytic model knows about.
pub fn add_lights(scene: &mut Scene) {
    scene.add(Object3D::light(Light::Ambient(AmbientLight::new(c(AMBIENT), 1.0))));
    scene.add(Object3D::light(Light::Hemisphere(HemisphereLight::new(
        c(HEMI_SKY),
        c(HEMI_GROUND),
        1.0,
    ))));
    scene.add(Object3D::light(Light::Directional(
        DirectionalLight::new(Color::WHITE, DIR_COLOR[0] as f32).with_direction(v3(DIR_TRAVEL)),
    )));
    let mut p = PointLight::new(c(mul(POINT_COLOR, 1.0 / 6.0)), 6.0);
    p.decay = 2.0;
    p.distance = 0.0;
    let mut po = Object3D::light(Light::Point(p));
    po.position = v3(POINT_POS);
    scene.add(po);
    let mut s = SpotLight::new(c(mul(SPOT_COLOR, 1.0 / 10.0)), 10.0);
    s.angle = SPOT_ANGLE as f32;
    s.penumbra = SPOT_PENUMBRA as f32;
    s.decay = 2.0;
    s.distance = 0.0;
    s.direction = v3(norm(sub(SPOT_TARGET, SPOT_POS)));
    let mut so = Object3D::light(Light::Spot(s));
    so.position = v3(SPOT_POS);
    scene.add(so);
}

/// The unit sphere in `material`, finely tessellated so interpolated normals
/// are the sphere's own, under every light.
pub fn sphere_scene(material: Material) -> Scene {
    let mut scene = Scene::new();
    scene.background = Color::from_hex(0x000000);
    scene.add(Object3D::mesh(Mesh::new(SphereGeometry::new(1.0, 192, 96), material)));
    add_lights(&mut scene);
    scene
}

/// The surface the ray through pixel `(x, y)`'s centre meets: (point, normal,
/// direction to the camera). `None` off the sphere or within `margin` of its
/// silhouette (in units of the sphere's radius, measured on n·v).
pub fn hit(x: u32, y: u32) -> Option<(V3, V3, V3)> {
    let t = (FOV_DEG.to_radians() / 2.0).tan();
    let ndc_x = (x as f64 + 0.5) / N as f64 * 2.0 - 1.0;
    let ndc_y = 1.0 - (y as f64 + 0.5) / N as f64 * 2.0;
    let o = [0.0, 0.0, CAM_Z];
    let d = norm([ndc_x * t, ndc_y * t, -1.0]);
    // |o + s d|² = 1
    let b = dot(o, d);
    let cc = dot(o, o) - 1.0;
    let disc = b * b - cc;
    if disc <= 0.0 {
        return None;
    }
    let s = -b - disc.sqrt();
    let p = add(o, mul(d, s));
    let n = norm(p);
    let v = norm(sub(o, p));
    if dot(n, v) < 0.25 {
        return None;
    }
    Some((p, n, v))
}

/// One direct light's incident direction and colour after attenuation
/// (three.js's `IncidentLight`: `direction`, `color`).
fn direct_lights(p: V3) -> Vec<(V3, V3)> {
    let mut out = Vec::new();
    out.push((norm(mul(DIR_TRAVEL, -1.0)), DIR_COLOR));
    // getPointLightInfo: color * getDistanceAttenuation(d, 0, 2).
    let lv = sub(POINT_POS, p);
    let d = dot(lv, lv).sqrt();
    out.push((norm(lv), mul(POINT_COLOR, 1.0 / d.powf(2.0).max(0.01))));
    // getSpotLightInfo: spotAttenuation = smoothstep(coneCos, penumbraCos, angleCos).
    let lv = sub(SPOT_POS, p);
    let d = dot(lv, lv).sqrt();
    let l = norm(lv);
    let axis = norm(sub(SPOT_TARGET, SPOT_POS));
    let angle_cos = dot(l, mul(axis, -1.0));
    let cone_cos = SPOT_ANGLE.cos();
    let penumbra_cos = (SPOT_ANGLE * (1.0 - SPOT_PENUMBRA)).cos();
    let spot = if angle_cos > cone_cos {
        smoothstep(cone_cos, penumbra_cos, angle_cos)
    } else {
        0.0
    };
    out.push((l, mul(SPOT_COLOR, spot / d.powf(2.0).max(0.01))));
    out
}

/// Ambient + hemisphere irradiance (`getAmbientLightIrradiance`,
/// `getHemisphereLightIrradiance`).
fn indirect_irradiance(n: V3) -> V3 {
    let w = 0.5 * n[1] + 0.5;
    add(AMBIENT, add(mul(HEMI_GROUND, 1.0 - w), mul(HEMI_SKY, w)))
}

fn f_schlick(f0: V3, f90: f64, dot_vh: f64) -> V3 {
    let fresnel = 2f64.powf((-5.55473 * dot_vh - 6.98316) * dot_vh);
    add(mul(f0, 1.0 - fresnel), [f90 * fresnel; 3])
}

/// A three.js material, by the parameters its shading reads.
#[derive(Clone, Copy, Debug)]
pub enum Model {
    Basic { color: V3 },
    Lambert { color: V3 },
    Phong { color: V3, specular: V3, shininess: f64 },
    /// No gradient map: three.js's two-tone 0.7 / 1.0 ramp.
    Toon { color: V3 },
    /// Standard, and Physical at its defaults (ior 1.5, specularIntensity 1).
    Standard { color: V3, roughness: f64, metalness: f64 },
    Normal,
    /// No matcap texture: three.js's default `mix(0.2, 0.8, uv.y)`.
    Matcap { color: V3 },
}

/// What three.js writes for `model` at the surface (`p`, `n`, `v`): linear
/// outgoing radiance for the colour materials (encode with [`encode`]), the
/// packed normal itself for `Normal`. `None` where the answer depends on
/// screen-space derivatives (the toon ramp's antialiased step).
pub fn shade(model: Model, p: V3, n: V3, v: V3) -> Option<V3> {
    shade_with(model, n, v, &direct_lights(p), indirect_irradiance(n))
}

/// [`shade`] under explicit lights: `lights` are three.js `IncidentLight`s
/// (direction to the light, colour after attenuation and shadow),
/// `indirect` the ambient + hemisphere irradiance.
pub fn shade_with(model: Model, n: V3, v: V3, lights: &[(V3, V3)], indirect: V3) -> Option<V3> {
    let inv_pi = 1.0 / std::f64::consts::PI;
    Some(match model {
        Model::Basic { color } => color,
        Model::Lambert { color } => {
            let mut irr = indirect;
            for (l, lc) in lights {
                irr = add(irr, mul(*lc, saturate(dot(n, *l))));
            }
            mul(hadamard(irr, color), inv_pi)
        }
        Model::Phong { color, specular, shininess } => {
            let mut irr = indirect;
            let mut spec = [0.0; 3];
            for (l, lc) in lights {
                let ndl = saturate(dot(n, *l));
                let e = mul(*lc, ndl);
                irr = add(irr, e);
                let h = norm(add(*l, v));
                let ndh = saturate(dot(n, h));
                let vdh = saturate(dot(v, h));
                let f = f_schlick(specular, 1.0, vdh);
                let d = inv_pi * (shininess * 0.5 + 1.0) * ndh.powf(shininess);
                spec = add(spec, hadamard(e, mul(f, 0.25 * d)));
            }
            add(mul(hadamard(irr, color), inv_pi), spec)
        }
        Model::Toon { color } => {
            let mut irr = indirect;
            for (l, lc) in lights {
                let coord = dot(n, *l) * 0.5 + 0.5;
                if (coord - 0.7).abs() < 0.03 && lc.iter().any(|&x| x > 1e-3) {
                    return None;
                }
                let g = if coord < 0.7 { 0.7 } else { 1.0 };
                irr = add(irr, mul(*lc, g));
            }
            mul(hadamard(irr, color), inv_pi)
        }
        Model::Standard { color, roughness, metalness } => {
            let diffuse = mul(color, 1.0 - metalness);
            let f0 = add(mul([0.04; 3], 1.0 - metalness), mul(color, metalness));
            let r = roughness.clamp(0.0525, 1.0);
            let alpha = r * r;
            let a2 = alpha * alpha;
            let mut out = mul(hadamard(indirect, diffuse), inv_pi);
            for (l, lc) in lights {
                let ndl = saturate(dot(n, *l));
                let e = mul(*lc, ndl);
                let h = norm(add(*l, v));
                let ndv = saturate(dot(n, v));
                let ndh = saturate(dot(n, h));
                let vdh = saturate(dot(v, h));
                let f = f_schlick(f0, 1.0, vdh);
                let gv = ndl * (a2 + (1.0 - a2) * ndv * ndv).sqrt();
                let gl = ndv * (a2 + (1.0 - a2) * ndl * ndl).sqrt();
                let vis = 0.5 / (gv + gl).max(1e-6);
                let denom = ndh * ndh * (a2 - 1.0) + 1.0;
                let d = inv_pi * a2 / (denom * denom);
                let brdf = add(mul(f, vis * d), mul(diffuse, inv_pi));
                out = add(out, hadamard(e, brdf));
            }
            out
        }
        Model::Normal => {
            // View space is world space translated: the camera is unrotated.
            add(mul(n, 0.5), [0.5; 3])
        }
        Model::Matcap { color } => {
            let view_dir = v;
            let x = norm([view_dir[2], 0.0, -view_dir[0]]);
            let y = cross(view_dir, x);
            let uv_y = dot(y, n) * 0.495 + 0.5;
            let m = 0.2 + (0.8 - 0.2) * uv_y;
            mul(color, m)
        }
    })
}

/// three.js's `sRGBTransferOETF` on a clamped value, to a byte.
pub fn encode(c: V3) -> [u8; 3] {
    let e = |x: f64| {
        let x = saturate(x);
        let s = if x <= 0.0031308 {
            x * 12.92
        } else {
            1.055 * x.powf(1.0 / 2.4) - 0.055
        };
        (s * 255.0).round() as u8
    };
    [e(c[0]), e(c[1]), e(c[2])]
}

/// The packed normal written straight to the bytes (no transfer curve).
pub fn raw(c: V3) -> [u8; 3] {
    let e = |x: f64| (saturate(x) * 255.0).round() as u8;
    [e(c[0]), e(c[1]), e(c[2])]
}

/// The bytes three.js writes for `model` at pixel `(x, y)`, or `None` where
/// the pixel is off the sphere, at its silhouette, or at a derivative-
/// dependent ramp step.
pub fn expected(model: Model, x: u32, y: u32) -> Option<[u8; 3]> {
    let (p, n, v) = hit(x, y)?;
    let c = shade(model, p, n, v)?;
    Some(match model {
        Model::Normal => raw(c),
        _ => encode(c),
    })
}

/// Compare a rendered `N × N` RGBA8 frame with three.js's at every sphere
/// pixel; returns (pixels compared, worst channel error, where).
pub fn compare(model: Model, rgba: &[u8]) -> (usize, i32, (u32, u32), [u8; 3], [u8; 3]) {
    let mut worst = (0usize, -1i32, (0u32, 0u32), [0u8; 3], [0u8; 3]);
    for y in 0..N {
        for x in 0..N {
            let Some(want) = expected(model, x, y) else {
                continue;
            };
            let i = ((y * N + x) * 4) as usize;
            let got = [rgba[i], rgba[i + 1], rgba[i + 2]];
            let err = (0..3)
                .map(|k| (got[k] as i32 - want[k] as i32).abs())
                .max()
                .unwrap();
            worst.0 += 1;
            if err > worst.1 {
                worst = (worst.0, err, (x, y), got, want);
            }
        }
    }
    worst
}

/// The materials of the comparison, threers side, with the model three.js
/// evaluates for them.
pub fn cases() -> Vec<(&'static str, Material, Model)> {
    use threers::materials::{
        BasicMaterial, LambertMaterial, MatcapMaterial, NormalMaterial, PhongMaterial,
        PhysicalMaterial, StandardMaterial, ToonMaterial,
    };
    let albedo: V3 = [0.8, 0.45, 0.2];
    let mut out: Vec<(&'static str, Material, Model)> = Vec::new();
    out.push(("basic", BasicMaterial::new(c(albedo)).into(), Model::Basic { color: albedo }));
    out.push(("lambert", LambertMaterial::new(c(albedo)).into(), Model::Lambert { color: albedo }));
    let spec: V3 = [0.5, 0.5, 0.5];
    out.push((
        "phong",
        PhongMaterial::new(c(albedo)).with_specular(c(spec), 20.0).into(),
        Model::Phong { color: albedo, specular: spec, shininess: 20.0 },
    ));
    out.push(("toon", ToonMaterial::new(c(albedo)).into(), Model::Toon { color: albedo }));
    let mut st = StandardMaterial::new(c(albedo));
    st.roughness = 1.0;
    st.metalness = 0.0;
    out.push(("standard", st.into(), Model::Standard { color: albedo, roughness: 1.0, metalness: 0.0 }));
    let mut st = StandardMaterial::new(c(albedo));
    st.roughness = 1.0;
    st.metalness = 0.6;
    out.push((
        "standard-metal",
        st.into(),
        Model::Standard { color: albedo, roughness: 1.0, metalness: 0.6 },
    ));
    let mut ph = PhysicalMaterial::new(c(albedo));
    ph.roughness = 1.0;
    ph.metalness = 0.3;
    out.push((
        "physical",
        ph.into(),
        Model::Standard { color: albedo, roughness: 1.0, metalness: 0.3 },
    ));
    out.push(("normal", NormalMaterial::new().into(), Model::Normal));
    let mc = MatcapMaterial {
        color: c(albedo),
        ..Default::default()
    };
    out.push(("matcap", mc.into(), Model::Matcap { color: albedo }));
    out
}

// ---------------------------------------------------------------- shadows

/// Which light casts in [`shadow_scene`].
#[derive(Clone, Copy, Debug)]
pub enum Caster {
    Directional,
    Spot,
    Point,
}

pub const SHADOW_CAM_Z: f64 = 5.0;
pub const SHADOW_FOV_DEG: f64 = 40.0;
pub const SHADOW_AMBIENT: V3 = [0.15, 0.15, 0.15];
const SHADOW_DIR_TRAVEL: V3 = [-0.3, -0.2, -1.0];
const SHADOW_DIR_COLOR: V3 = [1.2, 1.2, 1.2];
const SHADOW_LAMP_POS: V3 = [0.9, 0.6, 4.5];
const SHADOW_LAMP_COLOR: V3 = [25.0, 25.0, 25.0];
/// Floor points: one deep in the caster's shadow (for every caster) and in
/// view past it, one lit by every caster.
pub const IN_SHADOW: V3 = [-0.6, -0.5, 0.0];
pub const IN_LIGHT: V3 = [0.6, 0.5, 0.0];

pub fn shadow_camera() -> PerspectiveCamera {
    let mut cam = PerspectiveCamera::new(SHADOW_FOV_DEG as f32, 1.0, 0.1, 100.0);
    cam.position = Vector3::new(0.0, 0.0, SHADOW_CAM_Z as f32);
    cam.look_at(Vector3::ZERO);
    cam
}

/// A floor (the plane z = 0, facing the camera) in `material`, receiving
/// shadows, under an ambient light and one shadow-casting `caster`, with a
/// small box at z = 1.5 casting onto it.
pub fn shadow_scene(material: Material, caster: Caster) -> Scene {
    use threers::geometries::{BoxGeometry, PlaneGeometry};
    use threers::materials::LambertMaterial;
    let mut scene = Scene::new();
    scene.background = Color::new(0.5, 0.5, 0.5);
    scene.add(Object3D::light(Light::Ambient(AmbientLight::new(c(SHADOW_AMBIENT), 1.0))));
    match caster {
        Caster::Directional => {
            let mut l = DirectionalLight::new(Color::WHITE, SHADOW_DIR_COLOR[0] as f32)
                .with_direction(v3(SHADOW_DIR_TRAVEL));
            l.cast_shadow = true;
            l.shadow.camera_size = 6.0;
            scene.add(Object3D::light(Light::Directional(l)));
        }
        Caster::Spot => {
            let mut l = SpotLight::new(Color::WHITE, SHADOW_LAMP_COLOR[0] as f32);
            l.angle = 0.5;
            l.penumbra = 0.0;
            l.decay = 2.0;
            l.distance = 0.0;
            l.direction = v3(norm(mul(SHADOW_LAMP_POS, -1.0)));
            l.cast_shadow = true;
            l.shadow.camera_near = 0.5;
            l.shadow.camera_far = 10.0;
            let mut o = Object3D::light(Light::Spot(l));
            o.position = v3(SHADOW_LAMP_POS);
            scene.add(o);
        }
        Caster::Point => {
            let mut l = PointLight::new(Color::WHITE, SHADOW_LAMP_COLOR[0] as f32);
            l.decay = 2.0;
            l.distance = 0.0;
            l.cast_shadow = true;
            l.shadow.camera_near = 0.5;
            l.shadow.camera_far = 10.0;
            let mut o = Object3D::light(Light::Point(l));
            o.position = v3(SHADOW_LAMP_POS);
            scene.add(o);
        }
    }
    let mut floor = Object3D::mesh(Mesh::new(PlaneGeometry::new(4.0, 4.0), material));
    floor.receive_shadow = true;
    scene.add(floor);
    let mut b = Object3D::mesh(Mesh::new(
        BoxGeometry::new(0.6, 0.6, 0.6),
        LambertMaterial::new(Color::new(0.2, 0.9, 0.2)).into(),
    ));
    b.position = Vector3::new(0.0, 0.0, 1.5);
    b.cast_shadow = true;
    scene.add(b);
    scene
}

/// The pixel nearest floor point `target`, and the floor point its centre's
/// ray actually meets.
pub fn floor_pixel(target: V3) -> ((u32, u32), V3) {
    let t = (SHADOW_FOV_DEG.to_radians() / 2.0).tan();
    let to_px = |w: f64| (w / (SHADOW_CAM_Z * t) + 1.0) / 2.0 * N as f64;
    let x = to_px(target[0]).floor() as u32;
    let y = (N as f64 - to_px(target[1])).floor() as u32;
    let ndc_x = (x as f64 + 0.5) / N as f64 * 2.0 - 1.0;
    let ndc_y = 1.0 - (y as f64 + 0.5) / N as f64 * 2.0;
    let p = [ndc_x * t * SHADOW_CAM_Z, ndc_y * t * SHADOW_CAM_Z, 0.0];
    ((x, y), p)
}

/// three.js's colour at floor point `p` for `model`, with the caster's light
/// shadowed or not.
pub fn shadow_expected(model: Model, caster: Caster, p: V3, shadowed: bool) -> [u8; 3] {
    let n = [0.0, 0.0, 1.0];
    let v = norm(sub([0.0, 0.0, SHADOW_CAM_Z], p));
    let vis = if shadowed { 0.0 } else { 1.0 };
    let light = match caster {
        Caster::Directional => (norm(mul(SHADOW_DIR_TRAVEL, -1.0)), mul(SHADOW_DIR_COLOR, vis)),
        Caster::Spot | Caster::Point => {
            let lv = sub(SHADOW_LAMP_POS, p);
            let d2 = dot(lv, lv);
            (norm(lv), mul(SHADOW_LAMP_COLOR, vis / d2))
        }
    };
    encode(shade_with(model, n, v, &[light], SHADOW_AMBIENT).expect("no ramp step on a flat floor"))
}

/// Every material's shadow receipt under every caster: the floor point in
/// the box's shadow and the lit one, against three.js. Lambert, Phong, Toon,
/// Standard and Physical receive (the caster's light goes to zero in the
/// shadow); Basic does not (three.js's MeshBasicMaterial reads no lights);
/// ShadowMaterial draws its colour at `opacity · (1 − shadow)` — black in the
/// shadow, the background (mid grey) outside it. Returns the failures.
pub fn check_shadows(
    backend: &str,
    render: &mut dyn FnMut(&mut Scene, &PerspectiveCamera) -> Vec<u8>,
) -> Vec<String> {
    use threers::materials::{
        BasicMaterial, LambertMaterial, PhongMaterial, PhysicalMaterial, StandardMaterial,
        ToonMaterial,
    };
    let albedo: V3 = [0.8, 0.45, 0.2];
    let spec: V3 = [0.5, 0.5, 0.5];
    let mut st = StandardMaterial::new(c(albedo));
    st.roughness = 1.0;
    let mut ph = PhysicalMaterial::new(c(albedo));
    ph.roughness = 1.0;
    ph.metalness = 0.3;
    let mut shadow_mat = BasicMaterial::new(Color::BLACK);
    shadow_mat.shadow_only = true;
    shadow_mat.transparent = true;
    // (name, material, model; `None` = ShadowMaterial)
    let cases: Vec<(&str, Material, Option<Model>)> = vec![
        ("basic", BasicMaterial::new(c(albedo)).into(), Some(Model::Basic { color: albedo })),
        ("lambert", LambertMaterial::new(c(albedo)).into(), Some(Model::Lambert { color: albedo })),
        (
            "phong",
            PhongMaterial::new(c(albedo)).with_specular(c(spec), 20.0).into(),
            Some(Model::Phong { color: albedo, specular: spec, shininess: 20.0 }),
        ),
        ("toon", ToonMaterial::new(c(albedo)).into(), Some(Model::Toon { color: albedo })),
        (
            "standard",
            st.into(),
            Some(Model::Standard { color: albedo, roughness: 1.0, metalness: 0.0 }),
        ),
        (
            "physical",
            ph.into(),
            Some(Model::Standard { color: albedo, roughness: 1.0, metalness: 0.3 }),
        ),
        ("shadow", shadow_mat.into(), None),
    ];
    let cam = shadow_camera();
    let mut failures = Vec::new();
    for caster in [Caster::Directional, Caster::Spot, Caster::Point] {
        for (name, material, model) in &cases {
            let rgba = render(&mut shadow_scene(material.clone(), caster), &cam);
            for (target, shadowed) in [(IN_SHADOW, true), (IN_LIGHT, false)] {
                let ((x, y), p) = floor_pixel(target);
                let i = ((y * N + x) * 4) as usize;
                let got = [rgba[i], rgba[i + 1], rgba[i + 2]];
                let want = match model {
                    // Basic reads no light at all, shadowed or not.
                    Some(m @ Model::Basic { .. }) => shadow_expected(*m, caster, p, false),
                    Some(m) => shadow_expected(*m, caster, p, shadowed),
                    None if shadowed => [0, 0, 0],
                    None => encode([0.5; 3]),
                };
                let err = (0..3).map(|k| (got[k] as i32 - want[k] as i32).abs()).max().unwrap();
                eprintln!(
                    "{backend} {caster:?} {name:>9} {}: got {got:?} want {want:?}",
                    if shadowed { "shadow" } else { "lit   " }
                );
                if err > 2 {
                    failures.push(format!(
                        "{caster:?} {name} {}: got {got:?}, three.js {want:?}",
                        if shadowed { "in shadow" } else { "lit" }
                    ));
                }
            }
        }
    }
    failures
}

// ------------------------------------------------------- instanced shadows

const INST_DIR_TRAVEL: V3 = [-1.0, 0.0, -1.0];
const INST_LAMP_POS: V3 = [-3.0, 0.2, 3.0];
const INST_BOX: f64 = 0.3;
const INST_Z: f64 = 0.6;
/// The two instances' centres, and the morph target's shift (an absolute
/// target: every position moved this far in x), which only instance 1 takes.
const INST_CENTRES: [V3; 2] = [[0.2, 0.6, INST_Z], [0.0, -0.7, INST_Z]];
const INST_MORPH_SHIFT: f64 = -0.8;

/// Where the shadow of a caster point `p` lands on the floor (z = 0): along
/// the directional light's travel, or along the ray from the lamp.
fn shadow_on_floor(caster: Caster, p: V3) -> V3 {
    match caster {
        Caster::Directional => {
            let t = p[2] / -INST_DIR_TRAVEL[2];
            add(p, mul(INST_DIR_TRAVEL, t))
        }
        Caster::Spot | Caster::Point => {
            let k = INST_LAMP_POS[2] / (INST_LAMP_POS[2] - p[2]);
            add(INST_LAMP_POS, mul(sub(p, INST_LAMP_POS), k))
        }
    }
}

/// A Lambert floor receiving the shadows of an `InstancedMesh` of two small
/// boxes — three.js draws an instanced mesh into the shadow map instance by
/// instance (`WebGLShadowMap` renders it with the instanced depth material) —
/// with per-instance morph targets (`setMorphAt`): instance 1 is drawn
/// shifted `INST_MORPH_SHIFT` in x by its morph, instance 0 not at all.
pub fn instanced_shadow_scene(caster: Caster) -> Scene {
    use std::sync::Arc;
    use threers::core::{BufferAttribute, InstancedMesh};
    use threers::geometries::{BoxGeometry, PlaneGeometry};
    use threers::materials::LambertMaterial;
    use threers::math::Matrix4;
    let mut scene = Scene::new();
    scene.background = Color::new(0.5, 0.5, 0.5);
    scene.add(Object3D::light(Light::Ambient(AmbientLight::new(c(SHADOW_AMBIENT), 1.0))));
    match caster {
        Caster::Directional => {
            let mut l = DirectionalLight::new(Color::WHITE, SHADOW_DIR_COLOR[0] as f32)
                .with_direction(v3(INST_DIR_TRAVEL));
            l.cast_shadow = true;
            l.shadow.camera_size = 6.0;
            scene.add(Object3D::light(Light::Directional(l)));
        }
        Caster::Spot => {
            let mut l = SpotLight::new(Color::WHITE, SHADOW_LAMP_COLOR[0] as f32);
            l.angle = 0.9;
            l.penumbra = 0.0;
            l.decay = 2.0;
            l.distance = 0.0;
            l.direction = v3(norm(mul(INST_LAMP_POS, -1.0)));
            l.cast_shadow = true;
            l.shadow.camera_near = 0.5;
            l.shadow.camera_far = 12.0;
            let mut o = Object3D::light(Light::Spot(l));
            o.position = v3(INST_LAMP_POS);
            scene.add(o);
        }
        Caster::Point => {
            let mut l = PointLight::new(Color::WHITE, SHADOW_LAMP_COLOR[0] as f32);
            l.decay = 2.0;
            l.distance = 0.0;
            l.cast_shadow = true;
            l.shadow.camera_near = 0.5;
            l.shadow.camera_far = 12.0;
            let mut o = Object3D::light(Light::Point(l));
            o.position = v3(INST_LAMP_POS);
            scene.add(o);
        }
    }
    let mut floor = Object3D::mesh(Mesh::new(
        PlaneGeometry::new(4.0, 4.0),
        threers::materials::LambertMaterial::new(c([0.8, 0.45, 0.2])).into(),
    ));
    floor.receive_shadow = true;
    scene.add(floor);
    let s = INST_BOX as f32;
    let mut g = BoxGeometry::new(s, s, s);
    let base = g.get_attribute("position").unwrap().array.clone();
    let shifted: Vec<f32> = base
        .chunks_exact(3)
        .flat_map(|p| [p[0] + INST_MORPH_SHIFT as f32, p[1], p[2]])
        .collect();
    g.set_morph_attribute("position", vec![BufferAttribute::new(shifted, 3)]);
    let mut im = InstancedMesh::from_arc(
        Arc::new(g),
        Arc::new(LambertMaterial::new(Color::new(0.2, 0.9, 0.2)).into()),
        INST_CENTRES.len(),
    );
    for (i, p) in INST_CENTRES.iter().enumerate() {
        im.set_matrix_at(i, Matrix4::translation(v3(*p)));
        im.set_morph_at(i, &[if i == 1 { 1.0 } else { 0.0 }]);
    }
    let mut o = Object3D::instanced_mesh(im);
    o.cast_shadow = true;
    scene.add(o);
    scene
}

/// An instanced caster's shadows land where three.js puts them, for every
/// light type: in shadow at the floor point under each instance's (morphed)
/// centre, lit where instance 1's centre would be without its morph. The
/// colours are three.js's Lambert floor, shadowed or not. Returns the
/// failures.
pub fn check_instanced_shadows(
    backend: &str,
    render: &mut dyn FnMut(&mut Scene, &PerspectiveCamera) -> Vec<u8>,
) -> Vec<String> {
    let cam = shadow_camera();
    let albedo: V3 = [0.8, 0.45, 0.2];
    let mut failures = Vec::new();
    for caster in [Caster::Directional, Caster::Spot, Caster::Point] {
        let rgba = render(&mut instanced_shadow_scene(caster), &cam);
        let morphed = add(INST_CENTRES[1], [INST_MORPH_SHIFT, 0.0, 0.0]);
        let probes = [
            ("instance 0", shadow_on_floor(caster, INST_CENTRES[0]), true),
            ("instance 1, morphed", shadow_on_floor(caster, morphed), true),
            ("instance 1 unmorphed", shadow_on_floor(caster, INST_CENTRES[1]), false),
        ];
        for (what, target, shadowed) in probes {
            let ((x, y), p) = floor_pixel(target);
            let i = ((y * N + x) * 4) as usize;
            let got = [rgba[i], rgba[i + 1], rgba[i + 2]];
            let n = [0.0, 0.0, 1.0];
            let v = norm(sub([0.0, 0.0, SHADOW_CAM_Z], p));
            let vis = if shadowed { 0.0 } else { 1.0 };
            let light = match caster {
                Caster::Directional => (norm(mul(INST_DIR_TRAVEL, -1.0)), mul(SHADOW_DIR_COLOR, vis)),
                Caster::Spot | Caster::Point => {
                    let lv = sub(INST_LAMP_POS, p);
                    (norm(lv), mul(SHADOW_LAMP_COLOR, vis / dot(lv, lv)))
                }
            };
            let want = encode(
                shade_with(Model::Lambert { color: albedo }, n, v, &[light], SHADOW_AMBIENT).unwrap(),
            );
            let err = (0..3).map(|k| (got[k] as i32 - want[k] as i32).abs()).max().unwrap();
            eprintln!(
                "{backend} instanced {caster:?} {what} at {target:?} px ({x}, {y}): got {got:?} want {want:?}"
            );
            if err > 2 {
                failures.push(format!(
                    "{caster:?} {what} ({}): got {got:?}, three.js {want:?}",
                    if shadowed { "in shadow" } else { "lit" }
                ));
            }
        }
    }
    failures
}
