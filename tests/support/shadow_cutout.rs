//! Cut-out shadows: three.js's shadow depth material takes a caster's `map` and
//! `alphaTest` (`WebGLShadowMap.getDepthMaterial`), so a quad whose map is
//! transparent on one half casts the shadow of the other half only.
//!
//! A horizontal 1×1 quad hovers at height 1 over a Lambert floor, its map's
//! left half alpha 0. A light shines past it at an angle so its shadow lands
//! beside it, where a camera looking straight down sees the floor. For each
//! light kind the floor is probed under the shadow of the opaque half (dark)
//! and of the cut half (lit), and again with `alphaTest` 0, where both halves
//! cast and both probes are dark — so the lit probe is known to be one the
//! shadow would otherwise cover. The depth material takes the caster's
//! `alphaMap` too, so an `alphaMap` (green 0 on the left half) with no `map`
//! cuts the same shadow.
//!
//! Shared by `tests/alpha_test.rs` (wgpu) and `tests/metal_backend.rs`.
#![allow(dead_code)]

use std::sync::Arc;
use threers::cameras::{Camera, OrthographicCamera};
use threers::core::{Mesh, Object3D};
use threers::geometries::PlaneGeometry;
use threers::lights::{AmbientLight, DirectionalLight, Light, PointLight, SpotLight};
use threers::materials::{BasicMaterial, LambertMaterial, Material};
use threers::math::{Color, Euler, Vector3};
use threers::scene::Scene;
use threers::textures::{Texture, TextureFilter, TextureFormat};

pub const N: u32 = 128;
const HALF_VIEW: f32 = 2.0;

#[derive(Clone, Copy, Debug)]
pub enum Kind {
    Directional,
    Spot,
    Point,
}

pub fn camera() -> OrthographicCamera {
    let mut cam = OrthographicCamera::new(-HALF_VIEW, HALF_VIEW, HALF_VIEW, -HALF_VIEW, 0.1, 20.0);
    cam.position = Vector3::new(0.0, 6.0, 0.0);
    cam.up = Vector3::new(0.0, 0.0, -1.0);
    cam.look_at(Vector3::ZERO);
    cam
}

fn half_cut() -> Arc<Texture> {
    let mut t = Texture::new(2, 1, TextureFormat::Rgba8Unorm, vec![255, 255, 255, 0, 255, 255, 255, 255]);
    t.mag_filter = TextureFilter::Nearest;
    t.min_filter = TextureFilter::Nearest;
    Arc::new(t)
}

/// Left half green 0, right half green 255 — an `alphaMap`'s channel.
fn green_cut() -> Arc<Texture> {
    let mut t = Texture::new(2, 1, TextureFormat::Rgba8Unorm, vec![255, 0, 255, 255, 255, 255, 255, 255]);
    t.mag_filter = TextureFilter::Nearest;
    t.min_filter = TextureFilter::Nearest;
    Arc::new(t)
}

/// What cuts the caster: its `map`'s alpha or its `alphaMap`'s green.
#[derive(Clone, Copy)]
pub enum Mask {
    Map,
    AlphaMap,
}

/// Where a lamp (spot or point) sits: up and to the left, so a caster point at
/// height 1 throws its shadow to floor x = 2·x + 1 (z unchanged).
const LAMP: [f32; 3] = [-1.0, 2.0, 0.0];

pub fn scene(kind: Kind, alpha_test: f32) -> Scene {
    scene_sided(kind, alpha_test, 2)
}

/// The scene with the caster's material `side` chosen: three.js's shadow pass
/// records a front-sided caster's back faces only (`shadowSide`), so a
/// front-sided plane facing the light casts nothing — the double-sided one does.
pub fn scene_sided(kind: Kind, alpha_test: f32, side: u32) -> Scene {
    scene_full(kind, alpha_test, side, true)
}

/// `cast: false` is the reference: the same light and floor with the caster
/// casting nothing, so a probe can be compared with its own unshadowed value
/// (a spot's or point's falloff makes probes differ from one another).
pub fn scene_full(kind: Kind, alpha_test: f32, side: u32, cast: bool) -> Scene {
    scene_with(kind, alpha_test, side, cast, false)
}

/// `transparent`: the caster blends at opacity 0.4. three.js's shadow map
/// ignores that — a transparent caster casts its whole shadow.
pub fn scene_with(kind: Kind, alpha_test: f32, side: u32, cast: bool, transparent: bool) -> Scene {
    scene_masked(kind, alpha_test, side, cast, transparent, Mask::Map)
}

pub fn scene_masked(kind: Kind, alpha_test: f32, side: u32, cast: bool, transparent: bool, mask: Mask) -> Scene {
    let mut scene = Scene::new();
    scene.background = Color::from_hex(0x000000);
    scene.add_light(AmbientLight::new(Color::WHITE, 0.6));
    match kind {
        Kind::Directional => {
            // Travelling (1, −1, 0): a caster point at height 1 shadows floor x + 1.
            let mut l = DirectionalLight::new(Color::WHITE, 2.5).with_direction(Vector3::new(1.0, -1.0, 0.0));
            l.cast_shadow = true;
            l.shadow.camera_size = 6.0;
            scene.add(Object3D::light(Light::Directional(l)));
        }
        Kind::Spot => {
            let mut l = SpotLight::new(Color::WHITE, 12.0);
            l.angle = 1.2;
            l.penumbra = 0.0;
            l.decay = 2.0;
            l.distance = 0.0;
            l.direction = Vector3::new(2.0, -2.0, 0.0).normalize();
            l.cast_shadow = true;
            l.shadow.camera_near = 0.3;
            l.shadow.camera_far = 10.0;
            let mut o = Object3D::light(Light::Spot(l));
            o.position = Vector3::new(LAMP[0], LAMP[1], LAMP[2]);
            scene.add(o);
        }
        Kind::Point => {
            let mut l = PointLight::new(Color::WHITE, 12.0);
            l.decay = 2.0;
            l.distance = 0.0;
            l.cast_shadow = true;
            l.shadow.camera_near = 0.3;
            l.shadow.camera_far = 10.0;
            let mut o = Object3D::light(Light::Point(l));
            o.position = Vector3::new(LAMP[0], LAMP[1], LAMP[2]);
            scene.add(o);
        }
    }
    let mut floor = Object3D::mesh(Mesh::new(
        PlaneGeometry::new(2.0 * HALF_VIEW, 2.0 * HALF_VIEW),
        LambertMaterial::new(Color::WHITE).into(),
    ));
    floor.set_rotation(Euler::new(-std::f32::consts::FRAC_PI_2, 0.0, 0.0));
    floor.receive_shadow = true;
    scene.add(floor);

    let mut caster_mat = BasicMaterial::new(Color::from_hex(0x00ff00));
    match mask {
        Mask::Map => caster_mat.map = Some(half_cut()),
        Mask::AlphaMap => caster_mat.alpha_map = Some(green_cut()),
    }
    caster_mat.side = side;
    if transparent {
        caster_mat.opacity = 0.4;
        caster_mat.transparent = true;
    }
    let mut caster_mat: Material = caster_mat.into();
    caster_mat.set_alpha_test(alpha_test);
    let mut caster = Object3D::mesh(Mesh::new(PlaneGeometry::new(1.0, 1.0), caster_mat));
    caster.set_rotation(Euler::new(-std::f32::consts::FRAC_PI_2, 0.0, 0.0));
    caster.position = Vector3::new(0.0, 1.0, 0.0);
    caster.cast_shadow = cast;
    scene.add(caster);
    scene
}

/// The floor x a caster point at (x, 1, 0) throws its shadow to.
fn shadow_x(kind: Kind, x: f32) -> f32 {
    match kind {
        Kind::Directional => x + 1.0,
        Kind::Spot | Kind::Point => 2.0 * x + 1.0,
    }
}

fn floor_px(rgba: &[u8], wx: f32, wz: f32) -> [u8; 4] {
    let px = (((wx + HALF_VIEW) / (2.0 * HALF_VIEW)) * N as f32) as u32;
    let py = (((wz + HALF_VIEW) / (2.0 * HALF_VIEW)) * N as f32) as u32;
    let i = ((py.min(N - 1) * N + px.min(N - 1)) * 4) as usize;
    [rgba[i], rgba[i + 1], rgba[i + 2], rgba[i + 3]]
}

/// Every failure, as a line; empty when each light kind casts the cut-out.
pub fn check(label: &str, render: &mut dyn FnMut(&mut Scene, &dyn Camera) -> Vec<u8>) -> Vec<String> {
    let cam = camera();
    let mut failures = Vec::new();
    for kind in [Kind::Directional, Kind::Spot, Kind::Point] {
        // Caster points well inside each half: x = +0.25 (opaque), −0.25 (cut).
        let (dark_x, cut_x) = (shadow_x(kind, 0.25), shadow_x(kind, -0.25));
        let mut probe = |s: &mut Scene| {
            let img = render(s, &cam);
            (floor_px(&img, dark_x, 0.0)[0] as i32, floor_px(&img, cut_x, 0.0)[0] as i32)
        };
        let shadowed = |v: i32, lit: i32| v + 20 < lit;
        let lit = |v: i32, lit: i32| (v - lit).abs() <= 3;
        let (ref_dark, ref_cut) = probe(&mut scene_full(kind, 0.0, 2, false));
        let (cut_dark, cut_cut) = probe(&mut scene(kind, 0.5));
        let (whole_dark, whole_cut) = probe(&mut scene(kind, 0.0));
        let (front_dark, _) = probe(&mut scene_sided(kind, 0.0, 0));
        let (glass_dark, glass_cut) = probe(&mut scene_with(kind, 0.0, 2, true, true));
        if !shadowed(glass_dark, ref_dark) || !shadowed(glass_cut, ref_cut) {
            failures.push(format!(
                "{label} {kind:?}: a transparent caster cast no shadow ({glass_dark}, {glass_cut}) vs lit ({ref_dark}, {ref_cut}); \
                 three.js's shadow map ignores transparency"
            ));
        }
        if !shadowed(whole_dark, ref_dark) || !shadowed(whole_cut, ref_cut) {
            failures.push(format!(
                "{label} {kind:?}: without alphaTest both halves should shadow: ({whole_dark}, {whole_cut}) vs lit ({ref_dark}, {ref_cut})"
            ));
        }
        if !shadowed(cut_dark, ref_dark) {
            failures.push(format!("{label} {kind:?}: the opaque half cast no shadow with alphaTest ({cut_dark} vs lit {ref_dark})"));
        }
        if !lit(cut_cut, ref_cut) {
            failures.push(format!("{label} {kind:?}: the cut half still casts ({cut_cut} vs lit {ref_cut})"));
        }
        let (am_dark, am_cut) = probe(&mut scene_masked(kind, 0.5, 2, true, false, Mask::AlphaMap));
        if !shadowed(am_dark, ref_dark) || !lit(am_cut, ref_cut) {
            failures.push(format!(
                "{label} {kind:?}: an alphaMap caster should shadow its green half only: ({am_dark}, {am_cut}) vs lit ({ref_dark}, {ref_cut})"
            ));
        }
        if !lit(front_dark, ref_dark) {
            failures.push(format!(
                "{label} {kind:?}: a front-sided plane facing the light cast a shadow ({front_dark} vs lit {ref_dark}); \
                 three.js's shadowSide records only its back faces"
            ));
        }
    }
    failures
}
