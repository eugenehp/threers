//! Energy-conservation checks for the image-based-lighting path.
//!
//! These use a *uniform* environment, which turns the renderer into something
//! with an exact analytic answer:
//!
//! - A perfectly white Lambertian surface (`metalness = 0`, `roughness = 1`,
//!   albedo 1.0) sitting inside a uniform environment of radiance L must render
//!   at exactly L. It cannot be brighter than its surroundings — that would be
//!   reflecting more light than reaches it.
//! - A metal (`metalness = 1`) renders at L weighted by its F0 per channel, so
//!   the output channel ratios must reproduce the material's F0 ratios.
//!
//! Both previously failed: the environment's diffuse contribution was added
//! twice (once from the RE_IndirectSpecular port and once from the
//! RE_IndirectDiffuse port), giving a white surface a gain of exactly 2.0.

use std::sync::Arc;

use threers::{
    Color, CubeTexture, HeadlessRenderer, Material, Mesh, Object3D, PerspectiveCamera,
    PhysicalMaterial, Scene, SphereGeometry, TextureFormat, Vector3,
};

const W: u32 = 128;

fn lin_to_srgb_byte(l: f32) -> u8 {
    let s = if l <= 0.0031308 {
        l * 12.92
    } else {
        1.055 * l.powf(1.0 / 2.4) - 0.055
    };
    (s.clamp(0.0, 1.0) * 255.0).round() as u8
}

fn srgb_to_lin(b: f32) -> f32 {
    let s = b / 255.0;
    if s <= 0.04045 {
        s / 12.92
    } else {
        ((s + 0.055) / 1.055).powf(2.4)
    }
}

/// Mean over a disc well inside the sphere's silhouette.
fn mean_disc(rgba: &[u8]) -> (f32, f32, f32) {
    let (c, rad) = (W as f32 / 2.0, W as f32 * 0.30);
    let (mut n, mut a) = (0f32, [0f32; 3]);
    for y in 0..W {
        for x in 0..W {
            let (dx, dy) = (x as f32 - c, y as f32 - c);
            if dx * dx + dy * dy > rad * rad {
                continue;
            }
            let i = ((y * W + x) * 4) as usize;
            a[0] += rgba[i] as f32;
            a[1] += rgba[i + 1] as f32;
            a[2] += rgba[i + 2] as f32;
            n += 1.0;
        }
    }
    (a[0] / n, a[1] / n, a[2] / n)
}

fn uniform_env(level: f32) -> Arc<CubeTexture> {
    let byte = lin_to_srgb_byte(level);
    let sz = 64usize;
    let face: Vec<u8> = (0..sz * sz).flat_map(|_| [byte, byte, byte, 255]).collect();
    let faces: [Vec<u8>; 6] = std::array::from_fn(|_| face.clone());
    Arc::new(CubeTexture::new(
        sz as u32,
        TextureFormat::Rgba8UnormSrgb,
        faces,
    ))
}

/// Render `mat` in a uniform environment; returns the mean linear RGB.
///
/// The target is deliberately `Rgba8Unorm`, not `Rgba8UnormSrgb`: the mesh
/// shader performs its own linear→sRGB encode, so an sRGB target would encode a
/// second time and every reading here would be wrong by ~2.5–5×.
fn render_in_uniform_env(
    renderer: &mut HeadlessRenderer,
    env: &Arc<CubeTexture>,
    mat: PhysicalMaterial,
) -> (f32, f32, f32) {
    let mut scene = Scene::new();
    scene.environment = Some(env.clone());
    scene.add(Object3D::mesh(Mesh::new(
        SphereGeometry::new(1.0, 64, 32),
        Material::Physical(mat),
    )));
    let mut cam = PerspectiveCamera::new(32.0, 1.0, 0.1, 100.0);
    cam.position = Vector3::new(0.0, 0.0, 4.2);
    cam.look_at(Vector3::ZERO);
    scene.update_world();

    let out = renderer.render_to_rgba(&mut scene, &cam);
    let (r, g, b) = mean_disc(&out);
    (srgb_to_lin(r), srgb_to_lin(g), srgb_to_lin(b))
}

fn build_renderer() -> Option<HeadlessRenderer> {
    HeadlessRenderer::builder()
        .size(W, W)
        .color_format(wgpu::TextureFormat::Rgba8Unorm)
        .build()
        .ok()
}

#[test]
fn white_lambertian_cannot_outshine_a_uniform_environment() {
    let Some(mut renderer) = build_renderer() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    // Hold every environment alive for the whole test: the renderer caches the
    // uploaded cubemap by `Arc::as_ptr`, so a dropped Arc whose address gets
    // reused would silently keep the previous upload.
    let envs: Vec<(f32, Arc<CubeTexture>)> = [0.05f32, 0.10, 0.20]
        .iter()
        .map(|&l| (l, uniform_env(l)))
        .collect();

    for (level, env) in &envs {
        let mat = PhysicalMaterial::new(Color::new(1.0, 1.0, 1.0))
            .with_metalness(0.0)
            .with_roughness(1.0);
        let (r, _, _) = render_in_uniform_env(&mut renderer, env, mat);
        let gain = r / level;
        assert!(
            (gain - 1.0).abs() < 0.12,
            "white Lambertian in a uniform env of {level} rendered gain {gain:.3}; \
             expected ~1.0 (a gain of ~2.0 means the env diffuse is double-counted)"
        );
    }
}

#[test]
fn metal_reproduces_its_f0_ratios_under_ibl() {
    let Some(mut renderer) = build_renderer() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let env = uniform_env(0.10);
    // Gold: F0 = (1.000, 0.766, 0.336).
    let gold = PhysicalMaterial::new(Color::new(1.000, 0.766, 0.336))
        .with_metalness(1.0)
        .with_roughness(0.3);
    let (r, g, b) = render_in_uniform_env(&mut renderer, &env, gold);

    let gain = r / 0.10;
    assert!(
        (gain - 1.0).abs() < 0.15,
        "gold red-channel gain {gain:.3}, expected ~1.0 (F0.r = 1.0)"
    );
    // Fresnel lifts grazing angles toward white, so the measured ratios sit a
    // little above F0; they must still be far closer to gold than to neutral.
    let (gr, br) = (g / r, b / r);
    assert!(
        (0.66..=0.85).contains(&gr),
        "gold g/r = {gr:.3}, expected ~0.77 (neutral would be 1.0)"
    );
    assert!(
        (0.24..=0.45).contains(&br),
        "gold b/r = {br:.3}, expected ~0.34 (neutral would be 1.0)"
    );
}

/// A `RectAreaLight` must actually light the scene.
///
/// This previously rendered pure black: the renderer's light-gathering pass had
/// `Light::RectArea(_) => { /* TODO LTC */ }`, so the light was collected and
/// then silently dropped before ever reaching the shader.
#[test]
fn rect_area_light_illuminates() {
    let Some(mut renderer) = build_renderer() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut scene = Scene::new();
    scene.background = Color::new(0.0, 0.0, 0.0);

    let mut light = Object3D::light(threers::RectAreaLight::new(
        Color::new(1.0, 1.0, 1.0),
        5.0,
        4.0,
        4.0,
    ));
    // In front of the sphere, facing it down -Z.
    light.position = Vector3::new(0.0, 0.0, 3.0);
    scene.add(light);

    let mat = PhysicalMaterial::new(Color::new(0.8, 0.8, 0.8))
        .with_metalness(0.0)
        .with_roughness(0.5);
    scene.add(Object3D::mesh(Mesh::new(
        SphereGeometry::new(1.0, 64, 32),
        Material::Physical(mat),
    )));

    let mut cam = PerspectiveCamera::new(32.0, 1.0, 0.1, 100.0);
    cam.position = Vector3::new(0.0, 0.0, 4.2);
    cam.look_at(Vector3::ZERO);
    scene.update_world();

    let out = renderer.render_to_rgba(&mut scene, &cam);
    let (r, g, b) = mean_disc(&out);
    assert!(
        r > 8.0 && g > 8.0 && b > 8.0,
        "RectAreaLight produced almost no light: mean rgb = ({r:.1}, {g:.1}, {b:.1})"
    );
}
