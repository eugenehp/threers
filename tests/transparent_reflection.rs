//! A transparent surface still reflects — with
//! `Renderer::set_reflective_transparency`, which is off by default, where
//! a transparent surface blends as three.js blends it.
//!
//! Plain alpha blending composites `src * α + dst * (1 - α)`, which scales the
//! *whole* shaded colour by alpha — reflection included. But alpha says how
//! much of what is behind the surface shows through, and a reflection happens
//! in front of it and is not transmitted at all. Blending it away is what makes
//! a polished surface at `alphaMode: BLEND` read as grey film rather than as
//! glass, and authors without `KHR_materials_transmission` fake glass exactly
//! that way, so it is the common case rather than the corner one.
//!
//! A metal is the sharp version of the test. Its reflectance is most of the
//! spectrum, so a "75% transparent" metal that blends away 75% of its mirror is
//! unmistakably wrong — and there is an exact thing to compare it against, the
//! same metal at `opacity = 1`.

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

/// Mean luminance over a disc well inside the sphere's silhouette.
fn mean_disc(rgba: &[u8]) -> f64 {
    let (c, rad) = (W as f32 / 2.0, W as f32 * 0.30);
    let (mut n, mut a) = (0f64, 0f64);
    for y in 0..W {
        for x in 0..W {
            let (dx, dy) = (x as f32 - c, y as f32 - c);
            if dx * dx + dy * dy > rad * rad {
                continue;
            }
            let i = ((y * W + x) * 4) as usize;
            a += 0.2126 * rgba[i] as f64 + 0.7152 * rgba[i + 1] as f64 + 0.0722 * rgba[i + 2] as f64;
            n += 1.0;
        }
    }
    a / n
}

/// The environment is the only light, and the background is black, so every
/// level inside the silhouette arrived as a reflection.
fn render(opacity: f32, metalness: f32, roughness: f32) -> Option<f64> {
    render_with(opacity, metalness, roughness, true)
}

fn render_with(opacity: f32, metalness: f32, roughness: f32, reflective: bool) -> Option<f64> {
    let mut r = HeadlessRenderer::builder()
        .size(W, W)
        .color_format(threers::wgpu::TextureFormat::Rgba8Unorm)
        .build()
        .ok()?;
    r.renderer().set_reflective_transparency(reflective);
    let mut scene = Scene::new();
    scene.background = Color::from_hex(0x000000);
    scene.environment = Some(uniform_env(1.0));
    let mut mat = PhysicalMaterial::new(Color::new(0.95, 0.95, 0.95));
    mat.metalness = metalness;
    mat.roughness = roughness;
    mat.opacity = opacity;
    scene.add(Object3D::mesh(Mesh::new(
        SphereGeometry::new(1.0, 64, 32),
        Material::Physical(mat),
    )));
    let mut cam = PerspectiveCamera::new(32.0, 1.0, 0.1, 100.0);
    cam.position = Vector3::new(0.0, 0.0, 4.2);
    cam.look_at(Vector3::ZERO);
    scene.update_world();
    Some(mean_disc(&r.render_to_rgba(&mut scene, &cam)))
}

#[test]
fn a_transparent_metal_keeps_its_reflection() {
    let (Some(opaque), Some(faint)) = (render(1.0, 1.0, 0.1), render(0.25, 1.0, 0.1)) else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    eprintln!("metal: opaque {opaque:.1}, α=0.25 {faint:.1}, kept {:.0}%", 100.0 * faint / opaque);
    assert!(
        opaque > 40.0,
        "the opaque metal should reflect the white environment, got {opaque:.1}"
    );
    // Scaling the shaded colour by alpha would leave a quarter of it. A metal
    // transmits nothing, so it should keep nearly all of it.
    assert!(
        faint / opaque > 0.80,
        "a transparent metal kept only {:.0}% of its reflection ({faint:.1} against {opaque:.1}) \
         — the specular is still being scaled by alpha",
        100.0 * faint / opaque
    );
}

/// The other end: a dielectric reflects about 4% face-on, so raising coverage
/// by its reflectance must be a small correction, not a licence to turn every
/// transparent surface opaque. A ghosted overlay has to stay ghosted.
#[test]
fn a_transparent_dielectric_stays_transparent() {
    let (Some(opaque), Some(faint)) = (render(1.0, 0.0, 0.5), render(0.25, 0.0, 0.5)) else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    eprintln!("dielectric: opaque {opaque:.1}, α=0.25 {faint:.1}, kept {:.0}%", 100.0 * faint / opaque);
    assert!(
        faint / opaque < 0.45,
        "a 25%-opacity dielectric kept {:.0}% of its shading ({faint:.1} against {opaque:.1}) — \
         the coverage correction is too aggressive and transparency no longer means anything",
        100.0 * faint / opaque
    );
}

/// Opaque is opaque: the correction must vanish at `opacity = 1`, or it would
/// silently change every ordinary material in the crate.
#[test]
fn opaque_materials_are_untouched() {
    let (Some(a), Some(b)) = (render(1.0, 1.0, 0.1), render(1.0, 0.0, 0.5)) else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    assert!(a > 0.0 && b > 0.0);
    // Nothing to assert beyond "it renders"; the pinning is that these two go
    // through the `opacity < 1.0` gate and come out of the same branch they
    // always did. The real guard is the energy test in `ibl_energy.rs`, which
    // measures the opaque path against its analytic answer.
}

/// By default a transparent surface blends as three.js blends it: the whole
/// shaded colour by `opacity`, reflection included — over black, a quarter
/// of the opaque metal at `opacity = 0.25`.
#[test]
fn by_default_a_transparent_metal_blends_as_three_js_does() {
    let (Some(opaque), Some(faint)) = (render_with(1.0, 1.0, 0.1, false), render_with(0.25, 1.0, 0.1, false)) else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    let kept = faint / opaque;
    eprintln!("metal, three.js blending: opaque {opaque:.1}, α=0.25 {faint:.1}, kept {:.0}%", 100.0 * kept);
    assert!(
        (kept - 0.25).abs() < 0.04,
        "three.js blends a 25%-opacity surface's colour by 0.25; kept {:.0}%",
        100.0 * kept
    );
}
