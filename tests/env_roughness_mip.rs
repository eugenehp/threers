//! A rough surface has to see a blurrier environment than a smooth one.
//!
//! When the environment is a plain cube with no PMREM atlas, the shader walks
//! the cube's own mip chain. That mapping was `(roughness + 2) / 10 * 7`, which
//! puts the whole of `0..1` on mips 1.40 to 2.10 — seven tenths of one level.
//! Every material from mirror to chalk came back the same blur, and a renderer
//! that ignores roughness in its reflections has no way to tell metal from
//! plastic.

use std::sync::Arc;

use threers::{
    Color, CubeTexture, HeadlessRenderer, Material, Mesh, Object3D, PerspectiveCamera,
    PhysicalMaterial, Scene, SphereGeometry, TextureFormat, Vector3,
};

const W: u32 = 96;

/// A cube with a hard black-and-white checker on every face — all its energy in
/// the high frequencies, so blurring it is unmistakable.
fn checker_env() -> Arc<CubeTexture> {
    let sz = 64usize;
    let face: Vec<u8> = (0..sz * sz)
        .flat_map(|i| {
            let (x, y) = (i % sz, i / sz);
            let on = ((x / 4) + (y / 4)) % 2 == 0;
            let v = if on { 255u8 } else { 0 };
            [v, v, v, 255]
        })
        .collect();
    let faces: [Vec<u8>; 6] = std::array::from_fn(|_| face.clone());
    Arc::new(CubeTexture::new(
        sz as u32,
        TextureFormat::Rgba8UnormSrgb,
        faces,
    ))
}

/// Contrast across a mirror sphere's reflection. A sharp environment gives a
/// high spread; a blurred one averages toward grey.
fn reflected_contrast(roughness: f32) -> Option<f64> {
    let mut r = HeadlessRenderer::builder()
        .size(W, W)
        .color_format(threers::wgpu::TextureFormat::Rgba8Unorm)
        .build()
        .ok()?;
    let mut scene = Scene::new();
    scene.background = Color::from_hex(0x000000);
    scene.environment = Some(checker_env());
    let mut mat = PhysicalMaterial::new(Color::new(0.95, 0.95, 0.95));
    mat.metalness = 1.0;
    mat.roughness = roughness;
    scene.add(Object3D::mesh(Mesh::new(
        SphereGeometry::new(1.0, 64, 32),
        Material::Physical(mat),
    )));
    let mut cam = PerspectiveCamera::new(32.0, 1.0, 0.1, 100.0);
    cam.position = Vector3::new(0.0, 0.0, 4.2);
    cam.look_at(Vector3::ZERO);
    scene.update_world();
    let px = r.render_to_rgba(&mut scene, &cam);

    // Standard deviation over a disc well inside the silhouette.
    let (c, rad) = (W as f32 / 2.0, W as f32 * 0.28);
    let mut v = Vec::new();
    for y in 0..W {
        for x in 0..W {
            let (dx, dy) = (x as f32 - c, y as f32 - c);
            if dx * dx + dy * dy > rad * rad {
                continue;
            }
            let i = ((y * W + x) * 4) as usize;
            v.push(0.2126 * px[i] as f64 + 0.7152 * px[i + 1] as f64 + 0.0722 * px[i + 2] as f64);
        }
    }
    let m = v.iter().sum::<f64>() / v.len() as f64;
    Some((v.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / v.len() as f64).sqrt())
}

#[test]
fn roughness_blurs_a_plain_cube_environment() {
    let Some(sharp) = reflected_contrast(0.02) else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    // Partly blurred: the 4-pixel checker on a 64-pixel face is grey by
    // roughness 0.25 (contrast 0.75), so a middle sample above that compares
    // two flat greys and measures noise — 0.28 against 0.34 on NVIDIA, 0.31
    // against 0.23 on Apple, for 0.35 and 0.9.
    let mid = reflected_contrast(0.15).unwrap();
    let rough = reflected_contrast(0.9).unwrap();
    eprintln!("reflected contrast: r=0.02 {sharp:.2}, r=0.15 {mid:.2}, r=0.9 {rough:.2}");

    assert!(
        sharp > 8.0,
        "a mirror should reflect the checker sharply, got {sharp:.2}"
    );
    assert!(
        rough < sharp * 0.6,
        "a rough metal reflected the environment at {rough:.2} against a mirror's {sharp:.2} — \
         roughness is not reaching the mip selection"
    );
    assert!(
        mid < sharp && mid > rough,
        "the blur should be monotonic in roughness: {sharp:.2}, {mid:.2}, {rough:.2}"
    );
}
