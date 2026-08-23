//! The renderer's caption overlay pass must produce the same picture as the
//! CPU painter, and must leave the rendered scene untouched everywhere the
//! overlay is transparent.
//!
//! Skipped when no GPU adapter is available.

#![cfg(feature = "captions")]

use threers::captions::{
    CaptionAnchor, CaptionOverlay, CaptionPainter, CaptionStyle, CaptionTrack,
};
use threers::{
    AmbientLight, BoxGeometry, Color, HeadlessRenderer, Material, Mesh, Object3D,
    PerspectiveCamera, Scene, StandardMaterial, Vector3,
};

const W: u32 = 320;
const H: u32 = 180;

fn renderer() -> Option<HeadlessRenderer> {
    HeadlessRenderer::builder()
        .size(W, H)
        // Unorm so the overlay's sRGB-authored colors land unchanged.
        .color_format(wgpu::TextureFormat::Rgba8Unorm)
        .build()
        .ok()
}

fn scene() -> Scene {
    let mut scene = Scene::new();
    scene.background = Color::from_hex(0x203050);
    let material = StandardMaterial::new(Color::from_hex(0xcc5544));
    scene.add(Object3D::mesh(Mesh::new(
        BoxGeometry::new(1.5, 1.5, 1.5),
        Material::Standard(material),
    )));
    scene.add_light(AmbientLight::new(Color::from_hex(0xffffff), 1.0));
    scene
}

fn camera() -> PerspectiveCamera {
    let mut c = PerspectiveCamera::new(50.0, W as f32 / H as f32, 0.1, 100.0);
    c.position = Vector3::new(0.0, 0.0, 4.0);
    c.look_at(Vector3::ZERO);
    c
}

fn style() -> CaptionStyle {
    CaptionStyle::default()
        .font_size(14.0)
        .margin(8.0)
        .padding(4.0)
        .outline([0, 0, 0, 255], 1.0)
        .background([0, 0, 0, 180])
        .anchor(CaptionAnchor::Bottom)
}

fn track() -> CaptionTrack {
    CaptionTrack::new().cue(0.0, 2.0, "Overlay pass")
}

#[test]
fn gpu_overlay_matches_the_cpu_painter() {
    let Some(mut renderer) = renderer() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    let (mut scene, camera) = (scene(), camera());

    // GPU: the renderer blends the overlay texture into the target.
    let mut overlay = CaptionOverlay::with_painter(track(), CaptionPainter::new().style(style()));
    let gpu = renderer.render_to_rgba_with_captions(&mut scene, &camera, &mut overlay, 1.0);

    // CPU: the same painter composites into the read-back buffer.
    let mut cpu = renderer.render_to_rgba(&mut scene, &camera);
    CaptionPainter::new()
        .style(style())
        .burn_in(&mut cpu, W, H, &track(), 1.0);

    assert_eq!(gpu.len(), cpu.len());
    // Blending happens in 8-bit on both sides but through different rounding,
    // so allow a channel or two of slack rather than requiring bit equality.
    let mut worst = 0i32;
    let mut differing = 0usize;
    for (a, b) in gpu.iter().zip(cpu.iter()) {
        let d = (*a as i32 - *b as i32).abs();
        worst = worst.max(d);
        if d > 0 {
            differing += 1;
        }
    }
    assert!(worst <= 2, "GPU and CPU overlays diverge by {worst}");
    let total = gpu.len();
    assert!(
        differing * 20 < total,
        "{differing}/{total} bytes differ — more than rounding can explain"
    );
}

#[test]
fn overlay_only_touches_caption_pixels() {
    let Some(mut renderer) = renderer() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    let (mut scene, camera) = (scene(), camera());

    let plain = renderer.render_to_rgba(&mut scene, &camera);
    let mut overlay = CaptionOverlay::with_painter(track(), CaptionPainter::new().style(style()));
    let captioned = renderer.render_to_rgba_with_captions(&mut scene, &camera, &mut overlay, 1.0);

    // Everything above the caption band must be pixel-identical.
    let band_top = (H as f32 * 0.7) as u32;
    let cut = (band_top * W * 4) as usize;
    assert_eq!(
        &plain[..cut],
        &captioned[..cut],
        "the overlay must not disturb the scene above the caption"
    );
    assert_ne!(
        &plain[cut..],
        &captioned[cut..],
        "the caption band must have changed"
    );
    // The target stays opaque.
    assert!(captioned.chunks_exact(4).all(|px| px[3] == 255));
}

#[test]
fn no_active_cue_leaves_the_frame_untouched() {
    let Some(mut renderer) = renderer() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    let (mut scene, camera) = (scene(), camera());

    let plain = renderer.render_to_rgba(&mut scene, &camera);
    let mut overlay = CaptionOverlay::with_painter(track(), CaptionPainter::new().style(style()));
    // t = 9s is past the only cue.
    let after = renderer.render_to_rgba_with_captions(&mut scene, &camera, &mut overlay, 9.0);
    assert!(!overlay.is_visible());
    assert_eq!(plain, after, "an empty overlay must be a no-op");
}

#[test]
fn overlay_lands_on_the_resolved_target_under_msaa() {
    let Some(mut renderer) = renderer() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    // MSAA takes the surface-style path, which resolves into the target's
    // single-sample color view — the one the overlay pass writes to.
    renderer.set_msaa(4);
    let (mut scene, camera) = (scene(), camera());

    let plain = renderer.render_to_rgba(&mut scene, &camera);
    let mut overlay = CaptionOverlay::with_painter(track(), CaptionPainter::new().style(style()));
    let captioned = renderer.render_to_rgba_with_captions(&mut scene, &camera, &mut overlay, 1.0);

    assert!(overlay.is_visible());
    assert_ne!(
        plain, captioned,
        "the caption must reach the resolved target"
    );
    let band_top = (H as f32 * 0.7) as u32;
    let cut = (band_top * W * 4) as usize;
    assert_eq!(
        &plain[..cut],
        &captioned[..cut],
        "scene must be undisturbed"
    );
}

#[test]
fn overlay_follows_a_resize() {
    let Some(mut renderer) = renderer() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    let (mut scene, camera) = (scene(), camera());
    let mut overlay = CaptionOverlay::with_painter(track(), CaptionPainter::new().style(style()));

    let first = renderer.render_to_rgba_with_captions(&mut scene, &camera, &mut overlay, 1.0);
    assert_eq!(overlay.size(), (W, H));
    assert_eq!(first.len(), (W * H * 4) as usize);

    // Drawing again at the same size must reuse the raster and repeat exactly.
    let second = renderer.render_to_rgba_with_captions(&mut scene, &camera, &mut overlay, 1.2);
    assert_eq!(first, second, "a cached overlay must draw identically");
}
