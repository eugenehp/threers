//! three.js's `map`, `aoMap` and `emissiveMap` on Lambert, Phong and Toon, on
//! the wgpu renderer (scenes and expectations in `tests/support/lit_maps.rs`;
//! `tests/metal_backend.rs` holds Metal to the same). Skipped without a GPU
//! adapter.

#[path = "support/lit_maps.rs"]
mod lit;

#[test]
fn lambert_phong_toon_shade_their_maps_as_three_js_does() {
    let mut failures = Vec::new();
    for format in [wgpu::TextureFormat::Rgba8UnormSrgb, wgpu::TextureFormat::Rgba8Unorm] {
        let Ok(mut r) = threers::HeadlessRenderer::builder()
            .size(lit::N, lit::N)
            .color_format(format)
            .build()
        else {
            eprintln!("skipping: no GPU adapter");
            return;
        };
        failures.extend(lit::check(&format!("wgpu {format:?}"), &mut |s, c| r.render_to_rgba(s, c)));
    }
    assert!(failures.is_empty(), "{} failures:\n{}", failures.len(), failures.join("\n"));
}
