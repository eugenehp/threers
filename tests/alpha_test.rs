//! three.js `alphaTest` and `transparent` on every material kind, on the wgpu
//! renderer (scenes and expectations in `tests/support/alpha_test.rs`;
//! `tests/metal_backend.rs` holds Metal to the same). Skipped without a GPU
//! adapter.

#[path = "support/alpha_test.rs"]
mod alpha;

#[test]
fn alpha_test_and_transparent_follow_three_js() {
    let mut failures = Vec::new();
    for format in [wgpu::TextureFormat::Rgba8UnormSrgb, wgpu::TextureFormat::Rgba8Unorm] {
        for msaa in [1, 4] {
            let Ok(mut r) = threers::HeadlessRenderer::builder()
                .size(alpha::N, alpha::N)
                .color_format(format)
                .build()
            else {
                eprintln!("skipping: no GPU adapter");
                return;
            };
            r.set_msaa(msaa);
            failures.extend(alpha::check(&format!("wgpu {format:?} msaa {msaa}"), &mut |s, c| {
                r.render_to_rgba(s, c)
            }));
        }
    }
    assert!(failures.is_empty(), "{} failures:\n{}", failures.len(), failures.join("\n"));
}

#[path = "support/shadow_cutout.rs"]
mod cutout;

/// A caster cut by `alphaTest` and its map casts only what is left of it, for
/// directional, spot and point shadows.
#[test]
fn alpha_tested_casters_cast_cut_out_shadows() {
    let Ok(mut r) = threers::HeadlessRenderer::builder().size(cutout::N, cutout::N).build() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    let failures = cutout::check("wgpu", &mut |s, c| r.render_to_rgba(s, c));
    assert!(failures.is_empty(), "{} failures:\n{}", failures.len(), failures.join("\n"));
}
