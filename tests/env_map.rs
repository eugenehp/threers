//! three.js `envMap` on Basic, Lambert and Phong, on wgpu. See
//! `tests/support/env_map.rs`. Skipped without a GPU adapter.

#[path = "support/env_map.rs"]
mod support;

#[test]
fn env_maps_reflect_and_combine_as_three_js_does() {
    let Ok(mut r) = threers::HeadlessRenderer::builder().size(support::N, support::N).build() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    let failures = support::check("wgpu", &mut |s, c| r.render_to_rgba(s, c));
    assert!(failures.is_empty(), "{} failures:\n{}", failures.len(), failures.join("\n"));
}
