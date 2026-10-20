//! three.js `envMap`, `envMapIntensity`, `envMapRotation` and the scene's
//! `environmentIntensity` / `environmentRotation` on Standard materials, on
//! wgpu. See `tests/support/pbr_env.rs`. Skipped without a GPU adapter.

#[path = "support/pbr_env.rs"]
mod support;

#[test]
fn standard_materials_read_their_environment_as_three_js_does() {
    let Ok(mut r) = threers::HeadlessRenderer::builder().size(support::N, support::N).build() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    let failures = support::check("wgpu", &mut |s, c| r.render_to_rgba(s, c));
    assert!(failures.is_empty(), "{} failures:\n{}", failures.len(), failures.join("\n"));
}
