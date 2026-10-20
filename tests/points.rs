//! three.js `Points` on wgpu. See `tests/support/points.rs`. Skipped without a
//! GPU adapter.

#[path = "support/points.rs"]
mod support;

#[test]
fn points_draw_as_three_js_does() {
    let Ok(mut r) = threers::HeadlessRenderer::builder().size(support::N, support::N).build() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    let failures = support::check("wgpu", &mut |s, c| r.render_to_rgba(s, c));
    assert!(failures.is_empty(), "{} failures:\n{}", failures.len(), failures.join("\n"));
}
