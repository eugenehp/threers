//! three.js `polygonOffset` on the wgpu backend (see
//! `tests/support/polygon_offset.rs`). Skipped without a GPU adapter.

#[path = "support/polygon_offset.rs"]
mod decal;

#[test]
fn polygon_offset_keeps_a_decal_over_its_surface() {
    let Ok(mut r) = threers::HeadlessRenderer::builder().size(decal::N, decal::N).build() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    for msaa in [1, 4] {
        r.set_msaa(msaa);
        let [(blue0, red0), (blue1, red1)] = decal::measure(&mut |s, c| r.render_to_rgba(s, c));
        eprintln!("wgpu msaa {msaa}: no offset {blue0} blue / {red0} red; offset {blue1} blue / {red1} red");
        assert!(blue1 > 1000, "msaa {msaa}: decal interior is {blue1} pixels");
        assert_eq!(red1, 0, "msaa {msaa}: with polygonOffset the surface shows through the decal");
        assert!(red0 * 20 > blue0 + red0, "msaa {msaa}: without an offset the two should z-fight");
    }
}
