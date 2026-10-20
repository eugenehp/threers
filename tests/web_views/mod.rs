//! Multi-view rendering in a real browser, shared by the WebGL2 suite
//! (`tests/webgl.rs`) and the WebGPU one (`tests/webgpu.rs`).
//!
//! What these check is what the native tests cannot: viewport, scissor and the
//! partial-clear pipeline compiled by the browser's own shader compiler (Tint
//! on WebGPU, ANGLE's GLSL on WebGL2), and several renders per animation frame
//! landing on one *canvas* — where each render acquires and presents the
//! surface on its own, so an earlier pane survives only if the later renders
//! load rather than clear.
//!
//! Each including file supplies `super::views_renderer()`, which hands back the
//! one shared renderer for its backend (a browser caps live contexts per page)
//! and the canvas it draws to.
//!
//! The colours are chosen with red equal to blue, or one of them zero in both:
//! a WebGPU canvas in Chrome is `bgra8unorm`, render-target read-back is not
//! swizzled, and these colours read the same either way round.

use threers::wasm::{
    WebBufferAttribute, WebBufferGeometry, WebCamera, WebColor, WebGeometry, WebMaterial, WebMesh,
    WebQuadView, WebRenderTarget, WebRenderer, WebScene,
};
use wasm_bindgen::{JsCast, JsValue};
use wasm_bindgen_test::wasm_bindgen_test;

const W: u32 = 64;
const H: u32 = 64;

/// Pull a render target back to the CPU (top-left origin, 4 bytes a pixel).
async fn read_target(renderer: &WebRenderer, target_id: u32, w: u32, h: u32) -> Vec<u8> {
    let promise = renderer.read_render_target_pixels(target_id, 0, 0, w, h);
    let value = wasm_bindgen_futures::JsFuture::from(promise)
        .await
        .expect("read-back rejected");
    js_sys::Uint8Array::new(&value).to_vec()
}

/// The canvas's current picture, RGBA, top-left origin.
///
/// Copied through a 2D canvas, in the same task as the renders: a WebGL2 or
/// WebGPU canvas only promises its drawing buffer until the browser composites
/// it, and a `drawImage` before that sees what the frame drew. Done through
/// `Reflect` so the test needs no extra `web-sys` features.
fn read_canvas(canvas: &web_sys::HtmlCanvasElement) -> Vec<u8> {
    let document = web_sys::window().unwrap().document().unwrap();
    let copy = document
        .create_element("canvas")
        .unwrap()
        .dyn_into::<web_sys::HtmlCanvasElement>()
        .unwrap();
    let (w, h) = (canvas.width(), canvas.height());
    copy.set_width(w);
    copy.set_height(h);
    let ctx: JsValue = copy
        .get_context("2d")
        .expect("2d context")
        .expect("2d context")
        .into();
    let call = |name: &str, args: &[JsValue]| -> JsValue {
        let f: js_sys::Function = js_sys::Reflect::get(&ctx, &name.into())
            .unwrap()
            .dyn_into()
            .unwrap();
        let arr = js_sys::Array::new();
        for a in args {
            arr.push(a);
        }
        f.apply(&ctx, &arr).expect(name)
    };
    call("drawImage", &[canvas.into(), 0.into(), 0.into()]);
    let image = call(
        "getImageData",
        &[0.into(), 0.into(), (w as f64).into(), (h as f64).into()],
    );
    let data = js_sys::Reflect::get(&image, &"data".into()).unwrap();
    js_sys::Uint8ClampedArray::new(&data).to_vec()
}

/// Pixel at `(x, y)` with a **bottom-left** origin — the viewport convention —
/// out of a top-left-origin buffer.
fn px(buf: &[u8], w: u32, h: u32, x: u32, y: u32) -> [u8; 3] {
    let i = (((h - 1 - y) * w + x) * 4) as usize;
    [buf[i], buf[i + 1], buf[i + 2]]
}

fn is_green(p: [u8; 3]) -> bool {
    p[1] > 200 && p[0] < 40 && p[2] < 40
}
fn is_magenta(p: [u8; 3]) -> bool {
    p[0] > 200 && p[2] > 200 && p[1] < 40
}
fn is_white(p: [u8; 3]) -> bool {
    p.iter().all(|&c| c > 200)
}
fn is_black(p: [u8; 3]) -> bool {
    p.iter().all(|&c| c < 40)
}
/// The mid grey used for gaps: neutral and nowhere near black or white.
fn is_grey(p: [u8; 3]) -> bool {
    let (lo, hi) = (*p.iter().min().unwrap(), *p.iter().max().unwrap());
    hi - lo < 12 && lo > 90 && hi < 220
}

fn grey() -> WebColor {
    WebColor::new(0.3, 0.3, 0.3)
}

/// Put the renderer back the way every test expects to find it.
fn reset(renderer: &mut WebRenderer) {
    renderer.set_render_target(0);
    renderer.reset_viewport();
    renderer.set_scissor_test(false);
    renderer.set_auto_clear(true);
    renderer.set_pixel_ratio(1.0);
}

/// Straight down the Z axis at the XY plane, ±1 filling the target.
fn front_ortho() -> WebCamera {
    let mut c = WebCamera::orthographic(-1.0, 1.0, 1.0, -1.0, 0.1, 10.0);
    c.set_position(0.0, 0.0, 5.0);
    c.look_at(0.0, 0.0, 0.0);
    c
}

/// A unit quad in XY as two indexed triangles, split along its 0–2 diagonal,
/// optionally declared as one polygon.
fn quad_buffer_geometry(polygon: bool) -> WebGeometry {
    let mut g = WebBufferGeometry::new();
    g.set_attribute(
        "position",
        WebBufferAttribute::new(
            vec![
                -0.5, -0.5, 0.0, 0.5, -0.5, 0.0, 0.5, 0.5, 0.0, -0.5, 0.5, 0.0,
            ],
            3,
        ),
    );
    g.set_attribute(
        "normal",
        WebBufferAttribute::new(
            vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
            3,
        ),
    );
    g.set_index(vec![0, 1, 2, 0, 2, 3]);
    if polygon {
        g.set_polygons(vec![2])
            .expect("two triangles make one quad");
        assert_eq!(g.polygon_counts(), Some(vec![2]));
    }
    WebGeometry::from_buffer_geometry(g)
}

fn white_wire() -> WebMaterial {
    let mut m = WebMaterial::basic(&WebColor::new(1.0, 1.0, 1.0));
    m.set_wireframe(true);
    m
}

/// Lit pixels in the `(2r+1)²` window centred on `(cx, cy)`.
fn lit_near(buf: &[u8], cx: u32, cy: u32, r: u32) -> usize {
    let mut n = 0;
    for y in cy - r..=cy + r {
        for x in cx - r..=cx + r {
            if px(buf, W, H, x, y).iter().any(|&c| c > 100) {
                n += 1;
            }
        }
    }
    n
}

/// Render `scene` from `camera` into a fresh `W`×`H` target and read it back.
async fn draw(renderer: &mut WebRenderer, scene: &mut WebScene, camera: &WebCamera) -> Vec<u8> {
    let target = WebRenderTarget::new(renderer, W, H);
    renderer.set_render_target(target.id);
    renderer.render(scene, camera);
    renderer.render(scene, camera);
    let pixels = read_target(renderer, target.id, W, H).await;
    renderer.set_render_target(0);
    pixels
}

/// `setPolygons` throws on a table that does not cover the triangles, and
/// leaves the geometry ungrouped.
#[wasm_bindgen_test]
fn set_polygons_rejects_a_mismatched_table() {
    let mut g = WebBufferGeometry::new();
    g.set_attribute("position", WebBufferAttribute::new(vec![0.0; 12], 3));
    g.set_index(vec![0, 1, 2, 0, 2, 3]);
    assert!(
        g.set_polygons(vec![3]).is_err(),
        "3 triangles claimed, 2 drawn"
    );
    assert!(
        g.set_polygons(vec![2, 0]).is_err(),
        "a polygon of zero triangles"
    );
    assert_eq!(g.polygon_counts(), None, "a rejected table must not stick");
    g.set_polygons(vec![1, 1])
        .expect("two triangles, two polygons");
    assert_eq!(g.polygon_counts(), Some(vec![1, 1]));
    // A new index is new topology: the grouping goes with the old one.
    g.set_index(vec![0, 1, 2, 0, 2, 3]);
    assert_eq!(g.polygon_counts(), None);
}

/// A quad declared as one polygon draws its outline in wireframe and not the
/// diagonal; the same two triangles without the declaration draw both.
#[wasm_bindgen_test]
async fn a_polygon_wireframe_draws_the_outline_not_the_diagonal() {
    let (shared, _canvas) = super::views_renderer().await;
    let mut guard = shared.borrow_mut();
    let renderer: &mut WebRenderer = &mut guard;
    reset(renderer);
    let camera = front_ortho();

    let mut results = Vec::new();
    for polygon in [true, false] {
        let mut scene = WebScene::new();
        scene.set_background(&WebColor::new(0.0, 0.0, 0.0));
        scene.add(&WebMesh::new(&quad_buffer_geometry(polygon), &white_wire()));
        let pixels = draw(renderer, &mut scene, &camera).await;
        // The quad spans pixels 16..48. The outline's left edge is near x = 16;
        // the 0–2 diagonal crosses the centre.
        let left_edge: usize = (24..40).map(|y| lit_near(&pixels, 16, y, 1).min(1)).sum();
        let centre = lit_near(&pixels, 32, 32, 2);
        let off_diagonal = lit_near(&pixels, 24, 40, 2);
        results.push((polygon, left_edge, centre, off_diagonal));
    }
    let (_, edge_poly, centre_poly, off_poly) = results[0];
    let (_, edge_tris, centre_tris, _) = results[1];
    assert!(
        edge_poly >= 12 && edge_tris >= 12,
        "the outline should draw either way: {results:?}"
    );
    assert_eq!(
        centre_poly, 0,
        "the polygon's fan diagonal was drawn: {results:?}"
    );
    assert_eq!(off_poly, 0, "something drew inside the quad: {results:?}");
    assert!(
        centre_tris > 0,
        "without polygons the diagonal should draw — otherwise the check above \
         proves nothing: {results:?}"
    );
}

/// `QuadGeometry` (through `WebGeometry.quad`) fills its four corners, and its
/// wireframe has no diagonal because it carries its cells as polygons.
#[wasm_bindgen_test]
async fn quad_geometry_draws_filled_and_as_an_outline() {
    let (shared, _canvas) = super::views_renderer().await;
    let mut guard = shared.borrow_mut();
    let renderer: &mut WebRenderer = &mut guard;
    reset(renderer);
    let camera = front_ortho();
    // A skewed quad, not an axis-aligned rectangle: x from -0.6 to 0.6 at the
    // bottom, -0.4 to 0.4 at the top.
    let corners = [
        -0.6, -0.5, 0.0, 0.6, -0.5, 0.0, 0.4, 0.5, 0.0, -0.4, 0.5, 0.0,
    ];
    let quad = WebGeometry::quad(&corners, 1, 1).expect("twelve coordinates");
    assert!(
        WebGeometry::quad(&corners[..9], 1, 1).is_err(),
        "nine coordinates is three corners"
    );
    assert_eq!(quad.to_buffer_geometry().polygon_counts(), Some(vec![2]));

    let mut scene = WebScene::new();
    scene.set_background(&WebColor::new(0.0, 0.0, 0.0));
    scene.add(&WebMesh::new(
        &quad,
        &WebMaterial::basic(&WebColor::new(0.0, 1.0, 0.0)),
    ));
    let filled = draw(renderer, &mut scene, &camera).await;
    assert!(
        is_green(px(&filled, W, H, 32, 32)),
        "centre {:?}",
        px(&filled, W, H, 32, 32)
    );
    // Bottom corners at x = ±0.6 → pixels 13 and 51; the top is narrower.
    assert!(
        is_green(px(&filled, W, H, 15, 18)),
        "bottom-left {:?}",
        px(&filled, W, H, 15, 18)
    );
    assert!(
        is_black(px(&filled, W, H, 15, 46)),
        "outside the narrow top {:?}",
        px(&filled, W, H, 15, 46)
    );
    assert!(
        is_black(px(&filled, W, H, 2, 2)),
        "corner {:?}",
        px(&filled, W, H, 2, 2)
    );

    let mut scene = WebScene::new();
    scene.set_background(&WebColor::new(0.0, 0.0, 0.0));
    scene.add(&WebMesh::new(&quad, &white_wire()));
    let wire = draw(renderer, &mut scene, &camera).await;
    assert_eq!(
        lit_near(&wire, 32, 32, 2),
        0,
        "QuadGeometry's wireframe drew a diagonal"
    );
    let bottom: usize = (24..40).map(|x| lit_near(&wire, x, 16, 1).min(1)).sum();
    assert!(
        bottom >= 12,
        "the bottom edge of the outline is missing ({bottom})"
    );
}

/// `WebBufferGeometry.subdivide` on a 2-unit cube: the result is quads (24 at
/// level 1, 96 at 2), it renders filled with its corners rounded off, and its
/// wireframe outlines the quads without their fan diagonals.
///
/// Seen head-on, one round puts the front face's centre at the origin, its
/// edge points at (±0.75, 0) and (0, ±0.75) and its corners at (±0.56, ±0.56);
/// the silhouette's corner is the vertical edges' edge point at (±0.75,
/// ±0.75). Each sub-quad is wound [vertex, edge, face, edge], so its fan
/// diagonal runs from a corner to the centre.
#[wasm_bindgen_test]
async fn a_subdivided_cube_renders_and_outlines_its_quads() {
    let (shared, _canvas) = super::views_renderer().await;
    let mut guard = shared.borrow_mut();
    let renderer: &mut WebRenderer = &mut guard;
    reset(renderer);
    let camera = front_ortho();

    let cube = WebGeometry::box_(2.0, 2.0, 2.0).to_buffer_geometry();
    assert_eq!(cube.polygon_counts(), Some(vec![2; 6]), "the box is six quads");
    let level1 = cube.subdivide(Some(1), None).expect("level 1");
    assert_eq!(level1.polygon_counts(), Some(vec![2; 24]));
    let level2 = cube.subdivide(Some(2), None).expect("level 2");
    assert_eq!(level2.polygon_counts(), Some(vec![2; 96]));
    assert!(cube.subdivide(Some(7), None).is_err(), "level 7 is refused");
    assert!(level1.has_attribute("normal"));
    // Every face a quad: `faces()` lists 24 loops of four.
    let faces = level1.faces();
    assert_eq!(faces.len(), 24 * 5);
    assert!(faces.chunks(5).all(|f| f[0] == 4), "{faces:?}");

    let mut scene = WebScene::new();
    scene.set_background(&WebColor::new(0.0, 0.0, 0.0));
    scene.add(&WebMesh::new(
        &WebGeometry::from_buffer_geometry(level1),
        &WebMaterial::basic(&WebColor::new(0.0, 1.0, 0.0)),
    ));
    let filled = draw(renderer, &mut scene, &camera).await;
    let drawn = filled.chunks(4).filter(|p| is_green([p[0], p[1], p[2]])).count();
    assert!(drawn > 1000, "only {drawn} pixels drawn");
    assert!(
        is_green(px(&filled, W, H, 32, 32)),
        "centre {:?}",
        px(&filled, W, H, 32, 32)
    );
    // (±0.875, ±0.875): inside the cage, outside the rounded surface.
    for (x, y) in [(60, 60), (4, 4), (60, 4), (4, 60)] {
        assert!(
            is_black(px(&filled, W, H, x, y)),
            "corner ({x}, {y}) was not rounded off: {:?}",
            px(&filled, W, H, x, y)
        );
    }

    // The same triangles with and without the quad grouping, in wireframe.
    let grouped = cube.subdivide(Some(1), None).expect("level 1");
    let mut ungrouped = cube.subdivide(Some(1), None).expect("level 1");
    ungrouped.set_index(grouped.get_index_array().expect("indexed"));
    assert_eq!(ungrouped.polygon_counts(), None);
    let mut lit = Vec::new();
    for g in [grouped, ungrouped] {
        let mut scene = WebScene::new();
        scene.set_background(&WebColor::new(0.0, 0.0, 0.0));
        scene.add(&WebMesh::new(&WebGeometry::from_buffer_geometry(g), &white_wire()));
        let wire = draw(renderer, &mut scene, &camera).await;
        // Inside each front sub-quad (4..14 px off the centre on both axes),
        // clear of its outline; the corner-to-centre diagonal crosses it.
        let mut inside = 0;
        for (sx, sy) in [(1i32, 1i32), (-1, 1), (1, -1), (-1, -1)] {
            for d in 4..=14 {
                // The centre sits on the corner of pixels 31 and 32, so the
                // pixel `d` out along a diagonal is 32 + d or 31 - d.
                let at = |s: i32| (if s > 0 { 32 + d } else { 31 - d }) as u32;
                let (x, y) = (at(sx), at(sy));
                if px(&wire, W, H, x, y).iter().any(|&c| c > 100) {
                    inside += 1;
                }
            }
        }
        // The outline through the centre: the edges to the edge points.
        let spokes: usize = (20..28).map(|x| lit_near(&wire, x, 32, 1).min(1)).sum();
        lit.push((inside, spokes));
    }
    let ((quads_inside, quads_spokes), (tris_inside, _)) = (lit[0], lit[1]);
    assert!(quads_spokes >= 6, "the quad outlines are missing: {lit:?}");
    assert_eq!(quads_inside, 0, "a fan diagonal was drawn: {lit:?}");
    assert!(
        tris_inside >= 20,
        "without the grouping the diagonals should draw — otherwise the check \
         above proves nothing: {lit:?}"
    );
}

/// Viewport + scissor + `clear`: each pane gets its own colour, a render with
/// auto-clear clears only its own pane, and the gap keeps the first clear.
#[wasm_bindgen_test]
async fn viewport_scissor_and_partial_clear_leave_the_gaps_alone() {
    let (shared, _canvas) = super::views_renderer().await;
    let mut guard = shared.borrow_mut();
    let renderer: &mut WebRenderer = &mut guard;
    reset(renderer);
    let target = WebRenderTarget::new(renderer, W, H);
    renderer.set_render_target(target.id);

    // Whole target grey first (scissor test off, so the clear is full).
    renderer.clear(&grey(), 1.0);

    renderer.set_scissor_test(true);
    renderer.set_auto_clear(false);
    let panes = [(0.0, 0.0), (34.0, 0.0), (0.0, 34.0)];
    let colours = [
        WebColor::new(0.0, 1.0, 0.0),
        WebColor::new(1.0, 0.0, 1.0),
        WebColor::new(1.0, 1.0, 1.0),
    ];
    for ((x, y), c) in panes.iter().zip(&colours) {
        renderer.set_viewport(*x, *y, 30.0, 30.0);
        renderer.set_scissor(*x, *y, 30.0, 30.0);
        renderer.clear(c, 1.0);
    }
    // The fourth pane through `render` with auto-clear on: the scene background
    // fills the scissor, a mesh draws inside the viewport.
    renderer.set_auto_clear(true);
    renderer.set_viewport(34.0, 34.0, 30.0, 30.0);
    renderer.set_scissor(34.0, 34.0, 30.0, 30.0);
    let mut scene = WebScene::new();
    scene.set_background(&WebColor::new(0.0, 0.0, 0.0));
    let mesh = WebMesh::new(
        &WebGeometry::plane(0.5, 0.5),
        &WebMaterial::basic(&WebColor::new(0.0, 1.0, 0.0)),
    );
    scene.add(&mesh);
    renderer.render(&mut scene, &front_ortho());

    let pixels = read_target(renderer, target.id, W, H).await;
    reset(renderer);

    let at = |x, y| px(&pixels, W, H, x, y);
    assert!(is_green(at(15, 15)), "bottom-left pane {:?}", at(15, 15));
    assert!(is_magenta(at(49, 15)), "bottom-right pane {:?}", at(49, 15));
    assert!(is_white(at(15, 49)), "top-left pane {:?}", at(15, 49));
    // Rendered pane: black background at its edge, the green plane at its centre
    // (the plane is a quarter of the ±1 view, so ~7 px across the 30 px pane).
    assert!(
        is_black(at(37, 37)),
        "rendered pane background {:?}",
        at(37, 37)
    );
    assert!(is_green(at(49, 49)), "rendered pane mesh {:?}", at(49, 49));
    // The gaps, the borders between panes and the cross in the middle.
    for (x, y) in [
        (32, 15),
        (32, 49),
        (15, 32),
        (49, 32),
        (32, 32),
        (31, 2),
        (2, 31),
    ] {
        assert!(
            is_grey(at(x, y)),
            "gap pixel ({x}, {y}) was touched: {:?}",
            at(x, y)
        );
    }
    // The pane edges are exact: 29 is inside, 30 is gap.
    assert!(is_green(at(29, 29)) && is_grey(at(30, 29)) && is_grey(at(29, 30)));
}

/// Several `render` calls in one frame to the **canvas** keep each other's
/// panes. Each call acquires and presents the surface by itself, so this only
/// holds if a pane render loads what is there instead of clearing the frame.
#[wasm_bindgen_test]
async fn several_renders_to_the_canvas_in_one_frame_keep_earlier_panes() {
    let (shared, canvas) = super::views_renderer().await;
    let mut guard = shared.borrow_mut();
    let renderer: &mut WebRenderer = &mut guard;
    reset(renderer);
    let (w, h) = (canvas.width() as f32, canvas.height() as f32);
    let half = (w / 2.0).floor();

    let mut left = WebScene::new();
    left.set_background(&WebColor::new(0.0, 1.0, 0.0));
    let mut right = WebScene::new();
    right.set_background(&WebColor::new(1.0, 0.0, 1.0));
    let camera = front_ortho();

    renderer.set_scissor_test(true);
    renderer.set_viewport(0.0, 0.0, half, h);
    renderer.set_scissor(0.0, 0.0, half, h);
    renderer.render(&mut left, &camera);
    renderer.set_viewport(half, 0.0, w - half, h);
    renderer.set_scissor(half, 0.0, w - half, h);
    renderer.render(&mut right, &camera);
    // A third pane over the bottom of the first, so a picture that is upside
    // down — not only one missing a pane — fails.
    let mut bottom = WebScene::new();
    bottom.set_background(&WebColor::new(1.0, 1.0, 1.0));
    renderer.set_viewport(0.0, 0.0, half, (h / 2.0).floor());
    renderer.set_scissor(0.0, 0.0, half, (h / 2.0).floor());
    renderer.render(&mut bottom, &camera);
    let pixels = read_canvas(&canvas);
    reset(renderer);

    let (cw, ch) = (canvas.width(), canvas.height());
    let at = |x, y| px(&pixels, cw, ch, x, y);
    let q = cw / 4;
    let (top, low) = (3 * ch / 4, ch / 4);
    assert!(
        is_green(at(q, top)),
        "the first pane did not survive the later renders: {:?} (right {:?}, bottom-left {:?})",
        at(q, top),
        at(3 * q, top),
        at(q, low)
    );
    assert!(
        is_magenta(at(3 * q, top)) && is_magenta(at(3 * q, low)),
        "second pane {:?}",
        at(3 * q, top)
    );
    assert!(
        is_white(at(q, low)),
        "third pane (bottom-left) {:?}",
        at(q, low)
    );
}

/// The scene the `QuadView` tests draw: one white box at the origin, which the
/// orthographic panes frame and the perspective camera looks at.
fn quad_view_scene() -> (WebScene, WebCamera, WebQuadView) {
    let mut scene = WebScene::new();
    scene.set_background(&WebColor::new(0.0, 0.0, 0.0));
    scene.add(&WebMesh::new(
        &WebGeometry::box_(1.0, 1.0, 1.0),
        &WebMaterial::basic(&WebColor::new(1.0, 1.0, 1.0)),
    ));
    let mut persp = WebCamera::perspective(40.0, 1.0, 0.1, 100.0);
    persp.set_position(3.0, 2.5, 3.5);
    persp.look_at(0.0, 0.0, 0.0);

    let mut quad = WebQuadView::new();
    assert!(
        quad.frame_scene(&mut scene),
        "a scene with a box has bounds"
    );
    quad.set_gap(4.0);
    quad.set_gap_color(&grey());
    quad.set_pane_background("top", &WebColor::new(0.0, 1.0, 0.0))
        .unwrap();
    quad.set_pane_background("front", &WebColor::new(1.0, 0.0, 1.0))
        .unwrap();
    quad.set_pane_background("side", &WebColor::new(0.0, 0.0, 0.0))
        .unwrap();
    quad.set_pane_background("perspective", &WebColor::new(0.2, 0.0, 0.2))
        .unwrap();
    (scene, persp, quad)
}

/// Check the four panes and the gap of a `QuadView` frame read back as `buf`.
fn assert_quad_view(buf: &[u8], quad: &WebQuadView, w: u32, h: u32, what: &str) {
    let at = |x: u32, y: u32| px(buf, w, h, x, y);
    let is_dim_magenta =
        |p: [u8; 3]| p[1] < 40 && (60..170).contains(&p[0]) && (60..170).contains(&p[2]);
    let checks: [(&str, fn([u8; 3]) -> bool); 3] =
        [("top", is_green), ("front", is_magenta), ("side", is_black)];
    let mut all: Vec<(&str, Box<dyn Fn([u8; 3]) -> bool>)> = checks
        .iter()
        .map(|&(n, f)| (n, Box::new(f) as Box<dyn Fn([u8; 3]) -> bool>))
        .collect();
    all.push(("perspective", Box::new(is_dim_magenta)));
    for (pane, bg) in &all {
        let r = quad
            .pane_rect(pane, w as f32, h as f32)
            .unwrap()
            .unwrap_or_else(|| panic!("{pane} pane hidden"));
        let (x0, y0) = (r[0] as u32, r[1] as u32);
        let (x1, y1) = ((r[0] + r[2]) as u32 - 1, (r[1] + r[3]) as u32 - 1);
        // Corners of the pane show its background (inset past the edge).
        for (x, y) in [(x0 + 2, y0 + 2), (x1 - 2, y1 - 2)] {
            assert!(
                bg(at(x, y)),
                "{what}: {pane} background at ({x}, {y}) is {:?}",
                at(x, y)
            );
        }
        // The framed box is in the middle of every pane.
        let (cx, cy) = ((x0 + x1) / 2, (y0 + y1) / 2);
        assert!(
            is_white(at(cx, cy)),
            "{what}: {pane} centre is {:?}, not the box",
            at(cx, cy)
        );
        assert_eq!(
            quad.pane_at(cx as f32, cy as f32, w as f32, h as f32)
                .as_deref(),
            Some(*pane)
        );
    }
    // The gap between the left and right columns, and the cross in the middle.
    let top_left = quad.pane_rect("top", w as f32, h as f32).unwrap().unwrap();
    let gap_x = (top_left[0] + top_left[2]) as u32 + 1;
    for y in [3, h / 2, h - 4] {
        assert!(
            is_grey(at(gap_x, y)),
            "{what}: gap at ({gap_x}, {y}) is {:?}",
            at(gap_x, y)
        );
    }
    assert_eq!(quad.pane_at(gap_x as f32, 3.0, w as f32, h as f32), None);
}

/// `QuadView` draws four panes — each its background, the box framed in the
/// middle of each — with the gap colour between them, into a render target.
#[wasm_bindgen_test]
async fn quad_view_renders_four_panes_into_a_target() {
    let (shared, canvas) = super::views_renderer().await;
    let mut guard = shared.borrow_mut();
    let renderer: &mut WebRenderer = &mut guard;
    reset(renderer);
    let (w, h) = (canvas.width(), canvas.height());
    let (mut scene, persp, mut quad) = quad_view_scene();
    // The layout follows the renderer's size, which is the canvas's: a target
    // the same size reads back the same frame.
    let target = WebRenderTarget::new(renderer, w, h);
    renderer.set_render_target(target.id);
    quad.render(renderer, &mut scene, &persp)
        .expect("perspective camera");
    let pixels = read_target(renderer, target.id, w, h).await;
    reset(renderer);
    assert_quad_view(&pixels, &quad, w, h, "render target");

    let ortho = quad.ortho_camera("top").unwrap();
    assert!(
        quad.render(renderer, &mut scene, &ortho).is_err(),
        "an orthographic camera for the perspective pane should be refused"
    );
    assert!(quad.ortho_camera("perspective").is_err());
    assert!(quad.set_pane_background("nope", &grey()).is_err());
}

/// The same on the canvas: one acquire, four panes, one present.
#[wasm_bindgen_test]
async fn quad_view_renders_four_panes_to_the_canvas() {
    let (shared, canvas) = super::views_renderer().await;
    let mut guard = shared.borrow_mut();
    let renderer: &mut WebRenderer = &mut guard;
    reset(renderer);
    let (w, h) = (canvas.width(), canvas.height());
    let (mut scene, persp, mut quad) = quad_view_scene();
    quad.render(renderer, &mut scene, &persp)
        .expect("perspective camera");
    let pixels = read_canvas(&canvas);
    reset(renderer);
    assert_quad_view(&pixels, &quad, w, h, "canvas");

    // Maximise: one pane over the whole canvas.
    quad.toggle_maximize("front").unwrap();
    assert_eq!(quad.maximized().as_deref(), Some("front"));
    assert_eq!(quad.pane_rect("top", w as f32, h as f32).unwrap(), None);
    quad.render(renderer, &mut scene, &persp).unwrap();
    let pixels = read_canvas(&canvas);
    reset(renderer);
    assert!(
        is_magenta(px(&pixels, w, h, 2, 2)),
        "maximised corner {:?}",
        px(&pixels, w, h, 2, 2)
    );
    assert!(
        is_white(px(&pixels, w, h, w / 2, h / 2)),
        "maximised centre"
    );
    quad.toggle_maximize("front").unwrap();
    assert_eq!(quad.maximized(), None);
}

/// Viewport rectangles are in logical pixels times the pixel ratio, as in
/// three.js: at ratio 2 a 16×16 viewport covers 32×32 device pixels.
#[wasm_bindgen_test]
async fn pixel_ratio_scales_the_viewport() {
    let (shared, _canvas) = super::views_renderer().await;
    let mut guard = shared.borrow_mut();
    let renderer: &mut WebRenderer = &mut guard;
    reset(renderer);
    let target = WebRenderTarget::new(renderer, W, H);
    renderer.set_render_target(target.id);
    renderer.clear(&grey(), 1.0);
    renderer.set_pixel_ratio(2.0);
    renderer.set_scissor_test(true);
    renderer.set_scissor(0.0, 0.0, 16.0, 16.0);
    renderer.clear(&WebColor::new(0.0, 1.0, 0.0), 1.0);
    let pixels = read_target(renderer, target.id, W, H).await;
    reset(renderer);
    assert!(
        is_green(px(&pixels, W, H, 30, 30)),
        "inside 32×32: {:?}",
        px(&pixels, W, H, 30, 30)
    );
    assert!(
        is_grey(px(&pixels, W, H, 33, 10)),
        "outside 32×32: {:?}",
        px(&pixels, W, H, 33, 10)
    );
}

/// three.js's multi-view idiom, on the canvas: auto-clear off, scissor test
/// on, and per view `setViewport` + `setScissor` + `clear` + `render`. The
/// clears and the draws each present separately, and every view survives.
#[wasm_bindgen_test]
async fn the_threejs_multi_view_idiom_on_the_canvas() {
    let (shared, canvas) = super::views_renderer().await;
    let mut guard = shared.borrow_mut();
    let renderer: &mut WebRenderer = &mut guard;
    reset(renderer);
    let (w, h) = (canvas.width() as f32, canvas.height() as f32);
    let half = (w / 2.0).floor();

    let mut scene = WebScene::new();
    // Must not show: with auto-clear off a render draws over what is there.
    scene.set_background(&WebColor::new(1.0, 0.0, 0.0));
    scene.add(&WebMesh::new(
        &WebGeometry::plane(0.5, 0.5),
        &WebMaterial::basic(&WebColor::new(1.0, 1.0, 1.0)),
    ));
    let camera = front_ortho();

    renderer.set_auto_clear(false);
    renderer.set_scissor_test(true);
    let views = [
        (0.0, half, WebColor::new(0.0, 1.0, 0.0)),
        (half, w - half, WebColor::new(1.0, 0.0, 1.0)),
    ];
    for (x, width, colour) in &views {
        renderer.set_viewport(*x, 0.0, *width, h);
        renderer.set_scissor(*x, 0.0, *width, h);
        renderer.clear(colour, 1.0);
        renderer.render(&mut scene, &camera);
    }
    let pixels = read_canvas(&canvas);
    reset(renderer);

    let (cw, ch) = (canvas.width(), canvas.height());
    let at = |x, y| px(&pixels, cw, ch, x, y);
    let q = cw / 4;
    assert!(
        is_green(at(3, ch - 4)),
        "left view's clear {:?}",
        at(3, ch - 4)
    );
    assert!(
        is_magenta(at(cw - 4, 3)),
        "right view's clear {:?}",
        at(cw - 4, 3)
    );
    assert!(
        is_white(at(q, ch / 2)),
        "left view's plane {:?}",
        at(q, ch / 2)
    );
    assert!(
        is_white(at(3 * q, ch / 2)),
        "right view's plane {:?}",
        at(3 * q, ch / 2)
    );
}

/// `QuadView` as the first thing a fresh canvas draws — no viewport, scissor
/// or clear call before it — which on WebGL2 takes the direct path rather
/// than the back buffer the other canvas tests leave switched on.
#[wasm_bindgen_test]
async fn quad_view_on_a_fresh_canvas() {
    let backend = {
        let (shared, _canvas) = super::views_renderer().await;
        let b = shared.borrow().backend();
        b
    };
    let document = web_sys::window().unwrap().document().unwrap();
    let canvas = document
        .create_element("canvas")
        .unwrap()
        .dyn_into::<web_sys::HtmlCanvasElement>()
        .unwrap();
    canvas.set_width(128);
    canvas.set_height(96);
    document.body().unwrap().append_child(&canvas).unwrap();
    let mut renderer = WebRenderer::new(canvas.clone()).await.expect("renderer");
    assert_eq!(renderer.backend(), backend);
    let (mut scene, persp, mut quad) = quad_view_scene();
    quad.render(&mut renderer, &mut scene, &persp).unwrap();
    let pixels = read_canvas(&canvas);
    assert_quad_view(&pixels, &quad, 128, 96, "fresh canvas");
}

/// Edge pixels of the box in the perspective pane — a mix of the white box
/// and the dim-magenta background, which only anti-aliasing produces.
fn mixed_edge_pixels(buf: &[u8], quad: &WebQuadView, w: u32, h: u32) -> usize {
    let r = quad
        .pane_rect("perspective", w as f32, h as f32)
        .unwrap()
        .unwrap();
    let (x0, y0) = (r[0] as u32, r[1] as u32);
    let (x1, y1) = ((r[0] + r[2]) as u32, (r[1] + r[3]) as u32);
    (y0..y1)
        .flat_map(|y| (x0..x1).map(move |x| (x, y)))
        .filter(|&(x, y)| (40..200).contains(&px(buf, w, h, x, y)[1]))
        .count()
}

/// MSAA in the panes, on the canvas: the perspective pane's box edges carry
/// intermediate coverage at 4 samples and none at 1, and the panes and gaps
/// are as they are without it.
#[wasm_bindgen_test]
async fn quad_view_panes_are_antialiased_with_msaa() {
    let (shared, canvas) = super::views_renderer().await;
    let mut guard = shared.borrow_mut();
    let renderer: &mut WebRenderer = &mut guard;
    let (w, h) = (canvas.width(), canvas.height());
    let mut counts = Vec::new();
    for samples in [1, 4] {
        reset(renderer);
        renderer.set_msaa(samples);
        let (mut scene, persp, mut quad) = quad_view_scene();
        quad.render(renderer, &mut scene, &persp).unwrap();
        let pixels = read_canvas(&canvas);
        renderer.set_msaa(1);
        reset(renderer);
        assert_quad_view(&pixels, &quad, w, h, &format!("msaa {samples}"));
        counts.push(mixed_edge_pixels(&pixels, &quad, w, h));
    }
    assert_eq!(counts[0], 0, "no MSAA, yet {} mixed pixels", counts[0]);
    assert!(counts[1] > 8, "MSAA 4 left only {} mixed edge pixels", counts[1]);
}

/// The interaction bindings: a drag in the top pane pans the shared centre,
/// the wheel there zooms all three orthographic panes, a drag in the
/// perspective pane orbits the camera and only it, and a double-click
/// maximises and restores. Canvas coordinates, top-left origin.
#[wasm_bindgen_test]
fn quad_view_pointer_bindings() {
    let (_scene, mut persp, mut quad) = quad_view_scene();
    let (w, h) = (128.0f32, 96.0f32);
    // Top pane: the top-left quadrant of the canvas.
    let (tx, ty) = (w * 0.25, h * 0.25);
    assert_eq!(
        quad.pointer_down(&mut persp, tx, ty, 0, w, h).unwrap().as_deref(),
        Some("top")
    );
    assert!(quad.pointer_move(&mut persp, tx + 10.0, ty).unwrap());
    quad.pointer_up();
    let t = quad.target();
    assert!(t[0] < -1e-3 && t[1].abs() < 1e-6, "pan moved {t:?}");
    assert!(quad.wheel(&mut persp, tx, ty, -120.0, w, h).unwrap());
    assert!(quad.zoom() > 1.05, "zoom {}", quad.zoom());
    // Perspective pane: top-right.
    let before = persp.read_position();
    let (px_, py_) = (w * 0.75, h * 0.25);
    assert_eq!(
        quad.pointer_down(&mut persp, px_, py_, 0, w, h).unwrap().as_deref(),
        Some("perspective")
    );
    let zoom = quad.zoom();
    assert!(quad.pointer_move(&mut persp, px_ + 12.0, py_ + 3.0).unwrap());
    quad.pointer_up();
    let after = persp.read_position();
    assert!(
        (after.x() - before.x()).abs() + (after.z() - before.z()).abs() > 0.05,
        "orbit"
    );
    assert_eq!(quad.zoom(), zoom, "orbit leaves the ortho panes alone");
    // Double-click maximises the side pane (bottom-right), and restores.
    assert_eq!(quad.dblclick(w * 0.75, h * 0.75, w, h).as_deref(), Some("side"));
    assert_eq!(quad.maximized().as_deref(), Some("side"));
    quad.dblclick(5.0, 5.0, w, h);
    assert_eq!(quad.maximized(), None);
}

/// three.js `renderer.shadowMap.autoUpdate` / `needsUpdate` reach wasm.
#[wasm_bindgen_test]
async fn shadow_map_flags_round_trip() {
    let (shared, _canvas) = super::views_renderer().await;
    let mut guard = shared.borrow_mut();
    let renderer: &mut WebRenderer = &mut guard;
    assert!(renderer.shadow_map_auto_update());
    renderer.set_shadow_map_auto_update(false);
    renderer.set_shadow_map_needs_update(true);
    assert!(!renderer.shadow_map_auto_update());
    assert!(renderer.shadow_map_needs_update());
    renderer.set_shadow_map_auto_update(true);
    renderer.set_shadow_map_needs_update(false);
}

// ---------------------------------------------------------------------------
// WebRaycaster: three.js-shaped hit objects out of the wasm raycaster. No
// drawing — these run the same on either backend.
// ---------------------------------------------------------------------------

use threers::wasm::{WebEuler, WebRaycaster, WebVector3};

/// `hit[key]`.
fn field(hit: &JsValue, key: &str) -> JsValue {
    js_sys::Reflect::get(hit, &key.into()).unwrap()
}

fn num(hit: &JsValue, key: &str) -> f64 {
    field(hit, key)
        .as_f64()
        .unwrap_or_else(|| panic!("{key} is not a number"))
}

/// `hit[key].x` and so on.
fn xyz(hit: &JsValue, key: &str) -> [f64; 3] {
    let v = field(hit, key);
    ["x", "y", "z"].map(|k| field(&v, k).as_f64().unwrap_or(f64::NAN))
}

fn absent(hit: &JsValue, keys: &[&str]) {
    for k in keys {
        assert!(field(hit, k).is_undefined(), "{k} should be undefined");
    }
}

fn hits(
    raycaster: &WebRaycaster,
    scene: &mut WebScene,
    ids: Vec<f64>,
    recursive: bool,
) -> Vec<JsValue> {
    raycaster
        .intersect_objects(scene, ids, Some(recursive))
        .expect("intersectObjects")
        .iter()
        .collect()
}

fn ray_down_z(x: f32, y: f32) -> WebRaycaster {
    let mut r = WebRaycaster::new();
    r.set(
        &WebVector3::new(x, y, 5.0),
        &WebVector3::new(0.0, 0.0, -2.0),
    );
    r
}

#[wasm_bindgen_test]
fn raycaster_reports_three_js_mesh_fields() {
    let mut scene = WebScene::new();
    let plane = scene.add(&WebMesh::new(
        &WebGeometry::plane(2.0, 2.0),
        &WebMaterial::basic(&WebColor::new(1.0, 1.0, 1.0)),
    ));
    let r = ray_down_z(0.5, -0.25);
    assert!((r.direction().z() + 1.0).abs() < 1e-6, "set() normalises");
    let root = scene.root_id();
    let found = r.intersect_object(&mut scene, root, None).unwrap();
    assert_eq!(found.length(), 1);
    let h = found.get(0);
    assert!((num(&h, "distance") - 5.0).abs() < 1e-5);
    let p = xyz(&h, "point");
    assert!((p[0] - 0.5).abs() < 1e-5 && (p[1] + 0.25).abs() < 1e-5 && p[2].abs() < 1e-5);
    assert_eq!(num(&h, "objectId"), plane.id());
    assert!(scene.handle_of(num(&h, "objectId")).is_some());
    // PlaneGeometry: two triangles, one quad, uv over [0, 1]².
    let face = field(&h, "face");
    let face_index = num(&h, "faceIndex");
    assert!(face_index == 0.0 || face_index == 1.0);
    for k in ["a", "b", "c", "materialIndex"] {
        assert!(field(&face, k).as_f64().is_some(), "face.{k}");
    }
    assert_eq!(num(&face, "materialIndex"), 0.0);
    let fnormal = xyz(&face, "normal");
    assert!((fnormal[2] - 1.0).abs() < 1e-5, "face normal {fnormal:?}");
    let n = xyz(&h, "normal");
    assert!((n[2] - 1.0).abs() < 1e-5, "normal faces the ray: {n:?}");
    let uv = field(&h, "uv");
    assert!((num(&uv, "x") - 0.75).abs() < 1e-5 && (num(&uv, "y") - 0.375).abs() < 1e-5);
    let w = xyz(&h, "barycoord");
    assert!((w[0] + w[1] + w[2] - 1.0).abs() < 1e-5);
    assert_eq!(num(&h, "polygonIndex"), 0.0);
    absent(&h, &["uv1", "instanceId", "index", "distanceToRay"]);
}

#[wasm_bindgen_test]
fn raycaster_honours_material_side() {
    let mut scene = WebScene::new();
    let mut back = WebMaterial::basic(&WebColor::new(1.0, 1.0, 1.0));
    back.set_side(1);
    let mut double = WebMaterial::basic(&WebColor::new(1.0, 1.0, 1.0));
    double.set_side(2);
    let front = scene.add(&WebMesh::new(
        &WebGeometry::plane(1.0, 1.0),
        &WebMaterial::basic(&WebColor::new(1.0, 1.0, 1.0)),
    ));
    let back = scene.add(&WebMesh::new(&WebGeometry::plane(1.0, 1.0), &back));
    let double = scene.add(&WebMesh::new(&WebGeometry::plane(1.0, 1.0), &double));
    // From the front (+z): front and double-sided hit; from behind, back and
    // double-sided.
    let mut r = WebRaycaster::new();
    for (z, dz, expect) in [
        (5.0, -1.0, [true, false, true]),
        (-5.0, 1.0, [false, true, true]),
    ] {
        // Off the plane's diagonal: a ray along a shared edge hits both triangles.
        r.set(
            &WebVector3::new(0.1, 0.3, z),
            &WebVector3::new(0.0, 0.0, dz),
        );
        for (handle, hit) in [&front, &back, &double].into_iter().zip(expect) {
            let n = hits(&r, &mut scene, vec![handle.id()], false).len();
            assert_eq!(n, hit as usize, "z = {z}, object {}", handle.id());
        }
    }
}

#[wasm_bindgen_test]
fn raycaster_hits_lines_points_sprites_and_instances() {
    let mut scene = WebScene::new();
    let white = WebColor::new(1.0, 1.0, 1.0);
    // A segment along x at y = 0.3, z = 0.
    let mut seg = WebBufferGeometry::new();
    seg.set_attribute(
        "position",
        WebBufferAttribute::new(vec![-1.0, 0.3, 0.0, 1.0, 0.3, 0.0], 3),
    );
    let line = scene.add_line_segments(&seg, &WebMaterial::line(&white));
    // Two points.
    let mut pts = WebBufferGeometry::new();
    pts.set_attribute(
        "position",
        WebBufferAttribute::new(vec![3.0, 0.0, 0.0, 3.2, 0.1, 0.0], 3),
    );
    let points = scene.add_points(&pts, &WebMaterial::points(&white, 1.0));
    // Three instances of a plane along x at 6, 7, 8.
    let mut transforms = Vec::new();
    for i in 0..3 {
        let mut m = [0.0f32; 16];
        m[0] = 1.0;
        m[5] = 1.0;
        m[10] = 1.0;
        m[15] = 1.0;
        m[12] = 6.0 + i as f32;
        transforms.extend_from_slice(&m);
    }
    let inst = scene.add_instanced_mesh(
        &WebGeometry::plane(0.8, 0.8),
        &WebMaterial::basic(&white),
        transforms,
    );

    let mut r = WebRaycaster::new();
    r.set_line_threshold(0.1);
    r.set_points_threshold(0.15);
    // The line, 0.05 off its segment.
    r.set(
        &WebVector3::new(0.2, 0.35, 5.0),
        &WebVector3::new(0.0, 0.0, -1.0),
    );
    let h = hits(&r, &mut scene, vec![line.id()], false);
    assert_eq!(h.len(), 1, "line within the threshold");
    assert_eq!(num(&h[0], "index"), 0.0);
    let p = xyz(&h[0], "point");
    assert!(
        (p[1] - 0.3).abs() < 1e-5,
        "the point is on the segment: {p:?}"
    );
    absent(
        &h[0],
        &["face", "faceIndex", "uv", "distanceToRay", "instanceId"],
    );
    r.set_line_threshold(0.01);
    assert!(
        hits(&r, &mut scene, vec![line.id()], false).is_empty(),
        "outside the threshold"
    );
    assert!((r.line_threshold() - 0.01).abs() < 1e-9);

    // The second point, 0.1 off the ray.
    r.set(
        &WebVector3::new(3.2, 0.0, 5.0),
        &WebVector3::new(0.0, 0.0, -1.0),
    );
    let h = hits(&r, &mut scene, vec![points.id()], false);
    assert_eq!(h.len(), 1);
    assert_eq!(num(&h[0], "index"), 1.0);
    assert!((num(&h[0], "distanceToRay") - 0.1).abs() < 1e-5);
    absent(&h[0], &["face", "faceIndex", "instanceId"]);

    // The middle instance.
    r.set(
        &WebVector3::new(7.1, 0.1, 5.0),
        &WebVector3::new(0.0, 0.0, -1.0),
    );
    let root = scene.root_id();
    let h = hits(&r, &mut scene, vec![root], true);
    assert_eq!(h.len(), 1);
    assert_eq!(num(&h[0], "instanceId"), 1.0);
    assert_eq!(num(&h[0], "objectId"), inst.id());

    // A sprite needs the camera: setFromCamera supplies it.
    let mut scene = WebScene::new();
    let sprite = scene.add_sprite(&WebMaterial::sprite(&white));
    let mut cam = WebCamera::perspective(50.0, 1.0, 0.1, 100.0);
    cam.set_position(0.0, 0.0, 5.0);
    cam.look_at(0.0, 0.0, 0.0);
    let mut r = WebRaycaster::new();
    r.set(
        &WebVector3::new(0.1, 0.1, 5.0),
        &WebVector3::new(0.0, 0.0, -1.0),
    );
    assert!(
        hits(&r, &mut scene, vec![sprite.id()], false).is_empty(),
        "no camera, no sprite hits"
    );
    r.set_from_camera(0.0, 0.0, &cam);
    let h = hits(&r, &mut scene, vec![sprite.id()], false);
    assert_eq!(h.len(), 1);
    let uv = field(&h[0], "uv");
    assert!((num(&uv, "x") - 0.5).abs() < 1e-4 && (num(&uv, "y") - 0.5).abs() < 1e-4);
    assert!((num(&h[0], "distance") - 5.0).abs() < 1e-4);
    absent(&h[0], &["face", "faceIndex", "normal"]);
}

#[wasm_bindgen_test]
fn raycaster_lists_near_far_recursion_and_first_hit() {
    let mut scene = WebScene::new();
    let white = WebColor::new(1.0, 1.0, 1.0);
    let group = scene.add_group();
    let mut double = WebMaterial::basic(&white);
    double.set_side(2);
    let mesh = WebMesh::new(&WebGeometry::plane(1.0, 1.0), &double);
    // Three stacked planes under the group, at z = 0, -1, -2.
    let mut layers = Vec::new();
    for z in [0.0, -1.0, -2.0] {
        let h = scene.add_mesh_to(&group, &mesh);
        scene.set_transform(
            &h,
            &WebVector3::new(0.0, 0.0, z),
            &WebEuler::new(0.0, 0.0, 0.0),
        );
        layers.push(h);
    }
    let root = scene.root_id();
    let mut r = ray_down_z(0.1, 0.3);
    let all = hits(&r, &mut scene, vec![group.id()], true);
    let order: Vec<f64> = all.iter().map(|h| num(h, "objectId")).collect();
    assert_eq!(
        order,
        layers.iter().map(|h| h.id()).collect::<Vec<_>>(),
        "nearest first"
    );
    assert!(
        hits(&r, &mut scene, vec![group.id()], false).is_empty(),
        "a group has no geometry"
    );
    // A list of objects, sorted together.
    let two = hits(&r, &mut scene, vec![layers[2].id(), layers[0].id()], false);
    assert_eq!(num(&two[0], "objectId"), layers[0].id());
    // near / far keep the middle one only.
    r.set_near(5.5);
    r.set_far(6.5);
    let mid = hits(&r, &mut scene, vec![root], true);
    assert_eq!(mid.len(), 1);
    assert_eq!(num(&mid[0], "objectId"), layers[1].id());
    // firstHitOnly: just the nearest, across objects.
    r.set_near(0.0);
    r.set_far(f32::INFINITY);
    r.set_first_hit_only(true);
    let first = hits(&r, &mut scene, vec![root], true);
    assert_eq!(first.len(), 1);
    assert_eq!(num(&first[0], "objectId"), layers[0].id());
    // A bad id is an error, not an empty list.
    assert!(r
        .intersect_objects(&mut scene, vec![123456789.0], None)
        .is_err());
}

/// Transparent surfaces blend on the canvas as three.js's default renderer
/// blends them: on the canvas's sRGB-*encoded* bytes, so 50% red over blue is
/// (128, 0, 128) — not the (188, 0, 188) a blend in linear light gives. A
/// WebGPU canvas is `Unorm` and the shader encodes, so its hardware blend is
/// already three.js's; a WebGL2 canvas is sRGB and cannot be viewed as
/// anything else, so its blended draws have to get there another way.
#[wasm_bindgen_test]
async fn transparent_surfaces_blend_on_the_canvas_as_three_js_does() {
    let (shared, canvas) = super::views_renderer().await;
    let mut guard = shared.borrow_mut();
    let renderer: &mut WebRenderer = &mut guard;
    reset(renderer);
    let mut scene = WebScene::new();
    scene.set_background(&WebColor::new(0.0, 0.0, 0.0));
    let blue = WebMaterial::basic(&WebColor::new(0.0, 0.0, 1.0));
    let back = scene.add(&WebMesh::new(&WebGeometry::plane(4.0, 4.0), &blue));
    scene.set_transform(
        &back,
        &threers::wasm::WebVector3::new(0.0, 0.0, -0.5),
        &threers::wasm::WebEuler::new(0.0, 0.0, 0.0),
    );
    let mut red = WebMaterial::basic(&WebColor::new(1.0, 0.0, 0.0));
    red.set_transparent(true);
    red.set_opacity(0.5);
    let front = scene.add(&WebMesh::new(&WebGeometry::plane(4.0, 4.0), &red));
    scene.set_transform(
        &front,
        &threers::wasm::WebVector3::new(0.0, 0.0, 0.5),
        &threers::wasm::WebEuler::new(0.0, 0.0, 0.0),
    );
    renderer.render(&mut scene, &front_ortho());
    let pixels = read_canvas(&canvas);
    let (cw, ch) = (canvas.width(), canvas.height());
    let got = px(&pixels, cw, ch, cw / 2, ch / 2);
    let want = [128u8, 0, 128];
    let err = (0..3).map(|k| (got[k] as i32 - want[k] as i32).abs()).max().unwrap();
    assert!(
        err <= 1,
        "{}: 50% red over blue on the canvas is {got:?}; three.js writes {want:?}",
        renderer.backend()
    );
}
