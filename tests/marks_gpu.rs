//! Marks on a real GPU: each kind lands where, and as large as, it was asked
//! to, hides behind the frame's geometry, and picks back to its own row.
//!
//! The camera is orthographic with one world unit to a pixel and the origin in
//! the middle of a 256×128 target, so a mark at world `(x, y)` is centred on
//! pixel `(128 + x, 64 - y)` and every expected pixel below is arithmetic, not
//! a reference image.
//!
//! Skipped when no GPU adapter is available.

use threers::marks::{
    DensityMarks, Mark, MarkRenderer, MarkView, Origin, PointMarks, SegmentMarks, ShaderHooks,
    ShapeMarks, SizeUnits,
};
use threers::materials::{BasicMaterial, Material};
use threers::{
    BoxGeometry, Color, HeadlessRenderer, OrthographicCamera, PerspectiveCamera, PolygonGeometry,
    Scene,
};

const W: u32 = 256;
const H: u32 = 128;

fn host() -> Option<HeadlessRenderer> {
    HeadlessRenderer::builder().size(W, H).build().ok()
}

fn camera(half_w: f32, half_h: f32) -> OrthographicCamera {
    let mut c = OrthographicCamera::new(-half_w, half_w, half_h, -half_h, 0.1, 100.0);
    c.position = threers::Vector3::new(0.0, 0.0, 10.0);
    c
}

fn scene() -> Scene {
    let mut s = Scene::new();
    s.background = Color::BLACK;
    s
}

fn px(rgba: &[u8], x: u32, y: u32) -> [u8; 4] {
    let o = ((y * W + x) * 4) as usize;
    [rgba[o], rgba[o + 1], rgba[o + 2], rgba[o + 3]]
}

fn is_bg(p: [u8; 4]) -> bool {
    p[0] < 8 && p[1] < 8 && p[2] < 8
}

/// Render `scene`, draw `marks` over it, read back.
fn frame(
    hr: &mut HeadlessRenderer,
    marks_renderer: &mut MarkRenderer,
    scene: &mut Scene,
    cam: &OrthographicCamera,
    origin: Origin,
    marks: &mut [&mut dyn Mark],
) -> Vec<u8> {
    hr.render(scene, cam);
    let device = hr.device().clone();
    let queue = hr.queue().clone();
    let view = MarkView::new(cam, origin, (W, H));
    let target = hr.mark_target();
    marks_renderer.draw(&device, &queue, &target, &view, marks);
    hr.read_rgba()
}

fn two_points() -> PointMarks {
    let mut p = PointMarks::new();
    p.set_positions(&[[-64.0, 0.0, 0.0], [64.0, 0.0, 0.0]]);
    p.set_radii(&[10.0, 4.0]);
    p.set_colors(&[[255, 0, 0, 255], [0, 255, 0, 255]]);
    p
}

#[test]
fn points_land_where_and_as_large_as_asked() {
    let Some(mut hr) = host() else { return };
    let mut mr = MarkRenderer::new(hr.device());
    let mut points = two_points();
    let img = frame(
        &mut hr,
        &mut mr,
        &mut scene(),
        &camera(128.0, 64.0),
        Origin::ZERO,
        &mut [&mut points],
    );
    assert_eq!(px(&img, 64, 64), [255, 0, 0, 255]);
    assert_eq!(px(&img, 64 + 8, 64)[0], 255, "inside the 10 px radius");
    assert!(is_bg(px(&img, 64 + 12, 64)), "outside it");
    assert_eq!(px(&img, 192, 64)[1], 255);
    assert!(is_bg(px(&img, 192 + 6, 64)));

    // World units: halve the view and a radius of 5 is 10 pixels.
    points.style_mut().units = SizeUnits::World;
    points.set_radii(&[5.0, 5.0]);
    points.set_positions(&[[-32.0, 0.0, 0.0], [32.0, 0.0, 0.0]]);
    let img = frame(
        &mut hr,
        &mut mr,
        &mut scene(),
        &camera(64.0, 32.0),
        Origin::ZERO,
        &mut [&mut points],
    );
    assert_eq!(px(&img, 64 + 8, 64)[0], 255);
    assert!(is_bg(px(&img, 64 + 12, 64)));
}

#[test]
fn a_partial_update_moves_only_that_row() {
    let Some(mut hr) = host() else { return };
    let mut mr = MarkRenderer::new(hr.device());
    let mut points = two_points();
    let cam = camera(128.0, 64.0);
    frame(
        &mut hr,
        &mut mr,
        &mut scene(),
        &cam,
        Origin::ZERO,
        &mut [&mut points],
    );
    points.update_positions(1, &[[64.0, 40.0, 0.0]]);
    let img = frame(
        &mut hr,
        &mut mr,
        &mut scene(),
        &cam,
        Origin::ZERO,
        &mut [&mut points],
    );
    assert!(is_bg(px(&img, 192, 64)), "row 1 left its old place");
    assert_eq!(px(&img, 192, 24)[1], 255, "and arrived at its new one");
    assert_eq!(px(&img, 64, 64)[0], 255, "row 0 did not move");
}

#[test]
fn marks_hide_behind_the_frames_geometry() {
    let Some(mut hr) = host() else { return };
    let mut mr = MarkRenderer::new(hr.device());
    let mut s = scene();
    // A blue wall at z = 0 over the left half.
    let mut wall = BoxGeometry::new(100.0, 100.0, 1.0);
    for v in wall
        .attributes
        .get_mut("position")
        .unwrap()
        .array
        .chunks_mut(3)
    {
        v[0] -= 64.0;
    }
    s.add_mesh(
        wall,
        Material::Basic(BasicMaterial::new(Color::from_hex(0x0000ff))),
    );
    let mut points = PointMarks::new();
    // Behind the wall by more than `depth_lift` brings it forward (two
    // radii), and in front of it.
    points.set_positions(&[[-64.0, 0.0, -20.0], [-64.0, 30.0, 5.0]]);
    points.set_radii(&[6.0, 6.0]);
    points.set_default_color([255, 0, 0, 255]);
    let cam = camera(128.0, 64.0);
    let img = frame(
        &mut hr,
        &mut mr,
        &mut s,
        &cam,
        Origin::ZERO,
        &mut [&mut points],
    );
    let hidden = px(&img, 64, 64);
    assert!(
        hidden[2] > 200 && hidden[0] < 20,
        "behind the wall: {hidden:?}"
    );
    assert_eq!(px(&img, 64, 34)[0], 255, "in front of it");

    // Without the depth test both show.
    points.style_mut().depth_test = false;
    let img = frame(
        &mut hr,
        &mut mr,
        &mut s,
        &cam,
        Origin::ZERO,
        &mut [&mut points],
    );
    assert_eq!(px(&img, 64, 64)[0], 255);
}

#[test]
fn positions_far_from_zero_keep_sub_unit_spacing() {
    let Some(mut hr) = host() else { return };
    let mut mr = MarkRenderer::new(hr.device());
    // Ten million units out, f32 steps are a whole unit; these are a quarter
    // apart. Two world units across the screen: 128 px a unit, 32 px a step.
    let base = 10_000_000.0;
    let mut points = PointMarks::new();
    let rows: Vec<[f64; 3]> = (0..4).map(|k| [base + k as f64 * 0.25, 0.0, 0.0]).collect();
    points.set_positions(&rows);
    points.set_radii(&[3.0; 4]);
    points.set_default_color([255, 255, 255, 255]);
    let origin = Origin::new(base, 0.0, 0.0);
    let img = frame(
        &mut hr,
        &mut mr,
        &mut scene(),
        &camera(1.0, 0.5),
        origin,
        &mut [&mut points],
    );
    for k in 0..4 {
        let x = 128 + k * 32;
        assert_eq!(px(&img, x, 64)[0], 255, "row {k} at x = {x}");
        assert!(is_bg(px(&img, x + 16, 64)), "gap after row {k}");
    }
}

#[test]
fn a_range_hook_filters_on_the_gpu() {
    let Some(mut hr) = host() else { return };
    let mut mr = MarkRenderer::new(hr.device());
    let mut points = PointMarks::new();
    points.set_positions(&[[-64.0, 0.0, 0.0], [0.0, 0.0, 0.0], [64.0, 0.0, 0.0]]);
    points.set_radii(&[5.0; 3]);
    points.set_default_color([255, 255, 255, 255]);
    points.common.set_custom(0, 1, &[0.0, 5.0, 10.0]);
    points.common.hooks = ShaderHooks::range_filter(0, 0);
    points.style_mut().params[0] = [4.0, 6.0, 0.0, 0.0];
    let cam = camera(128.0, 64.0);
    let img = frame(
        &mut hr,
        &mut mr,
        &mut scene(),
        &cam,
        Origin::ZERO,
        &mut [&mut points],
    );
    assert!(is_bg(px(&img, 64, 64)));
    assert_eq!(px(&img, 128, 64)[0], 255);
    assert!(is_bg(px(&img, 192, 64)));
    // Moving the range is a uniform write: same pipeline, different rows.
    let pipelines = mr.pipeline_count();
    points.style_mut().params[0] = [9.0, 11.0, 0.0, 0.0];
    let img = frame(
        &mut hr,
        &mut mr,
        &mut scene(),
        &cam,
        Origin::ZERO,
        &mut [&mut points],
    );
    assert!(is_bg(px(&img, 128, 64)));
    assert_eq!(px(&img, 192, 64)[0], 255);
    assert_eq!(mr.pipeline_count(), pipelines);
}

/// Both custom columns, four wide, on the kinds with the most columns of their
/// own: a pipeline over WebGPU's eight vertex buffers fails to build.
#[test]
fn every_kind_takes_both_custom_columns() {
    let Some(mut hr) = host() else { return };
    let mut mr = MarkRenderer::new(hr.device());
    let cam = camera(128.0, 64.0);
    // Filter on custom1.y: row 0 has 0 and is hidden, row 1 has 5 and is kept.
    let custom = |c: &mut threers::marks::MarkCommon| {
        c.set_custom(0, 4, &[0.0; 8]);
        c.set_custom(1, 4, &[0.0, 0.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0]);
        c.hooks = ShaderHooks::range_filter(5, 0);
        c.style.params[0] = [4.0, 6.0, 0.0, 0.0];
    };

    let mut paths = SegmentMarks::new();
    paths.set_paths(&[
        vec![[-100.0, 30.0, 0.0], [-20.0, 30.0, 0.0], [-20.0, 50.0, 0.0]],
        vec![[20.0, -30.0, 0.0], [100.0, -30.0, 0.0]],
    ]);
    paths.set_widths(&[6.0; 3]);
    // Per segment: the first path is rows 0 and 1, the second row 2.
    paths.common.set_custom(0, 4, &[0.0; 12]);
    paths.common.set_custom(
        1,
        4,
        &[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0],
    );
    paths.common.hooks = ShaderHooks::range_filter(5, 0);
    paths.common.style.params[0] = [4.0, 6.0, 0.0, 0.0];

    let mut boxes = ShapeMarks::new(&BoxGeometry::new(10.0, 10.0, 10.0));
    boxes.set_positions(&[[-60.0, -30.0, 0.0], [60.0, 30.0, 0.0]]);
    custom(&mut boxes.common);

    let img = frame(
        &mut hr,
        &mut mr,
        &mut scene(),
        &cam,
        Origin::ZERO,
        &mut [&mut paths, &mut boxes],
    );
    assert!(
        is_bg(px(&img, 128 - 60, 64 - 30)),
        "first path filtered out"
    );
    assert!(px(&img, 128 + 60, 64 + 30)[0] > 200, "second path drawn");
    assert!(is_bg(px(&img, 128 - 60, 64 + 30)), "first box filtered out");
    assert!(!is_bg(px(&img, 128 + 60, 64 - 30)), "second box drawn");
}

/// A row's rotation turns its mesh: a long box a quarter turn about the
/// view axis stands upright.
#[test]
fn shapes_turn_by_their_rotation() {
    let Some(mut hr) = host() else { return };
    let mut mr = MarkRenderer::new(hr.device());
    let cam = camera(128.0, 64.0);
    let mut bar = ShapeMarks::new(&BoxGeometry::new(60.0, 6.0, 6.0));
    bar.set_positions(&[[0.0, 0.0, 0.0]]);
    bar.set_default_color([255, 255, 255, 255]);
    bar.ambient = 1.0;
    let img = frame(&mut hr, &mut mr, &mut scene(), &cam, Origin::ZERO, &mut [&mut bar]);
    assert!(!is_bg(px(&img, 128 + 25, 64)) && is_bg(px(&img, 128, 64 - 25)), "lying down");
    let h = std::f32::consts::FRAC_1_SQRT_2;
    bar.set_rotations(&[[0.0, 0.0, h, h]]);
    let img = frame(&mut hr, &mut mr, &mut scene(), &cam, Origin::ZERO, &mut [&mut bar]);
    assert!(is_bg(px(&img, 128 + 25, 64)) && !is_bg(px(&img, 128, 64 - 25)), "standing up");
}

/// A line flat on the ground, seen from above at a slant, is foreshortened
/// across its width as the ground is; facing the camera, it keeps its width.
#[test]
fn flat_lines_lie_on_the_ground() {
    let Some(mut hr) = host() else { return };
    let mut mr = MarkRenderer::new(hr.device());
    // Looking down at the ground (world y up) from 30° above it.
    let mut cam = camera(128.0, 64.0);
    let (s, c) = 30f32.to_radians().sin_cos();
    cam.position = threers::Vector3::new(0.0, 50.0 * s, 50.0 * c);
    threers::Camera::look_at(&mut cam, threers::Vector3::new(0.0, 0.0, 0.0));
    let mut thickness = |flat: bool| {
        let mut line = SegmentMarks::new();
        line.set_segments(&[[-80.0, 0.0, 0.0]], &[[80.0, 0.0, 0.0]]);
        line.set_widths(&[20.0]);
        line.set_default_color([255, 255, 255, 255]);
        line.common.style.units = SizeUnits::World;
        line.flat = flat;
        let img = frame(&mut hr, &mut mr, &mut scene(), &cam, Origin::ZERO, &mut [&mut line]);
        (0..H).filter(|&y| px(&img, 128, y)[0] > 128).count() as f32
    };
    let (standing, flat) = (thickness(false), thickness(true));
    assert!((standing - 20.0).abs() <= 1.5, "facing the camera: {standing} px");
    // sin 30° of the width.
    assert!((flat - 10.0).abs() <= 1.5, "on the ground: {flat} px");
}

/// A whole path lights up, not one of its segments.
#[test]
fn a_whole_path_highlights() {
    let Some(mut hr) = host() else { return };
    let mut mr = MarkRenderer::new(hr.device());
    let cam = camera(128.0, 64.0);
    let mut paths = SegmentMarks::new();
    paths.set_paths(&[
        vec![[-100.0, 30.0, 0.0], [-50.0, 30.0, 0.0], [0.0, 30.0, 0.0]],
        vec![[-100.0, -30.0, 0.0], [-50.0, -30.0, 0.0], [0.0, -30.0, 0.0], [50.0, -30.0, 0.0]],
    ]);
    paths.set_widths(&[6.0; 5]);
    paths.set_default_color([0, 0, 255, 255]);
    paths.common.style.highlight_color = [255, 0, 0, 255];
    paths.highlight_paths(Some(1..2));
    let img = frame(&mut hr, &mut mr, &mut scene(), &cam, Origin::ZERO, &mut [&mut paths]);
    // Every segment of the second path is red; the first stays blue.
    for x in [-75, -25, 25] {
        assert!(px(&img, (128 + x) as u32, 64 + 30)[0] > 200, "second path at {x}");
    }
    assert!(px(&img, 128 - 75, 64 - 30)[2] > 200 && px(&img, 128 - 75, 64 - 30)[0] < 50, "first path");
}

/// Each point its own stroke: colour and width per row.
#[test]
fn points_take_their_own_strokes() {
    let Some(mut hr) = host() else { return };
    let mut mr = MarkRenderer::new(hr.device());
    let cam = camera(128.0, 64.0);
    let mut p = PointMarks::new();
    p.set_positions(&[[-50.0, 0.0, 0.0], [50.0, 0.0, 0.0]]);
    p.set_radii(&[20.0, 20.0]);
    p.set_default_color([0, 0, 0, 255]);
    p.stroke_width = 4.0;
    p.set_stroke_colors(&[[255, 0, 0, 255], [0, 255, 0, 255]]);
    p.set_stroke_widths(&[1.0, 3.0]);
    let img = frame(&mut hr, &mut mr, &mut scene(), &cam, Origin::ZERO, &mut [&mut p]);
    // The left ring: red, 4 px wide about r = 20; the right: green, 12 px.
    assert!(px(&img, 128 - 50 + 20, 64)[0] > 200, "red ring");
    assert!(px(&img, 128 - 50 + 24, 64)[0] < 30, "red ring is thin");
    assert!(px(&img, 128 + 50 + 20, 64)[1] > 200, "green ring");
    assert!(px(&img, 128 + 50 + 24, 64)[1] > 200, "green ring is wide");
}

/// A mesh textured from an atlas by its uvs, and tinted by its vertex
/// colours.
#[test]
fn shapes_take_textures_and_vertex_colours() {
    use threers::marks::TextureAtlas;
    let Some(mut hr) = host() else { return };
    let mut mr = MarkRenderer::new(hr.device());
    let cam = camera(128.0, 64.0);
    // Left half red, right half green.
    let mut img = Vec::new();
    for _y in 0..4 {
        for x in 0..4 {
            img.extend_from_slice(if x < 2 { &[255, 0, 0, 255] } else { &[0, 255, 0, 255] });
        }
    }
    let mut atlas = TextureAtlas::new(64);
    atlas.insert("halves", 4, 4, &img).unwrap();
    let atlas = atlas.shared();
    // A quad in the view plane, uvs across it.
    let mut quad = threers::BufferGeometry::new();
    let p = [[-1.0f32, -1.0, 0.0], [1.0, -1.0, 0.0], [1.0, 1.0, 0.0], [-1.0, 1.0, 0.0]];
    quad.set_attribute("position", threers::BufferAttribute::new(p.iter().flatten().copied().collect(), 3));
    quad.set_attribute("uv", threers::BufferAttribute::new(vec![0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0], 2));
    quad.set_index(vec![0, 1, 2, 0, 2, 3]);
    let mut shapes = ShapeMarks::new(&quad);
    shapes.set_positions(&[[0.0, 0.0, 0.0]]);
    shapes.set_default_color([255, 255, 255, 255]);
    shapes.ambient = 1.0;
    shapes.common.style.size_scale = 40.0;
    shapes.set_texture(Some((atlas, "halves".into())));
    let img = frame(&mut hr, &mut mr, &mut scene(), &cam, Origin::ZERO, &mut [&mut shapes]);
    let (l, r) = (px(&img, 128 - 20, 64), px(&img, 128 + 20, 64));
    assert!(l[0] > 200 && l[1] < 60, "left: {l:?}");
    assert!(r[1] > 200 && r[0] < 60, "right: {r:?}");
    // Vertex colours: blue everywhere turns the green half dark, the red
    // half too.
    let mut tinted = quad.clone();
    tinted.set_attribute("color", threers::BufferAttribute::new([0.0f32, 0.0, 1.0].repeat(4), 3));
    shapes.set_geometry(&tinted);
    let img = frame(&mut hr, &mut mr, &mut scene(), &cam, Origin::ZERO, &mut [&mut shapes]);
    assert!(px(&img, 128 + 20, 64)[1] < 30, "tinted out");
}

/// Points binned into screen cells on the GPU, counts and weights, by the
/// pixel each projects to.
#[test]
fn points_bin_into_screen_cells_on_the_gpu() {
    use threers::marks::ScreenBinner;
    let Some(hr) = host() else { return };
    let device = hr.device().clone();
    let queue = hr.queue().clone();
    if !ScreenBinner::supported(&device) {
        return;
    }
    let cam = camera(128.0, 64.0);
    let view = MarkView::new(&cam, Origin::ZERO, (W, H));
    // 1 unit a pixel, (0, 0) at the middle (128, 64): cells of 32 pixels.
    let points = [[-100.0, 0.0, 0.0], [-99.0, -1.0, 0.0], [10.0, 10.0, 0.0], [500.0, 0.0, 0.0]];
    let binner = ScreenBinner::new(&device, &points, Some(&[1.0, 2.5, 4.0, 8.0]));
    let bins = binner.bin(&device, &queue, view.view_proj, (W, H), 32.0).wait(&device).unwrap();
    assert_eq!((bins.cols, bins.rows), (8, 4));
    // (-100, 0) is pixel (28, 64): cell (0, 2); (10, 10) is (138, 54): (4, 1).
    assert_eq!(bins.counts[2 * 8], 2);
    assert!((bins.sums[2 * 8] - 3.5).abs() < 1e-3);
    assert_eq!(bins.counts[8 + 4], 1);
    // Off screen: in no cell.
    assert_eq!(bins.counts.iter().sum::<u32>(), 3);
}

/// Every layer's uniforms share one buffer: each reads its own slot — a pick
/// reports the layer id from its uniforms, and the size comes from them too —
/// through the arena growing and shrinking back.
#[test]
fn each_layer_reads_its_own_uniforms() {
    let Some(mut hr) = host() else { return };
    let mut mr = MarkRenderer::new(hr.device());
    let device = hr.device().clone();
    let queue = hr.queue().clone();
    let cam = camera(128.0, 64.0);
    for n in [5usize, 60, 5] {
        let mut layers: Vec<PointMarks> = (0..n)
            .map(|i| {
                let mut p = PointMarks::new();
                p.set_positions(&[[-118.0 + i as f64 * 4.0, 0.0, 0.0]]);
                p.set_radii(&[1.0]);
                p.set_default_color([255, 255, 255, 255]);
                // Every other layer twice the size, from the uniforms alone.
                p.style_mut().size_scale = if i % 2 == 0 { 1.0 } else { 2.0 };
                p
            })
            .collect();
        let mut refs: Vec<&mut dyn Mark> = layers.iter_mut().map(|l| l as &mut dyn Mark).collect();
        let view = MarkView::new(&cam, Origin::ZERO, (W, H));
        for i in 0..n {
            let x = 128.0 - 118.0 + i as f32 * 4.0 + 0.5;
            let hit = mr
                .pick(&device, &queue, &view, &mut refs, None, x, 64.5, 0)
                .wait(&device)
                .unwrap_or_else(|| panic!("{n} layers: layer {i} missed"));
            assert_eq!(hit.mark, i, "{n} layers");
        }
        let img = frame(
            &mut hr,
            &mut mr,
            &mut scene(),
            &cam,
            Origin::ZERO,
            &mut refs,
        );
        for i in 0..n {
            let x = 10 + i as u32 * 4;
            // 1.5 px out: inside a 2 px radius, outside a 1 px one.
            let lit = !is_bg(px(&img, x, 62)) || !is_bg(px(&img, x, 66));
            assert_eq!(lit, i % 2 == 1, "{n} layers: layer {i}'s size");
        }
    }
}

#[test]
fn picking_finds_the_row_under_the_cursor() {
    let Some(hr) = host() else { return };
    let mut mr = MarkRenderer::new(hr.device());
    let device = hr.device().clone();
    let queue = hr.queue().clone();
    let mut lines = SegmentMarks::new();
    lines.set_segments(&[[-100.0, 40.0, 0.0]], &[[100.0, 40.0, 0.0]]);
    lines.set_widths(&[6.0]);
    let mut points = two_points();
    let cam = camera(128.0, 64.0);
    let view = MarkView::new(&cam, Origin::ZERO, (W, H));
    let pick = |mr: &mut MarkRenderer, x: f32, y: f32, marks: &mut [&mut dyn Mark]| {
        mr.pick(&device, &queue, &view, marks, None, x, y, 2)
            .wait(&device)
    };
    let hit = pick(&mut mr, 64.5, 64.5, &mut [&mut lines, &mut points]).unwrap();
    assert_eq!((hit.mark, hit.index), (1, 0));
    // Five pixels from the 4 px point: off its edge, but within the pick radius.
    let hit = pick(&mut mr, 192.5 + 5.0, 64.5, &mut [&mut lines, &mut points]).unwrap();
    assert_eq!((hit.mark, hit.index), (1, 1));
    let hit = pick(&mut mr, 60.5, 24.5, &mut [&mut lines, &mut points]).unwrap();
    assert_eq!((hit.mark, hit.index), (0, 0));
    assert!(pick(&mut mr, 10.5, 110.5, &mut [&mut lines, &mut points]).is_none());
    points.style_mut().pickable = false;
    assert!(pick(&mut mr, 64.5, 64.5, &mut [&mut lines, &mut points]).is_none());
}

#[test]
fn a_point_resting_on_the_ground_draws_whole() {
    let Some(mut hr) = host() else { return };
    let mut mr = MarkRenderer::new(hr.device());
    let mut s = scene();
    let ground = [
        [-500.0, -500.0],
        [500.0, -500.0],
        [500.0, 500.0],
        [-500.0, 500.0],
    ];
    s.add_mesh(
        PolygonGeometry::new(&ground, &[], 0.0, 0.0),
        Material::Basic(BasicMaterial::new(Color::from_hex(0x0000ff))),
    );
    // Looking down at the origin from 30° above the ground: below the
    // point's centre on screen the ground is nearer than the point, by
    // 1.7 radii at the disc's lower edge — inside the default lift of 2.
    let mut cam = PerspectiveCamera::new(60.0, W as f32 / H as f32, 1.0, 1000.0);
    cam.position = threers::Vector3::new(0.0, 50.0, 86.6);
    cam.target = threers::Vector3::ZERO;
    let mut points = PointMarks::new();
    points.set_positions(&[[0.0, 0.0, 0.0]]);
    points.set_radii(&[10.0]);
    points.set_default_color([255, 0, 0, 255]);
    let mut draw = |hr: &mut HeadlessRenderer, points: &mut PointMarks| {
        hr.render(&mut s, &cam);
        let (device, queue) = (hr.device().clone(), hr.queue().clone());
        let view = MarkView::new(&cam, Origin::ZERO, (W, H));
        mr.draw(&device, &queue, &hr.mark_target(), &view, &mut [points]);
        hr.read_rgba()
    };

    let img = draw(&mut hr, &mut points);
    assert_eq!(px(&img, 128, 57)[0], 255, "upper half");
    assert_eq!(px(&img, 128, 71)[0], 255, "lower half");

    // Without the lift the lower half sinks into the ground.
    points.style_mut().depth_lift = 0.0;
    let img = draw(&mut hr, &mut points);
    assert_eq!(px(&img, 128, 57)[0], 255, "upper half, unlifted");
    let lower = px(&img, 128, 71);
    assert!(
        lower[0] < 20 && lower[2] > 200,
        "lower half, unlifted: {lower:?}"
    );
}

#[test]
fn picking_respects_the_frames_depth() {
    let Some(mut hr) = host() else { return };
    let mut mr = MarkRenderer::new(hr.device());
    let mut s = scene();
    let mut wall = BoxGeometry::new(100.0, 100.0, 1.0);
    for v in wall
        .attributes
        .get_mut("position")
        .unwrap()
        .array
        .chunks_mut(3)
    {
        v[0] -= 64.0;
    }
    s.add_mesh(
        wall,
        Material::Basic(BasicMaterial::new(Color::from_hex(0x0000ff))),
    );
    // Behind the wall by more than `depth_lift` brings it forward.
    let mut points = PointMarks::new();
    points.set_positions(&[[-64.0, 0.0, -20.0]]);
    points.set_radii(&[6.0]);
    let cam = camera(128.0, 64.0);
    hr.render(&mut s, &cam);
    let device = hr.device().clone();
    let queue = hr.queue().clone();
    let view = MarkView::new(&cam, Origin::ZERO, (W, H));
    let depth = &hr.target().depth_view;
    let hit = mr
        .pick(
            &device,
            &queue,
            &view,
            &mut [&mut points],
            Some(depth),
            64.5,
            64.5,
            0,
        )
        .wait(&device);
    assert!(hit.is_none(), "hidden behind the wall");
    let hit = mr
        .pick(
            &device,
            &queue,
            &view,
            &mut [&mut points],
            None,
            64.5,
            64.5,
            0,
        )
        .wait(&device);
    assert!(hit.is_some(), "found without the occluder");
}

#[test]
fn segments_are_as_wide_as_asked_and_arcs_rise() {
    let Some(mut hr) = host() else { return };
    let mut mr = MarkRenderer::new(hr.device());
    let mut line = SegmentMarks::new();
    line.set_segments(&[[-100.0, 0.0, 0.0]], &[[100.0, 0.0, 0.0]]);
    line.set_widths(&[10.0]);
    let cam = camera(128.0, 64.0);
    let img = frame(
        &mut hr,
        &mut mr,
        &mut scene(),
        &cam,
        Origin::ZERO,
        &mut [&mut line],
    );
    for dy in [-4i32, 0, 4] {
        assert_eq!(
            px(&img, 128, (64 + dy) as u32)[0],
            255,
            "dy {dy} is inside the 10 px width"
        );
    }
    assert!(is_bg(px(&img, 128, 64 + 7)));
    assert_eq!(px(&img, 128 + 103, 64)[0], 255, "round cap past the end");
    assert!(is_bg(px(&img, 128 + 107, 64)));

    // An arc a quarter as high as it is long peaks 50 px up, and does not
    // draw the chord.
    line.arc_height = 0.25;
    line.set_widths(&[4.0]);
    let img = frame(
        &mut hr,
        &mut mr,
        &mut scene(),
        &cam,
        Origin::ZERO,
        &mut [&mut line],
    );
    assert_eq!(px(&img, 128, 64 - 50)[0], 255, "the peak");
    assert!(is_bg(px(&img, 128, 64)), "no chord");
}

/// A line thinner than a pixel takes the share of each pixel it covers: a
/// 0.4-pixel line puts 0.4 of its ink in each column wherever it falls
/// between pixel centres, and never more than 0.4 in one pixel — where the
/// edge ramp drew it at 0.7–0.9 over two pixels, heavier on a centre than
/// between two. Along a row and on a 45° diagonal, whose columns each cross
/// √2 of its width.
#[test]
fn thin_segments_keep_their_weight_between_pixels() {
    let Some(mut hr) = host() else { return };
    let mut mr = MarkRenderer::new(hr.device());
    let cam = camera(128.0, 64.0);
    // Coverage of a white line on black, linear.
    let lin = |b: u8| {
        let c = b as f32 / 255.0;
        if c <= 0.04045 { c / 12.92 } else { ((c + 0.055) / 1.055).powf(2.4) }
    };
    let width = 0.4f32;
    for diagonal in [false, true] {
        let mut sums = Vec::new();
        for k in 0..8 {
            let phase = k as f64 / 8.0;
            let mut line = SegmentMarks::new();
            if diagonal {
                line.set_segments(&[[-40.0 + phase, -40.0, 0.0]], &[[40.0 + phase, 40.0, 0.0]]);
            } else {
                line.set_segments(&[[-100.0, phase, 0.0]], &[[100.0, phase, 0.0]]);
            }
            line.set_widths(&[width]);
            line.set_default_color([255, 255, 255, 255]);
            let img = frame(&mut hr, &mut mr, &mut scene(), &cam, Origin::ZERO, &mut [&mut line]);
            for x in [118u32, 128, 137] {
                let col: Vec<f32> = (0..H).map(|y| lin(px(&img, x, y)[0])).collect();
                let peak = col.iter().cloned().fold(0.0, f32::max);
                let expect = if diagonal { width * std::f32::consts::SQRT_2 } else { width };
                assert!(peak <= expect.min(1.0) + 0.03, "diagonal {diagonal} phase {phase} x {x}: peak {peak}");
                sums.push(col.iter().sum::<f32>() / expect);
            }
        }
        for s in &sums {
            assert!((s - 1.0).abs() < 0.06, "diagonal {diagonal}: column ink {s} of the width, all {sums:?}");
        }
    }
}

/// Round joints, translucent: each segment keeps its side of the joint, so
/// the disc there is no brighter than the legs, and as round.
#[test]
fn translucent_round_joints_are_not_doubled() {
    use threers::marks::LineJoin;
    let Some(mut hr) = host() else { return };
    let mut mr = MarkRenderer::new(hr.device());
    let mut path = SegmentMarks::new();
    path.set_paths(&[vec![[-60.0, -20.0, 0.0], [0.0, -20.0, 0.0], [0.0, 40.0, 0.0]]]);
    path.set_widths(&[10.0, 10.0]);
    path.set_default_color([255, 0, 0, 128]);
    path.join = LineJoin::Round;
    let cam = camera(128.0, 64.0);
    let img = frame(&mut hr, &mut mr, &mut scene(), &cam, Origin::ZERO, &mut [&mut path]);
    let leg = px(&img, 128 - 30, 64 + 20)[0];
    // The joint's centre, and the outside of the turn 3.5 px out (pixel
    // centres are at +0.5): inside the disc.
    for (x, y) in [(128, 64 + 20), (128 + 2, 64 + 22)] {
        let r = px(&img, x, y)[0];
        assert!(r.abs_diff(leg) <= 3, "({x}, {y}): {r} against the leg's {leg}");
    }
    // Past the disc on the outside of the turn: nothing.
    assert!(px(&img, 128 + 6, 64 + 26)[0] < 20, "the disc is round, not a mitre");
}

#[test]
fn translucent_paths_are_mitred_not_overlapped() {
    let Some(mut hr) = host() else { return };
    let mut mr = MarkRenderer::new(hr.device());
    // An L: along y = -20 to the origin's column, then up. Half-transparent
    // red over black, so any pixel covered twice comes out brighter.
    let mut path = SegmentMarks::new();
    path.set_paths(&[vec![
        [-60.0, -20.0, 0.0],
        [0.0, -20.0, 0.0],
        [0.0, 40.0, 0.0],
    ]]);
    path.set_widths(&[10.0, 10.0]);
    path.set_default_color([255, 0, 0, 128]);
    let cam = camera(128.0, 64.0);
    let img = frame(
        &mut hr,
        &mut mr,
        &mut scene(),
        &cam,
        Origin::ZERO,
        &mut [&mut path],
    );
    let leg = px(&img, 128 - 30, 64 + 20)[0];
    let other_leg = px(&img, 128, 64 - 10)[0];
    let joint = px(&img, 128, 64 + 20)[0];
    assert!(leg > 100 && leg < 200, "one layer of 50% red: {leg}");
    assert!(leg.abs_diff(other_leg) <= 2, "{leg} vs {other_leg}");
    assert!(
        leg.abs_diff(joint) <= 2,
        "the joint is one layer too: {joint}"
    );
    // (4.5, -24.5) is 6.4 px from the corner: outside a round join of radius
    // 5, inside the mitre's square corner.
    let corner = px(&img, 128 + 4, 64 + 24)[0];
    assert!(
        leg.abs_diff(corner) <= 2,
        "the mitre fills the corner: {corner}"
    );
    // The open ends still get round caps.
    assert!(px(&img, 128 - 63, 64 + 20)[0] > 100, "cap at the start");
    assert!(is_bg(px(&img, 128 - 67, 64 + 20)));
}

#[test]
fn polygons_fill_extrude_and_pick_per_feature() {
    use threers::marks::{PolygonFeature, PolygonMarks};
    let Some(mut hr) = host() else { return };
    let mut mr = MarkRenderer::new(hr.device());
    let mut s = scene();
    // A blue ground at the height the flat fill lies at.
    let ground = [
        [-200.0, -200.0],
        [200.0, -200.0],
        [200.0, 200.0],
        [-200.0, 200.0],
    ];
    s.add_mesh(
        PolygonGeometry::new(&ground, &[], 0.0, 0.0),
        Material::Basic(BasicMaterial::new(Color::from_hex(0x0000ff))),
    );
    // Straight down, north up the screen: ground (x, y) is pixel
    // (128 + x, 64 - y).
    let mut cam = OrthographicCamera::new(-128.0, 128.0, 64.0, -64.0, 0.1, 200.0);
    cam.position = threers::Vector3::new(0.0, 100.0, 0.0);
    cam.target = threers::Vector3::ZERO;
    cam.up = threers::Vector3::new(0.0, 0.0, -1.0);

    let rect = |x0: f64, y0: f64, x1: f64, y1: f64| vec![[x0, y0], [x1, y0], [x1, y1], [x0, y1]];
    let flat = PolygonFeature::new(rect(-80.0, -20.0, -40.0, 20.0));
    let mut tower = PolygonFeature::new(rect(40.0, -20.0, 80.0, 20.0));
    tower.holes.push(rect(55.0, -5.0, 65.0, 5.0));
    let mut polygons = PolygonMarks::new();
    polygons.set_polygons(&[flat, tower], 0.0);
    polygons.set_colors(&[[255, 0, 0, 255], [0, 255, 0, 255]]);
    polygons.set_elevations(&[0.0, 30.0]);
    // A marker at height 20 over the tower's roof, which is at 30.
    let mut marker = PointMarks::new();
    marker.set_positions(&[[48.0, 20.0, 0.0]]);
    marker.set_radii(&[3.0]);
    marker.set_default_color([255, 255, 255, 255]);

    let img = frame(
        &mut hr,
        &mut mr,
        &mut s,
        &cam,
        Origin::ZERO,
        &mut [&mut polygons, &mut marker],
    );
    assert_eq!(
        &px(&img, 128 - 60, 64)[..3],
        &[255, 0, 0],
        "a flat fill on coincident ground, its colour exact"
    );
    assert_eq!(&px(&img, 128 + 45, 64 + 10)[..3], &[0, 255, 0], "the roof");
    assert_eq!(
        &px(&img, 128 + 60, 64)[..3],
        &[0, 0, 255],
        "ground through the hole"
    );
    assert_eq!(
        &px(&img, 128 + 48, 64)[..3],
        &[0, 255, 0],
        "marker under the roof"
    );

    let device = hr.device().clone();
    let queue = hr.queue().clone();
    let view = MarkView::new(&cam, Origin::ZERO, (W, H));
    let hit = mr
        .pick(
            &device,
            &queue,
            &view,
            &mut [&mut polygons],
            None,
            128.5 + 45.0,
            74.5,
            0,
        )
        .wait(&device)
        .unwrap();
    assert_eq!((hit.mark, hit.index), (0, 1));
    let hole = mr
        .pick(
            &device,
            &queue,
            &view,
            &mut [&mut polygons],
            None,
            128.5 + 60.0,
            64.5,
            0,
        )
        .wait(&device);
    assert!(hole.is_none());

    // Lower the tower below the marker: one feature's vertices re-upload.
    polygons.update_elevations(1, &[10.0]);
    let img = frame(
        &mut hr,
        &mut mr,
        &mut s,
        &cam,
        Origin::ZERO,
        &mut [&mut polygons, &mut marker],
    );
    assert_eq!(
        &px(&img, 128 + 48, 64)[..3],
        &[255, 255, 255],
        "marker above it now"
    );
    assert_eq!(
        &px(&img, 128 - 60, 64)[..3],
        &[255, 0, 0],
        "the other feature as it was"
    );

    // At a slant and far from the origin, as a map is: the fill's position
    // is rebuilt on the GPU from split f64s while the ground's is local f32,
    // so equal heights come out a rounding apart. Without the fill's depth
    // bias the ground wins 240 to 504 of these 600 pixels.
    let far = Origin::new(-13_627_000.37, 0.0, 4_548_000.81);
    let lift = 3.3;
    let (ox, oy) = (far.position[0], -far.position[2]);
    let mut fill = PolygonMarks::new();
    fill.set_polygons(
        &[PolygonFeature::new(rect(
            ox - 80.13,
            oy - 20.29,
            ox - 40.07,
            oy + 20.41,
        ))],
        lift,
    );
    fill.set_default_color([255, 0, 0, 255]);
    let mut slant = threers::PerspectiveCamera::new(50.0, W as f32 / H as f32, 1.0, 1000.0);
    slant.position = threers::Vector3::new(-60.0, 40.0, 90.0);
    slant.target = threers::Vector3::new(-60.0, 0.0, 0.0);
    let mut raised = scene();
    raised.add_mesh(
        PolygonGeometry::new(&ground, &[], lift as f32, 0.0),
        Material::Basic(BasicMaterial::new(Color::from_hex(0x0000ff))),
    );
    let img = {
        hr.render(&mut raised, &slant);
        let view = MarkView::new(&slant, far, (W, H));
        mr.draw(&device, &queue, &hr.mark_target(), &view, &mut [&mut fill]);
        hr.read_rgba()
    };
    // Inside the fill's footprint on screen, which spans about x 95..160,
    // y 53..77.
    let fought = (58..73)
        .flat_map(|y| (108..148).map(move |x| (x, y)))
        .filter(|&(x, y)| px(&img, x, y)[..3] != [255, 0, 0])
        .count();
    assert_eq!(fought, 0, "pixels of the flat fill lost to the ground");
}

#[test]
fn a_trail_hook_cuts_paths_to_the_time_window() {
    let Some(mut hr) = host() else { return };
    let mut mr = MarkRenderer::new(hr.device());
    // Four 50 px legs from x = -100 to 100, reached at t = 0, 1, 2, 3, 4.
    let mut trip = SegmentMarks::new();
    trip.set_paths(&[(0..5)
        .map(|k| [-100.0 + 50.0 * k as f64, 0.0, 0.0])
        .collect()]);
    trip.set_widths(&[6.0; 4]);
    trip.set_default_color([255, 255, 255, 255]);
    trip.common
        .set_custom(0, 2, &[0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0]);
    trip.common.hooks = ShaderHooks::trail(0, 0);
    // Now 2.5 with a trail of 1: t in 1.5..=2.5, which is x in -25..=25.
    trip.common.style.params[0] = [2.5, 1.0, 0.0, 0.0];
    let cam = camera(128.0, 64.0);
    let img = frame(
        &mut hr,
        &mut mr,
        &mut scene(),
        &cam,
        Origin::ZERO,
        &mut [&mut trip],
    );
    let at = |x: i32| px(&img, (128 + x) as u32, 64)[0];
    assert!(
        at(-50) == 0 && at(-30) == 0,
        "past the tail: {} {}",
        at(-50),
        at(-30)
    );
    assert!(
        at(30) == 0 && at(60) == 0,
        "ahead of the head: {} {}",
        at(30),
        at(60)
    );
    // Opacity runs from 0 at the tail to 1 at the head.
    let (tail, mid, head) = (at(-15), at(0), at(20));
    assert!(
        tail > 0 && tail < mid && mid < head,
        "{tail} < {mid} < {head}"
    );
    assert!(head > 200, "near the head: {head}");

    // The clock moves with a uniform write and the head goes with it.
    let pipelines = mr.pipeline_count();
    trip.common.style.params[0] = [3.5, 1.0, 0.0, 0.0];
    let img = frame(
        &mut hr,
        &mut mr,
        &mut scene(),
        &cam,
        Origin::ZERO,
        &mut [&mut trip],
    );
    assert!(px(&img, 128 + 70, 64)[0] > 200, "head now at x = 75");
    assert_eq!(px(&img, 128, 64)[0], 0, "x = 0 is past the tail now");
    assert_eq!(mr.pipeline_count(), pipelines);
}

/// A mark kind defined outside threers, the way a downstream crate would:
/// its own WGSL over the shared preamble, its own columns, and its own bind
/// group — a 2×2 texture each square samples. Hooks, custom columns and
/// picking come with it for free.
mod textured_squares {
    use threers::marks::{
        Column, ColumnFormat, CustomShader, Mark, MarkCommon, MarkKind, PrepareContext,
    };

    const WGSL: &str = r#"
@group(2) @binding(0) var sq_tex : texture_2d<f32>;
@group(2) @binding(1) var sq_samp : sampler;

struct SqIn {
    @builtin(vertex_index) vid : u32,
    @builtin(instance_index) iid : u32,
    @location(0) pos_hi : vec3<f32>,
    @location(1) pos_lo : vec3<f32>,
    /*CUSTOM_IN*/
};
struct SqOut {
    @builtin(position) clip : vec4<f32>,
    @location(0) uv : vec2<f32>,
    @location(1) @interpolate(flat) index : u32,
};

@vertex
fn vs_main(in : SqIn) -> SqOut {
    var i : MarkInstance;
    i.position = mark_relative(in.pos_hi, in.pos_lo);
    i.position2 = i.position;
    i.size = 1.0;
    i.color = vec4<f32>(1.0);
    i.color2 = i.color;
    i.custom0 = /*CUSTOM0*/;
    i.custom1 = /*CUSTOM1*/;
    i.visible = 1.0;
    i.index = in.iid;
    i = hook_instance(i);
    var out : SqOut;
    let clip = frame.view_proj * vec4<f32>(i.position, 1.0);
    if (i.visible <= 0.0) {
        out.clip = mark_hidden();
        return out;
    }
    let c = mark_corner_signed(mark_quad_corner(in.vid));
    out.clip = mark_offset_px(clip, c * mark_size_px(i.size, clip.w));
    out.uv = c * 0.5 + 0.5;
    out.index = i.index;
    return out;
}

@fragment
fn fs_main(in : SqOut) -> @location(0) vec4<f32> {
    let t = textureSample(sq_tex, sq_samp, vec2<f32>(in.uv.x, 1.0 - in.uv.y));
    return mark_out(hook_color(vec4<f32>(t.rgb, 1.0), vec4<f32>(0.0), vec4<f32>(0.0)));
}

@fragment
fn fs_pick(in : SqOut) -> @location(0) vec2<u32> {
    return vec2<u32>(layer.ids.x, in.index);
}
"#;

    pub struct TexturedSquares {
        pub common: MarkCommon,
        shader: CustomShader,
        positions: Column,
        layout: Option<wgpu::BindGroupLayout>,
        bind_group: Option<wgpu::BindGroup>,
    }

    impl TexturedSquares {
        pub fn new(positions: &[[f64; 3]]) -> Self {
            let mut p = Column::new(ColumnFormat::PositionF64);
            p.set_positions(positions);
            let mut common = MarkCommon::default();
            common.style.size_scale = 8.0;
            Self {
                common,
                shader: CustomShader::new("textured squares", WGSL, 2),
                positions: p,
                layout: None,
                bind_group: None,
            }
        }
        pub fn shader(&self) -> &CustomShader {
            &self.shader
        }
    }

    impl Mark for TexturedSquares {
        fn kind(&self) -> MarkKind {
            self.shader.kind()
        }
        fn common(&self) -> &MarkCommon {
            &self.common
        }
        fn common_mut(&mut self) -> &mut MarkCommon {
            &mut self.common
        }
        fn instance_count(&self) -> usize {
            self.positions.len()
        }
        fn columns(&mut self) -> Vec<&mut Column> {
            vec![&mut self.positions]
        }
        fn custom_shader(&self) -> Option<&CustomShader> {
            Some(&self.shader)
        }
        fn extra_layout(&self) -> Option<&wgpu::BindGroupLayout> {
            self.layout.as_ref()
        }
        fn extra_bind_group(&self) -> Option<&wgpu::BindGroup> {
            self.bind_group.as_ref()
        }
        fn prepare(&mut self, ctx: &PrepareContext) {
            if self.bind_group.is_some() {
                return;
            }
            // Top row red, green; bottom row blue, white.
            let texels: [u8; 16] = [
                255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 255, 255, 255, 255,
            ];
            let tex = ctx.device.create_texture(&wgpu::TextureDescriptor {
                label: None,
                size: wgpu::Extent3d {
                    width: 2,
                    height: 2,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            ctx.queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &tex,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                &texels,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(8),
                    rows_per_image: Some(2),
                },
                wgpu::Extent3d {
                    width: 2,
                    height: 2,
                    depth_or_array_layers: 1,
                },
            );
            let layout = ctx
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: None,
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Texture {
                                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                                view_dimension: wgpu::TextureViewDimension::D2,
                                multisampled: false,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                            count: None,
                        },
                    ],
                });
            let view = tex.create_view(&Default::default());
            // Its own sampler — nearest, so each quarter is one texel — beside
            // the renderer's shared linear one.
            let _shared = ctx.linear_sampler();
            let nearest = ctx
                .device
                .create_sampler(&wgpu::SamplerDescriptor::default());
            let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&nearest),
                    },
                ],
            });
            self.layout = Some(layout);
            self.bind_group = Some(bg);
        }
    }
}

#[test]
fn a_mark_kind_defined_outside_threers_draws_filters_and_picks() {
    use textured_squares::TexturedSquares;
    let Some(mut hr) = host() else { return };
    let mut mr = MarkRenderer::new(hr.device());
    let mut sq = TexturedSquares::new(&[[-40.0, 0.0, 0.0], [40.0, 0.0, 0.0]]);
    sq.shader().validate(&ShaderHooks::none()).unwrap();
    let cam = camera(128.0, 64.0);
    let img = frame(
        &mut hr,
        &mut mr,
        &mut scene(),
        &cam,
        Origin::ZERO,
        &mut [&mut sq],
    );
    // 16 px squares, each quarter one texel: top-left red, bottom-right white.
    assert_eq!(
        &px(&img, 128 - 40 - 4, 64 - 4)[..3],
        &[255, 0, 0],
        "top left"
    );
    assert_eq!(
        &px(&img, 128 - 40 + 4, 64 - 4)[..3],
        &[0, 255, 0],
        "top right"
    );
    assert_eq!(
        &px(&img, 128 - 40 - 4, 64 + 4)[..3],
        &[0, 0, 255],
        "bottom left"
    );
    assert_eq!(
        &px(&img, 128 + 40 + 4, 64 + 4)[..3],
        &[255, 255, 255],
        "second square"
    );
    assert!(is_bg(px(&img, 128, 64)));

    // A user's range hook applies to it like any built-in kind.
    sq.common.set_custom(0, 1, &[0.0, 9.0]);
    sq.common.hooks = ShaderHooks::range_filter(0, 0);
    sq.common.style.params[0] = [5.0, 10.0, 0.0, 0.0];
    let img = frame(
        &mut hr,
        &mut mr,
        &mut scene(),
        &cam,
        Origin::ZERO,
        &mut [&mut sq],
    );
    assert!(is_bg(px(&img, 128 - 40 - 4, 64 - 4)), "first filtered out");
    assert!(!is_bg(px(&img, 128 + 40 + 4, 64 + 4)), "second kept");

    let device = hr.device().clone();
    let queue = hr.queue().clone();
    let view = MarkView::new(&cam, Origin::ZERO, (W, H));
    let hit = mr
        .pick(
            &device,
            &queue,
            &view,
            &mut [&mut sq],
            None,
            128.5 + 40.0,
            64.5,
            0,
        )
        .wait(&device)
        .unwrap();
    assert_eq!((hit.mark, hit.index), (0, 1));
}

#[test]
fn sprites_and_images_draw_from_an_atlas() {
    use threers::marks::{ImageMarks, ImageQuad, SpriteMarks, SpriteMode, TextureAtlas};
    let Some(mut hr) = host() else { return };
    let mut mr = MarkRenderer::new(hr.device());
    let solid = |w: u32, h: u32, c: [u8; 4]| -> Vec<u8> {
        c.iter()
            .copied()
            .cycle()
            .take((w * h * 4) as usize)
            .collect()
    };
    let mut atlas = TextureAtlas::new(64);
    let red = atlas
        .insert("red", 8, 8, &solid(8, 8, [255, 0, 0, 255]))
        .unwrap();
    // 8 wide, 16 tall: green over blue.
    let mut two = solid(8, 8, [0, 255, 0, 255]);
    two.extend(solid(8, 8, [0, 0, 255, 255]));
    let tall = atlas.insert("tall", 8, 16, &two).unwrap();
    // Left half opaque white, right half clear.
    let mut half = Vec::new();
    for _ in 0..8 {
        half.extend(solid(4, 1, [255, 255, 255, 255]));
        half.extend(solid(4, 1, [0, 0, 0, 0]));
    }
    atlas.insert("half", 8, 8, &half).unwrap();
    let atlas = atlas.shared();

    let mut sprites = SpriteMarks::new(atlas.clone());
    sprites.set_positions(&[
        [-80.0, 0.0, 0.0],
        [-40.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [40.0, 0.0, 0.0],
    ]);
    sprites.set_images(&["red", "tall", "tall", "half"]);
    sprites.set_angles(&[0.0, 0.0, std::f32::consts::FRAC_PI_2, 0.0]);
    sprites.style_mut().size_scale = 16.0;
    let cam = camera(128.0, 64.0);
    let img = frame(
        &mut hr,
        &mut mr,
        &mut scene(),
        &cam,
        Origin::ZERO,
        &mut [&mut sprites],
    );
    assert_eq!(
        &px(&img, 128 - 80, 64)[..3],
        &[255, 0, 0],
        "a 16 px red square"
    );
    assert!(
        is_bg(px(&img, 128 - 80 + 10, 64)),
        "8 px from centre is outside it"
    );
    // 16 tall and 8 wide, green on top.
    assert_eq!(&px(&img, 128 - 40, 64 - 4)[..3], &[0, 255, 0]);
    assert_eq!(&px(&img, 128 - 40, 64 + 4)[..3], &[0, 0, 255]);
    assert!(is_bg(px(&img, 128 - 40 + 6, 64)), "only 8 wide");
    // Turned a quarter counter-clockwise: green now on the left.
    assert_eq!(
        &px(&img, 128 - 4, 64)[..3],
        &[0, 255, 0],
        "rotated: top is left"
    );
    assert_eq!(&px(&img, 128 + 4, 64)[..3], &[0, 0, 255]);
    // Transparent texels are not drawn.
    assert_eq!(&px(&img, 128 + 40 - 4, 64)[..3], &[255, 255, 255]);
    assert!(is_bg(px(&img, 128 + 40 + 4, 64)));

    // ... nor picked.
    let device = hr.device().clone();
    let queue = hr.queue().clone();
    let view = MarkView::new(&cam, Origin::ZERO, (W, H));
    let pick = |mr: &mut MarkRenderer, s: &mut SpriteMarks, x: f32| {
        mr.pick(&device, &queue, &view, &mut [s], None, x, 64.5, 0)
            .wait(&device)
    };
    assert_eq!(
        pick(&mut mr, &mut sprites, 128.5 + 40.0 - 4.0).map(|h| h.index),
        Some(3)
    );
    assert_eq!(pick(&mut mr, &mut sprites, 128.5 + 40.0 + 4.0), None);

    // Mask mode recolours by the row's colour; an anchor of the bottom
    // middle stands the icon on its position.
    sprites.mode = SpriteMode::Mask;
    sprites.anchor = [0.0, -1.0];
    sprites.set_colors(&[[255, 0, 255, 255]; 4]);
    let img = frame(
        &mut hr,
        &mut mr,
        &mut scene(),
        &cam,
        Origin::ZERO,
        &mut [&mut sprites],
    );
    assert_eq!(
        &px(&img, 128 - 80, 64 - 8)[..3],
        &[255, 0, 255],
        "above the position"
    );
    assert!(is_bg(px(&img, 128 - 80, 64 + 4)), "nothing below it");

    // An image quad: 40 px wide, 40 tall, the tall image stretched over it.
    let mut images = ImageMarks::new(atlas.clone());
    images.set_quads(&[ImageQuad {
        corner: [-20.0, 20.0, 0.0],
        right: [40.0, 0.0, 0.0],
        down: [0.0, -40.0, 0.0],
        rect: tall,
    }]);
    let img = frame(
        &mut hr,
        &mut mr,
        &mut scene(),
        &cam,
        Origin::ZERO,
        &mut [&mut images],
    );
    assert_eq!(&px(&img, 128, 64 - 10)[..3], &[0, 255, 0], "upper half");
    assert_eq!(&px(&img, 128, 64 + 10)[..3], &[0, 0, 255], "lower half");
    assert!(is_bg(px(&img, 128 + 24, 64)));

    // The atlas grows under a layer that already drew from it.
    atlas
        .write()
        .unwrap()
        .insert("big", 60, 100, &solid(60, 100, [255, 255, 0, 255]))
        .unwrap();
    assert!(atlas.read().unwrap().size().1 > 64);
    images.update_quads(
        0,
        &[ImageQuad {
            corner: [-20.0, 20.0, 0.0],
            right: [40.0, 0.0, 0.0],
            down: [0.0, -40.0, 0.0],
            rect: red,
        }],
    );
    let img = frame(
        &mut hr,
        &mut mr,
        &mut scene(),
        &cam,
        Origin::ZERO,
        &mut [&mut images],
    );
    assert_eq!(
        &px(&img, 128, 64)[..3],
        &[255, 0, 0],
        "old rect, new texture"
    );
}

#[test]
fn local_positions_draw_where_absolute_ones_do() {
    let Some(mut hr) = host() else { return };
    let mut mr = MarkRenderer::new(hr.device());
    // A tile's corner ten million units out, and the view's origin near it.
    let corner = [10_000_000.0, 0.0, 0.0];
    let origin = Origin::new(10_000_040.25, 0.0, 0.0);
    let local: Vec<[f32; 3]> = vec![[-30.0, 10.0, 0.0], [10.5, -20.0, 0.0], [70.0, 0.0, 0.0]];
    let absolute: Vec<[f64; 3]> = local
        .iter()
        .map(|p| [corner[0] + p[0] as f64, p[1] as f64, p[2] as f64])
        .collect();
    let cam = camera(128.0, 64.0);
    let draw = |hr: &mut HeadlessRenderer, mr: &mut MarkRenderer, local_layers: bool| {
        let mut pts = PointMarks::new();
        let mut seg = SegmentMarks::new();
        let mut path = SegmentMarks::new();
        let path_local: Vec<[f32; 3]> =
            vec![[0.0, 40.0, 0.0], [40.0, 40.0, 0.0], [40.0, 55.0, 0.0]];
        if local_layers {
            pts.set_local_positions(corner, &local);
            seg.set_local_segments(corner, &local[..2], &local[1..]);
            path.set_local_paths(corner, &[path_local]);
        } else {
            pts.set_positions(&absolute);
            seg.set_segments(&absolute[..2], &absolute[1..]);
            let abs: Vec<[f64; 3]> = path_local
                .iter()
                .map(|p| [corner[0] + p[0] as f64, p[1] as f64, 0.0])
                .collect();
            path.set_paths(&[abs]);
        }
        pts.set_radii(&[5.0; 3]);
        pts.set_default_color([255, 0, 0, 255]);
        seg.set_widths(&[3.0, 3.0]);
        path.set_widths(&[4.0, 4.0]);
        path.set_default_color([0, 255, 0, 255]);
        frame(
            hr,
            mr,
            &mut scene(),
            &cam,
            origin,
            &mut [&mut seg, &mut path, &mut pts],
        )
    };
    let a = draw(&mut hr, &mut mr, false);
    let b = draw(&mut hr, &mut mr, true);
    let drawn = a
        .chunks(4)
        .filter(|p| p[0] > 0 || p[1] > 0 || p[2] > 0)
        .count();
    assert!(drawn > 300, "something to compare: {drawn} pixels");
    let differ = a.chunks(4).zip(b.chunks(4)).filter(|(x, y)| x != y).count();
    assert_eq!(
        differ, 0,
        "{differ} pixels differ between local and absolute"
    );
}

#[test]
fn released_layers_report_zero_and_draw_again() {
    use threers::marks::{PolygonFeature, PolygonMarks};
    let Some(mut hr) = host() else { return };
    let mut mr = MarkRenderer::new(hr.device());
    let mut pts = two_points();
    let mut poly = PolygonMarks::new();
    poly.set_polygons(
        &[PolygonFeature::new(vec![
            [-20.0, -5.0],
            [20.0, -5.0],
            [20.0, 5.0],
            [-20.0, 5.0],
        ])],
        0.0,
    );
    poly.set_default_color([0, 0, 255, 255]);
    poly.common.style.depth_test = false;
    // Seen from +Z the flat fill is edge-on, so lift it: extrude 30 up.
    poly.set_elevations(&[30.0]);
    let cam = camera(128.0, 64.0);
    let first = frame(
        &mut hr,
        &mut mr,
        &mut scene(),
        &cam,
        Origin::ZERO,
        &mut [&mut poly, &mut pts],
    );
    assert!(pts.gpu_bytes() > 0 && poly.gpu_bytes() > 0);
    pts.release_gpu();
    poly.release_gpu();
    assert_eq!(
        (pts.gpu_bytes(), poly.gpu_bytes()),
        (0, 0),
        "nothing held after release"
    );
    let again = frame(
        &mut hr,
        &mut mr,
        &mut scene(),
        &cam,
        Origin::ZERO,
        &mut [&mut poly, &mut pts],
    );
    assert!(first.chunks(4).any(|p| p[2] > 200), "the polygon drew");
    assert_eq!(first, again, "the same picture after uploading again");
}

#[test]
fn draw_order_stacks_coplanar_layers_and_clip_cuts_at_a_tile_edge() {
    use threers::marks::{PolygonFeature, PolygonMarks};
    let Some(mut hr) = host() else { return };
    let mut mr = MarkRenderer::new(hr.device());
    let rect = |x0: f64, y0: f64, x1: f64, y1: f64| vec![[x0, y0], [x1, y0], [x1, y1], [x0, y1]];
    // Far out and seen at a slant, where coplanar fills fight hardest.
    let far = Origin::new(-13_627_000.37, 0.0, 4_548_000.81);
    let (ox, oy) = (far.position[0], -far.position[2]);
    let fill = |c: [u8; 4], order: i32| {
        let mut p = PolygonMarks::new();
        p.set_polygons(
            &[PolygonFeature::new(rect(
                ox - 50.0,
                oy - 50.0,
                ox + 50.0,
                oy + 50.0,
            ))],
            0.0,
        );
        p.set_default_color(c);
        p.common.style.depth_order = order;
        p
    };
    let mut slant = threers::PerspectiveCamera::new(50.0, W as f32 / H as f32, 1.0, 1000.0);
    slant.position = threers::Vector3::new(0.0, 40.0, 90.0);
    slant.target = threers::Vector3::ZERO;
    let draw = |hr: &mut HeadlessRenderer, mr: &mut MarkRenderer, layers: &mut [&mut dyn Mark]| {
        hr.render(&mut scene(), &slant);
        let view = MarkView::new(&slant, far, (W, H));
        let (d, q) = (hr.device().clone(), hr.queue().clone());
        mr.draw(&d, &q, &hr.mark_target(), &view, layers);
        hr.read_rgba()
    };
    let count = |img: &[u8], c: [u8; 3]| img.chunks(4).filter(|p| p[..3] == c).count();
    // Blue is drawn first but ordered above: it keeps every pixel.
    let (mut red, mut blue) = (fill([255, 0, 0, 255], 0), fill([0, 0, 255, 255], 1));
    let img = draw(&mut hr, &mut mr, &mut [&mut blue, &mut red]);
    assert!(count(&img, [0, 0, 255]) > 2000, "blue drawn");
    assert_eq!(count(&img, [255, 0, 0]), 0, "red never wins where blue is");
    // Swap the orders and red takes all of it.
    red.common.style.depth_order = 2;
    let img = draw(&mut hr, &mut mr, &mut [&mut blue, &mut red]);
    assert_eq!(count(&img, [0, 0, 255]), 0, "now blue never wins");
    assert!(count(&img, [255, 0, 0]) > 2000);

    // Clip: straight down, ground (x, y) at pixel (128 + x, 64 - y).
    let mut top = OrthographicCamera::new(-128.0, 128.0, 64.0, -64.0, 0.1, 200.0);
    top.position = threers::Vector3::new(0.0, 100.0, 0.0);
    top.up = threers::Vector3::new(0.0, 0.0, -1.0);
    let mut big = PolygonMarks::new();
    big.set_polygons(
        &[PolygonFeature::new(rect(-100.0, -50.0, 100.0, 50.0))],
        0.0,
    );
    big.set_default_color([0, 255, 0, 255]);
    big.common.style.clip = Some([-40.0, -20.0, 40.0, 30.0]);
    let mut pts = PointMarks::new();
    // Anchors inside, and outside, the clip rectangle.
    pts.set_positions(&[[-30.0, 1.0, 10.0], [60.0, 1.0, 0.0]]);
    pts.set_radii(&[4.0, 4.0]);
    pts.set_default_color([255, 255, 255, 255]);
    pts.common.style.clip = big.common.style.clip;
    pts.common.style.depth_test = false;
    let img = frame(
        &mut hr,
        &mut mr,
        &mut scene(),
        &top,
        Origin::ZERO,
        &mut [&mut big, &mut pts],
    );
    assert_eq!(&px(&img, 128, 64)[..3], &[0, 255, 0], "inside the clip");
    assert!(is_bg(px(&img, 128 + 50, 64)), "the fill is cut at x = 40");
    assert!(is_bg(px(&img, 128, 64 - 35)), "and at y = 30");
    assert!(
        !is_bg(px(&img, 128 + 38, 64 - 28)),
        "right up to the corner"
    );
    assert_eq!(
        &px(&img, 128 - 30, 64 + 10)[..3],
        &[255, 255, 255],
        "point inside drawn"
    );
    assert!(is_bg(px(&img, 128 + 60, 64)), "point outside dropped whole");
}

#[test]
fn lines_take_dashes_caps_and_a_casing() {
    use threers::marks::LineCap;
    let Some(mut hr) = host() else { return };
    let mut mr = MarkRenderer::new(hr.device());
    let cam = camera(128.0, 64.0);
    // Two segments, x = -100 to 0 to 100, 10 px wide: along the path is
    // x + 100 pixels.
    let path = || {
        let mut p = SegmentMarks::new();
        p.set_paths(&[vec![[-100.0, 0.0, 0.0], [0.0, 0.0, 0.0], [100.0, 0.0, 0.0]]]);
        p.set_widths(&[10.0, 10.0]);
        p.set_default_color([255, 0, 0, 255]);
        p
    };
    let at = |img: &[u8], x: i32, y: i32| px(img, (128 + x) as u32, (64 + y) as u32);

    // Dashes two widths long, gaps one: 20 px on, 10 px off, in phase across
    // the joint at x = 0.
    let mut dashed = path();
    dashed.dash = Some([2.0, 1.0]);
    dashed.cap = LineCap::Butt;
    let img = frame(
        &mut hr,
        &mut mr,
        &mut scene(),
        &cam,
        Origin::ZERO,
        &mut [&mut dashed],
    );
    for (along, on) in [
        (10, true),
        (25, false),
        (35, true),
        (55, false),
        (100, true),
        (115, false),
        (130, true),
    ] {
        let p = at(&img, along - 100, 0);
        assert_eq!(p[0] > 200, on, "{along} px along: {p:?}");
    }
    // Round caps round the dashes' ends: 4 px into a gap is inside the
    // 5 px cap on the centre line, outside it 4 px off it (pixel centres
    // at +0.5).
    dashed.cap = LineCap::Round;
    let img = frame(&mut hr, &mut mr, &mut scene(), &cam, Origin::ZERO, &mut [&mut dashed]);
    assert!(at(&img, 24 - 100, 0)[0] > 200, "a dash's round end");
    assert!(at(&img, 24 - 100, 4)[0] < 100, "round, not square");
    // Justified: a whole number of periods per segment (100 px: 3 of 33 px,
    // 20 on), half a dash at each end.
    dashed.cap = LineCap::Butt;
    dashed.dash_justified = true;
    dashed.dash_mode = threers::marks::DashMode::Segment;
    let img = frame(&mut hr, &mut mr, &mut scene(), &cam, Origin::ZERO, &mut [&mut dashed]);
    for (along, on) in [(2, true), (20, false), (33, true), (50, false), (98, true)] {
        let p = at(&img, along - 100, 0);
        assert_eq!(p[0] > 200, on, "justified, {along} px along: {p:?}");
    }

    // Caps: a butt end stops at the point, round and square reach past it,
    // and only the square fills the corner.
    let mut caps = |cap: LineCap| {
        let mut p = path();
        p.cap = cap;
        frame(
            &mut hr,
            &mut mr,
            &mut scene(),
            &cam,
            Origin::ZERO,
            &mut [&mut p],
        )
    };
    let (butt, round, square) = (
        caps(LineCap::Butt),
        caps(LineCap::Round),
        caps(LineCap::Square),
    );
    assert!(is_bg(at(&butt, 102, 0)), "butt stops at x = 100");
    assert!(at(&round, 102, 0)[0] > 200, "round reaches past");
    assert!(is_bg(at(&round, 104, 4)), "round leaves the corner");
    assert!(at(&square, 104, 4)[0] > 200, "square fills it");

    // A 3 px casing in black around the red.
    let mut cased = path();
    cased.outline_width = 3.0;
    cased.outline_color = [0, 0, 0, 255];
    let img = frame(
        &mut hr,
        &mut mr,
        &mut scene(),
        &cam,
        Origin::ZERO,
        &mut [&mut cased],
    );
    assert_eq!(&at(&img, -50, 0)[..3], &[255, 0, 0], "red down the middle");
    let edge = at(&img, -50, 4);
    assert!(edge[0] < 60, "dark inside the edge: {edge:?}");
}

#[test]
fn box_picks_report_every_row_with_its_feature_and_highlight_needs_no_upload() {
    let Some(mut hr) = host() else { return };
    let mut mr = MarkRenderer::new(hr.device());
    let mut pts = PointMarks::new();
    pts.set_positions(&[
        [-60.0, 0.0, 0.0],
        [-20.0, 10.0, 0.0],
        [20.0, -10.0, 0.0],
        [60.0, 0.0, 0.0],
    ]);
    pts.set_radii(&[5.0; 4]);
    pts.set_default_color([255, 255, 255, 255]);
    pts.common.set_feature_ids(vec![1001, 1002, 1003, 1004]);
    let cam = camera(128.0, 64.0);
    let device = hr.device().clone();
    let queue = hr.queue().clone();
    let view = MarkView::new(&cam, Origin::ZERO, (W, H));
    // A box over the middle two.
    let mut hits = mr
        .pick_rect(
            &device,
            &queue,
            &view,
            &mut [&mut pts],
            None,
            (128 - 40, 40),
            (128 + 40, 90),
        )
        .wait(&device);
    hits.sort_by_key(|h| h.index);
    let got: Vec<(u32, Option<u64>)> = hits.iter().map(|h| (h.index, h.feature)).collect();
    assert_eq!(
        got,
        vec![(1, Some(1002)), (2, Some(1003))],
        "each row once, with its id"
    );
    let none = mr
        .pick_rect(
            &device,
            &queue,
            &view,
            &mut [&mut pts],
            None,
            (0, 0),
            (10, 10),
        )
        .wait(&device);
    assert!(none.is_empty());

    // Highlight row 2: a uniform, so no column is waiting to upload.
    let _ = frame(
        &mut hr,
        &mut mr,
        &mut scene(),
        &cam,
        Origin::ZERO,
        &mut [&mut pts],
    );
    pts.style_mut().highlight = Some(2);
    pts.style_mut().highlight_color = [255, 0, 0, 255];
    assert!(pts.positions().pending_upload().is_none());
    let img = frame(
        &mut hr,
        &mut mr,
        &mut scene(),
        &cam,
        Origin::ZERO,
        &mut [&mut pts],
    );
    assert_eq!(
        &px(&img, 128 + 20, 64 + 10)[..3],
        &[255, 0, 0],
        "row 2 highlighted"
    );
    assert_eq!(
        &px(&img, 128 - 20, 64 - 10)[..3],
        &[255, 255, 255],
        "row 1 as it was"
    );
}

#[test]
fn a_transition_hook_moves_rows_from_old_to_new_and_combines() {
    use threers::marks::transition_offsets;
    let Some(mut hr) = host() else { return };
    let mut mr = MarkRenderer::new(hr.device());
    let old = [[-60.0, 0.0, 0.0], [40.0, 30.0, 0.0]];
    let new = [[60.0, 0.0, 0.0], [40.0, -30.0, 0.0]];
    let mut pts = PointMarks::new();
    pts.set_positions(&new);
    pts.set_radii(&[6.0, 6.0]);
    pts.set_default_color([255, 255, 255, 255]);
    // Old position offsets, then old radius (half the new) in component 3.
    let mut custom = Vec::new();
    for (k, d) in transition_offsets(&old, &new).chunks(3).enumerate() {
        custom.extend_from_slice(d);
        custom.push([3.0, 3.0][k]);
    }
    pts.common.set_custom(0, 4, &custom);
    pts.common.hooks = ShaderHooks::transition(0, Some(3), 1);
    let cam = camera(128.0, 64.0);
    let mut at_t = |t: f32, pts: &mut PointMarks| {
        pts.common.style.params[1] = [t, 0.0, 0.0, 0.0];
        frame(
            &mut hr,
            &mut mr,
            &mut scene(),
            &cam,
            Origin::ZERO,
            &mut [pts],
        )
    };
    let img = at_t(0.0, &mut pts);
    assert!(
        !is_bg(px(&img, 128 - 60, 64)) && is_bg(px(&img, 128 + 60, 64)),
        "at the start, where it was"
    );
    assert!(
        !is_bg(px(&img, 128 - 60 + 2, 64)) && is_bg(px(&img, 128 - 60 + 5, 64)),
        "with the old 3 px radius"
    );
    let img = at_t(0.5, &mut pts);
    assert!(
        !is_bg(px(&img, 128, 64)),
        "half way: the first point at x = 0"
    );
    assert!(!is_bg(px(&img, 128 + 40, 64)), "and the second at (40, 0)");
    let img = at_t(1.0, &mut pts);
    assert!(
        !is_bg(px(&img, 128 + 60, 64)) && !is_bg(px(&img, 128 + 60 + 5, 64)),
        "at the end, full size"
    );
    assert!(is_bg(px(&img, 128 - 60, 64)));

    // With a filter on a second custom column: row 1 dropped mid-way.
    pts.common.set_custom(1, 1, &[1.0, 9.0]);
    pts.common.hooks =
        ShaderHooks::transition(0, Some(3), 1).combine(&ShaderHooks::range_filter(4, 0));
    pts.common.style.params[0] = [0.0, 5.0, 0.0, 0.0];
    let img = at_t(0.5, &mut pts);
    assert!(!is_bg(px(&img, 128, 64)), "row 0 still moving");
    assert!(is_bg(px(&img, 128 + 40, 64)), "row 1 filtered out");
}

#[test]
fn globe_marks_follow_the_sphere_and_hide_behind_it() {
    use threers::marks::geo::{lon_lat_to_globe, GlobeGround};
    use threers::marks::{
        ImageMarks, ImageQuad, PolygonFeature, PolygonMarks, PolygonMesh, TextureAtlas,
    };
    let Some(mut hr) = host() else { return };
    let mut mr = MarkRenderer::new(hr.device());
    // Looking down -Y at a sphere of radius 30 on the origin, -Z up the
    // screen: world (x, _, z) is pixel (128 + x, 64 + z), so how far out
    // along +Z a mark reaches shows as how far down the screen it goes.
    let mut top = OrthographicCamera::new(-128.0, 128.0, 64.0, -64.0, 0.1, 400.0);
    top.position = threers::Vector3::new(0.0, 200.0, 0.0);
    top.up = threers::Vector3::new(0.0, 0.0, -1.0);
    let on = |lon: f64, lat: f64, alt: f64| lon_lat_to_globe([lon, lat, alt], 30.0);
    let at = |img: &[u8], x: i32, z: i32| px(img, (128 + x) as u32, (64 + z) as u32);

    // A great circle along the equator from lon -60 to 60 peaks at z = 30;
    // the chord between its ends only reaches z = 15.
    let mut arc = SegmentMarks::new();
    arc.set_segments(&[on(-60.0, 0.0, 0.0)], &[on(60.0, 0.0, 0.0)]);
    arc.set_widths(&[3.0]);
    arc.globe = Some([0.0; 3]);
    arc.common.style.depth_test = false;
    let img = frame(
        &mut hr,
        &mut mr,
        &mut scene(),
        &top,
        Origin::ZERO,
        &mut [&mut arc],
    );
    assert!(!is_bg(at(&img, 0, 30)), "the arc follows the surface");
    assert!(is_bg(at(&img, 0, 15)), "not the chord");

    // A polygon on the globe, extruded 15: it rises away from the centre,
    // out to z = 45 at lon 0 — a +Y extrusion would point at the camera.
    let globe = GlobeGround {
        radius: 30.0,
        centre: [0.0; 3],
        max_edge_degrees: 2.0,
    };
    let square = vec![[-5.0, -5.0], [5.0, -5.0], [5.0, 5.0], [-5.0, 5.0]];
    let mut block = PolygonMarks::new();
    block.set_mesh(PolygonMesh::build_with(
        &[PolygonFeature::new(square)],
        0.0,
        &globe,
    ));
    block.set_elevations(&[15.0]);
    block.set_default_color([0, 200, 255, 255]);
    let img = frame(
        &mut hr,
        &mut mr,
        &mut scene(),
        &top,
        Origin::ZERO,
        &mut [&mut block],
    );
    assert!(!is_bg(at(&img, 0, 42)), "the block reaches out radially");
    assert!(is_bg(at(&img, 0, 48)), "to its roof and no further");

    // An image quad whose corners span the chord, bent onto the sphere.
    let mut atlas = TextureAtlas::new(16);
    let rect = atlas.insert("w", 4, 4, &[255; 64]).unwrap();
    let mut img_marks = ImageMarks::new(atlas.shared());
    let (a, b) = (on(-60.0, 10.0, 0.0), on(60.0, 10.0, 0.0));
    let c = on(-60.0, -10.0, 0.0);
    img_marks.set_quads(&[ImageQuad {
        corner: a,
        right: [
            (b[0] - a[0]) as f32,
            (b[1] - a[1]) as f32,
            (b[2] - a[2]) as f32,
        ],
        down: [
            (c[0] - a[0]) as f32,
            (c[1] - a[1]) as f32,
            (c[2] - a[2]) as f32,
        ],
        rect,
    }]);
    img_marks.subdivisions = 16;
    img_marks.sphere = Some(([0.0; 3], 30.0));
    img_marks.common.style.depth_test = false;
    let img = frame(
        &mut hr,
        &mut mr,
        &mut scene(),
        &top,
        Origin::ZERO,
        &mut [&mut img_marks],
    );
    assert!(
        !is_bg(at(&img, 0, 28)),
        "the tile bulges out to the surface"
    );

    // Horizon: from a perspective eye on +Z, a point on the far side is
    // culled even with depth testing off; one on the near side is not.
    let mut persp = threers::PerspectiveCamera::new(50.0, W as f32 / H as f32, 1.0, 1000.0);
    persp.position = threers::Vector3::new(0.0, 0.0, 150.0);
    persp.target = threers::Vector3::ZERO;
    let mut pts = PointMarks::new();
    pts.set_positions(&[on(0.0, 20.0, 1.0), on(180.0, -20.0, 1.0)]);
    pts.set_radii(&[4.0, 4.0]);
    pts.set_default_color([255, 255, 255, 255]);
    pts.common.style.depth_test = false;
    let count = |img: &[u8]| img.chunks(4).filter(|p| p[0] > 200).count();
    let both = {
        hr.render(&mut scene(), &persp);
        let view = MarkView::new(&persp, Origin::ZERO, (W, H));
        let (d, q) = (hr.device().clone(), hr.queue().clone());
        mr.draw(&d, &q, &hr.mark_target(), &view, &mut [&mut pts]);
        count(&hr.read_rgba())
    };
    pts.common.style.horizon = Some(([0.0; 3], 30.0));
    let front = {
        hr.render(&mut scene(), &persp);
        let view = MarkView::new(&persp, Origin::ZERO, (W, H));
        let (d, q) = (hr.device().clone(), hr.queue().clone());
        mr.draw(&d, &q, &hr.mark_target(), &view, &mut [&mut pts]);
        count(&hr.read_rgba())
    };
    assert!(
        front > 20 && front * 2 <= both + 4,
        "the far point culled: {front} of {both} px"
    );
}

#[test]
fn tone_mapped_marks_match_the_scene_and_plain_ones_keep_their_colour() {
    let Some(mut hr) = host() else { return };
    hr.renderer()
        .set_tone_mapping(threers::ToneMapping::AcesFilmic, 1.6);
    let mut mr = MarkRenderer::new(hr.device());
    let mut s = scene();
    // A basic-material square on the left, in the colour the marks use.
    let mut square = BoxGeometry::new(30.0, 30.0, 1.0);
    for v in square
        .attributes
        .get_mut("position")
        .unwrap()
        .array
        .chunks_mut(3)
    {
        v[0] -= 60.0;
    }
    s.add_mesh(
        square,
        Material::Basic(BasicMaterial::new(Color::from_hex(0x3399cc))),
    );
    let mut pts = PointMarks::new();
    pts.set_positions(&[[0.0, 0.0, 0.0], [60.0, 0.0, 0.0]]);
    pts.set_radii(&[8.0, 8.0]);
    pts.set_default_color([0x33, 0x99, 0xcc, 255]);
    pts.common.style.depth_test = false;
    let cam = camera(128.0, 64.0);
    hr.render(&mut s, &cam);
    let (device, queue) = (hr.device().clone(), hr.queue().clone());
    let mut draw = |hr: &mut HeadlessRenderer, pts: &mut PointMarks, tone: bool| {
        pts.common.style.tone_mapped = tone;
        hr.render(&mut s, &cam);
        let view = MarkView::from_renderer(&cam, Origin::ZERO, hr.renderer());
        mr.draw(&device, &queue, &hr.mark_target(), &view, &mut [pts]);
        hr.read_rgba()
    };
    let img = draw(&mut hr, &mut pts, true);
    let mesh = px(&img, 128 - 60, 64);
    let mark = px(&img, 128, 64);
    for c in 0..3 {
        assert!(
            mesh[c].abs_diff(mark[c]) <= 2,
            "tone mapped like the scene: {mesh:?} vs {mark:?}"
        );
    }
    assert_ne!(
        &mesh[..3],
        &[0x33, 0x99, 0xcc],
        "the scene really is tone mapped"
    );
    let img = draw(&mut hr, &mut pts, false);
    let plain = px(&img, 128, 64);
    for (c, want) in [0x33u8, 0x99, 0xcc].iter().enumerate() {
        assert!(
            plain[c].abs_diff(*want) <= 1,
            "exact data colour: {plain:?}"
        );
    }
}

#[test]
fn curved_paths_are_mitred_at_their_joints_too() {
    let Some(mut hr) = host() else { return };
    let mut mr = MarkRenderer::new(hr.device());
    // An L on a big sphere centred far behind the screen: great circles, so
    // every segment is drawn in pieces, and the joint between the legs is a
    // path joint, not a piece boundary.
    let r = 10_000.0f64;
    let on = |x: f64, y: f64| [x, y, (r * r - x * x - y * y).sqrt() - r];
    let mut path = SegmentMarks::new();
    path.set_paths(&[vec![on(-60.0, -20.0), on(0.0, -20.0), on(0.0, 40.0)]]);
    path.globe = Some([0.0, 0.0, -r]);
    path.set_widths(&[10.0, 10.0]);
    path.set_default_color([255, 0, 0, 128]);
    let cam = camera(128.0, 64.0);
    let img = frame(
        &mut hr,
        &mut mr,
        &mut scene(),
        &cam,
        Origin::ZERO,
        &mut [&mut path],
    );
    let leg = px(&img, 128 - 30, 64 + 20)[0];
    let joint = px(&img, 128, 64 + 20)[0];
    let corner = px(&img, 128 + 4, 64 + 24)[0];
    assert!(leg > 100 && leg < 200, "one layer of 50% red: {leg}");
    assert!(
        leg.abs_diff(joint) <= 3,
        "the joint is one layer: {leg} vs {joint}"
    );
    assert!(
        leg.abs_diff(corner) <= 3,
        "and the mitre fills its corner: {corner}"
    );
}

#[test]
fn shapes_are_instanced_and_lit() {
    let Some(mut hr) = host() else { return };
    let mut mr = MarkRenderer::new(hr.device());
    let mut boxes = ShapeMarks::new(&BoxGeometry::new(1.0, 1.0, 1.0));
    boxes.set_positions(&[[-64.0, 0.0, 0.0], [64.0, 0.0, 0.0]]);
    boxes.set_scales(&[[20.0, 20.0, 20.0], [10.0, 60.0, 10.0]]);
    boxes.set_default_color([255, 255, 255, 255]);
    let cam = camera(128.0, 64.0);
    let img = frame(
        &mut hr,
        &mut mr,
        &mut scene(),
        &cam,
        Origin::ZERO,
        &mut [&mut boxes],
    );
    let a = px(&img, 64, 64);
    assert!(!is_bg(a) && a[0] < 255, "lit, not flat white: {a:?}");
    assert!(!is_bg(px(&img, 64 + 8, 64)));
    assert!(is_bg(px(&img, 64 + 13, 64)));
    // The tall one is 60 tall and 10 wide.
    assert!(!is_bg(px(&img, 192, 64 - 28)));
    assert!(is_bg(px(&img, 192 + 8, 64)));
}

#[test]
fn density_colours_the_cluster_and_nothing_else() {
    let Some(mut hr) = host() else { return };
    let mut mr = MarkRenderer::new(hr.device());
    let mut heat = DensityMarks::new();
    let rows: Vec<[f64; 3]> = (0..40)
        .map(|i| {
            let a = i as f64 * 0.7;
            [-60.0 + a.cos() * 4.0, a.sin() * 4.0, 0.0]
        })
        .collect();
    heat.set_positions(&rows);
    heat.style_mut().size_scale = 20.0;
    heat.max_weight = 10.0;
    let cam = camera(128.0, 64.0);
    let img = frame(
        &mut hr,
        &mut mr,
        &mut scene(),
        &cam,
        Origin::ZERO,
        &mut [&mut heat],
    );
    let hot = px(&img, 68, 64);
    assert!(!is_bg(hot), "the cluster is coloured: {hot:?}");
    assert!(is_bg(px(&img, 200, 64)), "far from it is not");
    // Hotter in the middle than at the edge of the kernel.
    let edge = px(&img, 68 + 18, 64);
    let lum = |p: [u8; 4]| p[0] as u32 + p[1] as u32 + p[2] as u32;
    assert!(lum(hot) > lum(edge), "{hot:?} vs {edge:?}");
}

/// `auto_max` scales to the densest pixel, measured on the GPU: the peak
/// reaches the top of the ramp whatever the weights, and a half puts it
/// there at half the peak.
#[test]
fn density_scales_to_its_own_maximum() {
    let Some(mut hr) = host() else { return };
    let mut mr = MarkRenderer::new(hr.device());
    let cam = camera(128.0, 64.0);
    let lum = |img: &[u8], x: u32| {
        let p = px(img, x, 64);
        p[0] as u32 + p[1] as u32 + p[2] as u32
    };
    let draw =
        |hr: &mut HeadlessRenderer, mr: &mut MarkRenderer, weight: f32, auto: Option<f32>| {
            let mut heat = DensityMarks::new();
            // Two clusters: one kernel on the left, three stacked on the right.
            heat.set_positions(&[
                [-60.0, 0.0, 0.0],
                [60.0, 0.0, 0.0],
                [60.0, 0.0, 0.0],
                [60.0, 0.0, 0.0],
            ]);
            heat.set_weights(&[weight; 4]);
            heat.style_mut().size_scale = 24.0;
            heat.max_weight = 1.0;
            heat.threshold = 1e-3;
            heat.auto_max = auto;
            heat.set_ramp(
                &threers::math::ColorRamp::new(vec![
                    (0.0, Color::BLACK),
                    (1.0, Color::from_hex(0xffffff)),
                ])
                .in_linear_rgb(),
            );
            frame(hr, mr, &mut scene(), &cam, Origin::ZERO, &mut [&mut heat])
        };
    // Fixed: a hundredth of the weight barely shows.
    let fixed = draw(&mut hr, &mut mr, 0.01, None);
    assert!(
        lum(&fixed, 188) < 150,
        "fixed scale, light weights: {}",
        lum(&fixed, 188)
    );
    // Scaled: the stack reaches white, the lone kernel a third of the way
    // (in linear light) — at either weight.
    for w in [0.01, 100.0] {
        let img = draw(&mut hr, &mut mr, w, Some(1.0));
        assert!(lum(&img, 188) > 740, "weight {w}: peak {}", lum(&img, 188));
        let lone = px(&img, 68, 64)[0] as f32 / 255.0;
        // sRGB 0.6 is linear 0.32.
        assert!(
            (0.55..0.66).contains(&lone),
            "weight {w}: lone kernel {lone}"
        );
    }
    // Half the peak at the top: the lone kernel two thirds of the way.
    let half = draw(&mut hr, &mut mr, 1.0, Some(0.5));
    let lone = px(&half, 68, 64)[0] as f32 / 255.0;
    assert!((0.78..0.9).contains(&lone), "half: lone kernel {lone}");
}

#[test]
fn a_coarse_density_field_stays_round_and_smooth() {
    let Some(mut hr) = host() else { return };
    let mut mr = MarkRenderer::new(hr.device());
    // One kernel 48 px in radius: the automatic field is a quarter of the
    // viewport, twelve texels across the radius. Through a black-to-white
    // ramp, so the colour is the field.
    let mut heat = DensityMarks::new();
    heat.set_positions(&[[0.0, 0.0, 0.0]]);
    heat.style_mut().size_scale = 48.0;
    heat.max_weight = 1.0;
    heat.threshold = 1e-3;
    heat.set_ramp(
        &threers::math::ColorRamp::new(vec![(0.0, Color::BLACK), (1.0, Color::from_hex(0xffffff))])
            .in_linear_rgb(),
    );
    assert_eq!(heat.field_scale(), 0.25);
    let cam = camera(128.0, 64.0);
    let img = frame(
        &mut hr,
        &mut mr,
        &mut scene(),
        &cam,
        Origin::ZERO,
        &mut [&mut heat],
    );
    let lum = |x: i32, y: i32| {
        let p = px(&img, (128 + x) as u32, (64 + y) as u32);
        p[0] as i32 + p[1] as i32 + p[2] as i32
    };
    // Round: the same at the same distance in four directions. The kernel's
    // centre is the corner between four pixels, so pixel `-r - 1` mirrors `r`.
    for r in [10, 20, 30] {
        let ring = [lum(r, 0), lum(-r - 1, 0), lum(0, r), lum(0, -r - 1)];
        let spread = ring.iter().max().unwrap() - ring.iter().min().unwrap();
        assert!(spread <= 12, "radius {r}: {ring:?}");
    }
    // And it is the full-resolution field to within 2.5 % of the ramp, in
    // linear light — sRGB bytes would magnify the dark rim tenfold.
    heat.resolution = Some(1.0);
    let full = frame(
        &mut hr,
        &mut mr,
        &mut scene(),
        &cam,
        Origin::ZERO,
        &mut [&mut heat],
    );
    let linear = |b: u8| {
        let s = b as f32 / 255.0;
        if s <= 0.04045 {
            s / 12.92
        } else {
            ((s + 0.055) / 1.055).powf(2.4)
        }
    };
    let mut worst = (0.0f32, 0, 0);
    for y in 64 - 50..64 + 50 {
        for x in 128 - 50..128 + 50 {
            let d = (linear(px(&img, x, y)[0]) - linear(px(&full, x, y)[0])).abs();
            if d > worst.0 {
                worst = (d, x, y);
            }
        }
    }
    assert!(
        worst.0 <= 0.025,
        "worst difference {} at {:?}",
        worst.0,
        (worst.1, worst.2)
    );
}

#[cfg(feature = "captions")]
#[test]
fn labels_and_sprites_collide_through_a_shared_index() {
    use threers::captions::CaptionFont;
    use threers::marks::{CollisionIndex, SpriteMarks, TextLabel, TextLabels, TextureAtlas};
    let Some(mut hr) = host() else { return };
    let mut mr = MarkRenderer::new(hr.device());
    let mut atlas = TextureAtlas::new(32);
    atlas.insert("dot", 8, 8, &[255; 8 * 8 * 4]).unwrap();
    let atlas = atlas.shared();
    let index = CollisionIndex::new().shared();
    let mut icons = SpriteMarks::new(atlas);
    icons.set_positions(&[[0.0, 0.0, 0.0], [6.0, 0.0, 0.0], [-80.0, 0.0, 0.0]]);
    icons.set_images(&["dot", "dot", "dot"]);
    icons.style_mut().size_scale = 16.0;
    icons.collision = true;
    // Within the layer the second icon outranks the first it overlaps.
    icons.set_priorities(&[0.0, 1.0, 0.0]);
    icons.collision_index = Some(index.clone());
    let mut labels = TextLabels::new(CaptionFont::builtin());
    labels.font_size = 16.0;
    labels.set_labels(vec![TextLabel::new([6.0, 0.0, 0.0], "HI")]);
    labels.collision_index = Some(index.clone());
    let cam = camera(128.0, 64.0);

    index.lock().unwrap().clear();
    frame(
        &mut hr,
        &mut mr,
        &mut scene(),
        &cam,
        Origin::ZERO,
        &mut [&mut icons, &mut labels],
    );
    assert_eq!(
        icons.shown(),
        vec![false, true, true],
        "priority within the layer"
    );
    assert_eq!(
        labels.visible(),
        &[false],
        "the icon drawn first wins the label's spot"
    );

    // The next frame, labels first: now the label wins and the icon under
    // it goes.
    index.lock().unwrap().clear();
    frame(
        &mut hr,
        &mut mr,
        &mut scene(),
        &cam,
        Origin::ZERO,
        &mut [&mut labels, &mut icons],
    );
    assert_eq!(labels.visible(), &[true]);
    assert_eq!(icons.shown(), vec![false, false, true]);
}

#[cfg(feature = "captions")]
#[test]
fn a_label_along_a_line_draws_turned_with_it() {
    use threers::captions::CaptionFont;
    use threers::marks::{TextLabel, TextLabels};
    let Some(mut hr) = host() else { return };
    let mut mr = MarkRenderer::new(hr.device());
    let cam = camera(128.0, 64.0);
    // The correlation of ink x and y on screen: negative when the ink runs
    // up to the right (screen y grows downwards), near zero when level.
    let slope = |img: &[u8]| {
        let ink: Vec<(f32, f32)> = (0..H)
            .flat_map(|y| (0..W).map(move |x| (x, y)))
            .filter(|&(x, y)| px(img, x, y)[0] > 128)
            .map(|(x, y)| (x as f32, y as f32))
            .collect();
        let n = ink.len() as f32;
        let (mx, my) = ink
            .iter()
            .fold((0.0, 0.0), |a, p| (a.0 + p.0 / n, a.1 + p.1 / n));
        let (sxy, sxx, syy) = ink.iter().fold((0.0, 0.0, 0.0), |a, p| {
            let (dx, dy) = (p.0 - mx, p.1 - my);
            (a.0 + dx * dy, a.1 + dx * dx, a.2 + dy * dy)
        });
        (ink.len(), sxy / (sxx * syy).sqrt().max(1e-6))
    };
    let mut labels = TextLabels::new(CaptionFont::builtin());
    labels.font_size = 14.0;
    labels.set_labels(vec![TextLabel::along(
        vec![[-90.0, -45.0, 0.0], [90.0, 45.0, 0.0]],
        "DIAGONAL",
    )]);
    let img = frame(
        &mut hr,
        &mut mr,
        &mut scene(),
        &cam,
        Origin::ZERO,
        &mut [&mut labels],
    );
    let (count, r) = slope(&img);
    assert!(count > 150, "text drawn: {count} px");
    assert!(r < -0.6, "running up to the right with the line: r = {r}");
    // The same text level for comparison.
    labels.set_labels(vec![TextLabel::new([0.0, 0.0, 0.0], "DIAGONAL")]);
    let img = frame(
        &mut hr,
        &mut mr,
        &mut scene(),
        &cam,
        Origin::ZERO,
        &mut [&mut labels],
    );
    let (_, level) = slope(&img);
    assert!(level.abs() < 0.2, "level text: r = {level}");
}

#[cfg(feature = "captions")]
#[test]
fn labels_draw_text_and_cull_overlaps() {
    use threers::captions::CaptionFont;
    use threers::marks::{TextLabel, TextLabels};
    let Some(mut hr) = host() else { return };
    let mut mr = MarkRenderer::new(hr.device());
    let mut labels = TextLabels::new(CaptionFont::builtin());
    labels.font_size = 24.0;
    let mut a = TextLabel::new([0.0, 0.0, 0.0], "HELLO");
    a.priority = 1.0;
    let b = TextLabel::new([4.0, 2.0, 0.0], "WORLD");
    let c = TextLabel::new([0.0, 45.0, 0.0], "UP");
    labels.set_labels(vec![a, b, c]);
    let cam = camera(128.0, 64.0);
    let img = frame(
        &mut hr,
        &mut mr,
        &mut scene(),
        &cam,
        Origin::ZERO,
        &mut [&mut labels],
    );
    assert_eq!(labels.visible(), &[true, false, true]);
    let ink = |x0: u32, x1: u32, y0: u32, y1: u32| {
        (y0..y1)
            .flat_map(|y| (x0..x1).map(move |x| (x, y)))
            .filter(|&(x, y)| px(&img, x, y)[0] > 128)
            .count()
    };
    // "HELLO" at 24 px is about 120 px wide, centred on the middle.
    assert!(ink(70, 186, 52, 76) > 200, "text drawn around the anchor");
    assert_eq!(ink(0, 50, 100, 128), 0, "and nowhere near the corner");
    assert!(ink(100, 156, 5, 30) > 30, "the label above");
}

#[test]
fn marks_draw_over_a_multisampled_frame_and_its_depth() {
    let Some(mut hr) = host() else { return };
    hr.set_msaa(4);
    let mut mr = MarkRenderer::new(hr.device());
    let mut s = scene();
    let mut wall = BoxGeometry::new(100.0, 100.0, 1.0);
    for v in wall
        .attributes
        .get_mut("position")
        .unwrap()
        .array
        .chunks_mut(3)
    {
        v[0] -= 64.0;
    }
    s.add_mesh(
        wall,
        Material::Basic(BasicMaterial::new(Color::from_hex(0x0000ff))),
    );
    let mut points = PointMarks::new();
    points.set_positions(&[[-64.0, 0.0, -20.0], [-64.0, 30.0, 5.0]]);
    points.set_radii(&[6.0, 6.0]);
    points.set_default_color([255, 0, 0, 255]);
    let cam = camera(128.0, 64.0);
    hr.render(&mut s, &cam);
    assert_eq!(
        hr.mark_target().samples,
        4,
        "drawn into the multisampled attachments"
    );
    let img = frame(
        &mut hr,
        &mut mr,
        &mut s,
        &cam,
        Origin::ZERO,
        &mut [&mut points],
    );
    let hidden = px(&img, 64, 64);
    assert!(
        hidden[2] > 200 && hidden[0] < 20,
        "behind the wall: {hidden:?}"
    );
    assert_eq!(px(&img, 64, 34)[0], 255, "in front of it");
}

#[cfg(feature = "captions")]
fn dejavu() -> threers::captions::CaptionFont {
    threers::captions::CaptionFont::from_ttf_bytes(include_bytes!("../src/openscad/DejaVuSans.ttf"))
        .unwrap()
}

/// Bright pixels, white text on black.
#[cfg(feature = "captions")]
fn ink(img: &[u8], x0: u32, y0: u32, x1: u32, y1: u32) -> usize {
    (y0..y1)
        .flat_map(|y| (x0..x1).map(move |x| (x, y)))
        .filter(|&(x, y)| px(img, x, y)[0] > 128)
        .count()
}

/// A "T" drawn ten times larger than it was rasterized: the single distance
/// field rounds the crossbar's corner off, the multi-channel one keeps it.
#[cfg(feature = "captions")]
#[test]
fn msdf_text_keeps_its_corners_when_enlarged() {
    use threers::marks::{GlyphField, TextLabel, TextLabels};
    let Some(mut hr) = host() else { return };
    let mut mr = MarkRenderer::new(hr.device());
    let cam = camera(128.0, 64.0);
    let corner_ink = |hr: &mut HeadlessRenderer, mr: &mut MarkRenderer, field: GlyphField| {
        let mut labels = TextLabels::new(dejavu());
        labels.font_size = 110.0;
        labels.raster_size = 11.0;
        labels.glyph_field = field;
        labels.set_labels(vec![TextLabel::new([0.0, 0.0, 0.0], "T")]);
        let img = frame(hr, mr, &mut scene(), &cam, Origin::ZERO, &mut [&mut labels]);
        // Ink at over about 40% coverage: the crossbar's top edge can fall
        // halfway through a row, which white text puts at an encoded 128
        // (blended as CoreText does), and a cut at exactly half would take
        // that row as inked or not by a rounding.
        let inked = |x: u32, y: u32| px(&img, x, y)[0] > 96;
        // The crossbar's top-left corner: the first inked column and row.
        let (mut x0, mut y0) = (W, H);
        for y in 0..H {
            for x in 0..W {
                if inked(x, y) {
                    x0 = x0.min(x);
                    y0 = y0.min(y);
                }
            }
        }
        assert!(x0 < W, "{field:?}: something drawn");
        (y0..y0 + 6).flat_map(|y| (x0..x0 + 6).map(move |x| (x, y))).filter(|&(x, y)| inked(x, y)).count()
    };
    let sdf = corner_ink(&mut hr, &mut mr, GlyphField::Sdf);
    let msdf = corner_ink(&mut hr, &mut mr, GlyphField::Msdf);
    assert!(msdf >= 32, "msdf corner square: {msdf} of 36");
    assert!(msdf >= sdf + 4, "msdf {msdf} against sdf {sdf}");
}

/// The bitmap face has no omega; a fallback face draws it.
#[cfg(feature = "captions")]
#[test]
fn a_fallback_face_draws_what_the_first_lacks() {
    use threers::captions::CaptionFont;
    use threers::marks::{TextLabel, TextLabels};
    let Some(mut hr) = host() else { return };
    let mut mr = MarkRenderer::new(hr.device());
    let cam = camera(128.0, 64.0);
    let mut labels = TextLabels::new(CaptionFont::builtin());
    labels.font_size = 32.0;
    labels.set_labels(vec![TextLabel::new([0.0, 0.0, 0.0], "Ω")]);
    let img = frame(
        &mut hr,
        &mut mr,
        &mut scene(),
        &cam,
        Origin::ZERO,
        &mut [&mut labels],
    );
    let stand_in = img;
    labels.add_fallback(dejavu());
    let img = frame(
        &mut hr,
        &mut mr,
        &mut scene(),
        &cam,
        Origin::ZERO,
        &mut [&mut labels],
    );
    let with = ink(&img, 0, 0, W, H);
    let changed = img
        .chunks(4)
        .zip(stand_in.chunks(4))
        .filter(|(a, b)| a[0].abs_diff(b[0]) > 128)
        .count();
    // DejaVu's own omega, drawn by itself.
    let mut own = TextLabels::new(dejavu());
    own.font_size = 32.0;
    own.set_labels(vec![TextLabel::new([0.0, 0.0, 0.0], "Ω")]);
    let img = frame(
        &mut hr,
        &mut mr,
        &mut scene(),
        &cam,
        Origin::ZERO,
        &mut [&mut own],
    );
    let reference = ink(&img, 0, 0, W, H);
    assert!(
        (with as f32 / reference as f32 - 1.0).abs() < 0.1,
        "the fallback draws DejaVu's omega: {with} px against {reference}"
    );
    assert!(
        changed > 100,
        "not the bitmap face's missing-glyph box: {changed} px changed"
    );
}

/// `\n` starts a line 1.2 em lower; a shorter line is centred under a
/// centred label's longest.
#[cfg(feature = "captions")]
#[test]
fn a_label_breaks_lines() {
    use threers::captions::CaptionFont;
    use threers::marks::{TextLabel, TextLabels};
    let Some(mut hr) = host() else { return };
    let mut mr = MarkRenderer::new(hr.device());
    let cam = camera(128.0, 64.0);
    let mut bounds = |text: &str| {
        let mut labels = TextLabels::new(CaptionFont::builtin());
        labels.font_size = 16.0;
        labels.set_labels(vec![TextLabel::new([0.0, 0.0, 0.0], text)]);
        let img = frame(&mut hr, &mut mr, &mut scene(), &cam, Origin::ZERO, &mut [&mut labels]);
        let (mut x0, mut y0, mut x1, mut y1) = (W, H, 0, 0);
        for y in 0..H {
            for x in 0..W {
                if img[((y * W + x) * 4) as usize] > 128 {
                    (x0, y0, x1, y1) = (x0.min(x), y0.min(y), x1.max(x), y1.max(y));
                }
            }
        }
        (x0 as f32, y0 as f32, x1 as f32, y1 as f32)
    };
    let one = bounds("HHHH");
    let two = bounds("HHHH\nH");
    let under = bounds("    \nH");
    let taller = (two.3 - two.1) - (one.3 - one.1);
    assert!((taller - 19.2).abs() <= 1.5, "a line lower: {taller} px");
    assert!((one.1 - two.1 - 9.6).abs() <= 1.5, "the block centred: {one:?} {two:?}");
    let (mid_one, mid_under) = ((one.0 + one.2) * 0.5, (under.0 + under.2) * 0.5);
    assert!((mid_one - mid_under).abs() <= 1.0, "the short line centred: {mid_one} {mid_under}");
}

/// Labels wrap to a width at spaces, and can sit on a padded, bordered
/// box.
#[cfg(feature = "captions")]
#[test]
fn labels_wrap_and_take_backgrounds() {
    use threers::captions::CaptionFont;
    use threers::marks::{LabelBackground, TextLabel, TextLabels};
    let Some(mut hr) = host() else { return };
    let mut mr = MarkRenderer::new(hr.device());
    let cam = camera(128.0, 64.0);
    let mut extent = |max_width: Option<f32>| {
        let mut labels = TextLabels::new(CaptionFont::builtin());
        labels.font_size = 16.0;
        labels.max_width = max_width;
        labels.set_labels(vec![TextLabel::new([0.0, 0.0, 0.0], "HHHH HHHH")]);
        let img = frame(&mut hr, &mut mr, &mut scene(), &cam, Origin::ZERO, &mut [&mut labels]);
        let lit: Vec<(u32, u32)> = (0..H).flat_map(|y| (0..W).map(move |x| (x, y))).filter(|&(x, y)| px(&img, x, y)[0] > 128).collect();
        let (x0, x1) = (lit.iter().map(|p| p.0).min().unwrap(), lit.iter().map(|p| p.0).max().unwrap());
        let (y0, y1) = (lit.iter().map(|p| p.1).min().unwrap(), lit.iter().map(|p| p.1).max().unwrap());
        (x1 - x0, y1 - y0)
    };
    let (w1, h1) = extent(None);
    let (w2, h2) = extent(Some(80.0));
    assert!(w2 < w1 * 6 / 10 && h2 > h1 + 10, "wrapped onto two lines: {w1}x{h1} to {w2}x{h2}");

    let mut labels = TextLabels::new(CaptionFont::builtin());
    labels.font_size = 16.0;
    let mut label = TextLabel::new([0.0, 0.0, 0.0], "HI");
    label.color = [255, 255, 255, 255];
    label.background = Some(LabelBackground { color: [255, 0, 0, 255], padding: [10.0; 4], border_color: [0, 255, 0, 255], border_width: 3.0 });
    labels.set_labels(vec![label]);
    let img = frame(&mut hr, &mut mr, &mut scene(), &cam, Origin::ZERO, &mut [&mut labels]);
    let reds = img.chunks(4).filter(|p| p[0] > 200 && p[1] < 50).count();
    let greens = img.chunks(4).filter(|p| p[1] > 200 && p[0] < 50).count();
    let whites = img.chunks(4).filter(|p| p[0] > 200 && p[1] > 200).count();
    assert!(reds > 300 && greens > 100 && whites > 20, "box {reds}, border {greens}, text {whites}");
}

/// Glyphs that overlap (joined scripts do): every halo goes under every
/// fill, so one glyph's halo does not eat into the ink of the one before.
#[cfg(feature = "captions")]
#[test]
fn halos_go_under_overlapping_glyphs() {
    use std::collections::HashMap;
    use threers::captions::{CaptionFont, SdfGlyph, SdfGlyphs};
    use threers::marks::{TextLabel, TextLabels};
    let Some(mut hr) = host() else { return };
    let mut mr = MarkRenderer::new(hr.device());
    let cam = camera(128.0, 64.0);
    // A 12-pixel square advancing only 6: each glyph covers half the last.
    let (w, h, buffer) = (12u32, 12u32, 3u32);
    let (fw, fh) = (w + 2 * buffer, h + 2 * buffer);
    let field = (0..fw * fh)
        .map(|i| {
            let (x, y) = ((i % fw) as f32 + 0.5, (i / fw) as f32 + 0.5);
            let inside = (x - buffer as f32).min(y - buffer as f32).min((fw - buffer) as f32 - x).min((fh - buffer) as f32 - y);
            (192.0 + inside * 32.0).clamp(0.0, 255.0) as u8
        })
        .collect();
    let mut glyphs = HashMap::new();
    glyphs.insert(0x4e00, SdfGlyph { width: w, height: h, left: 0, top: -10, advance: 6.0, field });
    let font = std::sync::Arc::new(SdfGlyphs { size: 24.0, buffer, top_origin: 26.0, glyphs });
    let mut white = |halo: f32| {
        let mut labels = TextLabels::new(CaptionFont::from_sdf_glyphs(font.clone()));
        labels.font_size = 48.0;
        labels.halo_width = halo;
        labels.halo_color = [255, 0, 0, 255];
        labels.set_labels(vec![TextLabel::new([0.0, 0.0, 0.0], "\u{4e00}\u{4e00}\u{4e00}")]);
        let img = frame(&mut hr, &mut mr, &mut scene(), &cam, Origin::ZERO, &mut [&mut labels]);
        img.chunks(4).filter(|p| p[0] > 200 && p[1] > 200 && p[2] > 200).count()
    };
    let (plain, haloed) = (white(0.0), white(4.0));
    assert!(plain > 500, "{plain} px of ink");
    assert!(haloed as f32 >= plain as f32 * 0.97, "the ink under the halos: {haloed} px against {plain}");
}

/// A label layer replaced while the view holds still (a map tile styled
/// again): the new one takes the spot, not blocked by the old one's boxes.
#[cfg(feature = "captions")]
#[test]
fn a_replaced_label_layer_gives_back_its_spot() {
    use threers::captions::CaptionFont;
    use threers::marks::{CollisionIndex, TextLabel, TextLabels};
    let Some(mut hr) = host() else { return };
    let mut mr = MarkRenderer::new(hr.device());
    let index = CollisionIndex::new().shared();
    let cam = camera(128.0, 64.0);
    let view = MarkView::new(&cam, Origin::ZERO, (W, H));
    let make = || {
        let mut l = TextLabels::new(CaptionFont::builtin());
        l.font_size = 16.0;
        l.collision = true;
        l.set_labels(vec![TextLabel::new([0.0, 0.0, 0.0], "HI")]);
        l.collision_index = Some(index.clone());
        l
    };
    let mut old = make();
    for _ in 0..2 {
        index.lock().unwrap().begin_frame(&view);
        frame(&mut hr, &mut mr, &mut scene(), &cam, Origin::ZERO, &mut [&mut old]);
    }
    assert_eq!(old.visible(), &[true]);
    drop(old);
    let mut new = make();
    index.lock().unwrap().begin_frame(&view);
    frame(&mut hr, &mut mr, &mut scene(), &cam, Origin::ZERO, &mut [&mut new]);
    assert_eq!(new.visible(), &[true], "not hidden behind the dropped layer's box");
}

/// A distance-field face first (MapLibre's glyph PBFs hold only the ranges
/// fetched) and a TrueType face for the rest: both draw.
#[cfg(feature = "captions")]
#[test]
fn a_distance_field_face_and_a_fallback_both_draw() {
    use std::collections::HashMap;
    use threers::captions::{CaptionFont, SdfGlyph, SdfGlyphs};
    use threers::marks::{TextLabel, TextLabels};
    let Some(mut hr) = host() else { return };
    let mut mr = MarkRenderer::new(hr.device());
    let cam = camera(128.0, 64.0);
    // A 12-pixel square, a 3-pixel field around it.
    let (w, h, buffer) = (12u32, 12u32, 3u32);
    let (fw, fh) = (w + 2 * buffer, h + 2 * buffer);
    let field = (0..fw * fh)
        .map(|i| {
            let (x, y) = ((i % fw) as f32 + 0.5, (i / fw) as f32 + 0.5);
            let inside = (x - buffer as f32).min(y - buffer as f32).min((fw - buffer) as f32 - x).min((fh - buffer) as f32 - y);
            (192.0 + inside * 32.0).clamp(0.0, 255.0) as u8
        })
        .collect();
    let mut glyphs = HashMap::new();
    glyphs.insert(0x4e00, SdfGlyph { width: w, height: h, left: 1, top: -10, advance: 14.0, field });
    let set = SdfGlyphs { size: 24.0, buffer, top_origin: 26.0, glyphs };
    let mut draw = |labels: &mut TextLabels, text: &str| {
        labels.font_size = 32.0;
        labels.halo_width = 2.0;
        labels.halo_color = [255, 0, 0, 255];
        labels.collision = true;
        labels.set_labels(vec![TextLabel::new([0.0, 0.0, 0.0], text)]);
        let img = frame(&mut hr, &mut mr, &mut scene(), &cam, Origin::ZERO, &mut [labels]);
        ink(&img, 0, 0, W, H)
    };
    let mut both = TextLabels::new(CaptionFont::from_sdf_glyphs(std::sync::Arc::new(set)));
    both.add_fallback(dejavu());
    let with = draw(&mut both, "A\u{4e00}");
    let alone = draw(&mut TextLabels::new(dejavu()), "A");
    // The square is 16 px across at 32 px per em.
    assert!(with > alone + 200, "the square beside DejaVu's A: {with} px against {alone}");
}

/// An icon beside the text: drawn in its own colour to the left, and part of
/// the symbol's box, so a label overlapping only the icon is culled.
#[cfg(feature = "captions")]
#[test]
fn an_icon_draws_beside_its_text_and_collides_as_one() {
    use threers::marks::{LabelIcon, TextLabel, TextLabels};
    let Some(mut hr) = host() else { return };
    let mut mr = MarkRenderer::new(hr.device());
    let cam = camera(128.0, 64.0);
    let mut labels = TextLabels::new(dejavu());
    labels.font_size = 20.0;
    labels.add_icon("square", 8, 8, &[255; 64]);
    let mut icon = LabelIcon::new("square", 20.0);
    icon.color = [0, 255, 0, 255];
    let symbol = TextLabel::new([0.0, 0.0, 0.0], "HI").with_icon(icon);
    labels.set_labels(vec![symbol.clone()]);
    let img = frame(
        &mut hr,
        &mut mr,
        &mut scene(),
        &cam,
        Origin::ZERO,
        &mut [&mut labels],
    );
    let green: Vec<u32> = (0..W)
        .filter(|&x| {
            let p = px(&img, x, 64);
            p[1] > 200 && p[0] < 60
        })
        .collect();
    let white: Vec<u32> = (0..W)
        .filter(|&x| (54..74).any(|y| px(&img, x, y)[0] > 200 && px(&img, x, y)[1] > 200))
        .collect();
    assert!(
        green.len() >= 16,
        "the icon is about 20 px wide: {}",
        green.len()
    );
    assert!(!white.is_empty());
    assert!(
        green.iter().max() < white.iter().min(),
        "icon {green:?} left of text {white:?}"
    );
    // A second label over the icon only, ranked lower: culled by the
    // symbol's box. Moved clear, it shows.
    let icon_x = (green[0] + green[green.len() - 1]) as f64 / 2.0 - 128.0;
    let mut low = TextLabel::new([icon_x, 0.0, 0.0], ".");
    low.priority = -1.0;
    labels.set_labels(vec![symbol.clone(), low.clone()]);
    frame(
        &mut hr,
        &mut mr,
        &mut scene(),
        &cam,
        Origin::ZERO,
        &mut [&mut labels],
    );
    assert_eq!(labels.visible(), &[true, false]);
    low.position = [icon_x, 40.0, 0.0];
    labels.set_labels(vec![symbol, low]);
    frame(
        &mut hr,
        &mut mr,
        &mut scene(),
        &cam,
        Origin::ZERO,
        &mut [&mut labels],
    );
    assert_eq!(labels.visible(), &[true, true]);
}

/// Lit points take their colour from their normals: one facing the light
/// is brighter than one facing away, and one without lighting is as set.
#[test]
fn lit_points_shade_by_their_normals() {
    let Some(mut hr) = host() else { return };
    let mut mr = MarkRenderer::new(hr.device());
    let mut points = PointMarks::new();
    points.set_positions(&[[-64.0, 0.0, 0.0], [64.0, 0.0, 0.0]]);
    points.set_radii(&[10.0, 10.0]);
    points.set_default_color([200, 200, 200, 255]);
    let unlit = frame(&mut hr, &mut mr, &mut scene(), &camera(128.0, 64.0), Origin::ZERO, &mut [&mut points]);
    assert_eq!(px(&unlit, 64, 64)[0], 200);
    points.lighting = Some(threers::marks::Lighting {
        ambient: 0.2,
        diffuse: 0.8,
        specular: 0.0,
        shininess: 1.0,
        lights: [([0.0, 0.0, 1.0], 1.0), ([0.0, 0.0, 1.0], 0.0)],
        srgb: true,
        pbr: None,
        scene: false,
        specular_color: [0.0; 3],
    });
    points.set_normals(&[[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]]);
    let lit = frame(&mut hr, &mut mr, &mut scene(), &camera(128.0, 64.0), Origin::ZERO, &mut [&mut points]);
    let (toward, away) = (px(&lit, 64, 64)[0], px(&lit, 192, 64)[0]);
    assert!((toward as i32 - 200).abs() <= 1, "facing the light: ambient + diffuse = 1 ({toward})");
    assert!((away as i32 - 40).abs() <= 1, "facing away: ambient only ({away})");
}

/// A target's rectangle: marks drawn within it only, its view the
/// rectangle's size, and a clear that leaves the rest alone.
#[test]
fn marks_draw_and_clear_within_a_rect() {
    let Some(mut hr) = host() else { return };
    let mut mr = MarkRenderer::new(hr.device());
    let mut points = PointMarks::new();
    points.set_positions(&[[0.0, 0.0, 0.0]]);
    points.set_radii(&[200.0]);
    points.set_default_color([255, 0, 0, 255]);
    let cam = camera(64.0, 64.0);
    hr.render(&mut scene(), &cam);
    let device = hr.device().clone();
    let queue = hr.queue().clone();
    // The right half: a view 128 wide, cleared green, then a huge red dot
    // that would cover the whole target.
    let view = MarkView::new(&cam, Origin::ZERO, (W / 2, H));
    {
        let target = hr.mark_target().with_rect([W / 2, 0, W / 2, H]);
        mr.clear(&device, &queue, &target, Some(threers::wgpu::Color::GREEN), Some(1.0));
    }
    let before = hr.read_rgba();
    assert!(is_bg(px(&before, 10, 10)), "left half untouched by the clear");
    assert_eq!(px(&before, 200, 10)[1], 255, "right half cleared green");
    {
        let target = hr.mark_target().with_rect([W / 2, 0, W / 2, H]);
        mr.draw(&device, &queue, &target, &view, &mut [&mut points]);
    }
    let img = hr.read_rgba();
    assert!(is_bg(px(&img, 10, 64)), "nothing drawn left of the rect");
    assert_eq!(px(&img, 192, 64), [255, 0, 0, 255], "the dot fills the rect");
}
