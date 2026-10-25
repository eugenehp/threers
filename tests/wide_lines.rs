//! `LineBasicMaterial::line_width` draws lines that wide, in device pixels.
//!
//! three.js's `linewidth` does nothing under WebGL, and WebGPU has no line
//! width at all; threers draws a segment wider than a pixel as a round-capped
//! quad (`vs_wide_lines`). These render wide lines — indexed and not, opaque
//! and blended, with a draw range, per-vertex colours — and measure them.
//!
//! Skipped when no GPU adapter is available.

use threers::{
    BufferAttribute, BufferGeometry, Color, HeadlessRenderer, LineBasicMaterial, LineSegments, Material, Object3D,
    OrthographicCamera, PerspectiveCamera, Scene, Vector3,
};

const W: u32 = 128;
const H: u32 = 128;

fn renderer() -> Option<HeadlessRenderer> {
    HeadlessRenderer::builder().size(W, H).color_format(wgpu::TextureFormat::Rgba8Unorm).build().ok()
}

fn camera() -> OrthographicCamera {
    // world x, y in [-1, 1] fill the frame: 64 px a unit
    let mut c = OrthographicCamera::new(-1.0, 1.0, 1.0, -1.0, 0.1, 10.0);
    c.position = Vector3::new(0.0, 0.0, 5.0);
    c.look_at(Vector3::new(0.0, 0.0, 0.0));
    c
}

fn material(width: f32, opacity: f32) -> LineBasicMaterial {
    let mut m = LineBasicMaterial::default();
    m.color = Color::from_hex(0xffffff);
    m.opacity = opacity;
    m.line_width = width;
    m
}

fn lines(pos: &[f32], index: Option<Vec<u32>>, m: LineBasicMaterial) -> Object3D {
    let mut g = BufferGeometry::new();
    g.set_attribute("position", BufferAttribute::new(pos.to_vec(), 3));
    if let Some(i) = index {
        g.set_index(i);
    }
    Object3D::line_segments(LineSegments::new(g, Material::Line(m)))
}

fn scene_with(objs: Vec<Object3D>) -> Scene {
    let mut s = Scene::new();
    s.background = Color::from_hex(0x000000);
    for o in objs {
        s.add(o);
    }
    s
}

/// the red channel at pixel (u, v)
fn at(px: &[u8], u: u32, v: u32) -> u8 {
    px[((v * W + u) * 4) as usize]
}

/// how many pixels down column `u` are lit (red above 127)
fn thickness(px: &[u8], u: u32) -> u32 {
    (0..H).filter(|&v| at(px, u, v) > 127).count() as u32
}

/// a horizontal segment across the middle, from x = -0.6 to 0.6
const ACROSS: [f32; 6] = [-0.6, 0.0, 0.0, 0.6, 0.0, 0.0];

#[test]
fn a_line_is_as_wide_as_asked() {
    let Some(mut r) = renderer() else { return };
    // centred on a pixel boundary, an odd width covers one pixel centre
    // fewer or more than it is wide
    for (width, want) in [(1.0, 1..=2), (4.0, 4..=5), (9.0, 8..=10), (12.0, 12..=13)] {
        let px = r.render_to_rgba(&mut scene_with(vec![lines(&ACROSS, None, material(width, 1.0))]), &camera());
        let t = thickness(&px, W / 2);
        assert!(want.contains(&t), "width {width}: {t} px down the middle");
    }
}

#[test]
fn indexed_and_not_draw_the_same() {
    let Some(mut r) = renderer() else { return };
    let plain = r.render_to_rgba(&mut scene_with(vec![lines(&ACROSS, None, material(5.0, 1.0))]), &camera());
    // the same segment through an index, over vertices in another order
    let pos = [0.6, 0.0, 0.0, 0.0, 0.9, 0.0, -0.6, 0.0, 0.0];
    let indexed = r.render_to_rgba(&mut scene_with(vec![lines(&pos, Some(vec![2, 0]), material(5.0, 1.0))]), &camera());
    let differ = plain.chunks(4).zip(indexed.chunks(4)).filter(|(a, b)| a[0].abs_diff(b[0]) > 8).count();
    assert!(differ <= 4, "{differ} pixels differ");
    // and the unindexed vertex (0, 0.9) is joined to nothing
    assert_eq!(thickness(&indexed, W / 2), thickness(&plain, W / 2));
}

#[test]
fn ends_are_round_and_joints_closed() {
    let Some(mut r) = renderer() else { return };
    // a right angle out of two segments meeting at the origin
    let pos = [-0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.6, 0.0];
    let px = r.render_to_rgba(&mut scene_with(vec![lines(&pos, None, material(9.0, 1.0))]), &camera());
    let (cu, cv) = (W / 2, H / 2);
    // the joint's outer corner is filled — butt ends would leave a notch
    // there — and rounded: a square cap would fill a pixel further out
    assert!(at(&px, cu + 2, cv + 2) > 127, "joint filled");
    assert!(at(&px, cu + 3, cv + 3) < 128, "joint rounded");
    // the start's cap reaches half a width past it, no further
    let left = ((1.0 - 0.6) * 0.5 * W as f32) as u32;
    assert!(at(&px, left - 3, cv) > 127, "cap past the start");
    assert!(at(&px, left - 7, cv) < 128, "nothing beyond the cap");
}

#[test]
fn blended_lines_fade_at_the_edge() {
    let Some(mut r) = renderer() else { return };
    let px = r.render_to_rgba(&mut scene_with(vec![lines(&ACROSS, None, material(6.0, 0.5))]), &camera());
    let col: Vec<u8> = (0..H).map(|v| at(&px, W / 2, v)).collect();
    let peak = *col.iter().max().unwrap();
    // half opacity over black: the middle near half bright, not full
    assert!((100..=200).contains(&peak), "peak {peak}");
}

/// Thinner than a pixel and transparent, a line takes the share of each pixel
/// it covers rather than lighting whole pixels or none: a 0.4-pixel line puts
/// 0.4 of its colour in each column wherever it falls between pixel centres,
/// and no more than that in any one pixel. Along a row and on a 45°
/// diagonal, whose columns each cross √2 of its width.
#[test]
fn thin_transparent_lines_keep_their_weight_between_pixels() {
    let Some(mut r) = renderer() else { return };
    let width = 0.4f32;
    let px_world = 2.0 / W as f32;
    for diagonal in [false, true] {
        let mut sums = Vec::new();
        for k in 0..8 {
            let d = k as f32 / 8.0 * px_world;
            let pos = if diagonal { [-0.5 + d, -0.5, 0.0, 0.5 + d, 0.5, 0.0] } else { [-0.6, d, 0.0, 0.6, d, 0.0] };
            let mut m = material(width, 1.0);
            m.transparent = Some(true);
            let px = r.render_to_rgba(&mut scene_with(vec![lines(&pos, None, m)]), &camera());
            let expect = if diagonal { width * std::f32::consts::SQRT_2 } else { width };
            for u in [W / 2 - 10, W / 2, W / 2 + 9] {
                let col: Vec<f32> = (0..H).map(|v| at(&px, u, v) as f32 / 255.0).collect();
                let peak = col.iter().cloned().fold(0.0, f32::max);
                assert!(peak <= expect + 0.03, "diagonal {diagonal} step {k} column {u}: peak {peak}");
                sums.push(col.iter().sum::<f32>() / expect);
            }
        }
        for s in &sums {
            assert!((s - 1.0).abs() < 0.06, "diagonal {diagonal}: column ink {s} of the width, all {sums:?}");
        }
    }
    // Opaque, it stays a line primitive: whole pixels.
    let px = r.render_to_rgba(&mut scene_with(vec![lines(&ACROSS, None, material(width, 1.0))]), &camera());
    assert!((0..H).any(|v| at(&px, W / 2, v) == 255), "an opaque thin line still lights its pixel");
}

#[test]
fn a_draw_range_limits_the_segments() {
    let Some(mut r) = renderer() else { return };
    // two segments, one above the other; the range draws only the first
    let pos = [-0.6, 0.5, 0.0, 0.6, 0.5, 0.0, -0.6, -0.5, 0.0, 0.6, -0.5, 0.0];
    let mut g = BufferGeometry::new();
    g.set_attribute("position", BufferAttribute::new(pos.to_vec(), 3));
    g.set_draw_range(0, 2);
    let o = Object3D::line_segments(LineSegments::new(g, Material::Line(material(5.0, 1.0))));
    let px = r.render_to_rgba(&mut scene_with(vec![o]), &camera());
    let v_top = ((1.0 - (0.5 + 1.0) * 0.5) * H as f32) as u32;
    let v_bottom = ((1.0 - (-0.5 + 1.0) * 0.5) * H as f32) as u32;
    assert!(at(&px, W / 2, v_top) > 127);
    assert!(at(&px, W / 2, v_bottom) < 128);
}

#[test]
fn vertex_colours_colour_each_end() {
    let Some(mut r) = renderer() else { return };
    let mut g = BufferGeometry::new();
    g.set_attribute("position", BufferAttribute::new(ACROSS.to_vec(), 3));
    g.set_attribute("color", BufferAttribute::new(vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0], 3));
    // a colour attribute colours lines (no flag, unlike three.js's `vertexColors`)
    let o = Object3D::line_segments(LineSegments::new(g, Material::Line(material(6.0, 1.0))));
    let px = r.render_to_rgba(&mut scene_with(vec![o]), &camera());
    let i = |u: u32| ((H / 2 * W + u) * 4) as usize;
    let (l, rr) = (i(W / 2 - 30), i(W / 2 + 30));
    assert!(px[l] > px[l + 2], "red towards the start");
    assert!(px[rr + 2] > px[rr], "blue towards the end");
}

#[test]
fn width_holds_in_perspective_and_past_the_camera() {
    let Some(mut r) = renderer() else { return };
    let mut cam = PerspectiveCamera::new(60.0, 1.0, 0.1, 100.0);
    cam.position = Vector3::new(0.0, 0.0, 3.0);
    cam.look_at(Vector3::new(0.0, 0.0, 0.0));
    // one near, one far: the same width on screen
    let pos = [-0.5, 0.3, 2.0, 0.5, 0.3, 2.0, -3.0, -1.0, -10.0, 3.0, -1.0, -10.0];
    let px = r.render_to_rgba(&mut scene_with(vec![lines(&pos, None, material(6.0, 1.0))]), &cam);
    let col: Vec<u32> = (0..H).filter(|&v| at(&px, W / 2, v) > 127).collect();
    let runs: Vec<&[u32]> = col.chunk_by(|a, b| b - a == 1).collect();
    assert_eq!(runs.len(), 2, "two lines down the middle: {col:?}");
    for run in runs {
        assert!((5..=7).contains(&run.len()), "a run of {}", run.len());
    }
    // a segment running from in front of the camera to behind it still draws
    let through = [0.0, -0.2, 0.0, 0.0, -0.2, 10.0];
    let px = r.render_to_rgba(&mut scene_with(vec![lines(&through, None, material(4.0, 1.0))]), &cam);
    assert!(px.chunks(4).any(|p| p[0] > 127), "the part in front of the camera is drawn");
}


#[test]
fn every_target_draws_them() {
    // multisampled, an sRGB target (blended lines go through the encoded
    // blend pass there), half-float (the linear pipelines), the default
    // format: the same width each time
    let Some(mut plain) = renderer() else { return };
    let build = |f: wgpu::TextureFormat| HeadlessRenderer::builder().size(W, H).color_format(f).build().unwrap();
    let mut srgb = build(wgpu::TextureFormat::Rgba8UnormSrgb);
    let mut half = build(wgpu::TextureFormat::Rgba16Float);
    let mut msaa = build(wgpu::TextureFormat::Rgba8Unorm);
    msaa.set_msaa(4);
    for opacity in [1.0, 0.6] {
        for (name, r) in [("plain", &mut plain), ("srgb", &mut srgb), ("f16", &mut half), ("msaa", &mut msaa)] {
            let px = r.render_to_rgba(&mut scene_with(vec![lines(&ACROSS, None, material(6.0, opacity))]), &camera());
            let lit = (0..H).filter(|&v| at(&px, W / 2, v) > 40).count();
            assert!((6..=8).contains(&lit), "{name} at opacity {opacity}: {lit} px");
        }
    }
    // and the opaque line was drawn multisampled, not on a fallback path
    let _ = msaa.render_to_rgba(&mut scene_with(vec![lines(&ACROSS, None, material(6.0, 1.0))]), &camera());
    assert!(msaa.renderer().last_frame_multisampled());
}
