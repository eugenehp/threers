//! The quad view on the Metal backend, both ways it can be drawn:
//!
//! * `QuadView::metal_views` — four viewports in one `render_views` call, each
//!   pane drawn, the earlier panes surviving the later ones (with MSAA too),
//!   and nothing spilling into the gaps;
//! * `QuadView::render` through the Metal renderer's three.js viewport /
//!   scissor / auto-clear state — the same checks `tests/quad_view.rs` makes
//!   on wgpu, on real pixels, plus a pane-for-pane comparison against wgpu.
//!
//! Every probe is a value no other code path writes: each pane its own
//! background, the gap magenta, the markers unlit primaries — the fixture
//! `tests/quad_view.rs` uses, from `tests/quad_view_fixture`.
//!
//! Skipped without the `metal` feature or a Metal device.

#![cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]

use threers::cameras::PerspectiveCamera;
use threers::core::{Mesh, Object3D};
use threers::geometries::BoxGeometry;
use threers::helpers::{Pane, QuadView};
use threers::materials::{BasicMaterial, Material};
use threers::math::{Box3, Color, Vector3};
use threers::metal::{MetalError, MetalHeadlessRenderer};
use threers::scene::Scene;

#[path = "quad_view_fixture/mod.rs"]
mod quad_view_fixture;
use quad_view_fixture::{
    marker_scene, offset_of, persp, px, quad, rgb, shadow_scene, tilted_box_scene, tl, BLUE,
    GREEN, RED,
};

const W: u32 = 160;
const H: u32 = 120;

/// A `w`×`h` Metal renderer, or `None` (the test skips) without a device.
fn build(w: u32, h: u32, msaa: u32, ss: u32) -> Option<MetalHeadlessRenderer> {
    match MetalHeadlessRenderer::builder()
        .size(w, h)
        .msaa(msaa)
        .supersample(ss)
        .build()
    {
        Ok(r) => Some(r),
        Err(MetalError::NoDevice) => {
            eprintln!("skipping: no Metal device on this machine");
            None
        }
        Err(e) => panic!("Metal renderer could not be built: {e}"),
    }
}

fn renderer(msaa: u32) -> Option<MetalHeadlessRenderer> {
    build(W, H, msaa, 1)
}

fn scene() -> Scene {
    let mut scene = Scene::new();
    scene.background = Color::new(0.1, 0.2, 0.3);
    let mut add = |size: f32, at: Vector3, c: Color| {
        let mut o = Object3D::mesh(Mesh::new(
            BoxGeometry::new(size, size, size),
            Material::Basic(BasicMaterial::new(c)),
        ));
        o.position = at;
        scene.add(o);
    };
    add(1.0, Vector3::ZERO, Color::new(1.0, 1.0, 1.0));
    add(0.4, Vector3::new(1.5, 0.0, 0.0), Color::new(1.0, 0.0, 0.0));
    scene
}

#[test]
fn quad_view_renders_four_panes_on_metal() {
    for msaa in [1, 4] {
        let Some(mut hr) = renderer(msaa) else { return };
        assert_eq!(hr.render_size(), (W, H), "test assumes no supersample");
        let mut scene = scene();
        let mut q = QuadView::new();
        q.gap = 8.0;
        q.frame_box(&Box3::new(
            Vector3::new(-2.0, -2.0, -2.0),
            Vector3::new(2.0, 2.0, 2.0),
        ));
        let mut persp = PerspectiveCamera::new(45.0, 1.0, 0.1, 100.0);
        persp.position = Vector3::new(4.0, 3.0, 5.0);
        persp.look_at(Vector3::ZERO);
        let views = q.metal_views(&persp, W as f32, H as f32, 1.0);
        assert_eq!(views.len(), 4);
        let mut img = Vec::new();
        hr.render_views_into(&mut scene, &views, &mut img).unwrap();

        // The gap is the scene background (Metal clears once).
        let bg = px(&img, W, W / 2, H / 2);
        for (pane, r) in q.visible_panes(W as f32, H as f32) {
            let (x0, x1) = (r.x as u32, (r.x + r.width) as u32);
            let (y0, y1) = (H - (r.y + r.height) as u32, H - r.y as u32);
            // Every pane drew its cube at its centre...
            let c = px(&img, W, (x0 + x1) / 2, (y0 + y1) / 2);
            assert_ne!(c, bg, "msaa {msaa}: {pane:?} centre is background");
            assert!(c[0] > 200, "msaa {msaa}: {pane:?} centre {c:?}");
            // ...and its corner is still background, not garbage.
            assert_eq!(px(&img, W, x0 + 1, y0 + 1), bg, "msaa {msaa}: {pane:?} corner");
            // The red marker: right of centre in top and front, at the centre
            // (towards the camera) in the side view.
            let red = (y0..y1)
                .flat_map(|y| (x0..x1).map(move |x| (x, y)))
                .filter(|&(x, y)| {
                    let p = px(&img, W, x, y);
                    p[0] > 200 && p[1] < 60 && p[2] < 60
                })
                .map(|(x, _)| x as f32 - (x0 + x1) as f32 * 0.5)
                .collect::<Vec<_>>();
            let mean = red.iter().sum::<f32>() / red.len().max(1) as f32;
            match pane {
                Pane::Top | Pane::Front => assert!(mean > 5.0, "{pane:?} red at {mean}"),
                Pane::Side => assert!(mean.abs() < 2.0, "{pane:?} red at {mean}"),
                Pane::Perspective => assert!(!red.is_empty()),
            }
        }
        // A whole gap column: background top to bottom.
        let left = q.pane_rect(Pane::Front, W as f32, H as f32).unwrap();
        let gx = (left.x + left.width) as u32 + 3;
        for y in 0..H {
            assert_eq!(px(&img, W, gx, y), bg, "msaa {msaa}: gap column at y {y}");
        }
    }
}

// ------------------------------------------------- QuadView::render on Metal

const QW: u32 = 320;
const QH: u32 = 240;

fn metal(msaa: u32, ss: u32) -> Option<MetalHeadlessRenderer> {
    build(QW, QH, msaa, ss)
}

fn full_persp() -> PerspectiveCamera {
    let mut c = persp();
    c.aspect = QW as f32 / QH as f32;
    c
}

fn read(hr: &MetalHeadlessRenderer) -> Vec<u8> {
    hr.read_rgba().expect("readback")
}

/// Per-pane backgrounds and a magenta gap: every gap pixel magenta, every
/// pane corner its own background, something drawn in every pane — with and
/// without MSAA.
#[test]
fn render_keeps_gap_colour_and_pane_backgrounds() {
    for msaa in [1, 4] {
        let Some(mut hr) = metal(msaa, 1) else { return };
        let mut scene = marker_scene();
        let mut q = quad();
        q.frame_scene(&mut scene).unwrap();
        q.render(&mut hr, &mut scene, &persp());
        let img = read(&hr);
        let gap = rgb(q.gap_color.unwrap());

        let mut pane_pixels = 0usize;
        let rects: Vec<_> = q
            .visible_panes(QW as f32, QH as f32)
            .into_iter()
            .map(|(p, r)| (p, tl(r, QH, 1)))
            .collect();
        assert_eq!(rects.len(), 4);
        for &(pane, b) in &rects {
            let bg = rgb(q.backgrounds[pane.index()].unwrap());
            for (x, y) in [
                (b.0, b.1),
                (b.2 - 1, b.1),
                (b.0, b.3 - 1),
                (b.2 - 1, b.3 - 1),
            ] {
                assert_eq!(px(&img, QW, x, y), bg, "msaa {msaa}: {pane:?} corner ({x},{y})");
            }
            let drawn = (b.1..b.3)
                .flat_map(|y| (b.0..b.2).map(move |x| (x, y)))
                .filter(|&(x, y)| px(&img, QW, x, y) != bg)
                .count();
            assert!(drawn > 200, "msaa {msaa}: {pane:?}: only {drawn} pixels drawn");
            pane_pixels += ((b.2 - b.0) * (b.3 - b.1)) as usize;
        }
        let mut gap_pixels = 0usize;
        for y in 0..QH {
            for x in 0..QW {
                let inside = rects
                    .iter()
                    .any(|(_, b)| x >= b.0 && x < b.2 && y >= b.1 && y < b.3);
                if !inside {
                    assert_eq!(px(&img, QW, x, y), gap, "msaa {msaa}: gap pixel ({x},{y})");
                    gap_pixels += 1;
                }
            }
        }
        assert_eq!(pane_pixels + gap_pixels, (QW * QH) as usize);
        assert!(gap_pixels > 1000);
        // The renderer's state is handed back as it was.
        let r = hr.renderer();
        assert_eq!(r.viewport(), None);
        assert_eq!(r.scissor(), None);
        assert!(!r.scissor_test());
        assert!(r.auto_clear());
    }
}

#[test]
fn orthographic_panes_face_the_documented_way_on_metal() {
    let Some(mut hr) = metal(1, 1) else { return };
    let mut scene = marker_scene();
    let mut q = quad();
    q.frame_box(&Box3::new(
        Vector3::new(-2.0, -2.0, -2.0),
        Vector3::new(2.0, 2.0, 2.0),
    ));
    q.render(&mut hr, &mut scene, &persp());
    let img = read(&hr);
    let b = |p: Pane| tl(q.pane_rect(p, QW as f32, QH as f32).unwrap(), QH, 1);
    let off = |p: Pane, c: [u8; 3]| {
        offset_of(&img, QW, b(p), c).unwrap_or_else(|| panic!("{p:?}: no {c:?} pixels"))
    };
    let right = |o: (f32, f32)| o.0 > 10.0 && o.1.abs() < 3.0;
    let up = |o: (f32, f32)| o.1 > 10.0 && o.0.abs() < 3.0;
    let centre = |o: (f32, f32)| o.0.abs() < 3.0 && o.1.abs() < 3.0;
    assert!(right(off(Pane::Top, RED)), "{:?}", off(Pane::Top, RED));
    assert!(up(off(Pane::Top, GREEN)), "{:?}", off(Pane::Top, GREEN));
    assert!(centre(off(Pane::Top, BLUE)), "{:?}", off(Pane::Top, BLUE));
    assert!(right(off(Pane::Front, RED)), "{:?}", off(Pane::Front, RED));
    assert!(up(off(Pane::Front, BLUE)), "{:?}", off(Pane::Front, BLUE));
    assert!(offset_of(&img, QW, b(Pane::Front), GREEN).is_none());
    assert!(right(off(Pane::Side, GREEN)), "{:?}", off(Pane::Side, GREEN));
    assert!(up(off(Pane::Side, BLUE)), "{:?}", off(Pane::Side, BLUE));
    assert!(centre(off(Pane::Side, RED)), "{:?}", off(Pane::Side, RED));
    let (t, f) = (off(Pane::Top, RED).0, off(Pane::Front, RED).0);
    assert!((t - f).abs() < 1.0, "top {t} vs front {f}");
}

#[test]
fn maximised_pane_fills_the_target_on_metal() {
    let Some(mut hr) = metal(1, 1) else { return };
    let mut scene = marker_scene();
    let mut q = quad();
    q.frame_scene(&mut scene).unwrap();
    q.toggle_maximize(Pane::Front);
    q.render(&mut hr, &mut scene, &persp());
    let img = read(&hr);
    let bg = rgb(q.backgrounds[Pane::Front.index()].unwrap());
    assert_eq!(px(&img, QW, 4, 4), bg);
    assert_eq!(px(&img, QW, QW - 5, QH - 5), bg);
    assert_eq!(px(&img, QW, 0, 0), rgb(q.gap_color.unwrap()), "border");
    for p in [Pane::Top, Pane::Side, Pane::Perspective] {
        let other = rgb(q.backgrounds[p.index()].unwrap());
        assert!(img.chunks(4).all(|c| c[..3] != other), "{p:?} leaked");
    }
}

/// Supersampled: rectangles stay in output pixels, the target is twice the
/// size, and the gap scales with it.
#[test]
fn pixel_ratio_follows_the_supersample_on_metal() {
    let Some(mut hr) = metal(1, 2) else { return };
    assert_eq!(hr.renderer().pixel_ratio(), 2.0);
    assert_eq!(hr.renderer().size(), (QW as f32, QH as f32));
    let mut scene = marker_scene();
    let mut q = quad();
    q.frame_scene(&mut scene).unwrap();
    q.render(&mut hr, &mut scene, &persp());
    let img = read(&hr);
    let (w2, h2) = hr.render_size();
    assert_eq!((w2, h2), (2 * QW, 2 * QH));
    for (pane, rect) in q.visible_panes(QW as f32, QH as f32) {
        let b = tl(rect, h2, 2);
        let bg = rgb(q.backgrounds[pane.index()].unwrap());
        assert_eq!(px(&img, w2, b.0, b.1), bg, "{pane:?}");
        assert_eq!(px(&img, w2, b.2 - 1, b.3 - 1), bg, "{pane:?}");
        assert_eq!(px(&img, w2, b.2, b.3 - 1), rgb(q.gap_color.unwrap()), "{pane:?}");
    }
}

/// A scissored render must not touch a single pixel outside the scissor —
/// with MSAA too, whose full-target resolve would. Two of them, so the second
/// has a first to erase: a multisampled texture that was resolved and
/// discarded holds nothing of the first, and loading it would wipe it.
#[test]
fn scissored_render_leaves_the_rest_alone_on_metal() {
    for msaa in [1, 4] {
        let Some(mut hr) = metal(msaa, 1) else { return };
        let outside = Color::from_hex(0x13579b);
        hr.clear(outside, 1.0).unwrap();
        let mut scene = marker_scene();
        scene.background = Color::from_hex(0x2468ac);
        let r = hr.renderer();
        r.set_scissor_test(true);
        // Bottom-left origin: the lower-right part of the image.
        r.set_scissor(200.0, 10.0, 100.0, 80.0);
        hr.render(&mut scene, &full_persp()).unwrap();
        // The second: upper-left, its own background.
        let second_bg = Color::from_hex(0x8899aa);
        let mut second = marker_scene();
        second.background = second_bg;
        hr.renderer().set_scissor(20.0, 150.0, 80.0, 60.0);
        hr.render(&mut second, &full_persp()).unwrap();
        let img = read(&hr);
        let r = hr.renderer();
        r.set_scissor_test(false);
        r.reset_scissor();
        let (x0, x1, y0, y1) = (200, 300, QH - 90, QH - 10);
        let (sx0, sx1, sy0, sy1) = (20, 100, QH - 210, QH - 150);
        let (mut inside_bg, mut inside_drawn, mut second_px) = (0, 0, 0);
        for y in 0..QH {
            for x in 0..QW {
                let p = px(&img, QW, x, y);
                if x >= sx0 && x < sx1 && y >= sy0 && y < sy1 {
                    second_px += usize::from(p == rgb(second_bg));
                } else if x >= x0 && x < x1 && y >= y0 && y < y1 {
                    assert_ne!(p, rgb(outside), "msaa {msaa}: ({x},{y}) not cleared");
                    inside_bg += usize::from(p == rgb(scene.background));
                    inside_drawn += usize::from(p != rgb(scene.background));
                } else {
                    assert_eq!(p, rgb(outside), "msaa {msaa}: ({x},{y}) touched");
                }
            }
        }
        assert!(inside_bg > 0, "msaa {msaa}: scissor never cleared to the background");
        assert!(inside_drawn > 0, "msaa {msaa}: nothing drawn inside the scissor");
        assert!(second_px > 0, "msaa {msaa}: second scissor never cleared");
    }
}

/// A scissored `clear` fills the rectangle and nothing else, MSAA or not.
#[test]
fn scissored_clear_fills_only_the_rectangle() {
    for msaa in [1, 4] {
        let Some(mut hr) = metal(msaa, 1) else { return };
        let (base, rect) = (Color::from_hex(0x204060), Color::from_hex(0xc08040));
        hr.clear(base, 1.0).unwrap();
        hr.renderer().set_scissor_test(true);
        hr.renderer().set_scissor(10.0, 20.0, 30.0, 40.0);
        hr.clear(rect, 1.0).unwrap();
        hr.renderer().set_scissor_test(false);
        let img = read(&hr);
        for y in 0..QH {
            for x in 0..QW {
                let inside = (10..40).contains(&x) && (QH - 60..QH - 20).contains(&y);
                let want = if inside { rgb(rect) } else { rgb(base) };
                assert_eq!(px(&img, QW, x, y), want, "msaa {msaa}: ({x},{y})");
            }
        }
    }
}

/// `auto_clear` off: the render draws over what was there.
#[test]
fn auto_clear_off_keeps_the_previous_frame_on_metal() {
    for msaa in [1, 4] {
        let Some(mut hr) = metal(msaa, 1) else { return };
        let keep = Color::from_hex(0x7f3f1f);
        hr.clear(keep, 1.0).unwrap();
        hr.renderer().set_auto_clear(false);
        let mut scene = marker_scene();
        hr.render(&mut scene, &full_persp()).unwrap();
        hr.renderer().set_auto_clear(true);
        let img = read(&hr);
        assert_eq!(px(&img, QW, 0, 0), rgb(keep), "msaa {msaa}");
        assert_ne!(px(&img, QW, QW / 2, QH / 2), rgb(keep), "msaa {msaa}: the cube still draws");
    }
}

/// The default state takes the old path; so does a viewport set explicitly
/// to the whole target, byte for byte. And the staged path agrees with it:
/// a full-target render with auto-clear off over a target cleared to the
/// background is the same picture.
#[test]
fn default_and_full_viewport_are_byte_identical_on_metal() {
    for msaa in [1, 4] {
        let Some(mut hr) = metal(msaa, 1) else { return };
        let mut scene = marker_scene();
        let cam = full_persp();
        hr.render(&mut scene, &cam).unwrap();
        let plain = read(&hr);
        hr.renderer().set_viewport(0.0, 0.0, QW as f32, QH as f32);
        hr.render(&mut scene, &cam).unwrap();
        let explicit = read(&hr);
        hr.renderer().reset_viewport();
        assert!(plain == explicit, "msaa {msaa}");
        if msaa == 1 {
            hr.clear(scene.background, 1.0).unwrap();
            hr.renderer().set_auto_clear(false);
            hr.render(&mut scene, &cam).unwrap();
            hr.renderer().set_auto_clear(true);
            let staged = read(&hr);
            assert!(plain == staged, "staged path differs from the legacy one");
        }
    }
}

/// A viewport hanging off the left edge shows the same picture, shifted.
#[test]
fn a_viewport_off_the_edge_is_shifted_not_squashed_on_metal() {
    let Some(mut hr) = metal(1, 1) else { return };
    let mut scene = marker_scene();
    let cam = full_persp();
    hr.render(&mut scene, &cam).unwrap();
    let full = read(&hr);
    let shift = 80u32;
    hr.renderer()
        .set_viewport(-(shift as f32), 0.0, QW as f32, QH as f32);
    hr.render(&mut scene, &cam).unwrap();
    let shifted = read(&hr);
    hr.renderer().reset_viewport();
    let (mut differ, mut total) = (0, 0);
    for y in 0..QH {
        for x in 0..QW - shift {
            total += 1;
            if px(&shifted, QW, x, y) != px(&full, QW, x + shift, y) {
                differ += 1;
            }
        }
    }
    assert!(differ * 200 < total, "{differ} of {total} pixels differ");
}

/// wgpu and Metal draw the same quad view: every gap pixel identical, and
/// every pane within a small tolerance — rasterisation rules differ at edges.
#[test]
fn wgpu_and_metal_quad_views_agree_pane_for_pane() {
    let Some(mut mr) = metal(1, 1) else { return };
    let Ok(mut wr) = threers::HeadlessRenderer::builder()
        .size(QW, QH)
        .color_format(wgpu::TextureFormat::Rgba8UnormSrgb)
        .build()
    else {
        eprintln!("skipping: no wgpu adapter");
        return;
    };
    let mut q = quad();
    let mut scene = marker_scene();
    q.frame_scene(&mut scene).unwrap();
    q.render(&mut wr, &mut scene, &persp());
    let w = wr.read_rgba();
    q.render(&mut mr, &mut scene, &persp());
    let m = read(&mr);
    assert_eq!(w.len(), m.len());

    let rects: Vec<_> = q
        .visible_panes(QW as f32, QH as f32)
        .into_iter()
        .map(|(p, r)| (p, tl(r, QH, 1)))
        .collect();
    let mut gap_differ = 0;
    for y in 0..QH {
        for x in 0..QW {
            if !rects
                .iter()
                .any(|(_, b)| x >= b.0 && x < b.2 && y >= b.1 && y < b.3)
                && px(&w, QW, x, y) != px(&m, QW, x, y)
            {
                gap_differ += 1;
            }
        }
    }
    assert_eq!(gap_differ, 0, "gap pixels differ between backends");
    for (pane, b) in rects {
        let (mut sum, mut max, mut differ, mut n) = (0u64, 0u8, 0usize, 0usize);
        for y in b.1..b.3 {
            for x in b.0..b.2 {
                let (a, c) = (px(&w, QW, x, y), px(&m, QW, x, y));
                let d = (0..3).map(|i| a[i].abs_diff(c[i])).max().unwrap();
                sum += d as u64;
                max = max.max(d);
                differ += usize::from(d > 0);
                n += 1;
            }
        }
        let mean = sum as f64 / n as f64;
        let frac = differ as f64 / n as f64;
        eprintln!(
            "{pane:?}: mean |diff| {mean:.4}/255, max {max}, {differ} of {n} pixels differ ({:.2}%)",
            frac * 100.0
        );
        assert!(mean < 1.0, "{pane:?}: mean difference {mean}");
        assert!(frac < 0.02, "{pane:?}: {differ} of {n} pixels differ");
        // Same background in the corner, byte for byte.
        assert_eq!(px(&w, QW, b.0, b.1), px(&m, QW, b.0, b.1), "{pane:?} corner");
    }
}

/// MSAA in the panes on Metal: each pane drawn into a pane-sized
/// multisampled target and composited into its rectangle — intermediate
/// coverage on the box edges at 4 samples, none at 1, and the gap untouched.
#[test]
fn msaa_antialiases_every_pane_on_metal() {
    for msaa in [1, 4] {
        let Some(mut hr) = metal(msaa, 1) else { return };
        let mut scene = tilted_box_scene();
        let mut q = quad();
        q.frame_scene(&mut scene).unwrap();
        let mut cam = persp();
        cam.position = Vector3::new(2.0, 1.5, 2.5);
        cam.look_at(Vector3::ZERO);
        q.render(&mut hr, &mut scene, &cam);
        let img = read(&hr);
        let gap = rgb(q.gap_color.unwrap());
        let panes = q.visible_panes(QW as f32, QH as f32);
        for &(pane, rect) in &panes {
            let b = tl(rect, QH, 1);
            let bg = rgb(q.backgrounds[pane.index()].unwrap());
            let n = (b.1..b.3)
                .flat_map(|y| (b.0..b.2).map(move |x| (x, y)))
                .filter(|&(x, y)| {
                    let p = px(&img, QW, x, y);
                    p != bg && p != [255, 255, 255]
                })
                .count();
            if msaa == 1 {
                assert_eq!(n, 0, "{pane:?}: {n} intermediate pixels without MSAA");
            } else {
                assert!(n > 40, "{pane:?}: only {n} anti-aliased edge pixels");
            }
            assert_eq!(px(&img, QW, b.0, b.1), bg, "msaa {msaa} {pane:?}");
            assert_eq!(px(&img, QW, b.2 - 1, b.3 - 1), bg, "msaa {msaa} {pane:?}");
        }
        let rects: Vec<_> = panes.iter().map(|&(_, r)| tl(r, QH, 1)).collect();
        for y in 0..QH {
            for x in 0..QW {
                if !rects
                    .iter()
                    .any(|b| x >= b.0 && x < b.2 && y >= b.1 && y < b.3)
                {
                    assert_eq!(px(&img, QW, x, y), gap, "msaa {msaa}: gap ({x},{y})");
                }
            }
        }
    }
}

/// Shared shadow maps on Metal: identical panes, one update per quad view.
#[test]
fn shared_shadow_maps_draw_identical_panes_on_metal() {
    for msaa in [1, 4] {
        let Some(mut hr) = metal(msaa, 1) else { return };
        let mut scene = shadow_scene();
        let mut q = quad();
        q.frame_scene(&mut scene).unwrap();
        let mut cam = persp();
        cam.position = Vector3::new(5.0, 4.0, 6.0);
        cam.look_at(Vector3::ZERO);
        q.share_shadows = false;
        let n0 = hr.renderer().shadow_map_updates();
        q.render(&mut hr, &mut scene, &cam);
        let separate = read(&hr);
        let n1 = hr.renderer().shadow_map_updates();
        // The panes share one scene and its lights, so the maps are drawn for
        // the first and found unchanged for the other three.
        assert_eq!(n1 - n0, 1, "msaa {msaa}: drawn once, then reused");
        q.share_shadows = true;
        q.render(&mut hr, &mut scene, &cam);
        let shared = read(&hr);
        assert_eq!(hr.renderer().shadow_map_updates() - n1, 1, "msaa {msaa}");
        assert!(hr.renderer().shadow_map().auto_update);
        assert!(separate == shared, "msaa {msaa}: shared shadow maps changed a pane");
        for (_, obj) in scene.arena.nodes.iter_mut() {
            obj.cast_shadow = false;
        }
        q.render(&mut hr, &mut scene, &cam);
        assert!(read(&hr) != shared, "msaa {msaa}: no visible shadow");
    }
}
