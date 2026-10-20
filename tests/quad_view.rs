//! Viewports, scissors and the quad view, checked on real pixels.
//!
//! Every probe is a value no other code path writes: each pane gets its own
//! background, the gap its own colour, and the markers are unlit primaries, so
//! a pane that was cleared by the wrong render, a gap that a pane spilled
//! into, or a view facing the wrong way shows up as a wrong byte rather than
//! as "something was drawn".
//!
//! Skipped when no GPU adapter is available.

use threers::helpers::Pane;
use threers::{Color, HeadlessRenderer, Vector3};

#[path = "quad_view_fixture/mod.rs"]
mod quad_view_fixture;
use quad_view_fixture::{
    marker_scene as scene, offset_of, persp, px, quad, rgb, shadow_scene, tilted_box_scene, tl,
    BLUE, GREEN, RED,
};

const W: u32 = 320;
const H: u32 = 240;

fn renderer(ss: u32) -> Option<HeadlessRenderer> {
    HeadlessRenderer::builder()
        .size(W, H)
        .supersample(ss)
        .color_format(wgpu::TextureFormat::Rgba8Unorm)
        .build()
        .ok()
}

#[test]
fn every_pane_is_drawn_and_the_gaps_keep_their_colour() {
    let Some(mut hr) = renderer(1) else { return };
    let mut scene = scene();
    let mut q = quad();
    q.frame_scene(&mut scene).unwrap();
    q.render(&mut hr, &mut scene, &persp());
    let img = hr.read_rgba();
    let gap = rgb(q.gap_color.unwrap());

    let mut pane_pixels = 0usize;
    for (pane, rect) in q.visible_panes(W as f32, H as f32) {
        let b = tl(rect, H, 1);
        let bg = rgb(q.backgrounds[pane.index()].unwrap());
        // Its own background in the corners — not the next pane's, not the
        // scene's, not the gap colour.
        for (x, y) in [
            (b.0, b.1),
            (b.2 - 1, b.1),
            (b.0, b.3 - 1),
            (b.2 - 1, b.3 - 1),
        ] {
            assert_eq!(px(&img, W, x, y), bg, "{pane:?} corner ({x},{y})");
        }
        // And something drawn over it.
        let drawn = (b.1..b.3)
            .flat_map(|y| (b.0..b.2).map(move |x| (x, y)))
            .filter(|&(x, y)| px(&img, W, x, y) != bg)
            .count();
        assert!(drawn > 200, "{pane:?}: only {drawn} pixels drawn");
        pane_pixels += ((b.2 - b.0) * (b.3 - b.1)) as usize;
    }
    // Every pixel outside the panes is the gap colour.
    let rects: Vec<_> = q
        .visible_panes(W as f32, H as f32)
        .into_iter()
        .map(|(_, r)| tl(r, H, 1))
        .collect();
    let mut gap_pixels = 0usize;
    for y in 0..H {
        for x in 0..W {
            let inside = rects
                .iter()
                .any(|b| x >= b.0 && x < b.2 && y >= b.1 && y < b.3);
            if !inside {
                assert_eq!(px(&img, W, x, y), gap, "gap pixel ({x},{y})");
                gap_pixels += 1;
            }
        }
    }
    assert_eq!(pane_pixels + gap_pixels, (W * H) as usize);
    assert!(gap_pixels > 1000);
}

#[test]
fn orthographic_panes_face_the_documented_way() {
    let Some(mut hr) = renderer(1) else { return };
    let mut scene = scene();
    let mut q = quad();
    // Framed on a box centred on the origin, so the origin is each pane's
    // centre (the scene's own bounds are lopsided towards the markers).
    q.frame_box(&threers::Box3::new(
        Vector3::new(-2.0, -2.0, -2.0),
        Vector3::new(2.0, 2.0, 2.0),
    ));
    q.render(&mut hr, &mut scene, &persp());
    let img = hr.read_rgba();
    let b = |p: Pane| tl(q.pane_rect(p, W as f32, H as f32).unwrap(), H, 1);
    let at = |p: Pane, c: [u8; 3]| {
        offset_of(&img, W, b(p), c).unwrap_or_else(|| panic!("{p:?}: no {c:?} pixels"))
    };
    let right = |o: (f32, f32)| o.0 > 10.0 && o.1.abs() < 3.0;
    let up = |o: (f32, f32)| o.1 > 10.0 && o.0.abs() < 3.0;
    let centre = |o: (f32, f32)| o.0.abs() < 3.0 && o.1.abs() < 3.0;
    // Top: right is +X, up is −Z, and +Y is towards the camera.
    assert!(right(at(Pane::Top, RED)), "{:?}", at(Pane::Top, RED));
    assert!(up(at(Pane::Top, GREEN)), "{:?}", at(Pane::Top, GREEN));
    assert!(centre(at(Pane::Top, BLUE)), "{:?}", at(Pane::Top, BLUE));
    // Front: right is +X, up is +Y; −Z is hidden behind the cube.
    assert!(right(at(Pane::Front, RED)), "{:?}", at(Pane::Front, RED));
    assert!(up(at(Pane::Front, BLUE)), "{:?}", at(Pane::Front, BLUE));
    assert!(offset_of(&img, W, b(Pane::Front), GREEN).is_none());
    // Side (from +X): right is −Z, up is +Y, +X towards the camera.
    assert!(right(at(Pane::Side, GREEN)), "{:?}", at(Pane::Side, GREEN));
    assert!(up(at(Pane::Side, BLUE)), "{:?}", at(Pane::Side, BLUE));
    assert!(centre(at(Pane::Side, RED)), "{:?}", at(Pane::Side, RED));
    // One scale across the ortho panes: the red marker's distance from the
    // centre is the same in top and front (both show X across).
    let (t, f) = (at(Pane::Top, RED).0, at(Pane::Front, RED).0);
    assert!((t - f).abs() < 1.0, "top {t} vs front {f}");
}

#[test]
fn maximised_pane_fills_the_target() {
    let Some(mut hr) = renderer(1) else { return };
    let mut scene = scene();
    let mut q = quad();
    q.frame_scene(&mut scene).unwrap();
    q.toggle_maximize(Pane::Front);
    q.render(&mut hr, &mut scene, &persp());
    let img = hr.read_rgba();
    let bg = rgb(q.backgrounds[Pane::Front.index()].unwrap());
    assert_eq!(px(&img, W, 4, 4), bg);
    assert_eq!(px(&img, W, W - 5, H - 5), bg);
    assert_eq!(px(&img, W, 0, 0), rgb(q.gap_color.unwrap()), "border");
    // No other pane's background anywhere.
    for p in [Pane::Top, Pane::Side, Pane::Perspective] {
        let other = rgb(q.backgrounds[p.index()].unwrap());
        assert!(img.chunks(4).all(|c| c[..3] != other), "{p:?} leaked");
    }
}

/// Supersampled: rectangles stay in output pixels, the target is twice the
/// size, and the gap scales with it.
#[test]
fn pixel_ratio_follows_the_supersample() {
    let Some(mut hr) = renderer(2) else { return };
    let mut scene = scene();
    let mut q = quad();
    q.frame_scene(&mut scene).unwrap();
    q.render(&mut hr, &mut scene, &persp());
    let img = hr.read_rgba();
    let (w2, h2) = hr.render_size();
    assert_eq!((w2, h2), (2 * W, 2 * H));
    for (pane, rect) in q.visible_panes(W as f32, H as f32) {
        let b = tl(rect, h2, 2);
        let bg = rgb(q.backgrounds[pane.index()].unwrap());
        assert_eq!(px(&img, w2, b.0, b.1), bg, "{pane:?}");
        assert_eq!(px(&img, w2, b.2 - 1, b.3 - 1), bg, "{pane:?}");
        // One device pixel outside is the gap.
        assert_eq!(
            px(&img, w2, b.2, b.3 - 1),
            rgb(q.gap_color.unwrap()),
            "{pane:?}"
        );
    }
}

/// A scissored render must not touch a single pixel outside the scissor —
/// with MSAA asked for too, whose full-target resolve would.
#[test]
fn scissored_render_leaves_the_rest_alone() {
    for msaa in [1, 4] {
        let Some(mut hr) = renderer(1) else { return };
        hr.set_msaa(msaa);
        let outside = Color::from_hex(0x13579b);
        hr.clear(outside, 1.0);
        let mut scene = scene();
        scene.background = Color::from_hex(0x2468ac);
        let cam = {
            let mut c = persp();
            c.aspect = W as f32 / H as f32;
            c
        };
        let r = hr.renderer();
        r.set_scissor_test(true);
        // Bottom-left origin: this is the lower-right part of the image.
        r.set_scissor(200.0, 10.0, 100.0, 80.0);
        let img = hr.render_to_rgba(&mut scene, &cam);
        let r = hr.renderer();
        r.set_scissor_test(false);
        r.reset_scissor();
        let (x0, x1, y0, y1) = (200, 300, H - 90, H - 10);
        let mut inside_bg = 0;
        for y in 0..H {
            for x in 0..W {
                let p = px(&img, W, x, y);
                if x >= x0 && x < x1 && y >= y0 && y < y1 {
                    assert_ne!(p, rgb(outside), "msaa {msaa}: ({x},{y}) not cleared");
                    inside_bg += usize::from(p == rgb(scene.background));
                } else {
                    assert_eq!(p, rgb(outside), "msaa {msaa}: ({x},{y}) touched");
                }
            }
        }
        assert!(
            inside_bg > 0,
            "msaa {msaa}: scissor never cleared to the background"
        );
    }
}

/// `auto_clear` off: the second render draws over the first rather than
/// clearing it.
#[test]
fn auto_clear_off_keeps_the_previous_frame() {
    let Some(mut hr) = renderer(1) else { return };
    let keep = Color::from_hex(0x7f3f1f);
    hr.clear(keep, 1.0);
    hr.renderer().set_auto_clear(false);
    let mut scene = scene();
    let mut cam = persp();
    cam.aspect = W as f32 / H as f32;
    let img = hr.render_to_rgba(&mut scene, &cam);
    hr.renderer().set_auto_clear(true);
    assert_eq!(px(&img, W, 0, 0), rgb(keep));
    assert_ne!(px(&img, W, W / 2, H / 2), rgb(keep), "the cube still draws");
}

/// The default state is the old renderer, and so is a viewport explicitly set
/// to the whole target.
#[test]
fn full_viewport_is_byte_identical_to_none() {
    let Some(mut hr) = renderer(1) else { return };
    let mut scene = scene();
    let mut cam = persp();
    cam.aspect = W as f32 / H as f32;
    let plain = hr.render_to_rgba(&mut scene, &cam);
    hr.renderer().set_viewport(0.0, 0.0, W as f32, H as f32);
    let explicit = hr.render_to_rgba(&mut scene, &cam);
    hr.renderer().reset_viewport();
    assert!(plain == explicit);
}

/// A viewport hanging off the left edge shows the same picture, shifted, as
/// an unclipped one would — the projection is rescaled, not squashed.
#[test]
fn a_viewport_off_the_edge_is_shifted_not_squashed() {
    let Some(mut hr) = renderer(1) else { return };
    let mut scene = scene();
    let mut cam = persp();
    cam.aspect = W as f32 / H as f32;
    let full = hr.render_to_rgba(&mut scene, &cam);
    let shift = 80u32;
    hr.renderer()
        .set_viewport(-(shift as f32), 0.0, W as f32, H as f32);
    let shifted = hr.render_to_rgba(&mut scene, &cam);
    hr.renderer().reset_viewport();
    // shifted(x) == full(x + shift) for every visible column.
    let mut differ = 0;
    let mut total = 0;
    for y in 0..H {
        for x in 0..W - shift {
            total += 1;
            if px(&shifted, W, x, y) != px(&full, W, x + shift, y) {
                differ += 1;
            }
        }
    }
    assert!(differ * 200 < total, "{differ} of {total} pixels differ");
}

/// Pixels of a pane that are neither its background nor the box's white —
/// partial coverage, which only anti-aliasing produces on an unlit box.
fn intermediate(img: &[u8], w: u32, b: (u32, u32, u32, u32), bg: [u8; 3]) -> usize {
    (b.1..b.3)
        .flat_map(|y| (b.0..b.2).map(move |x| (x, y)))
        .filter(|&(x, y)| {
            let p = px(img, w, x, y);
            p != bg && p != [255, 255, 255]
        })
        .count()
}

/// MSAA in the panes: at 4 samples the box edges in every pane carry
/// intermediate coverage values, at 1 there are none; and the gaps keep
/// their colour either way — the pane's resolve lands in its rectangle only.
#[test]
fn msaa_antialiases_every_pane_and_keeps_the_gaps() {
    for msaa in [1, 4] {
        let Some(mut hr) = renderer(1) else { return };
        hr.set_msaa(msaa);
        let mut scene = tilted_box_scene();
        let mut q = quad();
        q.frame_scene(&mut scene).unwrap();
        let mut cam = persp();
        cam.position = Vector3::new(2.0, 1.5, 2.5);
        cam.look_at(Vector3::ZERO);
        q.render(&mut hr, &mut scene, &cam);
        let img = hr.read_rgba();
        let gap = rgb(q.gap_color.unwrap());
        let panes = q.visible_panes(W as f32, H as f32);
        for &(pane, rect) in &panes {
            let b = tl(rect, H, 1);
            let bg = rgb(q.backgrounds[pane.index()].unwrap());
            let n = intermediate(&img, W, b, bg);
            if msaa == 1 {
                assert_eq!(n, 0, "{pane:?}: {n} intermediate pixels without MSAA");
            } else {
                assert!(n > 40, "{pane:?}: only {n} anti-aliased edge pixels");
            }
            // Corners are the pane's own background.
            assert_eq!(px(&img, W, b.0, b.1), bg, "msaa {msaa} {pane:?}");
            assert_eq!(px(&img, W, b.2 - 1, b.3 - 1), bg, "msaa {msaa} {pane:?}");
        }
        let rects: Vec<_> = panes.iter().map(|&(_, r)| tl(r, H, 1)).collect();
        for y in 0..H {
            for x in 0..W {
                if !rects
                    .iter()
                    .any(|b| x >= b.0 && x < b.2 && y >= b.1 && y < b.3)
                {
                    assert_eq!(px(&img, W, x, y), gap, "msaa {msaa}: gap ({x},{y})");
                }
            }
        }
    }
}

/// A pane drawn through its own multisampled target is the same picture as
/// the same view drawn multisampled over a whole target of that size.
#[test]
fn msaa_pane_matches_a_full_target_render() {
    let (Some(mut full), Some(mut wide)) = (
        HeadlessRenderer::builder()
            .size(W, H)
            .color_format(wgpu::TextureFormat::Rgba8Unorm)
            .build()
            .ok(),
        HeadlessRenderer::builder()
            .size(2 * W, H)
            .color_format(wgpu::TextureFormat::Rgba8Unorm)
            .build()
            .ok(),
    ) else {
        return;
    };
    full.set_msaa(4);
    wide.set_msaa(4);
    let mut scene = tilted_box_scene();
    let mut cam = persp();
    cam.position = Vector3::new(2.0, 1.5, 2.5);
    cam.look_at(Vector3::ZERO);
    cam.aspect = W as f32 / H as f32;
    let reference = full.render_to_rgba(&mut scene, &cam);
    wide.clear(Color::from_hex(0x13579b), 1.0);
    let r = wide.renderer();
    r.set_scissor_test(true);
    r.set_viewport(W as f32, 0.0, W as f32, H as f32);
    r.set_scissor(W as f32, 0.0, W as f32, H as f32);
    let img = wide.render_to_rgba(&mut scene, &cam);
    let mut differ = 0;
    for y in 0..H {
        for x in 0..W {
            let (a, b) = (px(&reference, W, x, y), px(&img, 2 * W, W + x, y));
            if a.iter().zip(b).any(|(a, b)| a.abs_diff(b) > 1) {
                differ += 1;
            }
            assert_eq!(px(&img, 2 * W, x, y), rgb(Color::from_hex(0x13579b)));
        }
    }
    assert_eq!(
        differ, 0,
        "{differ} pixels differ from the full-target MSAA render"
    );
}

/// One set of shadow maps for the four panes: the pictures are identical to
/// drawing the maps per pane, the maps are drawn once rather than four times,
/// and the renderer's own setting is left as it was.
#[test]
fn shared_shadow_maps_draw_identical_panes() {
    for msaa in [1, 4] {
        let Some(mut hr) = renderer(1) else { return };
        hr.set_msaa(msaa);
        let mut scene = shadow_scene();
        let mut q = quad();
        q.frame_scene(&mut scene).unwrap();
        let mut cam = persp();
        cam.position = Vector3::new(5.0, 4.0, 6.0);
        cam.look_at(Vector3::ZERO);

        q.share_shadows = false;
        let n0 = hr.renderer().shadow_map_updates();
        q.render(&mut hr, &mut scene, &cam);
        let separate = hr.read_rgba();
        let n1 = hr.renderer().shadow_map_updates();
        // The panes share one scene and its lights, so the maps are drawn for
        // the first and found unchanged for the other three.
        assert_eq!(n1 - n0, 1, "msaa {msaa}: drawn once, then reused");

        q.share_shadows = true;
        q.render(&mut hr, &mut scene, &cam);
        let shared = hr.read_rgba();
        let n2 = hr.renderer().shadow_map_updates();
        assert_eq!(n2 - n1, 1, "msaa {msaa}: one update per quad view");
        assert!(hr.renderer().shadow_map().auto_update, "setting restored");
        assert!(
            separate == shared,
            "msaa {msaa}: shared shadow maps changed a pane"
        );

        // And there is a shadow to share: without casters the panes differ.
        for (_, obj) in scene.arena.nodes.iter_mut() {
            obj.cast_shadow = false;
        }
        q.render(&mut hr, &mut scene, &cam);
        assert!(hr.read_rgba() != shared, "msaa {msaa}: no visible shadow");
    }
}

/// three.js semantics on the renderer itself: with auto-update off the maps
/// are redrawn only when asked, and the request is consumed.
#[test]
fn shadow_map_auto_update_off_redraws_only_on_request() {
    let Some(mut hr) = renderer(1) else { return };
    let mut scene = shadow_scene();
    let mut cam = persp();
    cam.position = Vector3::new(5.0, 4.0, 6.0);
    cam.look_at(Vector3::ZERO);
    hr.renderer().shadow_map_mut().auto_update = false;
    let n = |hr: &mut HeadlessRenderer| hr.renderer().shadow_map_updates();
    // Nothing drawn yet: the first render has to draw them regardless.
    hr.render(&mut scene, &cam);
    assert_eq!(n(&mut hr), 1);
    let first = hr.read_rgba();
    hr.render(&mut scene, &cam);
    assert_eq!(n(&mut hr), 1, "reused");
    assert!(hr.read_rgba() == first);
    hr.renderer().shadow_map_mut().needs_update = true;
    hr.render(&mut scene, &cam);
    assert_eq!(n(&mut hr), 2);
    assert!(!hr.renderer().shadow_map().needs_update, "consumed");
    hr.render(&mut scene, &cam);
    assert_eq!(n(&mut hr), 2);
}

/// GPU cost of the shadow maps in a quad view, drawn per pane and shared.
/// Prints numbers; run with `--ignored --nocapture`.
#[test]
#[ignore]
fn measure_shared_shadow_maps() {
    use threers::geometries::SphereGeometry;
    use threers::materials::{Material, StandardMaterial};
    use threers::{Mesh, Object3D};
    let (w, h) = (1280u32, 960u32);
    let Some(mut hr) = HeadlessRenderer::builder().size(w, h).build().ok() else {
        return;
    };
    hr.set_msaa(4);
    let mut scene = shadow_scene();
    // Heavier casters: 64 dense spheres.
    for i in 0..64 {
        let mut o = Object3D::mesh(Mesh::new(
            SphereGeometry::new(0.18, 96, 64),
            Material::Standard(StandardMaterial::new(Color::from_hex(0x8899aa))),
        ));
        o.position = Vector3::new(
            (i % 8) as f32 * 0.6 - 2.1,
            0.4 + (i / 32) as f32 * 0.5,
            ((i / 8) % 4) as f32 * 0.6 - 1.0,
        );
        o.cast_shadow = true;
        scene.add(o);
    }
    let mut q = quad();
    q.frame_scene(&mut scene).unwrap();
    let mut cam = persp();
    cam.position = Vector3::new(5.0, 4.0, 6.0);
    cam.look_at(Vector3::ZERO);
    let device = hr.renderer().device_arc();
    for share in [false, true, false, true] {
        q.share_shadows = share;
        let mut shadow = Vec::new();
        let mut main = Vec::new();
        let n0 = hr.renderer().shadow_map_updates();
        let frames = 120;
        let t0 = std::time::Instant::now();
        for _ in 0..frames {
            q.render(&mut hr, &mut scene, &cam);
            let _ = device.poll(wgpu::PollType::wait_indefinitely());
            let (s, m) = (hr.renderer().gpu_shadow_ms(), hr.renderer().gpu_frame_ms());
            shadow.push(s);
            main.push(m);
        }
        let wall = t0.elapsed().as_secs_f64() * 1e3 / frames as f64;
        let updates = (hr.renderer().shadow_map_updates() - n0) as f64 / frames as f64;
        let med = |v: &mut Vec<f32>| {
            v.retain(|x| *x > 0.0);
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            v.get(v.len() / 2).copied().unwrap_or(0.0)
        };
        let s = med(&mut shadow);
        let m = med(&mut main);
        println!(
            "share_shadows={share}: {updates:.0} shadow updates/frame × {s:.3} ms = {:.3} ms shadow GPU/frame; \
             main pass {m:.3} ms/pane; wall {wall:.2} ms/frame",
            updates as f32 * s
        );
    }
}
