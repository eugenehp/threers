//! Reversed depth (`Renderer::set_reversed_z`) draws what standard depth
//! draws, everything that reads depth back included — and holds apart, far
//! away, surfaces that standard depth cannot.
//!
//! Each scene is rendered both ways and the frames compared. The flip touches
//! every depth test of the view, the clears, the sky, glass, transparency
//! modes, the post effects and TAA, and the marks drawn over the frame; a
//! reader of the depth buffer that missed the conversion shows up here as a
//! frame that changed.

use threers::cameras::PerspectiveCamera;
use threers::core::{Mesh, Object3D};
use threers::geometries::{BoxGeometry, PlaneGeometry, SphereGeometry};
use threers::lights::{AmbientLight, DirectionalLight, Light};
use threers::marks::{MarkRenderer, MarkView, Origin, PointMarks};
use threers::materials::{DepthFunc, Material, PhysicalMaterial, StandardMaterial};
use threers::math::{Color, Vector3};
use threers::scene::Scene;
use threers::HeadlessRenderer;

const W: u32 = 160;
const H: u32 = 120;

fn host() -> Option<HeadlessRenderer> {
    HeadlessRenderer::builder().size(W, H).build().ok()
}

fn camera() -> PerspectiveCamera {
    let mut c = PerspectiveCamera::new(50.0, W as f32 / H as f32, 0.5, 200.0);
    c.position = Vector3::new(4.0, 5.0, 9.0);
    c.target = Vector3::new(0.0, 0.5, 0.0);
    c
}

fn standard(hex: u32) -> StandardMaterial {
    let mut m = StandardMaterial::new(Color::from_hex(hex));
    m.roughness = 0.6;
    m
}

/// A floor, a box and a ball, lit by a shadow-casting sun.
fn lit_scene() -> Scene {
    let mut s = Scene::new();
    s.background = Color::from_hex(0x1a2030);
    s.add_light(AmbientLight::new(Color::WHITE, 0.25));
    let mut sun = DirectionalLight::new(Color::WHITE, 2.0);
    sun.direction = Vector3::new(-0.4, -1.0, -0.3).normalize();
    sun.cast_shadow = true;
    sun.shadow.map_size = 1024;
    sun.shadow.camera_size = 20.0;
    sun.shadow.camera_near = 0.5;
    sun.shadow.camera_far = 60.0;
    let mut light = Object3D::light(Light::Directional(sun));
    light.position = Vector3::new(4.0, 10.0, 3.0);
    s.add(light);
    let mut floor = Object3D::mesh(Mesh::new(
        PlaneGeometry::new(20.0, 20.0),
        standard(0x8899aa).into(),
    ));
    floor.rotate_x(-std::f32::consts::FRAC_PI_2);
    floor.receive_shadow = true;
    s.add(floor);
    let mut cube = Object3D::mesh(Mesh::new(
        BoxGeometry::new(2.0, 2.0, 2.0),
        standard(0xcc5533).into(),
    ));
    cube.position = Vector3::new(-1.0, 1.0, 0.0);
    cube.cast_shadow = true;
    s.add(cube);
    let mut ball = Object3D::mesh(Mesh::new(
        SphereGeometry::new(0.8, 32, 16),
        standard(0x33aa66).into(),
    ));
    ball.position = Vector3::new(1.6, 0.8, 1.0);
    ball.cast_shadow = true;
    s.add(ball);
    s
}

/// `(pixels differing by more than 24 in any channel, largest difference)`.
fn compare(a: &[u8], b: &[u8]) -> (usize, u8) {
    let mut n = 0;
    let mut worst = 0;
    for (p, q) in a.chunks(4).zip(b.chunks(4)) {
        let d = (0..3).map(|c| p[c].abs_diff(q[c])).max().unwrap();
        worst = worst.max(d);
        if d > 24 {
            n += 1;
        }
    }
    (n, worst)
}

/// Render `scene()` with `setup` applied, standard then reversed.
fn both(
    scene: impl Fn() -> Scene,
    setup: impl Fn(&mut HeadlessRenderer),
    frames: usize,
) -> Option<(Vec<u8>, Vec<u8>)> {
    let mut out = Vec::new();
    for reversed in [false, true] {
        let mut hr = host()?;
        setup(&mut hr);
        hr.renderer().set_reversed_z(reversed);
        let mut s = scene();
        let cam = camera();
        let mut img = Vec::new();
        for _ in 0..frames {
            // The resolved readback is the one that applies the post effects.
            img = hr.render_to_rgba_resolved(&mut s, &cam);
        }
        out.push(img);
    }
    let b = out.pop()?;
    let a = out.pop()?;
    Some((a, b))
}

fn assert_same(name: &str, (a, b): (Vec<u8>, Vec<u8>)) {
    let lit = a
        .chunks(4)
        .filter(|p| p[0] > 40 || p[1] > 40 || p[2] > 60)
        .count();
    assert!(
        lit > (W * H / 4) as usize,
        "{name}: the scene drew ({lit} lit px)"
    );
    let (n, worst) = compare(&a, &b);
    // Rounding at the odd silhouette pixel, no more.
    assert!(
        n <= (W * H / 400) as usize,
        "{name}: {n} pixels differ between standard and reversed depth (worst {worst})"
    );
}

#[test]
fn a_lit_shadowed_scene_is_the_same() {
    if let Some(r) = both(lit_scene, |_| {}, 1) {
        assert_same("lit", r);
    }
}

#[test]
fn a_sky_stays_behind_everything() {
    let scene = || {
        let mut s = lit_scene();
        // three.js's Sky: a big box pinned to the far plane by the shader.
        let sky = threers::materials::SkyMaterial {
            sun_position: Vector3::new(0.3, 0.4, -1.0),
            ..Default::default()
        };
        s.add(Object3D::mesh(Mesh::new(
            BoxGeometry::new(150.0, 150.0, 150.0),
            Material::Sky(sky),
        )));
        s
    };
    if let Some(r) = both(scene, |_| {}, 1) {
        assert_same("sky", r);
    }
}

#[test]
fn transparency_and_material_depth_state_are_the_same() {
    let scene = || {
        let mut s = lit_scene();
        // Blended, overlapping; one drawn through everything; one only where
        // it is behind what is already there.
        let mut glass = standard(0x4488ff);
        glass.transparent = Some(true);
        glass.opacity = 0.5;
        let mut pane = Object3D::mesh(Mesh::new(BoxGeometry::new(3.0, 2.0, 0.1), glass.into()));
        pane.position = Vector3::new(0.0, 1.5, 2.0);
        s.add(pane);
        let mut xray = standard(0xffee00);
        xray.render_state.depth_test = false;
        let mut through = Object3D::mesh(Mesh::new(BoxGeometry::new(0.4, 0.4, 0.4), xray.into()));
        through.position = Vector3::new(-1.0, 1.0, -1.5);
        s.add(through);
        // Order-independent transparency: its own pass and pipeline.
        let mut oit = PhysicalMaterial::new(Color::from_hex(0x66ff99))
            .with_transparency(threers::materials::TransparencyMode::Oit);
        oit.opacity = 0.5;
        oit.transparent = Some(true);
        let mut veil = Object3D::mesh(Mesh::new(
            BoxGeometry::new(2.0, 2.0, 0.1),
            Material::Physical(oit),
        ));
        veil.position = Vector3::new(1.5, 1.0, 2.5);
        s.add(veil);
        let mut behind = standard(0xff00ff);
        behind.render_state.depth_func = DepthFunc::Greater;
        let mut ghost = Object3D::mesh(Mesh::new(BoxGeometry::new(0.6, 0.6, 0.6), behind.into()));
        ghost.position = Vector3::new(-1.0, 1.0, -1.2);
        s.add(ghost);
        s
    };
    if let Some(r) = both(scene, |_| {}, 1) {
        assert_same("transparency", r);
    }
}

#[test]
fn glass_refraction_is_the_same() {
    let scene = || {
        let mut s = lit_scene();
        let mut lens = Object3D::mesh(Mesh::new(
            SphereGeometry::new(1.0, 48, 24),
            Material::Physical(threers::materials::presets::optical_glass()),
        ));
        lens.position = Vector3::new(0.5, 1.2, 2.5);
        s.add(lens);
        let _ = PhysicalMaterial::new(Color::WHITE);
        s
    };
    if let Some(r) = both(scene, |_| {}, 1) {
        assert_same("glass", r);
    }
}

#[test]
fn depth_reading_post_effects_are_the_same() {
    // Focused on the box and the ball, ten units out, so a depth read the
    // wrong way round blurs them; occlusion strong enough to see.
    let setup = |hr: &mut HeadlessRenderer| {
        hr.set_ssao(3.0, 1.5);
        hr.set_depth_of_field(10.0, 3.0, 6.0);
        hr.set_bloom(0.4, 0.8, 0.4);
    };
    if let Some(r) = both(lit_scene, setup, 1) {
        assert_same("post effects", r);
    }
}

/// The post settings above change the frame a lot, so the comparison is not
/// passing because they do nothing.
#[test]
fn the_post_effects_compared_are_visible() {
    let Some(mut plain) = host() else { return };
    let a = plain.render_to_rgba_resolved(&mut lit_scene(), &camera());
    let Some(mut fx) = host() else { return };
    fx.set_ssao(3.0, 1.5);
    fx.set_depth_of_field(10.0, 3.0, 6.0);
    let b = fx.render_to_rgba_resolved(&mut lit_scene(), &camera());
    let (n, _) = compare(&a, &b);
    assert!(n > 500, "depth of field and occlusion change {n} px");
}

#[test]
fn taa_and_msaa_are_the_same() {
    if let Some(r) = both(lit_scene, |hr| hr.set_taa(true), 6) {
        assert_same("taa", r);
    }
    if let Some(r) = both(lit_scene, |hr| hr.set_msaa(4), 1) {
        assert_same("msaa", r);
    }
}

/// TAA reprojects each pixel through the depth buffer, so the depth only
/// matters once the camera moves: held still, any depth lands back on the
/// same pixel. Orbit the camera between frames, over a scene with plenty of
/// fine contrast for a misplaced history sample to show against.
#[test]
fn taa_reprojects_a_moving_camera_the_same() {
    let scene = || {
        let mut s = lit_scene();
        // A row of thin posts in alternating colours.
        for i in 0..12 {
            let hex = if i % 2 == 0 { 0xf0f0f0 } else { 0x202060 };
            let mut post = Object3D::mesh(Mesh::new(
                BoxGeometry::new(0.18, 2.0, 0.18),
                standard(hex).into(),
            ));
            post.position = Vector3::new(-3.0 + i as f32 * 0.5, 1.0, -1.5);
            s.add(post);
        }
        s
    };
    let mut out = Vec::new();
    for reversed in [false, true] {
        let Some(mut hr) = host() else {
            return;
        };
        hr.set_taa(true);
        hr.renderer().set_reversed_z(reversed);
        let mut s = scene();
        let mut img = Vec::new();
        for f in 0..8 {
            let mut cam = camera();
            let a = f as f32 * 0.04;
            cam.position = Vector3::new(
                9.85 * a.sin() + 4.0 * a.cos(),
                5.0,
                9.0 * a.cos() - 4.0 * a.sin(),
            );
            img = hr.render_to_rgba_resolved(&mut s, &cam);
        }
        out.push(img);
    }
    let b = out.pop().unwrap();
    let a = out.pop().unwrap();
    assert_same("taa, moving camera", (a, b));
}

/// The composer's outline pass draws two depth passes of its own — the
/// selection and everything else — and compares them in a post effect: a
/// ball half behind the box comes out with the same visible and hidden edges.
#[test]
fn the_outline_pass_compares_reversed_depth() {
    use threers::postprocessing::{EffectComposer, OutlinePass, RenderPass};
    use threers::renderer::{readback, RenderTarget};
    const FORMAT: threers::wgpu::TextureFormat = threers::wgpu::TextureFormat::Rgba8Unorm;
    let frame = |reversed: bool, select: bool| -> Option<Vec<u8>> {
        let mut hr = HeadlessRenderer::builder()
            .size(W, H)
            .color_format(FORMAT)
            .build()
            .ok()?;
        hr.renderer().set_reversed_z(reversed);
        let mut s = lit_scene();
        let mut ball = Object3D::mesh(Mesh::new(
            SphereGeometry::new(0.9, 32, 16),
            standard(0x6688cc).into(),
        ));
        ball.position = Vector3::new(-2.3, 1.2, -1.8);
        let id = s.add(ball);
        let mut outline = OutlinePass::new(W, H, 0, 0, if select { vec![id] } else { Vec::new() });
        outline.visible_edge_color = Color::from_hex(0xff0000);
        outline.hidden_edge_color = Color::from_hex(0x00ff00);
        outline.edge_strength = 10.0;
        let mut composer =
            EffectComposer::new(hr.device().clone(), hr.queue().clone(), W, H, FORMAT);
        composer.add_pass(RenderPass::new());
        composer.add_pass(outline);
        let out = RenderTarget::new(hr.device(), W, H, FORMAT);
        composer.render(hr.renderer(), &mut s, &camera(), &out.color_view);
        Some(readback::texture_bytes(
            hr.device(),
            hr.queue(),
            &out.color_texture,
        ))
    };
    let (Some(plain), Some(a), Some(b)) =
        (frame(false, false), frame(false, true), frame(true, true))
    else {
        return;
    };
    let edges = |img: &[u8], hue: usize| {
        img.chunks(4)
            .zip(plain.chunks(4))
            .filter(|(p, q)| p[hue] as i32 - q[hue] as i32 > 60)
            .count()
    };
    assert!(edges(&a, 0) > 40, "visible edges drawn: {}", edges(&a, 0));
    assert!(edges(&a, 1) > 40, "hidden edges drawn: {}", edges(&a, 1));
    let (n, worst) = compare(&a, &b);
    assert!(
        n <= (W * H / 400) as usize,
        "outline: {n} pixels differ between standard and reversed depth (worst {worst})"
    );
}

/// The composer's depth-reading passes — each with its own depth renders
/// read back through the post-effect stage — draw the same either way, and
/// each visibly changes the frame, so the comparison means something.
#[test]
fn composer_depth_passes_are_the_same() {
    use threers::postprocessing::{
        BokehPass, EffectComposer, Pass, RenderPass, SaoPass, SsaoPass, SsrPass,
    };
    use threers::renderer::{readback, RenderTarget};
    const FORMAT: threers::wgpu::TextureFormat = threers::wgpu::TextureFormat::Rgba8Unorm;
    let frame = |reversed: bool, pass: Option<Box<dyn Pass>>| -> Option<Vec<u8>> {
        let mut hr = HeadlessRenderer::builder()
            .size(W, H)
            .color_format(FORMAT)
            .build()
            .ok()?;
        hr.renderer().set_reversed_z(reversed);
        let mut composer =
            EffectComposer::new(hr.device().clone(), hr.queue().clone(), W, H, FORMAT);
        composer.add_pass(RenderPass::new());
        if let Some(p) = pass {
            composer.add_boxed(p);
        }
        let out = RenderTarget::new(hr.device(), W, H, FORMAT);
        composer.render(hr.renderer(), &mut lit_scene(), &camera(), &out.color_view);
        Some(readback::texture_bytes(
            hr.device(),
            hr.queue(),
            &out.color_texture,
        ))
    };
    let Some(plain) = frame(false, None) else {
        return;
    };
    type Make = fn() -> Box<dyn Pass>;
    let passes: [(&str, Make); 4] = [
        ("ssao", || Box::new(SsaoPass::new(0, 0, W, H, 32))),
        ("sao", || Box::new(SaoPass::new(0, 0, W, H))),
        ("ssr", || Box::new(SsrPass::new(0, 0, W, H))),
        ("bokeh", || Box::new(BokehPass::new(0, 0, 4.0, 0.05, 0.02))),
    ];
    for (name, make) in passes {
        let a = frame(false, Some(make())).unwrap();
        let b = frame(true, Some(make())).unwrap();
        let (changed, _) = compare(&plain, &a);
        assert!(
            changed > (W * H / 100) as usize,
            "{name} changed the frame: {changed} px"
        );
        let (n, worst) = compare(&a, &b);
        assert!(
            n <= (W * H / 400) as usize,
            "{name}: {n} pixels differ between standard and reversed depth (worst {worst})"
        );
    }
}

/// A cube probe draws six faces through its own depth buffer, swapped in
/// for the frame's: the same faces either way.
#[test]
fn cube_faces_are_the_same() {
    const SIDE: u32 = 64;
    let faces = |reversed: bool| -> Option<Vec<u8>> {
        let mut hr = host()?;
        hr.renderer().set_reversed_z(reversed);
        let rt = threers::renderer::RenderTarget::new_cube(
            hr.device(),
            SIDE,
            threers::wgpu::TextureFormat::Rgba8UnormSrgb,
        );
        let mut s = lit_scene();
        let eye = Vector3::new(0.5, 2.5, 3.0);
        let looks = [
            (Vector3::new(1.0, 0.0, 0.0), Vector3::new(0.0, -1.0, 0.0)),
            (Vector3::new(-1.0, 0.0, 0.0), Vector3::new(0.0, -1.0, 0.0)),
            (Vector3::new(0.0, 1.0, 0.0), Vector3::new(0.0, 0.0, 1.0)),
            (Vector3::new(0.0, -1.0, 0.0), Vector3::new(0.0, 0.0, -1.0)),
            (Vector3::new(0.0, 0.0, 1.0), Vector3::new(0.0, -1.0, 0.0)),
            (Vector3::new(0.0, 0.0, -1.0), Vector3::new(0.0, -1.0, 0.0)),
        ];
        for (face, (dir, up)) in looks.into_iter().enumerate() {
            let mut cam = PerspectiveCamera::new(90.0, 1.0, 0.1, 100.0);
            cam.position = eye;
            cam.target = eye + dir;
            cam.up = up;
            hr.renderer().render_to_cube_face(&mut s, &cam, &rt, face);
        }
        let faces = hr.read_cube_faces(&rt)?;
        Some(
            faces
                .iter()
                .flatten()
                .map(|&v| (v.powf(1.0 / 2.2).clamp(0.0, 1.0) * 255.0) as u8)
                .collect(),
        )
    };
    let (Some(a), Some(b)) = (faces(false), faces(true)) else {
        return;
    };
    let total = (6 * SIDE * SIDE) as usize;
    let distinct: std::collections::HashSet<_> = a
        .chunks(4)
        .map(|p| [p[0] / 16, p[1] / 16, p[2] / 16])
        .collect();
    assert!(
        distinct.len() > 8,
        "the probe saw the scene: {} colours",
        distinct.len()
    );
    let (n, worst) = compare(&a, &b);
    assert!(
        n <= total / 400,
        "cube: {n} pixels differ between standard and reversed depth (worst {worst})"
    );
}

/// Four panes drawn into one frame — viewport, scissor and a depth clear
/// per pane, perspective and orthographic — with and without MSAA.
#[test]
fn quad_view_panes_are_the_same() {
    use threers::helpers::QuadView;
    for msaa in [1, 4] {
        let mut out = Vec::new();
        for reversed in [false, true] {
            let Some(mut hr) = host() else {
                return;
            };
            hr.set_msaa(msaa);
            hr.renderer().set_reversed_z(reversed);
            let mut s = lit_scene();
            QuadView::new().render(&mut hr, &mut s, &camera());
            out.push(hr.read_rgba());
        }
        let b = out.pop().unwrap();
        let a = out.pop().unwrap();
        assert_same(&format!("quad view, msaa {msaa}"), (a, b));
    }
}

#[test]
fn marks_draw_and_pick_against_a_reversed_frame() {
    let mut frames = Vec::new();
    let mut picks = Vec::new();
    for reversed in [false, true] {
        let Some(mut hr) = host() else { return };
        hr.renderer().set_reversed_z(reversed);
        let mut s = lit_scene();
        let cam = camera();
        hr.render(&mut s, &cam);
        // One point in front of the box, one hidden behind it.
        let mut pts = PointMarks::new();
        pts.set_positions(&[[-1.0, 1.0, 1.6], [-1.0, 1.0, -1.6]]);
        pts.set_radii(&[6.0, 6.0]);
        pts.set_default_color([255, 255, 255, 255]);
        let mut mr = MarkRenderer::new(hr.device());
        let view = MarkView::from_renderer(&cam, Origin::ZERO, hr.renderer());
        assert_eq!(view.reversed_z, reversed);
        let (d, q) = (hr.device().clone(), hr.queue().clone());
        mr.draw(&d, &q, &hr.mark_target(), &view, &mut [&mut pts]);
        frames.push(hr.read_rgba());
        let front = view.project([-1.0, 1.0, 1.6]).unwrap();
        let back = view.project([-1.0, 1.0, -1.6]).unwrap();
        let depth = &hr.target().depth_view;
        let pick = |x: f32, y: f32, mr: &mut MarkRenderer, pts: &mut PointMarks| {
            mr.pick(&d, &q, &view, &mut [pts], Some(depth), x, y, 1)
                .wait(&d)
                .map(|h| h.index)
        };
        picks.push((
            pick(front[0], front[1], &mut mr, &mut pts),
            pick(back[0], back[1], &mut mr, &mut pts),
        ));
    }
    assert_eq!(
        picks[0],
        (Some(0), None),
        "standard: the front point picks, the hidden one does not"
    );
    assert_eq!(picks[1], picks[0], "reversed picks the same");
    let (n, worst) = compare(&frames[0], &frames[1]);
    assert!(n <= (W * H / 400) as usize, "{n} px differ (worst {worst})");
}

/// Two walls a metre apart, 20 km away, seen with a near plane of 5 cm: the
/// nearer one should hide the farther completely. Standard `f32` depth cannot
/// tell them apart there — the metre is about 1e-10 of depth, against an
/// `f32` step of 6e-8 near 1 — and lets the back wall through in patches;
/// reversed depth keeps them separate. (At 2 km, 5x below the step rather
/// than 500x, NVIDIA's rasterizer still held the walls apart.)
#[test]
fn reversed_depth_holds_apart_what_standard_depth_cannot() {
    let scene = || {
        let mut s = Scene::new();
        s.background = Color::from_hex(0x000000);
        s.add_light(AmbientLight::new(Color::WHITE, 1.0));
        let wall = |hex: u32, z: f32| {
            let mut m = standard(hex);
            m.emissive = Color::from_hex(hex);
            let mut o = Object3D::mesh(Mesh::new(PlaneGeometry::new(40000.0, 40000.0), m.into()));
            o.position = Vector3::new(0.0, 0.0, z);
            // Tilted, so depth varies across the frame.
            o.rotate_x(-1.2);
            o
        };
        s.add(wall(0x0000ff, -20001.0));
        s.add(wall(0xff0000, -20000.0));
        s
    };
    let leak = |reversed: bool| -> Option<usize> {
        let mut hr = host()?;
        hr.renderer().set_reversed_z(reversed);
        let mut cam = PerspectiveCamera::new(50.0, W as f32 / H as f32, 0.05, 1e5);
        cam.position = Vector3::new(0.0, 0.0, 0.0);
        cam.target = Vector3::new(0.0, -0.3, -1.0);
        let img = hr.render_to_rgba(&mut scene(), &cam);
        // Blue showing where the red wall is in front.
        Some(img.chunks(4).filter(|p| p[2] > 128 && p[0] < 64).count())
    };
    let (Some(std), Some(rev)) = (leak(false), leak(true)) else {
        return;
    };
    // The control: that the scene is hard enough to need reversed depth.
    // Which wall wins a tie in standard depth is the driver's business, and
    // not a steady one — on NVIDIA (595.99) the back wall showed through in
    // 4000 px with the test run alone and in none after a few other tests
    // in the same process, its shaders by then recompiled in the
    // background. So the control is reported rather than required; Apple's
    // GPU shows the leak every time.
    if std <= 100 {
        eprintln!("note: standard depth leaked only {std} px here — the control shows nothing on this driver");
    }
    assert_eq!(rev, 0, "reversed depth does not: {rev} px leak");
}

/// Ambient occlusion on its own, judged finely: it darkens creases by a few
/// levels, which the coarse comparison above cannot see either way.
#[test]
fn ambient_occlusion_reads_reversed_depth() {
    let run = |ssao: bool, reversed: bool| -> Option<Vec<u8>> {
        let mut hr = host()?;
        if ssao {
            hr.set_ssao(8.0, 2.0);
        }
        hr.renderer().set_reversed_z(reversed);
        Some(hr.render_to_rgba_resolved(&mut lit_scene(), &camera()))
    };
    let (Some(none), Some(std), Some(rev)) = (run(false, false), run(true, false), run(true, true))
    else {
        return;
    };
    let finely = |a: &[u8], b: &[u8]| {
        a.chunks(4)
            .zip(b.chunks(4))
            .filter(|(p, q)| (0..3).any(|c| p[c].abs_diff(q[c]) > 3))
            .count()
    };
    let shown = finely(&none, &std);
    assert!(shown > 300, "occlusion is visible: {shown} px");
    let off = finely(&std, &rev);
    assert!(
        off <= shown / 20,
        "reversed occlusion matches: {off} px differ of {shown} it touches"
    );
}

/// A post effect applied through `apply_postfx` reading the hardware depth:
/// bokeh (kind 26) blurs by each pixel's distance from the focus. Judged
/// finely, as the blur is gentle at this size: first that moving the focus
/// changes the frame — the pass really reads depth — then that reversed depth
/// gives what standard does.
#[test]
fn apply_postfx_reads_reversed_hardware_depth() {
    use std::sync::Arc;
    use threers::renderer::{readback, PostFxCamera};
    use threers::RenderTarget;
    let fmt = threers::wgpu::TextureFormat::Rgba8UnormSrgb;
    let run = |reversed: bool, focus: f32| -> Option<Vec<u8>> {
        let mut hr = host()?;
        hr.renderer().set_reversed_z(reversed);
        let frame = Arc::new(RenderTarget::new(hr.device(), W, H, fmt));
        let out = Arc::new(RenderTarget::new(hr.device(), W, H, fmt));
        hr.renderer().register_render_target(1, &frame);
        hr.renderer().register_render_target(2, &out);
        let cam = camera();
        hr.renderer().render_to(&mut lit_scene(), &cam, &frame);
        let pc = PostFxCamera {
            near: cam.near,
            far: cam.far,
            ..PostFxCamera::default()
        };
        let p2 = [focus, 0.05, 0.1, W as f32 / H as f32];
        // Depth and normal both name the frame: the hardware-depth binding.
        hr.renderer().apply_postfx_by_id(
            1,
            2,
            &out.color_view,
            1,
            1,
            26,
            0.0,
            p2,
            [0.0; 4],
            W,
            H,
            false,
            pc,
        );
        Some(readback::texture_bytes(
            hr.device(),
            hr.queue(),
            &out.color_texture,
        ))
    };
    let finely = |a: &[u8], b: &[u8]| {
        a.chunks(4)
            .zip(b.chunks(4))
            .filter(|(p, q)| (0..3).any(|c| p[c].abs_diff(q[c]) > 3))
            .count()
    };
    let (Some(near), Some(far), Some(rev)) =
        (run(false, 10.0), run(false, 1000.0), run(true, 10.0))
    else {
        return;
    };
    let depends = finely(&near, &far);
    assert!(
        depends > 1000,
        "the blur follows depth: {depends} px change with the focus"
    );
    let off = finely(&near, &rev);
    assert!(
        off <= depends / 20,
        "reversed matches: {off} px differ of {depends}"
    );
}
