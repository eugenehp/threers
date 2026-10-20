//! The composer and its passes, natively: sequencing against three.js's
//! rules, each pass's effect measured in the pixels, and the seeded
//! randomness against three.js's draw order.

use super::*;
use crate::renderer::readback;
use crate::{
    AmbientLight, BasicMaterial, BoxGeometry, Color, DirectionalLight, Mesh, Object3D,
    OrthographicCamera, PerspectiveCamera, PlaneGeometry, Scene, StandardMaterial, Vector3,
};

const W: u32 = 64;
const H: u32 = 64;
const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

struct Rig {
    gpu: crate::gpu::GpuContext,
    renderer: crate::Renderer,
}

fn rig(what: &str) -> Option<Rig> {
    let gpu = crate::gpu::testing::context(what)?;
    let renderer = crate::Renderer::new(gpu.device.clone(), gpu.queue.clone(), FORMAT, W, H);
    Some(Rig { gpu, renderer })
}

impl Rig {
    /// One frame of `passes` after a RenderPass, into an 8-bit screen target;
    /// RGBA bytes, top row first.
    fn frame(&mut self, scene: &mut Scene, camera: &dyn crate::cameras::Camera, passes: Vec<Box<dyn Pass>>) -> Vec<u8> {
        let mut composer = EffectComposer::with_size(W, H);
        composer.add_pass(RenderPass::new());
        for p in passes {
            composer.add_boxed(p);
        }
        self.run(&mut composer, scene, camera)
    }

    /// Two frames a sixtieth of a second apart (so an accumulating pass has
    /// a history); the second's pixels.
    fn run(&mut self, composer: &mut EffectComposer, scene: &mut Scene, camera: &dyn crate::cameras::Camera) -> Vec<u8> {
        let screen = Target::new(&mut self.renderer, W, H, FORMAT);
        for _ in 0..2 {
            composer.render_frame(
                &mut self.renderer,
                &mut [&mut *scene],
                &[camera],
                &Screen::Target(screen.clone()),
                Some(1.0 / 60.0),
            );
        }
        readback::texture_bytes(&self.gpu.device, &self.gpu.queue, &screen.render_target().color_texture)
    }
}

fn px(bytes: &[u8], x: u32, y: u32) -> [u8; 4] {
    let i = ((y * W + x) * 4) as usize;
    [bytes[i], bytes[i + 1], bytes[i + 2], bytes[i + 3]]
}

fn front() -> OrthographicCamera {
    let mut c = OrthographicCamera::new(-1.0, 1.0, 1.0, -1.0, 0.1, 10.0);
    c.position = Vector3::new(0.0, 0.0, 5.0);
    c.look_at(Vector3::ZERO);
    c
}

/// A white square on a dark background, left of centre: (x 16..32, y 16..48).
fn square(background: Color) -> (Scene, crate::core::ObjectId) {
    let mut scene = Scene::new();
    scene.background = background;
    let mut q = Object3D::mesh(Mesh::new(
        PlaneGeometry::new(0.5, 1.0),
        BasicMaterial::new(Color::WHITE).into(),
    ));
    q.position = Vector3::new(-0.25, 0.0, 0.0);
    let id = scene.add(q);
    (scene, id)
}

fn differ(a: &[u8], b: &[u8]) -> usize {
    a.chunks_exact(4)
        .zip(b.chunks_exact(4))
        .filter(|(x, y)| x.iter().zip(y.iter()).take(3).any(|(p, q)| p.abs_diff(*q) > 2))
        .count()
}

// ---- the composer's rules --------------------------------------------------

/// three.js `EffectComposer.render`: the last *enabled* pass gets
/// `renderToScreen`, the buffers swap after a pass that needs it and not
/// after one that does not, and `insertPass` / `removePass` keep the order.
#[test]
fn the_composer_follows_three_js_sequencing() {
    let mut c = EffectComposer::with_size(W, H);
    c.add_pass(RenderPass::new());
    c.add_pass(ShaderPass::copy());
    let mut disabled = ShaderPass::copy();
    disabled.state.enabled = false;
    c.add_pass(disabled);
    assert!(!c.is_last_enabled_pass(0));
    assert!(c.is_last_enabled_pass(1), "a disabled pass after it does not count");
    c.insert_boxed(Box::new(FilmPass::default()), 1);
    assert_eq!(c.pass_names(), ["RenderPass", "FilmPass", "ShaderPass", "ShaderPass"]);
    let removed = c.remove_pass(1).expect("removed");
    assert_eq!(removed.name(), "FilmPass");
    c.set_pixel_ratio(2.0);
    assert_eq!(c.drawing_size(), (2 * W, 2 * H));

    let Some(mut r) = rig("composer sequencing") else { return };
    let (mut scene, _) = square(Color::BLACK);
    c.set_pixel_ratio(1.0);
    let before_write = {
        c.ensure_buffers(&mut r.renderer);
        c.write_buffer().unwrap().id()
    };
    let screen = Target::new(&mut r.renderer, W, H, FORMAT);
    c.render_frame(&mut r.renderer, &mut [&mut scene], &[&front()], &Screen::Target(screen), Some(0.0));
    // RenderPass does not swap; the copy does: one swap in all.
    assert_ne!(c.write_buffer().unwrap().id(), before_write);
    assert!(c.passes()[1].state().render_to_screen);
    assert!(!c.passes()[0].state().render_to_screen);
}

/// A pass that draws itself runs in the chain, with the read buffer as its
/// input and — as the last pass — the screen as its target.
#[test]
fn a_draw_pass_runs_in_the_chain() {
    let Some(mut r) = rig("draw pass") else { return };
    let calls = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let seen = calls.clone();
    let draw = DrawPass::new("flat", move |ctx| {
        seen.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        assert_eq!((ctx.width, ctx.height), (W, H));
        let _ = crate::renderer::fullscreen::begin_with(
            ctx.encoder,
            ctx.target,
            crate::renderer::fullscreen::Load::Clear,
            wgpu::Color { r: 0.0, g: 1.0, b: 0.0, a: 1.0 },
            "flat",
        );
    });
    let (mut scene, _) = square(Color::BLACK);
    let out = r.frame(&mut scene, &front(), vec![Box::new(draw)]);
    // `frame` renders twice.
    assert_eq!(calls.load(std::sync::atomic::Ordering::Relaxed), 2);
    let p = px(&out, W / 2, H / 2);
    assert!(p[1] > 200 && p[0] < 40, "the draw pass's green did not reach the screen: {p:?}");
}

// ---- randomness ------------------------------------------------------------

/// GlitchPass draws three.js's numbers in three.js's order: the heightmap,
/// sixteen for the UUIDs of the objects its constructor makes, the trigger,
/// then per frame the seed and the glitch.
#[test]
fn glitch_pass_draws_three_js_numbers_in_three_js_order() {
    let _guard = random_lock();
    random::seed_random(42);
    let p = GlitchPass::new(4);
    let mut lcg = Lcg(42);
    for &h in p.height_map() {
        assert_eq!(h, lcg.next() as f32);
    }
    for _ in 0..16 {
        lcg.next();
    }
    let rand_x = 120 + (lcg.next() * 121.0).floor() as i64;
    assert_eq!(p.rand_x, rand_x);
    let mut p = p;
    p.step();
    // Frame 0 is a trigger: seed, amount, angle, seed_x, seed_y, distortions.
    let seed = lcg.next() as f32;
    assert_eq!(p.uniforms.f("seed"), seed);
    assert_eq!(p.uniforms.f("amount"), (lcg.next() / 30.0) as f32);
    random::unseed_random();
}

/// SSAOPass builds three.js's kernel and noise from the stream: the kernel
/// first, then SimplexNoise's 256-entry permutation, then the coordinates.
#[test]
fn ssao_kernel_is_three_js_kernel() {
    let _guard = random_lock();
    random::seed_random(0x5a00426);
    let p = SsaoPass::new(0, 0, W, H, 4);
    let mut lcg = Lcg(0x5a00426);
    for (i, k) in p.kernel().iter().enumerate() {
        let (x, y, z) = (lcg.next() * 2.0 - 1.0, lcg.next() * 2.0 - 1.0, lcg.next());
        let len = (x * x + y * y + z * z).sqrt();
        let s = i as f64 / 4.0;
        let s = 0.1 + 0.9 * s * s;
        assert_eq!(*k, [(x / len * s) as f32, (y / len * s) as f32, (z / len * s) as f32]);
    }
    random::unseed_random();
    assert!(p.noise().iter().all(|v| v.is_finite() && v.abs() <= 1.0));
}

struct Lcg(u32);
impl Lcg {
    fn next(&mut self) -> f64 {
        self.0 = self.0.wrapping_mul(1664525).wrapping_add(1013904223);
        self.0 as f64 / 4294967296.0
    }
}

/// The stream is global: tests that draw from it take turns.
fn random_lock() -> std::sync::MutexGuard<'static, ()> {
    random::test_lock()
}

// ---- the passes, in pixels -------------------------------------------------

/// UnrealBloomPass lights the background beside a bright square and leaves
/// the far background alone; with a threshold above the square, nothing.
#[test]
fn unreal_bloom_glows_round_bright_areas() {
    let Some(mut r) = rig("unreal bloom") else { return };
    let (mut scene, _) = square(Color::new(0.02, 0.02, 0.02));
    // The baseline is the same pass at strength 0: as the last pass it copies
    // the frame with a MeshBasicMaterial (sRGB on the screen) first.
    let plain = r.frame(&mut scene, &front(), vec![Box::new(UnrealBloomPass::new(W, H, 0.0, 0.0, 0.5))]);
    let bloom = r.frame(&mut scene, &front(), vec![Box::new(UnrealBloomPass::new(W, H, 1.5, 0.0, 0.5))]);
    let none = r.frame(&mut scene, &front(), vec![Box::new(UnrealBloomPass::new(W, H, 1.5, 0.4, 1.5))]);
    let beside = |b: &[u8]| px(b, 34, 32)[0] as i32;
    assert!(beside(&bloom) > beside(&plain) + 10, "no glow: {} vs {}", beside(&bloom), beside(&plain));
    // Falling off: far less at the frame's far corner than beside it.
    let far = px(&bloom, 62, 1)[0] as i32 - px(&plain, 62, 1)[0] as i32;
    assert!(far * 4 < beside(&bloom) - beside(&plain), "no fall-off: {far} at the far corner");
    assert!(beside(&none) <= beside(&plain) + 2, "bloom below a threshold above the square");
    // `radius` moves the weight from the sharp mips to the wide ones.
    let mut wide = UnrealBloomPass::new(W, H, 1.5, 0.0, 0.5);
    assert!(wide.set("radius", ParamValue::Float(1.0)));
    let wide = r.frame(&mut scene, &front(), vec![Box::new(wide)]);
    assert!(differ(&bloom, &wide) > 50, "radius did not change the bloom");
}

/// Every SSAOPass output draws something of its own: Depth the linearised
/// depth (it used to copy a target nothing rendered), Normal the normals,
/// SSAO and Blur the occlusion, Default the beauty darkened in the crease.
#[test]
fn ssao_outputs_each_draw_their_buffer() {
    let _guard = random_lock();
    let Some(mut r) = rig("ssao outputs") else { return };
    let mut scene = Scene::new();
    scene.background = Color::new(0.5, 0.5, 0.5);
    scene.add_light(AmbientLight::new(Color::WHITE, 1.0));
    let mut floor = Object3D::mesh(Mesh::new(
        BoxGeometry::new(6.0, 0.2, 6.0),
        StandardMaterial::new(Color::WHITE).into(),
    ));
    floor.position = Vector3::new(0.0, -0.6, 0.0);
    scene.add(floor);
    scene.add(Object3D::mesh(Mesh::new(
        BoxGeometry::new(1.0, 1.0, 1.0),
        StandardMaterial::new(Color::WHITE).into(),
    )));
    let mut camera = PerspectiveCamera::new(50.0, 1.0, 0.1, 20.0);
    camera.position = Vector3::new(1.6, 1.2, 2.6);
    camera.look_at(Vector3::new(0.0, -0.2, 0.0));
    let frame = |r: &mut Rig, scene: &mut Scene, output: i32| {
        let mut p = SsaoPass::new(0, 0, W, H, 32);
        p.kernel_radius = 1.5;
        p.output = output;
        r.frame(scene, &camera, vec![Box::new(p)])
    };
    let outs: Vec<Vec<u8>> = (0..5).map(|o| frame(&mut r, &mut scene, o)).collect();
    for (a, an) in outs.iter().zip(["Default", "SSAO", "Blur", "Depth", "Normal"]) {
        let distinct: std::collections::HashSet<[u8; 4]> = a.chunks_exact(4).map(|p| [p[0], p[1], p[2], p[3]]).collect();
        assert!(distinct.len() > 3, "OUTPUT.{an} is flat ({} colours)", distinct.len());
    }
    for i in 0..5 {
        for j in i + 1..5 {
            assert!(differ(&outs[i], &outs[j]) > 100, "outputs {i} and {j} are the same picture");
        }
    }
    // Depth: nearer is brighter (1 - linear depth), the background black.
    let depth = &outs[3];
    assert!(px(depth, W / 2, H / 2)[0] > px(depth, 2, 2)[0]);
}

/// OutlinePass draws its edge round the selected object only, in the visible
/// colour, and in the hidden colour where something in front hides it.
#[test]
fn outline_draws_round_the_selection() {
    let Some(mut r) = rig("outline") else { return };
    let mut scene = Scene::new();
    scene.background = Color::BLACK;
    let mut a = Object3D::mesh(Mesh::new(PlaneGeometry::new(0.5, 0.5), BasicMaterial::new(Color::new(0.2, 0.2, 0.2)).into()));
    a.position = Vector3::new(-0.5, 0.0, 0.0);
    let selected = scene.add(a);
    let mut b = Object3D::mesh(Mesh::new(PlaneGeometry::new(0.5, 0.5), BasicMaterial::new(Color::new(0.2, 0.2, 0.2)).into()));
    b.position = Vector3::new(0.5, 0.0, 0.0);
    scene.add(b);
    // An occluder over the selection's top-left corner.
    let mut c = Object3D::mesh(Mesh::new(PlaneGeometry::new(0.4, 0.4), BasicMaterial::new(Color::new(0.2, 0.2, 0.2)).into()));
    c.position = Vector3::new(-0.75, 0.25, 1.0);
    scene.add(c);
    let mut outline = OutlinePass::new(W, H, 0, 0, vec![selected]);
    outline.visible_edge_color = Color::new(1.0, 0.0, 0.0);
    outline.hidden_edge_color = Color::new(0.0, 0.0, 1.0);
    outline.edge_strength = 6.0;
    let plain = r.frame(&mut scene, &front(), vec![Box::new(ShaderPass::copy())]);
    let out = r.frame(&mut scene, &front(), vec![Box::new(outline)]);
    // Bottom edge of the selection (y = -0.25 → row 40): red appears.
    let red_rows = (14..30).filter(|&x| px(&out, x, 40)[0] as i32 > px(&plain, x, 40)[0] as i32 + 30).count();
    assert!(red_rows > 6, "no visible edge under the selection");
    // The unselected square's edge is untouched.
    let other = (34..50).filter(|&x| differ(&out[((40 * W + x) * 4) as usize..][..4], &plain[((40 * W + x) * 4) as usize..][..4]) > 0).count();
    assert_eq!(other, 0, "the outline reached the unselected square");
    // The selection's top edge (row 24, from x = 8) runs under the occluder
    // (x up to 14.4, rows 17.6..30.4) for its first few pixels: hidden colour
    // there, the visible colour once it comes out (x = 20).
    let blue_over_red = |x, y| px(&out, x, y)[2] as i32 - px(&out, x, y)[0] as i32;
    let hidden = (20..24).any(|y| (9..14).any(|x| blue_over_red(x, y) > 20));
    assert!(hidden, "no hidden-colour edge behind the occluder");
    let visible = (20..24).any(|y| blue_over_red(20, y) < -20);
    assert!(visible, "no visible-colour edge where the top edge comes out");
}

/// MaskPass confines the passes up to ClearMaskPass to what its scene covers:
/// a ClearPass inside the mask clears only the square.
#[test]
fn mask_pass_confines_the_passes_between() {
    let Some(mut r) = rig("mask pass") else { return };
    let (mut scene, _) = square(Color::new(0.3, 0.3, 0.3));
    let mut composer = EffectComposer::with_size(W, H);
    composer.add_pass(RenderPass::new());
    composer.add_pass(MaskPass::new(0, 0));
    composer.add_pass(ClearPass::new(Some(Color::new(1.0, 0.0, 0.0)), 1.0));
    composer.add_pass(ClearMaskPass::new());
    composer.add_pass(ShaderPass::copy());
    let out = r.run(&mut composer, &mut scene, &front());
    let inside = px(&out, 24, 32);
    let outside = px(&out, 50, 32);
    assert!(inside[0] > 200 && inside[1] < 40, "inside the mask is not the clear colour: {inside:?}");
    assert!(outside[0] < 120 && outside[0] == outside[1], "outside the mask changed: {outside:?}");
}

/// Each knob moves the frame: a pass's parameter set by name reaches its
/// shader. One frame with the default, one with the knob moved.
#[test]
fn pass_parameters_reach_their_shaders() {
    let _guard = random_lock();
    let Some(mut r) = rig("pass parameters") else { return };
    let mut scene = Scene::new();
    scene.background = Color::new(0.1, 0.1, 0.1);
    scene.add_light(AmbientLight::new(Color::WHITE, 0.3));
    scene.add_light(DirectionalLight::new(Color::WHITE, 2.0).with_direction(Vector3::new(-0.5, -0.7, -0.5).normalize()));
    scene.add(Object3D::mesh(Mesh::new(crate::SphereGeometry::new(0.8, 32, 16), StandardMaterial::new(Color::from_hex(0x9aa0a6)).into())));
    // A wall just behind the sphere, for the occlusion passes to find.
    let mut wall = Object3D::mesh(Mesh::new(PlaneGeometry::new(8.0, 8.0), StandardMaterial::new(Color::WHITE).into()));
    wall.position = Vector3::new(0.0, 0.0, -0.9);
    scene.add(wall);
    let mut camera = PerspectiveCamera::new(50.0, 1.0, 0.1, 10.0);
    camera.position = Vector3::new(0.9, 0.9, 2.4);
    camera.look_at(Vector3::ZERO);
    type Make = fn() -> Box<dyn Pass>;
    let cases: Vec<(&str, Make, &str, ParamValue)> = vec![
        // Weak, so neither frame saturates: the additive blend multiplies
        // the bloom by its alpha, strength × Σ factors, as three.js's does.
        ("UnrealBloomPass", || Box::new(UnrealBloomPass::new(W, H, 0.02, 0.4, 0.1)), "strength", ParamValue::Float(0.1)),
        ("UnrealBloomPass", || Box::new(UnrealBloomPass::new(W, H, 0.1, 0.4, 0.1)), "threshold", ParamValue::Float(0.9)),
        // BloomPass draws into readBuffer only (three.js's never reaches the
        // screen itself): the frame shows it once something copies it on.
        ("BloomPass", || Box::new(BloomPass::default()), "strength", ParamValue::Float(3.0)),
        ("FilmPass", || Box::new(FilmPass::new(0.5, false)), "intensity", ParamValue::Float(1.0)),
        ("FilmPass", || Box::new(FilmPass::new(0.5, false)), "grayscale", ParamValue::Bool(true)),
        ("DotScreenPass", || Box::new(DotScreenPass::default()), "scale", ParamValue::Float(0.3)),
        ("HalftonePass", || Box::new(HalftonePass::new(W, H, &[])), "radius", ParamValue::Float(8.0)),
        ("OutputPass", || Box::new(OutputPass::new()), "toneMapping", ParamValue::Int(4)),
        ("ShaderPass(Copy)", || Box::new(ShaderPass::copy()), "opacity", ParamValue::Float(0.5)),
        ("ShaderPass(LuminosityHighPass)", || Box::new(ShaderPass::new(ShaderKind::LuminosityHighPass)), "luminosityThreshold", ParamValue::Float(0.0)),
        ("BokehPass", || Box::new(BokehPass::new(0, 0, 1.0, 0.025, 0.01)), "maxblur", ParamValue::Float(0.05)),
        ("SAOPass", || Box::new(SaoPass::new(0, 0, W, H)), "saoIntensity", ParamValue::Float(1.0)),
        ("RenderPixelatedPass", || Box::new(RenderPixelatedPass::new(2, 0, 0)), "pixelSize", ParamValue::Int(6)),
        ("AdaptiveToneMappingPass", || Box::new(AdaptiveToneMappingPass::new(false, 64)), "middleGrey", ParamValue::Float(0.1)),
    ];
    let mut dead = Vec::new();
    for (name, make, key, value) in cases {
        let tail = || -> Box<dyn Pass> { Box::new(ShaderPass::copy()) };
        let before = r.frame(&mut scene, &camera, vec![make(), tail()]);
        let mut p = make();
        assert!(p.set(key, value.clone()), "{name} has no parameter {key}");
        let after = r.frame(&mut scene, &camera, vec![p, tail()]);
        let n = differ(&before, &after);
        if n < 20 {
            dead.push(format!("{name}.{key}: {n} pixels"));
        }
    }
    assert!(dead.is_empty(), "parameters that do not reach the frame:\n  {}", dead.join("\n  "));
}

/// SMAA softens a hard diagonal edge; LUTPass maps colours through its cube
/// (an inverting LUT inverts); RenderTransitionPass shows one scene, the
/// other, or a mix by `mixRatio`.
#[test]
fn smaa_lut_and_transition_do_what_they_say() {
    let Some(mut r) = rig("smaa lut transition") else { return };
    let mut scene = Scene::new();
    scene.background = Color::BLACK;
    let mut q = Object3D::mesh(Mesh::new(PlaneGeometry::new(1.0, 1.0), BasicMaterial::new(Color::WHITE).into()));
    q.quaternion = crate::Quaternion::from_axis_angle(Vector3::new(0.0, 0.0, 1.0), 0.3);
    scene.add(q);
    let plain = r.frame(&mut scene, &front(), vec![Box::new(ShaderPass::copy())]);
    let smaa = r.frame(&mut scene, &front(), vec![Box::new(SmaaPass::new(W, H))]);
    let grey = |b: &[u8]| b.chunks_exact(4).filter(|p| p[0] > 20 && p[0] < 235).count();
    assert!(grey(&smaa) > grey(&plain) + 10, "SMAA added no intermediate edge pixels");

    let mut lut = LutPass::new(1.0);
    let size = 4u32;
    let mut cube = Vec::new();
    for z in 0..size {
        for y in 0..size {
            for x in 0..size {
                let f = |v: u32| 1.0 - v as f32 / (size - 1) as f32;
                cube.push([f(x), f(y), f(z), 1.0]);
            }
        }
    }
    lut.set_lut(size, Some(cube));
    let inverted = r.frame(&mut scene, &front(), vec![Box::new(lut)]);
    let (a, b) = (px(&plain, 2, 2)[0], px(&inverted, 2, 2)[0]);
    assert!(a < 20 && b > 200, "the inverting LUT left black at {b}");

    let (mut other, _) = square(Color::new(0.0, 0.0, 1.0));
    let mut camera = front();
    let mut show = |ratio: f32| -> Vec<u8> {
        let mut t = RenderTransitionPass::new(0, 0, 1, 0);
        t.use_texture = false;
        t.mix_ratio = ratio;
        let mut c = EffectComposer::with_size(W, H);
        c.add_pass(t);
        let screen = Target::new(&mut r.renderer, W, H, FORMAT);
        camera.position = Vector3::new(0.0, 0.0, 5.0);
        c.render_frame(&mut r.renderer, &mut [&mut scene, &mut other], &[&camera], &Screen::Target(screen.clone()), None);
        readback::texture_bytes(&r.gpu.device, &r.gpu.queue, &screen.render_target().color_texture)
    };
    let a_only = show(1.0);
    let b_only = show(0.0);
    let half = show(0.5);
    assert!(px(&b_only, 60, 4)[2] > 200, "mixRatio 0 is not scene B");
    assert!(px(&a_only, 60, 4)[2] < 20, "mixRatio 1 is not scene A");
    let mid = px(&half, 60, 4)[2];
    assert!(mid > 40 && mid < 220, "mixRatio 0.5 is not a mix: {mid}");
}


/// AfterimagePass keeps a fading trail of what moved: an object drawn last
/// frame and gone this one is still there, dimmer, with `damp` 0.96, and gone
/// with `damp` 0.
#[test]
fn afterimage_leaves_a_fading_trail() {
    let Some(mut r) = rig("afterimage") else { return };
    let camera = front();
    let mut trail = |damp: f32| -> u8 {
        let mut c = EffectComposer::with_size(W, H);
        c.add_pass(RenderPass::new());
        c.add_pass(AfterimagePass::new(damp));
        c.add_pass(ShaderPass::copy());
        let screen = Target::new(&mut r.renderer, W, H, FORMAT);
        let (mut with, _) = square(Color::BLACK);
        let mut without = Scene::new();
        without.background = Color::BLACK;
        for scene in [&mut with, &mut without] {
            c.render_frame(&mut r.renderer, &mut [scene], &[&camera], &Screen::Target(screen.clone()), Some(1.0 / 60.0));
        }
        let b = readback::texture_bytes(&r.gpu.device, &r.gpu.queue, &screen.render_target().color_texture);
        px(&b, 24, 32)[0]
    };
    let (kept, gone) = (trail(0.96), trail(0.0));
    assert!(kept > 150 && gone < 10, "trail {kept} with damp 0.96, {gone} with 0");
}
