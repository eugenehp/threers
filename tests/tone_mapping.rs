//! Every tone-mapping mode, on the GPU and on the CPU, against each other.
//!
//! There are three implementations of these curves in the crate — the WGSL the
//! rasteriser runs, the CPU form `SvgRenderer` and the path tracer's film share,
//! and whatever a `ShaderMaterial` does — and nothing was checking that any two
//! of them agreed. They drift in ways that look like art direction: while this
//! test was being written the CPU ACES carried a `* 0.6` pre-exposure the shader
//! does not, which is a whole stop, and the only symptom would have been an SVG
//! export that came out darker than its PNG.

use threers::{
    BoxGeometry, Color, HeadlessRenderer, Material, Mesh, Object3D, PerspectiveCamera, Scene,
    ShaderMaterial, ToneMapping, Vector3,
};

const W: u32 = 32;

const MODES: [(ToneMapping, &str); 5] = [
    (ToneMapping::None, "none"),
    (ToneMapping::Linear, "linear"),
    (ToneMapping::AcesFilmic, "aces"),
    (ToneMapping::Neutral, "neutral"),
    (ToneMapping::AgX, "agx"),
];

/// Draw a flat emissive colour through the renderer's own output path and read
/// the centre pixel back, sRGB-encoded exactly as the canvas would get it.
fn through_the_gpu(
    r: &mut HeadlessRenderer,
    linear: [f32; 3],
    mode: ToneMapping,
    exposure: f32,
) -> [u8; 3] {
    r.renderer().set_tone_mapping(mode, exposure);

    // A `ShaderMaterial` that ends the way the docs tell one to: the shared
    // dispatcher, then the shared encode. Nothing material-specific, so what
    // comes back is the renderer's curve and not a second copy of it.
    let material = ShaderMaterial::new(
        "@fragment\n\
         fn fs_main(in: VsOut) -> @location(0) vec4<f32> {\n\
         let hdr = u_data.data[0].rgb;\n\
         return vec4<f32>(framebuffer_encode(apply_tone_mapping(hdr)), 1.0);\n\
         }\n",
    )
    .with_data(vec![[linear[0], linear[1], linear[2], 1.0]]);

    let mut scene = Scene::new();
    scene.background = Color::from_hex(0x000000);
    scene.add(Object3D::mesh(Mesh::new(
        BoxGeometry::new(4.0, 4.0, 0.1),
        Material::Shader(material),
    )));
    let mut cam = PerspectiveCamera::new(45.0, 1.0, 0.1, 100.0);
    cam.position = Vector3::new(0.0, 0.0, 3.0);
    cam.look_at(Vector3::ZERO);
    scene.update_world();
    let px = r.render_to_rgba(&mut scene, &cam);
    let i = ((W / 2 * W + W / 2) * 4) as usize;
    [px[i], px[i + 1], px[i + 2]]
}

/// The CPU side of the same pipeline: exposure, the curve, then the sRGB OETF.
fn on_the_cpu(linear: [f32; 3], mode: ToneMapping, exposure: f32) -> [u8; 3] {
    let exposed = match mode {
        ToneMapping::None => linear,
        _ => [
            linear[0] * exposure,
            linear[1] * exposure,
            linear[2] * exposure,
        ],
    };
    let mapped = mode.apply(exposed);
    mapped.map(|v| {
        let v = v.clamp(0.0, 1.0);
        let s = if v <= 0.0031308 {
            v * 12.92
        } else {
            1.055 * v.powf(1.0 / 2.4) - 0.055
        };
        (s * 255.0 + 0.5) as u8
    })
}

#[test]
fn the_cpu_curves_match_the_ones_the_shader_runs() {
    // Dark, mid, bright-but-under-one, and well over one. The last is the only
    // place the curves differ much from each other, and the only place a
    // missing pre-exposure shows up.
    let samples: [[f32; 3]; 5] = [
        [0.02, 0.02, 0.02],
        [0.18, 0.18, 0.18],
        [0.75, 0.40, 0.10],
        [1.00, 1.00, 1.00],
        [4.00, 0.60, 0.20],
    ];
    let Ok(mut r) = HeadlessRenderer::builder()
        .size(W, W)
        .color_format(threers::wgpu::TextureFormat::Rgba8Unorm)
        .build()
    else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    let mut checked = 0;
    for (mode, name) in MODES {
        for exposure in [1.0f32, 0.5] {
            for c in samples {
                let gpu = through_the_gpu(&mut r, c, mode, exposure);
                let cpu = on_the_cpu(c, mode, exposure);
                for k in 0..3 {
                    let d = gpu[k] as i32 - cpu[k] as i32;
                    assert!(
                        d.abs() <= 2,
                        "{name} at exposure {exposure} on {c:?}: channel {k} is \
                         {} on the GPU and {} on the CPU",
                        gpu[k],
                        cpu[k]
                    );
                }
                checked += 1;
            }
        }
    }
    assert_eq!(checked, MODES.len() * 2 * 5);
}

/// The curves have to be different from one another, or the test above passes
/// by comparing two copies of the same thing.
#[test]
fn the_modes_are_actually_different_curves() {
    // Over-bright and saturated: this is the case the modes exist to disagree
    // about.
    let c = [4.0f32, 0.6, 0.2];
    let out: Vec<[u8; 3]> = MODES.iter().map(|(m, _)| on_the_cpu(c, *m, 1.0)).collect();
    for i in 0..out.len() {
        for j in i + 1..out.len() {
            if MODES[i].0 == ToneMapping::None && MODES[j].0 == ToneMapping::Linear {
                // At exposure 1 these two genuinely are the same thing.
                continue;
            }
            assert_ne!(
                out[i], out[j],
                "{} and {} produce identical output",
                MODES[i].1, MODES[j].1
            );
        }
    }
}

/// What `Neutral` is for, measured rather than assumed.
///
/// The first version of this test asserted that ACES shifts a saturated red
/// towards orange and Neutral does not. That is true of the full ACES RRT+ODT
/// and false of the Narkowicz fit everyone actually ships, which is per-channel
/// and has no cross-channel term at all — so it was ACES that held the hue and
/// Neutral that washed it out, which is the opposite of what the doc comment
/// claimed. These are the two properties that do separate them.
#[test]
fn neutral_leaves_the_midtones_alone_and_aces_does_not() {
    // Below the shoulder, Neutral subtracts a small black offset and otherwise
    // passes the value straight through.
    for x in [0.18f32, 0.35, 0.50] {
        let n = ToneMapping::Neutral.apply([x, x, x])[0];
        assert!(
            (n - (x - 0.04)).abs() < 0.01,
            "Neutral moved {x} to {n}; it should be about {}",
            x - 0.04
        );
        let a = ToneMapping::AcesFilmic.apply([x, x, x])[0];
        assert!(
            a > x * 1.15,
            "ACES was expected to lift the midtones, {x} came back as {a}"
        );
    }
}

#[test]
fn neutral_desaturates_a_blown_highlight_and_aces_keeps_the_ratio() {
    let red = [6.0f32, 0.05, 0.05];
    let aces = on_the_cpu(red, ToneMapping::AcesFilmic, 1.0);
    let neutral = on_the_cpu(red, ToneMapping::Neutral, 1.0);
    let ratio = |p: [u8; 3]| p[1] as f32 / p[0].max(1) as f32;

    // Per-channel, so a blown red stays a blown red.
    assert!(
        ratio(aces) < 0.35,
        "ACES is per-channel and should hold the ratio, got {aces:?}"
    );
    // Cross-channel, so it resolves towards white the way an over-exposure does.
    assert!(
        ratio(neutral) > ratio(aces) * 2.0,
        "Neutral should desaturate the highlight: {neutral:?} against {aces:?}"
    );
}

/// A half-float target must keep values above one, or there is nothing for the
/// post chain to bloom. `framebuffer_encode` used to clamp on that path, which
/// made an HDR render target behave like an 8-bit one.
#[test]
fn a_half_float_target_keeps_values_above_one() {
    let Some(mut r) = HeadlessRenderer::builder()
        .size(W, W)
        .color_format(threers::wgpu::TextureFormat::Rgba16Float)
        .build()
        .ok()
    else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    // No curve, so whatever the material writes is what the target should hold.
    r.renderer().set_tone_mapping(ToneMapping::None, 1.0);

    let material = ShaderMaterial::new(
        "@fragment\n\
         fn fs_main(in: VsOut) -> @location(0) vec4<f32> {\n\
         return vec4<f32>(framebuffer_encode(vec3<f32>(7.0, 0.5, 0.25)), 1.0);\n\
         }\n",
    );
    let mut scene = Scene::new();
    scene.background = Color::from_hex(0x000000);
    scene.add(Object3D::mesh(Mesh::new(
        BoxGeometry::new(4.0, 4.0, 0.1),
        Material::Shader(material),
    )));
    let mut cam = PerspectiveCamera::new(45.0, 1.0, 0.1, 100.0);
    cam.position = Vector3::new(0.0, 0.0, 3.0);
    cam.look_at(Vector3::ZERO);
    scene.update_world();

    // The 8-bit readback cannot show above one, so this reads the float target
    // directly: saturated white is what a clamp would leave behind, and it is
    // indistinguishable from a correct 1.0.
    let px = r.render_to_rgba(&mut scene, &cam);
    let i = ((W / 2 * W + W / 2) * 4) as usize;
    assert!(
        px[i] >= 254 && px[i + 1] < 250,
        "expected a clipped-red readback from an over-bright pixel, got \
         ({}, {}, {})",
        px[i],
        px[i + 1],
        px[i + 2]
    );
}

/// Bloom, ambient occlusion and the film grade have to reach the RGBA readback
/// as well as the RGB one.
///
/// They live in the compute pack that feeds a video encoder, and for a long
/// time asking for them and then reading RGBA — which is what every headless
/// example does — got a frame with none of them applied and no warning. A
/// setter that returns quietly and does nothing is the worst shape a setting
/// has.
#[test]
fn bloom_reaches_the_rgba_readback() {
    let frame = |bloom: bool| -> Option<Vec<u8>> {
        let mut r = HeadlessRenderer::builder()
            .size(64, 64)
            .color_format(threers::wgpu::TextureFormat::Rgba8Unorm)
            .build()
            .ok()?;
        r.renderer().set_tone_mapping(ToneMapping::None, 1.0);
        if bloom {
            r.set_bloom(2.5, 0.2, 1.0);
        }
        // A small bright quad on black: with bloom its light spreads outwards,
        // and the corner is far enough away to be untouched without it.
        let material = ShaderMaterial::new(
            "@fragment\n\
             fn fs_main(in: VsOut) -> @location(0) vec4<f32> {\n\
             return vec4<f32>(4.0, 4.0, 4.0, 1.0);\n\
             }\n",
        );
        let mut scene = Scene::new();
        scene.background = Color::from_hex(0x000000);
        scene.add(Object3D::mesh(Mesh::new(
            BoxGeometry::new(0.7, 0.7, 0.1),
            Material::Shader(material),
        )));
        let mut cam = PerspectiveCamera::new(45.0, 1.0, 0.1, 100.0);
        cam.position = Vector3::new(0.0, 0.0, 3.0);
        cam.look_at(Vector3::ZERO);
        scene.update_world();
        Some(r.render_to_rgba_resolved(&mut scene, &cam))
    };

    let Some(plain) = frame(false) else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    let bloomed = frame(true).expect("second render");
    assert_eq!(plain.len(), bloomed.len(), "the two paths disagree on size");

    // Everything outside the quad, which is black without bloom.
    let halo = |px: &[u8]| -> u32 {
        px.chunks_exact(4)
            .filter(|p| p[0] < 200)
            .map(|p| p[0] as u32)
            .sum()
    };
    let (a, b) = (halo(&plain), halo(&bloomed));
    assert!(
        b > a + 2000,
        "bloom added {} to the surround; it is not reaching this readback",
        b.saturating_sub(a)
    );
}

/// Depth of field blurs what is off the focus plane and leaves what is on it.
///
/// One quad at a time, because a fourteen-pixel blur on a ninety-six-pixel
/// frame spreads a long way: the first version of this put both quads in one
/// render and split the frame down the middle, and the far one's blur crossed
/// the line and was measured as a fringe on the near one.
#[test]
fn depth_of_field_blurs_the_far_plane_and_not_the_near_one() {
    // `z` is where the quad sits; the camera is at +3 and focus is on 3 units
    // away, which is the origin.
    let frame = |z: f32, size: f32, dof: bool| -> Option<Vec<u8>> {
        let mut r = HeadlessRenderer::builder()
            .size(96, 96)
            .color_format(threers::wgpu::TextureFormat::Rgba8Unorm)
            .build()
            .ok()?;
        r.renderer().set_tone_mapping(ToneMapping::None, 1.0);
        let (near, far) = (0.1f32, 100.0);
        if dof {
            r.set_depth_of_field(3.0, 1.2, 14.0);
        }
        let mut scene = Scene::new();
        scene.background = Color::from_hex(0x000000);
        let mut obj = Object3D::mesh(Mesh::new(
            BoxGeometry::new(size, size, 0.1),
            Material::Shader(ShaderMaterial::new(
                "@fragment\n\
                 fn fs_main(in: VsOut) -> @location(0) vec4<f32> {\n\
                 return vec4<f32>(1.0, 1.0, 1.0, 1.0);\n\
                 }\n",
            )),
        ));
        obj.position = Vector3::new(0.0, 0.0, z);
        scene.add(obj);
        let mut cam = PerspectiveCamera::new(45.0, 1.0, near, far);
        cam.position = Vector3::new(0.0, 0.0, 3.0);
        cam.look_at(Vector3::ZERO);
        scene.update_world();
        Some(r.render_to_rgba_resolved(&mut scene, &cam))
    };

    // How wide the transition is, as the count of pixels that are neither
    // background nor subject. A sharp edge has almost none; a blurred one is
    // mostly transition. Measured this way rather than as a sum of neighbour
    // differences, which counts sampling noise as detail and went *up* when the
    // first version of the gather was undersampled.
    let soft = |px: &[u8]| -> u32 {
        px.chunks_exact(4).filter(|p| (24..232).contains(&p[0])).count() as u32
    };

    // Six units behind the focus plane: fully defocused.
    let Some(far_sharp) = frame(-6.0, 2.0, false) else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    let far_blur = frame(-6.0, 2.0, true).expect("render");
    assert!(
        soft(&far_blur) > soft(&far_sharp) * 3 + 200,
        "the far quad kept its edge: {} soft pixels against {}",
        soft(&far_blur),
        soft(&far_sharp)
    );

    // And on the focus plane, where turning it on must change almost nothing.
    let near_sharp = frame(0.0, 0.7, false).expect("render");
    let near_blur = frame(0.0, 0.7, true).expect("render");
    assert!(
        soft(&near_blur) <= soft(&near_sharp) + 24,
        "the quad in focus was blurred too: {} against {}",
        soft(&near_blur),
        soft(&near_sharp)
    );
}

/// The post passes take the camera's clip planes from the frame, not from the
/// caller.
///
/// They used to be arguments with a doc comment admitting that passing the
/// wrong ones was "wrong silently" — and the one caller in the tree was passing
/// the wrong ones: `examples/simcity_render.rs` computed its field of view with
/// `camera.fov.to_radians()`, and `fov` is already radians.
///
/// The check reads oddly until you see what it rests on. Two cameras differing
/// only in their far plane put a subject at the same *distance* but at a
/// different *depth-buffer value*. A pass that linearises with the frame's own
/// planes recovers the same distance from both and blurs them identically; a
/// pass using a constant recovers two different distances and does not. So
/// agreement is the pass being right, and the second half is what stops two
/// blank frames agreeing by default.
#[test]
fn depth_of_field_reads_the_clip_planes_from_the_frame() {
    let frame = |far: f32, focus: f32| -> Option<Vec<u8>> {
        let mut r = HeadlessRenderer::builder()
            .size(64, 64)
            .color_format(threers::wgpu::TextureFormat::Rgba8Unorm)
            .build()
            .ok()?;
        r.renderer().set_tone_mapping(ToneMapping::None, 1.0);
        r.set_depth_of_field(focus, 1.5, 12.0);
        let mut scene = Scene::new();
        scene.background = Color::from_hex(0x000000);
        let mut obj = Object3D::mesh(Mesh::new(
            BoxGeometry::new(2.0, 2.0, 0.1),
            Material::Shader(ShaderMaterial::new(
                "@fragment\n\
                 fn fs_main(in: VsOut) -> @location(0) vec4<f32> {\n\
                 return vec4<f32>(1.0, 1.0, 1.0, 1.0);\n\
                 }\n",
            )),
        ));
        obj.position = Vector3::new(0.0, 0.0, -3.0);
        scene.add(obj);
        let mut cam = PerspectiveCamera::new(45.0, 1.0, 0.1, far);
        cam.position = Vector3::new(0.0, 0.0, 3.0);
        cam.look_at(Vector3::ZERO);
        scene.update_world();
        Some(r.render_to_rgba_resolved(&mut scene, &cam))
    };

    // The subject is six units out; focus on it, so it is near the transition
    // and a mis-linearised distance moves it.
    let Some(short) = frame(20.0, 6.0) else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    let long = frame(400.0, 6.0).expect("render");
    let lit = short.chunks_exact(4).filter(|p| p[0] > 40).count();
    assert!(lit > 200, "the subject did not draw: {lit} lit pixels");

    let differing = |a: &[u8], b: &[u8]| {
        a.chunks_exact(4)
            .zip(b.chunks_exact(4))
            .filter(|(x, y)| (x[0] as i32 - y[0] as i32).abs() > 6)
            .count()
    };
    let across_far_planes = differing(&short, &long);
    assert!(
        across_far_planes < 24,
        "a twentyfold change of far plane moved {across_far_planes} pixels; the \
         pass is linearising with something other than the frame's own planes"
    );

    // And the control: the same camera with the focus moved has to change a
    // great many more, or the comparison above is two identical blurs agreeing
    // about nothing.
    let refocused = frame(20.0, 1.2).expect("render");
    let across_focus = differing(&short, &refocused);
    assert!(
        across_focus > 400,
        "moving the focus changed only {across_focus} pixels, so the first \
         comparison proves nothing"
    );
}
