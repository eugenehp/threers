//! What can be asserted about a simulation whose entire state lives on the GPU.
//!
//! The pure functions — the grid sizing, the hue wheel, the generator — are
//! checked directly. For the solver itself the interesting properties are
//! invariants rather than values: that the projection actually removes
//! divergence, that dissipation actually dissipates, that a splat lands where
//! it was asked for (which is the only check on the texture-space convention
//! the whole port rests on), and that nothing reaches a NaN. All four read the
//! fields back, which is what the `COPY_SRC` on every target is for.

use super::*;

// -- the pure parts ------------------------------------------------------

#[test]
fn the_short_side_of_a_grid_is_the_resolution() {
    // Landscape: the resolution is the height, and the width follows aspect.
    assert_eq!(grid_size(128, 1920, 1080), (228, 128));
    // Portrait: the other way round, with the same ratio.
    assert_eq!(grid_size(128, 1080, 1920), (128, 228));
    // Square stays square.
    assert_eq!(grid_size(128, 512, 512), (128, 128));
    // A degenerate canvas must not produce a zero-sized texture, which wgpu
    // refuses outright.
    let (w, h) = grid_size(128, 0, 0);
    assert!(w >= 1 && h >= 1);
}

#[test]
fn the_hue_wheel_is_fully_saturated_everywhere() {
    for i in 0..60 {
        let h = i as f32 / 60.0;
        let c = hsv_to_rgb(h, 1.0, 1.0);
        let max = c.iter().copied().fold(f32::MIN, f32::max);
        let min = c.iter().copied().fold(f32::MAX, f32::min);
        // Value 1, saturation 1: the brightest channel is 1 and the dimmest 0.
        assert!((max - 1.0).abs() < 1e-5, "hue {h}: max {max}");
        assert!(min.abs() < 1e-5, "hue {h}: min {min}");
    }
    // The wrap at 1.0 lands back on the start rather than falling off the
    // `match`, which an `i as usize` index would do.
    assert_eq!(hsv_to_rgb(1.0, 1.0, 1.0), hsv_to_rgb(0.0, 1.0, 1.0));
}

#[test]
fn the_generator_repeats_from_a_seed_and_stays_in_range() {
    let draw = || {
        let mut rng = Rng::new(7);
        (0..512).map(|_| rng.next_f32()).collect::<Vec<_>>()
    };
    assert_eq!(draw(), draw(), "same seed, different numbers");
    let xs = draw();
    assert!(xs.iter().all(|x| (0.0..1.0).contains(x)));
    // A generator that is merely in range could still be constant.
    let mean = xs.iter().sum::<f32>() / xs.len() as f32;
    assert!((mean - 0.5).abs() < 0.05, "mean {mean}");
}

#[test]
fn quality_levels_only_go_up() {
    let mut last: Option<&FluidQuality> = None;
    for q in QUALITY_LEVELS {
        if let Some(prev) = last {
            assert!(
                q.sim_resolution > prev.sim_resolution,
                "{} vs {}",
                q.name,
                prev.name
            );
            assert!(
                q.dye_resolution > prev.dye_resolution,
                "{} vs {}",
                q.name,
                prev.name
            );
            assert!(
                q.pressure_iterations > prev.pressure_iterations,
                "{}",
                q.name
            );
            assert!(q.bloom_iterations >= prev.bloom_iterations, "{}", q.name);
        }
        last = Some(q);
    }
    // `high` is the original's defaults, and `FluidConfig::default` has to
    // agree with it or the two ways of asking for the same thing diverge.
    let high = quality("high").expect("a level called high");
    let default = FluidConfig::default();
    assert_eq!(default.sim_resolution, high.sim_resolution);
    assert_eq!(default.dye_resolution, high.dye_resolution);
    assert_eq!(default.pressure_iterations, high.pressure_iterations);
    assert_eq!(default.bloom_resolution, high.bloom_resolution);
    assert_eq!(default.sunrays_resolution, high.sunrays_resolution);
    assert_eq!(FluidConfig::default().with_quality(high), default);
    assert!(quality("enormous").is_none());
    assert_eq!(quality_names().count(), QUALITY_LEVELS.len());
}

/// Asking for a grid the device cannot allocate has to give back the largest
/// one it can *at the same shape*: scaling one axis alone would stretch every
/// vortex, which is worse than the resolution being lower than requested.
#[test]
fn an_oversized_grid_is_scaled_down_whole() {
    // 4096 short side on a 32:9 canvas wants 14563 on the long one.
    let (w, h) = grid_size_limited(4096, 3840, 1080, 8192);
    assert!(w <= 8192 && h <= 8192, "{w}x{h} is over the limit");
    let asked = grid_size(4096, 3840, 1080);
    let aspect_asked = asked.0 as f32 / asked.1 as f32;
    let aspect_got = w as f32 / h as f32;
    assert!(
        (aspect_asked - aspect_got).abs() < 0.01,
        "the clamp changed the shape: {aspect_asked} -> {aspect_got}"
    );
    // Under the limit it is the plain sizing, untouched.
    assert_eq!(
        grid_size_limited(128, 1920, 1080, 8192),
        grid_size(128, 1920, 1080)
    );
    // And a limit smaller than everything still leaves a usable texture.
    let (w, h) = grid_size_limited(4096, 3840, 1080, 4);
    assert!(w >= 1 && h >= 1 && w <= 4 && h <= 4, "{w}x{h}");
}

// -- the GPU parts -------------------------------------------------------

/// A device and the formats this adapter will carry the fields in, or `None`
/// after printing why — `gpu::testing` is the shared version of the bail-out
/// every GPU test in this crate needs.
fn device() -> Option<(Arc<wgpu::Device>, Arc<wgpu::Queue>, FluidFormats)> {
    let gpu = crate::gpu::testing::context("fluid")?;
    let formats = FluidFormats::for_adapter(&gpu.adapter)?;
    Some((gpu.device.clone(), gpu.queue.clone(), formats))
}

fn sim(width: u32, height: u32, config: FluidConfig) -> Option<FluidSim> {
    let (device, queue, formats) = device()?;
    FluidSim::with_formats(device, queue, width, height, config, formats).ok()
}

/// Read one channel of a half-float field back as `f32`.
fn read_channel(sim: &FluidSim, fbo: &Fbo, channel: usize) -> Vec<f32> {
    crate::renderer::readback::texture_channel_f32(&sim.device, &sim.queue, &fbo.texture, channel)
}

/// The mean |divergence| of the velocity field as it stands right now.
///
/// Runs the divergence pass on demand rather than reading the texture the last
/// step happened to leave: that one holds the divergence of a *mid-step* field,
/// taken before the projection, which is exactly the number that cannot answer
/// whether the projection worked.
fn divergence_now(sim: &mut FluidSim) -> f32 {
    let mut enc = sim.encoder("test divergence");
    sim.record_divergence(&mut enc);
    sim.queue.submit([enc.finish()]);
    mean_abs(&read_channel(sim, &sim.divergence, 0))
}

fn mean_abs(xs: &[f32]) -> f32 {
    xs.iter().map(|x| x.abs()).sum::<f32>() / xs.len() as f32
}

/// The brightest pixel of an RGBA8 frame, as a fraction of the way across and
/// down it.
fn brightest(rgba: &[u8], width: u32, height: u32) -> (f32, f32, u8) {
    let mut best = (0u32, 0u32, 0u8);
    for y in 0..height {
        for x in 0..width {
            let at = ((y * width + x) * 4) as usize;
            let v = rgba[at].max(rgba[at + 1]).max(rgba[at + 2]);
            if v > best.2 {
                best = (x, y, v);
            }
        }
    }
    (
        best.0 as f32 / width as f32,
        best.1 as f32 / height as f32,
        best.2,
    )
}

/// The one test that pins the texture-space convention the whole port rests
/// on: a splat asked for a quarter of the way across and a quarter of the way
/// *down* has to appear in the top left, not the bottom left. Get the
/// handedness wrong and everything still runs, looks like a fluid, and is
/// upside down.
#[test]
fn a_splat_lands_where_it_was_asked_for() {
    let config = FluidConfig {
        // No post-processing: bloom and sunrays spread the brightest pixel
        // around and this test is about where it is.
        bloom: false,
        sunrays: false,
        shading: false,
        ..FluidConfig::default()
    };
    let Some(mut sim) = sim(256, 256, config) else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    // Dye only, no velocity, so nothing carries it away from where it landed.
    sim.splat(0.25, 0.25, 0.0, 0.0, [0.6, 0.6, 0.6]);
    sim.update(1.0 / 60.0);

    let rgba = sim.capture_rgba(256, 256);
    let (x, y, v) = brightest(&rgba, 256, 256);
    assert!(v > 32, "nothing was drawn (brightest {v})");
    assert!((x - 0.25).abs() < 0.05, "x {x}");
    assert!((y - 0.25).abs() < 0.05, "y {y}");
}

/// How much of a splat's divergence the projection removes, both as shipped
/// and with the solve run closer to convergence.
///
/// The step is short on purpose. This solver projects and *then* advects, and
/// semi-Lagrangian advection of a divergence-free field does not leave it
/// divergence-free — so at a full frame's dt the two effects are the same size
/// and a measurement after the step says nothing about either.
#[test]
fn projection_removes_the_divergence_a_splat_creates() {
    let residual = |config: FluidConfig| -> Option<f32> {
        let mut sim = sim(256, 256, config)?;
        // A radial puff of velocity: pure divergence, and the most
        // compressible thing the simulation can be handed.
        sim.splat(0.5, 0.5, 400.0, 0.0, [0.3, 0.3, 0.3]);
        let raw = divergence_now(&mut sim);
        assert!(
            raw > 1e-4,
            "the splat produced no divergence to remove: {raw}"
        );
        sim.step(1.0 / 6000.0);
        Some(divergence_now(&mut sim) / raw)
    };
    let shipped = FluidConfig {
        // The confinement force is not divergence-free either, and this is a
        // measurement of the projection alone.
        curl: 0.0,
        ..FluidConfig::default()
    };
    let converged = FluidConfig {
        pressure_iterations: 320,
        gradient_scale: 0.5,
        ..shipped
    };
    let (Some(shipped), Some(converged)) = (residual(shipped), residual(converged)) else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    assert!(
        shipped < 0.75,
        "the default projection left {shipped:.3} of the divergence"
    );
    assert!(
        converged < 0.45,
        "a converged projection left {converged:.3} of the divergence"
    );
}

/// The one finding in this port that is worth a test of its own: the 0.5 the
/// original leaves off the pressure gradient is an over-relaxation that pays
/// for an unconverged solve, and it stops paying once the solve converges.
///
/// Both halves are asserted, because either alone would argue for the wrong
/// default. At twenty sweeps, doubling the correction removes more divergence
/// than the consistent coefficient does; at three hundred and twenty it removes
/// *less than nothing*, handing back a field more compressible than the one it
/// was given.
#[test]
fn the_missing_half_compensates_for_an_unconverged_solve() {
    let residual = |iterations: u32, scale: f32| -> Option<f32> {
        let mut sim = sim(
            256,
            256,
            FluidConfig {
                pressure_iterations: iterations,
                gradient_scale: scale,
                curl: 0.0,
                ..FluidConfig::default()
            },
        )?;
        sim.splat(0.5, 0.5, 400.0, 0.0, [0.3, 0.3, 0.3]);
        let raw = divergence_now(&mut sim);
        sim.step(1.0 / 6000.0);
        Some(divergence_now(&mut sim) / raw)
    };
    let (Some(few_half), Some(few_whole), Some(many_half), Some(many_whole)) = (
        residual(20, 0.5),
        residual(20, 1.0),
        residual(320, 0.5),
        residual(320, 1.0),
    ) else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    eprintln!("  20 sweeps: 0.5 -> {few_half:.3}, 1.0 -> {few_whole:.3}");
    eprintln!(" 320 sweeps: 0.5 -> {many_half:.3}, 1.0 -> {many_whole:.3}");
    assert!(
        few_whole < few_half,
        "at 20 sweeps the overshoot was expected to help: 1.0 left {few_whole:.3}, 0.5 left {few_half:.3}"
    );
    assert!(
        many_half < many_whole,
        "at 320 sweeps the consistent coefficient was expected to win: 0.5 left {many_half:.3}, 1.0 left {many_whole:.3}"
    );
    assert!(
        many_whole > 1.0,
        "at 320 sweeps the overshoot was expected to add divergence, but left {many_whole:.3}"
    );
}

/// Jacobi sweeps have to buy something, and with the consistent coefficient
/// they monotonically do. With the default one they stop doing so past about
/// eighty, which is the test above.
#[test]
fn more_sweeps_leave_less_divergence() {
    let run = |iterations: u32| -> Option<f32> {
        let mut sim = sim(
            256,
            256,
            FluidConfig {
                pressure_iterations: iterations,
                gradient_scale: 0.5,
                curl: 0.0,
                ..FluidConfig::default()
            },
        )?;
        sim.splat(0.5, 0.5, 400.0, 0.0, [0.3, 0.3, 0.3]);
        sim.step(1.0 / 6000.0);
        Some(divergence_now(&mut sim))
    };
    let (Some(few), Some(many)) = (run(1), run(80)) else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    assert!(
        many < few,
        "80 sweeps ({many:e}) did not beat 1 sweep ({few:e})"
    );
}

/// Dissipation is specified per second, not per frame, so the same elapsed
/// time has to fade the dye by the same amount whatever the frame rate. That
/// is the property that lets a tab drop to 20 fps without the picture changing
/// speed, and it only holds because the decay is `1 + dissipation * dt`.
#[test]
fn dye_fades_by_elapsed_time_not_by_frame_count() {
    let total = |steps: u32, dt: f32| -> Option<f32> {
        let mut sim = sim(
            256,
            256,
            FluidConfig {
                // Still dye: advection would move it around and the comparison
                // is about how much is left, not where.
                curl: 0.0,
                density_dissipation: 1.0,
                ..FluidConfig::default()
            },
        )?;
        sim.splat(0.5, 0.5, 0.0, 0.0, [0.5, 0.5, 0.5]);
        for _ in 0..steps {
            sim.step(dt);
        }
        let dye = read_channel(&sim, &sim.dye.read, 0);
        Some(dye.iter().sum::<f32>())
    };
    let (Some(fast), Some(slow), Some(none)) =
        (total(60, 1.0 / 60.0), total(20, 1.0 / 20.0), total(0, 0.0))
    else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    assert!(none > 0.0, "no dye was splatted");
    assert!(fast < none * 0.75, "a second of dissipation did nothing");
    // Both ran one second of simulated time. The decay is applied once per
    // step, so the two are not identical — but they are close, where a
    // per-frame decay would differ by 3x.
    let ratio = fast / slow;
    assert!(
        (0.8..1.25).contains(&ratio),
        "60 small steps and 20 large ones disagree by {ratio:.2}x"
    );
}

/// The velocity clamp in the vorticity pass is the only thing standing between
/// a half-float field and an infinity that advection then smears over the whole
/// canvas. This drives it hard enough to find out.
#[test]
fn a_hard_driven_field_stays_finite() {
    let Some(mut sim) = sim(
        512,
        256,
        FluidConfig {
            // Everything turned up: maximum confinement, no dissipation to
            // take energy back out, and a coarse grid so the forces are large
            // in texel terms.
            curl: 50.0,
            velocity_dissipation: 0.0,
            density_dissipation: 0.0,
            sim_resolution: 64,
            ..FluidConfig::default()
        },
    ) else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    for i in 0..40 {
        let t = i as f32 / 40.0;
        sim.splat(t, 1.0 - t, 3000.0, -3000.0, [1.0, 1.0, 1.0]);
        sim.update(1.0 / 60.0);
    }
    for _ in 0..200 {
        sim.update(1.0 / 60.0);
    }
    for channel in 0..2 {
        let v = read_channel(&sim, &sim.velocity.read, channel);
        assert!(
            v.iter().all(|x| x.is_finite()),
            "velocity channel {channel} went non-finite"
        );
        // The shader clamps to +-1000; half float would reach 65504 before
        // saturating, so a value past the clamp means the clamp was skipped.
        let worst = v.iter().copied().fold(0.0f32, |a, b| a.max(b.abs()));
        assert!(worst <= 1001.0, "velocity reached {worst}");
    }
    let dye = read_channel(&sim, &sim.dye.read, 0);
    assert!(dye.iter().all(|x| x.is_finite()), "dye went non-finite");
}

/// Same seed, same picture. Without this the headless example would drift and
/// the docs image would have to be regenerated every time anything moved.
#[test]
fn the_same_seed_renders_the_same_frame() {
    let run = || -> Option<Vec<u8>> {
        let mut sim = sim(192, 128, FluidConfig::default())?;
        sim.multiple_splats(12);
        for _ in 0..30 {
            sim.update(1.0 / 60.0);
        }
        Some(sim.capture_rgba(192, 128))
    };
    let (Some(a), Some(b)) = (run(), run()) else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    assert_eq!(a, b, "two runs of the same seed differ");
    // And a different seed does not accidentally give the same thing.
    let Some(mut other) = sim(
        192,
        128,
        FluidConfig {
            seed: 99,
            ..FluidConfig::default()
        },
    ) else {
        return;
    };
    other.multiple_splats(12);
    for _ in 0..30 {
        other.update(1.0 / 60.0);
    }
    assert_ne!(a, other.capture_rgba(192, 128), "the seed changed nothing");
}

/// A resize must not throw the fluid away: the grids are rebuilt at the new
/// aspect and the fields resampled into them.
#[test]
fn a_resize_keeps_the_fluid() {
    let Some(mut sim) = sim(256, 256, FluidConfig::default()) else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    sim.splat(0.5, 0.5, 0.0, 0.0, [0.8, 0.8, 0.8]);
    sim.update(1.0 / 60.0);
    let before = read_channel(&sim, &sim.dye.read, 0).iter().sum::<f32>();

    sim.resize(512, 256);
    assert_eq!(sim.size(), (512, 256));
    // Twice as many texels across, so the sum roughly doubles; what matters is
    // that the dye is still there at all.
    let after = read_channel(&sim, &sim.dye.read, 0).iter().sum::<f32>();
    assert!(
        after > before * 0.5,
        "the resize lost the dye ({before} -> {after})"
    );

    // And the new grids follow the new aspect.
    let (sw, sh) = (sim.velocity.width(), sim.velocity.height());
    assert!(sw > sh, "sim grid did not follow the canvas ({sw}x{sh})");
    sim.update(1.0 / 60.0);
}

/// Turning a display option on or off must not disturb the simulation, which
/// is what lets a UI toggle them mid-stroke.
#[test]
fn display_options_do_not_touch_the_fields() {
    let Some(mut sim) = sim(256, 256, FluidConfig::default()) else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    sim.multiple_splats(6);
    for _ in 0..10 {
        sim.update(1.0 / 60.0);
    }
    let before = read_channel(&sim, &sim.velocity.read, 0);
    sim.set_config(FluidConfig {
        bloom: false,
        sunrays: false,
        shading: false,
        ..*sim.config()
    });
    let after = read_channel(&sim, &sim.velocity.read, 0);
    assert_eq!(
        before, after,
        "a display toggle reallocated the velocity field"
    );

    // Changing a *resolution*, on the other hand, is expected to rebuild.
    sim.set_config(FluidConfig {
        sim_resolution: 64,
        ..*sim.config()
    });
    assert_eq!(sim.velocity.width().min(sim.velocity.height()), 64);
}

/// `pressure_iterations: 0` would leave the solve unentered and the pressure
/// field untouched, which is a silently wrong fluid rather than an error.
#[test]
fn zero_sweeps_is_refused() {
    let Some((device, queue, formats)) = device() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    let built = FluidSim::with_formats(
        device,
        queue,
        64,
        64,
        FluidConfig {
            pressure_iterations: 0,
            ..FluidConfig::default()
        },
        formats,
    );
    let Err(err) = built else {
        panic!("a simulation with no pressure sweeps was accepted");
    };
    assert!(err.contains("pressure_iterations"), "{err}");
}

/// The top quality level asks for grids a device may not have, and the answer
/// has to be a smaller fluid rather than a panic inside `create_texture`.
#[test]
fn the_top_quality_level_runs_or_degrades() {
    let extreme = quality("extreme").expect("a level called extreme");
    // Deliberately wide: at 32:9 the long side of a 4096 grid is 14563, which
    // is past what most devices will allocate.
    let Some(mut sim) = sim(3840, 1080, FluidConfig::default().with_quality(extreme)) else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    let max = sim.max_dim;
    let ((dw, dh), (sw, sh)) = sim.grid_sizes();
    assert!(
        dw <= max && dh <= max,
        "dye grid {dw}x{dh} over the limit {max}"
    );
    assert!(
        sw <= max && sh <= max,
        "sim grid {sw}x{sh} over the limit {max}"
    );
    eprintln!(
        "extreme on 3840x1080: dye {dw}x{dh}, sim {sw}x{sh}, {} MB",
        sim.memory_bytes() / (1024 * 1024)
    );
    // And it still simulates and composites.
    sim.multiple_splats(4);
    sim.update(1.0 / 60.0);
    let rgba = sim.capture_rgba(320, 90);
    assert!(
        rgba.iter().any(|&b| b > 8),
        "the top level rendered nothing"
    );
}

/// The dye, used as a material map on scene geometry.
///
/// This is the integration that makes the fluid part of threers rather than a
/// demo beside it, and it needs no new API: the fluid renders into a
/// [`RenderTarget`](crate::renderer::RenderTarget) like anything else, the
/// renderer registers that target under an id, and
/// [`Texture::from_render_target`](crate::textures::Texture::from_render_target)
/// turns the id into something a material can hold.
///
/// One thing has to line up for the colours to survive. The fluid's display
/// pass hands back *display-referred* values, and `register_render_target`
/// samples an 8-bit target through its **sRGB** view — so the target's format
/// must be the non-sRGB one. Then the fluid writes the bytes untouched, the
/// renderer decodes them to linear to shade with, and they come back out the
/// far end as they went in. Give it an sRGB target instead and the encode
/// happens twice.
#[test]
fn the_dye_can_be_a_material_map() {
    use crate::renderer::RenderTarget;
    use crate::textures::Texture;
    use crate::{
        BasicMaterial, Color, HeadlessRenderer, Mesh, Object3D, PerspectiveCamera, PlaneGeometry,
        Scene, TextureFormat, Vector3,
    };

    let Ok(mut hr) = HeadlessRenderer::builder().size(256, 256).build() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    let (device, queue) = (hr.device().clone(), hr.queue().clone());

    // Non-sRGB, deliberately: see the note above.
    let rt = Arc::new(RenderTarget::new(
        &device,
        256,
        256,
        wgpu::TextureFormat::Rgba8Unorm,
    ));
    let Ok(mut sim) = FluidSim::new(device, queue, 256, 256, FluidConfig::default()) else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    sim.multiple_splats(14);
    for _ in 0..16 {
        sim.update(1.0 / 60.0);
    }
    sim.render_capture(&rt.color_texture);

    const RT_ID: u32 = 4242;
    hr.renderer().register_render_target(RT_ID, &rt);
    let mut material = BasicMaterial::new(Color::WHITE);
    material.map = Some(Arc::new(Texture::from_render_target(
        RT_ID,
        256,
        256,
        TextureFormat::Rgba8Unorm,
    )));

    let mut scene = Scene::new();
    scene.background = Color::BLACK;
    scene.add(Object3D::mesh(Mesh::new(
        PlaneGeometry::new(2.0, 2.0),
        material.into(),
    )));
    // Close enough that the plane fills the frame, so every pixel sampled is
    // dye rather than background.
    let mut camera = PerspectiveCamera::new(60.0, 1.0, 0.1, 10.0);
    camera.position = Vector3::new(0.0, 0.0, 1.0);
    camera.look_at(Vector3::ZERO);

    let rgba = hr.render_to_rgba(&mut scene, &camera);

    // A missing texture gives a flat frame — the material's base colour, or
    // black. What proves the dye arrived is *hue*: several pixels whose
    // channels disagree strongly, which no flat fill can produce.
    let saturated = rgba
        .chunks_exact(4)
        .filter(|p| {
            let (hi, lo) = (p[0].max(p[1]).max(p[2]), p[0].min(p[1]).min(p[2]));
            hi > 40 && hi - lo > 40
        })
        .count();
    assert!(
        saturated > 500,
        "only {saturated} pixels carry a hue; the dye did not reach the material"
    );
}

/// The fluid as a pass in the post-processing chain, which is the other way
/// to put it in a frame: not a texture on a surface, but a layer over the
/// scene the composer has already drawn.
///
/// What this exercises is [`DrawPass`](crate::postprocessing::DrawPass) — the
/// composer's other passes are *descriptions* it compiles and runs, which
/// covers a filter over the previous buffer and nothing that owns GPU state of
/// its own. A simulation is the case that needs a hook.
#[test]
fn the_fluid_can_be_a_pass_in_the_composer() {
    use crate::postprocessing::{DrawPass, EffectComposer, RenderPass};
    use crate::renderer::{readback, RenderTarget};

    let Some(gpu) = crate::gpu::testing::context("fluid as a pass") else {
        return;
    };
    let format = wgpu::TextureFormat::Rgba8Unorm;
    let (w, h) = (128u32, 128u32);

    let mut sim = FluidSim::new(
        gpu.device.clone(),
        gpu.queue.clone(),
        w,
        h,
        FluidConfig {
            // Over whatever the scene left, rather than onto a flat colour of
            // its own — which is what makes this a layer and not a replacement.
            transparent: true,
            ..FluidConfig::default()
        },
    )
    .expect("fluid");
    sim.multiple_splats(10);
    for _ in 0..12 {
        sim.update(1.0 / 60.0);
    }

    let mut renderer = crate::Renderer::new(gpu.device.clone(), gpu.queue.clone(), format, w, h);
    let mut composer = EffectComposer::new(gpu.device.clone(), gpu.queue.clone(), w, h, format);
    composer.add_pass(RenderPass::new());
    composer.add_pass(DrawPass::new("fluid", move |ctx| {
        // The composer hands over the view the next pass will read, its format
        // and its size — everything `render_to_view` needs and nothing it has
        // to be told twice.
        sim.update(1.0 / 60.0);
        sim.render_into(ctx.target(), true);
    }));

    let out = RenderTarget::new(&gpu.device, w, h, format);
    let mut scene = crate::Scene::new();
    scene.background = crate::Color::BLACK;
    let mut camera = crate::PerspectiveCamera::new(50.0, 1.0, 0.1, 10.0);
    camera.position = crate::Vector3::new(0.0, 0.0, 3.0);
    composer.render(&mut renderer, &mut scene, &camera, &out.color_view);

    let pixels = readback::texture_bytes(&gpu.device, &gpu.queue, &out.color_texture);
    let coloured = pixels
        .chunks_exact(4)
        .filter(|p| {
            let (hi, lo) = (p[0].max(p[1]).max(p[2]), p[0].min(p[1]).min(p[2]));
            hi > 40 && hi - lo > 30
        })
        .count();
    assert!(
        coloured > 200,
        "only {coloured} pixels carry dye; the pass did not reach the final frame"
    );
}

/// What the multigrid solver is worth, against the Jacobi one it replaces.
///
/// A fluid is incompressible or it is not, and twenty Jacobi sweeps leaves
/// eighty-two per cent of a splat's divergence in the field. The number does
/// come down — but as a *rate*, not towards a floor: sixty-four times the
/// sweeps reaches twenty-four per cent, because a Jacobi sweep moves
/// information one texel and the pressure from a localised puff is spread
/// across the whole grid.
///
/// A V-cycle solves that part of the problem on a grid eight texels across,
/// where "spread across the whole grid" is a handful of neighbours.
#[test]
fn multigrid_beats_sixty_four_times_the_jacobi_sweeps() {
    let residual = |solver: PressureSolver, iterations: u32, cycles: u32| -> Option<f32> {
        let mut sim = sim(
            256,
            256,
            FluidConfig {
                pressure_solver: solver,
                pressure_iterations: iterations,
                multigrid_cycles: cycles,
                // The consistent coefficient, so this measures the solver and
                // not the compensating overshoot the default carries.
                gradient_scale: 0.5,
                // The confinement force is not divergence-free either.
                curl: 0.0,
                ..FluidConfig::default()
            },
        )?;
        sim.splat(0.5, 0.5, 400.0, 0.0, [0.3, 0.3, 0.3]);
        let raw = divergence_now(&mut sim);
        // dt = 0: advection's lookup lands exactly on texel centres, so it is
        // the identity, and what is measured is the projection alone.
        sim.step(0.0);
        Some(divergence_now(&mut sim) / raw)
    };
    let (Some(j20), Some(j1280), Some(mg1), Some(mg2)) = (
        residual(PressureSolver::Jacobi, 20, 1),
        residual(PressureSolver::Jacobi, 1280, 1),
        residual(PressureSolver::Multigrid, 20, 1),
        residual(PressureSolver::Multigrid, 20, 2),
    ) else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    eprintln!("  jacobi    20 sweeps: {j20:.3}");
    eprintln!("  jacobi  1280 sweeps: {j1280:.3}");
    eprintln!("  multigrid  1 cycle : {mg1:.3}");
    eprintln!("  multigrid  2 cycles: {mg2:.3}");

    // One cycle, at about two to three times the work of the default twenty
    // sweeps, beats sixty-four times them.
    assert!(
        mg1 < j1280,
        "one V-cycle left {mg1:.3} against 1280 Jacobi sweeps' {j1280:.3}"
    );
    assert!(
        mg1 < j20 * 0.5,
        "one V-cycle left {mg1:.3} against the default's {j20:.3}"
    );
    // And a second cycle is worth having, which is what says the cycle is
    // converging rather than hitting a limit of its own.
    assert!(
        mg2 < mg1 * 0.6,
        "a second cycle took {mg1:.3} only to {mg2:.3}"
    );
    assert!(mg2 < 0.15, "two cycles still leave {mg2:.3}");
}

/// Switching solver must not switch fluids: the dye still lands where it was
/// put, nothing reaches a NaN, and a hard-driven field stays bounded.
#[test]
fn the_multigrid_solver_is_still_a_fluid() {
    let Some(mut sim) = sim(
        256,
        256,
        FluidConfig {
            pressure_solver: PressureSolver::Multigrid,
            bloom: false,
            sunrays: false,
            shading: false,
            ..FluidConfig::default()
        },
    ) else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    sim.splat(0.25, 0.25, 0.0, 0.0, [0.6, 0.6, 0.6]);
    sim.update(1.0 / 60.0);
    let (x, y, v) = brightest(&sim.capture_rgba(256, 256), 256, 256);
    assert!(v > 32, "nothing was drawn (brightest {v})");
    assert!((x - 0.25).abs() < 0.05, "x {x}");
    assert!((y - 0.25).abs() < 0.05, "y {y}");

    for i in 0..60 {
        let t = i as f32 / 60.0;
        sim.splat(t, 1.0 - t, 2500.0, -2500.0, [1.0, 1.0, 1.0]);
        sim.update(1.0 / 60.0);
    }
    for channel in 0..2 {
        let field = read_channel(&sim, &sim.velocity.read, channel);
        assert!(
            field.iter().all(|x| x.is_finite()),
            "velocity channel {channel} went non-finite under the multigrid solver"
        );
        // Bounded, but not by the 1000 the vorticity pass clamps to: that
        // clamp is three passes from the end, and both the projection and the
        // advection touch the field after it. A better-converged projection
        // legitimately moves more energy around than a worse one, so this
        // checks the field has not run away rather than that it obeys a bound
        // it was never given.
        let worst = field.iter().copied().fold(0.0f32, |a, b| a.max(b.abs()));
        assert!(worst < 1.0e4, "velocity reached {worst}");
    }
}

/// A grid too small to coarsen falls back to sweeping rather than failing.
#[test]
fn a_grid_too_small_to_coarsen_still_solves() {
    let Some(mut sim) = sim(
        64,
        64,
        FluidConfig {
            pressure_solver: PressureSolver::Multigrid,
            sim_resolution: 8,
            ..FluidConfig::default()
        },
    ) else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    assert!(
        sim.mg.len() <= 1,
        "an 8-texel grid built {} levels",
        sim.mg.len()
    );
    sim.multiple_splats(4);
    sim.update(1.0 / 60.0);
    let field = read_channel(&sim, &sim.velocity.read, 0);
    assert!(field.iter().all(|x| x.is_finite()));
}

/// What MacCormack advection buys, under a flow that is identical for both.
///
/// The flow is a uniform translation, and that is the point: a constant
/// velocity field advects itself exactly, has zero divergence so the
/// projection does nothing, and zero curl so confinement does nothing. Both
/// schemes therefore carry the dye at exactly the same speed, and every
/// difference in what arrives is the scheme's own error.
///
/// Semi-Lagrangian's error is diffusion — the backward trace lands between
/// texels and the bilinear fetch averages them, every step — so after a
/// hundred steps its blob is lower and wider. That is the blur MacCormack
/// exists to remove, and it is most of why a long-running fluid goes muddy.
#[test]
fn maccormack_advection_diffuses_less() {
    let run = |advection: Advection| -> Option<(f32, f32)> {
        let mut sim = sim(
            256,
            256,
            FluidConfig {
                advection,
                // A blob a few texels across, which is where a first-order
                // scheme's diffusion actually shows. A wide smooth one is
                // carried almost exactly by either — bilinear interpolation
                // of something already smooth loses very little — and
                // comparing on one says nothing.
                dye_resolution: 256,
                splat_radius: 0.02,
                // Nothing may fade the dye or change the flow: the only thing
                // allowed to act on the blob is the advection itself.
                density_dissipation: 0.0,
                velocity_dissipation: 0.0,
                curl: 0.0,
                ..FluidConfig::default()
            },
        )?;
        sim.splat(0.25, 0.5, 0.0, 0.0, [1.0, 1.0, 1.0]);
        // Half a texel a step is the worst case for a bilinear fetch, and so
        // the fairest place to compare: a whole-texel step would interpolate
        // nothing and both schemes would be exact.
        sim.set_uniform_velocity(30.0, 0.0);
        for _ in 0..100 {
            sim.step(1.0 / 60.0);
        }
        let dye = read_channel(&sim, &sim.dye.read, 0);
        let peak = dye.iter().copied().fold(0.0f32, f32::max);
        let mass = dye.iter().sum::<f32>();
        Some((peak, mass))
    };
    let (Some((sl_peak, sl_mass)), Some((mc_peak, mc_mass))) =
        (run(Advection::SemiLagrangian), run(Advection::MacCormack))
    else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    eprintln!("  semi-Lagrangian: peak {sl_peak:.4}, mass {sl_mass:.1}");
    eprintln!("  MacCormack:      peak {mc_peak:.4}, mass {mc_mass:.1}");

    assert!(sl_peak > 0.0, "the dye vanished");
    // The blob keeps its height: diffusion is what flattens it, and this is
    // the whole claim.
    assert!(
        mc_peak > sl_peak * 1.3,
        "MacCormack held {mc_peak:.4} against semi-Lagrangian's {sl_peak:.4} \
         — not the sharper blob it is for"
    );
    // And it does not do it by inventing dye: the limiter clamps the
    // correction to the range of the texels the forward pass read, so the
    // total can fall (diffusion off the edge) but must not climb.
    assert!(
        mc_mass < sl_mass * 1.1,
        "MacCormack carries {mc_mass:.1} against {sl_mass:.1} — the limiter \
         is not holding"
    );
}

/// The plane draws its read-out too, and only where it says.
///
/// The same two claims as the volume's `the_read_out_draws_only_where_it_says`,
/// against the other solver — the glyphs are shared but the entry points and
/// the uniform are not, and a read-out wired to the wrong field draws a
/// confident wrong number.
#[test]
fn the_plane_draws_its_read_out_too() {
    let (w, h) = (128u32, 128u32);
    let render = |show: bool| -> Option<Vec<u8>> {
        let mut sim = sim(
            w,
            h,
            FluidConfig {
                show_fps: show,
                bloom: false,
                sunrays: false,
                ..FluidConfig::default()
            },
        )?;
        sim.multiple_splats(4);
        for _ in 0..4 {
            sim.update(1.0 / 60.0);
            // Twice, so there is a rate to report rather than a zero.
            sim.capture_rgba(w, h);
        }
        Some(sim.capture_rgba(w, h))
    };
    let Some(plain) = render(false) else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    let with = render(true).expect("a second simulation");

    // The read-out's two parts: three digits at 3x5 scaled by three, ten in
    // from each edge, and the chart under them — 64 samples two pixels wide,
    // 18 tall, starting at y = 29. Anything else must be untouched.
    let in_read_out = |x: u32, y: u32| {
        let digits = x < 10 + 12 * 3 && y < 10 + 5 * 3;
        let chart = (10..10 + 64 * 2).contains(&x) && (29..29 + 18).contains(&y);
        digits || chart
    };
    let (mut inside, mut outside) = (0usize, 0usize);
    for y in 0..h {
        for x in 0..w {
            let i = ((y * w + x) * 4) as usize;
            if plain[i..i + 4] == with[i..i + 4] {
                continue;
            }
            if in_read_out(x, y) {
                inside += 1;
            } else {
                outside += 1;
            }
        }
    }
    eprintln!("  the plane's read-out changed {inside} pixels inside its box, {outside} outside");
    assert!(inside > 20, "the read-out drew {inside} pixels");
    assert_eq!(outside, 0, "it changed {outside} pixels outside its box");
}

/// The read-out shows the number it was given.
///
/// Checked against values chosen here rather than against the clock,
/// because the interesting cases are the boundaries — one digit, two, three
/// — and a clock will not visit them on request. The first version of this
/// read-out was only ever seen at 133 fps, and the leading-zero suppression
/// was wrong below a hundred without anyone noticing.
#[test]
fn the_read_out_shows_the_number_it_was_given() {
    let (w, h) = (96u32, 48u32);
    let Some(mut sim) = sim(
        w,
        h,
        FluidConfig {
            show_fps: true,
            bloom: false,
            sunrays: false,
            transparent: false,
            back_color: [0.0; 3],
            ..FluidConfig::default()
        },
    ) else {
        eprintln!("skipping: no GPU adapter");
        return;
    };

    // Read the glyph grid back: 3x5 cells at scale 3, 10 in from the edge.
    let read = |px: &[u8]| -> String {
        let lit = |cx: usize, cy: usize| -> bool {
            let (x, y) = (10 + cx * 3 + 1, 10 + cy * 3 + 1);
            px[(y * w as usize + x) * 4] > 120
        };
        let shapes: [(&str, [[bool; 3]; 5]); 11] = [
            (" ", [[false; 3]; 5]),
            (
                "0",
                [
                    [true, true, true],
                    [true, false, true],
                    [true, false, true],
                    [true, false, true],
                    [true, true, true],
                ],
            ),
            (
                "1",
                [
                    [false, true, false],
                    [true, true, false],
                    [false, true, false],
                    [false, true, false],
                    [true, true, true],
                ],
            ),
            (
                "2",
                [
                    [true, true, true],
                    [false, false, true],
                    [true, true, true],
                    [true, false, false],
                    [true, true, true],
                ],
            ),
            (
                "3",
                [
                    [true, true, true],
                    [false, false, true],
                    [true, true, true],
                    [false, false, true],
                    [true, true, true],
                ],
            ),
            (
                "4",
                [
                    [true, false, true],
                    [true, false, true],
                    [true, true, true],
                    [false, false, true],
                    [false, false, true],
                ],
            ),
            (
                "5",
                [
                    [true, true, true],
                    [true, false, false],
                    [true, true, true],
                    [false, false, true],
                    [true, true, true],
                ],
            ),
            (
                "6",
                [
                    [true, true, true],
                    [true, false, false],
                    [true, true, true],
                    [true, false, true],
                    [true, true, true],
                ],
            ),
            (
                "7",
                [
                    [true, true, true],
                    [false, false, true],
                    [false, false, true],
                    [false, false, true],
                    [false, false, true],
                ],
            ),
            (
                "8",
                [
                    [true, true, true],
                    [true, false, true],
                    [true, true, true],
                    [true, false, true],
                    [true, true, true],
                ],
            ),
            (
                "9",
                [
                    [true, true, true],
                    [true, false, true],
                    [true, true, true],
                    [false, false, true],
                    [true, true, true],
                ],
            ),
        ];
        (0..3)
            .map(|col| {
                let seen: [[bool; 3]; 5] =
                    std::array::from_fn(|y| std::array::from_fn(|x| lit(col * 4 + x, y)));
                shapes
                    .iter()
                    .find(|(_, want)| *want == seen)
                    .map(|(name, _)| *name)
                    .unwrap_or("?")
            })
            .collect()
    };

    for (value, want) in [
        (7.0f32, "  7"),
        (42.0, " 42"),
        (99.0, " 99"),
        (100.0, "100"),
        (999.0, "999"),
    ] {
        sim.set_fps_for_test(value);
        let px = sim.capture_rgba(w, h);
        let got = read(&px);
        eprintln!("  {value:>5} shows {got:?}");
        assert_eq!(got, want, "the read-out showed {got:?} for {value}");
    }
}

/// The chart under the number shows the frame times it was given.
///
/// A read-out that draws something is not the same as a read-out that draws
/// the right thing — the first version of this one sampled the *dye* as its
/// history, because the overlay pass inherited the dye binding from the
/// pass beside it, and drew a confident chart of nothing at all.
#[test]
fn the_chart_shows_the_frame_times_it_was_given() {
    let (w, h) = (160u32, 64u32);
    let Some(mut sim) = sim(
        w,
        h,
        FluidConfig {
            show_fps: true,
            bloom: false,
            sunrays: false,
            transparent: false,
            back_color: [0.0; 3],
            ..FluidConfig::default()
        },
    ) else {
        eprintln!("skipping: no GPU adapter");
        return;
    };

    // How tall the bar is in a given column, in pixels, 0 if there is none.
    let bar = |px: &[u8], column: usize| -> usize {
        let x = 10 + column * 2;
        (0..18)
            .filter(|row| {
                let y = 29 + row;
                let i = (y * w as usize + x) * 4;
                // The target line is faint; a bar is not.
                px[i] > 60 || px[i + 2] > 60
            })
            .count()
    };

    // A flat history at the target, then one with a spike in it.
    sim.set_frame_times_for_test(&[1000.0 / 60.0; 64]);
    let flat = sim.capture_rgba(w, h);
    let mut spiky = [1000.0f32 / 60.0; 64];
    spiky[40] = 1000.0 / 10.0; // a tenth of a second: a visible hitch
    sim.set_frame_times_for_test(&spiky);
    let hitched = sim.capture_rgba(w, h);

    let steady: Vec<usize> = (0..64).map(|c| bar(&flat, c)).collect();
    let spiked: Vec<usize> = (0..64).map(|c| bar(&hitched, c)).collect();
    eprintln!(
        "  flat: bars {}..{} tall; with a hitch at 40: column 40 is {} and its neighbour {}",
        steady.iter().min().unwrap(),
        steady.iter().max().unwrap(),
        spiked[40],
        spiked[39]
    );
    // At the target, every bar reaches half height and no further.
    assert!(
        steady.iter().all(|t| (7..=10).contains(t)),
        "a steady {:.1} ms should sit at half of eighteen: {steady:?}",
        1000.0 / 60.0
    );
    // The hitch is taller, and only where it happened.
    assert!(
        spiked[40] > spiked[39] + 4,
        "the hitch did not show: {} against {}",
        spiked[40],
        spiked[39]
    );
    assert_eq!(
        spiked[39], steady[39],
        "a hitch in one frame changed another column"
    );
}
