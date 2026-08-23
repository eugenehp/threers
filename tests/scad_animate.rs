//! Animating OpenSCAD models: the `$t` loop, static-model caching, concurrent
//! evaluation, camera framing, and the render/export path.
//!
//! The GPU tests skip themselves when no adapter is available.
#![cfg(feature = "openscad")]

use threers::openscad::animate::{
    animation_bounds, ScadAnimation, ScadCamera, ScadKernel, ScadRender,
};
use threers::openscad::scad::scad_viewport;

/// A cube whose side grows with `$t`, so every frame differs measurably.
const GROWING: &str = "cube(10 + 40 * $t, center = true);";

fn size_of(frame: &threers::openscad::animate::ScadFrame) -> f32 {
    let (min, max) = frame.bounds().expect("frame has geometry");
    max[0] - min[0]
}

// ---------------------------------------------------------------------------
// The `$t` loop
// ---------------------------------------------------------------------------

#[test]
fn frames_step_t_across_the_loop() {
    let a = ScadAnimation::from_source(GROWING).frames(4);
    // Looping (the default) never reaches 1, so frame 0 of the next loop lines
    // up with frame 0 of this one.
    assert_eq!(a.t_at(0), 0.0);
    assert_eq!(a.t_at(1), 0.25);
    assert_eq!(a.t_at(3), 0.75);

    let once = ScadAnimation::from_source(GROWING).frames(4).looping(false);
    assert_eq!(once.t_at(0), 0.0);
    assert_eq!(once.t_at(3), 1.0);
}

#[test]
fn geometry_actually_changes_with_t() {
    let a = ScadAnimation::from_source(GROWING).frames(4);
    let first = size_of(&a.frame(0).unwrap());
    let last = size_of(&a.frame(3).unwrap());
    assert!((first - 10.0).abs() < 0.01, "t=0 → side 10, got {first}");
    assert!((last - 40.0).abs() < 0.01, "t=0.75 → side 40, got {last}");
}

#[test]
fn a_static_model_is_detected_and_shared_across_frames() {
    let mut still = ScadAnimation::from_source("cube(10);").frames(5);
    assert!(!still.is_animated());
    let frames = still.evaluate().unwrap();
    assert_eq!(frames.len(), 5);
    // One evaluation, aliased by every frame — not five copies.
    for f in &frames[1..] {
        assert!(
            std::sync::Arc::ptr_eq(&frames[0].parts, &f.parts),
            "a static model must be evaluated once"
        );
    }

    let mut moving = ScadAnimation::from_source(GROWING).frames(3);
    assert!(moving.is_animated());
    let frames = moving.evaluate().unwrap();
    assert!(!std::sync::Arc::ptr_eq(&frames[0].parts, &frames[1].parts));
}

#[test]
fn seeded_variables_drive_the_model_and_count_as_animated() {
    let mut a = ScadAnimation::from_source("cube([10, 10, HEIGHT]);")
        .frames(3)
        .var("HEIGHT", |t| 5.0 + 20.0 * t);
    assert!(
        a.is_animated(),
        "a seeded variable makes the model animated"
    );
    let frames = a.evaluate().unwrap();
    let z = |i: usize| frames[i].bounds().unwrap().1[2];
    assert!((z(0) - 5.0).abs() < 0.01, "{}", z(0));
    assert!(z(2) > z(0));

    // A model that mentions neither `$t` nor a seeded name is still static.
    let plain = ScadAnimation::from_source("cube([10, 10, 5]);")
        .frames(3)
        .var("HEIGHT", |t| t);
    assert!(!plain.is_animated());
}

#[test]
fn a_variable_name_must_match_whole_identifiers() {
    // `HEIGHTS` is a different name and must not make this look animated.
    let a = ScadAnimation::from_source("MAX_HEIGHTS = 3; cube(10);")
        .frames(2)
        .var("HEIGHT", |t| t);
    assert!(!a.is_animated(), "substring match should not count");
}

#[test]
fn closures_animate_the_rust_dsl() {
    use threers::{cube, sphere};
    let a = ScadAnimation::from_fn(|t| {
        cube([20.0, 20.0, 20.0]).difference(sphere(4.0 + 8.0 * t as f32))
    })
    .frames(3);
    assert!(a.is_animated(), "a closure is always treated as animated");
    assert!(a.frame(0).unwrap().triangle_count() > 0);
    assert!(a.frame(2).unwrap().triangle_count() > 0);
}

#[test]
fn evaluation_errors_surface_with_the_scad_message() {
    let a = ScadAnimation::from_source("this is not scad(((").frames(1);
    assert!(a.frame(0).is_err());
    let missing = ScadAnimation::from_file("no/such/file.scad").frames(1);
    let err = missing.frame(0).unwrap_err();
    assert!(err.contains("no/such/file.scad"), "{err}");
}

// ---------------------------------------------------------------------------
// Concurrency
// ---------------------------------------------------------------------------

#[test]
fn concurrent_evaluation_matches_serial_frame_for_frame() {
    let mut serial = ScadAnimation::from_source(GROWING).frames(8).concurrency(1);
    let mut concurrent = ScadAnimation::from_source(GROWING).frames(8).concurrency(4);
    let a = serial.evaluate().unwrap();
    let b = concurrent.evaluate().unwrap();

    assert_eq!(a.len(), b.len());
    for (x, y) in a.iter().zip(b.iter()) {
        assert_eq!(x.index, y.index, "frames must come back in order");
        assert_eq!(x.t, y.t);
        assert_eq!(x.triangle_count(), y.triangle_count());
        assert_eq!(x.bounds(), y.bounds());
    }
}

#[test]
fn progress_is_reported_once_per_frame() {
    use std::sync::{Arc, Mutex};
    let seen = Arc::new(Mutex::new(Vec::<(usize, usize)>::new()));
    let sink = seen.clone();
    let mut a = ScadAnimation::from_source(GROWING)
        .frames(6)
        .concurrency(3)
        .on_progress(move |p| sink.lock().unwrap().push((p.done, p.total)));
    a.evaluate().unwrap();

    let seen = seen.lock().unwrap().clone();
    assert_eq!(seen.len(), 6, "one report per frame: {seen:?}");
    assert!(seen.iter().all(|(_, total)| *total == 6));
    // `done` counts up monotonically even though frames finish out of order.
    let counts: Vec<usize> = seen.iter().map(|(d, _)| *d).collect();
    assert_eq!(counts, vec![1, 2, 3, 4, 5, 6], "{counts:?}");
}

#[test]
fn an_error_in_one_frame_fails_the_whole_evaluation() {
    // `assert()` fires only at the last frame, so most workers succeed.
    let mut a = ScadAnimation::from_source("assert($t < 0.6); cube(10);")
        .frames(5)
        .concurrency(3);
    assert!(a.evaluate().is_err());
}

// ---------------------------------------------------------------------------
// Viewport and camera
// ---------------------------------------------------------------------------

#[test]
fn viewport_variables_are_read_from_the_model() {
    let vp = scad_viewport(
        "$vpd = 250; $vpr = [60, 0, 360 * $t]; $vpt = [1, 2, 3]; cube(1);",
        0.5,
    );
    assert!(vp.explicit);
    assert_eq!(vp.distance, 250.0);
    assert_eq!(vp.rotation, [60.0, 0.0, 180.0]);
    assert_eq!(vp.target, [1.0, 2.0, 3.0]);

    // A model that sets none reports the OpenSCAD defaults, flagged inexplicit.
    let none = scad_viewport("cube(1);", 0.0);
    assert!(!none.explicit);
    assert_eq!(none.distance, 140.0);
}

#[test]
fn the_animation_exposes_the_models_viewport_per_frame() {
    let a = ScadAnimation::from_source("$vpd = 100 + 100 * $t; cube(10);").frames(4);
    assert_eq!(a.viewport_at(0.0).distance, 100.0);
    assert_eq!(a.viewport_at(0.5).distance, 150.0);
    assert_eq!(a.frame(2).unwrap().viewport.distance, 150.0);
}

#[test]
fn animation_bounds_cover_every_frame() {
    let mut a = ScadAnimation::from_source(GROWING).frames(4);
    let frames = a.evaluate().unwrap();
    let (min, max) = animation_bounds(&frames);
    // The largest frame is a 40 cube centred on the origin.
    assert!((max[0] - 20.0).abs() < 0.01, "{max:?}");
    assert!((min[0] + 20.0).abs() < 0.01, "{min:?}");
}

#[test]
fn the_camera_frames_the_model_and_keeps_it_in_front_of_the_near_plane() {
    use threers::cameras::Camera;
    let a = ScadAnimation::from_source("cylinder(h = 60, r = 8, $fn = 24);").frames(1);
    let frame = a.frame(0).unwrap();
    let fit = frame.bounds().unwrap();
    let render = ScadRender::new(640, 360).camera(ScadCamera::auto());
    let camera = render.build_camera(&frame, fit);

    // Every corner of the model must sit between the near and far planes and
    // inside the frustum — a near plane grazing the geometry slices the front
    // off the model, which is exactly the bug this guards.
    let view_proj = camera.projection_matrix().multiply(&camera.view_matrix());
    let e = &view_proj.elements;
    for i in 0..8 {
        let (px, py, pz) = (
            if i & 1 == 0 { fit.0[0] } else { fit.1[0] },
            if i & 2 == 0 { fit.0[1] } else { fit.1[1] },
            if i & 4 == 0 { fit.0[2] } else { fit.1[2] },
        );
        // Column-major, so w is the fourth row.
        let w = e[3] * px + e[7] * py + e[11] * pz + e[15];
        assert!(w > 0.0, "corner {i} is behind the camera");
        let x = (e[0] * px + e[4] * py + e[8] * pz + e[12]) / w;
        let y = (e[1] * px + e[5] * py + e[9] * pz + e[13]) / w;
        let z = (e[2] * px + e[6] * py + e[10] * pz + e[14]) / w;
        assert!(
            z > 0.0 && z < 1.0,
            "corner {i} outside the depth range: {z}"
        );
        assert!(x.abs() <= 1.0, "corner {i} off screen horizontally: {x}");
        assert!(y.abs() <= 1.0, "corner {i} off screen vertically: {y}");
    }
}

#[test]
fn the_camera_is_z_up_like_openscad() {
    let a = ScadAnimation::from_source("cylinder(h = 60, r = 4, $fn = 12);").frames(1);
    let frame = a.frame(0).unwrap();
    let camera = ScadRender::new(640, 360).build_camera(&frame, frame.bounds().unwrap());
    assert_eq!(camera.up, threers::Vector3::new(0.0, 0.0, 1.0));
}

#[test]
fn a_turntable_orbits_but_holds_its_distance() {
    let mut a = ScadAnimation::from_source("cube([40, 10, 10], center = true);").frames(4);
    let frames = a.evaluate().unwrap();
    let fit = animation_bounds(&frames);
    let render = ScadRender::new(640, 360).camera(ScadCamera::turntable());

    let cameras: Vec<_> = frames.iter().map(|f| render.build_camera(f, fit)).collect();
    let distance = |c: &threers::PerspectiveCamera| (c.position - c.target).length();
    let d0 = distance(&cameras[0]);
    for c in &cameras[1..] {
        assert!(
            (distance(c) - d0).abs() < 1e-3,
            "an orbit must not change distance, or the model breathes"
        );
    }
    // …and the eye actually moves.
    assert!((cameras[0].position - cameras[2].position).length() > 1.0);
}

#[test]
fn a_fixed_camera_is_used_verbatim() {
    let a = ScadAnimation::from_source("cube(10);").frames(1);
    let frame = a.frame(0).unwrap();
    let render = ScadRender::new(640, 360).camera(ScadCamera::Fixed {
        eye: [100.0, 0.0, 0.0],
        target: [0.0, 0.0, 5.0],
        fov: 30.0,
    });
    let camera = render.build_camera(&frame, frame.bounds().unwrap());
    assert_eq!(camera.position, threers::Vector3::new(100.0, 0.0, 0.0));
    assert_eq!(camera.target, threers::Vector3::new(0.0, 0.0, 5.0));
}

#[test]
fn zoom_moves_the_camera_closer() {
    let a = ScadAnimation::from_source("cube(10);").frames(1);
    let frame = a.frame(0).unwrap();
    let fit = frame.bounds().unwrap();
    let far = ScadRender::new(640, 360)
        .camera(ScadCamera::auto())
        .build_camera(&frame, fit);
    let near = ScadRender::new(640, 360)
        .camera(ScadCamera::auto().zoomed(0.5))
        .build_camera(&frame, fit);
    let d = |c: &threers::PerspectiveCamera| (c.position - c.target).length();
    assert!(d(&near) < d(&far) * 0.6, "{} vs {}", d(&near), d(&far));
}

// ---------------------------------------------------------------------------
// Scene building
// ---------------------------------------------------------------------------

#[test]
fn each_colored_part_becomes_its_own_mesh() {
    let a = ScadAnimation::from_source(
        "color(\"red\") cube(10); color(\"blue\") translate([20, 0, 0]) cube(10);",
    )
    .frames(1);
    let frame = a.frame(0).unwrap();
    assert_eq!(frame.parts.len(), 2);
    let render = ScadRender::new(320, 240);
    let camera = render.build_camera(&frame, frame.bounds().unwrap());
    let scene = render.build_scene(&frame, &camera);
    let meshes = scene
        .arena
        .nodes
        .values()
        .filter(|o| matches!(o.kind, threers::ObjectKind::Mesh(_)))
        .count();
    assert_eq!(meshes, 2);
}

#[test]
fn the_float_kernel_falls_back_rather_than_rendering_nothing() {
    // A model the float `CsgEvaluator` gives up on; the animation must still
    // produce geometry by re-running it through the robust kernel.
    let src = std::fs::read_to_string("examples/scad_animate.scad").expect("demo model");
    let plain = threers::parse_scad_at(&src, 0.0).unwrap();
    if !plain.parts_float().is_empty() {
        eprintln!("skipping: the float kernel now handles this model");
        return;
    }
    let a = ScadAnimation::from_source(src)
        .frames(1)
        .kernel(ScadKernel::Float);
    let frame = a.frame(0).unwrap();
    assert!(
        frame.triangle_count() > 0,
        "an empty float result must fall back to the exact kernel"
    );
}

// ---------------------------------------------------------------------------
// Rendering (needs a GPU adapter)
// ---------------------------------------------------------------------------

/// Whether a headless renderer can be created here.
fn gpu_available() -> bool {
    threers::HeadlessRenderer::builder()
        .size(16, 16)
        .build()
        .is_ok()
}

#[test]
fn rendering_produces_frames_with_the_model_in_them() {
    if !gpu_available() {
        eprintln!("skipping: no GPU adapter");
        return;
    }
    let (w, h) = (240u32, 160u32);
    let mut a = ScadAnimation::from_source(GROWING).frames(3);
    let render = ScadRender::new(w, h)
        .camera(ScadCamera::auto())
        .background([0.0, 0.0, 0.0, 1.0]);
    let frames = render.render_frames(&mut a).unwrap();

    assert_eq!(frames.len(), 3);
    for (i, rgba) in frames.iter().enumerate() {
        assert_eq!(rgba.len(), (w * h * 4) as usize);
        let lit = rgba.chunks_exact(4).filter(|px| px[0] > 20).count();
        assert!(lit > 100, "frame {i} looks empty ({lit} lit pixels)");
    }
    // The model grows, so later frames cover more of a fixed-fit view.
    let lit = |rgba: &Vec<u8>| rgba.chunks_exact(4).filter(|px| px[0] > 20).count();
    assert!(lit(&frames[2]) > lit(&frames[0]), "the cube should grow");
}

#[test]
fn supersampling_returns_the_requested_size() {
    if !gpu_available() {
        eprintln!("skipping: no GPU adapter");
        return;
    }
    let (w, h) = (160u32, 120u32);
    let mut a = ScadAnimation::from_source("cube(10);").frames(1);
    let frames = ScadRender::new(w, h)
        .supersample(2)
        .render_frames(&mut a)
        .unwrap();
    assert_eq!(frames[0].len(), (w * h * 4) as usize);
}

#[test]
fn png_and_sequence_export_write_readable_files() {
    if !gpu_available() {
        eprintln!("skipping: no GPU adapter");
        return;
    }
    let dir = std::env::temp_dir().join(format!("threers-scad-anim-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();

    let mut a = ScadAnimation::from_source(GROWING).frames(3);
    let render = ScadRender::new(160, 120);

    let single = dir.join("still.png");
    render.render_png(&a, 0, &single).unwrap();
    let decoded = threers::decode_png(&std::fs::read(&single).unwrap()).expect("valid png");
    assert_eq!((decoded.width, decoded.height), (160, 120));

    let paths = render.export_png_sequence(&mut a, &dir, "frame").unwrap();
    assert_eq!(paths.len(), 3);
    assert!(paths[0].ends_with("frame0000.png"));
    assert!(paths[2].ends_with("frame0002.png"));
    for p in &paths {
        assert!(threers::decode_png(&std::fs::read(p).unwrap()).is_ok());
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[cfg(all(feature = "video", feature = "native-codec"))]
#[test]
fn video_export_picks_the_codec_from_the_extension() {
    if !gpu_available() {
        eprintln!("skipping: no GPU adapter");
        return;
    }
    let dir = std::env::temp_dir().join(format!("threers-scad-vid-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();

    let mut a = ScadAnimation::from_source(GROWING).frames(4).fps(8);
    let out = dir.join("anim.gif");
    ScadRender::new(120, 80)
        .export_video(&mut a, &out)
        .expect("gif export");
    let bytes = std::fs::read(&out).unwrap();
    assert_eq!(&bytes[..3], b"GIF");
    let (_, frames) = threers::decode_gif(&bytes).expect("decode");
    assert_eq!(frames.len(), 4);
    let _ = std::fs::remove_dir_all(&dir);
}

// ---------------------------------------------------------------------------
// Colors
// ---------------------------------------------------------------------------

use threers::openscad::animate::{
    ScadColoring, ScadEasing, ScadMaterial, ScadPalette, ScadQuality,
};

/// Two parts: one the model colored red, one it left alone.
fn mixed_parts() -> threers::openscad::animate::ScadFrame {
    ScadAnimation::from_source("color(\"red\") cube(10); translate([20, 0, 0]) cube(10);")
        .frames(1)
        .frame(0)
        .unwrap()
}

#[test]
fn every_built_in_palette_is_named_and_resolvable() {
    let all = ScadPalette::all();
    assert!(all.len() >= 8);
    for p in &all {
        assert_eq!(p.name, p.name.to_ascii_lowercase(), "names are lower-case");
        assert!(!p.cycle.is_empty(), "{} has no colors", p.name);
        assert_eq!(
            ScadPalette::by_name(&p.name).as_ref(),
            Some(p),
            "{} must round-trip through by_name",
            p.name
        );
        // Case-insensitive.
        assert!(ScadPalette::by_name(&p.name.to_uppercase()).is_some());
    }
    assert!(ScadPalette::by_name("no such scheme").is_none());
}

#[test]
fn a_palette_cycle_repeats() {
    let p = ScadPalette::from_hex("two", 0x000000, &[0xFF0000, 0x00FF00]);
    assert_eq!(p.color_at(0), [1.0, 0.0, 0.0, 1.0]);
    assert_eq!(p.color_at(1), [0.0, 1.0, 0.0, 1.0]);
    assert_eq!(p.color_at(2), p.color_at(0));
    assert_eq!(p.color_at(97), p.color_at(1));
    // An empty cycle still answers, with the fallback.
    let empty = p.clone().colors(vec![]);
    assert_eq!(empty.color_at(3), empty.default_part);
}

#[test]
fn coloring_modes_choose_different_sources() {
    let frame = mixed_parts();
    assert_eq!(frame.parts.len(), 2);
    let red = [1.0, 0.0, 0.0, 1.0];
    let palette = ScadPalette::from_hex("p", 0x000000, &[0x0000FF, 0x00FF00]);
    let blue = [0.0, 0.0, 1.0, 1.0];
    let green = [0.0, 1.0, 0.0, 1.0];
    let material = ScadMaterial::default().color([0.5, 0.5, 0.5, 1.0]);

    let render = |mode| {
        ScadRender::new(64, 64)
            .material(material)
            .palette(palette.clone())
            .coloring(mode)
    };

    // Model: color() wins, untagged falls back to the material.
    let r = render(ScadColoring::Model);
    assert_eq!(r.color_for(&frame.parts[0], 0), red);
    assert_eq!(r.color_for(&frame.parts[1], 1), material.color);

    // ModelThenPalette: color() wins, untagged takes a palette color.
    let r = render(ScadColoring::ModelThenPalette);
    assert_eq!(r.color_for(&frame.parts[0], 0), red);
    assert_eq!(r.color_for(&frame.parts[1], 1), green);

    // Palette: color() is ignored entirely.
    let r = render(ScadColoring::Palette);
    assert_eq!(r.color_for(&frame.parts[0], 0), blue);
    assert_eq!(r.color_for(&frame.parts[1], 1), green);

    // Uniform: one color for everything.
    let r = render(ScadColoring::Uniform);
    assert_eq!(r.color_for(&frame.parts[0], 0), material.color);
    assert_eq!(r.color_for(&frame.parts[1], 1), material.color);
}

#[test]
fn choosing_a_palette_also_sets_the_background() {
    let render = ScadRender::new(64, 64).palette(ScadPalette::cornfield());
    assert_eq!(render.palette_ref().name, "cornfield");
    // …and an explicit background afterwards still wins.
    let overridden = ScadRender::new(64, 64)
        .palette(ScadPalette::cornfield())
        .background([0.0, 0.0, 0.0, 1.0]);
    let frame = mixed_parts();
    let camera = overridden.build_camera(&frame, frame.bounds().unwrap());
    assert_eq!(
        overridden.build_scene(&frame, &camera).background,
        threers::Color::BLACK
    );
}

#[test]
fn palette_named_ignores_an_unknown_name() {
    let base = ScadRender::new(64, 64).palette(ScadPalette::sunset());
    let kept = ScadRender::new(64, 64)
        .palette(ScadPalette::sunset())
        .palette_named("not a scheme");
    assert_eq!(kept.palette_ref(), base.palette_ref());
    assert_eq!(
        ScadRender::new(64, 64)
            .palette_named("MIDNIGHT")
            .palette_ref()
            .name,
        "midnight"
    );
}

#[test]
fn part_colors_overrides_the_models_own() {
    let frame = mixed_parts();
    let render =
        ScadRender::new(64, 64).part_colors(vec![[0.1, 0.2, 0.3, 1.0], [0.4, 0.5, 0.6, 1.0]]);
    assert_eq!(render.coloring_mode(), ScadColoring::Palette);
    assert_eq!(render.color_for(&frame.parts[0], 0), [0.1, 0.2, 0.3, 1.0]);
    assert_eq!(render.color_for(&frame.parts[1], 1), [0.4, 0.5, 0.6, 1.0]);
}

#[test]
fn palettes_change_what_actually_gets_rendered() {
    if !gpu_available() {
        eprintln!("skipping: no GPU adapter");
        return;
    }
    let render_with = |p: ScadPalette| {
        let mut model =
            ScadAnimation::from_source("cube(10); translate([20,0,0]) cube(10);").frames(1);
        ScadRender::new(120, 90)
            .palette(p)
            .coloring(ScadColoring::Palette)
            .render_frames(&mut model)
            .unwrap()
            .remove(0)
    };
    let cornfield = render_with(ScadPalette::cornfield());
    let midnight = render_with(ScadPalette::midnight());
    assert_ne!(
        cornfield, midnight,
        "different palettes must look different"
    );

    // Cornfield's background is near-white, midnight's near-black.
    let mean = |rgba: &Vec<u8>| {
        rgba.chunks_exact(4).map(|p| p[0] as u64).sum::<u64>() / (rgba.len() / 4) as u64
    };
    assert!(mean(&cornfield) > mean(&midnight));
}

// ---------------------------------------------------------------------------
// Speed
// ---------------------------------------------------------------------------

#[test]
fn speed_changes_the_rate_not_the_frames() {
    let base = ScadAnimation::from_source(GROWING).frames(60).fps(30);
    let fast = ScadAnimation::from_source(GROWING)
        .frames(60)
        .fps(30)
        .speed(2.0);
    let slow = ScadAnimation::from_source(GROWING)
        .frames(60)
        .fps(30)
        .speed(0.5);

    assert_eq!(base.frame_rate(), 30);
    assert_eq!(fast.frame_rate(), 60);
    assert_eq!(slow.frame_rate(), 15);
    assert_eq!(base.duration(), 2.0);
    assert_eq!(fast.duration(), 1.0);
    assert_eq!(slow.duration(), 4.0);

    // Same frame count, and the same `$t` at each index — only playback differs.
    assert_eq!(fast.frame_count(), base.frame_count());
    for i in [0, 17, 59] {
        assert_eq!(fast.t_at(i), base.t_at(i));
    }
}

#[test]
fn a_nonsense_speed_falls_back_to_normal() {
    for bad in [0.0, -2.0, f64::NAN, f64::INFINITY] {
        let a = ScadAnimation::from_source("cube(1);")
            .frames(10)
            .fps(20)
            .speed(bad);
        assert_eq!(a.frame_rate(), 20, "speed {bad} should be ignored");
    }
}

#[test]
fn seconds_retimes_the_loop_in_either_order() {
    let a = ScadAnimation::from_source("cube(1);")
        .seconds(4.0)
        .frames(120);
    let b = ScadAnimation::from_source("cube(1);")
        .frames(120)
        .seconds(4.0);
    assert_eq!(a.frame_rate(), 30);
    assert_eq!(b.frame_rate(), 30);
    assert_eq!(a.duration(), 4.0);
    // Speed still multiplies on top of an explicit duration.
    let quick = ScadAnimation::from_source("cube(1);")
        .frames(120)
        .seconds(4.0)
        .speed(2.0);
    assert_eq!(quick.frame_rate(), 60);
    assert_eq!(quick.duration(), 2.0);
}

#[test]
fn ping_pong_runs_t_out_and_back() {
    let a = ScadAnimation::from_source(GROWING)
        .frames(8)
        .ping_pong(true);
    let ts: Vec<f64> = (0..8).map(|i| a.t_at(i)).collect();
    assert_eq!(ts[0], 0.0);
    assert_eq!(
        ts[4], 1.0,
        "the far end is halfway through the loop: {ts:?}"
    );
    assert_eq!(ts[7], 0.25);
    // Rises then falls, and never leaves the range.
    assert!(ts.windows(2).take(4).all(|w| w[1] > w[0]), "{ts:?}");
    assert!(ts.windows(2).skip(4).all(|w| w[1] < w[0]), "{ts:?}");
    assert!(ts.iter().all(|t| (0.0..=1.0).contains(t)));
}

#[test]
fn easing_reshapes_the_motion_but_keeps_the_ends() {
    let curve = |e: ScadEasing| {
        let a = ScadAnimation::from_source(GROWING)
            .frames(9)
            .looping(false)
            .easing(e);
        (0..9).map(|i| a.t_at(i)).collect::<Vec<_>>()
    };
    let linear = curve(ScadEasing::Linear);
    let ease_in = curve(ScadEasing::EaseIn);
    let ease_out = curve(ScadEasing::EaseOut);

    for c in [&linear, &ease_in, &ease_out] {
        assert_eq!(c[0], 0.0);
        assert!((c[8] - 1.0).abs() < 1e-9, "{c:?}");
        assert!(c.windows(2).all(|w| w[1] >= w[0]), "must not go backwards");
    }
    assert!(ease_in[4] < linear[4], "ease-in lags");
    assert!(ease_out[4] > linear[4], "ease-out leads");
}

#[test]
fn easing_actually_changes_the_geometry_timeline() {
    // The same frame index lands at a different size under a different curve.
    let linear = ScadAnimation::from_source(GROWING).frames(9).looping(false);
    let eased = ScadAnimation::from_source(GROWING)
        .frames(9)
        .looping(false)
        .easing(ScadEasing::EaseIn);
    assert!(size_of(&eased.frame(4).unwrap()) < size_of(&linear.frame(4).unwrap()));
    // …but the ends still agree.
    assert!((size_of(&eased.frame(8).unwrap()) - size_of(&linear.frame(8).unwrap())).abs() < 0.01);
}

#[test]
fn quality_trades_detail_for_speed() {
    let tris = |q| {
        ScadAnimation::from_source("sphere(20);")
            .frames(1)
            .quality(q)
            .frame(0)
            .unwrap()
            .triangle_count()
    };
    let draft = tris(ScadQuality::Draft);
    let balanced = tris(ScadQuality::Balanced);
    let fine = tris(ScadQuality::Fine);
    assert!(draft < balanced, "{draft} vs {balanced}");
    assert!(balanced < fine, "{balanced} vs {fine}");

    // Render-side quality is the supersample factor.
    assert_eq!(ScadQuality::Draft.supersample(), 1);
    assert_eq!(ScadQuality::Fine.supersample(), 3);
    assert_eq!(ScadQuality::Draft.kernel(), ScadKernel::Float);
    assert_eq!(ScadQuality::Fine.kernel(), ScadKernel::Exact);
}

#[test]
fn a_model_that_pins_its_own_facets_ignores_the_quality_preset() {
    let tris = |q| {
        ScadAnimation::from_source("sphere(20, $fn = 32);")
            .frames(1)
            .quality(q)
            .frame(0)
            .unwrap()
            .triangle_count()
    };
    assert_eq!(
        tris(ScadQuality::Draft),
        tris(ScadQuality::Fine),
        "an explicit $fn must win, as it does in OpenSCAD"
    );
}

#[test]
fn quality_is_not_cumulative_across_calls() {
    let a = ScadAnimation::from_source("sphere(20);")
        .frames(1)
        .quality(ScadQuality::Draft)
        .quality(ScadQuality::Fine);
    let direct = ScadAnimation::from_source("sphere(20);")
        .frames(1)
        .quality(ScadQuality::Fine);
    assert_eq!(
        a.frame(0).unwrap().triangle_count(),
        direct.frame(0).unwrap().triangle_count()
    );
}

#[test]
fn constants_are_seeded_without_making_the_model_animated() {
    let a = ScadAnimation::from_source("cube([10, 10, H]);")
        .frames(4)
        .constant("H", 7.0);
    assert!(!a.is_animated(), "a constant is a parameter, not a driver");
    assert_eq!(a.frame(0).unwrap().bounds().unwrap().1[2], 7.0);
    assert_eq!(a.frame(3).unwrap().bounds().unwrap().1[2], 7.0);

    // An animated var of the same name takes precedence over the constant.
    let overridden = ScadAnimation::from_source("cube([10, 10, H]);")
        .frames(4)
        .constant("H", 7.0)
        .var("H", |t| 1.0 + 10.0 * t);
    assert!(overridden.is_animated());
    assert_eq!(overridden.frame(0).unwrap().bounds().unwrap().1[2], 1.0);
}

#[cfg(all(feature = "video", feature = "native-codec"))]
#[test]
fn the_encoded_video_uses_the_adjusted_frame_rate() {
    if !gpu_available() {
        eprintln!("skipping: no GPU adapter");
        return;
    }
    let dir = std::env::temp_dir().join(format!("threers-scad-speed-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();

    // 8 frames at 8 fps, played at 2× → 16 fps → a half-second loop.
    let mut a = ScadAnimation::from_source(GROWING)
        .frames(8)
        .fps(8)
        .speed(2.0);
    assert_eq!(a.frame_rate(), 16);
    let out = dir.join("fast.gif");
    ScadRender::new(96, 72).export_video(&mut a, &out).unwrap();

    let (_, frames) = threers::decode_gif(&std::fs::read(&out).unwrap()).unwrap();
    assert_eq!(frames.len(), 8);
    // GIF delays are centiseconds: 16 fps ≈ 6 cs, not the 12 cs of 8 fps.
    assert!(frames[0].delay_num <= 7, "delay {} cs", frames[0].delay_num);
    let _ = std::fs::remove_dir_all(&dir);
}

// ---------------------------------------------------------------------------
// Lighting
// ---------------------------------------------------------------------------

use threers::openscad::animate::{ScadLight, ScadLighting, ScadRig};

/// How a white cube comes out under `lighting`.
struct Exposure {
    darkest: u8,
    /// 95th percentile — the top of the *diffuse* range. A specular highlight
    /// is allowed to clip above this; a whole face is not.
    p95: u8,
    brightest: u8,
    /// Share of lit pixels pinned at 255.
    clipped: f64,
}

fn exposure(lighting: ScadLighting) -> Exposure {
    let mut model = ScadAnimation::from_source("cube(20, center = true);").frames(1);
    let frames = ScadRender::new(200, 150)
        .lighting(lighting)
        .coloring(ScadColoring::Uniform)
        .material(ScadMaterial::default().color([1.0, 1.0, 1.0, 1.0]))
        .background([0.0, 0.0, 0.0, 1.0])
        .render_frames(&mut model)
        .unwrap();
    let mut lit: Vec<u8> = frames[0]
        .chunks_exact(4)
        .filter(|px| px[0] > 4 || px[1] > 4 || px[2] > 4)
        .map(|px| px[0].max(px[1]).max(px[2]))
        .collect();
    assert!(!lit.is_empty(), "nothing was lit at all");
    lit.sort_unstable();
    let n = lit.len();
    Exposure {
        darkest: lit[0],
        p95: lit[n * 95 / 100],
        brightest: lit[n - 1],
        // `>= 255` on a byte is every value there is. Clipped means saturated.
        clipped: lit.iter().filter(|&&v| v == 255).count() as f64 * 100.0 / n as f64,
    }
}

#[test]
fn every_rig_lights_a_white_model_without_blowing_it_out() {
    if !gpu_available() {
        eprintln!("skipping: no GPU adapter");
        return;
    }
    for rig in [
        ScadLighting::studio(),
        ScadLighting::sun(215.0, 38.0),
        ScadLighting::flat(),
    ] {
        let name = format!("{:?}", rig.rig);
        let e = exposure(rig);
        // Bright enough to read as lit…
        assert!(
            e.brightest > 150,
            "{name}: too dim, brightest is {}",
            e.brightest
        );
        // …with the diffuse range below clipping, so faces keep their shading.
        assert!(
            e.p95 < 250,
            "{name}: diffuse is blown out, p95 is {}",
            e.p95
        );
        // A specular highlight may clip, but only a sliver of the image.
        assert!(
            e.clipped < 4.0,
            "{name}: {:.1}% of the model is clipped",
            e.clipped
        );
        // And the shaded side still carries light rather than going black.
        assert!(e.darkest > 12, "{name}: shadows crushed to {}", e.darkest);
    }
}

#[test]
fn intensity_and_ambient_move_the_exposure_the_way_you_would_expect() {
    if !gpu_available() {
        eprintln!("skipping: no GPU adapter");
        return;
    }
    let base = exposure(ScadLighting::studio());
    let dim = exposure(ScadLighting::studio().intensity(0.4));
    assert!(dim.p95 < base.p95, "lower intensity must be darker");

    // Ambient opens up the shaded side without touching the key.
    let open = exposure(ScadLighting::studio().ambient(1.2));
    assert!(
        open.darkest > base.darkest,
        "more ambient must lift the shadows"
    );
}

#[test]
fn the_studio_key_follows_the_camera() {
    // A camera-relative rig must keep the near face lit from every angle —
    // that is the whole point of it on a turntable.
    if !gpu_available() {
        eprintln!("skipping: no GPU adapter");
        return;
    }
    let mut model = ScadAnimation::from_source("cube(20, center = true);").frames(6);
    let frames = ScadRender::new(160, 120)
        .camera(ScadCamera::turntable())
        .coloring(ScadColoring::Uniform)
        .material(ScadMaterial::default().color([1.0, 1.0, 1.0, 1.0]))
        .background([0.0, 0.0, 0.0, 1.0])
        .render_frames(&mut model)
        .unwrap();

    let brightest: Vec<u8> = frames
        .iter()
        .map(|f| f.chunks_exact(4).map(|px| px[0]).max().unwrap())
        .collect();
    let lo = *brightest.iter().min().unwrap();
    let hi = *brightest.iter().max().unwrap();
    assert!(lo > 150, "some angle went dark: {brightest:?}");
    // Every angle should look about equally lit.
    assert!(
        hi - lo < 40,
        "exposure swings around the orbit: {brightest:?}"
    );
}

#[test]
fn a_world_fixed_sun_does_swing_around_the_orbit() {
    // The counterpart of the test above: `Sun` is deliberately world-fixed, so
    // the shading *should* change as the camera moves. If it did not, the two
    // rigs would be the same thing.
    if !gpu_available() {
        eprintln!("skipping: no GPU adapter");
        return;
    }
    let mut model = ScadAnimation::from_source("cube(20, center = true);").frames(4);
    let frames = ScadRender::new(160, 120)
        .camera(ScadCamera::turntable())
        .lighting(ScadLighting::sun(0.0, 25.0))
        .coloring(ScadColoring::Uniform)
        .material(ScadMaterial::default().color([1.0, 1.0, 1.0, 1.0]))
        .render_frames(&mut model)
        .unwrap();
    assert_ne!(
        frames[0], frames[2],
        "a fixed sun must relight as you orbit"
    );
}

#[test]
fn shadows_darken_the_model_and_can_be_turned_off() {
    if !gpu_available() {
        eprintln!("skipping: no GPU adapter");
        return;
    }
    // A post standing on a plate: the post must throw a shadow onto the plate.
    let src = "cylinder(h = 40, r = 4, $fn = 24); translate([0, 0, -3]) cylinder(h = 3, r = 30, $fn = 48);";
    let render = |shadows| {
        let mut model = ScadAnimation::from_source(src).frames(1);
        ScadRender::new(220, 165)
            .lighting(ScadLighting::sun(200.0, 30.0).shadows(shadows))
            .coloring(ScadColoring::Uniform)
            .material(ScadMaterial::default().color([0.9, 0.9, 0.9, 1.0]))
            .render_frames(&mut model)
            .unwrap()
            .remove(0)
    };
    let with = render(true);
    let without = render(false);
    assert_ne!(with, without, "shadows must change the image");

    // The shadowed render is darker overall — it removes light, never adds it.
    let mean =
        |f: &Vec<u8>| f.chunks_exact(4).map(|p| p[0] as u64).sum::<u64>() / (f.len() / 4) as u64;
    assert!(
        mean(&with) < mean(&without),
        "{} vs {}",
        mean(&with),
        mean(&without)
    );
}

#[test]
fn a_custom_rig_uses_only_the_lights_it_is_given() {
    if !gpu_available() {
        eprintln!("skipping: no GPU adapter");
        return;
    }
    // One red lamp and nothing else: the model can only come back red.
    let lighting = ScadLighting::custom(vec![ScadLight::Key {
        color: [1.0, 0.0, 0.0],
        intensity: 2.0,
        azimuth: 20.0,
        elevation: 25.0,
    }]);
    assert_eq!(lighting.rig, ScadRig::Custom);

    let mut model = ScadAnimation::from_source("sphere(10, $fn = 24);").frames(1);
    let frame = ScadRender::new(120, 90)
        .lighting(lighting)
        .coloring(ScadColoring::Uniform)
        .material(ScadMaterial::default().color([1.0, 1.0, 1.0, 1.0]))
        .background([0.0, 0.0, 0.0, 1.0])
        .render_frames(&mut model)
        .unwrap()
        .remove(0);

    let lit: Vec<&[u8]> = frame.chunks_exact(4).filter(|px| px[0] > 20).collect();
    assert!(!lit.is_empty(), "the lamp lit nothing");
    for px in lit {
        assert!(
            px[1] < 40 && px[2] < 40,
            "expected red-only light, got {px:?}"
        );
    }
}

#[test]
fn palettes_carry_their_own_ambient_light() {
    // Light should match the scheme, so a palette brings sky and ground with it.
    for p in ScadPalette::all() {
        let sky_sum: f32 = p.sky.iter().sum();
        let ground_sum: f32 = p.ground.iter().sum();
        assert!(
            sky_sum > ground_sum,
            "{}: sky should outshine ground",
            p.name
        );
        assert!(
            sky_sum > 0.5,
            "{}: sky is too dark to fill anything",
            p.name
        );
    }
    // …and an explicit hemisphere overrides it.
    let pinned = ScadLighting::studio().hemisphere([1.0, 0.0, 0.0], [0.0, 0.0, 1.0]);
    assert_eq!(pinned.hemisphere, Some(([1.0, 0.0, 0.0], [0.0, 0.0, 1.0])));
}


