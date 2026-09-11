//! Headless PNG of a procedurally generated city — the SimCity view, on a
//! machine with no display.
//!
//! ```text
//! cargo run --release --example simcity_render                 # → out/simcity.png
//! cargo run --release --example simcity_render -- --time 0.62  # dusk
//! cargo run --release --example simcity_render -- --strip      # 2x2 day cycle
//!
//! # a whole day as a WebM / GIF / APNG, encoded in-process by the pure-Rust
//! # codecs — no ffmpeg anywhere:
//! cargo run --release --example simcity_render --features native-codec \
//!     -- --anim simcity.webm
//! ```
//!
//! | Flag | Meaning |
//! |------|---------|
//! | `--seed N` | which city (any `u64`; the layout is fully deterministic) |
//! | `--blocks N` | blocks per side, 3–12 (default 6) |
//! | `--layout L` | `manhattan` (default), `boulevard`, `oldtown`, `waterfront`, `parkway` |
//! | `--no-postfx` | skip post-processing; `--bloom S`, `--bloom-threshold T`, `--bloom-radius R` tune it |
//! | `--no-bake` | skip the raycast light bake (needs `--features mesh-bvh` to do anything either way) |
//! | `--time T` | fraction of a day: `0` sunrise, `0.25` noon, `0.5` sunset (default `0.07`, mid-morning) |
//! | `--view V` | `aerial` (default), `skyline`, `close`, `street` |
//! | `--eye x,y,z` `--look x,y,z` `--fov D` | free camera, in metres from the city centre |
//! | `--follow N` | stand in front of pedestrian `N` |
//! | `--size WxH` | output resolution (default 1600x1000) |
//! | `--strip` | four times of day in one 2x2 sheet |
//! | `--sheet` | every street layout in one 4x2 sheet |
//! | `--anim PATH` | a whole day as an animation (needs `native-codec`) |
//! | `--anim-frames N` | frames in that animation (default 96) |
//! | `--anim-fps N` | playback rate (default 24) |
//! | `--anim-orbit D` | degrees the camera swings across the clip (default 40) |
//! | `--out PATH` | where to write the PNG |
//!
//! The city itself lives in `examples/simcity/`, shared with the interactive
//! `simcity` example and the Metal one, so the three cannot drift apart.
//! `cargo test --test simcity` checks its invariants.

use std::path::PathBuf;

use threers::{HeadlessRenderer, PerspectiveCamera, ToneMapping};

// The city itself lives in `examples/simcity/`, shared by all three
// entry points. `#[path]` because a directory example would otherwise
// need its own `main.rs`, and there are three of those.
#[path = "simcity/mod.rs"]
mod city;
use city::*;

fn arg(name: &str) -> Option<String> {
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        if a == name {
            return args.next();
        }
    }
    None
}

/// Render one frame, through the post-processed path when it is on.
///
/// Bloom and screen-space occlusion live on the crate's RGB readback path —
/// the one built for feeding a video encoder — and nothing outside the library
/// used it, so no example had ever turned either of them on. They are exactly
/// what this scene wants: bloom is what makes neon and lamp lenses read as
/// emitting rather than as bright paint, and SSAO grounds everything the
/// baked light cannot touch, which is every car, pedestrian and animal in the
/// city.
fn shoot(
    renderer: &mut HeadlessRenderer,
    scene: &mut Scene,
    camera: &threers::PerspectiveCamera,
    fx: PostFx,
    lights: f32,
) -> Vec<u8> {
    if !fx.on {
        return renderer.render_to_rgba_resolved(scene, camera);
    }
    if fx.bloom > 0.0 {
        // Threshold follows the clock. A neon sign at midnight should bleed;
        // sunlit glass at noon should not, and a fixed threshold low enough
        // for the first puts a haze round every tower in the second. This is
        // roughly what an adapting eye does.
        let threshold = mix(fx.threshold * 2.1, fx.threshold, lights);
        renderer.set_bloom(fx.bloom, threshold, fx.radius);
    }
    if fx.ssao > 0.0 {
        let tan_half = (camera.fov.to_radians() * 0.5).tan();
        renderer.set_ssao(
            fx.ssao,
            fx.ssao_radius,
            camera.near,
            camera.far,
            tan_half,
            camera.aspect,
        );
    }
    renderer.render(scene, camera);
    // The RGB path is pipelined for video: the call that queues a frame is not
    // the call that hands one back, and for a single still the first returns
    // nothing at all. Queue, then drain whatever is waiting.
    let to_rgba = |rgb: Vec<u8>| {
        let mut out = Vec::with_capacity(rgb.len() / 3 * 4);
        for px in rgb.chunks_exact(3) {
            out.extend_from_slice(&[px[0], px[1], px[2], 255]);
        }
        out
    };
    if let Some(frame) = renderer.read_rgb_frame() {
        let bytes = frame.bytes().to_vec();
        renderer.recycle(frame);
        return to_rgba(bytes);
    }
    if let Some(rgb) = renderer.drain_rgb_pixels() {
        return to_rgba(rgb);
    }
    // The path declines some formats; falling back is better than a panic and
    // the only visible difference is the effects.
    renderer.read_rgba_resolved()
}

#[derive(Clone, Copy)]
struct PostFx {
    on: bool,
    bloom: f32,
    threshold: f32,
    radius: f32,
    ssao: f32,
    ssao_radius: f32,
}

fn has_flag(name: &str) -> bool {
    std::env::args().any(|a| a == name)
}


fn main() {
    let out = PathBuf::from(arg("--out").unwrap_or_else(|| "out/simcity.png".into()));
    let view = arg("--view").unwrap_or_else(|| "aerial".into());
    let strip = has_flag("--strip");
    let (w, h) = arg("--size")
        .and_then(|s| {
            let (a, b) = s.split_once('x')?;
            Some((a.parse().ok()?, b.parse().ok()?))
        })
        .unwrap_or((1600u32, 1000u32));

    let fx = PostFx {
        on: !has_flag("--no-postfx"),
        bloom: if has_flag("--no-bloom") {
            0.0
        } else {
            arg("--bloom").and_then(|s| s.parse().ok()).unwrap_or(1.15)
        },
        // Linear luminance a pixel has to beat to bloom at all. Low enough
        // that neon and a lamp lens carry, high enough that a sunlit pavement
        // does not turn to soup.
        threshold: arg("--bloom-threshold")
            .and_then(|s| s.parse().ok())
            .unwrap_or(0.30),
        radius: arg("--bloom-radius")
            .and_then(|s| s.parse().ok())
            .unwrap_or(2.4),
        // Off by default, and not because it is not wanted. Two library bugs
        // sat in the way of this path: bloom could not create its sRGB view of
        // the resolve target, and occlusion was handed the renderer's depth
        // buffer when `render_to` had written the target's — an untouched
        // buffer, which yields a uniformly white factor that looks exactly
        // like "SSAO is on and subtle". Both are fixed. It still resolves to a
        // constant whatever radius it is given, so something further down is
        // wrong and shipping it on would be shipping a flag that does nothing.
        // `--ssao 1.0` turns it on for anyone chasing it.
        ssao: arg("--ssao").and_then(|s| s.parse().ok()).unwrap_or(0.0),
        // World units. A city seen from a hundred metres needs a radius in
        // metres, not centimetres: 1.4 covers a fraction of a pixel at that
        // distance and finds nothing to occlude.
        ssao_radius: arg("--ssao-radius")
            .and_then(|s| s.parse().ok())
            .unwrap_or(6.0),
    };
    let params = CityParams {
        seed: arg("--seed")
            .and_then(|s| s.parse().ok())
            .unwrap_or(CityParams::default().seed),
        blocks: arg("--blocks")
            .and_then(|s| s.parse().ok())
            .unwrap_or(CityParams::default().blocks),
        cars: CityParams::default().cars,
        // Raycast light transport. On by default; a whole contact sheet or a
        // long animation is a reason to want it off.
        bake: !has_flag("--no-bake"),
        layout: arg("--layout")
            .and_then(|s| Layout::from_name(&s))
            .unwrap_or(CityParams::default().layout),
    };

    let mut scene = Scene::new();
    let mut city = generate_city(&mut scene, &params);
    let s = &city.stats;
    println!(
        "{} · seed {} · {} blocks/side · {} buildings · {} cars · {} on foot · {} draws · {} static triangles + up to {} in movers · {:.0} m across",
        params.layout.name(),
        params.seed,
        s.blocks,
        s.buildings,
        s.cars,
        s.walkers,
        s.draws,
        s.triangles,
        s.mover_triangles,
        s.extent * 2.0
    );
    println!("  {}", s.bake.summary());
    println!(
        "  {} people sitting or standing about, {}/4 civic buildings placed, \
         {} blocks laid out as closes or crescents",
        s.idlers, s.civic, s.suburbs
    );
    let wk = &s.works;
    println!(
        "  works: {}, {} pylons, port {}",
        if wk.plant.is_some() { "power station" } else { "NO power station" },
        wk.pylons,
        if wk.port { "yes" } else { "no" },
    );
    println!(
        "  sport: stadium {}, ballpark {}, cricket {}, {} courts",
        if wk.stadium.is_some() { "yes" } else { "NO" },
        if wk.ballpark { "yes" } else { "NO" },
        if wk.cricket { "yes" } else { "NO" },
        wk.courts,
    );
    println!(
        "  retail: mall {}, {} groceries, {} strips, {} parking decks, {} cars in bays",
        if wk.mall { "yes" } else { "NO" },
        wk.grocery,
        wk.strips,
        wk.decks,
        wk.parked_cars,
    );
    println!(
        "  logistics: depot {}, rail yard {}, {} vehicles on the motorway",
        if wk.depot { "yes" } else { "NO" },
        if wk.rail_yard { "yes" } else { "NO" },
        wk.motorway_vehicles,
    );
    println!(
        "  {} surveillance cameras, {} bridges over the motorway, {} animals in the zoo",
        wk.cameras, wk.crossings, wk.zoo_animals
    );
    println!(
        "  {} aircraft on the apron, {} cyclists in the bike lanes",
        wk.aircraft, wk.cyclists
    );
    println!(
        "  {} parking meters, data centre {}, {} telecom masts",
        wk.meters,
        if wk.data_centre { "yes" } else { "NO" },
        wk.masts
    );
    println!("  {} echelon parking bays", wk.echelon_bays);
    println!(
        "  monuments: {}{}",
        if wk.monuments.is_empty() { "NONE".to_string() } else { wk.monuments.join(", ") },
        if wk.suspension { " + suspension bridge" } else { "" }
    );
    println!(
        "  {} access roads, {} mountain summits, {} road tunnels through them",
        wk.spurs, wk.peaks, wk.tunnels.len()
    );
    println!(
        "  {} footprints claimed, {} placements refused for overlapping",
        s.footprints.0, s.footprints.1
    );

    // Panels have to divide the sheet exactly, or a column and a row go
    // unwritten — and the renderer is built once, at panel size.
    let sheet = has_flag("--sheet");
    let bench: usize = arg("--bench").and_then(|v| v.parse().ok()).unwrap_or(0);
    let (cols, rows) = if sheet {
        (4u32, 2u32)
    } else if strip {
        (2, 2)
    } else {
        (1, 1)
    };
    let (w, h) = (w - w % cols, h - h % rows);
    let (tile_w, tile_h) = (w / cols, h / rows);
    let mut renderer = HeadlessRenderer::builder()
        .size(tile_w, tile_h)
        .supersample(2)
        // Rgba8Unorm, not the sRGB default: the mesh shader encodes to sRGB
        // itself, and an sRGB target would do it a second time.
        .color_format(wgpu::TextureFormat::Rgba8Unorm)
        .build()
        .expect("headless renderer (is a GPU adapter available?)");
    // The sun is far brighter than white and lit windows sit above it. ACES
    // rolls that off without shifting hue, so a blown highlight on glass still
    // reads as glass.
    renderer
        .renderer()
        .set_tone_mapping(ToneMapping::AcesFilmic, 1.0);

    // Raw wgpu: a compute pass rewrites the water mesh's vertex buffer in
    // place between frames. Clone the handles up front — `hr.renderer()`
    // borrows the whole renderer mutably.
    let device = renderer.device().clone();
    let queue = renderer.queue().clone();
    let mut water = city
        .water
        .as_ref()
        .map(|w| WaterSim::new(&device, w.vertices, w.level));

    let mut camera = frame_camera(&view, s.extent, tile_w as f32 / tile_h as f32, city.vista);
    // Free camera, for looking at one thing closely. Coordinates are metres,
    // origin at the middle of the city.
    let triple = |v: String| {
        let mut it = v.split(',').filter_map(|t| t.trim().parse::<f32>().ok());
        Some(Vector3::new(it.next()?, it.next()?, it.next()?))
    };
    if let Some(eye) = arg("--eye").and_then(triple) {
        camera.position = eye;
        camera.look_at(arg("--look").and_then(triple).unwrap_or(Vector3::ZERO));
    }
    // Stand in front of one pedestrian. Hunting for a good vantage by hand is
    // how you end up inside a building.
    if let Some(n) = arg("--follow").and_then(|v| v.parse::<usize>().ok()) {
        if let Some((pos, facing)) = city.mover_pose(Gait::Walk, n) {
            let side = Vector3::new(-facing.z, 0.0, facing.x);
            camera.position = pos + facing * 3.1 + side * 1.5 + Vector3::new(0.0, 1.55, 0.0);
            camera.look_at(pos + Vector3::new(0.0, 0.95, 0.0));
        }
    }
    // Independent of `--eye`, so any preset can be zoomed into.
    if let Some(fov) = arg("--fov").and_then(|v| v.parse::<f32>().ok()) {
        camera.fov = fov.to_radians();
    }
    // After every override, because the near plane has to suit wherever the
    // camera actually ended up — see `near_for`.
    {
        let target = arg("--look").and_then(triple).unwrap_or(Vector3::ZERO);
        camera.near = near_for(camera.position, target);
    }
    // The clock first: how much of the population is out depends on it, and
    // measuring before setting it reported the same count at every hour.
    let first_time = arg("--time").and_then(|s| s.parse().ok()).unwrap_or(0.07);
    city.apply_sky(&mut scene, first_time);
    // One tick, now that there is an eye to measure level of detail from,
    // so the traffic is on the roads rather than stacked at the instanced
    // meshes' identity transforms.
    let (near, total) = city.drive(&mut scene, 0.0, camera.position, 0.0);
    println!(
        "  {near}/{total} movers at full detail from here ({:.0}% of the \
         population is out at this hour)",
        city.activity * 100.0
    );
    // Handy when aiming `--eye`: a point that is definitely on tarmac.
    let v = city.vista.0;
    println!("  street level at {:.0},{:.0},{:.0}", v.x, v.y, v.z);
    let times: Vec<f32> = if strip {
        vec![0.04, 0.25, 0.47, 0.72]
    } else {
        vec![arg("--time").and_then(|s| s.parse().ok()).unwrap_or(0.07)]
    };

    if sheet {
        write_layout_sheet(&mut renderer, &params, &out, w, h, tile_w, tile_h, fx);
        return;
    }

    if let Some(path) = arg("--anim") {
        let frames: usize = arg("--anim-frames")
            .and_then(|s| s.parse().ok())
            .unwrap_or(96);
        let fps = arg("--anim-fps").and_then(|s| s.parse().ok()).unwrap_or(24);
        let orbit = arg("--anim-orbit")
            .and_then(|s| s.parse().ok())
            .unwrap_or(40.0);
        write_animation(
            &mut renderer,
            &mut scene,
            &mut city,
            &camera,
            &path,
            frames,
            fps,
            orbit,
            tile_w,
            tile_h,
            &device,
            &queue,
            &mut water,
            fx,
        );
        return;
    }

    // Steady-state frame timing, through exactly the path a viewer uses:
    // simulate, update the transforms, render. Warm-up frames are discarded
    // because the first few pay for pipeline and bind-group creation, which a
    // running viewport pays once and a benchmark should not count many times.
    if bench > 0 {
        let mut worst = 0.0f64;
        let mut part = [0.0f64; 6];
        for i in 0..bench + 4 {
            let clock = i as f32 * 0.02;
            let mut lap = [0.0f64; 6];
            let t0 = std::time::Instant::now();
            if let (Some(sim), Some(w)) = (&mut water, &city.water) {
                sim.step(&device, &queue, renderer.renderer(), &w.geometry, clock, 0.35);
            }
            let mut m = std::time::Instant::now();
            lap[0] = (m - t0).as_secs_f64() * 1000.0;
            let sky = city.apply_sky(&mut scene, first_time);
            lap[1] = m.elapsed().as_secs_f64() * 1000.0;
            m = std::time::Instant::now();
            city.drive(&mut scene, clock, camera.position, 0.0);
            lap[2] = m.elapsed().as_secs_f64() * 1000.0;
            m = std::time::Instant::now();
            city.place_lights(&mut scene, camera.position, sky.lights);
            city.animate_lights(&mut scene, clock);
            lap[3] = m.elapsed().as_secs_f64() * 1000.0;
            m = std::time::Instant::now();
            scene.update_world();
            lap[4] = m.elapsed().as_secs_f64() * 1000.0;
            m = std::time::Instant::now();
            let _ = shoot(&mut renderer, &mut scene, &camera, fx, sky.lights);
            lap[5] = m.elapsed().as_secs_f64() * 1000.0;
            let ms = t0.elapsed().as_secs_f64() * 1000.0;
            if i >= 4 {
                for k in 0..6 {
                    part[k] += lap[k];
                }
                worst = worst.max(ms);
            }
        }
        let n = bench as f64;
        let mean: f64 = part.iter().sum::<f64>() / n;
        println!(
            "bench {bench} frames at {tile_w}x{tile_h}: {mean:.2} ms/frame \
             ({:.0} fps), worst {worst:.2} ms",
            1000.0 / mean
        );
        println!(
            "  water {:.2}  sky {:.2}  drive {:.2}  lights {:.2}  world {:.2}  gpu {:.2}",
            part[0] / n,
            part[1] / n,
            part[2] / n,
            part[3] / n,
            part[4] / n,
            part[5] / n
        );
        return;
    }

    let mut sheet = vec![0u8; (w * h * 4) as usize];
    for (i, t) in times.iter().enumerate() {
        if let (Some(sim), Some(w)) = (&mut water, &city.water) {
            sim.step(&device, &queue, renderer.renderer(), &w.geometry, 11.0, 0.35);
        }
        let sky = city.apply_sky(&mut scene, *t);
        city.place_lights(&mut scene, camera.position, sky.lights);
        // A still wants the warning lights on rather than caught mid-blink.
        city.animate_lights(&mut scene, 0.0);
        scene.update_world();
        let rgba = shoot(&mut renderer, &mut scene, &camera, fx, sky.lights);
        // What the frustum threw away. Culling is defined to leave the image
        // alone, so this counter is the only way to see it working — and on a
        // loaded machine it is a far steadier measure than wall-clock time.
        // Re-render a few times and keep the FASTEST GPU time. This is an
        // integrated GPU sharing its memory bandwidth and power budget with
        // whatever else the machine is running, so the same frame measured
        // repeatedly here spread from 7.9 to 14.1 ms. The spread is real
        // contention rather than noise in the counter, and the minimum is the
        // least-contended sample — the closest thing to what the frame costs
        // when the box is quiet.
        let mut gpu_ms = f32::INFINITY;
        for _ in 0..48 {
            let _ = shoot(&mut renderer, &mut scene, &camera, fx, sky.lights);
            let ms = renderer.renderer().gpu_frame_ms();
            if ms > 0.0 {
                gpu_ms = gpu_ms.min(ms);
            }
        }
        let gpu_ms = if gpu_ms.is_finite() { gpu_ms } else { 0.0 };
        let (drawn, culled) = renderer.renderer().cull_stats();
        let total = drawn + culled;
        println!(
            "  t={t:.2} rendered · {drawn}/{total} batches drawn, {culled} culled ({:.0}%) \
             · main pass {gpu_ms:.2} ms on the GPU (best of 48)",
            if total > 0 {
                100.0 * culled as f32 / total as f32
            } else {
                0.0
            }
        );
        if !strip {
            sheet = rgba;
            break;
        }
        let (ox, oy) = ((i as u32 % 2) * tile_w, (i as u32 / 2) * tile_h);
        for row in 0..tile_h {
            let src = (row * tile_w * 4) as usize;
            let dst = (((oy + row) * w + ox) * 4) as usize;
            sheet[dst..dst + (tile_w * 4) as usize]
                .copy_from_slice(&rgba[src..src + (tile_w * 4) as usize]);
        }
    }

    let (fw, fh) = if strip { (w, h) } else { (tile_w, tile_h) };
    if let Some(dir) = std::path::Path::new(&out).parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    std::fs::write(&out, threers::encode_png(fw, fh, &sheet)).expect("write png");
    println!("wrote {} ({fw}x{fh})", out.display());
}

/// Render one whole day and encode it in-process.
///
/// Two deliberate choices. The traffic advances a tenth of a second a frame
/// while the sun does a full cycle: a true timelapse would smear the cars
/// into stripes, and the point of the shot is that both are moving. And the
/// camera swings through `--anim-orbit` degrees across the clip, because a
/// locked-off camera makes a day cycle look like a slideshow of one frame.
///
/// Frames are handed to the encoder one at a time rather than collected: a
/// 96-frame 1280x800 clip is 400 MB if you hold them all, and nothing here
/// needs more than one at once.
#[cfg(feature = "native-codec")]
#[allow(clippy::too_many_arguments)]
fn write_animation(
    renderer: &mut HeadlessRenderer,
    scene: &mut Scene,
    city: &mut City,
    camera: &PerspectiveCamera,
    path: &str,
    frames: usize,
    fps: u32,
    orbit_deg: f32,
    w: u32,
    h: u32,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    water: &mut Option<WaterSim>,
    fx: PostFx,
) {
    use threers::{encode_animation_rgba, AnimationEncodeOptions, BrowserCodec};

    let codec = match path.rsplit('.').next().unwrap_or("") {
        "gif" => BrowserCodec::Gif,
        "png" | "apng" => BrowserCodec::Apng,
        _ => BrowserCodec::Webm,
    };
    let frames = frames.clamp(2, 900);
    let target = camera.target;
    let offset = camera.position - target;
    let mut cam = camera.clone();
    let mut f = 0usize;

    println!("rendering {frames} frames at {w}x{h}, {fps} fps, {orbit_deg:.0}° orbit…");
    let stream = std::iter::from_fn(|| {
        if f >= frames {
            return None;
        }
        let t = f as f32 / frames as f32;
        let a = orbit_deg.to_radians() * t;
        let (sa, ca) = (a.sin(), a.cos());
        cam.position = target
            + Vector3::new(
                offset.x * ca + offset.z * sa,
                offset.y,
                -offset.x * sa + offset.z * ca,
            );
        cam.look_at(target);
        if let (Some(sim), Some(w)) = (water.as_mut(), &city.water) {
            sim.step(device, queue, renderer.renderer(), &w.geometry, f as f32 * 0.35, 0.35);
        }
        let sky = city.apply_sky(scene, t);
        city.place_lights(scene, cam.position, sky.lights);
        city.animate_lights(scene, f as f32 * 0.35);
        city.drive(scene, 0.10, cam.position, f as f32 * 0.35);
        scene.update_world();
        let rgba = shoot(renderer, scene, &cam, fx, sky.lights);
        if f.is_multiple_of(12) {
            println!("  frame {}/{frames}", f + 1);
        }
        f += 1;
        Some(rgba)
    });

    let opts = AnimationEncodeOptions {
        width: w,
        height: h,
        fps,
        codec,
        transparent: false,
        gif_colors: 256,
    };
    let bytes = encode_animation_rgba(&opts, stream).expect("encode animation");
    std::fs::write(path, &bytes).expect("write animation");
    println!("wrote {path} ({:.1} MB)", bytes.len() as f32 / 1.0e6);
}

/// Without the codecs there is nothing to encode with, and saying so beats
/// failing to compile the rest of the example.
#[cfg(not(feature = "native-codec"))]
#[allow(clippy::too_many_arguments)]
fn write_animation(
    _renderer: &mut HeadlessRenderer,
    _scene: &mut Scene,
    _city: &mut City,
    _camera: &PerspectiveCamera,
    _path: &str,
    _frames: usize,
    _fps: u32,
    _orbit_deg: f32,
    _w: u32,
    _h: u32,
    _device: &wgpu::Device,
    _queue: &wgpu::Queue,
    _water: &mut Option<WaterSim>,
    _fx: PostFx,
) {
    eprintln!("--anim needs the `native-codec` feature: cargo run --release \\");
    eprintln!("    --example simcity_render --features native-codec -- --anim out.webm");
}

/// One panel per street layout, cycling from whichever `--layout` was asked
/// for. Each panel is a city of its own — same seed, same camera rules,
/// different plan — which is the quickest way to see what the setting does.
#[allow(clippy::too_many_arguments)]
fn write_layout_sheet(
    renderer: &mut HeadlessRenderer,
    params: &CityParams,
    out: &std::path::Path,
    w: u32,
    h: u32,
    tw: u32,
    th: u32,
    fx: PostFx,
) {
    const COLS: u32 = 4;
    const ROWS: u32 = 2;
    let mut sheet = vec![0u8; (w * h * 4) as usize];
    let mut layout = params.layout;
    for panel in 0..COLS * ROWS {
        // Eight panels, seven layouts: the last repeats the first after dark.
        let night = panel == COLS * ROWS - 1;
        let mut scene = Scene::new();
        let mut city = generate_city(
            &mut scene,
            &CityParams {
                seed: params.seed,
                blocks: params.blocks,
                cars: params.cars,
                layout,
                bake: params.bake,
            },
        );
        let camera = frame_camera("aerial", city.stats.extent, tw as f32 / th as f32, city.vista);
        let sky = city.apply_sky(&mut scene, if night { 0.72 } else { 0.07 });
        city.place_lights(&mut scene, camera.position, sky.lights);
        city.animate_lights(&mut scene, 0.0);
        city.drive(&mut scene, 0.0, camera.position, 0.0);
        scene.update_world();
        if let Some(w) = &city.water {
            let device = renderer.device().clone();
            let queue = renderer.queue().clone();
            WaterSim::new(&device, w.vertices, w.level).step(
                &device,
                &queue,
                renderer.renderer(),
                &w.geometry,
                11.0,
                0.35,
            );
        }
        println!("  {} {}", layout.name(), if night { "(night)" } else { "" });
        let rgba = shoot(renderer, &mut scene, &camera, fx, sky.lights);

        let (ox, oy) = ((panel % COLS) * tw, (panel / COLS) * th);
        for row in 0..th {
            let src = (row * tw * 4) as usize;
            let dst = (((oy + row) * w + ox) * 4) as usize;
            sheet[dst..dst + (tw * 4) as usize]
                .copy_from_slice(&rgba[src..src + (tw * 4) as usize]);
        }
        // The renderer caches geometry by identity and nothing here will drop
        // the last city's buffers otherwise.
        renderer.renderer().clear_geometry_cache();
        if !night {
            layout = layout.next();
        }
    }
    std::fs::write(out, threers::encode_png(w, h, &sheet)).expect("write png");
    println!("wrote {} ({w}x{h})", out.display());
}
