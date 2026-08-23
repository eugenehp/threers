//! Spectral ocean water — GPU waves, foam, absorption, buoyancy, Preetham sky.
//!
//! ```text
//! cargo run --release --example ocean_water
//! cargo run --release --example ocean_water -- storm
//! cargo run --release --example ocean_water -- sunset --quality ultra
//! cargo run --release --example ocean_water -- moonlit --png ocean.png
//! cargo run --release --example ocean_water -- tropical --bench
//! ```
//!
//! Left-drag orbits, right-drag pans, scroll zooms. `--list` prints the presets.
//!
//! # How it works
//!
//! The surface is a **JONSWAP wave spectrum** ([Hasselmann et al. 1973]) sampled
//! on a full lattice and inverse-transformed into three tiling cascades — swell,
//! waves, ripples — by an **FFT on the GPU** ([`ocean_fft`]). Wind speed sets the
//! energy of the sea, the spectral peak sets the size of the dominant wave, and
//! the two are independent. Heights come out in metres and the motion runs in
//! real seconds, so a preset is a sea state rather than a bag of magic numbers.
//!
//! The cascades are then a *texture*, which is what makes 65 536 modes each
//! cheaper than forty analytic ones were. Everything reads them the same way,
//! through one shared sampler in [`waves_gpu::cascade_wgsl`]:
//!
//! - a **compute shader** displaces the mesh from them ([`waves_gpu`]),
//! - the **fragment shader** samples them for per-pixel normals and the
//!   breaking-wave mask ([`shader`]),
//! - a second compute pass reads them for persistent foam ([`surface_state`]),
//! - and the **CPU** answers buoyancy from the dominant modes, without a
//!   readback ([`ocean_fft::OceanFft::sample`]).
//!
//! Nothing is integrated — every mode is a fixed amplitude and a rotating phase —
//! so nothing drifts and two machines running the same preset agree frame for
//! frame. Every consumer band-limits to what its own sampler can resolve, by
//! fading out cascades finer than its footprint: sampling a wave finer than your
//! own spacing does not give you detail, it gives you a different and wrong
//! lower-frequency wave.
//!
//! Foam comes from the **Jacobian of the horizontal displacement**, where
//! choppiness folds the surface onto itself, plus a slope term for the
//! wind-driven whitecapping the Jacobian alone under-reports.
//!
//! # Shoaling
//!
//! An FFT field is homogeneous and depth is not, so the cascades are generated
//! for deep water and **shoaled per sample** against the analytic bed. Everything
//! follows from the finite-depth dispersion relation `w² = g k tanh(kd)`: waves
//! slow down, shorten, and — by Green's law — dip a few percent before growing as
//! `d^(-1/4)`. A wave taller than about 0.78 of its own depth breaks, which is
//! what stops the amplification running away and what draws the surf zone.
//!
//! Refraction is the first-order version: slowing down means the phase lags, and
//! a lag that varies across the shore turns the crests to face it. The lag is
//! integrated in from deep water along the depth gradient and applied as a shift
//! of the sample position — only where `kd < 1.2`, since past that the local
//! wavenumber is within a few percent of the deep-water one and the loop is the
//! most expensive thing in the shader.
//!
//! # What the surface does with the scene
//!
//! The water opts into threers' screen-space pass, so the captured opaque scene
//! is bound while it draws. That is what buys **refraction** (what is behind the
//! surface, bent), **transparency measured against real geometry** rather than an
//! analytic seabed — which is what makes the waterline soft and puts a buoy's wet
//! half underwater — and **screen-space reflection**, so the island appears in
//! the water instead of leaving a hole where its reflection should be. The sky
//! stays the reflection fallback, because only what is on screen can reflect.
//!
//! **Caustics** fall out of the same Jacobian the foam does: the surface is a
//! lens, and where it converges the floor is brighter. Below the waterline the
//! surface shades from underneath, with total internal reflection outside
//! Snell's window.
//!
//! One thing here genuinely carries state. Foam that broke seconds ago is still
//! there, drifting and dissolving, and no function of the current wave field can
//! say so — see [`surface_state`], which is also where wake lives.
//!
//! **Spray, rain and underwater motes** are a third compute pass writing
//! camera-facing quads ([`particles`]). Spray finds its crests by sampling the
//! same cascades, so it is thrown off the waves that are actually there.
//!
//! # Two halves of one column
//!
//! The sea floor is a `ShaderMaterial` too, and the split between it and the
//! water follows the physics: the **floor** attenuates the light coming *down*
//! through the column and focuses it into caustics; the **water** attenuates what
//! comes back *up* to the eye. Neither needs to know about the other's half, and
//! the floor being lit correctly is what removed a compensation hack from the
//! water shader. It also means caustics show when you look at the bed directly,
//! not only through the surface.
//!
//! The bed shades from the analytic height field rather than from its own mesh,
//! so its relief is finer than its 4 m tessellation, and it carries wave-built
//! sand ripples (underwater only, and only above the wave base), wet-sand
//! darkening at the waterline, and silting with depth.
//!
//! # Where the vertices live
//!
//! [`ShaderMaterial`](threers::ShaderMaterial) replaces the fragment stage only,
//! so there is no vertex hook to displace through. The way out is that this
//! example owns the `wgpu::Device` the renderer was built on: the water geometry
//! is marked [`gpu_writable`](threers::BufferGeometry::gpu_writable), its vertex
//! buffer is fetched with [`Renderer::vertex_buffer`](threers::Renderer::vertex_buffer),
//! and a compute pass writes positions and normals into it directly.
//!
//! The mesh is uploaded once and then never again. Per frame the CPU writes the
//! component table (a few hundred bytes) and sixteen bytes of parameters — the
//! vertices never come back across the bus.
//!
//! `--bench` reports three numbers, because they answer different questions: the
//! CPU cost per frame, the end-to-end frame (fill-rate bound at that resolution,
//! and readback-bound on top), and the kernels alone with the queue drained —
//! the only one of the three that can tell you whether a compute change helped.
//!
//! [Hasselmann et al. 1973]: https://repository.tudelft.nl/islandora/object/uuid:f204e188-13b9-49d8-a6dc-4fb7c20562fc

mod fft;
mod grid;
mod ocean_fft;
mod particles;
mod preset;
mod probe;
mod shader;
mod spectrum;
mod surface_state;
mod terrain;
mod waterline;
mod waves_gpu;
mod world;

use std::sync::Arc;
use std::time::Instant;

use threers::cameras::Camera;
use threers::{encode_png, HeadlessRenderer, OrbitControls, PointerEvent, Renderer};

use winit::{
    dpi::PhysicalPosition,
    event::{ElementState, Event, MouseButton, MouseScrollDelta, WindowEvent},
    event_loop::EventLoop,
    keyboard::Key,
    window::WindowBuilder,
};

use preset::{Preset, Quality, QUALITY_LEVELS};
use world::World;

// ---------------------------------------------------------------------------
// Command line
// ---------------------------------------------------------------------------

struct Args {
    preset: Preset,
    quality: Quality,
    png: Option<String>,
    bench: bool,
    /// Overrides the quality tier's cascade lattice, if given.
    cascade: Option<usize>,
    /// Largest cascade tile, metres — how far the swell runs before it repeats.
    max_scale: f32,
    /// Camera override for `--png`, so a still can be framed on the thing being
    /// looked at rather than on whatever the preset opens with.
    cam: Option<[f32; 3]>,
    look: Option<[f32; 3]>,
    /// Draw without the lens overlay, to see what the surface shader alone does
    /// at the waterline.
    no_waterline: bool,
    /// Override the preset's foam amount, for telling foam from geometry.
    foam: Option<f32>,
    /// Cut the water away inside this radius of the origin.
    mask: Option<f32>,
    /// Render through `Renderer::render` straight into a surface-shaped texture
    /// — the path the window and the browser use — rather than through the
    /// headless render-to-target path, and write the result out.
    direct: Option<String>,
}

fn parse_args() -> Args {
    let all = preset::all();
    let mut args = Args {
        preset: all[1],
        quality: QUALITY_LEVELS[2],
        png: None,
        bench: false,
        cascade: None,
        max_scale: ocean_fft::TILES[0],
        cam: None,
        look: None,
        no_waterline: false,
        foam: None,
        mask: None,
        direct: None,
    };

    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < argv.len() {
        match argv[i].as_str() {
            "--quality" | "-q" if i + 1 < argv.len() => {
                i += 1;
                match QUALITY_LEVELS.iter().find(|q| q.name == argv[i]) {
                    Some(q) => args.quality = *q,
                    None => {
                        let names: Vec<_> = QUALITY_LEVELS.iter().map(|q| q.name).collect();
                        fail(&format!(
                            "unknown quality {:?}; expected one of {names:?}",
                            argv[i]
                        ));
                    }
                }
            }
            "--cascade" | "-c" if i + 1 < argv.len() => {
                i += 1;
                match argv[i].parse::<usize>() {
                    Ok(n) => args.cascade = Some(n),
                    Err(_) => fail(&format!("--cascade wants a texel count, got {:?}", argv[i])),
                }
            }
            "--max-scale" if i + 1 < argv.len() => {
                i += 1;
                match argv[i].parse::<f32>() {
                    Ok(m) => args.max_scale = m,
                    Err(_) => fail(&format!("--max-scale wants metres, got {:?}", argv[i])),
                }
            }
            "--cam" | "--look" if i + 1 < argv.len() => {
                let which = argv[i].clone();
                i += 1;
                let v: Vec<f32> = argv[i]
                    .split(',')
                    .filter_map(|c| c.trim().parse().ok())
                    .collect();
                if v.len() != 3 {
                    fail(&format!("{which} wants x,y,z; got {:?}", argv[i]));
                }
                let t = [v[0], v[1], v[2]];
                if which == "--cam" {
                    args.cam = Some(t);
                } else {
                    args.look = Some(t);
                }
            }
            "--mask" if i + 1 < argv.len() => {
                i += 1;
                args.mask = argv[i].parse().ok();
            }
            "--foam" if i + 1 < argv.len() => {
                i += 1;
                args.foam = argv[i].parse().ok();
            }
            "--direct" if i + 1 < argv.len() => {
                i += 1;
                args.direct = Some(argv[i].clone());
            }
            "--png" if i + 1 < argv.len() => {
                i += 1;
                args.png = Some(argv[i].clone());
            }
            "--bench" => args.bench = true,
            "--no-waterline" => args.no_waterline = true,
            "--list" => {
                println!("presets:");
                for p in &all {
                    println!("  {:9} {}", p.name, p.blurb);
                }
                println!("quality levels:");
                for q in &QUALITY_LEVELS {
                    println!(
                        "  {:7} {}x{} grid = {} vertices, {}² cascades",
                        q.name,
                        q.rings,
                        q.sectors,
                        1 + q.rings * q.sectors,
                        q.cascade_n
                    );
                }
                println!("  --cascade N overrides the lattice, --max-scale M the swell's tile");
                println!("  --cam x,y,z and --look x,y,z frame a --png shot");
                std::process::exit(0);
            }
            name => match all.iter().find(|p| p.name == name) {
                Some(p) => args.preset = *p,
                None => {
                    let names: Vec<_> = all.iter().map(|p| p.name).collect();
                    fail(&format!(
                        "unknown preset {name:?}; expected one of {names:?}\n(run with --list for descriptions)"
                    ));
                }
            },
        }
        i += 1;
    }
    args
}

/// The lattice the flags ask for: the quality tier's, overridden where given.
fn cascades(args: &Args) -> ocean_fft::Cascades {
    ocean_fft::Cascades::default()
        .with_resolution(args.cascade.unwrap_or(args.quality.cascade_n))
        .with_max_scale(args.max_scale)
}

fn fail(msg: &str) -> ! {
    eprintln!("{msg}");
    std::process::exit(2);
}

fn report(world: &World, preset: &Preset, quality: &Quality) {
    println!("threers — ocean water");
    println!("  preset      {} ({})", preset.name, preset.blurb);
    println!("  quality     {}", quality.name);
    println!(
        "  sea state   {:.1} m/s wind, H_s {:.2} m, peak wavelength {:.0} m",
        preset.wind_speed, world.sea.significant_height, world.sea.peak_wavelength
    );
    println!(
        "  spectrum    {} cascades of {n}x{n}, tiles {:?} m, choppiness {:.2}",
        ocean_fft::CASCADES,
        world.cas.tiles,
        preset.choppiness,
        n = world.cas.n,
    );
    println!(
        "  grid        {} rings x {} sectors = {} vertices, displaced on the GPU",
        quality.rings,
        quality.sectors,
        world.base.len()
    );
}

// ---------------------------------------------------------------------------
// Entry points
// ---------------------------------------------------------------------------

fn main() {
    env_logger::init();
    let args = parse_args();
    if args.png.is_some() || args.bench || args.direct.is_some() {
        offscreen(&args);
    } else {
        pollster::block_on(windowed(args));
    }
}

/// Offscreen: one frame to a PNG, and/or the timing loop.
///
/// The PNG path is also the cheapest way to find out whether the WGSL still
/// compiles, which is why it exists at all.
fn offscreen(args: &Args) {
    let (w, h) = (1600u32, 900u32);
    let mut world = World::build_with(
        &args.preset,
        &args.quality,
        w as f32 / h as f32,
        cascades(args),
    );
    report(&world, &args.preset, &args.quality);

    let mut hr = HeadlessRenderer::builder()
        .size(w, h)
        .supersample(2)
        // Rgba8Unorm: the mesh shader already encodes sRGB itself.
        .color_format(wgpu::TextureFormat::Rgba8Unorm)
        .high_resolution(true)
        .build()
        .expect("headless renderer (is a GPU adapter available?)");
    // No TAA here on purpose: it needs a history of frames to accumulate, and a
    // one-shot still has none — it just renders a jittered frame against an empty
    // buffer. Supersampling is the right antialiasing for a still; TAA is for the
    // windowed path, where the frames keep coming.
    let device = hr.device().clone();
    let queue = hr.queue().clone();
    let waves = world.attach_waves(&device, &queue, hr.renderer());
    if let Some(f) = args.foam {
        let (e, c) = (world.preset.exposure, world.preset.choppiness);
        world.set_look(e, c, f);
    }
    if let Some(r) = args.mask {
        world.set_water_mask([0.0, 0.0], r, 12.0);
    }
    if args.no_waterline {
        if let Some(w) = &mut world.waterline {
            w.enabled = false;
        }
    }
    if let Some(c) = args.cam {
        world.camera.position = threers::Vector3::new(c[0], c[1], c[2]);
    }
    if let Some(l) = args.look {
        world
            .camera
            .look_at(threers::Vector3::new(l[0], l[1], l[2]));
    }

    // Before the timing loop, so `--png` produces the same frame with or without
    // `--bench` — the bench leaves the sea at whatever time it stopped at.
    if let Some(path) = &args.png {
        // Foam and wake are the one part of this ocean that carries state, so a
        // single frame would show them empty. Spin up a few seconds of it.
        for i in 0..90 {
            world.update(21.0 - (90 - i) as f32 / 60.0, &device, &queue, &waves);
        }
        world.update(21.0, &device, &queue, &waves);
        let rgba = hr.render_to_rgba_resolved(&mut world.scene, &world.camera);
        std::fs::write(path, encode_png(w, h, &rgba)).expect("write png");
        println!("  wrote       {path} ({w}x{h})");
    }
    if let Some(path) = &args.direct {
        direct(args, &device, &queue, path);
    }
    if args.bench {
        bench(&mut world, &device, &queue, &waves, &mut hr);
    }
}

/// Render the way a window — or a browser canvas — does, and read it back.
///
/// [`HeadlessRenderer`] draws through the renderer's render-to-target path,
/// which is *not* the path a swapchain uses: `render_to` hands the renderer a
/// sampleable colour and depth texture, and `render` does not. A picture only
/// ever checked through the first can be black through the second, and nobody
/// finds out until it is opened in a browser.
///
/// The format matters as much as the path. WebGPU canvases have no sRGB format,
/// so a browser gets `Bgra8Unorm` where a native window gets `Bgra8UnormSrgb` —
/// which is exactly the kind of difference that hides in a demo checked only on
/// one of them. This builds its own renderer so both can be reproduced here,
/// with no window and no screen.
fn direct(args: &Args, device: &Arc<wgpu::Device>, queue: &Arc<wgpu::Queue>, path: &str) {
    let (w, h) = (1280u32, 720u32);
    for (format, tag) in [
        (
            wgpu::TextureFormat::Bgra8Unorm,
            "bgra8unorm (browser canvas)",
        ),
        (
            wgpu::TextureFormat::Bgra8UnormSrgb,
            "bgra8unorm-srgb (native window)",
        ),
    ] {
        let mut renderer = Renderer::new(device.clone(), queue.clone(), format, w, h);
        let mut world = World::build_with(
            &args.preset,
            &args.quality,
            w as f32 / h as f32,
            cascades(args),
        );
        let waves = world.attach_waves(device, queue, &mut renderer);
        for i in 0..90 {
            world.update(21.0 - (90 - i) as f32 / 60.0, device, queue, &waves);
        }
        world.update(21.0, device, queue, &waves);

        let tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("direct target"),
            size: wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
        renderer.render(&mut world.scene, &world.camera, &view, false);

        let padded = (w * 4).div_ceil(256) * 256;
        let buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("direct readback"),
            size: (padded * h) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = device.create_command_encoder(&Default::default());
        enc.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &buf,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded),
                    rows_per_image: Some(h),
                },
            },
            wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
        );
        queue.submit([enc.finish()]);
        buf.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        let _ = device.poll(wgpu::PollType::wait_indefinitely());

        let data = buf.slice(..).get_mapped_range().expect("buffer range is mapped");
        let mut rgba = Vec::with_capacity((w * h * 4) as usize);
        let mut lit = 0u64;
        for y in 0..h {
            let row = (y * padded) as usize;
            for x in 0..w {
                let i = row + (x * 4) as usize;
                // Both formats here are BGRA on the wire.
                let (b, g, r, a) = (data[i], data[i + 1], data[i + 2], data[i + 3]);
                rgba.extend_from_slice(&[r, g, b, a]);
                lit += r as u64 + g as u64 + b as u64;
            }
        }
        drop(data);
        buf.unmap();

        let mean = lit as f64 / (w * h * 3) as f64;
        let out = if format == wgpu::TextureFormat::Bgra8Unorm {
            path.to_string()
        } else {
            path.replace(".png", "-srgb.png")
        };
        std::fs::write(&out, encode_png(w, h, &rgba)).expect("write png");
        println!("  direct      {tag}: mean luma {mean:.1} -> {out}");
        if mean < 0.5 {
            println!("              ^ black through Renderer::render");
        }
    }
}

/// Time the per-frame path.
///
/// Two numbers, because they answer different questions. *cpu* is the wall time
/// of everything this example does per frame — advancing the spectrum, writing
/// two small buffers, encoding the compute dispatch — and is what competes with
/// game logic. *frame* additionally waits for the GPU to finish, by reading the
/// target back, so it is the real end-to-end cost at this resolution.
fn bench(
    world: &mut World,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    waves: &waves_gpu::WaveCompute,
    hr: &mut HeadlessRenderer,
) {
    const WARMUP: u32 = 20;
    const FRAMES: u32 = 120;
    let step = |i: u32| 30.0 + i as f32 * 0.016;

    for i in 0..WARMUP {
        world.update(step(i), device, queue, waves);
        let _ = hr.render_to_rgba(&mut world.scene, &world.camera);
    }

    let t0 = Instant::now();
    for i in 0..FRAMES {
        world.update(step(i), device, queue, waves);
    }
    let cpu = t0.elapsed().as_secs_f64() * 1000.0 / FRAMES as f64;

    let t1 = Instant::now();
    for i in 0..FRAMES {
        world.update(step(i), device, queue, waves);
        // The readback is the only thing here that actually blocks on the GPU.
        let _ = hr.render_to_rgba(&mut world.scene, &world.camera);
    }
    let frame = t1.elapsed().as_secs_f64() * 1000.0 / FRAMES as f64;

    let (rw, rh) = hr.render_size();
    println!(
        "  bench       cpu {cpu:.3} ms/frame · frame {frame:.2} ms at {rw}x{rh} (readback included)"
    );

    // Kernel-only: submit a run of dispatches back to back and wait for the queue
    // to drain. The frame number above is fill-rate bound at this resolution and
    // says nothing about the compute, so it cannot be used to judge the kernels.
    const K: u32 = 400;
    // One encoder per iteration, drained at the end: what is timed is the kernel
    // plus one submission, which is the same shape the frame path uses.
    let disp = {
        let t = Instant::now();
        for _ in 0..K {
            let mut e = device.create_command_encoder(&Default::default());
            waves.dispatch(queue, &mut e, [0.0, 0.0]);
            queue.submit([e.finish()]);
        }
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        t.elapsed().as_secs_f64() * 1000.0 / K as f64
    };

    let mut foam = 0.0;
    if let Some(state) = &mut world.state {
        let t = Instant::now();
        for _ in 0..K {
            let mut e = device.create_command_encoder(&Default::default());
            state.dispatch(queue, &mut e, 1.0 / 60.0, [1.0, 0.0], 0.1, 0.3, 1.2, &[]);
            queue.submit([e.finish()]);
        }
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        foam = t.elapsed().as_secs_f64() * 1000.0 / K as f64;
    }

    let mut casc = 0.0;
    if let Some(fft) = &world.fft {
        let t = Instant::now();
        for i in 0..K {
            let mut e = device.create_command_encoder(&Default::default());
            fft.dispatch(queue, &mut e, 30.0 + i as f32 * 0.016, 1.0);
            queue.submit([e.finish()]);
        }
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        casc = t.elapsed().as_secs_f64() * 1000.0 / K as f64;
    }
    println!(
        "  kernels     cascades {casc:.3} ms · displace {disp:.3} ms · \
         surface-state {foam:.3} ms (GPU, queue-drained)"
    );
}

struct Input {
    rotating: bool,
    panning: bool,
    last: Option<PhysicalPosition<f64>>,
}

impl Input {
    fn motion(&mut self, pos: PhysicalPosition<f64>) -> (f32, f32) {
        let d = match self.last {
            Some(l) => ((pos.x - l.x) as f32, (pos.y - l.y) as f32),
            None => (0.0, 0.0),
        };
        self.last = Some(pos);
        d
    }
}

async fn windowed(args: Args) {
    let event_loop = EventLoop::new().expect("event loop");
    let window = Arc::new(
        WindowBuilder::new()
            .with_title(format!("threers — ocean water ({})", args.preset.name))
            .with_inner_size(winit::dpi::LogicalSize::new(1280, 720))
            .build(&event_loop)
            .expect("window"),
    );

    let instance = {
        // wgpu 30 dropped `Default` here; the display handle is only
        // consulted by GLES/Wayland, not Vulkan, Metal or DX12.
        let mut d = wgpu::InstanceDescriptor::new_without_display_handle();
        d.backends = wgpu::Backends::PRIMARY;
        wgpu::Instance::new(d)
    };
    let surface = instance.create_surface(window.clone()).expect("surface");
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
            apply_limit_buckets: false,
        })
        .await
        .expect("adapter");
    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: Some("threers device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::downlevel_defaults(),
                ..Default::default()
            },
        )
        .await
        .expect("device");
    let device = Arc::new(device);
    let queue = Arc::new(queue);

    let size = window.inner_size();
    let caps = surface.get_capabilities(&adapter);
    let format = caps
        .formats
        .iter()
        .copied()
        .find(|f| f.is_srgb())
        .unwrap_or(caps.formats[0]);
    let mut config = wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format,
        width: size.width.max(1),
        height: size.height.max(1),
        present_mode: caps.present_modes[0],
        alpha_mode: caps.alpha_modes[0],
        view_formats: vec![],
        desired_maximum_frame_latency: 2,
        color_space: wgpu::SurfaceColorSpace::Srgb,
    };
    surface.configure(&device, &config);

    let mut renderer = Renderer::new(
        device.clone(),
        queue.clone(),
        format,
        config.width,
        config.height,
    );

    let mut world = World::build_with(
        &args.preset,
        &args.quality,
        config.width as f32 / config.height as f32,
        cascades(&args),
    );
    let mut waves = world.attach_waves(&device, &queue, &mut renderer);
    report(&world, &args.preset, &args.quality);
    println!("  left-drag orbits · right-drag pans · scroll zooms");

    renderer.set_taa(true);

    let mut controls = OrbitControls::new(&world.camera);
    controls.min_distance = 8.0;
    controls.max_distance = 2500.0;
    // Let the camera go under. The surface is double-sided and shades from below,
    // so there is something down there now.
    controls.max_polar_angle = 3.05;
    controls.min_distance = 2.0;

    let mut input = Input {
        rotating: false,
        panning: false,
        last: None,
    };
    let start = Instant::now();
    let win = window.clone();
    let mut fps_frames = 0u32;
    let mut fps_since = Instant::now();

    // Live state, so the keys below can drive the runtime API rather than
    // needing the example restarted. This is the same API the browser build
    // exposes to JavaScript — exercised here so it cannot rot untested.
    let all_presets = preset::all();
    let mut cur_preset = args.preset;
    let mut cur_quality = args.quality;
    let mut cur_cas = cascades(&args);
    println!("  keys        1-8 preset · [ ] quality · c cascades · , . wind dir · - = wind speed");
    println!("              s w sun · k clouds · e r exposure · o p choppiness · h surface report");

    event_loop
        .run(move |event, target| {
            let Event::WindowEvent { event, window_id } = event else {
                return;
            };
            if window_id != win.id() {
                return;
            }
            match event {
                WindowEvent::CloseRequested => target.exit(),
                WindowEvent::Resized(s) => {
                    config.width = s.width.max(1);
                    config.height = s.height.max(1);
                    surface.configure(&device, &config);
                    renderer.resize(config.width, config.height);
                    world
                        .camera
                        .set_aspect(config.width as f32 / config.height as f32);
                }
                WindowEvent::MouseInput { state, button, .. } => {
                    let pressed = state == ElementState::Pressed;
                    match button {
                        MouseButton::Left => input.rotating = pressed,
                        MouseButton::Right => input.panning = pressed,
                        _ => {}
                    }
                    if !pressed {
                        input.last = None;
                    }
                }
                WindowEvent::CursorMoved { position, .. } => {
                    if !input.rotating && !input.panning {
                        return;
                    }
                    let (dx, dy) = input.motion(position);
                    controls.update(
                        PointerEvent {
                            dx,
                            dy,
                            wheel: 0.0,
                            rotating: input.rotating,
                            panning: input.panning,
                        },
                        &mut world.camera,
                        (config.width as f32, config.height as f32),
                    );
                }
                WindowEvent::MouseWheel { delta, .. } => {
                    let wheel = match delta {
                        MouseScrollDelta::LineDelta(_, y) => y * 40.0,
                        MouseScrollDelta::PixelDelta(p) => p.y as f32,
                    };
                    controls.update(
                        PointerEvent {
                            dx: 0.0,
                            dy: 0.0,
                            wheel,
                            rotating: false,
                            panning: false,
                        },
                        &mut world.camera,
                        (config.width as f32, config.height as f32),
                    );
                }
                WindowEvent::KeyboardInput { event: key, .. } => {
                    if key.state != ElementState::Pressed {
                        return;
                    }
                    let Key::Character(c) = key.logical_key.as_ref() else {
                        return;
                    };
                    // The three costs, kept visibly distinct: `rebuild` recreates
                    // the world, `set_wind` rewrites one texture, and the rest are
                    // uniform slots.
                    let mut rebuild = false;
                    match c {
                        "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" => {
                            let i = c.parse::<usize>().unwrap_or(1) - 1;
                            if i < all_presets.len() {
                                cur_preset = all_presets[i];
                                cur_cas = cur_cas.with_resolution(cur_quality.cascade_n);
                                rebuild = true;
                            }
                        }
                        "[" | "]" => {
                            let i = QUALITY_LEVELS
                                .iter()
                                .position(|q| q.name == cur_quality.name)
                                .unwrap_or(2);
                            let n = if c == "]" {
                                (i + 1).min(QUALITY_LEVELS.len() - 1)
                            } else {
                                i.saturating_sub(1)
                            };
                            cur_quality = QUALITY_LEVELS[n];
                            cur_cas = cur_cas.with_resolution(cur_quality.cascade_n);
                            rebuild = true;
                        }
                        "c" => {
                            let n = match cur_cas.n {
                                n if n < 256 => 256,
                                256 => 512,
                                _ => 128,
                            };
                            cur_cas = cur_cas.with_resolution(n);
                            rebuild = true;
                        }
                        "," | "." => {
                            let d = world.preset.wind_dir + if c == "." { 0.2 } else { -0.2 };
                            let sp = world.preset.wind_speed;
                            world.set_wind(&queue, d, sp);
                            println!(
                                "  wind        {:.2} rad, {sp:.1} m/s",
                                world.preset.wind_dir
                            );
                        }
                        "e" | "r" => {
                            let e = (world.preset.exposure + if c == "r" { 0.08 } else { -0.08 })
                                .max(0.02);
                            let (ch, fo) = (world.preset.choppiness, world.preset.foam_amount);
                            world.set_look(e, ch, fo);
                            println!("  exposure    {e:.2}");
                        }
                        "o" | "p" => {
                            let ch = world.preset.choppiness + if c == "p" { 0.1 } else { -0.1 };
                            let (e, fo) = (world.preset.exposure, world.preset.foam_amount);
                            world.set_look(e, ch, fo);
                            println!("  choppiness  {:.2}", world.preset.choppiness);
                        }
                        "s" | "w" => {
                            let p = world.preset;
                            let elev = (p.sun_elevation + if c == "w" { 3.0 } else { -3.0 })
                                .clamp(-6.0, 89.0);
                            world.set_sky(
                                elev,
                                p.sun_azimuth,
                                p.turbidity,
                                p.rayleigh,
                                p.mie_coefficient,
                                p.mie_directional_g,
                                p.clouds,
                            );
                            println!("  sun         {elev:.1}° elevation");
                        }
                        "k" => {
                            let p = world.preset;
                            let clouds = match p.clouds {
                                x if x < 0.05 => 0.45,
                                x if x < 0.7 => 0.95,
                                _ => 0.0,
                            };
                            world.set_sky(
                                p.sun_elevation,
                                p.sun_azimuth,
                                p.turbidity,
                                p.rayleigh,
                                p.mie_coefficient,
                                p.mie_directional_g,
                                clouds,
                            );
                            println!("  clouds      {clouds:.2}");
                        }
                        "h" => {
                            let t = start.elapsed().as_secs_f32();
                            let cam = world.camera.position;
                            let (y, g) = world.sample_surface(cam.x, cam.z, t);
                            let probed = world.probe.as_ref().map(|p| p.read().len()).unwrap_or(0);
                            println!(
                                "  surface     {y:+.2} m, slope ({:+.3}, {:+.3}) under the camera \
                                 · quality {} · {probed} probe taps live",
                                g[0], g[1], world.quality.name
                            );
                        }
                        "-" | "=" => {
                            let sp = world.preset.wind_speed + if c == "=" { 1.0 } else { -1.0 };
                            let d = world.preset.wind_dir;
                            world.set_wind(&queue, d, sp);
                            println!(
                                "  wind        {d:.2} rad, {:.1} m/s, H_s {:.2} m",
                                world.preset.wind_speed, world.sea.significant_height
                            );
                        }
                        _ => {}
                    }
                    if rebuild {
                        let t0 = Instant::now();
                        waves = world.reconfigure(
                            &device,
                            &queue,
                            &mut renderer,
                            &cur_preset,
                            &cur_quality,
                            cur_cas,
                        );
                        println!(
                            "  rebuilt     {} · {} · {}² cascades in {:.0} ms",
                            cur_preset.name,
                            cur_quality.name,
                            cur_cas.n,
                            t0.elapsed().as_secs_f64() * 1000.0
                        );
                    }
                }
                WindowEvent::RedrawRequested => {
                    // The compute dispatch is submitted here, before the draw
                    // that reads its output; queue order does the rest.
                    world.update(start.elapsed().as_secs_f32(), &device, &queue, &waves);
                    match surface.get_current_texture() {
                        wgpu::CurrentSurfaceTexture::Success(frame)
                            | wgpu::CurrentSurfaceTexture::Suboptimal(frame) => {
                            let view = frame
                                .texture
                                .create_view(&wgpu::TextureViewDescriptor::default());
                            renderer.render(&mut world.scene, &world.camera, &view, false);
                            queue.present(frame);
                        }
                        wgpu::CurrentSurfaceTexture::Lost | wgpu::CurrentSurfaceTexture::Outdated => {
                            surface.configure(&device, &config);
                        }
                        e => log::error!("surface error: {e:?}"),
                    }
                    // Frame rate in the title bar: the bench number is
                    // readback-bound and says little about the interactive path.
                    fps_frames += 1;
                    let dt = fps_since.elapsed().as_secs_f32();
                    if dt >= 0.5 {
                        win.set_title(&format!(
                            "threers — ocean water ({}) · {:.0} fps{}",
                            cur_preset.name,
                            fps_frames as f32 / dt,
                            if world.submerged {
                                " · underwater"
                            } else {
                                ""
                            }
                        ));
                        fps_frames = 0;
                        fps_since = Instant::now();
                    }
                    win.request_redraw();
                }
                _ => {}
            }
        })
        .expect("event loop");
}
