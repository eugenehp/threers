//! The Metal backend off the main thread.
//!
//! Every renderer here draws on a thread spawned with `std::thread`, which —
//! unlike the main thread or a GCD queue — has no autorelease pool of its own.
//! An Objective-C object autoreleased there with no pool on the stack is not
//! freed until the thread exits, which for a render thread is a leak per frame.
//! `no_object_is_autoreleased_without_a_pool` reruns this file under
//! `OBJC_DEBUG_MISSING_POOLS=YES`, which has the runtime report each one.
//!
//! The pixels are checked too: a frame drawn on another thread, or after the
//! renderer changed threads, is the frame drawn on this one, byte for byte.

#![cfg(metal_backend)]

use std::ffi::c_void;
use std::sync::{mpsc, Arc, Barrier};
use std::thread;

use threers::cameras::PerspectiveCamera;
use threers::core::{Mesh, Object3D};
use threers::geometries::{BoxGeometry, PlaneGeometry};
use threers::lights::{AmbientLight, DirectionalLight};
use threers::materials::{BasicMaterial, Material, StandardMaterial};
use threers::math::{Color, Vector3};
use threers::metal::enums::pixel_format;
use threers::metal::objc::{get_class, get_sel, msg0, AutoreleasePool, Owned};
use threers::metal::{
    MetalDevice, MetalError, MetalHeadlessRenderer, MetalJobError, MetalRenderPool,
    MetalRenderPoolConfig, MetalRenderTarget, MetalRenderer, MetalSurface, MetalWorker,
};
use threers::scene::Scene;
use threers::textures::{Texture, TextureFormat};

const SIZE: u32 = 48;

/// Under the pool check, name this thread as one that runs threers code.
///
/// Metal and Core Animation autorelease on their own GCD threads too — a
/// command buffer finishing drops its resources there — and those reports are
/// Apple's, not this backend's. The check counts only threads named here.
fn track() {
    unsafe extern "C" {
        fn pthread_self() -> usize;
    }
    if std::env::var_os("THREERS_POOL_CHECK").is_some() {
        eprintln!("THREERS_THREAD {:#x}", unsafe { pthread_self() });
    }
}

/// `thread::spawn`, tracked.
fn spawn<T: Send + 'static>(f: impl FnOnce() -> T + Send + 'static) -> thread::JoinHandle<T> {
    thread::spawn(move || {
        track();
        f()
    })
}

fn device() -> Option<MetalDevice> {
    match MetalDevice::new() {
        Ok(d) => Some(d),
        Err(MetalError::NoDevice) => {
            eprintln!("skipping: no Metal device on this machine");
            None
        }
        Err(e) => panic!("Metal device could not be built: {e}"),
    }
}

/// A box over a floor, shadowed, with a texture and a transparent pane: the
/// upload, shadow, blend and readback paths in one frame.
fn scene() -> Scene {
    let mut s = Scene::new();
    s.background = Color::from_hex(0x102030);
    s.add_light(AmbientLight::new(Color::WHITE, 0.3));
    let mut light = DirectionalLight::new(Color::WHITE, 2.5);
    light.direction = Vector3::new(-0.3, -1.0, -0.4);
    light.cast_shadow = true;
    light.shadow.map_size = 256;
    s.add_light(light);

    let mut floor = Object3D::mesh(Mesh::new(
        PlaneGeometry::new(6.0, 6.0),
        StandardMaterial::new(Color::WHITE).into(),
    ));
    floor.rotate_x(-std::f32::consts::FRAC_PI_2);
    floor.receive_shadow = true;
    s.add(floor);

    let checker: Vec<u8> = (0..16)
        .flat_map(|i| {
            if (i + i / 4) % 2 == 0 {
                [255, 128, 0, 255]
            } else {
                [0, 64, 255, 255]
            }
        })
        .collect();
    let mut mat = StandardMaterial::new(Color::WHITE);
    mat.map = Some(Arc::new(Texture::new(
        4,
        4,
        TextureFormat::Rgba8UnormSrgb,
        checker,
    )));
    let mut cube = Object3D::mesh(Mesh::new(BoxGeometry::new(1.0, 1.0, 1.0), mat.into()));
    cube.position = Vector3::new(0.0, 0.5, 0.0);
    cube.cast_shadow = true;
    s.add(cube);

    let mut glass = BasicMaterial::new(Color::from_hex(0xff0000));
    glass.transparent = true;
    glass.opacity = 0.4;
    let mut pane = Object3D::mesh(Mesh::new(
        PlaneGeometry::new(1.0, 1.0),
        Material::Basic(glass),
    ));
    pane.position = Vector3::new(0.6, 0.8, 1.0);
    s.add(pane);
    s
}

fn camera() -> PerspectiveCamera {
    let mut c = PerspectiveCamera::new(50.0, 1.0, 0.1, 100.0);
    c.position = Vector3::new(2.5, 2.5, 3.5);
    c.target = Vector3::new(0.0, 0.3, 0.0);
    c
}

/// A renderer and the target it draws into.
struct Rig {
    renderer: MetalRenderer,
    target: MetalRenderTarget,
}

impl Rig {
    fn new(device: &MetalDevice) -> Self {
        Self {
            renderer: MetalRenderer::with_device(device.clone()).unwrap(),
            target: MetalRenderTarget::new(device, SIZE, SIZE, pixel_format::RGBA8_UNORM_SRGB, 4)
                .unwrap(),
        }
    }

    fn draw(&mut self, scene: &mut Scene) -> Vec<u8> {
        self.renderer
            .render(scene, &camera(), &self.target.attachments())
            .unwrap();
        self.target.read_rgba(self.renderer.device()).unwrap()
    }
}

/// The frame every test compares against, drawn on the test's own thread.
fn reference(device: &MetalDevice) -> Vec<u8> {
    let rgba = Rig::new(device).draw(&mut scene());
    // Not a blank frame: the background, the lit floor and the cube all show.
    let distinct: std::collections::HashSet<_> = rgba.as_chunks::<4>().0.iter().collect();
    assert!(distinct.len() > 20, "reference frame is nearly flat");
    rgba
}

fn headless() -> MetalHeadlessRenderer {
    let mut hr = MetalHeadlessRenderer::builder()
        .size(SIZE, SIZE)
        .msaa(4)
        .build()
        .unwrap();
    hr.set_bloom(0.6, 0.5, 0.3);
    hr
}

#[test]
fn renders_on_a_thread_without_a_pool() {
    track();
    let Some(device) = device() else { return };
    let want = reference(&device);
    let got = spawn(move || {
        let mut rig = Rig::new(&device);
        let mut scene = scene();
        (0..8).map(|_| rig.draw(&mut scene)).last().unwrap()
    })
    .join()
    .unwrap();
    assert_eq!(got, want, "a frame drawn off the test thread differs");
}

#[test]
fn a_renderer_built_on_one_thread_draws_on_another() {
    track();
    let Some(device) = device() else { return };
    let want = reference(&device);
    // Built here, with its caches warm, then handed over.
    let mut rig = Rig::new(&device);
    let mut scene = scene();
    assert_eq!(rig.draw(&mut scene), want);
    let (mut rig, mut scene) = spawn(move || {
        assert_eq!(
            rig.draw(&mut scene),
            want,
            "after moving to the render thread"
        );
        (rig, scene)
    })
    .join()
    .unwrap();
    // And back again.
    assert_eq!(rig.draw(&mut scene), reference(rig.renderer.device()));
}

#[test]
fn a_renderer_can_change_threads_every_frame() {
    track();
    let Some(device) = device() else { return };
    let want = reference(&device);
    // Three workers pass one renderer round a ring, a frame each, the way a
    // job system hands a render task to whichever thread is free.
    const WORKERS: usize = 3;
    const FRAMES: usize = 12;
    let (senders, receivers): (Vec<_>, Vec<_>) = (0..WORKERS)
        .map(|_| mpsc::channel::<(Rig, Scene, usize)>())
        .unzip();
    let (done_tx, done_rx) = mpsc::channel();
    let handles: Vec<_> = receivers
        .into_iter()
        .enumerate()
        .map(|(i, rx)| {
            let next = senders[(i + 1) % WORKERS].clone();
            let done = done_tx.clone();
            let want = want.clone();
            spawn(move || {
                while let Ok((mut rig, mut scene, frame)) = rx.recv() {
                    assert_eq!(rig.draw(&mut scene), want, "frame {frame} on worker {i}");
                    if frame + 1 == FRAMES {
                        done.send(rig).unwrap();
                        return;
                    }
                    next.send((rig, scene, frame + 1)).unwrap();
                }
            })
        })
        .collect();
    senders[0].send((Rig::new(&device), scene(), 0)).unwrap();
    let rig = done_rx.recv().unwrap();
    drop(senders);
    for h in handles {
        h.join().unwrap();
    }
    // Twelve frames on three threads cached what one frame on one thread does.
    let mut once = Rig::new(&device);
    once.draw(&mut scene());
    assert_eq!(rig.renderer.cache_sizes(), once.renderer.cache_sizes());
}

#[test]
fn threads_draw_at_once_on_one_device() {
    track();
    let Some(device) = device() else { return };
    let want = Arc::new(reference(&device));
    // One device and one command queue, shared; a renderer per thread, all
    // encoding and committing at the same time.
    let device = Arc::new(device);
    const THREADS: usize = 6;
    let start = Arc::new(Barrier::new(THREADS));
    let handles: Vec<_> = (0..THREADS)
        .map(|t| {
            let (device, start, want) = (device.clone(), start.clone(), want.clone());
            spawn(move || {
                let mut rig = Rig::new(&device);
                let mut scene = scene();
                start.wait();
                for frame in 0..10 {
                    assert!(rig.draw(&mut scene) == *want, "thread {t}, frame {frame}");
                }
            })
        })
        .collect();
    for h in handles {
        h.join().unwrap();
    }
}

#[test]
fn a_headless_renderer_moves_with_a_readback_in_flight() {
    track();
    let Some(_) = device() else { return };
    // Same calls on one thread and spread over two must give the same frames,
    // including the post chain's readback that is queued on one call and
    // collected on the next.
    let frames = |split: bool| {
        let mut hr = headless();
        let mut scene = scene();
        let mut out: Vec<Vec<u8>> = (0..3)
            .map(|_| hr.render_to_rgba(&mut scene, &camera()).unwrap())
            .collect();
        let rest = |hr: &mut MetalHeadlessRenderer, scene: &mut Scene| {
            (0..3)
                .map(|_| hr.render_to_rgba(scene, &camera()).unwrap())
                .collect::<Vec<_>>()
        };
        if split {
            out.extend(spawn(move || rest(&mut hr, &mut scene)).join().unwrap());
        } else {
            out.extend(rest(&mut hr, &mut scene));
        }
        out
    };
    assert_eq!(frames(true), frames(false));
}

#[test]
fn renderers_drop_cleanly_on_other_threads() {
    track();
    let Some(device) = device() else { return };
    // Made here, dropped there — and the reverse — with their GPU work
    // possibly still running.
    let mut rig = Rig::new(&device);
    rig.renderer
        .render(&mut scene(), &camera(), &rig.target.attachments())
        .unwrap();
    spawn(move || drop(rig)).join().unwrap();
    let rig = spawn(move || {
        let mut rig = Rig::new(&device);
        rig.renderer
            .render(&mut scene(), &camera(), &rig.target.attachments())
            .unwrap();
        rig
    })
    .join()
    .unwrap();
    drop(rig);
}

/// `[CAMetalLayer layer]`, retained — what a `UIView` with
/// `layerClass = CAMetalLayer.self` carries.
fn metal_layer() -> Owned {
    let _pool = AutoreleasePool::new();
    unsafe {
        let layer = msg0(get_class("CAMetalLayer\0"), get_sel("layer\0"));
        Owned::retain(layer).expect("CAMetalLayer")
    }
}

#[test]
fn a_surface_presents_from_a_render_thread() {
    track();
    let Some(device) = device() else { return };
    let layer = metal_layer();
    let layer_ptr = layer.id() as usize;
    spawn(move || {
        // Adopting the layer, sizing it and drawing are all render-thread work.
        let mut surface =
            unsafe { MetalSurface::from_layer(&device, layer_ptr as *mut c_void, 64, 32, 4) }
                .unwrap();
        let mut renderer = MetalRenderer::with_device(device.clone()).unwrap();
        let mut scene = scene();
        let mut presented = 0;
        for frame in 0..6 {
            if frame == 3 {
                surface.resize(80, 40).unwrap();
            }
            if let Some(f) = surface.next_frame() {
                renderer
                    .render(&mut scene, &camera(), &f.attachments())
                    .unwrap();
                presented += 1;
            }
        }
        assert!(presented > 0, "the layer never vended a drawable");
        assert_eq!(surface.size(), (80, 40));
        // The surface itself can change threads too.
        spawn(move || {
            let f = surface
                .next_frame()
                .expect("drawable after moving the surface");
            renderer
                .render(&mut scene, &camera(), &f.attachments())
                .unwrap();
        })
        .join()
        .unwrap();
    })
    .join()
    .unwrap();
    drop(layer);
}

#[test]
fn view_calls_refuse_to_run_off_the_main_thread() {
    track();
    let Some(device) = device() else { return };
    spawn(move || {
        let err = unsafe { MetalSurface::from_ns_view(&device, std::ptr::null_mut(), 8, 8, 1) }
            .unwrap_err();
        assert!(
            matches!(&err, MetalError::Unsupported(m) if m.contains("main thread")),
            "{err}"
        );
        #[cfg(any(target_os = "ios", target_os = "tvos", target_os = "visionos"))]
        {
            let err = unsafe { MetalSurface::from_ui_view(&device, std::ptr::null_mut(), 8, 8, 1) }
                .unwrap_err();
            assert!(
                matches!(&err, MetalError::Unsupported(m) if m.contains("main thread")),
                "{err}"
            );
        }
    })
    .join()
    .unwrap();
}

#[test]
fn a_layer_that_is_not_metal_is_refused() {
    track();
    let Some(device) = device() else { return };
    let _pool = AutoreleasePool::new();
    let plain =
        unsafe { msg0::<threers::metal::objc::Id>(get_class("CALayer\0"), get_sel("layer\0")) };
    for layer in [plain, std::ptr::null_mut()] {
        let err = unsafe { MetalSurface::from_layer(&device, layer.cast(), 8, 8, 1) }.unwrap_err();
        assert!(
            matches!(&err, MetalError::Unsupported(m) if m.contains("CAMetalLayer")),
            "{err}"
        );
    }
}

#[cfg(feature = "fluid")]
#[test]
fn the_fluid_steps_on_another_thread() {
    track();
    use threers::fluid::FluidConfig;
    use threers::metal::{fluid_shader_source, MetalFluid};
    let Some(_) = device() else { return };
    let device = MetalDevice::with_shader_source(&fluid_shader_source()).unwrap();
    let config = FluidConfig {
        bloom: false,
        sunrays: false,
        colorful: false,
        ..FluidConfig::default()
    };
    let run = |device: &MetalDevice, fluid: &mut MetalFluid| {
        for _ in 0..4 {
            fluid.step(device, 1.0 / 60.0).unwrap();
        }
        fluid.capture_rgba(device).unwrap()
    };
    let mut here = MetalFluid::new(&device, 64, 64, config).unwrap();
    let mut there = MetalFluid::new(&device, 64, 64, config).unwrap();
    let want = run(&device, &mut here);
    let got = spawn(move || run(&device, &mut there)).join().unwrap();
    assert_eq!(got, want);
}

/// One job on a pool worker: draw [`scene`] into the worker's own target.
/// Returns the worker, the frame, and how many textures the frame uploaded.
fn pool_frame(worker: &mut MetalWorker) -> (usize, Vec<u8>, usize) {
    track();
    let device = worker.device().clone();
    let target: &mut Option<MetalRenderTarget> = worker.local();
    let attachments = target
        .get_or_insert_with(|| {
            MetalRenderTarget::new(&device, SIZE, SIZE, pixel_format::RGBA8_UNORM_SRGB, 4).unwrap()
        })
        .attachments();
    // The scene is the worker's too, so its texture is the same object from
    // job to job and the renderer's upload cache can recognise it.
    let mut scene = worker.local::<Option<Scene>>().take().unwrap_or_else(scene);
    let stats = worker
        .renderer()
        .unwrap()
        .render(&mut scene, &camera(), &attachments)
        .unwrap();
    *worker.local::<Option<Scene>>() = Some(scene);
    let target: &mut Option<MetalRenderTarget> = worker.local();
    let rgba = target.as_ref().unwrap().read_rgba(&device).unwrap();
    (worker.index(), rgba, stats.texture_uploads as usize)
}

#[test]
fn a_pool_draws_the_same_frame_on_every_worker() {
    track();
    let Some(device) = device() else { return };
    let want = reference(&device);
    let pool = MetalRenderPool::new(&device, MetalRenderPoolConfig::with_threads(3));
    let jobs: Vec<_> = (0..24).map(|_| pool.submit(pool_frame).unwrap()).collect();
    let mut workers = std::collections::HashSet::new();
    let mut uploads = 0;
    for (i, job) in jobs.into_iter().enumerate() {
        let (worker, rgba, up) = job.wait().unwrap();
        assert!(rgba == want, "job {i} on worker {worker}");
        workers.insert(worker);
        uploads += up;
    }
    assert!(workers.iter().all(|&w| w < 3));
    // Each worker's renderer is kept between jobs: the texture went up once
    // per worker that ran a job, not once per job.
    assert_eq!(uploads, workers.len(), "workers {workers:?}");
}

#[test]
fn a_full_pool_refuses_or_waits() {
    track();
    let Some(device) = device() else { return };
    let config = MetalRenderPoolConfig {
        queue_depth: 0,
        ..MetalRenderPoolConfig::with_threads(1)
    };
    let pool = MetalRenderPool::new(&device, config);
    let (started_tx, started_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel::<()>();
    let busy = pool
        .submit(move |_: &mut MetalWorker| {
            started_tx.send(()).unwrap();
            release_rx.recv().unwrap();
            7
        })
        .unwrap();
    started_rx.recv().unwrap();
    // The one worker is busy and there is no queue: drop the frame.
    assert_eq!(
        pool.try_submit(|_: &mut MetalWorker| ()).err(),
        Some(MetalJobError::QueueFull)
    );
    release_tx.send(()).unwrap();
    assert_eq!(busy.wait(), Ok(7));
    // `submit` waits for the worker instead.
    assert_eq!(
        pool.submit(pool_frame).unwrap().wait().unwrap().1,
        reference(&device)
    );
}

#[test]
fn a_panicking_job_does_not_take_its_worker_down() {
    track();
    let Some(device) = device() else { return };
    let pool = MetalRenderPool::new(&device, MetalRenderPoolConfig::with_threads(1));
    pool.submit(pool_frame).unwrap().wait().unwrap();
    let err = pool
        .submit(|w: &mut MetalWorker| {
            w.renderer().unwrap();
            panic!("mid-frame")
        })
        .unwrap()
        .wait()
        .unwrap_err();
    assert!(
        matches!(&err, MetalJobError::Panicked(m) if m == "mid-frame"),
        "{err}"
    );
    // Same worker, fresh renderer: it uploads the texture again and draws the
    // same frame.
    let (_, rgba, uploads) = pool.submit(pool_frame).unwrap().wait().unwrap();
    assert_eq!(rgba, reference(&device));
    assert_eq!(uploads, 1);
}

#[test]
fn dropping_a_pool_finishes_its_queue() {
    track();
    let Some(device) = device() else { return };
    let pool = MetalRenderPool::new(
        &device,
        MetalRenderPoolConfig {
            queue_depth: 8,
            ..MetalRenderPoolConfig::with_threads(2)
        },
    );
    let jobs: Vec<_> = (0..8).map(|_| pool.submit(pool_frame).unwrap()).collect();
    drop(pool);
    let want = reference(&device);
    for job in jobs {
        assert_eq!(job.wait().unwrap().1, want);
    }
}

#[test]
fn the_default_pool_fits_the_platform() {
    track();
    let config = MetalRenderPoolConfig::for_this_device();
    let cap = if cfg!(target_os = "macos") { 4 } else { 2 };
    assert!((1..=cap).contains(&config.threads), "{config:?}");
    assert_eq!(config.queue_depth, config.threads);
    assert_eq!(MetalRenderPoolConfig::with_threads(0).threads, 1);
}

/// Reruns the rest of this file with the runtime reporting every object
/// autoreleased where no pool is in place.
#[cfg(target_os = "macos")]
#[test]
fn no_object_is_autoreleased_without_a_pool() {
    track();
    if std::env::var_os("THREERS_POOL_CHECK").is_some() || device().is_none() {
        return;
    }
    let out = std::process::Command::new(std::env::current_exe().unwrap())
        .args([
            "--skip",
            "no_object_is_autoreleased_without_a_pool",
            "--test-threads",
            "4",
            "--nocapture",
        ])
        .env("OBJC_DEBUG_MISSING_POOLS", "YES")
        .env("THREERS_POOL_CHECK", "1")
        .output()
        .unwrap();
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "the rerun failed:\n{stderr}");
    let ours: std::collections::HashSet<&str> = stderr
        .lines()
        .filter_map(|l| l.strip_prefix("THREERS_THREAD "))
        .collect();
    assert!(
        ours.len() > 10,
        "the rerun named too few threads: {}",
        ours.len()
    );
    // `objc[pid]: MISSING POOLS: (0x16d42f000) Object …` — the thread is the
    // parenthesised `pthread_t`.
    let missing: Vec<_> = stderr
        .lines()
        .filter(|l| l.contains("MISSING POOLS"))
        .filter(|l| {
            l.split('(')
                .nth(1)
                .and_then(|r| r.split(')').next())
                .is_some_and(|t| ours.contains(t))
        })
        .collect();
    assert!(
        missing.is_empty(),
        "{} objects autoreleased with no pool, first:\n{}",
        missing.len(),
        missing
            .iter()
            .take(5)
            .copied()
            .collect::<Vec<_>>()
            .join("\n")
    );
}
