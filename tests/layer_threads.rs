//! `LayerRenderer` off the main thread: drawing an app's `CAMetalLayer` from
//! render threads, handing a renderer between threads, several layers on one
//! device at once — and no autoreleased object left without a pool.
//!
//! The layers here are made with `[CAMetalLayer layer]` and never put in a
//! window, which is enough for Metal to vend drawables, so this runs headless
//! on a Mac and in the iOS, tvOS and visionOS simulators alike.

#![cfg(any(target_os = "macos", target_os = "ios", target_os = "tvos", target_os = "visionos"))]

use std::ffi::{c_char, c_void};
use std::sync::{Arc, Barrier};
use std::thread;

use threers::cameras::PerspectiveCamera;
use threers::core::{Mesh, Object3D};
use threers::geometries::BoxGeometry;
use threers::lights::DirectionalLight;
use threers::materials::StandardMaterial;
use threers::math::{Color, Vector3};
use threers::renderer::{LayerGpu, LayerRenderer};
use threers::scene::Scene;

#[link(name = "objc", kind = "dylib")]
extern "C" {
    fn objc_getClass(name: *const c_char) -> *mut c_void;
    fn sel_registerName(name: *const c_char) -> *const c_void;
    fn objc_msgSend();
    fn objc_retain(obj: *mut c_void) -> *mut c_void;
    fn objc_release(obj: *mut c_void);
    fn objc_autoreleasePoolPush() -> *mut c_void;
    fn objc_autoreleasePoolPop(pool: *mut c_void);
}

#[link(name = "QuartzCore", kind = "framework")]
extern "C" {}

/// A retained `+[class layer]`, released on drop. Held as an address so it
/// can be handed to the thread that draws it.
struct Layer(usize);

impl Layer {
    fn new(class: &std::ffi::CStr) -> Self {
        unsafe {
            let pool = objc_autoreleasePoolPush();
            let send: unsafe extern "C" fn(*mut c_void, *const c_void) -> *mut c_void =
                std::mem::transmute(objc_msgSend as unsafe extern "C" fn());
            let layer = send(objc_getClass(class.as_ptr()), sel_registerName(c"layer".as_ptr()));
            assert!(!layer.is_null(), "{class:?}");
            let layer = objc_retain(layer);
            objc_autoreleasePoolPop(pool);
            Self(layer as usize)
        }
    }

    fn metal() -> Self {
        Self::new(c"CAMetalLayer")
    }

    fn ptr(&self) -> *mut c_void {
        self.0 as *mut c_void
    }
}

impl Drop for Layer {
    fn drop(&mut self) {
        unsafe {
            let pool = objc_autoreleasePoolPush();
            objc_release(self.ptr());
            objc_autoreleasePoolPop(pool);
        }
    }
}

/// Under the pool check, name this thread as one that runs threers code (see
/// `tests/metal_threads.rs`: Metal's own GCD threads report too).
fn track() {
    extern "C" {
        fn pthread_self() -> usize;
    }
    if std::env::var_os("THREERS_POOL_CHECK").is_some() {
        eprintln!("THREERS_THREAD {:#x}", unsafe { pthread_self() });
    }
}

fn spawn<T: Send + 'static>(f: impl FnOnce() -> T + Send + 'static) -> thread::JoinHandle<T> {
    thread::spawn(move || {
        track();
        f()
    })
}

fn gpu() -> Option<LayerGpu> {
    match LayerGpu::new() {
        Ok(g) => Some(g),
        Err(e) => {
            eprintln!("skipping: {e}");
            None
        }
    }
}

fn scene() -> Scene {
    let mut s = Scene::new();
    s.background = Color::from_hex(0x203040);
    s.add_light(DirectionalLight::new(Color::WHITE, 2.0));
    s.add(Object3D::mesh(Mesh::new(
        BoxGeometry::new(1.0, 1.0, 1.0),
        StandardMaterial::new(Color::from_hex(0xff8844)).into(),
    )));
    s
}

fn camera() -> PerspectiveCamera {
    let mut c = PerspectiveCamera::new(50.0, 1.5, 0.1, 100.0);
    c.position = Vector3::new(2.0, 1.5, 3.0);
    c.target = Vector3::ZERO;
    c
}

/// Draw `frames` frames, returning how many reached the layer.
fn draw(view: &mut LayerRenderer, scene: &mut Scene, frames: usize) -> usize {
    (0..frames).filter(|_| view.render(scene, &camera())).count()
}

#[test]
fn layers_draw_from_render_threads_at_once() {
    track();
    let Some(gpu) = gpu() else { return };
    // One device, a layer and a render thread each, all drawing together.
    let gpu = Arc::new(gpu);
    const THREADS: usize = 4;
    let start = Arc::new(Barrier::new(THREADS));
    let handles: Vec<_> = (0..THREADS)
        .map(|t| {
            let (gpu, start) = (gpu.clone(), start.clone());
            spawn(move || {
                let layer = Layer::metal();
                let mut view = unsafe { LayerRenderer::new(&gpu, layer.ptr(), 96, 64, 4) }.unwrap();
                let mut scene = scene();
                start.wait();
                let mut drawn = draw(&mut view, &mut scene, 8);
                view.resize(120, 80);
                assert_eq!(view.size(), (120, 80));
                drawn += draw(&mut view, &mut scene, 8);
                assert!(drawn >= 8, "thread {t} drew {drawn} of 16 frames");
                drop(view);
                drop(layer);
            })
        })
        .collect();
    for h in handles {
        h.join().unwrap();
    }
}

#[test]
fn a_layer_renderer_changes_threads() {
    track();
    let Some(gpu) = gpu() else { return };
    let gpu = Arc::new(gpu);
    let layer = Layer::metal();
    // Made here, drawn on two other threads in turn, then back here.
    let mut view = unsafe { LayerRenderer::new(&gpu, layer.ptr(), 64, 64, 1) }.unwrap();
    let mut scene = scene();
    assert!(draw(&mut view, &mut scene, 3) > 0);
    for _ in 0..2 {
        let (v, s, n) = spawn(move || {
            let n = draw(&mut view, &mut scene, 3);
            (view, scene, n)
        })
        .join()
        .unwrap();
        assert!(n > 0, "no frame on the render thread");
        (view, scene) = (v, s);
    }
    assert!(draw(&mut view, &mut scene, 3) > 0);
    // Dropped on another thread than the one that made it.
    spawn(move || drop(view)).join().unwrap();
}

#[test]
fn a_layer_that_is_not_metal_is_an_error_not_a_crash() {
    track();
    let Some(gpu) = gpu() else { return };
    let plain = Layer::new(c"CALayer");
    for layer in [plain.ptr(), std::ptr::null_mut()] {
        let err = unsafe { LayerRenderer::new(&gpu, layer, 8, 8, 1) }.err().expect("an error");
        assert!(err.contains("CAMetalLayer"), "{err}");
    }
}

/// Reruns the rest of this file with the runtime reporting every object
/// autoreleased where no pool is in place, on the threads that ran our code.
#[cfg(target_os = "macos")]
#[test]
fn no_object_is_autoreleased_without_a_pool() {
    if std::env::var_os("THREERS_POOL_CHECK").is_some() || gpu().is_none() {
        return;
    }
    let out = std::process::Command::new(std::env::current_exe().unwrap())
        .args(["--skip", "no_object_is_autoreleased_without_a_pool", "--nocapture"])
        .env("OBJC_DEBUG_MISSING_POOLS", "YES")
        .env("THREERS_POOL_CHECK", "1")
        .output()
        .unwrap();
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "the rerun failed:\n{stderr}");
    let ours: std::collections::HashSet<&str> =
        stderr.lines().filter_map(|l| l.strip_prefix("THREERS_THREAD ")).collect();
    assert!(ours.len() >= 6, "the rerun named too few threads: {}", ours.len());
    let missing: Vec<_> = stderr
        .lines()
        .filter(|l| l.contains("MISSING POOLS"))
        .filter(|l| {
            l.split('(').nth(1).and_then(|r| r.split(')').next()).is_some_and(|t| ours.contains(t))
        })
        .collect();
    assert!(
        missing.is_empty(),
        "{} objects autoreleased with no pool, first:\n{}",
        missing.len(),
        missing.iter().take(5).copied().collect::<Vec<_>>().join("\n")
    );
}
