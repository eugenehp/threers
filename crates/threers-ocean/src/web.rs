//! The browser shell: canvas → WebGPU surface → `requestAnimationFrame`.
//!
//! This is the only part of the demo that is web-specific. It does what the
//! native example's winit loop does — make a surface, build the world, pump
//! frames, feed the orbit controls — with the platform's own primitives.

use std::cell::RefCell;
use std::rc::Rc;

use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;

use threers::cameras::Camera;
use threers::{OrbitControls, PerspectiveCamera, PointerEvent, Renderer};

use crate::preset::{self, Preset, Quality, QUALITY_LEVELS};
use crate::waves_gpu::WaveCompute;
use crate::world::World;

/// Everything the frame callback needs, behind one handle so the closure can own
/// it. The browser gives no place to put state, so it lives here.
struct App {
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    device: std::sync::Arc<wgpu::Device>,
    queue: std::sync::Arc<wgpu::Queue>,
    renderer: Renderer,
    world: World,
    waves: WaveCompute,
    controls: OrbitControls,
    /// Drag state, mirroring the native example's `Input`.
    rotating: bool,
    panning: bool,
    last: Option<(f32, f32)>,
    /// Page time at start, so the sea begins at t = 0.
    t0: f64,
    frames: u32,
    fps_since: f64,
}

fn now() -> f64 {
    web_sys::window()
        .and_then(|w| w.performance())
        .map(|p| p.now())
        .unwrap_or(0.0)
}

fn lookup(name: Option<String>, presets: &[Preset]) -> Preset {
    match name {
        Some(n) => presets
            .iter()
            .find(|p| p.name == n)
            .copied()
            .unwrap_or(presets[1]),
        None => presets[1],
    }
}

fn lookup_quality(name: Option<String>) -> Quality {
    match name {
        Some(n) => QUALITY_LEVELS
            .iter()
            .find(|q| q.name == n)
            .copied()
            .unwrap_or(QUALITY_LEVELS[2]),
        None => QUALITY_LEVELS[2],
    }
}

/// Start the demo on `canvas`. Resolves once the first frame is scheduled;
/// the loop then runs until the page goes away.
#[wasm_bindgen]
pub async fn start(
    canvas: web_sys::HtmlCanvasElement,
    preset_name: Option<String>,
    quality_name: Option<String>,
) -> Result<(), JsValue> {
    console_error_panic_hook::set_once();

    let presets = preset::all();
    let preset = lookup(preset_name, &presets);
    let quality = lookup_quality(quality_name);

    let width = canvas.width().max(1);
    let height = canvas.height().max(1);

    let instance = {
        // wgpu 30 dropped `Default` here; the display handle is only
        // consulted by GLES/Wayland, not Vulkan, Metal or DX12.
        let mut d = wgpu::InstanceDescriptor::new_without_display_handle();
        d.backends = wgpu::Backends::BROWSER_WEBGPU;
        wgpu::Instance::new(d)
    };
    let surface = instance
        .create_surface(wgpu::SurfaceTarget::Canvas(canvas.clone()))
        .map_err(|e| JsValue::from_str(&format!("no WebGPU surface: {e}")))?;

    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
            apply_limit_buckets: false,
        })
        .await
        .ok_or_else(|| JsValue::from_str("no WebGPU adapter (does this browser support WebGPU?)"))?;

    // The browser's own defaults, not `downlevel_defaults`: this demo wants
    // storage textures and a handful of storage buffers per stage, and WebGPU's
    // baseline already guarantees them.
    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: Some("threers ocean"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default().using_resolution(adapter.limits()),
                ..Default::default()
            },
        )
        .await
        .map_err(|e| JsValue::from_str(&format!("no WebGPU device: {e}")))?;
    let device = std::sync::Arc::new(device);
    let queue = std::sync::Arc::new(queue);

    let caps = surface.get_capabilities(&adapter);
    let format = caps
        .formats
        .iter()
        .copied()
        .find(|f| f.is_srgb())
        .unwrap_or(caps.formats[0]);
    let config = wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format,
        width,
        height,
        present_mode: wgpu::PresentMode::Fifo,
        alpha_mode: caps.alpha_modes[0],
        view_formats: vec![],
        desired_maximum_frame_latency: 2,
        color_space: wgpu::SurfaceColorSpace::Srgb,
    };
    surface.configure(&device, &config);

    let mut renderer = Renderer::new(device.clone(), queue.clone(), format, width, height);
    renderer.set_taa(true);

    let mut world = World::build(&preset, &quality, width as f32 / height as f32);
    let waves = world.attach_waves(&device, &queue, &mut renderer);

    let mut controls = OrbitControls::new(&world.camera);
    controls.min_distance = 2.0;
    controls.max_distance = 2500.0;
    controls.max_polar_angle = 3.05;

    let app = Rc::new(RefCell::new(App {
        surface,
        config,
        device,
        queue,
        renderer,
        world,
        waves,
        controls,
        rotating: false,
        panning: false,
        last: None,
        t0: now(),
        frames: 0,
        fps_since: now(),
    }));

    install_input(&canvas, &app)?;
    start_frames(canvas, app);
    Ok(())
}

/// Mouse drag and wheel, wired to the same `OrbitControls` the desktop build
/// uses — so the two behave identically rather than merely similarly.
fn install_input(canvas: &web_sys::HtmlCanvasElement, app: &Rc<RefCell<App>>) -> Result<(), JsValue> {
    let target: &web_sys::EventTarget = canvas.as_ref();

    let a = app.clone();
    let down = Closure::<dyn FnMut(web_sys::MouseEvent)>::new(move |e: web_sys::MouseEvent| {
        let mut s = a.borrow_mut();
        match e.button() {
            0 => s.rotating = true,
            2 => s.panning = true,
            _ => {}
        }
        s.last = None;
        e.prevent_default();
    });
    target.add_event_listener_with_callback("mousedown", down.as_ref().unchecked_ref())?;
    down.forget();

    let a = app.clone();
    let up = Closure::<dyn FnMut(web_sys::MouseEvent)>::new(move |_e: web_sys::MouseEvent| {
        let mut s = a.borrow_mut();
        s.rotating = false;
        s.panning = false;
        s.last = None;
    });
    target.add_event_listener_with_callback("mouseup", up.as_ref().unchecked_ref())?;
    up.forget();

    let a = app.clone();
    let mv = Closure::<dyn FnMut(web_sys::MouseEvent)>::new(move |e: web_sys::MouseEvent| {
        let mut s = a.borrow_mut();
        if !s.rotating && !s.panning {
            return;
        }
        let p = (e.client_x() as f32, e.client_y() as f32);
        let (dx, dy) = match s.last {
            Some(l) => (p.0 - l.0, p.1 - l.1),
            None => (0.0, 0.0),
        };
        s.last = Some(p);
        let (w, h) = (s.config.width as f32, s.config.height as f32);
        let ev = PointerEvent {
            dx,
            dy,
            wheel: 0.0,
            rotating: s.rotating,
            panning: s.panning,
        };
        let App {
            controls, world, ..
        } = &mut *s;
        controls.update(ev, &mut world.camera, (w, h));
    });
    target.add_event_listener_with_callback("mousemove", mv.as_ref().unchecked_ref())?;
    mv.forget();

    let a = app.clone();
    let wheel = Closure::<dyn FnMut(web_sys::WheelEvent)>::new(move |e: web_sys::WheelEvent| {
        let mut s = a.borrow_mut();
        let (w, h) = (s.config.width as f32, s.config.height as f32);
        let ev = PointerEvent {
            dx: 0.0,
            dy: 0.0,
            // Browsers report wheel deltas an order larger than winit's lines.
            wheel: -e.delta_y() as f32 * 0.6,
            rotating: false,
            panning: false,
        };
        let App {
            controls, world, ..
        } = &mut *s;
        controls.update(ev, &mut world.camera, (w, h));
        e.prevent_default();
    });
    target.add_event_listener_with_callback("wheel", wheel.as_ref().unchecked_ref())?;
    wheel.forget();

    // Right-drag pans, so the context menu has to go.
    let ctx = Closure::<dyn FnMut(web_sys::MouseEvent)>::new(|e: web_sys::MouseEvent| {
        e.prevent_default();
    });
    target.add_event_listener_with_callback("contextmenu", ctx.as_ref().unchecked_ref())?;
    ctx.forget();

    Ok(())
}

/// The frame loop. `requestAnimationFrame` re-arms itself through an `Rc` the
/// closure holds a weak half of, which is the standard way to express a
/// self-scheduling callback without leaking it on every frame.
fn start_frames(canvas: web_sys::HtmlCanvasElement, app: Rc<RefCell<App>>) {
    let f: Rc<RefCell<Option<Closure<dyn FnMut()>>>> = Rc::new(RefCell::new(None));
    let g = f.clone();

    *g.borrow_mut() = Some(Closure::<dyn FnMut()>::new(move || {
        {
            let mut s = app.borrow_mut();

            // Follow the canvas if CSS resized it.
            let (w, h) = (canvas.width().max(1), canvas.height().max(1));
            if w != s.config.width || h != s.config.height {
                s.config.width = w;
                s.config.height = h;
                let cfg = s.config.clone();
                let dev = s.device.clone();
                s.surface.configure(&dev, &cfg);
                s.renderer.resize(w, h);
                s.world.camera.set_aspect(w as f32 / h as f32);
            }

            let t = ((now() - s.t0) / 1000.0) as f32;
            let App {
                world,
                device,
                queue,
                waves,
                ..
            } = &mut *s;
            world.update(t, device, queue, waves);

            if let wgpu::CurrentSurfaceTexture::Success(frame)
            | wgpu::CurrentSurfaceTexture::Suboptimal(frame) =
            s.surface.get_current_texture() {
                let view = frame
                    .texture
                    .create_view(&wgpu::TextureViewDescriptor::default());
                let App {
                    renderer, world, ..
                } = &mut *s;
                renderer.render(&mut world.scene, &world.camera, &view, false);
                queue.present(frame);
            }

            // Frame rate into the document title, same as the native build puts
            // it in the window title.
            s.frames += 1;
            let dt = now() - s.fps_since;
            if dt >= 500.0 {
                let fps = s.frames as f64 * 1000.0 / dt;
                let submerged = s.world.submerged;
                if let Some(doc) = web_sys::window().and_then(|w| w.document()) {
                    doc.set_title(&format!(
                        "threers — ocean water · {fps:.0} fps{}",
                        if submerged { " · underwater" } else { "" }
                    ));
                }
                s.frames = 0;
                s.fps_since = now();
            }
        }
        request_frame(f.borrow().as_ref().unwrap());
    }));

    request_frame(g.borrow().as_ref().unwrap());
}

fn request_frame(f: &Closure<dyn FnMut()>) {
    if let Some(w) = web_sys::window() {
        let _ = w.request_animation_frame(f.as_ref().unchecked_ref());
    }
}
