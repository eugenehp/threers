//! Parity scene `glitch` — native Rust (winit + wgpu).
//!
//! Generated from `tests/parity/scenes/threers-glitch.html`.
//! Compare with the JavaScript tab above; adjust imports before copying to `examples/`.
//!
//! ```text
//! cargo run --example glitch
//! ```

use std::sync::Arc;

use threers::cameras::Camera;
use threers::{
    BasicMaterial,
    BoxGeometry,
    Color,
    Mesh,
    Object3D,
    PerspectiveCamera,
    RenderTarget,
    Renderer,
    Scene,
    TextureFormat,
    Vector3,
};

use winit::{
    event::{Event, WindowEvent},
    event_loop::ActiveEventLoop,
};

struct ParityRng { state: u32 }

impl ParityRng {
    fn new(seed: u32) -> Self {
        Self { state: seed }
    }
    fn next_f32(&mut self) -> f32 {
        self.state = self.state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        self.state as f32 / 4_294_967_296.0
    }
    fn rand_float(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.next_f32()
    }
    fn rand_int(&mut self, lo: i32, hi: i32) -> i32 {
        lo + (self.next_f32() * (hi - lo + 1) as f32) as i32
    }
}

fn sample_height_linear(data: &[f32], size: u32, u: f32, v: f32) -> f32 {
    let stx = u * size as f32 - 0.5;
    let sty = v * size as f32 - 0.5;
    let i0x = stx.floor() as i32;
    let i0y = sty.floor() as i32;
    let fx = stx - i0x as f32;
    let fy = sty - i0y as f32;
    let max = size as i32 - 1;
    let load = |ix: i32, iy: i32| {
        let cx = ix.clamp(0, max) as u32;
        let cy = iy.clamp(0, max) as u32;
        data[(cy * size + cx) as usize]
    };
    let c0 = load(i0x, i0y) * (1.0 - fx) + load(i0x + 1, i0y) * fx;
    let c1 = load(i0x, i0y + 1) * (1.0 - fx) + load(i0x + 1, i0y + 1) * fx;
    c0 * (1.0 - fy) + c1 * fy
}

/// Full-screen disp bake — same as `_glslGlitchDispBake` / wasm kind 26.
fn bake_glitch_disp(w: u32, h: u32, seed: f32, height: &[f32], dt: u32) -> Vec<f32> {
    let mut out = vec![0f32; (w * h) as usize];
    for y in 0..h {
        for x in 0..w {
            let u = (x as f32 + 0.5) / w as f32;
            let v = (y as f32 + 0.5) / h as f32;
            out[(y * w + x) as usize] =
                sample_height_linear(height, dt, u * seed * seed, v * seed * seed);
        }
    }
    out
}

fn main() {
    env_logger::init();
    let attributes = winit::window::Window::default_attributes()
        .with_title("threers — glitch")
        .with_inner_size(winit::dpi::LogicalSize::new(800, 600));
    threers::sketch::run_window(attributes, |window| pollster::block_on(setup(window)));
}

/// The scene for a window that has just opened, and the handler that draws it.
async fn setup(window: Arc<winit::window::Window>) -> impl FnMut(Event<()>, &ActiveEventLoop) {

    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::PRIMARY,
        ..Default::default()
    });
    let surface = instance.create_surface(window.clone()).expect("surface");

    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::default(),
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        })
        .await
        .expect("adapter");

    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: Some("threers"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::downlevel_defaults(),
            },
            None,
        )
        .await
        .expect("device");

    let device = Arc::new(device);
    let queue = Arc::new(queue);

    let size = window.inner_size();
    let caps = surface.get_capabilities(&adapter);
    let format = caps.formats.iter().copied().find(|f| f.is_srgb()).unwrap_or(caps.formats[0]);

    let mut config = wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format,
        width: size.width.max(1),
        height: size.height.max(1),
        present_mode: caps.present_modes[0],
        alpha_mode: caps.alpha_modes[0],
        view_formats: vec![],
        desired_maximum_frame_latency: 2,
    };
    surface.configure(&device, &config);

    let mut renderer = Renderer::new(device.clone(), queue.clone(), format, config.width, config.height);

    // --- scene (matches threers parity iframe) ---
    let mut scene = Scene::new();
    scene.background = Color::from_hex(0x101010);
    scene.add(Object3D::mesh(Mesh::new(
        BoxGeometry::new(1.0, 1.0, 1.0),
        BasicMaterial::new(Color::from_hex(0x00ff88)),
    )));
    let mut camera = PerspectiveCamera::new(45.0, 800.0 / 600.0, 0.1, 100.0);
    camera.position = Vector3::new(0.0, 0.0, 3.0);
    camera.look_at(Vector3::ZERO);

    // DigitalGlitch parity — same LCG + frame-0 uniforms as three.js GlitchPass (seed 42).
    let mut rng = ParityRng::new(42);
    let dt_size = 64u32;
    let mut height_data = vec![0f32; (dt_size * dt_size) as usize];
    for v in &mut height_data {
        *v = rng.rand_float(0.0, 1.0);
    }
    for _ in 0..16 {
        rng.next_f32();
    }
    let _rand_x = rng.rand_int(120, 240);
    let glitch_seed = rng.next_f32();
    let glitch_amount = rng.next_f32() / 30.0;
    let glitch_angle = rng.rand_float(-std::f32::consts::PI, std::f32::consts::PI);
    let glitch_seed_x = rng.rand_float(-1.0, 1.0);
    let glitch_seed_y = rng.rand_float(-1.0, 1.0);
    let glitch_dist_x = rng.next_f32();
    let glitch_dist_y = rng.next_f32();
    let glitch_byp = 0.0f32;

    renderer.set_glitch_disp(&height_data, dt_size);
    let disp_bake = bake_glitch_disp(config.width, config.height, glitch_seed, &height_data, dt_size);
    renderer.set_glitch_snow(&disp_bake, config.width, config.height);
    let scene_rt = RenderTarget::new(
        &device,
        config.width,
        config.height,
        wgpu::TextureFormat::Rgba16Float,
    );

    scene.update_world();
    let window_for_loop = window.clone();

    move |event, target| {
            match event {
                Event::WindowEvent { event, window_id } if window_id == window_for_loop.id() => {
                    match event {
                        WindowEvent::CloseRequested => target.exit(),
                        WindowEvent::Resized(new_size) => {
                            config.width = new_size.width.max(1);
                            config.height = new_size.height.max(1);
                            surface.configure(&device, &config);
                            renderer.resize(config.width, config.height);
                            camera.set_aspect(config.width as f32 / config.height as f32);
                        }
                        WindowEvent::RedrawRequested => {
                            scene.update_world();
                            match surface.get_current_texture() {
                                Ok(frame) => {
                                    let view = frame.texture.create_view(&wgpu::TextureViewDescriptor::default());
                                    renderer.render_to(&mut scene, &camera, &scene_rt);
                                    renderer.apply_postfx(
                                        &scene_rt.color_view,
                                        &view,
                                        26,
                                        glitch_seed,
                                        [
                                            glitch_amount,
                                            glitch_angle,
                                            glitch_dist_x,
                                            glitch_dist_y,
                                        ],
                                        [glitch_seed_x, glitch_seed_y, 0.05, glitch_byp],
                                        config.width,
                                        config.height,
                                    );
                                    frame.present();
                                }
                                Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                                    surface.configure(&device, &config);
                                }
                                Err(e) => log::error!("surface: {e:?}"),
                            }
                        }
                        _ => {}
                    }
                }
                Event::AboutToWait => window_for_loop.request_redraw(),
                _ => {}
            }
        }
}
