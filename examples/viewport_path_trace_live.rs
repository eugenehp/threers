//! Interactive path-traced viewport: orbit the camera and watch the image refine.
//!
//! ```text
//! cargo run --release --example viewport_path_trace_live --features raytrace
//! cargo run --release --example viewport_path_trace_live --features raytrace -- --gpu
//! ```
//!
//! Left-drag: orbit · Right-drag: pan · Scroll: zoom · Release mouse to refine.
//! Keys: `R` reset · `D` denoise · `A` adaptive · `S` redistribute · `Tab` AOV
//! · `H` error · `E` PNG · `X` EXR · `C`/`L` checkpoint · `[`/`]` exposure
//! · `=`/`-` sample budget · `F` firefly clamp · `G` glass caustics · `Space` pause.

use std::sync::Arc;

use threers::cameras::Camera;
use threers::raytrace::{gpu::GpuBackend, Aov, CpuBackend, RaytraceRenderer, RaytraceSettings};
use threers::{
    AmbientLight, BoxGeometry, Color, DirectionalLight, Material, Mesh, Object3D, OrbitControls,
    PerspectiveCamera, PhysicalMaterial, PlaneGeometry, PointerEvent, Scene, SphereGeometry,
    StandardMaterial, Vector3,
};

use winit::{
    dpi::PhysicalSize,
    event::{ElementState, Event, KeyEvent, MouseButton, MouseScrollDelta, WindowEvent},
    event_loop::EventLoop,
    keyboard::{KeyCode, PhysicalKey},
    window::WindowBuilder,
};

struct InputState {
    rotating: bool,
    panning: bool,
    last: Option<winit::dpi::PhysicalPosition<f64>>,
    camera_dirty: bool,
    paused: bool,
    view: ViewMode,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ViewMode {
    Beauty,
    Albedo,
    Normal,
    Depth,
    Error,
    Samples,
}

impl ViewMode {
    fn next(self) -> Self {
        match self {
            Self::Beauty => Self::Albedo,
            Self::Albedo => Self::Normal,
            Self::Normal => Self::Depth,
            Self::Depth => Self::Error,
            Self::Error => Self::Samples,
            Self::Samples => Self::Beauty,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Beauty => "beauty",
            Self::Albedo => "albedo",
            Self::Normal => "normal",
            Self::Depth => "depth",
            Self::Error => "error",
            Self::Samples => "samples",
        }
    }
}

impl InputState {
    fn new() -> Self {
        Self {
            rotating: false,
            panning: false,
            last: None,
            camera_dirty: false,
            paused: false,
            view: ViewMode::Beauty,
        }
    }

    fn motion(&mut self, pos: winit::dpi::PhysicalPosition<f64>) -> (f32, f32) {
        let (dx, dy) = match self.last {
            Some(last) => ((pos.x - last.x) as f32, (pos.y - last.y) as f32),
            None => (0.0, 0.0),
        };
        self.last = Some(pos);
        (dx, dy)
    }
}

struct RgbaPresenter {
    texture: wgpu::Texture,
    #[allow(dead_code)]
    view: wgpu::TextureView,
    bind_group: wgpu::BindGroup,
    pipeline: wgpu::RenderPipeline,
    width: u32,
    height: u32,
}

impl RgbaPresenter {
    fn new(device: &wgpu::Device, format: wgpu::TextureFormat, width: u32, height: u32) -> Self {
        let w = width.max(1);
        let h = height.max(1);
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("path trace preview"),
            size: wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("path trace preview sampler"),
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("path trace blit layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("path trace blit bind group"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("path trace blit"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(
                r#"
@group(0) @binding(0) var t: texture_2d<f32>;
@group(0) @binding(1) var s: sampler;

struct VOut { @builtin(position) p: vec4<f32>, @location(0) uv: vec2<f32> };

@vertex fn vs(@builtin(vertex_index) i: u32) -> VOut {
    var p = array<vec2<f32>, 3>(vec2(-1.0, -1.0), vec2(3.0, -1.0), vec2(-1.0, 3.0));
    var o: VOut;
    o.p = vec4(p[i], 0.0, 1.0);
    o.uv = p[i] * vec2(0.5, -0.5) + vec2(0.5, 0.5);
    return o;
}

@fragment fn fs(i: VOut) -> @location(0) vec4<f32> {
    return textureSample(t, s, i.uv);
}
"#,
            )),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("path trace blit pipeline layout"),
            bind_group_layouts: &[Some(&layout)],
            ..Default::default()
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("path trace blit"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs"),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });
        Self {
            texture,
            view,
            bind_group,
            pipeline,
            width: w,
            height: h,
        }
    }

    fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32, format: wgpu::TextureFormat) {
        *self = Self::new(device, format, width, height);
    }

    fn upload(&self, queue: &wgpu::Queue, rgba: &[u8]) {
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            rgba,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(self.width * 4),
                rows_per_image: Some(self.height),
            },
            wgpu::Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
        );
    }

    fn draw(&self, encoder: &mut wgpu::CommandEncoder, target: &wgpu::TextureView) {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("path trace blit pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            ..Default::default()
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.draw(0..3, 0..1);
    }
}

fn build_scene() -> Scene {
    let mut scene = Scene::new();
    scene.background = Color::new(0.04, 0.05, 0.08);

    let floor = StandardMaterial::new(Color::new(0.55, 0.55, 0.58)).with_roughness(0.85);
    let mut floor_obj = Object3D::mesh(Mesh::new(
        PlaneGeometry::new(14.0, 14.0),
        Material::Standard(floor),
    ));
    floor_obj.rotate_x(-std::f32::consts::FRAC_PI_2);
    scene.add(floor_obj);

    let glass = PhysicalMaterial::new(Color::new(0.92, 0.96, 1.0))
        .with_roughness(0.04)
        .with_metalness(0.0)
        .with_transmission(1.0, 1.45, 0.5);
    scene.add(Object3D::mesh(Mesh::new(
        SphereGeometry::new(0.55, 32, 24),
        Material::Physical(glass),
    )));

    let mut hero = Object3D::mesh(Mesh::new(
        BoxGeometry::new(1.2, 1.2, 1.2),
        Material::Standard(StandardMaterial::new(Color::new(0.78, 0.35, 0.18)).with_roughness(0.35)),
    ));
    hero.position = Vector3::new(-1.2, 0.6, 0.0);
    scene.add(hero);

    scene.add_light(AmbientLight::new(Color::new(0.25, 0.28, 0.32), 0.35));
    let mut key = Object3D::light(DirectionalLight::new(Color::new(1.0, 0.96, 0.88), 2.2));
    key.position = Vector3::new(4.0, 6.0, 3.0);
    scene.add(key);
    scene
}

fn use_gpu() -> bool {
    std::env::args().any(|a| a == "--gpu")
}

fn main() {
    env_logger::init();
    pollster::block_on(run());
}

async fn run() {
    let event_loop = EventLoop::new().expect("event loop");
    let window = Arc::new(
        WindowBuilder::new()
            .with_title("threers — live path trace")
            .with_inner_size(PhysicalSize::new(960, 720))
            .build(&event_loop)
            .expect("window"),
    );

    let instance = {
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
                label: Some("viewport path trace"),
                required_features: wgpu::Features::empty(),
                required_limits: adapter.limits(),
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

    let mut scene = build_scene();
    let mut camera = PerspectiveCamera::new(
        42.0,
        config.width as f32 / config.height as f32,
        0.1,
        100.0,
    );
    camera.position = Vector3::new(4.5, 2.8, 5.5);
    camera.look_at(Vector3::new(0.0, 0.6, 0.0));

    let settings = RaytraceSettings::preview()
        .with_samples(256)
        .with_denoise(true)
        .with_adaptive(0.02)
        .with_sample_redistribution(true);

    let mut path = if use_gpu() {
        let backend = GpuBackend::with_device(device.clone(), queue.clone()).with_samples_per_dispatch(4);
        let mut r = RaytraceRenderer::with_backend(config.width, config.height, Box::new(backend));
        r.set_settings(settings);
        r.prepare(&mut scene, &camera);
        eprintln!("backend: gpu");
        r
    } else {
        let mut r = RaytraceRenderer::with_backend(
            config.width,
            config.height,
            Box::new(CpuBackend::new()),
        );
        r.set_settings(settings);
        r.prepare(&mut scene, &camera);
        eprintln!("backend: cpu");
        r
    };

    let mut presenter = RgbaPresenter::new(&device, format, config.width, config.height);
    let mut controls = OrbitControls::new(&camera);
    let mut input = InputState::new();
    let mut title_timer = std::time::Instant::now();
    let window_for_loop = window.clone();

    event_loop
        .run(move |event, target| match event {
            Event::WindowEvent { event, window_id } if window_id == window_for_loop.id() => {
                match event {
                    WindowEvent::CloseRequested => target.exit(),
                    WindowEvent::Resized(new_size) => {
                        config.width = new_size.width.max(1);
                        config.height = new_size.height.max(1);
                        surface.configure(&device, &config);
                        camera.set_aspect(config.width as f32 / config.height as f32);
                        let (w, h) = path.set_size(config.width, config.height);
                        config.width = w;
                        config.height = h;
                        path.prepare_if_changed(&mut scene, &camera);
                        presenter.resize(&device, config.width, config.height, format);
                        input.camera_dirty = false;
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
                            if input.camera_dirty {
                                path.prepare_if_changed(&mut scene, &camera);
                                input.camera_dirty = false;
                            }
                        }
                    }
                    WindowEvent::CursorMoved { position, .. } => {
                        if !input.rotating && !input.panning {
                            return;
                        }
                        let (dx, dy) = input.motion(position);
                        let ev = PointerEvent {
                            dx,
                            dy,
                            wheel: 0.0,
                            rotating: input.rotating,
                            panning: input.panning,
                        };
                        controls.update(
                            ev,
                            &mut camera,
                            (config.width as f32, config.height as f32),
                        );
                        input.camera_dirty = true;
                        // Clear + coarse preview while dragging so the film
                        // matches the new view instead of showing a stale frame.
                        path.prepare_if_changed(&mut scene, &camera);
                        let _ = path.accumulate_unconverged_tiles_budgeted(0, 1, 2, 8);
                        input.camera_dirty = false;
                    }
                    WindowEvent::KeyboardInput {
                        event:
                            KeyEvent {
                                physical_key: PhysicalKey::Code(code),
                                state: ElementState::Pressed,
                                repeat: false,
                                ..
                            },
                        ..
                    } => match code {
                        KeyCode::KeyR => {
                            path.reset_film();
                            path.prepare_if_changed(&mut scene, &camera);
                        }
                        KeyCode::KeyD => {
                            path.set_denoise(!path.settings().denoise);
                        }
                        KeyCode::KeyA => {
                            let t = path.settings().adaptive_threshold;
                            path.set_adaptive(if t > 0.0 { 0.0 } else { 0.02 });
                        }
                        KeyCode::KeyS => {
                            let on = !path.settings().sample_redistribution;
                            path.set_sample_redistribution(on);
                        }
                        KeyCode::KeyH => {
                            input.view = if input.view == ViewMode::Error {
                                ViewMode::Beauty
                            } else {
                                ViewMode::Error
                            };
                        }
                        KeyCode::Tab => {
                            input.view = input.view.next();
                        }
                        KeyCode::KeyE => {
                            let (w, h) = path.size();
                            let rgba = path.resolve_rgba();
                            let png = threers::encode_png(w, h, &rgba);
                            let out = "out/viewport_path_trace_live.png";
                            if let Some(dir) = std::path::Path::new(out).parent() {
                                let _ = std::fs::create_dir_all(dir);
                            }
                            match std::fs::write(out, png) {
                                Ok(()) => eprintln!("wrote {out} ({w}×{h})"),
                                Err(e) => eprintln!("save png failed: {e}"),
                            }
                        }
                        KeyCode::KeyX => {
                            match path.save_exr("out/viewport_path_trace_live.exr") {
                                Ok(()) => eprintln!(
                                    "wrote viewport_path_trace_live.exr ({} spp)",
                                    path.samples()
                                ),
                                Err(e) => eprintln!("save exr failed: {e}"),
                            }
                        }
                        KeyCode::KeyC => {
                            match path.save_film("viewport_path_trace_live.rtfc") {
                                Ok(()) => eprintln!(
                                    "wrote viewport_path_trace_live.rtfc ({} spp)",
                                    path.samples()
                                ),
                                Err(e) => eprintln!("save checkpoint failed: {e}"),
                            }
                        }
                        KeyCode::KeyL => {
                            match path.load_film("viewport_path_trace_live.rtfc") {
                                Ok(()) => eprintln!(
                                    "loaded viewport_path_trace_live.rtfc ({} spp)",
                                    path.samples()
                                ),
                                Err(e) => eprintln!("load checkpoint failed: {e}"),
                            }
                        }
                        KeyCode::BracketLeft => {
                            let e = (path.settings().exposure / 1.25).max(0.05);
                            path.set_exposure(e);
                        }
                        KeyCode::BracketRight => {
                            let e = (path.settings().exposure * 1.25).min(32.0);
                            path.set_exposure(e);
                        }
                        KeyCode::Equal => {
                            let spp = path.settings().samples_per_pixel.saturating_mul(2).max(1);
                            path.set_sample_budget(spp);
                        }
                        KeyCode::Minus => {
                            let spp = (path.settings().samples_per_pixel / 2).max(1);
                            path.set_sample_budget(spp);
                        }
                        KeyCode::KeyF => {
                            let c = path.settings().clamp_indirect;
                            path.set_clamp_indirect(if c > 0.0 { 0.0 } else { 10.0 });
                        }
                        KeyCode::KeyG => {
                            let on = !path.settings().caustic_glass_shadows;
                            path.set_caustic_glass_shadows(on);
                        }
                        KeyCode::Space => {
                            input.paused = !input.paused;
                        }
                        _ => {}
                    },
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
                            &mut camera,
                            (config.width as f32, config.height as f32),
                        );
                        input.camera_dirty = true;
                        path.prepare_if_changed(&mut scene, &camera);
                        input.camera_dirty = false;
                    }
                    WindowEvent::RedrawRequested => {
                        const MAX_TILES: u32 = 16;
                        const BUDGET_MS: u32 = 14;
                        if !input.rotating && !input.panning && !input.paused && path.needs_more_samples() {
                            let _ = path.accumulate_interactive(
                                &mut scene,
                                &camera,
                                0,
                                1,
                                MAX_TILES,
                                BUDGET_MS,
                            );
                        }

                        let rgba = match input.view {
                            ViewMode::Beauty => path.resolve_rgba(),
                            ViewMode::Albedo => path.resolve_rgba_aov(Aov::Albedo),
                            ViewMode::Normal => path.resolve_rgba_aov(Aov::Normal),
                            ViewMode::Depth => path.resolve_rgba_aov(Aov::Depth),
                            ViewMode::Error => path.resolve_rgba_error_overlay(0.65),
                            ViewMode::Samples => path.resolve_rgba_samples(),
                        };
                        presenter.upload(&queue, &rgba);

                        match surface.get_current_texture() {
                            wgpu::CurrentSurfaceTexture::Success(frame)
                            | wgpu::CurrentSurfaceTexture::Suboptimal(frame) => {
                                let view = frame
                                    .texture
                                    .create_view(&wgpu::TextureViewDescriptor::default());
                                let mut encoder =
                                    device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                                        label: Some("viewport path trace frame"),
                                    });
                                presenter.draw(&mut encoder, &view);
                                queue.submit(Some(encoder.finish()));
                                queue.present(frame);
                            }
                            wgpu::CurrentSurfaceTexture::Lost
                            | wgpu::CurrentSurfaceTexture::Outdated => {
                                surface.configure(&device, &config);
                            }
                            e => log::error!("surface: {e:?}"),
                        }

                        if title_timer.elapsed().as_millis() > 250 {
                            title_timer = std::time::Instant::now();
                            let spp = path.samples();
                            let total = path.settings().samples_per_pixel;
                            let tile = path.recommended_tile_size();
                            let stats = path.render_stats_for_tile_size(tile);
                            let eff = stats.efficiency * 100.0;
                            let conv = stats.converged_fraction * 100.0;
                            let tiles = stats.unconverged_tile_count;
                            let ms = path.last_tile_ms();
                            let denoise = if path.settings().denoise { "denoise" } else { "raw" };
                            let adapt = if path.settings().adaptive_threshold > 0.0 {
                                "adapt"
                            } else {
                                "fixed"
                            };
                            let redist = if path.settings().sample_redistribution {
                                "+redist"
                            } else {
                                ""
                            };
                            let clamp = if path.settings().clamp_indirect > 0.0 {
                                " · clamp"
                            } else {
                                ""
                            };
                            let caustic = if path.settings().caustic_glass_shadows {
                                " · caustic"
                            } else {
                                ""
                            };
                            let pause = if input.paused { " · paused" } else { "" };
                            let exp = path.settings().exposure;
                            let view = input.view.label();
                            let max_side = path
                                .max_film_side()
                                .map(|n| format!("{n}px"))
                                .unwrap_or_else(|| "n/a".into());
                            window_for_loop.set_title(&format!(
                                "threers — path trace {spp}/{total} spp · {view} · eff {eff:.0}% · converged {conv:.0}% · tiles {tiles} · {ms:.1}ms/tile · exp {exp:.2} · max {max_side} · {denoise} · {adapt}{redist}{clamp}{caustic}{pause}"
                            ));
                        }

                        let keep_going = (!input.paused && path.needs_more_samples())
                            || input.rotating
                            || input.panning;
                        if keep_going {
                            window_for_loop.request_redraw();
                        }
                    }
                    _ => {}
                }
            }
            Event::AboutToWait
                if ((!input.paused && path.needs_more_samples())
                    || input.rotating
                    || input.panning)
                => {
                    window_for_loop.request_redraw();
                }
            _ => {}
        })
        .expect("event loop");
}
