//! A window, a scene and a loop, in one call — p5's `setup`/`draw`.
//!
//! The shortest path from nothing to a moving picture used to be about eighty
//! lines: an event loop, an instance, an adapter, a device with two limits
//! raised by hand, a surface, a surface configuration, a renderer, and then a
//! `match` on window events with resize and redraw handled. Every interactive
//! example in this crate opens with the same eighty lines. None of it is
//! interesting and all of it has to be right.
//!
//! ```no_run
//! use threers::prelude::*;
//! use threers::sketch::{sketch, Frame, SketchOptions};
//!
//! sketch(SketchOptions::new("spinning cube"), |s| {
//!     let cube = s.scene.add_mesh(
//!         BoxGeometry::new(1.2, 1.2, 1.2),
//!         StandardMaterial::new(Color::from_hex(0xff6633)).into(),
//!     );
//!     s.scene.add_light(AmbientLight::new(Color::WHITE, 0.4));
//!     s.camera.position = Vector3::new(2.5, 2.0, 3.5);
//!     s.camera.look_at(Vector3::ZERO);
//!
//!     // Setup returns the draw closure, so whatever setup made is simply
//!     // captured — no shared state to thread through, and no `Option` that
//!     // is `None` until the first frame.
//!     move |f: &mut Frame| {
//!         if let Some(obj) = f.scene.get_mut(cube) {
//!             obj.quaternion = Euler::new(f.elapsed * 0.6, f.elapsed * 0.9, 0.0).to_quaternion();
//!         }
//!     }
//! });
//! ```
//!
//! # The overlay
//!
//! [`Frame::overlay`](crate::sketch::Frame::overlay) is a 2D canvas over the 3D
//! view — see [`crate::overlay`]. Written out in full because this module's docs
//! are concatenated with the `/// ` on `pub mod sketch` in `lib.rs` and then
//! resolved in *that* file's scope, where `Frame` is not in scope.
//! Cleared before `draw` and composited after it, and a frame that draws nothing
//! on it costs nothing:
//!
//! ```no_run
//! # use threers::prelude::*;
//! # use threers::sketch::{sketch, Frame, SketchOptions};
//! sketch(SketchOptions::new("hud"), |_s| {
//!     move |f: &mut Frame| {
//!         f.overlay.fill(Color::WHITE);
//!         f.overlay.text(12.0, 12.0, &format!("{:.0} fps", 1.0 / f.delta.max(1e-6)));
//!     }
//! });
//! ```
//!
//! # Four views
//!
//! [`SketchOptions::quad_view`](crate::sketch::SketchOptions::quad_view) draws
//! every frame through a [`QuadView`](crate::helpers::QuadView) — top, front
//! and side orthographic panes plus the sketch's perspective camera. `setup`
//! and `draw` get it as `quad_view`, to frame the scene or maximise a pane:
//!
//! ```no_run
//! # use threers::prelude::*;
//! # use threers::helpers::QuadView;
//! # use threers::sketch::{sketch, Frame, SketchOptions};
//! sketch(SketchOptions::new("four views").quad_view(QuadView::new()), |s| {
//!     s.scene.add_mesh(
//!         BoxGeometry::new(1.0, 1.0, 1.0),
//!         BasicMaterial::new(Color::WHITE).into(),
//!     );
//!     s.camera.position = Vector3::new(3.0, 2.0, 4.0);
//!     s.camera.look_at(Vector3::ZERO);
//!     if let Some(q) = s.quad_view.as_deref_mut() {
//!         q.frame_scene(s.scene);
//!     }
//!     move |_f: &mut Frame| {}
//! });
//! ```
//!
//! The mouse drives it through
//! [`QuadViewControls`](crate::helpers::QuadViewControls): left-drag in the
//! perspective pane orbits the camera and right/middle-drag pans it, any drag
//! in an orthographic pane pans all three together, the wheel zooms (one scale
//! for the three orthographic panes, about the cursor), and a double-click
//! maximises a pane or restores the four. Turn it off with
//! [`SketchOptions::quad_controls`](crate::sketch::SketchOptions::quad_controls()).
//!
//! Pane rectangles are logical pixels with a bottom-left origin, while
//! `Frame::pointer` is physical pixels from the top-left: divide by
//! `scale_factor` and flip y before calling `QuadView::pane_at`.
//!
//! This is a convenience, not a framework. It owns the window and the loop and
//! nothing else: `Frame` hands out `&mut Scene`, `&mut PerspectiveCamera`, the
//! overlay and the clock, and anything it does not cover is a reason to open the
//! event loop yourself rather than a reason to grow this.

use std::sync::Arc;
use std::time::Instant;

use crate::cameras::Camera;
use crate::helpers::{PointerButton, QuadView, QuadViewControls, ViewSurface};
use crate::math::{Color, Vector2};
use crate::overlay::Overlay;
use crate::{PerspectiveCamera, Renderer, Scene};

/// Window and renderer settings. Everything has a default worth starting from.
#[derive(Debug, Clone)]
pub struct SketchOptions {
    pub title: String,
    pub width: u32,
    pub height: u32,
    /// Cleared to this before every frame.
    pub background: Color,
    /// Vertical field of view, in degrees.
    pub fov: f32,
    /// Multisampling. `1` is off; `4` is the usual choice when edges matter.
    /// Quad-view panes are multisampled too.
    pub msaa: u32,
    /// Draw through this quad view instead of one full-window view. See
    /// [`quad_view`](Self::quad_view).
    pub quad_view: Option<QuadView>,
    /// With a quad view: the mouse orbits, pans, zooms and maximises (see
    /// the [module docs](self#four-views)). On by default.
    pub quad_controls: bool,
}


impl SketchOptions {
    pub fn new(title: impl Into<String>) -> Self {
        Self {
            title: title.into(),
            ..Default::default()
        }
    }

    /// Window size in **logical** pixels — points of desk space, which is what
    /// asking for a window size means, and on a 2x display a 960x720 window is
    /// a 1920x1440 framebuffer.
    ///
    /// Everything handed to `draw` is physical instead: [`Frame::width`], the
    /// overlay and [`Frame::pointer`] are all real pixels and all agree with
    /// each other. [`Frame::scale_factor`] is the ratio between the two.
    pub fn size(mut self, width: u32, height: u32) -> Self {
        self.width = width;
        self.height = height;
        self
    }

    pub fn background(mut self, color: Color) -> Self {
        self.background = color;
        self
    }

    pub fn fov(mut self, degrees: f32) -> Self {
        self.fov = degrees;
        self
    }

    pub fn msaa(mut self, samples: u32) -> Self {
        self.msaa = samples;
        self
    }

    /// Split the window into `quad`'s four panes: top, front and side
    /// orthographic views, and the sketch camera in the perspective pane.
    /// `setup` and `draw` get it back as `quad_view` to frame or rearrange.
    pub fn quad_view(mut self, quad: QuadView) -> Self {
        self.quad_view = Some(quad);
        self
    }

    /// Whether the mouse drives the quad view (on by default).
    pub fn quad_controls(mut self, on: bool) -> Self {
        self.quad_controls = on;
        self
    }
}

impl Default for SketchOptions {
    fn default() -> Self {
        Self {
            title: "threers".into(),
            width: 960,
            height: 720,
            background: Color::from_hex(0x101018),
            fov: 60.0,
            msaa: 1,
            quad_view: None,
            quad_controls: true,
        }
    }
}

/// What `setup` is given: an empty scene and a camera, both to be filled in.
pub struct Setup<'a> {
    pub scene: &'a mut Scene,
    pub camera: &'a mut PerspectiveCamera,
    /// Physical pixels per logical pixel. See [`Frame::scale_factor`].
    pub scale_factor: f32,
    /// The 2D overlay, for settings that only need saying once — the type size,
    /// say. Anything drawn here is erased before the first frame, because the
    /// overlay is cleared per frame.
    pub overlay: &'a mut Overlay,
    /// The quad view, when [`SketchOptions::quad_view`] asked for one.
    pub quad_view: Option<&'a mut QuadView>,
    pub width: u32,
    pub height: u32,
}

/// What `draw` is given, once per frame.
pub struct Frame<'a> {
    pub scene: &'a mut Scene,
    pub camera: &'a mut PerspectiveCamera,
    /// Immediate-mode 2D over the top of the render. Cleared before every
    /// `draw`, so draw what this frame should show and nothing persists.
    pub overlay: &'a mut Overlay,
    /// The quad view the frame is drawn through, when
    /// [`SketchOptions::quad_view`] asked for one. Its pane rectangles are in
    /// logical pixels, bottom-left origin.
    pub quad_view: Option<&'a mut QuadView>,
    /// Frames drawn so far. p5's `frameCount`.
    pub frame: u64,
    /// Seconds since the first frame.
    pub elapsed: f32,
    /// Seconds since the previous frame. Use this and not a fixed step, or the
    /// sketch runs at a different speed on a different monitor.
    pub delta: f32,
    pub width: u32,
    pub height: u32,
    /// Pointer position in pixels, origin top-left. p5's `mouseX`/`mouseY`.
    pub pointer: Vector2,
    /// Pointer position as `-1..1` with y up, which is the frame most 3D maths
    /// wants and saves every sketch the same two lines.
    pub pointer_ndc: Vector2,
    pub pointer_down: bool,
    /// Whether the pointer is over the window. When it leaves, `pointer` keeps
    /// the last position it had, which is rarely what hover effects want.
    pub pointer_inside: bool,
    /// Show the system cursor over the window. Set it to `false` to draw your
    /// own; it stays as last set until changed.
    pub cursor_visible: bool,
    /// Physical pixels per logical pixel — p5's `pixelDensity()`. `1.0` on an
    /// ordinary display and `2.0` on a Retina one.
    ///
    /// [`width`](Self::width), [`height`](Self::height),
    /// [`pointer`](Self::pointer) and the overlay are all in physical pixels,
    /// so they agree with each other and with
    /// [`Camera::world_to_screen`]
    /// without conversion. What they do *not* do is look the same size on two
    /// displays: a 15-pixel label is half as tall on a 2x screen as the window
    /// size asked for. Multiply anything meant to be read — type, line weights,
    /// hit radii — by this:
    ///
    /// ```no_run
    /// # use threers::sketch::Frame;
    /// # fn draw(f: &mut Frame) {
    /// f.overlay.text_size(15.0 * f.scale_factor);
    /// # }
    /// ```
    pub scale_factor: f32,
    /// Set this to stop the loop and close the window.
    pub exit: bool,
}

/// Open a window and run `setup`, then its returned closure every frame.
///
/// Blocks until the window closes. Native only — the browser has its own loop
/// and `WebRenderer` is the way in there.
pub fn sketch<D, S>(options: SketchOptions, setup: S)
where
    S: FnOnce(&mut Setup) -> D,
    D: FnMut(&mut Frame) + 'static,
{
    use winit::event::{ElementState, Event, MouseButton, MouseScrollDelta, WindowEvent};
    use winit::event_loop::EventLoop;
    use winit::window::WindowBuilder;

    let event_loop = EventLoop::new().expect("event loop");
    let window = Arc::new(
        WindowBuilder::new()
            .with_title(options.title.clone())
            .with_inner_size(winit::dpi::LogicalSize::new(options.width, options.height))
            .build(&event_loop)
            .expect("window"),
    );

    let instance = {
        // The display handle only matters for GLES/Wayland presentation; the
        // backends asked for here ignore it.
        let mut d = wgpu::InstanceDescriptor::new_without_display_handle();
        d.backends = wgpu::Backends::PRIMARY;
        wgpu::Instance::new(d)
    };
    let surface = instance.create_surface(window.clone()).expect("surface");
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::default(),
        compatible_surface: Some(&surface),
        force_fallback_adapter: false,
        apply_limit_buckets: false,
    }))
    .expect("no GPU adapter");

    // The two limits every caller of `Renderer::new` has had to raise by hand.
    // Downlevel defaults allow sixteen sampled textures and sixteen samplers
    // per stage; the main pass declares twenty of each, so without these the
    // renderer fails pipeline-layout validation before a frame is drawn. The
    // resolution bump is for the 4096 shadow map against a downlevel cap of
    // 2048.
    let adapter_limits = adapter.limits();
    let mut limits = wgpu::Limits::downlevel_defaults().using_resolution(adapter_limits.clone());
    limits.max_sampled_textures_per_shader_stage =
        adapter_limits.max_sampled_textures_per_shader_stage;
    limits.max_samplers_per_shader_stage = adapter_limits.max_samplers_per_shader_stage;

    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("threers sketch"),
        required_features: wgpu::Features::empty(),
        required_limits: limits,
        ..Default::default()
    }))
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
    // The overlay is blended by fixed-function hardware, which mixes in
    // whatever space the attachment is in: on an sRGB view it decodes the
    // destination, mixes light and re-encodes, and on a plain `Unorm` one it
    // mixes the stored bytes. The 3D pass writes encoded colour either way, so
    // the second case blends encodings — half coverage of white on black lands
    // on 128 where the light says 188, and anti-aliased text reads thin.
    // `format` above is sRGB on every surface that offers one; declaring the
    // sRGB variant here covers the surfaces that do not, and costs nothing when
    // it is already sRGB or has no sRGB form at all.
    let srgb_format = format.add_srgb_suffix();
    let mut config = wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format,
        width: size.width.max(1),
        height: size.height.max(1),
        present_mode: caps.present_modes[0],
        alpha_mode: caps.alpha_modes[0],
        view_formats: if srgb_format == format {
            vec![]
        } else {
            vec![srgb_format]
        },
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
    if options.msaa > 1 {
        renderer.set_msaa(options.msaa);
    }
    // Viewport and scissor rectangles — the quad view's panes — are logical
    // pixels, like the window size asked for; this maps them to the surface.
    renderer.set_pixel_ratio(window.scale_factor() as f32);
    let mut quad_view = options.quad_view.clone();

    let mut scene = Scene::new();
    scene.background = options.background;
    let mut camera = PerspectiveCamera::new(
        options.fov,
        config.width as f32 / config.height as f32,
        0.1,
        1000.0,
    );

    let mut overlay = Overlay::new(config.width, config.height);
    let mut draw = {
        let mut s = Setup {
            scene: &mut scene,
            camera: &mut camera,
            overlay: &mut overlay,
            quad_view: quad_view.as_mut(),
            scale_factor: window.scale_factor() as f32,
            width: config.width,
            height: config.height,
        };
        setup(&mut s)
    };

    let start = Instant::now();
    let mut last = start;
    let mut frame_index: u64 = 0;
    let mut pointer = Vector2::new(0.0, 0.0);
    let mut pointer_down = false;
    let mut pointer_inside = false;
    let mut cursor_visible = true;
    let win = window.clone();
    // Mouse → quad view, after `setup` has placed the camera.
    let mut quad_controls = (options.quad_controls && quad_view.is_some())
        .then(|| QuadViewControls::new(&camera));
    // The last left press, for telling a double-click (winit reports none).
    let mut last_click: Option<(Instant, Vector2)> = None;

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
                    overlay.resize(config.width, config.height);
                    camera.set_aspect(config.width as f32 / config.height as f32);
                    renderer.set_pixel_ratio(win.scale_factor() as f32);
                }
                WindowEvent::ScaleFactorChanged { scale_factor, .. } => {
                    // The `Resized` that follows reconfigures the surface.
                    renderer.set_pixel_ratio(scale_factor as f32);
                }
                WindowEvent::CursorEntered { .. } => pointer_inside = true,
                WindowEvent::CursorLeft { .. } => pointer_inside = false,
                WindowEvent::CursorMoved { position, .. } => {
                    // Some platforms send no `CursorEntered` for a pointer
                    // already over the window when it opens; a move proves it.
                    pointer_inside = true;
                    pointer = Vector2::new(position.x as f32, position.y as f32);
                    if let (Some(ctl), Some(quad)) = (quad_controls.as_mut(), quad_view.as_mut()) {
                        let s = win.scale_factor() as f32;
                        ctl.pointer_move(quad, &mut camera, pointer.x / s, pointer.y / s);
                    }
                }
                WindowEvent::MouseInput { state, button, .. } => {
                    if button == MouseButton::Left {
                        pointer_down = state == ElementState::Pressed;
                    }
                    if let (Some(ctl), Some(quad)) = (quad_controls.as_mut(), quad_view.as_mut()) {
                        let s = win.scale_factor() as f32;
                        ctl.set_size(config.width as f32 / s, config.height as f32 / s);
                        let (x, y) = (pointer.x / s, pointer.y / s);
                        let button = match button {
                            MouseButton::Left => Some(PointerButton::Primary),
                            MouseButton::Middle => Some(PointerButton::Middle),
                            MouseButton::Right => Some(PointerButton::Secondary),
                            _ => None,
                        };
                        match (state, button) {
                            (ElementState::Pressed, Some(PointerButton::Primary)) => {
                                let now = Instant::now();
                                let double = last_click.is_some_and(|(t, at)| {
                                    now.duration_since(t).as_millis() < 400
                                        && (at - pointer).length() < 6.0 * s
                                });
                                if double {
                                    last_click = None;
                                    ctl.dblclick(quad, x, y);
                                } else {
                                    last_click = Some((now, pointer));
                                    ctl.pointer_down(quad, &camera, x, y, PointerButton::Primary);
                                }
                            }
                            (ElementState::Pressed, Some(b)) => {
                                ctl.pointer_down(quad, &camera, x, y, b);
                            }
                            (ElementState::Released, Some(_)) => ctl.pointer_up(),
                            _ => {}
                        }
                    }
                }
                WindowEvent::MouseWheel { delta, .. } => {
                    if let (Some(ctl), Some(quad)) = (quad_controls.as_mut(), quad_view.as_mut()) {
                        let s = win.scale_factor() as f32;
                        ctl.set_size(config.width as f32 / s, config.height as f32 / s);
                        // As DOM `deltaY`: positive scrolls down (zooms out).
                        // winit's positive is up; a line is taken as 100 px,
                        // which is what a browser reports for a wheel notch.
                        let d = match delta {
                            MouseScrollDelta::LineDelta(_, y) => -y * 100.0,
                            MouseScrollDelta::PixelDelta(p) => -(p.y as f32) / s,
                        };
                        ctl.wheel(quad, &mut camera, pointer.x / s, pointer.y / s, d);
                    }
                }
                WindowEvent::RedrawRequested => {
                    let now = Instant::now();
                    let delta = now.duration_since(last).as_secs_f32();
                    last = now;

                    // Immediate mode: this frame draws what this frame shows.
                    overlay.clear();
                    let mut f = Frame {
                        scene: &mut scene,
                        camera: &mut camera,
                        overlay: &mut overlay,
                        quad_view: quad_view.as_mut(),
                        frame: frame_index,
                        elapsed: start.elapsed().as_secs_f32(),
                        delta,
                        width: config.width,
                        height: config.height,
                        pointer,
                        pointer_ndc: Vector2::new(
                            pointer.x / config.width.max(1) as f32 * 2.0 - 1.0,
                            1.0 - pointer.y / config.height.max(1) as f32 * 2.0,
                        ),
                        pointer_down,
                        pointer_inside,
                        cursor_visible,
                        // Read rather than cached, so dragging the window to a
                        // display with a different density needs no event: the
                        // `Resized` that follows already reconfigures the
                        // surface, and this follows it for free.
                        scale_factor: win.scale_factor() as f32,
                        exit: false,
                    };
                    draw(&mut f);
                    let wants_exit = f.exit;
                    if f.cursor_visible != cursor_visible {
                        cursor_visible = f.cursor_visible;
                        win.set_cursor_visible(cursor_visible);
                    }
                    frame_index += 1;

                    // A minimised window has a zero-sized surface and acquiring
                    // from it is an error, not a frame to skip quietly.
                    if config.width > 0 && config.height > 0 {
                        if let wgpu::CurrentSurfaceTexture::Success(frame)
                        | wgpu::CurrentSurfaceTexture::Suboptimal(frame) =
                            surface.get_current_texture()
                        {
                            let view = frame
                                .texture
                                .create_view(&wgpu::TextureViewDescriptor::default());
                            match quad_view.as_mut() {
                                Some(quad) => quad.render(
                                    &mut ViewSurface {
                                        renderer: &mut renderer,
                                        view: &view,
                                        linear_framebuffer: false,
                                    },
                                    &mut scene,
                                    &camera,
                                ),
                                None => renderer.render(&mut scene, &camera, &view, false),
                            }
                            // Over the top of the render, and skipped entirely
                            // on a frame that drew nothing 2D.
                            if overlay.touched() {
                                let srgb_view = if srgb_format == config.format {
                                    None
                                } else {
                                    Some(frame.texture.create_view(
                                        &wgpu::TextureViewDescriptor {
                                            format: Some(srgb_format),
                                            ..Default::default()
                                        },
                                    ))
                                };
                                // Uploads only the rows that changed, which for
                                // a read-out in the corner is a tenth of them.
                                renderer.draw_overlay(
                                    &mut overlay,
                                    srgb_view.as_ref().unwrap_or(&view),
                                    srgb_format,
                                );
                            }
                            queue.present(frame);
                        }
                    }
                    if wants_exit {
                        target.exit();
                    }
                    win.request_redraw();
                }
                _ => {}
            }
        })
        .expect("event loop");
}
