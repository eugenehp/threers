//! A UI and a 3D view, each drawn on its own thread at its own rate.
//!
//! The 3D view renders on a thread of its own ([`RenderThread`]); the window's
//! thread lays out and draws the UI, and shows the view's newest frame in a
//! box ([`SurfaceView`]) without waiting for it. Make the 3D frames slow and
//! the UI stays smooth, showing the newest frame it has.
//!
//! Keys:
//! - `-` / `=`: 3D frame rate down / up
//! - `[` / `]`: UI frame rate cap down / up
//! - `h`: make each 3D frame take 80 ms longer, as a heavy scene would
//! - `space`: pause the 3D view (the UI keeps running)
//! - `a`: adaptive resolution: the 3D view draws fewer pixels while its GPU
//!   frames run over the time its frame rate allows
//! - `o`: UI on change: draw only for input, a change in the UI or a new 3D
//!   frame (pause the 3D view and the UI stops drawing altogether)
//! - drag in the 3D view: orbit (sent to the render thread)
//!
//! ```text
//! cargo run --release --features ui --example threaded_view
//! ```

use std::time::{Duration, Instant};

use threers::cameras::Camera;
use threers::core::ObjectId;
use threers::renderer::{
    AdaptiveScale, RenderLoop, RenderThread, UiPass, ViewId, ViewOptions, ViewProducer,
    WindowRenderer,
};
use threers::style;
use threers::ui::redraw::{RedrawMode, UiRedraw};
use threers::ui::surface_view::SurfaceView;
use threers::ui::Ui;
use threers::{
    AmbientLight, Color, DirectionalLight, Euler, Mesh, Object3D, PerspectiveCamera, Scene,
    StandardMaterial, TorusKnotGeometry, Vector3,
};
use winit::event::{ElementState, Event, MouseButton, WindowEvent};
use winit::event_loop::ControlFlow;
use winit::keyboard::{Key, NamedKey};

/// Everything the render thread owns.
struct World {
    scene: Scene,
    knot: ObjectId,
    camera: PerspectiveCamera,
    yaw: f32,
    pitch: f32,
    stall_ms: u64,
}

const RATES: [u32; 7] = [5, 10, 15, 24, 30, 60, 120];

fn step(rate: u32, up: bool) -> u32 {
    let i = RATES.iter().position(|&r| r == rate).unwrap_or(3);
    RATES[if up { (i + 1).min(RATES.len() - 1) } else { i.saturating_sub(1) }]
}

fn world() -> World {
    let mut scene = Scene::new();
    scene.background = Color::from_hex(0x101418);
    scene.add_light(AmbientLight::new(Color::from_hex(0xffffff), 0.4));
    let mut key = Object3D::light(DirectionalLight::new(Color::from_hex(0xffffff), 1.2));
    key.position = Vector3::new(3.0, 4.0, 5.0);
    scene.add(key);
    let knot = scene.add(Object3D::mesh(Mesh::new(
        TorusKnotGeometry::new(0.8, 0.28, 256, 32, 2, 3),
        StandardMaterial::new(Color::from_hex(0xd0763a))
            .with_roughness(0.35)
            .into(),
    )));
    World {
        scene,
        knot,
        camera: PerspectiveCamera::new(45.0, 1.0, 0.1, 100.0),
        yaw: 0.0,
        pitch: 0.0,
        stall_ms: 0,
    }
}

fn main() {
    env_logger::init();
    let attributes = winit::window::Window::default_attributes()
        .with_title("threers — threaded view")
        .with_inner_size(winit::dpi::LogicalSize::new(1000, 640));
    threers::sketch::run_window(attributes, move |window| {
        let mut view = WindowRenderer::new(window.clone()).expect("a GPU for the window");
        view.set_present_mode(wgpu::PresentMode::Mailbox);
        let gpu = view.shared_gpu().expect("a GPU to share");
        let mut pass = UiPass::new(&gpu.device);

        // The render thread: what one frame draws, and nothing else.
        let (producer, frames) = ViewProducer::on(&gpu, ViewOptions::default());
        let render = RenderThread::spawn(RenderLoop::new(producer, world(), |w: &mut World, view| {
            let (yaw, pitch) = (w.yaw, w.pitch);
            w.camera.position = Vector3::new(
                4.0 * pitch.cos() * yaw.sin(),
                4.0 * pitch.sin(),
                4.0 * pitch.cos() * yaw.cos(),
            );
            w.camera.look_at(Vector3::new(0.0, 0.0, 0.0));
            w.camera.set_aspect(view.aspect());
            if let Some(obj) = w.scene.get_mut(w.knot) {
                obj.quaternion = Euler::new(view.time * 0.5, view.time * 0.8, 0.0).to_quaternion();
            }
            std::thread::sleep(Duration::from_millis(w.stall_ms));
            view.render(&mut w.scene, &w.camera);
        }));
        let (mut fps_3d, mut stall, mut paused, mut adaptive) = (24u32, false, false, false);
        render.set_fps(ViewId::PRIMARY, fps_3d as f32);

        // The UI: a sidebar of readouts beside the 3D box.
        let mut ui = Ui::new();
        let root = ui.root();
        ui.set_style(
            root,
            style! { display: flex; flex-direction: row; background: #12151b; color: #d6dae3; font-family: system-ui; font-size: 14px; },
        );
        let side = ui.add_box(
            root,
            style! { display: flex; flex-direction: column; gap: 8px; padding: 16px; width: 240px; },
        );
        let line = |ui: &mut Ui| ui.add_text(side, "");
        let (ui_line, view_line, state_line, scale_line, clock_line) =
            (line(&mut ui), line(&mut ui), line(&mut ui), line(&mut ui), line(&mut ui));
        ui.add_text(side, "- / = 3D rate, [ / ] UI rate, h heavy 3D, space pause, a adaptive, o UI on change, drag to orbit");
        let mut surface = SurfaceView::new(
            &mut ui,
            root,
            style! { flex-grow: 1; margin: 12px; border-radius: 10px; background: #1b1f27; },
            frames,
        );

        let mut redraw = UiRedraw::new(RedrawMode::Continuous, 60.0);
        // A new 3D frame wakes the UI, from the render thread.
        let waker = window.clone();
        surface.frames().on_frame(move || waker.request_redraw());
        let start = Instant::now();
        let (mut ui_frames, mut tick, mut ui_fps) = (0u32, Instant::now(), 0.0f32);
        let (mut cursor, mut dragging) = ((0.0f32, 0.0f32), false);
        let window_for_loop = window.clone();

        move |event, target| match event {
            Event::WindowEvent { event, window_id } if window_id == window_for_loop.id() => {
                if event != WindowEvent::RedrawRequested {
                    redraw.request();
                }
                match event {
                    WindowEvent::CloseRequested => target.exit(),
                    WindowEvent::Resized(size) => view.resize(size.width, size.height),
                    WindowEvent::CursorMoved { position, .. } => {
                        let scale = window_for_loop.scale_factor() as f32;
                        let at = (position.x as f32 / scale, position.y as f32 / scale);
                        if dragging {
                            let (dx, dy) = (at.0 - cursor.0, at.1 - cursor.1);
                            render.update(move |w| {
                                w.yaw -= dx * 0.01;
                                w.pitch = (w.pitch + dy * 0.01).clamp(-1.4, 1.4);
                            });
                        }
                        cursor = at;
                    }
                    WindowEvent::MouseInput { state, button: MouseButton::Left, .. } => {
                        // A drag starts on the view and goes on wherever the pointer goes.
                        dragging = state == ElementState::Pressed
                            && surface.point(&ui, cursor.0, cursor.1).is_some();
                    }
                    WindowEvent::KeyboardInput { event, .. } if event.state == ElementState::Pressed => {
                        let ui_cap = redraw.max_fps().unwrap_or(60.0).round() as u32;
                        match event.logical_key.as_ref() {
                            Key::Character("-") | Key::Character("=") => {
                                fps_3d = step(fps_3d, event.logical_key == Key::Character("=".into()));
                                render.set_fps(ViewId::PRIMARY, fps_3d as f32);
                            }
                            Key::Character("[") => redraw.set_max_fps(step(ui_cap, false) as f32),
                            Key::Character("]") => redraw.set_max_fps(step(ui_cap, true) as f32),
                            Key::Character("h") => {
                                stall = !stall;
                                let ms = if stall { 80 } else { 0 };
                                render.update(move |w| w.stall_ms = ms);
                            }
                            Key::Character("a") => {
                                adaptive = !adaptive;
                                let budget = 1000.0 / fps_3d as f32;
                                render.with_loop(move |l| {
                                    let mut v = l.producer().view(ViewId::PRIMARY);
                                    if adaptive {
                                        v.set_adaptive_scale(Some(AdaptiveScale::new(budget)));
                                    } else {
                                        v.set_render_scale(1.0);
                                    }
                                });
                            }
                            Key::Character("o") => redraw.set_mode(match redraw.mode() {
                                RedrawMode::Continuous => RedrawMode::OnChange,
                                RedrawMode::OnChange => RedrawMode::Continuous,
                            }),
                            Key::Named(NamedKey::Space) => {
                                paused = !paused;
                                render.set_paused(ViewId::PRIMARY, paused);
                            }
                            _ => {}
                        }
                    }
                    WindowEvent::RedrawRequested => {
                        if !redraw.should_draw(&ui, &[surface.frames()], Instant::now()) {
                            return;
                        }
                        ui_frames += 1;
                        if tick.elapsed() >= Duration::from_millis(500) {
                            ui_fps = ui_frames as f32 / tick.elapsed().as_secs_f32();
                            (ui_frames, tick) = (0, Instant::now());
                        }
                        let stats = surface.stats();
                        let cap = redraw.max_fps().map_or("off".into(), |f| format!("{f:.0}"));
                        let mode = match redraw.mode() {
                            RedrawMode::Continuous => "",
                            RedrawMode::OnChange => ", on change",
                        };
                        ui.set_text(ui_line, format!("UI: {ui_fps:.0} fps (cap {cap}{mode})"));
                        ui.set_text(view_line, format!("3D: {:.0} fps (cap {fps_3d})", stats.fps));
                        ui.set_text(
                            state_line,
                            match (surface.failure(), paused, stall) {
                                (Some(why), ..) => why,
                                (_, true, _) => "3D paused".into(),
                                (_, _, false) => "3D frames: light".into(),
                                (_, _, true) => "3D frames: +80 ms each".into(),
                            },
                        );
                        let mode = if adaptive { "adaptive" } else { "fixed" };
                        ui.set_text(
                            scale_line,
                            format!(
                                "3D resolution: {:.0}% ({mode}), {:.1} ms a frame",
                                stats.scale * 100.0,
                                stats.frame_ms
                            ),
                        );
                        // A clock that ticks only in continuous mode: on change,
                        // a still UI really is still.
                        if redraw.mode() == RedrawMode::Continuous {
                            ui.set_text(clock_line, format!("UI clock: {:.2} s", start.elapsed().as_secs_f32()));
                        }

                        let scale = window_for_loop.scale_factor() as f32;
                        let (w, h) = view.size();
                        ui.layout(w as f32 / scale, h as f32 / scale, &mut pass.text_measure(scale));
                        surface.update(&ui, &mut pass, scale);
                        let scene = ui.scene();
                        view.present_with(|device, queue, target, size, format| {
                            pass.draw_retained(device, queue, &scene, scale, target, size, format, [0.07, 0.08, 0.1, 1.0]);
                        });
                    }
                    _ => {}
                }
            }
            Event::AboutToWait => match redraw.next_wake(&ui, &[surface.frames()]) {
                Some(at) if at <= Instant::now() => window_for_loop.request_redraw(),
                Some(at) => target.set_control_flow(ControlFlow::WaitUntil(at)),
                None => target.set_control_flow(ControlFlow::Wait),
            },
            _ => {}
        }
    });
}
