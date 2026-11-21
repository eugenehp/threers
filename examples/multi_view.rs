//! Four views of one scene — perspective, top, front, side — drawn on one
//! render thread by one renderer, shown in a UI that draws only when
//! something changes.
//!
//! One [`RenderLoop`] draws all four: the mesh is uploaded once and every
//! pipeline built once. Each view has its own frame rate; the one under the
//! pointer gets priority; a view hidden in the UI isn't drawn at all; under a
//! shared GPU budget the views trade resolution, least important first. Each
//! box shows its view's numbers.
//!
//! Keys:
//! - `1`–`4`: hide or show a view (a hidden view costs nothing)
//! - `b`: share an 8 ms GPU budget between the views
//! - `s`: the side views draw only when something changes, or at 15 fps
//! - `space`: stop or start the knot turning
//! - `o`: the UI draws continuously, or only on change
//!
//! ```text
//! cargo run --release --features ui --example multi_view
//! ```

use std::time::Instant;

use threers::cameras::Camera;
use threers::core::ObjectId;
use threers::renderer::{
    Redraw, RenderLoop, RenderThread, UiPass, ViewId, ViewOptions, ViewProducer, WindowRenderer,
};
use threers::style;
use threers::ui::redraw::{RedrawMode, UiRedraw};
use threers::ui::surface_view::SurfaceView;
use threers::ui::{NodeId, Ui};
use threers::{
    AmbientLight, Color, DirectionalLight, Euler, GridHelper, Mesh, Object3D, OrthographicCamera,
    PerspectiveCamera, Scene, StandardMaterial, TorusKnotGeometry, Vector3,
};
use winit::event::{ElementState, Event, WindowEvent};
use winit::event_loop::ControlFlow;
use winit::keyboard::{Key, NamedKey};

const NAMES: [&str; 4] = ["Perspective", "Top", "Front", "Side"];

struct World {
    scene: Scene,
    knot: ObjectId,
    /// One camera per view, by view index.
    cameras: Vec<Box<dyn Camera + Send>>,
    spinning: bool,
    angle: f32,
}

fn world() -> World {
    let mut scene = Scene::new();
    scene.background = Color::from_hex(0x101418);
    scene.add_light(AmbientLight::new(Color::from_hex(0xffffff), 0.4));
    let mut key = Object3D::light(DirectionalLight::new(Color::from_hex(0xffffff), 1.2));
    key.position = Vector3::new(3.0, 4.0, 5.0);
    scene.add(key);
    scene.add(GridHelper::new(4.0, 8, Color::from_hex(0x3a4150), Color::from_hex(0x262b35)));
    let knot = scene.add(Object3D::mesh(Mesh::new(
        TorusKnotGeometry::new(0.8, 0.28, 256, 32, 2, 3),
        StandardMaterial::new(Color::from_hex(0xd0763a))
            .with_roughness(0.35)
            .into(),
    )));
    let mut persp = PerspectiveCamera::new(45.0, 1.0, 0.1, 100.0);
    persp.position = Vector3::new(3.0, 2.2, 3.5);
    persp.look_at(Vector3::ZERO);
    let ortho = |at: Vector3| -> Box<dyn Camera + Send> {
        let mut c = OrthographicCamera::new(-1.6, 1.6, 1.6, -1.6, 0.1, 100.0);
        c.position = at;
        c.look_at(Vector3::ZERO);
        Box::new(c)
    };
    World {
        scene,
        knot,
        cameras: vec![
            Box::new(persp),
            ortho(Vector3::new(0.0, 6.0, 0.001)),
            ortho(Vector3::new(0.0, 0.0, 6.0)),
            ortho(Vector3::new(6.0, 0.0, 0.0)),
        ],
        spinning: true,
        angle: 0.0,
    }
}

/// One cell of the grid: a label over the view's box.
struct Cell {
    cell: NodeId,
    label: NodeId,
    surface: SurfaceView,
    id: ViewId,
    hidden: bool,
}

fn main() {
    env_logger::init();
    let attributes = winit::window::Window::default_attributes()
        .with_title("threers — four views, one render thread")
        .with_inner_size(winit::dpi::LogicalSize::new(1200, 800));
    threers::sketch::run_window(attributes, move |window| {
        let mut view = WindowRenderer::new(window.clone()).expect("a GPU for the window");
        view.set_present_mode(wgpu::PresentMode::Mailbox);
        let gpu = view.shared_gpu().expect("a GPU to share");
        let mut pass = UiPass::new(&gpu.device);

        // One producer, four views, one loop drawing whichever is due.
        let (mut producer, first) = ViewProducer::on(&gpu, ViewOptions::default());
        let mut frames = vec![(ViewId::PRIMARY, first)];
        for _ in 1..4 {
            frames.push(producer.add_view());
        }
        let render = RenderThread::spawn(RenderLoop::new(producer, world(), |w: &mut World, view| {
            // Turn the knot once a round, on the perspective view's clock.
            if view.id() == ViewId::PRIMARY && w.spinning {
                w.angle += view.dt * 0.6;
                let a = w.angle;
                if let Some(knot) = w.scene.get_mut(w.knot) {
                    knot.quaternion = Euler::new(a * 0.5, a, 0.0).to_quaternion();
                }
            }
            let camera = &mut w.cameras[view.id().index()];
            camera.set_aspect(view.aspect());
            view.render(&mut w.scene, camera.as_ref());
        }));
        render.set_fps(ViewId::PRIMARY, 60.0);
        for (id, _) in &frames[1..] {
            render.set_fps(*id, 15.0);
        }

        // The UI: a sidebar and a 2×2 grid of views.
        let mut ui = Ui::new();
        let root = ui.root();
        ui.set_style(
            root,
            style! { display: flex; flex-direction: row; background: #12151b; color: #d6dae3; font-family: system-ui; font-size: 13px; },
        );
        let side = ui.add_box(
            root,
            style! { display: flex; flex-direction: column; gap: 8px; padding: 16px; width: 220px; },
        );
        let status = ui.add_text(side, "");
        ui.add_text(side, "1–4 hide/show, b budget, s side views on change, space spin, o UI on change");
        let grid = ui.add_box(
            root,
            style! { flex-grow: 1; display: flex; flex-direction: column; gap: 8px; padding: 8px; },
        );
        let rows = [
            ui.add_box(grid, style! { flex-grow: 1; display: flex; flex-direction: row; gap: 8px; }),
            ui.add_box(grid, style! { flex-grow: 1; display: flex; flex-direction: row; gap: 8px; }),
        ];
        let mut cells: Vec<Cell> = frames
            .into_iter()
            .enumerate()
            .map(|(i, (id, frames))| {
                let cell = ui.add_box(
                    rows[i / 2],
                    style! { flex-grow: 1; display: flex; flex-direction: column; gap: 4px; },
                );
                let label = ui.add_text(cell, NAMES[i]);
                let surface = SurfaceView::new(
                    &mut ui,
                    cell,
                    style! { flex-grow: 1; border-radius: 8px; background: #1b1f27; },
                    frames,
                );
                Cell { cell, label, surface, id, hidden: false }
            })
            .collect();

        let mut redraw = UiRedraw::new(RedrawMode::OnChange, 60.0);
        for c in &cells {
            let waker = window.clone();
            c.surface.frames().on_frame(move || waker.request_redraw());
        }
        let (mut budget, mut side_on_change, mut spinning) = (false, false, true);
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
                        let (x, y) = (position.x as f32 / scale, position.y as f32 / scale);
                        // The view under the pointer matters most.
                        for c in &cells {
                            let over = c.surface.point(&ui, x, y).is_some();
                            c.surface.frames().set_priority(if over { 10 } else { 0 });
                        }
                    }
                    WindowEvent::KeyboardInput { event, .. } if event.state == ElementState::Pressed => {
                        match event.logical_key.as_ref() {
                            Key::Character(k @ ("1" | "2" | "3" | "4")) => {
                                let c = &mut cells[k.parse::<usize>().unwrap() - 1];
                                c.hidden = !c.hidden;
                                let style = if c.hidden {
                                    style! { display: none; }
                                } else {
                                    style! { flex-grow: 1; display: flex; flex-direction: column; gap: 4px; }
                                };
                                ui.set_style(c.cell, style);
                            }
                            Key::Character("b") => {
                                budget = !budget;
                                render.set_gpu_budget(budget.then_some(8.0));
                            }
                            Key::Character("s") => {
                                side_on_change = !side_on_change;
                                let redraw_mode = if side_on_change { Redraw::OnDemand } else { Redraw::Continuous };
                                for c in &cells[1..] {
                                    render.set_redraw(c.id, redraw_mode);
                                }
                            }
                            Key::Named(NamedKey::Space) => {
                                spinning = !spinning;
                                render.update(move |w| w.spinning = spinning);
                            }
                            Key::Character("o") => redraw.set_mode(match redraw.mode() {
                                RedrawMode::Continuous => RedrawMode::OnChange,
                                RedrawMode::OnChange => RedrawMode::Continuous,
                            }),
                            _ => {}
                        }
                    }
                    WindowEvent::RedrawRequested => {
                        let views: Vec<_> = cells.iter().map(|c| c.surface.frames()).collect();
                        if !redraw.should_draw(&ui, &views, Instant::now()) {
                            return;
                        }
                        for (i, c) in cells.iter().enumerate() {
                            let s = c.surface.stats();
                            let text = match c.surface.failure() {
                                Some(why) => format!("{}: {why}", NAMES[i]),
                                None if !c.surface.frames().is_visible() => format!("{}: hidden, not drawn", NAMES[i]),
                                None => format!(
                                    "{}  {:.0} fps  {:.0}%  {:.1} ms",
                                    NAMES[i],
                                    s.fps,
                                    s.scale * 100.0,
                                    s.frame_ms
                                ),
                            };
                            ui.set_text(c.label, text);
                        }
                        let (drawn, idle) = redraw.counts();
                        ui.set_text(
                            status,
                            format!(
                                "UI {} · drawn {drawn} · idle {idle}\nbudget {} · side views {}",
                                match redraw.mode() {
                                    RedrawMode::Continuous => "continuous",
                                    RedrawMode::OnChange => "on change",
                                },
                                if budget { "8 ms" } else { "off" },
                                if side_on_change { "on change" } else { "15 fps" },
                            ),
                        );
                        let scale = window_for_loop.scale_factor() as f32;
                        let (w, h) = view.size();
                        ui.layout(w as f32 / scale, h as f32 / scale, &mut pass.text_measure(scale));
                        for c in &mut cells {
                            c.surface.update(&ui, &mut pass, scale);
                        }
                        let scene = ui.scene();
                        view.present_with(|device, queue, target, size, format| {
                            pass.draw_retained(device, queue, &scene, scale, target, size, format, [0.07, 0.08, 0.1, 1.0]);
                        });
                    }
                    _ => {}
                }
            }
            Event::AboutToWait => {
                let views: Vec<_> = cells.iter().map(|c| c.surface.frames()).collect();
                match redraw.next_wake(&ui, &views) {
                    Some(at) if at <= Instant::now() => window_for_loop.request_redraw(),
                    Some(at) => target.set_control_flow(ControlFlow::WaitUntil(at)),
                    None => target.set_control_flow(ControlFlow::Wait),
                }
            }
            _ => {}
        }
    });
}
