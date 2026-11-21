//! A UI laid out, drawn and presented on a thread of its own.
//!
//! The main thread only runs the event loop and forwards window events to a
//! [`WindowThread`]. The UI thread owns the `Ui`, its `UiPass` and the
//! window's renderer. Make the main thread slow and the UI keeps
//! animating. Its input arrives late, but no frame waits for it.
//!
//! Keys:
//! - `b`: make the main thread busy, stalling 250 ms each time round its loop
//! - `[` / `]`: UI frame rate down / up
//! - click the button: counts clicks, handled on the UI thread
//!
//! ```text
//! cargo run --release --features ui --example ui_thread
//! ```

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::time::{Duration, Instant};

use threers::renderer::{FramePacer, ThreadEvent, UiPass, WindowRenderer, WindowThread};
use threers::style;
use threers::ui::input::{Event as UiEvent, Input, PointerButton};
use threers::ui::{Length, Ui};
use winit::event::{ElementState, Event, MouseButton, WindowEvent};
use winit::event_loop::ControlFlow;
use winit::keyboard::Key;

/// What the main thread and the UI thread share.
struct Shared {
    /// Whether the main thread stalls.
    busy: AtomicBool,
    /// The main thread's last trip round its loop, in ms.
    loop_ms: AtomicU32,
}

const RATES: [u32; 6] = [10, 24, 30, 60, 120, 240];

fn main() {
    env_logger::init();
    let attributes = winit::window::Window::default_attributes()
        .with_title("threers — UI on its own thread")
        .with_inner_size(winit::dpi::LogicalSize::new(720, 420));
    threers::sketch::run_window(attributes, move |window| {
        // Made on the main thread, then handed to the UI thread.
        let mut renderer = WindowRenderer::new(window.clone()).expect("a GPU for the window");
        renderer.set_present_mode(wgpu::PresentMode::Mailbox);
        let shared = Arc::new(Shared {
            busy: AtomicBool::new(false),
            loop_ms: AtomicU32::new(0),
        });

        let on_thread = shared.clone();
        let thread = WindowThread::spawn(window.clone(), renderer, move |ctl| {
            // Everything from here on lives on the UI thread.
            let shared = on_thread;
            let device = ctl.renderer().renderer().expect("the wgpu backend").device_arc();
            let mut pass = UiPass::new(&device);
            let mut ui = Ui::new();
            let mut input: Input<u32> = Input::new();
            let mut clicks = 0u32;

            let root = ui.root();
            ui.set_style(
                root,
                style! { display: flex; flex-direction: column; gap: 10px; padding: 24px; background: #14171d; color: #d6dae3; font-family: system-ui; font-size: 15px; },
            );
            let line = |ui: &mut Ui| ui.add_text(root, "");
            let (fps_line, main_line, clock_line) = (line(&mut ui), line(&mut ui), line(&mut ui));
            ui.add_text(root, "b busy main thread, [ / ] UI rate");
            let track = ui.add_box(
                root,
                style! { position: relative; height: 28px; border-radius: 14px; background: #232833; },
            );
            let dot = ui.add_box(track, style! { position: absolute; top: 2px; width: 24px; height: 24px; border-radius: 12px; background: #e0874a; });
            let button = ui.add_box(
                root,
                style! { padding: 10px 16px; border-radius: 8px; background: #2f6fd0; color: #ffffff; align-self: flex-start; cursor: pointer; },
            );
            let label = ui.add_text(button, "Clicked 0 times");
            input.on_click(button, move |n, ui, _| {
                *n += 1;
                ui.set_text(label, format!("Clicked {n} times"));
            });

            let start = Instant::now();
            let mut pacer = FramePacer::new(60.0);
            let (mut frames, mut tick, mut fps) = (0u32, Instant::now(), 0.0f32);
            let mut cursor = (0.0f32, 0.0f32);

            move |event, ctl| match event {
                ThreadEvent::Window(event) => {
                    let s = ctl.scale_factor() as f32;
                    let ui_event = match event {
                        WindowEvent::CursorMoved { position, .. } => {
                            cursor = (position.x as f32 / s, position.y as f32 / s);
                            Some(UiEvent::PointerMove { x: cursor.0, y: cursor.1 })
                        }
                        WindowEvent::MouseInput { state, button: MouseButton::Left, .. } => {
                            let (x, y, button) = (cursor.0, cursor.1, PointerButton::Primary);
                            Some(match state {
                                ElementState::Pressed => UiEvent::PointerDown { x, y, button },
                                ElementState::Released => UiEvent::PointerUp { x, y, button },
                            })
                        }
                        WindowEvent::KeyboardInput { event, .. } if event.state == ElementState::Pressed => {
                            if let Key::Character(c @ ("[" | "]")) = event.logical_key.as_ref() {
                                let now = pacer.fps().unwrap_or(60.0).round() as u32;
                                let i = RATES.iter().position(|&r| r == now).unwrap_or(3);
                                let i = if c == "]" { (i + 1).min(RATES.len() - 1) } else { i.saturating_sub(1) };
                                pacer.set_fps(RATES[i] as f32);
                            }
                            None
                        }
                        _ => None,
                    };
                    if let Some(e) = ui_event {
                        input.dispatch(&mut ui, &mut clicks, e);
                    }
                    ctl.request_redraw();
                }
                ThreadEvent::Redraw => {
                    // Held to the cap whoever asked; the next frame is due then.
                    let now = Instant::now();
                    if !pacer.due(now) {
                        ctl.redraw_at(pacer.next());
                        return;
                    }
                    ctl.redraw_at(pacer.next());
                    frames += 1;
                    if tick.elapsed() >= Duration::from_millis(500) {
                        fps = frames as f32 / tick.elapsed().as_secs_f32();
                        (frames, tick) = (0, Instant::now());
                    }
                    let t = start.elapsed().as_secs_f32();
                    input.tick(&mut ui, &mut clicks, t as f64);
                    let cap = pacer.fps().map_or("off".into(), |f| format!("{f:.0}"));
                    ui.set_text(fps_line, format!("UI thread: {fps:.0} fps (cap {cap})"));
                    let busy = if shared.busy.load(Ordering::Relaxed) { "busy" } else { "idle" };
                    let ms = shared.loop_ms.load(Ordering::Relaxed);
                    ui.set_text(main_line, format!("Main thread: {busy}, {ms} ms a trip round its loop"));
                    ui.set_text(clock_line, format!("UI clock: {t:.2} s"));

                    let (w, h) = ctl.logical_size();
                    let room = (w - 48.0 - 28.0).max(0.0);
                    let x = 2.0 + room * (0.5 - 0.5 * (t * 2.0).cos());
                    let mut moved = ui.style(dot).clone();
                    moved.left = Length::Px(x);
                    ui.set_style(dot, moved);

                    let s = ctl.scale_factor() as f32;
                    ui.layout(w, h, &mut pass.text_measure(s));
                    let scene = ui.scene();
                    ctl.renderer().present_with(|device, queue, target, size, format| {
                        pass.draw_retained(device, queue, &scene, s, target, size, format, [0.08, 0.09, 0.11, 1.0]);
                    });
                }
                _ => {}
            }
        })
        .expect("the wgpu backend: unset THREERS_BACKEND");

        let mut thread = Some(thread);
        let mut last = Instant::now();
        move |event, target| match event {
            Event::WindowEvent { event, .. } => match event {
                WindowEvent::CloseRequested => {
                    if let Some(t) = thread.take() {
                        t.close();
                    }
                    target.exit();
                }
                WindowEvent::KeyboardInput { ref event, .. }
                    if event.state == ElementState::Pressed
                        && event.logical_key.as_ref() == Key::Character("b") =>
                {
                    shared.busy.fetch_xor(true, Ordering::Relaxed);
                }
                event => {
                    if let Some(t) = &thread {
                        t.send(event);
                    }
                }
            },
            Event::AboutToWait => {
                if shared.busy.load(Ordering::Relaxed) {
                    // Work the main thread does that a frame never waits for.
                    std::thread::sleep(Duration::from_millis(250));
                    target.set_control_flow(ControlFlow::Poll);
                } else {
                    target.set_control_flow(ControlFlow::Wait);
                }
                let now = Instant::now();
                shared.loop_ms.store(now.duration_since(last).as_millis() as u32, Ordering::Relaxed);
                last = now;
                if thread.as_ref().is_some_and(WindowThread::is_finished) {
                    target.exit();
                }
            }
            _ => {}
        }
    });
}
