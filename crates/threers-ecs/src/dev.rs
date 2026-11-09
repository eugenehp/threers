//! Looking inside a running game.
//!
//! | Key | Does |
//! |---|---|
//! | F2 | Fly a free camera, the game paused: WASD, Q and E, Shift for speed, the right mouse button or arrows to look |
//! | Click | With the panel up, select what is under the pointer |
//! | F3 | Show frame time, the slowest systems, any that failed, and the selected entity |
//! | F4 | Show colliders, and for each car its tyres' grip, suspension and speed (`physics` feature) |
//! | F5 | Pause or resume |
//! | F6 | While paused, run one fixed step |
//! | F7 | Select the next entity with registered components |
//! | F8 | Move to the selected entity's next field |
//! | F9, F10 | Lower or raise that field; a yes/no field flips |
//! | F12 | Start or stop recording input, to `recordings/` (see [`replay`](crate::replay)) |
//!
//! Selecting works from the list [`World::register`] keeps, so the panel shows
//! each field by name with nothing extra written per component.

use crate::app::{App, Plugin, Time};
use crate::hud::with_hud;
use crate::input::{Input, Key};
use crate::registry::FieldValue;
use crate::world::{Entity, World};
use threers::math::Color;

/// Adds the tools, and turns on [`App::catch_panics`]: a system that panics is
/// switched off and named on the panel, and the game goes on. Leave it out of a
/// release build.
#[derive(Default)]
pub struct DevTools;

/// What the tools are showing.
#[derive(Debug, Default)]
pub struct DevState {
    /// The panel of timings and counts, F3.
    pub panel: bool,
    /// Collider outlines, F4.
    pub colliders: bool,
    /// What F7 picked, for the inspector.
    pub selected: Option<Entity>,
    /// Which of the selected entity's fields F9 and F10 change.
    pub field: usize,
    /// Where the last recording went, or why it could not.
    pub saved: Option<String>,
    /// The free camera, while flying: where it is and which way it looks, the
    /// game's camera to put back, and whether the game was paused already.
    pub flying: Option<Flight>,
    /// Recent frame times, in milliseconds.
    frames: Vec<f32>,
    #[cfg(feature = "physics")]
    lines: Option<Entity>,
}

/// A free camera's state.
#[derive(Debug, Clone)]
pub struct Flight {
    /// Where the free camera is.
    pub position: threers::math::Vector3,
    /// Which way it faces about the vertical, in radians.
    pub yaw: f32,
    /// How far it looks up or down, in radians.
    pub pitch: f32,
    saved: threers::cameras::PerspectiveCamera,
    was_paused: bool,
}

impl DevState {
    /// Mean frame time over the recent frames, in milliseconds.
    pub fn frame_ms(&self) -> f32 {
        self.frames.iter().sum::<f32>() / self.frames.len().max(1) as f32
    }
}

impl Plugin for DevTools {
    fn build(self, app: &mut App) {
        // A bug in one system should not end the session it is being found in.
        app.catch_panics(true);
        app.plugin(crate::console::Console);
        let step =
            crate::schedule::System::new(dev_step).after_if_added(crate::camera::chase_camera);
        app.resource(DevState::default()).system(step);
    }
}

/// Read the keys and draw the panel.
pub fn dev_step(w: &mut World) {
    let input = w.resource::<Input>();
    let keys = [
        Key::F3,
        Key::F4,
        Key::F5,
        Key::F6,
        Key::F7,
        Key::F8,
        Key::F9,
        Key::F10,
        Key::F12,
    ]
    .map(|key| input.was_pressed(key));
    let [panel, colliders, pause, step, next, next_field, lower, raise, record] = keys;
    let time = *w.resource::<Time>();

    let inspectable = w.inspectable();
    let state = w.resource_mut::<DevState>();
    state.panel ^= panel;
    state.colliders ^= colliders;
    state.frames.push(time.real_delta * 1000.0);
    if state.frames.len() > 60 {
        state.frames.remove(0);
    }
    if next && !inspectable.is_empty() {
        let at = state
            .selected
            .and_then(|id| inspectable.iter().position(|&e| e == id));
        state.selected = Some(inspectable[at.map_or(0, |i| (i + 1) % inspectable.len())]);
        state.field = 0;
    }
    if next_field {
        state.field += 1;
    }
    let (show, selected, frame_ms) = (state.panel, state.selected, state.frame_ms());

    let clock = w.resource_mut::<Time>();
    clock.paused ^= pause;
    if step && clock.paused {
        clock.step_once += 1;
    }
    let paused = clock.paused;

    if record {
        toggle_recording(w);
    }
    if w.resource::<Input>().was_pressed(Key::F2) {
        toggle_flight(w);
    }
    fly(w);
    if show
        && w.resource::<Input>()
            .was_pressed(crate::input::MouseButton::Left)
        && let Some(id) = pick(w)
    {
        let state = w.resource_mut::<DevState>();
        state.selected = Some(id);
        state.field = 0;
    }
    if let (Some(id), true) = (selected, lower || raise) {
        nudge(w, id, if raise { 1.0 } else { -1.0 });
    }

    #[cfg(feature = "physics")]
    draw_colliders(w);
    #[cfg(feature = "physics")]
    if w.resource::<DevState>().colliders {
        draw_vehicles(w);
    }

    if show {
        let mut lines = vec![
            format!("{frame_ms:.1} ms  {:.0} fps", 1000.0 / frame_ms.max(0.001)),
            format!(
                "{} nodes, step {}",
                w.scene.arena.nodes.len(),
                time.fixed_steps
            ),
        ];
        if paused {
            lines.push("paused: F5 resumes, F6 steps".to_owned());
        }
        for failure in &w.resource::<crate::profile::Failures>().0 {
            let mut message = failure.message.clone();
            message.truncate(60);
            lines.push(format!(
                "FAILED {}: {message}",
                crate::profile::short(failure.system)
            ));
        }
        let profile = w.resource::<crate::profile::Profile>();
        lines.push(format!("systems {:.2} ms", profile.total_ms()));
        for (name, time) in profile.slowest().into_iter().take(5) {
            lines.push(format!(
                "  {:>6.2} ms  {}",
                time.ms,
                crate::profile::short(name)
            ));
        }
        if w.is_recording() {
            lines.push("recording: F12 stops".to_owned());
        } else if let Some(saved) = &w.resource::<DevState>().saved {
            lines.push(saved.clone());
        }
        if let Some(id) = selected {
            lines.push(format!("{id:?}  F8 field, F9/F10 change"));
            let cursor = w.resource::<DevState>().field;
            let mut index = 0;
            let total: usize = w.inspect(id).iter().map(|(_, f)| f.len()).sum();
            for (name, fields) in w.inspect(id) {
                lines.push(format!(" {name}"));
                for (field, value) in fields {
                    let mark = if total > 0 && index == cursor % total {
                        ">"
                    } else {
                        " "
                    };
                    lines.push(format!(" {mark}{field}: {value}"));
                    index += 1;
                }
            }
        }
        with_hud(w, |hud| {
            hud.no_stroke();
            hud.fill_alpha(Color::from_hex(0x000000), 0.6);
            hud.rect(8.0, 8.0, 420.0, 12.0 + 18.0 * lines.len() as f32);
            hud.fill(Color::WHITE);
            hud.text_size(14.0);
            for (index, line) in lines.iter().enumerate() {
                hud.text(16.0, 26.0 + 18.0 * index as f32, line);
            }
        });
    }
}

/// A step that suits a number's size: a tenth of its leading digit's place.
fn step_for(value: f64) -> f64 {
    if value == 0.0 {
        return 0.1;
    }
    let place = 10f64.powf(value.abs().log10().floor());
    let step = place / 10.0;
    if value.fract() == 0.0 {
        step.max(1.0)
    } else {
        step
    }
}

/// Move the field under the cursor by a step, or flip it.
fn nudge(w: &mut World, id: Entity, direction: f64) {
    let cursor = w.resource::<DevState>().field;
    let fields: Vec<(&'static str, &'static str, FieldValue)> = w
        .inspect(id)
        .into_iter()
        .flat_map(|(component, fields)| fields.into_iter().map(move |(f, v)| (component, f, v)))
        .collect();
    if fields.is_empty() {
        return;
    }
    let (component, field, value) = &fields[cursor % fields.len()];
    let changed = match value {
        FieldValue::Number(v) => FieldValue::Number(v + direction * step_for(*v)),
        FieldValue::Bool(v) => FieldValue::Bool(!v),
        _ => return,
    };
    w.set_field(id, component, field, &changed);
}

fn toggle_flight(w: &mut World) {
    use crate::app::MainCamera;
    let Some(camera) = w.get_resource::<MainCamera>().map(|c| c.0.clone()) else {
        return;
    };
    let flying = w.resource_mut::<DevState>().flying.take();
    match flying {
        Some(flight) => {
            w.resource_mut::<MainCamera>().0 = flight.saved;
            w.resource_mut::<Time>().paused = flight.was_paused;
        }
        None => {
            let look = (camera.target - camera.position).normalize();
            let flight = Flight {
                position: camera.position,
                yaw: look.x.atan2(-look.z),
                pitch: look.y.clamp(-1.0, 1.0).asin(),
                saved: camera,
                was_paused: w.resource::<Time>().paused,
            };
            w.resource_mut::<Time>().paused = true;
            w.resource_mut::<DevState>().flying = Some(flight);
        }
    }
}

/// Move the free camera by the keys and the mouse, in real time, as the game
/// is paused.
fn fly(w: &mut World) {
    use crate::app::MainCamera;
    use crate::input::MouseButton;
    use threers::math::Vector3;
    let Some(mut flight) = w.resource::<DevState>().flying.clone() else {
        return;
    };
    let input = w.resource::<Input>();
    let dt = w.resource::<Time>().real_delta.min(0.1);
    let key = |k: Key| if input.is_down(k) { 1.0f32 } else { 0.0 };
    let speed = if input.is_down(Key::ShiftLeft) {
        40.0
    } else {
        10.0
    } * dt;
    let turn = 1.6 * dt;
    flight.yaw += (key(Key::ArrowRight) - key(Key::ArrowLeft)) * turn;
    flight.pitch += (key(Key::ArrowUp) - key(Key::ArrowDown)) * turn;
    if input.is_down(MouseButton::Right) {
        flight.yaw += input.mouse_delta.x * 0.004;
        flight.pitch -= input.mouse_delta.y * 0.004;
    }
    flight.pitch = flight.pitch.clamp(-1.5, 1.5);
    let look = Vector3::new(
        flight.yaw.sin() * flight.pitch.cos(),
        flight.pitch.sin(),
        -flight.yaw.cos() * flight.pitch.cos(),
    );
    let right = Vector3::new(flight.yaw.cos(), 0.0, flight.yaw.sin());
    let along = key(Key::KeyW) - key(Key::KeyS);
    let across = key(Key::KeyD) - key(Key::KeyA);
    let rise = key(Key::KeyE) - key(Key::KeyQ);
    flight.position =
        flight.position + (look * along + right * across + Vector3::UP * rise) * speed;
    let camera = &mut w.resource_mut::<MainCamera>().0;
    camera.position = flight.position;
    camera.look_at(flight.position + look);
    w.resource_mut::<DevState>().flying = Some(flight);
}

/// The entity under the pointer: the nearest mesh hit, or the nearest of its
/// ancestors with components, which is what there is to inspect.
fn pick(w: &World) -> Option<Entity> {
    use crate::app::MainCamera;
    use threers::core::Raycaster;
    use threers::math::Vector2;
    let camera = &w.get_resource::<MainCamera>()?.0;
    let (width, height) = w.get_resource::<crate::hud::Hud>()?.size();
    let at = w.resource::<Input>().mouse_position;
    let ndc = Vector2::new(
        at.x / width as f32 * 2.0 - 1.0,
        1.0 - at.y / height as f32 * 2.0,
    );
    let mut ray = Raycaster::default();
    ray.set_from_camera(ndc, camera);
    let hits = ray.intersect_object(&w.scene.arena, w.scene.root, true);
    let inspectable = w.inspectable();
    for hit in hits {
        if w.node(hit.object).is_some_and(|n| n.name == "gizmos") {
            continue;
        }
        let mut id = Some(hit.object);
        while let Some(at) = id {
            if inspectable.contains(&at) {
                return Some(at);
            }
            id = w.node(at).and_then(|n| n.parent);
        }
        return Some(hit.object);
    }
    None
}

fn toggle_recording(w: &mut World) {
    if !w.is_recording() {
        w.start_recording();
        return;
    }
    let Some(take) = w.stop_recording() else {
        return;
    };
    #[cfg(not(target_arch = "wasm32"))]
    let message = {
        let dir = std::path::Path::new("recordings");
        let count = std::fs::read_dir(dir).map_or(0, |d| d.count());
        let path = dir.join(format!("take-{:03}.txt", count + 1));
        match std::fs::create_dir_all(dir).and_then(|_| std::fs::write(&path, take.to_text())) {
            Ok(()) => format!("saved {} ({:.1} s)", path.display(), take.seconds()),
            Err(error) => format!("could not save the recording: {error}"),
        }
    };
    #[cfg(target_arch = "wasm32")]
    let message = format!("recorded {:.1} s (no files in a browser)", take.seconds());
    w.resource_mut::<DevState>().saved = Some(message);
}

/// Each car's tyres where they touch, green gripping to red sliding, the
/// suspension from wheel to road, and its velocity as an arrow.
#[cfg(feature = "physics")]
fn draw_vehicles(w: &mut World) {
    use crate::car::Car;
    use crate::gizmos::Gizmos;
    use crate::physics::{Body, Physics};
    use threers::math::{Color, Vector3};
    let mut lines: Vec<(Vector3, Vector3, Color, bool)> = Vec::new();
    {
        let physics = w.resource::<Physics>();
        for (car, body) in w.iter::<(&Car, &Body)>() {
            let Some(chassis) = physics.body(body.0) else {
                continue;
            };
            let centre = chassis.position.translation;
            lines.push((
                centre,
                centre + chassis.linear_velocity * 0.25,
                Color::from_hex(0x3a8bff),
                true,
            ));
            for (index, wheel) in car.vehicle.wheels.iter().enumerate() {
                let Some(contact) = &wheel.contact else {
                    continue;
                };
                let grip = Color::new(
                    wheel.slip.clamp(0.0, 1.0),
                    1.0 - wheel.slip.clamp(0.0, 1.0),
                    0.1,
                );
                lines.push((
                    contact.point,
                    contact.point + Vector3::UP * 0.4,
                    grip,
                    false,
                ));
                if let Some(pose) = car.vehicle.wheel_transform(physics, index) {
                    lines.push((
                        pose.translation,
                        contact.point,
                        Color::from_hex(0xffffff),
                        false,
                    ));
                }
            }
        }
    }
    let gizmos = w.resource_mut::<Gizmos>();
    for (a, b, colour, arrow) in lines {
        if arrow {
            gizmos.arrow(a, b, colour);
        } else {
            gizmos.line(a, b, colour);
            if colour != Color::from_hex(0xffffff) {
                gizmos.point(a, 0.3, colour);
            }
        }
    }
}

#[cfg(feature = "physics")]
fn draw_colliders(w: &mut World) {
    use crate::physics::Physics;
    use std::sync::Arc;
    use threers::core::{LineSegments, Object3D, ObjectKind};
    use threers::materials::{LineBasicMaterial, Material};
    use threers_physics::debug::DebugRenderer;

    let (on, lines) = {
        let state = w.resource::<DevState>();
        (state.colliders, state.lines)
    };
    if !on {
        if let Some(id) = lines {
            w.despawn(id);
            w.resource_mut::<DevState>().lines = None;
        }
        return;
    }
    let Some(physics) = w.get_resource::<Physics>() else {
        return;
    };
    let mut debug = DebugRenderer::new();
    debug.render(physics);
    let geometry = debug.to_geometry();
    match lines.and_then(|id| w.scene.get_mut(id)) {
        Some(node) => {
            if let ObjectKind::LineSegments(segments) = &mut node.kind {
                segments.geometry = Arc::new(geometry);
            }
        }
        None => {
            let material = Material::Line(LineBasicMaterial::default());
            let segments = LineSegments::new(geometry, material);
            let id = w.scene.arena.insert(Object3D::line_segments(segments));
            let root = w.scene.root;
            w.scene.arena.add_child(root, id);
            w.resource_mut::<DevState>().lines = Some(id);
        }
    }
}
