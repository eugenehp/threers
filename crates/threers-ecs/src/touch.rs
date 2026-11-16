//! Playing with thumbs.
//!
//! Sticks and buttons on the screen, feeding the same actions as the keyboard,
//! so game code needs no change: a stick on the left and buttons on the right,
//! or a stick under each thumb for a twin-stick game. They stay hidden until
//! the first touch, so a desktop player never sees them.
//!
//! ```
//! use threers_ecs::prelude::*;
//! use threers_ecs::touch::{TouchButton, TouchControls};
//!
//! let mut app = App::new();
//! app.plugin(
//!     TouchControls::new()
//!         .stick("move_x", "move_y")
//!         .button(TouchButton::new("jump", "A", 0.88, 0.75)),
//! );
//!
//! // Twin sticks: one under each thumb.
//! let mut app = App::new();
//! app.plugin(
//!     TouchControls::new()
//!         .stick_in("move_x", "move_y", 0.0, 0.5)
//!         .stick_in("look_x", "look_y", 0.5, 1.0),
//! );
//! ```

use crate::app::{App, Plugin};
use crate::hud::with_hud;
use crate::input::{Button, Input, TOUCH_STICKS};
use crate::world::World;
use threers::math::{Color, Vector2};

/// One on-screen button.
#[derive(Debug, Clone)]
pub struct TouchButton {
    action: String,
    /// What is written on it.
    pub label: String,
    /// Centre, as fractions of the screen.
    pub x: f32,
    /// Centre, as a fraction of the screen's height.
    pub y: f32,
    /// In pixels.
    pub radius: f32,
}

impl TouchButton {
    /// A button for `action` with its centre at a fraction of the screen.
    pub fn new(action: &str, label: &str, x: f32, y: f32) -> Self {
        Self {
            action: action.to_owned(),
            label: label.to_owned(),
            x,
            y,
            radius: 48.0,
        }
    }
}

/// One floating stick: the axes it drives and the strip of the screen it
/// takes thumbs in.
#[derive(Debug, Clone)]
struct Stick {
    axes: (String, String),
    /// From and to, as fractions of the screen's width.
    area: (f32, f32),
    held: Option<StickState>,
}

/// The on-screen controls, as a plugin and then a resource.
#[derive(Debug, Clone)]
pub struct TouchControls {
    sticks: Vec<Stick>,
    /// The buttons, in the order [`Button::Touch`] numbers them.
    pub buttons: Vec<TouchButton>,
    /// How far a thumb travels for full deflection, in pixels.
    pub stick_radius: f32,
    /// A [`stick`](Self::stick) takes touches left of this fraction of the
    /// screen.
    pub stick_area: f32,
    /// Screen size in pixels. A host with a HUD keeps this current.
    pub screen: (f32, f32),
    used: bool,
}

#[derive(Debug, Clone, Copy)]
struct StickState {
    touch: u64,
    origin: Vector2,
}

impl TouchControls {
    /// No stick and no buttons; add them with [`stick`](Self::stick) and [`button`](Self::button).
    pub fn new() -> Self {
        Self {
            sticks: Vec::new(),
            buttons: Vec::new(),
            stick_radius: 70.0,
            stick_area: 0.5,
            screen: (640.0, 360.0),
            used: false,
        }
    }

    /// A floating stick for these two axes, on the left of the screen (up to
    /// [`stick_area`](Self::stick_area)): it centres wherever the thumb lands.
    pub fn stick(self, x_axis: &str, y_axis: &str) -> Self {
        // Read when the plugin is built, so a stick_area set after counts.
        self.stick_in(x_axis, y_axis, 0.0, f32::NAN)
    }

    /// A floating stick for these two axes, taking thumbs that land between
    /// `from` and `to`, fractions of the screen's width. Up to
    /// [`TOUCH_STICKS`] of them; the first stick added is stick 0.
    pub fn stick_in(mut self, x_axis: &str, y_axis: &str, from: f32, to: f32) -> Self {
        if self.sticks.len() < TOUCH_STICKS {
            self.sticks.push(Stick {
                axes: (x_axis.to_owned(), y_axis.to_owned()),
                area: (from, to),
                held: None,
            });
        }
        self
    }

    /// Add a button.
    pub fn button(mut self, button: TouchButton) -> Self {
        self.buttons.push(button);
        self
    }

    fn centre(&self, button: &TouchButton) -> Vector2 {
        Vector2::new(button.x * self.screen.0, button.y * self.screen.1)
    }

    fn over(&self, button: &TouchButton, at: Vector2) -> bool {
        let (dx, dy) = (at.x - self.centre(button).x, at.y - self.centre(button).y);
        dx * dx + dy * dy <= button.radius * button.radius
    }
}

impl Default for TouchControls {
    fn default() -> Self {
        Self::new()
    }
}

impl Plugin for TouchControls {
    fn build(mut self, app: &mut App) {
        // A plain `stick` reaches across to `stick_area`, as it is now.
        for stick in &mut self.sticks {
            if stick.area.1.is_nan() {
                stick.area.1 = self.stick_area;
            }
        }
        let input = app.world.resource_mut::<Input>();
        for (n, stick) in self.sticks.iter().enumerate() {
            let (x, y) = &stick.axes;
            input
                .bind_touch_stick(x, 2 * n)
                .bind_touch_stick(y, 2 * n + 1);
        }
        for (index, button) in self.buttons.iter().enumerate() {
            input.bind(&button.action, Button::Touch(index as u8));
        }
        app.resource(self).system(touch_step);
    }
}

/// Turn fingers into stick and button state, and draw the controls.
pub fn touch_step(w: &mut World) {
    w.with_resource::<TouchControls, _>(|w, controls| {
        if let Some(hud) = w.get_resource::<crate::hud::Hud>() {
            let (width, height) = hud.0.size();
            controls.screen = (width as f32, height as f32);
        }
        let input = w.resource_mut::<Input>();
        let touches = input.touches.clone();
        controls.used |= !touches.is_empty();

        // A button keeps the touch that lands on it, so the stick must not claim it.
        for index in 0..controls.buttons.len() {
            let held = touches
                .iter()
                .any(|touch| controls.over(&controls.buttons[index], touch.position));
            let button = Button::Touch(index as u8);
            if held {
                input.press(button);
            } else {
                input.release(button);
            }
        }

        let mut moved = Vec::new();
        for n in 0..controls.sticks.len() {
            // A stick lets go when its thumb lifts.
            if let Some(held) = controls.sticks[n].held
                && !touches.iter().any(|touch| touch.id == held.touch)
            {
                controls.sticks[n].held = None;
            }
            if controls.sticks[n].held.is_none() {
                let (from, to) = controls.sticks[n].area;
                let (from, to) = (from * controls.screen.0, to * controls.screen.0);
                let taken: Vec<u64> = controls
                    .sticks
                    .iter()
                    .filter_map(|s| s.held.map(|h| h.touch))
                    .collect();
                let free = touches.iter().find(|touch| {
                    touch.start.x >= from
                        && touch.start.x < to
                        && !taken.contains(&touch.id)
                        && !controls
                            .buttons
                            .iter()
                            .any(|b| controls.over(b, touch.start))
                });
                controls.sticks[n].held = free.map(|touch| StickState {
                    touch: touch.id,
                    origin: touch.start,
                });
            }
            let (mut x, mut y) = (0.0, 0.0);
            if let Some(held) = controls.sticks[n].held
                && let Some(touch) = touches.iter().find(|touch| touch.id == held.touch)
            {
                x = (touch.position.x - held.origin.x) / controls.stick_radius;
                // The screen's y runs down; a stick's up is positive.
                y = (held.origin.y - touch.position.y) / controls.stick_radius;
            }
            input.set_touch_stick_at(n, x, y);
            moved.push((x, y));
        }

        if controls.used {
            draw(w, controls, &moved);
        }
    });
}

fn draw(w: &mut World, controls: &TouchControls, moved: &[(f32, f32)]) {
    with_hud(w, |hud| {
        hud.no_stroke();
        for button in &controls.buttons {
            let at = controls.centre(button);
            hud.fill_alpha(Color::WHITE, 0.25);
            hud.circle(at.x, at.y, button.radius);
            hud.fill_alpha(Color::WHITE, 0.8);
            hud.text_size(24.0);
            hud.text(at.x - 8.0, at.y + 8.0, &button.label);
        }
        for (stick, (x, y)) in controls.sticks.iter().zip(moved) {
            let Some(held) = stick.held else { continue };
            hud.fill_alpha(Color::WHITE, 0.2);
            hud.circle(held.origin.x, held.origin.y, controls.stick_radius);
            hud.fill_alpha(Color::WHITE, 0.5);
            let (dx, dy) = (x.clamp(-1.0, 1.0), y.clamp(-1.0, 1.0));
            hud.circle(
                held.origin.x + dx * controls.stick_radius,
                held.origin.y - dy * controls.stick_radius,
                24.0,
            );
        }
    });
}
