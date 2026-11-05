//! Playing with thumbs.
//!
//! A stick on the left of the screen and buttons on the right, feeding the same
//! actions as the keyboard, so game code needs no change. They stay hidden until
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
//! ```

use crate::app::{App, Plugin};
use crate::hud::with_hud;
use crate::input::{Button, Input};
use crate::world::World;
use threers::math::{Color, Vector2};

/// One on-screen button.
#[derive(Debug, Clone)]
pub struct TouchButton {
    action: String,
    pub label: String,
    /// Centre, as fractions of the screen.
    pub x: f32,
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

/// The on-screen controls, as a plugin and then a resource.
#[derive(Debug, Clone)]
pub struct TouchControls {
    axes: Option<(String, String)>,
    pub buttons: Vec<TouchButton>,
    /// How far a thumb travels for full deflection, in pixels.
    pub stick_radius: f32,
    /// The stick takes touches left of this fraction of the screen.
    pub stick_area: f32,
    /// Screen size in pixels. A host with a HUD keeps this current.
    pub screen: (f32, f32),
    stick: Option<StickState>,
    used: bool,
}

#[derive(Debug, Clone, Copy)]
struct StickState {
    touch: u64,
    origin: Vector2,
}

impl TouchControls {
    pub fn new() -> Self {
        Self {
            axes: None,
            buttons: Vec::new(),
            stick_radius: 70.0,
            stick_area: 0.5,
            screen: (640.0, 360.0),
            stick: None,
            used: false,
        }
    }

    /// A floating stick for these two axes: it centres wherever the thumb lands.
    pub fn stick(mut self, x_axis: &str, y_axis: &str) -> Self {
        self.axes = Some((x_axis.to_owned(), y_axis.to_owned()));
        self
    }

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
    fn build(self, app: &mut App) {
        let input = app.world.resource_mut::<Input>();
        if let Some((x, y)) = &self.axes {
            input.bind_touch_stick(x, 0).bind_touch_stick(y, 1);
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

        if let Some(stick) = controls.stick {
            if !touches.iter().any(|touch| touch.id == stick.touch) {
                controls.stick = None;
            }
        }
        if controls.stick.is_none() && controls.axes.is_some() {
            let area = controls.stick_area * controls.screen.0;
            let free = touches.iter().find(|touch| {
                touch.start.x < area
                    && !controls
                        .buttons
                        .iter()
                        .any(|b| controls.over(b, touch.start))
            });
            controls.stick = free.map(|touch| StickState {
                touch: touch.id,
                origin: touch.start,
            });
        }
        let (mut x, mut y) = (0.0, 0.0);
        if let Some(stick) = controls.stick {
            if let Some(touch) = touches.iter().find(|touch| touch.id == stick.touch) {
                x = (touch.position.x - stick.origin.x) / controls.stick_radius;
                // The screen's y runs down; a stick's up is positive.
                y = (stick.origin.y - touch.position.y) / controls.stick_radius;
            }
        }
        input.set_touch_stick(x, y);

        if controls.used {
            draw(w, controls, x, y);
        }
    });
}

fn draw(w: &mut World, controls: &TouchControls, x: f32, y: f32) {
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
        if let Some(stick) = controls.stick {
            hud.fill_alpha(Color::WHITE, 0.2);
            hud.circle(stick.origin.x, stick.origin.y, controls.stick_radius);
            hud.fill_alpha(Color::WHITE, 0.5);
            let (dx, dy) = (x.clamp(-1.0, 1.0), y.clamp(-1.0, 1.0));
            hud.circle(
                stick.origin.x + dx * controls.stick_radius,
                stick.origin.y - dy * controls.stick_radius,
                24.0,
            );
        }
    });
}
