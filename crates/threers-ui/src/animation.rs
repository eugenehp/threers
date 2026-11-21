//! Animations attached to nodes:
//! a property moving between two values over a duration, along an easing,
//! once, a number of times or for ever, and back again if it alternates.
//! The tree runs them; the app only says how much time has passed.
//!
//! ```
//! use threers_ui::animation::{Animation, Property, Value};
//! use threers_ui::{style, FixedWidthText, Ui};
//!
//! let mut ui = Ui::new();
//! let root = ui.root();
//! let toast = ui.add_box(root, style! { width: 100px; height: 20px; opacity: 0; });
//! ui.animate(toast, Animation::new(Property::Opacity, Value::Number(0.0), Value::Number(1.0), 0.2));
//! assert!(ui.step(0.1));
//! assert_eq!(ui.style(toast).opacity, 0.5);
//! assert!(!ui.step(0.1)); // arrived: no more frames needed
//! assert_eq!(ui.style(toast).opacity, 1.0);
//! ```
//!
//! [`Ui::transition`] makes a node's state styles (hover, focus, …) move
//! there rather than jump, as CSS `transition` does.

use threers_math::easing::Easing;

use super::style::{Length, Rgba};
use super::tree::{NodeId, Ui};

/// What an animation moves.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Property {
    Opacity,
    Background,
    Color,
    BorderColor,
    /// `width`, in pixels.
    Width,
    /// `height`, in pixels.
    Height,
    /// `left`, in pixels.
    Left,
    /// `top`, in pixels.
    Top,
    /// `right`, in pixels.
    Right,
    /// `bottom`, in pixels.
    Bottom,
    /// A box's turn about its horizontal axis, in radians (see
    /// [`depth`](crate::depth)).
    RotateX,
    /// About its vertical axis.
    RotateY,
    /// How far it is lifted toward the viewer, in pixels.
    Depth,
    /// A scrolling box's `scroll_top`: smooth scrolling.
    ScrollTop,
    /// Its `scroll_left`: settling on a snap point.
    ScrollLeft,
    /// A path node's turn, in radians (see
    /// [`Ui::set_path_transform`](crate::Ui::set_path_transform)): a spinner.
    Rotate,
    /// A box's size as drawn, 1 as laid out (see [`depth`](crate::depth)).
    Scale,
    /// A box's turn in its own plane, in radians.
    RotateZ,
    /// How far a box is drawn to the right of where it is laid out, in
    /// pixels.
    TranslateX,
    /// How far below where it is laid out, in pixels.
    TranslateY,
}

impl Property {
    /// Whether the property moves boxes (and so needs layout), rather than
    /// only how they are painted.
    pub fn affects_layout(self) -> bool {
        !matches!(
            self,
            Property::Opacity
                | Property::Background
                | Property::Color
                | Property::BorderColor
                | Property::Rotate
                | Property::RotateX
                | Property::RotateY
                | Property::Depth
                | Property::Scale
                | Property::RotateZ
                | Property::TranslateX
                | Property::TranslateY
        )
    }
}

/// A value of a [`Property`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Value {
    Number(f32),
    Color(Rgba),
}

impl Value {
    fn lerp(self, to: Value, t: f32) -> Value {
        match (self, to) {
            (Value::Number(a), Value::Number(b)) => Value::Number(a + (b - a) * t),
            (Value::Color(a), Value::Color(b)) => {
                Value::Color(std::array::from_fn(|i| a[i] + (b[i] - a[i]) * t))
            }
            (_, b) => b,
        }
    }
}

/// How many times an animation runs.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum Repeat {
    #[default]
    Once,
    Times(u32),
    Forever,
}

/// An animation of one property of one node. See the [module docs](self).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Animation {
    pub property: Property,
    pub from: Value,
    pub to: Value,
    /// Seconds one run takes.
    pub duration: f32,
    pub easing: Easing,
    pub repeat: Repeat,
    /// Every other run goes back from `to` to `from`.
    pub alternate: bool,
    /// Seconds before it starts (the property holds `from` meanwhile).
    pub delay: f32,
}

impl Animation {
    pub fn new(property: Property, from: Value, to: Value, duration: f32) -> Self {
        Animation {
            property,
            from,
            to,
            duration: duration.max(1e-4),
            easing: Easing::Linear,
            repeat: Repeat::Once,
            alternate: false,
            delay: 0.0,
        }
    }
    pub fn easing(mut self, easing: Easing) -> Self {
        self.easing = easing;
        self
    }
    pub fn repeat(mut self, repeat: Repeat) -> Self {
        self.repeat = repeat;
        self
    }
    pub fn alternate(mut self) -> Self {
        self.alternate = true;
        self
    }
    pub fn delay(mut self, seconds: f32) -> Self {
        self.delay = seconds.max(0.0);
        self
    }

    /// The value `elapsed` seconds in, and whether it has finished.
    fn at(&self, elapsed: f32) -> (Value, bool) {
        let t = (elapsed - self.delay).max(0.0) / self.duration;
        let runs = match self.repeat {
            Repeat::Once => Some(1.0),
            Repeat::Times(n) => Some(n.max(1) as f32),
            Repeat::Forever => None,
        };
        let done = runs.is_some_and(|r| t >= r);
        let t = runs.map_or(t, |r| t.min(r));
        let run = (t.floor() as u32).min(runs.map_or(u32::MAX, |r| r as u32 - 1));
        let mut f = if done { 1.0 } else { t - t.floor() };
        if self.alternate && run % 2 == 1 {
            f = 1.0 - f;
        }
        (
            self.from
                .lerp(self.to, self.easing.apply(f.clamp(0.0, 1.0))),
            done,
        )
    }
}

/// An animation running on a node.
#[derive(Debug, Clone)]
pub(crate) struct Running {
    pub node: NodeId,
    pub animation: Animation,
    pub elapsed: f32,
}

impl Ui {
    /// Starts `animation` on `id`, replacing one already moving the same
    /// property there. The property takes its `from` value now.
    pub fn animate(&mut self, id: NodeId, animation: Animation) {
        self.node(id);
        // A spring on the property gives way.
        self.springs
            .retain(|s| !(s.node == id && s.property == animation.property));
        if self.motion_skipped(animation.property) {
            self.animations
                .retain(|r| !(r.node == id && r.animation.property == animation.property));
            self.write(id, animation.property, animation.to);
            return;
        }
        self.animations
            .retain(|r| !(r.node == id && r.animation.property == animation.property));
        self.write(id, animation.property, animation.from);
        self.animations.push(Running {
            node: id,
            animation,
            elapsed: 0.0,
        });
    }

    /// Stops `id`'s animations of `property` (all of them with `None`),
    /// leaving the values where they are.
    pub fn stop_animations(&mut self, id: NodeId, property: Option<Property>) {
        self.animations
            .retain(|r| !(r.node == id && property.is_none_or(|p| p == r.animation.property)));
    }

    /// Whether `id` has an animation running.
    pub fn is_animating(&self, id: NodeId) -> bool {
        self.animations.iter().any(|r| r.node == id)
    }

    /// Moves every animation on by `dt` seconds and writes their values.
    /// Returns whether any is still running (draw another frame).
    pub fn step(&mut self, dt: f32) -> bool {
        self.clock += dt.max(0.0) as f64;
        crate::charts::step_glides(self, dt);
        let mut running = std::mem::take(&mut self.animations);
        running.retain(|r| self.contains(r.node));
        let mut keep = Vec::with_capacity(running.len());
        for mut r in running {
            // One that runs for ever (a spinner) waits, unmoved, while it
            // is out of sight: nothing to draw, and nothing to lay out.
            if self.idles(&r) {
                keep.push(r);
                continue;
            }
            r.elapsed += dt.max(0.0);
            let (value, done) = r.animation.at(r.elapsed);
            self.write(r.node, r.animation.property, value);
            if !done {
                keep.push(r);
            }
        }
        // Animations started by what was written run from the next step.
        keep.append(&mut self.animations);
        self.animations = keep;
        // `auto` scrollbars show while their boxes scroll, and then fade.
        self.step_keyframes(dt.max(0.0));
        let bars = self.watch_scrollbars();
        self.step_leaving(dt.max(0.0));
        let sprung = self.step_springs(dt.max(0.0));
        self.step_motions();
        self.step_solids();
        self.step_beams();
        self.plan_effects();
        let flickers = self.step_flickers();
        self.animations.iter().any(|r| !self.idles(r))
            || flickers
            || bars
            || !self.leaving.is_empty()
            || sprung
            || !self.env.keyframes.is_empty()
    }

    /// Whether `r` runs for ever on a node that does not show, as last
    /// laid out: it waits for it to.
    fn idles(&self, r: &Running) -> bool {
        r.animation.repeat == Repeat::Forever && !self.is_on_screen(r.node)
    }

    /// The current value of `property` on `id`.
    pub fn read(&self, id: NodeId, property: Property) -> Value {
        let s = self.style(id);
        let px = |l: Length| match l {
            Length::Px(v) => v,
            _ => 0.0,
        };
        match property {
            Property::Opacity => Value::Number(s.opacity),
            Property::Background => Value::Color(s.background.unwrap_or([0.0; 4])),
            Property::Color => Value::Color(s.color),
            Property::BorderColor => Value::Color(s.border_color),
            Property::Width => Value::Number(px(s.width)),
            Property::Height => Value::Number(px(s.height)),
            Property::Left => Value::Number(px(s.left)),
            Property::Top => Value::Number(px(s.top)),
            Property::Right => Value::Number(px(s.right)),
            Property::Bottom => Value::Number(px(s.bottom)),
            Property::ScrollTop => Value::Number(s.scroll_top),
            Property::ScrollLeft => Value::Number(s.scroll_left),
            Property::Rotate => Value::Number(self.path_transform(id).rotate),
            Property::RotateX => Value::Number(self.transform3d(id).map_or(0.0, |t| t.rotate_x)),
            Property::RotateY => Value::Number(self.transform3d(id).map_or(0.0, |t| t.rotate_y)),
            Property::Depth => Value::Number(self.transform3d(id).map_or(0.0, |t| t.depth)),
            Property::Scale => Value::Number(self.transform3d(id).map_or(1.0, |t| t.scale)),
            Property::RotateZ => Value::Number(self.transform3d(id).map_or(0.0, |t| t.rotate_z)),
            Property::TranslateX => {
                Value::Number(self.transform3d(id).map_or(0.0, |t| t.translate_x))
            }
            Property::TranslateY => {
                Value::Number(self.transform3d(id).map_or(0.0, |t| t.translate_y))
            }
        }
    }

    /// Sets `property` on `id` to `value`, marking for layout only what
    /// changes boxes.
    pub(crate) fn write(&mut self, id: NodeId, property: Property, value: Value) {
        if self.read(id, property) == value {
            return;
        }
        if property.affects_layout() {
            self.touch(id);
        }
        // Paint-only properties change the look, not the layout: no touch,
        // but the hit index and the scene see the new style anyway.
        // A node with state styles is restyled from its base whenever the
        // pointer or focus moves: an animation of where it is, how big or
        // how opaque goes into the base too, or the next restyle would put
        // it back. Colours are what state transitions animate between
        // their own styles: those stay out of it.
        if !matches!(
            property,
            Property::Background
                | Property::Color
                | Property::BorderColor
                | Property::Rotate
                | Property::RotateX
                | Property::RotateY
                | Property::Depth
                | Property::Scale
                | Property::RotateZ
                | Property::TranslateX
                | Property::TranslateY
        ) && let Some(base) = self.nodes[id.0]
            .states
            .as_mut()
            .and_then(|s| s.base.as_mut())
        {
            Self::write_style(base, property, value);
        }
        let s = &mut self.nodes[id.0].style;
        match (property, value) {
            (Property::Opacity, Value::Number(v)) => s.opacity = v.clamp(0.0, 1.0),
            (Property::Background, Value::Color(c)) => s.background = Some(c),
            (Property::Color, Value::Color(c)) => s.color = c,
            (Property::BorderColor, Value::Color(c)) => s.border_color = c,
            (Property::Width, Value::Number(v)) => s.width = Length::Px(v),
            (Property::Height, Value::Number(v)) => s.height = Length::Px(v),
            (Property::Left, Value::Number(v)) => s.left = Length::Px(v),
            (Property::Top, Value::Number(v)) => s.top = Length::Px(v),
            (Property::Right, Value::Number(v)) => s.right = Length::Px(v),
            (Property::Bottom, Value::Number(v)) => s.bottom = Length::Px(v),
            (Property::ScrollTop, Value::Number(v)) => s.scroll_top = v,
            (Property::ScrollLeft, Value::Number(v)) => s.scroll_left = v,
            (Property::Rotate, Value::Number(v)) => {
                let t = crate::transform::Transformation {
                    rotate: v,
                    ..self.path_transform(id)
                };
                self.set_path_transform(id, t);
            }
            (
                Property::RotateX
                | Property::RotateY
                | Property::Depth
                | Property::Scale
                | Property::RotateZ
                | Property::TranslateX
                | Property::TranslateY,
                Value::Number(v),
            ) => {
                let t = self.env.transforms3d.entry(id).or_default();
                match property {
                    Property::RotateX => t.rotate_x = v,
                    Property::RotateY => t.rotate_y = v,
                    Property::Scale => t.scale = v,
                    Property::RotateZ => t.rotate_z = v,
                    Property::TranslateX => t.translate_x = v,
                    Property::TranslateY => t.translate_y = v,
                    _ => t.depth = v,
                }
            }
            _ => {}
        }
    }

    /// `property` set to `value` in style `s` (all but a path's turn).
    fn write_style(s: &mut crate::Style, property: Property, value: Value) {
        match (property, value) {
            (Property::Opacity, Value::Number(v)) => s.opacity = v.clamp(0.0, 1.0),
            (Property::Background, Value::Color(c)) => s.background = Some(c),
            (Property::Color, Value::Color(c)) => s.color = c,
            (Property::BorderColor, Value::Color(c)) => s.border_color = c,
            (Property::Width, Value::Number(v)) => s.width = Length::Px(v),
            (Property::Height, Value::Number(v)) => s.height = Length::Px(v),
            (Property::Left, Value::Number(v)) => s.left = Length::Px(v),
            (Property::Top, Value::Number(v)) => s.top = Length::Px(v),
            (Property::Right, Value::Number(v)) => s.right = Length::Px(v),
            (Property::Bottom, Value::Number(v)) => s.bottom = Length::Px(v),
            (Property::ScrollTop, Value::Number(v)) => s.scroll_top = v,
            (Property::ScrollLeft, Value::Number(v)) => s.scroll_left = v,
            _ => {}
        }
    }

    /// Whether the app should draw another frame for the tree: an animation
    /// is running, or something changed since the last layout. An app that
    /// asks this (and
    /// [`Input::needs_frame`](super::input::Input::needs_frame)) and sleeps
    /// otherwise draws nothing while nothing happens.
    pub fn needs_frame(&self) -> bool {
        self.animations.iter().any(|r| !self.idles(r))
            || !self.springs.is_empty()
            || !self.leaving.is_empty()
            || self.nodes[0].dirty
            || self.effects_animate()
            || self.flickers_animate()
            || self.motions_animate()
            || self.env.repaint.get()
            || self.jobs_pending()
            || !self.env.keyframes.is_empty()
            || !self.env.glides.is_empty()
            || self.scrollbar_use.keys().any(|&n| {
                self.contains(n) && self.fades_scrollbar(n) && self.scrollbar_opacity(n) > 0.0
            })
    }

    /// Scrolls `container` to `to` over `duration` seconds, eased out.
    pub fn scroll_smoothly(&mut self, container: NodeId, to: f32, duration: f32) {
        let from = self.style(container).scroll_top;
        let (content, view, _) = self.scroll_extent(container);
        let to = to.clamp(0.0, (content - view).max(0.0));
        self.animate(
            container,
            Animation::new(
                Property::ScrollTop,
                Value::Number(from),
                Value::Number(to),
                duration,
            )
            .easing(Easing::CubicOut),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repeats_alternate_and_delay() {
        let a = Animation::new(
            Property::Opacity,
            Value::Number(0.0),
            Value::Number(1.0),
            1.0,
        )
        .repeat(Repeat::Times(2))
        .alternate()
        .delay(0.5);
        assert_eq!(a.at(0.25), (Value::Number(0.0), false));
        assert_eq!(a.at(1.0), (Value::Number(0.5), false));
        assert_eq!(a.at(1.75), (Value::Number(0.75), false), "back again");
        assert_eq!(
            a.at(9.0),
            (Value::Number(0.0), true),
            "two runs, ending where the second does"
        );
        let f = Animation::new(Property::Top, Value::Number(0.0), Value::Number(10.0), 2.0)
            .repeat(Repeat::Forever);
        assert_eq!(f.at(1001.0), (Value::Number(5.0), false));
    }
}
