//! Properties moved by springs: [`Ui::spring_to`] pulls a node's property
//! toward a target, and a new target (the pointer moved on, the card is
//! sent back mid-turn) bends its path from where it is going and how fast,
//! instead of starting again from rest — the way physics-based interfaces
//! follow a finger and can be reversed half way.
//!
//! ```
//! use threers_ui::anim::Spring;
//! use threers_ui::animation::Property;
//! use threers_ui::{style, Length, Ui};
//!
//! let mut ui = Ui::new();
//! let root = ui.root();
//! let b = ui.add_box(root, style! { position: relative; left: 0px; });
//! ui.spring_to(b, Property::Left, 100.0, Spring::settling_in(0.3));
//! ui.step(0.05);
//! let moving = ui.spring_velocity(b, Property::Left);
//! assert!(moving > 0.0);
//! // Sent back half way: it carries on a little before it turns.
//! ui.spring_to(b, Property::Left, 0.0, Spring::settling_in(0.3));
//! assert_eq!(ui.spring_velocity(b, Property::Left), moving, "the speed it had");
//! for _ in 0..40 {
//!     ui.step(1.0 / 60.0);
//! }
//! assert_eq!(ui.style(b).left, Length::Px(0.0));
//! ```

use crate::anim::Spring;
use crate::animation::{Property, Value};
use crate::tree::{NodeId, Ui};

/// A property being pulled to its target.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Sprung {
    pub(crate) node: NodeId,
    pub(crate) property: Property,
    target: f32,
    value: f32,
    velocity: f32,
    spring: Spring,
}

impl Ui {
    /// Pulls `property` of `node` (a number: a place, a size, a turn, an
    /// opacity) toward `target` by `spring`, keeping the speed it has. With
    /// reduced motion it moves there at once. A tween on the property gives
    /// way.
    pub fn spring_to(&mut self, node: NodeId, property: Property, target: f32, spring: Spring) {
        self.stop_animations(node, Some(property));
        if self.motion_skipped(property) {
            self.springs
                .retain(|s| !(s.node == node && s.property == property));
            self.write(node, property, Value::Number(target));
            return;
        }
        if let Some(s) = self
            .springs
            .iter_mut()
            .find(|s| s.node == node && s.property == property)
        {
            s.target = target;
            s.spring = spring;
            return;
        }
        let value = match self.read(node, property) {
            Value::Number(v) => v,
            Value::Color(_) => return,
        };
        self.springs.push(Sprung {
            node,
            property,
            target,
            value,
            velocity: 0.0,
            spring,
        });
    }

    /// How fast `property` of `node` is moving under a spring (0 at rest).
    pub fn spring_velocity(&self, node: NodeId, property: Property) -> f32 {
        self.springs
            .iter()
            .find(|s| s.node == node && s.property == property)
            .map_or(0.0, |s| s.velocity)
    }

    /// The springs moved on by `dt` (from [`step`](Self::step)); whether
    /// any still moves.
    pub(crate) fn step_springs(&mut self, dt: f32) -> bool {
        if self.springs.is_empty() {
            return false;
        }
        let mut springs = std::mem::take(&mut self.springs);
        springs.retain(|s| self.contains(s.node));
        for s in &mut springs {
            let (x, v) = s.spring.step(s.value - s.target, s.velocity, dt);
            s.value = s.target + x;
            s.velocity = v;
            // At rest within a hundredth (of a pixel, a radian): there.
            if x.abs() < 0.01 && v.abs() < 0.01 {
                s.value = s.target;
                s.velocity = 0.0;
            }
            self.write(s.node, s.property, Value::Number(s.value));
        }
        springs.retain(|s| s.velocity != 0.0 || s.value != s.target);
        let any = !springs.is_empty();
        // Springs started while these were written run from the next step.
        springs.append(&mut self.springs);
        self.springs = springs;
        any
    }
}
