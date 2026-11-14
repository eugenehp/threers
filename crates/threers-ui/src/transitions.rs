//! Nodes that come and go with an animation:
//! [`Ui::appear`] animates a node in where it was added, and
//! [`Ui::remove_with`] animates it out and then removes it. While it
//! leaves it takes no pointer and is hidden from assistive technology.
//! With [reduced motion](Ui::set_reduce_motion) a move is skipped and only
//! the fade runs.
//!
//! ```
//! use threers_ui::transitions::Transition;
//! use threers_ui::{style, Ui};
//!
//! let mut ui = Ui::new();
//! let root = ui.root();
//! let notice = ui.add_box(root, style! { padding: 8px; });
//! ui.appear(notice, Transition::Fade, 0.2);
//! ui.step(0.1);
//! assert!(ui.style(notice).opacity > 0.0 && ui.style(notice).opacity < 1.0);
//! ui.remove_with(notice, Transition::Fade, 0.2);
//! ui.step(0.25);
//! assert!(ui.contains(notice) && ui.is_leaving(notice), "removed on the next step");
//! ui.step(0.0);
//! assert!(!ui.contains(notice));
//! ```

use crate::animation::{Animation, Property, Value};
use crate::tree::{NodeId, Ui};
use crate::{Length, Position};

/// How a node comes and goes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Transition {
    /// Its opacity from or to 0.
    Fade,
    /// From or to `(dx, dy)` pixels off its place (it is positioned
    /// relative to it for that), fading too.
    Move { dx: f32, dy: f32 },
}

impl Transition {
    /// From the side: `Move` by `distance` pixels to the right.
    pub fn from_right(distance: f32) -> Self {
        Transition::Move {
            dx: distance,
            dy: 0.0,
        }
    }

    /// From below: `Move` by `distance` pixels down.
    pub fn from_below(distance: f32) -> Self {
        Transition::Move {
            dx: 0.0,
            dy: distance,
        }
    }
}

impl Ui {
    /// Animates `node` in over `seconds`, from `transition`'s start.
    pub fn appear(&mut self, node: NodeId, transition: Transition, seconds: f32) {
        self.run_transition(node, transition, seconds, true);
    }

    /// Animates `node` out over `seconds`, then removes it (on the step
    /// after the animation ends). Ignored for a node already leaving.
    pub fn remove_with(&mut self, node: NodeId, transition: Transition, seconds: f32) {
        if !self.contains(node) || self.is_leaving(node) {
            return;
        }
        self.update_style(node, |s| s.pointer_events = false);
        self.semantics_mut(node).hidden = true;
        self.run_transition(node, transition, seconds, false);
        self.leaving.push((node, seconds.max(0.0)));
    }

    /// Whether `node` is on its way out.
    pub fn is_leaving(&self, node: NodeId) -> bool {
        self.leaving.iter().any(|&(n, _)| n == node)
    }

    fn run_transition(&mut self, node: NodeId, transition: Transition, seconds: f32, coming: bool) {
        // Coming, from just above transparent: a fully transparent node is
        // hidden from hit tests and assistive technology, and one appearing
        // is there to use from the start.
        let (a, b) = if coming { (0.02, 1.0) } else { (1.0, 0.0) };
        let opacity = if coming {
            self.style(node).opacity.max(0.0)
        } else {
            self.style(node).opacity
        };
        let full = if coming && opacity == 0.0 {
            1.0
        } else {
            opacity
        };
        self.animate(
            node,
            Animation::new(
                Property::Opacity,
                Value::Number(a * full),
                Value::Number(b * full),
                seconds,
            ),
        );
        if let Transition::Move { dx, dy } = transition {
            if self.style(node).position == Position::Static {
                self.update_style(node, |s| s.position = Position::Relative);
            }
            let px = |l: Length| if let Length::Px(v) = l { v } else { 0.0 };
            let s = self.style(node);
            // A box placed by its right or bottom edge (and not its left or
            // top) moves by that edge: giving it a left or a top would
            // stretch it between the two.
            let by_right = s.left == Length::Auto && s.right != Length::Auto;
            let by_bottom = s.top == Length::Auto && s.bottom != Length::Auto;
            let across = if by_right {
                (Property::Right, px(s.right), -dx)
            } else {
                (Property::Left, px(s.left), dx)
            };
            let down = if by_bottom {
                (Property::Bottom, px(s.bottom), -dy)
            } else {
                (Property::Top, px(s.top), dy)
            };
            for (p, home, off) in [across, down] {
                if off != 0.0 {
                    let (from, to) = if coming {
                        (home + off, home)
                    } else {
                        (home, home + off)
                    };
                    self.animate(
                        node,
                        Animation::new(p, Value::Number(from), Value::Number(to), seconds),
                    );
                }
            }
        }
    }

    /// Removes the nodes whose leaving is over (from [`step`](Self::step)).
    pub(crate) fn step_leaving(&mut self, dt: f32) {
        if self.leaving.is_empty() {
            return;
        }
        let mut gone = vec![];
        self.leaving.retain_mut(|(n, left)| {
            if *left <= 0.0 {
                gone.push(*n);
                return false;
            }
            *left -= dt;
            true
        });
        for n in gone {
            self.remove(n);
        }
        let leaving = std::mem::take(&mut self.leaving);
        self.leaving = leaving
            .into_iter()
            .filter(|&(n, _)| self.contains(n))
            .collect();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{style, FixedWidthText};

    /// A box anchored to the bottom slides up by its bottom edge, and keeps
    /// its own height.
    #[test]
    fn a_bottom_anchored_box_slides_by_its_bottom() {
        let mut ui = Ui::new();
        let root = ui.root();
        let panel = ui.add_box(
            root,
            style! { position: absolute; left: 0px; bottom: 0px; width: 100%; height: 100px; },
        );
        ui.appear(panel, Transition::from_below(80.0), 0.2);
        ui.layout(400.0, 600.0, &mut FixedWidthText(7.0));
        assert_eq!(ui.rect(panel).y, 580.0, "starting 80 px lower");
        ui.step(0.3);
        ui.layout(400.0, 600.0, &mut FixedWidthText(7.0));
        assert_eq!((ui.rect(panel).y, ui.rect(panel).height), (500.0, 100.0));
        assert_eq!(ui.style(panel).top, Length::Auto);
    }
}
