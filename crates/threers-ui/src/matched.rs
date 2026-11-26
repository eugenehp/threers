//! A box that moves to match another's place and size, animating between
//! them: the selected tab's underline sliding to the tab picked, a
//! highlight following the row chosen. The indicator is an absolutely
//! placed box under any parent; [`Ui::match_geometry`] ties it to a target
//! node, and [`Ui::set_match_target`] moves it on to another.
//!
//! ```
//! use threers_ui::matched::MatchShape;
//! use threers_ui::{style, FixedWidthText, Ui};
//!
//! let mut ui = Ui::new();
//! let root = ui.root();
//! let row = ui.add_box(root, style! { display: flex; position: relative; });
//! let a = ui.add_box(row, style! { width: 50px; height: 20px; });
//! let b = ui.add_box(row, style! { width: 80px; height: 20px; });
//! let line = ui.add_box(row, style! { position: absolute; background: rgb(37, 99, 235); });
//! ui.match_geometry(line, a, MatchShape::Underline(2.0), 0.2);
//! ui.layout(400.0, 100.0, &mut FixedWidthText(7.0));
//! assert_eq!((ui.rect(line).x, ui.rect(line).width, ui.rect(line).y), (0.0, 50.0, 18.0));
//! ui.set_match_target(line, b);
//! ui.layout(400.0, 100.0, &mut FixedWidthText(7.0));
//! ui.step(0.1);
//! ui.layout(400.0, 100.0, &mut FixedWidthText(7.0));
//! assert!(ui.rect(line).x > 0.0 && ui.rect(line).x < 50.0, "on its way");
//! ui.step(0.2);
//! ui.layout(400.0, 100.0, &mut FixedWidthText(7.0));
//! assert_eq!((ui.rect(line).x, ui.rect(line).width), (50.0, 80.0));
//! ```

use crate::animation::{Animation, Property, Value};
use crate::tree::{NodeId, Ui};
use crate::{Length, Rect};

/// What of the target the indicator takes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MatchShape {
    /// Its whole box.
    Fill,
    /// A line this thick along its bottom.
    Underline(f32),
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct Matched {
    indicator: NodeId,
    target: NodeId,
    shape: MatchShape,
    seconds: f32,
    last: Option<Rect>,
}

impl Ui {
    /// Places `indicator` over `target` (as `shape` says) after each
    /// layout, animating over `seconds` when the place changes.
    pub fn match_geometry(
        &mut self,
        indicator: NodeId,
        target: NodeId,
        shape: MatchShape,
        seconds: f32,
    ) {
        self.matched.retain(|m| m.indicator != indicator);
        self.matched.push(Matched {
            indicator,
            target,
            shape,
            seconds,
            last: None,
        });
        self.invalidate_layout();
    }

    /// Moves `indicator` on to `target`.
    pub fn set_match_target(&mut self, indicator: NodeId, target: NodeId) {
        if let Some(m) = self.matched.iter_mut().find(|m| m.indicator == indicator)
            && m.target != target
        {
            m.target = target;
            self.invalidate_layout();
        }
    }

    /// Moves each indicator after layout; whether any moved.
    pub(crate) fn place_matched(&mut self) -> bool {
        if self.matched.is_empty() {
            return false;
        }
        let mut all = std::mem::take(&mut self.matched);
        all.retain(|m| self.contains(m.indicator) && self.contains(m.target));
        let mut moved = false;
        for m in &mut all {
            let Some(parent) = self.parent(m.indicator) else {
                continue;
            };
            let (p, mut t) = (self.rect(parent), self.rect(m.target));
            // Laid out left to right, then mirrored: placed in the unmirrored.
            if self.nodes[parent.0].rtl {
                t.x = p.x + p.width - (t.x - p.x) - t.width;
            }
            let ps = self.style(parent);
            let node = &self.nodes[parent.0];
            // Inside the parent's border, and where its content would be
            // unscrolled: the indicator scrolls with it.
            let (ox, oy) = (
                p.x + ps.border.left - node.scrolled_x,
                p.y + ps.border.top - node.scrolled,
            );
            let want = match m.shape {
                MatchShape::Fill => Rect {
                    x: t.x - ox,
                    y: t.y - oy,
                    width: t.width,
                    height: t.height,
                },
                MatchShape::Underline(h) => Rect {
                    x: t.x - ox,
                    y: t.y + t.height - h - oy,
                    width: t.width,
                    height: h,
                },
            };
            if m.last == Some(want) {
                continue;
            }
            let first = m.last.is_none();
            m.last = Some(want);
            moved = true;
            let props = [
                (Property::Left, want.x),
                (Property::Top, want.y),
                (Property::Width, want.width),
                (Property::Height, want.height),
            ];
            if first || m.seconds <= 0.0 {
                self.update_style(m.indicator, |s| {
                    (s.left, s.top, s.width, s.height) = (
                        Length::Px(want.x),
                        Length::Px(want.y),
                        Length::Px(want.width),
                        Length::Px(want.height),
                    );
                });
            } else {
                for (p, to) in props {
                    let from = self.read(m.indicator, p);
                    self.animate(
                        m.indicator,
                        Animation::new(p, from, Value::Number(to), m.seconds)
                            .easing(threers_math::easing::Easing::CubicOut),
                    );
                }
            }
        }
        self.matched = all;
        moved
    }
}
