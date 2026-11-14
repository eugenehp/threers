//! Items of a scrolling box that change as they enter and leave its view:
//! fading, shrinking, tilting back over the edge they cross, by how far
//! out of view they are ([`Ui::scroll_transition`]). An item wholly in view
//! is as it is; one half out is halfway changed.
//!
//! ```
//! use threers_ui::scroll_transition::ScrollTransition;
//! use threers_ui::{style, FixedWidthText, Ui};
//!
//! let mut ui = Ui::new();
//! let root = ui.root();
//! let list = ui.add_box(root, style! { height: 100px; overflow: auto; });
//! let rows: Vec<_> = (0..10).map(|_| ui.add_box(list, style! { height: 40px; flex-shrink: 0; })).collect();
//! ui.scroll_transition(list, ScrollTransition::default());
//! ui.layout(300.0, 300.0, &mut FixedWidthText(7.0));
//! assert_eq!(ui.style(rows[0]).opacity, 1.0);
//! assert!(ui.style(rows[2]).opacity < 1.0, "half out of view at the bottom");
//! ```

use crate::depth::Transform3d;
use crate::tree::{NodeId, Ui};

/// How a scrolling box's items change by how far out of view they are.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScrollTransition {
    /// Opacity lost when wholly out (0 to 1).
    pub fade: f32,
    /// Size lost when wholly out (0.2: drawn at 80%).
    pub shrink: f32,
    /// Radians turned back over the edge when wholly out.
    pub tilt: f32,
}

impl Default for ScrollTransition {
    /// Fading and shrinking a little, as a list's rows go.
    fn default() -> Self {
        ScrollTransition {
            fade: 0.7,
            shrink: 0.12,
            tilt: 0.0,
        }
    }
}

impl ScrollTransition {
    /// Rows that turn back over the edges, as on a wheel.
    pub fn tilted() -> Self {
        ScrollTransition {
            fade: 0.5,
            shrink: 0.06,
            tilt: 0.9,
        }
    }
}

impl Ui {
    /// Changes `container`'s items by how far out of its view each is (see
    /// the [module docs](self)), kept up after each layout; `None` stops it
    /// (the items are left as they are then).
    pub fn scroll_transition(
        &mut self,
        container: NodeId,
        transition: impl Into<Option<ScrollTransition>>,
    ) {
        self.env.scroll_transitions.retain(|(c, _)| *c != container);
        if let Some(t) = transition.into() {
            self.env.scroll_transitions.push((container, t));
            self.touch(container);
        }
    }

    /// Where an item is against its box's view: −1 wholly out above, 0
    /// wholly in, 1 wholly out below (in between by how much is out).
    pub fn scroll_phase(&self, container: NodeId, item: NodeId) -> f32 {
        let (v, r) = (self.rect(container), self.rect(item));
        let h = r.height.max(1.0);
        let above = ((v.y - r.y) / h).clamp(0.0, 1.0);
        let below = ((r.y + r.height - (v.y + v.height)) / h).clamp(0.0, 1.0);
        if above > 0.0 {
            -above
        } else {
            below
        }
    }

    /// The items placed by where they now are (after each layout).
    pub(crate) fn place_scroll_transitions(&mut self) {
        let all: Vec<(NodeId, ScrollTransition)> = self
            .env
            .scroll_transitions
            .iter()
            .copied()
            .filter(|(c, _)| self.contains(*c))
            .collect();
        self.env.scroll_transitions = all.clone();
        for (c, t) in all {
            for item in self.children(c).to_vec() {
                let phase = self.scroll_phase(c, item);
                let a = phase.abs();
                self.write(
                    item,
                    crate::animation::Property::Opacity,
                    crate::animation::Value::Number(1.0 - a * t.fade.clamp(0.0, 1.0)),
                );
                let now = self.transform3d(item);
                if t.shrink == 0.0 && t.tilt == 0.0 && now.is_none() {
                    continue;
                }
                let base = now.unwrap_or_default();
                // Turned on the edge it is crossing: the top edge for one
                // leaving at the top.
                let origin = (0.5, if phase < 0.0 { 1.0 } else { 0.0 });
                let turned = Transform3d {
                    scale: 1.0 - a * t.shrink,
                    rotate_x: phase * t.tilt,
                    origin,
                    perspective: base.perspective.max(600.0),
                    ..base
                };
                self.env.transforms3d.insert(item, turned);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{style, FixedWidthText};

    #[test]
    fn items_change_by_how_far_out_of_view_and_follow_the_scroll() {
        let mut ui = Ui::new();
        let root = ui.root();
        let list = ui.add_box(root, style! { height: 100px; overflow: auto; });
        let rows: Vec<_> = (0..10)
            .map(|_| ui.add_box(list, style! { height: 40px; flex-shrink: 0; }))
            .collect();
        ui.scroll_transition(list, ScrollTransition::tilted());
        ui.layout(300.0, 300.0, &mut FixedWidthText(7.0));
        // Rows at 0, 40 in; 80..120 half out below; 120.. out.
        assert_eq!(ui.scroll_phase(list, rows[0]), 0.0);
        assert_eq!(ui.scroll_phase(list, rows[2]), 0.5);
        assert_eq!(ui.scroll_phase(list, rows[5]), 1.0);
        let t = ui.transform3d(rows[2]).unwrap();
        assert!(
            (t.rotate_x - 0.45).abs() < 1e-4 && (t.scale - 0.97).abs() < 1e-4,
            "{t:?}"
        );
        assert!((ui.style(rows[2]).opacity - 0.75).abs() < 1e-4);
        // Scrolled down 60: the first row half out at the top, turned the
        // other way.
        ui.update_style(list, |s| s.scroll_top = 60.0);
        ui.layout(300.0, 300.0, &mut FixedWidthText(7.0));
        assert_eq!(ui.scroll_phase(list, rows[1]), -0.5);
        assert!(ui.transform3d(rows[1]).unwrap().rotate_x < 0.0);
        assert_eq!(ui.style(rows[3]).opacity, 1.0);
    }
}
