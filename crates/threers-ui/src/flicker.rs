//! Nodes that flicker at a given frequency: on and off as a square wave, or
//! up and down as a sine (see [`Flicker`]).
//!
//! The flicker multiplies the node's opacity when it is painted, by the
//! level at the tree's clock (the seconds [`Ui::step`] has been given), so
//! it reaches everything inside the node, and its reflection, as
//! `opacity` does. It leaves the style alone: an animation or a state style
//! can move the opacity of a flickering node, and the flicker multiplies
//! whatever that comes to. A node flickered off still takes the pointer and
//! still stands in the accessibility tree, as one at `opacity: 0` does.
//!
//! ```
//! use threers_ui::flicker::Flicker;
//! use threers_ui::{style, Ui};
//!
//! let mut ui = Ui::new();
//! let root = ui.root();
//! let target = ui.add_box(root, style! { width: 100px; height: 100px; background: white; });
//! ui.set_flicker(target, Some(Flicker::square(12.0)));
//! assert!(ui.needs_frame());
//! assert_eq!(ui.flicker_level(target), 1.0);
//! ui.step(1.0 / 20.0); // 0.6 of a cycle in: the off half
//! assert_eq!(ui.flicker_level(target), 0.0);
//! ```

pub use threers_math::flicker::{Flicker, Waveform};

use super::tree::{NodeId, Ui};

impl Ui {
    /// Makes `id` flicker, or (`None`) stop. A node removed takes its
    /// flicker with it.
    pub fn set_flicker(&mut self, id: NodeId, flicker: Option<Flicker>) {
        assert!(self.contains(id), "set_flicker on a removed node");
        let was = self.flickers.iter().position(|(n, _)| *n == id);
        match (was, flicker) {
            (Some(i), Some(f)) => self.flickers[i].1 = f,
            (Some(i), None) => {
                self.flickers.swap_remove(i);
            }
            (None, Some(f)) => self.flickers.push((id, f)),
            (None, None) => return,
        }
        self.env.repaint.set(true);
    }

    /// The flicker `id` was given, if any.
    pub fn flicker(&self, id: NodeId) -> Option<Flicker> {
        self.flickers
            .iter()
            .find(|(n, _)| *n == id)
            .map(|(_, f)| *f)
    }

    /// How far `id`'s flicker shows it at the tree's clock, 0..=1: 1 for a
    /// node that does not flicker.
    pub fn flicker_level(&self, id: NodeId) -> f32 {
        self.flickers
            .iter()
            .find(|(n, _)| *n == id)
            .map_or(1.0, |(_, f)| f.level(self.clock))
    }

    /// The node's own opacity as painted: its style's, times its flicker.
    pub(crate) fn painted_opacity(&self, id: NodeId) -> f32 {
        let own = self.nodes[id.0].style.opacity;
        if self.flickers.is_empty() {
            own
        } else {
            own * self.flicker_level(id)
        }
    }

    /// Drops the flickers of removed nodes, and says whether one still
    /// on screen needs frames.
    pub(crate) fn step_flickers(&mut self) -> bool {
        if self.flickers.is_empty() {
            return false;
        }
        let flickers = std::mem::take(&mut self.flickers);
        self.flickers = flickers
            .into_iter()
            .filter(|(n, _)| self.contains(*n))
            .collect();
        self.flickers_animate()
    }

    /// Whether a flicker that changes over time is on a node on screen.
    pub(crate) fn flickers_animate(&self) -> bool {
        self.flickers.iter().any(|(n, f)| {
            f.frequency > 0.0
                && f.low.clamp(0.0, 1.0) != f.high.clamp(0.0, 1.0)
                && self.is_on_screen(*n)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{damage, FixedWidthText, Painter, Rect, Rgba, Style, TextMeasure, TextStyle};

    #[derive(Default)]
    struct Alphas(Vec<f32>);

    impl TextMeasure for Alphas {
        fn measure(&mut self, text: &str, style: &Style) -> (f32, f32) {
            FixedWidthText(7.0).measure(text, style)
        }
    }

    impl Painter for Alphas {
        fn fill_rect(&mut self, _: Rect, c: Rgba) {
            self.0.push(c[3]);
        }
        fn text(&mut self, _: Rect, _: &str, _: &TextStyle) {}
        fn path(&mut self, _: Rect, _: &[String], _: [f32; 4], _: Option<f32>, _: bool, _: Rgba) {}
    }

    fn tree() -> (Ui, NodeId, NodeId) {
        let mut ui = Ui::new();
        let root = ui.root();
        let outer = ui.add_box(
            root,
            Style {
                width: crate::Length::Px(40.0),
                height: crate::Length::Px(40.0),
                background: Some([1.0; 4]),
                opacity: 0.5,
                ..Style::default()
            },
        );
        let inner = ui.add_box(
            outer,
            Style {
                width: crate::Length::Px(10.0),
                height: crate::Length::Px(10.0),
                background: Some([1.0; 4]),
                ..Style::default()
            },
        );
        ui.layout(100.0, 100.0, &mut Alphas::default());
        (ui, outer, inner)
    }

    fn paint(ui: &Ui) -> Vec<f32> {
        let mut p = Alphas::default();
        ui.paint(&mut p);
        p.0
    }

    #[test]
    fn a_square_flicker_shows_and_hides_the_node_and_what_is_inside() {
        let (mut ui, outer, _) = tree();
        assert_eq!(paint(&ui), [0.5, 0.5]);
        ui.set_flicker(outer, Some(Flicker::square(10.0)));
        assert_eq!(paint(&ui), [0.5, 0.5], "a cycle starts on");
        ui.step(0.06);
        assert!(paint(&ui).is_empty(), "off for the second half");
        ui.step(0.05);
        assert_eq!(paint(&ui), [0.5, 0.5]);
        ui.set_flicker(outer, None);
        ui.step(0.06);
        assert_eq!(paint(&ui), [0.5, 0.5], "stopped");
        assert!(!ui.needs_frame());
    }

    #[test]
    fn a_sine_flicker_multiplies_the_style_opacity() {
        let (mut ui, outer, inner) = tree();
        ui.set_flicker(inner, Some(Flicker::sine(1.0).range(0.2, 1.0)));
        ui.step(0.5);
        let a = paint(&ui);
        assert_eq!(a[0], 0.5);
        assert!((a[1] - 0.1).abs() < 1e-5, "{a:?}");
        // An animation moving the opacity is multiplied, not overwritten.
        ui.update_style(outer, |s| s.opacity = 1.0);
        let a = paint(&ui);
        assert!((a[1] - 0.2).abs() < 1e-5, "{a:?}");
        assert_eq!(ui.style(inner).opacity, 1.0);
    }

    #[test]
    fn flickering_keeps_frames_coming_and_damages_the_node() {
        let (mut ui, outer, inner) = tree();
        ui.step(0.0);
        assert!(!ui.needs_frame());
        ui.set_flicker(inner, Some(Flicker::square(10.0)));
        let before = damage::collect(&ui);
        assert!(ui.step(0.06), "step asks for the next frame");
        assert!(ui.needs_frame());
        let after = damage::collect(&ui);
        assert_ne!(
            damage::diff(&before, &after, 100.0, 100.0),
            damage::Damage::None
        );
        // A flicker that cannot change asks for nothing.
        ui.set_flicker(inner, Some(Flicker::square(0.0)));
        assert!(!ui.step(0.01));
        paint(&ui);
        assert!(!ui.needs_frame());
        // Removed with its node.
        ui.set_flicker(inner, Some(Flicker::square(10.0)));
        ui.remove(outer);
        assert!(!ui.step(0.01));
        assert!(ui.flickers.is_empty());
    }
}
