//! Scrolling that settles where it should, and a scroll position kept as
//! state:
//!
//! - [`Ui::set_scroll_snap`]: a box that scrolls sideways comes to rest
//!   with an item (or a page) at its start, as a carousel does. A flick
//!   picks the item nearest where its glide would have stopped and eases
//!   there; a drag or a trackpad lets go onto the nearest.
//! - [`Ui::snap_offset_x`]: where it would settle from an offset.
//! - [`Input::bind_scroll`] and [`Input::bind_scroll_x`]: a box's scroll
//!   offset bound to the state both ways, to save and restore it, or to
//!   scroll by setting it.
//!
//! ```
//! use threers_ui::scroll_snap::Snap;
//! use threers_ui::{style, FixedWidthText, Ui};
//!
//! let mut ui = Ui::new();
//! let root = ui.root();
//! let strip = ui.add_box(root, style! { display: flex; overflow-x: auto; width: 200px; });
//! for _ in 0..10 {
//!     ui.add_box(strip, style! { width: 80px; height: 40px; flex-shrink: 0; });
//! }
//! ui.set_scroll_snap(strip, Some(Snap::Items));
//! ui.layout(400.0, 100.0, &mut FixedWidthText(7.0));
//! assert_eq!(ui.snap_offset_x(strip, 130.0), 160.0, "the third item's start is the nearest");
//! assert_eq!(ui.snap_offset_x(strip, 900.0), 600.0, "no further than the end");
//! ```

use crate::animation::{Animation, Property, Value};
use crate::binding::Binding;
use crate::input::Input;
use crate::tree::{NodeId, Ui};

/// Where a box scrolling sideways comes to rest.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Snap {
    /// With one of its children's starts at its own.
    Items,
    /// At a whole number of its widths.
    Pages,
}

/// How long settling on a snap point takes, at most.
const SETTLE_SECONDS: f32 = 0.35;

impl Ui {
    /// Makes `container` (which scrolls sideways) settle on `snap` points
    /// (`None`: anywhere).
    pub fn set_scroll_snap(&mut self, container: NodeId, snap: Option<Snap>) {
        let keep: Vec<(NodeId, Snap)> = self
            .snaps
            .iter()
            .copied()
            .filter(|(c, _)| *c != container && self.contains(*c))
            .collect();
        self.snaps = keep;
        if let Some(s) = snap {
            self.snaps.push((container, s));
        }
    }

    pub fn scroll_snap(&self, container: NodeId) -> Option<Snap> {
        self.snaps
            .iter()
            .find(|(c, _)| *c == container)
            .map(|(_, s)| *s)
    }

    /// The snap point of `container` nearest `offset` (its offset itself
    /// when it does not snap), within its extent.
    pub fn snap_offset_x(&self, container: NodeId, offset: f32) -> f32 {
        let (content, view, scrolled) = self.scroll_extent_x(container);
        let max = (content - view).max(0.0);
        let points: Vec<f32> = match self.scroll_snap(container) {
            None => return offset.clamp(0.0, max),
            Some(Snap::Pages) => {
                let page = view.max(1.0);
                (0..=(max / page).ceil() as usize)
                    .map(|i| (i as f32 * page).min(max))
                    .collect()
            }
            Some(Snap::Items) => {
                let r = self.rect(container);
                let s = self.style(container);
                let start = r.x + s.border.left + s.padding.left;
                let mut p: Vec<f32> = self
                    .children(container)
                    .iter()
                    .filter(|&&c| self.style(c).display != crate::Display::None)
                    .map(|&c| {
                        let q = self.rect(c);
                        let along = match self.nodes[container.0].rtl {
                            false => q.x - start,
                            true => r.x + r.width - s.border.left - s.padding.left - q.x - q.width,
                        };
                        (along + scrolled).clamp(0.0, max)
                    })
                    .collect();
                p.push(max);
                p
            }
        };
        points
            .into_iter()
            .min_by(|a, b| (a - offset).abs().total_cmp(&(b - offset).abs()))
            .unwrap_or(0.0)
    }

    /// Eases `container` onto the snap point nearest `aim`; whether it
    /// snaps at all.
    pub(crate) fn settle_x(&mut self, container: NodeId, aim: f32) -> bool {
        if self.scroll_snap(container).is_none() {
            return false;
        }
        let to = self.snap_offset_x(container, aim);
        let now = self.style(container).scroll_left;
        if (to - now).abs() > 0.5 {
            let seconds =
                (SETTLE_SECONDS * ((to - now).abs() / 300.0).sqrt()).clamp(0.12, SETTLE_SECONDS);
            self.animate(
                container,
                Animation::new(
                    Property::ScrollLeft,
                    Value::Number(now),
                    Value::Number(to),
                    seconds,
                )
                .easing(threers_math::easing::Easing::CubicOut),
            );
        } else if now != to {
            self.style_mut(container).scroll_left = to;
        }
        true
    }
}

impl<S: 'static> Input<S> {
    /// Keeps `container`'s scroll offset (down) and `offset` in the state
    /// the same: scrolling writes it; setting it scrolls.
    pub fn bind_scroll(&mut self, container: NodeId, offset: Binding<S, f32>) {
        self.bind_offset(container, offset, false);
    }

    /// [`bind_scroll`](Self::bind_scroll), sideways.
    pub fn bind_scroll_x(&mut self, container: NodeId, offset: Binding<S, f32>) {
        self.bind_offset(container, offset, true);
    }

    fn bind_offset(&mut self, container: NodeId, offset: Binding<S, f32>, sideways: bool) {
        // The offset last agreed between the box and the state.
        let mut agreed: Option<f32> = None;
        self.on_after_dispatch(move |_, ui, state| {
            if !ui.contains(container) {
                return false;
            }
            let read = |ui: &Ui| {
                if sideways {
                    ui.style(container).scroll_left
                } else {
                    ui.style(container).scroll_top
                }
            };
            let want = offset.get(state);
            if agreed != Some(want) {
                // The state asks for somewhere new (or this is the first
                // look): scroll there.
                if sideways {
                    ui.scroll_to_x(container, want);
                } else {
                    ui.scroll_to(container, want);
                }
                let reached = read(ui);
                // Before the box is laid out its extent is nothing and the
                // offset cannot be reached: try again later.
                let (content, _, _) = if sideways {
                    ui.scroll_extent_x(container)
                } else {
                    ui.scroll_extent(container)
                };
                if reached == want || content > 0.0 {
                    if reached != want {
                        offset.set(state, reached);
                    }
                    agreed = Some(reached);
                }
            } else if read(ui) != want {
                // The box scrolled: the state follows.
                let at = read(ui);
                offset.set(state, at);
                agreed = Some(at);
            }
            true
        });
    }
}
