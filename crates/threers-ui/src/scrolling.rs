//! Scrolling by program and by scrollbar: bring a node into view, scroll to an offset
//! (smoothly, with [`Ui::scroll_smoothly`]), and scrollbars that are drawn
//! over a scrolling box's edge and can be dragged.
//!
//! Scrollbars are off unless asked for (CSS `scrollbar-width: thin | auto`),
//! so a UI built to match a page with overlay scrollbars keeps matching it.
//!
//! ```
//! use threers_ui::{style, FixedWidthText, Ui};
//!
//! let mut ui = Ui::new();
//! let root = ui.root();
//! let list = ui.add_box(root, style! { height: 100px; overflow: auto; scrollbar-width: thin; });
//! let rows: Vec<_> = (0..50).map(|_| ui.add_box(list, style! { height: 20px; })).collect();
//! ui.layout(200.0, 200.0, &mut FixedWidthText(7.0));
//! ui.scroll_into_view(rows[30]);
//! ui.layout(200.0, 200.0, &mut FixedWidthText(7.0));
//! assert_eq!(ui.scroll_extent(list).2, 520.0); // row 30's bottom at the view's bottom
//! let bar = ui.scrollbar(list).unwrap();
//! assert!(bar.thumb.height < bar.track.height);
//! ```

use super::style::{Rgba, Scrollbar, ScrollbarVisibility};
use super::tree::{NodeId, Rect, Ui};

/// Where a scrolling box's scrollbar is, as last laid out.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScrollbarGeometry {
    pub track: Rect,
    pub thumb: Rect,
}

/// A part of a scrollbar.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScrollbarPart {
    Thumb,
    Track,
}

/// Where to scroll something to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ScrollStrategy {
    /// Into view by the least: not at all if it shows.
    #[default]
    Nearest,
    /// To the top of the view.
    Top,
    /// To the middle of the view.
    Center,
    /// To the bottom of the view.
    Bottom,
}

/// A thumb is never shorter than this, to stay grabbable.
const MIN_THUMB: f32 = 20.0;

impl Ui {
    /// Scrolls `container` to `offset` (clamped to its content).
    pub fn scroll_to(&mut self, container: NodeId, offset: f32) {
        let (content, view, _) = self.scroll_extent(container);
        let to = offset.clamp(0.0, (content - view).max(0.0));
        if self.style(container).scroll_top != to {
            self.style_mut(container).scroll_top = to;
        }
    }

    /// Scrolls every scrolling box around `id`, innermost first, by as
    /// little as brings it into view (its top, if it is taller than the
    /// view). Lay out again to see it.
    /// Scrolls `id`'s nearest scrolling ancestor to put it where `strategy`
    /// says, and the boxes further out by the least, to show that box.
    pub fn scroll_into_view_with(&mut self, id: NodeId, strategy: ScrollStrategy) {
        if strategy == ScrollStrategy::Nearest {
            return self.scroll_into_view(id);
        }
        let container = self.ancestors(id).skip(1).find(|&n| self.style(n).scroll);
        if let Some(c) = container {
            let target = self.rect(id);
            let r = self.rect(c);
            let b = self.style(c).border;
            let (view_top, view_h) = (r.y + b.top, (r.height - b.vertical()).max(0.0));
            let (_, _, now) = self.scroll_extent(c);
            let delta = match strategy {
                ScrollStrategy::Top => target.y - view_top,
                ScrollStrategy::Bottom => target.y + target.height - (view_top + view_h),
                _ => target.y + target.height / 2.0 - (view_top + view_h / 2.0),
            };
            self.scroll_to(c, now + delta);
            self.scroll_into_view(c);
        }
    }

    pub fn scroll_into_view(&mut self, id: NodeId) {
        self.scroll_into_view_x(id);
        let target = self.rect(id);
        let (mut top, mut bottom) = (target.y, target.y + target.height);
        let containers: Vec<NodeId> = self
            .ancestors(id)
            .skip(1)
            .filter(|&n| self.style(n).scroll)
            .collect();
        for c in containers {
            let r = self.rect(c);
            let b = self.style(c).border;
            let (view_top, view_h) = (r.y + b.top, (r.height - b.vertical()).max(0.0));
            // The rects are as laid out, at the offset layout applied.
            let (_, _, now) = self.scroll_extent(c);
            let delta = if top < view_top {
                top - view_top
            } else if bottom > view_top + view_h {
                ((bottom - (view_top + view_h)).min(top - view_top)).max(0.0)
            } else {
                0.0
            };
            // Out to whole device pixels, as layout puts the offset: else
            // the rounding can leave a fraction of it out of view.
            let ratio = self.pixel_ratio;
            let delta = if delta > 0.0 { (delta * ratio - 1e-3).ceil() / ratio } else { (delta * ratio + 1e-3).floor() / ratio };
            if delta != 0.0 {
                self.scroll_to(c, now + delta);
                let moved = self.style(c).scroll_top - now;
                top -= moved;
                bottom -= moved;
            }
            // Further out, the container is what has to show.
            top = top.max(r.y);
            bottom = bottom.min(r.y + r.height);
        }
    }

    /// `container`'s scrollbar, if it shows one: it scrolls, asks for a
    /// scrollbar, and has more content than fits.
    pub fn scrollbar(&self, container: NodeId) -> Option<ScrollbarGeometry> {
        let s = self.style(container);
        let width = match s.scrollbar {
            Scrollbar::None => return None,
            Scrollbar::Thin => 6.0,
            Scrollbar::Auto => 10.0,
        };
        if !s.scroll {
            return None;
        }
        let (content, view, scrolled) = self.scroll_extent(container);
        if content <= view + 0.5 {
            return None;
        }
        let r = self.rect(container);
        let track = Rect {
            x: r.x + r.width - s.border.right - width,
            y: r.y + s.border.top,
            width,
            height: view,
        };
        let h = (view / content * track.height).max(MIN_THUMB).min(track.height);
        let travel = (content - view).max(1e-6);
        let thumb = Rect {
            y: track.y + scrolled / travel * (track.height - h),
            height: h,
            ..track
        };
        Some(ScrollbarGeometry { track, thumb })
    }

    /// The scrollbar part at `(x, y)`, and whose it is: the innermost
    /// scrolling box under the point with a scrollbar there.
    pub fn scrollbar_at(&self, x: f32, y: f32) -> Option<(NodeId, ScrollbarPart)> {
        self.scrollbar_reach(x, y, 0.0)
    }

    /// [`scrollbar_at`](Self::scrollbar_at), with the thumb taking points up
    /// to `reach` to its left too (for a finger).
    pub(crate) fn scrollbar_reach(&self, x: f32, y: f32, reach: f32) -> Option<(NodeId, ScrollbarPart)> {
        let hit = self.hit_test(x, y)?;
        self.ancestors(hit).find_map(|n| {
            let g = self.scrollbar(n)?;
            let grab = Rect { x: g.thumb.x - reach, width: g.thumb.width + reach, ..g.thumb };
            if grab.contains(x, y) {
                Some((n, ScrollbarPart::Thumb))
            } else if g.track.contains(x, y) {
                Some((n, ScrollbarPart::Track))
            } else {
                None
            }
        })
    }

    /// The scroll offset that puts the thumb's top at `thumb_y`.
    pub(crate) fn scroll_for_thumb(&self, container: NodeId, thumb_y: f32) -> Option<f32> {
        let g = self.scrollbar(container)?;
        let (content, view, _) = self.scroll_extent(container);
        let room = (g.track.height - g.thumb.height).max(1e-6);
        Some(((thumb_y - g.track.y) / room).clamp(0.0, 1.0) * (content - view).max(0.0))
    }

    /// Shows `container`'s `auto` scrollbar as if just used: the pointer is
    /// over it, or drags its thumb. It fades again after `scrollbar-fade`.
    pub fn reveal_scrollbar(&mut self, container: NodeId) {
        if self.contains(container) && self.style(container).scrollbar_visibility == ScrollbarVisibility::Auto {
            let top = self.style(container).scroll_top;
            self.scrollbar_use.insert(container, (top, self.clock));
        }
    }

    /// How much of `container`'s scrollbar shows, 0 to 1: always 1 for an
    /// `always` one; for an `auto` one, 1 until `scrollbar-fade`'s delay
    /// after it was last used, then less, to 0 at the end of the fade.
    pub fn scrollbar_opacity(&self, container: NodeId) -> f32 {
        let s = self.style(container);
        if s.scrollbar_visibility == ScrollbarVisibility::Always {
            return 1.0;
        }
        let Some(&(_, used)) = self.scrollbar_use.get(&container) else {
            return 0.0;
        };
        let (delay, duration) = s.scrollbar_fade;
        let t = (self.clock - used) as f32 - delay;
        if t <= 0.0 {
            1.0
        } else if duration <= 0.0 {
            0.0
        } else {
            (1.0 - t / duration).clamp(0.0, 1.0)
        }
    }

    /// Whether `container` scrolls with an `auto` scrollbar.
    pub(crate) fn fades_scrollbar(&self, container: NodeId) -> bool {
        let s = self.style(container);
        s.scroll && s.scrollbar_visibility == ScrollbarVisibility::Auto
    }

    /// Notes which `auto` scrollbars' boxes scrolled since last time (their
    /// bars show again). Returns whether any bar is still showing, so more
    /// frames are wanted to fade it.
    pub(crate) fn watch_scrollbars(&mut self) -> bool {
        // Removed, or no longer an `auto` scrollbar (a restyle can change
        // that): nothing to fade.
        let gone: Vec<NodeId> = self
            .scrollbar_use
            .keys()
            .copied()
            .filter(|&n| !self.contains(n) || !self.fades_scrollbar(n))
            .collect();
        for n in gone {
            self.scrollbar_use.remove(&n);
        }
        let mut showing = false;
        for i in 0..self.nodes.len() {
            let node = &self.nodes[i];
            if node.removed || !node.style.scroll || node.style.scrollbar_visibility != ScrollbarVisibility::Auto {
                continue;
            }
            let id = self.id_at(i);
            let top = node.style.scroll_top;
            let clock = self.clock;
            let used = self.scrollbar_use.entry(id).or_insert((top, f64::NEG_INFINITY));
            if used.0 != top {
                *used = (top, clock);
            }
            showing |= self.scrollbar_opacity(id) > 0.0;
        }
        showing
    }

    pub(crate) fn scrollbar_colors(&self, container: NodeId) -> (Rgba, Rgba) {
        self.style(container)
            .scrollbar_color
            .unwrap_or(([0.5, 0.5, 0.5, 0.6], [0.0, 0.0, 0.0, 0.0]))
    }
}

impl Ui {
    /// Sideways, as [`scroll_into_view`](Self::scroll_into_view) is down:
    /// every box around `id` that scrolls sideways, by as little as shows
    /// it (its start, if it is wider).
    fn scroll_into_view_x(&mut self, id: NodeId) {
        let target = self.rect(id);
        let (mut left, mut right) = (target.x, target.x + target.width);
        let containers: Vec<NodeId> = self.ancestors(id).skip(1).filter(|&n| self.style(n).scroll_x).collect();
        for c in containers {
            let r = self.rect(c);
            let b = self.style(c).border;
            let (view_left, view_w) = (r.x + b.left, (r.width - b.horizontal()).max(0.0));
            let now = self.style(c).scroll_left;
            let delta = if left < view_left {
                left - view_left
            } else if right > view_left + view_w {
                ((right - (view_left + view_w)).min(left - view_left)).max(0.0)
            } else {
                0.0
            };
            let ratio = self.pixel_ratio;
            let delta = if delta > 0.0 { (delta * ratio - 1e-3).ceil() / ratio } else { (delta * ratio + 1e-3).floor() / ratio };
            if delta != 0.0 {
                self.scroll_to_x(c, now + delta);
                let moved = self.style(c).scroll_left - now;
                left -= moved;
                right -= moved;
            }
            left = left.max(r.x);
            right = right.min(r.x + r.width);
        }
    }
}

