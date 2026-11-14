//! A UI in a headset, where the system knows where a person looks and the
//! app does not. Apple Vision Pro keeps the gaze private: it highlights
//! what is looked at itself, from regions the app declares. A UI drawn by
//! its own renderer, as this one is, says where its controls are with
//! [`Ui::hover_regions`], each frame: what can be pressed, where it shows
//! (cut by what clips it), how round its corners are, and an id that stays
//! the same from frame to frame so the highlight follows the control.
//! threers' visionOS session draws them into the compositor's tracking
//! areas (`threers::xr::PanelPass::draw_hover_regions`).
//!
//! Targets are larger in a headset: the eyes point less precisely than a
//! finger. [`SPATIAL_MIN_TARGET`] is visionOS's 60 points, the size
//! [`Ui::set_min_target`] has the accessibility audit hold controls to.
//!
//! ```
//! use threers_ui::{style, FixedWidthText, Role, Semantics, Ui};
//!
//! let mut ui = Ui::new();
//! let root = ui.root();
//! let save = ui.add_box(root, style! { width: 120px; height: 60px; border-radius: 12px; });
//! ui.set_semantics(save, Semantics::new(Role::Button).label("Save"));
//! ui.add_text(root, "Not a control");
//! ui.layout(400.0, 300.0, &mut FixedWidthText(7.0));
//! let regions = ui.hover_regions();
//! assert_eq!(regions.len(), 1);
//! assert_eq!((regions[0].rect.width, regions[0].radius), (120.0, 12.0));
//! ```

use crate::tree::{NodeId, Ui};
use crate::{Display, Rect, Role};

/// visionOS's smallest comfortable target, in points: a UI pixel on a
/// threers panel is about a point.
pub const SPATIAL_MIN_TARGET: f32 = 60.0;

/// A control the system may highlight when it is looked at. See the
/// [module docs](self).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HoverRegion {
    pub node: NodeId,
    /// Never 0, and the same for the node from frame to frame (a node
    /// removed and its slot reused gets another).
    pub id: u64,
    /// Where it shows, in UI pixels, cut by the boxes that clip it.
    pub rect: Rect,
    /// Its corners' radius, at most half its shorter side.
    pub radius: f32,
    pub role: Role,
}

impl Ui {
    /// The controls on screen that can be pressed or focused, in the order
    /// they are painted (so where two overlap, the later is on top). A
    /// control is a node whose role is one (a button, a link, a field, a
    /// tab, an option, …) or which takes focus, not disabled, shown, and
    /// with pointer events. See the [module docs](self).
    pub fn hover_regions(&self) -> Vec<HoverRegion> {
        let mut out = vec![];
        let mut roots = vec![self.root()];
        roots.extend(self.deferred_order());
        for &root in &roots {
            // Deferred subtrees (menus, popovers) are drawn above all and
            // ignore their ancestors' clips.
            let mut stack = vec![(root, None::<Rect>)];
            while let Some((id, clip)) = stack.pop() {
                let n = &self.nodes[id.0];
                let deferred = n.deferred.is_some() && id != root;
                if n.removed
                    || n.style.display == Display::None
                    || n.style.opacity <= 0.0
                    || deferred
                {
                    continue;
                }
                if n.semantics.as_ref().is_some_and(|s| s.hidden) {
                    continue;
                }
                let shown = match clip {
                    Some(c) => intersect(n.rect, c),
                    None => Some(n.rect).filter(|r| r.width > 0.0 && r.height > 0.0),
                };
                if let (Some(rect), Some(s)) = (shown, &n.semantics)
                    && n.style.pointer_events
                    && !s.disabled
                    && (is_control(s.role) || s.is_focusable())
                {
                    let full = n.rect;
                    let radius = n
                        .style
                        .border_radius
                        .resolve(Some(full.width.min(full.height)))
                        .unwrap_or(0.0);
                    out.push(HoverRegion {
                        node: id,
                        id: ((id.0 as u64 + 1) << 32) | id.1 as u64,
                        rect,
                        radius: radius.clamp(0.0, rect.width.min(rect.height) / 2.0),
                        role: s.role,
                    });
                }
                let inner = if n.style.clip {
                    match clip {
                        Some(c) => Some(intersect(n.rect, c).unwrap_or(Rect {
                            x: c.x,
                            y: c.y,
                            width: 0.0,
                            height: 0.0,
                        })),
                        None => Some(n.rect),
                    }
                } else {
                    clip
                };
                // A box clipped to nothing has nothing inside to show.
                if inner.is_some_and(|c| c.width <= 0.0 || c.height <= 0.0) {
                    continue;
                }
                let order: Vec<NodeId> = self.paint_order_iter(id).collect();
                stack.extend(order.into_iter().rev().map(|c| (c, inner)));
            }
        }
        out
    }

    /// The smallest target, in UI pixels, the
    /// [accessibility audit](Ui::accessibility_audit) holds controls to:
    /// WCAG 2.2's 24 by default, [`SPATIAL_MIN_TARGET`] in a headset (0
    /// puts the default back).
    pub fn set_min_target(&mut self, pixels: f32) {
        self.env.min_target = pixels.max(0.0);
    }

    /// See [`set_min_target`](Self::set_min_target).
    pub fn min_target(&self) -> f32 {
        if self.env.min_target > 0.0 {
            self.env.min_target
        } else {
            crate::a11y_core::MIN_TARGET
        }
    }
}

/// Roles that are pressed or typed into.
pub(crate) fn is_control(role: Role) -> bool {
    matches!(
        role,
        Role::Button
            | Role::Link
            | Role::Checkbox
            | Role::Radio
            | Role::Switch
            | Role::Slider
            | Role::TextBox
            | Role::TextArea
            | Role::ComboBox
            | Role::SearchBox
            | Role::SpinButton
            | Role::Tab
            | Role::MenuItem
            | Role::Option
    )
}

fn intersect(a: Rect, b: Rect) -> Option<Rect> {
    let (x0, y0) = (a.x.max(b.x), a.y.max(b.y));
    let (x1, y1) = (
        (a.x + a.width).min(b.x + b.width),
        (a.y + a.height).min(b.y + b.height),
    );
    (x1 > x0 && y1 > y0).then_some(Rect {
        x: x0,
        y: y0,
        width: x1 - x0,
        height: y1 - y0,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{style, FixedWidthText, Semantics};

    #[test]
    fn regions_are_cut_by_clips_ordered_as_painted_and_keep_their_ids() {
        let mut ui = Ui::new();
        let root = ui.root();
        let list = ui.add_box(
            root,
            style! { width: 200px; height: 100px; overflow: hidden; flex-direction: column; },
        );
        let mut rows = vec![];
        for i in 0..4 {
            let r = ui.add_box(list, style! { width: 200px; height: 40px; flex-shrink: 0; });
            ui.set_semantics(r, Semantics::new(Role::Button).label(format!("Row {i}")));
            rows.push(r);
        }
        let off = ui.add_box(root, style! { width: 50px; height: 50px; });
        ui.set_semantics(
            off,
            Semantics {
                disabled: true,
                ..Semantics::new(Role::Button).label("Off")
            },
        );
        let gone = ui.add_box(root, style! { width: 50px; height: 50px; display: none; });
        ui.set_semantics(gone, Semantics::new(Role::Button).label("Gone"));
        let over = ui.add_box(
            root,
            style! { position: absolute; left: 10px; top: 10px; width: 30px; height: 30px; },
        );
        ui.set_semantics(over, Semantics::new(Role::Link).label("Over"));
        ui.layout(400.0, 300.0, &mut FixedWidthText(7.0));
        let r = ui.hover_regions();
        let names: Vec<NodeId> = r.iter().map(|h| h.node).collect();
        // The third row is cut to what shows; the fourth does not show.
        assert_eq!(names, vec![rows[0], rows[1], rows[2], over]);
        assert_eq!(r[2].rect.height, 20.0);
        // Positioned: painted last, so last.
        assert_eq!(r.last().unwrap().role, Role::Link);
        let ids: Vec<u64> = r.iter().map(|h| h.id).collect();
        assert!(ids.iter().all(|&i| i != 0));
        ui.layout(401.0, 300.0, &mut FixedWidthText(7.0));
        assert_eq!(
            ui.hover_regions().iter().map(|h| h.id).collect::<Vec<_>>(),
            ids
        );
    }

    #[test]
    fn the_audit_holds_targets_to_the_minimum_set() {
        let mut ui = Ui::new();
        let root = ui.root();
        let b = ui.add_box(root, style! { width: 44px; height: 44px; });
        ui.set_semantics(b, Semantics::new(Role::Button).label("OK"));
        ui.layout(400.0, 300.0, &mut FixedWidthText(7.0));
        let small = |ui: &Ui| {
            ui.accessibility_audit()
                .iter()
                .any(|i| i.kind == crate::a11y_core::Audit::SmallTarget)
        };
        assert!(!small(&ui));
        ui.set_min_target(SPATIAL_MIN_TARGET);
        assert!(small(&ui));
    }
}
