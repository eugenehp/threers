//! What a UI does for accessibility without being asked:
//!
//! - **Modal dialogs keep assistive technology in.** While one is open
//!   ([`Input::set_modal`](crate::input::Input::set_modal) says so), the
//!   accessibility tree is that dialog alone, marked modal, so a screen
//!   reader's cursor cannot wander behind it.
//! - **Announcements.** [`Ui::announce`] has a message read without a node
//!   to carry it ("Saved", "3 results"); [`Ui::announce_assertive`]
//!   interrupts. The DOM mirror reads them out through live regions.
//! - **A focus ring.** Where focus came from the keyboard and the focused
//!   node has no focus style of its own, a ring is drawn round it in the
//!   focus colour ([`Ui::set_focus_ring`] turns it off).
//! - **An audit.** [`Ui::accessibility_audit`] lists what will trouble
//!   someone using assistive technology or a keyboard: controls with no
//!   name, pictures with no description, targets too small to touch,
//!   things that can take focus but are hidden from assistive technology,
//!   headings that skip a level, and text without enough contrast.
//!
//! ```
//! use threers_ui::a11y_core::Audit;
//! use threers_ui::{style, FixedWidthText, Role, Semantics, Ui};
//!
//! let mut ui = Ui::new();
//! let root = ui.root();
//! let b = ui.add_box(root, style! { width: 12px; height: 12px; });
//! ui.set_semantics(b, Semantics::new(Role::Button));
//! ui.layout(400.0, 300.0, &mut FixedWidthText(7.0));
//! let found: Vec<Audit> = ui.accessibility_audit().into_iter().map(|i| i.kind).collect();
//! assert!(found.contains(&Audit::NoName));
//! assert!(found.contains(&Audit::SmallTarget));
//! ```

use crate::environment::Token;
use crate::tree::{NodeId, Ui};
use crate::{Painter, Rect, Role};

/// What an audit found wrong.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Audit {
    /// A control (a button, a link, a field, a checkbox, …) with no name:
    /// a screen reader can only say what it is, not what it does.
    NoName,
    /// A picture with no description.
    ImageNoName,
    /// A control smaller than 24 × 24 pixels (WCAG 2.2's target size), or
    /// than the [minimum set](Ui::set_min_target).
    SmallTarget,
    /// Hidden from assistive technology, yet able to take focus (by a
    /// press, or from the app).
    HiddenFocusable,
    /// A heading more than one level below the one before it.
    HeadingSkipsLevel,
    /// Text with too little contrast against what is behind it.
    LowContrast,
}

/// One thing an audit found.
#[derive(Debug, Clone, PartialEq)]
pub struct AuditIssue {
    pub kind: Audit,
    pub id: NodeId,
    /// What it is, for a person reading the list.
    pub detail: String,
}

/// The smallest target WCAG 2.2 (level AA) asks for, in pixels.
pub(crate) const MIN_TARGET: f32 = 24.0;

impl Ui {
    /// Keeps the accessibility tree to `dialog` while it is open (`None`:
    /// the whole UI). [`Input::set_modal`](crate::input::Input::set_modal)
    /// calls it.
    pub fn set_accessibility_modal(&mut self, dialog: Option<NodeId>) {
        self.a11y_modal = dialog;
    }

    /// Has `message` read out politely: after what is being read.
    pub fn announce(&mut self, message: impl Into<String>) {
        self.push_announcement(message.into(), false);
    }

    /// Has `message` read out at once, interrupting (an error).
    pub fn announce_assertive(&mut self, message: impl Into<String>) {
        self.push_announcement(message.into(), true);
    }

    fn push_announcement(&mut self, message: String, assertive: bool) {
        self.announced += 1;
        self.announcements
            .push((self.announced, message, assertive));
        // A few are kept, for a mirror that looks only now and then.
        let n = self.announcements.len();
        if n > 8 {
            self.announcements.drain(..n - 8);
        }
    }

    /// The announcements after the one numbered `after` (0 for all kept):
    /// each one's number, message, and whether it interrupts.
    pub fn announcements_since(&self, after: u64) -> Vec<(u64, String, bool)> {
        self.announcements
            .iter()
            .filter(|a| a.0 > after)
            .cloned()
            .collect()
    }

    /// Whether a ring is drawn round a node focused from the keyboard that
    /// has no focus style of its own (on by default).
    pub fn set_focus_ring(&mut self, on: bool) {
        self.focus_ring = on;
    }

    /// The ring, if one is due: after everything else is painted.
    pub(crate) fn paint_focus_ring(&self, painter: &mut dyn Painter) {
        if !self.focus_ring || !self.interaction.focus_visible {
            return;
        }
        let Some(f) = self.interaction.focused.filter(|&f| self.contains(f)) else {
            return;
        };
        let styled = self.nodes[f.0]
            .states
            .as_ref()
            .is_some_and(|s| !s.focus_visible.is_empty() || !s.focus.is_empty());
        if styled || self.style(f).display == crate::Display::None {
            return;
        }
        let r = self.rect(f);
        if r.width <= 0.0 || r.height <= 0.0 {
            return;
        }
        let radius = self
            .style(f)
            .border_radius
            .resolve(Some(r.width.min(r.height)))
            .unwrap_or(0.0);
        let c = self.token(Token::Focus);
        // A 2 px ring 2 px outside the box, as a browser's focus ring.
        let (gap, w) = (2.0, 2.0);
        let outer = Rect {
            x: r.x - gap - w,
            y: r.y - gap - w,
            width: r.width + 2.0 * (gap + w),
            height: r.height + 2.0 * (gap + w),
        };
        let rr = radius + gap + w;
        for side in [
            Rect { height: w, ..outer },
            Rect {
                y: outer.y + outer.height - w,
                height: w,
                ..outer
            },
            Rect { width: w, ..outer },
            Rect {
                x: outer.x + outer.width - w,
                width: w,
                ..outer
            },
        ] {
            painter.fill_rounded_rect(side, rr.min(w / 2.0), rr.min(w / 2.0), c);
        }
    }

    /// What will trouble someone using assistive technology or a keyboard,
    /// in the UI as last laid out (contrast is checked against `page`, the
    /// colour behind the root).
    pub fn accessibility_audit(&self) -> Vec<AuditIssue> {
        let tree = self.accessibility_tree();
        let mut out = vec![];
        let mut last_heading: Option<u8> = None;
        let min = self.min_target();
        for n in &tree {
            let control = matches!(
                n.role,
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
            );
            let what = format!("{:?} {:?}", n.role, n.name);
            if control && n.name.trim().is_empty() {
                out.push(AuditIssue {
                    kind: Audit::NoName,
                    id: n.id,
                    detail: what.clone(),
                });
            }
            if n.role == Role::Image && n.name.trim().is_empty() {
                out.push(AuditIssue {
                    kind: Audit::ImageNoName,
                    id: n.id,
                    detail: what.clone(),
                });
            }
            if control
                && !n.disabled
                && n.bounds.width > 0.0
                && (n.bounds.width < min || n.bounds.height < min)
            {
                out.push(AuditIssue {
                    kind: Audit::SmallTarget,
                    id: n.id,
                    detail: format!(
                        "{what}: {}×{}",
                        n.bounds.width.round(),
                        n.bounds.height.round()
                    ),
                });
            }
            if let Role::Heading(level) = n.role {
                if last_heading.is_some_and(|l| level > l + 1)
                    || (last_heading.is_none() && level > 2)
                {
                    out.push(AuditIssue {
                        kind: Audit::HeadingSkipsLevel,
                        id: n.id,
                        detail: format!("h{level} {:?}", n.name),
                    });
                }
                last_heading = Some(level);
            }
        }
        // Able to take focus (by a press, or from the app; Tab skips it),
        // yet hidden from assistive technology: focus lands where a screen
        // reader says nothing.
        for i in 0..self.nodes.len() {
            let n = &self.nodes[i];
            let Some(s) = n.semantics.as_ref().filter(|_| !n.removed) else {
                continue;
            };
            let takes_focus = !s.disabled && s.focusable.unwrap_or(s.role.focusable_by_default());
            if !takes_focus || !self.style(self.id_at(i)).pointer_events {
                continue;
            }
            let id = self.id_at(i);
            let shown = self
                .ancestors(id)
                .all(|a| self.style(a).display != crate::Display::None);
            let hidden = self
                .ancestors(id)
                .any(|a| self.nodes[a.0].semantics.as_ref().is_some_and(|s| s.hidden));
            if shown && hidden {
                out.push(AuditIssue {
                    kind: Audit::HiddenFocusable,
                    id,
                    detail: format!("{:?} {:?}", s.role, s.label),
                });
            }
        }
        let page = self
            .style(self.root())
            .background
            .unwrap_or([1.0, 1.0, 1.0, 1.0]);
        for c in self.contrast_issues(page, false) {
            let hex = |c: [f32; 3]| {
                format!(
                    "#{:02x}{:02x}{:02x}",
                    (c[0] * 255.0).round() as u8,
                    (c[1] * 255.0).round() as u8,
                    (c[2] * 255.0).round() as u8
                )
            };
            out.push(AuditIssue {
                kind: Audit::LowContrast,
                id: c.id,
                detail: format!(
                    "{:?}: {:.2}:1, {} on {} (needs {:.1}:1)",
                    c.text,
                    c.ratio,
                    hex(c.foreground),
                    hex(c.background),
                    c.required
                ),
            });
        }
        out
    }
}
