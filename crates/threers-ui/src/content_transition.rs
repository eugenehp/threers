//! Text that changes with motion, not a cut: [`rolling_text`] rolls each
//! changed character over (digits up when a number grows, down when it
//! shrinks, as an odometer does); [`crossfade_text`] fades from the old
//! text to the new. With reduced motion both only fade.
//!
//! Each is a node of its own that the app gives new text with `set`; it
//! inherits its font from where it is put. Assistive technology hears the
//! text, not the characters in motion.
//!
//! ```
//! use threers_ui::content_transition::rolling_text;
//! use threers_ui::{FixedWidthText, Ui};
//!
//! let mut ui = Ui::new();
//! let root = ui.root();
//! let mut count = rolling_text(&mut ui, root, "Unread", "9");
//! ui.layout(400.0, 100.0, &mut FixedWidthText(7.0));
//! count.set(&mut ui, "10");
//! assert_eq!(count.text(), "10");
//! assert!(ui.needs_frame(), "rolling");
//! ```

use threers_math::easing::Easing;

use crate::animation::{Animation, Property, Value};
use crate::tree::{NodeId, Ui};
use crate::{style, Role, Semantics};

/// Seconds a character takes to roll over or a text to fade.
const DURATION: f32 = 0.32;

/// One character's place: in the flow, a hidden copy of the character it
/// shows (which sizes it), and over it the character leaving and the one
/// arriving.
struct Cell {
    cell: NodeId,
    sizer: NodeId,
    leaving: NodeId,
    arriving: NodeId,
}

/// Text whose characters roll over as it changes. Made by
/// [`rolling_text`].
pub struct RollingText {
    pub node: NodeId,
    cells: Vec<Cell>,
    shown: Vec<Option<char>>,
}

/// Text that rolls each changed character over as it changes (see the
/// [module docs](self)), under `parent`, showing `text`; assistive
/// technology hears it as `label`, live.
pub fn rolling_text(ui: &mut Ui, parent: NodeId, label: &str, text: &str) -> RollingText {
    let node = ui.add_box(parent, style! { display: flex; flex-shrink: 0; });
    let mut sem = Semantics::new(Role::Status).label(label);
    sem.live = true;
    sem.value = Some(text.to_owned());
    ui.set_semantics(node, sem);
    let mut r = RollingText {
        node,
        cells: vec![],
        shown: vec![],
    };
    r.place(ui, &text.chars().map(Some).collect::<Vec<_>>(), 0.0);
    r
}

/// A box holding a text, positioned over the cell's.
fn layer_text(ui: &mut Ui, parent: NodeId, flow: bool) -> NodeId {
    let b = if flow {
        ui.add_box(parent, style! { opacity: 0; })
    } else {
        ui.add_box(parent, style! { position: absolute; left: 0px; top: 0px; })
    };
    ui.add_text(b, "");
    ui.semantics_mut(b).hidden = true;
    b
}

fn text_of(ui: &mut Ui, holder: NodeId, c: Option<char>) {
    let t = ui.children(holder)[0];
    ui.set_text(t, c.map(String::from).unwrap_or_default());
}

impl RollingText {
    /// Shows `text`, rolling each character that changes. Numbers line up
    /// on their last digit and roll up as they grow, down as they shrink.
    pub fn set(&mut self, ui: &mut Ui, text: &str) {
        let old: String = self.text();
        if old == text {
            return;
        }
        ui.semantics_mut(self.node).value = Some(text.to_owned());
        // Which way: a number that grew comes up from below.
        let dir = match (old.trim().parse::<f64>(), text.trim().parse::<f64>()) {
            (Ok(a), Ok(b)) if b < a => -1.0,
            _ => 1.0,
        };
        let numbers = old.trim().parse::<f64>().is_ok() && text.trim().parse::<f64>().is_ok();
        let mut want: Vec<Option<char>> = text.chars().map(Some).collect();
        let n = want.len().max(self.shown.len());
        // Numbers align right (units over units), other text left.
        if numbers {
            let pad = n - want.len();
            want.splice(0..0, std::iter::repeat_n(None, pad));
            let pad = n - self.shown.len();
            self.shown.splice(0..0, std::iter::repeat_n(None, pad));
            for _ in 0..pad {
                let c = self.new_cell(ui);
                ui.move_child(self.node, c.cell, 0);
                self.cells.insert(0, c);
            }
        } else {
            want.resize(n, None);
            self.shown.resize(n, None);
        }
        self.place(ui, &want, dir);
    }

    /// What it shows (or is rolling to).
    pub fn text(&self) -> String {
        self.shown.iter().flatten().collect()
    }

    fn new_cell(&self, ui: &mut Ui) -> Cell {
        let cell = ui.add_box(
            self.node,
            style! { position: relative; overflow: hidden; flex-shrink: 0; },
        );
        let sizer = layer_text(ui, cell, true);
        let leaving = layer_text(ui, cell, false);
        let arriving = layer_text(ui, cell, false);
        Cell {
            cell,
            sizer,
            leaving,
            arriving,
        }
    }

    /// Sets each cell to `want`, rolling those that change by `dir` (0: as
    /// they are made, without motion).
    fn place(&mut self, ui: &mut Ui, want: &[Option<char>], dir: f32) {
        while self.cells.len() < want.len() {
            let c = self.new_cell(ui);
            self.cells.push(c);
        }
        self.shown.resize(want.len(), None);
        let reduce = ui.reduce_motion();
        for (i, &new) in want.iter().enumerate() {
            let old = self.shown[i];
            let c = &self.cells[i];
            text_of(ui, c.sizer, new);
            if old == new && dir != 0.0 {
                continue;
            }
            text_of(ui, c.leaving, old);
            text_of(ui, c.arriving, new);
            let h = ui
                .rect(c.cell)
                .height
                .max(ui.style(c.cell).font_size * 1.2)
                .max(1.0);
            if dir == 0.0 {
                ui.update_style(c.leaving, |s| s.opacity = 0.0);
                continue;
            }
            let ease = Easing::CubicOut;
            let run = |p: Property, a: f32, b: f32| {
                Animation::new(p, Value::Number(a), Value::Number(b), DURATION).easing(ease)
            };
            ui.animate(c.leaving, run(Property::Opacity, 1.0, 0.0));
            ui.animate(c.arriving, run(Property::Opacity, 0.0, 1.0));
            if !reduce {
                ui.animate(c.leaving, run(Property::Top, 0.0, -dir * h));
                ui.animate(c.arriving, run(Property::Top, dir * h, 0.0));
            }
        }
        self.shown = want.to_vec();
    }
}

/// Text that fades from what it showed to what it shows. Made by
/// [`crossfade_text`].
pub struct FadingText {
    pub node: NodeId,
    sizer: NodeId,
    leaving: NodeId,
    arriving: NodeId,
    shown: String,
}

/// Text under `parent` that fades from the old to the new as it changes,
/// showing `text` (see the [module docs](self)).
pub fn crossfade_text(ui: &mut Ui, parent: NodeId, text: &str) -> FadingText {
    let node = ui.add_box(parent, style! { position: relative; flex-shrink: 0; });
    let sizer = ui.add_box(node, style! { opacity: 0; });
    ui.add_text(sizer, text);
    let leaving = ui.add_box(
        node,
        style! { position: absolute; left: 0px; top: 0px; opacity: 0; white-space: nowrap; },
    );
    ui.add_text(leaving, "");
    let arriving = ui.add_box(
        node,
        style! { position: absolute; left: 0px; top: 0px; white-space: nowrap; },
    );
    ui.add_text(arriving, text);
    for n in [sizer, leaving] {
        ui.semantics_mut(n).hidden = true;
    }
    FadingText {
        node,
        sizer,
        leaving,
        arriving,
        shown: text.to_owned(),
    }
}

impl FadingText {
    /// Shows `text`, fading over from what it showed.
    pub fn set(&mut self, ui: &mut Ui, text: &str) {
        if self.shown == text {
            return;
        }
        let old = std::mem::replace(&mut self.shown, text.to_owned());
        for (holder, s) in [
            (self.sizer, text),
            (self.leaving, old.as_str()),
            (self.arriving, text),
        ] {
            let t = ui.children(holder)[0];
            ui.set_text(t, s);
        }
        let run = |a: f32, b: f32| {
            Animation::new(
                Property::Opacity,
                Value::Number(a),
                Value::Number(b),
                DURATION,
            )
            .easing(Easing::SineInOut)
        };
        ui.animate(self.leaving, run(1.0, 0.0));
        ui.animate(self.arriving, run(0.0, 1.0));
    }

    pub fn text(&self) -> &str {
        &self.shown
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::FixedWidthText;

    fn settle(ui: &mut Ui) {
        for _ in 0..60 {
            ui.step(1.0 / 60.0);
        }
        ui.layout(400.0, 100.0, &mut FixedWidthText(7.0));
    }

    fn shown_text(ui: &Ui, holder: NodeId) -> String {
        ui.text_lines_of(ui.children(holder)[0]).join("")
    }

    #[test]
    fn a_growing_number_rolls_its_changed_digits_up_and_lines_up_on_the_units() {
        let mut ui = Ui::new();
        let root = ui.root();
        let mut r = rolling_text(&mut ui, root, "Count", "99");
        ui.layout(400.0, 100.0, &mut FixedWidthText(7.0));
        r.set(&mut ui, "100");
        assert_eq!(r.text(), "100");
        assert_eq!(ui.semantics_mut(r.node).value.as_deref(), Some("100"));
        // Mid-roll: the arriving units digit is below its place, rising.
        ui.step(0.05);
        ui.layout(400.0, 100.0, &mut FixedWidthText(7.0));
        let units = &r.cells[2];
        let (cell, arriving) = (ui.rect(units.cell), ui.rect(units.arriving));
        assert!(
            arriving.y > cell.y,
            "coming up from below: {arriving:?} in {cell:?}"
        );
        assert_eq!(shown_text(&ui, units.arriving), "0");
        settle(&mut ui);
        let units = &r.cells[2];
        assert_eq!(ui.rect(units.arriving).y, ui.rect(units.cell).y);
        // Down when it shrinks.
        r.set(&mut ui, "99");
        ui.step(0.05);
        ui.layout(400.0, 100.0, &mut FixedWidthText(7.0));
        let units = &r.cells[2];
        assert!(
            ui.rect(units.arriving).y < ui.rect(units.cell).y,
            "coming down from above"
        );
        settle(&mut ui);
        assert!(!ui.needs_frame());
        // The hundreds' cell empty, and as narrow as nothing.
        assert_eq!(ui.rect(r.cells[0].cell).width, 0.0);
    }

    #[test]
    fn with_reduced_motion_it_only_fades_and_a_crossfade_fades() {
        let mut ui = Ui::new();
        ui.set_reduce_motion(true);
        let root = ui.root();
        let mut r = rolling_text(&mut ui, root, "Count", "1");
        ui.layout(400.0, 100.0, &mut FixedWidthText(7.0));
        r.set(&mut ui, "2");
        ui.step(0.05);
        ui.layout(400.0, 100.0, &mut FixedWidthText(7.0));
        assert_eq!(ui.rect(r.cells[0].arriving).y, ui.rect(r.cells[0].cell).y);
        let mut f = crossfade_text(&mut ui, root, "Hello");
        ui.layout(400.0, 100.0, &mut FixedWidthText(7.0));
        f.set(&mut ui, "Goodbye");
        ui.step(0.1);
        let (a, b) = (ui.style(f.leaving).opacity, ui.style(f.arriving).opacity);
        assert!(a > 0.0 && a < 1.0 && b > 0.0 && b < 1.0, "{a} {b}");
        settle(&mut ui);
        assert_eq!(
            (ui.style(f.leaving).opacity, ui.style(f.arriving).opacity),
            (0.0, 1.0)
        );
        assert_eq!(f.text(), "Goodbye");
    }
}
