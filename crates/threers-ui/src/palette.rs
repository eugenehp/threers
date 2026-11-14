//! A command palette: a modal list of an app's commands with
//! their shortcut labels, narrowed by fuzzy matching as the user types,
//! worked from the keyboard (up and down, Enter to run, Escape to close)
//! or picked with the pointer. Assistive technology hears a dialog with a
//! `combobox` (the query) over a `listbox` of `option`s, the highlighted
//! one `selected`.
//!
//! ```
//! use threers_ui::input::Input;
//! use threers_ui::palette::{open_palette, Command};
//! use threers_ui::{FixedWidthText, Ui};
//!
//! let mut ui = Ui::new();
//! let mut input: Input<Vec<String>> = Input::new();
//! let mut ran = vec![];
//! let commands = vec![Command::new("save", "Save File").shortcut("⌘S"), Command::new("theme", "Toggle Theme")];
//! open_palette(&mut ui, &mut input, &mut ran, commands, |ran, _, id| ran.push(id.to_owned()));
//! ui.layout(600.0, 400.0, &mut FixedWidthText(7.0));
//! input.simulate_keystrokes(&mut ui, &mut ran, "t t enter").unwrap();
//! assert_eq!(ran, ["theme"]);
//! ```

use std::cell::RefCell;
use std::rc::Rc;

use crate::environment::Token;
use crate::input::Input;
use crate::tree::{NodeId, Ui};
use crate::{style, Display, Role, Semantics};

/// A command the palette offers.
#[derive(Debug, Clone, PartialEq)]
pub struct Command {
    pub id: String,
    pub label: String,
    /// Its key binding's label (`⌘S`), shown at the end.
    pub shortcut: Option<String>,
}

impl Command {
    pub fn new(id: impl Into<String>, label: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            label: label.into(),
            shortcut: None,
        }
    }

    pub fn shortcut(mut self, label: impl Into<String>) -> Self {
        self.shortcut = Some(label.into());
        self
    }
}

/// How well `query` matches `label`, its letters in order (case aside);
/// `None` if they are not all there. Letters at the start of words and
/// letters in a row score more, so `ts` finds "Toggle Sidebar" first.
pub fn fuzzy_score(query: &str, label: &str) -> Option<i32> {
    let label: Vec<char> = label.chars().collect();
    let mut score = 0;
    let mut at = 0;
    let mut last: Option<usize> = None;
    for q in query.chars().filter(|c| !c.is_whitespace()) {
        let q = q.to_lowercase().next().unwrap_or(q);
        let found = (at..label.len()).find(|&i| label[i].to_lowercase().next() == Some(q))?;
        score += 1;
        if found == 0 || !label[found - 1].is_alphanumeric() {
            score += 8;
        }
        if last == Some(found.wrapping_sub(1)) {
            score += 5;
        }
        score -= (found - at) as i32 / 4;
        last = Some(found);
        at = found + 1;
    }
    Some(score)
}

/// Sizes and centres `panel` (an absolutely placed box under the root) in
/// the window as last laid out: `width` wide, or the window less a margin
/// if that is narrower; `top` of the window's height down; no taller than
/// the rest of the window. Through its base style (it has state styles).
pub(crate) fn place_in_window(ui: &mut Ui, panel: NodeId, width: f32, top: f32) {
    const MARGIN: f32 = 8.0;
    let window = ui.rect(ui.root());
    let w = width.min(window.width - 2.0 * MARGIN).max(0.0);
    let (left, top) = (
        ((window.width - w) / 2.0).max(MARGIN),
        (window.height * top).max(MARGIN),
    );
    let max_h = (window.height - top - MARGIN).max(0.0);
    let s = ui.style(panel);
    let want = (
        crate::Length::Px(w),
        crate::Length::Px(left),
        crate::Length::Px(top),
        crate::Length::Px(max_h),
    );
    if (s.width, s.left, s.top, s.max_height) != want {
        ui.update_style(panel, |s| (s.width, s.left, s.top, s.max_height) = want);
    }
}

/// Rows shown at once.
const ROWS: usize = 8;

struct Palette {
    commands: Vec<Command>,
    query: String,
    matches: Vec<usize>,
    highlighted: usize,
    query_text: NodeId,
    rows: Vec<(NodeId, NodeId, NodeId)>,
    closing: bool,
}

impl Palette {
    fn refilter(&mut self) {
        let mut scored: Vec<(i32, usize)> = self
            .commands
            .iter()
            .enumerate()
            .filter_map(|(i, c)| fuzzy_score(&self.query, &c.label).map(|s| (s, i)))
            .collect();
        scored.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
        self.matches = scored.into_iter().map(|(_, i)| i).collect();
        self.highlighted = 0;
    }

    fn show(&self, ui: &mut Ui) {
        let shown = if self.query.is_empty() {
            "Type a command…".to_owned()
        } else {
            self.query.clone()
        };
        ui.set_text(self.query_text, shown);
        for (k, &(row, label, keys)) in self.rows.iter().enumerate() {
            match self.matches.get(k).map(|&i| &self.commands[i]) {
                Some(c) => {
                    ui.update_style(row, |s| s.display = Display::Flex);
                    ui.set_text(label, c.label.clone());
                    ui.set_text(keys, c.shortcut.clone().unwrap_or_default());
                    let on = k == self.highlighted;
                    ui.style_mut(row).background = on.then(|| ui.token(Token::Accent));
                    ui.style_mut(row).color = if on {
                        ui.token(Token::OnAccent)
                    } else {
                        [1.0; 4]
                    };
                    // The shortcut, readable on the highlight.
                    if let Some(k) = ui.parent(keys) {
                        ui.token_style(
                            k,
                            &[(
                                "color",
                                if on {
                                    Token::OnAccent
                                } else {
                                    Token::SecondaryText
                                },
                            )],
                        );
                    }
                    let mut sem = Semantics::new(Role::Option)
                        .label(c.label.clone())
                        .selected(on);
                    sem.description = c.shortcut.clone();
                    ui.set_semantics(row, sem);
                }
                None => ui.update_style(row, |s| s.display = Display::None),
            }
        }
    }
}

/// Opens the palette over the UI with `commands`; `on_run` gets the id of
/// the one run. It closes then, on Escape, and on a click outside it.
pub fn open_palette<S: 'static>(
    ui: &mut Ui,
    input: &mut Input<S>,
    state: &mut S,
    commands: Vec<Command>,
    on_run: impl FnMut(&mut S, &mut Ui, &str) + 'static,
) -> NodeId {
    let root = ui.root();
    let panel = ui.add_box(
        root,
        style! {
            position: absolute; display: flex; flex-direction: column; width: 440px; padding: 6px; gap: 2px; box-sizing: border-box;
            border-radius: 12px; background: #1f2430; border: 1px solid #3a4150; color: #e5e7eb;
            box-shadow: 0 18px 44px rgba(0,0,0,0.55); font-size: 13px;
        },
    );
    ui.token_style(
        panel,
        &[
            ("background", Token::Raised),
            ("border-color", Token::Border),
            ("color", Token::Text),
        ],
    );
    place_in_window(ui, panel, 440.0, 0.12);
    ui.set_semantics(panel, Semantics::new(Role::Dialog).label("Command palette"));
    ui.defer(panel, 960);
    ui.appear(panel, crate::transitions::Transition::Fade, 0.12);
    let field = ui.add_box(
        panel,
        style! { padding: 8px 10px; border-bottom: 1px solid #3a4150; },
    );
    ui.token_style(
        field,
        &[
            ("border-color", Token::Separator),
            ("color", Token::SecondaryText),
        ],
    );
    let query_text = ui.add_text(field, "");
    let mut sem = Semantics::new(Role::ComboBox)
        .label("Command")
        .tab_index(-1);
    sem.expanded = Some(true);
    ui.set_semantics(field, sem);
    // Scrolls when the window is too short for every row.
    let list = ui.add_box(panel, style! { display: flex; flex-direction: column; min-height: 0; flex-shrink: 1; overflow: auto; });
    ui.set_semantics(list, Semantics::new(Role::ListBox).label("Commands"));
    let rows: Vec<(NodeId, NodeId, NodeId)> = (0..ROWS)
        .map(|_| {
            let r = ui.add_box(list, style! { display: flex; justify-content: space-between; gap: 16px; padding: 7px 10px; border-radius: 6px; cursor: pointer; });
            // A long name gives way (cut short), not the shortcut.
            let lb = ui.add_box(r, style! { min-width: 0; flex-shrink: 1; overflow: hidden; white-space: nowrap; text-overflow: ellipsis; });
            let l = ui.add_text(lb, "");
            let k = ui.add_box(r, style! { flex-shrink: 0; white-space: nowrap; });
            ui.token_style(k, &[("color", Token::SecondaryText)]);
            let kt = ui.add_text(k, "");
            (r, l, kt)
        })
        .collect();
    let mut p = Palette {
        commands,
        query: String::new(),
        matches: vec![],
        highlighted: 0,
        query_text,
        rows: rows.clone(),
        closing: false,
    };
    p.refilter();
    p.show(ui);
    let p = Rc::new(RefCell::new(p));
    let run = Rc::new(RefCell::new(on_run));
    let pick = {
        let (p, run) = (p.clone(), run.clone());
        Rc::new(move |s: &mut S, ui: &mut Ui, k: usize| {
            let id = {
                let b = p.borrow();
                b.matches.get(k).map(|&i| b.commands[i].id.clone())
            };
            if let Some(id) = id {
                p.borrow_mut().closing = true;
                (run.borrow_mut())(s, ui, &id);
            }
        })
    };
    for (k, &(row, _, _)) in rows.iter().enumerate() {
        let pk = pick.clone();
        input.on_click(row, move |s, ui, _| pk(s, ui, k));
    }
    let (pp, pk) = (p.clone(), pick.clone());
    input.on_key(field, move |s, ui, key, m, _| {
        let mut b = pp.borrow_mut();
        match key {
            "Escape" => b.closing = true,
            "Enter" => {
                let k = b.highlighted;
                drop(b);
                pk(s, ui, k);
                return true;
            }
            "ArrowDown" => {
                b.highlighted = (b.highlighted + 1).min(b.matches.len().min(ROWS).saturating_sub(1))
            }
            "ArrowUp" => b.highlighted = b.highlighted.saturating_sub(1),
            "Backspace" => {
                b.query.pop();
                b.refilter();
            }
            k if k.chars().count() == 1 && !m.control && !m.meta => {
                b.query.push_str(k);
                b.refilter();
            }
            _ => return false,
        }
        b.show(ui);
        true
    });
    let pp = p.clone();
    input.on_click_outside(panel, move |_, _, _, _| pp.borrow_mut().closing = true);
    let before = input.focused();
    input.set_modal(ui, state, Some(panel));
    input.set_focus(ui, state, Some(field));
    input.on_after_dispatch(move |input, ui, state| {
        // Closed, or removed from outside (see `Input::present`).
        if ui.contains(panel) && !p.borrow().closing {
            place_in_window(ui, panel, 440.0, 0.12);
            return true;
        }
        if input.modal().is_none_or(|m| m == panel || !ui.contains(m)) {
            input.set_modal(ui, state, None);
        }
        ui.remove(panel);
        input.set_focus(ui, state, before.filter(|&b| ui.contains(b)));
        false
    });
    panel
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fuzzy_matching_prefers_word_starts_and_runs() {
        assert!(fuzzy_score("xyz", "Save").is_none());
        let ts_sidebar = fuzzy_score("ts", "Toggle Sidebar").unwrap();
        let ts_settings = fuzzy_score("ts", "Open Settings").unwrap();
        assert!(ts_sidebar > ts_settings, "{ts_sidebar} {ts_settings}");
        assert!(fuzzy_score("fil", "File").unwrap() > fuzzy_score("fil", "Fit Inline").unwrap());
        assert!(fuzzy_score("", "Anything").is_some());
    }
}
