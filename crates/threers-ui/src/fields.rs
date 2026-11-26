//! Fields beyond a line of text: a [`password_field`], a [`tag_input`] and a
//! [`date_picker`]. Each shows its caret or opens its calendar itself, and
//! is bound to the state (see [`binding`](mod@crate::binding)).
//!
//! ```
//! use threers_ui::binding::Binding;
//! use threers_ui::fields::tag_input;
//! use threers_ui::input::Input;
//! use threers_ui::{FixedWidthText, Ui};
//!
//! let mut ui = Ui::new();
//! let mut input: Input<Vec<String>> = Input::new();
//! let mut tags = vec![];
//! let root = ui.root();
//! let all = Binding::new(|s: &Vec<String>| s.clone(), |s: &mut Vec<String>, v| *s = v);
//! let f = tag_input(&mut ui, &mut input, &tags, root, "Tags", "Remove {tag}", all);
//! ui.layout(300.0, 100.0, &mut FixedWidthText(7.0));
//! input.set_focus(&mut ui, &mut tags, Some(f));
//! input.simulate_keystrokes(&mut ui, &mut tags, "a b enter c enter backspace").unwrap();
//! assert_eq!(tags, ["ab"]);
//! ```

use std::cell::{Cell, RefCell};
use std::rc::Rc;

use crate::binding::Binding;
use crate::components::{popover, PopoverHandle};
use crate::environment::Token;
use crate::input::{Event, Input, Phase};
use crate::text_field::{FieldKey, KeyModifiers, TextField};
use crate::text_shortcuts::{text_command, TextCommand};
use crate::tree::{NodeId, Ui};
use crate::view::{calendar_picking, CalendarWords, Day, View};
use crate::{style, Display, Role, Semantics};

/// What stands for each character of a password, to assistive technology:
/// on screen it is a drawn dot, which no font can lack.
const BULLET: char = '•';

/// A line of text with its selection and caret, or a placeholder.
#[derive(Clone, Copy)]
struct Line {
    row: NodeId,
    texts: [NodeId; 3],
    selected: NodeId,
    caret: NodeId,
}

impl Line {
    fn new(ui: &mut Ui, parent: NodeId) -> Line {
        // Scrolls to its caret; not page text to select.
        let row = ui.add_box(
            parent,
            style! { display: flex; flex-direction: row; align-items: center; flex-grow: 1; min-width: 0; min-height: 18px; overflow-x: auto; scrollbar-width: none; white-space: pre; user-select: none; font-size: 13px; },
        );
        let part = |ui: &mut Ui| {
            let b = ui.add_box(
                row,
                style! { flex-shrink: 0; white-space: pre; font-size: 13px; },
            );
            (b, ui.add_text(b, ""))
        };
        let (_, before) = part(ui);
        let (selected, inside) = part(ui);
        let caret = ui.add_box(
            row,
            style! { width: 1px; height: 16px; flex-shrink: 0; display: none; },
        );
        ui.token_style(caret, &[("background", Token::Text)]);
        let (_, after) = part(ui);
        // Read out as the field's value, not as three texts.
        ui.set_semantics(row, Semantics::new(Role::Generic).hidden());
        Line {
            row,
            texts: [before, inside, after],
            selected,
            caret,
        }
    }

    /// `text` with `anchor..caret` selected while `focused`; `placeholder`
    /// when there is none. `masked`, a dot for each character instead.
    fn show(
        &self,
        ui: &mut Ui,
        text: &str,
        masked: bool,
        (anchor, caret): (usize, usize),
        focused: bool,
        placeholder: &str,
    ) {
        let chars: Vec<char> = text.chars().collect();
        let n = chars.len();
        let (a, z) = (anchor.min(caret).min(n), anchor.max(caret).min(n));
        let mut parts = [&chars[..a], &chars[a..z], &chars[z..]].map(String::from_iter);
        let empty = n == 0;
        let dots = if masked { [a, z - a, n - z] } else { [0; 3] };
        if masked {
            parts = Default::default();
        }
        if empty {
            parts[2] = placeholder.to_owned();
        }
        for (t, s) in self.texts.iter().zip(parts) {
            if ui.text_content(*t) != s {
                ui.set_text(*t, s);
            }
        }
        let ink = if empty { Token::SecondaryText } else { Token::Text };
        ui.token_style(self.row, &[("color", ink)]);
        for (&t, dots) in self.texts.iter().zip(dots) {
            if let Some(b) = ui.parent(t) {
                ui.token_style(b, &[("color", ink)]);
                // After the text, which is empty while there are any.
                while ui.children(b).len() > dots + 1 {
                    let last = ui.children(b)[dots + 1];
                    ui.remove(last);
                }
                while ui.children(b).len() < dots + 1 {
                    dot(ui, b);
                }
            }
        }
        if focused && a != z {
            ui.token_style(
                self.selected,
                &[("background", Token::Accent), ("color", Token::OnAccent)],
            );
        } else {
            ui.clear_token_style(self.selected, "background");
        }
        let on = if focused { Token::OnAccent } else { Token::Text };
        for cell in ui.children(self.selected).iter().skip(1).copied().collect::<Vec<_>>() {
            let d = ui.children(cell)[0];
            ui.token_style(d, &[("background", on)]);
        }
        let d = if focused && a == z { Display::Flex } else { Display::None };
        ui.update_style(self.caret, |s| s.display = d);
        if focused {
            // (where the text just set puts it: after layout)
            ui.reveal_x_after_layout(self.caret);
        }
    }
}

/// A character of a password, drawn in `part`.
fn dot(ui: &mut Ui, part: NodeId) {
    ui.update_style(part, |s| {
        let row = style! { display: flex; flex-direction: row; };
        (s.display, s.flex_direction) = (row.display, row.flex_direction);
    });
    let cell = ui.add_box(
        part,
        style! { display: flex; align-items: center; justify-content: center; width: 8px; height: 18px; flex-shrink: 0; },
    );
    let d = ui.add_box(
        cell,
        style! { width: 6px; height: 6px; border-radius: 3px; flex-shrink: 0; },
    );
    ui.token_style(d, &[("background", Token::Text)]);
}

/// The box a field's line goes in, its ring while it has focus.
fn field_box(ui: &mut Ui, parent: NodeId) -> NodeId {
    let b = ui.add_box(
        parent,
        style! { display: flex; flex-direction: row; align-items: center; flex-grow: 1; min-width: 0; min-height: 30px; box-sizing: border-box; padding: 5px 8px; border: 1px solid; border-radius: 7px; cursor: text; },
    );
    // An edge that can be seen: the field is all edge (WCAG 1.4.11).
    ui.token_style(
        b,
        &[("background", Token::Surface), ("border-color", Token::Track)],
    );
    ui.focus_style(b, "box-shadow: 0 0 0 2px var(--focus)");
    b
}

/// `key` in a field whose text is hidden: nothing that copies it out, and
/// nothing that tells where its words are.
fn hidden_key(f: &mut TextField, key: &str, m: KeyModifiers) -> FieldKey {
    use TextCommand as T;
    let Some(c) = text_command(key, m) else {
        return f.key(key, m, |c| !c.is_control());
    };
    let c = match c {
        // Taken, and nothing done: not the page's to copy either.
        T::Copy | T::Cut => return FieldKey::Moved,
        T::WordLeft { extend } => T::TextStart { extend },
        T::WordRight { extend } => T::TextEnd { extend },
        T::DeleteWordBack => T::DeleteToLineStart,
        T::DeleteWordForward => T::DeleteToLineEnd,
        c => c,
    };
    f.command(c, |c| !c.is_control())
}

/// A password, bound to `text`: a bullet for each character until the eye
/// beside it, named `names` (to show, to hide), is pressed. Hidden, it
/// cannot be copied or cut, its words cannot be told apart, and assistive
/// technology is given the bullets. `label` names it and shows while it is
/// empty. A host mirroring it knows to keep it secret by
/// [`Input::focused_text_is_secret`]. Returns the field, the node to focus.
#[allow(clippy::too_many_arguments)]
pub fn password_field<S: 'static>(
    ui: &mut Ui,
    input: &mut Input<S>,
    state: &S,
    parent: NodeId,
    label: &str,
    names: (&str, &str),
    text: Binding<S, String>,
) -> NodeId {
    let row = ui.add_box(
        parent,
        style! { display: flex; flex-direction: row; align-items: center; gap: 4px; align-self: stretch; min-width: 0; },
    );
    let b = field_box(ui, row);
    let line = Line::new(ui, b);
    ui.set_semantics(b, Semantics::new(Role::TextBox).label(label).secret());
    // Beside the field, not in it: its Space is not the field's.
    let eye = ui.add_box(
        row,
        style! { display: flex; align-items: center; justify-content: center; width: 28px; height: 28px; border-radius: 7px; cursor: pointer; flex-shrink: 0; },
    );
    ui.hover_style(eye, "background: var(--hover)");
    const EYE: &str =
        r#"<path d="M2 12s4-7 10-7 10 7 10 7-4 7-10 7S2 12 2 12z"/><circle cx="12" cy="12" r="3"/>"#;
    let [open, shut] = [EYE.to_owned(), format!(r#"{EYE}<path d="M4 20 20 4"/>"#)].map(|m| {
        let i = crate::svg::add_icon(ui, eye, style! { width: 16px; height: 16px; }, &m, 2.0);
        ui.token_style(i, &[("color", Token::SecondaryText)]);
        i
    });
    let names = [names.0.to_owned(), names.1.to_owned()];
    ui.set_semantics(eye, Semantics::new(Role::Button).label(names[0].clone()));

    let field = Rc::new(RefCell::new(TextField::new(text.get(state))));
    let shown = Rc::new(Cell::new(false));
    let r = shown.clone();
    input.on_click(eye, move |_, _, _| r.set(!r.get()));
    let (f, t, r) = (field.clone(), text.clone(), shown.clone());
    input.on_key(b, move |s, _, key, m, _| {
        // Enter is the form's, and Tab goes on.
        if matches!(key, "Enter" | "Tab" | "Escape") {
            return false;
        }
        let mut field = f.borrow_mut();
        let now = t.get(s);
        if field.text != now {
            *field = TextField::new(&now);
        }
        let did = if r.get() {
            field.key(key, m, |c| !c.is_control())
        } else {
            hidden_key(&mut field, key, m)
        };
        if field.text != now {
            t.set(s, field.text.clone());
        }
        did != FieldKey::Ignored
    });
    register(input, b, field.clone(), text.clone());
    let label = label.to_owned();
    let mut last = None;
    let mut show = move |ui: &mut Ui, s: &S, focused: bool| {
        let now = text.get(s);
        let mut f = field.borrow_mut();
        if f.text != now {
            // Changed by the app.
            *f = TextField::new(&now);
        }
        let state = (now, f.anchor, f.caret, focused, shown.get());
        if last.as_ref() == Some(&state) {
            return;
        }
        let (text, anchor, caret, focused, seen) = &state;
        let text = if *seen {
            text.clone()
        } else {
            text.chars().map(|_| BULLET).collect()
        };
        line.show(ui, &text, !seen, (*anchor, *caret), *focused, &label);
        for (icon, on) in [(open, !seen), (shut, *seen)] {
            ui.update_style(icon, |s| {
                s.display = if on { Display::Flex } else { Display::None }
            });
        }
        ui.semantics_mut(eye).label = Some(names[usize::from(*seen)].clone());
        ui.semantics_mut(b).value = Some(text);
        last = Some(state);
    };
    show(ui, state, false);
    input.on_after_dispatch(move |input, ui, s| {
        if !ui.contains(b) {
            return false;
        }
        show(ui, s, input.focused() == Some(b));
        true
    });
    b
}

/// A line of text to edit, bound to `text`: its caret drawn, what is
/// selected marked, and scrolled to keep the caret in view (a line longer
/// than the field). `label` names it and shows while it is empty. Enter,
/// Tab and Escape are left to what is round it (a form, `on_submit`).
/// Returns the field, the node to focus.
pub fn line_field<S: 'static>(
    ui: &mut Ui,
    input: &mut Input<S>,
    state: &S,
    parent: NodeId,
    label: &str,
    text: Binding<S, String>,
) -> NodeId {
    let b = field_box(ui, parent);
    let line = Line::new(ui, b);
    ui.set_semantics(b, Semantics::new(Role::TextBox).label(label));
    let field = Rc::new(RefCell::new(TextField::new(text.get(state))));
    let (f, t) = (field.clone(), text.clone());
    input.on_key(b, move |s, _, key, m, _| {
        if matches!(key, "Enter" | "Tab" | "Escape") {
            return false;
        }
        let mut field = f.borrow_mut();
        let now = t.get(s);
        if field.text != now {
            *field = TextField::new(&now);
        }
        let did = field.key(key, m, |c| !c.is_control());
        if field.text != now {
            t.set(s, field.text.clone());
        }
        did != FieldKey::Ignored
    });
    register(input, b, field.clone(), text.clone());
    let label = label.to_owned();
    let mut last = None;
    let mut show = move |ui: &mut Ui, s: &S, focused: bool| {
        let now = text.get(s);
        let mut f = field.borrow_mut();
        if f.text != now {
            // Changed by the app: the caret at its end.
            *f = TextField::new(&now);
        }
        let state = (now, f.anchor, f.caret, focused);
        if last.as_ref() == Some(&state) {
            return;
        }
        let (now, anchor, caret, focused) = &state;
        line.show(ui, now, false, (*anchor, *caret), *focused, &label);
        ui.semantics_mut(b).value = Some(now.clone());
        last = Some(state);
    };
    show(ui, state, false);
    input.on_after_dispatch(move |input, ui, s| {
        if !ui.contains(b) {
            return false;
        }
        show(ui, s, input.focused() == Some(b));
        true
    });
    b
}

/// A host's own text input (a phone's keyboard) edits `text` too.
fn register<S: 'static>(
    input: &mut Input<S>,
    node: NodeId,
    field: Rc<RefCell<TextField>>,
    text: Binding<S, String>,
) {
    let (f, t) = (field.clone(), text.clone());
    input.register_text(
        node,
        move |s| {
            let (f, now) = (f.borrow(), t.get(s));
            if f.text == now {
                (now, f.anchor, f.caret)
            } else {
                let n = now.chars().count();
                (now, n, n)
            }
        },
        move |s, _, new, anchor, caret| {
            // (a line: what is pasted over several is one)
            let new: String = new.chars().filter(|c| !c.is_control()).collect();
            let mut f = field.borrow_mut();
            *f = TextField::new(&new);
            let n = new.chars().count();
            (f.anchor, f.caret) = (anchor.min(n), caret.min(n));
            text.set(s, new);
        },
    );
}

/// `tags` with the words of `typed` added: those between commas and line
/// breaks, trimmed, none empty and none there already.
fn with_tags(mut tags: Vec<String>, typed: &str) -> Vec<String> {
    for t in typed.split([',', '\n', '\r', '、', '،', '，']) {
        let t: String = t.trim().chars().filter(|c| !c.is_control()).collect();
        if !t.is_empty() && !tags.contains(&t) {
            tags.push(t);
        }
    }
    tags
}

/// Words kept as chips, bound to `tags`: Enter or a comma keeps what is
/// typed, Backspace with nothing typed removes the last, and each has a ✕
/// named `remove` (`"Remove {tag}"`). What is there already is not added
/// again; a list pasted is added word by word. `label` names it and shows
/// while it is empty. Returns the field, the node to focus.
#[allow(clippy::too_many_arguments)]
pub fn tag_input<S: 'static>(
    ui: &mut Ui,
    input: &mut Input<S>,
    state: &S,
    parent: NodeId,
    label: &str,
    remove: &str,
    tags: Binding<S, Vec<String>>,
) -> NodeId {
    let row = ui.add_box(
        parent,
        style! { display: flex; flex-direction: row; flex-wrap: wrap; align-items: center; gap: 6px; align-self: stretch; min-width: 0; },
    );
    let chips = ui.add_box(
        row,
        style! { display: flex; flex-direction: row; flex-wrap: wrap; align-items: center; gap: 6px; min-width: 0; },
    );
    ui.set_semantics(chips, Semantics::new(Role::List).label(label));
    let b = field_box(ui, row);
    ui.update_style(b, |s| s.min_width = crate::Length::Px(120.0));
    let line = Line::new(ui, b);
    ui.set_semantics(b, Semantics::new(Role::TextBox).label(label));
    let field = Rc::new(RefCell::new(TextField::default()));
    // What is typed, taken as tags as far as its last comma.
    let keep = {
        let tags = tags.clone();
        Rc::new(move |s: &mut S, f: &mut TextField, all: bool| {
            let is_end = |c: char| [',', '\n', '\r', '、', '،', '，'].contains(&c);
            let upto = if all {
                f.text.len()
            } else {
                match f.text.rfind(is_end) {
                    Some(i) => i + f.text[i..].chars().next().map_or(0, char::len_utf8),
                    None => return,
                }
            };
            let was = tags.get(s);
            let now = with_tags(was.clone(), &f.text[..upto]);
            if now != was {
                tags.set(s, now);
            }
            let rest = f.text[upto..].trim_start().to_owned();
            *f = TextField::new(rest);
        })
    };
    let (f, k, t) = (field.clone(), keep.clone(), tags.clone());
    input.on_key(b, move |s, _, key, m, _| {
        let mut f = f.borrow_mut();
        match key {
            "Tab" | "Escape" => return false,
            "Enter" if f.text.trim().is_empty() => return false,
            "Enter" => {
                k(s, &mut f, true);
                return true;
            }
            "Backspace" if f.is_empty() => {
                let mut all = t.get(s);
                if all.pop().is_some() {
                    t.set(s, all);
                }
                return true;
            }
            _ => {}
        }
        let did = f.key(key, m, |_| true);
        if did == FieldKey::Edited {
            k(s, &mut f, false);
        }
        did != FieldKey::Ignored
    });
    let (f, f2, k) = (field.clone(), field.clone(), keep);
    input.register_text(
        b,
        move |_| {
            let f = f.borrow();
            (f.text.clone(), f.anchor, f.caret)
        },
        move |s, _, new, anchor, caret| {
            let mut f = f2.borrow_mut();
            let n = new.chars().count();
            *f = TextField::new(new);
            (f.anchor, f.caret) = (anchor.min(n), caret.min(n));
            k(s, &mut f, false);
        },
    );
    let (label, remove) = (label.to_owned(), remove.to_owned());
    let mut last_tags: Option<Vec<String>> = None;
    let mut last = None;
    let mut show = move |ui: &mut Ui, input: &mut Input<S>, s: &S, focused: bool| {
        let now = tags.get(s);
        if last_tags.as_ref() != Some(&now) {
            ui.clear_children(chips);
            let d = if now.is_empty() { Display::None } else { Display::Flex };
            ui.update_style(chips, |s| s.display = d);
            for (i, tag) in now.iter().enumerate() {
                let chip = ui.add_box(
                    chips,
                    style! { display: flex; flex-direction: row; align-items: center; gap: 2px; padding: 0 2px 0 10px; border: 1px solid; border-radius: 13px; max-width: 100%; },
                );
                ui.token_style(
                    chip,
                    &[("border-color", Token::Border), ("color", Token::Text)],
                );
                let mut sem = Semantics::new(Role::ListItem).label(tag.clone());
                sem.position = Some((i + 1, now.len()));
                ui.set_semantics(chip, sem);
                let words = ui.add_box(
                    chip,
                    style! { font-size: 12px; min-width: 0; overflow: hidden; white-space: nowrap; text-overflow: ellipsis; },
                );
                ui.add_text(words, tag.clone());
                let x = ui.add_box(
                    chip,
                    style! { display: flex; align-items: center; justify-content: center; width: 24px; height: 24px; border-radius: 12px; cursor: pointer; flex-shrink: 0; },
                );
                ui.hover_style(x, "background: var(--hover)");
                let cross = crate::svg::add_icon(
                    ui,
                    x,
                    style! { width: 12px; height: 12px; },
                    r#"<path d="M18 6 6 18M6 6l12 12"/>"#,
                    2.0,
                );
                ui.token_style(cross, &[("color", Token::SecondaryText)]);
                let name = remove.replace("{tag}", tag);
                ui.set_semantics(x, Semantics::new(Role::Button).label(name));
                let (tags, tag) = (tags.clone(), tag.clone());
                input.on_click(x, move |s, _, cx| {
                    let mut all = tags.get(s);
                    all.retain(|t| *t != tag);
                    tags.set(s, all);
                    // Its button goes: on to the field, not nowhere.
                    cx.focus(b);
                });
            }
            last_tags = Some(now);
        }
        let f = field.borrow();
        let state = (f.text.clone(), f.anchor, f.caret, focused);
        if last.as_ref() != Some(&state) {
            line.show(ui, &f.text, false, (f.anchor, f.caret), focused, &label);
            ui.semantics_mut(b).value = Some(f.text.clone());
            last = Some(state);
        }
    };
    show(ui, input, state, false);
    input.on_after_dispatch(move |input, ui, s| {
        if !ui.contains(b) {
            return false;
        }
        let focused = input.focused() == Some(b);
        show(ui, input, s, focused);
        true
    });
    b
}

/// A day, bound to `day`: a button showing it as `words` write it
/// (`placeholder` while there is none), which opens a
/// [`calendar`](crate::view::calendar) under itself. The arrow keys move the
/// day; a click on one, or Enter, takes it and closes; Escape or a press
/// outside closes and puts back the day there was. Returns the button.
#[allow(clippy::too_many_arguments)]
pub fn date_picker<S: 'static>(
    ui: &mut Ui,
    input: &mut Input<S>,
    state: &S,
    parent: NodeId,
    label: &str,
    placeholder: &str,
    day: Binding<S, Option<Day>>,
    today: Day,
    words: CalendarWords,
) -> NodeId {
    let b = ui.add_box(
        parent,
        style! { display: flex; flex-direction: row; align-items: center; gap: 8px; min-height: 30px; box-sizing: border-box; padding: 5px 10px; border: 1px solid; border-radius: 8px; cursor: pointer; max-width: 100%; },
    );
    ui.token_style(
        b,
        &[("border-color", Token::Track), ("color", Token::Text)],
    );
    ui.hover_style(b, "background: var(--hover)");
    let icon = crate::svg::add_icon(
        ui,
        b,
        style! { width: 14px; height: 14px; flex-shrink: 0; },
        r#"<path d="M4 6h16v15H4zM4 11h16M8 3v5M16 3v5"/>"#,
        2.0,
    );
    ui.token_style(icon, &[("color", Token::SecondaryText)]);
    let t = ui.add_box(
        b,
        style! { font-size: 13px; min-width: 0; overflow: hidden; white-space: nowrap; text-overflow: ellipsis; },
    );
    let t = ui.add_text(t, "");
    ui.set_semantics(
        b,
        Semantics::new(Role::ComboBox).label(label).expanded(false),
    );

    /// What is open: the panel, its view, what it last showed, and the day
    /// there was before.
    struct Open<S> {
        panel: PopoverHandle,
        view: View<S>,
        drawn: Option<(Option<Day>, Day)>,
        before: Option<Day>,
        /// The days, to focus once they are there.
        fresh: bool,
    }
    let open: Rc<RefCell<Option<Open<S>>>> = Rc::default();
    let wanted = Rc::new(Cell::new(false));
    // Set by taking a day; unset, closing puts back the one before.
    let taken = Rc::new(Cell::new(false));
    let month = Rc::new(Cell::new(today));
    let (o, w) = (open.clone(), wanted.clone());
    input.on_click(b, move |_, _, _| match &*o.borrow() {
        Some(p) => p.panel.close(),
        None => w.set(true),
    });
    let (o, w) = (open.clone(), wanted.clone());
    input.on_key(b, move |_, _, key, m, _| {
        let opens = o.borrow().is_none() && crate::menu::opens_combo_box(key, m);
        if opens {
            w.set(true);
        }
        opens
    });
    let (label, placeholder) = (label.to_owned(), placeholder.to_owned());
    let (d, words2) = (day.clone(), words.clone());
    let mut last = None;
    let mut show = move |ui: &mut Ui, s: &S, is_open: bool| {
        let now = (d.get(s), is_open);
        if last != Some(now) {
            let said = now.0.map(|d| words2.day(d));
            ui.set_text(t, said.clone().unwrap_or_else(|| placeholder.clone()));
            let sem = ui.semantics_mut(b);
            sem.value = said;
            sem.expanded = Some(is_open);
            last = Some(now);
        }
    };
    show(ui, state, false);
    input.on_after_dispatch(move |input, ui, s| {
        let here = ui.contains(b);
        let mut o = open.borrow_mut();
        if wanted.take() && here && o.is_none() {
            let before = day.get(s);
            month.set(before.unwrap_or(today));
            taken.set(false);
            let mut view = None;
            let panel = popover(ui, input, s, b, &label, |ui, p| {
                view = Some(View::new(
                    ui,
                    p,
                    "display: flex; flex-direction: column",
                ));
            });
            // Enter takes the day the keys came to.
            let (done, t) = (panel.clone(), taken.clone());
            input.on(panel.panel, Phase::Bubble, move |_, _, e, cx| {
                if let Event::Key { key, .. } = e
                    && key == "Enter"
                {
                    t.set(true);
                    done.close();
                    cx.stop_propagation();
                    cx.prevent_default();
                }
            });
            *o = view.map(|view| Open {
                panel,
                view,
                drawn: None,
                before,
                fresh: true,
            });
        }
        if let Some(p) = o.as_mut() {
            if !here {
                p.panel.close();
            }
            if p.panel.is_closing(ui) {
                // Closed without taking one: as it was.
                if here && !taken.get() && day.get(s) != p.before {
                    day.set(s, p.before);
                }
                *o = None;
            } else {
                let now = (day.get(s), month.get());
                if p.drawn != Some(now) {
                    let (d, m) = (day.clone(), month.clone());
                    let (d2, t, done) = (day.clone(), taken.clone(), p.panel.clone());
                    let m2 = month.clone();
                    let el = calendar_picking(
                        now.1,
                        now.0,
                        Some(today),
                        &words,
                        move |s: &mut S, to| {
                            m.set(to);
                            d.set(s, Some(to));
                        },
                        move |s: &mut S, to| {
                            d2.set(s, Some(to));
                            t.set(true);
                            done.close();
                        },
                        move |_: &mut S, to| m2.set(to),
                    );
                    p.view.update_with(ui, input, s, vec![el]);
                    p.drawn = Some(now);
                }
                if std::mem::take(&mut p.fresh) {
                    let days = ui
                        .children(p.view.root())
                        .first()
                        .and_then(|&c| ui.children(c).last().copied());
                    if let Some(days) = days {
                        input.set_focus(ui, s, Some(days));
                    }
                }
            }
        }
        if here {
            show(ui, s, o.is_some());
        }
        here || o.is_some()
    });
    b
}
