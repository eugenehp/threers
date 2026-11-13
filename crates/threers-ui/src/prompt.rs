//! Asking the user something: a modal dialog with
//! a message and buttons, answered by a click, Enter (the first button,
//! the usual answer) or Escape (the last, the way out). Focus goes to the
//! first button, stays in the dialog while it is open (it is modal), and
//! goes back after. Screen readers hear it as an `alertdialog` whose
//! message is its description.
//!
//! ```
//! use threers_ui::input::Input;
//! use threers_ui::prompt::prompt;
//! use threers_ui::{FixedWidthText, Ui};
//!
//! let mut ui = Ui::new();
//! let mut input: Input<Vec<usize>> = Input::new();
//! let mut answers = vec![];
//! prompt(&mut ui, &mut input, &mut answers, "Discard changes?", "They will be lost.", &["Discard", "Cancel"], |a, _, i| a.push(i));
//! ui.layout(400.0, 300.0, &mut FixedWidthText(7.0));
//! input.simulate_keystrokes(&mut ui, &mut answers, "escape").unwrap();
//! assert_eq!(answers, [1], "Escape: the last, Cancel");
//! ```

use std::cell::Cell;
use std::rc::Rc;

use crate::input::Input;
use crate::button::{button, ButtonRole};
use crate::environment::Token;
use crate::tree::{NodeId, Ui};
use crate::{style, Role, Semantics};

/// An open prompt: its dialog, gone once it is answered.
#[derive(Debug, Clone, Copy)]
pub struct PromptHandle {
    pub dialog: NodeId,
}

/// Asks `title` (with `message` under it), answered by one of `buttons`:
/// `on_answer` gets its index. The first is the primary answer and, with
/// more than one, the last is the way out. See the [module docs](self).
#[allow(clippy::too_many_arguments)]
pub fn prompt<S: 'static>(
    ui: &mut Ui,
    input: &mut Input<S>,
    state: &mut S,
    title: &str,
    message: &str,
    buttons: &[&str],
    on_answer: impl FnMut(&mut S, &mut Ui, usize) + 'static,
) -> PromptHandle {
    let n = buttons.len();
    let roled: Vec<(&str, ButtonRole)> = buttons
        .iter()
        .enumerate()
        .map(|(i, &b)| {
            let role = if i == 0 {
                ButtonRole::Primary
            } else if i + 1 == n {
                ButtonRole::Cancel
            } else {
                ButtonRole::Normal
            };
            (b, role)
        })
        .collect();
    prompt_with(ui, input, state, title, message, &roled, on_answer)
}

/// [`prompt`] with each button's [`ButtonRole`]: Escape presses the
/// cancel button (else the last), Enter the one with focus, which starts
/// on the primary one (else the first). A destructive one reads so.
#[allow(clippy::too_many_arguments)]
pub fn prompt_with<S: 'static>(
    ui: &mut Ui,
    input: &mut Input<S>,
    state: &mut S,
    title: &str,
    message: &str,
    buttons: &[(&str, ButtonRole)],
    on_answer: impl FnMut(&mut S, &mut Ui, usize) + 'static,
) -> PromptHandle {
    let root = ui.root();
    let backdrop = ui.add_box(root, style! { position: absolute; left: 0px; top: 0px; width: 100%; height: 100%; background: rgba(0,0,0,0.45); });
    ui.defer(backdrop, 950);
    let dialog = ui.add_box(
        root,
        style! {
            position: absolute; display: flex; flex-direction: column; gap: 12px; width: 320px; padding: 18px; box-sizing: border-box;
            border-radius: 12px; background: #1f2430; border: 1px solid #3a4150; color: #e5e7eb; box-shadow: 0 16px 40px rgba(0,0,0,0.55);
        },
    );
    ui.token_style(dialog, &[("background", Token::Raised), ("border-color", Token::Border), ("color", Token::Text)]);
    ui.defer(dialog, 951);
    ui.appear(dialog, crate::transitions::Transition::Fade, 0.12);
    ui.appear(backdrop, crate::transitions::Transition::Fade, 0.12);
    crate::palette::place_in_window(ui, dialog, 320.0, 0.25);
    let t = ui.add_box(dialog, style! { font-size: 15px; font-weight: bold; });
    ui.add_text(t, title);
    if !message.is_empty() {
        let m = ui.add_box(dialog, style! { white-space: normal; });
        ui.token_style(m, &[("color", Token::SecondaryText)]);
        ui.add_text(m, message);
    }
    let mut sem = Semantics::new(Role::AlertDialog).label(title);
    sem.description = (!message.is_empty()).then(|| message.to_owned());
    ui.set_semantics(dialog, sem);
    let row = ui.add_box(dialog, style! { display: flex; gap: 8px; justify-content: flex-end; });
    let answered: Rc<Cell<Option<usize>>> = Rc::default();
    let on_answer = Rc::new(std::cell::RefCell::new(on_answer));
    let mut nodes = vec![];
    for (i, &(label, role)) in buttons.iter().enumerate() {
        let (a, f) = (answered.clone(), on_answer.clone());
        let b = button(ui, input, row, label, role, move |s, ui| {
            if a.get().is_none() {
                a.set(Some(i));
                (f.borrow_mut())(s, ui, i);
            }
        });
        nodes.push(b);
    }
    let cancel = buttons.iter().position(|b| b.1 == ButtonRole::Cancel).unwrap_or(buttons.len().saturating_sub(1));
    let primary = buttons.iter().position(|b| b.1 == ButtonRole::Primary).unwrap_or(0);
    let (a, f, all) = (answered.clone(), on_answer.clone(), nodes.clone());
    input.on_key(dialog, move |s, ui, key, _, _| {
        let i = match key {
            "Escape" => cancel,
            // Enter presses the button with focus, or else the primary one.
            "Enter" => ui.interaction().focused.and_then(|f| all.iter().position(|&n| n == f)).unwrap_or(primary),
            _ => return false,
        };
        if a.get().is_none() {
            a.set(Some(i));
            (f.borrow_mut())(s, ui, i);
        }
        true
    });
    let before = input.focused();
    input.set_modal(ui, state, Some(dialog));
    if let Some(&first) = nodes.get(primary).or(nodes.first()) {
        input.set_focus(ui, state, Some(first));
    }
    input.on_after_dispatch(move |input, ui, state| {
        // Answered, or removed from outside (see `Input::present`).
        if ui.contains(dialog) && answered.get().is_none() {
            // The window may have changed size.
            crate::palette::place_in_window(ui, dialog, 320.0, 0.25);
            return true;
        }
        if input.modal().is_none_or(|m| m == dialog || !ui.contains(m)) {
            input.set_modal(ui, state, None);
        }
        ui.remove(dialog);
        ui.remove(backdrop);
        input.set_focus(ui, state, before.filter(|&b| ui.contains(b)));
        false
    });
    PromptHandle { dialog }
}
