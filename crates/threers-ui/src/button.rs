//! Buttons by role and style. A [`ButtonRole`] says what a button does —
//! the usual answer, one that deletes or discards, or the way out — and
//! dialogs read it: [`prompt_with`](crate::prompt::prompt_with) answers
//! Escape with the cancel button and Enter with the primary one. A
//! [`ButtonStyle`] set as an [environment](crate::environment) value on a
//! box restyles every [`button`] built inside it.
//!
//! ```
//! use threers_ui::button::{button, ButtonRole, ButtonStyle};
//! use threers_ui::input::Input;
//! use threers_ui::{style, Ui};
//!
//! let mut ui = Ui::new();
//! let mut input: Input<()> = Input::new();
//! let root = ui.root();
//! let bar = ui.add_box(root, style! { display: flex; });
//! ui.set_environment(bar, ButtonStyle::Borderless);
//! let b = button(&mut ui, &mut input, bar, "Delete", ButtonRole::Destructive, |_, _| {});
//! assert_eq!(ui.style(b).background, None, "borderless: no fill");
//! ```

use crate::environment::Token;
use crate::input::Input;
use crate::tree::{NodeId, Ui};
use crate::{style, Role, Semantics};

/// What a button does.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ButtonRole {
    #[default]
    Normal,
    /// The usual answer: filled with the accent.
    Primary,
    /// Deletes or discards: in the destructive colour.
    Destructive,
    /// The way out: what Escape presses.
    Cancel,
}

/// How buttons look, set on a box for every button inside it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ButtonStyle {
    /// Outlined; a primary or destructive one filled.
    #[default]
    Bordered,
    /// Every one filled.
    Prominent,
    /// Text alone, in its role's colour.
    Borderless,
}

/// A button saying `label`, styled by its `role` and the [`ButtonStyle`]
/// in effect at `parent`, calling `on_click` when pressed.
pub fn button<S: 'static>(
    ui: &mut Ui,
    input: &mut Input<S>,
    parent: NodeId,
    label: &str,
    role: ButtonRole,
    mut on_click: impl FnMut(&mut S, &mut Ui) + 'static,
) -> NodeId {
    let style = ui.environment::<ButtonStyle>(parent).unwrap_or_default();
    let b = ui.add_box(parent, style! { padding: 6px 14px; border-radius: 6px; cursor: pointer; border: 1px solid transparent; });
    ui.add_text(b, label);
    ui.set_semantics(b, Semantics::new(Role::Button).label(label));
    let colour = match role {
        ButtonRole::Destructive => Token::Destructive,
        ButtonRole::Normal | ButtonRole::Cancel => Token::Text,
        ButtonRole::Primary => Token::Accent,
    };
    let filled = match style {
        ButtonStyle::Prominent => true,
        ButtonStyle::Bordered => matches!(role, ButtonRole::Primary | ButtonRole::Destructive),
        ButtonStyle::Borderless => false,
    };
    if filled {
        let fill = if role == ButtonRole::Destructive {
            Token::DestructiveFill
        } else {
            Token::Accent
        };
        ui.token_style(b, &[("background", fill), ("color", Token::OnAccent)]);
    } else if style == ButtonStyle::Bordered {
        ui.token_style(b, &[("border-color", Token::Border), ("color", colour)]);
    } else {
        ui.token_style(
            b,
            &[(
                "color",
                if role == ButtonRole::Normal {
                    Token::Accent
                } else {
                    colour
                },
            )],
        );
    }
    ui.hover_style(
        b,
        if filled {
            "opacity: 0.88"
        } else {
            "background: rgba(128,138,160,0.2)"
        },
    );
    ui.active_style(b, "opacity: 0.75");
    ui.focus_visible_style(b, "box-shadow: 0 0 0 2px #93c5fd");
    input.on_click(b, move |s, ui, _| on_click(s, ui));
    b
}
