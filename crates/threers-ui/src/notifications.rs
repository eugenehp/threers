//! Notifications stacked in a corner: each a title, a
//! body, perhaps an action, and a close button; gone after a while unless
//! it asks to stay; the oldest dropped past a few. Each is announced to
//! screen readers as it appears (a live `status`, or an `alert` for an
//! error).

use std::cell::RefCell;
use std::rc::Rc;

use crate::environment::Token;
use crate::input::Input;
use crate::states::Appearance;
use crate::transitions::Transition;
use crate::tree::{NodeId, Ui};
use crate::{style, Role, Semantics};

/// Seconds a notice takes to fade out.
const LEAVE: f32 = 0.2;

/// How a notice reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Tone {
    #[default]
    Info,
    Success,
    /// An alert: announced at once, and it stays until closed.
    Error,
}

/// A notification to show.
#[derive(Debug, Clone, Default)]
pub struct Notice {
    pub title: String,
    pub body: String,
    /// A button on it, and what it is called.
    pub action: Option<String>,
    pub tone: Tone,
    /// Seconds it shows for; `None` until closed (an error's default).
    pub timeout: Option<f64>,
}

impl Notice {
    pub fn new(title: impl Into<String>, body: impl Into<String>) -> Self {
        Self {
            title: title.into(),
            body: body.into(),
            timeout: Some(5.0),
            ..Self::default()
        }
    }

    pub fn action(mut self, label: impl Into<String>) -> Self {
        self.action = Some(label.into());
        self
    }

    pub fn tone(mut self, tone: Tone) -> Self {
        self.tone = tone;
        if tone == Tone::Error {
            self.timeout = None;
        }
        self
    }

    pub fn timeout(mut self, seconds: Option<f64>) -> Self {
        self.timeout = seconds;
        self
    }
}

type ActionFn<S> = Box<dyn FnMut(&mut S, &mut Ui)>;

/// The stack. See the [module docs](self).
pub struct Notifications<S> {
    stack: NodeId,
    shown: Rc<RefCell<Vec<NodeId>>>,
    /// At most this many show; the oldest goes for a new one.
    pub max: usize,
    _state: std::marker::PhantomData<S>,
}

impl<S: 'static> Notifications<S> {
    /// An empty stack in the bottom-right corner, over everything.
    pub fn new(ui: &mut Ui) -> Self {
        let root = ui.root();
        let stack = ui.add_box(
            root,
            style! { position: absolute; right: 16px; bottom: 16px; width: 300px; display: flex; flex-direction: column; gap: 8px; pointer-events: none; },
        );
        ui.set_semantics(stack, Semantics::new(Role::Region).label("Notifications"));
        ui.defer(stack, 800);
        Self {
            stack,
            shown: Rc::default(),
            max: 4,
            _state: std::marker::PhantomData,
        }
    }

    /// Shows `notice`; `on_action` runs when its action is pressed (and it
    /// closes). Returns its node.
    pub fn push(
        &mut self,
        ui: &mut Ui,
        input: &mut Input<S>,
        notice: Notice,
        on_action: Option<ActionFn<S>>,
    ) -> NodeId {
        // The oldest goes when there are too many.
        {
            let mut shown = self.shown.borrow_mut();
            shown.retain(|&n| ui.contains(n) && !ui.is_leaving(n));
            while shown.len() >= self.max.max(1) {
                let old = shown.remove(0);
                ui.remove_with(old, Transition::Fade, LEAVE);
            }
        }
        let card = ui.add_box(
            self.stack,
            style! {
                display: flex; flex-direction: column; gap: 4px; padding: 10px 12px; border-radius: 10px; pointer-events: auto;
                border: 1px solid; box-shadow: 0 8px 22px rgba(0,0,0,0.45);
            },
        );
        let tone = match notice.tone {
            Tone::Info => Token::Accent,
            Tone::Success => Token::Success,
            Tone::Error => Token::Destructive,
        };
        ui.token_style(
            card,
            &[
                ("background", Token::Raised),
                ("color", Token::Text),
                ("border-color", tone),
            ],
        );
        ui.appearance_style(
            card,
            Appearance::Light,
            "box-shadow: 0 8px 22px rgba(0,0,0,0.15)",
        );
        // Through the base style: the card has a light one over it.
        ui.update_style(card, |s| s.border.left = 3.0);
        let head = ui.add_box(
            card,
            style! { display: flex; align-items: center; gap: 8px; },
        );
        let title = ui.add_box(head, style! { font-weight: bold; flex-grow: 1; });
        ui.add_text(title, notice.title.clone());
        let close = ui.add_box(head, style! { display: flex; align-items: center; justify-content: center; min-width: 24px; min-height: 24px; border-radius: 6px; cursor: pointer; });
        ui.hover_style(close, "background: var(--hover)");
        ui.token_style(close, &[("color", Token::SecondaryText)]);
        ui.add_text(close, "✕");
        ui.set_semantics(
            close,
            Semantics::new(Role::Button).label(crate::i18n::word(crate::i18n::Word::Close)),
        );
        if !notice.body.is_empty() {
            let body = ui.add_box(card, style! { font-size: 12px; white-space: normal; });
            ui.token_style(body, &[("color", Token::SecondaryText)]);
            ui.add_text(body, notice.body.clone());
        }
        let role = if notice.tone == Tone::Error {
            Role::Alert
        } else {
            Role::Status
        };
        let name = if notice.body.is_empty() {
            notice.title.clone()
        } else {
            format!("{}. {}", notice.title, notice.body)
        };
        let mut sem = Semantics::new(role).label(name);
        sem.live = true;
        ui.set_semantics(card, sem);
        input.on_click(close, move |_, ui, cx| {
            ui.remove_with(card, Transition::Fade, LEAVE);
            cx.stop_propagation();
        });
        if let Some(label) = notice.action {
            let b = crate::components::button(
                ui,
                card,
                &label,
                crate::components::ButtonKind::Primary,
            );
            ui.apply(b, &crate::patch::Patch::css("align-self: flex-start"));
            let mut f = on_action;
            input.on_click(b, move |s, ui, _| {
                if let Some(f) = &mut f {
                    f(s, ui);
                }
                ui.remove_with(card, Transition::Fade, LEAVE);
            });
        }
        if let Some(t) = notice.timeout {
            // Gone with the card if it is closed first.
            input.after_for(card, t, move |_, ui| {
                ui.remove_with(card, Transition::Fade, LEAVE)
            });
        }
        ui.appear(card, Transition::from_right(24.0), 0.2);
        self.shown.borrow_mut().push(card);
        card
    }

    /// The notices showing, oldest first.
    pub fn shown(&self, ui: &Ui) -> Vec<NodeId> {
        self.shown
            .borrow()
            .iter()
            .copied()
            .filter(|&n| ui.contains(n) && !ui.is_leaving(n))
            .collect()
    }
}
