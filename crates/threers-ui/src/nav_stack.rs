//! Screens pushed on and popped off a stack, driven by a path in the state:
//! a header with the top screen's title and a back button, each screen
//! sliding in from the right as it is pushed and back out as it is popped,
//! and on a touch screen a drag from the left edge to go back. The path is
//! the truth: push by adding to it, pop by taking from it (the back
//! button, the edge swipe and Alt+Left all do), jump by setting it.
//!
//! A screen names itself with a [`NavTitle`] preference (see
//! [`preferences`](crate::preferences)), so what is inside it can retitle
//! it.
//!
//! ```
//! use threers_ui::input::Input;
//! use threers_ui::nav_stack::{nav_stack, NavTitle};
//! use threers_ui::{binding, FixedWidthText, Ui};
//!
//! #[derive(Default)]
//! struct S { path: Vec<String> }
//!
//! let mut ui = Ui::new();
//! let mut input: Input<S> = Input::new();
//! let mut s = S::default();
//! let root = ui.root();
//! let stack = nav_stack(&mut ui, &mut input, &s, root, "Mail", binding!(|s: S| s.path), |ui, _, _, page, screen| {
//!     ui.add_text(screen, format!("Message {page}"));
//!     ui.set_preference(screen, NavTitle(format!("Message {page}")));
//! }, |ui, _, _, screen| {
//!     ui.add_text(screen, "Inbox");
//! });
//! ui.layout(400.0, 600.0, &mut FixedWidthText(7.0));
//! s.path.push("7".into());
//! input.tick(&mut ui, &mut s, 0.0);
//! ui.layout(400.0, 600.0, &mut FixedWidthText(7.0));
//! assert_eq!(stack.title(&ui), "Message 7");
//! input.set_focus(&mut ui, &mut s, Some(stack.back));
//! input.simulate_keystrokes(&mut ui, &mut s, "alt-left").unwrap();
//! assert!(s.path.is_empty());
//! ```

use std::cell::{Cell, RefCell};
use std::rc::Rc;

use crate::environment::LayoutDirection;
use crate::binding::Binding;
use crate::environment::Token;
use crate::input::{Event, GesturePhase, Input, Phase};
use crate::transitions::Transition;
use crate::tree::{NodeId, Ui};
use crate::{style, Display, Length, Role, Semantics};

/// How screens come and go, set on the stack's container (or anything
/// around it) as an [environment](crate::environment) value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum NavTransition {
    /// In from the right, out to the right.
    #[default]
    Slide,
    /// The faces of a cube, turning about the edge they share.
    Cube,
    /// The screen on top turns over like a page, about its left edge.
    PageTurn,
}

fn transition_at(ui: &Ui, container: NodeId) -> NavTransition {
    if ui.reduce_motion() {
        return NavTransition::Slide;
    }
    ui.environment::<NavTransition>(container)
        .unwrap_or_default()
}

/// How a 3D turn of a screen moves.
fn turn_spring() -> crate::anim::Spring {
    crate::anim::Spring::new(170.0, 24.0, 1.0)
}

/// `screen` comes in over `below`.
fn push_screen(ui: &mut Ui, how: NavTransition, screen: NodeId, below: NodeId, body: NodeId) {
    use crate::animation::Property;
    use crate::depth::Transform3d;
    use std::f32::consts::{FRAC_PI_2, PI};
    let w = ui.rect(body).width.max(1.0);
    match how {
        NavTransition::Slide => ui.appear(screen, Transition::Move { dx: w, dy: 0.0 }, SLIDE),
        NavTransition::Cube => {
            // The new face turns in from the right, the old one away to
            // the left, about the edge between them.
            ui.set_transform3d(
                screen,
                Transform3d {
                    rotate_y: FRAC_PI_2,
                    origin: (0.0, 0.5),
                    shade: true,
                    perspective: w * 1.5,
                    ..Transform3d::default()
                },
            );
            ui.spring_to(screen, Property::RotateY, 0.0, turn_spring());
            ui.set_transform3d(
                below,
                Transform3d {
                    origin: (1.0, 0.5),
                    shade: true,
                    perspective: w * 1.5,
                    ..Transform3d::default()
                },
            );
            ui.spring_to(below, Property::RotateY, -FRAC_PI_2, turn_spring());
        }
        NavTransition::PageTurn => {
            // The old page lies on top and turns over, showing the new one
            // under it.
            if let Some(at) = ui.children(body).iter().position(|&c| c == below) {
                ui.move_child(body, screen, at);
            }
            ui.set_transform3d(
                below,
                Transform3d {
                    origin: (0.0, 0.5),
                    shade: true,
                    hide_backface: true,
                    perspective: w * 2.0,
                    ..Transform3d::default()
                },
            );
            ui.spring_to(below, Property::RotateY, -PI, turn_spring());
        }
    }
}

/// `screen` goes, showing `below` again.
fn pop_screen(ui: &mut Ui, how: NavTransition, screen: NodeId, below: NodeId, body: NodeId) {
    use crate::animation::Property;
    use crate::depth::Transform3d;
    use std::f32::consts::FRAC_PI_2;
    let w = ui.rect(body).width.max(1.0);
    match how {
        NavTransition::Slide => ui.remove_with(screen, Transition::Move { dx: w, dy: 0.0 }, SLIDE),
        NavTransition::Cube => {
            ui.set_transform3d(
                screen,
                Transform3d {
                    origin: (0.0, 0.5),
                    shade: true,
                    perspective: w * 1.5,
                    ..ui.transform3d(screen).unwrap_or_default()
                },
            );
            ui.spring_to(screen, Property::RotateY, FRAC_PI_2, turn_spring());
            ui.remove_with(screen, Transition::Fade, 0.45);
            ui.spring_to(below, Property::RotateY, 0.0, turn_spring());
        }
        NavTransition::PageTurn => {
            // The page turns back over the one leaving.
            ui.spring_to(below, Property::RotateY, 0.0, turn_spring());
            ui.remove_with(screen, Transition::Fade, 0.5);
        }
    }
}

/// A screen's title, stated as a preference by the screen or anything in it.
#[derive(Debug, Clone, PartialEq)]
pub struct NavTitle(pub String);

/// How far from the left edge a drag must start to go back.
const EDGE: f32 = 24.0;
/// Seconds a push or pop slides for.
const SLIDE: f32 = 0.28;

/// A navigation stack's nodes.
#[derive(Debug, Clone, Copy)]
pub struct NavStack {
    pub container: NodeId,
    pub header: NodeId,
    pub back: NodeId,
    title_text: NodeId,
    /// Where screens go.
    pub body: NodeId,
}

impl NavStack {
    /// The title showing.
    pub fn title(&self, ui: &Ui) -> String {
        ui.text_lines_of(self.title_text).concat()
    }
}

type Destination<S> = Box<dyn FnMut(&mut Ui, &mut Input<S>, &S, &str, NodeId)>;

/// A stack under `parent`: `build_root` fills the bottom screen (titled
/// `root_title`), and `destination` builds the screen for each entry of
/// `path`, given its entry and the screen box.
#[allow(clippy::too_many_arguments)]
pub fn nav_stack<S: 'static>(
    ui: &mut Ui,
    input: &mut Input<S>,
    state: &S,
    parent: NodeId,
    root_title: &str,
    path: Binding<S, Vec<String>>,
    destination: impl FnMut(&mut Ui, &mut Input<S>, &S, &str, NodeId) + 'static,
    build_root: impl FnOnce(&mut Ui, &mut Input<S>, &S, NodeId),
) -> NavStack {
    let container = ui.add_box(parent, style! { display: flex; flex-direction: column; flex-grow: 1; min-height: 0; overflow: hidden; });
    ui.set_semantics(
        container,
        Semantics::new(Role::Navigation).label(root_title),
    );
    let header = ui.add_box(container, style! { display: flex; align-items: center; gap: 8px; min-height: 40px; padding: 0 8px; border-bottom: 1px solid; flex-shrink: 0; });
    ui.token_style(header, &[("border-color", Token::Separator)]);
    let back = ui.add_box(
        header,
        style! { padding: 4px 8px; border-radius: 6px; cursor: pointer; display: none; },
    );
    ui.token_style(back, &[("color", Token::AccentText)]);
    ui.add_text(back, crate::i18n::back_label());
    ui.set_semantics(
        back,
        Semantics::new(Role::Button).label(crate::i18n::word(crate::i18n::Word::Back)),
    );
    ui.hover_style(back, "background: var(--hover)");
    let title = ui.add_box(header, style! { font-weight: bold; flex-grow: 1; min-width: 0; overflow: hidden; white-space: nowrap; text-overflow: ellipsis; });
    ui.set_semantics(title, Semantics::new(Role::Heading(2)));
    let title_text = ui.add_text(title, root_title);
    let body = ui.add_box(
        container,
        style! { position: relative; flex-grow: 1; min-height: 0; overflow: hidden; },
    );
    let screen_style = style! { position: absolute; left: 0px; top: 0px; width: 100%; height: 100%; display: flex; flex-direction: column; overflow: auto; box-sizing: border-box; padding: 8px; };
    let root_screen = ui.add_box(body, screen_style.clone());
    ui.token_style(root_screen, &[("background", Token::Surface)]);
    ui.set_preference(root_screen, NavTitle(root_title.to_owned()));
    build_root(ui, input, state, root_screen);
    let nav = NavStack {
        container,
        header,
        back,
        title_text,
        body,
    };

    // Pop: the path without its last entry.
    let pop = {
        let path = path.clone();
        Rc::new(move |s: &mut S| {
            let mut p = path.get(s);
            p.pop();
            path.set(s, p);
        })
    };
    let pp = pop.clone();
    input.on_click(back, move |s, _, _| pp(s));
    let pp = pop.clone();
    input.on_key(container, move |s, ui, key, m, _| {
        if ui.reading_key(container, key) == "ArrowLeft" && m.alt {
            pp(s);
            return true;
        }
        false
    });

    // The screens built, over the root one: their path entries and boxes.
    let built: Rc<RefCell<Vec<(String, NodeId)>>> = Rc::default();
    // A drag from the left edge: how far the top screen has been pulled.
    let edge: Rc<Cell<Option<f32>>> = Rc::default();
    let (b, e, pp) = (built.clone(), edge.clone(), pop.clone());
    input.on(body, Phase::Bubble, move |s, ui, ev, cx| {
        let Event::Gesture(crate::input::Gesture::Pan { x, dx, phase, .. }) = *ev else {
            return;
        };
        // From the edge the screen came over, and how far from there.
        let r = ui.rect(body);
        let (x, dx) = match ui.direction_of(body) {
            LayoutDirection::LeftToRight => (x - r.x, dx),
            LayoutDirection::RightToLeft => (r.x + r.width - x, -dx),
        };
        let Some(&(_, top)) = b.borrow().last() else {
            return;
        };
        match phase {
            GesturePhase::Began if x - dx <= EDGE && dx > 0.0 => e.set(Some(dx)),
            GesturePhase::Changed if e.get().is_some() => {
                e.set(Some((e.get().unwrap_or(0.0) + dx).max(0.0)))
            }
            GesturePhase::Ended if e.get().is_some() => {
                let pulled = e.take().unwrap_or(0.0);
                if pulled > ui.rect(body).width / 3.0 {
                    pp(s);
                } else {
                    let from = ui.read(top, crate::animation::Property::Left);
                    ui.animate(
                        top,
                        crate::animation::Animation::new(
                            crate::animation::Property::Left,
                            from,
                            crate::animation::Value::Number(0.0),
                            0.2,
                        ),
                    );
                }
                cx.prevent_default();
                return;
            }
            _ => return,
        }
        // Pulled: the screen follows the finger, and nothing scrolls.
        let pulled = e.get().unwrap_or(0.0);
        ui.update_style(top, |st| st.left = Length::Px(pulled));
        cx.prevent_default();
    });

    // The path in the state, and the screens built, kept the same.
    let mut destination: Destination<S> = Box::new(destination);
    let screen_style2 = screen_style;
    input.on_after_dispatch(move |input, ui, s| {
        if !ui.contains(container) {
            return false;
        }
        let want = path.get(s);
        let mut changed = false;
        // Pop what is no longer on the path (or differs from it).
        loop {
            let keep = {
                let b = built.borrow();
                b.iter()
                    .zip(&want)
                    .take_while(|((p, _), w)| p == *w)
                    .count()
            };
            if built.borrow().len() <= keep {
                break;
            }
            let Some((_, screen)) = built.borrow_mut().pop() else {
                break;
            };
            let below = built.borrow().last().map_or(root_screen, |&(_, n)| n);
            pop_screen(ui, transition_at(ui, container), screen, below, body);
            changed = true;
        }
        // Push what is new, sliding in the last only.
        let have = built.borrow().len();
        for (i, entry) in want.iter().enumerate().skip(have) {
            let screen = ui.add_box(body, screen_style2.clone());
            ui.token_style(screen, &[("background", Token::Surface)]);
            ui.set_preference(screen, NavTitle(entry.clone()));
            destination(ui, input, s, entry, screen);
            if i + 1 == want.len() {
                let below = built.borrow().last().map_or(root_screen, |&(_, n)| n);
                push_screen(ui, transition_at(ui, container), screen, below, body);
            }
            built.borrow_mut().push((entry.clone(), screen));
            changed = true;
        }
        if changed {
            // Only the top screen is there for the pointer and assistive
            // technology; the ones under it wait.
            let tops: Vec<NodeId> = std::iter::once(root_screen)
                .chain(built.borrow().iter().map(|&(_, n)| n))
                .collect();
            let last = tops.len() - 1;
            for (i, &n) in tops.iter().enumerate() {
                let on = i == last;
                ui.semantics_mut(n).hidden = !on;
                if ui.style(n).pointer_events != on {
                    ui.update_style(n, |st| st.pointer_events = on);
                }
            }
            let depth = built.borrow().len();
            ui.update_style(back, |st| {
                st.display = if depth > 0 {
                    Display::Flex
                } else {
                    Display::None
                }
            });
        }
        // The title: the top screen's (or something in it), as it says now.
        let top = built.borrow().last().map_or(root_screen, |&(_, n)| n);
        let title = ui
            .preference::<NavTitle>(top)
            .map_or_else(String::new, |t| t.0);
        if nav_title(ui, title_text) != title {
            ui.set_text(title_text, title);
        }
        true
    });
    nav
}

fn nav_title(ui: &Ui, text: NodeId) -> String {
    match ui.kind(text) {
        crate::NodeKind::Text(t) => t.clone(),
        _ => String::new(),
    }
}
