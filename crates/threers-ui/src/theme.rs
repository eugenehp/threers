//! Light, dark, or the system's — for a whole UI and for any part of it.
//!
//! - **The whole UI** follows [`Ui::set_theme`]: [`Theme::System`] (the
//!   default) follows the system's appearance as the host reports it
//!   ([`Ui::set_system_appearance`]: a browser's `prefers-color-scheme`, an
//!   OS's dark mode), `Light` and `Dark` hold one whatever the system says.
//! - **A picker** for it, [`theme_picker`]: System, Light and Dark as a
//!   segmented control, bound to the app's state, and the UI kept to it.
//!   [`Theme::name`] and [`Theme::parse`] keep the choice between runs.
//! - **Any part** can hold its own, as CSS's `color-scheme` on an element
//!   does: [`Ui::set_subtree_theme`] — a dark sidebar in a light app, a
//!   preview of both side by side, a dialog over a photo. Everything inside
//!   it coloured by [`Token`] or styled by appearance
//!   ([`Ui::appearance_style`]) takes that part's appearance;
//!   [`Ui::appearance_of`] says which a node is in.
//!
//! ```
//! use threers_ui::environment::Token;
//! use threers_ui::states::{Appearance, Theme};
//! use threers_ui::{style, Ui};
//!
//! let mut ui = Ui::new();
//! ui.set_system_appearance(Appearance::Light);
//! let root = ui.root();
//! let sidebar = ui.add_box(root, style! { width: 200px; });
//! ui.token_style(sidebar, &[("background", Token::Surface)]);
//! ui.set_subtree_theme(sidebar, Some(Theme::Dark));
//! assert_eq!(ui.appearance_of(root), Appearance::Light);
//! assert_eq!(ui.appearance_of(sidebar), Appearance::Dark);
//! // The system turning dark changes the rest; the sidebar was dark already.
//! ui.set_system_appearance(Appearance::Dark);
//! assert_eq!(ui.appearance_of(root), Appearance::Dark);
//! ```

use crate::binding::Binding;
use crate::environment::Token;
use crate::input::Input;
use crate::states::{Appearance, Theme};
use crate::tree::{NodeId, Ui};
use crate::Rgba;

impl Theme {
    /// The three, in the order a picker shows them.
    pub const ALL: [Theme; 3] = [Theme::System, Theme::Light, Theme::Dark];

    /// Its name, to show and to store: "System", "Light", "Dark".
    pub fn name(self) -> &'static str {
        match self {
            Theme::System => "System",
            Theme::Light => "Light",
            Theme::Dark => "Dark",
        }
    }

    /// A name back ([`name`](Self::name)'s, any case); `None` for others.
    pub fn parse(name: &str) -> Option<Theme> {
        Theme::ALL.into_iter().find(|t| t.name().eq_ignore_ascii_case(name.trim()))
    }
}

/// System, Light and Dark as a segmented control under `parent`, bound to
/// `theme` in the app's state (`Binding::local(Theme::System)` where the app
/// keeps none); the UI's theme follows it after every event. Named
/// "Appearance" to assistive technology, a radio group.
pub fn theme_picker<S: 'static>(ui: &mut Ui, input: &mut Input<S>, state: &S, parent: NodeId, theme: Binding<S, Theme>) -> NodeId {
    let (get, set) = (theme.clone(), theme.clone());
    let index = Binding::new(
        move |s: &S| Theme::ALL.iter().position(|&t| t == get.get(s)).unwrap_or(0),
        move |s: &mut S, i: usize| set.set(s, Theme::ALL[i.min(2)]),
    );
    let names = Theme::ALL.map(Theme::name);
    let node = crate::pickers::segmented(ui, input, state, parent, "Appearance", &names, index);
    let want = theme.get(state);
    if ui.theme() != want {
        ui.set_theme(want);
    }
    input.on_after_dispatch(move |_, ui, s| {
        if !ui.contains(node) {
            return false;
        }
        let want = theme.get(s);
        if ui.theme() != want {
            ui.set_theme(want);
        }
        true
    });
    node
}

impl Ui {
    /// `node` and everything inside it in `theme`, whatever the rest of the
    /// UI is in (`Some(Theme::System)`: the system's, whatever the UI's
    /// theme); `None` takes it back to the UI's (or a part round it).
    pub fn set_subtree_theme(&mut self, node: NodeId, theme: Option<Theme>) {
        let before = self.env.subtree_themes.get(&node).copied();
        if before == theme {
            return;
        }
        match theme {
            Some(t) => self.env.subtree_themes.insert(node, t),
            None => self.env.subtree_themes.remove(&node),
        };
        // Everything inside styled by appearance or by token, again.
        let mut stack = vec![node];
        while let Some(n) = stack.pop() {
            if self.nodes[n.0].states.is_some() {
                self.restyle(n);
            }
            stack.extend(self.children(n).iter().copied());
        }
    }

    /// The theme `node` was given of its own, if any.
    pub fn subtree_theme(&self, node: NodeId) -> Option<Theme> {
        self.env.subtree_themes.get(&node).copied()
    }

    /// The appearance `node` shows: the nearest theme given to it or a
    /// part round it, else the UI's.
    pub fn appearance_of(&self, node: NodeId) -> Appearance {
        if !self.env.subtree_themes.is_empty() && self.contains(node) {
            for a in self.ancestors(node) {
                if let Some(&t) = self.env.subtree_themes.get(&a) {
                    return match t {
                        Theme::Light => Appearance::Light,
                        Theme::Dark => Appearance::Dark,
                        Theme::System => self.system_appearance,
                    };
                }
            }
        }
        self.appearance()
    }

    /// `token`'s colour where `node` is: in its [appearance](Self::appearance_of).
    pub fn token_of(&self, node: NodeId, token: Token) -> Rgba {
        self.token_in(self.appearance_of(node), token)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{style, FixedWidthText};

    #[test]
    fn a_part_keeps_its_theme_as_the_rest_follows_the_system() {
        let mut ui = Ui::new();
        ui.set_system_appearance(Appearance::Light);
        let root = ui.root();
        let card = ui.add_box(root, style! {});
        ui.token_style(card, &[("background", Token::Surface)]);
        let dark = ui.add_box(root, style! {});
        let inside = ui.add_box(dark, style! {});
        ui.token_style(inside, &[("background", Token::Surface), ("color", Token::Text)]);
        ui.appearance_style(inside, Appearance::Dark, "border-radius: 9px");
        ui.set_subtree_theme(dark, Some(Theme::Dark));
        let light_surface = ui.token_in(Appearance::Light, Token::Surface);
        let dark_surface = ui.token_in(Appearance::Dark, Token::Surface);
        let near = |a: Option<Rgba>, b: Rgba| a.is_some_and(|a| (0..4).all(|i| (a[i] - b[i]).abs() < 1.0 / 255.0));
        assert!(near(ui.style(card).background, light_surface));
        assert!(near(ui.style(inside).background, dark_surface), "inside the dark part, dark");
        assert_eq!(ui.style(inside).border_radius, crate::Length::Px(9.0), "and its dark appearance style");
        // The system goes dark, then the app holds Light: the part stays dark.
        ui.set_system_appearance(Appearance::Dark);
        assert!(near(ui.style(card).background, dark_surface));
        ui.set_theme(Theme::Light);
        assert!(near(ui.style(card).background, light_surface));
        assert!(near(ui.style(inside).background, dark_surface));
        // Given back: it follows the UI again.
        ui.set_subtree_theme(dark, None);
        assert!(near(ui.style(inside).background, light_surface));
        assert_eq!(ui.appearance_of(inside), Appearance::Light);
    }

    #[test]
    fn the_picker_sets_the_theme_and_names_round_trip() {
        #[derive(Default)]
        struct S {
            theme: Theme,
        }
        let mut ui = Ui::new();
        let mut input: Input<S> = Input::new();
        let mut s = S::default();
        let root = ui.root();
        let picker = theme_picker(&mut ui, &mut input, &s, root, crate::binding!(|s: S| s.theme));
        ui.layout(400.0, 200.0, &mut FixedWidthText(7.0));
        assert_eq!(ui.theme(), Theme::System, "System by default");
        // Choosing Dark: a click on its segment.
        let dark = ui.accessibility_tree().into_iter().find(|a| a.name == "Dark").expect("a Dark choice");
        let r = dark.bounds;
        let (x, y) = (r.x + r.width / 2.0, r.y + r.height / 2.0);
        let b = crate::input::PointerButton::Primary;
        input.dispatch(&mut ui, &mut s, crate::input::Event::PointerDown { x, y, button: b });
        input.dispatch(&mut ui, &mut s, crate::input::Event::PointerUp { x, y, button: b });
        assert_eq!(s.theme, Theme::Dark);
        assert_eq!(ui.theme(), Theme::Dark);
        assert!(ui.contains(picker));
        for t in Theme::ALL {
            assert_eq!(Theme::parse(t.name()), Some(t));
        }
        assert_eq!(Theme::parse(" dark "), Some(Theme::Dark));
        assert_eq!(Theme::parse("Sepia"), None);
    }
}
