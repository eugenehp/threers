//! Styles for states, as CSS's `:hover`, `:active`,
//! `:focus-visible` and `.group:hover .x` do: declarations a node takes
//! while it is in a state, over its own style, put on and taken off by the
//! tree as [`Input`](super::input::Input) reports what the pointer and the
//! keyboard are doing. With a [transition](Ui::transition) the colours and
//! opacity move there instead of jumping.
//!
//! ```
//! use threers_ui::input::Input;
//! use threers_ui::{style, FixedWidthText, Ui};
//!
//! let mut ui = Ui::new();
//! let root = ui.root();
//! let row = ui.add_box(root, style! { height: 20px; background: #222; });
//! let delete = ui.add_box(row, style! { width: 20px; height: 20px; opacity: 0; });
//! ui.hover_style(row, "background: #333");
//! ui.group(row, "row");
//! ui.group_hover_style(delete, "row", "opacity: 1");
//! ui.layout(200.0, 100.0, &mut FixedWidthText(7.0));
//!
//! let mut input = Input::new();
//! input.dispatch(&mut ui, &mut (), threers_ui::input::Event::PointerMove { x: 100.0, y: 10.0 });
//! assert_eq!(ui.style(row).background, Some([0.2, 0.2, 0.2, 1.0]));
//! assert_eq!(ui.style(delete).opacity, 1.0, "the row is hovered, so its delete button shows");
//! ```

use super::animation::{Animation, Property};
use super::style::Style;
use super::tree::{NodeId, Ui};

/// CSS declarations, parsed into `(property, value)` pairs.
pub(crate) type Declarations = Vec<(String, String)>;

/// Reads `"background: #333; color: white"`.
pub(crate) fn declarations(css: &str) -> Declarations {
    css.split(';')
        .filter_map(|d| {
            let (name, value) = d.split_once(':')?;
            let (name, value) = (name.trim(), value.trim());
            (!name.is_empty() && !value.is_empty()).then(|| (name.to_owned(), value.to_owned()))
        })
        .collect()
}

/// Light or dark (CSS's `prefers-color-scheme`):
/// what a UI shows, by its [`Theme`] and the system's (see
/// [`Ui::appearance`]); styled for with [`Ui::appearance_style`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Appearance {
    #[default]
    Light,
    Dark,
}

/// The appearance a UI is asked for: light, dark, or the system's (the
/// default), which follows the system as it changes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Theme {
    Light,
    Dark,
    #[default]
    System,
}

/// A node's styles by state.
#[derive(Debug, Clone, Default)]
pub(crate) struct StateStyles {
    /// Its style out of every state.
    pub base: Option<Style>,
    pub hover: Declarations,
    pub active: Declarations,
    pub focus: Declarations,
    pub focus_visible: Declarations,
    pub disabled: Declarations,
    /// Styles while an ancestor group of this name is hovered.
    pub group_hover: Vec<(String, Declarations)>,
    /// Styles in the light or the dark appearance.
    pub appearance: Vec<(Appearance, Declarations)>,
    /// Colours by purpose (see [`Ui::token_style`]).
    pub tokens: Vec<(String, super::environment::Token)>,
    /// Seconds colours and opacity take to move to a state's.
    pub transition: f32,
}

/// What the pointer and keyboard are doing, as [`Input`](super::input::Input) last said.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Interaction {
    /// The node under the pointer and its ancestors.
    pub hovered: Vec<NodeId>,
    /// The node pressed and its ancestors.
    pub pressed: Vec<NodeId>,
    pub focused: Option<NodeId>,
    /// Focus came from the keyboard (CSS `:focus-visible`).
    pub focus_visible: bool,
}

impl Ui {
    pub(crate) fn states_mut(&mut self, id: NodeId) -> &mut StateStyles {
        let style = self.style(id).clone();
        let s = self.node_mut(id).states.get_or_insert_with(Default::default);
        s.base.get_or_insert(style);
        s
    }

    fn add_state(&mut self, id: NodeId, css: &str, pick: impl FnOnce(&mut StateStyles) -> &mut Declarations) {
        let d = declarations(css);
        pick(self.states_mut(id)).extend(d);
        self.restyle(id);
    }

    /// Declarations `id` takes while the pointer is over it or over
    /// something inside it (`:hover`).
    pub fn hover_style(&mut self, id: NodeId, css: &str) {
        self.add_state(id, css, |s| &mut s.hover);
    }

    /// Replaces what `id` takes on hover (where [`hover_style`](Self::hover_style)
    /// adds to it): for a hover that depends on another state, such as a
    /// selected tab's darker accent where the others turn grey.
    pub fn set_hover_style(&mut self, id: NodeId, css: &str) {
        self.states_mut(id).hover = declarations(css);
        self.restyle(id);
    }

    /// While it is pressed (`:active`).
    pub fn active_style(&mut self, id: NodeId, css: &str) {
        self.add_state(id, css, |s| &mut s.active);
    }

    /// While it has focus (`:focus`).
    pub fn focus_style(&mut self, id: NodeId, css: &str) {
        self.add_state(id, css, |s| &mut s.focus);
    }

    /// While it has focus that came from the keyboard (`:focus-visible`):
    /// the focus ring, which a click should not show.
    pub fn focus_visible_style(&mut self, id: NodeId, css: &str) {
        self.add_state(id, css, |s| &mut s.focus_visible);
    }

    /// While its [`Semantics`](super::Semantics) say it is disabled.
    pub fn disabled_style(&mut self, id: NodeId, css: &str) {
        self.add_state(id, css, |s| &mut s.disabled);
    }

    /// Names `id` a group, for [`group_hover_style`](Self::group_hover_style).
    pub fn group(&mut self, id: NodeId, name: &str) {
        self.node_mut(id).group = Some(name.to_owned());
        self.restyle_all();
    }

    /// Declarations `id` takes while an ancestor named `group` is hovered:
    /// a row's buttons showing when the row is.
    pub fn group_hover_style(&mut self, id: NodeId, group: &str, css: &str) {
        let d = declarations(css);
        self.states_mut(id).group_hover.push((group.to_owned(), d));
        self.restyle(id);
    }

    /// Declarations `id` takes in `appearance` (CSS's
    /// `@media (prefers-color-scheme: …)`): a UI styled for one gives the
    /// other's colours here.
    pub fn appearance_style(&mut self, id: NodeId, appearance: Appearance, css: &str) {
        let d = declarations(css);
        self.states_mut(id).appearance.push((appearance, d));
        self.restyle(id);
    }

    /// The system's appearance: the host tells it at the start, and when
    /// it changes (a browser's `prefers-color-scheme` media query). The
    /// `System` theme follows it.
    pub fn set_system_appearance(&mut self, appearance: Appearance) {
        let before = self.system_appearance;
        self.system_appearance = appearance;
        // The UI's appearance, or a part that follows the system, changed.
        if appearance != before && (self.theme == Theme::System || self.env.subtree_themes.values().any(|t| *t == Theme::System)) {
            self.restyle_all();
        }
    }

    /// Light, dark, or the system's (the default).
    pub fn set_theme(&mut self, theme: Theme) {
        let before = self.appearance();
        self.theme = theme;
        if self.appearance() != before {
            self.restyle_all();
        }
    }

    pub fn theme(&self) -> Theme {
        self.theme
    }

    /// The appearance shown: the theme's, or with the `System` theme, the
    /// system's. Every node styled for it takes that appearance's styles.
    pub fn appearance(&self) -> Appearance {
        match self.theme {
            Theme::Light => Appearance::Light,
            Theme::Dark => Appearance::Dark,
            Theme::System => self.system_appearance,
        }
    }

    /// Colours and opacity move to a state's over `seconds` (CSS
    /// `transition`), rather than at once.
    pub fn transition(&mut self, id: NodeId, seconds: f32) {
        self.states_mut(id).transition = seconds.max(0.0);
    }

    /// Takes in what the pointer and keyboard are doing and restyles every
    /// node with state styles to match. [`Input`](super::input::Input) calls
    /// it after each event.
    pub fn set_interaction(&mut self, interaction: Interaction) {
        if self.interaction != interaction {
            self.interaction = interaction;
            self.restyle_all();
        }
    }

    pub fn interaction(&self) -> &Interaction {
        &self.interaction
    }

    pub(crate) fn restyle_all(&mut self) {
        let ids: Vec<NodeId> = (0..self.nodes.len())
            .filter(|&i| !self.nodes[i].removed && self.nodes[i].states.is_some())
            .map(|i| self.id_at(i))
            .collect();
        for id in ids {
            self.restyle(id);
        }
    }

    /// The style `id` should have in the current interaction.
    fn state_style(&self, id: NodeId) -> Option<Style> {
        let states = self.nodes[id.0].states.as_ref()?;
        let mut style = states.base.clone()?;
        let i = &self.interaction;
        let apply = |s: &mut Style, d: &Declarations| {
            for (name, value) in d {
                super::css::apply(s, name, value);
            }
        };
        // The appearance first (its own part's: see `theme`): states go
        // over it.
        let appearance = self.appearance_of(id);
        for (a, d) in &states.appearance {
            if *a == appearance {
                apply(&mut style, d);
            }
        }
        for (name, token) in &states.tokens {
            super::css::apply(&mut style, name, &super::environment::css_color(self.token_in(appearance, *token)));
        }
        if i.hovered.contains(&id) {
            apply(&mut style, &states.hover);
        }
        for (group, d) in &states.group_hover {
            let hovered = self
                .ancestors(id)
                .any(|a| self.nodes[a.0].group.as_deref() == Some(group) && i.hovered.contains(&a));
            if hovered {
                apply(&mut style, d);
            }
        }
        if i.focused == Some(id) {
            apply(&mut style, &states.focus);
            if i.focus_visible {
                apply(&mut style, &states.focus_visible);
            }
        }
        if i.pressed.contains(&id) {
            apply(&mut style, &states.active);
        }
        if self.nodes[id.0].semantics.as_ref().is_some_and(|s| s.disabled) {
            apply(&mut style, &states.disabled);
        }
        Some(style)
    }

    pub(crate) fn restyle(&mut self, id: NodeId) {
        let Some(mut want) = self.state_style(id) else { return };
        // Where a box is scrolled to is its state, not its style.
        want.scroll_top = self.style(id).scroll_top;
        want.scroll_left = self.style(id).scroll_left;
        if *self.style(id) == want {
            return;
        }
        let transition = self.nodes[id.0].states.as_ref().map_or(0.0, |s| s.transition);
        // Colours and opacity, from where they are now.
        const MOVING: [Property; 4] = [Property::Opacity, Property::Background, Property::Color, Property::BorderColor];
        let from = MOVING.map(|p| self.read(id, p));
        if self.style(id).same_layout(&want) {
            self.touch_paint();
        } else {
            self.touch(id);
        }
        self.nodes[id.0].style = want;
        self.inherit_text_style(id);
        if transition > 0.0 {
            for (p, from) in MOVING.into_iter().zip(from) {
                let to = self.read(id, p);
                if from != to {
                    self.animate(id, Animation::new(p, from, to, transition));
                }
            }
        }
    }
}
