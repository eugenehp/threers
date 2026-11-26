//! Gradients and glass as fills: what SwiftUI's shape styles and
//! `.glassEffect` are, for any node and, by theme colour, for every
//! component drawn in it.
//!
//! [`Ui::set_token_fill`] has a theme colour painted as a gradient or as
//! glass wherever it is a background: with the accent a gradient, every
//! switch that is on, progress bar, chosen day and default button is one;
//! with the surface glass, every card is. `view::El` has the same for one
//! element (`background_gradient`, `glass`) or everything in one (`tint`).
//!
//! ```
//! use threers_ui::environment::Token;
//! use threers_ui::fill::{Fill, Gradient};
//! use threers_ui::{style, Ui};
//!
//! let mut ui = Ui::new();
//! let root = ui.root();
//! let bar = ui.add_box(root, style! { height: 8px; });
//! ui.token_style(bar, &[("background", Token::Accent)]);
//! ui.set_token_fill(
//!     Token::Accent,
//!     Some(Fill::Gradient(Gradient::linear(90.0, ["#7c3aed", "#ec4899"]))),
//! );
//! assert_eq!(ui.style(bar).background_gradient.len(), 2);
//! ui.set_token_fill(Token::Accent, None);
//! assert!(ui.style(bar).background_gradient.is_empty());
//! ```

use crate::environment::Token;
use crate::material::Material;
use crate::states::Appearance;
use crate::tree::{NodeId, Ui};
use crate::{Rgba, Style};

/// A colour of a [`Gradient`]: one of the theme's, or CSS.
#[derive(Debug, Clone, PartialEq)]
pub enum Paint {
    Token(Token),
    Css(String),
}

impl Paint {
    fn css(&self) -> String {
        match self {
            Paint::Token(t) => format!("var(--{})", t.css_name()),
            Paint::Css(c) => c.clone(),
        }
    }
}

impl From<Token> for Paint {
    fn from(t: Token) -> Self {
        Paint::Token(t)
    }
}

impl From<&str> for Paint {
    fn from(c: &str) -> Self {
        Paint::Css(c.to_owned())
    }
}

impl From<String> for Paint {
    fn from(c: String) -> Self {
        Paint::Css(c)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Shape {
    /// Degrees, as CSS: 90 left to right, 180 down.
    Linear(f32),
    Radial,
    /// Degrees it starts at.
    Conic(f32),
}

/// Colours running into each other, in the theme's colours or CSS's.
#[derive(Debug, Clone, PartialEq)]
pub struct Gradient {
    shape: Shape,
    stops: Vec<(Paint, Option<f32>)>,
}

impl Gradient {
    fn new<P: Into<Paint>>(shape: Shape, colors: impl IntoIterator<Item = P>) -> Self {
        Gradient {
            shape,
            stops: colors.into_iter().map(|c| (c.into(), None)).collect(),
        }
    }

    /// Along a line at `degrees` (CSS's: 90 left to right, 180 down), the
    /// colours evenly spaced (SwiftUI's `LinearGradient`).
    pub fn linear<P: Into<Paint>>(degrees: f32, colors: impl IntoIterator<Item = P>) -> Self {
        Self::new(Shape::Linear(degrees), colors)
    }

    /// From the middle out (`RadialGradient`).
    pub fn radial<P: Into<Paint>>(colors: impl IntoIterator<Item = P>) -> Self {
        Self::new(Shape::Radial, colors)
    }

    /// Round the middle, from `degrees` (`AngularGradient`).
    pub fn angular<P: Into<Paint>>(degrees: f32, colors: impl IntoIterator<Item = P>) -> Self {
        Self::new(Shape::Conic(degrees), colors)
    }

    /// `token` with a little light on it from above, as SwiftUI's
    /// `Color.gradient`.
    pub fn of(token: Token) -> Self {
        let c = format!("var(--{})", token.css_name());
        Self::linear(
            180.0,
            [format!("color-mix(in srgb, {c} 91%, #ffffff)"), c.clone()],
        )
    }

    /// Where each colour is, from 0 to 1 (those past the end keep their
    /// even spacing).
    pub fn at(mut self, places: impl IntoIterator<Item = f32>) -> Self {
        for (stop, p) in self.stops.iter_mut().zip(places) {
            stop.1 = Some(p.clamp(0.0, 1.0));
        }
        self
    }

    /// As CSS, its theme colours by name. One colour is that colour
    /// throughout.
    pub fn css(&self) -> String {
        let mut stops: Vec<String> = self
            .stops
            .iter()
            .map(|(p, at)| match at {
                Some(at) => format!("{} {}%", p.css(), at * 100.0),
                None => p.css(),
            })
            .collect();
        if stops.len() == 1 {
            stops.push(stops[0].clone());
        }
        if stops.is_empty() {
            stops = vec!["transparent".into(), "transparent".into()];
        }
        let stops = stops.join(", ");
        match self.shape {
            Shape::Linear(d) => format!("linear-gradient({d}deg, {stops})"),
            Shape::Radial => format!("radial-gradient({stops})"),
            Shape::Conic(d) => format!("conic-gradient(from {d}deg, {stops})"),
        }
    }

    /// Its colour at `t` (0 to 1) along it, as CSS.
    pub fn css_at(&self, t: f32) -> String {
        let n = self.stops.len();
        let even = |i: usize| i as f32 / (n.max(2) - 1) as f32;
        let place = |i: usize| self.stops[i].1.unwrap_or_else(|| even(i));
        match n {
            0 => "transparent".into(),
            1 => self.stops[0].0.css(),
            _ if t <= place(0) => self.stops[0].0.css(),
            _ => {
                let Some(i) = (1..n).find(|&i| t <= place(i)) else {
                    return self.stops[n - 1].0.css();
                };
                let (a, b) = (place(i - 1), place(i));
                let k = ((t - a) / (b - a).max(1e-6)).clamp(0.0, 1.0) * 100.0;
                format!(
                    "color-mix(in srgb, {} {k:.1}%, {})",
                    self.stops[i].0.css(),
                    self.stops[i - 1].0.css()
                )
            }
        }
    }

    /// Its first colour.
    pub(crate) fn first(&self) -> Option<&Paint> {
        self.stops.first().map(|(p, _)| p)
    }

    /// Its colours and where they are, in the theme of `ui` at `node`.
    pub(crate) fn resolve(&self, ui: &Ui) -> Vec<(f32, Rgba)> {
        let css = crate::environment::resolve_colors(&self.css(), |t| ui.token(t)).into_owned();
        let mut s = Style::default();
        crate::css::apply(&mut s, "background", &css);
        s.background_gradient
    }
}

/// What a background is painted as, instead of one colour.
#[derive(Debug, Clone, PartialEq)]
pub enum Fill {
    Gradient(Gradient),
    /// What is behind, blurred, under a tint (see [`Material`]).
    Glass(Material),
}

/// `material`'s glass in `tint` (the colour the background would have
/// been), as `s`'s background.
pub(crate) fn glass(s: &mut Style, material: Material, tint: Rgba, dark: bool) {
    let (blur, saturate, light_tint, dark_tint) = material.recipe();
    s.backdrop_blur = blur;
    s.backdrop_saturate = saturate;
    let a = if dark { dark_tint } else { light_tint };
    s.background = Some([tint[0], tint[1], tint[2], tint[3] * a]);
}

impl Ui {
    /// Paints `token` as `fill` wherever it is a node's background, in
    /// every component; `None` as its plain colour again.
    pub fn set_token_fill(&mut self, token: Token, fill: Option<Fill>) {
        self.env.fills.retain(|(t, _)| *t != token);
        if let Some(f) = fill {
            self.env.fills.push((token, f));
        }
        self.restyle_all();
    }

    /// How `token` is painted as a background, if not as its colour
    /// (`AccentText` as the accent is, unless it has a fill of its own).
    pub fn token_fill(&self, token: Token) -> Option<&Fill> {
        let of = |token| self.env.fills.iter().find(|(t, _)| *t == token);
        of(token)
            .or_else(|| of(Token::Accent).filter(|_| token == Token::AccentText))
            .map(|(_, f)| f)
    }

    /// Everything from `node` in has `token` painted as `fill` where it is
    /// a background, whatever the theme does (SwiftUI's `.tint`); for what
    /// is there now.
    pub fn set_fill_in(&mut self, node: NodeId, token: Token, fill: Option<Fill>) {
        let mut stack = vec![node];
        while let Some(n) = stack.pop() {
            stack.extend_from_slice(self.children(n));
            let has = self.nodes[n.0].states.as_ref().is_some_and(|s| {
                s.tokens
                    .iter()
                    .any(|(p, t)| {
                        p == "background"
                            && (*t == token || (token, *t) == (Token::Accent, Token::AccentText))
                    })
            });
            if has
                && self.nodes[n.0]
                    .states
                    .as_ref()
                    .is_none_or(|s| s.fill != fill)
            {
                self.states_mut(n).fill = fill.clone();
                self.restyle(n);
            }
        }
    }

    /// `fill` as `s`'s background in place of `token`'s colour.
    pub(crate) fn apply_fill(
        &self,
        s: &mut Style,
        fill: &Fill,
        appearance: Appearance,
        token: Token,
    ) {
        match fill {
            Fill::Gradient(g) => {
                let css =
                    crate::environment::resolve_colors(&g.css(), |t| self.token_in(appearance, t))
                        .into_owned();
                crate::css::apply(s, "background", &css);
            }
            Fill::Glass(m) => glass(
                s,
                *m,
                self.token_in(appearance, token),
                appearance == Appearance::Dark,
            ),
        }
    }

    /// Puts `node` in the middle of its parent, across, down or both, when
    /// the parent lays out in a row or a column.
    pub fn center(&mut self, node: NodeId, across: bool, down: bool) {
        let column = self
            .parent(node)
            .is_none_or(|p| self.style(p).flex_direction != crate::FlexDirection::Row);
        self.update_style(node, |s| center(s, column, across, down));
    }
}

/// Makes `s` sit in the middle of a parent that is a column (or a row).
fn center(s: &mut Style, column: bool, across: bool, down: bool) {
    // Auto margins share what is free on both sides, in either direction;
    // what was stretched across the parent is as wide as its own content.
    if (column && across) || (!column && down) {
        s.align_self = Some(crate::AlignItems::Center);
    }
    if across {
        (s.margin_auto[1], s.margin_auto[3]) = (true, true);
    }
    if down {
        (s.margin_auto[0], s.margin_auto[2]) = (true, true);
    }
}
