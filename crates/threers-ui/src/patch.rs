//! Styles to build, combine, take apart and reuse: a [`Patch`] is part of a
//! style (only the properties it names), with the theme's colours and the
//! looks for hover, press, focus and the dark appearance, written with
//! typed builders, as CSS text, or both.
//!
//! - **Build** with methods (`Patch::new().padding(8.0).radius(6.0)`), or
//!   from CSS (`Patch::css("padding: 8px; border-radius: 6px")`).
//! - **Combine** with `+` or [`Patch::with`]: the right one wins where both
//!   set a property.
//! - **Take apart** with [`Patch::without`] and [`Patch::only`].
//! - **Reuse** as a class: [`Ui::define_class`] names one,
//!   [`Patch::extends`] builds on a class (and classes on classes), and
//!   [`Ui::add_class`] puts it on a node. Redefining a class restyles every
//!   node that has it.
//! - **Apply** to a node with [`Ui::apply`], over what it has.
//!
//! ```
//! use threers_ui::environment::Token;
//! use threers_ui::patch::Patch;
//! use threers_ui::{FixedWidthText, Ui};
//!
//! let mut ui = Ui::new();
//! let card = Patch::new().padding(12.0).radius(10.0).token("background", Token::Surface).column().gap(6.0);
//! let warm = Patch::new().background("#fff4e5").border(1.0, "#f59e0b");
//! ui.define_class("card", card.clone());
//! ui.define_class("warning", Patch::extends("card") + warm);
//!
//! let root = ui.root();
//! let note = ui.add_box(root, Patch::new().width(200.0).into());
//! ui.add_class(note, "warning");
//! assert_eq!(ui.style(note).padding.left, 12.0, "from the card");
//! assert_eq!(ui.style(note).border.left, 1.0, "its own");
//! assert_eq!(ui.style(note).width, threers_ui::Length::Px(200.0), "the node's own kept");
//!
//! // Restyle every card, warnings included.
//! ui.define_class("card", card.padding(20.0));
//! assert_eq!(ui.style(note).padding.left, 20.0);
//! ui.remove_class(note, "warning");
//! assert_eq!(ui.style(note).padding.left, 0.0);
//! ```

use std::collections::HashMap;
use std::ops::Add;

use crate::environment::Token;
use crate::states::Appearance;
use crate::tree::{NodeId, Ui};
use crate::{Rgba, Style};

/// When a [`Patch`]'s state part applies.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum State {
    /// The pointer over it.
    Hover,
    /// Pressed.
    Active,
    /// Focused, however.
    Focus,
    /// Focused from the keyboard.
    FocusVisible,
    Disabled,
    /// In the dark appearance.
    Dark,
    /// In the light appearance.
    Light,
}

/// Part of a style. See the [module docs](self).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Patch {
    /// Classes it builds on, applied first.
    extends: Vec<String>,
    /// CSS declarations, in order: a later one wins.
    decls: Vec<(String, String)>,
    /// Properties coloured by the theme.
    tokens: Vec<(String, Token)>,
    /// What changes in a state.
    states: Vec<(State, Patch)>,
}

/// A colour for a [`Patch`]: CSS text (`"#2563eb"`, `"rgb(…)"`, a name),
/// an [`Rgba`], or `0xRRGGBB`.
pub trait IntoCss {
    fn into_css(self) -> String;
}

impl IntoCss for &str {
    fn into_css(self) -> String {
        self.to_owned()
    }
}

impl IntoCss for String {
    fn into_css(self) -> String {
        self
    }
}

impl IntoCss for Rgba {
    fn into_css(self) -> String {
        let b = |v: f32| (v.clamp(0.0, 1.0) * 255.0).round() as u8;
        format!(
            "#{:02x}{:02x}{:02x}{:02x}",
            b(self[0]),
            b(self[1]),
            b(self[2]),
            b(self[3])
        )
    }
}

impl IntoCss for u32 {
    fn into_css(self) -> String {
        format!("#{:06x}", self & 0xffffff)
    }
}

impl Patch {
    /// Nothing yet.
    pub fn new() -> Self {
        Self::default()
    }

    /// From CSS declarations: `"padding: 8px; border-radius: 6px"`.
    pub fn css(css: &str) -> Self {
        let mut p = Self::new();
        for d in css.split(';') {
            if let Some((n, v)) = d.split_once(':') {
                let (n, v) = (n.trim(), v.trim());
                if !n.is_empty() {
                    p.decls.push((n.to_owned(), v.to_owned()));
                }
            }
        }
        p
    }

    /// Building on the class `name` (as defined when it is applied).
    pub fn extends(name: &str) -> Self {
        Self {
            extends: vec![name.to_owned()],
            ..Self::default()
        }
    }

    /// Sets the CSS property `name` to `value`.
    pub fn set(mut self, name: &str, value: impl Into<String>) -> Self {
        self.decls.retain(|(n, _)| n != name);
        self.decls.push((name.to_owned(), value.into()));
        self
    }

    /// `property` (`background`, `color`, `border-color`) in the theme's
    /// `token` colour, following light and dark.
    pub fn token(mut self, property: &str, token: Token) -> Self {
        self.tokens.retain(|(n, _)| n != property);
        self.tokens.push((property.to_owned(), token));
        self
    }

    /// What changes in `state`: `.on(State::Hover, Patch::new().opacity(0.8))`.
    pub fn on(mut self, state: State, patch: Patch) -> Self {
        match self.states.iter_mut().find(|(s, _)| *s == state) {
            Some((_, p)) => *p = std::mem::take(p) + patch,
            None => self.states.push((state, patch)),
        }
        self
    }

    pub fn hover(self, patch: Patch) -> Self {
        self.on(State::Hover, patch)
    }

    pub fn pressed(self, patch: Patch) -> Self {
        self.on(State::Active, patch)
    }

    pub fn focused(self, patch: Patch) -> Self {
        self.on(State::FocusVisible, patch)
    }

    pub fn disabled(self, patch: Patch) -> Self {
        self.on(State::Disabled, patch)
    }

    pub fn dark(self, patch: Patch) -> Self {
        self.on(State::Dark, patch)
    }

    /// `other` over this: where both set a property, `other`'s.
    pub fn with(mut self, other: Patch) -> Self {
        for c in other.extends {
            if !self.extends.contains(&c) {
                self.extends.push(c);
            }
        }
        for (n, v) in other.decls {
            self = self.set(&n, v);
        }
        for (n, t) in other.tokens {
            self = self.token(&n, t);
        }
        for (s, p) in other.states {
            self = self.on(s, p);
        }
        self
    }

    /// Without the properties `names` (and their theme colours).
    pub fn without(mut self, names: &[&str]) -> Self {
        self.decls.retain(|(n, _)| !names.contains(&n.as_str()));
        self.tokens.retain(|(n, _)| !names.contains(&n.as_str()));
        self
    }

    /// Only the properties `names`: one part of it, to use elsewhere.
    pub fn only(&self, names: &[&str]) -> Patch {
        Patch {
            extends: vec![],
            decls: self
                .decls
                .iter()
                .filter(|(n, _)| names.contains(&n.as_str()))
                .cloned()
                .collect(),
            tokens: self
                .tokens
                .iter()
                .filter(|(n, _)| names.contains(&n.as_str()))
                .cloned()
                .collect(),
            states: vec![],
        }
    }

    /// What it sets, as CSS (its own declarations; not its classes, theme
    /// colours or states).
    pub fn to_css(&self) -> String {
        self.decls
            .iter()
            .map(|(n, v)| format!("{n}: {v}"))
            .collect::<Vec<_>>()
            .join("; ")
    }

    /// The value it sets `name` to, if it does.
    pub fn get(&self, name: &str) -> Option<&str> {
        self.decls
            .iter()
            .rev()
            .find(|(n, _)| n == name)
            .map(|(_, v)| v.as_str())
    }

    /// Its declarations over `style`.
    pub fn apply_to(&self, style: &mut Style) {
        for (n, v) in &self.decls {
            crate::css::apply(style, n, v);
        }
    }

    // Typed builders for what is set most.

    pub fn width(self, px: f32) -> Self {
        self.set("width", format!("{px}px"))
    }
    pub fn height(self, px: f32) -> Self {
        self.set("height", format!("{px}px"))
    }
    pub fn size(self, w: f32, h: f32) -> Self {
        self.width(w).height(h)
    }
    pub fn fill_width(self) -> Self {
        self.set("width", "100%")
    }
    pub fn padding(self, px: f32) -> Self {
        self.set("padding", format!("{px}px"))
    }
    pub fn padding_xy(self, x: f32, y: f32) -> Self {
        self.set("padding", format!("{y}px {x}px"))
    }
    pub fn margin(self, px: f32) -> Self {
        self.set("margin", format!("{px}px"))
    }
    pub fn gap(self, px: f32) -> Self {
        self.set("gap", format!("{px}px"))
    }
    pub fn radius(self, px: f32) -> Self {
        self.set("border-radius", format!("{px}px"))
    }
    /// Round ends: half its shorter side.
    pub fn pill(self) -> Self {
        self.set("border-radius", "9999px")
    }
    pub fn background(self, c: impl IntoCss) -> Self {
        self.set("background", c.into_css())
    }
    /// A gradient from left to right through `colours`, spread evenly.
    pub fn gradient(self, colours: &[&str]) -> Self {
        self.set(
            "background",
            format!("linear-gradient(to right, {})", colours.join(", ")),
        )
    }
    pub fn color(self, c: impl IntoCss) -> Self {
        self.set("color", c.into_css())
    }
    pub fn border(self, width: f32, c: impl IntoCss) -> Self {
        self.set("border", format!("{width}px solid {}", c.into_css()))
    }
    /// A soft shadow below, `blur` pixels across.
    pub fn shadow(self, blur: f32) -> Self {
        self.set(
            "box-shadow",
            format!("0 {}px {blur}px rgba(0,0,0,0.25)", (blur * 0.3).round()),
        )
    }
    pub fn opacity(self, o: f32) -> Self {
        self.set("opacity", format!("{o}"))
    }
    pub fn font_size(self, px: f32) -> Self {
        self.set("font-size", format!("{px}px"))
    }
    pub fn font(self, family: &str) -> Self {
        self.set("font-family", family.to_owned())
    }
    pub fn bold(self) -> Self {
        self.set("font-weight", "bold")
    }
    pub fn weight(self, w: u16) -> Self {
        self.set("font-weight", format!("{w}"))
    }
    pub fn italic(self) -> Self {
        self.set("font-style", "italic")
    }
    pub fn align_text(self, align: &str) -> Self {
        self.set("text-align", align.to_owned())
    }
    /// Its children in a row.
    pub fn row(self) -> Self {
        self.set("display", "flex").set("flex-direction", "row")
    }
    /// Its children in a column.
    pub fn column(self) -> Self {
        self.set("display", "flex").set("flex-direction", "column")
    }
    /// Its children in its middle, both ways.
    pub fn center(self) -> Self {
        self.set("display", "flex")
            .set("align-items", "center")
            .set("justify-content", "center")
    }
    pub fn grow(self) -> Self {
        self.set("flex-grow", "1")
    }
    pub fn wrap(self) -> Self {
        self.set("flex-wrap", "wrap")
    }
    pub fn clip(self) -> Self {
        self.set("overflow", "hidden")
    }
}

/// Steps of space, in UI pixels, on a 4-pixel grid: the same few gaps,
/// paddings and margins everywhere keep a UI in rhythm.
pub struct Space;

impl Space {
    pub const NONE: f32 = 0.0;
    pub const XXS: f32 = 2.0;
    pub const XS: f32 = 4.0;
    pub const S: f32 = 8.0;
    pub const M: f32 = 12.0;
    pub const L: f32 = 16.0;
    pub const XL: f32 = 24.0;
    pub const XXL: f32 = 32.0;
    pub const XXXL: f32 = 48.0;
}

/// How a box's children line up across its main direction (CSS
/// `align-items`), or how one child does (`align-self`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Align {
    Start,
    Center,
    End,
    /// As long as the box is across.
    Stretch,
}

impl Align {
    fn css(self) -> &'static str {
        match self {
            Align::Start => "flex-start",
            Align::Center => "center",
            Align::End => "flex-end",
            Align::Stretch => "stretch",
        }
    }
}

/// How a box's children share its main direction (CSS `justify-content`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Justify {
    Start,
    Center,
    End,
    /// The first at the start, the last at the end, the rest evenly between.
    Between,
    /// Equal room round each.
    Around,
    /// Equal room between each and at both ends.
    Evenly,
}

impl Justify {
    fn css(self) -> &'static str {
        match self {
            Justify::Start => "flex-start",
            Justify::Center => "center",
            Justify::End => "flex-end",
            Justify::Between => "space-between",
            Justify::Around => "space-around",
            Justify::Evenly => "space-evenly",
        }
    }
}

/// The `justify-content` placing children as `a` places them across.
fn justify_of(a: Align) -> Justify {
    match a {
        Align::Start | Align::Stretch => Justify::Start,
        Align::Center => Justify::Center,
        Align::End => Justify::End,
    }
}

/// The four sides of a `padding` or `margin` it sets (top, right, bottom,
/// left), from its shorthand.
fn sides(p: &Patch, name: &str) -> [f32; 4] {
    let v: Vec<f32> = p
        .get(name)
        .unwrap_or("0")
        .split_whitespace()
        .filter_map(|t| t.trim_end_matches("px").parse().ok())
        .collect();
    match v.as_slice() {
        [a] => [*a; 4],
        [y, x] => [*y, *x, *y, *x],
        [t, x, b] => [*t, *x, *b, *x],
        [t, r, b, l] => [*t, *r, *b, *l],
        _ => [0.0; 4],
    }
}

fn with_sides(p: Patch, name: &str, f: impl FnOnce(&mut [f32; 4])) -> Patch {
    let mut s = sides(&p, name);
    f(&mut s);
    p.set(name, format!("{}px {}px {}px {}px", s[0], s[1], s[2], s[3]))
}

/// Spacing and flexbox, each side on its own (they compose with the
/// shorthands: `.padding(8.0).padding_top(16.0)`).
impl Patch {
    pub fn padding_x(self, px: f32) -> Self {
        with_sides(self, "padding", |s| (s[1], s[3]) = (px, px))
    }
    pub fn padding_y(self, px: f32) -> Self {
        with_sides(self, "padding", |s| (s[0], s[2]) = (px, px))
    }
    pub fn padding_top(self, px: f32) -> Self {
        with_sides(self, "padding", |s| s[0] = px)
    }
    pub fn padding_right(self, px: f32) -> Self {
        with_sides(self, "padding", |s| s[1] = px)
    }
    pub fn padding_bottom(self, px: f32) -> Self {
        with_sides(self, "padding", |s| s[2] = px)
    }
    pub fn padding_left(self, px: f32) -> Self {
        with_sides(self, "padding", |s| s[3] = px)
    }
    /// Top, right, bottom, left.
    pub fn padding_sides(self, top: f32, right: f32, bottom: f32, left: f32) -> Self {
        with_sides(self, "padding", |s| *s = [top, right, bottom, left])
    }
    pub fn margin_x(self, px: f32) -> Self {
        with_sides(self, "margin", |s| (s[1], s[3]) = (px, px))
    }
    pub fn margin_y(self, px: f32) -> Self {
        with_sides(self, "margin", |s| (s[0], s[2]) = (px, px))
    }
    pub fn margin_top(self, px: f32) -> Self {
        with_sides(self, "margin", |s| s[0] = px)
    }
    pub fn margin_right(self, px: f32) -> Self {
        with_sides(self, "margin", |s| s[1] = px)
    }
    pub fn margin_bottom(self, px: f32) -> Self {
        with_sides(self, "margin", |s| s[2] = px)
    }
    pub fn margin_left(self, px: f32) -> Self {
        with_sides(self, "margin", |s| s[3] = px)
    }
    /// Top, right, bottom, left.
    pub fn margin_sides(self, top: f32, right: f32, bottom: f32, left: f32) -> Self {
        with_sides(self, "margin", |s| *s = [top, right, bottom, left])
    }
    /// Its children across its main direction.
    pub fn align(self, a: Align) -> Self {
        self.set("align-items", a.css())
    }
    /// Itself across its parent's main direction.
    pub fn align_self(self, a: Align) -> Self {
        self.set("align-self", a.css())
    }
    /// Its children along its main direction.
    pub fn justify(self, j: Justify) -> Self {
        self.set("justify-content", j.css())
    }
    pub fn row_gap(self, px: f32) -> Self {
        self.set("row-gap", format!("{px}px"))
    }
    pub fn column_gap(self, px: f32) -> Self {
        self.set("column-gap", format!("{px}px"))
    }
    /// How much it gives up when there is too little room (0: none).
    pub fn shrink(self, f: f32) -> Self {
        self.set("flex-shrink", format!("{f}"))
    }
    /// Never smaller than its content along the row or column.
    pub fn no_shrink(self) -> Self {
        self.shrink(0.0)
    }
    pub fn basis(self, px: f32) -> Self {
        self.set("flex-basis", format!("{px}px"))
    }
    pub fn min_width(self, px: f32) -> Self {
        self.set("min-width", format!("{px}px"))
    }
    pub fn max_width(self, px: f32) -> Self {
        self.set("max-width", format!("{px}px"))
    }
    pub fn min_height(self, px: f32) -> Self {
        self.set("min-height", format!("{px}px"))
    }
    pub fn max_height(self, px: f32) -> Self {
        self.set("max-height", format!("{px}px"))
    }
    /// Lines of text this far apart, as a multiple of the font size.
    pub fn line_height(self, times: f32) -> Self {
        self.set("line-height", format!("{times}"))
    }
    /// Text kept on one line, cut with an ellipsis if it does not fit.
    pub fn one_line(self) -> Self {
        self.set("white-space", "nowrap")
            .set("overflow", "hidden")
            .set("text-overflow", "ellipsis")
    }
    /// Its children placed across the page's width, whichever way they
    /// run: in a row that is along it (`justify-content`), in a column
    /// across it (`align-items`). A patch that sets no direction is a row.
    pub fn align_horizontal(self, a: Align) -> Self {
        if self.is_column() {
            self.align(a)
        } else {
            self.justify(justify_of(a))
        }
    }

    /// Its children placed up and down the page, whichever way they run:
    /// in a row across it (`align-items`: a row's text and icons lined up
    /// through their middles with `Align::Center`), in a column along it.
    pub fn align_vertical(self, a: Align) -> Self {
        if self.is_column() {
            self.justify(justify_of(a))
        } else {
            self.align(a)
        }
    }

    /// Its children in its middle, both ways (a flex box).
    pub fn center_both(self) -> Self {
        let d = if self.is_column() { "column" } else { "row" };
        self.set("display", "flex")
            .set("flex-direction", d)
            .set("align-items", "center")
            .set("justify-content", "center")
    }

    fn is_column(&self) -> bool {
        self.get("flex-direction")
            .is_some_and(|d| d.starts_with("column"))
    }

    /// A column, `gap` between its children.
    pub fn stack(self, gap: f32) -> Self {
        self.column().gap(gap)
    }
    /// A row, `gap` between its children, lined up through their middles
    /// (text and icons alike).
    pub fn hstack(self, gap: f32) -> Self {
        self.row().gap(gap).align(Align::Center)
    }
}

impl Add for Patch {
    type Output = Patch;
    fn add(self, other: Patch) -> Patch {
        self.with(other)
    }
}

impl From<&str> for Patch {
    fn from(css: &str) -> Self {
        Patch::css(css)
    }
}

/// A patch as a whole style (over the defaults), to build a node with.
impl From<Patch> for Style {
    fn from(p: Patch) -> Style {
        let mut s = Style::default();
        p.apply_to(&mut s);
        s
    }
}

/// Classes and the nodes that have them.
#[derive(Clone, Default)]
pub(crate) struct Classes {
    pub defined: HashMap<String, Patch>,
    /// Each classed node: its style before its classes, and their names.
    pub nodes: HashMap<NodeId, (Style, Vec<String>)>,
}

impl Ui {
    /// `patch` over `node`'s style: its declarations, its theme colours, and
    /// its looks in states (the classes it [extends](Patch::extends)
    /// first).
    pub fn apply(&mut self, node: NodeId, patch: &Patch) {
        let flat = self.resolve(patch, 0);
        let mut style = self.style(node).clone();
        flat.apply_to(&mut style);
        self.set_style(node, style);
        self.apply_extras(node, &flat);
    }

    /// Names `patch` as the class `name`, restyling every node that has it
    /// (or a class built on it).
    pub fn define_class(&mut self, name: &str, patch: Patch) {
        self.env.classes.defined.insert(name.to_owned(), patch);
        let nodes: Vec<NodeId> = self.env.classes.nodes.keys().copied().collect();
        for n in nodes {
            if self.contains(n) {
                self.restyle_classes(n);
            } else {
                self.env.classes.nodes.remove(&n);
            }
        }
    }

    /// The class `name`, as defined.
    pub fn class(&self, name: &str) -> Option<&Patch> {
        self.env.classes.defined.get(name)
    }

    /// Gives `node` the class `name` (over its own style and the classes it
    /// has, in the order they were added).
    pub fn add_class(&mut self, node: NodeId, name: &str) {
        let own = self.style(node).clone();
        let entry = self
            .env
            .classes
            .nodes
            .entry(node)
            .or_insert_with(|| (own, vec![]));
        if !entry.1.iter().any(|c| c == name) {
            entry.1.push(name.to_owned());
        }
        self.restyle_classes(node);
    }

    /// Takes the class `name` off `node`: its own style back, with the
    /// classes it still has. (Theme colours and state looks a class added
    /// stay until replaced.)
    pub fn remove_class(&mut self, node: NodeId, name: &str) {
        if let Some(entry) = self.env.classes.nodes.get_mut(&node) {
            entry.1.retain(|c| c != name);
            self.restyle_classes(node);
        }
    }

    pub fn has_class(&self, node: NodeId, name: &str) -> bool {
        self.env
            .classes
            .nodes
            .get(&node)
            .is_some_and(|(_, c)| c.iter().any(|c| c == name))
    }

    /// `node`'s style: its own, with its classes over it.
    fn restyle_classes(&mut self, node: NodeId) {
        let Some((own, names)) = self.env.classes.nodes.get(&node).cloned() else {
            return;
        };
        let mut style = own;
        let mut extras = Patch::new();
        for name in &names {
            let flat = self.resolve(&Patch::extends(name), 0);
            flat.apply_to(&mut style);
            extras = extras + flat;
        }
        self.set_style(node, style);
        self.apply_extras(node, &extras);
    }

    /// `patch` with the classes it extends folded in, theirs first.
    fn resolve(&self, patch: &Patch, depth: usize) -> Patch {
        let mut out = Patch::new();
        if depth < 16 {
            for c in &patch.extends {
                if let Some(p) = self.env.classes.defined.get(c) {
                    out = out + self.resolve(p, depth + 1);
                }
            }
        }
        out + Patch {
            extends: vec![],
            ..patch.clone()
        }
    }

    /// A resolved patch's theme colours and state looks onto `node`.
    fn apply_extras(&mut self, node: NodeId, flat: &Patch) {
        if !flat.tokens.is_empty() {
            let tokens: Vec<(&str, Token)> =
                flat.tokens.iter().map(|(n, t)| (n.as_str(), *t)).collect();
            self.token_style(node, &tokens);
        }
        for (state, p) in &flat.states {
            let css = p.to_css();
            match state {
                State::Hover => self.hover_style(node, &css),
                State::Active => self.active_style(node, &css),
                State::Focus => self.focus_style(node, &css),
                State::FocusVisible => self.focus_visible_style(node, &css),
                State::Disabled => self.disabled_style(node, &css),
                State::Dark => self.appearance_style(node, Appearance::Dark, &css),
                State::Light => self.appearance_style(node, Appearance::Light, &css),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_right_one_wins_and_parts_come_away() {
        let a = Patch::new().padding(4.0).radius(2.0).background("#000");
        let b = Patch::css("padding: 8px; color: red");
        let c = a.clone() + b;
        assert_eq!(c.get("padding"), Some("8px"));
        assert_eq!(c.get("border-radius"), Some("2px"));
        assert_eq!(c.get("color"), Some("red"));
        let shape = c.only(&["border-radius", "padding"]);
        assert_eq!(shape.to_css(), "border-radius: 2px; padding: 8px");
        assert_eq!(c.without(&["padding"]).get("padding"), None);
    }

    #[test]
    fn colours_in_any_form() {
        let s: Style = Patch::new()
            .background([1.0, 0.0, 0.0, 0.5])
            .color(0x00ff00)
            .into();
        assert_eq!(s.background, Some([1.0, 0.0, 0.0, 128.0 / 255.0]));
        assert_eq!(s.color, [0.0, 1.0, 0.0, 1.0]);
    }

    #[test]
    fn classes_build_on_classes_and_states_apply() {
        let mut ui = Ui::new();
        ui.define_class(
            "base",
            Patch::new().padding(2.0).hover(Patch::new().opacity(0.5)),
        );
        ui.define_class("mid", Patch::extends("base").radius(4.0));
        ui.define_class("top", Patch::extends("mid").padding(6.0));
        // A loop is cut, not followed for ever.
        ui.define_class("loop", Patch::extends("loop").gap(3.0));
        let root = ui.root();
        let n = ui.add_box(root, Style::default());
        ui.add_class(n, "top");
        ui.add_class(n, "loop");
        assert_eq!(ui.style(n).padding.left, 6.0);
        assert_eq!(ui.style(n).border_radius, crate::Length::Px(4.0));
        assert!(!ui.has_class(n, "mid") && ui.has_class(n, "top"));
    }

    #[test]
    fn sides_compose_with_the_shorthand() {
        let p = Patch::new()
            .padding(8.0)
            .padding_top(16.0)
            .padding_x(Space::XS);
        assert_eq!(p.get("padding"), Some("16px 4px 8px 4px"));
        let s: Style = p
            .margin_y(Space::S)
            .hstack(Space::M)
            .justify(Justify::Between)
            .into();
        assert_eq!(
            (
                s.padding.top,
                s.padding.right,
                s.padding.bottom,
                s.padding.left
            ),
            (16.0, 4.0, 8.0, 4.0)
        );
        assert_eq!((s.margin.top, s.margin.left), (8.0, 0.0));
        assert_eq!(s.align_items, crate::AlignItems::Center);
        assert_eq!(s.justify_content, crate::JustifyContent::SpaceBetween);
    }

    #[test]
    fn vertical_and_horizontal_follow_the_direction() {
        let row: Style = Patch::new()
            .row()
            .align_vertical(Align::End)
            .align_horizontal(Align::Center)
            .into();
        assert_eq!(
            (row.align_items, row.justify_content),
            (crate::AlignItems::FlexEnd, crate::JustifyContent::Center)
        );
        let col: Style = Patch::new()
            .column()
            .align_vertical(Align::End)
            .align_horizontal(Align::Center)
            .into();
        assert_eq!(
            (col.align_items, col.justify_content),
            (crate::AlignItems::Center, crate::JustifyContent::FlexEnd)
        );
    }
}
