//! SwiftUI-style views and modifiers for [`El`]: stacks, spacers and
//! dividers; text, labels, buttons, toggles, pickers, links and progress; group
//! boxes, forms and sections, labelled values, lists, text fields and
//! editors (which keep their text, caret and focus), and what shows when
//! there is nothing ([`content_unavailable`]) — each with its role for
//! assistive technology, coloured by the theme's tokens; and the modifiers
//! that adjust any of them (`.padding(8.0)`, `.font(Font::Headline)`,
//! `.foreground(Token::SecondaryText)`, `.disabled(true)`, `.help(…)`,
//! `.when(cond, …)`, …), and the containers that arrange them: `for_each`
//! by identity, `disclosure_group`, `adaptive_grid`, `title_bar`, an inline
//! `confirmation`.
//!
//! ```
//! use threers_ui::environment::Token;
//! use threers_ui::input::Input;
//! use threers_ui::view::{button, hstack, spacer, text, toggle, vstack, El, Font, View};
//! use threers_ui::{FixedWidthText, Ui};
//!
//! #[derive(Default)]
//! struct Settings { sounds: bool, saved: u32 }
//!
//! fn body(s: &Settings) -> Vec<El<Settings>> {
//!     vec![vstack(8.0, [
//!         text("Settings").font(Font::Title),
//!         toggle("Sounds", s.sounds, |s: &mut Settings, on| s.sounds = on),
//!         hstack(8.0, [
//!             text(format!("Saved {} times", s.saved)).foreground(Token::SecondaryText),
//!             spacer(),
//!             button("Save", |s: &mut Settings| s.saved += 1),
//!         ]),
//!     ])
//!     .padding(16.0)]
//! }
//!
//! let mut ui = Ui::new();
//! let mut input: Input<Settings> = Input::new();
//! let root = ui.root();
//! let mut view = View::new(&mut ui, root, "");
//! view.update(&mut ui, &mut input, body(&Settings::default()));
//! ui.layout(400.0, 300.0, &mut FixedWidthText(7.0));
//! ```
//!
//! A modifier on [`El::label`] (a bare text node) has nothing to style:
//! [`text`] is text in a box, which the modifiers style.

use super::El;
use crate::components::ButtonKind;
use crate::environment::Token;
use crate::{Role, Semantics};

/// Where a column's children line up across it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum HorizontalAlignment {
    Leading,
    #[default]
    Center,
    Trailing,
}

/// Where a row's children line up across it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum VerticalAlignment {
    Top,
    #[default]
    Center,
    Bottom,
}

/// Which sides [`El::padding_edges`] pads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Edge {
    All,
    /// Left and right.
    Horizontal,
    /// Top and bottom.
    Vertical,
    Top,
    Bottom,
}

/// Text styles by purpose, as SwiftUI names them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Font {
    LargeTitle,
    Title,
    Title2,
    Title3,
    /// Bold, at body size.
    Headline,
    Body,
    Callout,
    Subheadline,
    Footnote,
    Caption,
    Caption2,
}

impl Font {
    /// Its size in UI pixels and weight.
    pub fn size_and_weight(self) -> (f32, u16) {
        match self {
            Font::LargeTitle => (28.0, 700),
            Font::Title => (22.0, 700),
            Font::Title2 => (18.0, 700),
            Font::Title3 => (16.0, 600),
            Font::Headline => (14.0, 600),
            Font::Body => (14.0, 400),
            Font::Callout => (13.0, 400),
            Font::Subheadline => (12.0, 400),
            Font::Footnote => (11.0, 400),
            Font::Caption => (11.0, 400),
            Font::Caption2 => (10.0, 400),
        }
    }
}

fn align_items(h: HorizontalAlignment) -> &'static str {
    match h {
        HorizontalAlignment::Leading => "flex-start",
        HorizontalAlignment::Center => "center",
        HorizontalAlignment::Trailing => "flex-end",
    }
}

/// The modifiers.
impl<S: 'static> El<S> {
    /// `decls` after its own declarations (a later one wins).
    pub fn css(mut self, decls: &str) -> Self {
        let own = self.css.trim_end();
        if !own.is_empty() && !own.ends_with(';') {
            self.css = format!("{own};");
        }
        if !self.css.is_empty() {
            self.css.push(' ');
        }
        self.css.push_str(decls);
        self
    }

    /// `px` of room inside it, on every side.
    pub fn padding(self, px: f32) -> Self {
        self.padding_edges(Edge::All, px)
    }

    /// `px` of room inside it on `edges`.
    pub fn padding_edges(self, edges: Edge, px: f32) -> Self {
        let p = |side: &str| format!("padding-{side}: {px}px");
        let decls = match edges {
            Edge::All => vec![format!("padding: {px}px")],
            Edge::Horizontal => vec![p("left"), p("right")],
            Edge::Vertical => vec![p("top"), p("bottom")],
            Edge::Top => vec![p("top")],
            Edge::Bottom => vec![p("bottom")],
        };
        self.css(&decls.join("; "))
    }

    /// Its size: `width` and `height` in UI pixels, each as its content
    /// makes it when `None`.
    pub fn frame(self, width: Option<f32>, height: Option<f32>) -> Self {
        let mut d = vec![];
        if let Some(w) = width {
            d.push(format!("width: {w}px"));
        }
        if let Some(h) = height {
            d.push(format!("height: {h}px"));
        }
        self.css(&d.join("; "))
    }

    /// As wide as its parent lets it be (SwiftUI's
    /// `.frame(maxWidth: .infinity)`).
    pub fn fill_width(self) -> Self {
        self.css("flex-grow: 1; align-self: stretch; min-width: 0")
    }

    /// Behind it, in the theme's `token` colour.
    pub fn background(self, token: Token) -> Self {
        self.token("background", token)
    }

    /// Its text in the theme's `token` colour.
    pub fn foreground(self, token: Token) -> Self {
        self.token("color", token)
    }

    /// `property` (`background`, `color`, `border-color`) in a colour that
    /// is no theme token (a status's amber, a swatch): CSS text, in place of
    /// any token it had there (a token would win over it).
    pub fn fixed_color(mut self, property: &'static str, css: &str) -> Self {
        self.tokens.retain(|(p, _)| *p != property);
        self.css(&format!("{property}: {css}"))
    }

    /// A line of `width` px round it, in the theme's `token` colour.
    pub fn border(self, token: Token, width: f32) -> Self {
        self.css(&format!("border: {width}px solid"))
            .token("border-color", token)
    }

    pub fn corner_radius(self, px: f32) -> Self {
        self.css(&format!("border-radius: {px}px"))
    }

    pub fn font(self, font: Font) -> Self {
        let (size, weight) = font.size_and_weight();
        self.css(&format!("font-size: {size}px; font-weight: {weight}"))
    }

    pub fn bold(self) -> Self {
        self.css("font-weight: 600")
    }

    pub fn opacity(self, o: f32) -> Self {
        self.css(&format!("opacity: {o}"))
    }

    /// Its text wraps onto more lines (SwiftUI's `.lineLimit(nil)`), at
    /// the width its stack gives it: across a column (even a leading-aligned
    /// one), and shrinking in a row.
    pub fn wraps(self) -> Self {
        self.css("white-space: normal; align-self: stretch; min-width: 0")
    }

    /// It as a row's child: what [`fill_width`](Self::fill_width) or
    /// [`wraps`](Self::wraps) stretched is as tall as its own content and
    /// lines up as the row says (SwiftUI's: they say nothing of height).
    /// What shows how tall it is (a background, a border), scrolls or is
    /// hosted stays stretched.
    pub(super) fn in_row(mut self) -> Self {
        let shows = ["background", "border", "overflow"];
        if matches!(self.kind, super::Kind::Box)
            && !self.children.is_empty()
            && self.css.contains("align-self: stretch")
            && self.tokens.iter().all(|(p, _)| *p == "color")
            && !shows.iter().any(|p| self.css.contains(p))
        {
            self.css.push_str("; align-self: auto");
        }
        self
    }

    /// Where a stack's children line up across a column.
    pub fn alignment(self, a: HorizontalAlignment) -> Self {
        self.css(&format!("align-items: {}", align_items(a)))
    }

    /// Where a stack's children line up across a row.
    pub fn vertical_alignment(self, a: VerticalAlignment) -> Self {
        let v = match a {
            VerticalAlignment::Top => "flex-start",
            VerticalAlignment::Center => "center",
            VerticalAlignment::Bottom => "flex-end",
        };
        self.css(&format!("align-items: {v}"))
    }

    /// Unusable when `disabled`: dimmed, not pressed, skipped by Tab and
    /// said to be so.
    pub fn disabled(mut self, disabled: bool) -> Self {
        if !disabled {
            return self;
        }
        self.on_click = None;
        self.hover = None;
        let s = self
            .semantics
            .take()
            .unwrap_or_else(|| Semantics::new(Role::Generic));
        self.semantics = Some(s.disabled(true));
        // A filled button faded is its label faded into it: a quiet fill
        // and a label that reads, as `components::button`'s.
        if self
            .tokens
            .iter()
            .any(|(p, t)| *p == "background" && *t == Token::Prominent)
        {
            return self
                .background(Token::Separator)
                .foreground(Token::SecondaryText)
                .token("border-color", Token::Border)
                .css("border: 1px solid; cursor: default");
        }
        self.css("opacity: 0.65; cursor: default")
    }

    /// There, taking its room, but not seen nor said.
    pub fn hidden(mut self) -> Self {
        self.semantics = Some(Semantics::new(Role::Generic).hidden());
        self.on_click = None;
        self.css("opacity: 0")
    }

    /// The name assistive technology says for it.
    pub fn accessibility_label(mut self, label: impl Into<String>) -> Self {
        let s = self
            .semantics
            .take()
            .unwrap_or_else(|| Semantics::new(Role::Generic));
        self.semantics = Some(s.label(label.into()));
        self
    }

    /// A button's look ([`ButtonKind`]): its colours and its looks under the
    /// pointer, in place of the ones it had.
    pub fn button_style(mut self, kind: ButtonKind) -> Self {
        let (css, tokens) = kind.css();
        self.tokens = tokens;
        self.hover = Some(kind.state_css().0.to_owned());
        // Text alone lines up with the text around it in a column.
        let text_only = matches!(kind, ButtonKind::Plain | ButtonKind::DestructivePlain);
        let disabled = self.semantics.as_ref().is_some_and(|s| s.disabled);
        self.css(css)
            .line_up_in_column(if text_only { 8.0 } else { 0.0 })
            .disabled(disabled)
    }
}

/// Modifiers that compose others.
impl<S: 'static> El<S> {
    /// `f` applied to it (a group of modifiers kept together, reused).
    pub fn modify(self, f: impl FnOnce(Self) -> Self) -> Self {
        f(self)
    }

    /// `f` applied to it when `cond` holds (a modifier only sometimes:
    /// `.when(chosen, |e| e.foreground(Token::Accent))`).
    pub fn when(self, cond: bool, f: impl FnOnce(Self) -> Self) -> Self {
        if cond {
            f(self)
        } else {
            self
        }
    }

    /// `f` applied with the value when there is one (`.when_some(action,
    /// |e, a| e.on_click(…))`).
    pub fn when_some<T>(self, value: Option<T>, f: impl FnOnce(Self, T) -> Self) -> Self {
        match value {
            Some(v) => f(self, v),
            None => self,
        }
    }

    /// A small pill of `text` after it (a count, "new"), as SwiftUI's
    /// `.badge`; none when `text` is empty.
    pub fn badge(self, text: impl Into<String>) -> Self {
        let text = text.into();
        if text.is_empty() {
            return self;
        }
        hstack(
            8.0,
            [
                self.fill_width(),
                El::div("padding: 1px 7px; border-radius: 9px; font-size: 11px; font-weight: 700; flex-shrink: 0")
                    .background(Token::Accent)
                    .foreground(Token::OnAccent)
                    .text(text),
            ],
        )
        .css("align-self: stretch")
    }

    /// `view` over it, centred (SwiftUI's `.overlay`): it keeps its size,
    /// `view` takes none of its own.
    pub fn overlay(self, view: El<S>) -> Self {
        zstack([self, view])
    }
}

/// A view for each of `items`, each known by `id(item)` (SwiftUI's
/// `ForEach`): as items come, go and move, each view keeps its own nodes
/// (a field its text, a row its state) instead of taking its neighbour's.
pub fn for_each<S: 'static, T>(
    items: impl IntoIterator<Item = T>,
    id: impl Fn(&T) -> String,
    view: impl Fn(T) -> El<S>,
) -> Vec<El<S>> {
    items
        .into_iter()
        .map(|item| {
            let key = id(&item);
            view(item).key(key)
        })
        .collect()
}

/// A heading that shows `content` under it while `expanded` (SwiftUI's
/// `DisclosureGroup`): pressed, it calls `toggle` with the other state. A
/// button to assistive technology, its state `expanded`.
pub fn disclosure_group<S: 'static>(
    title: impl Into<String>,
    expanded: bool,
    toggle: impl Fn(&mut S, bool) + 'static,
    content: impl IntoIterator<Item = El<S>>,
) -> El<S> {
    let title = title.into();
    let head = hstack(
        6.0,
        [
            chevron(
                if expanded { Chevron::Down } else { Chevron::Forward },
                12.0,
            )
            .foreground(Token::SecondaryText),
            text(title.clone()).bold(),
        ],
    )
    .css("cursor: pointer; align-self: stretch; border-radius: 6px; padding: 4px 0")
    .hover("background: var(--hover)")
    .semantics(Semantics::new(Role::Button).label(title).expanded(expanded))
    .on_click(move |s, _| toggle(s, !expanded));
    let mut parts = vec![head];
    if expanded {
        parts.extend(content);
    }
    vstack(8.0, parts)
        .alignment(HorizontalAlignment::Leading)
        .css("align-self: stretch")
}

/// `children` in as many columns as fit, each at least `min_width` wide
/// and sharing what is left (SwiftUI's `LazyVGrid` with adaptive columns),
/// `spacing` apart both ways; a short last row keeps the columns' width.
pub fn adaptive_grid<S: 'static>(
    min_width: f32,
    spacing: f32,
    children: impl IntoIterator<Item = El<S>>,
) -> El<S> {
    El::div(&format!(
        "display: grid; grid-template-columns: repeat(auto-fill, minmax({min_width}px, 1fr)); gap: {spacing}px; \
         align-self: stretch"
    ))
    .children(children)
}

/// A page's title (and `subtitle` under it), its `trailing` actions at the
/// end of the line (SwiftUI's navigation title with toolbar items).
pub fn title_bar<S: 'static>(
    title: impl Into<String>,
    subtitle: impl Into<String>,
    trailing: impl IntoIterator<Item = El<S>>,
) -> El<S> {
    let (title, subtitle) = (title.into(), subtitle.into());
    let mut head = vec![text(title.clone())
        .font(Font::Title)
        .semantics(Semantics::new(Role::Heading(1)).label(title))];
    if !subtitle.is_empty() {
        head.push(
            text(subtitle)
                .font(Font::Subheadline)
                .foreground(Token::SecondaryText)
                .wraps(),
        );
    }
    let mut row = vec![vstack(4.0, head)
        .alignment(HorizontalAlignment::Leading)
        .fill_width()];
    row.extend(trailing);
    hstack(8.0, row)
        .vertical_alignment(VerticalAlignment::Top)
        .css("align-self: stretch")
}

/// "Delete this?" in place (SwiftUI's `confirmationDialog`, inline): the
/// `question`, said at once, then `cancel` and the destructive `yes`, the
/// safe choice first.
pub fn confirmation<S: 'static>(
    question: impl Into<String>,
    cancel: impl Into<String>,
    yes: impl Into<String>,
    on_cancel: impl Fn(&mut S) + 'static,
    on_yes: impl Fn(&mut S) + 'static,
) -> El<S> {
    let question = question.into();
    vstack(
        4.0,
        [
            text(question.clone())
                .css("font-size: 12px")
                .foreground(Token::Destructive)
                .wraps()
                .semantics(Semantics::new(Role::Alert).label(question)),
            hstack(
                8.0,
                [
                    button(cancel, on_cancel).button_style(ButtonKind::Plain),
                    button(yes, on_yes).button_style(ButtonKind::Destructive),
                ],
            ),
        ],
    )
    .alignment(HorizontalAlignment::Leading)
    .css("align-self: stretch; min-width: 0")
}

/// A column: `children` one under another, `spacing` apart, centred across
/// (see [`El::alignment`]).
pub fn vstack<S: 'static>(spacing: f32, children: impl IntoIterator<Item = El<S>>) -> El<S> {
    El::div(&format!(
        "display: flex; flex-direction: column; align-items: center; gap: {spacing}px"
    ))
    .children(children)
}

/// A row: `children` side by side, `spacing` apart, lined up through their
/// middles (see [`El::vertical_alignment`]).
pub fn hstack<S: 'static>(spacing: f32, children: impl IntoIterator<Item = El<S>>) -> El<S> {
    El::div(&format!(
        "display: flex; flex-direction: row; align-items: center; gap: {spacing}px"
    ))
    .children(children.into_iter().map(El::in_row))
}

/// `children` over one another (the first at the back), each centred, the
/// stack as big as the biggest.
pub fn zstack<S: 'static>(children: impl IntoIterator<Item = El<S>>) -> El<S> {
    El::div("display: grid").children(children.into_iter().map(|c| {
        El::div("grid-column: 1; grid-row: 1; display: flex; align-items: center; justify-content: center")
            .child(c)
    }))
}

/// Room that grows to take what its stack has left, pushing what follows
/// to the end.
pub fn spacer<S: 'static>() -> El<S> {
    El::div("flex-grow: 1; min-width: 0; min-height: 0")
}

/// A thin line across a column (or down a row), in the theme's separator
/// colour.
pub fn divider<S: 'static>() -> El<S> {
    El::div("align-self: stretch; min-width: 1px; min-height: 1px; flex-shrink: 0")
        .background(Token::Separator)
        .semantics(Semantics::new(Role::Separator))
}

/// Text, in a box the modifiers style.
pub fn text<S: 'static>(s: impl Into<String>) -> El<S> {
    El::div("").text(s)
}

/// A `title` after its `symbol` (an emoji, a glyph), as SwiftUI's `Label`.
pub fn label<S: 'static>(title: impl Into<String>, symbol: impl Into<String>) -> El<S> {
    let title = title.into();
    hstack(
        6.0,
        [text(symbol).accessibility_label(""), text(title.clone())],
    )
    .accessibility_label(title)
}

/// A button: `title`, running `action` when pressed (a click, a tap, Space
/// or Enter). Outlined; see [`El::button_style`].
pub fn button<S: 'static>(title: impl Into<String>, action: impl Fn(&mut S) + 'static) -> El<S> {
    let title = title.into();
    El::div("")
        .text(title.clone())
        .role(Role::Button, title)
        .on_click(move |s, _| action(s))
        .button_style(ButtonKind::Secondary)
}

/// A link: `title` in the accent colour, running `action` when followed.
pub fn link<S: 'static>(title: impl Into<String>, action: impl Fn(&mut S) + 'static) -> El<S> {
    let title = title.into();
    El::div("cursor: pointer")
        .text(title.clone())
        .foreground(Token::AccentText)
        .hover("opacity: 0.8")
        .role(Role::Link, title)
        .on_click(move |s, _| action(s))
}

/// A switch with its `title`, showing `is_on`; pressed anywhere on the row,
/// it calls `set` with the other value (role `switch`, its state
/// `checked`).
pub fn toggle<S: 'static>(
    title: impl Into<String>,
    is_on: bool,
    set: impl Fn(&mut S, bool) + 'static,
) -> El<S> {
    let title = title.into();
    toggle_with(title.clone(), text(title), is_on, set)
}

/// A [`toggle`] whose label is any view (a title over a description, say),
/// named `name` for assistive technology.
pub fn toggle_with<S: 'static>(
    name: impl Into<String>,
    label: El<S>,
    is_on: bool,
    set: impl Fn(&mut S, bool) + 'static,
) -> El<S> {
    let (fill, dot) = crate::components::switch_colors(is_on);
    let knob = El::div(
        "width: 18px; height: 18px; border-radius: 9px; box-shadow: 0 1px 3px rgba(0,0,0,0.25)",
    )
    .background(dot);
    let track = El::div(&format!(
        "display: flex; flex-direction: row; width: 38px; height: 22px; border-radius: 11px; padding: 2px; \
         box-sizing: border-box; flex-shrink: 0; justify-content: {}",
        if is_on { "flex-end" } else { "flex-start" }
    ))
    .background(fill)
    .child(knob);
    hstack(12.0, [label.fill_width().wraps(), track])
        .css("align-self: stretch; cursor: pointer; min-height: 24px")
        .semantics(
            Semantics::new(Role::Switch)
                .label(name.into())
                .checked(is_on),
        )
        .on_click(move |s, _| set(s, !is_on))
        .as_row()
}

/// One of `options` to choose, as buttons that wrap onto more lines, the
/// `selected` one in the accent (a radio group to assistive technology;
/// SwiftUI's `Picker`, inline). Pressing one calls `set` with its index.
pub fn picker<S: 'static, L: Into<String>>(
    options: impl IntoIterator<Item = L>,
    selected: Option<usize>,
    set: impl Fn(&mut S, usize) + 'static,
) -> El<S> {
    let set = std::rc::Rc::new(set);
    let items = options.into_iter().enumerate().map(|(i, o)| {
        let o = o.into();
        let on = selected == Some(i);
        let set = set.clone();
        // The chosen one outlined and washed in the accent (not filled: a
        // fill reads as the page's action).
        button(o.clone(), move |s: &mut S| set(s, i))
            .button_style(ButtonKind::Secondary)
            // A long one wraps in a narrow column rather than run out of it.
            .css("max-width: 100%; white-space: normal")
            .when(on, |e| {
                e.css("background: color-mix(in srgb, var(--accent) 14%, var(--surface))")
                    .token("border-color", Token::AccentText)
                    .bold()
            })
            .semantics(Semantics::new(Role::Radio).label(o).checked(on))
    });
    // As wide as its column, so it wraps there.
    hstack(8.0, items)
        .css("flex-wrap: wrap; align-self: stretch; min-width: 0")
        .semantics(Semantics::new(Role::RadioGroup))
}

/// A screen's header: its `icon` (SVG markup) on a tile in the accent, its
/// `title` and what it is for, in a card washed with the accent — a
/// settings pane's head, an app's welcome. `stat` (a figure and what it
/// counts) goes at the end, large, in the accent.
pub fn hero<S: 'static>(
    icon_markup: &str,
    title: impl Into<String>,
    subtitle: impl Into<String>,
    stat: Option<(String, String)>,
) -> El<S> {
    let title = title.into();
    let tile = icon(icon_markup, 24.0, 2.0)
        .foreground(Token::OnAccent)
        .background(Token::Accent)
        .css("padding: 11px; border-radius: 11px; flex-shrink: 0");
    let mut row = vec![
        tile,
        vstack(
            4.0,
            [
                text(title.clone())
                    .css("font-size: 17px; font-weight: 600")
                    .semantics(Semantics::new(Role::Heading(1)).label(title)),
                text(subtitle.into())
                    .css("font-size: 13px")
                    .foreground(Token::SecondaryText)
                    .wraps(),
            ],
        )
        .alignment(HorizontalAlignment::Leading)
        .fill_width(),
    ];
    if let Some((figure, caption)) = stat {
        row.push(
            vstack(
                2.0,
                [
                    text(figure)
                        .css("font-size: 28px; font-weight: 700")
                        .foreground(Token::AccentText),
                    text(caption)
                        .css("font-size: 11px")
                        .foreground(Token::SecondaryText),
                ],
            )
            .alignment(HorizontalAlignment::Trailing),
        );
    }
    hstack(16.0, row)
        .padding(16.0)
        .corner_radius(12.0)
        .border(Token::Border, 1.0)
        .background(Token::Surface)
        .css("align-self: stretch")
}

/// One of `options` chosen, the choices joined in a pill with the chosen
/// one raised (SwiftUI's `Picker` with the segmented style; a toolbar's
/// switch between views): `set` gets the index picked. Role `radiogroup`,
/// each option a `radio`.
pub fn segmented<S: 'static, L: Into<String>>(
    options: impl IntoIterator<Item = L>,
    selected: Option<usize>,
    set: impl Fn(&mut S, usize) + 'static,
) -> El<S> {
    let set = std::rc::Rc::new(set);
    let items = options.into_iter().enumerate().map(|(i, o)| {
        let o = o.into();
        let on = selected == Some(i);
        let set = set.clone();
        text(o.clone())
            .css("padding: 3px 13px; border: 1px solid transparent; min-height: 25px; box-sizing: border-box; border-radius: 7px; font-size: 13px; font-weight: 500; white-space: nowrap; cursor: pointer")
            .foreground(Token::Text)
            .when(on, |e| {
                // (an edge: on glass its fill alone is hardly seen)
                e.background(Token::Thumb)
                    .css("border-color: var(--border); box-shadow: 0 1px 3px rgba(0,0,0,0.18)")
            })
            .when(!on, |e| e.hover("opacity: 0.7"))
            .semantics(Semantics::new(Role::Radio).label(o).checked(on))
            .on_click(move |s: &mut S, _| set(s, i))
    });
    hstack(2.0, items)
        .padding(2.0)
        .corner_radius(9.0)
        .background(Token::Separator)
        .semantics(Semantics::new(Role::RadioGroup))
}

/// A button that opens `options` in a menu under it (SwiftUI's `Menu`):
/// `pick` gets the index chosen. `label` is what the button says (an icon's
/// name for assistive technology when `icon` is given: SVG markup, as
/// [`icon`]'s, drawn in place of the words).
pub fn menu<S: 'static>(
    label: impl Into<String>,
    icon: Option<&str>,
    options: Vec<String>,
    pick: impl Fn(&mut S, usize) + 'static,
) -> El<S> {
    let label = label.into();
    let icon = icon.map(str::to_owned);
    let shown = label.clone();
    let id = format!("menu:{label}:{icon:?}:{}", options.join("\u{1f}"));
    let pick = std::rc::Rc::new(pick);
    El::host(id, move |ui, input, _, node| {
        match &icon {
            Some(markup) => {
                crate::svg::add_icon(
                    ui,
                    node,
                    super::style_of("width: 18px; height: 18px"),
                    markup,
                    1.8,
                );
            }
            None => {
                let t = ui.add_box(node, super::style_of(""));
                ui.add_text(t, shown.clone());
            }
        }
        let items: Vec<crate::menu::MenuItem> = options
            .iter()
            .enumerate()
            .map(|(i, o)| crate::menu::MenuItem::action(i.to_string(), o.clone()))
            .collect();
        let pick = pick.clone();
        crate::menu::popover_menu(
            input,
            node,
            move |_| items.clone(),
            move |s: &mut S, _, id: &str| {
                if let Ok(i) = id.parse() {
                    pick(s, i);
                }
            },
        );
    })
    .css("display: flex; align-items: center; padding: 5px 8px; border-radius: 7px; cursor: pointer; font-size: 13px")
    .foreground(Token::SecondaryText)
    .hover("background: var(--hover)")
    .semantics(Semantics::new(Role::Button).label(label))
}

/// How far along something is: `value` from 0 to 1, or under way when
/// `None` (role `progressbar`).
pub fn progress_view<S: 'static>(value: Option<f32>) -> El<S> {
    let fill = value.map_or(30.0, |v| v.clamp(0.0, 1.0) * 100.0);
    let mut sem = Semantics::new(Role::ProgressBar);
    if let Some(v) = value {
        sem = sem.value(format!("{:.0}%", v.clamp(0.0, 1.0) * 100.0));
    }
    // As `pickers::progress`: an outlined bar, the outline outside its 6 px.
    El::div(
        "align-self: stretch; box-sizing: border-box; height: 8px; margin: -1px 0; \
         border-radius: 4px; overflow: hidden",
    )
    .background(Token::Surface)
    .border(Token::Track, 1.0)
    .semantics(sem)
    .child(
        El::div(&format!("height: 6px; border-radius: 3px; width: {fill}%"))
            .background(Token::AccentText),
    )
}

/// A card of `children` under a small `title`, as wide as its column
/// (SwiftUI's `GroupBox`; the look of [`crate::components::section`]).
pub fn group_box<S: 'static>(
    title: impl Into<String>,
    children: impl IntoIterator<Item = El<S>>,
) -> El<S> {
    let title = title.into();
    let mut items = vec![];
    if !title.is_empty() {
        // As `components::section`'s: small capitals, a heading.
        items.push(
            text(title.to_uppercase())
                .css("font-size: 12px; font-weight: 700")
                .foreground(Token::SecondaryText)
                .semantics(Semantics::new(Role::Heading(2)).label(title)),
        );
    }
    items.extend(children);
    vstack(8.0, items)
        .alignment(HorizontalAlignment::Leading)
        .padding(16.0)
        .corner_radius(10.0)
        .border(Token::Border, 1.0)
        .background(Token::Surface)
        // As wide as its column, as a form's groups are.
        .css("align-self: stretch; min-width: 0")
}

/// `children` in a rounded card with `title` above it, outside, in small
/// capitals (SwiftUI's inset-grouped `Section` header; a settings screen's
/// look). No title, just the card.
pub fn inset_group<S: 'static>(
    title: impl Into<String>,
    children: impl IntoIterator<Item = El<S>>,
) -> El<S> {
    let title = title.into();
    let mut rows: Vec<El<S>> = vec![];
    for child in children {
        // A hairline between one row and the next.
        if child.row && rows.last().is_some_and(|r: &El<S>| r.row) {
            rows.push(
                El::div("align-self: stretch; min-height: 1px; flex-shrink: 0")
                    .background(Token::Border),
            );
        }
        rows.push(child);
    }
    let card = vstack(8.0, rows)
        .alignment(HorizontalAlignment::Leading)
        .padding_edges(Edge::Vertical, 14.0)
        .padding_edges(Edge::Horizontal, 16.0)
        .corner_radius(12.0)
        .border(Token::Border, 1.0)
        .background(Token::Surface)
        .css("align-self: stretch; min-width: 0");
    if title.is_empty() {
        return card;
    }
    vstack(
        8.0,
        [
            text(title.clone())
                .css("font-size: 13px; font-weight: 600; padding: 0 4px")
                .semantics(Semantics::new(Role::Heading(2)).label(title)),
            card,
        ],
    )
    .alignment(HorizontalAlignment::Leading)
    // Further from what is above it than its caption is from its card.
    .css("align-self: stretch; min-width: 0; padding-top: 8px")
}

/// Settings laid out as a form: its [`section`]s one under another.
pub fn form<S: 'static>(sections: impl IntoIterator<Item = El<S>>) -> El<S> {
    vstack(16.0, sections)
        .alignment(HorizontalAlignment::Leading)
        .css("align-self: stretch")
        .semantics(Semantics::new(Role::Region))
}

/// A form's group: its `header` (small capitals, a heading), then a card of
/// `rows` with lines between them.
pub fn section<S: 'static>(
    header: impl Into<String>,
    rows: impl IntoIterator<Item = El<S>>,
) -> El<S> {
    let header = header.into();
    let mut parts = vec![];
    if !header.is_empty() {
        parts.push(
            text(header.to_uppercase())
                .font(Font::Footnote)
                .bold()
                .foreground(Token::SecondaryText)
                .padding_edges(Edge::Horizontal, 4.0)
                .semantics(Semantics::new(Role::Heading(2)).label(header)),
        );
    }
    parts.push(
        separated(rows)
            .padding_edges(Edge::Horizontal, 12.0)
            .corner_radius(10.0)
            .border(Token::Border, 1.0)
            .background(Token::Surface),
    );
    vstack(6.0, parts)
        .alignment(HorizontalAlignment::Leading)
        .css("align-self: stretch")
}

/// `rows` in a column, a [`divider`] between each two.
fn separated<S: 'static>(rows: impl IntoIterator<Item = El<S>>) -> El<S> {
    let mut items = vec![];
    for (i, r) in rows.into_iter().enumerate() {
        if i > 0 {
            items.push(divider());
        }
        items.push(r.padding_edges(Edge::Vertical, 8.0));
    }
    vstack(0.0, items)
        .alignment(HorizontalAlignment::Leading)
        .css("align-self: stretch")
}

/// A `label` and its `value` on one line, the value at the end and muted
/// (SwiftUI's `LabeledContent`).
pub fn labeled_content<S: 'static>(label: impl Into<String>, value: impl Into<String>) -> El<S> {
    hstack(
        12.0,
        [
            text(label).fill_width().wraps(),
            text(value).foreground(Token::SecondaryText),
        ],
    )
    .css("align-self: stretch")
}

/// `rows` as a list (role `list`, each a `listitem`), a line between each
/// two.
pub fn list<S: 'static>(rows: impl IntoIterator<Item = El<S>>) -> El<S> {
    let items = rows.into_iter().map(|r| {
        let s = r.semantics.clone();
        // An item keeps a role of its own (a button row) inside a list item.
        match s {
            Some(_) => El::div("align-self: stretch")
                .semantics(Semantics::new(Role::ListItem))
                .child(r),
            None => r
                .css("align-self: stretch")
                .semantics(Semantics::new(Role::ListItem)),
        }
    });
    separated(items).semantics(Semantics::new(Role::List))
}

/// What shows where there is nothing yet: a `symbol`, a `title` and a line
/// of `description` (SwiftUI's `ContentUnavailableView`), said at once.
pub fn content_unavailable<S: 'static>(
    symbol: impl Into<String>,
    title: impl Into<String>,
    description: impl Into<String>,
) -> El<S> {
    let (title, description) = (title.into(), description.into());
    vstack(
        6.0,
        [
            text(symbol)
                .css("font-size: 32px")
                .foreground(Token::SecondaryText),
            text(title.clone()).font(Font::Title3),
            text(description.clone())
                .font(Font::Subheadline)
                .foreground(Token::SecondaryText)
                .wraps(),
        ],
    )
    .padding_edges(Edge::Vertical, 24.0)
    .css("align-self: stretch; text-align: center")
    .semantics(Semantics::new(Role::Status).label(format!("{title}. {description}")))
}

/// A line of text to edit, bound to `text` (SwiftUI's `TextField`):
/// `placeholder` shows while it is empty and names it. It keeps its text,
/// caret and focus while the view updates round it; mount it with
/// [`View::update_with`](super::View::update_with). See
/// [`El::on_submit`].
pub fn text_field<S: 'static>(
    placeholder: impl Into<String>,
    text: crate::binding::Binding<S, String>,
) -> El<S> {
    let placeholder = placeholder.into();
    // (a line, its caret shown and kept in view: see `fields::line_field`)
    El::host(format!("field:{placeholder}"), move |ui, input, state, node| {
        crate::fields::line_field(ui, input, state, node, &placeholder, text.clone());
    })
    .css("display: flex; flex-direction: column; align-self: stretch; min-width: 0")
}

/// A search box bound to `query` (SwiftUI's `.searchable`, as a field): a
/// magnifier, the text, a ✕ to empty it; `placeholder` names it. Hosted,
/// so it keeps its text, caret and focus. See [`El::on_submit`].
pub fn search_field<S: 'static>(
    placeholder: impl Into<String>,
    query: crate::binding::Binding<S, String>,
) -> El<S> {
    let placeholder = placeholder.into();
    El::host(
        format!("search:{placeholder}"),
        move |ui, input, state, node| {
            crate::pickers::search_field(ui, input, state, node, &placeholder, query.clone());
        },
    )
    .css("display: flex; flex-direction: column; align-self: stretch; min-width: 0")
}

/// Text of several lines to edit (`rows` tall at least), as
/// [`text_field`] (SwiftUI's `TextEditor`).
pub fn text_editor<S: 'static>(
    placeholder: impl Into<String>,
    text: crate::binding::Binding<S, String>,
    rows: u32,
) -> El<S> {
    let placeholder = placeholder.into();
    El::host(
        format!("text:{placeholder}"),
        move |ui, input, state, node| {
            crate::pickers::text_area(ui, input, state, node, &placeholder, text.clone(), rows);
        },
    )
    .css("display: flex; flex-direction: column; align-self: stretch; min-width: 0")
}

/// A value in `range` set by dragging along a track, `step` at a time
/// (SwiftUI's `Slider`): the arrows step it, Home and End go to the ends.
/// Hosted, so it keeps its drag while the view updates round it (mount it
/// with [`View::update_with`](super::View::update_with)).
pub fn slider<S: 'static>(
    label: impl Into<String>,
    value: crate::binding::Binding<S, f32>,
    range: std::ops::RangeInclusive<f32>,
    step: f32,
) -> El<S> {
    let label = label.into();
    El::host(format!("slider:{label}"), move |ui, input, state, node| {
        crate::pickers::slider(
            ui,
            input,
            state,
            node,
            &label,
            value.clone(),
            range.clone(),
            step,
        );
    })
    .css("display: flex; flex-direction: column; align-self: stretch")
}

/// A knob, turned to a value in `range`, `step` at a time, `size` pixels
/// across (see [`pickers::knob`](crate::pickers::knob)): for a spatial UI's
/// dials, and a mixer's. Hosted.
pub fn knob<S: 'static>(
    label: impl Into<String>,
    value: crate::binding::Binding<S, f32>,
    range: std::ops::RangeInclusive<f32>,
    step: f32,
    size: f32,
) -> El<S> {
    let label = label.into();
    El::host(format!("knob:{label}"), move |ui, input, state, node| {
        crate::pickers::knob(ui, input, state, node, &label, value.clone(), range.clone(), step, size);
    })
    .css("display: flex; flex-direction: column; align-items: center; flex-shrink: 0")
}

/// A pad holding a point, `(x, y)` each 0 to 1, `size` pixels square (see
/// [`pickers::xy_pad`](crate::pickers::xy_pad)): a trackpad, a joystick, two
/// axes at once. Hosted.
pub fn xy_pad<S: 'static>(
    label: impl Into<String>,
    value: crate::binding::Binding<S, (f32, f32)>,
    size: f32,
) -> El<S> {
    let label = label.into();
    El::host(format!("pad:{label}"), move |ui, input, state, node| {
        crate::pickers::xy_pad(ui, input, state, node, &label, value.clone(), size);
    })
    .css("display: flex; flex-direction: column; align-items: center; flex-shrink: 0")
}

/// An icon from SVG markup (SwiftUI's `Image(systemName:)`): the inside of
/// an `<svg>` — `path`, `circle`, `rect`, `line`, `polyline`… — in a 24 × 24
/// view box (Lucide's, Feather's), stroked `stroke` units wide, `size`
/// pixels square, in the colour of the text around it ([`El::foreground`]
/// colours it). Drawn the same in every font, where a glyph (▸, ✓) is each
/// font's own. See [`crate::svg`].
pub fn icon<S: 'static>(markup: &str, size: f32, stroke: f32) -> El<S> {
    let markup = markup.to_owned();
    let id = format!("icon:{markup}:{size}:{stroke}");
    El::host(id, move |ui, _, _, node| {
        // as large as the text round it: given at the scale now, kept as
        // its own at none
        let at = ui.scaled(size);
        ui.update_style(node, |s| (s.width, s.height) = (crate::Length::Px(at), crate::Length::Px(at)));
        let drawn = crate::svg::add_icon(
            ui,
            node,
            super::style_of(&format!("width: {at}px; height: {at}px")),
            &markup,
            stroke,
        );
        ui.scale_with_text(node);
        ui.scale_with_text(drawn);
    })
    .css("flex-shrink: 0")
}

/// A chevron pointing `towards` the line's end (`›`), down (`⌄`) or back.
pub fn chevron<S: 'static>(towards: Chevron, size: f32) -> El<S> {
    let d = match towards {
        Chevron::Forward => r#"<path d="M9 18l6-6-6-6"/>"#,
        Chevron::Down => r#"<path d="M6 9l6 6 6-6"/>"#,
        Chevron::Back => r#"<path d="M15 18l-6-6 6-6"/>"#,
    };
    // (forward is to the left where that is the way lines read)
    icon(d, size, 2.4).flips_for_right_to_left()
}

/// Where a [`chevron`] points.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Chevron {
    Forward,
    Down,
    Back,
}

/// A picture, `css` sizing it (SwiftUI's `Image`); `id` names this one, so
/// another picture there is drawn anew.
pub fn image<S: 'static>(
    id: impl Into<String>,
    picture: crate::image::ImageRef,
    css: &str,
) -> El<S> {
    let css = css.to_owned();
    El::host(format!("image:{}", id.into()), move |ui, _, _, node| {
        ui.add_image(node, super::style_of(&css), picture.clone());
    })
}

/// `values` as a line, `size` (width, height) big, named `name` for
/// assistive technology (with its range); drawn anew when they change.
pub fn line_chart<S: 'static>(
    name: impl Into<String>,
    values: Vec<f32>,
    size: (f32, f32),
) -> El<S> {
    line_chart_in(name.into(), values, size, false, None, None)
}

/// The most points a line is drawn through.
const LINE_POINTS: usize = 240;

/// `values` with no more than [`LINE_POINTS`]: of each stretch its lowest and
/// highest, in their order, so no spike is lost. (What a line costs to draw
/// grows with its points, and more than there are pixels are not seen.)
fn thinned(values: Vec<f32>) -> Vec<f32> {
    if values.len() <= LINE_POINTS {
        return values;
    }
    let stretches = LINE_POINTS / 2;
    (0..stretches)
        .flat_map(|i| {
            let from = i * values.len() / stretches;
            let part = &values[from..(i + 1) * values.len() / stretches];
            let (mut low, mut high) = (0, 0);
            for (k, v) in part.iter().enumerate() {
                if *v < part[low] {
                    low = k;
                }
                if *v > part[high] {
                    high = k;
                }
            }
            [part[low.min(high)], part[low.max(high)]]
        })
        .collect()
}

#[cfg(test)]
mod line_tests {
    use super::*;

    #[test]
    fn a_long_line_is_thinned_and_keeps_its_spikes() {
        let mut v: Vec<f32> = (0..5000).map(|i| (i as f32 * 0.01).sin()).collect();
        v[1234] = 9.0;
        v[4321] = -7.0;
        let t = thinned(v);
        assert!(t.len() <= LINE_POINTS, "{}", t.len());
        assert!(t.contains(&9.0) && t.contains(&-7.0));
        assert_eq!(thinned(vec![1.0, 2.0, 3.0]), vec![1.0, 2.0, 3.0]);
    }
}

fn line_chart_in<S: 'static>(
    name: String,
    values: Vec<f32>,
    size: (f32, f32),
    wide: bool,
    color: Option<String>,
    range: Option<(f32, f32)>,
) -> El<S> {
    use std::hash::{Hash, Hasher};
    let values = thinned(values);
    let mut h = std::collections::hash_map::DefaultHasher::new();
    for v in &values {
        v.to_bits().hash(&mut h);
    }
    El::host(
        format!("line:{name}:{wide}:{color:?}:{range:?}:{:x}", h.finish()),
        move |ui, _, _, node| {
            let chart = crate::charts::line_chart_between(ui, node, &name, &values, size, range);
            if wide {
                let layers = ui.children(chart).to_vec();
                // (its end's dot would be stretched with it)
                if let Some(&dot) = layers.get(2) {
                    ui.update_style(dot, |s| s.display = crate::Display::None);
                }
                for n in std::iter::once(chart).chain(layers) {
                    ui.update_style(n, |s| s.width = crate::Length::Percent(100.0));
                }
            }
            if let Some(c) = &color {
                for n in ui.children(chart).to_vec() {
                    ui.clear_token_style(n, "color");
                    ui.update_style(n, |s| {
                        crate::css::apply(s, "color", c);
                    });
                }
            }
        },
    )
}

/// A [`line_chart_wide`] in `color` (CSS) rather than the accent, on the
/// scale `range` if given (see [`crate::charts::line_chart_between`]).
pub fn line_chart_in_color<S: 'static>(
    name: impl Into<String>,
    values: Vec<f32>,
    height: f32,
    color: &str,
    range: Option<(f32, f32)>,
) -> El<S> {
    line_chart_in(name.into(), values, (320.0, height), true, Some(color.to_owned()), range)
        .css("display: flex; flex-direction: column; align-self: stretch; min-width: 0")
}

/// A [`line_chart`] as wide as what it is in, `height` tall.
pub fn line_chart_wide<S: 'static>(
    name: impl Into<String>,
    values: Vec<f32>,
    height: f32,
) -> El<S> {
    // Drawn at a size near what it will be, and stretched across to it.
    line_chart_in(name.into(), values, (320.0, height), true, None, None)
        .css("display: flex; flex-direction: column; align-self: stretch; min-width: 0")
}

/// Text that follows the state: `f(state)`, kept up to date after every
/// event without the view updating (the number beside a slider as it is
/// dragged). `id` tells two apart.
pub fn live_text<S: 'static>(id: impl Into<String>, f: impl Fn(&S) -> String + 'static) -> El<S> {
    let f = std::rc::Rc::new(f);
    El::host(
        format!("live:{}", id.into()),
        move |ui, input, state, node| {
            let mut shown = f(state);
            let text = ui.add_text(node, shown.clone());
            let f = f.clone();
            input.on_after_dispatch(move |_, ui, s| {
                if !ui.contains(text) {
                    return false;
                }
                let now = f(s);
                if now != shown {
                    ui.set_text(text, now.clone());
                    shown = now;
                }
                true
            });
        },
    )
}

/// The modifiers of what is typed in.
impl<S: 'static> El<S> {
    /// Calls `submit` when Enter is pressed in it (a [`text_field`]:
    /// SwiftUI's `.onSubmit`).
    pub fn on_submit(self, submit: impl Fn(&mut S) + 'static) -> Self {
        use crate::input::{Event, Phase};
        let submit = std::rc::Rc::new(submit);
        self.after_build(move |_, input, _, node| {
            let submit = submit.clone();
            // Before the field takes Enter for itself (Shift+Enter is still
            // its own).
            input.on(node, Phase::Capture, move |s, _, e, cx| {
                if let Event::Key { key, modifiers } = e
                    && key == "Enter"
                    && !modifiers.shift
                {
                    submit(s);
                    cx.stop_propagation();
                    cx.prevent_default();
                }
            });
        })
    }
}

/// `children` in a column that scrolls when they do not fit.
pub fn scroll_view<S: 'static>(children: impl IntoIterator<Item = El<S>>) -> El<S> {
    El::div("display: flex; flex-direction: column; overflow: auto; flex-grow: 1; min-height: 0")
        .children(children)
}
