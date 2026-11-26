//! What flows down a UI to everything in it:
//!
//! - **Values** of any type set on a node and read by anything inside it
//!   ([`Ui::set_environment`], [`Ui::environment`]), the nearest setting
//!   winning, and an app-wide [global](crate::globals) under them all.
//! - **Colours by purpose** ([`Token`]): a node styled with
//!   [`Ui::token_style`] takes the colour its purpose has in the appearance
//!   showing, from the [`Tokens`] the UI is given — so components carry no
//!   colours of their own, and a new palette restyles everything.
//! - **Text size** ([`Ui::set_text_scale`]): every text node's font and
//!   line height scaled, as browsers' text zoom does.
//! - **Reduced motion** ([`Ui::set_reduce_motion`]): animations of place
//!   and size jump to their end; fades still run.
//!
//! ```
//! use threers_ui::environment::Token;
//! use threers_ui::states::Appearance;
//! use threers_ui::{style, Ui};
//!
//! let mut ui = Ui::new();
//! let root = ui.root();
//! let card = ui.add_box(root, style! { padding: 8px; });
//! ui.token_style(card, &[("background", Token::Surface), ("color", Token::Text)]);
//! ui.set_system_appearance(Appearance::Light);
//! assert_eq!(ui.style(card).background, Some(ui.tokens().light.surface));
//! ui.set_system_appearance(Appearance::Dark);
//! assert_eq!(ui.style(card).background, Some(ui.tokens().dark.surface));
//! ```

use std::any::{Any, TypeId};
use std::collections::HashMap;
use std::rc::Rc;

use crate::animation::Property;
use crate::states::Appearance;
use crate::tree::{NodeId, Ui};
use crate::{Edges, Length, NodeKind, Rgba};

/// A colour's purpose. See [`TokenSet`] for what each is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Token {
    Background,
    Surface,
    Raised,
    Text,
    SecondaryText,
    Border,
    Separator,
    Accent,
    OnAccent,
    Hover,
    Destructive,
    /// Behind white text: a filled destructive button.
    DestructiveFill,
    Success,
    Focus,
    /// Behind the default button's label: the accent, unless the theme
    /// keeps its accent for highlights and draws buttons plainer.
    Prominent,
    /// The default button's label, on [`Prominent`](Token::Prominent).
    OnProminent,
    /// What to mind: amber, as text and as a wash.
    Warning,
    /// The accent as text (a link, a text-only button): moved toward the
    /// text's colour as far as it takes to read. See
    /// [`TokenSet::accent_text`].
    AccentText,
    /// What a control is drawn on that has to be seen (a slider's rail):
    /// the border's colour, stronger. See [`TokenSet::track`].
    Track,
    /// What is raised off a control's own ground (a segmented control's
    /// chosen part). See [`TokenSet::thumb`].
    Thumb,
}

/// The colours of one appearance, by purpose.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TokenSet {
    /// Behind everything.
    pub background: Rgba,
    /// Panels and cards on it.
    pub surface: Rgba,
    /// What floats over the rest: menus, popovers, dialogs, notices.
    pub raised: Rgba,
    pub text: Rgba,
    /// Captions, hints, shortcut labels.
    pub secondary_text: Rgba,
    pub border: Rgba,
    pub separator: Rgba,
    /// The highlight: selection, the default button, a switch that is on.
    pub accent: Rgba,
    /// Text on the accent.
    pub on_accent: Rgba,
    /// Under the pointer.
    pub hover: Rgba,
    pub destructive: Rgba,
    /// The destructive colour as a fill under [`on_accent`](Self::on_accent)
    /// text: darker where `destructive` is a light red for text on dark.
    pub destructive_fill: Rgba,
    pub success: Rgba,
    /// The focus ring.
    pub focus: Rgba,
    /// The default button's fill (often the accent; a neutral one where
    /// the accent only highlights) and its label.
    pub prominent: Rgba,
    pub on_prominent: Rgba,
    /// What to mind (not an error).
    pub warning: Rgba,
}

impl TokenSet {
    pub fn get(&self, token: Token) -> Rgba {
        match token {
            Token::Background => self.background,
            Token::Surface => self.surface,
            Token::Raised => self.raised,
            Token::Text => self.text,
            Token::SecondaryText => self.secondary_text,
            Token::Border => self.border,
            Token::Separator => self.separator,
            Token::Accent => self.accent,
            Token::OnAccent => self.on_accent,
            Token::Hover => self.hover,
            Token::Destructive => self.destructive,
            Token::DestructiveFill => self.destructive_fill,
            Token::Success => self.success,
            Token::Focus => self.focus,
            Token::Prominent => self.prominent,
            Token::OnProminent => self.on_prominent,
            Token::Warning => self.warning,
            Token::AccentText => self.accent_text(),
            Token::Track => self.track(),
            Token::Thumb => self.thumb(),
        }
    }

    /// The accent as text: itself where it is 4.5:1 on the background, the
    /// surface and what is raised (WCAG AA), else mixed with the text's
    /// colour until it is. (One accent cannot be both dark enough under
    /// white labels and light enough on a dark page.)
    pub fn accent_text(&self) -> Rgba {
        self.toward_text(self.accent, 4.5)
    }

    /// The border's colour as a control's rail: mixed with the text's until
    /// it is 3:1 on the background, the surface and what is raised (WCAG
    /// 1.4.11).
    pub fn track(&self) -> Rgba {
        self.toward_text(self.border, 3.0)
    }

    /// Lighter than the separator's colour, as what is raised off it is in
    /// either appearance: the surface where that is lighter, else the
    /// separator part of the way to white.
    pub fn thumb(&self) -> Rgba {
        let light = |c: Rgba| c[0] + c[1] + c[2];
        if light(self.surface) > light(self.separator) {
            self.surface
        } else {
            let s = self.separator;
            [0, 1, 2, 3].map(|i| if i == 3 { 1.0 } else { s[i] + (1.0 - s[i]) * 0.2 })
        }
    }

    /// `from`, mixed with the text's colour as little as makes it `ratio`
    /// against the background, the surface and what is raised.
    fn toward_text(&self, from: Rgba, ratio: f32) -> Rgba {
        // A see-through colour as it looks on the surface.
        let alpha = from[3];
        let mut from = from;
        if alpha < 1.0 {
            from = std::array::from_fn(|i| self.surface[i] + (from[i] - self.surface[i]) * alpha);
            from[3] = 1.0;
        }
        let rgb = |c: Rgba| [c[0], c[1], c[2]];
        let mixed =
            |k: f32| -> Rgba { std::array::from_fn(|i| from[i] + (self.text[i] - from[i]) * k) };
        let reads = |c: &Rgba| {
            [self.background, self.surface, self.raised]
                .iter()
                .all(|&b| crate::contrast_ratio(rgb(*c), rgb(b)) >= ratio)
        };
        (0..=32)
            .map(|i| mixed(i as f32 / 32.0))
            .find(reads)
            .unwrap_or(self.text)
    }
}

fn hex(v: u32) -> Rgba {
    let c = |s: u32| ((v >> s) & 0xff) as f32 / 255.0;
    [c(16), c(8), c(0), 1.0]
}

/// The colours of both appearances.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Tokens {
    pub light: TokenSet,
    pub dark: TokenSet,
}

impl Default for Tokens {
    fn default() -> Self {
        Self {
            light: TokenSet {
                background: hex(0xf2f4f7),
                surface: hex(0xffffff),
                raised: hex(0xffffff),
                text: hex(0x1f2329),
                secondary_text: hex(0x5b6270),
                border: hex(0xd5d9e0),
                separator: hex(0xe2e5ea),
                accent: hex(0x2563eb),
                on_accent: [0.99, 0.99, 0.99, 1.0],
                hover: [0.5, 0.54, 0.63, 0.18],
                // 5.9:1 on the page's grey as text; under white text, the
                // fill's 4.7:1.
                destructive: hex(0xb91c1c),
                destructive_fill: hex(0xdc2626),
                // 4.5:1 or more as text; the ring 3:1 or more (WCAG 1.4.11):
                // on white, the dark theme's pale ones are not.
                success: hex(0x047857),
                focus: hex(0x2563eb),
                prominent: hex(0x2563eb),
                on_prominent: [0.99, 0.99, 0.99, 1.0],
                warning: hex(0xb45309),
            },
            dark: TokenSet {
                background: hex(0x12151b),
                surface: hex(0x1b1f27),
                raised: hex(0x1f2430),
                text: hex(0xe5e7eb),
                secondary_text: hex(0x979eaf),
                border: hex(0x3a4150),
                separator: hex(0x2a303c),
                accent: hex(0x2563eb),
                on_accent: [0.99, 0.99, 0.99, 1.0],
                hover: [0.5, 0.54, 0.63, 0.28],
                // 5.6:1 or more as text on every dark surface; the fill
                // darker, 4.7:1 under white text (#ef4444 was 3.7:1).
                destructive: hex(0xf87171),
                destructive_fill: hex(0xdc2626),
                success: hex(0x10b981),
                focus: hex(0x93c5fd),
                prominent: hex(0x2563eb),
                on_prominent: [0.99, 0.99, 0.99, 1.0],
                warning: hex(0xf59e0b),
            },
        }
    }
}

impl Tokens {
    /// The lightest glass these are made to be read on, sRGB: glass over a
    /// bright room, darkened as `threers-ios`'s pieces darken it.
    pub const GLASS_GROUND: Rgba = [0.36, 0.36, 0.36, 1.0];

    /// The colours for a UI on glass (a visionOS window's, a piece's): white
    /// text and pale accents, what holds others a darker, see-through
    /// layer that lets the glass show, and what floats over the rest nearly
    /// solid. On glass no lighter than [`GLASS_GROUND`](Self::GLASS_GROUND),
    /// text is 4.5:1 or more and what has to be seen 3:1. The same light
    /// and dark: the glass is what it is.
    pub fn glass() -> Self {
        let grey = |v: f32, a: f32| [v, v, v + 0.015, a];
        let white = |a: f32| [1.0, 1.0, 1.0, a];
        let on_pale = hex(0x06142e);
        let set = TokenSet {
            background: grey(0.365, 0.0),
            // shades, not greys: coloured glass keeps its colour through them
            // (the surface a hair the lighter, which makes it the thumb's)
            surface: grey(0.0, 0.14),
            raised: grey(0.2, 0.95),
            text: white(1.0),
            secondary_text: hex(0xf2f2f6),
            border: white(0.5),
            separator: [0.0, 0.0, 0.0, 0.17],
            accent: hex(0x9cc8ff),
            on_accent: on_pale,
            hover: white(0.1),
            destructive: hex(0xffc9c4),
            destructive_fill: hex(0xff9c94),
            success: hex(0xa6f4b8),
            focus: white(1.0),
            prominent: hex(0x9cc8ff),
            on_prominent: on_pale,
            warning: hex(0xffd980),
        };
        Self { light: set, dark: set }
    }

    /// The darkest pale glass [`pale_glass`](Self::pale_glass) is made to be
    /// read on, sRGB: pale glass over a dark room, lightened as
    /// `threers-ios`'s pieces lighten it.
    pub const PALE_GLASS_GROUND: Rgba = [0.68, 0.68, 0.68, 1.0];

    /// [`glass`](Self::glass)'s for pale glass: dark text and deep accents,
    /// as readable on glass no darker than
    /// [`PALE_GLASS_GROUND`](Self::PALE_GLASS_GROUND).
    pub fn pale_glass() -> Self {
        let black = |a: f32| [0.0, 0.0, 0.0, a];
        let white = |a: f32| [1.0, 1.0, 1.0, a];
        // (not quite white: pure white is a style's default, which inherits)
        let on_deep = [0.99, 0.99, 0.99, 1.0];
        let set = TokenSet {
            background: [0.68, 0.68, 0.68, 0.0],
            surface: white(0.3),
            raised: [0.95, 0.95, 0.96, 0.96],
            text: hex(0x08080c),
            secondary_text: hex(0x222228),
            border: black(0.5),
            separator: black(0.12),
            accent: hex(0x07306e),
            on_accent: on_deep,
            hover: black(0.08),
            destructive: hex(0x6a0808),
            destructive_fill: hex(0xa01515),
            success: hex(0x06331a),
            focus: hex(0x07306e),
            prominent: hex(0x07306e),
            on_prominent: on_deep,
            warning: hex(0x452600),
        };
        Self { light: set, dark: set }
    }
}

/// The environment's state in a [`Ui`].
#[derive(Clone, Default)]
pub(crate) struct Env {
    pub tokens: Tokens,
    /// Theme colours painted as something else (see [`fill`](crate::fill)).
    pub fills: Vec<(Token, crate::fill::Fill)>,
    /// Values set on nodes, by type.
    pub values: HashMap<NodeId, HashMap<TypeId, Rc<dyn Any>>>,
    /// 1 when unset (`0.0` reads as 1).
    pub text_scale: f32,
    /// Text nodes' own font size and line height, before the scale.
    pub unscaled: HashMap<NodeId, (f32, Option<f32>)>,
    pub reduce_motion: bool,
    /// The smallest target the audit allows, in UI pixels; 0 when unset
    /// (WCAG's 24).
    pub min_target: f32,
    /// Bold Text and Increase Contrast (see [`legibility`](crate::legibility)),
    /// and each text node's weight before Bold Text.
    pub bold_text: bool,
    pub increase_contrast: bool,
    pub unbolded: HashMap<NodeId, crate::FontWeight>,
    /// Parts of the UI in a theme of their own (see [`theme`](crate::theme)).
    pub subtree_themes: HashMap<NodeId, crate::states::Theme>,
    /// Names nodes are linked to by (see [`anchors`](crate::anchors)).
    pub anchors: HashMap<NodeId, String>,
    /// Gauges gliding to a new value.
    pub glides: Vec<crate::charts::Glide>,
    /// Boxes whose spacing and fixed sizes follow the text size, with
    /// their own (see [`Ui::scale_with_text`]).
    pub scaled: HashMap<NodeId, Metrics>,
    /// The window's edges the system covers (see [`Ui::set_safe_area`]).
    pub safe_area: Edges,
    /// Boxes padded clear of them: their own padding and which edges.
    pub safe_nodes: HashMap<NodeId, (Edges, [bool; 4])>,
    /// Subtrees drawn as placeholders (see [`Ui::set_redacted`]).
    pub redacted: std::collections::HashSet<NodeId>,
    pub direction: LayoutDirection,
    /// Boxes turned and lifted in 3D (see [`depth`](crate::depth)).
    pub transforms3d: HashMap<NodeId, crate::depth::Transform3d>,
    /// Boxes moving on their own, and their transforms without the motion
    /// (see [`motion`](crate::motion)).
    pub motions: HashMap<NodeId, (crate::motion::MotionSpec, Option<crate::depth::Transform3d>)>,
    /// Named styles and the nodes that have them (see [`patch`](crate::patch)).
    pub classes: crate::patch::Classes,
    /// Boxes drawn solid (see [`solid`](crate::solid)), and the turn and
    /// room their shader was last given.
    pub solids: HashMap<NodeId, crate::solid::SolidState>,
    /// Layers moving by their depth when the box around them tilts.
    pub parallax: HashMap<NodeId, f32>,
    /// What is to be scrolled into view sideways once laid out (a field's
    /// caret, after its text changed).
    pub reveal_x: Vec<NodeId>,
    /// Strips whose items turn away from their middle.
    pub carousels: Vec<NodeId>,
    /// Where light falls from, for glare and shading.
    pub light: crate::depth::Light,
    /// How the device is held: its tilt forward and sideways, in degrees.
    pub device_tilt: (f32, f32),
    /// Lists whose rows roll over a drum.
    pub cylinders: Vec<NodeId>,
    /// Boxes mirrored below themselves: strength and height shown.
    pub reflections: HashMap<NodeId, (f32, f32)>,
    /// Boxes drawn through custom shaders.
    pub effects: HashMap<NodeId, crate::shader::Effect>,
    /// The beams on boxes' effects, as last put there (see
    /// [`border_beam`](crate::border_beam)).
    pub beams: HashMap<NodeId, crate::border_beam::BeamState>,
    /// Reduced motion as the beams were last brought up to date with.
    pub beams_still: bool,
    /// A style or the tree changed since the beams were last brought up
    /// to date with them.
    pub beams_stale: bool,
    /// How much the effects may animate (see [`Ui::set_animation_budget`](crate::Ui::set_animation_budget)).
    pub budget: crate::shader::AnimationBudget,
    /// The animated effects' rates under it, frames a second (infinite:
    /// every frame; 0: held), as the last step planned them.
    pub effect_rates: HashMap<NodeId, f32>,
    /// The time each animated effect was last drawn at.
    pub effect_drawn: std::cell::RefCell<HashMap<NodeId, f32>>,
    /// What boxes show changed, not where they are: paint again.
    pub repaint: std::cell::Cell<bool>,
    /// Heights set by `aspect-ratio`, by box (so that a height the app set
    /// is told from one that follows the width).
    pub aspect_heights: HashMap<NodeId, f32>,
    /// Scrolling boxes whose items change by where they are in view.
    pub scroll_transitions: Vec<(NodeId, crate::scroll_transition::ScrollTransition)>,
    /// Keyframes playing (see [`keyframes`](crate::keyframes)).
    pub keyframes: Vec<crate::keyframes::Playing>,
    /// Work off the UI thread and what to do when it is done.
    pub jobs: Rc<std::cell::RefCell<crate::jobs::Jobs>>,
    /// Painting a reflection now (it does not reflect again).
    pub reflecting: std::cell::Cell<bool>,
}

/// A box's spacing and fixed sizes as given, before the text scale.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Metrics {
    padding: Edges,
    gap: f32,
    row_gap: Option<f32>,
    width: Length,
    height: Length,
    min_width: Length,
    min_height: Length,
    radius: Length,
}

/// Which way a UI reads (see [`Ui::set_layout_direction`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LayoutDirection {
    #[default]
    LeftToRight,
    /// Arabic, Hebrew, Persian: rows start at the right.
    RightToLeft,
}

impl std::fmt::Debug for Env {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Env")
            .field("text_scale", &self.text_scale)
            .field("reduce_motion", &self.reduce_motion)
            .finish()
    }
}

impl Token {
    /// Every token.
    pub const ALL: [Token; 20] = [
        Token::Background,
        Token::Surface,
        Token::Raised,
        Token::Text,
        Token::SecondaryText,
        Token::Border,
        Token::Separator,
        Token::Accent,
        Token::OnAccent,
        Token::Hover,
        Token::Destructive,
        Token::DestructiveFill,
        Token::Success,
        Token::Focus,
        Token::Prominent,
        Token::OnProminent,
        Token::Warning,
        Token::AccentText,
        Token::Track,
        Token::Thumb,
    ];

    /// Its name as a CSS custom property, without the dashes: in a state's
    /// look (`hover_style(b, "background: var(--hover)")`) it is the
    /// theme's colour as it is at the time, following a new theme.
    pub fn css_name(self) -> &'static str {
        match self {
            Token::Background => "background",
            Token::Surface => "surface",
            Token::Raised => "raised",
            Token::Text => "text",
            Token::SecondaryText => "secondary-text",
            Token::Border => "border",
            Token::Separator => "separator",
            Token::Accent => "accent",
            Token::OnAccent => "on-accent",
            Token::Hover => "hover",
            Token::Destructive => "destructive",
            Token::DestructiveFill => "destructive-fill",
            Token::Success => "success",
            Token::Focus => "focus",
            Token::Prominent => "prominent",
            Token::OnProminent => "on-prominent",
            Token::Warning => "warning",
            Token::AccentText => "accent-text",
            Token::Track => "track",
            Token::Thumb => "thumb",
        }
    }

    /// The token named `name` (see [`css_name`](Self::css_name)).
    pub fn from_css_name(name: &str) -> Option<Token> {
        Token::ALL.into_iter().find(|t| t.css_name() == name)
    }
}

/// `value` with each `var(--token)` in the colours of `tokens`, and a
/// `color-mix(in srgb, A p%, B)` of two colours mixed (as CSS mixes them:
/// premultiplied, so a colour mixed with `transparent` keeps its hue and
/// takes `p`% opacity). Anything else is as it was.
pub(crate) fn resolve_colors<'a>(
    value: &'a str,
    tokens: impl Fn(Token) -> Rgba,
) -> std::borrow::Cow<'a, str> {
    if !value.contains("var(--") && !value.contains("color-mix(") {
        return std::borrow::Cow::Borrowed(value);
    }
    let mut out = String::with_capacity(value.len());
    let mut rest = value;
    while let Some(i) = rest.find("var(--") {
        out.push_str(&rest[..i]);
        let after = &rest[i + 6..];
        let Some(end) = after.find(')') else {
            out.push_str(&rest[i..]);
            rest = "";
            break;
        };
        match Token::from_css_name(after[..end].trim()) {
            Some(t) => out.push_str(&css_color(tokens(t))),
            None => out.push_str(&rest[i..i + 6 + end + 1]),
        }
        rest = &after[end + 1..];
    }
    out.push_str(rest);
    // Every `color-mix(…)`, wherever it is: a gradient's stops, a shadow, a
    // whole declaration list.
    let mut mixed = String::with_capacity(out.len());
    let mut rest = out.as_str();
    while let Some(i) = rest.find("color-mix(") {
        mixed.push_str(&rest[..i]);
        let call = &rest[i..];
        // Its closing bracket, past the ones its colours hold.
        let mut depth = 0;
        let end = call.char_indices().find_map(|(j, c)| {
            match c {
                '(' => depth += 1,
                ')' => {
                    depth -= 1;
                    if depth == 0 {
                        return Some(j + 1);
                    }
                }
                _ => {}
            }
            None
        });
        let Some(end) = end else {
            mixed.push_str(call);
            rest = "";
            break;
        };
        match color_mix(&call[..end]) {
            Some(c) => mixed.push_str(&css_color(c)),
            None => mixed.push_str(&call[..end]),
        }
        rest = &call[end..];
    }
    mixed.push_str(rest);
    std::borrow::Cow::Owned(mixed)
}

/// `color-mix(in srgb, A p%, B q%)`, either percentage left out.
fn color_mix(v: &str) -> Option<Rgba> {
    let inner = v.strip_prefix("color-mix(")?.strip_suffix(')')?;
    // Commas outside brackets (`rgba(1,2,3,1)` is one colour).
    let mut parts = vec![];
    let (mut depth, mut start) = (0, 0);
    for (i, ch) in inner.char_indices() {
        match ch {
            '(' => depth += 1,
            ')' => depth -= 1,
            ',' if depth == 0 => {
                parts.push(inner[start..i].trim());
                start = i + 1;
            }
            _ => {}
        }
    }
    parts.push(inner[start..].trim());
    let [space, a, b] = parts[..] else {
        return None;
    };
    if space != "in srgb" {
        return None;
    }
    let side = |p: &str| -> Option<(Rgba, Option<f32>)> {
        match p.rsplit_once(' ') {
            Some((c, pc)) if pc.ends_with('%') => Some((
                crate::css::color(c.trim())?,
                Some(pc.trim_end_matches('%').parse::<f32>().ok()? / 100.0),
            )),
            _ => Some((crate::css::color(p)?, None)),
        }
    };
    let ((ca, pa), (cb, pb)) = (side(a)?, side(b)?);
    let (pa, pb) = match (pa, pb) {
        (Some(x), Some(y)) => (x, y),
        (Some(x), None) => (x, 1.0 - x),
        (None, Some(y)) => (1.0 - y, y),
        (None, None) => (0.5, 0.5),
    };
    let total = pa + pb;
    if total <= 0.0 {
        return None;
    }
    let (pa, pb) = (pa / total, pb / total);
    let alpha = ca[3] * pa + cb[3] * pb;
    if alpha <= 0.0 {
        return Some([0.0; 4]);
    }
    let ch = |i: usize| (ca[i] * ca[3] * pa + cb[i] * cb[3] * pb) / alpha;
    // Fewer than 100% in all: the alpha scales down (as CSS).
    Some([ch(0), ch(1), ch(2), alpha * total.min(1.0)])
}

/// A colour as CSS, for a declaration.
pub(crate) fn css_color(c: Rgba) -> String {
    let b = |v: f32| (v.clamp(0.0, 1.0) * 255.0).round() as u8;
    format!("rgba({},{},{},{})", b(c[0]), b(c[1]), b(c[2]), c[3])
}

impl Ui {
    /// Sets `value` for `node` and everything inside it (until something
    /// inside sets its own).
    pub fn set_environment<T: 'static>(&mut self, node: NodeId, value: T) {
        self.env.values.retain(|n, _| {
            self.nodes
                .get(n.0)
                .is_some_and(|x| !x.removed && x.generation == n.1)
        });
        self.env
            .values
            .entry(node)
            .or_default()
            .insert(TypeId::of::<T>(), Rc::new(value));
    }

    /// The `T` set on `node` itself, if one is (not inherited).
    pub fn own_environment<T: Clone + 'static>(&self, node: NodeId) -> Option<T> {
        self.env
            .values
            .get(&node)
            .and_then(|m| m.get(&TypeId::of::<T>()))
            .and_then(|v| v.downcast_ref::<T>().cloned())
    }

    /// The `T` in effect at `node`: set on it or its nearest ancestor that
    /// has one, else the app-wide [global](Self::global)'s value.
    pub fn environment<T: Clone + 'static>(&self, node: NodeId) -> Option<T> {
        let mut at = Some(node);
        while let Some(n) = at.filter(|&n| self.contains(n)) {
            if let Some(v) = self
                .env
                .values
                .get(&n)
                .and_then(|m| m.get(&TypeId::of::<T>()))
            {
                return v.downcast_ref::<T>().cloned();
            }
            at = self.parent(n);
        }
        Some(self.global::<T>()?.read().clone())
    }

    /// The colours by purpose.
    pub fn tokens(&self) -> &Tokens {
        &self.env.tokens
    }

    /// New colours by purpose: everything styled by token takes them.
    pub fn set_tokens(&mut self, tokens: Tokens) {
        if self.env.tokens != tokens {
            self.env.tokens = tokens;
            self.restyle_all();
        }
    }

    /// `token`'s colour in the appearance showing.
    pub fn token(&self, token: Token) -> Rgba {
        self.token_in(self.appearance(), token)
    }

    /// `token`'s colour in `appearance` (with the contrast asked for).
    pub fn token_in(&self, appearance: Appearance, token: Token) -> Rgba {
        let set = match appearance {
            Appearance::Light => &self.env.tokens.light,
            Appearance::Dark => &self.env.tokens.dark,
        };
        let c = set.get(token);
        if self.env.increase_contrast {
            crate::legibility::stronger(token, c, set.get(Token::Text))
        } else {
            c
        }
    }

    /// Gives `node` colours by purpose: each CSS property (`background`,
    /// `color`, `border-color`, …) the token's colour in the appearance
    /// showing, kept as the appearance and the tokens change. They go over
    /// its own style and its appearance styles, under its state styles. A
    /// property given again takes its new token.
    pub fn token_style(&mut self, node: NodeId, props: &[(&str, Token)]) {
        let tokens = &mut self.states_mut(node).tokens;
        for (p, t) in props {
            // A property given again takes the new token.
            tokens.retain(|(q, _)| q != p);
            tokens.push((p.to_string(), *t));
        }
        self.restyle(node);
    }

    /// Takes back the token `node` had for `property`: its own style's value
    /// shows again.
    pub fn clear_token_style(&mut self, node: NodeId, property: &str) {
        if let Some(s) = self.nodes[node.0].states.as_mut() {
            s.tokens.retain(|(p, _)| p != property);
        }
        self.restyle(node);
    }

    /// The text size, as a multiple of each text node's own (1 by default).
    pub fn text_scale(&self) -> f32 {
        if self.env.text_scale > 0.0 {
            self.env.text_scale
        } else {
            1.0
        }
    }

    /// Scales every text node's font size and line height by `scale`
    /// (a user's larger text). Text added later is scaled as it is added;
    /// a text node's own style given after should be given unscaled
    /// through [`set_style`](Self::set_style).
    pub fn set_text_scale(&mut self, scale: f32) {
        let scale = if scale.is_finite() && scale > 0.0 {
            scale
        } else {
            1.0
        };
        if scale == self.text_scale() {
            return;
        }
        self.env.text_scale = scale;
        let texts: Vec<NodeId> = (0..self.nodes.len())
            .filter(|&i| !self.nodes[i].removed && matches!(self.nodes[i].kind, NodeKind::Text(_)))
            .map(|i| self.id_at(i))
            .collect();
        for id in texts {
            self.scale_text(id);
        }
        self.apply_scaled();
    }

    /// Sets a text node's font size and line height from its own and the
    /// scale.
    pub(crate) fn scale_text(&mut self, id: NodeId) {
        let mine = (self.style(id).font_size, self.style(id).line_height);
        let own = *self.env.unscaled.entry(id).or_insert(mine);
        let k = self.text_scale();
        let (size, line) = (own.0 * k, own.1.map(|l| l * k));
        if self.style(id).font_size != size || self.style(id).line_height != line {
            let st = self.style_mut(id);
            st.font_size = size;
            st.line_height = line;
        }
    }

    /// A text node's style was given anew: that is its own size now.
    pub(crate) fn rescale_text_after_set(&mut self, id: NodeId) {
        if self.text_scale() != 1.0 {
            let s = self.style(id);
            self.env.unscaled.insert(id, (s.font_size, s.line_height));
            self.scale_text(id);
        }
    }

    pub fn reduce_motion(&self) -> bool {
        self.env.reduce_motion
    }

    /// Animations of place, size, turn and scrolling jump to their end, for
    /// users who ask for less motion; fades and colour changes still run.
    pub fn set_reduce_motion(&mut self, on: bool) {
        self.env.reduce_motion = on;
    }

    /// `value` (a padding, a size) at the text scale: for spacing set in
    /// code that should grow with the text.
    pub fn scaled(&self, value: f32) -> f32 {
        value * self.text_scale()
    }

    /// Makes `node`'s padding, gaps, fixed sizes and corner radius follow
    /// the text size, as its text does: at twice the text, twice the
    /// padding and a row twice as tall, so larger text still fits. Its
    /// sizes now are its own; one given later through
    /// [`set_style`](Self::set_style) is taken as scaled.
    pub fn scale_with_text(&mut self, node: NodeId) {
        let s = self.style(node);
        let k = self.text_scale();
        let own = |l: Length| {
            if let Length::Px(v) = l {
                Length::Px(v / k)
            } else {
                l
            }
        };
        let m = Metrics {
            padding: Edges {
                top: s.padding.top / k,
                right: s.padding.right / k,
                bottom: s.padding.bottom / k,
                left: s.padding.left / k,
            },
            gap: s.gap / k,
            row_gap: s.row_gap.map(|g| g / k),
            width: own(s.width),
            height: own(s.height),
            min_width: own(s.min_width),
            min_height: own(s.min_height),
            radius: own(s.border_radius),
        };
        self.env.scaled.insert(node, m);
    }

    /// [`scale_with_text`](Self::scale_with_text) for `node` and every box
    /// inside it now.
    pub fn scale_tree_with_text(&mut self, node: NodeId) {
        let mut stack = vec![node];
        while let Some(n) = stack.pop() {
            if matches!(self.kind(n), NodeKind::Box) {
                self.scale_with_text(n);
            }
            stack.extend(self.children(n).iter().copied());
        }
    }

    fn apply_scaled(&mut self) {
        let k = self.text_scale();
        let nodes: Vec<(NodeId, Metrics)> = self.env.scaled.iter().map(|(n, m)| (*n, *m)).collect();
        for (n, m) in nodes {
            if !self.contains(n) {
                self.env.scaled.remove(&n);
                continue;
            }
            let px = |l: Length| {
                if let Length::Px(v) = l {
                    Length::Px(v * k)
                } else {
                    l
                }
            };
            self.update_style(n, |s| {
                s.padding = Edges {
                    top: m.padding.top * k,
                    right: m.padding.right * k,
                    bottom: m.padding.bottom * k,
                    left: m.padding.left * k,
                };
                s.gap = m.gap * k;
                s.row_gap = m.row_gap.map(|g| g * k);
                s.width = px(m.width);
                s.height = px(m.height);
                s.min_width = px(m.min_width);
                s.min_height = px(m.min_height);
                s.border_radius = px(m.radius);
            });
        }
    }

    /// How far in from each edge of the window the system draws over it (a
    /// phone's notch, its home indicator, rounded corners): the page's
    /// `env(safe-area-inset-*)`. Boxes given [`pad_safe_area`](Self::pad_safe_area)
    /// keep their content clear of it.
    pub fn set_safe_area(&mut self, insets: Edges) {
        if self.env.safe_area == insets {
            return;
        }
        self.env.safe_area = insets;
        let nodes: Vec<NodeId> = self.env.safe_nodes.keys().copied().collect();
        for n in nodes {
            self.apply_safe(n);
        }
    }

    pub fn safe_area(&self) -> Edges {
        self.env.safe_area
    }

    /// Adds the safe area's insets to `node`'s padding on the edges asked
    /// for (`[top, right, bottom, left]`): its own padding plus the inset.
    pub fn pad_safe_area(&mut self, node: NodeId, edges: [bool; 4]) {
        let own = self.style(node).padding;
        self.env.safe_nodes.insert(node, (own, edges));
        self.apply_safe(node);
    }

    fn apply_safe(&mut self, node: NodeId) {
        let Some(&(own, [t, r, b, l])) = self.env.safe_nodes.get(&node) else {
            return;
        };
        if !self.contains(node) {
            self.env.safe_nodes.remove(&node);
            return;
        }
        let i = self.env.safe_area;
        let on = |yes: bool, v: f32| if yes { v } else { 0.0 };
        let want = Edges {
            top: own.top + on(t, i.top),
            right: own.right + on(r, i.right),
            bottom: own.bottom + on(b, i.bottom),
            left: own.left + on(l, i.left),
        };
        if self.style(node).padding != want {
            self.update_style(node, |s| s.padding = want);
        }
    }

    /// Draws `node`'s text and images as plain bars (a placeholder while
    /// what it shows loads), or as themselves again.
    pub fn set_redacted(&mut self, node: NodeId, on: bool) {
        let changed = if on {
            self.env.redacted.insert(node)
        } else {
            self.env.redacted.remove(&node)
        };
        if changed {
            self.touch(node);
        }
    }

    /// Whether `node` or a box around it is drawn as a placeholder.
    pub fn is_redacted(&self, node: NodeId) -> bool {
        !self.env.redacted.is_empty()
            && self.ancestors(node).any(|a| self.env.redacted.contains(&a))
    }

    /// Lays rows out from the right, as Arabic and Hebrew read: everything
    /// is mirrored across the window, so what starts a row is at its right
    /// end, and a sideways strip starts at the right.
    pub fn set_layout_direction(&mut self, direction: LayoutDirection) {
        if self.env.direction != direction {
            self.env.direction = direction;
            self.invalidate_layout();
        }
    }

    pub fn layout_direction(&self) -> LayoutDirection {
        self.env.direction
    }

    /// Which way `node` reads: as it says itself (`direction`), else as
    /// what it is in, else as the whole UI.
    pub fn direction_of(&self, node: NodeId) -> LayoutDirection {
        self.ancestors(node)
            .find_map(|n| self.own_direction(n))
            .unwrap_or(self.env.direction)
    }

    /// Which way `node` says itself that it reads.
    fn own_direction(&self, node: NodeId) -> Option<LayoutDirection> {
        let s = self.style(node);
        if !s.direction_auto {
            return s.direction;
        }
        // The first text in it, in order, that has a letter.
        let mut stack = vec![node];
        while let Some(n) = stack.pop() {
            if let crate::NodeKind::Text(t) = self.kind(n)
                && let Some(d) = crate::bidi::first_strong(t)
            {
                return Some(d);
            }
            stack.extend(self.children(n).iter().rev());
        }
        None
    }

    /// `key` with the left and right arrows exchanged where `node` reads
    /// right to left: the key as "towards the end" and "towards the start".
    pub fn reading_key<'a>(&self, node: NodeId, key: &'a str) -> &'a str {
        match (key, self.direction_of(node)) {
            ("ArrowLeft", LayoutDirection::RightToLeft) => "ArrowRight",
            ("ArrowRight", LayoutDirection::RightToLeft) => "ArrowLeft",
            _ => key,
        }
    }

    /// Before layout: which way each node reads.
    pub(crate) fn resolve_directions(&mut self) {
        let plain = self.env.direction == LayoutDirection::LeftToRight
            && self.nodes.iter().all(|n| n.style.direction.is_none() && !n.style.direction_auto);
        if plain {
            for n in &mut self.nodes {
                n.rtl = false;
                n.flipped = false;
            }
            return;
        }
        let mut stack = vec![(self.root(), self.env.direction, false)];
        while let Some((id, around, flips)) = stack.pop() {
            let d = self.own_direction(id).unwrap_or(around);
            let n = &mut self.nodes[id.0];
            n.rtl = d == LayoutDirection::RightToLeft;
            let flips = flips || n.style.flips_rtl;
            n.flipped = flips && n.rtl;
            stack.extend(n.children.iter().map(|&c| (c, d, flips)));
        }
    }

    /// After layout: what is in a box that reads right to left mirrored
    /// across it.
    pub(crate) fn mirror_layout(&mut self) {
        self.resolve_directions();
        if self.nodes.iter().all(|n| !n.rtl) {
            return;
        }
        // Each with how far it was moved, and its box as it was laid out.
        let root = self.root();
        let mut stack = vec![(root, self.nodes[root.0].rect, self.nodes[root.0].rtl)];
        let window = (self.nodes[root.0].rect, self.nodes[root.0].rtl);
        while let Some((id, was, rtl)) = stack.pop() {
            let now = self.nodes[id.0].rect;
            for c in self.nodes[id.0].children.clone() {
                let n = &mut self.nodes[c.0];
                let laid = n.rect;
                // A fixed box is in the window, whatever it is inside.
                let (was, now, rtl) = match n.style.position {
                    crate::Position::Fixed => (window.0, window.0, window.1),
                    _ => (was, now, rtl),
                };
                n.rect.x = if rtl && !n.style.unmirrored {
                    now.x + now.width - (laid.x - was.x) - laid.width
                } else {
                    now.x + (laid.x - was.x)
                };
                stack.push((c, laid, n.rtl));
            }
        }
    }

    /// Whether reduced motion skips animating `property`.
    pub(crate) fn motion_skipped(&self, property: Property) -> bool {
        self.env.reduce_motion
            && matches!(
                property,
                Property::Left
                    | Property::Top
                    | Property::Right
                    | Property::Bottom
                    | Property::Width
                    | Property::Height
                    | Property::ScrollTop
                    | Property::ScrollLeft
                    | Property::Rotate
                    | Property::RotateX
                    | Property::RotateY
                    | Property::Depth
                    | Property::Scale
                    | Property::RotateZ
                    | Property::TranslateX
                    | Property::TranslateY
            )
    }
}

#[cfg(test)]
mod tests {
    /// `var(--token)` takes the theme's colour; `color-mix` with
    /// `transparent` keeps the hue at that opacity; anything else is left.
    #[test]
    fn theme_colours_by_name_and_mixed() {
        let tok = |t: Token| match t {
            Token::Hover => [0.2, 0.4, 0.6, 1.0],
            _ => [1.0, 0.0, 0.0, 1.0],
        };
        assert_eq!(resolve_colors("background: none", tok), "background: none");
        assert_eq!(
            resolve_colors("var(--hover)", tok),
            css_color([0.2, 0.4, 0.6, 1.0])
        );
        assert_eq!(
            resolve_colors("0 0 0 2px var(--focus)", tok),
            format!("0 0 0 2px {}", css_color([1.0, 0.0, 0.0, 1.0]))
        );
        assert_eq!(
            resolve_colors("var(--nope)", tok),
            "var(--nope)",
            "an unknown name is left"
        );
        assert_eq!(
            resolve_colors(
                "color-mix(in srgb, var(--destructive) 12%, transparent)",
                tok
            ),
            css_color([1.0, 0.0, 0.0, 0.12])
        );
        assert_eq!(
            resolve_colors("color-mix(in srgb, #000000, #ffffff)", tok),
            css_color([0.5, 0.5, 0.5, 1.0]),
            "half and half"
        );
        // Each of several, in a gradient: the rest left as it was.
        let g = resolve_colors(
            "linear-gradient(90deg, color-mix(in srgb, #000000, #ffffff), color-mix(in srgb, #ff0000 50%, transparent))",
            tok,
        );
        assert!(!g.contains("color-mix"), "{g}");
        assert!(g.starts_with("linear-gradient(90deg, "), "{g}");
        assert!(Token::ALL
            .iter()
            .all(|t| Token::from_css_name(t.css_name()) == Some(*t)));
    }

    use super::*;
    use crate::{style, FixedWidthText};

    #[test]
    fn values_flow_down_and_the_nearest_wins() {
        #[derive(Clone, Debug, PartialEq)]
        struct Density(u8);
        let mut ui = Ui::new();
        let root = ui.root();
        let a = ui.add_box(root, style! {});
        let b = ui.add_box(a, style! {});
        let c = ui.add_box(b, style! {});
        assert_eq!(ui.environment::<Density>(c), None);
        ui.set_global(Density(1));
        assert_eq!(ui.environment::<Density>(c), Some(Density(1)));
        ui.set_environment(a, Density(2));
        assert_eq!(ui.environment::<Density>(c), Some(Density(2)));
        ui.set_environment(b, Density(3));
        assert_eq!(ui.environment::<Density>(c), Some(Density(3)));
        assert_eq!(ui.environment::<Density>(a), Some(Density(2)));
    }

    #[test]
    fn tokens_follow_the_appearance_and_new_tokens_and_states_go_over_them() {
        let mut ui = Ui::new();
        let root = ui.root();
        let b = ui.add_box(root, style! {});
        ui.token_style(b, &[("background", Token::Surface)]);
        ui.hover_style(b, "background: #ff0000");
        let surface = ui.token(Token::Surface);
        assert_eq!(ui.style(b).background, Some(surface));
        let mut t = Tokens::default();
        t.dark.surface = [0.0, 0.5, 0.0, 1.0];
        t.light.surface = [0.0, 0.5, 0.0, 1.0];
        ui.set_tokens(t);
        assert_eq!(
            ui.style(b).background.map(|c| (c[1] * 255.0).round()),
            Some(128.0)
        );
        ui.set_interaction(crate::states::Interaction {
            hovered: vec![b],
            ..Default::default()
        });
        assert_eq!(ui.style(b).background, Some([1.0, 0.0, 0.0, 1.0]));
    }

    #[test]
    fn text_scales_and_motion_reduces() {
        let mut ui = Ui::new();
        let root = ui.root();
        let b = ui.add_box(root, style! { font-size: 10px; });
        let t = ui.add_text(b, "abc");
        ui.layout(400.0, 300.0, &mut FixedWidthText(7.0));
        let h = ui.rect(t).height;
        ui.set_text_scale(2.0);
        assert_eq!(ui.style(t).font_size, 20.0);
        let late = ui.add_text(b, "later");
        assert_eq!(ui.style(late).font_size, 20.0, "scaled as it is added");
        ui.layout(400.0, 300.0, &mut FixedWidthText(7.0));
        assert_eq!(ui.rect(t).height, 2.0 * h);
        ui.set_text_scale(1.0);
        assert_eq!(ui.style(t).font_size, 10.0);
        ui.set_reduce_motion(true);
        ui.animate(
            b,
            crate::animation::Animation::new(
                Property::Left,
                crate::animation::Value::Number(0.0),
                crate::animation::Value::Number(50.0),
                1.0,
            ),
        );
        assert!(!ui.is_animating(b));
        assert_eq!(ui.style(b).left, crate::Length::Px(50.0));
    }
}

#[cfg(test)]
mod animated_state_tests {
    use crate::animation::{Animation, Property, Value};
    use crate::{style, Length, Ui};

    /// A node with state styles keeps where an animation put it when the
    /// pointer moves and it is restyled.
    #[test]
    fn a_restyle_keeps_what_an_animation_moved() {
        let mut ui = Ui::new();
        let root = ui.root();
        let b = ui.add_box(root, style! { position: absolute; left: 0px; });
        ui.token_style(b, &[("background", super::Token::Accent)]);
        ui.hover_style(b, "opacity: 0.5");
        ui.animate(
            b,
            Animation::new(
                Property::Left,
                Value::Number(0.0),
                Value::Number(100.0),
                0.2,
            ),
        );
        ui.step(0.3);
        assert_eq!(ui.style(b).left, Length::Px(100.0));
        ui.set_interaction(crate::states::Interaction {
            hovered: vec![b],
            ..Default::default()
        });
        assert_eq!(ui.style(b).left, Length::Px(100.0), "not back at 0");
        assert_eq!(ui.style(b).opacity, 0.5, "and its hover style on top");
    }
}
