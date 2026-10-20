//! CSS declarations onto a [`Style`]: what [`style!`](crate::style) expands
//! to.
//!
//! Values are CSS text (`20px`, `50%`, `auto`, `#9a7518`,
//! `rgba(0, 0, 0, 0.3)`, `1px solid #ccc`, `blur(8px)`) or, written `{expr}`
//! in the macro, a Rust value of the property's type ([`Value`]).

use super::style::{AlignItems, BoxSizing, Display, Edges, FlexDirection, FontWeight, JustifyContent, Length, Position, Rgba, Style};

/// A computed value for a property, given as a Rust expression.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Number(f32),
    Length(Length),
    Color(Rgba),
    OptionalColor(Option<Rgba>),
    Edges(Edges),
    Text(String),
    Bool(bool),
    Display(Display),
    Position(Position),
    BoxSizing(BoxSizing),
    FlexDirection(FlexDirection),
    AlignItems(AlignItems),
    JustifyContent(JustifyContent),
    FontWeight(FontWeight),
    Gradient(Vec<(f32, Rgba)>),
}

macro_rules! from {
    ($($t:ty => $v:ident),* $(,)?) => {
        $(impl From<$t> for Value {
            fn from(v: $t) -> Self {
                Value::$v(v.into())
            }
        })*
    };
}

from! {
    f32 => Number,
    Length => Length,
    Rgba => Color,
    Option<Rgba> => OptionalColor,
    Edges => Edges,
    String => Text,
    &str => Text,
    bool => Bool,
    Display => Display,
    Position => Position,
    BoxSizing => BoxSizing,
    FlexDirection => FlexDirection,
    AlignItems => AlignItems,
    JustifyContent => JustifyContent,
    FontWeight => FontWeight,
    Vec<(f32, Rgba)> => Gradient,
}

fn bad(name: &str, value: &str) -> ! {
    panic!("style!: unsupported `{name}: {value}`")
}

/// `stringify!` spaces tokens apart (`sans - serif`, `- 1px`, `# 222`,
/// `rgba (0 , 0)`); put CSS back together.
fn normalize(text: &str) -> String {
    let tokens: Vec<&str> = text.split_whitespace().collect();
    let mut out = String::with_capacity(text.len());
    let mut glue = true;
    for (i, t) in tokens.iter().enumerate() {
        let next = tokens.get(i + 1).copied().unwrap_or("");
        let joins_previous = matches!(t.chars().next(), Some(',' | ')' | '(' | '%'))
            // `sans - serif`: a hyphen inside a word.
            || (*t == "-" && next.starts_with(|c: char| c.is_alphabetic()) && out.ends_with(|c: char| c.is_alphanumeric()))
            || (out.ends_with('-') && t.starts_with(|c: char| c.is_alphabetic()) && out.len() > 1 && !out.ends_with(" -"));
        if !out.is_empty() && !glue && !joins_previous {
            out.push(' ');
        }
        out.push_str(t);
        // No space after a sign, `#` or `(`.
        glue = matches!(*t, "-" | "#" | "(") || t.ends_with('(');
    }
    out
}

/// Splits a value at top-level whitespace (not inside parentheses).
fn words(v: &str) -> Vec<&str> {
    let (mut out, mut depth, mut start) = (vec![], 0, None);
    for (i, c) in v.char_indices() {
        match c {
            '(' => depth += 1,
            ')' => depth -= 1,
            ' ' if depth == 0 => {
                if let Some(s) = start.take() {
                    out.push(&v[s..i]);
                }
                continue;
            }
            _ => {}
        }
        start.get_or_insert(i);
    }
    if let Some(s) = start {
        out.push(&v[s..]);
    }
    out
}

/// A CSS length: `12px`, `12`, `50%`, `auto`. (`px` is the only unit.)
pub fn length(v: &str) -> Option<Length> {
    let v = v.trim();
    if v == "auto" {
        return Some(Length::Auto);
    }
    if let Some(p) = v.strip_suffix('%') {
        return p.trim().parse().ok().map(Length::Percent);
    }
    v.strip_suffix("px").unwrap_or(v).trim().parse().ok().map(Length::Px)
}

fn px(v: &str) -> Option<f32> {
    match length(v)? {
        Length::Px(p) => Some(p),
        _ => None,
    }
}

/// A CSS colour: `#rgb`, `#rrggbb`, `#rrggbbaa`, `rgb(…)`, `rgba(…)` (0–255
/// and an alpha 0–1), or `white`, `black`, `transparent`, `orange`.
pub fn color(v: &str) -> Option<Rgba> {
    let v = v.trim();
    let named = |r, g, b| Some([r as f32 / 255.0, g as f32 / 255.0, b as f32 / 255.0, 1.0]);
    match v {
        "white" => return named(255, 255, 255),
        "black" => return named(0, 0, 0),
        "orange" => return named(255, 165, 0),
        "yellow" => return named(255, 255, 0),
        "red" => return named(255, 0, 0),
        "lime" => return named(0, 255, 0),
        "blue" => return named(0, 0, 255),
        "gray" | "grey" => return named(128, 128, 128),
        "transparent" => return Some([0.0; 4]),
        _ => {}
    }
    if let Some(hex) = v.strip_prefix('#') {
        let digits: Vec<u8> = hex.chars().map(|c| c.to_digit(16).map(|d| d as u8)).collect::<Option<_>>()?;
        let channels: Vec<u8> = match digits.len() {
            3 | 4 => digits.iter().map(|d| d * 17).collect(),
            6 | 8 => digits.chunks(2).map(|p| p[0] * 16 + p[1]).collect(),
            _ => return None,
        };
        let a = channels.get(3).map_or(1.0, |&a| a as f32 / 255.0);
        return Some([channels[0] as f32 / 255.0, channels[1] as f32 / 255.0, channels[2] as f32 / 255.0, a]);
    }
    let args = v.strip_prefix("rgba(").or_else(|| v.strip_prefix("rgb("))?.strip_suffix(')')?;
    let n: Vec<f32> = args.split(',').map(|a| a.trim().parse().ok()).collect::<Option<_>>()?;
    match n[..] {
        [r, g, b] => Some([r / 255.0, g / 255.0, b / 255.0, 1.0]),
        [r, g, b, a] => Some([r / 255.0, g / 255.0, b / 255.0, a]),
        _ => None,
    }
}

/// `margin`/`padding`/`border-width`: one to four lengths, CSS order.
fn edges(v: &str) -> Option<Edges> {
    let n: Vec<f32> = words(v).iter().map(|w| px(w)).collect::<Option<_>>()?;
    Some(match n[..] {
        [a] => Edges::all(a),
        [v, h] => Edges::new(v, h, v, h),
        [t, h, b] => Edges::new(t, h, b, h),
        [t, r, b, l] => Edges::new(t, r, b, l),
        _ => return None,
    })
}

fn edge(e: &mut Edges, side: &str, v: f32) -> bool {
    match side {
        "top" => e.top = v,
        "right" => e.right = v,
        "bottom" => e.bottom = v,
        "left" => e.left = v,
        _ => return false,
    }
    true
}

fn align(v: &str) -> Option<AlignItems> {
    Some(match v {
        "stretch" => AlignItems::Stretch,
        "flex-start" | "start" => AlignItems::FlexStart,
        "center" => AlignItems::Center,
        "flex-end" | "end" => AlignItems::FlexEnd,
        _ => return None,
    })
}

/// Sets property `name` from CSS text.
pub fn apply(s: &mut Style, name: &str, value: &str) {
    let value = normalize(value);
    let v = value.as_str();
    let ok = (|| -> Option<()> {
        match name {
            "display" => {
                s.display = match v {
                    "block" | "inline-block" => Display::Block,
                    "flex" | "inline-flex" => Display::Flex,
                    "none" => Display::None,
                    _ => return None,
                }
            }
            "position" => {
                s.position = match v {
                    "static" => Position::Static,
                    "relative" => Position::Relative,
                    "absolute" => Position::Absolute,
                    _ => return None,
                }
            }
            "box-sizing" => {
                s.box_sizing = match v {
                    "content-box" => BoxSizing::ContentBox,
                    "border-box" => BoxSizing::BorderBox,
                    _ => return None,
                }
            }
            "flex-direction" => {
                s.flex_direction = match v {
                    "row" => FlexDirection::Row,
                    "column" => FlexDirection::Column,
                    _ => return None,
                }
            }
            "flex-grow" => s.flex_grow = v.parse().ok()?,
            "flex-shrink" => s.flex_shrink = v.parse().ok()?,
            "flex-basis" => s.flex_basis = length(v)?,
            // `flex: <grow> [<shrink>] [<basis>]`; one number is `n 1 0%`.
            "flex" => {
                let w = words(v);
                s.flex_grow = w.first()?.parse().ok()?;
                s.flex_shrink = w.get(1).map_or(Some(1.0), |x| x.parse().ok())?;
                s.flex_basis = w.get(2).map_or(Some(Length::Percent(0.0)), |x| length(x))?;
            }
            "align-items" => s.align_items = align(v)?,
            "align-self" => s.align_self = if v == "auto" { None } else { Some(align(v)?) },
            "justify-content" => {
                s.justify_content = match v {
                    "flex-start" | "start" => JustifyContent::FlexStart,
                    "center" => JustifyContent::Center,
                    "flex-end" | "end" | "right" => JustifyContent::FlexEnd,
                    "space-between" => JustifyContent::SpaceBetween,
                    _ => return None,
                }
            }
            "gap" => s.gap = px(v)?,
            "width" => s.width = length(v)?,
            "height" => s.height = length(v)?,
            "min-width" => s.min_width = length(v)?,
            "min-height" => s.min_height = length(v)?,
            "max-width" => s.max_width = length(v)?,
            "max-height" => s.max_height = length(v)?,
            "margin" => s.margin = edges(v)?,
            "padding" => s.padding = edges(v)?,
            "border-width" => s.border = edges(v)?,
            "border-color" => s.border_color = color(v)?,
            // `border: <width> [solid] [<color>]`.
            "border" => {
                for w in words(v) {
                    if let Some(p) = px(w) {
                        s.border = Edges::all(p);
                    } else if let Some(c) = color(w) {
                        s.border_color = c;
                    } else if w != "solid" {
                        return None;
                    }
                }
            }
            // `border-left: <width> [solid] [<color>]`, …: one edge's width
            // (the box has one border colour).
            "border-top" | "border-right" | "border-bottom" | "border-left" => {
                for w in words(v) {
                    if let Some(p) = px(w) {
                        match name {
                            "border-top" => s.border.top = p,
                            "border-right" => s.border.right = p,
                            "border-bottom" => s.border.bottom = p,
                            _ => s.border.left = p,
                        }
                    } else if let Some(c) = color(w) {
                        s.border_color = c;
                    } else if w != "solid" {
                        return None;
                    }
                }
            }
            "border-radius" => s.border_radius = length(v)?,
            "background" | "background-color" => s.background = if v == "none" { None } else { Some(color(v)?) },
            "color" => s.color = color(v)?,
            "font-family" => s.font_family = v.trim_matches('"').to_owned(),
            "font-size" => s.font_size = px(v)?,
            "font-weight" => {
                s.font_weight = match v {
                    "normal" => FontWeight::Normal,
                    "bold" => FontWeight::Bold,
                    n => FontWeight::Weight(n.parse().ok()?),
                }
            }
            "line-height" => s.line_height = if v == "normal" { None } else { Some(px(v)?) },
            "top" => s.top = length(v)?,
            "left" => s.left = length(v)?,
            "right" => s.right = length(v)?,
            "bottom" => s.bottom = length(v)?,
            "overflow" | "overflow-y" => {
                (s.clip, s.scroll) = match v {
                    "hidden" | "clip" => (true, false),
                    "auto" | "scroll" => (true, true),
                    "visible" => (false, false),
                    _ => return None,
                }
            }
            "font-style" => s.italic = v == "italic" || v == "oblique",
            "white-space" => s.wrap = v != "nowrap" && v != "pre",
            "word-wrap" | "overflow-wrap" => s.wrap |= v == "break-word" || v == "anywhere",
            "writing-mode" => s.vertical = v.starts_with("vertical"),
            "background-clip" => {
                s.background_clip_content = match v {
                    "content-box" => true,
                    "border-box" | "padding-box" => false,
                    _ => return None,
                }
            }
            "opacity" => s.opacity = v.parse().ok()?,
            "text-decoration" | "text-decoration-line" => s.line_through = v == "line-through",
            "filter" => s.blur = px(v.strip_prefix("blur(")?.strip_suffix(')')?)?,
            _ => {
                let (base, side) = name.rsplit_once('-')?;
                match base {
                    "margin" => edge(&mut s.margin, side, px(v)?).then_some(())?,
                    "padding" => edge(&mut s.padding, side, px(v)?).then_some(())?,
                    _ => return None,
                }
            }
        }
        Some(())
    })();
    if ok.is_none() {
        bad(name, v);
    }
}

/// Sets property `name` from a Rust value.
pub fn set(s: &mut Style, name: &str, value: Value) {
    use Value as V;
    let ok = match (name, value) {
        ("display", V::Display(d)) => {
            s.display = d;
            true
        }
        ("position", V::Position(p)) => {
            s.position = p;
            true
        }
        ("box-sizing", V::BoxSizing(b)) => {
            s.box_sizing = b;
            true
        }
        ("flex-direction", V::FlexDirection(d)) => {
            s.flex_direction = d;
            true
        }
        ("align-items", V::AlignItems(a)) => {
            s.align_items = a;
            true
        }
        ("align-self", V::AlignItems(a)) => {
            s.align_self = Some(a);
            true
        }
        ("justify-content", V::JustifyContent(j)) => {
            s.justify_content = j;
            true
        }
        ("font-weight", V::FontWeight(w)) => {
            s.font_weight = w;
            true
        }
        ("margin", V::Edges(e)) => {
            s.margin = e;
            true
        }
        ("padding", V::Edges(e)) => {
            s.padding = e;
            true
        }
        ("border-width", V::Edges(e)) => {
            s.border = e;
            true
        }
        ("background" | "background-color", V::OptionalColor(c)) => {
            s.background = c;
            true
        }
        ("background" | "background-color", V::Color(c)) => {
            s.background = Some(c);
            true
        }
        ("background" | "background-image", V::Gradient(g)) => {
            s.background_gradient = g;
            true
        }
        ("color", V::Color(c)) => {
            s.color = c;
            true
        }
        ("border-color", V::Color(c)) => {
            s.border_color = c;
            true
        }
        ("font-family", V::Text(t)) => {
            s.font_family = t;
            true
        }
        ("overflow", V::Bool(b)) => {
            s.clip = b;
            true
        }
        ("text-decoration", V::Bool(b)) => {
            s.line_through = b;
            true
        }
        (name, V::Length(l)) => set_length(s, name, l),
        (name, V::Number(n)) => match name {
            "flex-grow" => {
                s.flex_grow = n;
                true
            }
            "flex-shrink" => {
                s.flex_shrink = n;
                true
            }
            "opacity" => {
                s.opacity = n;
                true
            }
            "font-size" => {
                s.font_size = n;
                true
            }
            "line-height" => {
                s.line_height = Some(n);
                true
            }
            "gap" => {
                s.gap = n;
                true
            }
            "filter" => {
                s.blur = n;
                true
            }
            "scroll-top" => {
                s.scroll_top = n;
                true
            }
            // A bare number is pixels, as in CSS-in-JS.
            name => set_length(s, name, Length::Px(n)),
        },
        (name, V::Text(t)) => {
            apply(s, name, &t);
            true
        }
        _ => false,
    };
    if !ok {
        panic!("style!: `{name}` does not take that kind of value");
    }
}

fn set_length(s: &mut Style, name: &str, l: Length) -> bool {
    let px = |l: Length| match l {
        Length::Px(p) => p,
        _ => panic!("style!: `{name}` takes pixels"),
    };
    match name {
        "width" => s.width = l,
        "height" => s.height = l,
        "min-width" => s.min_width = l,
        "min-height" => s.min_height = l,
        "max-width" => s.max_width = l,
        "max-height" => s.max_height = l,
        "flex-basis" => s.flex_basis = l,
        "top" => s.top = l,
        "left" => s.left = l,
        "right" => s.right = l,
        "bottom" => s.bottom = l,
        "border-radius" => s.border_radius = l,
        "margin" => s.margin = Edges::all(px(l)),
        "padding" => s.padding = Edges::all(px(l)),
        "border-width" => s.border = Edges::all(px(l)),
        _ => {
            let Some((base, side)) = name.rsplit_once('-') else { return false };
            let v = px(l);
            return match base {
                "margin" => edge(&mut s.margin, side, v),
                "padding" => edge(&mut s.padding, side, v),
                _ => false,
            };
        }
    }
    true
}

/// Builds a [`Style`] from CSS declarations.
///
/// ```
/// use threers::style;
/// use threers::ui::{Edges, Length, Display};
///
/// let width = 120.0;
/// let s = style! {
///     display: flex;
///     flex-direction: column;
///     margin: 0 0 3px;
///     margin-left: -5px;
///     background: #222;
///     border: 1px solid rgba(255, 255, 255, 0.5);
///     font-family: "monospace";
///     width: {width};
/// };
/// assert_eq!(s.display, Display::Flex);
/// assert_eq!(s.margin, Edges::new(0.0, 0.0, 3.0, -5.0));
/// assert_eq!(s.background, Some([34.0 / 255.0, 34.0 / 255.0, 34.0 / 255.0, 1.0]));
/// assert_eq!(s.width, Length::Px(120.0));
/// ```
///
/// A declaration's value is CSS (parsed when the style is built; a mistake
/// panics, naming it) or `{expr}`, a Rust value: a number (pixels, or the
/// property's number), a [`Length`](super::Length), colour, [`Edges`], text,
/// `bool` (`overflow` → clip, `text-decoration` → line-through) or one of the
/// style enums. `..base;` first starts from another style instead of the
/// default.
#[macro_export]
macro_rules! style {
    ($($body:tt)*) => {{
        // `..base;` replaces the default before it is read.
        #[allow(unused_mut, unused_assignments)]
        let mut style = $crate::ui::Style::default();
        $crate::__style_declarations!(style; $($body)*);
        style
    }};
}

#[doc(hidden)]
#[macro_export]
macro_rules! __style_declarations {
    ($s:ident;) => {};
    ($s:ident; .. $base:expr; $($rest:tt)*) => {
        $s = $base;
        $crate::__style_declarations!($s; $($rest)*);
    };
    ($s:ident; $($name:ident)-+ : { $value:expr } $(; $($rest:tt)*)?) => {
        $crate::ui::css::set(&mut $s, &[$(stringify!($name)),+].join("-"), $crate::ui::css::Value::from($value));
        $crate::__style_declarations!($s; $($($rest)*)?);
    };
    ($s:ident; $($name:ident)-+ : $($rest:tt)*) => {
        $crate::__style_value!($s; [$(stringify!($name)),+] [] $($rest)*);
    };
}

/// Collects a value's tokens up to `;` (or the end).
#[doc(hidden)]
#[macro_export]
macro_rules! __style_value {
    ($s:ident; [$($n:expr),+] [$($v:tt)*] ; $($rest:tt)*) => {
        $crate::ui::css::apply(&mut $s, &[$($n),+].join("-"), stringify!($($v)*));
        $crate::__style_declarations!($s; $($rest)*);
    };
    ($s:ident; [$($n:expr),+] [$($v:tt)*]) => {
        $crate::ui::css::apply(&mut $s, &[$($n),+].join("-"), stringify!($($v)*));
    };
    ($s:ident; $n:tt [$($v:tt)*] $t:tt $($rest:tt)*) => {
        $crate::__style_value!($s; $n [$($v)* $t] $($rest)*);
    };
}

/// Adds boxes and text to a [`Ui`](super::Ui) as nested markup, and evaluates
/// to the first node added.
///
/// ```
/// use threers::{style, ui};
/// use threers::ui::{FixedWidthText, Ui};
///
/// let mut tree = Ui::new();
/// let root = tree.root();
/// let bar = ui!(tree, root,
///     div(display: flex; height: 20px; background: #222) {
///         div(padding: 0 2px; font-size: 13px) { "x" }
///         div(flex: 1) {}
///     }
/// );
/// tree.layout(200.0, 100.0, &mut FixedWidthText(7.0));
/// assert_eq!(tree.rect(bar).height, 20.0);
/// assert_eq!(tree.children(bar).len(), 2);
/// ```
///
/// A node is `div(<declarations>) { <children> }`, a string literal (a text
/// run in its parent's font) or `{expr}` (text from a Rust expression).
#[macro_export]
macro_rules! ui {
    ($tree:expr, $parent:expr, $($nodes:tt)+) => {{
        let tree: &mut $crate::ui::Ui = &mut $tree;
        let parent: $crate::ui::NodeId = $parent;
        let mut first: Option<$crate::ui::NodeId> = None;
        $crate::__ui_nodes!(tree, parent, first; $($nodes)+);
        first.expect("ui!: no nodes")
    }};
}

#[doc(hidden)]
#[macro_export]
macro_rules! __ui_nodes {
    ($tree:ident, $parent:ident, $first:ident;) => {};
    ($tree:ident, $parent:ident, $first:ident; div ( $($style:tt)* ) { $($children:tt)* } $($rest:tt)*) => {
        let node = $tree.add_box($parent, $crate::style!($($style)*));
        $first.get_or_insert(node);
        {
            #[allow(unused_variables, unused_mut)]
            let mut first: Option<$crate::ui::NodeId> = None;
            $crate::__ui_nodes!($tree, node, first; $($children)*);
        }
        $crate::__ui_nodes!($tree, $parent, $first; $($rest)*);
    };
    ($tree:ident, $parent:ident, $first:ident; $text:literal $($rest:tt)*) => {
        let node = $tree.add_text($parent, $text);
        $first.get_or_insert(node);
        $crate::__ui_nodes!($tree, $parent, $first; $($rest)*);
    };
    ($tree:ident, $parent:ident, $first:ident; { $text:expr } $($rest:tt)*) => {
        let node = $tree.add_text($parent, $text);
        $first.get_or_insert(node);
        $crate::__ui_nodes!($tree, $parent, $first; $($rest)*);
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_css_values() {
        assert_eq!(length("12px"), Some(Length::Px(12.0)));
        assert_eq!(length("-1.5px"), Some(Length::Px(-1.5)));
        assert_eq!(length("20%"), Some(Length::Percent(20.0)));
        assert_eq!(color("#9a7518"), Some([154.0 / 255.0, 117.0 / 255.0, 24.0 / 255.0, 1.0]));
        assert_eq!(color("#f00"), Some([1.0, 0.0, 0.0, 1.0]));
        assert_eq!(color("rgba(0, 0, 0, 0.3)"), Some([0.0, 0.0, 0.0, 0.3]));
        assert_eq!(edges("1px 1px 1px 5px"), Some(Edges::new(1.0, 1.0, 1.0, 5.0)));
        // One edge's border.
        let mut e = Style::default();
        apply(&mut e, "border-left", "2px solid #181818");
        apply(&mut e, "border-bottom", "3px");
        assert_eq!(e.border, Edges::new(0.0, 0.0, 3.0, 2.0));
        assert_eq!(e.border_color, color("#181818").unwrap());
        assert_eq!(normalize("- 1 px"), "-1 px");
        assert_eq!(normalize("rgba (0 , 0 , 0 , 0.3)"), "rgba(0, 0, 0, 0.3)");
        assert_eq!(normalize("# 222"), "#222");
        assert_eq!(normalize("sans - serif"), "sans-serif");
        assert_eq!(normalize("0 - 5px"), "0 -5px");
        assert_eq!(normalize("flex - start"), "flex-start");
    }

    #[test]
    fn the_macro_reads_like_css() {
        let hover = true;
        let s = crate::style! {
            position: absolute;
            top: 0; left: 0; right: 0; bottom: 16px;
            min-width: 18px;
            padding: 0 2px;
            margin-top: -1px;
            align-items: center;
            justify-content: space-between;
            border-radius: 20%;
        };
        let _ = s;
        let t = crate::style! {
            ..crate::ui::Style::flex(FlexDirection::Column);
            font-family: monospace;
            font-size: 13px;
            font-weight: 900;
            color: {if hover { [1.0; 4] } else { [0.0, 0.0, 0.0, 1.0] }};
            background: {None};
            border: 1px solid #ccc;
            overflow: hidden;
            filter: blur(9.6px);
            text-decoration: line-through;
            flex: 1;
        };
        assert_eq!(t.flex_direction, FlexDirection::Column);
        assert_eq!(t.font_family, "monospace");
        assert_eq!(t.font_weight, FontWeight::Weight(900));
        assert_eq!(t.color, [1.0; 4]);
        assert_eq!(t.border, Edges::all(1.0));
        assert!(t.clip && t.line_through);
        assert_eq!(t.blur, 9.6);
        assert_eq!((t.flex_grow, t.flex_shrink, t.flex_basis), (1.0, 1.0, Length::Percent(0.0)));
    }
}
