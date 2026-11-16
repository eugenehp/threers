//! CSS declarations onto a [`Style`]: what [`style!`](crate::style) expands
//! to.
//!
//! Values are CSS text (`20px`, `50%`, `auto`, `#9a7518`,
//! `rgba(0, 0, 0, 0.3)`, `1px solid #ccc`, `blur(8px)`) or, written `{expr}`
//! in the macro, a Rust value of the property's type ([`Value`]).

use super::style::{
    AlignContent, AlignItems, BoxShadow, BoxSizing, Cursor, Display, Edges, FlexDirection,
    FlexWrap, FontWeight, GridPlacement, JustifyContent, Length, ObjectFit, Position, Rgba,
    Scrollbar, ScrollbarVisibility, Style, Track, TrackSize, UserSelect,
};

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
    Cursor(Cursor),
    ObjectFit(ObjectFit),
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
    Cursor => Cursor,
    ObjectFit => ObjectFit,
}

/// `solid` or `dashed`: whether dashed.
fn dashed(v: &str) -> Option<bool> {
    match v {
        "solid" => Some(false),
        "dashed" => Some(true),
        _ => None,
    }
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
pub(crate) fn words(v: &str) -> Vec<&str> {
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
    v.strip_suffix("px")
        .unwrap_or(v)
        .trim()
        .parse()
        .ok()
        .map(Length::Px)
}

/// A CSS time, `300ms` or `1.5s`, in seconds.
pub(crate) fn seconds(v: &str) -> Option<f32> {
    let v = v.trim();
    let t = if let Some(ms) = v.strip_suffix("ms") {
        ms.trim().parse::<f32>().ok()? / 1000.0
    } else {
        v.strip_suffix('s')?.trim().parse::<f32>().ok()?
    };
    (t >= 0.0 && t.is_finite()).then_some(t)
}

pub(crate) fn px(v: &str) -> Option<f32> {
    match length(v)? {
        Length::Px(p) => Some(p),
        _ => None,
    }
}

/// A CSS colour: `#rgb`, `#rrggbb`, `#rrggbbaa`, `rgb(…)`/`rgba(…)` (0–255
/// or percentages; commas, or spaces and `/ alpha`), `hsl(…)`, `oklch(…)`,
/// or `white`, `black`, `transparent`, `orange` and a few more names.
/// `linear-gradient(…)`, `radial-gradient(…)` or `conic-gradient(…)`:
/// the shape and the stops (positions given or spread evenly).
fn gradient(v: &str) -> Option<(crate::style::GradientShape, Vec<(f32, Rgba)>)> {
    use crate::style::GradientShape;
    let (kind, rest) = v.split_once('(')?;
    let body = rest.strip_suffix(')')?;
    // Arguments at the top level (`rgb(…)` holds commas of its own).
    let mut args = vec![];
    let (mut depth, mut start) = (0, 0);
    for (i, c) in body.char_indices() {
        match c {
            '(' => depth += 1,
            ')' => depth -= 1,
            ',' if depth == 0 => {
                args.push(body[start..i].trim());
                start = i + 1;
            }
            _ => {}
        }
    }
    args.push(body[start..].trim());
    // Where: `50% 50%`, `center`, `top left` and the like.
    let place = |at: &str| -> Option<(f32, f32)> {
        let mut xy = (0.5, 0.5);
        let words: Vec<&str> = at.split_whitespace().collect();
        for (i, w) in words.iter().enumerate() {
            match *w {
                "left" => xy.0 = 0.0,
                "right" => xy.0 = 1.0,
                "top" => xy.1 = 0.0,
                "bottom" => xy.1 = 1.0,
                "center" => {}
                p => {
                    let f = p.strip_suffix('%')?.parse::<f32>().ok()? / 100.0;
                    if i == 0 {
                        xy.0 = f
                    } else {
                        xy.1 = f
                    }
                }
            }
        }
        Some(xy)
    };
    let angle = |a: &str| -> Option<f32> {
        if let Some(d) = a.strip_suffix("deg") {
            return d.trim().parse().ok();
        }
        if let Some(t) = a.strip_suffix("turn") {
            return t.trim().parse::<f32>().ok().map(|t| t * 360.0);
        }
        None
    };
    let first = args.first().copied().unwrap_or("");
    let (shape, from) = match kind.trim() {
        "linear-gradient" => {
            let side = match first {
                "to top" => Some(0.0),
                "to right" => Some(90.0),
                "to bottom" => Some(180.0),
                "to left" => Some(270.0),
                "to top right" | "to right top" => Some(45.0),
                "to bottom right" | "to right bottom" => Some(135.0),
                "to bottom left" | "to left bottom" => Some(225.0),
                "to top left" | "to left top" => Some(315.0),
                a => angle(a),
            };
            // A gradient with no direction runs top to bottom, as in CSS.
            (
                GradientShape::Linear(side.unwrap_or(180.0)),
                usize::from(side.is_some()),
            )
        }
        "radial-gradient" => {
            let config = !first.starts_with('#') && color(first).is_none();
            let (cx, cy) = match first.split_once("at ") {
                Some((_, at)) if config => place(at)?,
                _ => (0.5, 0.5),
            };
            (GradientShape::Radial { cx, cy }, usize::from(config))
        }
        "conic-gradient" => {
            let config = !first.starts_with('#') && color(first).is_none();
            let mut from = 0.0;
            let mut at = (0.5, 0.5);
            if config {
                let (head, tail) = first
                    .split_once("at ")
                    .map_or((first, None), |(h, t)| (h, Some(t)));
                if let Some(f) = head.trim().strip_prefix("from ") {
                    from = angle(f.trim())?;
                }
                if let Some(t) = tail {
                    at = place(t)?;
                }
            }
            (
                GradientShape::Conic {
                    cx: at.0,
                    cy: at.1,
                    from,
                },
                usize::from(config),
            )
        }
        _ => return None,
    };
    let stops: Vec<(Rgba, Option<f32>)> = args[from..]
        .iter()
        .map(|a| match a.rsplit_once(' ') {
            Some((c, p)) if p.ends_with('%') => Some((
                color(c)?,
                Some(p.strip_suffix('%')?.parse::<f32>().ok()? / 100.0),
            )),
            _ => Some((color(a)?, None)),
        })
        .collect::<Option<_>>()?;
    if stops.len() < 2 {
        return None;
    }
    let n = stops.len() - 1;
    Some((
        shape,
        stops
            .iter()
            .enumerate()
            .map(|(i, &(c, p))| (p.unwrap_or(i as f32 / n as f32), c))
            .collect(),
    ))
}

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
        let digits: Vec<u8> = hex
            .chars()
            .map(|c| c.to_digit(16).map(|d| d as u8))
            .collect::<Option<_>>()?;
        let channels: Vec<u8> = match digits.len() {
            3 | 4 => digits.iter().map(|d| d * 17).collect(),
            6 | 8 => digits.chunks(2).map(|p| p[0] * 16 + p[1]).collect(),
            _ => return None,
        };
        let a = channels.get(3).map_or(1.0, |&a| a as f32 / 255.0);
        return Some([
            channels[0] as f32 / 255.0,
            channels[1] as f32 / 255.0,
            channels[2] as f32 / 255.0,
            a,
        ]);
    }
    let (function, args) = v.strip_suffix(')')?.split_once('(')?;
    // `rgb(1, 2, 3, 0.5)` or `rgb(1 2 3 / 50%)`: three channels and an alpha.
    let (channels, alpha) = match args.split_once('/') {
        Some((c, a)) => (c, Some(a.trim())),
        None => (args, None),
    };
    let mut parts: Vec<&str> = if channels.contains(',') {
        channels.split(',').map(str::trim).collect()
    } else {
        channels.split_whitespace().collect()
    };
    let alpha = match (alpha, parts.len()) {
        (Some(a), 3) => a,
        (None, 4) => parts.pop()?,
        (None, 3) => "1",
        _ => return None,
    };
    let alpha = match alpha.strip_suffix('%') {
        Some(p) => p.trim().parse::<f32>().ok()? / 100.0,
        None => alpha.parse::<f32>().ok()?,
    }
    .clamp(0.0, 1.0);
    // A number, or a percentage of `full`.
    let num = |s: &str, full: f32| -> Option<f32> {
        match s.strip_suffix('%') {
            Some(p) => Some(p.trim().parse::<f32>().ok()? / 100.0 * full),
            None => s.trim_end_matches("deg").parse().ok(),
        }
    };
    let [r, g, b] = match function.trim() {
        "rgb" | "rgba" => [
            num(parts[0], 255.0)? / 255.0,
            num(parts[1], 255.0)? / 255.0,
            num(parts[2], 255.0)? / 255.0,
        ],
        "hsl" | "hsla" => hsl_to_rgb(
            num(parts[0], 360.0)?,
            num(parts[1], 1.0)?,
            num(parts[2], 1.0)?,
        ),
        // Lightness 0–1 (or a percentage), chroma about 0–0.4, hue in degrees.
        "oklch" => oklch_to_rgb(
            num(parts[0], 1.0)?,
            num(parts[1], 0.4)?,
            num(parts[2], 360.0)?,
        ),
        _ => return None,
    };
    Some([
        r.clamp(0.0, 1.0),
        g.clamp(0.0, 1.0),
        b.clamp(0.0, 1.0),
        alpha,
    ])
}

/// HSL (hue in degrees, saturation and lightness 0–1) to sRGB.
fn hsl_to_rgb(h: f32, s: f32, l: f32) -> [f32; 3] {
    let (s, l) = (s.clamp(0.0, 1.0), l.clamp(0.0, 1.0));
    let a = s * l.min(1.0 - l);
    let f = |n: f32| {
        let k = (n + h.rem_euclid(360.0) / 30.0) % 12.0;
        l - a * (k - 3.0).min(9.0 - k).clamp(-1.0, 1.0)
    };
    [f(0.0), f(8.0), f(4.0)]
}

/// OKLCH to sRGB (CSS Color 4), clipped to the gamut.
fn oklch_to_rgb(l: f32, c: f32, h: f32) -> [f32; 3] {
    let (l, c, h) = (l as f64, c.max(0.0) as f64, (h as f64).to_radians());
    let (a, b) = (c * h.cos(), c * h.sin());
    let l_ = (l + 0.3963377774 * a + 0.2158037573 * b).powi(3);
    let m_ = (l - 0.1055613458 * a - 0.0638541728 * b).powi(3);
    let s_ = (l - 0.0894841775 * a - 1.2914855480 * b).powi(3);
    let linear = [
        4.0767416621 * l_ - 3.3077115913 * m_ + 0.2309699292 * s_,
        -1.2684380046 * l_ + 2.6097574011 * m_ - 0.3413193965 * s_,
        -0.0041960863 * l_ - 0.7034186147 * m_ + 1.7076147010 * s_,
    ];
    linear.map(|v| {
        let v = v.clamp(0.0, 1.0);
        (if v <= 0.0031308 {
            12.92 * v
        } else {
            1.055 * v.powf(1.0 / 2.4) - 0.055
        }) as f32
    })
}

/// `margin`/`padding`/`border-width`: one to four lengths, CSS order.
/// `box-shadow`: shadows separated by commas, each two to four lengths
/// (x, y, blur, spread) and a colour, in either order. `inset` is not
/// supported, and refused.
fn box_shadows(v: &str) -> Option<Vec<BoxShadow>> {
    if v == "none" {
        return Some(vec![]);
    }
    let mut out = vec![];
    let (mut depth, mut start) = (0, 0);
    let mut parts = vec![];
    for (i, c) in v.char_indices() {
        match c {
            '(' => depth += 1,
            ')' => depth -= 1,
            ',' if depth == 0 => {
                parts.push(&v[start..i]);
                start = i + 1;
            }
            _ => {}
        }
    }
    parts.push(&v[start..]);
    for part in parts {
        let mut lengths = vec![];
        let mut col = None;
        for w in words(part.trim()) {
            if w == "inset" {
                return None;
            }
            match px(w) {
                Some(n) => lengths.push(n),
                None => col = Some(color(w)?),
            }
        }
        if !(2..=4).contains(&lengths.len()) {
            return None;
        }
        out.push(BoxShadow {
            x: lengths[0],
            y: lengths[1],
            blur: lengths.get(2).copied().unwrap_or(0.0).max(0.0),
            spread: lengths.get(3).copied().unwrap_or(0.0),
            color: col.unwrap_or([0.0, 0.0, 0.0, 1.0]),
        });
    }
    Some(out)
}

/// A track size: `100px`, `25%`, `1fr`, `auto`.
fn track_size(v: &str) -> Option<TrackSize> {
    if v == "auto" || v == "min-content" || v == "max-content" {
        return Some(TrackSize::Auto);
    }
    if let Some(n) = v.strip_suffix("fr") {
        return n.parse().ok().map(TrackSize::Fr);
    }
    if let Some(n) = v.strip_suffix('%') {
        return n.parse::<f32>().ok().map(|p| TrackSize::Percent(p / 100.0));
    }
    px(v).map(TrackSize::Px)
}

/// One track: a size or `minmax(a, b)`.
fn track(v: &str) -> Option<Track> {
    if let Some(inner) = v.strip_prefix("minmax(").and_then(|r| r.strip_suffix(')')) {
        let (a, b) = inner.split_once(',')?;
        return Some(Track {
            min: track_size(a.trim())?,
            max: track_size(b.trim())?,
        });
    }
    Some(match track_size(v)? {
        TrackSize::Fr(f) => Track::fr(f),
        s => Track { min: s, max: s },
    })
}

/// A track list, `repeat(n, …)` expanded: `200px repeat(2, 1fr) auto`.
fn tracks(v: &str) -> Option<Vec<Track>> {
    if v == "none" {
        return Some(vec![]);
    }
    let mut out = vec![];
    for w in words(v) {
        if let Some(inner) = w.strip_prefix("repeat(").and_then(|r| r.strip_suffix(')')) {
            let (n, list) = inner.split_once(',')?;
            let n: usize = n.trim().parse().ok()?;
            let one = tracks(list.trim())?;
            for _ in 0..n {
                out.extend(one.iter().copied());
            }
        } else {
            out.push(track(w)?);
        }
    }
    Some(out)
}

/// `grid-column` / `grid-row`: `2`, `1 / 3`, `1 / -1`, `span 2`,
/// `2 / span 3`, `auto`.
fn placement(v: &str) -> Option<GridPlacement> {
    let side = |s: &str| -> Option<(Option<i32>, u32)> {
        let s = s.trim();
        if s == "auto" {
            Some((None, 0))
        } else if let Some(n) = s.strip_prefix("span") {
            Some((None, n.trim().parse().ok()?))
        } else {
            Some((Some(s.parse().ok()?), 0))
        }
    };
    let (a, b) = match v.split_once('/') {
        Some((a, b)) => (side(a)?, Some(side(b)?)),
        None => (side(v)?, None),
    };
    let mut p = GridPlacement {
        start: a.0,
        end: None,
        span: a.1,
    };
    if let Some((end, span)) = b {
        p.end = end;
        if span > 0 {
            p.span = span;
        }
    }
    Some(p)
}

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
        "baseline" | "first baseline" => AlignItems::Baseline,
        _ => return None,
    })
}

/// Sets property `name` from CSS text.
pub fn apply(s: &mut Style, name: &str, value: &str) {
    if !try_apply(s, name, value) {
        bad(name, &normalize(value));
    }
}

/// [`apply`], saying whether the property and value are supported rather
/// than panicking. Where they are not, `s` may be partly changed.
pub fn try_apply(s: &mut Style, name: &str, value: &str) -> bool {
    let value = normalize(value);
    let v = value.as_str();
    let ok = (|| -> Option<()> {
        match name {
            "display" => {
                s.display = match v {
                    "block" | "inline-block" => Display::Block,
                    "flex" | "inline-flex" => Display::Flex,
                    "grid" | "inline-grid" => Display::Grid,
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
            "cursor" => s.cursor = Cursor::parse(v)?,
            "grid-template-columns" => s.grid_columns = tracks(v)?,
            "grid-template-rows" => s.grid_rows = tracks(v)?,
            "grid-column" => s.grid_column = placement(v)?,
            "grid-row" => s.grid_row = placement(v)?,
            "row-gap" => s.row_gap = Some(px(v)?),
            "scrollbar-width" => {
                s.scrollbar = match v {
                    "none" => Scrollbar::None,
                    "thin" => Scrollbar::Thin,
                    "auto" => Scrollbar::Auto,
                    _ => return None,
                }
            }
            "scrollbar-visibility" => {
                s.scrollbar_visibility = match v {
                    "always" => ScrollbarVisibility::Always,
                    "auto" => ScrollbarVisibility::Auto,
                    _ => return None,
                }
            }
            "scrollbar-fade" => {
                s.scrollbar_fade = match words(v).as_slice() {
                    [delay] => (seconds(delay)?, s.scrollbar_fade.1),
                    [delay, duration] => (seconds(delay)?, seconds(duration)?),
                    _ => return None,
                }
            }
            "scrollbar-color" => {
                s.scrollbar_color = match words(v).as_slice() {
                    ["auto"] => None,
                    [thumb, track] => Some((color(thumb)?, color(track)?)),
                    _ => return None,
                }
            }
            "line-clamp" | "-webkit-line-clamp" => {
                s.line_clamp = match v {
                    "none" => None,
                    n => Some(n.parse::<u32>().ok().filter(|&n| n > 0)?),
                }
            }
            "user-select" | "-webkit-user-select" => {
                s.user_select = match v {
                    "text" | "all" => UserSelect::Text,
                    "none" => UserSelect::None,
                    "auto" => UserSelect::Auto,
                    _ => return None,
                }
            }
            "pointer-events" => {
                s.pointer_events = match v {
                    "none" => false,
                    "auto" => true,
                    _ => return None,
                }
            }
            "object-fit" => {
                s.object_fit = match v {
                    "fill" => ObjectFit::Fill,
                    "contain" => ObjectFit::Contain,
                    "cover" => ObjectFit::Cover,
                    "none" => ObjectFit::None,
                    "scale-down" => ObjectFit::ScaleDown,
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
                (s.flex_direction, s.flex_reverse) = match v {
                    "row" => (FlexDirection::Row, false),
                    "column" => (FlexDirection::Column, false),
                    "row-reverse" => (FlexDirection::Row, true),
                    "column-reverse" => (FlexDirection::Column, true),
                    _ => return None,
                }
            }
            "flex-wrap" => {
                s.flex_wrap = match v {
                    "nowrap" => FlexWrap::NoWrap,
                    "wrap" => FlexWrap::Wrap,
                    "wrap-reverse" => FlexWrap::WrapReverse,
                    _ => return None,
                }
            }
            // `flex-flow: <direction> <wrap>`, either or both, any order.
            "flex-flow" => {
                for w in words(v) {
                    match w {
                        "nowrap" => s.flex_wrap = FlexWrap::NoWrap,
                        "wrap" => s.flex_wrap = FlexWrap::Wrap,
                        "wrap-reverse" => s.flex_wrap = FlexWrap::WrapReverse,
                        "row" => (s.flex_direction, s.flex_reverse) = (FlexDirection::Row, false),
                        "column" => {
                            (s.flex_direction, s.flex_reverse) = (FlexDirection::Column, false)
                        }
                        "row-reverse" => {
                            (s.flex_direction, s.flex_reverse) = (FlexDirection::Row, true)
                        }
                        "column-reverse" => {
                            (s.flex_direction, s.flex_reverse) = (FlexDirection::Column, true)
                        }
                        _ => return None,
                    }
                }
            }
            "order" => s.order = v.parse().ok()?,
            "align-content" => {
                s.align_content = match v {
                    "stretch" | "normal" => AlignContent::Stretch,
                    "flex-start" | "start" => AlignContent::FlexStart,
                    "center" => AlignContent::Center,
                    "flex-end" | "end" => AlignContent::FlexEnd,
                    "space-between" => AlignContent::SpaceBetween,
                    "space-around" => AlignContent::SpaceAround,
                    "space-evenly" => AlignContent::SpaceEvenly,
                    _ => return None,
                }
            }
            "column-gap" => s.column_gap = Some(px(v)?),
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
                    "space-around" => JustifyContent::SpaceAround,
                    "space-evenly" => JustifyContent::SpaceEvenly,
                    _ => return None,
                }
            }
            // `gap: <row> [<column>]`.
            "gap" => {
                let w = words(v);
                match w.as_slice() {
                    [g] => s.gap = px(g)?,
                    [r, c] => {
                        s.row_gap = Some(px(r)?);
                        s.column_gap = Some(px(c)?);
                        s.gap = px(c)?;
                    }
                    _ => return None,
                }
            }
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
            "border-style" => s.border_dashed = dashed(v)?,
            // `border: <width> [solid|dashed] [<color>]`.
            "border" => {
                for w in words(v) {
                    if let Some(p) = px(w) {
                        s.border = Edges::all(p);
                    } else if let Some(c) = color(w) {
                        s.border_color = c;
                    } else {
                        s.border_dashed = dashed(w)?;
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
            "background" | "background-color" | "background-image" if v.contains("gradient(") => {
                let (shape, stops) = gradient(v)?;
                s.gradient_shape = shape;
                s.background_gradient = stops;
            }
            "background" | "background-color" => {
                s.background = if v == "none" { None } else { Some(color(v)?) }
            }
            "color" => {
                s.color = color(v)?;
                s.color_set = true;
            }
            "font-family" => {
                s.font_family = v.trim_matches('"').to_owned();
                s.font_family_set = true;
            }
            "font-size" => {
                s.font_size = px(v)?;
                s.font_size_set = true;
            }
            "font-weight" => {
                s.font_weight = match v {
                    "normal" => FontWeight::Normal,
                    "bold" => FontWeight::Bold,
                    n => FontWeight::Weight(n.parse().ok()?),
                };
                s.font_weight_set = true;
            }
            // A bare number is times the font size (as set by then).
            "line-height" => {
                s.line_height = match v.parse::<f32>() {
                    _ if v == "normal" => None,
                    Ok(n) => Some(n * s.font_size),
                    Err(_) => Some(px(v)?),
                };
                s.line_height_set = true;
            }
            "top" => s.top = length(v)?,
            "left" => s.left = length(v)?,
            "right" => s.right = length(v)?,
            "bottom" => s.bottom = length(v)?,
            "overflow" | "overflow-y" | "overflow-x" => {
                let (clip, scrolls) = match v {
                    "hidden" | "clip" => (true, false),
                    "auto" | "scroll" => (true, true),
                    "visible" => (false, false),
                    _ => return None,
                };
                s.clip = clip || (name != "overflow" && s.clip);
                if name != "overflow-x" {
                    s.scroll = scrolls;
                }
                if name != "overflow-y" {
                    s.scroll_x = scrolls;
                }
            }
            "font-style" => s.italic = v == "italic" || v == "oblique",
            "text-overflow" => s.ellipsis = v == "ellipsis",
            "white-space" => {
                s.wrap = v != "nowrap" && v != "pre";
                s.keep_newlines = matches!(v, "pre" | "pre-wrap" | "pre-line" | "break-spaces");
            }
            "text-align" => {
                s.text_align = match v {
                    "center" => crate::style::TextAlign::Center,
                    "right" | "end" => crate::style::TextAlign::End,
                    "left" | "start" | "justify" => crate::style::TextAlign::Start,
                    _ => return None,
                };
                s.text_align_set = true;
            }
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
            "text-decoration" | "text-decoration-line" => {
                let w = words(v);
                s.line_through = w.contains(&"line-through");
                s.underline = w.contains(&"underline");
                if w.contains(&"wavy") {
                    s.decoration_wavy = true;
                }
                if let Some(c) = w.iter().find_map(|w| color(w)) {
                    s.decoration_color = Some(c);
                }
            }
            "text-decoration-style" => s.decoration_wavy = v == "wavy",
            "text-decoration-color" => s.decoration_color = Some(color(v)?),
            "box-shadow" => s.box_shadow = box_shadows(v)?,
            "filter" => s.blur = px(v.strip_prefix("blur(")?.strip_suffix(')')?)?,
            "mix-blend-mode" => {
                use crate::style::BlendMode;
                s.mix_blend_mode = match v {
                    "normal" => BlendMode::Normal,
                    "multiply" => BlendMode::Multiply,
                    "screen" => BlendMode::Screen,
                    "plus-lighter" | "plus" => BlendMode::Plus,
                    "darken" => BlendMode::Darken,
                    "lighten" => BlendMode::Lighten,
                    _ => return None,
                };
            }
            "mask-image" | "mask" => {
                s.mask = if v == "none" {
                    None
                } else {
                    Some(gradient(v)?)
                };
            }
            "clip-path" => {
                use crate::style::ClipShape;
                s.clip_path = match v {
                    "none" => None,
                    "capsule" => Some(ClipShape::Capsule),
                    _ if v.starts_with("circle(") => Some(ClipShape::Circle),
                    _ if v.starts_with("ellipse(") => Some(ClipShape::Ellipse),
                    _ => {
                        // `inset(<edges> [round <radius>])`.
                        let inner = v.strip_prefix("inset(")?.strip_suffix(')')?;
                        let (e, r) = match inner.split_once(" round ") {
                            Some((e, r)) => (e.trim(), px(r.trim())?),
                            None => (inner.trim(), 0.0),
                        };
                        Some(ClipShape::Inset(edges(e)?, r))
                    }
                };
            }
            "aspect-ratio" => {
                // `16 / 9`, `1.5`, or `auto`.
                s.aspect_ratio = match v.split_once('/') {
                    _ if v == "auto" => 0.0,
                    Some((a, b)) => {
                        a.trim().parse::<f32>().ok()? / b.trim().parse::<f32>().ok()?.max(1e-6)
                    }
                    None => v.parse::<f32>().ok()?,
                }
                .max(0.0);
            }
            "backdrop-filter" => {
                // `blur(20px) saturate(180%)`, either or both, or `none`.
                s.backdrop_blur = 0.0;
                s.backdrop_saturate = 1.0;
                for f in v
                    .split(')')
                    .map(str::trim)
                    .filter(|f| !f.is_empty() && *f != "none")
                {
                    let (name, arg) = f.split_once('(')?;
                    match name.trim() {
                        "blur" => s.backdrop_blur = px(arg.trim())?.max(0.0),
                        "saturate" => {
                            let a = arg.trim();
                            s.backdrop_saturate = match a.strip_suffix('%') {
                                Some(p) => p.trim().parse::<f32>().ok()? / 100.0,
                                None => a.parse().ok()?,
                            };
                        }
                        _ => return None,
                    }
                }
            }
            n if n.starts_with("border-beam") => crate::border_beam::css(s, n, v)?,
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
    ok.is_some()
}

/// Sets property `name` from a Rust value.
pub fn set(s: &mut Style, name: &str, value: Value) {
    use Value as V;
    let ok = match (name, value) {
        ("display", V::Display(d)) => {
            s.display = d;
            true
        }
        ("cursor", V::Cursor(c)) => {
            s.cursor = c;
            true
        }
        ("object-fit", V::ObjectFit(f)) => {
            s.object_fit = f;
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
            s.font_weight_set = true;
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
            s.font_family_set = true;
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
                s.font_size_set = true;
                true
            }
            "line-height" => {
                s.line_height = Some(n);
                s.line_height_set = true;
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
            let Some((base, side)) = name.rsplit_once('-') else {
                return false;
            };
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
/// use threers_ui::style;
/// use threers_ui::{Edges, Length, Display};
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
        let mut style = $crate::Style::default();
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
        $crate::css::set(&mut $s, &[$(stringify!($name)),+].join("-"), $crate::css::Value::from($value));
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
        $crate::css::apply(&mut $s, &[$($n),+].join("-"), stringify!($($v)*));
        $crate::__style_declarations!($s; $($rest)*);
    };
    ($s:ident; [$($n:expr),+] [$($v:tt)*]) => {
        $crate::css::apply(&mut $s, &[$($n),+].join("-"), stringify!($($v)*));
    };
    ($s:ident; $n:tt [$($v:tt)*] $t:tt $($rest:tt)*) => {
        $crate::__style_value!($s; $n [$($v)* $t] $($rest)*);
    };
}

/// Adds boxes and text to a [`Ui`](super::Ui) as nested markup, and evaluates
/// to the first node added.
///
/// ```
/// use threers_ui::{style, ui};
/// use threers_ui::{FixedWidthText, Ui};
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
/// A node is `div(<declarations>) { <children> }`, `div["<tailwind classes>"]
/// { <children> }` (see [`Ui::tw`](super::Ui::tw)), a string literal (a text
/// run in its parent's font) or `{expr}` (text from a Rust expression).
///
/// ```
/// use threers_ui::{ui, FixedWidthText, Ui};
///
/// let mut tree = Ui::new();
/// let root = tree.root();
/// let card = ui!(tree, root,
///     div["flex flex-col gap-2 p-4 rounded-lg bg-white hover:bg-slate-50"] {
///         div["text-lg font-semibold text-slate-900"] { "Title" }
///         div(height: 1px; background: #e5e7eb) {}
///     }
/// );
/// tree.layout(400.0, 300.0, &mut FixedWidthText(7.0));
/// assert_eq!(tree.style(card).padding.top, 16.0);
/// ```
#[macro_export]
macro_rules! ui {
    ($tree:expr, $parent:expr, $($nodes:tt)+) => {{
        let tree: &mut $crate::Ui = &mut $tree;
        let parent: $crate::NodeId = $parent;
        let mut first: Option<$crate::NodeId> = None;
        $crate::__ui_nodes!(tree, parent, first; $($nodes)+);
        first.expect("ui!: no nodes")
    }};
}

#[doc(hidden)]
#[macro_export]
macro_rules! __ui_nodes {
    ($tree:ident, $parent:ident, $first:ident;) => {};
    ($tree:ident, $parent:ident, $first:ident; div [ $classes:expr ] { $($children:tt)* } $($rest:tt)*) => {
        let node = $tree.add_tw($parent, $classes);
        $first.get_or_insert(node);
        {
            #[allow(unused_variables, unused_mut)]
            let mut first: Option<$crate::NodeId> = None;
            $crate::__ui_nodes!($tree, node, first; $($children)*);
        }
        $crate::__ui_nodes!($tree, $parent, $first; $($rest)*);
    };
    ($tree:ident, $parent:ident, $first:ident; div ( $($style:tt)* ) { $($children:tt)* } $($rest:tt)*) => {
        let node = $tree.add_box($parent, $crate::style!($($style)*));
        $first.get_or_insert(node);
        {
            #[allow(unused_variables, unused_mut)]
            let mut first: Option<$crate::NodeId> = None;
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
    fn a_bare_line_height_is_times_the_font_size() {
        let mut s = Style::default();
        apply(&mut s, "font-size", "20px");
        apply(&mut s, "line-height", "1.5");
        assert_eq!(s.line_height, Some(30.0));
        apply(&mut s, "line-height", "18px");
        assert_eq!(s.line_height, Some(18.0));
        apply(&mut s, "line-height", "normal");
        assert_eq!(s.line_height, None);
    }

    #[test]
    fn parses_css_values() {
        assert_eq!(length("12px"), Some(Length::Px(12.0)));
        assert_eq!(length("-1.5px"), Some(Length::Px(-1.5)));
        assert_eq!(length("20%"), Some(Length::Percent(20.0)));
        assert_eq!(
            color("#9a7518"),
            Some([154.0 / 255.0, 117.0 / 255.0, 24.0 / 255.0, 1.0])
        );
        assert_eq!(color("#f00"), Some([1.0, 0.0, 0.0, 1.0]));
        assert_eq!(color("rgba(0, 0, 0, 0.3)"), Some([0.0, 0.0, 0.0, 0.3]));
        assert_eq!(
            edges("1px 1px 1px 5px"),
            Some(Edges::new(1.0, 1.0, 1.0, 5.0))
        );
        // One edge's border.
        let mut e = Style::default();
        apply(&mut e, "border-left", "2px solid #181818");
        apply(&mut e, "border-bottom", "3px");
        assert_eq!(e.border, Edges::new(0.0, 0.0, 3.0, 2.0));
        assert_eq!(e.border_color, color("#181818").unwrap());
        // dashed, in the shorthand or on its own
        apply(&mut e, "border", "1px dashed #000000");
        assert!(e.border_dashed && e.border == Edges::all(1.0));
        apply(&mut e, "border-style", "solid");
        assert!(!e.border_dashed);
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
            ..crate::Style::flex(FlexDirection::Column);
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
        assert_eq!(
            (t.flex_grow, t.flex_shrink, t.flex_basis),
            (1.0, 1.0, Length::Percent(0.0))
        );
    }
}
