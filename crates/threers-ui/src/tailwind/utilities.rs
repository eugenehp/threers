//! One utility class into declarations: the class split into a root
//! (`border-t`, `min-w`, `rotate-y`) and a value, the longest known root
//! first, so `border-teal-500` is a colour and `border-t-2` an edge.

use crate::style::Rgba;

use super::theme::{Paint, Tailwind};
use super::{arbitrary, split_top};

/// A declaration's value: one text, or one for each appearance.
#[derive(Debug, Clone, PartialEq)]
pub(super) enum Val {
    Text(String),
    Adaptive { light: String, dark: String },
}

impl Val {
    pub fn get(&self, dark: bool) -> &str {
        match self {
            Val::Text(t) => t,
            Val::Adaptive { light, dark: d } => {
                if dark {
                    d
                } else {
                    light
                }
            }
        }
    }

    /// From a colour (or colours) put into text by `f`.
    fn paint(paints: &[Paint], f: impl Fn(&[Rgba]) -> String) -> Val {
        let light: Vec<Rgba> = paints.iter().map(|p| p.in_dark(false)).collect();
        if !paints.iter().any(Paint::is_adaptive) {
            return Val::Text(f(&light));
        }
        let dark: Vec<Rgba> = paints.iter().map(|p| p.in_dark(true)).collect();
        Val::Adaptive {
            light: f(&light),
            dark: f(&dark),
        }
    }
}

impl From<String> for Val {
    fn from(s: String) -> Self {
        Val::Text(s)
    }
}

impl From<&str> for Val {
    fn from(s: &str) -> Self {
        Val::Text(s.to_owned())
    }
}

#[derive(Debug, Clone)]
pub(super) struct Decl {
    pub name: String,
    pub value: Val,
    /// The class it came from, for errors and for ordering.
    pub class: String,
}

/// What one condition's classes say, before what took several classes is
/// joined.
#[derive(Debug, Clone, Default)]
pub(super) struct Group {
    pub decls: Vec<Decl>,
    pub gradient: Gradient,
    pub backdrop: Backdrop,
    pub effects: Effects,
    pub columns: Placement,
    pub rows: Placement,
    pub border_width: bool,
    pub border_color: bool,
}

impl Group {
    /// The group with what took several classes put over `base`'s, part
    /// by part (`hover:from-blue-600` keeps the base's direction and `to-`),
    /// and turned into declarations.
    pub fn joined(
        mut self,
        base: Option<&Group>,
        theme: &Tailwind,
    ) -> Result<Vec<Decl>, (String, String)> {
        if let Some(base) = base {
            self.gradient = base.gradient.under(&self.gradient);
            self.backdrop = base.backdrop.under(&self.backdrop);
            self.effects = base.effects.under(&self.effects);
            self.columns = base.columns.under(&self.columns);
            self.rows = base.rows.under(&self.rows);
        }
        let mut decls = self.decls;
        decls.extend(self.gradient.declaration()?);
        decls.extend(self.backdrop.declaration());
        decls.extend(self.effects.declaration(theme));
        for (p, prop) in [(&self.columns, "grid-column"), (&self.rows, "grid-row")] {
            if let Some(value) = p.css() {
                decls.push(Decl {
                    name: prop.into(),
                    value: value.into(),
                    class: p.class.clone(),
                });
            }
        }
        // Preflight: a border is drawn in the default colour.
        if self.border_width && !self.border_color && !base.is_some_and(|b| b.border_color) {
            let class = decls
                .iter()
                .find(|d| d.name.starts_with("border"))
                .map(|d| d.class.clone())
                .unwrap_or_default();
            decls.push(Decl {
                name: "border-color".into(),
                value: Val::paint(&[theme.border_color], |c| css_color(c[0])),
                class,
            });
        }
        Ok(decls)
    }
}

/// `bg-gradient-to-r from-… via-… to-…`: one `background`.
#[derive(Debug, Clone, Default)]
pub(super) struct Gradient {
    direction: Option<(String, String)>,
    /// From, via, to: colour and position (percent).
    stops: [(Option<Paint>, Option<f32>); 3],
    /// The first stop class, for errors.
    class: Option<String>,
}

impl Gradient {
    fn touched(&self) -> bool {
        self.class.is_some() || self.direction.is_some()
    }

    fn under(&self, top: &Gradient) -> Gradient {
        if !top.touched() {
            return Gradient::default();
        }
        let mut g = self.clone();
        if top.direction.is_some() {
            g.direction = top.direction.clone();
        }
        for (mine, theirs) in g.stops.iter_mut().zip(&top.stops) {
            mine.0 = theirs.0.or(mine.0);
            mine.1 = theirs.1.or(mine.1);
        }
        g.class = top.class.clone().or(g.class);
        g
    }

    fn declaration(&self) -> Result<Option<Decl>, (String, String)> {
        if !self.stops.iter().any(|s| s.0.is_some()) {
            return Ok(None);
        }
        let Some((direction, dir_class)) = &self.direction else {
            let class = self.class.clone().unwrap_or_default();
            return Err((
                class,
                "gradient stops need a direction: `bg-gradient-to-r` or the like".into(),
            ));
        };
        let from = self.stops[0]
            .0
            .or(self.stops[2].0.map(Paint::transparent))
            .unwrap_or(Paint::fixed([0.0; 4]));
        let to = self.stops[2].0.unwrap_or(from.transparent());
        let via = self.stops[1].0;
        let mut paints = vec![from];
        paints.extend(via);
        paints.push(to);
        let value = Val::paint(&paints, |c| {
            let stop = |c: Rgba, p: Option<f32>| match p {
                Some(p) => format!("{} {p}%", css_color(c)),
                None => css_color(c),
            };
            let mut parts = vec![direction.clone(), stop(c[0], self.stops[0].1)];
            if c.len() == 3 {
                parts.push(stop(c[1], self.stops[1].1));
            }
            parts.push(stop(c[c.len() - 1], self.stops[2].1));
            format!("linear-gradient({})", parts.join(", "))
        });
        Ok(Some(Decl {
            name: "background".into(),
            value,
            class: dir_class.clone(),
        }))
    }
}

/// `backdrop-blur-*` and `backdrop-saturate-*`: one property.
#[derive(Debug, Clone, Default)]
pub(super) struct Backdrop {
    blur: Option<String>,
    saturate: Option<String>,
    class: Option<String>,
}

impl Backdrop {
    fn under(&self, top: &Backdrop) -> Backdrop {
        if top.class.is_none() {
            return Backdrop::default();
        }
        Backdrop {
            blur: top.blur.clone().or(self.blur.clone()),
            saturate: top.saturate.clone().or(self.saturate.clone()),
            class: top.class.clone(),
        }
    }

    fn declaration(&self) -> Option<Decl> {
        let class = self.class.clone()?;
        let mut parts = vec![];
        if let Some(b) = &self.blur {
            parts.push(format!("blur({b})"));
        }
        if let Some(s) = &self.saturate {
            parts.push(format!("saturate({s})"));
        }
        Some(Decl {
            name: "backdrop-filter".into(),
            value: parts.join(" ").into(),
            class,
        })
    }
}

/// `shadow-*`, `ring-*` and `ring-offset-*`: one `box-shadow`, the offset
/// over the ring over the shadow, as Tailwind stacks them.
#[derive(Debug, Clone, Default)]
pub(super) struct Effects {
    shadow: Option<String>,
    ring: Option<f32>,
    ring_color: Option<Paint>,
    offset: Option<f32>,
    offset_color: Option<Paint>,
    class: Option<String>,
}

impl Effects {
    fn under(&self, top: &Effects) -> Effects {
        if top.class.is_none() {
            return Effects::default();
        }
        Effects {
            shadow: top.shadow.clone().or(self.shadow.clone()),
            ring: top.ring.or(self.ring),
            ring_color: top.ring_color.or(self.ring_color),
            offset: top.offset.or(self.offset),
            offset_color: top.offset_color.or(self.offset_color),
            class: top.class.clone(),
        }
    }

    fn declaration(&self, theme: &Tailwind) -> Option<Decl> {
        let class = self.class.clone()?;
        let ring = self.ring.filter(|w| *w > 0.0);
        if self.shadow.is_none() && self.ring.is_none() {
            // Only a colour or an offset so far: a state may add the ring.
            return None;
        }
        let offset = self.offset.unwrap_or(0.0);
        let ring_color = self.ring_color.unwrap_or(theme.ring_color);
        let offset_color = self.offset_color.unwrap_or(Paint::fixed([1.0; 4]));
        let value = Val::paint(&[offset_color, ring_color], |c| {
            let mut parts = vec![];
            if let Some(w) = ring {
                if offset > 0.0 {
                    parts.push(format!("0 0 0 {offset}px {}", css_color(c[0])));
                }
                parts.push(format!("0 0 0 {}px {}", offset + w, css_color(c[1])));
            }
            if let Some(s) = self.shadow.as_ref().filter(|s| *s != "none") {
                parts.push(s.clone());
            }
            if parts.is_empty() {
                "none".into()
            } else {
                parts.join(", ")
            }
        });
        Some(Decl {
            name: "box-shadow".into(),
            value,
            class,
        })
    }
}

/// `col-span-2`, `col-start-1`, `col-end-3`: one `grid-column`.
#[derive(Debug, Clone, Default)]
pub(super) struct Placement {
    start: Option<i32>,
    end: Option<i32>,
    span: Option<u32>,
    class: String,
}

impl Placement {
    fn under(&self, top: &Placement) -> Placement {
        if top.class.is_empty() {
            return Placement::default();
        }
        Placement {
            start: top.start.or(self.start),
            end: top.end.or(self.end),
            span: top.span.or(self.span),
            class: top.class.clone(),
        }
    }

    fn css(&self) -> Option<String> {
        let start = match (self.start, self.span) {
            (Some(s), _) => s.to_string(),
            (None, Some(n)) if self.end.is_some() => format!("span {n}"),
            (None, Some(n)) => return Some(format!("span {n}")),
            (None, None) if self.end.is_some() => "auto".into(),
            (None, None) => return None,
        };
        Some(match (self.end, self.span) {
            (Some(e), _) => format!("{start} / {e}"),
            (None, Some(n)) if self.start.is_some() => format!("{start} / span {n}"),
            _ => start,
        })
    }
}

/// A colour as CSS, to a thousandth of a channel step: exact for colours
/// given in hex, and close for `oklch(…)`.
pub(super) fn css_color(c: Rgba) -> String {
    let ch = |v: f32| (v.clamp(0.0, 1.0) * 255_000.0).round() / 1000.0;
    format!(
        "rgba({},{},{},{})",
        ch(c[0]),
        ch(c[1]),
        ch(c[2]),
        c[3].clamp(0.0, 1.0)
    )
}

fn px(n: f32) -> String {
    format!("{n}px")
}

/// Classes that are a fixed list of declarations.
fn keyword(u: &str) -> Option<&'static [(&'static str, &'static str)]> {
    Some(match u {
        "block" | "inline-block" => &[("display", "block")],
        "flex" | "inline-flex" => &[("display", "flex")],
        "grid" | "inline-grid" => &[("display", "grid")],
        "hidden" => &[("display", "none")],
        "static" => &[("position", "static")],
        "relative" => &[("position", "relative")],
        "absolute" => &[("position", "absolute")],
        "box-border" => &[("box-sizing", "border-box")],
        "box-content" => &[("box-sizing", "content-box")],
        "flex-row" => &[("flex-direction", "row")],
        "flex-row-reverse" => &[("flex-direction", "row-reverse")],
        "flex-col" => &[("flex-direction", "column")],
        "flex-col-reverse" => &[("flex-direction", "column-reverse")],
        "flex-wrap" => &[("flex-wrap", "wrap")],
        "flex-wrap-reverse" => &[("flex-wrap", "wrap-reverse")],
        "flex-nowrap" => &[("flex-wrap", "nowrap")],
        "flex-1" => &[("flex", "1 1 0%")],
        "flex-auto" => &[("flex", "1 1 auto")],
        "flex-initial" => &[("flex", "0 1 auto")],
        "flex-none" => &[("flex", "0 0 auto")],
        "grow" | "flex-grow" => &[("flex-grow", "1")],
        "shrink" | "flex-shrink" => &[("flex-shrink", "1")],
        "order-first" => &[("order", "-9999")],
        "order-last" => &[("order", "9999")],
        "order-none" => &[("order", "0")],
        "italic" => &[("font-style", "italic")],
        "not-italic" => &[("font-style", "normal")],
        "underline" => &[("text-decoration-line", "underline")],
        "line-through" => &[("text-decoration-line", "line-through")],
        "no-underline" => &[("text-decoration-line", "none")],
        "truncate" => &[
            ("overflow", "hidden"),
            ("text-overflow", "ellipsis"),
            ("white-space", "nowrap"),
        ],
        "text-ellipsis" => &[("text-overflow", "ellipsis")],
        "text-clip" => &[("text-overflow", "clip")],
        "text-wrap" => &[("white-space", "normal")],
        "text-nowrap" => &[("white-space", "nowrap")],
        "break-words" => &[("overflow-wrap", "break-word")],
        "col-auto" => &[("grid-column", "auto")],
        "row-auto" => &[("grid-row", "auto")],
        "border-solid" => &[("border-style", "solid")],
        "border-dashed" => &[("border-style", "dashed")],
        "bg-none" => &[("background", "none")],
        "bg-clip-border" => &[("background-clip", "border-box")],
        "bg-clip-padding" => &[("background-clip", "padding-box")],
        "bg-clip-content" => &[("background-clip", "content-box")],
        "pointer-events-none" => &[("pointer-events", "none")],
        "pointer-events-auto" => &[("pointer-events", "auto")],
        "aspect-auto" => &[("aspect-ratio", "auto")],
        "aspect-square" => &[("aspect-ratio", "1")],
        "aspect-video" => &[("aspect-ratio", "16 / 9")],
        "line-clamp-none" => &[("line-clamp", "none")],
        "decoration-solid" => &[("text-decoration-style", "solid")],
        "decoration-wavy" => &[("text-decoration-style", "wavy")],
        "grid-cols-none" => &[("grid-template-columns", "none")],
        "grid-rows-none" => &[("grid-template-rows", "none")],
        // Nothing here draws an outline, and children compose in 3D anyway.
        "outline-none" | "transform" | "transform-gpu" | "transform-cpu" | "transform-3d"
        | "transform-flat" => &[],
        "transform-none" => &[
            ("rotate-x", "0deg"),
            ("rotate-y", "0deg"),
            ("rotate", "0deg"),
            ("translate-x", "0px"),
            ("translate-y", "0px"),
            ("translate-z", "0px"),
            ("scale", "1"),
        ],
        "backface-hidden" => &[("backface-visibility", "hidden")],
        "backface-visible" => &[("backface-visibility", "visible")],
        // threers' own: the glare, shading and lift shadow of `depth`.
        "glare" => &[("glare", "on")],
        "shading" => &[("shading", "on")],
        "depth-shadow" => &[("depth-shadow", "on")],
        _ => return None,
    })
}

/// Roots of classes with values.
const ROOTS: &[&str] = &[
    "p",
    "px",
    "py",
    "pt",
    "pr",
    "pb",
    "pl",
    "ps",
    "pe",
    "m",
    "mx",
    "my",
    "mt",
    "mr",
    "mb",
    "ml",
    "ms",
    "me",
    "inset",
    "inset-x",
    "inset-y",
    "top",
    "right",
    "bottom",
    "left",
    "start",
    "end",
    "w",
    "h",
    "size",
    "min-w",
    "min-h",
    "max-w",
    "max-h",
    "basis",
    "flex",
    "grow",
    "shrink",
    "flex-grow",
    "flex-shrink",
    "order",
    "gap",
    "gap-x",
    "gap-y",
    "items",
    "self",
    "justify",
    "content",
    "grid-cols",
    "grid-rows",
    "col-span",
    "col-start",
    "col-end",
    "row-span",
    "row-start",
    "row-end",
    "text",
    "font",
    "leading",
    "line-clamp",
    "decoration",
    "whitespace",
    "select",
    "bg",
    "bg-gradient-to",
    "bg-linear-to",
    "from",
    "via",
    "to",
    "border",
    "border-x",
    "border-y",
    "border-t",
    "border-r",
    "border-b",
    "border-l",
    "border-s",
    "border-e",
    "rounded",
    "shadow",
    "ring",
    "ring-offset",
    "opacity",
    "blur",
    "backdrop-blur",
    "backdrop-saturate",
    "mix-blend",
    "rotate",
    "rotate-x",
    "rotate-y",
    "rotate-z",
    "translate",
    "translate-x",
    "translate-y",
    "translate-z",
    "scale",
    "scale-x",
    "scale-y",
    "scale-z",
    "perspective",
    "origin",
    "skew-x",
    "skew-y",
    "overflow",
    "overflow-x",
    "overflow-y",
    "cursor",
    "object",
    "scrollbar",
    "aspect",
];

/// Roots a `-` may go on.
const NEGATABLE: &[&str] = &[
    "m",
    "mx",
    "my",
    "mt",
    "mr",
    "mb",
    "ml",
    "ms",
    "me",
    "inset",
    "inset-x",
    "inset-y",
    "top",
    "right",
    "bottom",
    "left",
    "start",
    "end",
    "order",
    "rotate",
    "rotate-x",
    "rotate-y",
    "rotate-z",
    "translate",
    "translate-x",
    "translate-y",
    "translate-z",
];

/// Tailwind classes this UI has no property for, and what to do instead.
fn unsupported(u: &str, root: Option<&str>) -> Option<&'static str> {
    let starts = |prefixes: &[&str]| {
        prefixes
            .iter()
            .any(|p| u == *p || u.starts_with(&format!("{p}-")))
    };
    Some(match root {
        Some("scale-x" | "scale-y" | "scale-z") => "scales are even: `scale-*`",
        Some("skew-x" | "skew-y") => "no skew",
        Some(_) => return None,
        None if starts(&["fixed", "sticky"]) => {
            "no fixed or sticky positioning: `absolute` in a box that does not scroll"
        }
        None if starts(&["z"]) => "no z-index: later siblings paint over earlier ones",
        None if starts(&["space-x", "space-y", "divide-x", "divide-y", "divide"]) => {
            "no child selectors: `gap-*` on a flex or grid box"
        }
        None if starts(&["tracking"]) => "no letter spacing",
        None if starts(&["uppercase", "lowercase", "capitalize", "normal-case"]) => {
            "no text transform: change the text"
        }
        None if starts(&["ease"]) => "transitions take one easing curve",
        None if starts(&["delay"]) => "transitions start at once",
        None if starts(&["animate"]) => "no animation classes: `keyframes` or `Ui::animate`",
        None if starts(&["invisible", "visible", "collapse"]) => {
            "no visibility: `hidden` or `opacity-0`"
        }
        None if starts(&["outline"]) => "no outlines: `ring-*`",
        None if starts(&["underline-offset"]) => "no underline offset or thickness",
        None if starts(&["inline", "table", "contents", "flow-root", "list-item"]) => {
            "boxes are block, flex or grid"
        }
        None if starts(&["float", "clear"]) => "no floats: flex",
        None if starts(&["sr-only", "not-sr-only"]) => {
            "painted boxes reach screen readers through semantics: `Ui::set_semantics`"
        }
        None if starts(&[
            "place-content",
            "place-items",
            "place-self",
            "justify-items",
            "justify-self",
        ]) =>
        {
            "no `place-*` or `justify-items`: `items-*`, `justify-*` and `self-*`"
        }
        None => return None,
    })
}

/// The longest root `u` starts with, and the value after it.
fn split_root(u: &str) -> Option<(&'static str, &str)> {
    ROOTS
        .iter()
        .filter_map(|r| {
            let rest = u.strip_prefix(r)?;
            match rest.strip_prefix('-') {
                Some(v) => Some((*r, v)),
                None if rest.is_empty() => Some((*r, "")),
                None => None,
            }
        })
        .max_by_key(|(r, _)| r.len())
}

/// The nearest of `names` to `word`, if near enough to be a typo.
pub(super) fn nearest<'a>(word: &str, names: impl Iterator<Item = &'a str>) -> Option<&'a str> {
    fn distance(a: &str, b: &str) -> usize {
        let b: Vec<char> = b.chars().collect();
        let mut row: Vec<usize> = (0..=b.len()).collect();
        for (i, ca) in a.chars().enumerate() {
            let mut prev = row[0];
            row[0] = i + 1;
            for j in 0..b.len() {
                let cur = row[j + 1];
                row[j + 1] = (prev + usize::from(ca != b[j]))
                    .min(row[j] + 1)
                    .min(cur + 1);
                prev = cur;
            }
        }
        row[b.len()]
    }
    let limit = (word.len() / 4).clamp(1, 2);
    names
        .map(|n| (distance(word, n), n))
        .filter(|(d, _)| *d <= limit)
        .min()
        .map(|(_, n)| n)
}

/// Reads one utility (variants and `-` taken off) into `g`; why not, if
/// it cannot.
pub(super) fn utility(
    tw: &Tailwind,
    u: &str,
    neg: bool,
    class: &str,
    g: &mut Group,
) -> Result<(), String> {
    let mut emit = |name: &str, value: Val| {
        g.decls.push(Decl {
            name: name.to_owned(),
            value,
            class: class.to_owned(),
        })
    };
    // An arbitrary property: `[mask-image:linear-gradient(...)]`.
    if let Some(inner) = arbitrary(u) {
        let (name, value) = inner
            .split_once(':')
            .ok_or("an arbitrary property is `[name:value]`")?;
        emit(name.trim(), value.trim().into());
        return Ok(());
    }
    if let Some(list) = keyword(u).filter(|_| !neg) {
        for (name, value) in list {
            emit(name, (*value).into());
        }
        return Ok(());
    }
    let Some((root, v)) = split_root(u) else {
        return Err(unsupported(u, None)
            .map(str::to_owned)
            .unwrap_or_else(|| "not a class this UI knows".into()));
    };
    if let Some(reason) = unsupported(u, Some(root)) {
        return Err(reason.into());
    }
    if neg && !NEGATABLE.contains(&root) {
        return Err(
            "`-` goes only on margins, offsets, translations, rotations and `order`".into(),
        );
    }
    // `px` → padding-left and -right; `border-t` → border-top.
    let sides = |prop: &str| -> Vec<String> {
        let side = |s: &str| format!("{prop}-{s}");
        let which = root
            .strip_prefix("border-")
            .unwrap_or(if root.starts_with("border") {
                ""
            } else {
                &root[1..]
            });
        match which {
            "" => vec![prop.to_owned()],
            "x" => vec![side("left"), side("right")],
            "y" => vec![side("top"), side("bottom")],
            "t" => vec![side("top")],
            "r" | "e" => vec![side("right")],
            "b" => vec![side("bottom")],
            _ => vec![side("left")],
        }
    };
    let found = (|| -> Option<()> {
        match root {
            "p" | "px" | "py" | "pt" | "pr" | "pb" | "pl" | "ps" | "pe" => {
                let value = tw.space(v, false)?;
                for p in sides("padding") {
                    emit(&p, value.clone().into());
                }
            }
            "m" | "mx" | "my" | "mt" | "mr" | "mb" | "ml" | "ms" | "me" => {
                let value = tw.space(v, neg)?;
                for p in sides("margin") {
                    emit(&p, value.clone().into());
                }
            }
            "inset" | "inset-x" | "inset-y" | "top" | "right" | "bottom" | "left" | "start"
            | "end" => {
                let value = tw.size(v, neg)?;
                let props: &[&str] = match root {
                    "inset" => &["top", "right", "bottom", "left"],
                    "inset-x" => &["left", "right"],
                    "inset-y" => &["top", "bottom"],
                    "top" => &["top"],
                    "right" | "end" => &["right"],
                    "bottom" => &["bottom"],
                    _ => &["left"],
                };
                for p in props {
                    emit(p, value.clone().into());
                }
            }
            "w" | "h" | "size" | "min-w" | "min-h" | "max-h" | "basis" => {
                let value = match v {
                    // Shrink-wrapped, as `auto` is here.
                    "fit" | "min" | "max" => "auto".into(),
                    "none" if root == "max-h" => "auto".into(),
                    _ => tw.size(v, false)?,
                };
                let props: &[&str] = match root {
                    "w" => &["width"],
                    "h" => &["height"],
                    "size" => &["width", "height"],
                    "min-w" => &["min-width"],
                    "min-h" => &["min-height"],
                    "max-h" => &["max-height"],
                    _ => &["flex-basis"],
                };
                for p in props {
                    emit(p, value.clone().into());
                }
            }
            "max-w" => {
                let value = match v {
                    "none" => "auto".into(),
                    "full" => "100%".into(),
                    "fit" | "min" | "max" => "auto".into(),
                    _ => match tw.max_widths.get(v) {
                        Some(&w) => px(w),
                        None => match v
                            .strip_prefix("screen-")
                            .and_then(|s| tw.screens.iter().find(|(n, _)| n == s))
                        {
                            Some(&(_, w)) => px(w),
                            None => tw.space(v, false)?,
                        },
                    },
                };
                emit("max-width", value.into());
            }
            "flex" => emit("flex", arbitrary(v)?.into()),
            "grow" | "shrink" | "flex-grow" | "flex-shrink" => {
                let prop = if root.ends_with("grow") {
                    "flex-grow"
                } else {
                    "flex-shrink"
                };
                emit(prop, arbitrary(v).unwrap_or_else(|| v.to_owned()).into());
            }
            "order" => {
                let n: i32 = arbitrary(v).unwrap_or_else(|| v.to_owned()).parse().ok()?;
                emit("order", (if neg { -n } else { n }).to_string().into());
            }
            "gap" | "gap-x" | "gap-y" => {
                let prop = match root {
                    "gap" => "gap",
                    "gap-x" => "column-gap",
                    _ => "row-gap",
                };
                emit(prop, tw.space(v, false)?.into());
            }
            "items" | "self" | "justify" | "content" => {
                let value = match v {
                    "start" | "end" => format!("flex-{v}"),
                    "between" | "around" | "evenly" if root == "justify" || root == "content" => {
                        format!("space-{v}")
                    }
                    "center" => v.into(),
                    "baseline" if root != "justify" && root != "content" => v.into(),
                    "stretch" if root != "justify" => v.into(),
                    "auto" if root == "self" => v.into(),
                    "normal" if root == "content" => v.into(),
                    _ => return None,
                };
                let prop = match root {
                    "items" => "align-items",
                    "self" => "align-self",
                    "justify" => "justify-content",
                    _ => "align-content",
                };
                emit(prop, value.into());
            }
            "grid-cols" | "grid-rows" => {
                let value = match arbitrary(v) {
                    Some(a) => a,
                    None => format!(
                        "repeat({}, minmax(0, 1fr))",
                        v.parse::<u32>().ok().filter(|&n| n > 0)?
                    ),
                };
                emit(
                    if root == "grid-cols" {
                        "grid-template-columns"
                    } else {
                        "grid-template-rows"
                    },
                    value.into(),
                );
            }
            "col-span" | "col-start" | "col-end" | "row-span" | "row-start" | "row-end" => {
                let p = if root.starts_with("col") {
                    &mut g.columns
                } else {
                    &mut g.rows
                };
                let line = |v: &str| {
                    arbitrary(v)
                        .unwrap_or_else(|| v.to_owned())
                        .parse::<i32>()
                        .ok()
                };
                match (&root[4..], v) {
                    ("span", "full") => {
                        *p = Placement {
                            start: Some(1),
                            end: Some(-1),
                            span: None,
                            class: String::new(),
                        }
                    }
                    ("span", v) => p.span = Some(line(v).filter(|&n| n > 0)? as u32),
                    ("start", v) => p.start = Some(line(v)?),
                    (_, v) => p.end = Some(line(v)?),
                }
                p.class = class.to_owned();
            }
            "text" => {
                if matches!(v, "left" | "center" | "right" | "justify" | "start" | "end") {
                    emit("text-align", v.into());
                    return Some(());
                }
                if let Some(c) = tw.paint(v) {
                    emit("color", Val::paint(&[c], |c| css_color(c[0])));
                    return Some(());
                }
                // A size, and `/6` or `/[18px]` for its line height.
                let (size, leading) = match split_top(v, '/').as_slice() {
                    [s, l] => (*s, Some(*l)),
                    _ => (v, None),
                };
                match (arbitrary(size), tw.font_sizes.get(size)) {
                    (Some(a), _) if crate::css::length(&a).is_some() => emit("font-size", a.into()),
                    (None, Some((size, lh))) => {
                        emit("font-size", px(*size).into());
                        if let (Some(lh), None) = (lh, leading) {
                            emit("line-height", lh.clone().into());
                        }
                    }
                    _ => return None,
                }
                if let Some(l) = leading {
                    emit("line-height", tw.leading(l)?.into());
                }
            }
            "font" => {
                if let Some(&w) = tw.font_weights.get(v) {
                    emit("font-weight", w.to_string().into());
                } else if let Some(f) = tw.font_families.get(v) {
                    emit("font-family", f.clone().into());
                } else {
                    let a = arbitrary(v)?;
                    emit(
                        if a.parse::<u16>().is_ok() {
                            "font-weight"
                        } else {
                            "font-family"
                        },
                        a.into(),
                    );
                }
            }
            "leading" => emit("line-height", tw.leading(v)?.into()),
            "line-clamp" => emit(
                "line-clamp",
                arbitrary(v).unwrap_or_else(|| v.to_owned()).into(),
            ),
            "decoration" => emit(
                "text-decoration-color",
                Val::paint(&[tw.paint(v)?], |c| css_color(c[0])),
            ),
            "whitespace" => emit("white-space", v.into()),
            "select" => emit("user-select", v.into()),
            "bg-gradient-to" | "bg-linear-to" => {
                let side = |c: u8| match c {
                    b't' => Some("top"),
                    b'r' => Some("right"),
                    b'b' => Some("bottom"),
                    b'l' => Some("left"),
                    _ => None,
                };
                let dir = match v.as_bytes() {
                    [a] => format!("to {}", side(*a)?),
                    [a, b] => format!("to {} {}", side(*a)?, side(*b)?),
                    _ => return None,
                };
                g.gradient.direction = Some((dir, class.to_owned()));
            }
            "from" | "via" | "to" => {
                let stop = &mut g.gradient.stops[match root {
                    "from" => 0,
                    "via" => 1,
                    _ => 2,
                }];
                match v.strip_suffix('%').and_then(|p| p.parse::<f32>().ok()) {
                    Some(p) => stop.1 = Some(p),
                    None => stop.0 = Some(tw.paint(v)?),
                }
                g.gradient.class.get_or_insert_with(|| class.to_owned());
            }
            "bg" => match tw.paint(v) {
                Some(c) => emit("background-color", Val::paint(&[c], |c| css_color(c[0]))),
                None => emit("background", arbitrary(v)?.into()),
            },
            "border" | "border-x" | "border-y" | "border-t" | "border-r" | "border-b"
            | "border-l" | "border-s" | "border-e" => {
                let width = match v {
                    "" => Some(px(1.0)),
                    w => match arbitrary(w) {
                        Some(a) if crate::css::length(&a).is_some() => Some(a),
                        Some(_) => None,
                        None => w.parse::<f32>().ok().filter(|n| *n >= 0.0).map(px),
                    },
                };
                if let Some(w) = width {
                    let props = if root == "border" {
                        vec!["border-width".to_owned()]
                    } else {
                        sides("border")
                    };
                    for p in props {
                        emit(&p, w.clone().into());
                    }
                    g.border_width = true;
                } else if root == "border" {
                    emit(
                        "border-color",
                        Val::paint(&[tw.paint(v)?], |c| css_color(c[0])),
                    );
                    g.border_color = true;
                } else {
                    return None;
                }
            }
            "rounded" => {
                let corner = [
                    "t", "b", "l", "r", "s", "e", "tl", "tr", "bl", "br", "ss", "se", "es", "ee",
                ]
                .iter()
                .any(|c| v == *c || v.strip_prefix(c).is_some_and(|r| r.starts_with('-')));
                if corner {
                    return None;
                }
                let r = match tw.radii.get(v) {
                    Some(&r) => px(r),
                    None => arbitrary(v)?,
                };
                emit("border-radius", r.into());
            }
            "shadow" => {
                let css = tw.shadows.get(v).cloned().or_else(|| arbitrary(v))?;
                if css.contains("inset") {
                    return None;
                }
                g.effects.shadow = Some(css);
                g.effects.class = Some(class.to_owned());
            }
            "ring" | "ring-offset" => {
                let width = match v {
                    "" if root == "ring" => Some(tw.ring_width),
                    w => match arbitrary(w) {
                        Some(a) => crate::css::length(&a).and_then(|l| match l {
                            crate::Length::Px(p) => Some(p),
                            _ => None,
                        }),
                        None => w.parse::<f32>().ok().filter(|n| *n >= 0.0),
                    },
                };
                let e = &mut g.effects;
                match (root, width) {
                    ("ring", Some(w)) => e.ring = Some(w),
                    ("ring-offset", Some(w)) => e.offset = Some(w),
                    ("ring", None) if v != "inset" => e.ring_color = Some(tw.paint(v)?),
                    ("ring-offset", None) => e.offset_color = Some(tw.paint(v)?),
                    _ => return None,
                }
                e.class = Some(class.to_owned());
            }
            "opacity" => {
                let o = match arbitrary(v) {
                    Some(a) => a,
                    None => (v.parse::<f32>().ok()? / 100.0).to_string(),
                };
                emit("opacity", o.into());
            }
            "blur" => emit("filter", format!("blur({})", tw.blur_px(v)?).into()),
            "backdrop-blur" => {
                g.backdrop.blur = Some(tw.blur_px(v)?);
                g.backdrop.class.get_or_insert_with(|| class.to_owned());
            }
            "backdrop-saturate" => {
                g.backdrop.saturate = Some(match arbitrary(v) {
                    Some(a) => a,
                    None => format!("{}%", v.parse::<f32>().ok()?),
                });
                g.backdrop.class.get_or_insert_with(|| class.to_owned());
            }
            "mix-blend" => emit("mix-blend-mode", v.into()),
            // 3D: turned, lifted, scaled and seen in perspective.
            "rotate" | "rotate-x" | "rotate-y" | "rotate-z" => {
                let deg = match arbitrary(v) {
                    Some(a) => a,
                    None => format!("{}deg", v.parse::<f32>().ok()?),
                };
                let prop = match root {
                    "rotate-x" => "rotate-x",
                    "rotate-y" => "rotate-y",
                    _ => "rotate",
                };
                emit(prop, (if neg { format!("-{deg}") } else { deg }).into());
            }
            "translate" | "translate-x" | "translate-y" | "translate-z" => {
                if v.contains('/') || v == "full" {
                    return None;
                }
                let value = tw.space(v, neg)?;
                let props: &[&str] = match root {
                    "translate" => &["translate-x", "translate-y"],
                    "translate-x" => &["translate-x"],
                    "translate-y" => &["translate-y"],
                    _ => &["translate-z"],
                };
                for p in props {
                    emit(p, value.clone().into());
                }
            }
            "scale" => {
                let s = match arbitrary(v) {
                    Some(a) => a,
                    None => (v.parse::<f32>().ok()? / 100.0).to_string(),
                };
                emit("scale", s.into());
            }
            "perspective" => {
                let p = match (v, tw.perspectives.get(v)) {
                    ("none", _) => "none".into(),
                    (_, Some(&p)) => px(p),
                    _ => arbitrary(v)?,
                };
                emit("perspective", p.into());
            }
            "origin" => {
                let o = match arbitrary(v) {
                    Some(a) => a,
                    None => v.replace('-', " "),
                };
                emit("transform-origin", o.into());
            }
            "overflow" | "overflow-x" | "overflow-y" | "cursor" | "object" | "scrollbar"
            | "aspect" => {
                let prop = match root {
                    "cursor" => "cursor",
                    "object" => "object-fit",
                    "scrollbar" => "scrollbar-width",
                    "aspect" => "aspect-ratio",
                    overflow => overflow,
                };
                emit(prop, arbitrary(v).unwrap_or_else(|| v.to_owned()).into());
            }
            _ => return None,
        }
        Some(())
    })();
    found.ok_or_else(|| why_not(tw, root, v))
}

/// Why `<root>-<v>` was not read: a fraction for a translation, a corner
/// radius, or a value not in the theme (and the nearest that is).
fn why_not(tw: &Tailwind, root: &str, v: &str) -> String {
    match root {
        "translate" | "translate-x" | "translate-y" if v.contains('/') || v == "full" => {
            "translations are in pixels, not fractions of the box".into()
        }
        "rounded" => format!("one radius per box: `rounded-*` (the theme has {})", list(tw.radii.keys().map(String::as_str))),
        "shadow" if v == "inner" || v.contains("inset") => "no inset shadows".into(),
        "ring" if v == "inset" => "no inset rings".into(),
        "m" | "mx" | "my" | "mt" | "mr" | "mb" | "ml" | "ms" | "me" if v == "auto" => {
            "no auto margins: centre with `justify-center`, `items-center` or `self-center` on the flex parent".into()
        }
        "w" | "h" | "min-w" | "min-h" | "max-w" | "max-h" | "size" if matches!(v, "screen" | "dvh" | "svh" | "lvh" | "dvw") => {
            "no viewport units: `w-full`/`h-full` in a box as large as the window".into()
        }
        _ => {
            let colour = matches!(root, "bg" | "text" | "border" | "ring" | "ring-offset" | "decoration" | "from" | "via" | "to");
            let name = v.split('/').next().unwrap_or(v);
            match nearest(name, tw.color_names()).filter(|_| colour) {
                Some(n) => format!("no colour `{name}` in the theme: did you mean `{root}-{n}`?"),
                None => format!("`{v}` is not a value `{root}-` takes here"),
            }
        }
    }
}

fn list<'a>(names: impl Iterator<Item = &'a str>) -> String {
    let mut names: Vec<String> = names
        .map(|n| {
            if n.is_empty() {
                "`rounded`".to_owned()
            } else {
                format!("`{n}`")
            }
        })
        .collect();
    names.sort();
    names.join(", ")
}

impl Tailwind {
    /// Spacing: `4` (steps), `px`, a name, `[13px]`. Negated with `neg`.
    pub(super) fn space(&self, v: &str, neg: bool) -> Option<String> {
        let sign = if neg { "-" } else { "" };
        if let Some(a) = arbitrary(v) {
            return Some(format!("{sign}{a}"));
        }
        let n = match self.spacing.get(v) {
            Some(&n) => n,
            None => {
                let steps: f32 = v.parse().ok()?;
                (steps >= 0.0).then_some(steps * self.spacing_unit)?
            }
        };
        Some(px(if neg { -n } else { n }))
    }

    /// Spacing, or `full`, `auto`, a fraction (`1/2`).
    pub(super) fn size(&self, v: &str, neg: bool) -> Option<String> {
        if let Some(s) = self.space(v, neg) {
            return Some(s);
        }
        let sign = if neg { "-" } else { "" };
        match v {
            "full" => Some(format!("{sign}100%")),
            "auto" if !neg => Some("auto".into()),
            _ => {
                let (a, b) = v.split_once('/')?;
                let (a, b): (f32, f32) = (a.parse().ok()?, b.parse().ok()?);
                (b > 0.0).then(|| format!("{sign}{}%", a / b * 100.0))
            }
        }
    }

    /// A colour: `red-500`, `primary`, `[#1da1f2]`, with `/50` or `/[0.37]`
    /// for its opacity.
    pub(super) fn paint(&self, v: &str) -> Option<Paint> {
        let (name, alpha) = match split_top(v, '/').as_slice() {
            [n, a] => {
                let a = match arbitrary(a) {
                    Some(a) => a.parse::<f32>().ok()?,
                    None => a.parse::<f32>().ok()? / 100.0,
                };
                (*n, Some(a.clamp(0.0, 1.0)))
            }
            _ => (v, None),
        };
        let p = match arbitrary(name) {
            Some(a) => Paint::fixed(crate::css::color(&a)?),
            None => *self.colors.get(name)?,
        };
        Some(match alpha {
            Some(a) => p.faded(a),
            None => p,
        })
    }

    fn leading(&self, v: &str) -> Option<String> {
        if let Some(l) = self.line_heights.get(v) {
            return Some(l.clone());
        }
        if let Some(a) = arbitrary(v) {
            return Some(a);
        }
        self.space(v, false)
    }

    fn blur_px(&self, v: &str) -> Option<String> {
        match (self.blurs.get(v), arbitrary(v)) {
            (Some(&b), _) => Some(px(b)),
            (None, Some(a)) => Some(a),
            _ => None,
        }
    }
}
