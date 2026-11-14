//! Tailwind's utility classes: `"flex items-center gap-2 px-4 py-2
//! rounded-lg bg-blue-500 hover:bg-blue-600 text-white"` on a node, turned
//! into the CSS declarations [`style!`](crate::style) reads, conditional
//! state styles ([`Ui::when_style`]) and 3D transforms
//! ([`depth`](crate::depth)).
//!
//! ```
//! use threers_ui::{FixedWidthText, Length, Ui};
//!
//! let mut ui = Ui::new();
//! let root = ui.root();
//! let bar = ui.add_tw(root, "flex flex-row items-center gap-2 h-12 px-4 bg-slate-900");
//! let button = ui.add_tw(bar, "h-8 px-3 py-1 rounded-md bg-blue-500 hover:bg-blue-600 text-sm text-white");
//! ui.add_text(button, "Save");
//! ui.layout(800.0, 600.0, &mut FixedWidthText(7.0));
//! assert_eq!(ui.rect(bar).height, 48.0);
//! assert_eq!(ui.rect(button).height, 32.0, "border-box, as Tailwind's preflight has it");
//! assert_eq!(ui.style(button).border_radius, Length::Px(6.0));
//! ```
//!
//! # Themes
//!
//! The scales are Tailwind 3.4's defaults: its colour palette (`slate`
//! through `rose`, `50` to `950`), spacing in quarter-rems (`p-4` is 16px,
//! and any number works, as in Tailwind 4), font sizes with their line
//! heights, radii, shadows, blurs, perspectives and breakpoints.
//! [`Tailwind::shadcn`] is shadcn/ui's look instead: its neutral theme's
//! colours (`bg-background`, `text-primary-foreground`, `border-input`,
//! `ring-ring`, … each with its own value in the dark appearance) and its
//! components as classes (`button button-outline button-sm`, `card`,
//! `input`, `badge`, …). A [`Tailwind`] set as the UI's global
//! ([`Ui::set_global`]) is what [`Ui::tw`] reads with: colours (fixed or
//! [adaptive](Tailwind::adaptive_color)), spacing, breakpoints and the
//! rest, and components made of other classes (CSS's `@apply`). With the
//! `json` feature, [`Tailwind::extend_json`] reads them from a
//! `tailwind.config`-shaped JSON file.
//!
//! # Classes
//!
//! - **Values:** arbitrary values (`w-[137px]`, `bg-[#1da1f2]`,
//!   `grid-cols-[200px_1fr]`, underscores for spaces), arbitrary properties
//!   (`[mask-image:linear-gradient(...)]`), negatives (`-mt-2`), fractions
//!   (`w-1/2`), colour opacity (`bg-black/50`), `!important` (read, and
//!   ignored: the later class wins anyway).
//! - **Variants,** which stack (`dark:hover:`, `md:focus-visible:`):
//!   `hover:`, `active:`, `focus:`, `focus-visible:`, `focus-within:`,
//!   `disabled:`, `dark:`, `group-hover:`, `group-active:`, `group-focus:`
//!   and `group-focus-within:` (with `group` on an ancestor, or
//!   `group/name` and `group-hover/name:`), and the breakpoints `sm:` to
//!   `2xl:`, `max-sm:` to `max-2xl:`, `min-[600px]:` and `max-[600px]:`
//!   (widths of the window, the last layout's viewport). Where several
//!   match, they go on in Tailwind's order: breakpoints over the dark
//!   appearance over group states over the node's own.
//! - **Rings:** `ring`, `ring-2`, `ring-blue-500/50`, `ring-offset-2`,
//!   `ring-offset-white` stack over `shadow-*` in one `box-shadow`.
//! - **3D:** `rotate-x-*`, `rotate-y-*`, `rotate-*` (degrees),
//!   `translate-x-*`, `translate-y-*`, `translate-z-*` (spacing, the last
//!   toward the viewer), `scale-*` (percent), `perspective-*`
//!   (`dramatic` 100 to `distant` 1200, or `[650px]`), `origin-*`,
//!   `backface-hidden`, `transform-none`; and, threers' own, `glare`,
//!   `shading`, `depth-shadow` and `parallax-*` (see [`depth`](crate::depth)).
//!   With variants they move: `transition hover:-translate-y-1
//!   hover:scale-105`.
//! - **Motion:** `transition` (`-colors`, `-transform`, …) with
//!   `duration-*` moves colours, opacity and transforms into a state
//!   ([`Ui::transition`]).
//!
//! Where two classes set one property, the longhand wins over the shorthand
//! whatever their order (`p-4 px-2`: 8px across), as in Tailwind's CSS, and
//! otherwise the later class. As Tailwind's preflight does, a box measures
//! with `border-box`, and a border width with no colour gets the theme's
//! border colour.
//!
//! What this UI has no property for is refused, naming the class and saying
//! what to use instead, rather than dropped: `fixed`, `z-10`, `mx-auto`,
//! `space-x-2`, `tracking-wide`, `uppercase`, `rounded-t-lg` (one radius
//! per box), `shadow-inner`, `ease-in`, `translate-x-1/2`, … A misspelt
//! colour gets a suggestion. [`Ui::tw`] panics, as `style!` does;
//! [`Ui::try_tw`] returns the error.

#[cfg(feature = "json")]
mod json;
mod shadcn;
#[cfg(test)]
mod tests;
mod theme;
mod utilities;

use std::fmt;
use std::sync::OnceLock;

pub use theme::{Paint, Tailwind};

use super::states::Appearance;
pub use super::states::{GroupState, When};
use super::style::Style;
use super::tree::{NodeId, Ui};
use utilities::{nearest, utility, Decl, Group, Val};

/// Declarations, `(property, value)` in CSS text.
pub type Declarations = Vec<(String, String)>;

/// A class that could not be read, and why.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClassError {
    pub class: String,
    pub reason: String,
}

impl fmt::Display for ClassError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "tailwind: `{}`: {}", self.class, self.reason)
    }
}

impl std::error::Error for ClassError {}

fn refuse<T>(class: &str, reason: impl Into<String>) -> Result<T, ClassError> {
    Err(ClassError {
        class: class.to_owned(),
        reason: reason.into(),
    })
}

/// Declarations under a condition.
#[derive(Debug, Clone, PartialEq)]
pub struct Rule {
    pub when: When,
    pub declarations: Declarations,
    /// Only the dark values of adaptive colours set elsewhere: the rule
    /// that switches a theme's colours with the appearance.
    pub adaptive: bool,
}

/// A class list, read: declarations for the node, and rules for its
/// states, in the order they go on.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Classes {
    pub base: Declarations,
    pub rules: Vec<Rule>,
    /// The group this node is (`group`, `group/name`).
    pub group: Option<String>,
    /// Seconds colours, opacity and transforms take into a state
    /// (`transition`).
    pub transition: Option<f32>,
    /// How far in front of the box around it this one is (`parallax-4`).
    pub parallax: Option<f32>,
}

/// `[…]`'s inside, underscores as spaces (`\_` for an underscore).
fn arbitrary(v: &str) -> Option<String> {
    let inner = v.strip_prefix('[')?.strip_suffix(']')?;
    Some(
        inner
            .replace("\\_", "\u{0}")
            .replace('_', " ")
            .replace('\u{0}', "_"),
    )
}

/// Splits at `sep` outside brackets and parentheses.
fn split_top(v: &str, sep: char) -> Vec<&str> {
    let (mut out, mut depth, mut start) = (vec![], 0i32, 0);
    for (i, c) in v.char_indices() {
        match c {
            '[' | '(' => depth += 1,
            ']' | ')' => depth -= 1,
            c if c == sep && depth == 0 => {
                out.push(&v[start..i]);
                start = i + c.len_utf8();
            }
            _ => {}
        }
    }
    out.push(&v[start..]);
    out
}

/// Joins declarations as CSS text.
fn css_text(d: &Declarations) -> String {
    d.iter()
        .map(|(n, v)| format!("{n}: {v}"))
        .collect::<Vec<_>>()
        .join("; ")
}

/// What the node is rather than how it looks.
#[derive(Debug, Clone, Default)]
struct NodeClasses {
    group: Option<String>,
    /// `transition` (on) or `transition-none`.
    transition: Option<bool>,
    duration: Option<f32>,
    parallax: Option<f32>,
}

/// A class list as it is read: each condition's classes, base first.
struct Reading {
    groups: Vec<(When, Group)>,
    node: NodeClasses,
}

const STATES: [&str; 6] = [
    "hover",
    "active",
    "focus",
    "focus-visible",
    "focus-within",
    "disabled",
];

impl Tailwind {
    /// The default theme, shared rather than built for each use.
    fn shared() -> &'static Tailwind {
        static DEFAULT: OnceLock<Tailwind> = OnceLock::new();
        DEFAULT.get_or_init(Tailwind::default)
    }

    /// Reads a class list.
    pub fn parse(&self, classes: &str) -> Result<Classes, ClassError> {
        let mut r = Reading {
            groups: vec![(When::default(), Group::default())],
            node: NodeClasses::default(),
        };
        for class in classes.split_whitespace() {
            self.class(class, &[], &mut r, 0)?;
        }
        self.finish(r)
    }

    /// The style a class list gives, from the default style, in the light
    /// appearance. States, groups and transitions need a node; they are
    /// refused here (see [`Ui::tw`]).
    pub fn style(&self, classes: &str) -> Result<Style, ClassError> {
        self.style_in(classes, Appearance::Light)
    }

    /// [`style`](Self::style) in `appearance`: adaptive colours take its
    /// values.
    pub fn style_in(&self, classes: &str, appearance: Appearance) -> Result<Style, ClassError> {
        let c = self.parse(classes)?;
        let needs_node = c.group.is_some()
            || c.transition.is_some()
            || c.parallax.is_some()
            || c.rules
                .iter()
                .any(|r| !r.adaptive || r.when != When::default().dark());
        if needs_node {
            let node_only = |c: &&str| {
                (c.contains(':') && !c.starts_with('['))
                    || ["group", "transition", "duration-", "parallax-"]
                        .iter()
                        .any(|p| c.starts_with(p))
            };
            let class = classes
                .split_whitespace()
                .find(node_only)
                .unwrap_or(classes);
            return refuse(
                class,
                "states, groups and transitions need a node: `Ui::tw`",
            );
        }
        if let Some(c) = classes
            .split_whitespace()
            .find(|c| super::depth::is_property(c))
        {
            return refuse(c, "a 3D transform needs a node: `Ui::tw`");
        }
        let mut s = Style::default();
        let mut decls = c.base;
        if appearance == Appearance::Dark {
            decls.extend(c.rules.into_iter().flat_map(|r| r.declarations));
        }
        for (name, value) in &decls {
            if super::depth::is_property(name) {
                return refuse(classes, "a 3D transform needs a node: `Ui::tw`");
            }
            super::css::apply(&mut s, name, value);
        }
        Ok(s)
    }

    /// Reads one class (with the variants of the component it came from).
    fn class(
        &self,
        class: &str,
        outer: &[&str],
        r: &mut Reading,
        depth: usize,
    ) -> Result<(), ClassError> {
        let parts = split_top(class, ':');
        let (utility_text, own) = parts.split_last().expect("split gives one part at least");
        let mut variants: Vec<&str> = outer.to_vec();
        variants.extend(own);
        let u = utility_text.trim_start_matches('!').trim_end_matches('!');
        if u.is_empty() {
            return refuse(class, "no utility");
        }
        if let Some(classes) = self.components.get(u) {
            if depth > 8 {
                return refuse(class, "components nest too deep (does one include itself?)");
            }
            for inner in classes.split_whitespace() {
                self.class(inner, &variants, r, depth + 1)
                    .map_err(|e| ClassError {
                        class: if e.class == inner {
                            format!("{class} ({inner})")
                        } else {
                            e.class
                        },
                        reason: e.reason,
                    })?;
            }
            return Ok(());
        }
        let when = self.when(class, &variants)?;
        if self.node_class(class, u, &when, &mut r.node)? {
            return Ok(());
        }
        let i = match r.groups.iter().position(|(w, _)| *w == when) {
            Some(i) => i,
            None => {
                r.groups.push((when, Group::default()));
                r.groups.len() - 1
            }
        };
        let (neg, bare) = match u.strip_prefix('-') {
            Some(b) => (true, b),
            None => (false, u),
        };
        utility(self, bare, neg, class, &mut r.groups[i].1).or_else(|reason| refuse(class, reason))
    }

    /// Classes that say what the node is: a group, its transition, its
    /// parallax. Whether `u` was one.
    fn node_class(
        &self,
        class: &str,
        u: &str,
        when: &When,
        node: &mut NodeClasses,
    ) -> Result<bool, ClassError> {
        let is_node = u == "group"
            || u.starts_with("group/")
            || u.starts_with("transition")
            || u.starts_with("duration-")
            || u.trim_start_matches('-').starts_with("parallax-");
        if !is_node {
            return Ok(false);
        }
        if *when != When::default() {
            return refuse(
                class,
                "groups, transitions and parallax cannot be conditional",
            );
        }
        if u == "group" || u.starts_with("group/") {
            node.group = Some(u.strip_prefix("group/").unwrap_or("group").to_owned());
        } else if let Some(rest) = u.strip_prefix("transition") {
            node.transition = Some(match rest {
                "-none" => false,
                "" | "-all" | "-colors" | "-opacity" | "-transform" | "-shadow" => true,
                r if r.strip_prefix('-').and_then(arbitrary).is_some() => true,
                _ => return refuse(class, "`transition`, `transition-colors`, `-opacity`, `-transform`, `-shadow`, `-all`, `-[…]` or `-none`"),
            });
        } else if let Some(ms) = u.strip_prefix("duration-") {
            let ms = match arbitrary(ms) {
                Some(a) => a
                    .strip_suffix("ms")
                    .and_then(|m| m.parse::<f32>().ok())
                    .or_else(|| {
                        a.strip_suffix('s')
                            .and_then(|s| s.parse::<f32>().ok())
                            .map(|s| s * 1000.0)
                    }),
                None => ms.parse::<f32>().ok(),
            };
            let ms = ms.filter(|m| *m >= 0.0).ok_or_else(|| ClassError {
                class: class.into(),
                reason: "a duration is milliseconds: `duration-300`, `duration-[1.5s]`".into(),
            })?;
            node.duration = Some(ms / 1000.0);
        } else {
            let (neg, v) = match u.strip_prefix('-') {
                Some(r) => (true, r.strip_prefix("parallax-").unwrap_or(r)),
                None => (false, u.strip_prefix("parallax-").unwrap_or(u)),
            };
            let px = self
                .space(v, neg)
                .and_then(|s| crate::css::length(&s))
                .and_then(|l| match l {
                    crate::Length::Px(p) => Some(p),
                    _ => None,
                });
            node.parallax = Some(px.ok_or_else(|| ClassError {
                class: class.into(),
                reason: "parallax is a spacing value: `parallax-4`, `-parallax-2`".into(),
            })?);
        }
        Ok(true)
    }

    /// The condition `variants` make together.
    fn when(&self, class: &str, variants: &[&str]) -> Result<When, ClassError> {
        let mut w = When::default();
        let screen = |name: &str| -> Option<f32> {
            if let Some(a) = name.strip_prefix('[').and_then(|_| arbitrary(name)) {
                return crate::css::length(&a).and_then(|l| match l {
                    crate::Length::Px(p) => Some(p),
                    _ => None,
                });
            }
            self.screens.iter().find(|(n, _)| n == name).map(|s| s.1)
        };
        for v in variants {
            match *v {
                "hover" => w.hover = true,
                "active" => w.active = true,
                "focus" => w.focus = true,
                "focus-visible" => w.focus_visible = true,
                "focus-within" => w.focus_within = true,
                "disabled" => w.disabled = true,
                "dark" => w.appearance = Some(Appearance::Dark),
                _ => {
                    if let Some(rest) = v.strip_prefix("group-") {
                        let (state, name) = rest.split_once('/').unwrap_or((rest, "group"));
                        let state = match state {
                            "hover" => GroupState::Hover,
                            "active" => GroupState::Active,
                            "focus" => GroupState::Focus,
                            "focus-within" => GroupState::FocusWithin,
                            _ => {
                                return refuse(
                                    class,
                                    format!(
                                    "`{v}:`: groups are hovered, active, focused or focused within"
                                ),
                                )
                            }
                        };
                        if w.group.is_some() {
                            return refuse(class, "one group variant per class");
                        }
                        w.group = Some((name.to_owned(), state));
                    } else if let Some(width) =
                        screen(v).or_else(|| v.strip_prefix("min-").and_then(screen))
                    {
                        w = w.min_width(width);
                    } else if let Some(width) = v.strip_prefix("max-").and_then(screen) {
                        w = w.max_width(width);
                    } else {
                        return refuse(class, self.unknown_variant(v));
                    }
                }
            }
        }
        Ok(w)
    }

    fn unknown_variant(&self, v: &str) -> String {
        let first = v.split(['-', '[']).next().unwrap_or(v);
        let why = match first {
            "peer" => "no sibling variants: name a group on the parent and use `group-*:`",
            "first" | "last" | "odd" | "even" | "only" | "nth" => "no child-position variants: give those children their classes",
            "aria" | "data" | "has" | "in" | "not" | "open" | "checked" | "invalid" | "required" => "no attribute variants: the states are hover, active, focus, focus-visible, focus-within and disabled",
            "before" | "after" | "placeholder" | "file" | "marker" | "selection" | "backdrop" => "no pseudo-elements: add a node",
            "motion" => "no reduced-motion variant: `Ui::reduce_motion` stops transitions anyway",
            "print" | "portrait" | "landscape" | "contrast" | "forced" | "rtl" | "ltr" | "supports" | "starting" => "no such media variant here",
            _ => {
                let known = STATES.iter().copied().chain(["dark"]).chain(self.screens.iter().map(|(n, _)| n.as_str()));
                return match nearest(v, known) {
                    Some(n) => format!("`{v}:` is not a variant: did you mean `{n}:`?"),
                    None => format!("`{v}:` is not a variant this UI knows"),
                };
            }
        };
        format!("`{v}:`: {why}")
    }

    /// Joins up each condition's classes, orders the rules, and checks
    /// every declaration.
    fn finish(&self, r: Reading) -> Result<Classes, ClassError> {
        let n = r.node;
        let mut out = Classes {
            group: n.group,
            transition: n.transition.map(|on| {
                if on {
                    n.duration.unwrap_or(self.duration)
                } else {
                    0.0
                }
            }),
            parallax: n.parallax,
            ..Classes::default()
        };
        let base_group = r.groups[0].1.clone();
        let mut rules = vec![];
        for (i, (when, g)) in r.groups.into_iter().enumerate() {
            let base = (i > 0).then_some(&base_group);
            let mut decls = g
                .joined(base, self)
                .map_err(|(class, reason)| ClassError { class, reason })?;
            if i == 0 && self.preflight {
                decls.insert(
                    0,
                    Decl {
                        name: "box-sizing".into(),
                        value: Val::Text("border-box".into()),
                        class: String::new(),
                    },
                );
            }
            decls.sort_by_key(rank);
            check(&decls)?;
            let in_dark = when.appearance == Some(Appearance::Dark);
            let main: Declarations = decls
                .iter()
                .map(|d| (d.name.clone(), d.value.get(in_dark).to_owned()))
                .collect();
            // A theme's colours switch with the appearance: their dark
            // values in a rule of their own, right after this one. Only
            // where the colour is the last word on its property: in
            // `button button-ghost`, `bg-transparent` replaces `bg-primary`
            // in the dark too.
            let last = |i: usize| {
                !decls[i + 1..]
                    .iter()
                    .any(|later| later.name == decls[i].name)
            };
            let dark: Declarations = (0..decls.len())
                .filter(|&i| {
                    when.appearance.is_none()
                        && matches!(decls[i].value, Val::Adaptive { .. })
                        && last(i)
                })
                .map(|i| (decls[i].name.clone(), decls[i].value.get(true).to_owned()))
                .collect();
            let key = order(&when);
            if i == 0 {
                out.base = main;
            } else if !main.is_empty() {
                rules.push((
                    key,
                    0,
                    Rule {
                        when: when.clone(),
                        declarations: main,
                        adaptive: false,
                    },
                ));
            }
            if !dark.is_empty() {
                rules.push((
                    key,
                    1,
                    Rule {
                        when: when.dark(),
                        declarations: dark,
                        adaptive: true,
                    },
                ));
            }
        }
        rules.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
        out.rules = rules.into_iter().map(|(_, _, r)| r).collect();
        Ok(out)
    }
}

/// Where a declaration goes among the others: shorthands, then
/// longhands, then the line height a font size brings, then `leading-*`'s.
fn rank(d: &Decl) -> u8 {
    match d.name.as_str() {
        "padding" | "margin" | "border" | "border-width" | "flex" | "gap" | "overflow"
        | "background" | "font-size" => 0,
        "line-height"
            if d.class
                .rsplit(':')
                .next()
                .unwrap_or("")
                .trim_start_matches('!')
                .starts_with("text-") =>
        {
            2
        }
        "line-height" => 3,
        _ => 1,
    }
}

/// Tailwind's order for rules that apply together: breakpoints (wider
/// later; `max-*` narrower later), then the dark appearance, then group
/// states, then the node's own (`focus-within`, `hover`, `focus`,
/// `focus-visible`, `active`, `disabled`).
fn order(w: &When) -> [i64; 5] {
    let own = [
        w.focus_within,
        w.hover,
        w.focus,
        w.focus_visible,
        w.active,
        w.disabled,
    ]
    .iter()
    .enumerate()
    .filter(|(_, on)| **on)
    .map(|(i, _)| i as i64 + 1)
    .max()
    .unwrap_or(0);
    let group = w.group.as_ref().map_or(0, |(_, s)| match s {
        GroupState::FocusWithin => 1,
        GroupState::Hover => 2,
        GroupState::Focus => 3,
        GroupState::Active => 4,
    });
    [
        w.min_width.map_or(-1, |v| v as i64),
        w.max_width.map_or(i64::MIN, |v| -(v as i64)),
        i64::from(w.appearance.is_some()),
        group,
        own,
    ]
}

/// Every declaration, in each appearance, is one this UI reads.
fn check(decls: &[Decl]) -> Result<(), ClassError> {
    let mut style = Style::default();
    let mut transform = super::depth::Transform3d::default();
    for d in decls {
        for dark in [false, true] {
            let v = d.value.get(dark);
            let ok = if super::depth::is_property(&d.name) {
                super::depth::apply_property(&mut transform, &d.name, v)
            } else {
                super::css::try_apply(&mut style, &d.name, v)
            };
            if !ok {
                return refuse(&d.class, format!("`{}: {v}` is not supported", d.name));
            }
        }
    }
    Ok(())
}

/// The style a class list gives, read with Tailwind's default theme.
/// Panics on a class it cannot read, as [`style!`](crate::style) does on
/// CSS; states, groups and transforms need a node (see [`Ui::tw`]).
///
/// ```
/// use threers_ui::{tw, Length};
///
/// let s = tw("flex flex-col gap-4 p-6 w-1/2 rounded-xl bg-white");
/// assert_eq!(s.gap, 16.0);
/// assert_eq!(s.width, Length::Percent(50.0));
/// ```
#[track_caller]
pub fn tw(classes: &str) -> Style {
    Tailwind::shared()
        .style(classes)
        .unwrap_or_else(|e| panic!("{e}"))
}

impl Ui {
    /// Gives `node` Tailwind classes, read with the UI's [`Tailwind`]
    /// global or the default theme: its own style and 3D transform change,
    /// and its variants become [conditional styles](Ui::when_style) (see
    /// the [module docs](crate::tailwind)). Classes add to what the node
    /// has. Panics on a class it cannot read.
    #[track_caller]
    pub fn tw(&mut self, node: NodeId, classes: &str) {
        if let Err(e) = self.try_tw(node, classes) {
            panic!("{e}");
        }
    }

    /// [`tw`](Self::tw), returning a class it cannot read (and changing
    /// nothing) rather than panicking.
    pub fn try_tw(&mut self, node: NodeId, classes: &str) -> Result<(), ClassError> {
        let c = match self.global::<Tailwind>() {
            Some(theme) => theme.read().parse(classes)?,
            None => Tailwind::shared().parse(classes)?,
        };
        self.apply_classes(node, &c);
        Ok(())
    }

    /// Adds a box styled by Tailwind classes (see [`tw`](Self::tw)).
    #[track_caller]
    pub fn add_tw(&mut self, parent: NodeId, classes: &str) -> NodeId {
        let node = self.add_box(parent, Style::default());
        self.tw(node, classes);
        node
    }

    /// Gives `node` classes already read.
    pub fn apply_classes(&mut self, node: NodeId, c: &Classes) {
        use super::depth::{apply_property, is_property};
        let (moves, looks): (Declarations, Declarations) = c
            .base
            .iter()
            .cloned()
            .partition(|(name, _)| is_property(name));
        if !looks.is_empty() {
            self.update_style(node, |s| {
                for (name, value) in &looks {
                    super::css::apply(s, name, value);
                }
            });
        }
        if let Some(name) = &c.group {
            self.group(node, name);
        }
        if let Some(t) = c.transition {
            self.transition(node, t);
        }
        if let Some(depth) = c.parallax {
            self.set_parallax(node, depth);
        }
        let rules_move = c
            .rules
            .iter()
            .any(|r| r.declarations.iter().any(|(name, _)| is_property(name)));
        if !moves.is_empty() || rules_move {
            let states = self.node(node).states.as_ref();
            let mut t = states
                .and_then(|s| s.base_transform)
                .or(self.transform3d(node))
                .unwrap_or_default();
            for (name, value) in &moves {
                apply_property(&mut t, name, value);
            }
            if c.rules.is_empty() && states.is_none() {
                self.set_transform3d(node, t);
            } else {
                self.set_base_transform(node, t);
            }
        }
        for r in &c.rules {
            self.when_style(node, r.when.clone(), &css_text(&r.declarations));
        }
    }
}
