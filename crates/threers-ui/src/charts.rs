//! Gauges and charts: a value in its range as a bar or a dial
//! ([`gauge`]), values as bars ([`bar_chart`]) and as a line over an area
//! ([`line_chart`], small and bare: a sparkline). Each says what it shows
//! to assistive technology: a gauge its value, a chart its values in order
//! and their highest and lowest.
//!
//! ```
//! use threers_ui::charts::{bar_chart, gauge, GaugeKind};
//! use threers_ui::{FixedWidthText, Ui};
//!
//! let mut ui = Ui::new();
//! let root = ui.root();
//! let mut g = gauge(&mut ui, root, "Battery", 0.0..=100.0, GaugeKind::Circular);
//! g.set(&mut ui, 72.0);
//! bar_chart(&mut ui, root, "Steps", &[("Mon", 4.0), ("Tue", 7.5), ("Wed", 6.0)], 80.0);
//! ui.layout(400.0, 300.0, &mut FixedWidthText(7.0));
//! assert_eq!(g.value(), 72.0);
//! ```

use std::ops::RangeInclusive;

use crate::environment::Token;
use crate::tree::{NodeId, Ui};
use crate::style::Rgba;
use crate::{style, Length, Role, Semantics};

/// How a gauge shows its value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GaugeKind {
    /// A bar filled from the left, its name and value above it and its
    /// range's ends below.
    Linear,
    /// A thin track (the tint's colours along it) with a knob at the value:
    /// where a value sits in a scale, more than how full something is.
    Marker,
    /// Three quarters of a ring, filled clockwise, the value and its unit
    /// in its middle, the range's ends in the gap at the bottom, the name
    /// under it.
    Circular,
    /// A whole ring filled clockwise from the top, the share of the range
    /// in its middle: how full, how far along.
    Ring,
}

/// What colour a gauge's fill is.
#[derive(Debug, Clone, PartialEq)]
pub enum GaugeTint {
    /// The theme's accent.
    Accent,
    Color(Rgba),
    /// By how far along the range the value is: each colour from its
    /// fraction (0 to 1, in order) on, as a level turns amber, then red.
    Bands(Vec<(f32, Rgba)>),
    /// Colours along the range: a bar or a track shows all of them, a dial
    /// the colour at its value.
    Gradient(Vec<(f32, Rgba)>),
}

impl GaugeTint {
    /// Green, amber from 60 % of the range, red from 85 %: a level that
    /// should stay low.
    pub fn levels() -> GaugeTint {
        GaugeTint::Bands(vec![(0.0, GREEN), (0.6, AMBER), (0.85, RED)])
    }

    /// Green through amber to red along the range.
    pub fn heat() -> GaugeTint {
        GaugeTint::Gradient(vec![(0.0, GREEN), (0.5, AMBER), (1.0, RED)])
    }

    /// The colour at `f` (0 to 1) along the range; `None`: the accent.
    fn at(&self, f: f32) -> Option<Rgba> {
        match self {
            GaugeTint::Accent => None,
            GaugeTint::Color(c) => Some(*c),
            GaugeTint::Bands(b) => b.iter().rev().find(|(from, _)| f >= *from).or(b.first()).map(|(_, c)| *c),
            GaugeTint::Gradient(stops) => {
                let (first, last) = (stops.first()?, stops.last()?);
                if f <= first.0 {
                    return Some(first.1);
                }
                let i = stops.iter().position(|(at, _)| *at >= f).unwrap_or(stops.len() - 1);
                let ((a, ca), (b, cb)) = (stops[i.saturating_sub(1)], stops[i]);
                if f >= last.0 || b <= a {
                    return Some(if f >= last.0 { last.1 } else { cb });
                }
                let t = (f - a) / (b - a);
                Some(std::array::from_fn(|k| ca[k] + (cb[k] - ca[k]) * t))
            }
        }
    }
}

// Green 600, amber 600, red 600: legible on light and dark surfaces.
const GREEN: Rgba = [0.086, 0.639, 0.290, 1.0];
const AMBER: Rgba = [0.851, 0.467, 0.024, 1.0];
const RED: Rgba = [0.863, 0.149, 0.149, 1.0];

/// How a gauge looks, besides its kind. `GaugeStyle::default()` is the
/// accent, the range shown, the value in whole numbers or tenths, gliding.
#[derive(Debug, Clone, PartialEq)]
pub struct GaugeStyle {
    pub tint: GaugeTint,
    /// A dial's or a ring's width and height, in pixels (a bar takes its
    /// parent's width).
    pub size: f32,
    /// After the value ("%", " °C"); a ring shows a share, always "%".
    pub unit: String,
    /// Places after the point; `None`: none for a whole number, else one.
    pub decimals: Option<usize>,
    /// The range's ends shown (not a ring's, which is always 0 to 100 %).
    pub show_range: bool,
    /// A change glides to the new value (a quarter second, eased), unless
    /// motion is reduced.
    pub animate: bool,
    /// A dial's marks inside its ring: this many steps between its ends
    /// (0: none).
    pub ticks: u32,
    /// A dial points at its value with a needle, the number under it.
    pub needle: bool,
}

impl Default for GaugeStyle {
    fn default() -> Self {
        GaugeStyle { tint: GaugeTint::Accent, size: 96.0, unit: String::new(), decimals: None, show_range: true, animate: true, ticks: 0, needle: false }
    }
}

/// A value in a range, shown. Made by [`gauge`] or [`gauge_styled`].
pub struct Gauge {
    pub node: NodeId,
    /// What moves with the value (a bar's fill, a dial's arc).
    pub fill: NodeId,
    parts: Parts,
    value: f32,
}

/// A gauge's moving parts and how they show a value: what a glide steps.
#[derive(Debug, Clone)]
pub(crate) struct Parts {
    kind: GaugeKind,
    range: (f32, f32),
    style: GaugeStyle,
    fill: NodeId,
    /// A bar's gradient, as wide as the track inside its fill.
    inner: Option<NodeId>,
    knob: Option<NodeId>,
    /// A dial's needle.
    needle: Option<NodeId>,
    /// The value's text.
    readout: Option<NodeId>,
}

/// A glide from one value to another, stepped by [`Ui::step`].
#[derive(Debug, Clone)]
pub(crate) struct Glide {
    parts: Parts,
    from: f32,
    to: f32,
    elapsed: f32,
}

/// How long a glide takes, in seconds.
const GLIDE: f32 = 0.25;

/// Where the dial's ring starts and how far round it goes (radians,
/// clockwise from straight up).
const DIAL_START: f32 = -0.75 * std::f32::consts::PI;
const DIAL_SWEEP: f32 = 1.5 * std::f32::consts::PI;

/// An arc of a ring of radius 40 (in a 100 × 100 view box) from `start`,
/// `fraction` of `sweep` round (radians, clockwise from straight up). A
/// whole turn is drawn as two halves (one arc cannot close on itself).
fn arc(start: f32, sweep: f32, fraction: f32) -> Vec<String> {
    let f = fraction.clamp(0.0, 1.0);
    let point = |a: f32| (50.0 + 40.0 * a.sin(), 50.0 - 40.0 * a.cos());
    let (x0, y0) = point(start);
    let turn = sweep * f;
    if turn <= 1e-4 {
        return vec![];
    }
    if turn >= std::f32::consts::TAU - 1e-3 {
        let (xm, ym) = point(start + std::f32::consts::PI);
        return vec![format!("M{x0:.3} {y0:.3} A40 40 0 0 1 {xm:.3} {ym:.3} A40 40 0 0 1 {x0:.3} {y0:.3}")];
    }
    let (x1, y1) = point(start + turn);
    let large = u8::from(turn > std::f32::consts::PI);
    vec![format!("M{x0:.3} {y0:.3} A40 40 0 {large} 1 {x1:.3} {y1:.3}")]
}

/// The dial's arc from its start to `fraction` round.
#[cfg(test)]
fn dial_arc(fraction: f32) -> String {
    arc(DIAL_START, DIAL_SWEEP, fraction).pop().unwrap_or_default()
}

/// A gauge for a value in `range` under `parent`, named `label`, in the
/// default [`GaugeStyle`] (it starts at the range's low end; give it a
/// value with [`Gauge::set`]).
pub fn gauge(ui: &mut Ui, parent: NodeId, label: &str, range: RangeInclusive<f32>, kind: GaugeKind) -> Gauge {
    gauge_styled(ui, parent, label, range, kind, GaugeStyle::default())
}

/// [`gauge`] in `style`.
pub fn gauge_styled(ui: &mut Ui, parent: NodeId, label: &str, range: RangeInclusive<f32>, kind: GaugeKind, style: GaugeStyle) -> Gauge {
    let (lo, hi) = (*range.start(), *range.end());
    let text = |ui: &mut Ui, at: NodeId, s: &str, css: &str, token: Token| {
        let mut st = style! {};
        for d in css.split(';').filter(|d| !d.trim().is_empty()) {
            if let Some((n, v)) = d.split_once(':') {
                crate::css::apply(&mut st, n.trim(), v.trim());
            }
        }
        let b = ui.add_box(at, st);
        ui.token_style(b, &[("color", token)]);
        ui.add_text(b, s);
        ui.semantics_mut(b).hidden = true;
        b
    };
    let number = |v: f32| format_value(v, style.decimals, "");
    let mut parts = Parts { kind, range: (lo, hi), style: style.clone(), fill: node_placeholder(ui), inner: None, knob: None, needle: None, readout: None };
    let node = match kind {
        GaugeKind::Linear | GaugeKind::Marker => {
            let node = ui.add_box(parent, style! { display: flex; flex-direction: column; gap: 4px; min-width: 80px; });
            let head = ui.add_box(node, style! { display: flex; justify-content: space-between; align-items: flex-end; gap: 8px; });
            text(ui, head, label, "font-size: 12px", Token::SecondaryText);
            let value = text(ui, head, "", "font-size: 13px; font-weight: bold", Token::Text);
            parts.readout = Some(ui.children(value)[0]);
            if kind == GaugeKind::Linear {
                let track = ui.add_box(node, style! { position: relative; height: 8px; border-radius: 4px; overflow: hidden; });
                ui.token_style(track, &[("background", Token::Separator)]);
                let fill = ui.add_box(track, style! { position: absolute; left: 0px; top: 0px; height: 8px; width: 0%; border-radius: 4px; overflow: hidden; });
                parts.fill = fill;
                match &style.tint {
                    GaugeTint::Gradient(stops) => {
                        // The whole range's colours, the fill showing its share.
                        let inner = ui.add_box(fill, style! { position: absolute; left: 0px; top: 0px; height: 8px; width: 100%; });
                        ui.update_style(inner, |s| {
                            s.background_gradient = stops.clone();
                            s.gradient_shape = crate::GradientShape::Linear(90.0);
                        });
                        parts.inner = Some(inner);
                    }
                    GaugeTint::Accent => ui.token_style(fill, &[("background", Token::Accent)]),
                    _ => {}
                }
            } else {
                let track = ui.add_box(node, style! { position: relative; height: 16px; });
                let bar = ui.add_box(track, style! { position: absolute; left: 0px; right: 0px; top: 5px; height: 6px; border-radius: 3px; });
                match &style.tint {
                    GaugeTint::Gradient(stops) | GaugeTint::Bands(stops) => {
                        let stops = if let GaugeTint::Bands(b) = &style.tint { hard_stops(b) } else { stops.clone() };
                        ui.update_style(bar, |s| {
                            s.background_gradient = stops;
                            s.gradient_shape = crate::GradientShape::Linear(90.0);
                        });
                    }
                    _ => ui.token_style(bar, &[("background", Token::Separator)]),
                }
                let knob = ui.add_box(track, style! { position: absolute; top: 0px; left: 0%; margin-left: -8px; width: 10px; height: 10px; border: 3px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.35); });
                ui.token_style(knob, &[("background", Token::Surface)]);
                if style.tint == GaugeTint::Accent {
                    ui.token_style(knob, &[("border-color", Token::Accent)]);
                }
                parts.fill = knob;
                parts.knob = Some(knob);
            }
            if style.show_range {
                let ends = ui.add_box(node, style! { display: flex; justify-content: space-between; });
                text(ui, ends, &number(lo), "font-size: 11px", Token::SecondaryText);
                text(ui, ends, &number(hi), "font-size: 11px", Token::SecondaryText);
            }
            node
        }
        GaugeKind::Circular | GaugeKind::Ring => {
            let size = style.size.max(40.0);
            let node = ui.add_box(parent, style! { display: flex; flex-direction: column; align-items: center; gap: 2px; flex-shrink: 0; });
            let dial = ui.add_box(node, style! { position: relative; flex-shrink: 0; });
            ui.update_style(dial, |s| {
                s.width = Length::Px(size);
                s.height = Length::Px(size);
            });
            let whole = || {
                let mut s = style! { position: absolute; left: 0px; top: 0px; };
                s.width = Length::Px(size);
                s.height = Length::Px(size);
                s
            };
            let (start, sweep) = if kind == GaugeKind::Ring { (0.0, std::f32::consts::TAU) } else { (DIAL_START, DIAL_SWEEP) };
            let track = ui.add_path(dial, whole(), arc(start, sweep, 1.0), [0.0, 0.0, 100.0, 100.0], Some(10.0), false);
            ui.token_style(track, &[("color", Token::Separator)]);
            let fill = ui.add_path(dial, whole(), vec![], [0.0, 0.0, 100.0, 100.0], Some(10.0), false);
            if style.tint == GaugeTint::Accent {
                ui.token_style(fill, &[("color", Token::Accent)]);
            }
            parts.fill = fill;
            // Marks inside the ring, from end to end.
            if kind == GaugeKind::Circular && style.ticks > 0 {
                let mut d = String::new();
                for k in 0..=style.ticks {
                    let a = DIAL_START + DIAL_SWEEP * k as f32 / style.ticks as f32;
                    let (s, c) = a.sin_cos();
                    d.push_str(&format!("M{:.2} {:.2} L{:.2} {:.2} ", 50.0 + 27.0 * s, 50.0 - 27.0 * c, 50.0 + 31.0 * s, 50.0 - 31.0 * c));
                }
                let marks = ui.add_path(dial, whole(), vec![d], [0.0, 0.0, 100.0, 100.0], Some(1.6), false);
                ui.token_style(marks, &[("color", Token::SecondaryText)]);
                ui.semantics_mut(marks).hidden = true;
            }
            // A needle from the middle, pointing up until turned, and its pivot.
            if kind == GaugeKind::Circular && style.needle {
                let needle = ui.add_path(dial, whole(), vec!["M50 54 L50 20".to_owned()], [0.0, 0.0, 100.0, 100.0], Some(3.0), false);
                let pivot = ui.add_path(dial, whole(), vec!["M47 50 A3 3 0 0 1 53 50 A3 3 0 0 1 47 50".to_owned()], [0.0, 0.0, 100.0, 100.0], Some(6.0), false);
                for n in [needle, pivot] {
                    ui.token_style(n, &[("color", Token::Text)]);
                    ui.semantics_mut(n).hidden = true;
                }
                parts.needle = Some(needle);
            }
            // The value in the middle, the unit smaller under it; with a
            // needle, on a line of its own under the dial (the needle has
            // the middle).
            let needle = parts.needle.is_some();
            let mid = if needle {
                ui.add_box(node, style! { display: flex; align-items: flex-end; gap: 3px; margin-top: 2px; })
            } else {
                let m = ui.add_box(dial, whole());
                ui.update_style(m, |s| {
                    s.display = crate::Display::Flex;
                    s.align_items = crate::AlignItems::Center;
                    s.flex_direction = crate::FlexDirection::Column;
                    s.justify_content = crate::JustifyContent::Center;
                });
                m
            };
            let big = if needle { 0.15 } else { 0.24 };
            let value = text(ui, mid, "", &format!("font-size: {:.0}px; font-weight: bold; line-height: 1", (size * big).round()), Token::Text);
            parts.readout = Some(ui.children(value)[0]);
            let unit = if kind == GaugeKind::Ring { "%" } else { style.unit.trim() };
            if !unit.is_empty() {
                text(ui, mid, unit, &format!("font-size: {:.0}px", (size * if needle { 0.09 } else { 0.12 }).round().max(9.0)), Token::SecondaryText);
            }
            for n in [track, fill, mid] {
                ui.semantics_mut(n).hidden = true;
            }
            // The range's ends in the gap, under where the ring starts and stops.
            if kind == GaugeKind::Circular && style.show_range {
                let foot = 50.0 + 40.0 * (-DIAL_START).cos().abs();
                for (v, x) in [(lo, 50.0 + 40.0 * DIAL_START.sin()), (hi, 50.0 - 40.0 * DIAL_START.sin())] {
                    let css = format!("position: absolute; top: {:.1}px; left: {:.1}px; width: 40px; text-align: center; font-size: 11px", size * (foot + 4.0) / 100.0, size * x / 100.0 - 20.0);
                    text(ui, dial, &number(v), &css, Token::SecondaryText);
                }
            }
            text(ui, node, label, "font-size: 12px", Token::SecondaryText);
            node
        }
    };
    let mut sem = Semantics::new(Role::ProgressBar).label(label);
    sem.range = Some(crate::a11y::RangeValue { min: lo.min(hi), max: hi.max(lo), now: lo });
    ui.set_semantics(node, sem);
    show(ui, &parts, lo);
    let fill = parts.fill;
    let mut g = Gauge { node, fill, parts, value: f32::NAN };
    g.value = lo;
    g.say(ui);
    g
}

/// A stand-in until the fill is made (the root: never a gauge's part).
fn node_placeholder(ui: &Ui) -> NodeId {
    ui.root()
}

/// Bands as a gradient with hard edges, for a marker's track.
fn hard_stops(bands: &[(f32, Rgba)]) -> Vec<(f32, Rgba)> {
    let mut out = vec![];
    for (i, &(from, c)) in bands.iter().enumerate() {
        let to = bands.get(i + 1).map_or(1.0, |b| b.0);
        out.push((from.clamp(0.0, 1.0), c));
        out.push((to.clamp(0.0, 1.0), c));
    }
    out
}

/// `v` with `decimals` places (`None`: none for a whole number, else one),
/// then `unit`.
fn format_value(v: f32, decimals: Option<usize>, unit: &str) -> String {
    let n = match decimals {
        Some(d) => format!("{v:.d$}"),
        None if v.fract() == 0.0 => format!("{v}"),
        None => format!("{v:.1}"),
    };
    format!("{n}{unit}")
}

/// `parts` showing `v`: the fill, the knob, the colour, the number.
fn show(ui: &mut Ui, parts: &Parts, v: f32) {
    let (lo, hi) = parts.range;
    let f = if hi != lo { ((v - lo) / (hi - lo)).clamp(0.0, 1.0) } else { 0.0 };
    let colour = parts.style.tint.at(f);
    match parts.kind {
        GaugeKind::Linear => {
            ui.update_style(parts.fill, |s| {
                s.width = Length::Percent(f * 100.0);
                if let (Some(c), GaugeTint::Color(_) | GaugeTint::Bands(_)) = (colour, &parts.style.tint) {
                    s.background = Some(c);
                }
            });
            // The gradient as wide as the track: the fill's width over its share.
            if let Some(inner) = parts.inner {
                ui.update_style(inner, |s| s.width = Length::Percent(100.0 / f.max(1e-3)));
            }
        }
        GaugeKind::Marker => {
            let knob = parts.knob.unwrap_or(parts.fill);
            ui.update_style(knob, |s| {
                s.left = Length::Percent(f * 100.0);
                if let Some(c) = colour {
                    s.border_color = c;
                }
            });
        }
        GaugeKind::Circular | GaugeKind::Ring => {
            let (start, sweep) = if parts.kind == GaugeKind::Ring { (0.0, std::f32::consts::TAU) } else { (DIAL_START, DIAL_SWEEP) };
            ui.set_path_data(parts.fill, arc(start, sweep, f));
            if let Some(n) = parts.needle {
                ui.set_path_transform(n, crate::transform::Transformation { rotate: start + sweep * f, ..Default::default() });
            }
            if let Some(c) = colour {
                ui.update_style(parts.fill, |s| {
                    s.color = c;
                    s.color_set = true;
                });
            }
        }
    }
    if let Some(r) = parts.readout {
        let shown = match parts.kind {
            GaugeKind::Ring => format!("{}", (f * 100.0).round()),
            GaugeKind::Circular => format_value(v, parts.style.decimals, ""),
            _ => format_value(v, parts.style.decimals, &parts.style.unit),
        };
        ui.set_text(r, shown);
    }
}

impl Gauge {
    /// Shows `value` (held to the range): at once, or gliding there (see
    /// [`GaugeStyle::animate`]; [`Ui::step`] moves it).
    pub fn set(&mut self, ui: &mut Ui, value: f32) {
        let (lo, hi) = self.parts.range;
        let v = value.clamp(lo.min(hi), hi.max(lo));
        if v == self.value || v.is_nan() {
            return;
        }
        // From where it shows now: a glide under way, or the last value.
        let from = ui.env.glides.iter().find(|g| g.parts.fill == self.parts.fill).map_or(self.value, Glide::now);
        ui.env.glides.retain(|g| g.parts.fill != self.parts.fill);
        self.value = v;
        if self.parts.style.animate && !ui.reduce_motion() && from.is_finite() {
            ui.env.glides.push(Glide { parts: self.parts.clone(), from, to: v, elapsed: 0.0 });
            ui.env.repaint.set(true);
        } else {
            show(ui, &self.parts, v);
        }
        self.say(ui);
    }

    pub fn value(&self) -> f32 {
        self.value
    }

    /// The value, to assistive technology: at once, not as it glides.
    fn say(&self, ui: &mut Ui) {
        let unit = if self.parts.kind == GaugeKind::Ring { "" } else { self.parts.style.unit.as_str() };
        let s = ui.semantics_mut(self.node);
        s.value = Some(format_value(self.value, self.parts.style.decimals, unit));
        if let Some(r) = &mut s.range {
            r.now = self.value;
        }
    }
}

impl Glide {
    /// Where it is now, eased out.
    fn now(&self) -> f32 {
        let t = (self.elapsed / GLIDE).clamp(0.0, 1.0);
        let e = 1.0 - (1.0 - t).powi(3);
        self.from + (self.to - self.from) * e
    }
}

/// Moves the gauges gliding by `dt` seconds; returns whether any still are.
pub(crate) fn step_glides(ui: &mut Ui, dt: f32) -> bool {
    if ui.env.glides.is_empty() {
        return false;
    }
    let mut glides = std::mem::take(&mut ui.env.glides);
    glides.retain(|g| ui.contains(g.parts.fill));
    for g in &mut glides {
        g.elapsed += dt.max(0.0);
        show(ui, &g.parts, g.now());
    }
    glides.retain(|g| g.elapsed < GLIDE);
    // Glides started meanwhile run from the next step.
    glides.append(&mut ui.env.glides);
    ui.env.glides = glides;
    !ui.env.glides.is_empty()
}

/// What a chart of `values` says to assistive technology.
fn summary(label: &str, points: &[(String, f32)]) -> String {
    let list: Vec<String> = points.iter().map(|(n, v)| if n.is_empty() { format!("{v}") } else { format!("{n} {v}") }).collect();
    let max = points.iter().max_by(|a, b| a.1.total_cmp(&b.1));
    let min = points.iter().min_by(|a, b| a.1.total_cmp(&b.1));
    match (max, min) {
        (Some(hi), Some(lo)) => format!("{label}: {}. Highest {}, lowest {}.", list.join(", "), hi.1, lo.1),
        _ => format!("{label}: no values."),
    }
}

/// Bars for `data` (names and values, in order) `height` pixels tall at
/// the highest gridline, under `parent`, named `label`: each bar's value
/// on it and its name under it, light gridlines at round values with the
/// values beside them. The bars grow in when first shown (at once, under
/// reduced motion). To assistive technology a list, named for its values
/// in order and their highest and lowest, its items each a bar's name and
/// value, to step through.
pub fn bar_chart(ui: &mut Ui, parent: NodeId, label: &str, data: &[(&str, f32)], height: f32) -> NodeId {
    let points: Vec<(String, f32)> = data.iter().map(|(n, v)| (n.to_string(), *v)).collect();
    let node = ui.add_box(parent, style! { display: flex; flex-direction: column; gap: 3px; });
    ui.set_semantics(node, Semantics::new(Role::List).label(summary(label, &points)));
    let (step, top) = nice_scale(data.iter().map(|d| d.1).fold(0.0f32, f32::max));
    // Room for the axis's values on the left, and a value over the tallest bar.
    let axis = 28.0;
    let plot = ui.add_box(node, style! { position: relative; });
    ui.update_style(plot, |s| s.height = Length::Px(height + 14.0));
    let mut v = 0.0;
    while v <= top + step * 0.01 {
        let y = 14.0 + height * (1.0 - v / top);
        let line = ui.add_box(plot, style! { position: absolute; right: 0px; height: 1px; });
        ui.update_style(line, |s| {
            s.left = Length::Px(axis);
            s.top = Length::Px(y);
        });
        ui.token_style(line, &[("background", Token::Separator)]);
        let tick = ui.add_box(plot, style! { position: absolute; left: 0px; width: 24px; font-size: 10px; text-align: right; });
        ui.update_style(tick, |s| s.top = Length::Px(y - 6.0));
        ui.token_style(tick, &[("color", Token::SecondaryText)]);
        ui.add_text(tick, format_value(v, None, ""));
        for n in [line, tick] {
            ui.semantics_mut(n).hidden = true;
        }
        v += step;
    }
    let bars = ui.add_box(plot, style! { position: absolute; top: 0px; right: 0px; bottom: 0px; display: flex; align-items: flex-end; gap: 6px; });
    ui.update_style(bars, |s| s.left = Length::Px(axis + 4.0));
    let names = ui.add_box(node, style! { display: flex; gap: 6px; });
    ui.update_style(names, |s| s.padding.left = axis + 4.0);
    ui.semantics_mut(names).hidden = true;
    for (i, &(name, v)) in data.iter().enumerate() {
        let col = ui.add_box(bars, style! { display: flex; flex-direction: column; align-items: center; gap: 2px; flex-grow: 1; flex-basis: 0px; min-width: 14px; });
        ui.set_semantics(col, Semantics::new(Role::ListItem).label(format!("{name}, {}", format_value(v, None, ""))));
        let shown = text_box(ui, col, &format_value(v, None, ""), "font-size: 10px", Token::SecondaryText);
        ui.semantics_mut(shown).hidden = true;
        let bar = ui.add_box(col, style! { width: 100%; border-radius: 3px; });
        ui.token_style(bar, &[("background", Token::Accent)]);
        let h = (v.max(0.0) / top * height).max(1.0);
        ui.update_style(bar, |s| s.height = Length::Px(h));
        // Grown in, one after another.
        let mut grow = crate::animation::Animation::new(crate::animation::Property::Height, crate::animation::Value::Number(1.0), crate::animation::Value::Number(h), 0.45);
        grow.easing = threers_math::easing::Easing::CubicOut;
        grow.delay = 0.04 * i as f32;
        ui.animate(bar, grow);
        let n = ui.add_box(names, style! { flex-grow: 1; flex-basis: 0px; min-width: 14px; font-size: 10px; text-align: center; });
        ui.token_style(n, &[("color", Token::SecondaryText)]);
        ui.add_text(n, name);
    }
    node
}

/// A round step for gridlines under `top` (1, 2, 2.5 or 5 times a power of
/// ten, three or four of them), and the scale's top: `top` rounded up to a
/// step.
fn nice_scale(top: f32) -> (f32, f32) {
    let top = if top.is_finite() && top > 0.0 { top } else { 1.0 };
    let raw = top / 3.0;
    let pow = 10f32.powf(raw.log10().floor());
    let step = [1.0, 2.0, 2.5, 5.0, 10.0].into_iter().map(|m| m * pow).find(|&s| s >= raw).unwrap_or(10.0 * pow);
    (step, (top / step).ceil().max(1.0) * step)
}

/// A box of text under `parent`, styled by `css`, coloured by `token`.
fn text_box(ui: &mut Ui, parent: NodeId, s: &str, css: &str, token: Token) -> NodeId {
    let mut st = style! {};
    for d in css.split(';').filter(|d| !d.trim().is_empty()) {
        if let Some((n, v)) = d.split_once(':') {
            crate::css::apply(&mut st, n.trim(), v.trim());
        }
    }
    let b = ui.add_box(parent, st);
    ui.token_style(b, &[("color", token)]);
    ui.add_text(b, s);
    b
}

/// A layer over a box `w × h`, from its corner.
fn over(w: f32, h: f32) -> crate::Style {
    let mut s = style! { position: absolute; left: 0px; top: 0px; };
    s.width = Length::Px(w);
    s.height = Length::Px(h);
    s
}

/// A line through `values` (left to right, evenly spaced), the area under
/// it shaded, `size` pixels; small and bare, a sparkline.
pub fn line_chart(ui: &mut Ui, parent: NodeId, label: &str, values: &[f32], size: (f32, f32)) -> NodeId {
    let (w, h) = size;
    let node = ui.add_box(parent, style! { position: relative; flex-shrink: 0; });
    ui.update_style(node, |s| {
        s.width = Length::Px(w);
        s.height = Length::Px(h);
    });
    let points: Vec<(String, f32)> = values.iter().map(|v| (String::new(), *v)).collect();
    ui.set_semantics(node, Semantics::new(Role::Image).label(summary(label, &points)));
    if values.is_empty() {
        return node;
    }
    let (lo, hi) = values.iter().fold((f32::MAX, f32::MIN), |(a, b), &v| (a.min(v), b.max(v)));
    let span = (hi - lo).max(1e-6);
    // In a view box the size of the chart, a little in from its edges.
    let pad = 2.0;
    let at = |i: usize, v: f32| {
        let x = if values.len() > 1 { pad + (w - 2.0 * pad) * i as f32 / (values.len() - 1) as f32 } else { w / 2.0 };
        (x, pad + (h - 2.0 * pad) * (1.0 - (v - lo) / span))
    };
    let pts: Vec<(f32, f32)> = values.iter().enumerate().map(|(i, &v)| at(i, v)).collect();
    let line = smooth_line(&pts);
    let (x_end, y_end) = pts[pts.len() - 1];
    let area = format!("{line} L{x_end:.2} {h:.2} L{:.2} {h:.2} Z", pts[0].0);
    let shade = ui.add_path(node, over(w, h), vec![area], [0.0, 0.0, w, h], None, true);
    ui.token_style(shade, &[("color", Token::Accent)]);
    ui.update_style(shade, |s| s.opacity = 0.18);
    let stroke = ui.add_path(node, over(w, h), vec![line], [0.0, 0.0, w, h], Some(1.5), false);
    ui.token_style(stroke, &[("color", Token::Accent)]);
    // The latest value, marked: a disc (a ring as wide as it is round).
    let r = 1.5;
    let dot = format!("M{:.2} {y_end:.2} A{r} {r} 0 0 1 {:.2} {y_end:.2} A{r} {r} 0 0 1 {:.2} {y_end:.2}", x_end - r, x_end + r, x_end - r);
    let end = ui.add_path(node, over(w, h), vec![dot], [0.0, 0.0, w, h], Some(2.0 * r), false);
    ui.token_style(end, &[("color", Token::Accent)]);
    ui.semantics_mut(end).hidden = true;
    for n in [shade, stroke] {
        ui.semantics_mut(n).hidden = true;
    }
    node
}

/// SVG path data for a curve through `points` (in order along x): cubic
/// pieces with slopes that keep it monotone between points (Fritsch and
/// Carlson), so it never swings past a peak or below a trough the data do
/// not have. One point is a point; two, a line.
fn smooth_line(points: &[(f32, f32)]) -> String {
    let n = points.len();
    let mut d = format!("M{:.2} {:.2}", points[0].0, points[0].1);
    if n < 3 {
        for p in &points[1..] {
            d.push_str(&format!(" L{:.2} {:.2}", p.0, p.1));
        }
        return d;
    }
    // Each piece's slope, then each point's tangent.
    let slope: Vec<f32> = points.windows(2).map(|w| (w[1].1 - w[0].1) / (w[1].0 - w[0].0).max(1e-6)).collect();
    let mut m = vec![0.0f32; n];
    m[0] = slope[0];
    m[n - 1] = slope[n - 2];
    for i in 1..n - 1 {
        m[i] = if slope[i - 1] * slope[i] <= 0.0 { 0.0 } else { (slope[i - 1] + slope[i]) / 2.0 };
    }
    for i in 0..n - 1 {
        if slope[i] == 0.0 {
            m[i] = 0.0;
            m[i + 1] = 0.0;
            continue;
        }
        let (a, b) = (m[i] / slope[i], m[i + 1] / slope[i]);
        let t = a * a + b * b;
        if t > 9.0 {
            let k = 3.0 / t.sqrt();
            m[i] = k * a * slope[i];
            m[i + 1] = k * b * slope[i];
        }
    }
    for i in 0..n - 1 {
        let ((x0, y0), (x1, y1)) = (points[i], points[i + 1]);
        let dx = (x1 - x0) / 3.0;
        d.push_str(&format!(" C{:.2} {:.2} {:.2} {:.2} {x1:.2} {y1:.2}", x0 + dx, y0 + m[i] * dx, x1 - dx, y1 - m[i + 1] * dx));
    }
    d
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::FixedWidthText;

    #[test]
    fn a_smooth_line_stays_within_its_data() {
        let pts: Vec<(f32, f32)> = [11.0, 10.5, 10.0, 11.5, 14.0, 17.5, 19.0, 18.0, 16.5, 14.0, 12.5, 12.0].iter().enumerate().map(|(i, &v)| (i as f32 * 20.0, 100.0 - v * 4.0)).collect();
        let d = smooth_line(&pts);
        assert_eq!(d.matches('C').count(), 11);
        // Every control point between its piece's ends in y: no overshoot.
        let nums: Vec<f32> = d.split(|c: char| c.is_ascii_alphabetic() || c == ' ').filter(|t| !t.is_empty()).map(|t| t.parse().unwrap()).collect();
        for (i, piece) in nums[2..].chunks(6).enumerate() {
            let (ya, yb) = (pts[i].1, pts[i + 1].1);
            let (lo, hi) = (ya.min(yb) - 1e-3, ya.max(yb) + 1e-3);
            assert!((lo..=hi).contains(&piece[1]) && (lo..=hi).contains(&piece[3]), "piece {i}: {piece:?} outside {ya}..{yb}");
        }
    }

    #[test]
    fn a_gauge_fills_by_its_value_and_says_it() {
        let mut ui = Ui::new();
        let root = ui.root();
        let mut bar = gauge(&mut ui, root, "Done", 0.0..=10.0, GaugeKind::Linear);
        bar.set(&mut ui, 2.5);
        bar.set(&mut ui, 99.0);
        assert_eq!(bar.value(), 10.0, "held to the range");
        bar.set(&mut ui, 2.5);
        ui.layout(400.0, 100.0, &mut FixedWidthText(7.0));
        // Gliding: there once stepped past the glide.
        ui.step(0.1);
        ui.step(0.2);
        ui.layout(400.0, 100.0, &mut FixedWidthText(7.0));
        let track = ui.rect(ui.parent(bar.fill).unwrap());
        let fill = ui.rect(bar.fill);
        assert!((fill.width - track.width * 0.25).abs() < 0.5, "{fill:?} of {track:?}");
        assert_eq!(ui.semantics_mut(bar.node).value.as_deref(), Some("2.5"));
        // The dial's arc: three quarters of the ring is the whole range.
        assert!(dial_arc(1.0).contains(" 1 1 "), "the long way round: {}", dial_arc(1.0));
        assert!(dial_arc(0.25).contains(" 0 1 "));
        // A whole ring is two halves, closing where it began; none is nothing.
        let full = arc(0.0, std::f32::consts::TAU, 1.0);
        assert_eq!(full[0].matches('A').count(), 2, "{full:?}");
        assert!(full[0].starts_with("M50.000 10.000") && full[0].ends_with("50.000 10.000"));
        assert!(arc(0.0, std::f32::consts::TAU, 0.0).is_empty());
    }

    #[test]
    fn every_kind_and_tint_shows_its_value() {
        let mut ui = Ui::new();
        let root = ui.root();
        let kinds = [GaugeKind::Linear, GaugeKind::Marker, GaugeKind::Circular, GaugeKind::Ring];
        let tints = [GaugeTint::Accent, GaugeTint::Color(RED), GaugeTint::levels(), GaugeTint::heat()];
        let mut gauges = vec![];
        for kind in kinds {
            for tint in &tints {
                let st = GaugeStyle { tint: tint.clone(), unit: "%".into(), animate: false, ..Default::default() };
                let b = ui.add_box(root, style! { width: 200px; });
                gauges.push((kind, tint.clone(), gauge_styled(&mut ui, b, "Load", 0.0..=200.0, kind, st)));
            }
        }
        for (_, _, g) in &mut gauges {
            g.set(&mut ui, 180.0);
        }
        ui.layout(900.0, 2000.0, &mut FixedWidthText(7.0));
        let text_of = |ui: &Ui, n: NodeId| match ui.kind(n) {
            crate::NodeKind::Text(t) => t.clone(),
            _ => String::new(),
        };
        for (kind, tint, g) in &gauges {
            let readout = text_of(&ui, g.parts.readout.unwrap());
            let want = match kind {
                GaugeKind::Ring => "90",
                GaugeKind::Circular => "180",
                _ => "180%",
            };
            assert_eq!(readout, want, "{kind:?} {tint:?}");
            assert_eq!(ui.semantics(g.node).unwrap().value.as_deref(), Some(if *kind == GaugeKind::Ring { "180" } else { "180%" }));
            // Past 85 %: a level is red.
            if *tint == GaugeTint::levels() {
                let c = match kind {
                    GaugeKind::Linear => ui.style(g.fill).background.unwrap(),
                    GaugeKind::Marker => ui.style(g.fill).border_color,
                    _ => ui.style(g.fill).color,
                };
                assert_eq!(c, RED, "{kind:?}");
            }
        }
        // A marker's knob at 90 % along; a bar's gradient as wide as its track.
        let marker = &gauges[4].2;
        assert_eq!(ui.style(marker.fill).left, Length::Percent(90.0));
        let bar = &gauges[3].2;
        let inner = bar.parts.inner.unwrap();
        let track = ui.rect(ui.parent(bar.fill).unwrap());
        assert!((ui.rect(inner).width - track.width).abs() < 0.5, "{:?} {track:?}", ui.rect(inner));
        // A needle points at the value: 90 % of the way round the dial.
        let st = GaugeStyle { needle: true, ticks: 10, animate: false, ..Default::default() };
        let mut speedo = gauge_styled(&mut ui, root, "Speed", 0.0..=200.0, GaugeKind::Circular, st);
        speedo.set(&mut ui, 180.0);
        let turn = ui.path_transform(speedo.parts.needle.unwrap()).rotate;
        assert!((turn - (DIAL_START + DIAL_SWEEP * 0.9)).abs() < 1e-5, "{turn}");
        // The heat at 90 % is between amber and red.
        let c = GaugeTint::heat().at(0.9).unwrap();
        assert!(c[0] > AMBER[0] && c[1] < AMBER[1] && c[1] > RED[1], "{c:?}");
    }

    #[test]
    fn a_change_glides_unless_motion_is_reduced() {
        let mut ui = Ui::new();
        let root = ui.root();
        let mut g = gauge(&mut ui, root, "Level", 0.0..=100.0, GaugeKind::Circular);
        let readout = g.parts.readout.unwrap();
        let shown = |ui: &Ui| match ui.kind(readout) {
            crate::NodeKind::Text(t) => t.parse::<f32>().unwrap(),
            _ => f32::NAN,
        };
        g.set(&mut ui, 80.0);
        assert!(ui.needs_frame());
        assert_eq!(g.value(), 80.0, "the value at once");
        assert_eq!(ui.semantics(g.node).unwrap().value.as_deref(), Some("80"), "and what is said");
        ui.step(0.08);
        let mid = shown(&ui);
        assert!(mid > 0.0 && mid < 80.0, "on its way: {mid}");
        // Turned back midway: from where it shows, not from where it was.
        g.set(&mut ui, 10.0);
        ui.step(0.01);
        assert!((shown(&ui) - mid).abs() < 15.0);
        ui.step(0.3);
        assert_eq!(shown(&ui), 10.0);
        assert!(ui.env.glides.is_empty());
        ui.set_reduce_motion(true);
        g.set(&mut ui, 55.0);
        assert_eq!(shown(&ui), 55.0, "at once");
    }

    #[test]
    fn charts_say_their_values_highest_and_lowest() {
        let mut ui = Ui::new();
        let root = ui.root();
        let bars = bar_chart(&mut ui, root, "Steps", &[("Mon", 4.0), ("Tue", 8.0)], 80.0);
        let spark = line_chart(&mut ui, root, "Price", &[3.0, 1.0, 4.0, 1.5], (120.0, 30.0));
        ui.layout(400.0, 300.0, &mut FixedWidthText(7.0));
        let label = |n| ui.accessibility_tree().into_iter().find(|a| a.id == n).unwrap().name;
        assert_eq!(label(bars), "Steps: Mon 4, Tue 8. Highest 8, lowest 4.");
        assert_eq!(label(spark), "Price: 3, 1, 4, 1.5. Highest 4, lowest 1.");
        // The bars grown in, the taller twice the shorter, on a scale
        // rounded up to 10 (gridlines at 0, 5 and 10).
        for _ in 0..40 {
            ui.step(0.05);
        }
        ui.layout(400.0, 300.0, &mut FixedWidthText(7.0));
        let items: Vec<_> = ui.accessibility_tree().into_iter().filter(|a| a.role == Role::ListItem).collect();
        assert_eq!(items.iter().map(|a| a.name.as_str()).collect::<Vec<_>>(), ["Mon, 4", "Tue, 8"]);
        let h = |c: NodeId| ui.rect(ui.children(c)[1]).height;
        assert_eq!((h(items[0].id), h(items[1].id)), (32.0, 64.0));
        assert_eq!(nice_scale(8.0), (5.0, 10.0));
        assert_eq!(nice_scale(11.0), (5.0, 15.0));
        assert_eq!(nice_scale(0.0), (0.5, 1.0));
    }
}
