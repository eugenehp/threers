//! Plain indicators: no shader, no glow, the theme's colours.

use super::*;

// Plain indicators: no shader, no glow, the theme's colours, the shapes a
// platform's own controls have (SwiftUI's `ProgressView`, GPUI's
// indicators). Every painter draws them alike.

/// A plain indicator's shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum PlainKind {
    /// A thin rounded bar: filled to the value, or a segment sliding along
    /// it while busy.
    #[default]
    Linear,
    /// A ring: an arc from the top to the value, or a quarter arc turning
    /// while busy.
    Circular,
    /// The activity indicator: spokes round a centre, each fading in turn
    /// while busy; with a value, as many lit as it is far along.
    Spokes,
}

/// How a plain indicator looks. Start from [`PlainStyle::new`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PlainStyle {
    pub kind: PlainKind,
    /// A bar's height, a ring's or the spokes' width, in UI pixels.
    pub size: f32,
    /// A ring's stroke in UI pixels.
    pub stroke: f32,
    /// The fill's colour; `None` follows the theme (its accent, or for
    /// spokes its secondary text).
    pub tint: Option<Rgba>,
}

impl PlainStyle {
    /// The platform's sizes: a 4 pixel bar, a 20 pixel ring or spinner.
    pub fn new(kind: PlainKind) -> Self {
        let (size, stroke) = match kind {
            PlainKind::Linear => (4.0, 0.0),
            PlainKind::Circular => (20.0, 2.5),
            PlainKind::Spokes => (20.0, 0.0),
        };
        Self {
            kind,
            size,
            stroke,
            tint: None,
        }
    }

    pub fn size(self, size: f32) -> Self {
        Self {
            size: size.max(2.0),
            ..self
        }
    }

    pub fn stroke(self, stroke: f32) -> Self {
        Self {
            stroke: stroke.max(0.5),
            ..self
        }
    }

    pub fn tint(self, tint: Rgba) -> Self {
        Self {
            tint: Some(tint),
            ..self
        }
    }
}

/// How many spokes the activity indicator has.
pub const SPOKES: usize = 8;

/// A plain indicator's nodes: [`set`](Self::set) moves it on.
#[derive(Debug, Clone)]
pub struct ProgressView {
    /// Its box, with its role and value for assistive technology.
    pub node: NodeId,
    /// The bar's fill, the ring's arc, or the spokes.
    pub(super) parts: Vec<NodeId>,
    pub style: PlainStyle,
    pub(super) label: Option<LabelNode>,
}

/// A plain indicator in `parent` showing `value` (0 to 1), or busy with
/// `None`. A linear one is as wide as its parent lets it be.
///
/// ```
/// use threers_ui::indicators::{progress_view, PlainKind, PlainStyle};
/// use threers_ui::{FixedWidthText, Ui};
///
/// let mut ui = Ui::new();
/// let root = ui.root();
/// let spinner = progress_view(&mut ui, root, "Loading", PlainStyle::new(PlainKind::Spokes), None);
/// ui.layout(200.0, 100.0, &mut FixedWidthText(7.0));
/// assert!(ui.needs_frame(), "it turns");
/// spinner.set(&mut ui, Some(0.5));
/// assert_eq!(ui.semantics(spinner.node).unwrap().value.as_deref(), Some("50%"));
/// ```
pub fn progress_view(
    ui: &mut Ui,
    parent: NodeId,
    label: &str,
    style: PlainStyle,
    value: Option<f32>,
) -> ProgressView {
    use crate::environment::Token;
    let node = ui.add_box(parent, style! { position: relative; flex-shrink: 0; });
    ui.set_semantics(node, Semantics::new(Role::ProgressBar).label(label));
    let tint = |ui: &mut Ui, n: NodeId, prop: &str, token: Token| match style.tint {
        Some(c) => ui.update_style(n, |s| match prop {
            "background" => s.background = Some(c),
            _ => {
                s.color = c;
                s.color_set = true;
            }
        }),
        None => ui.token_style(n, &[(prop, token)]),
    };
    let whole = || style! { position: absolute; left: 0px; top: 0px; width: 100%; height: 100%; };
    let parts = match style.kind {
        PlainKind::Linear => {
            let r = style.size * 0.5;
            ui.update_style(node, |s| {
                s.height = Length::Px(style.size);
                s.align_self = Some(crate::AlignItems::Stretch);
                s.border_radius = Length::Px(r);
            });
            // The track clips the sliding fill (and not a label outside).
            let track = ui.add_box(node, whole());
            ui.update_style(track, |s| {
                s.border_radius = Length::Px(r);
                s.clip = true;
            });
            crate::pickers::outline_rail(ui, track);
            ui.semantics_mut(track).hidden = true;
            let fill = ui.add_box(
                track,
                style! { position: absolute; top: 0px; height: 100%; },
            );
            ui.update_style(fill, |s| s.border_radius = Length::Px(r));
            tint(ui, fill, "background", Token::AccentText);
            vec![fill]
        }
        PlainKind::Circular => {
            ui.update_style(node, |s| {
                s.width = Length::Px(style.size);
                s.height = Length::Px(style.size);
            });
            let stroke = 100.0 * style.stroke / style.size;
            let r = 50.0 - stroke * 0.5;
            let (edge, middle) = crate::pickers::ring_strokes(style.stroke);
            for (width, token) in [(Some(edge), Token::Track), (middle, Token::Surface)] {
                let Some(width) = width else { continue };
                let rail = ui.add_path(
                    node,
                    whole(),
                    arc(r, 0.0, 1.0),
                    [0.0, 0.0, 100.0, 100.0],
                    Some(100.0 * width / style.size),
                    false,
                );
                ui.token_style(rail, &[("color", token)]);
                ui.semantics_mut(rail).hidden = true;
            }
            let fill = ui.add_path(
                node,
                whole(),
                vec![],
                [0.0, 0.0, 100.0, 100.0],
                Some(stroke),
                false,
            );
            tint(ui, fill, "color", Token::AccentText);
            vec![fill]
        }
        PlainKind::Spokes => {
            ui.update_style(node, |s| {
                s.width = Length::Px(style.size);
                s.height = Length::Px(style.size);
            });
            (0..SPOKES)
                .map(|i| {
                    let a = i as f32 / SPOKES as f32 * std::f32::consts::TAU;
                    let (sn, cs) = a.sin_cos();
                    let d = format!(
                        "M{:.2} {:.2} L{:.2} {:.2}",
                        50.0 + 24.0 * sn,
                        50.0 - 24.0 * cs,
                        50.0 + 42.0 * sn,
                        50.0 - 42.0 * cs
                    );
                    let spoke = ui.add_path(
                        node,
                        whole(),
                        vec![d],
                        [0.0, 0.0, 100.0, 100.0],
                        Some(10.0),
                        false,
                    );
                    tint(ui, spoke, "color", Token::SecondaryText);
                    spoke
                })
                .collect()
        }
    };
    for &n in &parts {
        ui.semantics_mut(n).hidden = true;
    }
    let view = ProgressView {
        node,
        parts,
        style,
        label: None,
    };
    view.set(ui, value);
    view
}

impl ProgressView {
    /// With `label` on it (see [`Label`]): a ring's percentage in its
    /// middle, a bar's beside it.
    pub fn labelled(mut self, ui: &mut Ui, label: Label) -> Self {
        self.set_label(ui, Some(label));
        self
    }

    /// Puts `label` on it, or takes its label off.
    pub fn set_label(&mut self, ui: &mut Ui, label: Option<Label>) {
        let host = match self.style.kind {
            PlainKind::Linear => Host::Bar {
                height: self.style.size,
            },
            _ => Host::Round {
                size: self.style.size,
            },
        };
        let value = self.current(ui);
        match label {
            Some(l) => {
                self.label = Some(place_label(
                    ui,
                    self.node,
                    host,
                    l,
                    self.label.as_ref(),
                    value,
                ))
            }
            None => {
                if let Some(o) = self.label.take() {
                    ui.remove(o.node);
                }
            }
        }
    }

    /// What its label says, if it has one.
    pub fn label_text(&self, ui: &Ui) -> Option<String> {
        self.label.as_ref().and_then(|l| match ui.kind(l.text) {
            crate::NodeKind::Text(t) => Some(t.clone()),
            _ => None,
        })
    }

    /// Its value as last set (`None`: busy).
    pub fn current(&self, ui: &Ui) -> Option<f32> {
        ui.semantics(self.node)
            .and_then(|s| s.range.map(|r| r.now / 100.0))
    }

    /// How far along, 0 to 1; `None` while it is not known (busy).
    pub fn set(&self, ui: &mut Ui, value: Option<f32>) {
        let v = value.map(|v| v.clamp(0.0, 1.0));
        let sem = ui.semantics_mut(self.node);
        sem.range = v.map(|v| crate::a11y::RangeValue {
            min: 0.0,
            max: 100.0,
            now: v * 100.0,
        });
        sem.value = v.map(|v| format!("{}%", (v * 100.0).round()));
        if let Some(l) = &self.label {
            ui.set_text(l.text, l.label.render(v));
        }
        let st = self.style;
        match st.kind {
            PlainKind::Linear => {
                let fill = self.parts[0];
                match v {
                    Some(v) => {
                        stop_sliding(ui, fill);
                        ui.update_style(fill, |s| s.width = Length::Percent(v * 100.0));
                    }
                    None => slide(ui, self.node, fill, 0.3, 1.3),
                }
            }
            PlainKind::Circular => {
                let fill = self.parts[0];
                let stroke = 100.0 * st.stroke / st.size;
                let r = 50.0 - stroke * 0.5;
                match v {
                    Some(v) => {
                        ui.stop_animations(fill, Some(Property::Rotate));
                        ui.set_path_transform(fill, Default::default());
                        ui.set_path_data(fill, arc(r, 0.0, v));
                    }
                    None => {
                        ui.set_path_data(fill, arc(r, 0.0, 0.28));
                        if !ui.is_animating(fill) {
                            let tau = std::f32::consts::TAU;
                            ui.animate(
                                fill,
                                Animation::new(
                                    Property::Rotate,
                                    Value::Number(0.0),
                                    Value::Number(tau),
                                    0.9,
                                )
                                .repeat(Repeat::Forever),
                            );
                        }
                    }
                }
            }
            PlainKind::Spokes => {
                let n = self.parts.len();
                for (i, &spoke) in self.parts.iter().enumerate() {
                    match v {
                        Some(v) => {
                            ui.stop_animations(spoke, Some(Property::Opacity));
                            let lit = (i as f32) < (v * n as f32).round();
                            ui.update_style(spoke, |s| s.opacity = if lit { 1.0 } else { 0.22 });
                        }
                        None => {
                            // Each fades from full as the one before it
                            // lights: the light runs round clockwise.
                            if !ui.is_animating(spoke) {
                                let period = 0.8;
                                ui.update_style(spoke, |s| s.opacity = 0.22);
                                ui.animate(
                                    spoke,
                                    Animation::new(
                                        Property::Opacity,
                                        Value::Number(1.0),
                                        Value::Number(0.22),
                                        period,
                                    )
                                    .delay(period * i as f32 / n as f32)
                                    .repeat(Repeat::Forever),
                                );
                            }
                        }
                    }
                }
            }
        }
    }
}
