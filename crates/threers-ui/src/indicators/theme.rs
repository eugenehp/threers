//! Indicator styles set once on a part of the UI, for every indicator in it.

use super::*;

/// How the indicators in a part of the UI look, set once on a box for all
/// of them inside it (`ui.set_environment(panel, theme)`) and built with
/// [`indicator_themed`] and [`progress_view_themed`]. Each field left `None`
/// is the next theme out's, or the indicator's own default: themes nest,
/// an inner one changing only what it sets.
///
/// ```
/// use threers_ui::indicators::{indicator_themed, Depth, Fill, IndicatorTheme, Label, Palette, Shape};
/// use threers_ui::{style, Ui};
///
/// let mut ui = Ui::new();
/// let root = ui.root();
/// let panel = ui.add_box(root, style! { display: flex; });
/// ui.set_environment(panel, IndicatorTheme::new().palette(Palette::EMBER).depth(Depth::Solid).label(Label::percent()));
/// let card = ui.add_box(panel, style! { display: flex; });
/// ui.set_environment(card, IndicatorTheme::new().fill(Fill::Fire));
/// let ring = indicator_themed(&mut ui, card, "Render", Shape::Ring, Some(0.3));
/// assert_eq!((ring.style.palette, ring.style.depth, ring.style.fill), (Palette::EMBER, Depth::Solid, Fill::Fire));
/// assert_eq!(ring.label_text(&ui).as_deref(), Some("30%"));
/// ```
#[derive(Debug, Clone, Default, PartialEq)]
pub struct IndicatorTheme {
    pub palette: Option<Palette>,
    pub depth: Option<Depth>,
    pub fill: Option<Fill>,
    pub glow: Option<f32>,
    pub motion: Option<Motion>,
    pub shimmer: Option<bool>,
    /// A ring's stroke, as a share of its radius.
    pub thickness: Option<f32>,
    /// Text on each (a [`Label`]).
    pub label: Option<Label>,
    /// The plain views' colour (`None`: the theme's accent).
    pub tint: Option<Rgba>,
}

impl IndicatorTheme {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn palette(mut self, p: Palette) -> Self {
        self.palette = Some(p);
        self
    }
    pub fn depth(mut self, d: Depth) -> Self {
        self.depth = Some(d);
        self
    }
    pub fn fill(mut self, f: Fill) -> Self {
        self.fill = Some(f);
        self
    }
    pub fn glow(mut self, g: f32) -> Self {
        self.glow = Some(g.max(0.0));
        self
    }
    pub fn motion(mut self, m: Motion) -> Self {
        self.motion = Some(m);
        self
    }
    pub fn shimmer(mut self, on: bool) -> Self {
        self.shimmer = Some(on);
        self
    }
    pub fn thickness(mut self, t: f32) -> Self {
        self.thickness = Some(t);
        self
    }
    pub fn label(mut self, l: Label) -> Self {
        self.label = Some(l);
        self
    }
    pub fn tint(mut self, c: Rgba) -> Self {
        self.tint = Some(c);
        self
    }

    /// `inner` over this: its fields where it sets them.
    pub fn with(self, inner: IndicatorTheme) -> Self {
        Self {
            palette: inner.palette.or(self.palette),
            depth: inner.depth.or(self.depth),
            fill: inner.fill.or(self.fill),
            glow: inner.glow.or(self.glow),
            motion: inner.motion.or(self.motion),
            shimmer: inner.shimmer.or(self.shimmer),
            thickness: inner.thickness.or(self.thickness),
            label: inner.label.or(self.label),
            tint: inner.tint.or(self.tint),
        }
    }

    /// The theme in effect at `node`: every theme set on it and the boxes
    /// round it, the nearest over the rest.
    pub fn at(ui: &Ui, node: NodeId) -> IndicatorTheme {
        let mut chain: Vec<IndicatorTheme> = ui
            .ancestors(node)
            .filter_map(|n| ui.own_environment::<IndicatorTheme>(n))
            .collect();
        chain.reverse();
        chain.into_iter().fold(
            ui.global::<IndicatorTheme>()
                .map(|g| g.read().clone())
                .unwrap_or_default(),
            IndicatorTheme::with,
        )
    }

    /// `style` with this theme's settings over it.
    pub fn apply(&self, style: IndicatorStyle) -> IndicatorStyle {
        let mut s = style;
        if let Some(p) = self.palette {
            s.palette = p;
        }
        if let Some(d) = self.depth {
            s.depth = d;
        }
        if let Some(f) = self.fill {
            s.fill = f;
        }
        if let Some(g) = self.glow {
            s.glow = g;
        }
        if let Some(m) = self.motion {
            s.motion = m;
        }
        if let Some(on) = self.shimmer {
            s.shimmer = on;
        }
        if let Some(t) = self.thickness {
            s = s.thickness(t);
        }
        s
    }
}

/// An indicator of `shape` in `parent`, looking as the [`IndicatorTheme`]
/// in effect there says (and labelled, if it gives a label).
pub fn indicator_themed(
    ui: &mut Ui,
    parent: NodeId,
    label: &str,
    shape: Shape,
    value: Option<f32>,
) -> Indicator {
    let theme = IndicatorTheme::at(ui, parent);
    let ind = indicator(
        ui,
        parent,
        label,
        theme.apply(IndicatorStyle::new(shape)),
        value,
    );
    match theme.label {
        Some(l) => ind.labelled(ui, l),
        None => ind,
    }
}

/// A plain indicator of `kind` in `parent`, in the [`IndicatorTheme`]'s
/// tint (and label) in effect there.
pub fn progress_view_themed(
    ui: &mut Ui,
    parent: NodeId,
    label: &str,
    kind: PlainKind,
    value: Option<f32>,
) -> ProgressView {
    let theme = IndicatorTheme::at(ui, parent);
    let mut st = PlainStyle::new(kind);
    st.tint = theme.tint;
    let v = progress_view(ui, parent, label, st, value);
    match theme.label {
        Some(l) => v.labelled(ui, l),
        None => v,
    }
}
