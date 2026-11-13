//! Text on an indicator: how far along it is, or anything else.

use super::*;

/// Where an indicator's [`Label`] sits.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum LabelAlign {
    /// In the middle: of a ring, or along a bar.
    #[default]
    Center,
    /// Inside, at the start (a bar's left end, in left-to-right text).
    Start,
    /// Inside, at the end.
    End,
    /// Outside, over it.
    Above,
    /// Outside, under it.
    Below,
    /// Outside, after its end (beside a bar).
    After,
}

/// Text on an indicator: how far along it is, or anything else, and how it
/// looks. Build one and hand it to [`Indicator::labelled`] or
/// [`ProgressView::labelled`]:
///
/// ```
/// use threers_ui::indicators::{indicator, IndicatorStyle, Label, LabelAlign, Shape};
/// use threers_ui::Ui;
///
/// let mut ui = Ui::new();
/// let root = ui.root();
/// let ring = indicator(&mut ui, root, "Upload", IndicatorStyle::new(Shape::Ring), Some(0.42))
///     .labelled(&mut ui, Label::percent().size(20.0).bold().font("monospace"));
/// assert_eq!(ring.label_text(&ui).as_deref(), Some("42%"));
/// let bar = indicator(&mut ui, root, "Disk", IndicatorStyle::new(Shape::Bar), Some(0.25))
///     .labelled(&mut ui, Label::template("{value} of {max} GB").max(512.0).align(LabelAlign::Above));
/// assert_eq!(bar.label_text(&ui).as_deref(), Some("128 of 512 GB"));
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct Label {
    /// The text, with `{percent}` (0 to 100), `{value}` (0 to
    /// [`max`](Self::max)) and `{max}` filled in.
    pub template: String,
    /// What it says while the indicator is busy (no value).
    pub busy: String,
    /// What `{value}` runs up to.
    pub max: f32,
    /// Decimal places for `{percent}` and `{value}`.
    pub decimals: usize,
    /// Its size in UI pixels; `None` fits the indicator (a fifth of a
    /// ring, most of a bar's height, 12 outside one).
    pub size: Option<f32>,
    /// Its colour; `None` follows the theme's text (give a glowing
    /// indicator on a dark panel white, say).
    pub color: Option<Rgba>,
    pub weight: crate::FontWeight,
    pub italic: bool,
    /// A font family (`monospace` keeps digits from jumping about as they
    /// change; `serif`; a face by name); empty for the UI's own.
    pub font: String,
    pub align: LabelAlign,
}

impl Label {
    /// Text from a template: `{percent}`, `{value}` and `{max}` filled in.
    pub fn template(template: impl Into<String>) -> Self {
        Self { template: template.into(), busy: String::new(), max: 100.0, decimals: 0, size: None, color: None, weight: crate::FontWeight::Bold, italic: false, font: String::new(), align: LabelAlign::Center }
    }

    /// How far along: `42%`.
    pub fn percent() -> Self {
        Self::template("{percent}%")
    }

    /// Words that stay as they are.
    pub fn text(text: impl Into<String>) -> Self {
        let t = text.into();
        Self { busy: t.clone(), ..Self::template(t.replace('{', "{{").replace('}', "}}")) }
    }

    pub fn busy(mut self, text: impl Into<String>) -> Self {
        self.busy = text.into();
        self
    }

    pub fn max(mut self, max: f32) -> Self {
        self.max = max;
        self
    }

    pub fn decimals(mut self, decimals: usize) -> Self {
        self.decimals = decimals;
        self
    }

    pub fn size(mut self, size: f32) -> Self {
        self.size = Some(size.max(1.0));
        self
    }

    pub fn color(mut self, color: Rgba) -> Self {
        self.color = Some(color);
        self
    }

    pub fn weight(mut self, weight: crate::FontWeight) -> Self {
        self.weight = weight;
        self
    }

    pub fn bold(self) -> Self {
        self.weight(crate::FontWeight::Bold)
    }

    pub fn regular(self) -> Self {
        self.weight(crate::FontWeight::Normal)
    }

    pub fn italic(mut self) -> Self {
        self.italic = true;
        self
    }

    pub fn font(mut self, family: impl Into<String>) -> Self {
        self.font = family.into();
        self
    }

    pub fn align(mut self, align: LabelAlign) -> Self {
        self.align = align;
        self
    }

    /// What it says at `value` (`None`: busy).
    pub fn render(&self, value: Option<f32>) -> String {
        let Some(v) = value else { return self.busy.clone() };
        let d = self.decimals;
        self.template
            .replace("{percent}", &format!("{:.d$}", v * 100.0))
            .replace("{value}", &trim(format!("{:.d$}", v * self.max)))
            .replace("{max}", &trim(format!("{:.d$}", self.max)))
            .replace("{{", "{")
            .replace("}}", "}")
    }
}

/// `-0` as `0`.
pub(super) fn trim(s: String) -> String {
    if s.trim_start_matches('-').chars().all(|c| c == '0' || c == '.') { s.trim_start_matches('-').to_owned() } else { s }
}

/// A label's nodes on an indicator.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct LabelNode {
    pub(super) node: NodeId,
    pub(super) text: NodeId,
    pub(super) label: Label,
}

/// What an indicator is, for laying its label out.
#[derive(Clone, Copy)]
pub(super) enum Host {
    Bar { height: f32 },
    Round { size: f32 },
}

/// Puts `label` on the indicator box `host_node` (replacing `old`).
pub(super) fn place_label(ui: &mut Ui, host_node: NodeId, host: Host, label: Label, old: Option<&LabelNode>, value: Option<f32>) -> LabelNode {
    if let Some(o) = old {
        ui.remove(o.node);
    }
    let outside = matches!(label.align, LabelAlign::Above | LabelAlign::Below | LabelAlign::After);
    let size = label.size.unwrap_or(match (host, outside) {
        (_, true) => 12.0,
        (Host::Bar { height }, false) => (height * 0.72).max(6.0),
        (Host::Round { size }, false) => (size * 0.2).max(8.0),
    });
    let node = ui.add_box(host_node, style! { position: absolute; display: flex; align-items: center; white-space: nowrap; pointer-events: none; });
    ui.update_style(node, |s| {
        let full = Length::Percent(100.0);
        match label.align {
            LabelAlign::Center | LabelAlign::Start | LabelAlign::End => {
                s.left = Length::Px(0.0);
                s.top = Length::Px(0.0);
                s.width = full;
                s.height = full;
                s.justify_content = match label.align {
                    LabelAlign::Start => crate::JustifyContent::FlexStart,
                    LabelAlign::End => crate::JustifyContent::FlexEnd,
                    _ => crate::JustifyContent::Center,
                };
                let pad = if let Host::Bar { height } = host { (height * 0.5).max(4.0) } else { 0.0 };
                s.padding.left = pad;
                s.padding.right = pad;
                s.box_sizing = crate::BoxSizing::BorderBox;
            }
            LabelAlign::Above | LabelAlign::Below => {
                s.left = Length::Px(0.0);
                s.width = full;
                s.justify_content = crate::JustifyContent::Center;
                if label.align == LabelAlign::Above {
                    s.bottom = full;
                    s.margin.bottom = 4.0;
                } else {
                    s.top = full;
                    s.margin.top = 4.0;
                }
            }
            LabelAlign::After => {
                s.left = full;
                s.top = Length::Px(0.0);
                s.height = full;
                s.margin.left = 8.0;
            }
        }
        s.font_size = size;
        s.line_height = Some(size * 1.2);
        s.font_weight = label.weight;
        s.italic = label.italic;
        if !label.font.is_empty() {
            s.font_family = label.font.clone();
        }
    });
    match label.color {
        Some(c) => ui.update_style(node, |s| {
            s.color = c;
            s.color_set = true;
        }),
        None => ui.token_style(node, &[("color", crate::environment::Token::Text)]),
    }
    let text = ui.add_text(node, label.render(value));
    ui.semantics_mut(node).hidden = true;
    LabelNode { node, text, label }
}
