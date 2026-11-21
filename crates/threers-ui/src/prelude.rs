//! One import for what nearly every UI uses: the tree and its styles, input
//! (pointers, touches, keys) and bindings, a text field's editing, meaning
//! for assistive technology, the theme's colours, painters, and the gauges
//! and charts.
//!
//! ```
//! use threers_ui::prelude::*;
//!
//! #[derive(Default)]
//! struct State { level: f32 }
//!
//! let mut ui = Ui::new();
//! let mut input: Input<State> = Input::new();
//! let root = ui.root();
//! let card = ui.add_box(root, style! { display: flex; flex-direction: column; gap: 8px; padding: 12px; });
//! ui.token_style(card, &[("background", Token::Surface)]);
//! let mut level = gauge(&mut ui, card, "Level", 0.0..=100.0, GaugeKind::Circular);
//! level.set(&mut ui, 40.0);
//! let save = ui.add_box(card, style! { padding: 8px; });
//! ui.set_semantics(save, Semantics::new(Role::Button).label("Save"));
//! input.on(save, Phase::Bubble, |s: &mut State, _, e, _| {
//!     if matches!(e, Event::Click { .. }) {
//!         s.level += 10.0;
//!     }
//! });
//! ui.layout(400.0, 300.0, &mut FixedWidthText(7.0));
//! assert!(!ui.hover_regions().is_empty());
//! ```
//!
//! Its names are common words (`Event`, `Phase`, `Token`); a glob import
//! makes them yours. Everything else stays at its module's path
//! (`threers_ui::menu::…`, `threers_ui::table::…`).

pub use crate::binding::Binding;
pub use crate::charts::{
    bar_chart, gauge, gauge_styled, line_chart, Gauge, GaugeKind, GaugeStyle, GaugeTint,
};
pub use crate::environment::Token;
pub use crate::input::{Event, Input, Phase, PointerButton, TouchPhase};
pub use crate::spatial::{HoverRegion, SPATIAL_MIN_TARGET};
pub use crate::states::{Appearance, Theme};
pub use crate::theme::theme_picker;
pub use crate::{binding, style};
pub use crate::{
    AlignItems, ArcShape, Display, Edges, FixedWidthText, FlexDirection, FlexWrap, FontWeight,
    JustifyContent, Length, NodeId, NodeKind, Painter, Position, Rect, Rgba, Role, Semantics,
    Style, TextAlign, TextMeasure, TextStyle, Ui,
};
pub use crate::{FieldKey, KeyModifiers, TextField};
