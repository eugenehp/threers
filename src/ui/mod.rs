//! A small retained UI: boxes styled with a subset of CSS, laid out with
//! flexbox, painted onto a 2D canvas, hit-tested for input.
//!
//! It exists for apps whose interface is specified in CSS terms — an HTML UI
//! being reproduced, a design given in pixels and flex rules — and that want
//! the same layout natively and in the browser without a DOM. The tree is
//! plain data ([`Ui`]); [`Ui::layout`] computes every box with the same rules a
//! browser would for the supported properties; [`Ui::paint`] draws through a
//! [`Painter`], which can be threers' own [`Overlay`](crate::overlay::Overlay)
//! or, in a browser, the page's 2D canvas (so text and paths rasterize exactly
//! as the browser's own would).
//!
//! Painted pixels mean nothing to a screen reader, so nodes also carry
//! [`Semantics`] — role, name, value, state — which the built-in
//! [`controls`] set for themselves. [`Ui::accessibility_tree`] is what
//! assistive technology reads; in a browser, `a11y_dom::DomAccessibility` (wasm32)
//! puts it in the page. [`Ui::focus_order`] gives keyboard navigation and
//! [`Ui::contrast_issues`] checks text against WCAG's contrast minimum. See
//! [`a11y`].
//!
//! ```
//! use threers::ui::{FlexDirection, Length, Style, Ui};
//!
//! let mut ui = Ui::new();
//! let root = ui.root();
//! ui.set_style(root, Style::flex(FlexDirection::Column));
//! let bar = ui.add_box(root, Style { height: Length::Px(20.0), background: Some([0.13, 0.13, 0.13, 1.0]), ..Style::default() });
//! let rest = ui.add_box(root, Style { flex_grow: 1.0, ..Style::default() });
//! ui.layout(1200.0, 800.0, &mut threers::ui::FixedWidthText(7.0));
//! assert_eq!(ui.rect(bar).height, 20.0);
//! assert_eq!(ui.rect(rest).y, 20.0);
//! assert_eq!(ui.rect(rest).height, 780.0);
//! ```

pub mod a11y;
#[cfg(target_arch = "wasm32")]
pub mod a11y_dom;
#[cfg(target_arch = "wasm32")]
pub mod canvas;
pub mod code;
pub mod controls;
pub mod css;
mod layout;
mod paint;
mod style;
mod text_area;
mod text_field;
mod tree;

pub use a11y::{contrast_ratio, AccessNode, ContrastIssue, Role, Semantics};
pub use layout::{layout_unit, FixedWidthText, TextMeasure};
pub use paint::{circle_path, parse_circle, OverlayPainter, Painter, TextStyle};
pub use style::{
    AlignItems, BoxSizing, Display, Edges, FlexDirection, FontWeight, JustifyContent, Length, Position, Rgba, Style,
};
pub use text_area::TextArea;
pub use text_field::{FieldKey, KeyModifiers, TextField};
pub use tree::{NodeId, NodeKind, Rect, Ui};
