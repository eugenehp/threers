//! A small retained UI: boxes styled with a subset of CSS, laid out with
//! flexbox, painted onto a 2D canvas, hit-tested for input.
//!
//! It exists for apps whose interface is specified in CSS terms — an HTML UI
//! being reproduced, a design given in pixels and flex rules — and that want
//! the same layout natively and in the browser without a DOM. The tree is
//! plain data ([`Ui`]); [`Ui::layout`] computes every box with the same rules a
//! browser would for the supported properties; [`Ui::paint`] draws through a
//! [`Painter`], which can be threers' CPU overlay (`threers::ui::OverlayPainter`)
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
//! Beyond styling and layout:
//!
//! - **Input** ([`input`]): events go to the node under the pointer, or to
//!   the focused node, in a capture phase and then a bubble phase. Focus,
//!   clicks, keyboard activation, Tab order, wheel scrolling and pointer
//!   capture follow the DOM's defaults. Keys become actions of your own
//!   types through a [`keymap`], scoped by key contexts on the focus path.
//! - **Scenes** ([`scene`]): [`Ui::scene`] records a frame's painting as
//!   data, for a renderer to draw however it likes. threers' `UiPass` draws
//!   one on the GPU in a single instanced call.
//! - **Caching**: [`Ui::cache_layout`] marks a subtree whose layout is reused
//!   while nothing in it changes. Each node tracks whether it has changed.
//! - **Ids**: a [`NodeId`] is generational. A removed node's slot is reused,
//!   and its old id stops naming anything rather than naming the newcomer.
//! - **Lists and layers:** [`virtual_list`] builds only the rows in view.
//!   [`layers`] defers popups over everything and anchors them inside the
//!   window, and hit-testing uses a spatial index.
//! - **Text and pictures:** [`rich_text`] styles runs within a text node.
//!   [`image`] shows decoded images and loads them in the background.
//!   Surfaces show GPU textures. Styles have `box-shadow`, underlines
//!   (straight or wavy) and `cursor`.
//! - **Input:** [`input`] adds touch gestures, IME composition ([`ime`]),
//!   drag and drop, tooltips, and HTML's `tabindex` order.
//! - **Speed and state:** [`text_cache`] keeps text measurements across
//!   frames. [`state::ElementState`] keeps values across rebuilt UIs.
//!   [`responsive`] has container queries, and [`anim`] has springs.
//! - **Tooling:** [`inspector`] finds the node at a point and the line that
//!   made it. [`debug_overlay`] paints frame statistics, and [`testing`]
//!   makes random trees.
//! - **Interaction styles and motion:** [`states`] gives nodes hover,
//!   active, focus and group-hover styles. [`animation`] runs property
//!   animations and transitions, and `needs_frame` says when to draw.
//! - **Layout, scrolling and text:** CSS Grid. [`scrolling`] has scroll
//!   handles and draggable scrollbars. [`text_hit`] selects text with the
//!   pointer.
//! - **Keymaps and focus:** [`keymap`] has context expressions and JSON
//!   files. `Input::set_modal` traps focus in a dialog, and
//!   `on_click_outside` closes popups.
//! - **Native accessibility:** `a11y_accesskit`, with the `accesskit`
//!   feature.
//! - **Tailwind:** [`tailwind`] reads utility classes (`"flex gap-2 p-4
//!   bg-blue-500 hover:bg-blue-600 md:flex-row hover:rotate-y-12"`) into
//!   styles, stacking state and breakpoint styles, and 3D transforms, with
//!   Tailwind's theme, shadcn/ui's, or your own ([`Ui::tw`], [`tw`]).
//!
//! ```
//! use threers_ui::{FlexDirection, Length, Style, Ui};
//!
//! let mut ui = Ui::new();
//! let root = ui.root();
//! ui.set_style(root, Style::flex(FlexDirection::Column));
//! let bar = ui.add_box(root, Style { height: Length::Px(20.0), background: Some([0.13, 0.13, 0.13, 1.0]), ..Style::default() });
//! let rest = ui.add_box(root, Style { flex_grow: 1.0, ..Style::default() });
//! ui.layout(1200.0, 800.0, &mut threers_ui::FixedWidthText(7.0));
//! assert_eq!(ui.rect(bar).height, 20.0);
//! assert_eq!(ui.rect(rest).y, 20.0);
//! assert_eq!(ui.rect(rest).height, 780.0);
//! ```

pub mod a11y;
#[cfg(feature = "accesskit")]
pub mod a11y_accesskit;
pub mod a11y_core;
#[cfg(target_arch = "wasm32")]
pub mod a11y_dom;
pub mod anchors;
pub mod anim;
pub mod animation;
pub mod binding;
pub mod border_beam;
pub mod button;
pub mod candle;
#[cfg(target_arch = "wasm32")]
pub mod canvas;
pub mod charts;
pub mod code;
pub mod components;
pub mod content_transition;
pub mod controls;
pub mod css;
pub mod damage;
pub mod debug_overlay;
pub mod depth;
pub mod environment;
pub mod fits;
pub mod focus_state;
pub mod globals;
pub mod i18n;
pub mod image;
pub mod ime;
pub mod indicators;
pub mod input;
pub mod inspector;
pub mod jobs;
pub mod keyframes;
pub mod keymap;
pub mod layers;
mod layout;
pub mod legibility;
pub mod matched;
pub mod material;
pub mod menu;
pub mod model;
pub mod model3d;
pub mod motion;
pub mod native;
pub mod nav_stack;
pub mod navigation;
pub mod notifications;
mod paint;
pub mod palette;
pub mod patch;
pub mod pickers;
pub mod preferences;
pub mod prelude;
pub mod preview;
pub mod prompt;
pub mod region;
pub mod responsive;
pub mod rich_text;
pub mod scene;
pub mod scroll_snap;
pub mod scroll_transition;
pub mod scrolling;
pub mod search;
pub mod shader;
pub mod shader_lib;
pub mod sheet;
pub mod solid;
pub mod spatial;
pub mod split;
pub mod springs;
pub mod state;
pub mod states;
pub mod storage;
mod style;
pub mod svg;
pub mod table;
pub mod tailwind;
pub mod testing;
mod text_area;
pub mod text_cache;
#[cfg(feature = "text-touch")]
pub mod text_editing;
mod text_field;
pub mod text_shortcuts;
pub mod text_hit;
#[cfg(feature = "text-touch")]
pub mod text_touch;
pub mod theme;
pub mod transform;
pub mod transitions;
mod tree;
pub mod undo;
pub mod view;
pub mod virtual_keyboard;
pub mod virtual_list;

pub use a11y::{contrast_ratio, AccessNode, ContrastIssue, Role, Semantics};
pub use layout::{layout_unit, FixedWidthText, TextMeasure};
pub use paint::{
    circle_path, parse_circle, underline_position, ArcShape, Painter, TextStyle, WAVE_AMPLITUDE,
    WAVE_PERIOD,
};
pub use style::{
    gradient_at, AlignContent, AlignItems, BlendMode, BoxShadow, BoxSizing, ClipShape, Cursor,
    Display, Edges, FlexDirection, FlexWrap, FontWeight, GradientShape, GridPlacement,
    JustifyContent, Length, ObjectFit, Position, Rgba, Scrollbar, ScrollbarVisibility, Style,
    TextAlign, Track, TrackSize, UserSelect,
};
pub use tailwind::tw;
pub use text_area::TextArea;
pub use text_field::{FieldKey, KeyModifiers, TextField};
pub use tree::{NodeId, NodeKind, Rect, Ui};
