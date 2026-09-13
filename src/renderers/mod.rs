//! Alternate renderers. The primary `Renderer` lives in `crate::renderer` and
//! uses wgpu. This module exposes lightweight DOM-overlay renderers (CSS2D /
//! CSS3D) — their drawing surface is HTML, so on native they are data-only
//! descriptors that an integrator can pump into web_sys when compiled for
//! wasm — plus [`SvgRenderer`], which produces a complete SVG document and so
//! needs no host at all.

mod css2d;
mod css3d;
mod svg;

pub use css2d::Css2dRenderer;
pub use css3d::Css3dRenderer;
pub use svg::{svg_from_rgba, SvgOptions, SvgRenderer, SvgShading};
