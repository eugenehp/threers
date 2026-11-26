//! OpenType and TrueType fonts as FreeType renders and HarfBuzz shapes
//! them, written from the OpenType, TrueType and CFF specifications: the
//! sfnt tables, glyph outlines scaled to 26.6 pixels (TrueType's hinted
//! by a bytecode interpreter; CFF's unhinted), text shaping horizontally
//! and vertically (variable fonts' instances too), and the choice of a
//! face for a family name.

pub mod color;
pub mod fixed;
pub mod fontconfig;
pub mod glyph;
pub mod hint;
pub mod raster;
pub mod sfnt;
pub mod shape;
pub mod size;
mod cff;
mod trig;
pub mod var;

pub use glyph::{Glyph, Orientation, Outline};
pub use hint::Hinter;
pub use sfnt::{Error, Face};
pub use shape::{shape, shape_vertical, vertical_forms, GlyphPosition};
pub use size::{Size, SizeMetrics};
