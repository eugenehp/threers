//! TrueType fonts as FreeType renders and HarfBuzz shapes them, written
//! from the OpenType and TrueType specifications: the sfnt tables, glyph
//! outlines scaled and hinted to 26.6 pixels (with a TrueType bytecode
//! interpreter), text shaping, and the choice of a face for a family name.

pub mod fixed;
pub mod fontconfig;
pub mod glyph;
pub mod hint;
pub mod sfnt;
pub mod shape;
pub mod size;
mod trig;

pub use glyph::{Glyph, Orientation, Outline};
pub use hint::Hinter;
pub use sfnt::{Error, Face};
pub use shape::{shape, GlyphPosition};
pub use size::{Size, SizeMetrics};
