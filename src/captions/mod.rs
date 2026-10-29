//! Subtitles and captions — a timed-text track you can burn into exported
//! video, ship as a sidecar `.srt`/`.vtt`, mux as a soft subtitle track, or
//! draw on screen over a live render.
//!
//! Pure Rust with no dependencies and no OS/process/filesystem use, so it
//! builds and runs on `wasm32` as well as native (mirroring the `codec`
//! module, behind the `native-codec` feature).
//!
//! # Text is not a sequence of codepoints
//!
//! Between a caption's string and a row of glyphs there are three
//! transformations, all in [`order::visual`], and each omission was visible:
//! `"Noe\u{0308}l"` drew a floating diaeresis ([`compose`]), Hebrew read
//! backwards ([`bidi`]), and a bracket inside a right-to-left phrase opened the
//! wrong way (**L4**), and Arabic letters stood apart instead of joining
//! ([`crate::loaders::shape`]), and the dots a shaped Arabic letter decomposes into
//! sat at its origin rather than on it ([`crate::loaders::mark`]).
//!
//! ```
//! use threers::captions::{CaptionTrack, CaptionPainter};
//!
//! let track = CaptionTrack::parse_srt(
//!     "1\n00:00:00,000 --> 00:00:02,500\nHello, world!\n",
//! ).unwrap();
//! assert_eq!(track.active_at(1.0).len(), 1);
//!
//! // Burn the active cue into an RGBA frame.
//! let (w, h) = (320u32, 180u32);
//! let mut frame = vec![0u8; (w * h * 4) as usize];
//! let mut painter = CaptionPainter::new();
//! painter.burn_in(&mut frame, w, h, &track, 1.0);
//! ```
//!
//! | Piece | Role |
//! |-------|------|
//! | [`Cue`] / [`CaptionTrack`] | timed-text model, `active_at` lookup |
//! | [`srt`] / [`vtt`] | SubRip and WebVTT parse + serialize |
//! | [`CaptionStyle`] | type size, colors, outline, box, placement |
//! | [`CaptionFont`] | built-in 8×8 bitmap face, or a TrueType face |
//! | [`CaptionPainter`] | wraps text and composites it into RGBA pixels |
//! | [`CaptionOverlay`] | caches a frame-sized overlay for on-screen drawing |
//!
//! Video export wires these in through `VideoOptions::captions` (the `video`
//! feature); the wgpu
//! renderer draws them through
//! [`Renderer::draw_caption_overlay`](crate::renderer::Renderer::draw_caption_overlay).

pub mod bidi;
pub mod compose;
mod compose_tables;
pub mod font;
pub mod glyphs;
pub mod layout;
pub mod order;
pub mod overlay;
pub mod raster;

pub use font::{CaptionFont, GlyphBitmap, Hinting, SdfGlyph, SdfGlyphs};
pub use layout::{LaidOutText, LineBox};
pub use overlay::CaptionOverlay;
pub use raster::CaptionPainter;

// The caption data model (tracks, cues, SRT/WebVTT) lives in `threers-codec`,
// whose muxers write caption tracks; the rendering side stays here.
pub use threers_codec::captions::*;
