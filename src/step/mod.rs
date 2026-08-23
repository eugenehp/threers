//! STEP — ISO 10303 AP203/AP214 advanced B-rep exchange.
//!
//! The point of the B-rep stack is that a solid keeps its surfaces: a bore is a
//! cylinder with a parameter range, not a band of triangles. STEP is the format
//! that can carry that, and it is what every CAD system reads. Exporting a
//! tessellation to STL loses the model; exporting the [`Body`](crate::brep::Body)
//! to STEP does not.
//!
//! Two layers, kept apart:
//!
//! * [`crate::step::part21`] — the exchange *syntax*. A file is a header and a list of
//!   `#id = NAME(args);` instances, with no opinion about what they mean. Nearly
//!   every real-world STEP problem is lexical, and this layer is testable
//!   against a round-trip without constructing any geometry.
//! * [`mod@crate::step::export`] / [`mod@crate::step::import`] — the AP203 *mapping*, between those instances and
//!   `Body`.
//!
//! ## What is carried
//!
//! Planes, cylinders, spheres, cones and tori map to their STEP entities
//! directly and exactly, and so do the circles and lines that bound them. A
//! NURBS surface maps to `B_SPLINE_SURFACE_WITH_KNOTS`, rational or not.
//!
//! ## What is not
//!
//! The same rule as everywhere else in this stack: decline rather than
//! approximate. A surface with no STEP counterpart is reported, not silently
//! tessellated into one — see [`crate::step::export::Unsupported`]. Import likewise reports
//! the entities it skipped rather than quietly returning a partial solid.
//!
//! A *face* can be unstateable too, which is less obvious than a surface being
//! so and was twice found the hard way. AP203 gives a face its region through
//! the curves around it, and a closed curve on a closed surface bounds two
//! regions without saying which: one circle on a sphere is the edge of the cap
//! and equally of everything else. Where the face's own trim says something its
//! edges do not, that is `AmbiguousRegion`; where its boundary runs along the
//! surface's seam, `SeamBoundary`. Both are declined rather than written,
//! because a file this crate would misread is worse than no file.
//!
//! What that rule buys is checked rather than argued: a corpus of booleans is
//! written, read back and re-measured, and each must either return the size it
//! left or have the export say what it could not carry. Both times the rule was
//! broken, `skipped` was empty at *both* ends and a solid came back with the
//! wrong volume — once none at all — so nothing downstream could have known.

pub mod export;
pub mod import;
pub mod part21;

pub use export::{export, ExportReport, Unsupported};
pub use import::{import, ImportError, ImportReport};
pub use part21::{parse, Entity, ParseError, StepFile, Value};
