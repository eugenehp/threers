//! Polygon operations in integer coordinates: booleans ([`Boolean64`]),
//! offsetting ([`PathOffset`]), and simplification, with an optional `z`
//! carried through every operation by a callback.
//!
//! Results are deterministic to the point and to the order of points, so
//! they can be compared with another implementation of the same algorithms
//! byte for byte.
//!
//! # Polygons that nearly touch
//!
//! Turns and collinearity are decided with `f64` cross products, and on
//! polygons that meet in T-junctions — a vertex within a unit or so of
//! another polygon's edge, which is what rounding a shared edge to the grid
//! leaves — those decisions can come out wrong. Twenty-eight convex pieces
//! that tile without overlapping, covering 0.402, have unioned to 0.592, and
//! the C++ library these algorithms follow gives the same answer. Which
//! inputs fail moves with the scale, so a different scale is not a fix. Grow
//! each such polygon by a unit or two on its own first ([`PathOffset`],
//! mitred), so that near-touches become proper overlaps.
//!
//! The algorithms follow Angus Johnson's polygon clipping and offsetting
//! library (Copyright Angus Johnson 2010-2025), used under the Boost
//! Software License 1.0, whose text is in `LICENSE-BSL-1.0` next to this
//! file.

mod boolean;
mod core;
mod offset;

pub use self::core::*;
pub use boolean::{Boolean64, ClipType, PathType, PolyPath64, ZCallback64};
pub use offset::{ellipse, EndType, JoinType, PathOffset};
