//! Parametric curves. Mirror three.js's curve hierarchy: `Curve` base +
//! 2D and 3D concretes + `CurvePath`/`Path`/`Shape`.

mod bezier;
mod catmull_rom;
mod curve;
mod curve_path;
pub mod earcut;
mod ellipse;
mod line_curve;
mod nurbs;
mod path;
mod shape;
mod spline;
pub mod svg_path;
mod wire;

pub use bezier::{
    CubicBezierCurve, CubicBezierCurve3, QuadraticBezierCurve, QuadraticBezierCurve3,
};
pub use catmull_rom::CatmullRomCurve3;
pub use curve::{Curve2, Curve3};
pub use curve_path::CurvePath;
pub use ellipse::EllipseCurve;
pub use line_curve::{LineCurve, LineCurve3};
pub use nurbs::{NURBSCurve, NURBSSurface};
pub use path::Path;
pub use shape::Shape;
pub use spline::SplineCurve;
pub use svg_path::{parse_svg_subpaths, segments_to_path_data, PathSegment, SvgPathError};
pub use wire::{power_pair, Catenary3, Elastica3, ServiceHelix3, WireCode, WireSpec};
