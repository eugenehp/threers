//! Math primitives. Mirrors three.js's math module.

mod box2;
mod box3;
mod color;
mod color_ramp;
mod color_scale;
mod color_schemes;
mod color_space;
mod cylindrical;
mod euler;
mod frustum;
mod line3;
mod matrix3;
mod matrix4;
mod plane;
mod quaternion;
mod ray;
mod rng;
mod scale;
mod sphere;
mod spherical;
mod triangle;
mod triangulate;
mod vector2;
mod vector3;
mod vector4;

pub use box2::Box2;
pub use box3::Box3;
pub use color::Color;
pub use color_ramp::{ColorRamp, Palette, RampSpace, SchemeKind};
pub use color_scale::{ColorScale, ScaleKind};
pub use color_space::{ColorParseError, ColorSpace};
pub use cylindrical::Cylindrical;
pub use euler::{Euler, EulerOrder};
pub use frustum::Frustum;
pub use line3::Line3;
pub use matrix3::Matrix3;
pub use matrix4::Matrix4;
pub use plane::Plane;
pub use quaternion::Quaternion;
#[cfg(feature = "mesh-bvh")]
pub(crate) use ray::intersect_triangle_f64;
pub use ray::Ray;
pub use rng::Rng;
pub use scale::{
    inverse_lerp, lerp, map_range, map_range_clamped, smootherstep, smoothstep, tick_step, ticks,
    ScaleLinear, ScaleLog, ScalePow,
};
pub use sphere::Sphere;
pub use spherical::Spherical;
pub use triangle::Triangle;
pub(crate) use triangulate::open_ring;
pub use triangulate::{convex_hull, delaunay, polygon_points, triangulate_polygon, voronoi_cells};
pub use vector2::Vector2;
pub use vector3::Vector3;
pub use vector4::Vector4;
