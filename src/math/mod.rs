//! Math primitives. Mirrors three.js's math module.

mod box2;
mod box3;
mod color;
mod color_ramp;
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
pub use color_ramp::{ColorRamp, RampSpace};
pub use cylindrical::Cylindrical;
pub use euler::Euler;
pub use frustum::Frustum;
pub use line3::Line3;
pub use matrix3::Matrix3;
pub use matrix4::Matrix4;
pub use plane::Plane;
pub use quaternion::Quaternion;
pub use ray::Ray;
pub use rng::Rng;
pub use scale::{
    inverse_lerp, lerp, map_range, map_range_clamped, smootherstep, smoothstep, tick_step, ticks,
    ScaleLinear, ScaleLog, ScalePow,
};
pub use sphere::Sphere;
pub use spherical::Spherical;
pub use triangle::Triangle;
pub use triangulate::{convex_hull, delaunay, voronoi_cells};
pub use vector2::Vector2;
pub use vector3::Vector3;
pub use vector4::Vector4;
