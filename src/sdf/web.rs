//! [`Shape`] for JavaScript, as `WebSdf` (the shim's `THREE.SDF`).
//!
//! Every method returns a new field rather than changing this one, so a part
//! can be reused in several trees. Meshing and evaluation run in Rust: a
//! field built from JavaScript costs one call per mesh, not one per sample.

use wasm_bindgen::prelude::*;

use super::{Sdf3, Shape};
use crate::math::{Quaternion, Vector3};

#[wasm_bindgen]
#[derive(Clone)]
pub struct WebSdf {
    inner: Shape,
}

fn v(x: f32, y: f32, z: f32) -> Vector3 {
    Vector3::new(x, y, z)
}

impl WebSdf {
    fn of(inner: Shape) -> WebSdf {
        WebSdf { inner }
    }
    /// The field, for other bindings that build from it.
    pub(crate) fn shape(&self) -> &Shape {
        &self.inner
    }
}

#[wasm_bindgen]
impl WebSdf {
    pub fn sphere(r: f32) -> WebSdf {
        Self::of(Shape::sphere(r))
    }
    pub fn ellipsoid(rx: f32, ry: f32, rz: f32) -> WebSdf {
        Self::of(Shape::ellipsoid(v(rx, ry, rz)))
    }
    /// Half-extents, and how much the edges are rounded.
    #[wasm_bindgen(js_name = box)]
    pub fn cuboid(hx: f32, hy: f32, hz: f32, round: f32) -> WebSdf {
        Self::of(Shape::round_cuboid(v(hx, hy, hz), round))
    }
    /// From `a` to `b`, radius `ra` at `a` and `rb` at `b`.
    #[allow(clippy::too_many_arguments)]
    pub fn capsule(ax: f32, ay: f32, az: f32, bx: f32, by: f32, bz: f32, ra: f32, rb: f32) -> WebSdf {
        Self::of(Shape::tapered_capsule(v(ax, ay, az), v(bx, by, bz), ra, rb))
    }
    pub fn torus(major: f32, minor: f32) -> WebSdf {
        Self::of(Shape::torus(major, minor))
    }
    pub fn cylinder(half_height: f32, r: f32) -> WebSdf {
        Self::of(Shape::cylinder(half_height, r))
    }
    /// The half-space below `dot(p, n) + offset = 0`.
    pub fn plane(nx: f32, ny: f32, nz: f32, offset: f32) -> WebSdf {
        Self::of(Shape::plane(v(nx, ny, nz), offset))
    }

    pub fn union(&self, other: &WebSdf, k: f32) -> WebSdf {
        Self::of(self.inner.clone().union(other.inner.clone(), k))
    }
    pub fn intersect(&self, other: &WebSdf, k: f32) -> WebSdf {
        Self::of(self.inner.clone().intersect(other.inner.clone(), k))
    }
    pub fn subtract(&self, other: &WebSdf, k: f32) -> WebSdf {
        Self::of(self.inner.clone().subtract(other.inner.clone(), k))
    }
    pub fn round(&self, r: f32) -> WebSdf {
        Self::of(self.inner.clone().round(r))
    }
    pub fn onion(&self, t: f32) -> WebSdf {
        Self::of(self.inner.clone().onion(t))
    }
    pub fn translate(&self, x: f32, y: f32, z: f32) -> WebSdf {
        Self::of(self.inner.clone().translate(v(x, y, z)))
    }
    pub fn rotate(&self, qx: f32, qy: f32, qz: f32, qw: f32) -> WebSdf {
        Self::of(self.inner.clone().rotate(Quaternion::new(qx, qy, qz, qw).normalize()))
    }
    pub fn scale(&self, s: f32) -> WebSdf {
        Self::of(self.inner.clone().scale(s))
    }

    pub fn distance(&self, x: f32, y: f32, z: f32) -> f32 {
        self.inner.distance(v(x, y, z))
    }
    /// The field at every point of a flat `[x, y, z, …]` array.
    pub fn distances(&self, points: &[f32]) -> Vec<f32> {
        points.chunks_exact(3).map(|p| self.inner.distance(v(p[0], p[1], p[2]))).collect()
    }
    /// `[minX, minY, minZ, maxX, maxY, maxZ]`, or empty when unbounded.
    pub fn bounds(&self) -> Vec<f32> {
        self.inner.bounds().map_or_else(Vec::new, |(a, b)| vec![a.x, a.y, a.z, b.x, b.y, b.z])
    }
    /// Surface nets over the box (or the field's bounds, a cell of margin
    /// round them, when `bounds` is empty).
    #[wasm_bindgen(js_name = surfaceNets)]
    pub fn surface_nets(&self, cell: f32, bounds: &[f32]) -> Result<WebSurfaceNets, JsError> {
        let (lo, hi) = if bounds.len() == 6 {
            (v(bounds[0], bounds[1], bounds[2]), v(bounds[3], bounds[4], bounds[5]))
        } else {
            let (lo, hi) = self.inner.bounds().ok_or_else(|| JsError::new("this field has no end: give surfaceNets a box"))?;
            let m = v(cell * 1.5, cell * 1.5, cell * 1.5);
            (lo - m, hi + m)
        };
        let m = self.inner.to_surface_nets(cell, lo, hi);
        Ok(WebSurfaceNets {
            position: m.position.iter().flat_map(|p| [p.x, p.y, p.z]).collect(),
            normal: m.normal.iter().flat_map(|p| [p.x, p.y, p.z]).collect(),
            index: m.index,
        })
    }
}

/// A mesh from `WebSdf.surfaceNets`, as flat arrays.
#[wasm_bindgen]
pub struct WebSurfaceNets {
    position: Vec<f32>,
    normal: Vec<f32>,
    index: Vec<u32>,
}

#[wasm_bindgen]
impl WebSurfaceNets {
    #[wasm_bindgen(getter)]
    pub fn position(&self) -> Vec<f32> {
        self.position.clone()
    }
    #[wasm_bindgen(getter)]
    pub fn normal(&self) -> Vec<f32> {
        self.normal.clone()
    }
    #[wasm_bindgen(getter)]
    pub fn index(&self) -> Vec<u32> {
        self.index.clone()
    }
}
