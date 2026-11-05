//! `MeshBuilder` and `meshReport` for JavaScript: the Rust builder, with the
//! callbacks and arrays a JS caller has instead of closures and vectors.

use super::{WebBufferGeometry, WebGeometry};
use crate::geometries::{MeshBuilder, MeshReport};
use crate::math::{Color, Matrix4, Vector3};
use std::sync::Arc;
use wasm_bindgen::prelude::*;

/// Vertices and triangles, built into a geometry.
#[wasm_bindgen(js_name = MeshBuilder)]
pub struct WebMeshBuilder {
    inner: MeshBuilder,
}

fn points(flat: &[f32]) -> Result<Vec<Vector3>, JsError> {
    if flat.len() % 3 != 0 {
        return Err(JsError::new(&format!(
            "expected x, y, z triples, got {} numbers",
            flat.len()
        )));
    }
    Ok(flat
        .chunks_exact(3)
        .map(|p| Vector3::new(p[0], p[1], p[2]))
        .collect())
}

#[wasm_bindgen(js_class = MeshBuilder)]
impl WebMeshBuilder {
    #[wasm_bindgen(constructor)]
    pub fn new() -> WebMeshBuilder {
        WebMeshBuilder {
            inner: MeshBuilder::new(),
        }
    }

    /// The normal of the vertices to come, instead of one from the faces.
    pub fn normal(&mut self, x: f32, y: f32, z: f32) {
        self.inner.normal(Vector3::new(x, y, z));
    }

    #[wasm_bindgen(js_name = autoNormal)]
    pub fn auto_normal(&mut self) {
        self.inner.auto_normal();
    }

    pub fn uv(&mut self, u: f32, v: f32) {
        self.inner.uv(u, v);
    }

    /// Linear RGB, 0 to 1.
    pub fn color(&mut self, r: f32, g: f32, b: f32) {
        self.inner.color(Color::new(r, g, b));
    }

    /// Turn the triangles to come towards this side, whatever their order.
    pub fn facing(&mut self, x: f32, y: f32, z: f32) {
        self.inner.facing(Vector3::new(x, y, z));
    }

    #[wasm_bindgen(js_name = asWound)]
    pub fn as_wound(&mut self) {
        self.inner.as_wound();
    }

    /// Add a vertex, returning its index.
    pub fn vertex(&mut self, x: f32, y: f32, z: f32) -> u32 {
        self.inner.vertex(Vector3::new(x, y, z))
    }

    /// Throws if an index is not a vertex.
    pub fn triangle(&mut self, a: u32, b: u32, c: u32) -> Result<(), JsError> {
        self.check(&[a, b, c])?;
        self.inner.triangle(a, b, c);
        Ok(())
    }

    pub fn quad(&mut self, a: u32, b: u32, c: u32, d: u32) -> Result<(), JsError> {
        self.check(&[a, b, c, d])?;
        self.inner.quad(a, b, c, d);
        Ok(())
    }

    /// A strip between two edges given as flat x, y, z arrays, facing
    /// `(sx, sy, sz)`; `closed` joins the ends.
    pub fn ribbon(
        &mut self,
        left: &[f32],
        right: &[f32],
        sx: f32,
        sy: f32,
        sz: f32,
        closed: bool,
    ) -> Result<(), JsError> {
        let (left, right) = (points(left)?, points(right)?);
        if left.len() != right.len() {
            return Err(JsError::new(&format!(
                "a ribbon's edges must pair up: {} and {} points",
                left.len(),
                right.len()
            )));
        }
        self.inner
            .ribbon(&left, &right, Vector3::new(sx, sy, sz), closed);
        Ok(())
    }

    /// A height field over x and z, centred on the origin. `height` is a
    /// function `(x, z) => y`, an array of `(columns + 1) * (rows + 1)` heights
    /// row by row, or nothing for flat.
    pub fn grid(
        &mut self,
        width: f32,
        depth: f32,
        columns: u32,
        rows: u32,
        height: JsValue,
    ) -> Result<(), JsError> {
        let (columns, rows) = (columns.max(1), rows.max(1));
        if let Some(f) = height.dyn_ref::<js_sys::Function>() {
            let mut error = None;
            self.inner.grid(width, depth, columns, rows, |x, z| {
                match f
                    .call2(&JsValue::NULL, &JsValue::from(x), &JsValue::from(z))
                    .map(|v| v.as_f64())
                {
                    Ok(Some(y)) => y as f32,
                    _ => {
                        error.get_or_insert_with(|| {
                            JsError::new("the height function must return a number")
                        });
                        0.0
                    }
                }
            });
            return error.map_or(Ok(()), Err);
        }
        let heights: Vec<f32> = if height.is_undefined() || height.is_null() {
            Vec::new()
        } else {
            js_sys::Float32Array::new(&height).to_vec()
        };
        let want = ((columns + 1) * (rows + 1)) as usize;
        if !heights.is_empty() && heights.len() != want {
            return Err(JsError::new(&format!(
                "expected {want} heights for {columns} by {rows} cells, got {}",
                heights.len()
            )));
        }
        let mut at = 0;
        self.inner.grid(width, depth, columns, rows, |_, _| {
            let y = heights.get(at).copied().unwrap_or(0.0);
            at += 1;
            y
        });
        Ok(())
    }

    /// Copy a geometry in, moved by a 4x4 matrix in three.js's column-major
    /// `elements` order, or not moved without one.
    pub fn append(
        &mut self,
        geometry: &WebGeometry,
        matrix: Option<Vec<f32>>,
    ) -> Result<(), JsError> {
        let transform = match matrix {
            Some(m) if m.len() == 16 => Matrix4 {
                elements: m.try_into().unwrap(),
            },
            Some(m) => {
                return Err(JsError::new(&format!(
                    "a matrix has 16 numbers, not {}",
                    m.len()
                )))
            }
            None => Matrix4::identity(),
        };
        self.inner.append(&geometry.inner, &transform);
        Ok(())
    }

    #[wasm_bindgen(getter, js_name = vertexCount)]
    pub fn vertex_count(&self) -> usize {
        self.inner.vertex_count()
    }

    #[wasm_bindgen(getter, js_name = triangleCount)]
    pub fn triangle_count(&self) -> usize {
        self.inner.triangle_count()
    }

    /// The geometry, with vertices shared so smooth surfaces shade smooth.
    pub fn build(&self) -> WebBufferGeometry {
        WebBufferGeometry {
            inner: Arc::new(self.inner.build()),
        }
    }

    /// The geometry with each triangle shaded flat.
    #[wasm_bindgen(js_name = buildFlat)]
    pub fn build_flat(&self) -> WebBufferGeometry {
        WebBufferGeometry {
            inner: Arc::new(self.inner.build_flat()),
        }
    }

    fn check(&self, indices: &[u32]) -> Result<(), JsError> {
        let count = self.inner.vertex_count() as u32;
        match indices.iter().find(|&&i| i >= count) {
            Some(i) => Err(JsError::new(&format!(
                "vertex {i} does not exist: there are {count}"
            ))),
            None => Ok(()),
        }
    }
}

impl Default for WebMeshBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// What is wrong with a mesh, if anything: `{triangles, degenerate, openEdges,
/// flippedEdges, crowdedEdges, isSolid}`.
#[wasm_bindgen(js_name = meshReport)]
pub fn mesh_report(geometry: &WebGeometry) -> JsValue {
    let r = MeshReport::of(&geometry.inner);
    let out = js_sys::Object::new();
    let set = |k: &str, v: JsValue| {
        let _ = js_sys::Reflect::set(&out, &JsValue::from_str(k), &v);
    };
    set("triangles", (r.triangles as f64).into());
    set("degenerate", (r.degenerate as f64).into());
    set("openEdges", (r.open_edges as f64).into());
    set("flippedEdges", (r.flipped_edges as f64).into());
    set("crowdedEdges", (r.crowded_edges as f64).into());
    set("isSolid", r.is_solid().into());
    out.into()
}
