//! Bindings split out of `wasm.rs` by area (see the `mod` list there): the
//! scene data behind the shim's math value types. `Box3`, `Sphere` and
//! `Frustum` are JavaScript classes, as in three.js; what they cannot compute
//! from their own numbers — the bounds of a geometry, or of an object's
//! vertices as it is posed — comes from here, read off the Rust geometry and
//! the mirrored scene graph.
//!
//! Everything is `f64`, operation for operation as three.js r165 does it, so
//! a box or sphere built here equals the one three.js builds from the same
//! `Float32Array` data and the same (JavaScript, `f64`) matrices.
#![allow(unused_imports)]

use super::*;
use crate::core::{BufferAttribute, BufferGeometry, ObjectKind};

type V3 = [f64; 3];

/// `Vector3.fromBufferAttribute(attribute, i)`; components past the item
/// size read as 0.
fn read(a: &BufferAttribute, i: usize) -> V3 {
    let s = a.item_size.max(1);
    let o = i * s;
    let at = |k: usize| {
        if k < s {
            a.array.get(o + k).map_or(0.0, |&v| v as f64)
        } else {
            0.0
        }
    };
    [at(0), at(1), at(2)]
}

/// three.js `Box3` as `[minx, miny, minz, maxx, maxy, maxz]`.
#[derive(Clone, Copy)]
struct Bx([f64; 6]);

impl Bx {
    fn empty() -> Self {
        let i = f64::INFINITY;
        Bx([i, i, i, -i, -i, -i])
    }
    /// `expandByPoint`: `min.min(p)`, `max.max(p)` (`Math.min` / `Math.max`).
    fn expand(&mut self, p: V3) {
        for (k, &pk) in p.iter().enumerate() {
            self.0[k] = js_min(self.0[k], pk);
            self.0[k + 3] = js_max(self.0[k + 3], pk);
        }
    }
    fn min(&self) -> V3 {
        [self.0[0], self.0[1], self.0[2]]
    }
    fn max(&self) -> V3 {
        [self.0[3], self.0[4], self.0[5]]
    }
    /// `setFromBufferAttribute`.
    fn of_attribute(a: &BufferAttribute) -> Self {
        let mut b = Bx::empty();
        for i in 0..a.count() {
            b.expand(read(a, i));
        }
        b
    }
    fn is_empty(&self) -> bool {
        self.0[3] < self.0[0] || self.0[4] < self.0[1] || self.0[5] < self.0[2]
    }
    /// `getCenter`.
    fn center(&self) -> V3 {
        if self.is_empty() {
            [0.0; 3]
        } else {
            [
                (self.0[0] + self.0[3]) * 0.5,
                (self.0[1] + self.0[4]) * 0.5,
                (self.0[2] + self.0[5]) * 0.5,
            ]
        }
    }
}

/// `Math.min` / `Math.max`: NaN wins.
fn js_min(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else if a < b {
        a
    } else {
        b
    }
}
fn js_max(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else if a > b {
        a
    } else {
        b
    }
}

/// three.js `Sphere` as center and radius (`radius < 0` is empty).
struct Sp {
    c: V3,
    r: f64,
}

impl Sp {
    /// `Sphere.expandByPoint`.
    fn expand(&mut self, p: V3) {
        if self.r < 0.0 {
            self.c = p;
            self.r = 0.0;
            return;
        }
        let v = [p[0] - self.c[0], p[1] - self.c[1], p[2] - self.c[2]];
        let length_sq = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
        if length_sq > self.r * self.r {
            let length = length_sq.sqrt();
            let delta = (length - self.r) * 0.5;
            let s = delta / length;
            for (c, vk) in self.c.iter_mut().zip(v) {
                *c += vk * s;
            }
            self.r += delta;
        }
    }
}

/// `Vector3.applyMatrix4` (column-major `e`), with the perspective divide.
fn apply(e: &[f64], p: V3) -> V3 {
    let [x, y, z] = p;
    let w = 1.0 / (e[3] * x + e[7] * y + e[11] * z + e[15]);
    [
        (e[0] * x + e[4] * y + e[8] * z + e[12]) * w,
        (e[1] * x + e[5] * y + e[9] * z + e[13]) * w,
        (e[2] * x + e[6] * y + e[10] * z + e[14]) * w,
    ]
}

/// `BufferGeometry.computeBoundingBox()`; `None` without a position
/// attribute (three.js then makes the box empty).
fn geometry_box(g: &BufferGeometry) -> Option<Bx> {
    let position = g.get_attribute("position")?;
    let mut bb = Bx::of_attribute(position);
    for m in g.morph_attributes.get("position").into_iter().flatten() {
        let b = Bx::of_attribute(m);
        if g.morph_targets_relative {
            let (lo, blo) = (bb.min(), b.min());
            bb.expand([lo[0] + blo[0], lo[1] + blo[1], lo[2] + blo[2]]);
            let (hi, bhi) = (bb.max(), b.max());
            bb.expand([hi[0] + bhi[0], hi[1] + bhi[1], hi[2] + bhi[2]]);
        } else {
            bb.expand(b.min());
            bb.expand(b.max());
        }
    }
    Some(bb)
}

/// `BufferGeometry.computeBoundingSphere()`; `None` without a position
/// attribute (three.js then leaves the sphere as it was).
fn geometry_sphere(g: &BufferGeometry) -> Option<[f64; 4]> {
    let position = g.get_attribute("position")?;
    let morphs = g.morph_attributes.get("position");
    let mut bb = Bx::of_attribute(position);
    for m in morphs.into_iter().flatten() {
        let b = Bx::of_attribute(m);
        if g.morph_targets_relative {
            let (lo, blo) = (bb.min(), b.min());
            bb.expand([lo[0] + blo[0], lo[1] + blo[1], lo[2] + blo[2]]);
            let (hi, bhi) = (bb.max(), b.max());
            bb.expand([hi[0] + bhi[0], hi[1] + bhi[1], hi[2] + bhi[2]]);
        } else {
            bb.expand(b.min());
            bb.expand(b.max());
        }
    }
    let c = bb.center();
    let d2 = |p: V3| {
        let (dx, dy, dz) = (c[0] - p[0], c[1] - p[1], c[2] - p[2]);
        dx * dx + dy * dy + dz * dz
    };
    let mut max_sq: f64 = 0.0;
    for i in 0..position.count() {
        max_sq = js_max(max_sq, d2(read(position, i)));
    }
    for m in morphs.into_iter().flatten() {
        for j in 0..m.count() {
            let mut v = read(m, j);
            if g.morph_targets_relative {
                let o = read(position, j);
                v = [v[0] + o[0], v[1] + o[1], v[2] + o[2]];
            }
            max_sq = js_max(max_sq, d2(v));
        }
    }
    Some([c[0], c[1], c[2], max_sq.sqrt()])
}

/// three.js's shared `Sprite` geometry: a unit quad in the XY plane.
fn sprite_geometry() -> BufferGeometry {
    let mut g = BufferGeometry::new();
    g.set_attribute(
        "position",
        BufferAttribute::new(
            vec![
                -0.5, -0.5, 0.0, 0.5, -0.5, 0.0, 0.5, 0.5, 0.0, -0.5, 0.5, 0.0,
            ],
            3,
        ),
    );
    g
}

/// The vertices of `g` as three.js's `object.getVertexPosition(i)` gives them
/// for a mesh (morph targets blended by `influences`, then skinned by
/// `skin`), or `fromBufferAttribute` for anything else (`influences` and
/// `skin` absent) — then placed by `matrix` when one is given.
///
/// `skin` is `[bindMatrix, bindMatrixInverse, bone₀, bone₁, …]`, 16 numbers
/// each, `boneᵢ = bones[i].matrixWorld × boneInverses[i]`.
fn for_each_vertex(
    g: &BufferGeometry,
    influences: Option<&[f64]>,
    skin: Option<&[f64]>,
    matrix: Option<&[f64]>,
    mut f: impl FnMut(V3),
) -> bool {
    let Some(position) = g.get_attribute("position") else {
        return false;
    };
    let morphs = influences.and_then(|w| Some((w, g.morph_attributes.get("position")?)));
    let skin = skin
        .filter(|s| s.len() >= 32)
        .and_then(|s| Some((s, g.get_attribute("joint")?, g.get_attribute("weight")?)));
    for i in 0..position.count() {
        let mut p = read(position, i);
        if let Some((weights, targets)) = morphs {
            let mut acc = [0.0; 3];
            for (t, target) in targets.iter().enumerate() {
                let w = weights.get(t).copied().unwrap_or(f64::NAN);
                if w == 0.0 {
                    continue;
                }
                let mut v = read(target, i);
                if !g.morph_targets_relative {
                    v = [v[0] - p[0], v[1] - p[1], v[2] - p[2]];
                }
                for k in 0..3 {
                    acc[k] += v[k] * w;
                }
            }
            p = [p[0] + acc[0], p[1] + acc[1], p[2] + acc[2]];
        }
        if let Some((s, joints, weights)) = skin {
            let base = apply(&s[0..16], p);
            let ji = joints.item_size.max(1);
            let wi = weights.item_size.max(1);
            let mut acc = [0.0; 3];
            for k in 0..4 {
                let w = if k < wi {
                    weights.array.get(i * wi + k).map_or(0.0, |&v| v as f64)
                } else {
                    0.0
                };
                if w != 0.0 {
                    let bone = if k < ji {
                        joints.array.get(i * ji + k).map_or(0, |&v| v as usize)
                    } else {
                        0
                    };
                    let o = 32 + bone * 16;
                    let v = match s.get(o..o + 16) {
                        Some(m) => apply(m, base),
                        None => [f64::NAN; 3],
                    };
                    for c in 0..3 {
                        acc[c] += v[c] * w;
                    }
                }
            }
            p = apply(&s[16..32], acc);
        }
        if let Some(m) = matrix {
            p = apply(m, p);
        }
        f(p);
    }
    true
}

fn matrix16(m: &Option<Vec<f64>>) -> Option<&[f64]> {
    m.as_deref().filter(|m| m.len() >= 16)
}

#[wasm_bindgen]
impl WebGeometry {
    /// three.js `BufferGeometry.computeBoundingBox()`:
    /// `[minx, miny, minz, maxx, maxy, maxz]`, or undefined without a
    /// `position` attribute.
    #[wasm_bindgen(js_name = threeBoundingBox)]
    pub fn three_bounding_box(&self) -> Option<Vec<f64>> {
        geometry_box(&self.inner).map(|b| b.0.to_vec())
    }

    /// three.js `BufferGeometry.computeBoundingSphere()`:
    /// `[cx, cy, cz, radius]`, or undefined without a `position` attribute.
    #[wasm_bindgen(js_name = threeBoundingSphere)]
    pub fn three_bounding_sphere(&self) -> Option<Vec<f64>> {
        geometry_sphere(&self.inner).map(|s| s.to_vec())
    }
}

#[wasm_bindgen]
impl WebScene {
    /// The bounds of a node's vertices, one by one, as three.js walks them
    /// (`Box3.expandByObject(object, true)`, `SkinnedMesh.computeBoundingBox`
    /// / `computeBoundingSphere`): the node's geometry (a line's strip as
    /// authored, a sprite's unit quad), morphed by `influences` and skinned by
    /// `skin` (see the shim), then placed by `matrix` (16 numbers,
    /// column-major) when given. `mode` 0 returns a box
    /// `[minx, miny, minz, maxx, maxy, maxz]`, 1 a sphere grown point by point
    /// `[cx, cy, cz, radius]`. Undefined when the node has no positions.
    #[wasm_bindgen(js_name = vertexBounds)]
    pub fn vertex_bounds(
        &self,
        node: &WebObjectHandle,
        mode: u32,
        matrix: Option<Vec<f64>>,
        influences: Option<Vec<f64>>,
        skin: Option<Vec<f64>>,
    ) -> Option<Vec<f64>> {
        let o = self.inner.get(node.id)?;
        let sprite;
        let g: &BufferGeometry = match &o.kind {
            ObjectKind::Mesh(m) => &m.geometry,
            ObjectKind::SkinnedMesh(m) => &m.geometry,
            ObjectKind::InstancedMesh(m) => &m.geometry,
            ObjectKind::Points(p) => &p.geometry,
            ObjectKind::LineSegments(l) => l.strip.as_ref().map_or(&l.geometry, |s| &s.geometry),
            ObjectKind::Sprite(_) => {
                sprite = sprite_geometry();
                &sprite
            }
            ObjectKind::Group | ObjectKind::Light(_) => return None,
        };
        let m = matrix16(&matrix);
        let w = influences.as_deref();
        let s = skin.as_deref();
        if mode == 1 {
            let mut sp = Sp {
                c: [0.0; 3],
                r: -1.0,
            };
            for_each_vertex(g, w, s, m, |p| sp.expand(p))
                .then(|| vec![sp.c[0], sp.c[1], sp.c[2], sp.r])
        } else {
            let mut b = Bx::empty();
            for_each_vertex(g, w, s, m, |p| b.expand(p)).then(|| b.0.to_vec())
        }
    }

    /// three.js `geometry.computeBoundingBox()` / `computeBoundingSphere()`
    /// for the geometry a node draws (`mode` 0 / 1, results as on
    /// [`WebGeometry`]); for a sprite, its unit quad.
    #[wasm_bindgen(js_name = nodeGeometryBounds)]
    pub fn node_geometry_bounds(&self, node: &WebObjectHandle, mode: u32) -> Option<Vec<f64>> {
        let o = self.inner.get(node.id)?;
        let sprite;
        let g: &BufferGeometry = match &o.kind {
            ObjectKind::Mesh(m) => &m.geometry,
            ObjectKind::SkinnedMesh(m) => &m.geometry,
            ObjectKind::InstancedMesh(m) => &m.geometry,
            ObjectKind::Points(p) => &p.geometry,
            ObjectKind::LineSegments(l) => l.strip.as_ref().map_or(&l.geometry, |s| &s.geometry),
            ObjectKind::Sprite(_) => {
                sprite = sprite_geometry();
                &sprite
            }
            ObjectKind::Group | ObjectKind::Light(_) => return None,
        };
        if mode == 1 {
            geometry_sphere(g).map(|s| s.to_vec())
        } else {
            geometry_box(g).map(|b| b.0.to_vec())
        }
    }
}
