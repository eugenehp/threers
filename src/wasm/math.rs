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
            #[cfg(feature = "audio")]
            ObjectKind::Audio(_) => return None,
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
            #[cfg(feature = "audio")]
            ObjectKind::Audio(_) => return None,
        };
        if mode == 1 {
            geometry_sphere(g).map(|s| s.to_vec())
        } else {
            geometry_box(g).map(|b| b.0.to_vec())
        }
    }
}

// ---- math ----
#[wasm_bindgen]
#[derive(Clone, Copy)]
pub struct WebVector2 {
    pub x: f32,
    pub y: f32,
}
#[wasm_bindgen]
impl WebVector2 {
    #[wasm_bindgen(constructor)]
    pub fn new(x: f32, y: f32) -> WebVector2 {
        WebVector2 { x, y }
    }
    #[wasm_bindgen(js_name = lengthSq)]
    pub fn length_sq(&self) -> f32 {
        self.x * self.x + self.y * self.y
    }
    pub fn length(&self) -> f32 {
        self.length_sq().sqrt()
    }
    pub fn dot(&self, o: &WebVector2) -> f32 {
        self.x * o.x + self.y * o.y
    }
}

#[wasm_bindgen]
#[derive(Clone, Copy)]
pub struct WebVector4 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub w: f32,
}
#[wasm_bindgen]
impl WebVector4 {
    #[wasm_bindgen(constructor)]
    pub fn new(x: f32, y: f32, z: f32, w: f32) -> WebVector4 {
        WebVector4 { x, y, z, w }
    }
    pub fn length(&self) -> f32 {
        (self.x * self.x + self.y * self.y + self.z * self.z + self.w * self.w).sqrt()
    }
}

#[wasm_bindgen]
#[derive(Clone)]
pub struct WebMatrix3 {
    pub(super) inner: crate::Matrix3,
}
#[wasm_bindgen]
impl WebMatrix3 {
    #[wasm_bindgen(constructor)]
    pub fn new() -> WebMatrix3 {
        WebMatrix3 {
            inner: crate::Matrix3::identity(),
        }
    }
    pub fn identity() -> WebMatrix3 {
        WebMatrix3 {
            inner: crate::Matrix3::identity(),
        }
    }
    pub fn elements(&self) -> Vec<f32> {
        self.inner.elements.to_vec()
    }
}

#[wasm_bindgen]
#[derive(Clone)]
pub struct WebMatrix4 {
    pub(crate) inner: crate::Matrix4,
}
#[wasm_bindgen]
impl WebMatrix4 {
    #[wasm_bindgen(constructor)]
    pub fn new() -> WebMatrix4 {
        WebMatrix4 {
            inner: crate::Matrix4::identity(),
        }
    }
    pub fn identity() -> WebMatrix4 {
        WebMatrix4 {
            inner: crate::Matrix4::identity(),
        }
    }
    pub fn elements(&self) -> Vec<f32> {
        self.inner.elements.to_vec()
    }
    #[wasm_bindgen(js_name = makePerspective)]
    pub fn make_perspective(fov: f32, aspect: f32, near: f32, far: f32) -> WebMatrix4 {
        WebMatrix4 {
            inner: crate::Matrix4::perspective(fov, aspect, near, far),
        }
    }
    pub fn invert(&self) -> WebMatrix4 {
        WebMatrix4 {
            inner: self.inner.invert(),
        }
    }
}

#[wasm_bindgen]
#[derive(Clone, Copy)]
pub struct WebQuaternion {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub w: f32,
}
#[wasm_bindgen]
impl WebQuaternion {
    #[wasm_bindgen(constructor)]
    pub fn new(x: f32, y: f32, z: f32, w: f32) -> WebQuaternion {
        WebQuaternion { x, y, z, w }
    }
    pub fn identity() -> WebQuaternion {
        WebQuaternion {
            x: 0.0,
            y: 0.0,
            z: 0.0,
            w: 1.0,
        }
    }
    #[wasm_bindgen(js_name = setFromEuler)]
    pub fn set_from_euler(e: &WebEuler) -> WebQuaternion {
        let q = e.inner.to_quaternion();
        WebQuaternion {
            x: q.x,
            y: q.y,
            z: q.z,
            w: q.w,
        }
    }
}

#[wasm_bindgen]
#[derive(Clone, Copy)]
pub struct WebBox2 {
    pub(super) inner: crate::Box2,
}
#[wasm_bindgen]
impl WebBox2 {
    #[wasm_bindgen(constructor)]
    pub fn new(min: &WebVector2, max: &WebVector2) -> WebBox2 {
        WebBox2 {
            inner: crate::Box2::new(
                crate::Vector2::new(min.x, min.y),
                crate::Vector2::new(max.x, max.y),
            ),
        }
    }
    pub fn empty() -> WebBox2 {
        WebBox2 {
            inner: crate::Box2::empty(),
        }
    }
    #[wasm_bindgen(js_name = isEmpty)]
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
}

#[wasm_bindgen]
#[derive(Clone, Copy)]
pub struct WebBox3 {
    pub(crate) inner: crate::Box3,
}
#[wasm_bindgen]
impl WebBox3 {
    #[wasm_bindgen(constructor)]
    pub fn new(min: &WebVector3, max: &WebVector3) -> WebBox3 {
        WebBox3 {
            inner: crate::Box3::new(min.inner, max.inner),
        }
    }
    pub fn empty() -> WebBox3 {
        WebBox3 {
            inner: crate::Box3::empty(),
        }
    }
    #[wasm_bindgen(js_name = isEmpty)]
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
    #[wasm_bindgen(js_name = containsPoint)]
    pub fn contains_point(&self, p: &WebVector3) -> bool {
        self.inner.contains_point(p.inner)
    }
    #[wasm_bindgen(js_name = intersectsBox)]
    pub fn intersects_box(&self, o: &WebBox3) -> bool {
        self.inner.intersects_box(&o.inner)
    }
    pub fn center(&self) -> WebVector3 {
        WebVector3 {
            inner: self.inner.center(),
        }
    }
    pub fn size(&self) -> WebVector3 {
        WebVector3 {
            inner: self.inner.size(),
        }
    }
}

#[wasm_bindgen]
#[derive(Clone, Copy)]
pub struct WebSphere {
    pub(super) inner: crate::Sphere,
}
#[wasm_bindgen]
impl WebSphere {
    #[wasm_bindgen(constructor)]
    pub fn new(center: &WebVector3, radius: f32) -> WebSphere {
        WebSphere {
            inner: crate::Sphere::new(center.inner, radius),
        }
    }
    #[wasm_bindgen(js_name = containsPoint)]
    pub fn contains_point(&self, p: &WebVector3) -> bool {
        self.inner.contains_point(p.inner)
    }
}

#[wasm_bindgen]
#[derive(Clone, Copy)]
pub struct WebRay {
    pub(super) inner: crate::Ray,
}
#[wasm_bindgen]
impl WebRay {
    #[wasm_bindgen(constructor)]
    pub fn new(origin: &WebVector3, direction: &WebVector3) -> WebRay {
        WebRay {
            inner: crate::Ray::new(origin.inner, direction.inner),
        }
    }
    pub fn at(&self, t: f32) -> WebVector3 {
        WebVector3 {
            inner: self.inner.at(t),
        }
    }
}

#[wasm_bindgen]
#[derive(Clone, Copy)]
pub struct WebPlane {
    pub(super) inner: crate::Plane,
}
#[wasm_bindgen]
impl WebPlane {
    #[wasm_bindgen(constructor)]
    pub fn new(normal: &WebVector3, constant: f32) -> WebPlane {
        WebPlane {
            inner: crate::Plane::new(normal.inner, constant),
        }
    }
    #[wasm_bindgen(js_name = distanceToPoint)]
    pub fn distance_to_point(&self, p: &WebVector3) -> f32 {
        self.inner.distance_to_point(p.inner)
    }
}

#[wasm_bindgen]
#[derive(Clone, Copy)]
pub struct WebTriangle {
    pub(super) inner: crate::Triangle,
}
#[wasm_bindgen]
impl WebTriangle {
    #[wasm_bindgen(constructor)]
    pub fn new(a: &WebVector3, b: &WebVector3, c: &WebVector3) -> WebTriangle {
        WebTriangle {
            inner: crate::Triangle::new(a.inner, b.inner, c.inner),
        }
    }
    pub fn area(&self) -> f32 {
        self.inner.area()
    }
    pub fn normal(&self) -> WebVector3 {
        WebVector3 {
            inner: self.inner.normal(),
        }
    }
}

#[wasm_bindgen]
#[derive(Clone)]
pub struct WebFrustum {
    pub(super) inner: crate::Frustum,
}
#[wasm_bindgen]
impl WebFrustum {
    #[wasm_bindgen(constructor)]
    pub fn new() -> WebFrustum {
        WebFrustum {
            inner: crate::Frustum::default(),
        }
    }
    #[wasm_bindgen(js_name = setFromProjectionMatrix)]
    pub fn set_from_projection_matrix(m: &WebMatrix4) -> WebFrustum {
        WebFrustum {
            inner: crate::Frustum::from_projection_matrix(&m.inner),
        }
    }
    #[wasm_bindgen(js_name = containsPoint)]
    pub fn contains_point(&self, p: &WebVector3) -> bool {
        self.inner.contains_point(p.inner)
    }
}

#[wasm_bindgen]
#[derive(Clone, Copy)]
pub struct WebSpherical {
    pub(super) inner: crate::Spherical,
}
#[wasm_bindgen]
impl WebSpherical {
    #[wasm_bindgen(constructor)]
    pub fn new(radius: f32, phi: f32, theta: f32) -> WebSpherical {
        WebSpherical {
            inner: crate::Spherical::new(radius, phi, theta),
        }
    }
    #[wasm_bindgen(js_name = setFromVector3)]
    pub fn set_from_vector3(v: &WebVector3) -> WebSpherical {
        WebSpherical {
            inner: crate::Spherical::from_vector3(v.inner),
        }
    }
}

#[wasm_bindgen]
#[derive(Clone, Copy)]
pub struct WebCylindrical {
    pub(super) inner: crate::Cylindrical,
}
#[wasm_bindgen]
impl WebCylindrical {
    #[wasm_bindgen(constructor)]
    pub fn new(radius: f32, theta: f32, y: f32) -> WebCylindrical {
        WebCylindrical {
            inner: crate::Cylindrical::new(radius, theta, y),
        }
    }
}

#[wasm_bindgen]
#[derive(Clone, Copy)]
pub struct WebLine3 {
    pub(super) inner: crate::Line3,
}
#[wasm_bindgen]
impl WebLine3 {
    #[wasm_bindgen(constructor)]
    pub fn new(start: &WebVector3, end: &WebVector3) -> WebLine3 {
        WebLine3 {
            inner: crate::Line3::new(start.inner, end.inner),
        }
    }
    pub fn distance(&self) -> f32 {
        self.inner.distance()
    }
    pub fn center(&self) -> WebVector3 {
        WebVector3 {
            inner: self.inner.center(),
        }
    }
}

// ---- math method completeness ----
#[wasm_bindgen]
impl WebVector3 {
    pub fn add(&mut self, o: &WebVector3) -> WebVector3 {
        self.inner = self.inner + o.inner;
        WebVector3 { inner: self.inner }
    }
    pub fn sub(&mut self, o: &WebVector3) -> WebVector3 {
        self.inner = self.inner - o.inner;
        WebVector3 { inner: self.inner }
    }
    #[wasm_bindgen(js_name = multiplyScalar)]
    pub fn multiply_scalar(&mut self, s: f32) -> WebVector3 {
        self.inner = self.inner * s;
        WebVector3 { inner: self.inner }
    }
    pub fn length(&self) -> f32 {
        self.inner.length()
    }
    #[wasm_bindgen(js_name = lengthSq)]
    pub fn length_sq(&self) -> f32 {
        self.inner.length_sq()
    }
    pub fn normalize(&mut self) -> WebVector3 {
        self.inner = self.inner.normalize();
        WebVector3 { inner: self.inner }
    }
    pub fn dot(&self, o: &WebVector3) -> f32 {
        self.inner.dot(o.inner)
    }
    pub fn cross(&mut self, o: &WebVector3) -> WebVector3 {
        self.inner = self.inner.cross(o.inner);
        WebVector3 { inner: self.inner }
    }
    #[wasm_bindgen(js_name = distanceTo)]
    pub fn distance_to(&self, o: &WebVector3) -> f32 {
        self.inner.distance_to(o.inner)
    }
    pub fn lerp(&mut self, o: &WebVector3, t: f32) -> WebVector3 {
        self.inner = self.inner.lerp(o.inner, t);
        WebVector3 { inner: self.inner }
    }
    #[wasm_bindgen(js_name = applyMatrix4)]
    pub fn apply_matrix4(&mut self, m: &WebMatrix4) -> WebVector3 {
        self.inner = self.inner.apply_matrix4(&m.inner);
        WebVector3 { inner: self.inner }
    }
    #[wasm_bindgen(js_name = applyQuaternion)]
    pub fn apply_quaternion(&mut self, q: &WebQuaternion) -> WebVector3 {
        let qq = crate::Quaternion::new(q.x, q.y, q.z, q.w);
        self.inner = self.inner.apply_quaternion(qq);
        WebVector3 { inner: self.inner }
    }
}

#[wasm_bindgen]
impl WebColor {
    pub fn r(&self) -> f32 {
        self.inner.r
    }
    pub fn g(&self) -> f32 {
        self.inner.g
    }
    pub fn b(&self) -> f32 {
        self.inner.b
    }
    #[wasm_bindgen(js_name = setRGB)]
    pub fn set_rgb(&mut self, r: f32, g: f32, b: f32) -> WebColor {
        self.inner = crate::Color::new(r, g, b);
        *self
    }
    #[wasm_bindgen(js_name = setHex)]
    pub fn set_hex(&mut self, h: u32) -> WebColor {
        self.inner = crate::Color::from_hex(h);
        *self
    }
    pub fn lerp(&mut self, other: &WebColor, t: f32) -> WebColor {
        let r = self.inner.r + (other.inner.r - self.inner.r) * t;
        let g = self.inner.g + (other.inner.g - self.inner.g) * t;
        let b = self.inner.b + (other.inner.b - self.inner.b) * t;
        self.inner = crate::Color::new(r, g, b);
        *self
    }
    #[wasm_bindgen(js_name = getHex)]
    pub fn get_hex(&self) -> u32 {
        let r = (self.inner.r.clamp(0.0, 1.0) * 255.0) as u32;
        let g = (self.inner.g.clamp(0.0, 1.0) * 255.0) as u32;
        let b = (self.inner.b.clamp(0.0, 1.0) * 255.0) as u32;
        (r << 16) | (g << 8) | b
    }
}

#[wasm_bindgen]
impl WebMatrix4 {
    pub fn multiply(&mut self, m: &WebMatrix4) -> WebMatrix4 {
        self.inner = self.inner.multiply(&m.inner);
        WebMatrix4 { inner: self.inner }
    }
    #[wasm_bindgen(js_name = makeTranslation)]
    pub fn make_translation(x: f32, y: f32, z: f32) -> WebMatrix4 {
        WebMatrix4 {
            inner: crate::Matrix4::translation(crate::Vector3::new(x, y, z)),
        }
    }
    #[wasm_bindgen(js_name = makeScale)]
    pub fn make_scale(x: f32, y: f32, z: f32) -> WebMatrix4 {
        WebMatrix4 {
            inner: crate::Matrix4::scale(crate::Vector3::new(x, y, z)),
        }
    }
    #[wasm_bindgen(js_name = lookAt)]
    pub fn look_at(eye: &WebVector3, target: &WebVector3, up: &WebVector3) -> WebMatrix4 {
        WebMatrix4 {
            inner: crate::Matrix4::look_at(eye.inner, target.inner, up.inner),
        }
    }
    pub fn determinant(&self) -> f32 {
        self.inner.determinant()
    }
}

#[wasm_bindgen]
impl WebQuaternion {
    pub fn normalize(&mut self) -> WebQuaternion {
        let q = crate::Quaternion::new(self.x, self.y, self.z, self.w).normalize();
        self.x = q.x;
        self.y = q.y;
        self.z = q.z;
        self.w = q.w;
        *self
    }
    pub fn invert(&mut self) -> WebQuaternion {
        let q = crate::Quaternion::new(self.x, self.y, self.z, self.w).invert();
        self.x = q.x;
        self.y = q.y;
        self.z = q.z;
        self.w = q.w;
        *self
    }
    pub fn multiply(&mut self, o: &WebQuaternion) -> WebQuaternion {
        let a = crate::Quaternion::new(self.x, self.y, self.z, self.w);
        let b = crate::Quaternion::new(o.x, o.y, o.z, o.w);
        let c = a.multiply(b);
        self.x = c.x;
        self.y = c.y;
        self.z = c.z;
        self.w = c.w;
        *self
    }
    pub fn slerp(&mut self, o: &WebQuaternion, t: f32) -> WebQuaternion {
        let a = crate::Quaternion::new(self.x, self.y, self.z, self.w);
        let b = crate::Quaternion::new(o.x, o.y, o.z, o.w);
        let c = a.slerp(b, t);
        self.x = c.x;
        self.y = c.y;
        self.z = c.z;
        self.w = c.w;
        *self
    }
    pub fn dot(&self, o: &WebQuaternion) -> f32 {
        let a = crate::Quaternion::new(self.x, self.y, self.z, self.w);
        let b = crate::Quaternion::new(o.x, o.y, o.z, o.w);
        a.dot(b)
    }
    #[wasm_bindgen(js_name = setFromAxisAngle)]
    pub fn set_from_axis_angle(&mut self, axis: &WebVector3, angle: f32) -> WebQuaternion {
        let q = crate::Quaternion::from_axis_angle(axis.inner, angle);
        self.x = q.x;
        self.y = q.y;
        self.z = q.z;
        self.w = q.w;
        *self
    }
}

// ---- math utilities (three.mathutils) ----
#[wasm_bindgen]
pub struct WebMathUtils;

#[wasm_bindgen]
impl WebMathUtils {
    pub fn clamp(v: f32, min: f32, max: f32) -> f32 {
        v.clamp(min, max)
    }
    pub fn lerp(a: f32, b: f32, t: f32) -> f32 {
        a + (b - a) * t
    }
    #[wasm_bindgen(js_name = degToRad)]
    pub fn deg_to_rad(d: f32) -> f32 {
        d.to_radians()
    }
    #[wasm_bindgen(js_name = radToDeg)]
    pub fn rad_to_deg(r: f32) -> f32 {
        r.to_degrees()
    }
    #[wasm_bindgen(js_name = mapLinear)]
    pub fn map_linear(x: f32, a1: f32, a2: f32, b1: f32, b2: f32) -> f32 {
        b1 + (x - a1) * (b2 - b1) / (a2 - a1)
    }
    #[wasm_bindgen(js_name = smoothstep)]
    pub fn smoothstep(x: f32, min: f32, max: f32) -> f32 {
        if x <= min {
            return 0.0;
        }
        if x >= max {
            return 1.0;
        }
        let t = (x - min) / (max - min);
        t * t * (3.0 - 2.0 * t)
    }
    #[wasm_bindgen(js_name = euclideanModulo)]
    pub fn euclidean_modulo(n: f32, m: f32) -> f32 {
        ((n % m) + m) % m
    }
    #[wasm_bindgen(js_name = isPowerOfTwo)]
    pub fn is_power_of_two(n: u32) -> bool {
        n != 0 && (n & (n - 1)) == 0
    }
}
