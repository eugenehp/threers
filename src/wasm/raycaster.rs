//! Bindings split out of `wasm.rs` by area (see the `mod` list there): raycaster.

use super::*;

// ---- raycaster ----
#[wasm_bindgen]
pub struct WebRaycaster {
    pub(super) inner: crate::Raycaster,
}
#[wasm_bindgen]
impl WebRaycaster {
    #[wasm_bindgen(constructor)]
    pub fn new() -> WebRaycaster {
        WebRaycaster {
            inner: crate::Raycaster::default(),
        }
    }
    /// three.js `setFromCamera`: the ray through NDC (`x`, `y` in -1..1, +y
    /// up), perspective or orthographic, and remembers the camera, which
    /// sprites need.
    #[wasm_bindgen(js_name = setFromCamera)]
    pub fn set_from_camera(&mut self, ndc_x: f32, ndc_y: f32, camera: &WebCamera) {
        let ndc = crate::Vector2::new(ndc_x, ndc_y);
        match &camera.inner {
            CameraInner::Perspective(c) => self.inner.set_from_camera(ndc, c),
            CameraInner::Orthographic(c) => self.inner.set_from_camera(ndc, c),
        }
    }

    /// three.js `set`: a new origin and direction. The direction is
    /// normalised here (three.js assumes the caller did).
    pub fn set(&mut self, origin: &WebVector3, direction: &WebVector3) {
        self.inner.set(origin.inner, direction.inner.normalize());
    }

    /// The ray origin.
    #[wasm_bindgen(getter)]
    pub fn origin(&self) -> WebVector3 {
        WebVector3 {
            inner: self.inner.ray.origin,
        }
    }

    /// The (unit) ray direction.
    #[wasm_bindgen(getter)]
    pub fn direction(&self) -> WebVector3 {
        WebVector3 {
            inner: self.inner.ray.direction,
        }
    }

    /// Hits nearer than this are dropped (three.js `near`, default 0).
    #[wasm_bindgen(getter)]
    pub fn near(&self) -> f32 {
        self.inner.near
    }
    #[wasm_bindgen(setter)]
    pub fn set_near(&mut self, near: f32) {
        self.inner.near = near;
    }

    /// Hits farther than this are dropped (three.js `far`, default Infinity).
    #[wasm_bindgen(getter)]
    pub fn far(&self) -> f32 {
        self.inner.far
    }
    #[wasm_bindgen(setter)]
    pub fn set_far(&mut self, far: f32) {
        self.inner.far = far;
    }

    /// three.js `params.Line.threshold`: how close (world units) the ray must
    /// pass to a line segment to hit it. Default 1.
    #[wasm_bindgen(getter, js_name = lineThreshold)]
    pub fn line_threshold(&self) -> f32 {
        self.inner.params.line.threshold
    }
    #[wasm_bindgen(js_name = setLineThreshold)]
    pub fn set_line_threshold(&mut self, threshold: f32) {
        self.inner.params.line.threshold = threshold;
    }

    /// three.js `params.Points.threshold`. Default 1.
    #[wasm_bindgen(getter, js_name = pointsThreshold)]
    pub fn points_threshold(&self) -> f32 {
        self.inner.params.points.threshold
    }
    #[wasm_bindgen(js_name = setPointsThreshold)]
    pub fn set_points_threshold(&mut self, threshold: f32) {
        self.inner.params.points.threshold = threshold;
    }

    /// Return only the nearest hit of the whole cast (three-mesh-bvh's
    /// `firstHitOnly`, applied across objects rather than per mesh), and
    /// prune the search by it.
    #[wasm_bindgen(getter, js_name = firstHitOnly)]
    pub fn first_hit_only(&self) -> bool {
        self.inner.first_hit_only
    }
    #[wasm_bindgen(setter, js_name = firstHitOnly)]
    pub fn set_first_hit_only(&mut self, on: bool) {
        self.inner.first_hit_only = on;
    }

    /// three.js `intersectObject(object, recursive)`: hits on the object with
    /// id `object_id` (`WebObjectHandle.id`, or `scene.rootId` for the whole
    /// scene) and, when `recursive` (default true), its descendants, nearest
    /// first. Invisible objects and their subtrees are skipped.
    ///
    /// Each hit is a plain object with three.js's field names: `distance`,
    /// `point`, `objectId`, and per kind `faceIndex`, `face {a, b, c, normal,
    /// materialIndex}`, `normal`, `uv`, `uv1`, `instanceId`, `index`,
    /// `distanceToRay`, plus the threers extensions `polygonIndex` and
    /// `barycoord`. A field that does not apply is absent (undefined), as in
    /// three.js. Vectors are `{x, y, z}` / `{x, y}`.
    #[wasm_bindgen(js_name = intersectObject)]
    pub fn intersect_object(
        &self,
        scene: &mut WebScene,
        object_id: f64,
        recursive: Option<bool>,
    ) -> Result<js_sys::Array, JsValue> {
        self.intersect_objects(scene, vec![object_id], recursive)
    }

    /// three.js `intersectObjects(objects, recursive)`: as `intersectObject`
    /// over every id in `object_ids`, all hits sorted together.
    #[wasm_bindgen(js_name = intersectObjects)]
    pub fn intersect_objects(
        &self,
        scene: &mut WebScene,
        object_ids: Vec<f64>,
        recursive: Option<bool>,
    ) -> Result<js_sys::Array, JsValue> {
        let ids = object_ids
            .iter()
            .map(|&id| {
                object_id_from_f64(id)
                    .filter(|id| scene.inner.arena.get(*id).is_some())
                    .ok_or_else(|| JsValue::from_str(&format!("intersectObjects: no object {id}")))
            })
            .collect::<Result<Vec<_>, _>>()?;
        scene.inner.update_world();
        let hits =
            self.inner
                .intersect_object_list(&scene.inner.arena, &ids, recursive.unwrap_or(true));
        let out = js_sys::Array::new();
        for h in &hits {
            out.push(&intersection_to_js(h));
        }
        Ok(out)
    }
}

pub(super) fn js_set(obj: &js_sys::Object, key: &str, value: &JsValue) {
    let _ = js_sys::Reflect::set(obj, &JsValue::from_str(key), value);
}

pub(super) fn js_vec3(v: crate::Vector3) -> JsValue {
    let o = js_sys::Object::new();
    js_set(&o, "x", &v.x.into());
    js_set(&o, "y", &v.y.into());
    js_set(&o, "z", &v.z.into());
    o.into()
}

pub(super) fn js_vec2(v: crate::Vector2) -> JsValue {
    let o = js_sys::Object::new();
    js_set(&o, "x", &v.x.into());
    js_set(&o, "y", &v.y.into());
    o.into()
}

/// A hit as three.js reports it; fields that do not apply are left out.
pub(super) fn intersection_to_js(h: &crate::core::Intersection) -> JsValue {
    let o = js_sys::Object::new();
    js_set(&o, "distance", &h.distance.into());
    js_set(&o, "point", &js_vec3(h.point));
    js_set(&o, "objectId", &object_id_to_f64(h.object).into());
    if let Some(f) = &h.face {
        js_set(&o, "faceIndex", &(h.face_index as f64).into());
        let face = js_sys::Object::new();
        js_set(&face, "a", &(f.a as f64).into());
        js_set(&face, "b", &(f.b as f64).into());
        js_set(&face, "c", &(f.c as f64).into());
        js_set(&face, "normal", &js_vec3(f.normal));
        js_set(&face, "materialIndex", &(f.material_index as f64).into());
        js_set(&o, "face", &face.into());
    }
    if let Some(p) = h.polygon_index {
        js_set(&o, "polygonIndex", &(p as f64).into());
    }
    if let Some(n) = h.normal {
        js_set(&o, "normal", &js_vec3(n));
    }
    if let Some(uv) = h.uv {
        js_set(&o, "uv", &js_vec2(uv));
    }
    if let Some(uv) = h.uv1 {
        js_set(&o, "uv1", &js_vec2(uv));
    }
    if let Some(b) = h.barycoord {
        js_set(&o, "barycoord", &js_vec3(b));
    }
    if let Some(i) = h.instance_id {
        js_set(&o, "instanceId", &(i as f64).into());
    }
    if let Some(i) = h.index {
        js_set(&o, "index", &(i as f64).into());
    }
    if let Some(d) = h.distance_to_ray {
        js_set(&o, "distanceToRay", &d.into());
    }
    o.into()
}
