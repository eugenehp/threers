//! Bindings split out of `wasm.rs` by area (see the `mod` list there): the
//! controls whose logic is three.js r165's, run in Rust — `TransformControls`
//! (`controls/transform`), `FlyControls` (`controls/fly.rs`) — and
//! `OrbitControls`' `screenSpacePanning`, which `MapControls` turns off.
//!
//! The shim's classes are the DOM glue: they forward pointer and key events,
//! hand over the object's and camera's matrices (as `Float64Array`s, the
//! numbers JavaScript holds), and apply the poses and events that come back.
#![allow(unused_imports)]

use super::*;
use crate::controls::transform::{
    gizmo::{Group, MATERIALS},
    CameraState, ObjectState, Pointer, PropValue, TcEvent,
};
use crate::controls::{FlyControls, TransformControls};

fn m4(s: &[f64]) -> [f64; 16] {
    let mut m = [0.0; 16];
    for (d, v) in m.iter_mut().zip(s) {
        *d = *v;
    }
    m
}

/// `[matrixWorld ×16, projectionMatrixInverse ×16, orthographic, near, far,
/// fov, zoom, top, bottom]`.
fn camera_of(c: &[f64]) -> CameraState {
    let at = |i: usize, d: f64| c.get(i).copied().unwrap_or(d);
    CameraState {
        matrix_world: m4(c.get(0..16).unwrap_or(&[])),
        projection_matrix_inverse: m4(c.get(16..32).unwrap_or(&[])),
        orthographic: at(32, 0.0) != 0.0,
        near: at(33, 0.1),
        far: at(34, 2000.0),
        fov: at(35, 50.0),
        zoom: at(36, 1.0),
        top: at(37, 1.0),
        bottom: at(38, -1.0),
    }
}

/// `[position ×3, quaternion ×4, scale ×3, matrixWorld ×16, hasParent,
/// parent.matrixWorld ×16]`; `None` for no object.
fn object_of(o: &Option<Vec<f64>>) -> Option<ObjectState> {
    let o = o.as_deref()?;
    if o.len() < 27 {
        return None;
    }
    Some(ObjectState {
        position: [o[0], o[1], o[2]],
        quaternion: [o[3], o[4], o[5], o[6]],
        scale: [o[7], o[8], o[9]],
        matrix_world: m4(&o[10..26]),
        parent_matrix_world: (o[26] != 0.0 && o.len() >= 43).then(|| m4(&o[27..43])),
    })
}

fn prop_of(v: &JsValue) -> PropValue {
    if let Some(b) = v.as_bool() {
        PropValue::Bool(b)
    } else if let Some(n) = v.as_f64() {
        PropValue::Num(n)
    } else if let Some(s) = v.as_string() {
        PropValue::Str(s)
    } else {
        PropValue::Null
    }
}

fn prop_to_js(v: &PropValue) -> JsValue {
    match v {
        PropValue::Null => JsValue::NULL,
        PropValue::Bool(b) => JsValue::from_bool(*b),
        PropValue::Num(n) => JsValue::from_f64(*n),
        PropValue::Str(s) => JsValue::from_str(s),
    }
}

/// Events for JS, in order: `{ type: 'prop', name, value }`,
/// `{ type: 'position' | 'quaternion' | 'scale', value: [...] }`,
/// `{ type: 'change' | 'objectChange' }`, `{ type: 'mouseDown' | 'mouseUp', mode }`.
fn events_to_js(events: &[TcEvent]) -> js_sys::Array {
    let out = js_sys::Array::new();
    let nums = |v: &[f64]| -> JsValue { js_sys::Float64Array::from(v).into() };
    for e in events {
        let o = js_sys::Object::new();
        let (kind, extra): (&str, Option<(&str, JsValue)>) = match e {
            TcEvent::Prop(name, value) => {
                js_set(&o, "name", &JsValue::from_str(name));
                ("prop", Some(("value", prop_to_js(value))))
            }
            TcEvent::SetPosition(p) => ("position", Some(("value", nums(p)))),
            TcEvent::SetQuaternion(q) => ("quaternion", Some(("value", nums(q)))),
            TcEvent::SetScale(s) => ("scale", Some(("value", nums(s)))),
            TcEvent::Change => ("change", None),
            TcEvent::ObjectChange => ("objectChange", None),
            TcEvent::MouseDown(m) => ("mouseDown", Some(("mode", JsValue::from_str(m)))),
            TcEvent::MouseUp(m) => ("mouseUp", Some(("mode", JsValue::from_str(m)))),
        };
        js_set(&o, "type", &JsValue::from_str(kind));
        if let Some((k, v)) = extra {
            js_set(&o, k, &v);
        }
        out.push(&o.into());
    }
    out
}

fn pointer_of(has: bool, x: f64, y: f64, button: i32) -> Option<Pointer> {
    has.then_some(Pointer { x, y, button })
}

/// three.js r165 `TransformControls`, its gizmo and its drag plane.
#[wasm_bindgen]
pub struct WebTransformControls {
    inner: TransformControls,
}

#[wasm_bindgen]
impl WebTransformControls {
    #[wasm_bindgen(constructor)]
    pub fn new() -> WebTransformControls {
        WebTransformControls {
            inner: TransformControls::new(),
        }
    }

    /// How many handles the gizmo has (all modes and groups).
    #[wasm_bindgen(js_name = handleCount)]
    pub fn handle_count(&self) -> usize {
        self.inner.handles().len()
    }

    /// `{ group: 'gizmo'|'picker'|'helper', mode, name, tag, line, material }`
    /// for handle `i`, in the order three.js adds them.
    #[wasm_bindgen(js_name = handleInfo)]
    pub fn handle_info(&self, i: usize) -> JsValue {
        let Some(h) = self.inner.handles().get(i) else {
            return JsValue::UNDEFINED;
        };
        let o = js_sys::Object::new();
        let group = match h.group {
            Group::Gizmo => "gizmo",
            Group::Picker => "picker",
            Group::Helper => "helper",
        };
        js_set(&o, "group", &JsValue::from_str(group));
        js_set(&o, "mode", &JsValue::from_str(h.mode.name()));
        js_set(&o, "name", &JsValue::from_str(h.name));
        js_set(
            &o,
            "tag",
            &if h.helper {
                JsValue::from_str("helper")
            } else {
                JsValue::UNDEFINED
            },
        );
        js_set(&o, "line", &JsValue::from_bool(h.line));
        js_set(&o, "material", &JsValue::from_f64(h.material as f64));
        o.into()
    }

    /// Handle `i`'s baked geometry: `{ position, normal?, uv?, index? }`.
    #[wasm_bindgen(js_name = handleGeometry)]
    pub fn handle_geometry(&self, i: usize) -> JsValue {
        match self.inner.handles().get(i) {
            Some(h) => geometry_to_js(&h.geometry),
            None => JsValue::UNDEFINED,
        }
    }

    /// The drag plane's geometry, `PlaneGeometry(100000, 100000, 2, 2)`.
    #[wasm_bindgen(js_name = planeGeometry)]
    pub fn plane_geometry() -> JsValue {
        geometry_to_js(&crate::controls::transform::gizmo::plane())
    }

    /// `{ line, color, opacity }` for material `i` as three.js creates it.
    #[wasm_bindgen(js_name = materialInfo)]
    pub fn material_info(i: usize) -> JsValue {
        let Some(m) = MATERIALS.get(i) else {
            return JsValue::UNDEFINED;
        };
        let o = js_sys::Object::new();
        js_set(&o, "line", &JsValue::from_bool(m.line));
        js_set(&o, "color", &JsValue::from_f64(m.color as f64));
        js_set(&o, "opacity", &JsValue::from_f64(m.opacity));
        o.into()
    }

    #[wasm_bindgen(js_name = materialCount)]
    pub fn material_count() -> usize {
        MATERIALS.len()
    }

    /// A property by its three.js name (`axis`, `mode`, `space`, `size`,
    /// `dragging`, `enabled`, `translationSnap`, `rotationSnap`,
    /// `scaleSnap`, `showX/Y/Z`, `rotationAngle`).
    #[wasm_bindgen(js_name = getProperty)]
    pub fn get_property(&self, name: &str) -> JsValue {
        let t = &self.inner;
        let opt = |v: Option<f64>| v.map(JsValue::from_f64).unwrap_or(JsValue::NULL);
        let [x, y, z] = t.show();
        match name {
            "enabled" => JsValue::from_bool(t.enabled),
            "axis" => t.axis().map(JsValue::from_str).unwrap_or(JsValue::NULL),
            "mode" => JsValue::from_str(t.mode()),
            "space" => JsValue::from_str(t.space()),
            "size" => JsValue::from_f64(t.size()),
            "dragging" => JsValue::from_bool(t.dragging()),
            "translationSnap" => opt(t.translation_snap()),
            "rotationSnap" => opt(t.rotation_snap()),
            "scaleSnap" => opt(t.scale_snap()),
            "showX" => JsValue::from_bool(x),
            "showY" => JsValue::from_bool(y),
            "showZ" => JsValue::from_bool(z),
            "rotationAngle" => JsValue::from_f64(t.rotation_angle()),
            _ => JsValue::UNDEFINED,
        }
    }

    /// Set a property; the events three.js's setter dispatches.
    #[wasm_bindgen(js_name = setProperty)]
    pub fn set_property(&mut self, name: &str, value: JsValue) -> js_sys::Array {
        events_to_js(&self.inner.set_property(name, prop_of(&value)))
    }

    #[wasm_bindgen(js_name = setHasObject)]
    pub fn set_has_object(&mut self, on: bool) {
        self.inner.set_has_object(on);
    }

    pub fn detach(&mut self) -> js_sys::Array {
        events_to_js(&self.inner.detach())
    }

    /// `updateMatrixWorld`: the object (see the shim for the layout; omit
    /// for none), the camera, and the control's own `matrixWorld`.
    #[wasm_bindgen(js_name = updateMatrixWorld)]
    pub fn update_matrix_world(
        &mut self,
        object: Option<Vec<f64>>,
        camera: &[f64],
        controls_world: &[f64],
    ) {
        let o = object_of(&object);
        self.inner
            .update_matrix_world(o.as_ref(), &camera_of(camera), &m4(controls_world));
    }

    /// Per handle `[px, py, pz, qx, qy, qz, qw, sx, sy, sz, visible]`.
    #[wasm_bindgen(js_name = handleStates)]
    pub fn handle_states(&self) -> Vec<f64> {
        let mut out = Vec::with_capacity(self.inner.handle_states().len() * 11);
        for s in self.inner.handle_states() {
            out.extend_from_slice(&s.position);
            out.extend_from_slice(&s.quaternion);
            out.extend_from_slice(&s.scale);
            out.push(if s.visible { 1.0 } else { 0.0 });
        }
        out
    }

    /// Per material `[color, opacity]`.
    #[wasm_bindgen(js_name = materialStates)]
    pub fn material_states(&self) -> Vec<f64> {
        self.inner
            .material_states()
            .iter()
            .flat_map(|m| [m.color as f64, m.opacity])
            .collect()
    }

    /// The drag plane's `[px, py, pz, qx, qy, qz, qw]`.
    #[wasm_bindgen(js_name = planeTransform)]
    pub fn plane_transform(&self) -> Vec<f64> {
        let (p, q) = self.inner.plane_transform();
        vec![p[0], p[1], p[2], q[0], q[1], q[2], q[3]]
    }

    /// `worldPosition, worldPositionStart, worldQuaternion,
    /// worldQuaternionStart, cameraPosition, cameraQuaternion, pointStart,
    /// pointEnd, rotationAxis, eye` flattened (33 numbers).
    #[wasm_bindgen(js_name = worldState)]
    pub fn world_state(&self) -> Vec<f64> {
        let t = &self.inner;
        let mut v = Vec::with_capacity(33);
        v.extend_from_slice(&t.world_position);
        v.extend_from_slice(&t.world_position_start);
        v.extend_from_slice(&t.world_quaternion);
        v.extend_from_slice(&t.world_quaternion_start);
        v.extend_from_slice(&t.camera_position);
        v.extend_from_slice(&t.camera_quaternion);
        v.extend_from_slice(&t.point_start);
        v.extend_from_slice(&t.point_end);
        v.extend_from_slice(&t.rotation_axis);
        v.extend_from_slice(&t.eye);
        v
    }

    /// Whether `pointerHover` casts (an object attached, not dragging).
    #[wasm_bindgen(js_name = wantsHover)]
    pub fn wants_hover(&self) -> bool {
        self.inner.wants_hover()
    }

    /// `pointerHover` with the name of the picker handle the cast hit.
    #[wasm_bindgen(js_name = pointerHover)]
    pub fn pointer_hover(&mut self, hit: Option<String>) -> js_sys::Array {
        events_to_js(&self.inner.pointer_hover(hit.as_deref()))
    }

    #[wasm_bindgen(js_name = pointerDown)]
    #[allow(clippy::too_many_arguments)]
    pub fn pointer_down(
        &mut self,
        has_pointer: bool,
        x: f64,
        y: f64,
        button: i32,
        object: Option<Vec<f64>>,
        camera: &[f64],
    ) -> js_sys::Array {
        let o = object_of(&object);
        events_to_js(&self.inner.pointer_down(
            pointer_of(has_pointer, x, y, button),
            o.as_ref(),
            &camera_of(camera),
        ))
    }

    #[wasm_bindgen(js_name = pointerMove)]
    #[allow(clippy::too_many_arguments)]
    pub fn pointer_move(
        &mut self,
        has_pointer: bool,
        x: f64,
        y: f64,
        button: i32,
        object: Option<Vec<f64>>,
        camera: &[f64],
    ) -> js_sys::Array {
        let o = object_of(&object);
        events_to_js(&self.inner.pointer_move(
            pointer_of(has_pointer, x, y, button),
            o.as_ref(),
            &camera_of(camera),
        ))
    }

    #[wasm_bindgen(js_name = pointerUp)]
    pub fn pointer_up(&mut self, has_pointer: bool, button: i32) -> js_sys::Array {
        events_to_js(
            &self
                .inner
                .pointer_up(pointer_of(has_pointer, 0.0, 0.0, button)),
        )
    }

    pub fn reset(&mut self) -> js_sys::Array {
        events_to_js(&self.inner.reset())
    }
}

impl Default for WebTransformControls {
    fn default() -> Self {
        Self::new()
    }
}

fn geometry_to_js(g: &crate::controls::transform::gizmo::GizmoGeometry) -> JsValue {
    let o = js_sys::Object::new();
    js_set(
        &o,
        "position",
        &js_sys::Float32Array::from(&g.position[..]).into(),
    );
    if !g.normal.is_empty() {
        js_set(
            &o,
            "normal",
            &js_sys::Float32Array::from(&g.normal[..]).into(),
        );
    }
    if !g.uv.is_empty() {
        js_set(&o, "uv", &js_sys::Float32Array::from(&g.uv[..]).into());
    }
    if let Some(i) = &g.index {
        js_set(&o, "index", &js_sys::Uint32Array::from(&i[..]).into());
    }
    o.into()
}

/// three.js r165 `FlyControls`.
#[wasm_bindgen]
pub struct WebFlyControls {
    inner: FlyControls,
}

#[wasm_bindgen]
impl WebFlyControls {
    #[wasm_bindgen(constructor)]
    pub fn new() -> WebFlyControls {
        WebFlyControls {
            inner: FlyControls::new(),
        }
    }

    #[wasm_bindgen(getter)]
    pub fn enabled(&self) -> bool {
        self.inner.enabled
    }
    #[wasm_bindgen(setter)]
    pub fn set_enabled(&mut self, v: bool) {
        self.inner.enabled = v;
    }
    #[wasm_bindgen(getter, js_name = movementSpeed)]
    pub fn movement_speed(&self) -> f64 {
        self.inner.movement_speed
    }
    #[wasm_bindgen(setter, js_name = movementSpeed)]
    pub fn set_movement_speed(&mut self, v: f64) {
        self.inner.movement_speed = v;
    }
    #[wasm_bindgen(getter, js_name = rollSpeed)]
    pub fn roll_speed(&self) -> f64 {
        self.inner.roll_speed
    }
    #[wasm_bindgen(setter, js_name = rollSpeed)]
    pub fn set_roll_speed(&mut self, v: f64) {
        self.inner.roll_speed = v;
    }
    #[wasm_bindgen(getter, js_name = dragToLook)]
    pub fn drag_to_look(&self) -> bool {
        self.inner.drag_to_look
    }
    #[wasm_bindgen(setter, js_name = dragToLook)]
    pub fn set_drag_to_look(&mut self, v: bool) {
        self.inner.drag_to_look = v;
    }
    #[wasm_bindgen(getter, js_name = autoForward)]
    pub fn auto_forward(&self) -> bool {
        self.inner.auto_forward
    }
    #[wasm_bindgen(setter, js_name = autoForward)]
    pub fn set_auto_forward(&mut self, v: bool) {
        self.inner.auto_forward = v;
    }
    #[wasm_bindgen(getter, js_name = movementSpeedMultiplier)]
    pub fn movement_speed_multiplier(&self) -> f64 {
        self.inner.movement_speed_multiplier
    }
    #[wasm_bindgen(getter)]
    pub fn status(&self) -> i32 {
        self.inner.status
    }

    /// `moveState` as `[up, down, left, right, forward, back, pitchUp,
    /// pitchDown, yawLeft, yawRight, rollLeft, rollRight]`.
    #[wasm_bindgen(js_name = moveState)]
    pub fn move_state(&self) -> Vec<f64> {
        let s = &self.inner.move_state;
        vec![
            s.up,
            s.down,
            s.left,
            s.right,
            s.forward,
            s.back,
            s.pitch_up,
            s.pitch_down,
            s.yaw_left,
            s.yaw_right,
            s.roll_left,
            s.roll_right,
        ]
    }

    /// Replace `moveState` (same order as [`move_state`](Self::move_state)).
    #[wasm_bindgen(js_name = setMoveState)]
    pub fn set_move_state(&mut self, v: &[f64]) {
        if v.len() < 12 {
            return;
        }
        let s = &mut self.inner.move_state;
        [
            s.up,
            s.down,
            s.left,
            s.right,
            s.forward,
            s.back,
            s.pitch_up,
            s.pitch_down,
            s.yaw_left,
            s.yaw_right,
            s.roll_left,
            s.roll_right,
        ] = [
            v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8], v[9], v[10], v[11],
        ];
    }

    /// `[moveVector ×3, rotationVector ×3]`.
    pub fn vectors(&self) -> Vec<f64> {
        let (m, r) = (self.inner.move_vector, self.inner.rotation_vector);
        vec![m[0], m[1], m[2], r[0], r[1], r[2]]
    }

    #[wasm_bindgen(js_name = updateMovementVector)]
    pub fn update_movement_vector(&mut self) {
        self.inner.update_movement_vector();
    }
    #[wasm_bindgen(js_name = updateRotationVector)]
    pub fn update_rotation_vector(&mut self) {
        self.inner.update_rotation_vector();
    }

    #[wasm_bindgen(js_name = keyDown)]
    pub fn key_down(&mut self, code: &str, alt: bool) {
        self.inner.key_down(code, alt);
    }
    #[wasm_bindgen(js_name = keyUp)]
    pub fn key_up(&mut self, code: &str) {
        self.inner.key_up(code);
    }
    #[wasm_bindgen(js_name = pointerDown)]
    pub fn pointer_down(&mut self, button: i32) {
        self.inner.pointer_down(button);
    }
    #[wasm_bindgen(js_name = pointerMove)]
    #[allow(clippy::too_many_arguments)]
    pub fn pointer_move(&mut self, page_x: f64, page_y: f64, w: f64, h: f64, ox: f64, oy: f64) {
        self.inner.pointer_move(page_x, page_y, [w, h], [ox, oy]);
    }
    #[wasm_bindgen(js_name = pointerUp)]
    pub fn pointer_up(&mut self, button: i32) {
        self.inner.pointer_up(button);
    }
    #[wasm_bindgen(js_name = pointerCancel)]
    pub fn pointer_cancel(&mut self) {
        self.inner.pointer_cancel();
    }

    /// `update(delta)` on `[px, py, pz, qx, qy, qz, qw]`: the new pose and
    /// whether `change` fires (`[…7, changed]`), or `undefined` when disabled.
    pub fn update(&mut self, delta: f64, pose: &[f64]) -> Option<Vec<f64>> {
        if pose.len() < 7 {
            return None;
        }
        let (p, q, changed) = self.inner.update(
            delta,
            [pose[0], pose[1], pose[2]],
            [pose[3], pose[4], pose[5], pose[6]],
        )?;
        Some(vec![
            p[0],
            p[1],
            p[2],
            q[0],
            q[1],
            q[2],
            q[3],
            if changed { 1.0 } else { 0.0 },
        ])
    }
}

impl Default for WebFlyControls {
    fn default() -> Self {
        Self::new()
    }
}

#[wasm_bindgen]
impl WebOrbitControls {
    /// three.js `screenSpacePanning` (`MapControls` sets it false).
    #[wasm_bindgen(js_name = setScreenSpacePanning)]
    pub fn set_screen_space_panning(&mut self, on: bool) {
        self.inner.screen_space_panning = on;
    }
}
