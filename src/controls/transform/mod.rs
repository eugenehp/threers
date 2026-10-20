//! three.js r165 `TransformControls`: the translate / rotate / scale gizmo.
//!
//! Everything the control computes lives here, in `f64` and in three.js's
//! order of operations: the world state `updateMatrixWorld` derives from the
//! attached object and the camera, the appearance of every handle (placement,
//! camera-facing hiding, `showX/Y/Z`, highlighting), the orientation of the
//! invisible drag plane, the ray cast onto that plane, and the drag itself
//! (axis and plane translation in world or local space, snapping, uniform and
//! per-axis scaling, axis / trackball / screen rotation).
//!
//! The caller owns the scene: it hands in the attached object's transform and
//! the camera's matrices, the name of the picker handle a hover cast hit, and
//! applies what comes back — [`TcEvent`]s, in the order three.js would
//! dispatch them or touch the object. [`gizmo::handles`] builds the handles'
//! geometry exactly as three.js does, and [`TransformControls::handle_states`]
//! says where each one is and how it looks after an update.

pub mod gizmo;
pub mod math;

use gizmo::{Group, Handle, Mode, MATERIALS};
use math::{M4, Q, V3};

/// A property's value, as three.js's `<name>-changed` event carries it.
#[derive(Debug, Clone, PartialEq)]
pub enum PropValue {
    Null,
    Bool(bool),
    Num(f64),
    Str(String),
}

/// What the control does, in order: property changes (three.js dispatches
/// `<name>-changed` then `change` for each), the attached object's
/// position / quaternion / scale being written, and the control's own events.
#[derive(Debug, Clone, PartialEq)]
pub enum TcEvent {
    Prop(&'static str, PropValue),
    SetPosition(V3),
    SetQuaternion(Q),
    SetScale(V3),
    Change,
    ObjectChange,
    MouseDown(String),
    MouseUp(String),
}

/// The camera as the control reads it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CameraState {
    pub matrix_world: M4,
    pub projection_matrix_inverse: M4,
    pub orthographic: bool,
    pub near: f64,
    pub far: f64,
    /// Perspective: vertical field of view in degrees.
    pub fov: f64,
    pub zoom: f64,
    /// Orthographic frustum top / bottom.
    pub top: f64,
    pub bottom: f64,
}

impl Default for CameraState {
    fn default() -> Self {
        Self {
            matrix_world: math::IDENTITY_M4,
            projection_matrix_inverse: math::IDENTITY_M4,
            orthographic: false,
            near: 0.1,
            far: 2000.0,
            fov: 50.0,
            zoom: 1.0,
            top: 1.0,
            bottom: -1.0,
        }
    }
}

/// The attached object: its local transform, its `matrixWorld`, and its
/// parent's (`None` when it has no parent).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ObjectState {
    pub position: V3,
    pub quaternion: Q,
    pub scale: V3,
    pub matrix_world: M4,
    pub parent_matrix_world: Option<M4>,
}

/// A pointer in normalised device coordinates, with three.js's `button`
/// (`-1` for a move with no button, as `PointerEvent.button` reports).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Pointer {
    pub x: f64,
    pub y: f64,
    pub button: i32,
}

/// Where a handle is and how it shows after an update: local position,
/// quaternion and scale under the gizmo, and visibility.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HandleState {
    pub position: V3,
    pub quaternion: Q,
    pub scale: V3,
    pub visible: bool,
}

impl Default for HandleState {
    fn default() -> Self {
        Self {
            position: [0.0; 3],
            quaternion: math::IDENTITY_Q,
            scale: [1.0; 3],
            visible: true,
        }
    }
}

/// A gizmo material's colour and opacity after an update.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MaterialState {
    pub color: u32,
    pub opacity: f64,
}

const UNIT_X: V3 = [1.0, 0.0, 0.0];
const UNIT_Y: V3 = [0.0, 1.0, 0.0];
const UNIT_Z: V3 = [0.0, 0.0, 1.0];

fn unit(axis: &str) -> V3 {
    match axis {
        "X" => UNIT_X,
        "Y" => UNIT_Y,
        _ => UNIT_Z,
    }
}

/// `if (snap)`: set, non-zero, not NaN.
fn truthy(v: Option<f64>) -> Option<f64> {
    v.filter(|s| *s != 0.0 && !s.is_nan())
}

/// JavaScript's `Math.round`: half rounds up.
fn js_round(x: f64) -> f64 {
    let f = x.floor();
    if x - f >= 0.5 {
        f + 1.0
    } else {
        f
    }
}

/// three.js r165 `TransformControls` (with its `TransformControlsGizmo` and
/// `TransformControlsPlane`).
#[derive(Debug, Clone)]
pub struct TransformControls {
    // ---- defineProperty'd state ----
    pub enabled: bool,
    axis: Option<String>,
    mode: String,
    translation_snap: Option<f64>,
    rotation_snap: Option<f64>,
    scale_snap: Option<f64>,
    space: String,
    size: f64,
    dragging: bool,
    show_x: bool,
    show_y: bool,
    show_z: bool,
    has_object: bool,

    pub world_position: V3,
    pub world_position_start: V3,
    pub world_quaternion: Q,
    pub world_quaternion_start: Q,
    pub camera_position: V3,
    pub camera_quaternion: Q,
    pub point_start: V3,
    pub point_end: V3,
    pub rotation_axis: V3,
    rotation_angle: f64,
    pub eye: V3,

    // ---- private ----
    offset: V3,
    start_norm: V3,
    end_norm: V3,
    camera_scale: V3,
    parent_position: V3,
    parent_quaternion: Q,
    parent_quaternion_inv: Q,
    parent_scale: V3,
    world_scale_start: V3,
    world_quaternion_inv: Q,
    world_scale: V3,
    position_start: V3,
    quaternion_start: Q,
    scale_start: V3,

    // ---- gizmo and plane ----
    camera: CameraState,
    handles: Vec<Handle>,
    handle_states: Vec<HandleState>,
    materials: Vec<MaterialState>,
    plane_position: V3,
    plane_quaternion: Q,
    plane_matrix_world: M4,
    /// three.js's module-level `_dirVector`, which keeps its value between
    /// plane updates that do not set it.
    dir_vector: V3,
}

impl Default for TransformControls {
    fn default() -> Self {
        Self::new()
    }
}

impl TransformControls {
    pub fn new() -> Self {
        let handles = gizmo::handles();
        let handle_states = vec![HandleState::default(); handles.len()];
        Self {
            enabled: true,
            axis: None,
            mode: "translate".into(),
            translation_snap: None,
            rotation_snap: None,
            scale_snap: None,
            space: "world".into(),
            size: 1.0,
            dragging: false,
            show_x: true,
            show_y: true,
            show_z: true,
            has_object: false,
            world_position: [0.0; 3],
            world_position_start: [0.0; 3],
            world_quaternion: math::IDENTITY_Q,
            world_quaternion_start: math::IDENTITY_Q,
            camera_position: [0.0; 3],
            camera_quaternion: math::IDENTITY_Q,
            point_start: [0.0; 3],
            point_end: [0.0; 3],
            rotation_axis: [0.0; 3],
            rotation_angle: 0.0,
            eye: [0.0; 3],
            offset: [0.0; 3],
            start_norm: [0.0; 3],
            end_norm: [0.0; 3],
            camera_scale: [0.0; 3],
            parent_position: [0.0; 3],
            parent_quaternion: math::IDENTITY_Q,
            parent_quaternion_inv: math::IDENTITY_Q,
            parent_scale: [0.0; 3],
            world_scale_start: [0.0; 3],
            world_quaternion_inv: math::IDENTITY_Q,
            world_scale: [0.0; 3],
            position_start: [0.0; 3],
            quaternion_start: math::IDENTITY_Q,
            scale_start: [0.0; 3],
            camera: CameraState::default(),
            handles,
            handle_states,
            materials: MATERIALS
                .iter()
                .map(|m| MaterialState {
                    color: m.color,
                    opacity: m.opacity,
                })
                .collect(),
            plane_position: [0.0; 3],
            plane_quaternion: math::IDENTITY_Q,
            plane_matrix_world: math::IDENTITY_M4,
            dir_vector: [0.0; 3],
        }
    }

    // ------------------------------------------------------------------
    // Properties (three.js's defineProperty setters: an event on change)
    // ------------------------------------------------------------------

    pub fn axis(&self) -> Option<&str> {
        self.axis.as_deref()
    }
    pub fn mode(&self) -> &str {
        &self.mode
    }
    pub fn space(&self) -> &str {
        &self.space
    }
    pub fn size(&self) -> f64 {
        self.size
    }
    pub fn dragging(&self) -> bool {
        self.dragging
    }
    pub fn rotation_angle(&self) -> f64 {
        self.rotation_angle
    }
    pub fn translation_snap(&self) -> Option<f64> {
        self.translation_snap
    }
    pub fn rotation_snap(&self) -> Option<f64> {
        self.rotation_snap
    }
    pub fn scale_snap(&self) -> Option<f64> {
        self.scale_snap
    }
    pub fn show(&self) -> [bool; 3] {
        [self.show_x, self.show_y, self.show_z]
    }
    pub fn has_object(&self) -> bool {
        self.has_object
    }

    fn set_axis(&mut self, axis: Option<String>, ev: &mut Vec<TcEvent>) {
        if self.axis != axis {
            self.axis = axis;
            let v = self
                .axis
                .clone()
                .map(PropValue::Str)
                .unwrap_or(PropValue::Null);
            ev.push(TcEvent::Prop("axis", v));
        }
    }

    fn set_dragging(&mut self, on: bool, ev: &mut Vec<TcEvent>) {
        if self.dragging != on {
            self.dragging = on;
            ev.push(TcEvent::Prop("dragging", PropValue::Bool(on)));
        }
    }

    fn set_rotation_angle(&mut self, a: f64, ev: &mut Vec<TcEvent>) {
        // `!==`: NaN always counts as a change.
        if self.rotation_angle != a {
            self.rotation_angle = a;
            ev.push(TcEvent::Prop("rotationAngle", PropValue::Num(a)));
        }
    }

    /// Set a property by its three.js name. Returns the events three.js would
    /// dispatch (none when the value did not change). Unknown names and
    /// mistyped values are ignored.
    pub fn set_property(&mut self, name: &str, value: PropValue) -> Vec<TcEvent> {
        let mut ev = Vec::new();
        macro_rules! set {
            ($field:expr, $new:expr, $key:literal) => {{
                let new = $new;
                if $field != new {
                    $field = new;
                    ev.push(TcEvent::Prop($key, value.clone()));
                }
            }};
        }
        let num = |v: &PropValue| match v {
            PropValue::Num(n) => Some(*n),
            _ => None,
        };
        let flag = |v: &PropValue| match v {
            PropValue::Bool(b) => *b,
            PropValue::Num(n) => *n != 0.0 && !n.is_nan(),
            PropValue::Str(s) => !s.is_empty(),
            PropValue::Null => false,
        };
        match name {
            "enabled" => set!(self.enabled, flag(&value), "enabled"),
            "axis" => {
                let a = match &value {
                    PropValue::Str(s) => Some(s.clone()),
                    _ => None,
                };
                self.set_axis(a, &mut ev);
            }
            "mode" => {
                if let PropValue::Str(s) = &value {
                    set!(self.mode, s.clone(), "mode")
                }
            }
            "space" => {
                if let PropValue::Str(s) = &value {
                    set!(self.space, s.clone(), "space")
                }
            }
            "translationSnap" => set!(self.translation_snap, num(&value), "translationSnap"),
            "rotationSnap" => set!(self.rotation_snap, num(&value), "rotationSnap"),
            "scaleSnap" => set!(self.scale_snap, num(&value), "scaleSnap"),
            "size" => {
                if let Some(n) = num(&value) {
                    // `!==`: NaN always counts as a change.
                    if self.size != n {
                        self.size = n;
                        ev.push(TcEvent::Prop("size", value.clone()));
                    }
                }
            }
            "dragging" => self.set_dragging(flag(&value), &mut ev),
            "showX" => set!(self.show_x, flag(&value), "showX"),
            "showY" => set!(self.show_y, flag(&value), "showY"),
            "showZ" => set!(self.show_z, flag(&value), "showZ"),
            "rotationAngle" => {
                if let Some(n) = num(&value) {
                    self.set_rotation_angle(n, &mut ev);
                }
            }
            _ => {}
        }
        ev
    }

    /// `attach(object)`: whether an object is attached now. The caller keeps
    /// the object itself (and its `object-changed` event).
    pub fn set_has_object(&mut self, on: bool) {
        self.has_object = on;
    }

    /// `detach()`'s part here: `axis = null`.
    pub fn detach(&mut self) -> Vec<TcEvent> {
        let mut ev = Vec::new();
        self.has_object = false;
        self.set_axis(None, &mut ev);
        ev
    }

    // ------------------------------------------------------------------
    // updateMatrixWorld
    // ------------------------------------------------------------------

    /// `TransformControls.updateMatrixWorld` and its children's: world
    /// state from the object (already `updateMatrixWorld`'d) and camera, then
    /// every handle of the current mode and the drag plane. `controls_world`
    /// is the control's own `matrixWorld` (the plane's parent).
    pub fn update_matrix_world(
        &mut self,
        object: Option<&ObjectState>,
        camera: &CameraState,
        controls_world: &M4,
    ) {
        if let Some(o) = object {
            if let Some(pw) = &o.parent_matrix_world {
                let (p, q, s) = math::decompose(pw);
                self.parent_position = p;
                self.parent_quaternion = q;
                self.parent_scale = s;
            }
            let (p, q, s) = math::decompose(&o.matrix_world);
            self.world_position = p;
            self.world_quaternion = q;
            self.world_scale = s;
            self.parent_quaternion_inv = math::quat_invert(self.parent_quaternion);
            self.world_quaternion_inv = math::quat_invert(self.world_quaternion);
        }
        self.camera = *camera;
        let (p, q, s) = math::decompose(&camera.matrix_world);
        self.camera_position = p;
        self.camera_quaternion = q;
        self.camera_scale = s;
        if camera.orthographic {
            let e = &camera.matrix_world;
            // Camera.getWorldDirection, negated.
            let d = math::normalize([e[8], e[9], e[10]]);
            let d = [-d[0], -d[1], -d[2]];
            self.eye = [-d[0], -d[1], -d[2]];
        } else {
            self.eye = math::normalize(math::sub(self.camera_position, self.world_position));
        }
        self.update_gizmo();
        self.update_plane(controls_world);
    }

    fn update_gizmo(&mut self) {
        let Some(mode) = Mode::parse(&self.mode) else {
            return;
        };
        let space = if mode == Mode::Scale {
            "local"
        } else {
            self.space.as_str()
        };
        let quaternion = if space == "local" {
            self.world_quaternion
        } else {
            math::IDENTITY_Q
        };
        let factor = if self.camera.orthographic {
            (self.camera.top - self.camera.bottom) / self.camera.zoom
        } else {
            math::distance(self.world_position, self.camera_position)
                * (1.9 * math::js::tan(PI_F * self.camera.fov / 360.0) / self.camera.zoom).min(7.0)
        };
        // `_alignVector` and `_tempQuaternion` carry between handles.
        let order: Vec<usize> = [Group::Picker, Group::Gizmo, Group::Helper]
            .iter()
            .flat_map(|g| {
                self.handles
                    .iter()
                    .enumerate()
                    .filter(move |(_, h)| h.group == *g && h.mode == mode)
                    .map(|(i, _)| i)
            })
            .collect();
        let eye = self.eye;
        let axis = self.axis.clone();
        // three.js remembers each material's colour and opacity the first
        // time it sees it, in `_color` / `_opacity`; they are the table's.
        for i in order {
            let (name, helper, material) = {
                let h = &self.handles[i];
                (h.name, h.helper, h.material)
            };
            let mut st = HandleState {
                position: self.world_position,
                quaternion: math::IDENTITY_Q,
                scale: [factor * self.size / 4.0; 3],
                visible: true,
            };
            // `scale.set(1, 1, 1).multiplyScalar(f)` is `1 * f`.
            if helper {
                st.visible = false;
                match name {
                    "AXIS" => {
                        st.visible = axis.is_some();
                        match axis.as_deref() {
                            Some("X") => {
                                st.quaternion = math::quat_mul(
                                    quaternion,
                                    math::quat_from_euler_xyz(0.0, 0.0, 0.0),
                                );
                                if math::dot(math::apply_quat(UNIT_X, quaternion), eye).abs() > 0.9
                                {
                                    st.visible = false;
                                }
                            }
                            Some("Y") => {
                                st.quaternion = math::quat_mul(
                                    quaternion,
                                    math::quat_from_euler_xyz(0.0, 0.0, PI_F / 2.0),
                                );
                                if math::dot(math::apply_quat(UNIT_Y, quaternion), eye).abs() > 0.9
                                {
                                    st.visible = false;
                                }
                            }
                            Some("Z") => {
                                st.quaternion = math::quat_mul(
                                    quaternion,
                                    math::quat_from_euler_xyz(0.0, PI_F / 2.0, 0.0),
                                );
                                if math::dot(math::apply_quat(UNIT_Z, quaternion), eye).abs() > 0.9
                                {
                                    st.visible = false;
                                }
                            }
                            Some("XYZE") => {
                                let t = math::quat_from_euler_xyz(0.0, PI_F / 2.0, 0.0);
                                let mut m = math::IDENTITY_M4;
                                math::look_at(&mut m, [0.0; 3], self.rotation_axis, UNIT_Y);
                                st.quaternion =
                                    math::quat_mul(math::quat_from_rotation_matrix(&m), t);
                                st.visible = self.dragging;
                            }
                            Some("E") => st.visible = false,
                            _ => {}
                        }
                    }
                    "START" => {
                        st.position = self.world_position_start;
                        st.visible = self.dragging;
                    }
                    "END" => {
                        st.position = self.world_position;
                        st.visible = self.dragging;
                    }
                    "DELTA" => {
                        st.position = self.world_position_start;
                        st.quaternion = self.world_quaternion_start;
                        let mut t = math::add([1e-10; 3], self.world_position_start);
                        t = math::scale(math::sub(t, self.world_position), -1.0);
                        t = math::apply_quat(t, math::quat_invert(self.world_quaternion_start));
                        st.scale = t;
                        st.visible = self.dragging;
                    }
                    _ => {
                        st.quaternion = quaternion;
                        st.position = if self.dragging {
                            self.world_position_start
                        } else {
                            self.world_position
                        };
                        if let Some(a) = &axis {
                            st.visible = a.contains(name);
                        }
                    }
                }
                self.handle_states[i] = st;
                continue;
            }

            st.quaternion = quaternion;
            if mode == Mode::Translate || mode == Mode::Scale {
                const AXIS_HIDE_THRESHOLD: f64 = 0.99;
                const PLANE_HIDE_THRESHOLD: f64 = 0.2;
                let along = |u: V3| math::dot(math::apply_quat(u, quaternion), eye).abs();
                let hide = match name {
                    "X" => along(UNIT_X) > AXIS_HIDE_THRESHOLD,
                    "Y" => along(UNIT_Y) > AXIS_HIDE_THRESHOLD,
                    "Z" => along(UNIT_Z) > AXIS_HIDE_THRESHOLD,
                    "XY" => along(UNIT_Z) < PLANE_HIDE_THRESHOLD,
                    "YZ" => along(UNIT_X) < PLANE_HIDE_THRESHOLD,
                    "XZ" => along(UNIT_Y) < PLANE_HIDE_THRESHOLD,
                    _ => false,
                };
                if hide {
                    st.scale = [1e-10; 3];
                    st.visible = false;
                }
            } else {
                let q2 = quaternion;
                let align = math::apply_quat(eye, math::quat_invert(quaternion));
                if name.contains('E') {
                    let mut m = math::IDENTITY_M4;
                    math::look_at(&mut m, eye, [0.0; 3], UNIT_Y);
                    st.quaternion = math::quat_from_rotation_matrix(&m);
                }
                match name {
                    "X" => {
                        let t = math::quat_from_axis_angle(
                            UNIT_X,
                            math::js::atan2(-align[1], align[2]),
                        );
                        st.quaternion = math::quat_mul(q2, t);
                    }
                    "Y" => {
                        let t =
                            math::quat_from_axis_angle(UNIT_Y, math::js::atan2(align[0], align[2]));
                        st.quaternion = math::quat_mul(q2, t);
                    }
                    "Z" => {
                        let t =
                            math::quat_from_axis_angle(UNIT_Z, math::js::atan2(align[1], align[0]));
                        st.quaternion = math::quat_mul(q2, t);
                    }
                    _ => {}
                }
            }
            st.visible = st.visible && (!name.contains('X') || self.show_x);
            st.visible = st.visible && (!name.contains('Y') || self.show_y);
            st.visible = st.visible && (!name.contains('Z') || self.show_z);
            st.visible =
                st.visible && (!name.contains('E') || (self.show_x && self.show_y && self.show_z));
            self.handle_states[i] = st;

            let def = MATERIALS[material];
            let mut ms = MaterialState {
                color: def.color,
                opacity: def.opacity,
            };
            if self.enabled {
                if let Some(a) = &axis {
                    let single = a.chars().any(|c| name.len() == 1 && name.starts_with(c));
                    if name == a || single {
                        ms.color = 0xffff00;
                        ms.opacity = 1.0;
                    }
                }
            }
            self.materials[material] = ms;
        }
    }

    fn update_plane(&mut self, controls_world: &M4) {
        self.plane_position = self.world_position;
        let space = if self.mode == "scale" {
            "local"
        } else {
            self.space.as_str()
        };
        let q = if space == "local" {
            self.world_quaternion
        } else {
            math::IDENTITY_Q
        };
        let v1 = math::apply_quat(UNIT_X, q);
        let v2 = math::apply_quat(UNIT_Y, q);
        let v3 = math::apply_quat(UNIT_Z, q);
        let mut align = v2;
        match self.mode.as_str() {
            "translate" | "scale" => match self.axis.as_deref() {
                Some("X") => {
                    align = math::cross(self.eye, v1);
                    self.dir_vector = math::cross(v1, align);
                }
                Some("Y") => {
                    align = math::cross(self.eye, v2);
                    self.dir_vector = math::cross(v2, align);
                }
                Some("Z") => {
                    align = math::cross(self.eye, v3);
                    self.dir_vector = math::cross(v3, align);
                }
                Some("XY") => self.dir_vector = v3,
                Some("YZ") => self.dir_vector = v1,
                Some("XZ") => {
                    align = v3;
                    self.dir_vector = v2;
                }
                Some("XYZ") | Some("E") => self.dir_vector = [0.0; 3],
                _ => {}
            },
            _ => self.dir_vector = [0.0; 3],
        }
        if math::length(self.dir_vector) == 0.0 {
            self.plane_quaternion = self.camera_quaternion;
        } else {
            // three.js's module-level `_tempMatrix`: only `lookAt` ever
            // writes it, so its other elements stay the identity's.
            let mut m = math::IDENTITY_M4;
            math::look_at(&mut m, [0.0; 3], self.dir_vector, align);
            self.plane_quaternion = math::quat_from_rotation_matrix(&m);
        }
        let local = math::compose(self.plane_position, self.plane_quaternion, [1.0; 3]);
        self.plane_matrix_world = math::multiply(controls_world, &local);
    }

    /// Each handle's state after the last update, in [`handles`](Self::handles) order.
    pub fn handle_states(&self) -> &[HandleState] {
        &self.handle_states
    }

    /// The handles (group, mode, name, material, baked geometry).
    pub fn handles(&self) -> &[Handle] {
        &self.handles
    }

    /// Each material's colour and opacity after the last update.
    pub fn material_states(&self) -> &[MaterialState] {
        &self.materials
    }

    /// The drag plane's position and quaternion (under the control).
    pub fn plane_transform(&self) -> (V3, Q) {
        (self.plane_position, self.plane_quaternion)
    }

    // ------------------------------------------------------------------
    // Rays
    // ------------------------------------------------------------------

    /// `Raycaster.setFromCamera(pointer, camera)` → `(origin, direction)`.
    pub fn ray_from_camera(x: f64, y: f64, camera: &CameraState) -> (V3, V3) {
        let unproject = |v: V3| {
            math::apply_m4(
                math::apply_m4(v, &camera.projection_matrix_inverse),
                &camera.matrix_world,
            )
        };
        if camera.orthographic {
            let origin = unproject([
                x,
                y,
                (camera.near + camera.far) / (camera.near - camera.far),
            ]);
            let direction = math::transform_direction([0.0, 0.0, -1.0], &camera.matrix_world);
            (origin, direction)
        } else {
            let origin = math::matrix_position(&camera.matrix_world);
            let direction = math::normalize(math::sub(unproject([x, y, 0.5]), origin));
            (origin, direction)
        }
    }

    /// `intersectObjectWithRay(this._plane, raycaster, true)`: the drag
    /// plane mesh (`PlaneGeometry(1e5, 1e5, 2, 2)`, double-sided) as
    /// `Mesh.raycast` hits it, from its last `matrixWorld`. `None` for a ray
    /// parallel to it, pointing away, or missing it.
    pub fn intersect_plane(&self, origin: V3, direction: V3) -> Option<V3> {
        let mw = &self.plane_matrix_world;
        let inv = math::invert(mw);
        let lo = math::apply_m4(origin, &inv);
        let ld = math::transform_direction(direction, &inv);
        // Every triangle of the plane has edges along x and y of 5e4, so
        // its normal is (0, 0, ±2.5e9) and `intersectTriangle` reduces to
        // these products; the barycentric tests are the square's extent.
        let nz = 2.5e9;
        let mut ddn = ld[2] * nz;
        let sign = if ddn > 0.0 {
            1.0
        } else if ddn < 0.0 {
            ddn = -ddn;
            -1.0
        } else {
            return None;
        };
        let qdn = -sign * (lo[2] * nz);
        if qdn < 0.0 {
            return None;
        }
        let t = qdn / ddn;
        let p = [lo[0] + ld[0] * t, lo[1] + ld[1] * t, lo[2] + ld[2] * t];
        if p[0].abs() > 50000.0 || p[1].abs() > 50000.0 {
            return None;
        }
        Some(math::apply_m4(p, mw))
    }

    // ------------------------------------------------------------------
    // Pointer
    // ------------------------------------------------------------------

    /// Whether `pointerHover` would cast at all (an object, not dragging).
    pub fn wants_hover(&self) -> bool {
        self.has_object && !self.dragging
    }

    /// `pointerHover`: the caller cast at the current mode's pickers and
    /// passes the first visible hit's handle name (or `None`).
    pub fn pointer_hover(&mut self, hit: Option<&str>) -> Vec<TcEvent> {
        let mut ev = Vec::new();
        if !self.wants_hover() {
            return ev;
        }
        self.set_axis(hit.map(str::to_string), &mut ev);
        ev
    }

    /// `pointerDown`. `object` is read after its (and its parent's)
    /// `updateMatrixWorld`; `camera` as it is now (the ray).
    pub fn pointer_down(
        &mut self,
        pointer: Option<Pointer>,
        object: Option<&ObjectState>,
        camera: &CameraState,
    ) -> Vec<TcEvent> {
        let mut ev = Vec::new();
        if !self.has_object || self.dragging || pointer.is_some_and(|p| p.button != 0) {
            return ev;
        }
        if self.axis.is_some() {
            let hit = pointer.and_then(|p| {
                let (o, d) = Self::ray_from_camera(p.x, p.y, camera);
                self.intersect_plane(o, d)
            });
            if let (Some(point), Some(o)) = (hit, object) {
                self.position_start = o.position;
                self.quaternion_start = o.quaternion;
                self.scale_start = o.scale;
                let (p, q, s) = math::decompose(&o.matrix_world);
                self.world_position_start = p;
                self.world_quaternion_start = q;
                self.world_scale_start = s;
                self.point_start = math::sub(point, self.world_position_start);
            }
            self.set_dragging(true, &mut ev);
            ev.push(TcEvent::MouseDown(self.mode.clone()));
        }
        ev
    }

    /// `pointerMove`. `object` supplies the parent's current `matrixWorld`
    /// (for world-space snapping); `camera` the ray and its position.
    pub fn pointer_move(
        &mut self,
        pointer: Option<Pointer>,
        object: Option<&ObjectState>,
        camera: &CameraState,
    ) -> Vec<TcEvent> {
        let mut ev = Vec::new();
        let mode = self.mode.clone();
        let mut space = self.space.clone();
        let Some(axis) = self.axis.clone() else {
            return ev;
        };
        if mode == "scale" {
            space = "local".into();
        } else if axis == "E" || axis == "XYZE" || axis == "XYZ" {
            space = "world".into();
        }
        if !self.has_object || !self.dragging || pointer.is_some_and(|p| p.button != -1) {
            return ev;
        }
        let Some(p) = pointer else {
            return ev;
        };
        let (o, d) = Self::ray_from_camera(p.x, p.y, camera);
        let Some(point) = self.intersect_plane(o, d) else {
            return ev;
        };
        self.point_end = math::sub(point, self.world_position_start);
        let parent_world = object.and_then(|o| o.parent_matrix_world);

        if mode == "translate" {
            self.offset = math::sub(self.point_end, self.point_start);
            if space == "local" && axis != "XYZ" {
                self.offset = math::apply_quat(self.offset, self.world_quaternion_inv);
            }
            if !axis.contains('X') {
                self.offset[0] = 0.0;
            }
            if !axis.contains('Y') {
                self.offset[1] = 0.0;
            }
            if !axis.contains('Z') {
                self.offset[2] = 0.0;
            }
            if space == "local" && axis != "XYZ" {
                self.offset = math::div(
                    math::apply_quat(self.offset, self.quaternion_start),
                    self.parent_scale,
                );
            } else {
                self.offset = math::div(
                    math::apply_quat(self.offset, self.parent_quaternion_inv),
                    self.parent_scale,
                );
            }
            let mut position = math::add(self.offset, self.position_start);
            if let Some(snap) = truthy(self.translation_snap) {
                let round = |v: f64| js_round(v / snap) * snap;
                if space == "local" {
                    position = math::apply_quat(position, math::quat_invert(self.quaternion_start));
                    for (k, c) in ['X', 'Y', 'Z'].into_iter().enumerate() {
                        if axis.contains(c) {
                            position[k] = round(position[k]);
                        }
                    }
                    position = math::apply_quat(position, self.quaternion_start);
                }
                if space == "world" {
                    if let Some(pw) = &parent_world {
                        position = math::add(position, math::matrix_position(pw));
                    }
                    for (k, c) in ['X', 'Y', 'Z'].into_iter().enumerate() {
                        if axis.contains(c) {
                            position[k] = round(position[k]);
                        }
                    }
                    if let Some(pw) = &parent_world {
                        position = math::sub(position, math::matrix_position(pw));
                    }
                }
            }
            ev.push(TcEvent::SetPosition(position));
        } else if mode == "scale" {
            let factor = if axis.contains("XYZ") {
                let mut d = math::length(self.point_end) / math::length(self.point_start);
                if math::dot(self.point_end, self.point_start) < 0.0 {
                    d *= -1.0;
                }
                [d, d, d]
            } else {
                let a = math::apply_quat(self.point_start, self.world_quaternion_inv);
                let b = math::apply_quat(self.point_end, self.world_quaternion_inv);
                let mut f = math::div(b, a);
                if !axis.contains('X') {
                    f[0] = 1.0;
                }
                if !axis.contains('Y') {
                    f[1] = 1.0;
                }
                if !axis.contains('Z') {
                    f[2] = 1.0;
                }
                f
            };
            let mut scale = math::mul(self.scale_start, factor);
            if let Some(snap) = truthy(self.scale_snap) {
                for (k, c) in ['X', 'Y', 'Z'].into_iter().enumerate() {
                    if axis.contains(c) {
                        let v = js_round(scale[k] / snap) * snap;
                        // `|| this.scaleSnap`
                        scale[k] = if v == 0.0 || v.is_nan() { snap } else { v };
                    }
                }
            }
            ev.push(TcEvent::SetScale(scale));
        } else if mode == "rotate" {
            self.offset = math::sub(self.point_end, self.point_start);
            let rotation_speed = 20.0
                / math::distance(
                    self.world_position,
                    math::matrix_position(&camera.matrix_world),
                );
            let mut in_plane = false;
            if axis == "XYZE" {
                self.rotation_axis = math::normalize(math::cross(self.offset, self.eye));
                let a = math::dot(self.offset, math::cross(self.rotation_axis, self.eye))
                    * rotation_speed;
                self.set_rotation_angle(a, &mut ev);
            } else if axis == "X" || axis == "Y" || axis == "Z" {
                self.rotation_axis = unit(&axis);
                let mut t = unit(&axis);
                if space == "local" {
                    t = math::apply_quat(t, self.world_quaternion);
                }
                t = math::cross(t, self.eye);
                if math::length(t) == 0.0 {
                    in_plane = true;
                } else {
                    let a = math::dot(self.offset, math::normalize(t)) * rotation_speed;
                    self.set_rotation_angle(a, &mut ev);
                }
            }
            if axis == "E" || in_plane {
                self.rotation_axis = self.eye;
                let a = math::angle_to(self.point_end, self.point_start);
                self.set_rotation_angle(a, &mut ev);
                self.start_norm = math::normalize(self.point_start);
                self.end_norm = math::normalize(self.point_end);
                let s = if math::dot(math::cross(self.end_norm, self.start_norm), self.eye) < 0.0 {
                    1.0
                } else {
                    -1.0
                };
                self.set_rotation_angle(self.rotation_angle * s, &mut ev);
            }
            if let Some(snap) = truthy(self.rotation_snap) {
                let a = js_round(self.rotation_angle / snap) * snap;
                self.set_rotation_angle(a, &mut ev);
            }
            let q = if space == "local" && axis != "E" && axis != "XYZE" {
                math::quat_normalize(math::quat_mul(
                    self.quaternion_start,
                    math::quat_from_axis_angle(self.rotation_axis, self.rotation_angle),
                ))
            } else {
                self.rotation_axis =
                    math::apply_quat(self.rotation_axis, self.parent_quaternion_inv);
                math::quat_normalize(math::quat_mul(
                    math::quat_from_axis_angle(self.rotation_axis, self.rotation_angle),
                    self.quaternion_start,
                ))
            };
            ev.push(TcEvent::SetQuaternion(q));
        }
        ev.push(TcEvent::Change);
        ev.push(TcEvent::ObjectChange);
        ev
    }

    /// `pointerUp`.
    pub fn pointer_up(&mut self, pointer: Option<Pointer>) -> Vec<TcEvent> {
        let mut ev = Vec::new();
        if pointer.is_some_and(|p| p.button != 0) {
            return ev;
        }
        if self.dragging && self.axis.is_some() {
            ev.push(TcEvent::MouseUp(self.mode.clone()));
        }
        self.set_dragging(false, &mut ev);
        self.set_axis(None, &mut ev);
        ev
    }

    /// `reset()`: back to where the drag started.
    pub fn reset(&mut self) -> Vec<TcEvent> {
        let mut ev = Vec::new();
        if !self.enabled || !self.dragging {
            return ev;
        }
        ev.push(TcEvent::SetPosition(self.position_start));
        ev.push(TcEvent::SetQuaternion(self.quaternion_start));
        ev.push(TcEvent::SetScale(self.scale_start));
        ev.push(TcEvent::Change);
        ev.push(TcEvent::ObjectChange);
        self.point_start = self.point_end;
        ev
    }
}

const PI_F: f64 = std::f64::consts::PI;

#[cfg(test)]
mod tests;
