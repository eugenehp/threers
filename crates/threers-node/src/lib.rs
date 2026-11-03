//! Native Node.js bindings for threers (`npm i threers-node` via napi-rs).
//!
//! Deno and the browser use the wasm ESM package (`threers` / `crates/threers-js`).
//! This crate is the **native** Node addon — Neon is intentionally not used;
//! napi-rs provides `#[napi]` macros and generated TypeScript types.

#![deny(clippy::all)]

use napi::bindgen_prelude::*;
use napi_derive::napi;

use threers::cameras::PerspectiveCamera;
use threers::core::{Mesh, Object3D, ObjectId};
use threers::geometries::{BoxGeometry, QuadGeometry, SphereGeometry};
use threers::materials::{BasicMaterial, Material, StandardMaterial};
use threers::math::{Color, Vector3};
use threers::scene::Scene;

use threers::cameras::{Camera, OrthographicCamera};
use threers::core::{BufferAttribute, BufferGeometry, ObjectKind};
#[cfg(feature = "headless")]
use threers::lights::Light;
use threers::lights::{AmbientLight, DirectionalLight, HemisphereLight, PointLight, SpotLight};
use threers::materials::{LambertMaterial, NormalMaterial, PhongMaterial, PhysicalMaterial};
use threers::math::{Quaternion, Vector2};

#[napi]
pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

// ---- math ------------------------------------------------------------------

#[napi(js_name = "Vector3")]
#[derive(Clone, Copy)]
pub struct JsVector3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

#[napi]
impl JsVector3 {
    #[napi(constructor)]
    pub fn new(x: Option<f64>, y: Option<f64>, z: Option<f64>) -> Self {
        Self {
            x: x.unwrap_or(0.0),
            y: y.unwrap_or(0.0),
            z: z.unwrap_or(0.0),
        }
    }

    fn to_inner(self) -> Vector3 {
        Vector3::new(self.x as f32, self.y as f32, self.z as f32)
    }

    fn from_inner(v: Vector3) -> Self {
        Self {
            x: v.x as f64,
            y: v.y as f64,
            z: v.z as f64,
        }
    }

    #[napi]
    pub fn length(&self) -> f64 {
        self.to_inner().length() as f64
    }

    #[napi]
    pub fn normalized(&self) -> Self {
        Self::from_inner(self.to_inner().normalize())
    }
}

#[napi(js_name = "Color")]
#[derive(Clone, Copy)]
pub struct JsColor {
    pub r: f64,
    pub g: f64,
    pub b: f64,
}

#[napi]
impl JsColor {
    #[napi(constructor)]
    pub fn new(r: Option<f64>, g: Option<f64>, b: Option<f64>) -> Self {
        Self {
            r: r.unwrap_or(1.0),
            g: g.unwrap_or(1.0),
            b: b.unwrap_or(1.0),
        }
    }

    #[napi(factory)]
    pub fn from_hex(hex: u32) -> Self {
        let c = Color::from_hex(hex);
        Self {
            r: c.r as f64,
            g: c.g as f64,
            b: c.b as f64,
        }
    }
}

// ---- scene graph -----------------------------------------------------------

#[napi(js_name = "PerspectiveCamera")]
pub struct JsPerspectiveCamera {
    inner: PerspectiveCamera,
}

#[napi]
impl JsPerspectiveCamera {
    #[napi(constructor)]
    pub fn new(fov: Option<f64>, aspect: Option<f64>, near: Option<f64>, far: Option<f64>) -> Self {
        Self {
            inner: PerspectiveCamera::new(
                fov.unwrap_or(50.0) as f32,
                aspect.unwrap_or(1.0) as f32,
                near.unwrap_or(0.1) as f32,
                far.unwrap_or(2000.0) as f32,
            ),
        }
    }

    #[napi]
    pub fn set_position(&mut self, x: f64, y: f64, z: f64) {
        self.inner.position = Vector3::new(x as f32, y as f32, z as f32);
    }

    #[napi]
    pub fn look_at(&mut self, x: f64, y: f64, z: f64) {
        self.inner
            .look_at(Vector3::new(x as f32, y as f32, z as f32));
    }

    #[napi]
    pub fn set_aspect(&mut self, aspect: f64) {
        self.inner.aspect = aspect as f32;
    }
}

/// three.js `OrthographicCamera(left, right, top, bottom, near, far)`
/// (defaults -1, 1, 1, -1, 0.1, 2000). Renders through the same
/// `HeadlessRenderer` methods as `PerspectiveCamera`.
#[napi(js_name = "OrthographicCamera")]
pub struct JsOrthographicCamera {
    inner: OrthographicCamera,
}

#[napi]
impl JsOrthographicCamera {
    #[napi(constructor)]
    pub fn new(
        left: Option<f64>,
        right: Option<f64>,
        top: Option<f64>,
        bottom: Option<f64>,
        near: Option<f64>,
        far: Option<f64>,
    ) -> Result<Self> {
        let mut c = Self {
            inner: OrthographicCamera::new(-1.0, 1.0, 1.0, -1.0, 0.1, 2000.0),
        };
        c.set_frustum(
            left.unwrap_or(-1.0),
            right.unwrap_or(1.0),
            top.unwrap_or(1.0),
            bottom.unwrap_or(-1.0),
            near,
            far,
        )?;
        Ok(c)
    }

    /// Replace the view volume (three.js: assign `left` … `far`, then
    /// `updateProjectionMatrix()`).
    #[napi]
    pub fn set_frustum(
        &mut self,
        left: f64,
        right: f64,
        top: f64,
        bottom: f64,
        near: Option<f64>,
        far: Option<f64>,
    ) -> Result<()> {
        let near = near.unwrap_or(self.inner.near as f64);
        let far = far.unwrap_or(self.inner.far as f64);
        if ![left, right, top, bottom, near, far]
            .iter()
            .all(|v| v.is_finite())
        {
            return Err(bad("OrthographicCamera: values must be finite"));
        }
        if left == right || top == bottom || near >= far {
            return Err(bad(
                "OrthographicCamera: needs left != right, top != bottom and near < far",
            ));
        }
        let c = &mut self.inner;
        (c.left, c.right, c.top, c.bottom) = (left as f32, right as f32, top as f32, bottom as f32);
        (c.near, c.far) = (near as f32, far as f32);
        Ok(())
    }

    #[napi]
    pub fn set_position(&mut self, x: f64, y: f64, z: f64) -> Result<()> {
        let [x, y, z] = finite3(x, y, z, "setPosition")?;
        self.inner.position = Vector3::new(x, y, z);
        Ok(())
    }

    #[napi]
    pub fn look_at(&mut self, x: f64, y: f64, z: f64) -> Result<()> {
        let [x, y, z] = finite3(x, y, z, "lookAt")?;
        self.inner.look_at(Vector3::new(x, y, z));
        Ok(())
    }
}

// ---- class arguments checked with instanceof --------------------------------
//
// napi-rs 2 unwraps a `&T` class argument without checking its class, and its
// `Either`/`ClassInstance` validation looks the constructor up under the Rust
// struct name, which misses every class renamed with `js_name`. Arguments that
// take one of several classes, or that a caller could plausibly swap, go
// through `Instance<T>` instead, which checks `instanceof` against the
// registered JS constructor first.

/// A `#[napi]` class with its registered JS name (NUL-terminated).
trait JsClass: FromNapiRef + 'static {
    const JS_NAME: &'static str;
}

impl JsClass for JsPerspectiveCamera {
    const JS_NAME: &'static str = "PerspectiveCamera\0";
}
impl JsClass for JsOrthographicCamera {
    const JS_NAME: &'static str = "OrthographicCamera\0";
}
impl JsClass for JsGeometry {
    const JS_NAME: &'static str = "Geometry\0";
}
impl JsClass for JsMaterial {
    const JS_NAME: &'static str = "Material\0";
}

unsafe fn is_instance(
    env: sys::napi_env,
    val: sys::napi_value,
    js_name: &'static str,
) -> Result<bool> {
    let Some(ctor_ref) = get_class_constructor(js_name) else {
        return Ok(false);
    };
    let mut ctor = std::ptr::null_mut();
    check_status!(unsafe { sys::napi_get_reference_value(env, ctor_ref, &mut ctor) })?;
    let mut is = false;
    check_status!(unsafe { sys::napi_instanceof(env, val, ctor, &mut is) })?;
    Ok(is)
}

unsafe fn checked<T: JsClass>(
    env: sys::napi_env,
    val: sys::napi_value,
) -> Result<Option<&'static T>> {
    if unsafe { is_instance(env, val, T::JS_NAME) }? {
        Ok(Some(unsafe { T::from_napi_ref(env, val) }?))
    } else {
        Ok(None)
    }
}

fn class_name<T: JsClass>() -> &'static str {
    T::JS_NAME.trim_end_matches('\0')
}

/// A class argument of type `T`, checked with `instanceof`.
pub struct Instance<T: 'static>(&'static T);

impl<T: JsClass> TypeName for Instance<T> {
    fn type_name() -> &'static str {
        class_name::<T>()
    }
    fn value_type() -> ValueType {
        ValueType::Object
    }
}

impl<T: JsClass> FromNapiValue for Instance<T> {
    unsafe fn from_napi_value(env: sys::napi_env, val: sys::napi_value) -> Result<Self> {
        match unsafe { checked::<T>(env, val) }? {
            Some(v) => Ok(Self(v)),
            None => Err(bad(format!("expected a {}", class_name::<T>()))),
        }
    }
}

impl<T: JsClass> ValidateNapiValue for Instance<T> {}

impl<T> std::ops::Deref for Instance<T> {
    type Target = T;
    fn deref(&self) -> &T {
        self.0
    }
}

/// Either camera, as the renderer methods accept it.
pub enum AnyCamera {
    Perspective(&'static JsPerspectiveCamera),
    Orthographic(&'static JsOrthographicCamera),
}

impl TypeName for AnyCamera {
    fn type_name() -> &'static str {
        "PerspectiveCamera | OrthographicCamera"
    }
    fn value_type() -> ValueType {
        ValueType::Object
    }
}

impl FromNapiValue for AnyCamera {
    unsafe fn from_napi_value(env: sys::napi_env, val: sys::napi_value) -> Result<Self> {
        if let Some(c) = unsafe { checked::<JsPerspectiveCamera>(env, val) }? {
            return Ok(Self::Perspective(c));
        }
        if let Some(c) = unsafe { checked::<JsOrthographicCamera>(env, val) }? {
            return Ok(Self::Orthographic(c));
        }
        Err(bad("expected a PerspectiveCamera or OrthographicCamera"))
    }
}

fn camera_of(c: &AnyCamera) -> &dyn Camera {
    match c {
        AnyCamera::Perspective(p) => &p.inner,
        AnyCamera::Orthographic(o) => &o.inner,
    }
}

#[napi(js_name = "Scene")]
pub struct JsScene {
    inner: Scene,
    handles: Vec<ObjectId>,
}

impl Default for JsScene {
    fn default() -> Self {
        Self::new()
    }
}

#[napi]
impl JsScene {
    #[napi(constructor)]
    pub fn new() -> Self {
        Self {
            inner: Scene::new(),
            handles: Vec::new(),
        }
    }

    /// Add an axis-aligned box mesh. Returns a stable handle index.
    // Every dimension and placement value is optional on the JS side, so they
    // arrive as a flat list of `Option`s rather than an options object.
    #[allow(clippy::too_many_arguments)]
    #[napi]
    pub fn add_box(
        &mut self,
        width: Option<f64>,
        height: Option<f64>,
        depth: Option<f64>,
        color: Option<u32>,
        x: Option<f64>,
        y: Option<f64>,
        z: Option<f64>,
    ) -> u32 {
        let geom = BoxGeometry::new(
            width.unwrap_or(1.0) as f32,
            height.unwrap_or(1.0) as f32,
            depth.unwrap_or(1.0) as f32,
        );
        let mat = Material::Standard(StandardMaterial {
            color: Color::from_hex(color.unwrap_or(0xff6633)),
            ..StandardMaterial::default()
        });
        let mut obj = Object3D::mesh(Mesh::new(geom, mat));
        obj.position = Vector3::new(
            x.unwrap_or(0.0) as f32,
            y.unwrap_or(0.0) as f32,
            z.unwrap_or(0.0) as f32,
        );
        let id = self.inner.arena.insert(obj);
        self.inner.arena.add_child(self.inner.root, id);
        self.handles.push(id);
        (self.handles.len() - 1) as u32
    }

    #[napi]
    pub fn add_sphere(
        &mut self,
        radius: Option<f64>,
        color: Option<u32>,
        x: Option<f64>,
        y: Option<f64>,
        z: Option<f64>,
    ) -> u32 {
        let geom = SphereGeometry::new(radius.unwrap_or(0.5) as f32, 24, 16);
        let mat = Material::Basic(BasicMaterial::new(Color::from_hex(
            color.unwrap_or(0x6ea8fe),
        )));
        let mut obj = Object3D::mesh(Mesh::new(geom, mat));
        obj.position = Vector3::new(
            x.unwrap_or(0.0) as f32,
            y.unwrap_or(0.0) as f32,
            z.unwrap_or(0.0) as f32,
        );
        let id = self.inner.arena.insert(obj);
        self.inner.arena.add_child(self.inner.root, id);
        self.handles.push(id);
        (self.handles.len() - 1) as u32
    }

    /// Add a quad from four corners in winding order (`p0..p3` at UV
    /// `(0,0)`, `(1,0)`, `(1,1)`, `(0,1)`), meshed as a bilinear patch and
    /// shaded on both sides. `corners` is 12 numbers or 4 `{x, y, z}` objects
    /// (e.g. `Vector3`s). Returns a stable handle index.
    #[napi]
    pub fn add_quad(
        &mut self,
        corners: Vec<Either<f64, Vector3Like>>,
        color: Option<u32>,
        width_segments: Option<u32>,
        height_segments: Option<u32>,
    ) -> Result<u32> {
        let corners = parse_corners(&corners)?;
        let segments = |v: Option<u32>, name: &str| -> Result<usize> {
            let v = v.unwrap_or(1);
            if !(1..=MAX_QUAD_SEGMENTS).contains(&v) {
                return Err(Error::new(
                    Status::InvalidArg,
                    format!("{name} must be 1..={MAX_QUAD_SEGMENTS}, got {v}"),
                ));
            }
            Ok(v as usize)
        };
        let geom = QuadGeometry::with_segments(
            corners,
            segments(width_segments, "widthSegments")?,
            segments(height_segments, "heightSegments")?,
        );
        let mat = Material::Standard(StandardMaterial {
            color: Color::from_hex(color.unwrap_or(0xff6633)),
            side: DOUBLE_SIDE,
            ..StandardMaterial::default()
        });
        let id = self
            .inner
            .arena
            .insert(Object3D::mesh(Mesh::new(geom, mat)));
        self.inner.arena.add_child(self.inner.root, id);
        self.handles.push(id);
        Ok((self.handles.len() - 1) as u32)
    }

    #[napi]
    pub fn set_background(&mut self, hex: u32) {
        self.inner.background = Color::from_hex(hex);
    }

    #[napi]
    pub fn set_position(&mut self, handle: u32, x: f64, y: f64, z: f64) -> Result<()> {
        let id = *self
            .handles
            .get(handle as usize)
            .ok_or_else(|| Error::from_reason("bad object handle"))?;
        let obj = self
            .inner
            .arena
            .get_mut(id)
            .ok_or_else(|| Error::from_reason("object removed"))?;
        obj.position = Vector3::new(x as f32, y as f32, z as f32);
        Ok(())
    }

    /// Number of live handles (meshes, groups, lights and loaded models;
    /// removed objects are not counted).
    #[napi]
    pub fn object_count(&self) -> u32 {
        self.handles
            .iter()
            .filter(|id| self.inner.arena.get(**id).is_some())
            .count() as u32
    }

    // ---- generic meshes and groups ----

    /// Add a mesh built from a `Geometry` and a `Material`. Returns a stable
    /// handle index.
    #[napi]
    pub fn add_mesh(
        &mut self,
        #[napi(ts_arg_type = "Geometry")] geometry: Instance<JsGeometry>,
        #[napi(ts_arg_type = "Material")] material: Instance<JsMaterial>,
        x: Option<f64>,
        y: Option<f64>,
        z: Option<f64>,
    ) -> Result<u32> {
        if matches!(material.inner, Material::Line(_)) {
            return Err(bad("addMesh: a line material needs addLineSegments"));
        }
        let mut obj = Object3D::mesh(Mesh::new(geometry.inner.clone(), material.inner.clone()));
        obj.position = opt_point(x, y, z, "addMesh")?;
        Ok(self.push(obj))
    }

    /// three.js `LineSegments`: draw `geometry`'s vertices as pairs
    /// (`Geometry.edges` / `Geometry.wireframe` output, or any geometry with
    /// an even vertex count) with a `Material.line`.
    #[napi]
    pub fn add_line_segments(
        &mut self,
        #[napi(ts_arg_type = "Geometry")] geometry: Instance<JsGeometry>,
        #[napi(ts_arg_type = "Material")] material: Instance<JsMaterial>,
        x: Option<f64>,
        y: Option<f64>,
        z: Option<f64>,
    ) -> Result<u32> {
        if !matches!(material.inner, Material::Line(_)) {
            return Err(bad("addLineSegments: needs a Material.line"));
        }
        let g = &geometry.inner;
        let n = g.get_attribute("position").map_or(0, |a| a.count());
        if g.index.is_none() && !n.is_multiple_of(2) {
            return Err(bad(format!(
                "addLineSegments: needs an even vertex count, got {n}"
            )));
        }
        let mut obj = Object3D::line_segments(threers::core::LineSegments::new(
            g.clone(),
            material.inner.clone(),
        ));
        obj.position = opt_point(x, y, z, "addLineSegments")?;
        Ok(self.push(obj))
    }

    /// A copy of the geometry of a mesh or line-segments handle (for a
    /// loaded model, get the mesh handles with `meshes` first).
    #[napi]
    pub fn get_geometry(&self, handle: u32) -> Result<JsGeometry> {
        let id = self.live(handle)?;
        match &self.inner.arena.get(id).unwrap().kind {
            ObjectKind::Mesh(m) => Ok(JsGeometry::of((*m.geometry).clone())),
            ObjectKind::LineSegments(l) => Ok(JsGeometry::of((*l.geometry).clone())),
            _ => Err(bad(format!("getGeometry: object {handle} has no geometry"))),
        }
    }

    /// Handles for every mesh at or under `handle` (e.g. the parts of a
    /// loaded model), in scene-graph order. Repeated calls return the same
    /// handles.
    #[napi]
    pub fn meshes(&mut self, handle: u32) -> Result<Vec<u32>> {
        let id = self.live(handle)?;
        let mut ids = Vec::new();
        self.inner.arena.traverse(id, &mut |i, o| {
            if matches!(o.kind, ObjectKind::Mesh(_)) {
                ids.push(i);
            }
        });
        Ok(ids.into_iter().map(|i| self.handle_of(i)).collect())
    }

    /// Add an empty group (a transform node to parent other objects under).
    #[napi]
    pub fn add_group(&mut self, x: Option<f64>, y: Option<f64>, z: Option<f64>) -> Result<u32> {
        let mut obj = Object3D::group();
        obj.position = opt_point(x, y, z, "addGroup")?;
        Ok(self.push(obj))
    }

    /// Re-parent `child` under `parent` (omit `parent` for the scene root).
    /// The child's transform becomes relative to its new parent.
    #[napi]
    pub fn set_parent(&mut self, child: u32, parent: Option<u32>) -> Result<()> {
        let c = self.live(child)?;
        let p = match parent {
            None => self.inner.root,
            Some(p) => self.live(p)?,
        };
        if c == p || self.is_ancestor(c, p) {
            return Err(bad("setParent: would create a cycle"));
        }
        self.inner.arena.add_child(p, c);
        Ok(())
    }

    // ---- lights ----

    /// three.js `AmbientLight` (default white, intensity 1).
    #[napi]
    pub fn add_ambient_light(&mut self, color: Option<u32>, intensity: Option<f64>) -> Result<u32> {
        let light = AmbientLight::new(
            hex(color, 0xffffff, "color")?,
            intensity_arg(intensity, "addAmbientLight")?,
        );
        Ok(self.push(Object3D::light(light)))
    }

    /// three.js `DirectionalLight` shining from `position` (default
    /// `{x: 0, y: 1, z: 0}`) towards the origin. `castShadow` makes it the
    /// scene's directional shadow caster (meshes opt in with `setCastShadow` /
    /// `setReceiveShadow`).
    #[napi]
    pub fn add_directional_light(
        &mut self,
        color: Option<u32>,
        intensity: Option<f64>,
        position: Option<Vector3Like>,
        cast_shadow: Option<bool>,
    ) -> Result<u32> {
        let pos = opt_vec(
            position,
            Vector3::new(0.0, 1.0, 0.0),
            "addDirectionalLight: position",
        )?;
        if pos.length_sq() < 1e-12 {
            return Err(bad(
                "addDirectionalLight: position must not be the origin (the light aims at it)",
            ));
        }
        let mut light = DirectionalLight::new(
            hex(color, 0xffffff, "color")?,
            intensity_arg(intensity, "addDirectionalLight")?,
        );
        light.cast_shadow = cast_shadow.unwrap_or(false);
        let mut obj = Object3D::light(light);
        obj.position = pos;
        Ok(self.push(obj))
    }

    /// three.js `PointLight` (physically based: intensity in candela, default
    /// 1; `distance` 0 = infinite range; `decay` default 2).
    #[napi]
    pub fn add_point_light(
        &mut self,
        color: Option<u32>,
        intensity: Option<f64>,
        position: Option<Vector3Like>,
        distance: Option<f64>,
        decay: Option<f64>,
        cast_shadow: Option<bool>,
    ) -> Result<u32> {
        let mut light = PointLight::new(
            hex(color, 0xffffff, "color")?,
            intensity_arg(intensity, "addPointLight")?,
        )
        .with_distance(non_neg(distance, 0.0, "addPointLight: distance")?)
        .with_decay(non_neg(decay, 2.0, "addPointLight: decay")?);
        light.cast_shadow = cast_shadow.unwrap_or(false);
        let mut obj = Object3D::light(light);
        obj.position = opt_vec(position, Vector3::ZERO, "addPointLight: position")?;
        Ok(self.push(obj))
    }

    /// three.js `SpotLight` at `position` (default `{0, 1, 0}`) aimed at
    /// `target` (default origin). `angle` is the cone half-angle in radians
    /// (default π/3, max π/2); `penumbra` 0..1.
    #[allow(clippy::too_many_arguments)]
    #[napi]
    pub fn add_spot_light(
        &mut self,
        color: Option<u32>,
        intensity: Option<f64>,
        position: Option<Vector3Like>,
        target: Option<Vector3Like>,
        angle: Option<f64>,
        penumbra: Option<f64>,
        distance: Option<f64>,
        decay: Option<f64>,
        cast_shadow: Option<bool>,
    ) -> Result<u32> {
        let pos = opt_vec(
            position,
            Vector3::new(0.0, 1.0, 0.0),
            "addSpotLight: position",
        )?;
        let tgt = opt_vec(target, Vector3::ZERO, "addSpotLight: target")?;
        let dir = tgt - pos;
        if dir.length_sq() < 1e-12 {
            return Err(bad("addSpotLight: position and target must differ"));
        }
        let angle = angle.unwrap_or(std::f64::consts::FRAC_PI_3);
        if !(angle.is_finite() && angle > 0.0 && angle <= std::f64::consts::FRAC_PI_2) {
            return Err(bad(format!(
                "addSpotLight: angle must be in (0, π/2] radians, got {angle}"
            )));
        }
        let mut light = SpotLight::new(
            hex(color, 0xffffff, "color")?,
            intensity_arg(intensity, "addSpotLight")?,
        );
        light.direction = dir.normalize();
        light.angle = angle as f32;
        light.penumbra = unit(penumbra, 0.0, "addSpotLight: penumbra")?;
        light.distance = non_neg(distance, 0.0, "addSpotLight: distance")?;
        light.decay = non_neg(decay, 2.0, "addSpotLight: decay")?;
        light.cast_shadow = cast_shadow.unwrap_or(false);
        let mut obj = Object3D::light(light);
        obj.position = pos;
        Ok(self.push(obj))
    }

    /// three.js `HemisphereLight`: `skyColor` from above, `groundColor` from
    /// below (both default white), intensity default 1.
    #[napi]
    pub fn add_hemisphere_light(
        &mut self,
        sky_color: Option<u32>,
        ground_color: Option<u32>,
        intensity: Option<f64>,
    ) -> Result<u32> {
        let light = HemisphereLight::new(
            hex(sky_color, 0xffffff, "skyColor")?,
            hex(ground_color, 0xffffff, "groundColor")?,
            intensity_arg(intensity, "addHemisphereLight")?,
        );
        Ok(self.push(Object3D::light(light)))
    }

    /// Number of lights in the scene graph (including any a loaded model
    /// brought in).
    #[napi]
    pub fn light_count(&self) -> u32 {
        let mut n = 0;
        self.inner.arena.traverse(self.inner.root, &mut |_, o| {
            if matches!(o.kind, ObjectKind::Light(_)) {
                n += 1;
            }
        });
        n
    }

    // ---- handles ----

    /// Euler rotation in radians, XYZ order (three.js default).
    #[napi]
    pub fn set_rotation(&mut self, handle: u32, x: f64, y: f64, z: f64) -> Result<()> {
        let [x, y, z] = finite3(x, y, z, "setRotation")?;
        self.obj_mut(handle)?.quaternion = Quaternion::from_euler_xyz(x, y, z);
        Ok(())
    }

    #[napi]
    pub fn set_scale(&mut self, handle: u32, x: f64, y: f64, z: f64) -> Result<()> {
        let [x, y, z] = finite3(x, y, z, "setScale")?;
        self.obj_mut(handle)?.scale = Vector3::new(x, y, z);
        Ok(())
    }

    #[napi]
    pub fn set_visible(&mut self, handle: u32, visible: bool) -> Result<()> {
        self.obj_mut(handle)?.visible = visible;
        Ok(())
    }

    /// Include this object (and nothing else — not its children) in shadow
    /// maps.
    #[napi]
    pub fn set_cast_shadow(&mut self, handle: u32, cast: bool) -> Result<()> {
        self.obj_mut(handle)?.cast_shadow = cast;
        Ok(())
    }

    #[napi]
    pub fn set_receive_shadow(&mut self, handle: u32, receive: bool) -> Result<()> {
        self.obj_mut(handle)?.receive_shadow = receive;
        Ok(())
    }

    /// Replace a mesh's material. For a loaded model (a group), every mesh
    /// under it gets the material.
    #[napi]
    pub fn set_material(
        &mut self,
        handle: u32,
        #[napi(ts_arg_type = "Material")] material: Instance<JsMaterial>,
    ) -> Result<()> {
        let id = self.live(handle)?;
        let is_line = matches!(material.inner, Material::Line(_));
        if let ObjectKind::LineSegments(l) = &mut self.inner.arena.get_mut(id).unwrap().kind {
            if !is_line {
                return Err(bad("setMaterial: line segments need a Material.line"));
            }
            l.material = std::sync::Arc::new(material.inner.clone());
            return Ok(());
        }
        if is_line {
            return Err(bad(
                "setMaterial: a line material only applies to line segments",
            ));
        }
        if override_materials(&mut self.inner, id, &material.inner) == 0 {
            return Err(bad("setMaterial: handle has no mesh"));
        }
        Ok(())
    }

    #[napi]
    pub fn get_position(&self, handle: u32) -> Result<JsVector3> {
        let id = self.live(handle)?;
        Ok(JsVector3::from_inner(
            self.inner.arena.get(id).unwrap().position,
        ))
    }

    /// Remove an object and everything parented under it. Its handle (and
    /// any handles of its descendants) become invalid; other handles keep
    /// their indices.
    #[napi]
    pub fn remove(&mut self, handle: u32) -> Result<()> {
        let id = self.live(handle)?;
        remove_subtree(&mut self.inner, id);
        Ok(())
    }

    // ---- picking ----

    /// three.js `Raycaster.intersectObjects(scene.children, true)`: every
    /// visible object hit by the ray from `origin` along `direction` —
    /// meshes, line segments (within `options.lineThreshold`) — nearest
    /// first. `handle` is the object's handle (registered on the fly for
    /// meshes inside loaded models); the other fields are three.js's, plus
    /// `polygonIndex` (the polygon, e.g. the quad, the hit triangle belongs
    /// to) and `barycoord`. Fields that do not apply to a hit are undefined.
    /// `near` / `far` here win over `options.near` / `options.far`.
    #[napi]
    pub fn raycast(
        &mut self,
        origin: Vector3Like,
        direction: Vector3Like,
        near: Option<f64>,
        far: Option<f64>,
        options: Option<RaycastOptions>,
    ) -> Result<Vec<RaycastHit>> {
        let o = opt_vec(Some(origin), Vector3::ZERO, "raycast: origin")?;
        let d = opt_vec(Some(direction), Vector3::ZERO, "raycast: direction")?;
        if d.length_sq() < 1e-12 {
            return Err(bad("raycast: direction must not be zero"));
        }
        let mut rc = threers::core::Raycaster::default();
        rc.set(o, d.normalize());
        let options = options.unwrap_or_default();
        self.cast(
            rc,
            near.or(options.near),
            far.or(options.far),
            &options,
            "raycast",
        )
    }

    /// Pick through `camera` at normalised device coordinates `x`, `y`
    /// (-1..1, +y up) — three.js `setFromCamera`, perspective or
    /// orthographic, then `intersectObjects` as in `raycast`.
    #[napi]
    pub fn raycast_from_camera(
        &mut self,
        #[napi(ts_arg_type = "PerspectiveCamera | OrthographicCamera")] camera: AnyCamera,
        x: f64,
        y: f64,
        options: Option<RaycastOptions>,
    ) -> Result<Vec<RaycastHit>> {
        if !(x.is_finite() && y.is_finite()) {
            return Err(bad("raycastFromCamera: x and y must be finite"));
        }
        let mut rc = threers::core::Raycaster::default();
        rc.set_from_camera(Vector2::new(x as f32, y as f32), camera_of(&camera));
        // Pick what the camera draws: `set_from_camera` leaves the layers
        // alone (three.js), so take the camera's here.
        rc.layers = camera_of(&camera).layers();
        let options = options.unwrap_or_default();
        self.cast(rc, options.near, options.far, &options, "raycastFromCamera")
    }

    // ---- loaders ----

    /// Load a model file into the scene: `.glb`, `.gltf` (external buffers
    /// resolved next to the file), `.obj`, `.stl`, `.ply`, and `.usda` /
    /// `.usdc` / `.usdz` / `.usd` when built with the `usd` feature. Returns a
    /// handle: the mesh itself for the single-mesh formats (OBJ/STL/PLY), a
    /// group holding the model otherwise. `material` replaces the loaded
    /// materials (OBJ/STL/PLY default to a grey `Material.standard`).
    #[napi]
    pub fn load(
        &mut self,
        path: String,
        #[napi(ts_arg_type = "Material")] material: Option<Instance<JsMaterial>>,
    ) -> Result<u32> {
        let p = std::path::Path::new(&path);
        let fmt = format_from_path(p).map_err(bad)?;
        let bytes = std::fs::read(p)
            .map_err(|e| Error::new(Status::GenericFailure, format!("load: {path}: {e}")))?;
        let mat = material.map(|m| m.inner.clone());
        let id = load_into(&mut self.inner, &bytes, fmt, p.parent(), mat)
            .map_err(|e| bad(format!("load: {path}: {e}")))?;
        self.handles.push(id);
        Ok((self.handles.len() - 1) as u32)
    }

    /// Like `load`, from bytes. `format`: `'glb' | 'gltf' | 'obj' | 'stl' |
    /// 'ply' | 'usd'` (`'usda' | 'usdc' | 'usdz'` also accepted). A `.gltf`
    /// given as bytes must embed its buffers as data URIs.
    #[napi]
    pub fn load_bytes(
        &mut self,
        data: Buffer,
        #[napi(
            ts_arg_type = "'glb' | 'gltf' | 'obj' | 'stl' | 'ply' | 'usd' | 'usda' | 'usdc' | 'usdz'"
        )]
        format: String,
        #[napi(ts_arg_type = "Material")] material: Option<Instance<JsMaterial>>,
    ) -> Result<u32> {
        let fmt = parse_format(&format).map_err(bad)?;
        let mat = material.map(|m| m.inner.clone());
        let id = load_into(&mut self.inner, &data, fmt, None, mat)
            .map_err(|e| bad(format!("loadBytes: {e}")))?;
        self.handles.push(id);
        Ok((self.handles.len() - 1) as u32)
    }
}

impl JsScene {
    /// Insert under the root and hand out the next handle index.
    fn push(&mut self, obj: Object3D) -> u32 {
        let id = self.inner.arena.insert(obj);
        self.inner.arena.add_child(self.inner.root, id);
        self.handles.push(id);
        (self.handles.len() - 1) as u32
    }

    /// Apply `options` to `rc` and run it over the scene, or over
    /// `options.objects`.
    fn cast(
        &mut self,
        mut rc: threers::core::Raycaster,
        near: Option<f64>,
        far: Option<f64>,
        options: &RaycastOptions,
        what: &str,
    ) -> Result<Vec<RaycastHit>> {
        rc.near = non_neg(near, 0.0, &format!("{what}: near"))?;
        let far = far.unwrap_or(f64::INFINITY);
        if far.is_nan() || far <= rc.near as f64 {
            return Err(bad(format!("{what}: far must be > near, got {far}")));
        }
        rc.far = far as f32;
        rc.first_hit_only = options.first_hit_only.unwrap_or(false);
        rc.params.line.threshold = non_neg(
            options.line_threshold,
            1.0,
            &format!("{what}: lineThreshold"),
        )?;
        rc.params.points.threshold = non_neg(
            options.points_threshold,
            1.0,
            &format!("{what}: pointsThreshold"),
        )?;
        let roots = match &options.objects {
            None => vec![self.inner.root],
            Some(handles) => handles
                .iter()
                .map(|&h| self.live(h))
                .collect::<Result<Vec<_>>>()?,
        };
        self.inner.update_world();
        let hits =
            rc.intersect_object_list(&self.inner.arena, &roots, options.recursive.unwrap_or(true));
        Ok(hits.iter().map(|h| self.hit(h)).collect())
    }

    fn hit(&mut self, h: &threers::core::Intersection) -> RaycastHit {
        let v3 = |v: Vector3| Vector3Like {
            x: v.x as f64,
            y: v.y as f64,
            z: v.z as f64,
        };
        let v2 = |v: Vector2| Vector2Like {
            x: v.x as f64,
            y: v.y as f64,
        };
        RaycastHit {
            handle: self.handle_of(h.object),
            distance: h.distance as f64,
            point: v3(h.point),
            face_index: h.face.map(|_| h.face_index as u32),
            face: h.face.map(|f| RaycastFace {
                a: f.a as u32,
                b: f.b as u32,
                c: f.c as u32,
                normal: v3(f.normal),
                material_index: f.material_index as u32,
            }),
            normal: h.normal.map(v3),
            uv: h.uv.map(v2),
            uv1: h.uv1.map(v2),
            barycoord: h.barycoord.map(v3),
            instance_id: h.instance_id.map(|i| i as u32),
            index: h.index.map(|i| i as u32),
            distance_to_ray: h.distance_to_ray.map(|d| d as f64),
            polygon_index: h.polygon_index.map(|p| p as u32),
        }
    }

    /// The handle for `id`, registering it if it has none yet.
    fn handle_of(&mut self, id: ObjectId) -> u32 {
        match self.handles.iter().position(|h| *h == id) {
            Some(i) => i as u32,
            None => {
                self.handles.push(id);
                (self.handles.len() - 1) as u32
            }
        }
    }

    fn live(&self, handle: u32) -> Result<ObjectId> {
        let id = *self
            .handles
            .get(handle as usize)
            .ok_or_else(|| bad(format!("bad object handle {handle}")))?;
        if self.inner.arena.get(id).is_none() {
            return Err(bad(format!("object {handle} was removed")));
        }
        Ok(id)
    }

    fn obj_mut(&mut self, handle: u32) -> Result<&mut Object3D> {
        let id = self.live(handle)?;
        Ok(self.inner.arena.get_mut(id).unwrap())
    }

    /// Whether `a` is an ancestor of `b`.
    fn is_ancestor(&self, a: ObjectId, b: ObjectId) -> bool {
        let mut cur = self.inner.arena.get(b).and_then(|o| o.parent);
        while let Some(p) = cur {
            if p == a {
                return true;
            }
            cur = self.inner.arena.get(p).and_then(|o| o.parent);
        }
        false
    }
}

// ---- geometry --------------------------------------------------------------

/// Per-axis cap on generated geometry segment counts.
const MAX_SEGMENTS: u32 = 4096;
/// Each Catmull–Clark round multiplies the face count by about 4.
const MAX_SUBDIVISION_LEVELS: u32 = 6;

/// A triangle mesh's vertex data, built with the static factories
/// (three.js `*Geometry` classes, same parameters and defaults).
#[napi(js_name = "Geometry")]
pub struct JsGeometry {
    inner: BufferGeometry,
}

#[napi]
impl JsGeometry {
    #[napi(factory, js_name = "box")]
    pub fn box_geometry(
        width: Option<f64>,
        height: Option<f64>,
        depth: Option<f64>,
    ) -> Result<Self> {
        Ok(Self::of(BoxGeometry::new(
            non_neg(width, 1.0, "box: width")?,
            non_neg(height, 1.0, "box: height")?,
            non_neg(depth, 1.0, "box: depth")?,
        )))
    }

    #[napi(factory)]
    pub fn plane(
        width: Option<f64>,
        height: Option<f64>,
        width_segments: Option<u32>,
        height_segments: Option<u32>,
    ) -> Result<Self> {
        Ok(Self::of(threers::geometries::PlaneGeometry::with_segments(
            non_neg(width, 1.0, "plane: width")?,
            non_neg(height, 1.0, "plane: height")?,
            seg(width_segments, 1, 1, "plane: widthSegments")?,
            seg(height_segments, 1, 1, "plane: heightSegments")?,
        )))
    }

    #[napi(factory)]
    pub fn sphere(
        radius: Option<f64>,
        width_segments: Option<u32>,
        height_segments: Option<u32>,
    ) -> Result<Self> {
        Ok(Self::of(SphereGeometry::new(
            non_neg(radius, 1.0, "sphere: radius")?,
            seg(width_segments, 32, 3, "sphere: widthSegments")?,
            seg(height_segments, 16, 2, "sphere: heightSegments")?,
        )))
    }

    #[napi(factory)]
    pub fn cylinder(
        radius_top: Option<f64>,
        radius_bottom: Option<f64>,
        height: Option<f64>,
        radial_segments: Option<u32>,
        height_segments: Option<u32>,
        open_ended: Option<bool>,
    ) -> Result<Self> {
        Ok(Self::of(threers::geometries::CylinderGeometry::new(
            non_neg(radius_top, 1.0, "cylinder: radiusTop")?,
            non_neg(radius_bottom, 1.0, "cylinder: radiusBottom")?,
            non_neg(height, 1.0, "cylinder: height")?,
            seg(radial_segments, 32, 3, "cylinder: radialSegments")?,
            seg(height_segments, 1, 1, "cylinder: heightSegments")?,
            open_ended.unwrap_or(false),
            0.0,
            std::f32::consts::TAU,
        )))
    }

    #[napi(factory)]
    pub fn cone(
        radius: Option<f64>,
        height: Option<f64>,
        radial_segments: Option<u32>,
        height_segments: Option<u32>,
        open_ended: Option<bool>,
    ) -> Result<Self> {
        Ok(Self::of(threers::geometries::ConeGeometry::new(
            non_neg(radius, 1.0, "cone: radius")?,
            non_neg(height, 1.0, "cone: height")?,
            seg(radial_segments, 32, 3, "cone: radialSegments")?,
            seg(height_segments, 1, 1, "cone: heightSegments")?,
            open_ended.unwrap_or(false),
            0.0,
            std::f32::consts::TAU,
        )))
    }

    /// `arc` in radians, (0, 2π].
    #[napi(factory)]
    pub fn torus(
        radius: Option<f64>,
        tube: Option<f64>,
        radial_segments: Option<u32>,
        tubular_segments: Option<u32>,
        arc: Option<f64>,
    ) -> Result<Self> {
        let arc = arc.unwrap_or(std::f64::consts::TAU);
        if !(arc.is_finite() && arc > 0.0 && arc <= std::f64::consts::TAU + 1e-9) {
            return Err(bad(format!("torus: arc must be in (0, 2π], got {arc}")));
        }
        Ok(Self::of(threers::geometries::TorusGeometry::new(
            non_neg(radius, 1.0, "torus: radius")?,
            non_neg(tube, 0.4, "torus: tube")?,
            seg(radial_segments, 12, 2, "torus: radialSegments")?,
            seg(tubular_segments, 48, 3, "torus: tubularSegments")?,
            arc as f32,
        )))
    }

    #[napi(factory)]
    pub fn torus_knot(
        radius: Option<f64>,
        tube: Option<f64>,
        tubular_segments: Option<u32>,
        radial_segments: Option<u32>,
        p: Option<u32>,
        q: Option<u32>,
    ) -> Result<Self> {
        let (p, q) = (p.unwrap_or(2), q.unwrap_or(3));
        if p == 0 || q == 0 {
            return Err(bad("torusKnot: p and q must be >= 1"));
        }
        Ok(Self::of(threers::geometries::TorusKnotGeometry::new(
            non_neg(radius, 1.0, "torusKnot: radius")?,
            non_neg(tube, 0.4, "torusKnot: tube")?,
            seg(tubular_segments, 64, 3, "torusKnot: tubularSegments")?,
            seg(radial_segments, 8, 3, "torusKnot: radialSegments")?,
            p,
            q,
        )))
    }

    #[napi(factory)]
    pub fn ring(
        inner_radius: Option<f64>,
        outer_radius: Option<f64>,
        theta_segments: Option<u32>,
        phi_segments: Option<u32>,
    ) -> Result<Self> {
        let inner = non_neg(inner_radius, 0.5, "ring: innerRadius")?;
        let outer = non_neg(outer_radius, 1.0, "ring: outerRadius")?;
        if inner > outer {
            return Err(bad("ring: innerRadius must be <= outerRadius"));
        }
        Ok(Self::of(threers::geometries::RingGeometry::new(
            inner,
            outer,
            seg(theta_segments, 32, 3, "ring: thetaSegments")?,
            seg(phi_segments, 1, 1, "ring: phiSegments")?,
            0.0,
            std::f32::consts::TAU,
        )))
    }

    #[napi(factory)]
    pub fn circle(radius: Option<f64>, segments: Option<u32>) -> Result<Self> {
        Ok(Self::of(threers::geometries::CircleGeometry::new(
            non_neg(radius, 1.0, "circle: radius")?,
            seg(segments, 32, 3, "circle: segments")?,
            0.0,
            std::f32::consts::TAU,
        )))
    }

    #[napi(factory)]
    pub fn capsule(
        radius: Option<f64>,
        length: Option<f64>,
        cap_segments: Option<u32>,
        radial_segments: Option<u32>,
    ) -> Result<Self> {
        Ok(Self::of(threers::geometries::CapsuleGeometry::new(
            non_neg(radius, 1.0, "capsule: radius")?,
            non_neg(length, 1.0, "capsule: length")?,
            seg(cap_segments, 4, 1, "capsule: capSegments")?,
            seg(radial_segments, 8, 3, "capsule: radialSegments")?,
        )))
    }

    /// Revolve a profile around Y. `points`: flat `[x0, y0, x1, y1, …]` or
    /// `{x, y}` objects; at least 2.
    #[napi(factory)]
    pub fn lathe(points: Vec<Either<f64, Vector2Like>>, segments: Option<u32>) -> Result<Self> {
        let flat = flatten_points(&points, 2, "lathe: points")?;
        let pts: Vec<Vector2> = flat.chunks(2).map(|c| Vector2::new(c[0], c[1])).collect();
        Ok(Self::of(threers::geometries::LatheGeometry::new(
            &pts,
            seg(segments, 12, 1, "lathe: segments")?,
            0.0,
            std::f32::consts::TAU,
        )))
    }

    /// A tube along a Catmull-Rom curve through `points` (flat `[x, y, z, …]`
    /// or `{x, y, z}` objects; at least 2).
    #[napi(factory)]
    pub fn tube(
        points: Vec<Either<f64, Vector3Like>>,
        tubular_segments: Option<u32>,
        radius: Option<f64>,
        radial_segments: Option<u32>,
        closed: Option<bool>,
    ) -> Result<Self> {
        let flat = flatten_points(&points, 3, "tube: points")?;
        let pts: Vec<Vector3> = flat
            .chunks(3)
            .map(|c| Vector3::new(c[0], c[1], c[2]))
            .collect();
        let curve = threers::CatmullRomCurve3::new(pts);
        Ok(Self::of(threers::geometries::TubeGeometry::new(
            &curve,
            seg(tubular_segments, 64, 1, "tube: tubularSegments")?,
            non_neg(radius, 1.0, "tube: radius")?,
            seg(radial_segments, 8, 3, "tube: radialSegments")?,
            closed.unwrap_or(false),
        )))
    }

    #[napi(factory)]
    pub fn icosahedron(radius: Option<f64>, detail: Option<u32>) -> Result<Self> {
        let (r, d) = poly_args(radius, detail, "icosahedron")?;
        Ok(Self::of(threers::geometries::IcosahedronGeometry::new(
            r, d,
        )))
    }

    #[napi(factory)]
    pub fn octahedron(radius: Option<f64>, detail: Option<u32>) -> Result<Self> {
        let (r, d) = poly_args(radius, detail, "octahedron")?;
        Ok(Self::of(threers::geometries::OctahedronGeometry::new(r, d)))
    }

    #[napi(factory)]
    pub fn tetrahedron(radius: Option<f64>, detail: Option<u32>) -> Result<Self> {
        let (r, d) = poly_args(radius, detail, "tetrahedron")?;
        Ok(Self::of(threers::geometries::TetrahedronGeometry::new(
            r, d,
        )))
    }

    #[napi(factory)]
    pub fn dodecahedron(radius: Option<f64>, detail: Option<u32>) -> Result<Self> {
        let (r, d) = poly_args(radius, detail, "dodecahedron")?;
        Ok(Self::of(threers::geometries::DodecahedronGeometry::new(
            r, d,
        )))
    }

    /// Same corners as `Scene.addQuad`.
    #[napi(factory)]
    pub fn quad(
        corners: Vec<Either<f64, Vector3Like>>,
        width_segments: Option<u32>,
        height_segments: Option<u32>,
    ) -> Result<Self> {
        let corners = parse_corners(&corners)?;
        Ok(Self::of(QuadGeometry::with_segments(
            corners,
            seg(width_segments, 1, 1, "quad: widthSegments")?,
            seg(height_segments, 1, 1, "quad: heightSegments")?,
        )))
    }

    /// Custom geometry from flat `positions` (`x, y, z` per vertex), optional
    /// triangle `indices` and optional `normals` (computed when omitted).
    #[napi(factory)]
    pub fn from_arrays(
        positions: Vec<f64>,
        indices: Option<Vec<u32>>,
        normals: Option<Vec<f64>>,
    ) -> Result<Self> {
        let pos: Vec<f32> = positions.iter().map(|v| *v as f32).collect();
        let nrm = normals.map(|n| n.iter().map(|v| *v as f32).collect());
        geometry_from_arrays(pos, indices, nrm)
            .map(Self::of)
            .map_err(|e| bad(format!("fromArrays: {e}")))
    }

    /// three.js `EdgesGeometry`: the edges of `source` whose faces meet at
    /// `thresholdAngle` degrees or more (default 1), plus border edges, as a
    /// line geometry for `Scene.addLineSegments`. Edges inside one polygon
    /// (a quad's diagonal) are never emitted.
    #[napi(factory)]
    pub fn edges(
        #[napi(ts_arg_type = "Geometry")] source: Instance<JsGeometry>,
        threshold_angle: Option<f64>,
    ) -> Result<Self> {
        let deg = threshold_angle.unwrap_or(1.0);
        if !(deg.is_finite() && (0.0..=180.0).contains(&deg)) {
            return Err(bad(format!(
                "edges: thresholdAngle must be 0..=180 degrees, got {deg}"
            )));
        }
        Ok(Self::of(threers::geometries::EdgesGeometry::new(
            &source.inner,
            (deg as f32).to_radians(),
        )))
    }

    /// three.js `WireframeGeometry`: every edge of `source` once, as a line
    /// geometry. A geometry carrying polygons draws their outlines — quads
    /// without diagonals.
    #[napi(factory)]
    pub fn wireframe(#[napi(ts_arg_type = "Geometry")] source: Instance<JsGeometry>) -> Self {
        Self::of(threers::geometries::WireframeGeometry::new(&source.inner))
    }

    /// Flat vertex positions, `x, y, z` per vertex.
    #[napi]
    pub fn positions(&self) -> Vec<f64> {
        self.inner
            .get_attribute("position")
            .map(|a| a.array.iter().map(|v| *v as f64).collect())
            .unwrap_or_default()
    }

    /// Triangle vertex indices, or `null` for a non-indexed geometry.
    #[napi]
    pub fn index(&self) -> Option<Vec<u32>> {
        self.inner.index.clone()
    }

    /// Triangles per polygon, in triangle order (`2` is a quad), or `null`
    /// when the geometry carries no polygon grouping (every triangle stands
    /// alone). The grid generators, `quad`, and the OBJ / PLY / FBX / USD
    /// loaders attach one.
    #[napi]
    pub fn polygon_counts(&self) -> Option<Vec<u32>> {
        self.inner.polygon_counts().map(|p| p.to_vec())
    }

    /// Group consecutive triangles into polygons: `counts[i]` triangles form
    /// polygon `i`. The counts must be >= 1 and sum to `triangleCount()`.
    /// Wireframes and `Geometry.edges` then follow the polygon outlines.
    #[napi]
    pub fn set_polygons(&mut self, counts: Vec<u32>) -> Result<()> {
        check_polygons(&counts, self.inner.triangle_count()).map_err(bad)?;
        self.inner.set_polygons(counts);
        Ok(())
    }

    /// Pair adjacent triangles back into quads (Blender's "Tris to Quads"):
    /// two triangles merge when they fold by at most `maxFaceAngle` and every
    /// corner of the quad is within `maxShapeAngle` of 90° (degrees, both
    /// default 40). Only the index and the polygon grouping change; existing
    /// polygons stay whole. Returns how many quads were formed.
    #[napi]
    pub fn infer_quads(
        &mut self,
        max_face_angle: Option<f64>,
        max_shape_angle: Option<f64>,
    ) -> Result<u32> {
        let angle = |v: Option<f64>, name: &str| -> Result<f32> {
            let v = v.unwrap_or(40.0);
            if !(v.is_finite() && (0.0..=180.0).contains(&v)) {
                return Err(bad(format!(
                    "inferQuads: {name} must be 0..=180 degrees, got {v}"
                )));
            }
            Ok((v as f32).to_radians())
        };
        let face = angle(max_face_angle, "maxFaceAngle")?;
        let shape = angle(max_shape_angle, "maxShapeAngle")?;
        Ok(self.inner.infer_quads(face, shape) as u32)
    }

    /// Catmull–Clark subdivision, `levels` rounds (default 1, max 6): every
    /// polygon becomes quads (a quad → 4, a triangle → 3), as a new geometry
    /// with smooth normals and its quads grouped. The cage is this geometry's
    /// polygons (its triangles where there is no grouping), welded by
    /// position. `sharpBoundary` (default true) holds open borders in place.
    /// UVs and other attributes are not carried over.
    #[napi]
    pub fn subdivide(
        &self,
        levels: Option<u32>,
        sharp_boundary: Option<bool>,
    ) -> Result<JsGeometry> {
        let levels = levels.unwrap_or(1);
        if levels > MAX_SUBDIVISION_LEVELS {
            return Err(bad(format!(
                "subdivide: levels must be 0..={MAX_SUBDIVISION_LEVELS}, got {levels}"
            )));
        }
        let options = threers::geometries::SubdivisionOptions {
            sharp_boundary: sharp_boundary.unwrap_or(true),
            ..Default::default()
        };
        Ok(Self::of(
            self.inner.subdivide_with(levels as usize, &options),
        ))
    }

    /// Number of polygons (the triangle count when there is no grouping).
    #[napi]
    pub fn polygon_count(&self) -> u32 {
        self.inner.polygon_ranges().count() as u32
    }

    /// Polygons with exactly four corners.
    #[napi]
    pub fn quad_count(&self) -> u32 {
        polygon_loops(&self.inner)
            .iter()
            .filter(|l| l.as_ref().is_some_and(|l| l.len() == 4))
            .count() as u32
    }

    /// Each polygon's outline as vertex indices in winding order (a quad gives
    /// 4), or `null` for a polygon whose triangles do not form one simple
    /// loop.
    #[napi]
    pub fn polygon_loops(&self) -> Vec<Option<Vec<u32>>> {
        polygon_loops(&self.inner)
    }

    /// Polygon outline edges as `[a, b]` vertex-index pairs, each once —
    /// what a quad wireframe draws.
    #[napi]
    pub fn polygon_edges(&self) -> Vec<Vec<u32>> {
        self.inner
            .polygon_edges()
            .into_iter()
            .map(|e| e.to_vec())
            .collect()
    }

    #[napi]
    pub fn vertex_count(&self) -> u32 {
        self.inner
            .get_attribute("position")
            .map_or(0, |a| a.count()) as u32
    }

    #[napi]
    pub fn triangle_count(&self) -> u32 {
        self.inner.triangle_count() as u32
    }
}

impl JsGeometry {
    fn of(inner: BufferGeometry) -> Self {
        Self { inner }
    }
}

/// Anything with numeric `x`, `y`.
#[napi(object, js_name = "Vector2Like")]
pub struct Vector2Like {
    pub x: f64,
    pub y: f64,
}

trait PointLike {
    fn coords(&self) -> Vec<f64>;
}
impl PointLike for Vector2Like {
    fn coords(&self) -> Vec<f64> {
        vec![self.x, self.y]
    }
}
impl PointLike for Vector3Like {
    fn coords(&self) -> Vec<f64> {
        vec![self.x, self.y, self.z]
    }
}

/// Flat numbers (`dim` per point) or point objects, at least 2 points, all
/// finite.
fn flatten_points<P: PointLike>(
    pts: &[Either<f64, P>],
    dim: usize,
    what: &str,
) -> Result<Vec<f32>> {
    let mut flat = Vec::with_capacity(pts.len() * dim);
    let all_nums = pts.iter().all(|p| matches!(p, Either::A(_)));
    let all_pts = pts.iter().all(|p| matches!(p, Either::B(_)));
    if !(all_nums || all_pts) {
        return Err(bad(format!("{what}: mixes numbers and points")));
    }
    for p in pts {
        match p {
            Either::A(n) => flat.push(*n),
            Either::B(v) => flat.extend(v.coords()),
        }
    }
    if !flat.len().is_multiple_of(dim) {
        return Err(bad(format!(
            "{what}: {} numbers is not a whole number of {dim}D points",
            flat.len()
        )));
    }
    if flat.len() < 2 * dim {
        return Err(bad(format!("{what}: needs at least 2 points")));
    }
    if flat.iter().any(|v| !v.is_finite()) {
        return Err(bad(format!("{what}: non-finite coordinate")));
    }
    Ok(flat.into_iter().map(|v| v as f32).collect())
}

fn poly_args(radius: Option<f64>, detail: Option<u32>, what: &str) -> Result<(f32, usize)> {
    let d = detail.unwrap_or(0);
    if d > 32 {
        return Err(bad(format!("{what}: detail must be 0..=32, got {d}")));
    }
    Ok((
        non_neg(radius, 1.0, &format!("{what}: radius"))?,
        d as usize,
    ))
}

// ---- material --------------------------------------------------------------

/// A surface material, built with the static factories (three.js
/// `Mesh*Material`). Unlit: `basic`, `normal`. Lit (need a light to show):
/// `standard`, `physical`, `lambert`, `phong`.
#[napi(js_name = "Material")]
pub struct JsMaterial {
    inner: Material,
}

/// Material parameters. Each factory accepts the subset that applies to it
/// and rejects the rest.
#[napi(object)]
#[derive(Default)]
pub struct MaterialOptions {
    /// `0xRRGGBB` (sRGB), default white.
    pub color: Option<u32>,
    pub emissive: Option<u32>,
    pub emissive_intensity: Option<f64>,
    /// 0..1, default 1.
    pub roughness: Option<f64>,
    /// 0..1, default 0.
    pub metalness: Option<f64>,
    /// 0..1; below 1 the material is drawn transparent.
    pub opacity: Option<f64>,
    /// Force alpha blending (`basic`); the lit materials blend whenever
    /// `opacity < 1`.
    pub transparent: Option<bool>,
    pub wireframe: Option<bool>,
    #[napi(ts_type = "'front' | 'back' | 'double'")]
    pub side: Option<String>,
    /// `phong` specular colour.
    pub specular: Option<u32>,
    /// `phong` shininess, default 30.
    pub shininess: Option<f64>,
    pub clearcoat: Option<f64>,
    pub clearcoat_roughness: Option<f64>,
    pub transmission: Option<f64>,
    /// Index of refraction, 1..2.333, default 1.5.
    pub ior: Option<f64>,
    pub thickness: Option<f64>,
    /// `physical`: three.js `specularIntensity`, 0..1, default 1.
    pub specular_intensity: Option<f64>,
    /// `physical`: three.js `specularColor`, default white.
    pub specular_color: Option<u32>,
    /// `physical`: sheen layer weight, 0..1.
    pub sheen: Option<f64>,
    pub sheen_color: Option<u32>,
    /// 0..1, default 1.
    pub sheen_roughness: Option<f64>,
    /// Thin-film iridescence weight, 0..1.
    pub iridescence: Option<f64>,
    /// The film's index of refraction, 1..2.333, default 1.3.
    #[napi(js_name = "iridescenceIOR")]
    pub iridescence_ior: Option<f64>,
    /// The film's thickness in nm, `[min, max]`, default `[100, 400]`.
    #[napi(ts_type = "[number, number]")]
    pub iridescence_thickness_range: Option<Vec<f64>>,
    /// Anisotropic stretch of the highlight, 0..1, default 0.
    pub anisotropy: Option<f64>,
    /// Radians, default 0.
    pub anisotropy_rotation: Option<f64>,
    /// Volume absorption colour, default white.
    pub attenuation_color: Option<u32>,
    /// Distance light travels before it is tinted `attenuationColor`;
    /// default infinite.
    pub attenuation_distance: Option<f64>,
    /// Chromatic spread of the refraction, default 0.
    pub dispersion: Option<f64>,
    /// `line`: test against the depth buffer (default true); false draws the
    /// lines over everything.
    pub depth_test: Option<bool>,
}

#[derive(Clone, Copy, PartialEq)]
enum MatKind {
    Basic,
    Standard,
    Physical,
    Lambert,
    Phong,
    Toon,
    Normal,
    Line,
}

impl MatKind {
    fn name(self) -> &'static str {
        match self {
            MatKind::Basic => "basic",
            MatKind::Standard => "standard",
            MatKind::Physical => "physical",
            MatKind::Lambert => "lambert",
            MatKind::Phong => "phong",
            MatKind::Toon => "toon",
            MatKind::Normal => "normal",
            MatKind::Line => "line",
        }
    }
}

#[napi]
impl JsMaterial {
    /// three.js `MeshBasicMaterial` — unlit flat colour.
    #[napi(factory)]
    pub fn basic(opts: Option<MaterialOptions>) -> Result<Self> {
        Self::build(MatKind::Basic, opts)
    }

    /// three.js `MeshStandardMaterial` — metal/rough PBR.
    #[napi(factory)]
    pub fn standard(opts: Option<MaterialOptions>) -> Result<Self> {
        Self::build(MatKind::Standard, opts)
    }

    /// three.js `MeshPhysicalMaterial` — standard + clearcoat, transmission, IOR.
    #[napi(factory)]
    pub fn physical(opts: Option<MaterialOptions>) -> Result<Self> {
        Self::build(MatKind::Physical, opts)
    }

    /// three.js `MeshLambertMaterial` — diffuse only.
    #[napi(factory)]
    pub fn lambert(opts: Option<MaterialOptions>) -> Result<Self> {
        Self::build(MatKind::Lambert, opts)
    }

    /// three.js `MeshPhongMaterial` — diffuse + specular highlight.
    #[napi(factory)]
    pub fn phong(opts: Option<MaterialOptions>) -> Result<Self> {
        Self::build(MatKind::Phong, opts)
    }

    /// three.js `MeshToonMaterial` — cel shading: each light read through the
    /// `gradientMap` (`setMap('gradientMap', texture)`), two tones without
    /// one.
    #[napi(factory)]
    pub fn toon(opts: Option<MaterialOptions>) -> Result<Self> {
        Self::build(MatKind::Toon, opts)
    }

    /// three.js `MeshNormalMaterial` — normals as colours (unlit).
    #[napi(factory)]
    pub fn normal(opts: Option<MaterialOptions>) -> Result<Self> {
        Self::build(MatKind::Normal, opts)
    }

    /// three.js `LineBasicMaterial`, for `Scene.addLineSegments` (`color`,
    /// `opacity`, `depthTest`).
    #[napi(factory)]
    pub fn line(opts: Option<MaterialOptions>) -> Result<Self> {
        Self::build(MatKind::Line, opts)
    }

    /// `'basic' | 'standard' | 'physical' | 'lambert' | 'phong' | 'toon' |
    /// 'normal' | 'line'`.
    #[napi(getter)]
    pub fn kind(&self) -> &'static str {
        match self.inner {
            Material::Basic(_) => "basic",
            Material::Standard(_) => "standard",
            Material::Physical(_) => "physical",
            Material::Lambert(_) => "lambert",
            Material::Phong(_) => "phong",
            Material::Toon(_) => "toon",
            Material::Normal(_) => "normal",
            Material::Line(_) => "line",
            _ => "other",
        }
    }

    /// three.js `material[slot] = texture`: `'map'`, `'alphaMap'`,
    /// `'normalMap'`, `'roughnessMap'`, …, and Physical's `'clearcoatMap'`,
    /// `'sheenColorMap'`, `'transmissionMap'` and the rest. `null` empties the
    /// slot. Throws for a slot this material does not have.
    #[napi]
    pub fn set_map(
        &mut self,
        #[napi(
            ts_arg_type = "'map' | 'alphaMap' | 'normalMap' | 'aoMap' | 'emissiveMap' | 'roughnessMap' | 'metalnessMap' | 'lightMap' | 'bumpMap' | 'specularMap' | 'displacementMap' | 'matcap' | 'cloudShadowMap' | 'gradientMap' | 'clearcoatMap' | 'clearcoatRoughnessMap' | 'clearcoatNormalMap' | 'iridescenceMap' | 'iridescenceThicknessMap' | 'sheenColorMap' | 'sheenRoughnessMap' | 'specularIntensityMap' | 'specularColorMap' | 'transmissionMap' | 'thicknessMap' | 'anisotropyMap'"
        )]
        slot: String,
        #[napi(ts_arg_type = "Texture | null")] texture: Option<Instance<JsTexture>>,
    ) -> Result<()> {
        let kind = self.kind();
        match self.inner.map_slot_mut(&slot) {
            Some(s) => {
                *s = texture.map(|t| t.inner.clone());
                Ok(())
            }
            None => Err(bad(format!("Material.{kind} has no '{slot}' map"))),
        }
    }

    /// The texture in a map slot, or `null`.
    #[napi(ts_return_type = "Texture | null")]
    pub fn get_map(
        &self,
        #[napi(
            ts_arg_type = "'map' | 'alphaMap' | 'normalMap' | 'aoMap' | 'emissiveMap' | 'roughnessMap' | 'metalnessMap' | 'lightMap' | 'bumpMap' | 'specularMap' | 'displacementMap' | 'matcap' | 'cloudShadowMap' | 'gradientMap' | 'clearcoatMap' | 'clearcoatRoughnessMap' | 'clearcoatNormalMap' | 'iridescenceMap' | 'iridescenceThicknessMap' | 'sheenColorMap' | 'sheenRoughnessMap' | 'specularIntensityMap' | 'specularColorMap' | 'transmissionMap' | 'thicknessMap' | 'anisotropyMap'"
        )]
        slot: String,
    ) -> Result<Option<JsTexture>> {
        let mut m = self.inner.clone();
        match m.map_slot_mut(&slot) {
            Some(s) => Ok(s.clone().map(|inner| JsTexture { inner })),
            None => Err(bad(format!("Material.{} has no '{slot}' map", self.kind()))),
        }
    }
}

/// three.js `Texture`: RGBA8 pixels (rows top first) with the sampling state
/// three.js gives every texture — wrapping, filtering, the uv transform
/// (`offset`, `repeat`, `rotation`, `center`), the uv set it reads
/// (`channel`) and whether it holds sRGB colour or data. A material keeps the
/// state the texture had when `setMap` was called.
#[napi(js_name = "Texture")]
pub struct JsTexture {
    inner: std::sync::Arc<threers::textures::Texture>,
}

impl JsClass for JsTexture {
    const JS_NAME: &'static str = "Texture\0";
}

fn texture_format(color_space: Option<&str>) -> Result<threers::textures::TextureFormat> {
    match color_space {
        None | Some("srgb") => Ok(threers::textures::TextureFormat::Rgba8UnormSrgb),
        Some("linear") => Ok(threers::textures::TextureFormat::Rgba8Unorm),
        Some(s) => Err(bad(format!(
            "colorSpace must be 'srgb' or 'linear', got '{s}'"
        ))),
    }
}

fn parse_wrap(s: &str) -> Result<threers::textures::TextureWrap> {
    use threers::textures::TextureWrap::*;
    match s {
        "repeat" => Ok(Repeat),
        "mirror" => Ok(MirroredRepeat),
        "clamp" => Ok(ClampToEdge),
        s => Err(bad(format!(
            "wrap must be 'repeat', 'mirror' or 'clamp', got '{s}'"
        ))),
    }
}

fn wrap_name(w: threers::textures::TextureWrap) -> &'static str {
    use threers::textures::TextureWrap::*;
    match w {
        Repeat => "repeat",
        MirroredRepeat => "mirror",
        ClampToEdge => "clamp",
    }
}

fn parse_filter(s: &str) -> Result<threers::textures::TextureFilter> {
    match s {
        "linear" => Ok(threers::textures::TextureFilter::Linear),
        "nearest" => Ok(threers::textures::TextureFilter::Nearest),
        s => Err(bad(format!(
            "filter must be 'linear' or 'nearest', got '{s}'"
        ))),
    }
}

fn filter_name(f: threers::textures::TextureFilter) -> &'static str {
    match f {
        threers::textures::TextureFilter::Linear => "linear",
        threers::textures::TextureFilter::Nearest => "nearest",
    }
}

fn pair(v: &[f64], name: &str) -> Result<Vector2> {
    match v {
        &[x, y] if x.is_finite() && y.is_finite() => Ok(Vector2::new(x as f32, y as f32)),
        _ => Err(bad(format!("{name} takes [x, y], got {v:?}"))),
    }
}

impl JsTexture {
    fn edit(&mut self) -> &mut threers::textures::Texture {
        std::sync::Arc::make_mut(&mut self.inner)
    }
}

#[napi]
impl JsTexture {
    /// `width × height` RGBA8 pixels, rows top first. `colorSpace` `'srgb'`
    /// (colour — three.js's `SRGBColorSpace`, the default) or `'linear'`
    /// (data: normals, roughness, …).
    #[napi(constructor)]
    pub fn new(
        width: u32,
        height: u32,
        data: Uint8Array,
        #[napi(ts_arg_type = "'srgb' | 'linear'")] color_space: Option<String>,
    ) -> Result<Self> {
        let need = width as usize * height as usize * 4;
        if width == 0 || height == 0 || data.len() != need {
            return Err(bad(format!(
                "a {width}x{height} RGBA texture takes {need} bytes, got {}",
                data.len()
            )));
        }
        let mut t = threers::textures::Texture::new(
            width,
            height,
            texture_format(color_space.as_deref())?,
            data.to_vec(),
        );
        // Rows arrive top first; three.js's image upload flips them (`flipY`).
        t.flip_y = true;
        Ok(Self {
            inner: std::sync::Arc::new(t),
        })
    }

    /// A PNG file, as three.js's `TextureLoader` gives it.
    #[napi(factory, ts_return_type = "Texture")]
    pub fn from_file(
        path: String,
        #[napi(ts_arg_type = "'srgb' | 'linear'")] color_space: Option<String>,
    ) -> Result<Self> {
        let bytes = std::fs::read(&path).map_err(|e| bad(format!("{path}: {e}")))?;
        let png = threers::utils::png::decode_png(&bytes)
            .map_err(|e| bad(format!("{path}: not a PNG this loader reads ({e})")))?;
        Self::new(png.width, png.height, png.rgba.into(), color_space)
    }

    #[napi(getter)]
    pub fn width(&self) -> u32 {
        self.inner.width
    }
    #[napi(getter)]
    pub fn height(&self) -> u32 {
        self.inner.height
    }
    #[napi(getter, ts_return_type = "'srgb' | 'linear'")]
    pub fn color_space(&self) -> &'static str {
        if self.inner.format == threers::textures::TextureFormat::Rgba8UnormSrgb {
            "srgb"
        } else {
            "linear"
        }
    }
    #[napi(setter, ts_args_type = "v: 'srgb' | 'linear'")]
    pub fn set_color_space(&mut self, v: String) -> Result<()> {
        self.edit().format = texture_format(Some(&v))?;
        Ok(())
    }
    #[napi(getter, ts_return_type = "'repeat' | 'mirror' | 'clamp'")]
    pub fn wrap_s(&self) -> &'static str {
        wrap_name(self.inner.wrap_s)
    }
    #[napi(setter, ts_args_type = "v: 'repeat' | 'mirror' | 'clamp'")]
    pub fn set_wrap_s(&mut self, v: String) -> Result<()> {
        self.edit().wrap_s = parse_wrap(&v)?;
        Ok(())
    }
    #[napi(getter, ts_return_type = "'repeat' | 'mirror' | 'clamp'")]
    pub fn wrap_t(&self) -> &'static str {
        wrap_name(self.inner.wrap_t)
    }
    #[napi(setter, ts_args_type = "v: 'repeat' | 'mirror' | 'clamp'")]
    pub fn set_wrap_t(&mut self, v: String) -> Result<()> {
        self.edit().wrap_t = parse_wrap(&v)?;
        Ok(())
    }
    #[napi(getter, ts_return_type = "'linear' | 'nearest'")]
    pub fn mag_filter(&self) -> &'static str {
        filter_name(self.inner.mag_filter)
    }
    #[napi(setter, ts_args_type = "v: 'linear' | 'nearest'")]
    pub fn set_mag_filter(&mut self, v: String) -> Result<()> {
        self.edit().mag_filter = parse_filter(&v)?;
        Ok(())
    }
    #[napi(getter, ts_return_type = "'linear' | 'nearest'")]
    pub fn min_filter(&self) -> &'static str {
        filter_name(self.inner.min_filter)
    }
    #[napi(setter, ts_args_type = "v: 'linear' | 'nearest'")]
    pub fn set_min_filter(&mut self, v: String) -> Result<()> {
        self.edit().min_filter = parse_filter(&v)?;
        Ok(())
    }
    /// Between mip levels: `'linear'` (three.js's `*MipmapLinearFilter`, the
    /// default), `'nearest'` (`*MipmapNearestFilter`) or `null` (level 0
    /// only — `LinearFilter` / `NearestFilter`).
    #[napi(getter, ts_return_type = "'linear' | 'nearest' | null")]
    pub fn mipmap_filter(&self) -> Option<&'static str> {
        self.inner.mipmap_filter.map(filter_name)
    }
    #[napi(setter, ts_args_type = "v: 'linear' | 'nearest' | null")]
    pub fn set_mipmap_filter(&mut self, v: Option<String>) -> Result<()> {
        self.edit().mipmap_filter = v.as_deref().map(parse_filter).transpose()?;
        Ok(())
    }
    #[napi(getter, ts_return_type = "[number, number]")]
    pub fn offset(&self) -> Vec<f64> {
        vec![self.inner.offset.x as f64, self.inner.offset.y as f64]
    }
    #[napi(setter, ts_args_type = "v: [number, number]")]
    pub fn set_offset(&mut self, v: Vec<f64>) -> Result<()> {
        let p = pair(&v, "offset")?;
        let t = self.edit();
        t.offset = p;
        t.matrix_auto_update = true;
        Ok(())
    }
    #[napi(getter, ts_return_type = "[number, number]")]
    pub fn repeat(&self) -> Vec<f64> {
        vec![self.inner.repeat.x as f64, self.inner.repeat.y as f64]
    }
    #[napi(setter, ts_args_type = "v: [number, number]")]
    pub fn set_repeat(&mut self, v: Vec<f64>) -> Result<()> {
        let p = pair(&v, "repeat")?;
        let t = self.edit();
        t.repeat = p;
        t.matrix_auto_update = true;
        Ok(())
    }
    #[napi(getter)]
    pub fn rotation(&self) -> f64 {
        self.inner.rotation as f64
    }
    #[napi(setter)]
    pub fn set_rotation(&mut self, v: f64) -> Result<()> {
        if !v.is_finite() {
            return Err(bad(format!("rotation must be finite, got {v}")));
        }
        let t = self.edit();
        t.rotation = v as f32;
        t.matrix_auto_update = true;
        Ok(())
    }
    #[napi(getter, ts_return_type = "[number, number]")]
    pub fn center(&self) -> Vec<f64> {
        vec![self.inner.center.x as f64, self.inner.center.y as f64]
    }
    #[napi(setter, ts_args_type = "v: [number, number]")]
    pub fn set_center(&mut self, v: Vec<f64>) -> Result<()> {
        let p = pair(&v, "center")?;
        let t = self.edit();
        t.center = p;
        t.matrix_auto_update = true;
        Ok(())
    }
    /// three.js `channel`: the uv set read — 0 `uv`, 1–3 `uv1`–`uv3`.
    #[napi(getter)]
    pub fn channel(&self) -> u32 {
        self.inner.channel
    }
    #[napi(setter)]
    pub fn set_channel(&mut self, v: u32) -> Result<()> {
        if v > 3 {
            return Err(bad(format!("channel must be 0-3, got {v}")));
        }
        self.edit().channel = v;
        Ok(())
    }
    #[napi(getter)]
    pub fn flip_y(&self) -> bool {
        self.inner.flip_y
    }
    #[napi(setter)]
    pub fn set_flip_y(&mut self, v: bool) {
        self.edit().flip_y = v;
    }
}

impl JsMaterial {
    fn build(kind: MatKind, opts: Option<MaterialOptions>) -> Result<Self> {
        let o = opts.unwrap_or_default();
        use MatKind::*;
        let fields: [(&str, bool, &[MatKind]); 30] = [
            (
                "color",
                o.color.is_some(),
                &[Basic, Standard, Physical, Lambert, Phong, Toon, Line],
            ),
            (
                "emissive",
                o.emissive.is_some(),
                &[Standard, Physical, Lambert, Phong, Toon],
            ),
            (
                "emissiveIntensity",
                o.emissive_intensity.is_some(),
                &[Standard, Physical],
            ),
            ("roughness", o.roughness.is_some(), &[Standard, Physical]),
            ("metalness", o.metalness.is_some(), &[Standard, Physical]),
            (
                "opacity",
                o.opacity.is_some(),
                &[
                    Basic, Standard, Physical, Lambert, Phong, Toon, Normal, Line,
                ],
            ),
            (
                "transparent",
                o.transparent.is_some(),
                &[Basic, Standard, Physical, Lambert, Phong, Toon, Normal],
            ),
            (
                "wireframe",
                o.wireframe.is_some(),
                &[Basic, Standard, Physical, Lambert, Phong, Toon, Normal],
            ),
            (
                "side",
                o.side.is_some(),
                &[Basic, Standard, Physical, Lambert, Phong, Toon],
            ),
            ("specular", o.specular.is_some(), &[Phong]),
            ("shininess", o.shininess.is_some(), &[Phong]),
            ("clearcoat", o.clearcoat.is_some(), &[Physical]),
            (
                "clearcoatRoughness",
                o.clearcoat_roughness.is_some(),
                &[Physical],
            ),
            ("transmission", o.transmission.is_some(), &[Physical]),
            ("ior", o.ior.is_some(), &[Physical]),
            ("thickness", o.thickness.is_some(), &[Physical]),
            ("depthTest", o.depth_test.is_some(), &[Line]),
            (
                "specularIntensity",
                o.specular_intensity.is_some(),
                &[Physical],
            ),
            ("specularColor", o.specular_color.is_some(), &[Physical]),
            ("sheen", o.sheen.is_some(), &[Physical]),
            ("sheenColor", o.sheen_color.is_some(), &[Physical]),
            ("sheenRoughness", o.sheen_roughness.is_some(), &[Physical]),
            ("iridescence", o.iridescence.is_some(), &[Physical]),
            ("iridescenceIOR", o.iridescence_ior.is_some(), &[Physical]),
            (
                "iridescenceThicknessRange",
                o.iridescence_thickness_range.is_some(),
                &[Physical],
            ),
            ("anisotropy", o.anisotropy.is_some(), &[Physical]),
            (
                "anisotropyRotation",
                o.anisotropy_rotation.is_some(),
                &[Physical],
            ),
            (
                "attenuationColor",
                o.attenuation_color.is_some(),
                &[Physical],
            ),
            (
                "attenuationDistance",
                o.attenuation_distance.is_some(),
                &[Physical],
            ),
            ("dispersion", o.dispersion.is_some(), &[Physical]),
        ];
        for (name, set, allowed) in fields {
            if set && !allowed.contains(&kind) {
                return Err(bad(format!(
                    "Material.{}: '{name}' does not apply to this material",
                    kind.name()
                )));
            }
        }
        let color = hex(o.color, 0xffffff, "color")?;
        let emissive = hex(o.emissive, 0x000000, "emissive")?;
        let opacity = unit(o.opacity, 1.0, "opacity")?;
        let wireframe = o.wireframe.unwrap_or(false);
        let side = match o.side.as_deref() {
            None | Some("front") => 0,
            Some("back") => 1,
            Some("double") => 2,
            Some(s) => {
                return Err(bad(format!(
                    "side must be 'front', 'back' or 'double', got '{s}'"
                )))
            }
        };
        let roughness = unit(o.roughness, 1.0, "roughness")?;
        let metalness = unit(o.metalness, 0.0, "metalness")?;
        let emissive_intensity = non_neg(o.emissive_intensity, 1.0, "emissiveIntensity")?;
        let inner = match kind {
            Basic => Material::Basic(BasicMaterial {
                color,
                opacity,
                transparent: o.transparent.unwrap_or(false) || opacity < 1.0,
                wireframe,
                side,
                ..BasicMaterial::default()
            }),
            Standard => Material::Standard(StandardMaterial {
                color,
                emissive,
                emissive_intensity,
                roughness,
                metalness,
                opacity,
                wireframe,
                side,
                ..StandardMaterial::default()
            }),
            Physical => {
                let ior = o.ior.unwrap_or(1.5);
                if !(ior.is_finite() && (1.0..=2.333).contains(&ior)) {
                    return Err(bad(format!("ior must be in [1, 2.333], got {ior}")));
                }
                Material::Physical(PhysicalMaterial {
                    color,
                    emissive,
                    emissive_intensity,
                    roughness,
                    metalness,
                    opacity,
                    wireframe,
                    side,
                    clearcoat: unit(o.clearcoat, 0.0, "clearcoat")?,
                    clearcoat_roughness: unit(o.clearcoat_roughness, 0.0, "clearcoatRoughness")?,
                    transmission: unit(o.transmission, 0.0, "transmission")?,
                    ior: ior as f32,
                    thickness: non_neg(o.thickness, 0.0, "thickness")?,
                    specular_intensity: unit(o.specular_intensity, 1.0, "specularIntensity")?,
                    specular_color: hex(o.specular_color, 0xffffff, "specularColor")?,
                    sheen: unit(o.sheen, 0.0, "sheen")?,
                    sheen_color: hex(o.sheen_color, 0x000000, "sheenColor")?,
                    sheen_roughness: unit(o.sheen_roughness, 1.0, "sheenRoughness")?,
                    iridescence: unit(o.iridescence, 0.0, "iridescence")?,
                    iridescence_ior: {
                        let v = o.iridescence_ior.unwrap_or(1.3);
                        if !(v.is_finite() && (1.0..=2.333).contains(&v)) {
                            return Err(bad(format!(
                                "iridescenceIOR must be in [1, 2.333], got {v}"
                            )));
                        }
                        v as f32
                    },
                    iridescence_thickness_range: match o.iridescence_thickness_range.as_deref() {
                        None => [100.0, 400.0],
                        Some(&[lo, hi]) => [
                            non_neg(Some(lo), 0.0, "iridescenceThicknessRange")?,
                            non_neg(Some(hi), 0.0, "iridescenceThicknessRange")?,
                        ],
                        Some(v) => {
                            return Err(bad(format!(
                                "iridescenceThicknessRange takes [min, max], got {} numbers",
                                v.len()
                            )))
                        }
                    },
                    anisotropy: unit(o.anisotropy, 0.0, "anisotropy")?,
                    anisotropy_rotation: match o.anisotropy_rotation {
                        Some(v) if !v.is_finite() => {
                            return Err(bad(format!("anisotropyRotation must be finite, got {v}")))
                        }
                        v => v.unwrap_or(0.0) as f32,
                    },
                    attenuation_color: hex(o.attenuation_color, 0xffffff, "attenuationColor")?,
                    attenuation_distance: non_neg(
                        o.attenuation_distance,
                        f32::INFINITY,
                        "attenuationDistance",
                    )?,
                    dispersion: non_neg(o.dispersion, 0.0, "dispersion")?,
                    ..PhysicalMaterial::default()
                })
            }
            Lambert => Material::Lambert(LambertMaterial {
                color,
                emissive,
                opacity,
                wireframe,
                side,
                ..LambertMaterial::default()
            }),
            Toon => Material::Toon(threers::materials::ToonMaterial {
                color,
                emissive,
                opacity,
                wireframe,
                side,
                ..threers::materials::ToonMaterial::default()
            }),
            Phong => Material::Phong(PhongMaterial {
                color,
                emissive,
                specular: hex(o.specular, 0x111111, "specular")?,
                shininess: non_neg(o.shininess, 30.0, "shininess")?,
                opacity,
                wireframe,
                side,
                ..PhongMaterial::default()
            }),
            Normal => Material::Normal(NormalMaterial {
                opacity,
                wireframe,
                ..NormalMaterial::default()
            }),
            Line => Material::Line(threers::materials::LineBasicMaterial {
                color,
                opacity,
                render_state: threers::RenderState {
                    depth_test: o.depth_test.unwrap_or(true),
                    ..Default::default()
                },
                ..threers::materials::LineBasicMaterial::default()
            }),
        };
        Ok(Self { inner })
    }
}

// ---- argument helpers ------------------------------------------------------

fn bad(msg: impl Into<String>) -> Error {
    Error::new(Status::InvalidArg, msg.into())
}

/// `0xRRGGBB` → linear colour.
fn hex(v: Option<u32>, default: u32, name: &str) -> Result<Color> {
    let v = v.unwrap_or(default);
    if v > 0xff_ffff {
        return Err(bad(format!("{name} must be 0xRRGGBB, got {v:#x}")));
    }
    Ok(Color::from_hex(v))
}

fn non_neg(v: Option<f64>, default: f32, name: &str) -> Result<f32> {
    match v {
        None => Ok(default),
        Some(v) if v.is_finite() && v >= 0.0 => Ok(v as f32),
        Some(v) => Err(bad(format!("{name} must be a finite number >= 0, got {v}"))),
    }
}

fn unit(v: Option<f64>, default: f32, name: &str) -> Result<f32> {
    match v {
        None => Ok(default),
        Some(v) if (0.0..=1.0).contains(&v) => Ok(v as f32),
        Some(v) => Err(bad(format!("{name} must be in [0, 1], got {v}"))),
    }
}

fn intensity_arg(v: Option<f64>, what: &str) -> Result<f32> {
    non_neg(v, 1.0, &format!("{what}: intensity"))
}

fn seg(v: Option<u32>, default: u32, min: u32, name: &str) -> Result<usize> {
    let v = v.unwrap_or(default);
    if !(min..=MAX_SEGMENTS).contains(&v) {
        return Err(bad(format!(
            "{name} must be {min}..={MAX_SEGMENTS}, got {v}"
        )));
    }
    Ok(v as usize)
}

fn finite3(x: f64, y: f64, z: f64, what: &str) -> Result<[f32; 3]> {
    if ![x, y, z].iter().all(|v| v.is_finite()) {
        return Err(bad(format!("{what}: values must be finite")));
    }
    Ok([x as f32, y as f32, z as f32])
}

fn opt_point(x: Option<f64>, y: Option<f64>, z: Option<f64>, what: &str) -> Result<Vector3> {
    let [x, y, z] = finite3(x.unwrap_or(0.0), y.unwrap_or(0.0), z.unwrap_or(0.0), what)?;
    Ok(Vector3::new(x, y, z))
}

fn opt_vec(v: Option<Vector3Like>, default: Vector3, what: &str) -> Result<Vector3> {
    match v {
        None => Ok(default),
        Some(v) => {
            let [x, y, z] = finite3(v.x, v.y, v.z, what)?;
            Ok(Vector3::new(x, y, z))
        }
    }
}

// ---- scene helpers shared by the methods above -----------------------------

/// Set `material` on every mesh in the subtree at `id`; returns how many.
fn override_materials(scene: &mut Scene, id: ObjectId, material: &Material) -> usize {
    let mut ids = Vec::new();
    scene.arena.traverse(id, &mut |i, o| {
        if matches!(o.kind, ObjectKind::Mesh(_)) {
            ids.push(i);
        }
    });
    let mat = std::sync::Arc::new(material.clone());
    for i in &ids {
        if let Some(ObjectKind::Mesh(m)) = scene.arena.get_mut(*i).map(|o| &mut o.kind) {
            m.material = mat.clone();
        }
    }
    ids.len()
}

fn remove_subtree(scene: &mut Scene, id: ObjectId) {
    scene.arena.remove_child(id);
    let mut ids = Vec::new();
    scene.arena.traverse(id, &mut |i, _| ids.push(i));
    for i in ids {
        scene.arena.nodes.remove(i);
    }
}

fn geometry_from_arrays(
    positions: Vec<f32>,
    indices: Option<Vec<u32>>,
    normals: Option<Vec<f32>>,
) -> std::result::Result<BufferGeometry, String> {
    if positions.is_empty() || !positions.len().is_multiple_of(3) {
        return Err(format!(
            "positions needs a non-zero multiple of 3 numbers, got {}",
            positions.len()
        ));
    }
    if positions.iter().any(|v| !v.is_finite()) {
        return Err("positions has a non-finite value".into());
    }
    let n = (positions.len() / 3) as u32;
    let mut g = BufferGeometry::new();
    match &indices {
        Some(ix) => {
            if ix.is_empty() || !ix.len().is_multiple_of(3) {
                return Err(format!(
                    "indices needs a non-zero multiple of 3 entries, got {}",
                    ix.len()
                ));
            }
            if let Some(bad) = ix.iter().find(|i| **i >= n) {
                return Err(format!("index {bad} out of range for {n} vertices"));
            }
        }
        None if !n.is_multiple_of(3) => {
            return Err(format!(
                "without indices the vertex count must be a multiple of 3, got {n}"
            ))
        }
        None => {}
    }
    g.set_attribute("position", BufferAttribute::new(positions.clone(), 3));
    if let Some(ix) = indices {
        g.set_index(ix);
    }
    match normals {
        Some(nm) => {
            if nm.len() != positions.len() || nm.iter().any(|v| !v.is_finite()) {
                return Err(format!(
                    "normals needs {} finite numbers (one per position), got {}",
                    positions.len(),
                    nm.len()
                ));
            }
            g.set_attribute("normal", BufferAttribute::new(nm, 3));
        }
        None => threers::compute_vertex_normals(&mut g),
    }
    Ok(g)
}

fn check_polygons(counts: &[u32], triangles: usize) -> std::result::Result<(), String> {
    if counts.contains(&0) {
        return Err("setPolygons: every polygon needs at least 1 triangle".into());
    }
    let total: u64 = counts.iter().map(|&n| n as u64).sum();
    if total != triangles as u64 {
        return Err(format!(
            "setPolygons: counts sum to {total} triangles, the geometry has {triangles}"
        ));
    }
    Ok(())
}

fn polygon_loops(g: &BufferGeometry) -> Vec<Option<Vec<u32>>> {
    let ranges: Vec<_> = g.polygon_ranges().collect();
    ranges.into_iter().map(|r| g.polygon_loop(r)).collect()
}

// ---- loaders ---------------------------------------------------------------

#[derive(Clone, Copy)]
enum ModelFormat {
    Glb,
    Gltf,
    Obj,
    Stl,
    Ply,
    Fbx,
    Usd,
}

fn parse_format(name: &str) -> std::result::Result<ModelFormat, String> {
    Ok(match name.to_ascii_lowercase().as_str() {
        "glb" => ModelFormat::Glb,
        "gltf" => ModelFormat::Gltf,
        "obj" => ModelFormat::Obj,
        "stl" => ModelFormat::Stl,
        "ply" => ModelFormat::Ply,
        "fbx" => ModelFormat::Fbx,
        "usd" | "usda" | "usdc" | "usdz" => ModelFormat::Usd,
        other => {
            let known = "glb, gltf, obj, stl, ply, fbx, usd, usda, usdc, usdz";
            return Err(format!("unknown model format '{other}' ({known})"));
        }
    })
}

fn format_from_path(p: &std::path::Path) -> std::result::Result<ModelFormat, String> {
    let ext = p.extension().and_then(|e| e.to_str()).ok_or_else(|| {
        format!(
            "load: cannot tell the format of '{}' (no extension)",
            p.display()
        )
    })?;
    parse_format(ext)
}

/// Load `bytes` into `scene`; returns the id to hand out as the handle.
fn load_into(
    scene: &mut Scene,
    bytes: &[u8],
    fmt: ModelFormat,
    base_dir: Option<&std::path::Path>,
    material: Option<Material>,
) -> std::result::Result<ObjectId, String> {
    use threers::loaders::GltfLoader;
    let mesh_of = |mut g: BufferGeometry, material: Option<Material>| {
        if g.get_attribute("position").map_or(0, |a| a.count()) == 0 {
            return Err("file contains no geometry".to_string());
        }
        if g.get_attribute("normal").is_none() {
            threers::compute_vertex_normals(&mut g);
        }
        let mat = material.unwrap_or_else(|| {
            Material::Standard(StandardMaterial::new(Color::from_hex(0xcccccc)))
        });
        Ok(Object3D::mesh(Mesh::new(g, mat)))
    };
    let loaded: LoadedScene = match fmt {
        ModelFormat::Obj => {
            let text = std::str::from_utf8(bytes).map_err(|_| "OBJ is not UTF-8 text")?;
            let obj = mesh_of(threers::loaders::ObjLoader::parse(text), material)?;
            return Ok(scene.add(obj));
        }
        ModelFormat::Stl => {
            let obj = mesh_of(threers::loaders::StlLoader::parse(bytes), material)?;
            return Ok(scene.add(obj));
        }
        ModelFormat::Ply => {
            let obj = mesh_of(threers::loaders::PlyLoader::parse(bytes), material)?;
            return Ok(scene.add(obj));
        }
        ModelFormat::Glb => GltfLoader::parse_glb(bytes)
            .map_err(|e| format!("glTF: {e:?}"))?
            .into(),
        ModelFormat::Gltf => {
            let text = std::str::from_utf8(bytes).map_err(|_| "glTF JSON is not UTF-8")?;
            let mut external = std::collections::HashMap::new();
            if let Some(dir) = base_dir {
                for (i, uri) in gltf_buffer_uris(text).into_iter().enumerate() {
                    let Some(uri) = uri else { continue };
                    if uri.starts_with("data:") {
                        continue;
                    }
                    let path = dir.join(percent_decode(&uri));
                    let data = std::fs::read(&path)
                        .map_err(|e| format!("buffer {i} ({}): {e}", path.display()))?;
                    external.insert(i, data);
                }
            }
            GltfLoader::parse_gltf(text, &external)
                .map_err(|e| format!("glTF: {e:?}"))?
                .into()
        }
        ModelFormat::Fbx => {
            let s =
                threers::loaders::FbxLoader::parse_scene(bytes).map_err(|e| format!("FBX: {e}"))?;
            if s.roots.is_empty() {
                return Err("FBX: file contains no geometry".into());
            }
            LoadedScene {
                up_axis_z: s.up_axis == 2,
                scene: threers::loaders::GltfScene {
                    roots: s.roots,
                    arena: s.arena,
                    animations: Vec::new(),
                    skeletons: Vec::new(),
                },
            }
        }
        ModelFormat::Usd => load_usd(bytes)?,
    };
    let z_up = loaded.up_axis_z;
    let loaded = loaded.scene;
    let roots = threers::loaders::gltf_add_to_scene(scene, loaded);
    let mut group = Object3D::group();
    if z_up {
        // Bring a Z-up stage into three.js's Y-up world.
        group.quaternion = Quaternion::from_euler_xyz(-std::f32::consts::FRAC_PI_2, 0.0, 0.0);
    }
    let gid = scene.add(group);
    for r in roots {
        scene.arena.add_child(gid, r);
    }
    if let Some(m) = material {
        override_materials(scene, gid, &m);
    }
    Ok(gid)
}

#[cfg(feature = "usd")]
fn load_usd(bytes: &[u8]) -> std::result::Result<LoadedScene, String> {
    let s = threers::loaders::UsdLoader::parse(bytes).map_err(|e| format!("USD: {e}"))?;
    Ok(LoadedScene {
        up_axis_z: s.up_axis == 'Z',
        scene: threers::loaders::GltfScene {
            roots: s.roots,
            arena: s.arena,
            animations: Vec::new(),
            skeletons: Vec::new(),
        },
    })
}

#[cfg(not(feature = "usd"))]
fn load_usd(_bytes: &[u8]) -> std::result::Result<LoadedScene, String> {
    Err("this build has no USD support (enable the `usd` feature)".into())
}

/// A parsed scene graph ready to graft into a `Scene`.
struct LoadedScene {
    scene: threers::loaders::GltfScene,
    up_axis_z: bool,
}

impl From<threers::loaders::GltfScene> for LoadedScene {
    fn from(scene: threers::loaders::GltfScene) -> Self {
        Self {
            scene,
            up_axis_z: false,
        }
    }
}

/// The `uri` of each entry of a glTF document's top-level `buffers` array
/// (`None` for entries without one), found with a small string-aware scan
/// so external `.bin` files can be read before parsing.
fn gltf_buffer_uris(json: &str) -> Vec<Option<String>> {
    let b = json.as_bytes();
    let mut out = Vec::new();
    let Some(k) = json.find("\"buffers\"") else {
        return out;
    };
    let Some(open) = json[k..].find('[') else {
        return out;
    };
    let mut i = k + open + 1;
    let mut depth = 0usize; // 0 = directly inside the buffers array
    let mut last_str: Option<String> = None;
    let mut key: Option<String> = None;
    while i < b.len() {
        match b[i] {
            b'"' => {
                let mut raw = Vec::new();
                i += 1;
                while i < b.len() && b[i] != b'"' {
                    if b[i] == b'\\' && i + 1 < b.len() {
                        i += 1;
                    }
                    raw.push(b[i]);
                    i += 1;
                }
                let s = String::from_utf8_lossy(&raw).into_owned();
                if depth == 1 && key.as_deref() == Some("uri") {
                    if let Some(slot) = out.last_mut() {
                        *slot = Some(s.clone());
                    }
                    key = None;
                }
                last_str = Some(s);
            }
            b'{' | b'[' => {
                depth += 1;
                if depth == 1 && b[i] == b'{' {
                    out.push(None);
                }
            }
            b'}' | b']' => {
                if depth == 0 {
                    break;
                }
                depth -= 1;
            }
            b':' if depth == 1 => key = last_str.take(),
            b',' if depth == 1 => key = None,
            _ => {}
        }
        i += 1;
    }
    out
}

fn percent_decode(s: &str) -> String {
    let b = s.as_bytes();
    let mut out = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'%' && i + 2 < b.len() {
            let hex = std::str::from_utf8(&b[i + 1..i + 3]).ok();
            if let Some(v) = hex.and_then(|h| u8::from_str_radix(h, 16).ok()) {
                out.push(v);
                i += 3;
                continue;
            }
        }
        out.push(b[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// One `Scene.raycast` hit: three.js's intersection fields. Those that do
/// not apply to the kind of object hit are undefined, as in three.js.
#[napi(object)]
pub struct RaycastHit {
    /// The object hit.
    pub handle: u32,
    /// World-space distance from the ray origin.
    pub distance: f64,
    /// World-space hit point (for a line, the nearest point on the segment).
    pub point: Vector3Like,
    /// Triangle number within the mesh's geometry (mesh hits).
    pub face_index: Option<u32>,
    /// The triangle hit (mesh hits).
    pub face: Option<RaycastFace>,
    /// The geometry's `normal` attribute interpolated at the hit, in the
    /// object's local space, facing the ray (mesh hits with normals).
    pub normal: Option<Vector3Like>,
    /// The `uv` attribute interpolated at the hit.
    pub uv: Option<Vector2Like>,
    /// The `uv1` attribute interpolated at the hit.
    pub uv1: Option<Vector2Like>,
    /// threers extension: barycentric weights of the hit on `face.a`, `.b`,
    /// `.c`.
    pub barycoord: Option<Vector3Like>,
    /// The instance hit, for an instanced mesh.
    pub instance_id: Option<u32>,
    /// Vertex index: the first vertex of the segment for a line, the point
    /// for points.
    pub index: Option<u32>,
    /// For points: the point's distance from the ray (local space).
    pub distance_to_ray: Option<f64>,
    /// threers extension: polygon (quad) number, when the geometry carries a
    /// polygon grouping.
    pub polygon_index: Option<u32>,
}

/// three.js `intersection.face`.
#[napi(object)]
pub struct RaycastFace {
    pub a: u32,
    pub b: u32,
    pub c: u32,
    /// The triangle's geometric normal, in the object's local space.
    pub normal: Vector3Like,
    /// Material-array slot (always 0 here: no geometry groups).
    pub material_index: u32,
}

/// Options for `Scene.raycast` / `Scene.raycastFromCamera` — three.js
/// `Raycaster` settings.
#[napi(object)]
#[derive(Default)]
pub struct RaycastOptions {
    /// Drop hits nearer than this (default 0).
    pub near: Option<f64>,
    /// Drop hits farther than this (default Infinity).
    pub far: Option<f64>,
    /// Only the nearest hit, found with less work (three-mesh-bvh's
    /// `firstHitOnly`, across all objects).
    pub first_hit_only: Option<bool>,
    /// three.js `params.Line.threshold`, world units (default 1).
    pub line_threshold: Option<f64>,
    /// three.js `params.Points.threshold`, world units (default 1).
    pub points_threshold: Option<f64>,
    /// Test descendants too (default true).
    pub recursive: Option<bool>,
    /// Handles to test (three.js `intersectObjects(objects)`) instead of the
    /// whole scene.
    pub objects: Option<Vec<u32>>,
}

/// Anything with numeric `x`, `y`, `z` — a `Vector3` or a plain object.
#[napi(object, js_name = "Vector3Like")]
pub struct Vector3Like {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

/// `THREE.DoubleSide`.
const DOUBLE_SIDE: u32 = 2;
/// Per-axis cap on `addQuad` subdivision (1024² cells is already 2M triangles).
const MAX_QUAD_SEGMENTS: u32 = 1024;

fn parse_corners(corners: &[Either<f64, Vector3Like>]) -> Result<[Vector3; 4]> {
    let bad = |msg: String| Error::new(Status::InvalidArg, msg);
    let nums: Option<Vec<f64>> = corners
        .iter()
        .map(|c| match c {
            Either::A(n) => Some(*n),
            Either::B(_) => None,
        })
        .collect();
    let flat: Vec<f64> =
        match nums {
            Some(n) if n.len() == 12 => n,
            Some(n) => {
                return Err(bad(format!(
                    "addQuad: corners needs 12 numbers or 4 {{x, y, z}} points, got {} numbers",
                    n.len()
                )))
            }
            None => {
                let pts: Option<Vec<&Vector3Like>> = corners
                    .iter()
                    .map(|c| match c {
                        Either::B(v) => Some(v),
                        Either::A(_) => None,
                    })
                    .collect();
                match pts {
                    Some(p) if p.len() == 4 => p.iter().flat_map(|v| [v.x, v.y, v.z]).collect(),
                    Some(p) => {
                        return Err(bad(format!(
                            "addQuad: corners needs 4 points, got {}",
                            p.len()
                        )))
                    }
                    None => return Err(bad(
                        "addQuad: corners mixes numbers and points; pass 12 numbers or 4 points"
                            .into(),
                    )),
                }
            }
        };
    if let Some(i) = flat.iter().position(|v| !v.is_finite()) {
        return Err(bad(format!(
            "addQuad: corner {} has a non-finite coordinate",
            i / 3
        )));
    }
    let p = |i: usize| {
        Vector3::new(
            flat[3 * i] as f32,
            flat[3 * i + 1] as f32,
            flat[3 * i + 2] as f32,
        )
    };
    let c = [p(0), p(1), p(2), p(3)];
    // Twice the area vector: the diagonals' cross product.
    if (c[2] - c[0]).cross(c[3] - c[1]).length() <= 1e-12 {
        return Err(bad("addQuad: corners are degenerate (zero area)".into()));
    }
    Ok(c)
}

// ---- headless --------------------------------------------------------------

#[cfg(feature = "headless")]
#[napi(js_name = "HeadlessRenderer")]
pub struct JsHeadlessRenderer {
    inner: threers::HeadlessRenderer,
    supersample: u32,
    shadows: bool,
}

#[cfg(feature = "headless")]
#[napi]
impl JsHeadlessRenderer {
    #[napi(constructor)]
    pub fn new(width: Option<u32>, height: Option<u32>, supersample: Option<u32>) -> Result<Self> {
        let supersample = supersample.unwrap_or(1).max(1);
        let inner = build_headless(width.unwrap_or(640), height.unwrap_or(480), supersample)?;
        Ok(Self {
            inner,
            supersample,
            shadows: true,
        })
    }

    /// Render and return tightly packed RGBA8 bytes (`width * height * 4`).
    #[napi]
    pub fn render_rgba(
        &mut self,
        scene: &mut JsScene,
        #[napi(ts_arg_type = "PerspectiveCamera | OrthographicCamera")] camera: AnyCamera,
    ) -> Buffer {
        let shadows = self.shadows;
        let rgba = with_shadows(&mut scene.inner, shadows, |s| {
            self.inner.render_to_rgba(s, camera_of(&camera))
        });
        Buffer::from(rgba)
    }

    /// Render and return a PNG as bytes.
    #[napi]
    pub fn render_png(
        &mut self,
        scene: &mut JsScene,
        #[napi(ts_arg_type = "PerspectiveCamera | OrthographicCamera")] camera: AnyCamera,
    ) -> Buffer {
        let (w, h) = self.inner.render_size();
        let shadows = self.shadows;
        let rgba = with_shadows(&mut scene.inner, shadows, |s| {
            self.inner.render_to_rgba(s, camera_of(&camera))
        });
        let png = threers::encode_png(w, h, &rgba);
        Buffer::from(png)
    }

    #[napi]
    pub fn size(&self) -> Vec<u32> {
        let (w, h) = self.inner.render_size();
        vec![w, h]
    }

    /// Restrict drawing to a rectangle, in output pixels with a bottom-left
    /// origin (three.js `setViewport`).
    #[napi]
    pub fn set_viewport(&mut self, x: f64, y: f64, width: f64, height: f64) {
        self.inner
            .renderer()
            .set_viewport(x as f32, y as f32, width as f32, height as f32);
    }

    /// Scissor rectangle, same units as `setViewport`; only applies while
    /// `setScissorTest(true)`.
    #[napi]
    pub fn set_scissor(&mut self, x: f64, y: f64, width: f64, height: f64) {
        self.inner
            .renderer()
            .set_scissor(x as f32, y as f32, width as f32, height as f32);
    }

    #[napi]
    pub fn set_scissor_test(&mut self, enabled: bool) {
        self.inner.renderer().set_scissor_test(enabled);
    }

    /// Render a CAD-style quad view — top, front and side orthographic panes
    /// framed on the scene's meshes, plus `camera` as the perspective pane —
    /// at the configured output size (supersampling is resolved). Returns PNG
    /// bytes, or RGBA8 with `{ format: 'rgba' }`.
    #[napi]
    pub fn render_quad_view(
        &mut self,
        scene: &mut JsScene,
        camera: &JsPerspectiveCamera,
        opts: Option<QuadViewOptions>,
    ) -> Result<Buffer> {
        use threers::helpers::QuadView;
        let opts = opts.unwrap_or_default();
        let mut quad = QuadView::new();
        quad.maximized = match opts.maximized.as_deref() {
            None => None,
            Some(name) => Some(parse_pane(name)?),
        };
        if let Some(gap) = opts.gap {
            if !(gap.is_finite() && gap >= 0.0) {
                return Err(Error::new(
                    Status::InvalidArg,
                    format!("renderQuadView: gap must be >= 0, got {gap}"),
                ));
            }
            quad.gap = gap as f32;
        }
        if let Some(hex) = opts.gap_color {
            quad.gap_color = Some(Color::from_hex(hex));
        }
        if let Some(border) = opts.border {
            if !(border.is_finite() && border >= 0.0) {
                return Err(bad(format!(
                    "renderQuadView: border must be >= 0, got {border}"
                )));
            }
            quad.border = border as f32;
        }
        if let Some(margin) = opts.margin {
            if !(margin.is_finite() && margin >= 0.0) {
                return Err(bad(format!(
                    "renderQuadView: margin must be >= 0, got {margin}"
                )));
            }
            quad.margin = margin as f32;
        }
        if let Some(layout) = &opts.layout {
            quad.layout = parse_layout(layout).map_err(bad)?;
        }
        if let Some(bg) = &opts.backgrounds {
            use threers::helpers::Pane;
            for (pane, hex_v) in [
                (Pane::Top, bg.top),
                (Pane::Front, bg.front),
                (Pane::Side, bg.side),
                (Pane::Perspective, bg.perspective),
            ] {
                if let Some(v) = hex_v {
                    quad.backgrounds[pane.index()] =
                        Some(hex(Some(v), 0, &format!("backgrounds.{}", pane.name()))?);
                }
            }
        }
        let png = match opts.format.as_deref() {
            None | Some("png") => true,
            Some("rgba") => false,
            Some(other) => {
                return Err(Error::new(
                    Status::InvalidArg,
                    format!("renderQuadView: format must be 'png' or 'rgba', got '{other}'"),
                ))
            }
        };
        quad.frame_scene(&mut scene.inner);
        let shadows = self.shadows;
        with_shadows(&mut scene.inner, shadows, |s| {
            quad.render(&mut self.inner, s, &camera.inner)
        });
        let rgba = self.inner.read_rgba_resolved();
        if !png {
            return Ok(Buffer::from(rgba));
        }
        let cfg = self.inner.config();
        Ok(Buffer::from(threers::encode_png(
            cfg.width, cfg.height, &rgba,
        )))
    }

    /// Resize the output (three.js `setSize`). Rebuilds the render target, so
    /// the viewport and scissor reset; tone mapping and the shadow switch
    /// carry over.
    #[napi]
    pub fn set_size(&mut self, width: u32, height: u32) -> Result<()> {
        if width == 0 || height == 0 || width > 16384 || height > 16384 {
            return Err(bad(format!(
                "setSize: width and height must be 1..=16384, got {width}x{height}"
            )));
        }
        let (mode, exposure) = self.inner.renderer().tone_mapping();
        self.inner = build_headless(width, height, self.supersample)?;
        self.inner.renderer().set_tone_mapping(mode, exposure);
        Ok(())
    }

    /// `mode`: `'none'` (default), `'linear'`, `'aces'`, `'neutral'` or
    /// `'agx'`; `exposure` defaults to 1.
    #[napi]
    pub fn set_tone_mapping(
        &mut self,
        #[napi(ts_arg_type = "'none' | 'linear' | 'aces' | 'neutral' | 'agx'")] mode: String,
        exposure: Option<f64>,
    ) -> Result<()> {
        let mode = parse_tone_mapping(&mode).map_err(bad)?;
        let exposure = non_neg(exposure, 1.0, "setToneMapping: exposure")?;
        self.inner.renderer().set_tone_mapping(mode, exposure);
        Ok(())
    }

    #[napi]
    pub fn tone_mapping(&mut self) -> ToneMappingInfo {
        let (mode, exposure) = self.inner.renderer().tone_mapping();
        ToneMappingInfo {
            mode: tone_mapping_name(mode).to_string(),
            exposure: exposure as f64,
        }
    }

    /// Master shadow switch (three.js `shadowMap.enabled`), default on.
    /// Shadows still need a light with `castShadow` and meshes opted in with
    /// `setCastShadow` / `setReceiveShadow`.
    #[napi]
    pub fn set_shadows_enabled(&mut self, enabled: bool) {
        self.shadows = enabled;
    }

    #[napi]
    pub fn shadows_enabled(&self) -> bool {
        self.shadows
    }

    /// Undo `setViewport`: draw to the whole output again.
    #[napi]
    pub fn reset_viewport(&mut self) {
        self.inner.renderer().reset_viewport();
    }

    /// three.js `autoClear` (default on). Off, each render draws over what
    /// is already there, so several `setViewport` + render calls compose one
    /// image (call `clear` first).
    #[napi]
    pub fn set_auto_clear(&mut self, enabled: bool) {
        self.inner.renderer().set_auto_clear(enabled);
    }

    #[napi]
    pub fn auto_clear(&mut self) -> bool {
        self.inner.renderer().auto_clear()
    }

    /// Fill the output with `color` (`0xRRGGBB`, default black) at `alpha`
    /// (default 1) — only the scissor rectangle while the scissor test is on.
    #[napi]
    pub fn clear(&mut self, color: Option<u32>, alpha: Option<f64>) -> Result<()> {
        let c = hex(color, 0x000000, "clear: color")?;
        let a = unit(alpha, 1.0, "clear: alpha")?;
        self.inner.clear(c, a);
        Ok(())
    }
}

#[cfg(feature = "headless")]
#[napi(object)]
pub struct ToneMappingInfo {
    #[napi(ts_type = "'none' | 'linear' | 'aces' | 'neutral' | 'agx'")]
    pub mode: String,
    pub exposure: f64,
}

#[cfg(feature = "headless")]
fn build_headless(width: u32, height: u32, supersample: u32) -> Result<threers::HeadlessRenderer> {
    threers::HeadlessRenderer::builder()
        .size(width, height)
        .supersample(supersample)
        .build()
        .map_err(Error::from_reason)
}

#[cfg(feature = "headless")]
fn parse_tone_mapping(name: &str) -> std::result::Result<threers::ToneMapping, String> {
    use threers::ToneMapping as T;
    Ok(match name.to_ascii_lowercase().as_str() {
        "none" => T::None,
        "linear" => T::Linear,
        "aces" | "acesfilmic" => T::AcesFilmic,
        "neutral" => T::Neutral,
        "agx" => T::AgX,
        other => {
            return Err(format!(
                "unknown tone mapping '{other}' (none, linear, aces, neutral, agx)"
            ))
        }
    })
}

#[cfg(feature = "headless")]
fn tone_mapping_name(mode: threers::ToneMapping) -> &'static str {
    use threers::ToneMapping as T;
    match mode {
        T::None => "none",
        T::Linear => "linear",
        T::AcesFilmic => "aces",
        T::Neutral => "neutral",
        T::AgX => "agx",
        #[allow(unreachable_patterns)]
        _ => "other",
    }
}

/// Run `f` with every light's `cast_shadow` cleared when `enabled` is false,
/// restoring the flags afterwards.
#[cfg(feature = "headless")]
fn with_shadows<R>(scene: &mut Scene, enabled: bool, f: impl FnOnce(&mut Scene) -> R) -> R {
    if enabled {
        return f(scene);
    }
    let mut casters = Vec::new();
    scene.arena.traverse(scene.root, &mut |id, o| {
        if let ObjectKind::Light(l) = &o.kind {
            if light_casts(l) {
                casters.push(id);
            }
        }
    });
    let set = |scene: &mut Scene, on: bool| {
        for id in &casters {
            if let Some(ObjectKind::Light(l)) = scene.arena.get_mut(*id).map(|o| &mut o.kind) {
                match l {
                    Light::Directional(d) => d.cast_shadow = on,
                    Light::Point(p) => p.cast_shadow = on,
                    Light::Spot(s) => s.cast_shadow = on,
                    _ => {}
                }
            }
        }
    };
    set(scene, false);
    let out = f(scene);
    set(scene, true);
    out
}

#[cfg(feature = "headless")]
fn light_casts(l: &Light) -> bool {
    match l {
        Light::Directional(d) => d.cast_shadow,
        Light::Point(p) => p.cast_shadow,
        Light::Spot(s) => s.cast_shadow,
        _ => false,
    }
}

/// Options for `HeadlessRenderer.renderQuadView`.
#[cfg(feature = "headless")]
#[napi(object)]
#[derive(Default)]
pub struct QuadViewOptions {
    /// Show one pane full-size.
    #[napi(ts_type = "'top' | 'front' | 'side' | 'perspective'")]
    pub maximized: Option<String>,
    /// `'png'` (default) or `'rgba'`.
    #[napi(ts_type = "'png' | 'rgba'")]
    pub format: Option<String>,
    /// Space between panes, pixels (default 2).
    pub gap: Option<f64>,
    /// Colour of the gaps as `0xRRGGBB` (default `0x303030`).
    pub gap_color: Option<u32>,
    /// Space between the panes and the image edge, pixels (default 0).
    pub border: Option<f64>,
    /// Pane order in the grid: top-left, top-right, bottom-left,
    /// bottom-right (default `['top', 'perspective', 'front', 'side']`,
    /// third-angle).
    #[napi(ts_type = "Array<'top' | 'front' | 'side' | 'perspective'>")]
    pub layout: Option<Vec<String>>,
    /// Per-pane background colours (`0xRRGGBB`) overriding the scene's.
    pub backgrounds: Option<QuadViewBackgrounds>,
    /// Padding round the framed scene in the orthographic panes, as a
    /// fraction of its size.
    pub margin: Option<f64>,
}

/// Per-pane background colours for `renderQuadView`.
#[cfg(feature = "headless")]
#[napi(object)]
pub struct QuadViewBackgrounds {
    pub top: Option<u32>,
    pub front: Option<u32>,
    pub side: Option<u32>,
    pub perspective: Option<u32>,
}

/// Four distinct pane names → a `QuadView::layout`.
#[cfg(feature = "headless")]
fn parse_layout(names: &[String]) -> std::result::Result<[threers::helpers::Pane; 4], String> {
    use threers::helpers::Pane;
    if names.len() != 4 {
        return Err(format!("layout needs 4 pane names, got {}", names.len()));
    }
    let mut out = [Pane::Top; 4];
    for (i, n) in names.iter().enumerate() {
        let p = Pane::from_name(n)
            .ok_or_else(|| format!("unknown pane '{n}' (top, front, side, perspective)"))?;
        if out[..i].contains(&p) {
            return Err(format!("layout lists '{}' twice", p.name()));
        }
        out[i] = p;
    }
    Ok(out)
}

#[cfg(feature = "headless")]
fn parse_pane(name: &str) -> Result<threers::helpers::Pane> {
    threers::helpers::Pane::from_name(name).ok_or_else(|| {
        Error::new(
            Status::InvalidArg,
            format!("unknown pane '{name}' (top, front, side, perspective)"),
        )
    })
}

// ---- animation -------------------------------------------------------------

#[cfg(feature = "animation")]
#[napi(js_name = "Tween")]
pub struct JsTween {
    inner: threers_animation::tween::Tween<f32>,
}

#[cfg(feature = "animation")]
#[napi]
impl JsTween {
    #[napi(constructor)]
    pub fn new(
        start: Option<f64>,
        end: Option<f64>,
        duration: Option<f64>,
        easing: Option<String>,
    ) -> Result<Self> {
        use threers_animation::tween::Tween;
        let e = parse_easing(easing.as_deref().unwrap_or("cubicOut"))?;
        Ok(Self {
            inner: Tween::new(
                start.unwrap_or(0.0) as f32,
                end.unwrap_or(1.0) as f32,
                duration.unwrap_or(1.0) as f32,
            )
            .easing(e),
        })
    }

    #[napi]
    pub fn update(&mut self, dt: f64) -> f64 {
        self.inner.update(dt as f32) as f64
    }

    #[napi]
    pub fn value(&self) -> f64 {
        self.inner.value() as f64
    }

    #[napi]
    pub fn is_finished(&self) -> bool {
        self.inner.is_finished()
    }
}

#[cfg(feature = "animation")]
fn parse_easing(name: &str) -> Result<threers_animation::easing::Easing> {
    use threers_animation::easing::Easing;
    Ok(match name {
        "linear" => Easing::Linear,
        "quadIn" => Easing::QuadIn,
        "quadOut" => Easing::QuadOut,
        "quadInOut" => Easing::QuadInOut,
        "cubicIn" => Easing::CubicIn,
        "cubicOut" => Easing::CubicOut,
        "cubicInOut" => Easing::CubicInOut,
        other => {
            return Err(Error::from_reason(format!(
                "unknown easing '{other}' (try linear, cubicOut, cubicInOut, …)"
            )))
        }
    })
}

// ---- physics ---------------------------------------------------------------

#[cfg(feature = "physics")]
#[napi(js_name = "PhysicsWorld")]
pub struct JsPhysicsWorld {
    inner: threers_physics::prelude::World,
    bodies: Vec<threers_physics::prelude::BodyId>,
}

#[cfg(feature = "physics")]
impl Default for JsPhysicsWorld {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "physics")]
#[napi]
impl JsPhysicsWorld {
    #[napi(constructor)]
    pub fn new() -> Self {
        use threers_physics::prelude::*;
        let mut world = World::new();
        world.add_body(RigidBody::fixed().shape(Shape::ground()));
        Self {
            inner: world,
            bodies: Vec::new(),
        }
    }

    /// Add a dynamic ball. Returns a stable handle index.
    #[napi]
    pub fn add_ball(
        &mut self,
        radius: Option<f64>,
        x: Option<f64>,
        y: Option<f64>,
        z: Option<f64>,
        restitution: Option<f64>,
    ) -> u32 {
        use threers_physics::prelude::*;
        let id = self.inner.add_body(
            RigidBody::dynamic()
                .shape(Shape::ball(radius.unwrap_or(0.5) as f32))
                .translation(Vector3::new(
                    x.unwrap_or(0.0) as f32,
                    y.unwrap_or(5.0) as f32,
                    z.unwrap_or(0.0) as f32,
                ))
                .restitution(restitution.unwrap_or(0.4) as f32),
        );
        self.bodies.push(id);
        (self.bodies.len() - 1) as u32
    }

    #[napi]
    pub fn step(&mut self, dt: f64) {
        self.inner.step(dt as f32);
    }

    #[napi]
    pub fn body_translation(&self, handle: u32) -> Result<JsVector3> {
        let id = *self
            .bodies
            .get(handle as usize)
            .ok_or_else(|| Error::from_reason("bad body handle"))?;
        let body = self
            .inner
            .body(id)
            .ok_or_else(|| Error::from_reason("body removed"))?;
        Ok(JsVector3::from_inner(body.translation()))
    }

    #[napi]
    pub fn body_count(&self) -> u32 {
        self.bodies.len() as u32
    }
}
