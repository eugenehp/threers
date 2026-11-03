//! Python bindings for threers (`pip install threers` via maturin).
//!
//! ```bash
//! cd crates/threers-py
//! maturin develop --features headless,physics,animation
//! python -c "import threers; print(threers.__version__)"
//! ```

use pyo3::exceptions::PyTypeError;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList};

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

#[pyfunction]
fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

// ---- math ------------------------------------------------------------------

#[pyclass(name = "Vector3")]
#[derive(Clone, Copy)]
struct PyVector3 {
    inner: Vector3,
}

#[pymethods]
impl PyVector3 {
    #[new]
    #[pyo3(signature = (x=0.0, y=0.0, z=0.0))]
    fn new(x: f32, y: f32, z: f32) -> Self {
        Self {
            inner: Vector3::new(x, y, z),
        }
    }

    #[getter]
    fn x(&self) -> f32 {
        self.inner.x
    }
    #[getter]
    fn y(&self) -> f32 {
        self.inner.y
    }
    #[getter]
    fn z(&self) -> f32 {
        self.inner.z
    }

    #[setter]
    fn set_x(&mut self, v: f32) {
        self.inner.x = v;
    }
    #[setter]
    fn set_y(&mut self, v: f32) {
        self.inner.y = v;
    }
    #[setter]
    fn set_z(&mut self, v: f32) {
        self.inner.z = v;
    }

    fn length(&self) -> f32 {
        self.inner.length()
    }

    fn normalized(&self) -> Self {
        Self {
            inner: self.inner.normalize(),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "Vector3({:.4}, {:.4}, {:.4})",
            self.inner.x, self.inner.y, self.inner.z
        )
    }
}

#[pyclass(name = "Color")]
#[derive(Clone, Copy)]
struct PyColor {
    inner: Color,
}

#[pymethods]
impl PyColor {
    #[new]
    #[pyo3(signature = (r=1.0, g=1.0, b=1.0))]
    fn new(r: f32, g: f32, b: f32) -> Self {
        Self {
            inner: Color::new(r, g, b),
        }
    }

    #[staticmethod]
    fn from_hex(hex: u32) -> Self {
        Self {
            inner: Color::from_hex(hex),
        }
    }

    #[getter]
    fn r(&self) -> f32 {
        self.inner.r
    }
    #[getter]
    fn g(&self) -> f32 {
        self.inner.g
    }
    #[getter]
    fn b(&self) -> f32 {
        self.inner.b
    }

    fn __repr__(&self) -> String {
        format!(
            "Color({:.3}, {:.3}, {:.3})",
            self.inner.r, self.inner.g, self.inner.b
        )
    }
}

// ---- scene graph -----------------------------------------------------------

#[pyclass(name = "PerspectiveCamera", unsendable)]
struct PyPerspectiveCamera {
    inner: PerspectiveCamera,
}

#[pymethods]
impl PyPerspectiveCamera {
    #[new]
    #[pyo3(signature = (fov=50.0, aspect=1.0, near=0.1, far=2000.0))]
    fn new(fov: f32, aspect: f32, near: f32, far: f32) -> Self {
        Self {
            inner: PerspectiveCamera::new(fov, aspect, near, far),
        }
    }

    fn set_position(&mut self, x: f32, y: f32, z: f32) {
        self.inner.position = Vector3::new(x, y, z);
    }

    fn look_at(&mut self, x: f32, y: f32, z: f32) {
        self.inner.look_at(Vector3::new(x, y, z));
    }

    fn set_aspect(&mut self, aspect: f32) {
        self.inner.aspect = aspect;
    }
}

/// three.js ``OrthographicCamera(left, right, top, bottom, near, far)``.
/// Renders through the same ``HeadlessRenderer`` methods as
/// ``PerspectiveCamera``.
#[pyclass(name = "OrthographicCamera", unsendable)]
struct PyOrthographicCamera {
    inner: OrthographicCamera,
}

#[pymethods]
impl PyOrthographicCamera {
    #[new]
    #[pyo3(signature = (left=-1.0, right=1.0, top=1.0, bottom=-1.0, near=0.1, far=2000.0))]
    fn new(left: f64, right: f64, top: f64, bottom: f64, near: f64, far: f64) -> PyResult<Self> {
        let mut c = Self {
            inner: OrthographicCamera::new(-1.0, 1.0, 1.0, -1.0, 0.1, 2000.0),
        };
        c.set_frustum(left, right, top, bottom, Some(near), Some(far))?;
        Ok(c)
    }

    /// Replace the view volume (three.js: assign ``left`` … ``far``, then
    /// ``updateProjectionMatrix()``). ``near`` / ``far`` keep their values
    /// when omitted.
    #[pyo3(signature = (left, right, top, bottom, near=None, far=None))]
    fn set_frustum(
        &mut self,
        left: f64,
        right: f64,
        top: f64,
        bottom: f64,
        near: Option<f64>,
        far: Option<f64>,
    ) -> PyResult<()> {
        let near = near.unwrap_or(self.inner.near as f64);
        let far = far.unwrap_or(self.inner.far as f64);
        if ![left, right, top, bottom, near, far]
            .iter()
            .all(|v| v.is_finite())
        {
            return Err(PyValueError::new_err(
                "OrthographicCamera: values must be finite",
            ));
        }
        if left == right || top == bottom || near >= far {
            return Err(PyValueError::new_err(
                "OrthographicCamera: needs left != right, top != bottom and near < far",
            ));
        }
        let c = &mut self.inner;
        (c.left, c.right, c.top, c.bottom) = (left as f32, right as f32, top as f32, bottom as f32);
        (c.near, c.far) = (near as f32, far as f32);
        Ok(())
    }

    fn set_position(&mut self, x: f32, y: f32, z: f32) -> PyResult<()> {
        self.inner.position = finite_vec(x, y, z, "set_position")?;
        Ok(())
    }

    fn look_at(&mut self, x: f32, y: f32, z: f32) -> PyResult<()> {
        let t = finite_vec(x, y, z, "look_at")?;
        self.inner.look_at(t);
        Ok(())
    }
}

/// Borrow `camera` as either camera class for the duration of `f`.
fn with_camera<R>(camera: &Bound<'_, PyAny>, f: impl FnOnce(&dyn Camera) -> R) -> PyResult<R> {
    if let Ok(c) = camera.downcast::<PyPerspectiveCamera>() {
        return Ok(f(&c.borrow().inner));
    }
    if let Ok(c) = camera.downcast::<PyOrthographicCamera>() {
        return Ok(f(&c.borrow().inner));
    }
    Err(PyTypeError::new_err(
        "camera must be a PerspectiveCamera or OrthographicCamera",
    ))
}

#[pyclass(name = "Scene", unsendable)]
struct PyScene {
    inner: Scene,
    handles: Vec<ObjectId>,
    /// Animates the scene's fur and hair (`step_fur`).
    fur: threers::fur::FurAnimator,
}

#[pymethods]
impl PyScene {
    #[new]
    fn new() -> Self {
        Self {
            inner: Scene::new(),
            handles: Vec::new(),
            fur: threers::fur::FurAnimator::new(),
        }
    }

    /// Grow shell fur on ``geometry`` (a ``Geometry``) as ``fur`` (a
    /// ``FurMaterial``) says. Returns a stable handle index. It casts and
    /// receives shadows; ``step_fur`` animates it.
    #[pyo3(signature = (geometry, fur, x=0.0, y=0.0, z=0.0))]
    fn add_fur(
        &mut self,
        geometry: PyRef<'_, PyGeometry>,
        fur: PyRef<'_, PyFurMaterial>,
        x: f32,
        y: f32,
        z: f32,
    ) -> PyResult<usize> {
        let mut obj = threers::fur::fur_object(geometry.inner.clone(), &fur.inner);
        obj.position = finite_vec(x, y, z, "add_fur")?;
        let handle = self.push(obj);
        self.fur.track(self.handles[handle], &fur.inner);
        Ok(handle)
    }

    /// Grow strand hair on ``geometry``: ``count`` strands of ``segments``
    /// pieces, ribbons ``width`` wide at the root, shaded and moved as ``fur``
    /// (a ``FurMaterial``) says; ``shadow_strands`` of them (a share, 0 for
    /// none) cast its shadow. Returns a stable handle index.
    #[pyo3(signature = (geometry, fur, count=4000, segments=8, width=0.006, shadow_strands=0.25, x=0.0, y=0.0, z=0.0))]
    #[allow(clippy::too_many_arguments)]
    fn add_hair(
        &mut self,
        geometry: PyRef<'_, PyGeometry>,
        fur: PyRef<'_, PyFurMaterial>,
        count: usize,
        segments: usize,
        width: f32,
        shadow_strands: f32,
        x: f32,
        y: f32,
        z: f32,
    ) -> PyResult<usize> {
        if !(width.is_finite() && width > 0.0) {
            return Err(PyValueError::new_err("add_hair: width must be positive"));
        }
        let options = threers::fur::strands::StrandOptions {
            count,
            segments: segments.max(1),
            width,
            shadow_strands,
        };
        let mut obj = threers::fur::strands::hair_object(&geometry.inner, &fur.inner, &options);
        obj.position = finite_vec(x, y, z, "add_hair")?;
        let handle = self.push(obj);
        self.fur.track(self.handles[handle], &fur.inner);
        Ok(handle)
    }

    /// A hand in a fur or hair coat (``handle``): within ``radius`` of
    /// ``at`` (world), strands are pushed by ``push`` (a pull per unit of their
    /// length, along the hand's motion) and parted away from it by ``part``.
    /// Call it as the hand moves; ``step_fur`` fades it, the strands springing
    /// back.
    #[pyo3(signature = (handle, at, radius=0.3, push=(0.0, 0.0, 0.0), part=0.5))]
    fn stroke_fur(
        &mut self,
        handle: usize,
        at: (f32, f32, f32),
        radius: f32,
        push: (f32, f32, f32),
        part: f32,
    ) -> PyResult<()> {
        let id = self.live(handle)?;
        self.fur.stroke(
            id,
            Vector3::new(at.0, at.1, at.2),
            radius,
            Vector3::new(push.0, push.1, push.2),
            part,
        );
        Ok(())
    }

    /// Advance the scene's fur and hair — wind, flutter, inertia, their
    /// shadows — to ``time`` seconds, ``dt`` after the last step. Once a
    /// frame, after moving things and before rendering.
    #[pyo3(signature = (time, dt=1.0 / 60.0))]
    fn step_fur(&mut self, time: f32, dt: f32) {
        self.fur.update(&mut self.inner, time, dt);
    }

    /// Add an axis-aligned box mesh. Returns a stable handle index.
    #[pyo3(signature = (width=1.0, height=1.0, depth=1.0, color=0xff6633, x=0.0, y=0.0, z=0.0))]
    #[allow(clippy::too_many_arguments)]
    fn add_box(
        &mut self,
        width: f32,
        height: f32,
        depth: f32,
        color: u32,
        x: f32,
        y: f32,
        z: f32,
    ) -> usize {
        let geom = BoxGeometry::new(width, height, depth);
        let mat = Material::Standard(StandardMaterial {
            color: Color::from_hex(color),
            ..StandardMaterial::default()
        });
        let mut obj = Object3D::mesh(Mesh::new(geom, mat));
        obj.position = Vector3::new(x, y, z);
        let id = self.inner.arena.insert(obj);
        self.inner.arena.add_child(self.inner.root, id);
        self.handles.push(id);
        self.handles.len() - 1
    }

    #[pyo3(signature = (radius=0.5, color=0x6ea8fe, x=0.0, y=0.0, z=0.0))]
    fn add_sphere(&mut self, radius: f32, color: u32, x: f32, y: f32, z: f32) -> usize {
        let geom = SphereGeometry::new(radius, 24, 16);
        let mat = Material::Basic(BasicMaterial::new(Color::from_hex(color)));
        let mut obj = Object3D::mesh(Mesh::new(geom, mat));
        obj.position = Vector3::new(x, y, z);
        let id = self.inner.arena.insert(obj);
        self.inner.arena.add_child(self.inner.root, id);
        self.handles.push(id);
        self.handles.len() - 1
    }

    /// Add a quad from four corners in winding order (``p0..p3`` at UV
    /// ``(0,0)``, ``(1,0)``, ``(1,1)``, ``(0,1)``), meshed as a bilinear patch
    /// and shaded on both sides. ``corners`` is 12 numbers or 4 points (each a
    /// ``Vector3`` or an ``(x, y, z)`` sequence). Returns a stable handle index.
    #[pyo3(signature = (corners, color=0xff6633, width_segments=1, height_segments=1))]
    fn add_quad(
        &mut self,
        corners: &Bound<'_, PyAny>,
        color: u32,
        width_segments: u32,
        height_segments: u32,
    ) -> PyResult<usize> {
        let corners = parse_corners(corners)?;
        for (name, v) in [
            ("width_segments", width_segments),
            ("height_segments", height_segments),
        ] {
            if !(1..=MAX_QUAD_SEGMENTS).contains(&v) {
                return Err(PyValueError::new_err(format!(
                    "{name} must be 1..={MAX_QUAD_SEGMENTS}, got {v}"
                )));
            }
        }
        let geom =
            QuadGeometry::with_segments(corners, width_segments as usize, height_segments as usize);
        let mat = Material::Standard(StandardMaterial {
            color: Color::from_hex(color),
            side: DOUBLE_SIDE,
            ..StandardMaterial::default()
        });
        let id = self
            .inner
            .arena
            .insert(Object3D::mesh(Mesh::new(geom, mat)));
        self.inner.arena.add_child(self.inner.root, id);
        self.handles.push(id);
        Ok(self.handles.len() - 1)
    }

    fn set_background(&mut self, hex: u32) {
        self.inner.background = Color::from_hex(hex);
    }

    fn set_position(&mut self, handle: usize, x: f32, y: f32, z: f32) -> PyResult<()> {
        let id = *self
            .handles
            .get(handle)
            .ok_or_else(|| PyValueError::new_err("bad object handle"))?;
        let obj = self
            .inner
            .arena
            .get_mut(id)
            .ok_or_else(|| PyValueError::new_err("object removed"))?;
        obj.position = Vector3::new(x, y, z);
        Ok(())
    }

    /// Number of live handles (meshes, groups, lights and loaded models;
    /// removed objects are not counted).
    fn object_count(&self) -> usize {
        self.handles
            .iter()
            .filter(|id| self.inner.arena.get(**id).is_some())
            .count()
    }

    // ---- generic meshes and groups ----

    /// Add a mesh built from a ``Geometry`` and a ``Material``. Returns a
    /// stable handle index.
    #[pyo3(signature = (geometry, material, x=0.0, y=0.0, z=0.0))]
    fn add_mesh(
        &mut self,
        geometry: PyRef<'_, PyGeometry>,
        material: PyRef<'_, PyMaterial>,
        x: f32,
        y: f32,
        z: f32,
    ) -> PyResult<usize> {
        if matches!(material.inner, Material::Line(_)) {
            return Err(PyValueError::new_err(
                "add_mesh: a line material needs add_line_segments",
            ));
        }
        let mut obj = Object3D::mesh(Mesh::new(geometry.inner.clone(), material.inner.clone()));
        obj.position = finite_vec(x, y, z, "add_mesh")?;
        Ok(self.push(obj))
    }

    /// three.js ``LineSegments``: draw ``geometry``'s vertices as pairs
    /// (``Geometry.edges`` / ``Geometry.wireframe`` output, or any geometry
    /// with an even vertex count) with a ``Material.line``.
    #[pyo3(signature = (geometry, material, x=0.0, y=0.0, z=0.0))]
    fn add_line_segments(
        &mut self,
        geometry: PyRef<'_, PyGeometry>,
        material: PyRef<'_, PyMaterial>,
        x: f32,
        y: f32,
        z: f32,
    ) -> PyResult<usize> {
        if !matches!(material.inner, Material::Line(_)) {
            return Err(PyValueError::new_err(
                "add_line_segments: needs a Material.line",
            ));
        }
        let g = &geometry.inner;
        let n = g.get_attribute("position").map_or(0, |a| a.count());
        if g.index.is_none() && !n.is_multiple_of(2) {
            return Err(PyValueError::new_err(format!(
                "add_line_segments: needs an even vertex count, got {n}"
            )));
        }
        let mut obj = Object3D::line_segments(threers::core::LineSegments::new(
            g.clone(),
            material.inner.clone(),
        ));
        obj.position = finite_vec(x, y, z, "add_line_segments")?;
        Ok(self.push(obj))
    }

    /// A copy of the geometry of a mesh or line-segments handle (for a
    /// loaded model, get the mesh handles with ``meshes`` first).
    fn get_geometry(&self, handle: usize) -> PyResult<PyGeometry> {
        let id = self.live(handle)?;
        match &self.inner.arena.get(id).unwrap().kind {
            ObjectKind::Mesh(m) => Ok(PyGeometry::of((*m.geometry).clone())),
            ObjectKind::LineSegments(l) => Ok(PyGeometry::of((*l.geometry).clone())),
            _ => Err(PyValueError::new_err(format!(
                "get_geometry: object {handle} has no geometry"
            ))),
        }
    }

    /// Handles for every mesh at or under ``handle`` (e.g. the parts of a
    /// loaded model), in scene-graph order. Repeated calls return the same
    /// handles.
    fn meshes(&mut self, handle: usize) -> PyResult<Vec<usize>> {
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
    #[pyo3(signature = (x=0.0, y=0.0, z=0.0))]
    fn add_group(&mut self, x: f32, y: f32, z: f32) -> PyResult<usize> {
        let mut obj = Object3D::group();
        obj.position = finite_vec(x, y, z, "add_group")?;
        Ok(self.push(obj))
    }

    /// Re-parent ``child`` under ``parent`` (``None`` for the scene root).
    /// The child's transform becomes relative to its new parent.
    #[pyo3(signature = (child, parent=None))]
    fn set_parent(&mut self, child: usize, parent: Option<usize>) -> PyResult<()> {
        let c = self.live(child)?;
        let p = match parent {
            None => self.inner.root,
            Some(p) => self.live(p)?,
        };
        if c == p || self.is_ancestor(c, p) {
            return Err(PyValueError::new_err("set_parent: would create a cycle"));
        }
        self.inner.arena.add_child(p, c);
        Ok(())
    }

    // ---- lights ----

    /// three.js ``AmbientLight`` (default white, intensity 1).
    #[pyo3(signature = (color=0xffffff, intensity=1.0))]
    fn add_ambient_light(&mut self, color: u32, intensity: f64) -> PyResult<usize> {
        let light = AmbientLight::new(
            hex(color, "color")?,
            intensity_arg(intensity, "add_ambient_light")?,
        );
        Ok(self.push(Object3D::light(light)))
    }

    /// three.js ``DirectionalLight`` shining from ``position`` (a ``Vector3``
    /// or ``(x, y, z)``, default ``(0, 1, 0)``) towards the origin.
    /// ``cast_shadow`` makes it the scene's directional shadow caster (meshes
    /// opt in with ``set_cast_shadow`` / ``set_receive_shadow``).
    #[pyo3(signature = (color=0xffffff, intensity=1.0, position=None, cast_shadow=false))]
    fn add_directional_light(
        &mut self,
        color: u32,
        intensity: f64,
        position: Option<&Bound<'_, PyAny>>,
        cast_shadow: bool,
    ) -> PyResult<usize> {
        let pos = opt_vec(
            position,
            Vector3::new(0.0, 1.0, 0.0),
            "add_directional_light: position",
        )?;
        if pos.length_sq() < 1e-12 {
            return Err(PyValueError::new_err(
                "add_directional_light: position must not be the origin (the light aims at it)",
            ));
        }
        let mut light = DirectionalLight::new(
            hex(color, "color")?,
            intensity_arg(intensity, "add_directional_light")?,
        );
        light.cast_shadow = cast_shadow;
        let mut obj = Object3D::light(light);
        obj.position = pos;
        Ok(self.push(obj))
    }

    /// three.js ``PointLight`` (physically based: intensity in candela,
    /// default 1; ``distance`` 0 = infinite range; ``decay`` default 2).
    #[pyo3(signature = (color=0xffffff, intensity=1.0, position=None, distance=0.0, decay=2.0, cast_shadow=false))]
    fn add_point_light(
        &mut self,
        color: u32,
        intensity: f64,
        position: Option<&Bound<'_, PyAny>>,
        distance: f64,
        decay: f64,
        cast_shadow: bool,
    ) -> PyResult<usize> {
        let mut light = PointLight::new(
            hex(color, "color")?,
            intensity_arg(intensity, "add_point_light")?,
        )
        .with_distance(non_neg(distance, "add_point_light: distance")?)
        .with_decay(non_neg(decay, "add_point_light: decay")?);
        light.cast_shadow = cast_shadow;
        let mut obj = Object3D::light(light);
        obj.position = opt_vec(position, Vector3::ZERO, "add_point_light: position")?;
        Ok(self.push(obj))
    }

    /// three.js ``SpotLight`` at ``position`` (default ``(0, 1, 0)``) aimed at
    /// ``target`` (default origin). ``angle`` is the cone half-angle in
    /// radians (default π/3, max π/2); ``penumbra`` 0..1.
    #[pyo3(signature = (color=0xffffff, intensity=1.0, position=None, target=None, angle=std::f64::consts::FRAC_PI_3, penumbra=0.0, distance=0.0, decay=2.0, cast_shadow=false))]
    #[allow(clippy::too_many_arguments)]
    fn add_spot_light(
        &mut self,
        color: u32,
        intensity: f64,
        position: Option<&Bound<'_, PyAny>>,
        target: Option<&Bound<'_, PyAny>>,
        angle: f64,
        penumbra: f64,
        distance: f64,
        decay: f64,
        cast_shadow: bool,
    ) -> PyResult<usize> {
        let pos = opt_vec(
            position,
            Vector3::new(0.0, 1.0, 0.0),
            "add_spot_light: position",
        )?;
        let tgt = opt_vec(target, Vector3::ZERO, "add_spot_light: target")?;
        let dir = tgt - pos;
        if dir.length_sq() < 1e-12 {
            return Err(PyValueError::new_err(
                "add_spot_light: position and target must differ",
            ));
        }
        if !(angle.is_finite() && angle > 0.0 && angle <= std::f64::consts::FRAC_PI_2) {
            return Err(PyValueError::new_err(format!(
                "add_spot_light: angle must be in (0, π/2] radians, got {angle}"
            )));
        }
        let mut light = SpotLight::new(
            hex(color, "color")?,
            intensity_arg(intensity, "add_spot_light")?,
        );
        light.direction = dir.normalize();
        light.angle = angle as f32;
        light.penumbra = unit(penumbra, "add_spot_light: penumbra")?;
        light.distance = non_neg(distance, "add_spot_light: distance")?;
        light.decay = non_neg(decay, "add_spot_light: decay")?;
        light.cast_shadow = cast_shadow;
        let mut obj = Object3D::light(light);
        obj.position = pos;
        Ok(self.push(obj))
    }

    /// three.js ``HemisphereLight``: ``sky_color`` from above,
    /// ``ground_color`` from below (both default white), intensity default 1.
    #[pyo3(signature = (sky_color=0xffffff, ground_color=0xffffff, intensity=1.0))]
    fn add_hemisphere_light(
        &mut self,
        sky_color: u32,
        ground_color: u32,
        intensity: f64,
    ) -> PyResult<usize> {
        let light = HemisphereLight::new(
            hex(sky_color, "sky_color")?,
            hex(ground_color, "ground_color")?,
            intensity_arg(intensity, "add_hemisphere_light")?,
        );
        Ok(self.push(Object3D::light(light)))
    }

    /// Number of lights in the scene graph (including any a loaded model
    /// brought in).
    fn light_count(&self) -> usize {
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
    fn set_rotation(&mut self, handle: usize, x: f32, y: f32, z: f32) -> PyResult<()> {
        let r = finite_vec(x, y, z, "set_rotation")?;
        self.obj_mut(handle)?.quaternion = Quaternion::from_euler_xyz(r.x, r.y, r.z);
        Ok(())
    }

    fn set_scale(&mut self, handle: usize, x: f32, y: f32, z: f32) -> PyResult<()> {
        let s = finite_vec(x, y, z, "set_scale")?;
        self.obj_mut(handle)?.scale = s;
        Ok(())
    }

    fn set_visible(&mut self, handle: usize, visible: bool) -> PyResult<()> {
        self.obj_mut(handle)?.visible = visible;
        Ok(())
    }

    /// Include this object (not its children) in shadow maps.
    fn set_cast_shadow(&mut self, handle: usize, cast: bool) -> PyResult<()> {
        self.obj_mut(handle)?.cast_shadow = cast;
        Ok(())
    }

    fn set_receive_shadow(&mut self, handle: usize, receive: bool) -> PyResult<()> {
        self.obj_mut(handle)?.receive_shadow = receive;
        Ok(())
    }

    /// Replace a mesh's material. For a loaded model (a group), every mesh
    /// under it gets the material.
    fn set_material(&mut self, handle: usize, material: PyRef<'_, PyMaterial>) -> PyResult<()> {
        let id = self.live(handle)?;
        let is_line = matches!(material.inner, Material::Line(_));
        if let ObjectKind::LineSegments(l) = &mut self.inner.arena.get_mut(id).unwrap().kind {
            if !is_line {
                return Err(PyValueError::new_err(
                    "set_material: line segments need a Material.line",
                ));
            }
            l.material = std::sync::Arc::new(material.inner.clone());
            return Ok(());
        }
        if is_line {
            return Err(PyValueError::new_err(
                "set_material: a line material only applies to line segments",
            ));
        }
        if override_materials(&mut self.inner, id, &material.inner) == 0 {
            return Err(PyValueError::new_err("set_material: handle has no mesh"));
        }
        Ok(())
    }

    fn get_position(&self, handle: usize) -> PyResult<PyVector3> {
        let id = self.live(handle)?;
        Ok(PyVector3 {
            inner: self.inner.arena.get(id).unwrap().position,
        })
    }

    /// Remove an object and everything parented under it. Its handle (and
    /// any handles of its descendants) become invalid; other handles keep
    /// their indices.
    fn remove(&mut self, handle: usize) -> PyResult<()> {
        let id = self.live(handle)?;
        remove_subtree(&mut self.inner, id);
        Ok(())
    }

    // ---- picking ----

    /// three.js ``Raycaster.intersectObjects(scene.children, True)``: every
    /// visible object hit by the ray from ``origin`` along ``direction``
    /// (``Vector3`` or ``(x, y, z)``) — meshes, and line segments within
    /// ``line_threshold`` — nearest first, as dicts with three.js's fields in
    /// snake_case: ``handle``, ``distance``, ``point``, ``face_index``,
    /// ``face`` (``a``, ``b``, ``c``, ``normal``, ``material_index``),
    /// ``normal``, ``uv``, ``uv1``, ``instance_id``, ``index``,
    /// ``distance_to_ray``, plus ``polygon_index`` (the quad hit) and
    /// ``barycoord``. A field that does not apply to the hit is ``None``.
    /// ``objects`` (handles) limits the test to those objects (and, with
    /// ``recursive``, their descendants); ``first_hit_only`` returns only the
    /// nearest hit. Meshes inside loaded models get handles on the fly.
    #[pyo3(signature = (
        origin, direction, near=0.0, far=f64::INFINITY, *, first_hit_only=false,
        line_threshold=1.0, points_threshold=1.0, recursive=true, objects=None
    ))]
    #[allow(clippy::too_many_arguments)]
    fn raycast<'py>(
        &mut self,
        py: Python<'py>,
        origin: &Bound<'py, PyAny>,
        direction: &Bound<'py, PyAny>,
        near: f64,
        far: f64,
        first_hit_only: bool,
        line_threshold: f64,
        points_threshold: f64,
        recursive: bool,
        objects: Option<Vec<usize>>,
    ) -> PyResult<Bound<'py, PyList>> {
        let o = opt_vec(Some(origin), Vector3::ZERO, "raycast: origin")?;
        let d = opt_vec(Some(direction), Vector3::ZERO, "raycast: direction")?;
        if d.length_sq() < 1e-12 {
            return Err(PyValueError::new_err("raycast: direction must not be zero"));
        }
        let mut rc = threers::core::Raycaster::default();
        rc.set(o, d.normalize());
        let opts = CastOptions {
            near,
            far,
            first_hit_only,
            line_threshold,
            points_threshold,
            recursive,
            objects,
        };
        self.cast(py, rc, opts, "raycast")
    }

    /// Pick through ``camera`` at normalised device coordinates ``x``, ``y``
    /// (-1..1, +y up) — three.js ``setFromCamera``, perspective or
    /// orthographic — then as ``raycast``.
    #[pyo3(signature = (
        camera, x, y, *, near=0.0, far=f64::INFINITY, first_hit_only=false,
        line_threshold=1.0, points_threshold=1.0, recursive=true, objects=None
    ))]
    #[allow(clippy::too_many_arguments)]
    fn raycast_from_camera<'py>(
        &mut self,
        py: Python<'py>,
        camera: &Bound<'py, PyAny>,
        x: f32,
        y: f32,
        near: f64,
        far: f64,
        first_hit_only: bool,
        line_threshold: f64,
        points_threshold: f64,
        recursive: bool,
        objects: Option<Vec<usize>>,
    ) -> PyResult<Bound<'py, PyList>> {
        if !(x.is_finite() && y.is_finite()) {
            return Err(PyValueError::new_err(
                "raycast_from_camera: x and y must be finite",
            ));
        }
        let mut rc = threers::core::Raycaster::default();
        // Pick what the camera draws: `set_from_camera` leaves the layers
        // alone (three.js), so take the camera's here.
        with_camera(camera, |cam| {
            rc.set_from_camera(Vector2::new(x, y), cam);
            rc.layers = cam.layers();
        })?;
        let opts = CastOptions {
            near,
            far,
            first_hit_only,
            line_threshold,
            points_threshold,
            recursive,
            objects,
        };
        self.cast(py, rc, opts, "raycast_from_camera")
    }

    // ---- loaders ----

    /// Load a model file into the scene: ``.glb``, ``.gltf`` (external
    /// buffers resolved next to the file), ``.obj``, ``.stl``, ``.ply``, and
    /// ``.usda`` / ``.usdc`` / ``.usdz`` / ``.usd`` when built with the
    /// ``usd`` feature. Returns a handle: the mesh itself for the single-mesh
    /// formats (OBJ/STL/PLY), a group holding the model otherwise.
    /// ``material`` replaces the loaded materials (OBJ/STL/PLY default to a
    /// grey ``Material.standard``).
    #[pyo3(signature = (path, material=None))]
    fn load(
        &mut self,
        path: std::path::PathBuf,
        material: Option<PyRef<'_, PyMaterial>>,
    ) -> PyResult<usize> {
        let fmt = format_from_path(&path).map_err(PyValueError::new_err)?;
        let bytes = std::fs::read(&path)?;
        let mat = material.map(|m| m.inner.clone());
        let id = load_into(&mut self.inner, &bytes, fmt, path.parent(), mat)
            .map_err(|e| PyValueError::new_err(format!("load: {}: {e}", path.display())))?;
        self.handles.push(id);
        Ok(self.handles.len() - 1)
    }

    /// Like ``load``, from bytes. ``format``: ``"glb"``, ``"gltf"``,
    /// ``"obj"``, ``"stl"``, ``"ply"`` or ``"usd"`` (``"usda"``, ``"usdc"``,
    /// ``"usdz"`` also accepted). A ``.gltf`` given as bytes must embed its
    /// buffers as data URIs.
    #[pyo3(signature = (data, format, material=None))]
    fn load_bytes(
        &mut self,
        data: &[u8],
        format: &str,
        material: Option<PyRef<'_, PyMaterial>>,
    ) -> PyResult<usize> {
        let fmt = parse_format(format).map_err(PyValueError::new_err)?;
        let mat = material.map(|m| m.inner.clone());
        let id = load_into(&mut self.inner, data, fmt, None, mat)
            .map_err(|e| PyValueError::new_err(format!("load_bytes: {e}")))?;
        self.handles.push(id);
        Ok(self.handles.len() - 1)
    }
}

/// `Scene.raycast` keyword options — three.js `Raycaster` settings.
struct CastOptions {
    near: f64,
    far: f64,
    first_hit_only: bool,
    line_threshold: f64,
    points_threshold: f64,
    recursive: bool,
    objects: Option<Vec<usize>>,
}

impl PyScene {
    /// Insert under the root and hand out the next handle index.
    fn push(&mut self, obj: Object3D) -> usize {
        let id = self.inner.arena.insert(obj);
        self.inner.arena.add_child(self.inner.root, id);
        self.handles.push(id);
        self.handles.len() - 1
    }

    /// Apply `opts` to `rc` and run it over the scene, or over
    /// `opts.objects`.
    fn cast<'py>(
        &mut self,
        py: Python<'py>,
        mut rc: threers::core::Raycaster,
        opts: CastOptions,
        what: &str,
    ) -> PyResult<Bound<'py, PyList>> {
        rc.near = non_neg(opts.near, &format!("{what}: near"))?;
        if opts.far.is_nan() || opts.far <= rc.near as f64 {
            return Err(PyValueError::new_err(format!(
                "{what}: far must be > near, got {}",
                opts.far
            )));
        }
        rc.far = opts.far as f32;
        rc.first_hit_only = opts.first_hit_only;
        rc.params.line.threshold =
            non_neg(opts.line_threshold, &format!("{what}: line_threshold"))?;
        rc.params.points.threshold =
            non_neg(opts.points_threshold, &format!("{what}: points_threshold"))?;
        let roots = match &opts.objects {
            None => vec![self.inner.root],
            Some(handles) => handles
                .iter()
                .map(|&h| self.live(h))
                .collect::<PyResult<Vec<_>>>()?,
        };
        self.inner.update_world();
        let hits = rc.intersect_object_list(&self.inner.arena, &roots, opts.recursive);
        let out = PyList::empty(py);
        for h in hits {
            out.append(self.hit(py, &h)?)?;
        }
        Ok(out)
    }

    fn hit<'py>(
        &mut self,
        py: Python<'py>,
        h: &threers::core::Intersection,
    ) -> PyResult<Bound<'py, PyDict>> {
        let v3 = |v: Vector3| (v.x, v.y, v.z);
        let v2 = |v: Vector2| (v.x, v.y);
        let d = PyDict::new(py);
        d.set_item("handle", self.handle_of(h.object))?;
        d.set_item("distance", h.distance)?;
        d.set_item("point", v3(h.point))?;
        d.set_item("face_index", h.face.map(|_| h.face_index))?;
        let face = match &h.face {
            Some(f) => {
                let fd = PyDict::new(py);
                fd.set_item("a", f.a)?;
                fd.set_item("b", f.b)?;
                fd.set_item("c", f.c)?;
                fd.set_item("normal", v3(f.normal))?;
                fd.set_item("material_index", f.material_index)?;
                Some(fd)
            }
            None => None,
        };
        d.set_item("face", face)?;
        d.set_item("normal", h.normal.map(v3))?;
        d.set_item("uv", h.uv.map(v2))?;
        d.set_item("uv1", h.uv1.map(v2))?;
        d.set_item("barycoord", h.barycoord.map(v3))?;
        d.set_item("instance_id", h.instance_id)?;
        d.set_item("index", h.index)?;
        d.set_item("distance_to_ray", h.distance_to_ray)?;
        d.set_item("polygon_index", h.polygon_index)?;
        Ok(d)
    }

    /// The handle for `id`, registering it if it has none yet.
    fn handle_of(&mut self, id: ObjectId) -> usize {
        match self.handles.iter().position(|h| *h == id) {
            Some(i) => i,
            None => {
                self.handles.push(id);
                self.handles.len() - 1
            }
        }
    }

    fn live(&self, handle: usize) -> PyResult<ObjectId> {
        let id = *self
            .handles
            .get(handle)
            .ok_or_else(|| PyValueError::new_err(format!("bad object handle {handle}")))?;
        if self.inner.arena.get(id).is_none() {
            return Err(PyValueError::new_err(format!(
                "object {handle} was removed"
            )));
        }
        Ok(id)
    }

    fn obj_mut(&mut self, handle: usize) -> PyResult<&mut Object3D> {
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

/// `THREE.DoubleSide`.
const DOUBLE_SIDE: u32 = 2;
/// Per-axis cap on `add_quad` subdivision (1024² cells is already 2M triangles).
const MAX_QUAD_SEGMENTS: u32 = 1024;

fn parse_corners(obj: &Bound<'_, PyAny>) -> PyResult<[Vector3; 4]> {
    let items: Vec<Bound<'_, PyAny>> = obj
        .try_iter()
        .map_err(|_| PyTypeError::new_err("add_quad: corners must be a sequence"))?
        .collect::<PyResult<_>>()?;
    let flat: Vec<f32> = match items.len() {
        12 => items
            .iter()
            .map(|v| v.extract::<f32>())
            .collect::<PyResult<_>>()
            .map_err(|_| PyTypeError::new_err("add_quad: 12 corners values must all be numbers"))?,
        4 => {
            let mut flat = Vec::with_capacity(12);
            for (i, p) in items.iter().enumerate() {
                let v = if let Ok(v) = p.extract::<PyVector3>() {
                    v.inner
                } else if let Ok((x, y, z)) = p.extract::<(f32, f32, f32)>() {
                    Vector3::new(x, y, z)
                } else {
                    return Err(PyTypeError::new_err(format!(
                        "add_quad: corner {i} must be a Vector3 or an (x, y, z) sequence"
                    )));
                };
                flat.extend_from_slice(&[v.x, v.y, v.z]);
            }
            flat
        }
        n => {
            return Err(PyValueError::new_err(format!(
                "add_quad: corners needs 12 numbers or 4 points, got {n} items"
            )))
        }
    };
    if let Some(i) = flat.iter().position(|v| !v.is_finite()) {
        return Err(PyValueError::new_err(format!(
            "add_quad: corner {} has a non-finite coordinate",
            i / 3
        )));
    }
    let p = |i: usize| Vector3::new(flat[3 * i], flat[3 * i + 1], flat[3 * i + 2]);
    let c = [p(0), p(1), p(2), p(3)];
    // Twice the area vector: the diagonals' cross product.
    if (c[2] - c[0]).cross(c[3] - c[1]).length() <= 1e-12 {
        return Err(PyValueError::new_err(
            "add_quad: corners are degenerate (zero area)",
        ));
    }
    Ok(c)
}

// ---- geometry --------------------------------------------------------------

/// Per-axis cap on generated geometry segment counts.
const MAX_SEGMENTS: u32 = 4096;
/// Each Catmull–Clark round multiplies the face count by about 4.
const MAX_SUBDIVISION_LEVELS: u32 = 6;

// ---- fur ---------------------------------------------------------------------

/// Shell fur and strand hair's look and motion (``threers::fur::FurMaterial``):
/// every option a keyword, the rest as the Rust defaults.
#[pyclass(name = "FurMaterial", unsendable)]
#[derive(Clone)]
struct PyFurMaterial {
    inner: threers::fur::FurMaterial,
}

#[pymethods]
impl PyFurMaterial {
    #[new]
    #[pyo3(signature = (
        length=None, density=None, thickness=None, taper=None, gravity=None, curl=None,
        clumping=None, clump_size=None, length_variation=None, seed=None, shells=None,
        root_color=None, tip_color=None, occlusion=None, specular=None,
        specular_exponent=None, specular_shift=None, wind=None, wind_frequency=None,
        flutter=None, stiffness=None, damping=None, response=None, max_lag=None,
        placement=None, cell=None, shadow_height=None, lod=None, drag=None, turbulence=None
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        length: Option<f32>,
        density: Option<f32>,
        thickness: Option<f32>,
        taper: Option<f32>,
        gravity: Option<(f32, f32, f32)>,
        curl: Option<f32>,
        clumping: Option<f32>,
        clump_size: Option<f32>,
        length_variation: Option<f32>,
        seed: Option<u32>,
        shells: Option<u32>,
        root_color: Option<u32>,
        tip_color: Option<u32>,
        occlusion: Option<f32>,
        specular: Option<f32>,
        specular_exponent: Option<f32>,
        specular_shift: Option<f32>,
        wind: Option<(f32, f32, f32)>,
        wind_frequency: Option<f32>,
        flutter: Option<f32>,
        stiffness: Option<f32>,
        damping: Option<f32>,
        response: Option<f32>,
        max_lag: Option<f32>,
        placement: Option<&str>,
        cell: Option<f32>,
        shadow_height: Option<f32>,
        lod: Option<bool>,
        drag: Option<f32>,
        turbulence: Option<f32>,
    ) -> PyResult<Self> {
        let mut m = threers::fur::FurMaterial::default();
        let g = &mut m.groom;
        let v3 = |t: (f32, f32, f32)| Vector3::new(t.0, t.1, t.2);
        macro_rules! set {
            ($slot:expr, $v:expr) => {
                if let Some(v) = $v {
                    $slot = v;
                }
            };
        }
        set!(g.length, length);
        set!(g.density, density);
        set!(g.thickness, thickness);
        set!(g.taper, taper);
        set!(g.gravity, gravity.map(v3));
        set!(g.curl, curl);
        set!(g.clumping, clumping);
        set!(g.clump_size, clump_size);
        set!(g.length_variation, length_variation);
        set!(g.seed, seed);
        match placement {
            None | Some("uv") => {}
            Some("surface") => {
                g.placement = threers::fur::FurPlacement::Surface {
                    cell: cell.unwrap_or(0.02),
                }
            }
            Some(other) => {
                return Err(PyValueError::new_err(format!(
                    "FurMaterial: placement is 'uv' or 'surface', not {other:?}"
                )))
            }
        }
        set!(m.shells, shells);
        set!(m.root_color, root_color.map(Color::from_hex));
        set!(m.tip_color, tip_color.map(Color::from_hex));
        set!(m.occlusion, occlusion);
        set!(m.specular, specular);
        set!(m.specular_exponent, specular_exponent);
        set!(m.specular_shift, specular_shift);
        set!(m.wind, wind.map(v3));
        set!(m.wind_frequency, wind_frequency);
        set!(m.flutter, flutter);
        set!(m.dynamics.stiffness, stiffness);
        set!(m.dynamics.damping, damping);
        set!(m.dynamics.response, response);
        set!(m.dynamics.max_lag, max_lag);
        set!(m.shadow_height, shadow_height);
        set!(m.lod, lod);
        set!(m.drag, drag);
        set!(m.turbulence, turbulence);
        Ok(Self { inner: m })
    }

    #[getter]
    fn length(&self) -> f32 {
        self.inner.groom.length
    }
    #[getter]
    fn shells(&self) -> u32 {
        self.inner.shells
    }

    fn __repr__(&self) -> String {
        format!(
            "FurMaterial(length={}, density={}, shells={})",
            self.inner.groom.length, self.inner.groom.density, self.inner.shells
        )
    }
}

/// A triangle mesh's vertex data, built with the static factories (three.js
/// ``*Geometry`` classes, same parameters and defaults).
#[pyclass(name = "Geometry", unsendable)]
struct PyGeometry {
    inner: BufferGeometry,
}

#[pymethods]
impl PyGeometry {
    #[staticmethod]
    #[pyo3(name = "box", signature = (width=1.0, height=1.0, depth=1.0))]
    fn box_geometry(width: f64, height: f64, depth: f64) -> PyResult<Self> {
        Ok(Self::of(BoxGeometry::new(
            non_neg(width, "box: width")?,
            non_neg(height, "box: height")?,
            non_neg(depth, "box: depth")?,
        )))
    }

    #[staticmethod]
    #[pyo3(signature = (width=1.0, height=1.0, width_segments=1, height_segments=1))]
    fn plane(width: f64, height: f64, width_segments: u32, height_segments: u32) -> PyResult<Self> {
        Ok(Self::of(threers::geometries::PlaneGeometry::with_segments(
            non_neg(width, "plane: width")?,
            non_neg(height, "plane: height")?,
            seg(width_segments, 1, "plane: width_segments")?,
            seg(height_segments, 1, "plane: height_segments")?,
        )))
    }

    #[staticmethod]
    #[pyo3(signature = (radius=1.0, width_segments=32, height_segments=16))]
    fn sphere(radius: f64, width_segments: u32, height_segments: u32) -> PyResult<Self> {
        Ok(Self::of(SphereGeometry::new(
            non_neg(radius, "sphere: radius")?,
            seg(width_segments, 3, "sphere: width_segments")?,
            seg(height_segments, 2, "sphere: height_segments")?,
        )))
    }

    #[staticmethod]
    #[pyo3(signature = (radius_top=1.0, radius_bottom=1.0, height=1.0, radial_segments=32, height_segments=1, open_ended=false))]
    fn cylinder(
        radius_top: f64,
        radius_bottom: f64,
        height: f64,
        radial_segments: u32,
        height_segments: u32,
        open_ended: bool,
    ) -> PyResult<Self> {
        Ok(Self::of(threers::geometries::CylinderGeometry::new(
            non_neg(radius_top, "cylinder: radius_top")?,
            non_neg(radius_bottom, "cylinder: radius_bottom")?,
            non_neg(height, "cylinder: height")?,
            seg(radial_segments, 3, "cylinder: radial_segments")?,
            seg(height_segments, 1, "cylinder: height_segments")?,
            open_ended,
            0.0,
            std::f32::consts::TAU,
        )))
    }

    #[staticmethod]
    #[pyo3(signature = (radius=1.0, height=1.0, radial_segments=32, height_segments=1, open_ended=false))]
    fn cone(
        radius: f64,
        height: f64,
        radial_segments: u32,
        height_segments: u32,
        open_ended: bool,
    ) -> PyResult<Self> {
        Ok(Self::of(threers::geometries::ConeGeometry::new(
            non_neg(radius, "cone: radius")?,
            non_neg(height, "cone: height")?,
            seg(radial_segments, 3, "cone: radial_segments")?,
            seg(height_segments, 1, "cone: height_segments")?,
            open_ended,
            0.0,
            std::f32::consts::TAU,
        )))
    }

    /// ``arc`` in radians, (0, 2π].
    #[staticmethod]
    #[pyo3(signature = (radius=1.0, tube=0.4, radial_segments=12, tubular_segments=48, arc=std::f64::consts::TAU))]
    fn torus(
        radius: f64,
        tube: f64,
        radial_segments: u32,
        tubular_segments: u32,
        arc: f64,
    ) -> PyResult<Self> {
        if !(arc.is_finite() && arc > 0.0 && arc <= std::f64::consts::TAU + 1e-9) {
            return Err(PyValueError::new_err(format!(
                "torus: arc must be in (0, 2π], got {arc}"
            )));
        }
        Ok(Self::of(threers::geometries::TorusGeometry::new(
            non_neg(radius, "torus: radius")?,
            non_neg(tube, "torus: tube")?,
            seg(radial_segments, 2, "torus: radial_segments")?,
            seg(tubular_segments, 3, "torus: tubular_segments")?,
            arc as f32,
        )))
    }

    #[staticmethod]
    #[pyo3(signature = (radius=1.0, tube=0.4, tubular_segments=64, radial_segments=8, p=2, q=3))]
    fn torus_knot(
        radius: f64,
        tube: f64,
        tubular_segments: u32,
        radial_segments: u32,
        p: u32,
        q: u32,
    ) -> PyResult<Self> {
        if p == 0 || q == 0 {
            return Err(PyValueError::new_err("torus_knot: p and q must be >= 1"));
        }
        Ok(Self::of(threers::geometries::TorusKnotGeometry::new(
            non_neg(radius, "torus_knot: radius")?,
            non_neg(tube, "torus_knot: tube")?,
            seg(tubular_segments, 3, "torus_knot: tubular_segments")?,
            seg(radial_segments, 3, "torus_knot: radial_segments")?,
            p,
            q,
        )))
    }

    #[staticmethod]
    #[pyo3(signature = (inner_radius=0.5, outer_radius=1.0, theta_segments=32, phi_segments=1))]
    fn ring(
        inner_radius: f64,
        outer_radius: f64,
        theta_segments: u32,
        phi_segments: u32,
    ) -> PyResult<Self> {
        let inner = non_neg(inner_radius, "ring: inner_radius")?;
        let outer = non_neg(outer_radius, "ring: outer_radius")?;
        if inner > outer {
            return Err(PyValueError::new_err(
                "ring: inner_radius must be <= outer_radius",
            ));
        }
        Ok(Self::of(threers::geometries::RingGeometry::new(
            inner,
            outer,
            seg(theta_segments, 3, "ring: theta_segments")?,
            seg(phi_segments, 1, "ring: phi_segments")?,
            0.0,
            std::f32::consts::TAU,
        )))
    }

    #[staticmethod]
    #[pyo3(signature = (radius=1.0, segments=32))]
    fn circle(radius: f64, segments: u32) -> PyResult<Self> {
        Ok(Self::of(threers::geometries::CircleGeometry::new(
            non_neg(radius, "circle: radius")?,
            seg(segments, 3, "circle: segments")?,
            0.0,
            std::f32::consts::TAU,
        )))
    }

    #[staticmethod]
    #[pyo3(signature = (radius=1.0, length=1.0, cap_segments=4, radial_segments=8))]
    fn capsule(
        radius: f64,
        length: f64,
        cap_segments: u32,
        radial_segments: u32,
    ) -> PyResult<Self> {
        Ok(Self::of(threers::geometries::CapsuleGeometry::new(
            non_neg(radius, "capsule: radius")?,
            non_neg(length, "capsule: length")?,
            seg(cap_segments, 1, "capsule: cap_segments")?,
            seg(radial_segments, 3, "capsule: radial_segments")?,
        )))
    }

    /// Revolve a profile around Y. ``points``: flat ``[x0, y0, x1, y1, …]``
    /// or ``(x, y)`` pairs; at least 2.
    #[staticmethod]
    #[pyo3(signature = (points, segments=12))]
    fn lathe(points: &Bound<'_, PyAny>, segments: u32) -> PyResult<Self> {
        let flat = flatten_points(points, 2, "lathe: points")?;
        let pts: Vec<Vector2> = flat.chunks(2).map(|c| Vector2::new(c[0], c[1])).collect();
        Ok(Self::of(threers::geometries::LatheGeometry::new(
            &pts,
            seg(segments, 1, "lathe: segments")?,
            0.0,
            std::f32::consts::TAU,
        )))
    }

    /// A tube along a Catmull-Rom curve through ``points`` (flat
    /// ``[x, y, z, …]``, or ``Vector3`` / ``(x, y, z)`` points; at least 2).
    #[staticmethod]
    #[pyo3(signature = (points, tubular_segments=64, radius=1.0, radial_segments=8, closed=false))]
    fn tube(
        points: &Bound<'_, PyAny>,
        tubular_segments: u32,
        radius: f64,
        radial_segments: u32,
        closed: bool,
    ) -> PyResult<Self> {
        let flat = flatten_points(points, 3, "tube: points")?;
        let pts: Vec<Vector3> = flat
            .chunks(3)
            .map(|c| Vector3::new(c[0], c[1], c[2]))
            .collect();
        let curve = threers::CatmullRomCurve3::new(pts);
        Ok(Self::of(threers::geometries::TubeGeometry::new(
            &curve,
            seg(tubular_segments, 1, "tube: tubular_segments")?,
            non_neg(radius, "tube: radius")?,
            seg(radial_segments, 3, "tube: radial_segments")?,
            closed,
        )))
    }

    #[staticmethod]
    #[pyo3(signature = (radius=1.0, detail=0))]
    fn icosahedron(radius: f64, detail: u32) -> PyResult<Self> {
        let (r, d) = poly_args(radius, detail, "icosahedron")?;
        Ok(Self::of(threers::geometries::IcosahedronGeometry::new(
            r, d,
        )))
    }

    #[staticmethod]
    #[pyo3(signature = (radius=1.0, detail=0))]
    fn octahedron(radius: f64, detail: u32) -> PyResult<Self> {
        let (r, d) = poly_args(radius, detail, "octahedron")?;
        Ok(Self::of(threers::geometries::OctahedronGeometry::new(r, d)))
    }

    #[staticmethod]
    #[pyo3(signature = (radius=1.0, detail=0))]
    fn tetrahedron(radius: f64, detail: u32) -> PyResult<Self> {
        let (r, d) = poly_args(radius, detail, "tetrahedron")?;
        Ok(Self::of(threers::geometries::TetrahedronGeometry::new(
            r, d,
        )))
    }

    #[staticmethod]
    #[pyo3(signature = (radius=1.0, detail=0))]
    fn dodecahedron(radius: f64, detail: u32) -> PyResult<Self> {
        let (r, d) = poly_args(radius, detail, "dodecahedron")?;
        Ok(Self::of(threers::geometries::DodecahedronGeometry::new(
            r, d,
        )))
    }

    /// Same corners as ``Scene.add_quad``.
    #[staticmethod]
    #[pyo3(signature = (corners, width_segments=1, height_segments=1))]
    fn quad(
        corners: &Bound<'_, PyAny>,
        width_segments: u32,
        height_segments: u32,
    ) -> PyResult<Self> {
        let corners = parse_corners(corners)?;
        Ok(Self::of(QuadGeometry::with_segments(
            corners,
            seg(width_segments, 1, "quad: width_segments")?,
            seg(height_segments, 1, "quad: height_segments")?,
        )))
    }

    /// Custom geometry from flat ``positions`` (``x, y, z`` per vertex),
    /// optional triangle ``indices`` and optional ``normals`` (computed when
    /// omitted).
    #[staticmethod]
    #[pyo3(signature = (positions, indices=None, normals=None))]
    fn from_arrays(
        positions: Vec<f32>,
        indices: Option<Vec<u32>>,
        normals: Option<Vec<f32>>,
    ) -> PyResult<Self> {
        geometry_from_arrays(positions, indices, normals)
            .map(Self::of)
            .map_err(|e| PyValueError::new_err(format!("from_arrays: {e}")))
    }

    /// three.js ``EdgesGeometry``: the edges of ``source`` whose faces meet
    /// at ``threshold_angle`` degrees or more (default 1), plus border edges,
    /// as a line geometry for ``Scene.add_line_segments``. Edges inside one
    /// polygon (a quad's diagonal) are never emitted.
    #[staticmethod]
    #[pyo3(signature = (source, threshold_angle=1.0))]
    fn edges(source: PyRef<'_, PyGeometry>, threshold_angle: f64) -> PyResult<Self> {
        if !(threshold_angle.is_finite() && (0.0..=180.0).contains(&threshold_angle)) {
            return Err(PyValueError::new_err(format!(
                "edges: threshold_angle must be 0..=180 degrees, got {threshold_angle}"
            )));
        }
        Ok(Self::of(threers::geometries::EdgesGeometry::new(
            &source.inner,
            (threshold_angle as f32).to_radians(),
        )))
    }

    /// three.js ``WireframeGeometry``: every edge of ``source`` once, as a
    /// line geometry. A geometry carrying polygons draws their outlines —
    /// quads without diagonals.
    #[staticmethod]
    fn wireframe(source: PyRef<'_, PyGeometry>) -> Self {
        Self::of(threers::geometries::WireframeGeometry::new(&source.inner))
    }

    /// Flat vertex positions, ``x, y, z`` per vertex.
    fn positions(&self) -> Vec<f32> {
        self.inner
            .get_attribute("position")
            .map(|a| a.array.clone())
            .unwrap_or_default()
    }

    /// Triangle vertex indices, or ``None`` for a non-indexed geometry.
    fn index(&self) -> Option<Vec<u32>> {
        self.inner.index.clone()
    }

    /// Triangles per polygon, in triangle order (``2`` is a quad), or
    /// ``None`` when the geometry carries no polygon grouping. The grid
    /// generators, ``quad``, and the OBJ / PLY / FBX / USD loaders attach
    /// one.
    fn polygon_counts(&self) -> Option<Vec<u32>> {
        self.inner.polygon_counts().map(|p| p.to_vec())
    }

    /// Group consecutive triangles into polygons: ``counts[i]`` triangles
    /// form polygon ``i``. The counts must be >= 1 and sum to
    /// ``triangle_count()``. Wireframes and ``Geometry.edges`` then follow
    /// the polygon outlines.
    fn set_polygons(&mut self, counts: Vec<u32>) -> PyResult<()> {
        check_polygons(&counts, self.inner.triangle_count()).map_err(PyValueError::new_err)?;
        self.inner.set_polygons(counts);
        Ok(())
    }

    /// Pair adjacent triangles back into quads (Blender's "Tris to Quads"):
    /// two triangles merge when they fold by at most ``max_face_angle`` and
    /// every corner of the quad is within ``max_shape_angle`` of 90°
    /// (degrees). Only the index and the polygon grouping change; existing
    /// polygons stay whole. Returns how many quads were formed.
    #[pyo3(signature = (max_face_angle=40.0, max_shape_angle=40.0))]
    fn infer_quads(&mut self, max_face_angle: f64, max_shape_angle: f64) -> PyResult<usize> {
        let angle = |v: f64, name: &str| -> PyResult<f32> {
            if !(v.is_finite() && (0.0..=180.0).contains(&v)) {
                return Err(PyValueError::new_err(format!(
                    "infer_quads: {name} must be 0..=180 degrees, got {v}"
                )));
            }
            Ok((v as f32).to_radians())
        };
        let face = angle(max_face_angle, "max_face_angle")?;
        let shape = angle(max_shape_angle, "max_shape_angle")?;
        Ok(self.inner.infer_quads(face, shape))
    }

    /// Catmull–Clark subdivision, ``levels`` rounds (max 6): every polygon
    /// becomes quads (a quad → 4, a triangle → 3), as a new geometry with
    /// smooth normals and its quads grouped. The cage is this geometry's
    /// polygons (its triangles where there is no grouping), welded by
    /// position. ``sharp_boundary`` holds open borders in place. UVs and
    /// other attributes are not carried over.
    #[pyo3(signature = (levels=1, sharp_boundary=true))]
    fn subdivide(&self, levels: u32, sharp_boundary: bool) -> PyResult<PyGeometry> {
        if levels > MAX_SUBDIVISION_LEVELS {
            return Err(PyValueError::new_err(format!(
                "subdivide: levels must be 0..={MAX_SUBDIVISION_LEVELS}, got {levels}"
            )));
        }
        let options = threers::geometries::SubdivisionOptions {
            sharp_boundary,
            ..Default::default()
        };
        Ok(Self::of(
            self.inner.subdivide_with(levels as usize, &options),
        ))
    }

    /// Number of polygons (the triangle count when there is no grouping).
    fn polygon_count(&self) -> usize {
        self.inner.polygon_ranges().count()
    }

    /// Polygons with exactly four corners.
    fn quad_count(&self) -> usize {
        polygon_loops(&self.inner)
            .iter()
            .filter(|l| l.as_ref().is_some_and(|l| l.len() == 4))
            .count()
    }

    /// Each polygon's outline as vertex indices in winding order (a quad
    /// gives 4), or ``None`` for a polygon whose triangles do not form one
    /// simple loop.
    fn polygon_loops(&self) -> Vec<Option<Vec<u32>>> {
        polygon_loops(&self.inner)
    }

    /// Polygon outline edges as ``(a, b)`` vertex-index pairs, each once —
    /// what a quad wireframe draws.
    fn polygon_edges(&self) -> Vec<(u32, u32)> {
        self.inner
            .polygon_edges()
            .into_iter()
            .map(|[a, b]| (a, b))
            .collect()
    }

    fn vertex_count(&self) -> usize {
        self.inner
            .get_attribute("position")
            .map_or(0, |a| a.count())
    }

    fn triangle_count(&self) -> usize {
        self.inner.triangle_count()
    }

    fn __repr__(&self) -> String {
        format!(
            "Geometry(vertices={}, triangles={})",
            self.vertex_count(),
            self.triangle_count()
        )
    }
}

impl PyGeometry {
    fn of(inner: BufferGeometry) -> Self {
        Self { inner }
    }
}

/// Flat numbers (`dim` per point) or point sequences / `Vector3`s, at least
/// 2 points, all finite.
fn flatten_points(obj: &Bound<'_, PyAny>, dim: usize, what: &str) -> PyResult<Vec<f32>> {
    let items: Vec<Bound<'_, PyAny>> = obj
        .try_iter()
        .map_err(|_| PyTypeError::new_err(format!("{what}: must be a sequence")))?
        .collect::<PyResult<_>>()?;
    let mut flat = Vec::with_capacity(items.len() * dim);
    let all_nums = items.iter().all(|v| v.extract::<f64>().is_ok());
    if all_nums {
        for v in &items {
            flat.push(v.extract::<f64>()?);
        }
    } else {
        for (i, p) in items.iter().enumerate() {
            if dim == 3 {
                if let Ok(v) = p.extract::<PyVector3>() {
                    flat.extend([v.inner.x as f64, v.inner.y as f64, v.inner.z as f64]);
                    continue;
                }
            }
            let v: Vec<f64> = p.extract().map_err(|_| {
                PyTypeError::new_err(format!(
                    "{what}: item {i} must be a number or a {dim}-number point"
                ))
            })?;
            if v.len() != dim {
                return Err(PyValueError::new_err(format!(
                    "{what}: point {i} has {} coordinates, expected {dim}",
                    v.len()
                )));
            }
            flat.extend(v);
        }
    }
    if !flat.len().is_multiple_of(dim) {
        return Err(PyValueError::new_err(format!(
            "{what}: {} numbers is not a whole number of {dim}D points",
            flat.len()
        )));
    }
    if flat.len() < 2 * dim {
        return Err(PyValueError::new_err(format!(
            "{what}: needs at least 2 points"
        )));
    }
    if flat.iter().any(|v| !v.is_finite()) {
        return Err(PyValueError::new_err(format!(
            "{what}: non-finite coordinate"
        )));
    }
    Ok(flat.into_iter().map(|v| v as f32).collect())
}

fn poly_args(radius: f64, detail: u32, what: &str) -> PyResult<(f32, usize)> {
    if detail > 32 {
        return Err(PyValueError::new_err(format!(
            "{what}: detail must be 0..=32, got {detail}"
        )));
    }
    Ok((
        non_neg(radius, &format!("{what}: radius"))?,
        detail as usize,
    ))
}

// ---- material --------------------------------------------------------------

/// A surface material, built with the static factories (three.js
/// ``Mesh*Material``). Unlit: ``basic``, ``normal``. Lit (need a light to
/// show): ``standard``, ``physical``, ``lambert``, ``phong``. Each factory
/// takes keyword arguments and rejects the ones that do not apply to it.
#[pyclass(name = "Material", unsendable)]
struct PyMaterial {
    inner: Material,
}

/// Material keyword arguments, before validation.
#[derive(Default)]
struct MatOpts {
    color: Option<u32>,
    emissive: Option<u32>,
    emissive_intensity: Option<f64>,
    roughness: Option<f64>,
    metalness: Option<f64>,
    opacity: Option<f64>,
    transparent: Option<bool>,
    wireframe: Option<bool>,
    side: Option<String>,
    specular: Option<u32>,
    shininess: Option<f64>,
    clearcoat: Option<f64>,
    clearcoat_roughness: Option<f64>,
    transmission: Option<f64>,
    ior: Option<f64>,
    thickness: Option<f64>,
    depth_test: Option<bool>,
    // MeshPhysicalMaterial's remaining scalars and colours.
    specular_intensity: Option<f64>,
    specular_color: Option<u32>,
    sheen: Option<f64>,
    sheen_color: Option<u32>,
    sheen_roughness: Option<f64>,
    iridescence: Option<f64>,
    iridescence_ior: Option<f64>,
    iridescence_thickness_range: Option<(f64, f64)>,
    anisotropy: Option<f64>,
    anisotropy_rotation: Option<f64>,
    attenuation_color: Option<u32>,
    attenuation_distance: Option<f64>,
    dispersion: Option<f64>,
}

impl MatOpts {
    fn from_kwargs(kind: MatKind, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        let mut o = Self::default();
        let Some(kwargs) = kwargs else {
            return Ok(o);
        };
        for (k, v) in kwargs.iter() {
            let key: String = k.extract()?;
            let typed =
                |e: PyErr| PyTypeError::new_err(format!("Material.{}: {key}: {e}", kind.name()));
            match key.as_str() {
                "color" => o.color = Some(v.extract().map_err(typed)?),
                "emissive" => o.emissive = Some(v.extract().map_err(typed)?),
                "emissive_intensity" => o.emissive_intensity = Some(v.extract().map_err(typed)?),
                "roughness" => o.roughness = Some(v.extract().map_err(typed)?),
                "metalness" => o.metalness = Some(v.extract().map_err(typed)?),
                "opacity" => o.opacity = Some(v.extract().map_err(typed)?),
                "transparent" => o.transparent = Some(v.extract().map_err(typed)?),
                "wireframe" => o.wireframe = Some(v.extract().map_err(typed)?),
                "side" => o.side = Some(v.extract().map_err(typed)?),
                "specular" => o.specular = Some(v.extract().map_err(typed)?),
                "shininess" => o.shininess = Some(v.extract().map_err(typed)?),
                "clearcoat" => o.clearcoat = Some(v.extract().map_err(typed)?),
                "clearcoat_roughness" => o.clearcoat_roughness = Some(v.extract().map_err(typed)?),
                "transmission" => o.transmission = Some(v.extract().map_err(typed)?),
                "ior" => o.ior = Some(v.extract().map_err(typed)?),
                "thickness" => o.thickness = Some(v.extract().map_err(typed)?),
                "depth_test" => o.depth_test = Some(v.extract().map_err(typed)?),
                "specular_intensity" => o.specular_intensity = Some(v.extract().map_err(typed)?),
                "specular_color" => o.specular_color = Some(v.extract().map_err(typed)?),
                "sheen" => o.sheen = Some(v.extract().map_err(typed)?),
                "sheen_color" => o.sheen_color = Some(v.extract().map_err(typed)?),
                "sheen_roughness" => o.sheen_roughness = Some(v.extract().map_err(typed)?),
                "iridescence" => o.iridescence = Some(v.extract().map_err(typed)?),
                "iridescence_ior" => o.iridescence_ior = Some(v.extract().map_err(typed)?),
                "iridescence_thickness_range" => {
                    o.iridescence_thickness_range = Some(v.extract().map_err(typed)?)
                }
                "anisotropy" => o.anisotropy = Some(v.extract().map_err(typed)?),
                "anisotropy_rotation" => o.anisotropy_rotation = Some(v.extract().map_err(typed)?),
                "attenuation_color" => o.attenuation_color = Some(v.extract().map_err(typed)?),
                "attenuation_distance" => {
                    o.attenuation_distance = Some(v.extract().map_err(typed)?)
                }
                "dispersion" => o.dispersion = Some(v.extract().map_err(typed)?),
                _ => {
                    return Err(PyTypeError::new_err(format!(
                        "Material.{}() got an unexpected keyword argument '{key}'",
                        kind.name()
                    )))
                }
            }
        }
        Ok(o)
    }
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

#[pymethods]
impl PyMaterial {
    /// three.js ``MeshBasicMaterial`` — unlit flat colour.
    #[staticmethod]
    #[pyo3(signature = (**opts))]
    fn basic(opts: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        Self::build(MatKind::Basic, opts)
    }

    /// three.js ``MeshStandardMaterial`` — metal/rough PBR.
    #[staticmethod]
    #[pyo3(signature = (**opts))]
    fn standard(opts: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        Self::build(MatKind::Standard, opts)
    }

    /// three.js ``MeshPhysicalMaterial`` — standard + clearcoat, transmission, IOR.
    #[staticmethod]
    #[pyo3(signature = (**opts))]
    fn physical(opts: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        Self::build(MatKind::Physical, opts)
    }

    /// three.js ``MeshLambertMaterial`` — diffuse only.
    #[staticmethod]
    #[pyo3(signature = (**opts))]
    fn lambert(opts: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        Self::build(MatKind::Lambert, opts)
    }

    /// three.js ``MeshPhongMaterial`` — diffuse + specular highlight.
    #[staticmethod]
    #[pyo3(signature = (**opts))]
    fn phong(opts: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        Self::build(MatKind::Phong, opts)
    }

    /// three.js ``MeshToonMaterial`` — cel shading: each light read through
    /// the ``gradientMap`` (``set_map("gradientMap", texture)``), two tones
    /// without one.
    #[staticmethod]
    #[pyo3(signature = (**opts))]
    fn toon(opts: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        Self::build(MatKind::Toon, opts)
    }

    /// three.js ``MeshNormalMaterial`` — normals as colours (unlit).
    #[staticmethod]
    #[pyo3(signature = (**opts))]
    fn normal(opts: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        Self::build(MatKind::Normal, opts)
    }

    /// three.js ``LineBasicMaterial``, for ``Scene.add_line_segments``
    /// (``color``, ``opacity``, ``depth_test``).
    #[staticmethod]
    #[pyo3(signature = (**opts))]
    fn line(opts: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        Self::build(MatKind::Line, opts)
    }

    /// ``"basic"``, ``"standard"``, ``"physical"``, ``"lambert"``,
    /// ``"phong"``, ``"normal"`` or ``"line"``.
    #[getter]
    fn kind(&self) -> &'static str {
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

    fn __repr__(&self) -> String {
        format!("Material.{}(…)", self.kind())
    }

    /// three.js ``material[slot] = texture``: ``"map"``, ``"alphaMap"``,
    /// ``"normalMap"``, ``"aoMap"``, ``"emissiveMap"``, ``"roughnessMap"``,
    /// ``"metalnessMap"``, ``"lightMap"``, ``"bumpMap"``, ``"specularMap"``,
    /// ``"displacementMap"``, and Physical's ``"clearcoatMap"``,
    /// ``"sheenColorMap"``, ``"transmissionMap"`` and the rest. ``None``
    /// empties the slot. ValueError for a slot this material does not have.
    #[pyo3(signature = (slot, texture))]
    fn set_map(&mut self, slot: &str, texture: Option<PyRef<'_, PyTexture>>) -> PyResult<()> {
        let kind = self.kind();
        match self.inner.map_slot_mut(slot) {
            Some(s) => {
                *s = texture.map(|t| t.inner.clone());
                Ok(())
            }
            None => Err(PyValueError::new_err(format!(
                "Material.{kind} has no '{slot}' map"
            ))),
        }
    }

    /// The texture in a map slot, or ``None``.
    fn get_map(&self, slot: &str) -> PyResult<Option<PyTexture>> {
        let mut m = self.inner.clone();
        match m.map_slot_mut(slot) {
            Some(s) => Ok(s.clone().map(|inner| PyTexture { inner })),
            None => Err(PyValueError::new_err(format!(
                "Material.{} has no '{slot}' map",
                self.kind()
            ))),
        }
    }
}

/// three.js ``Texture``: an RGBA8 image, with the sampling state three.js
/// gives every texture — ``wrap_s``/``wrap_t``, ``mag_filter``/``min_filter``
/// and the mipmap filter, the uv transform (``offset``, ``repeat``,
/// ``rotation``, ``center``), the uv set it reads (``channel``) and whether it
/// holds sRGB colour or data.
#[pyclass(name = "Texture", module = "threers")]
#[derive(Clone)]
struct PyTexture {
    inner: std::sync::Arc<threers::textures::Texture>,
}

fn texture_format(color_space: &str) -> PyResult<threers::textures::TextureFormat> {
    match color_space {
        "srgb" => Ok(threers::textures::TextureFormat::Rgba8UnormSrgb),
        "linear" => Ok(threers::textures::TextureFormat::Rgba8Unorm),
        s => Err(PyValueError::new_err(format!(
            "color_space must be 'srgb' or 'linear', got '{s}'"
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

fn parse_wrap(s: &str) -> PyResult<threers::textures::TextureWrap> {
    use threers::textures::TextureWrap::*;
    match s {
        "repeat" => Ok(Repeat),
        "mirror" => Ok(MirroredRepeat),
        "clamp" => Ok(ClampToEdge),
        s => Err(PyValueError::new_err(format!(
            "wrap must be 'repeat', 'mirror' or 'clamp', got '{s}'"
        ))),
    }
}

fn parse_filter(s: &str) -> PyResult<threers::textures::TextureFilter> {
    match s {
        "linear" => Ok(threers::textures::TextureFilter::Linear),
        "nearest" => Ok(threers::textures::TextureFilter::Nearest),
        s => Err(PyValueError::new_err(format!(
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

impl PyTexture {
    fn edit(&mut self) -> &mut threers::textures::Texture {
        std::sync::Arc::make_mut(&mut self.inner)
    }
}

#[pymethods]
impl PyTexture {
    /// ``width × height`` RGBA8 pixels, rows top first. ``color_space``
    /// ``"srgb"`` (colour, decoded when sampled — three.js's
    /// ``SRGBColorSpace``) or ``"linear"`` (data: normals, roughness, …).
    #[new]
    #[pyo3(signature = (width, height, data, color_space = "srgb"))]
    fn new(width: u32, height: u32, data: &[u8], color_space: &str) -> PyResult<Self> {
        let need = width as usize * height as usize * 4;
        if width == 0 || height == 0 || data.len() != need {
            return Err(PyValueError::new_err(format!(
                "a {width}x{height} RGBA texture takes {need} bytes, got {}",
                data.len()
            )));
        }
        let mut t = threers::textures::Texture::new(
            width,
            height,
            texture_format(color_space)?,
            data.to_vec(),
        );
        // Rows arrive top first; three.js's image upload flips them (`flipY`).
        t.flip_y = true;
        Ok(Self {
            inner: std::sync::Arc::new(t),
        })
    }

    /// A PNG file, as three.js's ``TextureLoader`` gives it (``flip_y``).
    #[staticmethod]
    #[pyo3(signature = (path, color_space = "srgb"))]
    fn from_file(path: &str, color_space: &str) -> PyResult<Self> {
        let bytes =
            std::fs::read(path).map_err(|e| PyValueError::new_err(format!("{path}: {e}")))?;
        let png = threers::utils::png::decode_png(&bytes).map_err(|e| {
            PyValueError::new_err(format!("{path}: not a PNG this loader reads ({e})"))
        })?;
        Self::new(png.width, png.height, &png.rgba, color_space)
    }

    #[getter]
    fn width(&self) -> u32 {
        self.inner.width
    }
    #[getter]
    fn height(&self) -> u32 {
        self.inner.height
    }
    #[getter]
    fn color_space(&self) -> &'static str {
        if self.inner.format == threers::textures::TextureFormat::Rgba8UnormSrgb {
            "srgb"
        } else {
            "linear"
        }
    }
    #[setter]
    fn set_color_space(&mut self, v: &str) -> PyResult<()> {
        self.edit().format = texture_format(v)?;
        Ok(())
    }
    #[getter]
    fn wrap_s(&self) -> &'static str {
        wrap_name(self.inner.wrap_s)
    }
    #[setter]
    fn set_wrap_s(&mut self, v: &str) -> PyResult<()> {
        self.edit().wrap_s = parse_wrap(v)?;
        Ok(())
    }
    #[getter]
    fn wrap_t(&self) -> &'static str {
        wrap_name(self.inner.wrap_t)
    }
    #[setter]
    fn set_wrap_t(&mut self, v: &str) -> PyResult<()> {
        self.edit().wrap_t = parse_wrap(v)?;
        Ok(())
    }
    #[getter]
    fn mag_filter(&self) -> &'static str {
        filter_name(self.inner.mag_filter)
    }
    #[setter]
    fn set_mag_filter(&mut self, v: &str) -> PyResult<()> {
        self.edit().mag_filter = parse_filter(v)?;
        Ok(())
    }
    #[getter]
    fn min_filter(&self) -> &'static str {
        filter_name(self.inner.min_filter)
    }
    #[setter]
    fn set_min_filter(&mut self, v: &str) -> PyResult<()> {
        self.edit().min_filter = parse_filter(v)?;
        Ok(())
    }
    /// Between mip levels: ``"linear"`` (three.js's ``*MipmapLinearFilter``,
    /// the default), ``"nearest"`` (``*MipmapNearestFilter``) or ``None``
    /// (level 0 only — ``LinearFilter`` / ``NearestFilter``).
    #[getter]
    fn mipmap_filter(&self) -> Option<&'static str> {
        self.inner.mipmap_filter.map(filter_name)
    }
    #[setter]
    fn set_mipmap_filter(&mut self, v: Option<&str>) -> PyResult<()> {
        self.edit().mipmap_filter = v.map(parse_filter).transpose()?;
        Ok(())
    }
    #[getter]
    fn offset(&self) -> (f32, f32) {
        (self.inner.offset.x, self.inner.offset.y)
    }
    #[setter]
    fn set_offset(&mut self, v: (f32, f32)) {
        let t = self.edit();
        t.offset = Vector2::new(v.0, v.1);
        t.matrix_auto_update = true;
    }
    #[getter]
    fn repeat(&self) -> (f32, f32) {
        (self.inner.repeat.x, self.inner.repeat.y)
    }
    #[setter]
    fn set_repeat(&mut self, v: (f32, f32)) {
        let t = self.edit();
        t.repeat = Vector2::new(v.0, v.1);
        t.matrix_auto_update = true;
    }
    #[getter]
    fn rotation(&self) -> f32 {
        self.inner.rotation
    }
    #[setter]
    fn set_rotation(&mut self, v: f32) {
        let t = self.edit();
        t.rotation = v;
        t.matrix_auto_update = true;
    }
    #[getter]
    fn center(&self) -> (f32, f32) {
        (self.inner.center.x, self.inner.center.y)
    }
    #[setter]
    fn set_center(&mut self, v: (f32, f32)) {
        let t = self.edit();
        t.center = Vector2::new(v.0, v.1);
        t.matrix_auto_update = true;
    }
    /// three.js ``channel``: the uv set read — 0 ``uv``, 1–3 ``uv1``–``uv3``.
    #[getter]
    fn channel(&self) -> u32 {
        self.inner.channel
    }
    #[setter]
    fn set_channel(&mut self, v: u32) -> PyResult<()> {
        if v > 3 {
            return Err(PyValueError::new_err(format!(
                "channel must be 0-3, got {v}"
            )));
        }
        self.edit().channel = v;
        Ok(())
    }
    #[getter]
    fn flip_y(&self) -> bool {
        self.inner.flip_y
    }
    #[setter]
    fn set_flip_y(&mut self, v: bool) {
        self.edit().flip_y = v;
    }

    fn __repr__(&self) -> String {
        format!(
            "Texture({}x{}, {})",
            self.inner.width,
            self.inner.height,
            self.color_space()
        )
    }
}

impl PyMaterial {
    fn build(kind: MatKind, opts: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        let o = MatOpts::from_kwargs(kind, opts)?;
        use MatKind::*;
        let all: &[MatKind] = &[Basic, Standard, Physical, Lambert, Phong, Toon, Normal];
        let all_line: &[MatKind] = &[
            Basic, Standard, Physical, Lambert, Phong, Toon, Normal, Line,
        ];
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
                "emissive_intensity",
                o.emissive_intensity.is_some(),
                &[Standard, Physical],
            ),
            ("roughness", o.roughness.is_some(), &[Standard, Physical]),
            ("metalness", o.metalness.is_some(), &[Standard, Physical]),
            ("opacity", o.opacity.is_some(), all_line),
            ("transparent", o.transparent.is_some(), all),
            ("wireframe", o.wireframe.is_some(), all),
            (
                "side",
                o.side.is_some(),
                &[Basic, Standard, Physical, Lambert, Phong, Toon],
            ),
            ("specular", o.specular.is_some(), &[Phong]),
            ("shininess", o.shininess.is_some(), &[Phong]),
            ("clearcoat", o.clearcoat.is_some(), &[Physical]),
            (
                "clearcoat_roughness",
                o.clearcoat_roughness.is_some(),
                &[Physical],
            ),
            ("transmission", o.transmission.is_some(), &[Physical]),
            ("ior", o.ior.is_some(), &[Physical]),
            ("thickness", o.thickness.is_some(), &[Physical]),
            ("depth_test", o.depth_test.is_some(), &[Line]),
            (
                "specular_intensity",
                o.specular_intensity.is_some(),
                &[Physical],
            ),
            ("specular_color", o.specular_color.is_some(), &[Physical]),
            ("sheen", o.sheen.is_some(), &[Physical]),
            ("sheen_color", o.sheen_color.is_some(), &[Physical]),
            ("sheen_roughness", o.sheen_roughness.is_some(), &[Physical]),
            ("iridescence", o.iridescence.is_some(), &[Physical]),
            ("iridescence_ior", o.iridescence_ior.is_some(), &[Physical]),
            (
                "iridescence_thickness_range",
                o.iridescence_thickness_range.is_some(),
                &[Physical],
            ),
            ("anisotropy", o.anisotropy.is_some(), &[Physical]),
            (
                "anisotropy_rotation",
                o.anisotropy_rotation.is_some(),
                &[Physical],
            ),
            (
                "attenuation_color",
                o.attenuation_color.is_some(),
                &[Physical],
            ),
            (
                "attenuation_distance",
                o.attenuation_distance.is_some(),
                &[Physical],
            ),
            ("dispersion", o.dispersion.is_some(), &[Physical]),
        ];
        for (name, set, allowed) in fields {
            if set && !allowed.contains(&kind) {
                return Err(PyTypeError::new_err(format!(
                    "Material.{}: '{name}' does not apply to this material",
                    kind.name()
                )));
            }
        }
        let color = hex(o.color.unwrap_or(0xffffff), "color")?;
        let emissive = hex(o.emissive.unwrap_or(0), "emissive")?;
        let opacity = unit(o.opacity.unwrap_or(1.0), "opacity")?;
        let wireframe = o.wireframe.unwrap_or(false);
        let side = match o.side.as_deref() {
            None | Some("front") => 0,
            Some("back") => 1,
            Some("double") => 2,
            Some(s) => {
                return Err(PyValueError::new_err(format!(
                    "side must be 'front', 'back' or 'double', got '{s}'"
                )))
            }
        };
        let roughness = unit(o.roughness.unwrap_or(1.0), "roughness")?;
        let metalness = unit(o.metalness.unwrap_or(0.0), "metalness")?;
        let emissive_intensity =
            non_neg(o.emissive_intensity.unwrap_or(1.0), "emissive_intensity")?;
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
                    return Err(PyValueError::new_err(format!(
                        "ior must be in [1, 2.333], got {ior}"
                    )));
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
                    clearcoat: unit(o.clearcoat.unwrap_or(0.0), "clearcoat")?,
                    clearcoat_roughness: unit(
                        o.clearcoat_roughness.unwrap_or(0.0),
                        "clearcoat_roughness",
                    )?,
                    transmission: unit(o.transmission.unwrap_or(0.0), "transmission")?,
                    ior: ior as f32,
                    thickness: non_neg(o.thickness.unwrap_or(0.0), "thickness")?,
                    specular_intensity: unit(
                        o.specular_intensity.unwrap_or(1.0),
                        "specular_intensity",
                    )?,
                    specular_color: hex(o.specular_color.unwrap_or(0xffffff), "specular_color")?,
                    sheen: unit(o.sheen.unwrap_or(0.0), "sheen")?,
                    sheen_color: hex(o.sheen_color.unwrap_or(0x000000), "sheen_color")?,
                    sheen_roughness: unit(o.sheen_roughness.unwrap_or(1.0), "sheen_roughness")?,
                    iridescence: unit(o.iridescence.unwrap_or(0.0), "iridescence")?,
                    iridescence_ior: {
                        let v = o.iridescence_ior.unwrap_or(1.3);
                        if !(v.is_finite() && (1.0..=2.333).contains(&v)) {
                            return Err(PyValueError::new_err(format!(
                                "iridescence_ior must be in [1, 2.333], got {v}"
                            )));
                        }
                        v as f32
                    },
                    iridescence_thickness_range: {
                        let (lo, hi) = o.iridescence_thickness_range.unwrap_or((100.0, 400.0));
                        [
                            non_neg(lo, "iridescence_thickness_range")?,
                            non_neg(hi, "iridescence_thickness_range")?,
                        ]
                    },
                    anisotropy: unit(o.anisotropy.unwrap_or(0.0), "anisotropy")?,
                    anisotropy_rotation: o.anisotropy_rotation.unwrap_or(0.0) as f32,
                    attenuation_color: hex(
                        o.attenuation_color.unwrap_or(0xffffff),
                        "attenuation_color",
                    )?,
                    attenuation_distance: match o.attenuation_distance {
                        None => f32::INFINITY,
                        Some(d) => non_neg(d, "attenuation_distance")?,
                    },
                    dispersion: non_neg(o.dispersion.unwrap_or(0.0), "dispersion")?,
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
                specular: hex(o.specular.unwrap_or(0x111111), "specular")?,
                shininess: non_neg(o.shininess.unwrap_or(30.0), "shininess")?,
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

/// `0xRRGGBB` → linear colour.
fn hex(v: u32, name: &str) -> PyResult<Color> {
    if v > 0xff_ffff {
        return Err(PyValueError::new_err(format!(
            "{name} must be 0xRRGGBB, got {v:#x}"
        )));
    }
    Ok(Color::from_hex(v))
}

fn non_neg(v: f64, name: &str) -> PyResult<f32> {
    if v.is_finite() && v >= 0.0 {
        Ok(v as f32)
    } else {
        Err(PyValueError::new_err(format!(
            "{name} must be a finite number >= 0, got {v}"
        )))
    }
}

fn unit(v: f64, name: &str) -> PyResult<f32> {
    if (0.0..=1.0).contains(&v) {
        Ok(v as f32)
    } else {
        Err(PyValueError::new_err(format!(
            "{name} must be in [0, 1], got {v}"
        )))
    }
}

fn intensity_arg(v: f64, what: &str) -> PyResult<f32> {
    non_neg(v, &format!("{what}: intensity"))
}

fn seg(v: u32, min: u32, name: &str) -> PyResult<usize> {
    if !(min..=MAX_SEGMENTS).contains(&v) {
        return Err(PyValueError::new_err(format!(
            "{name} must be {min}..={MAX_SEGMENTS}, got {v}"
        )));
    }
    Ok(v as usize)
}

fn finite_vec(x: f32, y: f32, z: f32, what: &str) -> PyResult<Vector3> {
    if ![x, y, z].iter().all(|v| v.is_finite()) {
        return Err(PyValueError::new_err(format!(
            "{what}: values must be finite"
        )));
    }
    Ok(Vector3::new(x, y, z))
}

/// A `Vector3` or an `(x, y, z)` sequence; `default` when `None`.
fn opt_vec(v: Option<&Bound<'_, PyAny>>, default: Vector3, what: &str) -> PyResult<Vector3> {
    let Some(v) = v else {
        return Ok(default);
    };
    let p = if let Ok(p) = v.extract::<PyVector3>() {
        p.inner
    } else if let Ok((x, y, z)) = v.extract::<(f32, f32, f32)>() {
        Vector3::new(x, y, z)
    } else {
        return Err(PyTypeError::new_err(format!(
            "{what} must be a Vector3 or an (x, y, z) sequence"
        )));
    };
    finite_vec(p.x, p.y, p.z, what)
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
        return Err("set_polygons: every polygon needs at least 1 triangle".into());
    }
    let total: u64 = counts.iter().map(|&n| n as u64).sum();
    if total != triangles as u64 {
        return Err(format!(
            "set_polygons: counts sum to {total} triangles, the geometry has {triangles}"
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

// ---- headless --------------------------------------------------------------

#[cfg(feature = "headless")]
#[pyclass(name = "HeadlessRenderer", unsendable)]
struct PyHeadlessRenderer {
    inner: threers::HeadlessRenderer,
    supersample: u32,
    shadows: bool,
}

#[cfg(feature = "headless")]
#[pymethods]
impl PyHeadlessRenderer {
    #[new]
    #[pyo3(signature = (width=640, height=480, supersample=1))]
    fn new(width: u32, height: u32, supersample: u32) -> PyResult<Self> {
        let supersample = supersample.max(1);
        let inner = build_headless(width, height, supersample)?;
        Ok(Self {
            inner,
            supersample,
            shadows: true,
        })
    }

    /// Render and return tightly packed RGBA8 bytes (`width * height * 4`).
    /// ``camera`` is a ``PerspectiveCamera`` or an ``OrthographicCamera``.
    fn render_rgba<'py>(
        &mut self,
        py: Python<'py>,
        scene: &mut PyScene,
        camera: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let rgba = self.render(scene, camera)?;
        Ok(PyBytes::new(py, &rgba))
    }

    /// Render and return a PNG as bytes.
    fn render_png<'py>(
        &mut self,
        py: Python<'py>,
        scene: &mut PyScene,
        camera: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let (w, h) = self.inner.render_size();
        let rgba = self.render(scene, camera)?;
        let png = threers::encode_png(w, h, &rgba);
        Ok(PyBytes::new(py, &png))
    }

    fn size(&self) -> (u32, u32) {
        self.inner.render_size()
    }

    /// Multisampling: ``4`` for 4× MSAA, ``1`` for none (the default) —
    /// smooth edges, and fur and hair strands resolved per sample.
    fn set_msaa(&mut self, samples: u32) {
        self.inner.set_msaa(samples);
    }

    /// Restrict drawing to a rectangle, in output pixels with a bottom-left
    /// origin (three.js ``setViewport``).
    fn set_viewport(&mut self, x: f32, y: f32, width: f32, height: f32) {
        self.inner.renderer().set_viewport(x, y, width, height);
    }

    /// Scissor rectangle, same units as ``set_viewport``; only applies while
    /// ``set_scissor_test(True)``.
    fn set_scissor(&mut self, x: f32, y: f32, width: f32, height: f32) {
        self.inner.renderer().set_scissor(x, y, width, height);
    }

    fn set_scissor_test(&mut self, enabled: bool) {
        self.inner.renderer().set_scissor_test(enabled);
    }

    /// Render a CAD-style quad view — top, front and side orthographic panes
    /// framed on the scene's meshes, plus ``camera`` as the perspective pane —
    /// at the configured output size (supersampling is resolved). Returns PNG
    /// bytes, or RGBA8 with ``format="rgba"``. ``maximized`` shows one pane
    /// full-size: ``"top"``, ``"front"``, ``"side"`` or ``"perspective"``.
    ///
    /// ``layout`` orders the panes in the grid (top-left, top-right,
    /// bottom-left, bottom-right; default ``["top", "perspective", "front",
    /// "side"]``); ``border`` pads the image edge; ``backgrounds`` maps pane
    /// names to ``0xRRGGBB`` colours overriding the scene background;
    /// ``margin`` pads the framed scene in the orthographic panes (fraction
    /// of its size, default 0.1).
    #[pyo3(signature = (scene, camera, maximized=None, format="png", gap=None, gap_color=None, border=None, layout=None, backgrounds=None, margin=None))]
    #[allow(clippy::too_many_arguments)]
    fn render_quad_view<'py>(
        &mut self,
        py: Python<'py>,
        scene: &mut PyScene,
        camera: &PyPerspectiveCamera,
        maximized: Option<&str>,
        format: &str,
        gap: Option<f32>,
        gap_color: Option<u32>,
        border: Option<f32>,
        layout: Option<Vec<String>>,
        backgrounds: Option<std::collections::HashMap<String, u32>>,
        margin: Option<f32>,
    ) -> PyResult<Bound<'py, PyBytes>> {
        use threers::helpers::{Pane, QuadView};
        let mut quad = QuadView::new();
        quad.maximized = match maximized {
            None => None,
            Some(name) => Some(Pane::from_name(name).ok_or_else(|| {
                PyValueError::new_err(format!(
                    "unknown pane '{name}' (top, front, side, perspective)"
                ))
            })?),
        };
        let png = match format {
            "png" => true,
            "rgba" => false,
            other => {
                return Err(PyValueError::new_err(format!(
                    "format must be 'png' or 'rgba', got '{other}'"
                )))
            }
        };
        if let Some(gap) = gap {
            if !(gap.is_finite() && gap >= 0.0) {
                return Err(PyValueError::new_err(format!(
                    "gap must be >= 0, got {gap}"
                )));
            }
            quad.gap = gap;
        }
        if let Some(hex) = gap_color {
            quad.gap_color = Some(Color::from_hex(hex));
        }
        if let Some(border) = border {
            if !(border.is_finite() && border >= 0.0) {
                return Err(PyValueError::new_err(format!(
                    "border must be >= 0, got {border}"
                )));
            }
            quad.border = border;
        }
        if let Some(margin) = margin {
            if !(margin.is_finite() && margin >= 0.0) {
                return Err(PyValueError::new_err(format!(
                    "margin must be >= 0, got {margin}"
                )));
            }
            quad.margin = margin;
        }
        if let Some(layout) = &layout {
            quad.layout = parse_layout(layout).map_err(PyValueError::new_err)?;
        }
        for (name, v) in backgrounds.unwrap_or_default() {
            let pane = Pane::from_name(&name).ok_or_else(|| {
                PyValueError::new_err(format!(
                    "backgrounds: unknown pane '{name}' (top, front, side, perspective)"
                ))
            })?;
            quad.backgrounds[pane.index()] =
                Some(hex(v, &format!("backgrounds['{}']", pane.name()))?);
        }
        quad.frame_scene(&mut scene.inner);
        let shadows = self.shadows;
        with_shadows(&mut scene.inner, shadows, |s| {
            quad.render(&mut self.inner, s, &camera.inner)
        });
        let rgba = self.inner.read_rgba_resolved();
        if !png {
            return Ok(PyBytes::new(py, &rgba));
        }
        let cfg = self.inner.config();
        Ok(PyBytes::new(
            py,
            &threers::encode_png(cfg.width, cfg.height, &rgba),
        ))
    }

    /// Resize the output (three.js ``setSize``). Rebuilds the render target,
    /// so the viewport and scissor reset; tone mapping and the shadow switch
    /// carry over.
    fn set_size(&mut self, width: u32, height: u32) -> PyResult<()> {
        if width == 0 || height == 0 || width > 16384 || height > 16384 {
            return Err(PyValueError::new_err(format!(
                "set_size: width and height must be 1..=16384, got {width}x{height}"
            )));
        }
        let (mode, exposure) = self.inner.renderer().tone_mapping();
        self.inner = build_headless(width, height, self.supersample)?;
        self.inner.renderer().set_tone_mapping(mode, exposure);
        Ok(())
    }

    /// ``mode``: ``"none"`` (default), ``"linear"``, ``"aces"``,
    /// ``"neutral"`` or ``"agx"``.
    #[pyo3(signature = (mode, exposure=1.0))]
    fn set_tone_mapping(&mut self, mode: &str, exposure: f64) -> PyResult<()> {
        let mode = parse_tone_mapping(mode).map_err(PyValueError::new_err)?;
        let exposure = non_neg(exposure, "set_tone_mapping: exposure")?;
        self.inner.renderer().set_tone_mapping(mode, exposure);
        Ok(())
    }

    /// ``(mode, exposure)``.
    fn tone_mapping(&mut self) -> (&'static str, f32) {
        let (mode, exposure) = self.inner.renderer().tone_mapping();
        (tone_mapping_name(mode), exposure)
    }

    /// Master shadow switch (three.js ``shadowMap.enabled``), default on.
    /// Shadows still need a light with ``cast_shadow`` and meshes opted in
    /// with ``Scene.set_cast_shadow`` / ``set_receive_shadow``.
    fn set_shadows_enabled(&mut self, enabled: bool) {
        self.shadows = enabled;
    }

    fn shadows_enabled(&self) -> bool {
        self.shadows
    }

    /// Undo ``set_viewport``: draw to the whole output again.
    fn reset_viewport(&mut self) {
        self.inner.renderer().reset_viewport();
    }

    /// three.js ``autoClear`` (default on). Off, each render draws over what
    /// is already there — the background colour included, as in three.js —
    /// so several ``set_viewport`` + render calls compose one image (call
    /// ``clear`` first).
    fn set_auto_clear(&mut self, enabled: bool) {
        self.inner.renderer().set_auto_clear(enabled);
    }

    fn auto_clear(&mut self) -> bool {
        self.inner.renderer().auto_clear()
    }

    /// Fill the output with ``color`` (``0xRRGGBB``) at ``alpha`` — only the
    /// scissor rectangle while the scissor test is on.
    #[pyo3(signature = (color=0x000000, alpha=1.0))]
    fn clear(&mut self, color: u32, alpha: f64) -> PyResult<()> {
        let c = hex(color, "clear: color")?;
        let a = unit(alpha, "clear: alpha")?;
        self.inner.clear(c, a);
        Ok(())
    }
}

#[cfg(feature = "headless")]
impl PyHeadlessRenderer {
    fn render(&mut self, scene: &mut PyScene, camera: &Bound<'_, PyAny>) -> PyResult<Vec<u8>> {
        let shadows = self.shadows;
        let inner = &mut self.inner;
        with_camera(camera, |cam| {
            with_shadows(&mut scene.inner, shadows, |s| inner.render_to_rgba(s, cam))
        })
    }
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
fn build_headless(
    width: u32,
    height: u32,
    supersample: u32,
) -> PyResult<threers::HeadlessRenderer> {
    threers::HeadlessRenderer::builder()
        .size(width, height)
        .supersample(supersample)
        .build()
        .map_err(PyRuntimeError::new_err)
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

// ---- animation -------------------------------------------------------------

#[cfg(feature = "animation")]
#[pyclass(name = "Tween", unsendable)]
struct PyTween {
    inner: threers_animation::tween::Tween<f32>,
}

#[cfg(feature = "animation")]
#[pymethods]
impl PyTween {
    #[new]
    #[pyo3(signature = (start=0.0, end=1.0, duration=1.0, easing="cubicOut"))]
    fn new(start: f32, end: f32, duration: f32, easing: &str) -> PyResult<Self> {
        use threers_animation::tween::Tween;
        let e = parse_easing(easing)?;
        Ok(Self {
            inner: Tween::new(start, end, duration).easing(e),
        })
    }

    fn update(&mut self, dt: f32) -> f32 {
        self.inner.update(dt)
    }

    fn value(&self) -> f32 {
        self.inner.value()
    }

    fn is_finished(&self) -> bool {
        self.inner.is_finished()
    }
}

#[cfg(feature = "animation")]
fn parse_easing(name: &str) -> PyResult<threers_animation::easing::Easing> {
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
            return Err(PyValueError::new_err(format!(
                "unknown easing '{other}' (try linear, cubicOut, cubicInOut, …)"
            )))
        }
    })
}

// ---- physics ---------------------------------------------------------------

#[cfg(feature = "physics")]
#[pyclass(name = "PhysicsWorld", unsendable)]
struct PyPhysicsWorld {
    inner: threers_physics::prelude::World,
    bodies: Vec<threers_physics::prelude::BodyId>,
}

#[cfg(feature = "physics")]
#[pymethods]
impl PyPhysicsWorld {
    #[new]
    fn new() -> Self {
        use threers_physics::prelude::*;
        let mut world = World::new();
        world.add_body(RigidBody::fixed().shape(Shape::ground()));
        Self {
            inner: world,
            bodies: Vec::new(),
        }
    }

    /// Add a dynamic ball. Returns a stable handle index.
    #[pyo3(signature = (radius=0.5, x=0.0, y=5.0, z=0.0, restitution=0.4))]
    fn add_ball(&mut self, radius: f32, x: f32, y: f32, z: f32, restitution: f32) -> usize {
        use threers_physics::prelude::*;
        let id = self.inner.add_body(
            RigidBody::dynamic()
                .shape(Shape::ball(radius))
                .translation(Vector3::new(x, y, z))
                .restitution(restitution),
        );
        self.bodies.push(id);
        self.bodies.len() - 1
    }

    fn step(&mut self, dt: f32) {
        self.inner.step(dt);
    }

    fn body_translation(&self, handle: usize) -> PyResult<PyVector3> {
        let id = *self
            .bodies
            .get(handle)
            .ok_or_else(|| PyValueError::new_err("bad body handle"))?;
        let body = self
            .inner
            .body(id)
            .ok_or_else(|| PyValueError::new_err("body removed"))?;
        Ok(PyVector3 {
            inner: body.translation(),
        })
    }

    fn body_count(&self) -> usize {
        self.bodies.len()
    }
}

// ---- module ----------------------------------------------------------------

/// Native extension module. Imported as `threers._native` from the Python package.
#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_class::<PyVector3>()?;
    m.add_class::<PyColor>()?;
    m.add_class::<PyPerspectiveCamera>()?;
    m.add_class::<PyOrthographicCamera>()?;
    m.add_class::<PyScene>()?;
    m.add_class::<PyGeometry>()?;
    m.add_class::<PyMaterial>()?;
    m.add_class::<PyFurMaterial>()?;
    m.add_class::<PyTexture>()?;

    #[cfg(feature = "headless")]
    m.add_class::<PyHeadlessRenderer>()?;

    #[cfg(feature = "animation")]
    m.add_class::<PyTween>()?;

    #[cfg(feature = "physics")]
    m.add_class::<PyPhysicsWorld>()?;

    Ok(())
}
