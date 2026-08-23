//! Python bindings for threers (`pip install threers` via maturin).
//!
//! ```bash
//! cd crates/threers-py
//! maturin develop --features headless,physics,animation
//! python -c "import threers; print(threers.__version__)"
//! ```

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyBytes;

use threers::cameras::PerspectiveCamera;
use threers::core::{Mesh, Object3D, ObjectId};
use threers::geometries::{BoxGeometry, SphereGeometry};
use threers::materials::{BasicMaterial, Material, StandardMaterial};
use threers::math::{Color, Vector3};
use threers::scene::Scene;

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

#[pyclass(name = "Scene", unsendable)]
struct PyScene {
    inner: Scene,
    handles: Vec<ObjectId>,
}

#[pymethods]
impl PyScene {
    #[new]
    fn new() -> Self {
        Self {
            inner: Scene::new(),
            handles: Vec::new(),
        }
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

    fn object_count(&self) -> usize {
        self.handles.len()
    }
}

// ---- headless --------------------------------------------------------------

#[cfg(feature = "headless")]
#[pyclass(name = "HeadlessRenderer", unsendable)]
struct PyHeadlessRenderer {
    inner: threers::HeadlessRenderer,
}

#[cfg(feature = "headless")]
#[pymethods]
impl PyHeadlessRenderer {
    #[new]
    #[pyo3(signature = (width=640, height=480, supersample=1))]
    fn new(width: u32, height: u32, supersample: u32) -> PyResult<Self> {
        let inner = threers::HeadlessRenderer::builder()
            .size(width, height)
            .supersample(supersample.max(1))
            .build()
            .map_err(PyRuntimeError::new_err)?;
        Ok(Self { inner })
    }

    /// Render and return tightly packed RGBA8 bytes (`width * height * 4`).
    fn render_rgba<'py>(
        &mut self,
        py: Python<'py>,
        scene: &mut PyScene,
        camera: &PyPerspectiveCamera,
    ) -> Bound<'py, PyBytes> {
        let rgba = self
            .inner
            .render_to_rgba(&mut scene.inner, &camera.inner);
        PyBytes::new(py, &rgba)
    }

    /// Render and return a PNG as bytes.
    fn render_png<'py>(
        &mut self,
        py: Python<'py>,
        scene: &mut PyScene,
        camera: &PyPerspectiveCamera,
    ) -> Bound<'py, PyBytes> {
        let (w, h) = self.inner.render_size();
        let rgba = self
            .inner
            .render_to_rgba(&mut scene.inner, &camera.inner);
        let png = threers::encode_png(w, h, &rgba);
        PyBytes::new(py, &png)
    }

    fn size(&self) -> (u32, u32) {
        self.inner.render_size()
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
    m.add_class::<PyScene>()?;

    #[cfg(feature = "headless")]
    m.add_class::<PyHeadlessRenderer>()?;

    #[cfg(feature = "animation")]
    m.add_class::<PyTween>()?;

    #[cfg(feature = "physics")]
    m.add_class::<PyPhysicsWorld>()?;

    Ok(())
}
