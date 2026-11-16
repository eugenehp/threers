//! `threers.FemModel` and `threers.FemSurface`: finite and boundary element
//! analysis from threers-fem, through its box-region `flat` API.
//!
//! ```python
//! import threers
//! m = threers.FemModel.shape("bracket", 4.0)
//! m.set_material(200e3, 0.3, 7.85e-9)          # steel, mm / N / MPa
//! m.fix_box((-1, -1, -1), (0, 51, 31))
//! m.force_box((60, -1, -1), (61, 7, 31), (0, -500, 0))
//! r = m.solve()
//! print(r.max_von_mises, r.max_displacement)
//! ```

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use threers_fem::flat;

fn err(e: threers_fem::FemError) -> PyErr {
    PyValueError::new_err(e.to_string())
}

/// A meshed part with its material, supports and loads.
#[pyclass(name = "FemModel", unsendable)]
pub struct PyFemModel(flat::Model);

#[pymethods]
impl PyFemModel {
    /// The built-in shapes: `[(name, description), …]`.
    #[staticmethod]
    fn shapes() -> Vec<(&'static str, &'static str)> {
        flat::SHAPES.to_vec()
    }

    /// A built-in shape meshed at about `spacing`.
    #[staticmethod]
    fn shape(name: &str, spacing: f64) -> PyResult<Self> {
        flat::Model::shape(name, spacing).map(Self).map_err(err)
    }

    /// Mesh a closed triangle surface: flat `positions` (x, y, z per vertex)
    /// and `indices` (three per triangle).
    #[staticmethod]
    fn from_triangles(positions: Vec<f64>, indices: Vec<u32>, spacing: f64) -> PyResult<Self> {
        flat::Model::from_triangles(&positions, &indices, spacing)
            .map(Self)
            .map_err(err)
    }

    /// An exact box grid from `lo` to `hi` with `n` cubes per axis.
    #[staticmethod]
    fn box_grid(lo: [f64; 3], hi: [f64; 3], n: [usize; 3]) -> Self {
        Self(flat::Model::box_grid(lo, hi, n))
    }

    #[getter]
    fn node_count(&self) -> usize {
        self.0.node_count()
    }

    #[getter]
    fn tet_count(&self) -> usize {
        self.0.tet_count()
    }

    #[getter]
    fn volume(&self) -> f64 {
        self.0.volume()
    }

    /// `(min x, min y, min z, max x, max y, max z)`.
    fn bounds(&self) -> [f64; 6] {
        self.0.bounds()
    }

    /// Smallest and largest dihedral angle, degrees.
    fn dihedral_range(&self) -> [f64; 2] {
        self.0.dihedral_range()
    }

    /// x, y, z per node, flat.
    fn positions(&self) -> Vec<f64> {
        self.0.positions()
    }

    /// Boundary triangles as node indices, flat, wound outward.
    fn surface_indices(&self) -> Vec<u32> {
        self.0.surface_indices().to_vec()
    }

    #[pyo3(signature = (youngs, poisson, density=1.0))]
    fn set_material(&mut self, youngs: f64, poisson: f64, density: f64) {
        self.0.set_material(youngs, poisson, density);
    }

    /// Ten-node elements (default) or four-node.
    fn set_quadratic(&mut self, quadratic: bool) {
        self.0.set_quadratic(quadratic);
    }

    /// Remove every support, load and thermal condition.
    fn clear(&mut self) {
        self.0.clear();
    }

    /// Clamp every node inside the box.
    fn fix_box(&mut self, lo: [f64; 3], hi: [f64; 3]) {
        self.0.fix_box(lo, hi);
    }

    /// Hold displacement component `axis` (0, 1, 2) at `value` in the box.
    #[pyo3(signature = (lo, hi, axis, value=0.0))]
    fn hold_box(&mut self, lo: [f64; 3], hi: [f64; 3], axis: usize, value: f64) {
        self.0.hold_box(lo, hi, axis, value);
    }

    /// A total force spread by area over the boundary faces in the box.
    fn force_box(&mut self, lo: [f64; 3], hi: [f64; 3], force: [f64; 3]) {
        self.0.force_box(lo, hi, force);
    }

    /// Inward pressure on the boundary faces in the box.
    fn pressure_box(&mut self, lo: [f64; 3], hi: [f64; 3], pressure: f64) {
        self.0.pressure_box(lo, hi, pressure);
    }

    fn set_gravity(&mut self, g: [f64; 3]) {
        self.0.set_gravity(g);
    }

    fn temperature_box(&mut self, lo: [f64; 3], hi: [f64; 3], t: f64) {
        self.0.temperature_box(lo, hi, t);
    }

    fn heat_flux_box(&mut self, lo: [f64; 3], hi: [f64; 3], q: f64) {
        self.0.heat_flux_box(lo, hi, q);
    }

    fn convection_box(&mut self, lo: [f64; 3], hi: [f64; 3], h: f64, ambient: f64) {
        self.0.convection_box(lo, hi, h, ambient);
    }

    fn set_heat_source(&mut self, q: f64) {
        self.0.set_heat_source(q);
    }

    /// Static solve. Releases the GIL while it runs.
    fn solve(&self, py: Python<'_>) -> PyResult<PyFemStatic> {
        let m = &self.0;
        py.detach(|| m.solve()).map(PyFemStatic).map_err(err)
    }

    /// The lowest `count` natural frequencies and mode shapes.
    #[pyo3(signature = (count=4))]
    fn modes(&self, py: Python<'_>, count: usize) -> PyResult<PyFemModes> {
        let m = &self.0;
        py.detach(|| m.modes(count)).map(PyFemModes).map_err(err)
    }

    /// Steady conduction with conductivity `k`.
    fn solve_heat(&self, py: Python<'_>, k: f64) -> PyResult<PyFemHeat> {
        let m = &self.0;
        py.detach(|| m.solve_heat(k)).map(PyFemHeat).map_err(err)
    }

    fn __repr__(&self) -> String {
        format!("FemModel({} nodes, {} tetrahedra)", self.0.node_count(), self.0.tet_count())
    }
}

/// A static solution, per node.
#[pyclass(name = "FemStatic", unsendable)]
pub struct PyFemStatic(flat::StaticResult);

#[pymethods]
impl PyFemStatic {
    /// x, y, z per node, flat.
    #[getter]
    fn displacement(&self) -> Vec<f64> {
        self.0.displacement.clone()
    }

    #[getter]
    fn von_mises(&self) -> Vec<f64> {
        self.0.von_mises.clone()
    }

    /// Voigt xx, yy, zz, yz, xz, xy per node, flat.
    #[getter]
    fn stress(&self) -> Vec<f64> {
        self.0.stress.clone()
    }

    #[getter]
    fn max_von_mises(&self) -> f64 {
        self.0.max_von_mises
    }

    #[getter]
    fn max_displacement(&self) -> f64 {
        self.0.max_displacement
    }

    #[getter]
    fn reaction(&self) -> [f64; 3] {
        self.0.reaction
    }

    #[getter]
    fn compliance(&self) -> f64 {
        self.0.compliance
    }
}

/// Natural frequencies and mode shapes.
#[pyclass(name = "FemModes", unsendable)]
pub struct PyFemModes(flat::ModalResult);

#[pymethods]
impl PyFemModes {
    #[getter]
    fn frequencies(&self) -> Vec<f64> {
        self.0.frequencies.clone()
    }

    /// Mode `i`'s shape, x, y, z per node, flat; largest displacement 1.
    fn shape(&self, i: usize) -> PyResult<Vec<f64>> {
        self.0
            .shapes
            .get(i)
            .cloned()
            .ok_or_else(|| PyValueError::new_err("no such mode"))
    }

    fn __len__(&self) -> usize {
        self.0.frequencies.len()
    }
}

/// A temperature field, per node.
#[pyclass(name = "FemHeat", unsendable)]
pub struct PyFemHeat(flat::HeatResult);

#[pymethods]
impl PyFemHeat {
    #[getter]
    fn temperature(&self) -> Vec<f64> {
        self.0.temperature.clone()
    }

    /// x, y, z heat flux per node, flat.
    #[getter]
    fn flux(&self) -> Vec<f64> {
        self.0.flux.clone()
    }

    #[getter]
    fn min(&self) -> f64 {
        self.0.min
    }

    #[getter]
    fn max(&self) -> f64 {
        self.0.max
    }
}

/// A closed surface for the boundary element method.
#[pyclass(name = "FemSurface", unsendable)]
pub struct PyFemSurface(flat::SurfaceModel);

#[pymethods]
impl PyFemSurface {
    /// A box `size` across, centred on the origin.
    #[staticmethod]
    fn cuboid(size: [f64; 3], segments_per_unit: f64) -> PyResult<Self> {
        flat::SurfaceModel::cuboid(size, segments_per_unit)
            .map(Self)
            .map_err(err)
    }

    /// An icosphere with `20·(detail + 1)²` triangles.
    #[staticmethod]
    fn sphere(radius: f64, detail: usize) -> PyResult<Self> {
        flat::SurfaceModel::sphere(radius, detail).map(Self).map_err(err)
    }

    #[staticmethod]
    fn from_triangles(positions: Vec<f64>, indices: Vec<u32>) -> PyResult<Self> {
        flat::SurfaceModel::from_triangles(&positions, &indices)
            .map(Self)
            .map_err(err)
    }

    #[getter]
    fn triangle_count(&self) -> usize {
        self.0.triangle_count()
    }

    fn positions(&self) -> Vec<f64> {
        self.0.positions()
    }

    fn indices(&self) -> Vec<u32> {
        self.0.indices()
    }

    fn centroids(&self) -> Vec<f64> {
        self.0.centroids()
    }

    fn normals(&self) -> Vec<f64> {
        self.0.normals()
    }

    /// Capacitance in units of the permittivity.
    fn capacitance(&self, py: Python<'_>) -> PyResult<f64> {
        let s = &self.0;
        py.detach(|| s.capacitance()).map_err(err)
    }

    /// Per-triangle normal velocity for a piston inside the box, facing
    /// `facing`.
    fn piston(&self, lo: [f64; 3], hi: [f64; 3], facing: [f64; 3], velocity: f64) -> Vec<f64> {
        self.0.piston(lo, hi, facing, velocity)
    }

    /// Radiate at `frequency` from per-triangle normal velocity
    /// (`velocity_im` defaults to zero).
    #[pyo3(signature = (frequency, velocity_re, velocity_im=None, sound_speed=343.0, density=1.204))]
    fn radiate(
        &self,
        py: Python<'_>,
        frequency: f64,
        velocity_re: Vec<f64>,
        velocity_im: Option<Vec<f64>>,
        sound_speed: f64,
        density: f64,
    ) -> PyResult<PyFemAcoustic> {
        let im = velocity_im.unwrap_or_else(|| vec![0.0; velocity_re.len()]);
        let s = &self.0;
        py.detach(|| s.radiate(frequency, sound_speed, density, &velocity_re, &im))
            .map(PyFemAcoustic)
            .map_err(err)
    }
}

/// A radiated sound field.
#[pyclass(name = "FemAcoustic", unsendable)]
pub struct PyFemAcoustic(flat::AcousticResult);

#[pymethods]
impl PyFemAcoustic {
    #[getter]
    fn power(&self) -> f64 {
        self.0.power
    }

    #[getter]
    fn elements_per_wavelength(&self) -> f64 {
        self.0.elements_per_wavelength
    }

    /// Complex surface pressure per triangle.
    #[getter]
    fn pressure(&self) -> Vec<num_complex_shim::C> {
        self.0
            .pressure_re
            .iter()
            .zip(&self.0.pressure_im)
            .map(|(r, i)| num_complex_shim::C(*r, *i))
            .collect()
    }

    /// Complex pressure at each `(x, y, z)` point.
    fn field(&self, py: Python<'_>, points: Vec<[f64; 3]>) -> Vec<num_complex_shim::C> {
        let flat: Vec<f64> = points.iter().flatten().copied().collect();
        let s = &self.0;
        let (re, im) = py.detach(|| s.field(&flat));
        re.into_iter().zip(im).map(|(r, i)| num_complex_shim::C(r, i)).collect()
    }

    /// Far-field amplitude `|F|` per direction, with `p ≈ F e^{ikr}/r`.
    fn far_field(&self, directions: Vec<[f64; 3]>) -> Vec<f64> {
        let flat: Vec<f64> = directions.iter().flatten().copied().collect();
        self.0.far_field(&flat)
    }
}

/// Python `complex` out of a pair, without a num-complex dependency.
mod num_complex_shim {
    use pyo3::prelude::*;
    use pyo3::types::PyComplex;

    pub struct C(pub f64, pub f64);

    impl<'py> IntoPyObject<'py> for C {
        type Target = PyComplex;
        type Output = Bound<'py, PyComplex>;
        type Error = std::convert::Infallible;
        fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
            Ok(PyComplex::from_doubles(py, self.0, self.1))
        }
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyFemModel>()?;
    m.add_class::<PyFemStatic>()?;
    m.add_class::<PyFemModes>()?;
    m.add_class::<PyFemHeat>()?;
    m.add_class::<PyFemSurface>()?;
    m.add_class::<PyFemAcoustic>()?;
    Ok(())
}
