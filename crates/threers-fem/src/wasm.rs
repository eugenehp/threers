//! JavaScript bindings, through wasm-bindgen: the [`flat`](crate::flat) API
//! with camelCase names and typed arrays.
//!
//! ```js
//! import init, { FemModel } from "./pkg-fem/threers_fem.js";
//! await init();
//! const m = FemModel.shape("bracket", 3);
//! m.setMaterial(200e3, 0.3, 7.85e-9);           // steel in mm, N, MPa
//! m.fixBox([-1, -1, -1], [0, 51, 31]);
//! m.forceBox([60, -1, -1], [61, 7, 31], [0, -500, 0]);
//! const r = m.solve();
//! console.log(r.maxVonMises, r.maxDisplacement);
//! ```
//!
//! Points and boxes are `[x, y, z]` arrays (any array-like of numbers);
//! per-node results are `Float64Array`s, three entries per node for vectors.

use crate::flat;
use wasm_bindgen::prelude::*;

fn v3(a: &[f64]) -> Result<[f64; 3], JsError> {
    match a {
        [x, y, z] => Ok([*x, *y, *z]),
        _ => Err(JsError::new("expected [x, y, z]")),
    }
}

fn err(e: crate::FemError) -> JsError {
    JsError::new(&e.to_string())
}

/// The built-in shapes, as `name: description` lines.
#[wasm_bindgen(js_name = femShapes)]
pub fn fem_shapes() -> Vec<String> {
    flat::SHAPES.iter().map(|(n, d)| format!("{n}: {d}")).collect()
}

/// A meshed part with its material, supports and loads.
#[wasm_bindgen]
pub struct FemModel(flat::Model);

#[wasm_bindgen]
impl FemModel {
    /// A built-in shape — see `femShapes()` — meshed at about `spacing`.
    pub fn shape(name: &str, spacing: f64) -> Result<FemModel, JsError> {
        flat::Model::shape(name, spacing).map(Self).map_err(err)
    }

    /// Mesh a closed triangle surface (`positions` x, y, z per vertex).
    #[wasm_bindgen(js_name = fromTriangles)]
    pub fn from_triangles(positions: &[f64], indices: &[u32], spacing: f64) -> Result<FemModel, JsError> {
        flat::Model::from_triangles(positions, indices, spacing).map(Self).map_err(err)
    }

    /// An exact box grid, `n` cubes per axis.
    #[wasm_bindgen(js_name = boxGrid)]
    pub fn box_grid(lo: &[f64], hi: &[f64], nx: usize, ny: usize, nz: usize) -> Result<FemModel, JsError> {
        Ok(Self(flat::Model::box_grid(v3(lo)?, v3(hi)?, [nx, ny, nz])))
    }

    #[wasm_bindgen(getter, js_name = nodeCount)]
    pub fn node_count(&self) -> usize {
        self.0.node_count()
    }

    #[wasm_bindgen(getter, js_name = tetCount)]
    pub fn tet_count(&self) -> usize {
        self.0.tet_count()
    }

    #[wasm_bindgen(getter)]
    pub fn volume(&self) -> f64 {
        self.0.volume()
    }

    /// `[min x, min y, min z, max x, max y, max z]`.
    pub fn bounds(&self) -> Vec<f64> {
        self.0.bounds().to_vec()
    }

    /// Smallest and largest dihedral angle, degrees.
    #[wasm_bindgen(js_name = dihedralRange)]
    pub fn dihedral_range(&self) -> Vec<f64> {
        self.0.dihedral_range().to_vec()
    }

    /// x, y, z per node.
    pub fn positions(&self) -> Vec<f64> {
        self.0.positions()
    }

    /// Boundary triangles as node indices, wound outward.
    #[wasm_bindgen(js_name = surfaceIndices)]
    pub fn surface_indices(&self) -> Vec<u32> {
        self.0.surface_indices().to_vec()
    }

    #[wasm_bindgen(js_name = setMaterial)]
    pub fn set_material(&mut self, youngs: f64, poisson: f64, density: f64) {
        self.0.set_material(youngs, poisson, density);
    }

    /// Ten-node elements (default) or four-node.
    #[wasm_bindgen(js_name = setQuadratic)]
    pub fn set_quadratic(&mut self, quadratic: bool) {
        self.0.set_quadratic(quadratic);
    }

    pub fn clear(&mut self) {
        self.0.clear();
    }

    #[wasm_bindgen(js_name = fixBox)]
    pub fn fix_box(&mut self, lo: &[f64], hi: &[f64]) -> Result<(), JsError> {
        self.0.fix_box(v3(lo)?, v3(hi)?);
        Ok(())
    }

    /// Hold one displacement component (axis 0, 1, 2) at `value` in the box.
    #[wasm_bindgen(js_name = holdBox)]
    pub fn hold_box(&mut self, lo: &[f64], hi: &[f64], axis: usize, value: f64) -> Result<(), JsError> {
        self.0.hold_box(v3(lo)?, v3(hi)?, axis, value);
        Ok(())
    }

    #[wasm_bindgen(js_name = forceBox)]
    pub fn force_box(&mut self, lo: &[f64], hi: &[f64], force: &[f64]) -> Result<(), JsError> {
        self.0.force_box(v3(lo)?, v3(hi)?, v3(force)?);
        Ok(())
    }

    #[wasm_bindgen(js_name = pressureBox)]
    pub fn pressure_box(&mut self, lo: &[f64], hi: &[f64], pressure: f64) -> Result<(), JsError> {
        self.0.pressure_box(v3(lo)?, v3(hi)?, pressure);
        Ok(())
    }

    #[wasm_bindgen(js_name = setGravity)]
    pub fn set_gravity(&mut self, g: &[f64]) -> Result<(), JsError> {
        self.0.set_gravity(v3(g)?);
        Ok(())
    }

    #[wasm_bindgen(js_name = temperatureBox)]
    pub fn temperature_box(&mut self, lo: &[f64], hi: &[f64], t: f64) -> Result<(), JsError> {
        self.0.temperature_box(v3(lo)?, v3(hi)?, t);
        Ok(())
    }

    #[wasm_bindgen(js_name = heatFluxBox)]
    pub fn heat_flux_box(&mut self, lo: &[f64], hi: &[f64], q: f64) -> Result<(), JsError> {
        self.0.heat_flux_box(v3(lo)?, v3(hi)?, q);
        Ok(())
    }

    #[wasm_bindgen(js_name = convectionBox)]
    pub fn convection_box(&mut self, lo: &[f64], hi: &[f64], h: f64, ambient: f64) -> Result<(), JsError> {
        self.0.convection_box(v3(lo)?, v3(hi)?, h, ambient);
        Ok(())
    }

    #[wasm_bindgen(js_name = setHeatSource)]
    pub fn set_heat_source(&mut self, q: f64) {
        self.0.set_heat_source(q);
    }

    /// Static solve: displacements and stresses.
    pub fn solve(&self) -> Result<FemStatic, JsError> {
        self.0.solve().map(FemStatic).map_err(err)
    }

    /// The lowest `count` natural frequencies and mode shapes.
    pub fn modes(&self, count: usize) -> Result<FemModes, JsError> {
        self.0.modes(count).map(FemModes).map_err(err)
    }

    /// Steady conduction with conductivity `k`.
    #[wasm_bindgen(js_name = solveHeat)]
    pub fn solve_heat(&self, k: f64) -> Result<FemHeat, JsError> {
        self.0.solve_heat(k).map(FemHeat).map_err(err)
    }
}

/// A static solution, per node.
#[wasm_bindgen]
pub struct FemStatic(flat::StaticResult);

#[wasm_bindgen]
impl FemStatic {
    /// x, y, z per node.
    pub fn displacement(&self) -> Vec<f64> {
        self.0.displacement.clone()
    }

    #[wasm_bindgen(js_name = vonMises)]
    pub fn von_mises(&self) -> Vec<f64> {
        self.0.von_mises.clone()
    }

    /// Voigt xx, yy, zz, yz, xz, xy per node.
    pub fn stress(&self) -> Vec<f64> {
        self.0.stress.clone()
    }

    #[wasm_bindgen(getter, js_name = maxVonMises)]
    pub fn max_von_mises(&self) -> f64 {
        self.0.max_von_mises
    }

    #[wasm_bindgen(getter, js_name = maxDisplacement)]
    pub fn max_displacement(&self) -> f64 {
        self.0.max_displacement
    }

    /// Total support reaction, `[x, y, z]`.
    pub fn reaction(&self) -> Vec<f64> {
        self.0.reaction.to_vec()
    }

    #[wasm_bindgen(getter)]
    pub fn compliance(&self) -> f64 {
        self.0.compliance
    }
}

/// Natural frequencies and mode shapes.
#[wasm_bindgen]
pub struct FemModes(flat::ModalResult);

#[wasm_bindgen]
impl FemModes {
    pub fn frequencies(&self) -> Vec<f64> {
        self.0.frequencies.clone()
    }

    /// Mode `i`'s shape, x, y, z per node, largest displacement 1.
    pub fn shape(&self, i: usize) -> Result<Vec<f64>, JsError> {
        self.0
            .shapes
            .get(i)
            .cloned()
            .ok_or_else(|| JsError::new("no such mode"))
    }

    #[wasm_bindgen(getter)]
    pub fn count(&self) -> usize {
        self.0.frequencies.len()
    }
}

/// A temperature field, per node.
#[wasm_bindgen]
pub struct FemHeat(flat::HeatResult);

#[wasm_bindgen]
impl FemHeat {
    pub fn temperature(&self) -> Vec<f64> {
        self.0.temperature.clone()
    }

    /// x, y, z heat flux per node.
    pub fn flux(&self) -> Vec<f64> {
        self.0.flux.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn min(&self) -> f64 {
        self.0.min
    }

    #[wasm_bindgen(getter)]
    pub fn max(&self) -> f64 {
        self.0.max
    }
}

/// A closed surface for the boundary element method.
#[wasm_bindgen]
pub struct FemSurface(flat::SurfaceModel);

#[wasm_bindgen]
impl FemSurface {
    /// A box `size` across, centred on the origin.
    pub fn cuboid(size: &[f64], segments_per_unit: f64) -> Result<FemSurface, JsError> {
        flat::SurfaceModel::cuboid(v3(size)?, segments_per_unit)
            .map(Self)
            .map_err(err)
    }

    /// An icosphere with `20·(detail + 1)²` triangles.
    pub fn sphere(radius: f64, detail: usize) -> Result<FemSurface, JsError> {
        flat::SurfaceModel::sphere(radius, detail).map(Self).map_err(err)
    }

    #[wasm_bindgen(js_name = fromTriangles)]
    pub fn from_triangles(positions: &[f64], indices: &[u32]) -> Result<FemSurface, JsError> {
        flat::SurfaceModel::from_triangles(positions, indices)
            .map(Self)
            .map_err(err)
    }

    #[wasm_bindgen(getter, js_name = triangleCount)]
    pub fn triangle_count(&self) -> usize {
        self.0.triangle_count()
    }

    pub fn positions(&self) -> Vec<f64> {
        self.0.positions()
    }

    pub fn indices(&self) -> Vec<u32> {
        self.0.indices()
    }

    pub fn centroids(&self) -> Vec<f64> {
        self.0.centroids()
    }

    pub fn normals(&self) -> Vec<f64> {
        self.0.normals()
    }

    /// Capacitance in units of the permittivity.
    pub fn capacitance(&self) -> Result<f64, JsError> {
        self.0.capacitance().map_err(err)
    }

    /// Per-triangle velocities for a piston inside the box, facing `facing`.
    pub fn piston(&self, lo: &[f64], hi: &[f64], facing: &[f64], velocity: f64) -> Result<Vec<f64>, JsError> {
        Ok(self.0.piston(v3(lo)?, v3(hi)?, v3(facing)?, velocity))
    }

    /// Radiate at `frequency` from per-triangle normal velocity
    /// `velocity_re + i·velocity_im`.
    pub fn radiate(
        &self,
        frequency: f64,
        sound_speed: f64,
        density: f64,
        velocity_re: &[f64],
        velocity_im: &[f64],
    ) -> Result<FemAcoustic, JsError> {
        self.0
            .radiate(frequency, sound_speed, density, velocity_re, velocity_im)
            .map(FemAcoustic)
            .map_err(err)
    }
}

/// A radiated sound field.
#[wasm_bindgen]
pub struct FemAcoustic(flat::AcousticResult);

#[wasm_bindgen]
impl FemAcoustic {
    #[wasm_bindgen(getter)]
    pub fn power(&self) -> f64 {
        self.0.power
    }

    #[wasm_bindgen(getter, js_name = elementsPerWavelength)]
    pub fn elements_per_wavelength(&self) -> f64 {
        self.0.elements_per_wavelength
    }

    #[wasm_bindgen(js_name = pressureRe)]
    pub fn pressure_re(&self) -> Vec<f64> {
        self.0.pressure_re.clone()
    }

    #[wasm_bindgen(js_name = pressureIm)]
    pub fn pressure_im(&self) -> Vec<f64> {
        self.0.pressure_im.clone()
    }

    /// Complex pressure at `points` (x, y, z each), interleaved re, im.
    pub fn field(&self, points: &[f64]) -> Vec<f64> {
        let (re, im) = self.0.field(points);
        re.into_iter().zip(im).flat_map(|(a, b)| [a, b]).collect()
    }

    /// Far-field amplitude per direction (x, y, z each).
    #[wasm_bindgen(js_name = farField)]
    pub fn far_field(&self, directions: &[f64]) -> Vec<f64> {
        self.0.far_field(directions)
    }
}
