//! `FemModel` and `FemSurface`: finite and boundary element analysis from
//! threers-fem, through its box-region `flat` API.
//!
//! ```js
//! const { FemModel } = require("threers-node");
//! const m = FemModel.shape("bracket", 4);
//! m.setMaterial(200e3, 0.3, 7.85e-9);           // steel, mm / N / MPa
//! m.fixBox([-1, -1, -1], [0, 51, 31]);
//! m.forceBox([60, -1, -1], [61, 7, 31], [0, -500, 0]);
//! const r = m.solve();
//! console.log(r.maxVonMises, r.maxDisplacement);
//! ```

use napi::bindgen_prelude::*;
use napi_derive::napi;
use threers_fem::flat;

fn err(e: threers_fem::FemError) -> Error {
    Error::new(Status::InvalidArg, e.to_string())
}

fn v3(a: &[f64]) -> Result<[f64; 3]> {
    match a {
        [x, y, z] => Ok([*x, *y, *z]),
        _ => Err(Error::new(Status::InvalidArg, "expected [x, y, z]")),
    }
}

/// The built-in shapes, as `name: description` lines.
#[napi(js_name = "femShapes")]
pub fn fem_shapes() -> Vec<String> {
    flat::SHAPES.iter().map(|(n, d)| format!("{n}: {d}")).collect()
}

/// A meshed part with its material, supports and loads.
#[napi(js_name = "FemModel")]
pub struct JsFemModel(flat::Model);

#[napi]
impl JsFemModel {
    /// A built-in shape — see `femShapes()` — meshed at about `spacing`.
    #[napi(factory)]
    pub fn shape(name: String, spacing: f64) -> Result<Self> {
        flat::Model::shape(&name, spacing).map(Self).map_err(err)
    }

    /// Mesh a closed triangle surface (`positions` x, y, z per vertex).
    #[napi(factory)]
    pub fn from_triangles(positions: Vec<f64>, indices: Vec<u32>, spacing: f64) -> Result<Self> {
        flat::Model::from_triangles(&positions, &indices, spacing)
            .map(Self)
            .map_err(err)
    }

    /// An exact box grid with `n` (`[nx, ny, nz]`) cubes per axis.
    #[napi(factory)]
    pub fn box_grid(lo: Vec<f64>, hi: Vec<f64>, n: Vec<u32>) -> Result<Self> {
        let [nx, ny, nz] = match n.as_slice() {
            [a, b, c] => [*a as usize, *b as usize, *c as usize],
            _ => return Err(Error::new(Status::InvalidArg, "n must be [nx, ny, nz]")),
        };
        Ok(Self(flat::Model::box_grid(v3(&lo)?, v3(&hi)?, [nx, ny, nz])))
    }

    #[napi(getter)]
    pub fn node_count(&self) -> u32 {
        self.0.node_count() as u32
    }

    #[napi(getter)]
    pub fn tet_count(&self) -> u32 {
        self.0.tet_count() as u32
    }

    #[napi(getter)]
    pub fn volume(&self) -> f64 {
        self.0.volume()
    }

    /// `[min x, min y, min z, max x, max y, max z]`.
    #[napi]
    pub fn bounds(&self) -> Vec<f64> {
        self.0.bounds().to_vec()
    }

    /// Smallest and largest dihedral angle, degrees.
    #[napi]
    pub fn dihedral_range(&self) -> Vec<f64> {
        self.0.dihedral_range().to_vec()
    }

    /// x, y, z per node.
    #[napi]
    pub fn positions(&self) -> Float64Array {
        Float64Array::new(self.0.positions())
    }

    /// Boundary triangles as node indices, wound outward.
    #[napi]
    pub fn surface_indices(&self) -> Uint32Array {
        Uint32Array::new(self.0.surface_indices().to_vec())
    }

    #[napi]
    pub fn set_material(&mut self, youngs: f64, poisson: f64, density: Option<f64>) {
        self.0.set_material(youngs, poisson, density.unwrap_or(1.0));
    }

    /// Ten-node elements (default) or four-node.
    #[napi]
    pub fn set_quadratic(&mut self, quadratic: bool) {
        self.0.set_quadratic(quadratic);
    }

    /// Remove every support, load and thermal condition.
    #[napi]
    pub fn clear(&mut self) {
        self.0.clear();
    }

    /// Clamp every node inside the box.
    #[napi]
    pub fn fix_box(&mut self, lo: Vec<f64>, hi: Vec<f64>) -> Result<()> {
        self.0.fix_box(v3(&lo)?, v3(&hi)?);
        Ok(())
    }

    /// Hold displacement component `axis` (0, 1, 2) at `value` in the box.
    #[napi]
    pub fn hold_box(&mut self, lo: Vec<f64>, hi: Vec<f64>, axis: u32, value: Option<f64>) -> Result<()> {
        self.0.hold_box(v3(&lo)?, v3(&hi)?, axis as usize, value.unwrap_or(0.0));
        Ok(())
    }

    /// A total force spread by area over the boundary faces in the box.
    #[napi]
    pub fn force_box(&mut self, lo: Vec<f64>, hi: Vec<f64>, force: Vec<f64>) -> Result<()> {
        self.0.force_box(v3(&lo)?, v3(&hi)?, v3(&force)?);
        Ok(())
    }

    /// Inward pressure on the boundary faces in the box.
    #[napi]
    pub fn pressure_box(&mut self, lo: Vec<f64>, hi: Vec<f64>, pressure: f64) -> Result<()> {
        self.0.pressure_box(v3(&lo)?, v3(&hi)?, pressure);
        Ok(())
    }

    #[napi]
    pub fn set_gravity(&mut self, g: Vec<f64>) -> Result<()> {
        self.0.set_gravity(v3(&g)?);
        Ok(())
    }

    #[napi]
    pub fn temperature_box(&mut self, lo: Vec<f64>, hi: Vec<f64>, t: f64) -> Result<()> {
        self.0.temperature_box(v3(&lo)?, v3(&hi)?, t);
        Ok(())
    }

    #[napi]
    pub fn heat_flux_box(&mut self, lo: Vec<f64>, hi: Vec<f64>, q: f64) -> Result<()> {
        self.0.heat_flux_box(v3(&lo)?, v3(&hi)?, q);
        Ok(())
    }

    #[napi]
    pub fn convection_box(&mut self, lo: Vec<f64>, hi: Vec<f64>, h: f64, ambient: f64) -> Result<()> {
        self.0.convection_box(v3(&lo)?, v3(&hi)?, h, ambient);
        Ok(())
    }

    #[napi]
    pub fn set_heat_source(&mut self, q: f64) {
        self.0.set_heat_source(q);
    }

    /// Static solve: displacements and stresses.
    #[napi]
    pub fn solve(&self) -> Result<FemStatic> {
        let r = self.0.solve().map_err(err)?;
        Ok(FemStatic {
            displacement: Float64Array::new(r.displacement),
            von_mises: Float64Array::new(r.von_mises),
            stress: Float64Array::new(r.stress),
            max_von_mises: r.max_von_mises,
            max_displacement: r.max_displacement,
            reaction: r.reaction.to_vec(),
            compliance: r.compliance,
        })
    }

    /// The lowest `count` natural frequencies and mode shapes.
    #[napi]
    pub fn modes(&self, count: Option<u32>) -> Result<FemModes> {
        let r = self.0.modes(count.unwrap_or(4) as usize).map_err(err)?;
        Ok(FemModes {
            frequencies: r.frequencies,
            shapes: r.shapes.into_iter().map(Float64Array::new).collect(),
        })
    }

    /// Steady conduction with conductivity `k`.
    #[napi]
    pub fn solve_heat(&self, k: f64) -> Result<FemHeat> {
        let r = self.0.solve_heat(k).map_err(err)?;
        Ok(FemHeat {
            temperature: Float64Array::new(r.temperature),
            flux: Float64Array::new(r.flux),
            min: r.min,
            max: r.max,
        })
    }
}

/// A static solution, per node.
#[napi(object)]
pub struct FemStatic {
    /// x, y, z per node.
    pub displacement: Float64Array,
    pub von_mises: Float64Array,
    /// Voigt xx, yy, zz, yz, xz, xy per node.
    pub stress: Float64Array,
    pub max_von_mises: f64,
    pub max_displacement: f64,
    /// Total support reaction, `[x, y, z]`.
    pub reaction: Vec<f64>,
    pub compliance: f64,
}

/// Natural frequencies and mode shapes (x, y, z per node, largest 1).
#[napi(object)]
pub struct FemModes {
    pub frequencies: Vec<f64>,
    pub shapes: Vec<Float64Array>,
}

/// A temperature field, per node.
#[napi(object)]
pub struct FemHeat {
    pub temperature: Float64Array,
    /// x, y, z heat flux per node.
    pub flux: Float64Array,
    pub min: f64,
    pub max: f64,
}

/// A closed surface for the boundary element method.
#[napi(js_name = "FemSurface")]
pub struct JsFemSurface(flat::SurfaceModel);

#[napi]
impl JsFemSurface {
    /// A box `size` across, centred on the origin.
    #[napi(factory)]
    pub fn cuboid(size: Vec<f64>, segments_per_unit: f64) -> Result<Self> {
        flat::SurfaceModel::cuboid(v3(&size)?, segments_per_unit)
            .map(Self)
            .map_err(err)
    }

    /// An icosphere with `20·(detail + 1)²` triangles.
    #[napi(factory)]
    pub fn sphere(radius: f64, detail: u32) -> Result<Self> {
        flat::SurfaceModel::sphere(radius, detail as usize)
            .map(Self)
            .map_err(err)
    }

    #[napi(factory)]
    pub fn from_triangles(positions: Vec<f64>, indices: Vec<u32>) -> Result<Self> {
        flat::SurfaceModel::from_triangles(&positions, &indices)
            .map(Self)
            .map_err(err)
    }

    #[napi(getter)]
    pub fn triangle_count(&self) -> u32 {
        self.0.triangle_count() as u32
    }

    #[napi]
    pub fn positions(&self) -> Float64Array {
        Float64Array::new(self.0.positions())
    }

    #[napi]
    pub fn indices(&self) -> Uint32Array {
        Uint32Array::new(self.0.indices())
    }

    #[napi]
    pub fn centroids(&self) -> Float64Array {
        Float64Array::new(self.0.centroids())
    }

    #[napi]
    pub fn normals(&self) -> Float64Array {
        Float64Array::new(self.0.normals())
    }

    /// Capacitance in units of the permittivity.
    #[napi]
    pub fn capacitance(&self) -> Result<f64> {
        self.0.capacitance().map_err(err)
    }

    /// Per-triangle normal velocity for a piston inside the box, facing
    /// `facing`.
    #[napi]
    pub fn piston(&self, lo: Vec<f64>, hi: Vec<f64>, facing: Vec<f64>, velocity: f64) -> Result<Float64Array> {
        Ok(Float64Array::new(self.0.piston(v3(&lo)?, v3(&hi)?, v3(&facing)?, velocity)))
    }

    /// Radiate at `frequency` from per-triangle normal velocity
    /// `velocityRe + i·velocityIm` (air at 20 °C unless given).
    #[napi]
    pub fn radiate(
        &self,
        frequency: f64,
        velocity_re: Vec<f64>,
        velocity_im: Option<Vec<f64>>,
        sound_speed: Option<f64>,
        density: Option<f64>,
    ) -> Result<FemAcoustic> {
        let im = velocity_im.unwrap_or_else(|| vec![0.0; velocity_re.len()]);
        self.0
            .radiate(
                frequency,
                sound_speed.unwrap_or(343.0),
                density.unwrap_or(1.204),
                &velocity_re,
                &im,
            )
            .map(FemAcoustic)
            .map_err(err)
    }
}

/// A radiated sound field.
#[napi(js_name = "FemAcoustic")]
pub struct FemAcoustic(flat::AcousticResult);

#[napi]
impl FemAcoustic {
    #[napi(getter)]
    pub fn power(&self) -> f64 {
        self.0.power
    }

    #[napi(getter)]
    pub fn elements_per_wavelength(&self) -> f64 {
        self.0.elements_per_wavelength
    }

    #[napi]
    pub fn pressure_re(&self) -> Float64Array {
        Float64Array::new(self.0.pressure_re.clone())
    }

    #[napi]
    pub fn pressure_im(&self) -> Float64Array {
        Float64Array::new(self.0.pressure_im.clone())
    }

    /// Complex pressure at `points` (x, y, z each), interleaved re, im.
    #[napi]
    pub fn field(&self, points: Vec<f64>) -> Float64Array {
        let (re, im) = self.0.field(&points);
        Float64Array::new(re.into_iter().zip(im).flat_map(|(a, b)| [a, b]).collect())
    }

    /// Far-field amplitude per direction (x, y, z each).
    #[napi]
    pub fn far_field(&self, directions: Vec<f64>) -> Float64Array {
        Float64Array::new(self.0.far_field(&directions))
    }
}
