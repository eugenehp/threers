//! A flat API for language bindings: regions are boxes instead of closures,
//! inputs and results are plain arrays, and a few demonstration parts come
//! built in. The wasm, Python and Node bindings are thin layers over this.
//!
//! ```
//! use threers_fem::flat::Model;
//!
//! let mut m = Model::shape("beam", 0.5).unwrap(); // 10 × 1 × 1
//! m.set_material(1000.0, 0.3, 1.0);
//! m.fix_box([-1.0, -1.0, -1.0], [0.0, 2.0, 2.0]);
//! m.force_box([10.0, -1.0, -1.0], [11.0, 2.0, 2.0], [0.0, -1.0, 0.0]);
//! let r = m.solve().unwrap();
//! assert!(r.max_displacement > 3.9 && r.max_displacement < 4.3); // FL³/3EI = 4
//! assert_eq!(r.displacement.len(), 3 * m.node_count());
//! ```

use crate::bem::{Helmholtz, HelmholtzSolution, Laplace, Potential};
use crate::complex::Complex;
use crate::fem::{Elastic, Heat, Material, Order};
use crate::mesh::TetMesh;
use crate::surface::TriSurface;
use crate::vec3::*;
use crate::FemError;

/// `x, y, z` triples out of a flat array; a trailing partial triple is
/// dropped.
fn triples(flat: &[f64]) -> Vec<V3> {
    flat.as_chunks::<3>().0.to_vec()
}

/// Signed distance to an axis-aligned box.
fn cuboid(p: V3, lo: V3, hi: V3) -> f64 {
    let d = [0, 1, 2].map(|k| (lo[k] - p[k]).max(p[k] - hi[k]));
    let outside = d.iter().map(|v| v.max(0.0).powi(2)).sum::<f64>().sqrt();
    outside + d[0].max(d[1]).max(d[2]).min(0.0)
}

/// The built-in parts, by name, with a one-line description each.
pub const SHAPES: [(&str, &str); 4] = [
    ("beam", "10 × 1 × 1 bar on an exact grid"),
    ("plate", "100 × 40 × 5 plate with a 20 mm hole: a stress concentration"),
    ("bracket", "L-bracket, 60 mm arm on a 50 mm upright, 6 mm thick, with a hole"),
    ("hook", "a ring of 20 mm mean radius, 6 mm section, with a gap"),
];

enum Support {
    Fix([V3; 2], [Option<f64>; 3]),
}

enum Load {
    Force([V3; 2], V3),
    Pressure([V3; 2], f64),
}

enum Thermal {
    Temperature([V3; 2], f64),
    Flux([V3; 2], f64),
    Convection([V3; 2], f64, f64),
}

/// A meshed part, its material, and its supports and loads.
pub struct Model {
    pub mesh: TetMesh,
    surface: Vec<u32>,
    material: Material,
    order: Order,
    supports: Vec<Support>,
    loads: Vec<Load>,
    gravity: V3,
    thermal: Vec<Thermal>,
    source: f64,
    tol: f64,
}

/// What a static solve returns, per mesh node.
#[derive(Clone, Debug)]
pub struct StaticResult {
    /// `x, y, z` per node.
    pub displacement: Vec<f64>,
    pub von_mises: Vec<f64>,
    /// Voigt `xx, yy, zz, yz, xz, xy` per node.
    pub stress: Vec<f64>,
    pub max_von_mises: f64,
    pub max_displacement: f64,
    pub reaction: [f64; 3],
    pub compliance: f64,
}

/// What a modal solve returns.
#[derive(Clone, Debug)]
pub struct ModalResult {
    pub frequencies: Vec<f64>,
    /// Per mode, `x, y, z` per node, largest displacement 1.
    pub shapes: Vec<Vec<f64>>,
}

/// What a heat solve returns, per mesh node.
#[derive(Clone, Debug)]
pub struct HeatResult {
    pub temperature: Vec<f64>,
    /// `x, y, z` heat flux per node.
    pub flux: Vec<f64>,
    pub min: f64,
    pub max: f64,
}

impl Model {
    pub fn new(mesh: TetMesh) -> Self {
        let (s, back) = mesh.boundary_surface();
        let surface = s.triangles.iter().flat_map(|t| t.map(|v| back[v as usize])).collect();
        let (lo, hi) = mesh.bounds();
        Self {
            tol: 1e-6 * dist(lo, hi).max(1e-300),
            mesh,
            surface,
            material: Material::new(1.0, 0.3),
            order: Order::Quadratic,
            supports: Vec::new(),
            loads: Vec::new(),
            gravity: [0.0; 3],
            thermal: Vec::new(),
            source: 0.0,
        }
    }

    /// Mesh a closed triangle surface: `positions` as `x, y, z` triples,
    /// `indices` as triangles.
    pub fn from_triangles(positions: &[f64], indices: &[u32], spacing: f64) -> Result<Self, FemError> {
        if !positions.len().is_multiple_of(3) || !indices.len().is_multiple_of(3) {
            return Err(FemError::InvalidInput("positions and indices must come in threes"));
        }
        let pts = triples(positions);
        let s = TriSurface::welded(&pts, indices);
        Ok(Self::new(TetMesh::from_surface(&s, spacing)?))
    }

    /// An exact box grid from `lo` to `hi` with `n` cubes per axis.
    pub fn box_grid(lo: V3, hi: V3, n: [usize; 3]) -> Self {
        Self::new(TetMesh::box_grid(lo, hi, n))
    }

    /// One of the [`SHAPES`], meshed at about `spacing`.
    pub fn shape(name: &str, spacing: f64) -> Result<Self, FemError> {
        if spacing.is_nan() || spacing <= 0.0 {
            return Err(FemError::InvalidInput("spacing must be positive"));
        }
        let mesh = match name {
            "beam" => {
                let n = |l: f64| ((l / spacing).round() as usize).max(1);
                TetMesh::box_grid([0.0; 3], [10.0, 1.0, 1.0], [n(10.0), n(1.0), n(1.0)])
            }
            "plate" => TetMesh::from_sdf([0.0; 3], [100.0, 40.0, 5.0], spacing, |p| {
                let hole = 10.0 - ((p[0] - 50.0).powi(2) + (p[1] - 20.0).powi(2)).sqrt();
                cuboid(p, [0.0; 3], [100.0, 40.0, 5.0]).max(hole)
            })?,
            "bracket" => TetMesh::from_sdf([0.0; 3], [60.0, 50.0, 30.0], spacing, |p| {
                let upright = cuboid(p, [0.0; 3], [6.0, 50.0, 30.0]);
                let arm = cuboid(p, [0.0; 3], [60.0, 6.0, 30.0]);
                let hole = ((p[0] - 48.0).powi(2) + (p[2] - 15.0).powi(2)).sqrt() - 5.0;
                upright.min(arm).max(-hole)
            })?,
            "hook" => TetMesh::from_sdf([-26.0, -26.0, -3.0], [26.0, 26.0, 3.0], spacing, |p| {
                // A torus of mean radius 20 and tube radius 3, cut open on
                // the -y side by a 30° gap.
                let r = (p[0] * p[0] + p[1] * p[1]).sqrt();
                let ring = ((r - 20.0).powi(2) + p[2] * p[2]).sqrt() - 3.0;
                let ang = p[0].atan2(-p[1]).abs();
                let gap = (15f64.to_radians() - ang) * r.max(1.0);
                ring.max(gap)
            })?,
            _ => return Err(FemError::InvalidInput("unknown shape")),
        };
        Ok(Self::new(mesh))
    }

    pub fn node_count(&self) -> usize {
        self.mesh.nodes.len()
    }

    pub fn tet_count(&self) -> usize {
        self.mesh.tets.len()
    }

    /// `x, y, z` per node.
    pub fn positions(&self) -> Vec<f64> {
        self.mesh.nodes.iter().flatten().copied().collect()
    }

    /// Boundary triangles, as mesh node indices, wound outward.
    pub fn surface_indices(&self) -> &[u32] {
        &self.surface
    }

    /// `[min x, min y, min z, max x, max y, max z]`.
    pub fn bounds(&self) -> [f64; 6] {
        let (lo, hi) = self.mesh.bounds();
        [lo[0], lo[1], lo[2], hi[0], hi[1], hi[2]]
    }

    pub fn volume(&self) -> f64 {
        self.mesh.volume()
    }

    /// Smallest and largest dihedral angle, degrees.
    pub fn dihedral_range(&self) -> [f64; 2] {
        let q = self.mesh.quality();
        [q.min_dihedral, q.max_dihedral]
    }

    pub fn set_material(&mut self, youngs: f64, poisson: f64, density: f64) {
        self.material = Material::new(youngs, poisson).density(density);
    }

    /// Ten-node elements when true (the default), four-node when false.
    pub fn set_quadratic(&mut self, quadratic: bool) {
        self.order = if quadratic { Order::Quadratic } else { Order::Linear };
    }

    /// Remove every support, load and thermal condition.
    pub fn clear(&mut self) {
        self.supports.clear();
        self.loads.clear();
        self.thermal.clear();
        self.gravity = [0.0; 3];
        self.source = 0.0;
    }

    fn region(&self, lo: V3, hi: V3) -> [V3; 2] {
        [sub(lo, [self.tol; 3]), add(hi, [self.tol; 3])]
    }

    /// Clamp every node inside the box `lo..hi`.
    pub fn fix_box(&mut self, lo: V3, hi: V3) {
        let r = self.region(lo, hi);
        self.supports.push(Support::Fix(r, [Some(0.0); 3]));
    }

    /// Hold one displacement component (`axis` 0, 1 or 2) at `value` inside
    /// the box: a roller, or a symmetry plane at zero.
    pub fn hold_box(&mut self, lo: V3, hi: V3, axis: usize, value: f64) {
        let r = self.region(lo, hi);
        let mut d = [None; 3];
        d[axis.min(2)] = Some(value);
        self.supports.push(Support::Fix(r, d));
    }

    /// A total force spread by area over the boundary faces inside the box.
    pub fn force_box(&mut self, lo: V3, hi: V3, force: V3) {
        let r = self.region(lo, hi);
        self.loads.push(Load::Force(r, force));
    }

    /// Pressure, pushing inward, on the boundary faces inside the box.
    pub fn pressure_box(&mut self, lo: V3, hi: V3, pressure: f64) {
        let r = self.region(lo, hi);
        self.loads.push(Load::Pressure(r, pressure));
    }

    pub fn set_gravity(&mut self, g: V3) {
        self.gravity = g;
    }

    /// Hold the nodes inside the box at temperature `t`.
    pub fn temperature_box(&mut self, lo: V3, hi: V3, t: f64) {
        let r = self.region(lo, hi);
        self.thermal.push(Thermal::Temperature(r, t));
    }

    /// Heat flux per area into the body on the faces inside the box.
    pub fn heat_flux_box(&mut self, lo: V3, hi: V3, q: f64) {
        let r = self.region(lo, hi);
        self.thermal.push(Thermal::Flux(r, q));
    }

    /// Convection to `ambient` with film coefficient `h` on the faces inside
    /// the box.
    pub fn convection_box(&mut self, lo: V3, hi: V3, h: f64, ambient: f64) {
        let r = self.region(lo, hi);
        self.thermal.push(Thermal::Convection(r, h, ambient));
    }

    /// Uniform heat generation per volume.
    pub fn set_heat_source(&mut self, q: f64) {
        self.source = q;
    }

    fn elastic(&self) -> Elastic<'_> {
        let inside = |[lo, hi]: [V3; 2]| move |p: V3| (0..3).all(|k| p[k] >= lo[k] && p[k] <= hi[k]);
        let mut e = Elastic::new(&self.mesh, self.material).order(self.order);
        for s in &self.supports {
            let Support::Fix(r, d) = s;
            e = e.displace(inside(*r), *d);
        }
        for l in &self.loads {
            e = match l {
                Load::Force(r, f) => e.force(inside(*r), *f),
                Load::Pressure(r, p) => e.pressure(inside(*r), *p),
            };
        }
        if self.gravity != [0.0; 3] {
            e = e.gravity(self.gravity);
        }
        e
    }

    pub fn solve(&self) -> Result<StaticResult, FemError> {
        let s = self.elastic().solve()?;
        Ok(StaticResult {
            displacement: s.displacement.iter().flatten().copied().collect(),
            stress: s.stress.iter().flatten().copied().collect(),
            max_von_mises: s.max_von_mises(),
            max_displacement: s.max_displacement(),
            von_mises: s.von_mises,
            reaction: s.reaction,
            compliance: s.compliance,
        })
    }

    /// The lowest `count` natural frequencies and shapes under the
    /// supports (loads ignored).
    pub fn modes(&self, count: usize) -> Result<ModalResult, FemError> {
        let m = self.elastic().modes(count)?;
        Ok(ModalResult {
            frequencies: m.frequencies(),
            shapes: m.modes.iter().map(|m| m.shape.iter().flatten().copied().collect()).collect(),
        })
    }

    /// Steady conduction with conductivity `k` under the thermal conditions.
    pub fn solve_heat(&self, k: f64) -> Result<HeatResult, FemError> {
        let inside = |[lo, hi]: [V3; 2]| move |p: V3| (0..3).all(|k| p[k] >= lo[k] && p[k] <= hi[k]);
        let mut h = Heat::new(&self.mesh, k).order(self.order);
        for t in &self.thermal {
            h = match t {
                Thermal::Temperature(r, v) => h.temperature(inside(*r), *v),
                Thermal::Flux(r, q) => h.flux(inside(*r), *q),
                Thermal::Convection(r, c, a) => h.convection(inside(*r), *c, *a),
            };
        }
        if self.source != 0.0 {
            h = h.source(self.source);
        }
        let s = h.solve()?;
        Ok(HeatResult {
            min: s.min_temperature(),
            max: s.max_temperature(),
            flux: s.flux.iter().flatten().copied().collect(),
            temperature: s.temperature,
        })
    }
}

/// A closed surface for the boundary element method.
pub struct SurfaceModel {
    pub surface: TriSurface,
}

/// An acoustic solution: surface pressure, and the field anywhere outside.
pub struct AcousticResult {
    solution: HelmholtzSolution,
    /// Real and imaginary surface pressure per triangle.
    pub pressure_re: Vec<f64>,
    pub pressure_im: Vec<f64>,
    /// Radiated power, in the fluid it was solved for.
    pub power: f64,
    pub elements_per_wavelength: f64,
}

impl SurfaceModel {
    pub fn from_triangles(positions: &[f64], indices: &[u32]) -> Result<Self, FemError> {
        if !positions.len().is_multiple_of(3) || !indices.len().is_multiple_of(3) {
            return Err(FemError::InvalidInput("positions and indices must come in threes"));
        }
        let pts = triples(positions);
        let surface = TriSurface::welded(&pts, indices);
        if surface.defects() != 0 {
            return Err(FemError::InvalidSurface("surface is not closed"));
        }
        Ok(Self { surface })
    }

    /// A box `size` across, centred on the origin, `n` triangles' worth of
    /// segments per unit length on each axis (at least one).
    pub fn cuboid(size: V3, segments_per_unit: f64) -> Result<Self, FemError> {
        let seg = |l: f64| ((l * segments_per_unit).round() as u32).max(1);
        let g = threers::geometries::BoxGeometry::with_segments(
            size[0] as f32,
            size[1] as f32,
            size[2] as f32,
            seg(size[0]),
            seg(size[1]),
            seg(size[2]),
        );
        Ok(Self {
            surface: TriSurface::from_geometry(&g)?,
        })
    }

    /// An icosphere of `radius`, `20·(detail + 1)²` triangles.
    pub fn sphere(radius: f64, detail: usize) -> Result<Self, FemError> {
        let g = threers::geometries::IcosahedronGeometry::new(radius as f32, detail);
        Ok(Self {
            surface: TriSurface::from_geometry(&g)?,
        })
    }

    pub fn triangle_count(&self) -> usize {
        self.surface.triangles.len()
    }

    pub fn positions(&self) -> Vec<f64> {
        self.surface.vertices.iter().flatten().copied().collect()
    }

    pub fn indices(&self) -> Vec<u32> {
        self.surface.triangles.iter().flatten().copied().collect()
    }

    /// `x, y, z` centroid per triangle.
    pub fn centroids(&self) -> Vec<f64> {
        (0..self.surface.triangles.len())
            .flat_map(|t| {
                let [a, b, c] = self.surface.triangle(t);
                scale(add(add(a, b), c), 1.0 / 3.0)
            })
            .collect()
    }

    /// `x, y, z` outward normal per triangle.
    pub fn normals(&self) -> Vec<f64> {
        (0..self.surface.triangles.len())
            .flat_map(|t| self.surface.normal_area(t).0)
            .collect()
    }

    /// Capacitance in units of the permittivity.
    pub fn capacitance(&self) -> Result<f64, FemError> {
        Laplace::exterior(&self.surface)?.capacitance()
    }

    /// Exterior potential with the surface held at `potential` per triangle,
    /// evaluated at `points` (`x, y, z` triples).
    pub fn potential_field(&self, potential: &[f64], points: &[f64]) -> Result<Vec<f64>, FemError> {
        if potential.len() != self.triangle_count() {
            return Err(FemError::InvalidInput("one potential per triangle"));
        }
        let lap = Laplace::exterior(&self.surface)?;
        // The solvers ask for conditions in triangle order.
        let i = std::cell::Cell::new(0);
        let s = lap.solve(|_, _| {
            let k = i.replace(i.get() + 1);
            Potential::Value(potential[k])
        })?;
        let pts = triples(points);
        Ok(s.potential_at_points(&pts))
    }

    /// Sound radiated at `frequency` into a fluid of `sound_speed` and
    /// `density` by the surface vibrating with normal velocity
    /// `velocity_re + i·velocity_im` per triangle (outward positive).
    pub fn radiate(
        &self,
        frequency: f64,
        sound_speed: f64,
        density: f64,
        velocity_re: &[f64],
        velocity_im: &[f64],
    ) -> Result<AcousticResult, FemError> {
        let n = self.triangle_count();
        if velocity_re.len() != n || velocity_im.len() != n {
            return Err(FemError::InvalidInput("one velocity per triangle"));
        }
        let h = Helmholtz::acoustic(&self.surface, frequency, sound_speed)?;
        let i = std::cell::Cell::new(0);
        let s = h.radiate_velocity(density, sound_speed, |_, _| {
            let k = i.replace(i.get() + 1);
            Complex::new(velocity_re[k], velocity_im[k])
        })?;
        Ok(AcousticResult {
            pressure_re: s.pressure.iter().map(|p| p.re).collect(),
            pressure_im: s.pressure.iter().map(|p| p.im).collect(),
            power: s.sound_power(density, sound_speed),
            elements_per_wavelength: h.elements_per_wavelength(),
            solution: s,
        })
    }

    /// Per-triangle velocities for a piston: `velocity` on triangles whose
    /// centroid is inside the box and whose normal is within 45° of `facing`,
    /// zero elsewhere. Returns the real parts; the imaginary parts are zero.
    pub fn piston(&self, lo: V3, hi: V3, facing: V3, velocity: f64) -> Vec<f64> {
        let f = normalize(facing);
        (0..self.triangle_count())
            .map(|t| {
                let [a, b, c] = self.surface.triangle(t);
                let ctr = scale(add(add(a, b), c), 1.0 / 3.0);
                let (n, _) = self.surface.normal_area(t);
                let inside = (0..3).all(|k| ctr[k] >= lo[k] && ctr[k] <= hi[k]);
                if inside && dot(n, f) > std::f64::consts::FRAC_1_SQRT_2 { velocity } else { 0.0 }
            })
            .collect()
    }
}

impl AcousticResult {
    /// Complex pressure at `points` (`x, y, z` triples): real parts, then
    /// imaginary parts, each one per point.
    pub fn field(&self, points: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let pts = triples(points);
        let p = self.solution.pressure_at_points(&pts);
        (p.iter().map(|z| z.re).collect(), p.iter().map(|z| z.im).collect())
    }

    /// Far-field amplitude `|F|` per direction (`x, y, z` triples), with
    /// `p ≈ F e^{ikr}/r` far away.
    pub fn far_field(&self, directions: &[f64]) -> Vec<f64> {
        triples(directions)
            .into_iter()
            .map(|d| self.solution.far_field(d).abs())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_shape_meshes() {
        for (name, _) in SHAPES {
            let spacing = if name == "beam" { 0.5 } else { 4.0 };
            let m = Model::shape(name, spacing).unwrap();
            assert!(m.tet_count() > 50, "{name}");
            assert_eq!(m.mesh.defects(), 0, "{name}");
            assert_eq!(m.surface_indices().len() % 3, 0);
        }
    }

    #[test]
    fn plate_with_hole_concentrates_stress() {
        // Pull the plate along x: the peak at the hole is about Kt = 2.2 times
        // the nominal stress across the net section (Peterson, d/W = 0.5).
        let mut m = Model::shape("plate", 2.5).unwrap();
        m.set_material(200e3, 0.3, 7.85e-9);
        m.hold_box([-1.0, -1.0, -1.0], [0.0, 41.0, 6.0], 0, 0.0);
        m.hold_box([-1.0, -1.0, -1.0], [101.0, 41.0, 0.0], 2, 0.0);
        m.hold_box([-1.0, 19.0, -1.0], [0.0, 21.0, 6.0], 1, 0.0);
        let force = 20.0 * 5.0 * 100.0; // 100 MPa on the gross section
        m.force_box([100.0, -1.0, -1.0], [101.0, 41.0, 6.0], [force, 0.0, 0.0]);
        let r = m.solve().unwrap();
        let net = force / (20.0 * 5.0);
        let kt = r.max_von_mises / net;
        assert!(kt > 1.8 && kt < 2.6, "Kt {kt}");
    }

    #[test]
    fn heat_box_regions() {
        let mut m = Model::shape("beam", 0.5).unwrap();
        m.temperature_box([-1.0; 3], [0.0, 2.0, 2.0], 100.0);
        m.temperature_box([10.0, -1.0, -1.0], [11.0, 2.0, 2.0], 0.0);
        let r = m.solve_heat(1.0).unwrap();
        assert!((r.max - 100.0).abs() < 1e-9 && r.min.abs() < 1e-9);
    }

    #[test]
    fn piston_on_a_box_radiates() {
        let s = SurfaceModel::cuboid([0.2, 0.3, 0.2], 30.0).unwrap();
        let v = s.piston([-0.06, -0.06, 0.0], [0.06, 0.06, 1.0], [0.0, 0.0, 1.0], 1e-3);
        assert!(v.iter().filter(|x| **x != 0.0).count() > 4);
        let zero = vec![0.0; v.len()];
        let r = s.radiate(200.0, 343.0, 1.2, &v, &zero).unwrap();
        assert!(r.power > 0.0);
        let (re, im) = r.field(&[0.0, 0.0, 1.0, 0.0, 0.0, -1.0]);
        let front = re[0].hypot(im[0]);
        let back = re[1].hypot(im[1]);
        assert!(front > back && back > 0.0);
    }
}
