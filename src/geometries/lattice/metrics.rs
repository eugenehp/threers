//! What a lattice is worth to a fluid, a cell, or a bone.
//!
//! Stiffness per gram is one reason to build a lattice and not the only one.
//! A heat exchanger is chosen for the area it puts in the way of the flow and
//! the pressure it costs to push through; a scaffold for whether cells can get
//! in and whether the pores are the right size for the tissue; a filter for
//! both at once. Those are measurements, not adjectives, and this is where they
//! come from.
//!
//! ```
//! use threers::{Lattice, LatticeKind, Tpms, Vector3};
//!
//! let m = Lattice::new(LatticeKind::Tpms(Tpms::Gyroid))
//!     .size(Vector3::new(20.0, 20.0, 20.0))
//!     .cells([4, 4, 4])
//!     .fit_relative_density(0.3)
//!     .metrics();
//!
//! // Roughly a third solid, so roughly two thirds open.
//! assert!((m.porosity - 0.7).abs() < 0.1);
//! // And the pores are a fraction of the 5 mm cell.
//! assert!(m.pore_diameter > 0.5 && m.pore_diameter < 5.0);
//! // Open all the way through, on every axis.
//! assert!(m.percolates.iter().all(|&p| p));
//! ```
//!
//! # Porous, connected, and flowing are three questions
//!
//! They are routinely run together and they come apart badly. A closed-cell
//! foam is 80 % porous, mostly sealed, and carries no flow at all; a skinned
//! part is full of connected void that nothing can reach; a sheet TPMS is one
//! solid wall between *two* separate labyrinths, which is the entire reason to
//! put one in a heat exchanger and would look like a defect to anything that
//! only counted holes.
//!
//! So the void is flood-filled and asked three separate things:
//! [`open_porosity`] (can it be drained), [`percolates`] (can something flow
//! across), and [`largest_void_fraction`] (is it one space or many).
//!
//! [`open_porosity`]: LatticeMetrics::open_porosity
//! [`percolates`]: LatticeMetrics::percolates
//! [`largest_void_fraction`]: LatticeMetrics::largest_void_fraction
//!
//! # How the numbers are arrived at
//!
//! Areas are measured on the mesh that [`build`](super::Lattice::build) would
//! produce — the real triangles, so a grade or a skin is in the number. Volumes
//! and pore sizes are measured on a sampled grid, so they carry the sampling's
//! own error: a pore is never resolved finer than the spacing it was measured
//! at, and [`sample_spacing`](LatticeMetrics::sample_spacing) says what that
//! was.
//!
//! The distinction that matters most is between [`surface_area`] and
//! [`wetted_area`]: the first is every triangle in the mesh, the second drops
//! the ones lying in the part's own boundary, which are the outside of the part
//! rather than the inside of the lattice. A heat exchanger is sized on the
//! second.
//!
//! [`surface_area`]: LatticeMetrics::surface_area
//! [`wetted_area`]: LatticeMetrics::wetted_area

use super::Lattice;
use crate::math::Vector3;

/// Everything measurable about a built lattice that is not its stiffness.
///
/// Lengths are in the lattice's own units, so an area is length² and the
/// permeability is length².
#[derive(Clone, Copy, Debug, Default)]
pub struct LatticeMetrics {
    /// Fraction of the part that is material.
    pub relative_density: f32,
    /// Fraction of the part that is open — `1 - relative_density`.
    pub porosity: f32,
    /// The part of that porosity which connects to the outside of the part.
    ///
    /// The powder-removal question, and the fluid-access one: void that no path
    /// reaches cannot be drained, cleaned, infiltrated or flowed through. A
    /// [`skin`](super::Lattice::skin) closes a part off entirely and this says
    /// so — it drops to nearly zero.
    pub open_porosity: f32,
    /// The rest of it: sealed pockets. `open + closed = porosity`.
    pub closed_porosity: f32,
    /// Whether a connected void path crosses the part along x, y and z.
    ///
    /// The flow-through question, and stricter than [`open_porosity`]: one
    /// network has to open to the outside within the end tenth of the part at
    /// *both* ends of the axis. A closed-cell foam is porous, partly open where
    /// the surface cuts its bubbles, and percolates nowhere; an open-cell one
    /// does all three. A skinned part percolates nowhere either, whatever is
    /// inside it.
    ///
    /// [`open_porosity`]: Self::open_porosity
    pub percolates: [bool; 3],
    /// The largest single connected void network, as a fraction of all the
    /// void.
    ///
    /// 1 for an open-cell foam, where it is all one space. **About a half for a
    /// sheet TPMS**, which is not a defect but the reason they are used: the
    /// surface divides space into two labyrinths that never meet, so one heat
    /// exchanger can carry two fluids with a single wall between them. Near
    /// zero for a closed-cell foam, where every bubble is its own.
    pub largest_void_fraction: f32,
    /// Volume of the part, lattice and void together.
    pub part_volume: f32,
    /// Volume of the material in it.
    pub solid_volume: f32,
    /// Area of every triangle in the mesh, the part's outside included.
    pub surface_area: f32,
    /// Area of the lattice's internal surface: [`surface_area`] less the
    /// triangles lying in the part's own boundary.
    ///
    /// [`surface_area`]: Self::surface_area
    pub wetted_area: f32,
    /// Internal area per unit of part volume — the number a heat exchanger is
    /// bought for. Units of 1/length.
    pub specific_surface_area: f32,
    /// Internal area per unit of material. Rises as the walls thin, which is
    /// why a thin sheet beats a fat strut for anything surface-driven.
    pub surface_area_to_volume: f32,
    /// Diameter of the largest sphere that fits in the void — the pore size a
    /// scaffold or a filter is specified by.
    pub pore_diameter: f32,
    /// Diameter of the largest sphere that fits inside the material — the
    /// thickest ligament, and what a printer has to be able to cool.
    pub ligament_thickness: f32,
    /// `4 · void volume / wetted area`: the diameter of a round pipe with the
    /// same ratio, and the length scale a Reynolds or Nusselt number for this
    /// lattice is built on.
    pub hydraulic_diameter: f32,
    /// Kozeny–Carman estimate, `ε³ / (5 · S²)` with `S` the internal area per
    /// unit of part volume. Units of length².
    ///
    /// An estimate and not a solve: Kozeny–Carman was fitted to packed beds,
    /// and it is within a factor of two on an open lattice and further out on
    /// one with dead ends. It ranks two lattices reliably; it does not size a
    /// pump.
    pub permeability: f32,
    /// The grid spacing the volumes and pore sizes were measured at. No feature
    /// smaller than this was seen at all.
    pub sample_spacing: f32,
    /// How many samples fell across the wall — the same reading
    /// [`wall_samples`](super::Lattice::wall_samples) gives for the build grid,
    /// for the grid these numbers were measured on.
    ///
    /// Below about two, the sampling misses part of the wall: the density comes
    /// out low, and the pores come out enormous because two sides of a wall
    /// that was never sampled look like one big void. Infinite for a lattice
    /// with no wall to resolve.
    pub wall_samples: f32,
}

impl<'a> Lattice<'a> {
    /// Measure the lattice — see [`LatticeMetrics`].
    ///
    /// Builds the mesh once for the areas and samples a grid once for the
    /// volumes, so it costs about what a [`build`](Self::build) costs. Keep the
    /// result rather than calling it per field.
    pub fn metrics(&self) -> LatticeMetrics {
        // Sampled fine enough to see the wall, which is not a refinement but
        // the difference between a measurement and a fiction: a grid coarser
        // than the wall reads a lattice as half its density with pores the size
        // of the cell. Bounded, because a wall a thousandth of the cell would
        // otherwise ask for a grid nothing can hold — and `wall_samples` says
        // so when the bound bites.
        let base = self.resolution.min(16);
        let per_cell = match self.wall_width() {
            Some(width) if width > 0.0 => {
                let cell = self.resolved_cell();
                let widest = cell.x.max(cell.y).max(cell.z);
                let needed = (3.0 * widest / width).ceil();
                let needed = if needed.is_finite() && needed > 0.0 {
                    (needed as usize).min(256)
                } else {
                    base
                };
                base.max(needed)
            }
            _ => base,
        };
        self.metrics_at(per_cell)
    }

    /// The same, sampled at an explicit number of points per cell.
    ///
    /// Volumes converge quickly; pore and ligament sizes do not, because they
    /// are extrema rather than averages and a grid can only find a sphere it
    /// has a sample inside of. Twelve per cell is a reading, twenty-four is a
    /// number to quote.
    pub fn metrics_at(&self, samples_per_cell: usize) -> LatticeMetrics {
        let geometry = self.build();
        let (surface_area, wetted_area) = self.areas(&geometry);

        let cell = self.resolved_cell();
        let size = self.bounds.size();
        let res = samples_per_cell.max(2) as f32;
        let count = |extent: f32, cell: f32| (((extent / cell.max(1e-6)) * res).ceil() as usize).max(2);
        let mut dims = [
            count(size.x, cell.x),
            count(size.y, cell.y),
            count(size.z, cell.z),
        ];
        // The same budget the build grid answers to, for the same reason.
        while dims[0].saturating_mul(dims[1]).saturating_mul(dims[2]) > self.max_samples {
            let next = [
                (dims[0] / 2).max(2),
                (dims[1] / 2).max(2),
                (dims[2] / 2).max(2),
            ];
            if next == dims {
                break;
            }
            dims = next;
        }
        let spacing = Vector3::new(
            size.x / dims[0] as f32,
            size.y / dims[1] as f32,
            size.z / dims[2] as f32,
        );
        let sampler = self.sampler(cell, spacing, false);

        // Three states per sample: outside the part, void inside it, solid.
        let total = dims[0] * dims[1] * dims[2];
        let states = crate::utils::parallel::par_map_range(total, |n| {
            let i = n % dims[0];
            let j = (n / dims[0]) % dims[1];
            let k = n / (dims[0] * dims[1]);
            let p = Vector3::new(
                self.bounds.min.x + (i as f32 + 0.5) * spacing.x,
                self.bounds.min.y + (j as f32 + 0.5) * spacing.y,
                self.bounds.min.z + (k as f32 + 0.5) * spacing.z,
            );
            let cut = self.trim.as_deref().map(|t| t(p));
            if cut.is_some_and(|c| c <= 0.0) {
                0u8
            } else if sampler.value_at(p, cut) > 0.0 {
                2
            } else {
                1
            }
        });

        let connectivity = void_connectivity(&states, dims);

        let voxel = spacing.x * spacing.y * spacing.z;
        let inside = states.iter().filter(|&&s| s > 0).count();
        let part_volume = inside as f32 * voxel;
        // The density comes from the jittered estimate rather than by counting
        // this grid's solid samples. A regular grid is commensurate with the
        // lattice — every cell sampled at the same relative points — so it
        // reads the same wall the same way a few thousand times over and the
        // error does not average out. Jittered strata have no such alignment,
        // and give the number `fit_relative_density` solved against.
        let relative_density = self.relative_density();
        let solid_volume = part_volume * relative_density;
        let porosity = 1.0 - relative_density;

        // The largest sphere in the void, and the largest in the material.
        // Only inside the part: the space outside it is unbounded, and the
        // largest sphere that fits in it is not a property of the lattice.
        let pore_diameter = 2.0
            * max_inscribed(
                &states,
                dims,
                spacing,
                |s| s == 2 || s == 0, // walls, and the edge of the part
                |s| s == 1,
            );
        let ligament_thickness = 2.0 * max_inscribed(&states, dims, spacing, |s| s != 2, |s| s == 2);

        let void_volume = part_volume - solid_volume;
        let specific_surface_area = if part_volume > 0.0 {
            wetted_area / part_volume
        } else {
            0.0
        };
        let hydraulic_diameter = if wetted_area > 0.0 {
            4.0 * void_volume / wetted_area
        } else {
            0.0
        };
        let permeability = if specific_surface_area > 0.0 {
            porosity.powi(3) / (5.0 * specific_surface_area * specific_surface_area)
        } else {
            0.0
        };

        LatticeMetrics {
            relative_density,
            porosity,
            // As a share of the measured porosity rather than of the counted
            // void, so that the two add back up to it exactly — the porosity
            // itself comes from the jittered estimate and this grid from a
            // regular one, and they do not agree to the last digit.
            open_porosity: porosity * connectivity.open_fraction,
            closed_porosity: porosity * (1.0 - connectivity.open_fraction),
            percolates: connectivity.percolates,
            largest_void_fraction: connectivity.largest_fraction,
            part_volume,
            solid_volume,
            surface_area,
            wetted_area,
            specific_surface_area,
            surface_area_to_volume: if solid_volume > 0.0 {
                wetted_area / solid_volume
            } else {
                0.0
            },
            pore_diameter,
            ligament_thickness,
            hydraulic_diameter,
            permeability,
            sample_spacing: spacing.x.max(spacing.y).max(spacing.z),
            wall_samples: match self.wall_width() {
                Some(width) => width / spacing.x.max(spacing.y).max(spacing.z),
                None => f32::INFINITY,
            },
        }
    }

    /// Total triangle area, and the part of it that is not the part's own
    /// boundary.
    ///
    /// A triangle counts as boundary when its centroid sits within a sample
    /// step of the fill's surface — which is where the mesh closes over the
    /// cut, and where a skin's outside is. That is a threshold and not a proof:
    /// a lattice whose cell is only a couple of samples across will lose some
    /// genuine internal wall to it.
    fn areas(&self, geometry: &crate::core::BufferGeometry) -> (f32, f32) {
        let Some(positions) = geometry.get_attribute("position") else {
            return (0.0, 0.0);
        };
        let verts: Vec<Vector3> = positions
            .array
            .chunks_exact(3)
            .map(|c| Vector3::new(c[0], c[1], c[2]))
            .collect();
        let indices: Vec<usize> = match &geometry.index {
            Some(index) => index.iter().map(|i| *i as usize).collect(),
            None => (0..verts.len()).collect(),
        };
        let step = self.grid().0 .2;
        let near = step.x.max(step.y).max(step.z);
        let mut total = 0.0f32;
        let mut wetted = 0.0f32;
        for tri in indices.chunks_exact(3) {
            let (a, b, c) = (verts[tri[0]], verts[tri[1]], verts[tri[2]]);
            let area = (b - a).cross(c - a).length() * 0.5;
            if !area.is_finite() {
                continue;
            }
            total += area;
            let centroid = (a + b + c) * (1.0 / 3.0);
            let mut boundary = self.box_distance(centroid);
            if let Some(trim) = self.trim.as_deref() {
                boundary = boundary.min(trim(centroid));
            }
            if boundary.abs() > near {
                wetted += area;
            }
        }
        (total, wetted)
    }

    /// Distance into the bounding box, negative outside it.
    fn box_distance(&self, p: Vector3) -> f32 {
        let lo = self.bounds.min;
        let hi = self.bounds.max;
        (p.x - lo.x)
            .min(hi.x - p.x)
            .min(p.y - lo.y)
            .min(hi.y - p.y)
            .min(p.z - lo.z)
            .min(hi.z - p.z)
    }
}

/// What a flood fill through the void finds.
struct Connectivity {
    /// Share of the void that reaches the outside of the part.
    open_fraction: f32,
    /// Whether one connected network reaches the exterior on both sides of the
    /// part, per axis.
    percolates: [bool; 3],
    /// The largest network as a share of all the void.
    largest_fraction: f32,
}

/// Label the void's connected components and ask each what it touches.
///
/// Six-connected: two voxels meeting only at an edge or a corner are not
/// connected, because nothing can flow through a line or a point. A depth-first
/// fill with an explicit stack rather than recursion — a percolating network in
/// a fine grid is millions of voxels deep and would take the call stack with
/// it.
fn void_connectivity(states: &[u8], dims: [usize; 3]) -> Connectivity {
    let [nx, ny, nz] = dims;
    let total = nx * ny * nz;
    let mut label = vec![u32::MAX; total];
    let mut stack: Vec<u32> = Vec::new();
    let (mut void, mut open, mut largest) = (0usize, 0usize, 0usize);
    let mut percolates = [false; 3];
    let mut next = 0u32;

    for seed in 0..total {
        if states[seed] != 1 || label[seed] != u32::MAX {
            continue;
        }
        label[seed] = next;
        stack.push(seed as u32);
        let mut size = 0usize;
        let mut exterior_component = false;
        let mut low = [false; 3];
        let mut high = [false; 3];
        while let Some(v) = stack.pop() {
            let v = v as usize;
            size += 1;
            let at = [v % nx, (v / nx) % ny, v / (nx * ny)];
            let mut exterior = false;
            for axis in 0..3 {
                for step in [-1i32, 1] {
                    let moved = at[axis] as i32 + step;
                    if moved < 0 || moved >= dims[axis] as i32 {
                        // Off the sampled box, which is the part's own edge.
                        exterior = true;
                        continue;
                    }
                    let mut to = at;
                    to[axis] = moved as usize;
                    let index = (to[2] * ny + to[1]) * nx + to[0];
                    match states[index] {
                        // Outside the part: this void opens onto the world.
                        0 => exterior = true,
                        1 if label[index] == u32::MAX => {
                            label[index] = next;
                            stack.push(index as u32);
                        }
                        _ => {}
                    }
                }
            }
            if exterior {
                exterior_component = true;
                // Where the opening is. A flow across the part has to get in
                // near one end of the axis and out near the other, so only
                // contacts in the end tenths count — a bubble sitting in the
                // middle of a face opens to the world without going anywhere,
                // and a network that merely straddles the midpoint has not
                // crossed anything.
                for axis in 0..3 {
                    let end = (dims[axis] / 10).max(1);
                    if at[axis] < end {
                        low[axis] = true;
                    }
                    if at[axis] + end >= dims[axis] {
                        high[axis] = true;
                    }
                }
            }
        }
        void += size;
        if exterior_component {
            open += size;
        }
        for axis in 0..3 {
            percolates[axis] |= low[axis] && high[axis];
        }
        largest = largest.max(size);
        next = next.wrapping_add(1);
    }

    if void == 0 {
        return Connectivity {
            open_fraction: 0.0,
            percolates: [false; 3],
            largest_fraction: 0.0,
        };
    }
    Connectivity {
        open_fraction: open as f32 / void as f32,
        percolates,
        largest_fraction: largest as f32 / void as f32,
    }
}

/// The radius of the largest sphere that fits inside `interior`, where
/// `barrier` is what it may not touch.
///
/// A Euclidean distance transform, then a maximum. Anything not a barrier and
/// not interior — the outside of the part, when the pores are being measured —
/// bounds the sphere without being counted as somewhere it could be centred.
fn max_inscribed(
    states: &[u8],
    dims: [usize; 3],
    spacing: Vector3,
    barrier: impl Fn(u8) -> bool,
    interior: impl Fn(u8) -> bool,
) -> f32 {
    let squared = squared_distance_transform(states, dims, spacing, barrier);
    let mut best = 0.0f32;
    for (n, &s) in states.iter().enumerate() {
        if interior(s) {
            best = best.max(squared[n]);
        }
    }
    // Sample centre to sample centre, and the surface between them lies about
    // half a sample nearer than the far centre does. Without the correction
    // every measured feature is a sample wider than it is, which is most of a
    // thin wall.
    let half = (spacing.x + spacing.y + spacing.z) / 6.0;
    (best.max(0.0).sqrt() - half).max(0.0)
}

/// Squared distance from every sample to the nearest barrier, by the
/// Felzenszwalb–Huttenlocher separable transform.
///
/// Three one-dimensional passes, each the lower envelope of one parabola per
/// sample, which is linear in the samples rather than the quadratic a direct
/// search would be. The per-axis spacing goes into the parabolas' curvature, so
/// an anisotropic grid measures a true distance and not an index count.
fn squared_distance_transform(
    states: &[u8],
    dims: [usize; 3],
    spacing: Vector3,
    barrier: impl Fn(u8) -> bool,
) -> Vec<f32> {
    const FAR: f32 = 1e20;
    let [nx, ny, nz] = dims;
    let mut d: Vec<f32> = states
        .iter()
        .map(|&s| if barrier(s) { 0.0 } else { FAR })
        .collect();
    let step = [spacing.x, spacing.y, spacing.z];
    let stride = [1usize, nx, nx * ny];
    let len = [nx, ny, nz];

    let mut line = Vec::new();
    let mut out = Vec::new();
    for axis in 0..3 {
        let n = len[axis];
        line.resize(n, 0.0);
        out.resize(n, 0.0);
        let (outer_a, outer_b) = match axis {
            0 => (ny, nz),
            1 => (nx, nz),
            _ => (nx, ny),
        };
        let (stride_a, stride_b) = match axis {
            0 => (stride[1], stride[2]),
            1 => (stride[0], stride[2]),
            _ => (stride[0], stride[1]),
        };
        for b in 0..outer_b {
            for a in 0..outer_a {
                let base = a * stride_a + b * stride_b;
                for i in 0..n {
                    line[i] = d[base + i * stride[axis]];
                }
                envelope(&line, &mut out, step[axis]);
                for i in 0..n {
                    d[base + i * stride[axis]] = out[i];
                }
            }
        }
    }
    d
}

/// The lower envelope of the parabolas `f[q] + (spacing · (x - q))²`.
fn envelope(f: &[f32], out: &mut [f32], spacing: f32) {
    let n = f.len();
    if n == 0 {
        return;
    }
    let a = (spacing * spacing) as f64;
    // Which parabola is lowest in each stretch, and where the stretches meet.
    let mut hull = vec![0usize; n];
    let mut split = vec![0.0f64; n + 1];
    let mut k = 0usize;
    hull[0] = 0;
    split[0] = f64::NEG_INFINITY;
    split[1] = f64::INFINITY;
    for q in 1..n {
        if !f[q].is_finite() {
            continue;
        }
        loop {
            let p = hull[k];
            // Where parabola q crosses parabola p.
            let s = ((f[q] as f64 + a * (q * q) as f64) - (f[p] as f64 + a * (p * p) as f64))
                / (2.0 * a * (q as f64 - p as f64));
            if s <= split[k] {
                if k == 0 {
                    k = 0;
                    hull[0] = q;
                    split[0] = f64::NEG_INFINITY;
                    split[1] = f64::INFINITY;
                    break;
                }
                k -= 1;
            } else {
                k += 1;
                hull[k] = q;
                split[k] = s;
                split[k + 1] = f64::INFINITY;
                break;
            }
        }
    }
    let mut k = 0usize;
    for (x, o) in out.iter_mut().enumerate() {
        while split[k + 1] < x as f64 {
            k += 1;
        }
        let p = hull[k];
        let dx = x as f64 - p as f64;
        *o = (f[p] as f64 + a * dx * dx) as f32;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometries::lattice::{LatticeKind, LatticeStyle, Strut, Tpms};

    #[test]
    fn distance_transform_matches_brute_force() {
        let dims = [9usize, 7, 5];
        let spacing = Vector3::new(1.0, 2.0, 0.5);
        let mut states = vec![1u8; dims[0] * dims[1] * dims[2]];
        for (n, s) in states.iter_mut().enumerate() {
            // A scattering of barriers, deterministic.
            if n % 17 == 0 {
                *s = 2;
            }
        }
        let got = squared_distance_transform(&states, dims, spacing, |s| s == 2);
        for k in 0..dims[2] {
            for j in 0..dims[1] {
                for i in 0..dims[0] {
                    let mut want = f32::INFINITY;
                    for kk in 0..dims[2] {
                        for jj in 0..dims[1] {
                            for ii in 0..dims[0] {
                                if states[(kk * dims[1] + jj) * dims[0] + ii] != 2 {
                                    continue;
                                }
                                let d = Vector3::new(
                                    (i as f32 - ii as f32) * spacing.x,
                                    (j as f32 - jj as f32) * spacing.y,
                                    (k as f32 - kk as f32) * spacing.z,
                                );
                                want = want.min(d.length_sq());
                            }
                        }
                    }
                    let n = (k * dims[1] + j) * dims[0] + i;
                    assert!(
                        (got[n] - want).abs() < 1e-3 * want.max(1.0),
                        "{i},{j},{k}: {} vs {want}",
                        got[n]
                    );
                }
            }
        }
    }

    #[test]
    fn a_solid_block_has_no_pores_and_no_internal_area() {
        let m = crate::geometries::lattice::Lattice::new(LatticeKind::Tpms(Tpms::Gyroid))
            .size(Vector3::new(4.0, 4.0, 4.0))
            .cells([2, 2, 2])
            .style(LatticeStyle::Sheet)
            // A wall thicker than the cell fills it solid.
            .thickness(8.0)
            .metrics_at(8);
        assert!(m.relative_density > 0.99, "{}", m.relative_density);
        assert!(m.pore_diameter < 1e-3, "{}", m.pore_diameter);
        assert!(m.wetted_area < 0.05 * m.surface_area, "{} of {}", m.wetted_area, m.surface_area);
        // The block's outside is 6 × 4 × 4.
        assert!((m.surface_area - 96.0).abs() < 5.0, "{}", m.surface_area);
        assert!((m.part_volume - 64.0).abs() < 1.0, "{}", m.part_volume);
    }

    #[test]
    fn a_thin_sheet_has_more_area_per_gram_than_a_fat_strut() {
        let size = Vector3::new(20.0, 20.0, 20.0);
        let build = |kind| {
            crate::geometries::lattice::Lattice::new(kind)
                .size(size)
                .cells([4, 4, 4])
                .fit_relative_density(0.2)
                .metrics()
        };
        let sheet = build(LatticeKind::Tpms(Tpms::Gyroid));
        let strut = build(LatticeKind::Strut(Strut::Cubic));
        // The whole argument for a TPMS in a heat exchanger, as a number: the
        // same material, three times the surface.
        assert!(
            sheet.surface_area_to_volume > 3.0 * strut.surface_area_to_volume,
            "sheet {} vs strut {}",
            sheet.surface_area_to_volume,
            strut.surface_area_to_volume
        );
        for m in [sheet, strut] {
            // The density asked for is the density measured.
            assert!((m.relative_density - 0.2).abs() < 0.02, "{}", m.relative_density);
            assert!((m.porosity - 0.8).abs() < 0.02, "{}", m.porosity);
            assert!((m.part_volume - 8000.0).abs() < 100.0, "{}", m.part_volume);
            assert!(m.pore_diameter > 0.0 && m.pore_diameter < 20.0, "{}", m.pore_diameter);
            assert!(m.permeability > 0.0 && m.permeability.is_finite());
            assert!(m.hydraulic_diameter > 0.0);
            // And it sampled fine enough to mean any of it.
            assert!(m.wall_samples > 2.0, "{}", m.wall_samples);
        }
        // A gyroid sheet splits every cell into two channels; a simple-cubic
        // cell is one open room with struts only on its edges, and the sphere
        // that fits in it spans more than a cell. That is the same ordering
        // their permeabilities come in, and the reason to pick one or the
        // other.
        assert!(strut.pore_diameter > 2.0 * sheet.pore_diameter);
        assert!(strut.permeability > sheet.permeability);
    }

    #[test]
    fn a_sheet_tpms_has_two_labyrinths_and_an_open_cell_foam_has_one() {
        use crate::geometries::lattice::Stochastic;
        let size = Vector3::new(20.0, 20.0, 20.0);
        let build = |kind| {
            crate::geometries::lattice::Lattice::new(kind)
                .size(size)
                .cells([4, 4, 4])
                .seed(3)
                .fit_relative_density(0.25)
                .metrics()
        };

        // A gyroid surface cuts space in two and the two halves never meet.
        // That is the whole reason to put one in a heat exchanger.
        let gyroid = build(LatticeKind::Tpms(Tpms::Gyroid));
        assert!(
            (gyroid.largest_void_fraction - 0.5).abs() < 0.1,
            "{}",
            gyroid.largest_void_fraction
        );
        assert!(gyroid.percolates.iter().all(|&p| p), "{:?}", gyroid.percolates);
        assert!(gyroid.open_porosity > 0.95 * gyroid.porosity);

        // An open-cell foam is all one space.
        let foam = build(LatticeKind::Stochastic(Stochastic::Voronoi));
        assert!(foam.largest_void_fraction > 0.9, "{}", foam.largest_void_fraction);
        assert!(foam.percolates.iter().all(|&p| p), "{:?}", foam.percolates);
    }

    #[test]
    fn a_closed_cell_foam_is_porous_and_goes_nowhere() {
        use crate::geometries::lattice::Stochastic;
        let m = crate::geometries::lattice::Lattice::new(LatticeKind::Stochastic(
            Stochastic::VoronoiWall,
        ))
        .size(Vector3::new(24.0, 24.0, 24.0))
        .cells([4, 4, 4])
        .seed(5)
        .fit_relative_density(0.2)
        .metrics();

        // Four fifths open — and none of it goes anywhere. Every bubble is
        // its own space, so the largest is a fortieth of the void rather than
        // all of it, and nothing crosses the part.
        assert!(m.porosity > 0.75, "{}", m.porosity);
        assert!(m.largest_void_fraction < 0.1, "{}", m.largest_void_fraction);
        assert!(!m.percolates.iter().any(|&p| p), "{:?}", m.percolates);
        // A good share is sealed. Only a share, because the part's own surface
        // cuts the bubbles it passes through and those really are drainable —
        // and at four cells across, most bubbles are on a face.
        assert!(
            m.closed_porosity > 0.2 * m.porosity,
            "open {} closed {} of {}",
            m.open_porosity,
            m.closed_porosity,
            m.porosity
        );
        // And the two halves still add back up to the whole.
        assert!((m.open_porosity + m.closed_porosity - m.porosity).abs() < 1e-5);
    }

    #[test]
    fn a_skin_traps_the_powder() {
        let build = |skin: f32| {
            crate::geometries::lattice::Lattice::new(LatticeKind::Tpms(Tpms::Gyroid))
                .size(Vector3::new(16.0, 16.0, 16.0))
                .cells([4, 4, 4])
                .fit_relative_density(0.3)
                .skin(skin)
                .metrics()
        };
        let bare = build(0.0);
        let sealed = build(0.8);
        assert!(bare.open_porosity > 0.9 * bare.porosity, "{bare:?}");
        // Wrapped in a solid wall, there is no way out of any of it.
        assert!(
            sealed.open_porosity < 0.05 * sealed.porosity,
            "open {} of porosity {}",
            sealed.open_porosity,
            sealed.porosity
        );
        assert!(!sealed.percolates.iter().any(|&p| p), "{:?}", sealed.percolates);
    }

    #[test]
    fn thicker_walls_mean_fatter_ligaments_and_smaller_pores() {
        let build = |density: f32| {
            crate::geometries::lattice::Lattice::new(LatticeKind::Strut(Strut::Bcc))
                .size(Vector3::new(12.0, 12.0, 12.0))
                .cells([3, 3, 3])
                .fit_relative_density(density)
                .metrics()
        };
        let light = build(0.1);
        let heavy = build(0.4);
        assert!(heavy.ligament_thickness > light.ligament_thickness);
        assert!(heavy.pore_diameter < light.pore_diameter);
        assert!(heavy.permeability < light.permeability);
    }

    /// The measured ligament has to be the strut that was asked for.
    #[test]
    fn a_strut_measures_the_thickness_it_was_given() {
        let lattice = crate::geometries::lattice::Lattice::new(LatticeKind::Strut(Strut::Cubic))
            .size(Vector3::new(20.0, 20.0, 20.0))
            .cells([4, 4, 4])
            .thickness(1.5);
        let m = lattice.metrics();
        // A simple-cubic cell is round struts crossing at right angles, and
        // the largest sphere in it is the strut itself — three perpendicular
        // cylinders meeting at a node hold no bigger sphere than one of them
        // does. What is left is the grid's own error, a few percent of a
        // sample.
        assert!(
            (m.ligament_thickness - 1.5).abs() < 0.15,
            "{}",
            m.ligament_thickness
        );
    }
}
