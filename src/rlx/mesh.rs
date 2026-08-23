//! Mesh operations expressed as graphs over the positions tensor.
//!
//! Smoothing a mesh is a sparse matrix applied to a `[n, 3]` array — which is
//! the shape rlx is built for, and the reason the bridge converts attributes
//! and not only pixels. The topology (who neighbours whom) is derived once on
//! the host and becomes a `gather` index; the arithmetic then runs wherever
//! the graph runs, iteration after iteration, without the mesh coming back.
//!
//! ```no_run
//! use threers::geometries::SphereGeometry;
//! use threers::rlx::{mesh, preferred_device};
//!
//! let mut geometry = SphereGeometry::new(1.0, 32, 16);
//! mesh::taubin_smooth(&mut geometry, 20, 0.5, -0.53, preferred_device()).unwrap();
//! ```
//!
//! # Why Taubin as well as Laplacian
//!
//! Plain Laplacian smoothing moves every vertex towards the average of its
//! neighbours, and a closed surface that does that repeatedly *shrinks* — run
//! it long enough and a sphere collapses to its centre. [`crate::rlx::mesh::taubin_smooth`]
//! alternates a positive step with a slightly larger negative one, which
//! passes low frequencies and attenuates high ones instead of attenuating
//! everything. Use it unless you want the shrinkage.
//!
//! # Neighbours are the index buffer's, not the position's
//!
//! Two vertices at the same point are two vertices. A UV sphere's seam, a
//! cube's corners, any hard edge — these are split precisely so they can carry
//! different normals or UVs, and each copy smooths within its own half of the
//! surface. Welding them by position would silently undo the split the mesh
//! was built with, so this does not do it; weld first if that is what you
//! want.

use std::collections::BTreeSet;

use ::rlx::{DType, Device, Graph, GraphExt, Shape};

use crate::core::{BufferAttribute, BufferGeometry};
use crate::utils::compute_vertex_normals;

use super::session::GraphRunner;
use super::tensor::{Tensor, TensorError};

/// What a mesh operation can refuse.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MeshError {
    /// No `position` attribute, or one whose items are not 3 long.
    NoPositions,
    /// No index buffer. Smoothing needs to know which vertices are
    /// neighbours, and an unindexed mesh has no shared vertices to ask about
    /// — every triangle carries its own three, so the "surface" is dust.
    NotIndexed,
    /// A conversion refused; see [`TensorError`].
    Tensor(TensorError),
}

impl std::fmt::Display for MeshError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoPositions => write!(f, "geometry has no 3-component position attribute"),
            Self::NotIndexed => write!(f, "geometry has no index buffer"),
            Self::Tensor(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for MeshError {}

impl From<TensorError> for MeshError {
    fn from(e: TensorError) -> Self {
        Self::Tensor(e)
    }
}

/// Move every vertex a fraction `lambda` of the way towards the average of its
/// neighbours, `iterations` times.
///
/// `lambda` is the step: 0 does nothing, 1 replaces each vertex with the
/// average outright (and rings). Shrinks the surface — see the module note.
///
/// A geometry with no vertices is left alone. Smoothing nothing is nothing to
/// do, not an error — and not a graph either, since a zero-row tensor is not a
/// shape the backends accept.
pub fn laplacian_smooth(
    geometry: &mut BufferGeometry,
    iterations: usize,
    lambda: f32,
    device: Device,
) -> Result<(), MeshError> {
    let umbrella = Umbrella::of(geometry)?;
    if umbrella.count == 0 {
        return Ok(());
    }
    let mut runner = umbrella.compile(lambda, device);
    let mut positions = umbrella.positions(geometry)?;
    for _ in 0..iterations {
        positions = runner.run(&[("positions", &positions)]).remove(0);
    }
    write_back(geometry, &positions)
}

/// Taubin's λ|μ filter: a positive step of `lambda` followed by a negative
/// step of `mu` each iteration.
///
/// `mu` must be *more* negative than `lambda` is positive — the usual pair is
/// `(0.5, -0.53)`. The pass band is where `1 - λk` and `1 - μk` multiply to
/// about 1; making |μ| slightly larger is what puts that band at the low
/// frequencies and leaves the volume where it started.
pub fn taubin_smooth(
    geometry: &mut BufferGeometry,
    iterations: usize,
    lambda: f32,
    mu: f32,
    device: Device,
) -> Result<(), MeshError> {
    let umbrella = Umbrella::of(geometry)?;
    if umbrella.count == 0 {
        return Ok(());
    }
    let mut shrink = umbrella.compile(lambda, device);
    let mut inflate = umbrella.compile(mu, device);
    let mut positions = umbrella.positions(geometry)?;
    for _ in 0..iterations {
        positions = shrink.run(&[("positions", &positions)]).remove(0);
        positions = inflate.run(&[("positions", &positions)]).remove(0);
    }
    write_back(geometry, &positions)
}

/// The umbrella operator of a mesh, laid out for `gather`: for each vertex,
/// the indices of its neighbours and the weight each one carries.
///
/// Rows are padded to a common width because a tensor is rectangular and
/// valence is not. The padding entries point at vertex 0 and weigh **zero**,
/// so they gather a real (cheap, in-cache) value and then contribute nothing —
/// which is why the weights are `1/valence` per row rather than a single
/// `1/k`: the sum is over `k` terms and must still be the average of the
/// `valence` real ones.
struct Umbrella {
    count: usize,
    width: usize,
    /// `[count * width]`, f32-encoded vertex indices — the encoding `gather`
    /// takes.
    neighbours: Vec<f32>,
    /// `[count, width, 1]`, broadcast against `[count, width, 3]`.
    weights: Vec<f32>,
}

impl Umbrella {
    fn of(geometry: &BufferGeometry) -> Result<Self, MeshError> {
        let positions = geometry
            .get_attribute("position")
            .filter(|a| a.item_size == 3)
            .ok_or(MeshError::NoPositions)?;
        let index = geometry.index.as_ref().ok_or(MeshError::NotIndexed)?;
        let count = positions.count();

        let mut sets: Vec<BTreeSet<u32>> = vec![BTreeSet::new(); count];
        for tri in index.chunks_exact(3) {
            for (a, b) in [(0, 1), (1, 2), (2, 0)] {
                let (u, v) = (tri[a] as usize, tri[b] as usize);
                if u < count && v < count && u != v {
                    sets[u].insert(tri[b]);
                    sets[v].insert(tri[a]);
                }
            }
        }

        let width = sets.iter().map(|s| s.len()).max().unwrap_or(0).max(1);
        let mut neighbours = vec![0.0f32; count * width];
        let mut weights = vec![0.0f32; count * width];
        for (v, set) in sets.iter().enumerate() {
            if set.is_empty() {
                // A vertex no triangle references has nothing to move towards.
                // Its own neighbourhood is itself, which makes the update a
                // no-op — the alternative, an empty weighted sum, is an
                // "average" of zero, and would march the vertex to the origin
                // a fraction at a time. A UV sphere has two such vertices.
                neighbours[v * width] = v as f32;
                weights[v * width] = 1.0;
                continue;
            }
            let share = 1.0 / set.len() as f32;
            for (slot, n) in set.iter().enumerate() {
                neighbours[v * width + slot] = *n as f32;
                weights[v * width + slot] = share;
            }
        }
        Ok(Self {
            count,
            width,
            neighbours,
            weights,
        })
    }

    fn positions(&self, geometry: &BufferGeometry) -> Result<Tensor, MeshError> {
        let attr = geometry
            .get_attribute("position")
            .ok_or(MeshError::NoPositions)?;
        Ok(Tensor::from(attr))
    }

    /// `p ← p + step · (mean(neighbours) − p)`, compiled.
    fn compile(&self, step: f32, device: Device) -> GraphRunner {
        let mut g = Graph::new("umbrella");
        let p = g.input("positions", Shape::new(&[self.count, 3], DType::F32));
        let idx = g.param(
            "neighbours",
            Shape::new(&[self.count * self.width], DType::F32),
        );
        let w = g.param(
            "weights",
            Shape::new(&[self.count, self.width, 1], DType::F32),
        );

        let gathered = g.gather_(p, idx, 0); // [count·width, 3]
        let rows = g.reshape_(gathered, vec![self.count as i64, self.width as i64, 3]);
        // Weighted *sum*, not mean: the zero-weight padding must not count
        // towards the denominator, and `mean` over the axis would count it.
        let weighted = g.mul(rows, w);
        let average = g.sum(weighted, vec![1], false); // [count, 3]

        let delta = g.sub(average, p);
        let scaled = {
            let k = g.constant(step as f64, DType::F32);
            g.mul(delta, k)
        };
        let moved = g.add(p, scaled);
        g.set_outputs(vec![moved]);

        let mut runner = GraphRunner::new(g, device);
        runner.set_param("neighbours", &self.neighbours);
        runner.set_param("weights", &self.weights);
        runner
    }
}

/// Positions back onto the geometry, with normals recomputed.
///
/// Recomputed rather than kept: smoothing moves the surface, and a normal that
/// describes where the surface used to be is worse than no normal at all — it
/// shades the new geometry with the old lighting.
fn write_back(geometry: &mut BufferGeometry, positions: &Tensor) -> Result<(), MeshError> {
    let attr = BufferAttribute::try_from(positions)?;
    geometry.set_attribute("position", attr);
    compute_vertex_normals(geometry);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometries::SphereGeometry;
    use crate::rlx::preferred_device;

    /// Displace every vertex along its own direction by a deterministic
    /// pseudo-random amount — noise a smoother should remove.
    fn roughened_sphere() -> BufferGeometry {
        let mut geometry = SphereGeometry::new(1.0, 24, 12);
        let attr = geometry.get_attribute("position").unwrap().clone();
        let mut array = attr.array.clone();
        for (i, v) in array.chunks_exact_mut(3).enumerate() {
            let jitter = 1.0 + 0.12 * (((i * 2654435761) % 1000) as f32 / 1000.0 - 0.5);
            v[0] *= jitter;
            v[1] *= jitter;
            v[2] *= jitter;
        }
        geometry.set_attribute("position", BufferAttribute::new(array, 3));
        geometry
    }

    /// How bumpy the ball is, measured *locally*: the mean gap between a
    /// vertex's radius and its neighbours' average radius.
    ///
    /// Deviation from the global mean radius would be the wrong measure. A UV
    /// sphere's vertex spacing varies by latitude, so umbrella smoothing pulls
    /// the sparse equator inwards faster than the dense poles — the ball ends
    /// up smoother *and* less spherical, and a global metric reads that
    /// shrinkage as roughness. High-frequency noise is what smoothing claims
    /// to remove, so high-frequency noise is what the test measures.
    fn roughness(geometry: &BufferGeometry) -> f32 {
        let umbrella = Umbrella::of(geometry).unwrap();
        let radii: Vec<f32> = geometry
            .positions()
            .unwrap()
            .map(|p| (p.x * p.x + p.y * p.y + p.z * p.z).sqrt())
            .collect();
        let (mut total, mut counted) = (0.0f32, 0usize);
        for v in 0..umbrella.count {
            let row = v * umbrella.width..(v + 1) * umbrella.width;
            let neighbourhood: f32 = umbrella.neighbours[row.clone()]
                .iter()
                .zip(&umbrella.weights[row])
                .map(|(n, w)| w * radii[*n as usize])
                .sum();
            total += (radii[v] - neighbourhood).abs();
            counted += 1;
        }
        total / counted as f32
    }

    /// A square pyramid: four base corners and an apex, fanned into four
    /// triangles. One step at λ=1 must put the apex exactly on the average of
    /// the four corners it touches.
    #[test]
    fn one_full_step_lands_on_the_neighbour_average() {
        let mut geometry = BufferGeometry::new();
        geometry.set_attribute(
            "position",
            BufferAttribute::new(
                vec![
                    -1.0, 0.0, -1.0, // 0
                    1.0, 0.0, -1.0, // 1
                    1.0, 0.0, 1.0, // 2
                    -1.0, 0.0, 1.0, // 3
                    0.0, 4.0, 0.0, // 4 — the apex
                ],
                3,
            ),
        );
        geometry.set_index(vec![0, 1, 4, 1, 2, 4, 2, 3, 4, 3, 0, 4]);

        laplacian_smooth(&mut geometry, 1, 1.0, preferred_device()).unwrap();

        let p: Vec<f32> = geometry.get_attribute("position").unwrap().array.clone();
        let apex = &p[12..15];
        assert!(
            apex[0].abs() < 1e-5 && apex[1].abs() < 1e-5 && apex[2].abs() < 1e-5,
            "apex went to {apex:?}, not the centroid of the base"
        );
    }

    #[test]
    fn smoothing_removes_the_bumps() {
        let mut geometry = roughened_sphere();
        let before = roughness(&geometry);
        laplacian_smooth(&mut geometry, 8, 0.5, preferred_device()).unwrap();
        let after = roughness(&geometry);
        assert!(after < before * 0.5, "{before} → {after}");
    }

    #[test]
    fn laplacian_shrinks_and_taubin_does_not() {
        let volume = |g: &BufferGeometry| {
            g.positions()
                .unwrap()
                .map(|p| (p.x * p.x + p.y * p.y + p.z * p.z).sqrt())
                .sum::<f32>()
        };
        let device = preferred_device();

        let mut plain = roughened_sphere();
        let start = volume(&plain);
        laplacian_smooth(&mut plain, 24, 0.5, device).unwrap();
        let shrunk = volume(&plain);

        let mut taubin = roughened_sphere();
        taubin_smooth(&mut taubin, 12, 0.5, -0.53, device).unwrap();
        let held = volume(&taubin);

        assert!(shrunk < start * 0.97, "plain Laplacian did not shrink");
        assert!(
            held > start * 0.97,
            "Taubin lost volume: {start} → {held} (plain gave {shrunk})"
        );
    }

    #[test]
    fn normals_follow_the_new_surface() {
        let mut geometry = roughened_sphere();
        laplacian_smooth(&mut geometry, 4, 0.5, preferred_device()).unwrap();
        let normals = geometry.get_attribute("normal").expect("normals rebuilt");
        let positions = geometry.get_attribute("position").unwrap();
        // Only vertices some triangle uses: a normal is an average of face
        // normals, and a vertex with no faces has none to average. A UV
        // sphere carries two such vertices, and they are not evidence of
        // anything about smoothing.
        let used: std::collections::BTreeSet<u32> =
            geometry.index.as_ref().unwrap().iter().copied().collect();
        // A smoothed sphere's normals point outwards, i.e. along the position.
        for v in used {
            let n = &normals.array[v as usize * 3..v as usize * 3 + 3];
            let p = &positions.array[v as usize * 3..v as usize * 3 + 3];
            let dot = n[0] * p[0] + n[1] * p[1] + n[2] * p[2];
            assert!(dot > 0.0, "normal {n:?} at {p:?} points inwards");
        }
    }

    #[test]
    fn an_empty_geometry_is_left_alone_rather_than_compiled() {
        let mut geometry = BufferGeometry::new();
        geometry.set_attribute("position", BufferAttribute::new(Vec::new(), 3));
        geometry.set_index(Vec::new());
        let device = preferred_device();
        assert_eq!(laplacian_smooth(&mut geometry, 3, 0.5, device), Ok(()));
        assert_eq!(taubin_smooth(&mut geometry, 3, 0.5, -0.53, device), Ok(()));
        assert_eq!(geometry.get_attribute("position").unwrap().count(), 0);
    }

    #[test]
    fn an_unindexed_mesh_is_refused_rather_than_treated_as_dust() {
        let mut geometry = BufferGeometry::new();
        geometry.set_attribute("position", BufferAttribute::new(vec![0.0; 9], 3));
        assert_eq!(
            laplacian_smooth(&mut geometry, 1, 0.5, preferred_device()),
            Err(MeshError::NotIndexed)
        );
    }
}
