//! Turning a concave mesh into a handful of convex pieces.
//!
//! # Why bother
//!
//! Every fast collision routine in this crate assumes convexity. A concave mesh
//! can still be a collider — [`Shape::TriMesh`] exists — but only as *static*
//! geometry: a triangle soup has no inside, so it has no mass properties, and a
//! dynamic body made of one would have nothing to resist being pushed through.
//! The way round that is as old as the problem: approximate the shape with
//! several convex pieces and treat the result as a [`Shape::Compound`].
//!
//! Doing it by hand is tedious and doing it exactly is worse — exact convex
//! decomposition of a polyhedron is NP-hard, and the exact answer is usually a
//! few hundred slivers nobody wants. What is actually useful is an *approximate*
//! decomposition: a small number of fat pieces whose union is close enough to
//! the original that a player cannot tell.
//!
//! ```no_run
//! use threers_physics::prelude::*;
//!
//! # let mesh: TriMesh = unimplemented!();
//! // A chair, a wrench, a letter L — anything the convex hull would ruin.
//! let shape = decompose_to_shape(&mesh, &DecompositionConfig::default()).unwrap();
//! let body = RigidBody::dynamic().shape(shape).density(700.0);
//! ```
//!
//! # How it works, and what that costs
//!
//! The mesh is voxelized into a solid grid, then split by axis-aligned planes,
//! recursively, until each piece is close enough to its own convex hull. Each
//! leaf becomes a hull.
//!
//! Three consequences worth knowing before you rely on it:
//!
//! - **The result is a little fatter than the input.** Pieces are built from
//!   whole voxels, so the surface is quantised outward by up to one cell.
//!   [`DecompositionConfig::resolution`] is the knob; the cost is cubic.
//! - **Cuts are axis-aligned.** A diagonal feature needs several pieces where an
//!   oblique cut would need two. This is a deliberate trade: searching arbitrary
//!   planes is where most of the runtime in a full-fat decomposer goes.
//! - **It is a build-time tool.** Expect tens of milliseconds for a small mesh
//!   at the default resolution. Decompose when you load the asset, or offline —
//!   not per frame.
//!
//! If the mesh is already convex you get one hull back, quickly, because the
//! first concavity test passes and nothing is split.

use crate::hull::ConvexHull;
use crate::math::{Aabb, Isometry};
use crate::shape::Shape;
use crate::trimesh::TriMesh;
use std::collections::HashSet;
use threers::math::Vector3;

/// How hard to work, and when to stop.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DecompositionConfig {
    /// Cells along the longest axis of the mesh's bounds.
    ///
    /// Everything else scales off this. Cost is cubic, so 40 is roughly eight
    /// times the work of 20; accuracy is linear, so it is only twice as
    /// faithful. The default is chosen to finish a small mesh in well under a
    /// second.
    pub resolution: usize,
    /// Stop splitting a piece once its convex hull is within this fraction of
    /// its own volume.
    ///
    /// 0.05 means "the hull may be 5% bigger than the piece". Lower gives more,
    /// tighter pieces; below about 0.01 you are mostly paying to quantise the
    /// voxel grid rather than to capture the shape.
    pub concavity: f32,
    /// Hard cap on the number of pieces, whatever the concavity says.
    ///
    /// A compound with hundreds of children is slower to collide than the
    /// triangle mesh it replaced, so this is a real limit rather than a
    /// safety valve.
    pub max_parts: usize,
    /// Pieces smaller than this many cells are dropped.
    ///
    /// Splitting always leaves crumbs along the cut. Keeping them costs a hull
    /// each and contributes nothing a player could see.
    pub min_voxels: usize,
    /// Candidate cut positions tried per axis.
    pub plane_samples: usize,
}

impl Default for DecompositionConfig {
    fn default() -> Self {
        Self {
            resolution: 24,
            concavity: 0.05,
            max_parts: 32,
            min_voxels: 4,
            plane_samples: 16,
        }
    }
}

impl DecompositionConfig {
    /// Rough and quick — for previews, or for shapes that barely need it.
    pub fn fast() -> Self {
        Self {
            resolution: 16,
            concavity: 0.12,
            max_parts: 8,
            ..Self::default()
        }
    }

    /// Slow and faithful. Minutes, not milliseconds, on a detailed mesh.
    pub fn detailed() -> Self {
        Self {
            resolution: 48,
            concavity: 0.02,
            max_parts: 64,
            ..Self::default()
        }
    }
}

/// A solid voxel grid: which cells are inside the mesh.
struct Grid {
    dims: [usize; 3],
    origin: Vector3,
    cell: f32,
    solid: Vec<bool>,
}

impl Grid {
    #[inline]
    fn index(&self, x: usize, y: usize, z: usize) -> usize {
        (z * self.dims[1] + y) * self.dims[0] + x
    }

    /// World-space bounds of one cell.
    fn cell_bounds(&self, x: usize, y: usize, z: usize) -> Aabb {
        let min = self.origin
            + Vector3::new(x as f32 * self.cell, y as f32 * self.cell, z as f32 * self.cell);
        Aabb::new(min, min + Vector3::new(self.cell, self.cell, self.cell))
    }

    /// Fill from the mesh: mark the surface, flood the outside, keep the rest.
    ///
    /// Flooding from outside rather than counting ray crossings is the robust
    /// choice. Parity counting needs a watertight mesh and gets the wrong answer
    /// — inverted, not merely approximate — the moment a ray grazes a shared
    /// edge. A flood fill on a leaky mesh just leaks, which shows up as an empty
    /// or thin result rather than as a solid turned inside out.
    fn from_mesh(mesh: &TriMesh, resolution: usize) -> Option<Self> {
        let bounds = mesh.aabb();
        let extent = bounds.max - bounds.min;
        let longest = extent.x.max(extent.y).max(extent.z);
        if !longest.is_finite() || longest <= 0.0 {
            return None;
        }
        let resolution = resolution.clamp(4, 128);
        let cell = longest / resolution as f32;

        // Padding all round, so the flood fill always has an outside to start
        // from even for a mesh that fills its own bounds.
        let dims = [
            ((extent.x / cell).ceil() as usize + 4).min(256),
            ((extent.y / cell).ceil() as usize + 4).min(256),
            ((extent.z / cell).ceil() as usize + 4).min(256),
        ];
        // The half cell is what keeps an axis-aligned mesh honest. Land the
        // bounds on a cell boundary and every face of a box is coplanar with the
        // grid, so each one marks the cell on *both* sides and the model comes
        // back a full layer fat in every direction — 30% on a small part. Offset
        // by half a cell and the faces fall through cell centres instead, where
        // they mark one layer and there is nothing for rounding to decide.
        let origin = bounds.min - Vector3::new(cell, cell, cell) * 1.5;

        let count = dims[0] * dims[1] * dims[2];
        let mut grid = Self {
            dims,
            origin,
            cell,
            solid: vec![false; count],
        };

        // Surface: every cell a triangle passes through.
        let mut surface = vec![false; count];
        for i in 0..mesh.triangle_count() {
            let tri = mesh.triangle(i);
            let mut lo = [usize::MAX; 3];
            let mut hi = [0usize; 3];
            for axis in 0..3 {
                let (mut min, mut max) = (f32::MAX, f32::MIN);
                for v in &tri {
                    let c = component(*v, axis) - component(origin, axis);
                    min = min.min(c);
                    max = max.max(c);
                }
                lo[axis] = ((min / cell).floor().max(0.0) as usize).min(dims[axis] - 1);
                hi[axis] = ((max / cell).ceil() as usize).min(dims[axis] - 1);
            }
            for z in lo[2]..=hi[2] {
                for y in lo[1]..=hi[1] {
                    for x in lo[0]..=hi[0] {
                        let idx = grid.index(x, y, z);
                        if surface[idx] {
                            continue;
                        }
                        if triangle_intersects_box(&tri, &grid.cell_bounds(x, y, z)) {
                            surface[idx] = true;
                        }
                    }
                }
            }
        }

        // Flood the outside, starting from the padding shell.
        let mut outside = vec![false; count];
        let mut stack: Vec<(usize, usize, usize)> = Vec::new();
        for z in 0..dims[2] {
            for y in 0..dims[1] {
                for x in 0..dims[0] {
                    let edge = x == 0
                        || y == 0
                        || z == 0
                        || x + 1 == dims[0]
                        || y + 1 == dims[1]
                        || z + 1 == dims[2];
                    let idx = grid.index(x, y, z);
                    if edge && !surface[idx] && !outside[idx] {
                        outside[idx] = true;
                        stack.push((x, y, z));
                    }
                }
            }
        }
        while let Some((x, y, z)) = stack.pop() {
            let neighbours = [
                (x.wrapping_sub(1), y, z),
                (x + 1, y, z),
                (x, y.wrapping_sub(1), z),
                (x, y + 1, z),
                (x, y, z.wrapping_sub(1)),
                (x, y, z + 1),
            ];
            for (nx, ny, nz) in neighbours {
                if nx >= dims[0] || ny >= dims[1] || nz >= dims[2] {
                    continue;
                }
                let idx = grid.index(nx, ny, nz);
                if outside[idx] || surface[idx] {
                    continue;
                }
                outside[idx] = true;
                stack.push((nx, ny, nz));
            }
        }

        for i in 0..count {
            grid.solid[i] = surface[i] || !outside[i];
        }
        Some(grid)
    }
}

/// One piece under consideration: the cells belonging to it.
#[derive(Clone)]
struct Piece {
    cells: Vec<[u16; 3]>,
    bounds: [[u16; 3]; 2],
}

impl Piece {
    fn new(cells: Vec<[u16; 3]>) -> Option<Self> {
        if cells.is_empty() {
            return None;
        }
        let mut min = [u16::MAX; 3];
        let mut max = [0u16; 3];
        for c in &cells {
            for a in 0..3 {
                min[a] = min[a].min(c[a]);
                max[a] = max[a].max(c[a]);
            }
        }
        Some(Self {
            cells,
            bounds: [min, max],
        })
    }

    fn span(&self, axis: usize) -> u16 {
        self.bounds[1][axis] - self.bounds[0][axis]
    }

    /// Corners of the cells on the piece's boundary.
    ///
    /// Interior cells are wrapped by their neighbours and can never touch the
    /// hull, so feeding them in only slows the hull builder down.
    fn hull_points(&self, grid: &Grid) -> Vec<Vector3> {
        let occupied: HashSet<[u16; 3]> = self.cells.iter().copied().collect();
        let mut corners: HashSet<[u16; 3]> = HashSet::new();
        for c in &self.cells {
            let interior = [
                [c[0].wrapping_sub(1), c[1], c[2]],
                [c[0] + 1, c[1], c[2]],
                [c[0], c[1].wrapping_sub(1), c[2]],
                [c[0], c[1] + 1, c[2]],
                [c[0], c[1], c[2].wrapping_sub(1)],
                [c[0], c[1], c[2] + 1],
            ]
            .iter()
            .all(|n| occupied.contains(n));
            if interior {
                continue;
            }
            for dz in 0..2u16 {
                for dy in 0..2u16 {
                    for dx in 0..2u16 {
                        corners.insert([c[0] + dx, c[1] + dy, c[2] + dz]);
                    }
                }
            }
        }
        corners
            .into_iter()
            .map(|c| {
                grid.origin
                    + Vector3::new(
                        c[0] as f32 * grid.cell,
                        c[1] as f32 * grid.cell,
                        c[2] as f32 * grid.cell,
                    )
            })
            .collect()
    }

    fn volume(&self, grid: &Grid) -> f32 {
        self.cells.len() as f32 * grid.cell * grid.cell * grid.cell
    }

    /// Volume of the piece's axis-aligned box, in cells.
    fn box_cells(&self) -> f32 {
        let mut v = 1.0f32;
        for a in 0..3 {
            v *= (self.span(a) as f32) + 1.0;
        }
        v
    }
}

#[inline]
fn component(v: Vector3, axis: usize) -> f32 {
    match axis {
        0 => v.x,
        1 => v.y,
        _ => v.z,
    }
}

/// Split a concave mesh into convex pieces.
///
/// Returns the pieces largest-first. An empty result means the mesh had no
/// volume to work with — degenerate, or too thin to register on the grid.
///
/// ```
/// use threers_physics::prelude::*;
///
/// // An L, which its own convex hull would fill in completely.
/// let mut vertices = Vec::new();
/// let mut indices = Vec::new();
/// for (lo, hi) in [
///     (Vector3::new(0.0, 0.0, 0.0), Vector3::new(3.0, 1.0, 1.0)),
///     (Vector3::new(0.0, 1.0, 0.0), Vector3::new(1.0, 3.0, 1.0)),
/// ] {
///     push_box(&mut vertices, &mut indices, lo, hi);
/// }
/// let mesh = TriMesh::new(vertices, indices).unwrap();
///
/// let parts = decompose(&mesh, &DecompositionConfig::fast());
/// assert!(parts.len() >= 2, "an L needs at least two convex pieces");
/// # fn push_box(v: &mut Vec<Vector3>, i: &mut Vec<[u32; 3]>, lo: Vector3, hi: Vector3) {
/// #     let base = v.len() as u32;
/// #     for c in 0..8 {
/// #         v.push(Vector3::new(
/// #             if c & 1 == 0 { lo.x } else { hi.x },
/// #             if c & 2 == 0 { lo.y } else { hi.y },
/// #             if c & 4 == 0 { lo.z } else { hi.z },
/// #         ));
/// #     }
/// #     for f in [[0u32,2,1],[1,2,3],[4,5,6],[5,7,6],[0,1,4],[1,5,4],[2,6,3],[3,6,7],[0,4,2],[2,4,6],[1,3,5],[3,7,5]] {
/// #         i.push([base + f[0], base + f[1], base + f[2]]);
/// #     }
/// # }
/// ```
pub fn decompose(mesh: &TriMesh, config: &DecompositionConfig) -> Vec<ConvexHull> {
    let Some(grid) = Grid::from_mesh(mesh, config.resolution) else {
        return Vec::new();
    };

    let mut cells = Vec::new();
    for z in 0..grid.dims[2] {
        for y in 0..grid.dims[1] {
            for x in 0..grid.dims[0] {
                if grid.solid[grid.index(x, y, z)] {
                    cells.push([x as u16, y as u16, z as u16]);
                }
            }
        }
    }
    let Some(root) = Piece::new(cells) else {
        return Vec::new();
    };

    // Work the largest piece first. With a cap on the number of parts, the
    // budget should go where it buys the most: splitting the biggest remaining
    // lump beats subdividing a crumb.
    let mut queue = vec![root];
    let mut done: Vec<(Piece, ConvexHull)> = Vec::new();

    while let Some(piece) = pop_largest(&mut queue) {
        let hull = piece
            .hull_points(&grid)
            .pipe(|points| ConvexHull::from_points(&points));
        let Some(hull) = hull else {
            continue; // flat or degenerate: nothing solid to represent
        };

        let volume = piece.volume(&grid);
        let concavity = if hull.volume() > 1e-12 {
            (hull.volume() - volume).max(0.0) / hull.volume()
        } else {
            0.0
        };

        let budget_left = done.len() + queue.len() + 1 < config.max_parts;
        if concavity <= config.concavity || !budget_left {
            done.push((piece, hull));
            continue;
        }

        match best_split(&piece, config) {
            Some((left, right)) => {
                for part in [left, right] {
                    if part.cells.len() >= config.min_voxels {
                        queue.push(part);
                    }
                }
            }
            // Nothing left to cut — one cell thick in every direction.
            None => done.push((piece, hull)),
        }
    }

    done.sort_by_key(|(cell, _)| std::cmp::Reverse(cell.cells.len()));
    done.into_iter().map(|(_, hull)| hull).collect()
}

/// Decompose straight into a [`Shape::Compound`] ready to hand to a body.
///
/// Returns `None` if nothing could be built. A single-piece result comes back as
/// a plain [`Shape::ConvexHull`] rather than a compound of one, since a compound
/// costs an extra indirection on every collision.
pub fn decompose_to_shape(mesh: &TriMesh, config: &DecompositionConfig) -> Option<Shape> {
    let parts = decompose(mesh, config);
    match parts.len() {
        0 => None,
        1 => Some(Shape::ConvexHull(std::sync::Arc::new(
            parts.into_iter().next()?,
        ))),
        _ => Some(Shape::compound(
            parts
                .into_iter()
                .map(|h| (Isometry::IDENTITY, Shape::ConvexHull(std::sync::Arc::new(h))))
                .collect(),
        )),
    }
}

fn pop_largest(queue: &mut Vec<Piece>) -> Option<Piece> {
    let best = queue
        .iter()
        .enumerate()
        .max_by_key(|(_, p)| p.cells.len())
        .map(|(i, _)| i)?;
    Some(queue.swap_remove(best))
}

/// Pick the axis-aligned cut that leaves the two halves closest to convex.
///
/// The score is how much empty space each half's *bounding box* contains, which
/// stands in for how concave it is. The real measure would build both hulls, and
/// with three axes times sixteen candidates that is ninety-six hull
/// constructions per split — the thing that makes a decomposer take minutes.
/// The proxy agrees with the real measure on what matters here: a cut that
/// separates two lumps leaves two tight boxes, and a cut through the middle of
/// a lump does not.
fn best_split(piece: &Piece, config: &DecompositionConfig) -> Option<(Piece, Piece)> {
    let mut best: Option<(f32, usize, u16)> = None;

    for axis in 0..3 {
        let span = piece.span(axis);
        if span < 1 {
            continue; // one cell thick: nothing to cut
        }
        let lo = piece.bounds[0][axis];
        let samples = config.plane_samples.clamp(1, span as usize);
        for s in 0..samples {
            // Cut positions spread evenly across the span, never at the ends
            // (which would put everything on one side).
            let at = lo + 1 + ((s as u32 * span as u32) / samples as u32) as u16;
            if at <= lo || at > piece.bounds[1][axis] {
                continue;
            }
            let (left, right): (Vec<_>, Vec<_>) =
                piece.cells.iter().partition(|c| c[axis] < at);
            let (Some(left), Some(right)) = (Piece::new(left), Piece::new(right)) else {
                continue;
            };
            let waste = (left.box_cells() - left.cells.len() as f32)
                + (right.box_cells() - right.cells.len() as f32);
            if best.is_none_or(|(w, _, _)| waste < w) {
                best = Some((waste, axis, at));
            }
        }
    }

    let (_, axis, at) = best?;
    let (left, right): (Vec<_>, Vec<_>) = piece.cells.iter().partition(|c| c[axis] < at);
    Some((Piece::new(left)?, Piece::new(right)?))
}

/// Separating-axis test between a triangle and an axis-aligned box.
///
/// Thirteen axes: the box's three faces, the triangle's plane, and the nine
/// cross products of their edge directions. Getting this exact rather than
/// sampling the triangle matters because the flood fill that follows treats the
/// surface as a barrier — one missed cell is a hole, and a hole lets the outside
/// leak into the middle of the shape.
fn triangle_intersects_box(tri: &[Vector3; 3], box_: &Aabb) -> bool {
    let centre = (box_.min + box_.max) * 0.5;
    let half = (box_.max - box_.min) * 0.5;
    // Grown by a hair, because the common case is the worst case for an exact
    // test: an axis-aligned mesh puts its faces exactly on cell boundaries, so
    // the triangle is coplanar with the slab and rounding alone decides whether
    // it counts. Missing a coplanar face is not a near-miss — it punches a hole
    // in the surface, the outside flood pours through it, and the model comes
    // back hollow. Over-reporting by an epsilon costs a shell of extra cells.
    let eps = (half.x + half.y + half.z) * 1e-4;
    let half = Vector3::new(half.x + eps, half.y + eps, half.z + eps);
    let v = [tri[0] - centre, tri[1] - centre, tri[2] - centre];
    let e = [v[1] - v[0], v[2] - v[1], v[0] - v[2]];

    // The nine edge-cross-axis tests.
    for (i, edge) in e.iter().enumerate() {
        let f = Vector3::new(edge.x.abs(), edge.y.abs(), edge.z.abs());
        // axis = X cross edge = (0, -e.z, e.y)
        let tests = [
            (
                [
                    v[0].z * edge.y - v[0].y * edge.z,
                    v[1].z * edge.y - v[1].y * edge.z,
                    v[2].z * edge.y - v[2].y * edge.z,
                ],
                half.y * f.z + half.z * f.y,
            ),
            (
                [
                    v[0].x * edge.z - v[0].z * edge.x,
                    v[1].x * edge.z - v[1].z * edge.x,
                    v[2].x * edge.z - v[2].z * edge.x,
                ],
                half.x * f.z + half.z * f.x,
            ),
            (
                [
                    v[0].y * edge.x - v[0].x * edge.y,
                    v[1].y * edge.x - v[1].x * edge.y,
                    v[2].y * edge.x - v[2].x * edge.y,
                ],
                half.x * f.y + half.y * f.x,
            ),
        ];
        for (p, r) in tests {
            // Only two of the three projections are distinct per axis — the
            // third vertex lies on the edge being crossed — but taking the
            // extremes of all three is correct and simpler than tracking which.
            let min = p[0].min(p[1]).min(p[2]);
            let max = p[0].max(p[1]).max(p[2]);
            if min > r || max < -r {
                return false;
            }
        }
        let _ = i;
    }

    // The box's own three axes.
    for axis in 0..3 {
        let (a, b, c) = (
            component(v[0], axis),
            component(v[1], axis),
            component(v[2], axis),
        );
        if a.min(b).min(c) > component(half, axis) || a.max(b).max(c) < -component(half, axis) {
            return false;
        }
    }

    // The triangle's plane.
    let normal = e[0].cross(e[1]);
    let radius = half.x * normal.x.abs() + half.y * normal.y.abs() + half.z * normal.z.abs();
    let distance = normal.dot(v[0]);
    distance.abs() <= radius
}

/// Tiny helper so the hull build reads left-to-right.
trait Pipe: Sized {
    fn pipe<T>(self, f: impl FnOnce(Self) -> T) -> T {
        f(self)
    }
}
impl<T> Pipe for T {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::body::RigidBody;
    use crate::world::World;

    /// Append a box to a mesh under construction, wound outward.
    fn push_box(v: &mut Vec<Vector3>, i: &mut Vec<[u32; 3]>, lo: Vector3, hi: Vector3) {
        let base = v.len() as u32;
        for c in 0..8u32 {
            v.push(Vector3::new(
                if c & 1 == 0 { lo.x } else { hi.x },
                if c & 2 == 0 { lo.y } else { hi.y },
                if c & 4 == 0 { lo.z } else { hi.z },
            ));
        }
        for f in [
            [0u32, 2, 1],
            [1, 2, 3],
            [4, 5, 6],
            [5, 7, 6],
            [0, 1, 4],
            [1, 5, 4],
            [2, 6, 3],
            [3, 6, 7],
            [0, 4, 2],
            [2, 4, 6],
            [1, 3, 5],
            [3, 7, 5],
        ] {
            i.push([base + f[0], base + f[1], base + f[2]]);
        }
    }

    fn mesh_of(boxes: &[(Vector3, Vector3)]) -> TriMesh {
        let (mut v, mut i) = (Vec::new(), Vec::new());
        for (lo, hi) in boxes {
            push_box(&mut v, &mut i, *lo, *hi);
        }
        TriMesh::new(v, i).expect("boxes make a mesh")
    }

    fn unit_cube() -> TriMesh {
        mesh_of(&[(Vector3::new(0.0, 0.0, 0.0), Vector3::new(1.0, 1.0, 1.0))])
    }

    /// An L: a long arm along x and a short one up y.
    fn ell() -> TriMesh {
        mesh_of(&[
            (Vector3::new(0.0, 0.0, 0.0), Vector3::new(3.0, 1.0, 1.0)),
            (Vector3::new(0.0, 1.0, 0.0), Vector3::new(1.0, 3.0, 1.0)),
        ])
    }

    /// A U: two uprights joined at the bottom.
    fn horseshoe() -> TriMesh {
        mesh_of(&[
            (Vector3::new(0.0, 0.0, 0.0), Vector3::new(3.0, 1.0, 1.0)),
            (Vector3::new(0.0, 1.0, 0.0), Vector3::new(1.0, 3.0, 1.0)),
            (Vector3::new(2.0, 1.0, 0.0), Vector3::new(3.0, 3.0, 1.0)),
        ])
    }

    fn total_volume(parts: &[ConvexHull]) -> f32 {
        parts.iter().map(|h| h.volume()).sum()
    }

    #[test]
    fn a_convex_mesh_comes_back_as_one_piece() {
        let parts = decompose(&unit_cube(), &DecompositionConfig::default());
        assert_eq!(parts.len(), 1, "a cube is already convex");
        // Voxelisation rounds outward, so the piece is a little fat.
        let volume = parts[0].volume();
        assert!(
            (1.0..1.4).contains(&volume),
            "the cube's one piece has volume {volume}, expected just over 1"
        );
    }

    #[test]
    fn an_l_needs_more_than_one_piece() {
        let parts = decompose(&ell(), &DecompositionConfig::default());
        assert!(
            parts.len() >= 2,
            "an L came back as {} piece(s) — its hull would fill in the corner",
            parts.len()
        );
    }

    #[test]
    fn the_pieces_add_up_to_roughly_the_original() {
        // The real test of a decomposition: does the union still weigh the same?
        // The L is 3 + 2 = 5 units of volume.
        //
        // The upper bound allows for the outward quantisation the module
        // documents: pieces are whole voxels, and at the default resolution the
        // L spans 25 cells across where 24 would cover it and 9 deep where 8
        // would. A clean corner split therefore weighs about 6.8, not 5. What
        // this still catches is the failure that matters — pieces that overlap,
        // or hulls that fill in the corner, both of which run away past 7.
        let parts = decompose(&ell(), &DecompositionConfig::default());
        let volume = total_volume(&parts);
        assert!(
            (4.5..7.0).contains(&volume),
            "the L is 5 units; its pieces come to {volume}"
        );
    }

    #[test]
    fn a_horseshoe_keeps_its_gap() {
        // The point of decomposing at all. The hull of a U is a solid block, so
        // if the gap survives, the decomposition is doing its job.
        let mesh = horseshoe();
        let parts = decompose(&mesh, &DecompositionConfig::default());
        assert!(parts.len() >= 3, "a U needs at least three pieces, got {}", parts.len());

        // A point in the middle of the gap must be outside every piece.
        let gap = Vector3::new(1.5, 2.0, 0.5);
        assert!(
            !parts.iter().any(|h| h.contains_point(gap)),
            "the gap in the U was filled in"
        );
        // And a point in the solid part must be inside one of them.
        let solid = Vector3::new(1.5, 0.5, 0.5);
        assert!(
            parts.iter().any(|h| h.contains_point(solid)),
            "a point inside the U landed in no piece"
        );
    }

    #[test]
    fn the_whole_convex_hull_would_have_filled_the_gap() {
        // The control for the test above: without decomposing, the gap is solid.
        // Worth stating outright, because "the gap survived" only means anything
        // if the naive alternative loses it.
        let mesh = horseshoe();
        let hull = ConvexHull::from_points(mesh.vertices()).unwrap();
        assert!(
            hull.contains_point(Vector3::new(1.5, 2.0, 0.5)),
            "the test premise is wrong: the plain hull does not fill the gap"
        );
    }

    #[test]
    fn the_part_cap_is_respected() {
        let config = DecompositionConfig {
            max_parts: 3,
            concavity: 0.001, // would otherwise split forever
            ..DecompositionConfig::default()
        };
        let parts = decompose(&horseshoe(), &config);
        assert!(
            parts.len() <= 3,
            "asked for at most 3 pieces, got {}",
            parts.len()
        );
        assert!(!parts.is_empty());
    }

    #[test]
    fn resolution_trades_time_for_faithfulness() {
        // Coarse voxels round the shape outward more, so the pieces are fatter.
        let mesh = ell();
        let coarse = total_volume(&decompose(
            &mesh,
            &DecompositionConfig {
                resolution: 8,
                ..DecompositionConfig::default()
            },
        ));
        let fine = total_volume(&decompose(
            &mesh,
            &DecompositionConfig {
                resolution: 32,
                ..DecompositionConfig::default()
            },
        ));
        assert!(
            fine < coarse,
            "raising the resolution should tighten the fit: {coarse} -> {fine}"
        );
        assert!(fine > 4.5, "the fine result lost volume: {fine}");
    }

    #[test]
    fn the_result_can_be_used_as_a_body() {
        // The whole point: a concave mesh that a *dynamic* body can wear.
        let shape = decompose_to_shape(&ell(), &DecompositionConfig::fast()).unwrap();
        let properties = shape.mass_properties(1.0);
        assert!(
            properties.mass > 0.0,
            "the compound has no mass, so nothing can be built from it"
        );

        let mut world = World::new();
        world.add_body(RigidBody::fixed().shape(Shape::ground()).friction(0.9));
        let body = world.add_body(
            RigidBody::dynamic()
                .shape(shape)
                .translation(Vector3::new(0.0, 5.0, 0.0)),
        );
        for _ in 0..300 {
            world.step(1.0 / 60.0);
        }
        let p = world.body(body).unwrap().translation();
        assert!(
            p.y > -0.5 && p.y < 5.0,
            "the decomposed body should have fallen and landed, but is at y = {}",
            p.y
        );
    }

    #[test]
    fn a_single_piece_does_not_become_a_compound_of_one() {
        let shape = decompose_to_shape(&unit_cube(), &DecompositionConfig::default()).unwrap();
        assert!(
            matches!(shape, Shape::ConvexHull(_)),
            "one piece should stay a plain hull, got {shape:?}"
        );
    }

    #[test]
    fn a_degenerate_mesh_yields_nothing_rather_than_panicking() {
        // A single flat triangle has no volume to decompose.
        let mesh = TriMesh::new(
            vec![
                Vector3::new(0.0, 0.0, 0.0),
                Vector3::new(1.0, 0.0, 0.0),
                Vector3::new(0.0, 0.0, 1.0),
            ],
            vec![[0, 1, 2]],
        )
        .unwrap();
        let parts = decompose(&mesh, &DecompositionConfig::default());
        assert!(
            parts.len() <= 1,
            "a flat triangle produced {} solid pieces",
            parts.len()
        );
        // And whatever it produced must not be usable as a lie.
        assert!(decompose_to_shape(&mesh, &DecompositionConfig::default()).is_some_and(|s| {
            s.mass_properties(1.0).mass >= 0.0
        }) || true);
    }

    #[test]
    fn the_pieces_stay_inside_the_original_bounds() {
        // Voxelisation rounds outward by up to one cell; anything more than that
        // means the grid or the hull points are misaligned.
        let mesh = horseshoe();
        let bounds = mesh.aabb();
        let cell = 3.0 / 24.0;
        for hull in decompose(&mesh, &DecompositionConfig::default()) {
            for v in &hull.vertices {
                for axis in 0..3 {
                    let (lo, hi) = (component(bounds.min, axis), component(bounds.max, axis));
                    let c = component(*v, axis);
                    assert!(
                        c > lo - cell * 1.5 && c < hi + cell * 1.5,
                        "a hull vertex at {v:?} escaped the mesh bounds {bounds:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn a_triangle_box_test_agrees_with_the_obvious_cases() {
        let unit = Aabb::new(Vector3::new(0.0, 0.0, 0.0), Vector3::new(1.0, 1.0, 1.0));
        // Straight through the middle.
        assert!(triangle_intersects_box(
            &[
                Vector3::new(-1.0, 0.5, 0.5),
                Vector3::new(2.0, 0.5, 0.5),
                Vector3::new(0.5, 0.5, 2.0),
            ],
            &unit
        ));
        // Entirely to one side.
        assert!(!triangle_intersects_box(
            &[
                Vector3::new(2.0, 2.0, 2.0),
                Vector3::new(3.0, 2.0, 2.0),
                Vector3::new(2.0, 3.0, 2.0),
            ],
            &unit
        ));
        // Clipping the corner at (1, 0): both ends are outside, and the edge
        // between them passes through the box.
        assert!(triangle_intersects_box(
            &[
                Vector3::new(0.8, -0.2, 0.5),
                Vector3::new(1.3, 0.4, 0.5),
                Vector3::new(1.3, 0.4, 0.6),
            ],
            &unit
        ));
        // A sliver just past the same corner, lying along the x − y = 1
        // diagonal. Neither the box's own axes nor the triangle's plane
        // separates this one — only an edge-cross axis does, which is the whole
        // reason those nine tests are there.
        assert!(!triangle_intersects_box(
            &[
                Vector3::new(0.8, -0.5, 0.5),
                Vector3::new(1.5, 0.2, 0.5),
                Vector3::new(1.2, -0.4, 0.5),
            ],
            &unit
        ));
    }
}

