use std::collections::HashMap;

use crate::core::{BufferAttribute, BufferGeometry};
use crate::math::Vector3;

/// Marching-cubes isosurface extraction, as three.js's `MarchingCubes` does
/// it: sample a scalar field on a grid, put a vertex on every grid edge where
/// the field crosses `iso`, and join them into triangles by the 256-case
/// table (Paul Bourke's, the one three.js ships).
///
/// The field is *inside* where it is above `iso` — a metaball sum, or a
/// signed distance with its sign flipped (`radius - |p|` for a sphere). The
/// result is indexed: each grid edge's crossing is one vertex shared by every
/// triangle touching it, so a field closed inside the box gives a watertight
/// mesh. Normals are the field's gradient, pointing out of the inside, and
/// every triangle is wound counter-clockwise seen from outside.
///
/// ```
/// use threers::extras::MarchingCubes;
/// use threers::math::Vector3;
///
/// let ball = MarchingCubes::new(32, Vector3::new(2.0, 2.0, 2.0))
///     .extract(0.0, |x, y, z| 0.8 - (x * x + y * y + z * z).sqrt());
/// assert!(ball.get_attribute("normal").is_some());
/// assert!(ball.index.is_some());
/// ```
pub struct MarchingCubes {
    /// Cells along each side; the field is sampled at `resolution + 1`
    /// points per side.
    pub resolution: usize,
    /// The box sampled, centred on the origin.
    pub size: Vector3,
}

impl MarchingCubes {
    pub fn new(resolution: usize, size: Vector3) -> Self {
        Self { resolution, size }
    }

    /// The surface where `field(x, y, z) == iso`, in the box's coordinates.
    pub fn extract<F>(&self, iso: f32, field: F) -> BufferGeometry
    where
        F: Fn(f32, f32, f32) -> f32,
    {
        let res = self.resolution.max(2);
        let n = res + 1;
        let step = [
            self.size.x / res as f32,
            self.size.y / res as f32,
            self.size.z / res as f32,
        ];
        let at = |i: usize, axis: usize| (i as f32 - res as f32 * 0.5) * step[axis];
        let idx = |i: usize, j: usize, k: usize| i + n * (j + n * k);
        let mut grid = vec![0.0f32; n * n * n];
        for k in 0..n {
            for j in 0..n {
                for i in 0..n {
                    grid[idx(i, j, k)] = field(at(i, 0), at(j, 1), at(k, 2));
                }
            }
        }
        // The gradient at a sample, by central differences (one-sided on the
        // box's faces).
        let gradient = |p: [usize; 3]| -> [f32; 3] {
            [0, 1, 2].map(|axis| {
                let (mut lo, mut hi) = (p, p);
                lo[axis] = lo[axis].saturating_sub(1);
                hi[axis] = (hi[axis] + 1).min(res);
                let span = (hi[axis] - lo[axis]) as f32 * step[axis];
                (grid[idx(hi[0], hi[1], hi[2])] - grid[idx(lo[0], lo[1], lo[2])]) / span
            })
        };

        let mut positions: Vec<f32> = Vec::new();
        let mut normals: Vec<f32> = Vec::new();
        let mut indices: Vec<u32> = Vec::new();
        // One vertex per grid edge, keyed by its lower sample and its axis.
        let mut vertex_of: HashMap<(usize, u8), u32> = HashMap::new();
        for k in 0..res {
            for j in 0..res {
                for i in 0..res {
                    let mut case = 0usize;
                    for (c, o) in CORNERS.iter().enumerate() {
                        if grid[idx(i + o[0], j + o[1], k + o[2])] < iso {
                            case |= 1 << c;
                        }
                    }
                    if EDGE_TABLE[case] == 0 {
                        continue;
                    }
                    let mut tri = [0u32; 3];
                    for (m, &e) in TRI_TABLE[case].iter().enumerate() {
                        let (corner, axis) = EDGES[e as usize];
                        let o = CORNERS[corner];
                        let a = [i + o[0], j + o[1], k + o[2]];
                        let key = (idx(a[0], a[1], a[2]), axis);
                        tri[m % 3] = *vertex_of.entry(key).or_insert_with(|| {
                            let mut b = a;
                            b[axis as usize] += 1;
                            let fa = grid[idx(a[0], a[1], a[2])];
                            let fb = grid[idx(b[0], b[1], b[2])];
                            let t = if (fb - fa).abs() > f32::EPSILON {
                                ((iso - fa) / (fb - fa))
                                    .clamp(CROSSING_CLEARANCE, 1.0 - CROSSING_CLEARANCE)
                            } else {
                                0.5
                            };
                            for c in 0..3 {
                                positions.push(at(a[c], c) + t * (at(b[c], c) - at(a[c], c)));
                            }
                            let (ga, gb) = (gradient(a), gradient(b));
                            // Out of the inside: down the gradient.
                            let g = [0, 1, 2].map(|c| -(ga[c] + t * (gb[c] - ga[c])));
                            let len = (g[0] * g[0] + g[1] * g[1] + g[2] * g[2]).sqrt();
                            normals.extend(if len > 0.0 {
                                g.map(|c| c / len)
                            } else {
                                [0.0, 1.0, 0.0]
                            });
                            (positions.len() / 3 - 1) as u32
                        });
                        if m % 3 == 2 {
                            indices.extend(oriented(tri, &positions, &normals));
                        }
                    }
                }
            }
        }

        let mut g = BufferGeometry::new();
        if !positions.is_empty() {
            g.set_attribute("position", BufferAttribute::new(positions, 3));
            g.set_attribute("normal", BufferAttribute::new(normals, 3));
            g.set_index(indices);
        }
        g
    }
}

/// How close, as a fraction of an edge, a crossing may come to either end.
///
/// A sample that lands exactly on the surface — a face of a box on the grid,
/// the point where two spheres touch, anything centred on the origin at an even
/// resolution — would otherwise put the crossing of every edge meeting it at
/// that one point: separate vertices in the same place, and zero-area
/// triangles between them whose winding cannot be read. Held a ten-thousandth
/// of an edge off, every vertex has a position of its own and the topology is
/// still decided by the signs alone.
pub(crate) const CROSSING_CLEARANCE: f32 = 1e-4;

/// `tri` wound counter-clockwise seen from outside, its facet normal along
/// the field's (the table's own winding depends on which side it calls
/// inside).
fn oriented(tri: [u32; 3], positions: &[f32], normals: &[f32]) -> [u32; 3] {
    let p = |v: u32| {
        let v = v as usize * 3;
        [positions[v], positions[v + 1], positions[v + 2]]
    };
    let (a, b, c) = (p(tri[0]), p(tri[1]), p(tri[2]));
    let u = [0, 1, 2].map(|i| b[i] - a[i]);
    let w = [0, 1, 2].map(|i| c[i] - a[i]);
    let facet = [
        u[1] * w[2] - u[2] * w[1],
        u[2] * w[0] - u[0] * w[2],
        u[0] * w[1] - u[1] * w[0],
    ];
    let along: f32 = tri
        .iter()
        .map(|&v| {
            let v = v as usize * 3;
            facet[0] * normals[v] + facet[1] * normals[v + 1] + facet[2] * normals[v + 2]
        })
        .sum();
    if along < 0.0 {
        [tri[0], tri[2], tri[1]]
    } else {
        tri
    }
}

/// The cube's corners in the tables' order: the bottom face round, then the
/// top.
const CORNERS: [[usize; 3]; 8] = [
    [0, 0, 0],
    [1, 0, 0],
    [1, 1, 0],
    [0, 1, 0],
    [0, 0, 1],
    [1, 0, 1],
    [1, 1, 1],
    [0, 1, 1],
];

/// The cube's twelve edges in the tables' order, as the corner each starts
/// from and the axis it runs along from there.
const EDGES: [(usize, u8); 12] = [
    (0, 0),
    (1, 1),
    (3, 0),
    (0, 1),
    (4, 0),
    (5, 1),
    (7, 0),
    (4, 1),
    (0, 2),
    (1, 2),
    (2, 2),
    (3, 2),
];

/// Which edges each of the 256 corner cases crosses (Paul Bourke's table,
/// as three.js ships it).
#[rustfmt::skip]
const EDGE_TABLE: [u16; 256] = [
    0x000, 0x109, 0x203, 0x30a, 0x406, 0x50f, 0x605, 0x70c,
    0x80c, 0x905, 0xa0f, 0xb06, 0xc0a, 0xd03, 0xe09, 0xf00,
    0x190, 0x099, 0x393, 0x29a, 0x596, 0x49f, 0x795, 0x69c,
    0x99c, 0x895, 0xb9f, 0xa96, 0xd9a, 0xc93, 0xf99, 0xe90,
    0x230, 0x339, 0x033, 0x13a, 0x636, 0x73f, 0x435, 0x53c,
    0xa3c, 0xb35, 0x83f, 0x936, 0xe3a, 0xf33, 0xc39, 0xd30,
    0x3a0, 0x2a9, 0x1a3, 0x0aa, 0x7a6, 0x6af, 0x5a5, 0x4ac,
    0xbac, 0xaa5, 0x9af, 0x8a6, 0xfaa, 0xea3, 0xda9, 0xca0,
    0x460, 0x569, 0x663, 0x76a, 0x066, 0x16f, 0x265, 0x36c,
    0xc6c, 0xd65, 0xe6f, 0xf66, 0x86a, 0x963, 0xa69, 0xb60,
    0x5f0, 0x4f9, 0x7f3, 0x6fa, 0x1f6, 0x0ff, 0x3f5, 0x2fc,
    0xdfc, 0xcf5, 0xfff, 0xef6, 0x9fa, 0x8f3, 0xbf9, 0xaf0,
    0x650, 0x759, 0x453, 0x55a, 0x256, 0x35f, 0x055, 0x15c,
    0xe5c, 0xf55, 0xc5f, 0xd56, 0xa5a, 0xb53, 0x859, 0x950,
    0x7c0, 0x6c9, 0x5c3, 0x4ca, 0x3c6, 0x2cf, 0x1c5, 0x0cc,
    0xfcc, 0xec5, 0xdcf, 0xcc6, 0xbca, 0xac3, 0x9c9, 0x8c0,
    0x8c0, 0x9c9, 0xac3, 0xbca, 0xcc6, 0xdcf, 0xec5, 0xfcc,
    0x0cc, 0x1c5, 0x2cf, 0x3c6, 0x4ca, 0x5c3, 0x6c9, 0x7c0,
    0x950, 0x859, 0xb53, 0xa5a, 0xd56, 0xc5f, 0xf55, 0xe5c,
    0x15c, 0x055, 0x35f, 0x256, 0x55a, 0x453, 0x759, 0x650,
    0xaf0, 0xbf9, 0x8f3, 0x9fa, 0xef6, 0xfff, 0xcf5, 0xdfc,
    0x2fc, 0x3f5, 0x0ff, 0x1f6, 0x6fa, 0x7f3, 0x4f9, 0x5f0,
    0xb60, 0xa69, 0x963, 0x86a, 0xf66, 0xe6f, 0xd65, 0xc6c,
    0x36c, 0x265, 0x16f, 0x066, 0x76a, 0x663, 0x569, 0x460,
    0xca0, 0xda9, 0xea3, 0xfaa, 0x8a6, 0x9af, 0xaa5, 0xbac,
    0x4ac, 0x5a5, 0x6af, 0x7a6, 0x0aa, 0x1a3, 0x2a9, 0x3a0,
    0xd30, 0xc39, 0xf33, 0xe3a, 0x936, 0x83f, 0xb35, 0xa3c,
    0x53c, 0x435, 0x73f, 0x636, 0x13a, 0x033, 0x339, 0x230,
    0xe90, 0xf99, 0xc93, 0xd9a, 0xa96, 0xb9f, 0x895, 0x99c,
    0x69c, 0x795, 0x49f, 0x596, 0x29a, 0x393, 0x099, 0x190,
    0xf00, 0xe09, 0xd03, 0xc0a, 0xb06, 0xa0f, 0x905, 0x80c,
    0x70c, 0x605, 0x50f, 0x406, 0x30a, 0x203, 0x109, 0x000,
];

/// The triangles each case makes, as edges three at a time.
#[rustfmt::skip]
const TRI_TABLE: [&[u8]; 256] = [
    &[],
    &[0, 8, 3],
    &[0, 1, 9],
    &[1, 8, 3, 9, 8, 1],
    &[1, 2, 10],
    &[0, 8, 3, 1, 2, 10],
    &[9, 2, 10, 0, 2, 9],
    &[2, 8, 3, 2, 10, 8, 10, 9, 8],
    &[3, 11, 2],
    &[0, 11, 2, 8, 11, 0],
    &[1, 9, 0, 2, 3, 11],
    &[1, 11, 2, 1, 9, 11, 9, 8, 11],
    &[3, 10, 1, 11, 10, 3],
    &[0, 10, 1, 0, 8, 10, 8, 11, 10],
    &[3, 9, 0, 3, 11, 9, 11, 10, 9],
    &[9, 8, 10, 10, 8, 11],
    &[4, 7, 8],
    &[4, 3, 0, 7, 3, 4],
    &[0, 1, 9, 8, 4, 7],
    &[4, 1, 9, 4, 7, 1, 7, 3, 1],
    &[1, 2, 10, 8, 4, 7],
    &[3, 4, 7, 3, 0, 4, 1, 2, 10],
    &[9, 2, 10, 9, 0, 2, 8, 4, 7],
    &[2, 10, 9, 2, 9, 7, 2, 7, 3, 7, 9, 4],
    &[8, 4, 7, 3, 11, 2],
    &[11, 4, 7, 11, 2, 4, 2, 0, 4],
    &[9, 0, 1, 8, 4, 7, 2, 3, 11],
    &[4, 7, 11, 9, 4, 11, 9, 11, 2, 9, 2, 1],
    &[3, 10, 1, 3, 11, 10, 7, 8, 4],
    &[1, 11, 10, 1, 4, 11, 1, 0, 4, 7, 11, 4],
    &[4, 7, 8, 9, 0, 11, 9, 11, 10, 11, 0, 3],
    &[4, 7, 11, 4, 11, 9, 9, 11, 10],
    &[9, 5, 4],
    &[9, 5, 4, 0, 8, 3],
    &[0, 5, 4, 1, 5, 0],
    &[8, 5, 4, 8, 3, 5, 3, 1, 5],
    &[1, 2, 10, 9, 5, 4],
    &[3, 0, 8, 1, 2, 10, 4, 9, 5],
    &[5, 2, 10, 5, 4, 2, 4, 0, 2],
    &[2, 10, 5, 3, 2, 5, 3, 5, 4, 3, 4, 8],
    &[9, 5, 4, 2, 3, 11],
    &[0, 11, 2, 0, 8, 11, 4, 9, 5],
    &[0, 5, 4, 0, 1, 5, 2, 3, 11],
    &[2, 1, 5, 2, 5, 8, 2, 8, 11, 4, 8, 5],
    &[10, 3, 11, 10, 1, 3, 9, 5, 4],
    &[4, 9, 5, 0, 8, 1, 8, 10, 1, 8, 11, 10],
    &[5, 4, 0, 5, 0, 11, 5, 11, 10, 11, 0, 3],
    &[5, 4, 8, 5, 8, 10, 10, 8, 11],
    &[9, 7, 8, 5, 7, 9],
    &[9, 3, 0, 9, 5, 3, 5, 7, 3],
    &[0, 7, 8, 0, 1, 7, 1, 5, 7],
    &[1, 5, 3, 3, 5, 7],
    &[9, 7, 8, 9, 5, 7, 10, 1, 2],
    &[10, 1, 2, 9, 5, 0, 5, 3, 0, 5, 7, 3],
    &[8, 0, 2, 8, 2, 5, 8, 5, 7, 10, 5, 2],
    &[2, 10, 5, 2, 5, 3, 3, 5, 7],
    &[7, 9, 5, 7, 8, 9, 3, 11, 2],
    &[9, 5, 7, 9, 7, 2, 9, 2, 0, 2, 7, 11],
    &[2, 3, 11, 0, 1, 8, 1, 7, 8, 1, 5, 7],
    &[11, 2, 1, 11, 1, 7, 7, 1, 5],
    &[9, 5, 8, 8, 5, 7, 10, 1, 3, 10, 3, 11],
    &[5, 7, 0, 5, 0, 9, 7, 11, 0, 1, 0, 10, 11, 10, 0],
    &[11, 10, 0, 11, 0, 3, 10, 5, 0, 8, 0, 7, 5, 7, 0],
    &[11, 10, 5, 7, 11, 5],
    &[10, 6, 5],
    &[0, 8, 3, 5, 10, 6],
    &[9, 0, 1, 5, 10, 6],
    &[1, 8, 3, 1, 9, 8, 5, 10, 6],
    &[1, 6, 5, 2, 6, 1],
    &[1, 6, 5, 1, 2, 6, 3, 0, 8],
    &[9, 6, 5, 9, 0, 6, 0, 2, 6],
    &[5, 9, 8, 5, 8, 2, 5, 2, 6, 3, 2, 8],
    &[2, 3, 11, 10, 6, 5],
    &[11, 0, 8, 11, 2, 0, 10, 6, 5],
    &[0, 1, 9, 2, 3, 11, 5, 10, 6],
    &[5, 10, 6, 1, 9, 2, 9, 11, 2, 9, 8, 11],
    &[6, 3, 11, 6, 5, 3, 5, 1, 3],
    &[0, 8, 11, 0, 11, 5, 0, 5, 1, 5, 11, 6],
    &[3, 11, 6, 0, 3, 6, 0, 6, 5, 0, 5, 9],
    &[6, 5, 9, 6, 9, 11, 11, 9, 8],
    &[5, 10, 6, 4, 7, 8],
    &[4, 3, 0, 4, 7, 3, 6, 5, 10],
    &[1, 9, 0, 5, 10, 6, 8, 4, 7],
    &[10, 6, 5, 1, 9, 7, 1, 7, 3, 7, 9, 4],
    &[6, 1, 2, 6, 5, 1, 4, 7, 8],
    &[1, 2, 5, 5, 2, 6, 3, 0, 4, 3, 4, 7],
    &[8, 4, 7, 9, 0, 5, 0, 6, 5, 0, 2, 6],
    &[7, 3, 9, 7, 9, 4, 3, 2, 9, 5, 9, 6, 2, 6, 9],
    &[3, 11, 2, 7, 8, 4, 10, 6, 5],
    &[5, 10, 6, 4, 7, 2, 4, 2, 0, 2, 7, 11],
    &[0, 1, 9, 4, 7, 8, 2, 3, 11, 5, 10, 6],
    &[9, 2, 1, 9, 11, 2, 9, 4, 11, 7, 11, 4, 5, 10, 6],
    &[8, 4, 7, 3, 11, 5, 3, 5, 1, 5, 11, 6],
    &[5, 1, 11, 5, 11, 6, 1, 0, 11, 7, 11, 4, 0, 4, 11],
    &[0, 5, 9, 0, 6, 5, 0, 3, 6, 11, 6, 3, 8, 4, 7],
    &[6, 5, 9, 6, 9, 11, 4, 7, 9, 7, 11, 9],
    &[10, 4, 9, 6, 4, 10],
    &[4, 10, 6, 4, 9, 10, 0, 8, 3],
    &[10, 0, 1, 10, 6, 0, 6, 4, 0],
    &[8, 3, 1, 8, 1, 6, 8, 6, 4, 6, 1, 10],
    &[1, 4, 9, 1, 2, 4, 2, 6, 4],
    &[3, 0, 8, 1, 2, 9, 2, 4, 9, 2, 6, 4],
    &[0, 2, 4, 4, 2, 6],
    &[8, 3, 2, 8, 2, 4, 4, 2, 6],
    &[10, 4, 9, 10, 6, 4, 11, 2, 3],
    &[0, 8, 2, 2, 8, 11, 4, 9, 10, 4, 10, 6],
    &[3, 11, 2, 0, 1, 6, 0, 6, 4, 6, 1, 10],
    &[6, 4, 1, 6, 1, 10, 4, 8, 1, 2, 1, 11, 8, 11, 1],
    &[9, 6, 4, 9, 3, 6, 9, 1, 3, 11, 6, 3],
    &[8, 11, 1, 8, 1, 0, 11, 6, 1, 9, 1, 4, 6, 4, 1],
    &[3, 11, 6, 3, 6, 0, 0, 6, 4],
    &[6, 4, 8, 11, 6, 8],
    &[7, 10, 6, 7, 8, 10, 8, 9, 10],
    &[0, 7, 3, 0, 10, 7, 0, 9, 10, 6, 7, 10],
    &[10, 6, 7, 1, 10, 7, 1, 7, 8, 1, 8, 0],
    &[10, 6, 7, 10, 7, 1, 1, 7, 3],
    &[1, 2, 6, 1, 6, 8, 1, 8, 9, 8, 6, 7],
    &[2, 6, 9, 2, 9, 1, 6, 7, 9, 0, 9, 3, 7, 3, 9],
    &[7, 8, 0, 7, 0, 6, 6, 0, 2],
    &[7, 3, 2, 6, 7, 2],
    &[2, 3, 11, 10, 6, 8, 10, 8, 9, 8, 6, 7],
    &[2, 0, 7, 2, 7, 11, 0, 9, 7, 6, 7, 10, 9, 10, 7],
    &[1, 8, 0, 1, 7, 8, 1, 10, 7, 6, 7, 10, 2, 3, 11],
    &[11, 2, 1, 11, 1, 7, 10, 6, 1, 6, 7, 1],
    &[8, 9, 6, 8, 6, 7, 9, 1, 6, 11, 6, 3, 1, 3, 6],
    &[0, 9, 1, 11, 6, 7],
    &[7, 8, 0, 7, 0, 6, 3, 11, 0, 11, 6, 0],
    &[7, 11, 6],
    &[7, 6, 11],
    &[3, 0, 8, 11, 7, 6],
    &[0, 1, 9, 11, 7, 6],
    &[8, 1, 9, 8, 3, 1, 11, 7, 6],
    &[10, 1, 2, 6, 11, 7],
    &[1, 2, 10, 3, 0, 8, 6, 11, 7],
    &[2, 9, 0, 2, 10, 9, 6, 11, 7],
    &[6, 11, 7, 2, 10, 3, 10, 8, 3, 10, 9, 8],
    &[7, 2, 3, 6, 2, 7],
    &[7, 0, 8, 7, 6, 0, 6, 2, 0],
    &[2, 7, 6, 2, 3, 7, 0, 1, 9],
    &[1, 6, 2, 1, 8, 6, 1, 9, 8, 8, 7, 6],
    &[10, 7, 6, 10, 1, 7, 1, 3, 7],
    &[10, 7, 6, 1, 7, 10, 1, 8, 7, 1, 0, 8],
    &[0, 3, 7, 0, 7, 10, 0, 10, 9, 6, 10, 7],
    &[7, 6, 10, 7, 10, 8, 8, 10, 9],
    &[6, 8, 4, 11, 8, 6],
    &[3, 6, 11, 3, 0, 6, 0, 4, 6],
    &[8, 6, 11, 8, 4, 6, 9, 0, 1],
    &[9, 4, 6, 9, 6, 3, 9, 3, 1, 11, 3, 6],
    &[6, 8, 4, 6, 11, 8, 2, 10, 1],
    &[1, 2, 10, 3, 0, 11, 0, 6, 11, 0, 4, 6],
    &[4, 11, 8, 4, 6, 11, 0, 2, 9, 2, 10, 9],
    &[10, 9, 3, 10, 3, 2, 9, 4, 3, 11, 3, 6, 4, 6, 3],
    &[8, 2, 3, 8, 4, 2, 4, 6, 2],
    &[0, 4, 2, 4, 6, 2],
    &[1, 9, 0, 2, 3, 4, 2, 4, 6, 4, 3, 8],
    &[1, 9, 4, 1, 4, 2, 2, 4, 6],
    &[8, 1, 3, 8, 6, 1, 8, 4, 6, 6, 10, 1],
    &[10, 1, 0, 10, 0, 6, 6, 0, 4],
    &[4, 6, 3, 4, 3, 8, 6, 10, 3, 0, 3, 9, 10, 9, 3],
    &[10, 9, 4, 6, 10, 4],
    &[4, 9, 5, 7, 6, 11],
    &[0, 8, 3, 4, 9, 5, 11, 7, 6],
    &[5, 0, 1, 5, 4, 0, 7, 6, 11],
    &[11, 7, 6, 8, 3, 4, 3, 5, 4, 3, 1, 5],
    &[9, 5, 4, 10, 1, 2, 7, 6, 11],
    &[6, 11, 7, 1, 2, 10, 0, 8, 3, 4, 9, 5],
    &[7, 6, 11, 5, 4, 10, 4, 2, 10, 4, 0, 2],
    &[3, 4, 8, 3, 5, 4, 3, 2, 5, 10, 5, 2, 11, 7, 6],
    &[7, 2, 3, 7, 6, 2, 5, 4, 9],
    &[9, 5, 4, 0, 8, 6, 0, 6, 2, 6, 8, 7],
    &[3, 6, 2, 3, 7, 6, 1, 5, 0, 5, 4, 0],
    &[6, 2, 8, 6, 8, 7, 2, 1, 8, 4, 8, 5, 1, 5, 8],
    &[9, 5, 4, 10, 1, 6, 1, 7, 6, 1, 3, 7],
    &[1, 6, 10, 1, 7, 6, 1, 0, 7, 8, 7, 0, 9, 5, 4],
    &[4, 0, 10, 4, 10, 5, 0, 3, 10, 6, 10, 7, 3, 7, 10],
    &[7, 6, 10, 7, 10, 8, 5, 4, 10, 4, 8, 10],
    &[6, 9, 5, 6, 11, 9, 11, 8, 9],
    &[3, 6, 11, 0, 6, 3, 0, 5, 6, 0, 9, 5],
    &[0, 11, 8, 0, 5, 11, 0, 1, 5, 5, 6, 11],
    &[6, 11, 3, 6, 3, 5, 5, 3, 1],
    &[1, 2, 10, 9, 5, 11, 9, 11, 8, 11, 5, 6],
    &[0, 11, 3, 0, 6, 11, 0, 9, 6, 5, 6, 9, 1, 2, 10],
    &[11, 8, 5, 11, 5, 6, 8, 0, 5, 10, 5, 2, 0, 2, 5],
    &[6, 11, 3, 6, 3, 5, 2, 10, 3, 10, 5, 3],
    &[5, 8, 9, 5, 2, 8, 5, 6, 2, 3, 8, 2],
    &[9, 5, 6, 9, 6, 0, 0, 6, 2],
    &[1, 5, 8, 1, 8, 0, 5, 6, 8, 3, 8, 2, 6, 2, 8],
    &[1, 5, 6, 2, 1, 6],
    &[1, 3, 6, 1, 6, 10, 3, 8, 6, 5, 6, 9, 8, 9, 6],
    &[10, 1, 0, 10, 0, 6, 9, 5, 0, 5, 6, 0],
    &[0, 3, 8, 5, 6, 10],
    &[10, 5, 6],
    &[11, 5, 10, 7, 5, 11],
    &[11, 5, 10, 11, 7, 5, 8, 3, 0],
    &[5, 11, 7, 5, 10, 11, 1, 9, 0],
    &[10, 7, 5, 10, 11, 7, 9, 8, 1, 8, 3, 1],
    &[11, 1, 2, 11, 7, 1, 7, 5, 1],
    &[0, 8, 3, 1, 2, 7, 1, 7, 5, 7, 2, 11],
    &[9, 7, 5, 9, 2, 7, 9, 0, 2, 2, 11, 7],
    &[7, 5, 2, 7, 2, 11, 5, 9, 2, 3, 2, 8, 9, 8, 2],
    &[2, 5, 10, 2, 3, 5, 3, 7, 5],
    &[8, 2, 0, 8, 5, 2, 8, 7, 5, 10, 2, 5],
    &[9, 0, 1, 5, 10, 3, 5, 3, 7, 3, 10, 2],
    &[9, 8, 2, 9, 2, 1, 8, 7, 2, 10, 2, 5, 7, 5, 2],
    &[1, 3, 5, 3, 7, 5],
    &[0, 8, 7, 0, 7, 1, 1, 7, 5],
    &[9, 0, 3, 9, 3, 5, 5, 3, 7],
    &[9, 8, 7, 5, 9, 7],
    &[5, 8, 4, 5, 10, 8, 10, 11, 8],
    &[5, 0, 4, 5, 11, 0, 5, 10, 11, 11, 3, 0],
    &[0, 1, 9, 8, 4, 10, 8, 10, 11, 10, 4, 5],
    &[10, 11, 4, 10, 4, 5, 11, 3, 4, 9, 4, 1, 3, 1, 4],
    &[2, 5, 1, 2, 8, 5, 2, 11, 8, 4, 5, 8],
    &[0, 4, 11, 0, 11, 3, 4, 5, 11, 2, 11, 1, 5, 1, 11],
    &[0, 2, 5, 0, 5, 9, 2, 11, 5, 4, 5, 8, 11, 8, 5],
    &[9, 4, 5, 2, 11, 3],
    &[2, 5, 10, 3, 5, 2, 3, 4, 5, 3, 8, 4],
    &[5, 10, 2, 5, 2, 4, 4, 2, 0],
    &[3, 10, 2, 3, 5, 10, 3, 8, 5, 4, 5, 8, 0, 1, 9],
    &[5, 10, 2, 5, 2, 4, 1, 9, 2, 9, 4, 2],
    &[8, 4, 5, 8, 5, 3, 3, 5, 1],
    &[0, 4, 5, 1, 0, 5],
    &[8, 4, 5, 8, 5, 3, 9, 0, 5, 0, 3, 5],
    &[9, 4, 5],
    &[4, 11, 7, 4, 9, 11, 9, 10, 11],
    &[0, 8, 3, 4, 9, 7, 9, 11, 7, 9, 10, 11],
    &[1, 10, 11, 1, 11, 4, 1, 4, 0, 7, 4, 11],
    &[3, 1, 4, 3, 4, 8, 1, 10, 4, 7, 4, 11, 10, 11, 4],
    &[4, 11, 7, 9, 11, 4, 9, 2, 11, 9, 1, 2],
    &[9, 7, 4, 9, 11, 7, 9, 1, 11, 2, 11, 1, 0, 8, 3],
    &[11, 7, 4, 11, 4, 2, 2, 4, 0],
    &[11, 7, 4, 11, 4, 2, 8, 3, 4, 3, 2, 4],
    &[2, 9, 10, 2, 7, 9, 2, 3, 7, 7, 4, 9],
    &[9, 10, 7, 9, 7, 4, 10, 2, 7, 8, 7, 0, 2, 0, 7],
    &[3, 7, 10, 3, 10, 2, 7, 4, 10, 1, 10, 0, 4, 0, 10],
    &[1, 10, 2, 8, 7, 4],
    &[4, 9, 1, 4, 1, 7, 7, 1, 3],
    &[4, 9, 1, 4, 1, 7, 0, 8, 1, 8, 7, 1],
    &[4, 0, 3, 7, 4, 3],
    &[4, 8, 7],
    &[9, 10, 8, 10, 11, 8],
    &[3, 0, 9, 3, 9, 11, 11, 9, 10],
    &[0, 1, 10, 0, 10, 8, 8, 10, 11],
    &[3, 1, 10, 11, 3, 10],
    &[1, 2, 11, 1, 11, 9, 9, 11, 8],
    &[3, 0, 9, 3, 9, 11, 1, 2, 9, 2, 11, 9],
    &[0, 2, 11, 8, 0, 11],
    &[3, 2, 11],
    &[2, 3, 8, 2, 8, 10, 10, 8, 9],
    &[9, 10, 2, 0, 9, 2],
    &[2, 3, 8, 2, 8, 10, 0, 1, 8, 1, 10, 8],
    &[1, 10, 2],
    &[1, 3, 8, 9, 1, 8],
    &[0, 9, 1],
    &[0, 3, 8],
    &[],
];

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn ball(res: usize) -> BufferGeometry {
        MarchingCubes::new(res, Vector3::new(2.0, 2.0, 2.0))
            .extract(0.0, |x, y, z| 0.7 - (x * x + y * y + z * z).sqrt())
    }

    /// Positions, normals and triangles.
    type Parts = (Vec<[f32; 3]>, Vec<[f32; 3]>, Vec<[u32; 3]>);

    fn parts(g: &BufferGeometry) -> Parts {
        let p = &g.get_attribute("position").unwrap().array;
        let nm = &g.get_attribute("normal").unwrap().array;
        let ix = g.index.as_ref().unwrap();
        (
            p.chunks(3).map(|c| [c[0], c[1], c[2]]).collect(),
            nm.chunks(3).map(|c| [c[0], c[1], c[2]]).collect(),
            ix.chunks(3).map(|c| [c[0], c[1], c[2]]).collect(),
        )
    }

    /// Every vertex lies on the sphere, to within the linear interpolation's
    /// error, and its normal points away from the centre.
    #[test]
    fn a_sphere_comes_out_on_the_sphere_facing_out() {
        let (p, n, _) = parts(&ball(40));
        for (v, nv) in p.iter().zip(&n) {
            let r = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
            assert!((r - 0.7).abs() < 2e-3, "vertex at radius {r}");
            let out = (v[0] * nv[0] + v[1] * nv[1] + v[2] * nv[2]) / r;
            assert!(out > 0.99, "normal {nv:?} at {v:?}");
        }
    }

    /// Closed: every edge is shared by exactly two triangles, once each way
    /// round, so the winding is consistent; and the enclosed volume (by the
    /// divergence theorem) and area are the sphere's to within a percent.
    #[test]
    fn a_sphere_is_watertight_and_wound_outward() {
        let (p, _, t) = parts(&ball(40));
        let mut edges: HashMap<(u32, u32), i32> = HashMap::new();
        let (mut volume, mut area) = (0.0f64, 0.0f64);
        for tri in &t {
            for e in 0..3 {
                *edges.entry((tri[e], tri[(e + 1) % 3])).or_default() += 1;
            }
            let [a, b, c] = tri.map(|i| p[i as usize].map(|x| x as f64));
            let u = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
            let w = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
            let cross = [
                u[1] * w[2] - u[2] * w[1],
                u[2] * w[0] - u[0] * w[2],
                u[0] * w[1] - u[1] * w[0],
            ];
            area += 0.5 * (cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2]).sqrt();
            volume += (a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0])
                + a[2] * (b[0] * c[1] - b[1] * c[0]))
                / 6.0;
        }
        for (&(a, b), &count) in &edges {
            assert_eq!(
                count, 1,
                "edge {a}–{b} used {count} times the same way round"
            );
            assert_eq!(edges.get(&(b, a)), Some(&1), "edge {a}–{b} is open");
        }
        let r = 0.7f64;
        let (v, s) = (
            4.0 / 3.0 * std::f64::consts::PI * r.powi(3),
            4.0 * std::f64::consts::PI * r * r,
        );
        assert!(
            (volume / v - 1.0).abs() < 0.01,
            "volume {volume} against {v}"
        );
        assert!((area / s - 1.0).abs() < 0.01, "area {area} against {s}");
    }

    /// A box whose faces sit exactly on grid samples: every crossing next to
    /// a face sample used to land on it, as separate vertices in one place
    /// with zero-area triangles between them.
    #[test]
    fn samples_on_the_surface_make_no_coincident_vertices() {
        let g = MarchingCubes::new(64, Vector3::new(2.0, 2.0, 2.0))
            .extract(0.0, |x, y, z| 0.5 - x.abs().max(y.abs()).max(z.abs()));
        let (p, _, t) = parts(&g);
        let mut seen = std::collections::HashSet::new();
        for v in &p {
            assert!(seen.insert(v.map(f32::to_bits)), "two vertices at {v:?}");
        }
        for tri in &t {
            let [a, b, c] = tri.map(|i| p[i as usize]);
            let u = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
            let w = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
            let cross = [
                u[1] * w[2] - u[2] * w[1],
                u[2] * w[0] - u[0] * w[2],
                u[0] * w[1] - u[1] * w[0],
            ];
            assert!(
                cross.iter().any(|&x| x != 0.0),
                "zero-area triangle {tri:?}"
            );
        }
        // And the box still comes out where it is, to the clearance.
        for v in &p {
            let r = v[0].abs().max(v[1].abs()).max(v[2].abs());
            assert!((r - 0.5).abs() < 2.0 / 64.0 * 2e-4, "vertex at {r}");
        }
    }

    /// Nothing crosses: nothing out.
    #[test]
    fn a_field_with_no_crossing_gives_an_empty_geometry() {
        let g = MarchingCubes::new(8, Vector3::new(1.0, 1.0, 1.0)).extract(0.5, |_, _, _| 0.0);
        assert!(g.get_attribute("position").is_none());
    }
}
