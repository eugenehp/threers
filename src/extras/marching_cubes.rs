use crate::core::{BufferAttribute, BufferGeometry};
use crate::math::Vector3;

/// Marching-cubes isosurface extractor. Mirrors three.js's `MarchingCubes`
/// (build a volumetric field, then triangulate the iso-surface). This is a
/// minimal density-driven version — caller supplies a `field` function
/// returning a scalar per cell corner.
pub struct MarchingCubes {
    pub resolution: usize,
    pub size: Vector3,
}

impl MarchingCubes {
    pub fn new(resolution: usize, size: Vector3) -> Self {
        Self { resolution, size }
    }

    pub fn extract<F>(&self, iso: f32, field: F) -> BufferGeometry
    where
        F: Fn(f32, f32, f32) -> f32,
    {
        let res = self.resolution.max(2);
        let step_x = self.size.x / res as f32;
        let step_y = self.size.y / res as f32;
        let step_z = self.size.z / res as f32;

        // Sample the field on a uniform grid.
        let dim = res + 1;
        let mut grid = vec![0.0f32; dim * dim * dim];
        let idx = |i: usize, j: usize, k: usize| i + dim * (j + dim * k);
        for k in 0..dim {
            for j in 0..dim {
                for i in 0..dim {
                    let x = (i as f32 - res as f32 * 0.5) * step_x;
                    let y = (j as f32 - res as f32 * 0.5) * step_y;
                    let z = (k as f32 - res as f32 * 0.5) * step_z;
                    grid[idx(i, j, k)] = field(x, y, z);
                }
            }
        }

        // Cell-by-cell: emit a triangle per "edge crossing" using the simple
        // surface-net approximation (centroid of edge crossings). Not the full
        // 256-case marching-cubes table but produces a reasonable surface.
        let mut positions: Vec<f32> = Vec::new();
        for k in 0..res {
            for j in 0..res {
                for i in 0..res {
                    let corners = [
                        grid[idx(i, j, k)],
                        grid[idx(i + 1, j, k)],
                        grid[idx(i, j + 1, k)],
                        grid[idx(i + 1, j + 1, k)],
                        grid[idx(i, j, k + 1)],
                        grid[idx(i + 1, j, k + 1)],
                        grid[idx(i, j + 1, k + 1)],
                        grid[idx(i + 1, j + 1, k + 1)],
                    ];
                    let mut crossings = Vec::new();
                    let edges = [
                        (0, 1, [step_x, 0.0, 0.0]),
                        (2, 3, [step_x, 0.0, 0.0]),
                        (4, 5, [step_x, 0.0, 0.0]),
                        (6, 7, [step_x, 0.0, 0.0]),
                        (0, 2, [0.0, step_y, 0.0]),
                        (1, 3, [0.0, step_y, 0.0]),
                        (4, 6, [0.0, step_y, 0.0]),
                        (5, 7, [0.0, step_y, 0.0]),
                        (0, 4, [0.0, 0.0, step_z]),
                        (1, 5, [0.0, 0.0, step_z]),
                        (2, 6, [0.0, 0.0, step_z]),
                        (3, 7, [0.0, 0.0, step_z]),
                    ];
                    for &(a, b, axis) in &edges {
                        let va = corners[a];
                        let vb = corners[b];
                        if (va - iso) * (vb - iso) < 0.0 {
                            let t = (iso - va) / (vb - va);
                            let base = corner_pos(a, i, j, k, step_x, step_y, step_z, res);
                            crossings.push([
                                base[0] + axis[0] * t,
                                base[1] + axis[1] * t,
                                base[2] + axis[2] * t,
                            ]);
                        }
                    }
                    if crossings.len() < 3 {
                        continue;
                    }
                    // Fan-triangulate the crossings (rough surface-net).
                    let center = [
                        crossings.iter().map(|p| p[0]).sum::<f32>() / crossings.len() as f32,
                        crossings.iter().map(|p| p[1]).sum::<f32>() / crossings.len() as f32,
                        crossings.iter().map(|p| p[2]).sum::<f32>() / crossings.len() as f32,
                    ];
                    let n = crossings.len();
                    for x in 0..n {
                        let p1 = crossings[x];
                        let p2 = crossings[(x + 1) % n];
                        positions.extend_from_slice(&center);
                        positions.extend_from_slice(&p1);
                        positions.extend_from_slice(&p2);
                    }
                }
            }
        }

        let mut g = BufferGeometry::new();
        if !positions.is_empty() {
            g.set_attribute("position", BufferAttribute::new(positions, 3));
        }
        g
    }
}

#[allow(clippy::too_many_arguments)]
fn corner_pos(
    c: usize,
    i: usize,
    j: usize,
    k: usize,
    sx: f32,
    sy: f32,
    sz: f32,
    res: usize,
) -> [f32; 3] {
    let dx = if c & 1 != 0 { 1 } else { 0 };
    let dy = if c & 2 != 0 { 1 } else { 0 };
    let dz = if c & 4 != 0 { 1 } else { 0 };
    [
        ((i + dx) as f32 - res as f32 * 0.5) * sx,
        ((j + dy) as f32 - res as f32 * 0.5) * sy,
        ((k + dz) as f32 - res as f32 * 0.5) * sz,
    ]
}
