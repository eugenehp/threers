//! Light through smoke: how much of a system's own particles lies between
//! each particle and a light.
//!
//! The particles are spread into a coarse grid of extinction (each one's
//! optical density, `extinction × opacity` per metre, over its own sphere),
//! the grid lined up with the light so that the optical depth from the light
//! to every cell is a running sum down each column. Each particle then reads
//! the depth where the light enters it: the light reaching it has crossed
//! that much of the others. Once toward the sun and once toward the
//! sky, which is all the shader needs for Beer–Lambert light, multiple
//! scattering and occlusion of the skylight (see `PARTICLE_WGSL`).
//!
//! A few thousand particles take a fraction of a millisecond: the grid is at
//! most 32 cells a side, and a particle spreads over the cells its sphere
//! covers (a dozen or so a side at most), so it costs what its size does.

use crate::math::Vector3;

use super::ParticleSystem;

/// Cells along the grid's longest side.
const CELLS: usize = 32;

/// Each particle's optical depth toward `toward` (a unit vector from the
/// particles to the light) through the others in `centre` (radius
/// `radius`, extinction coefficient inside its sphere `sigma`, per metre).
pub fn optical_depths(
    centre: &[Vector3],
    radius: &[f32],
    sigma: &[f32],
    toward: Vector3,
) -> Vec<f32> {
    let n = centre.len();
    if n == 0 {
        return Vec::new();
    }
    // The light's frame: z toward the light.
    let z = toward.normalize();
    let helper = if z.y.abs() < 0.9 {
        Vector3::new(0.0, 1.0, 0.0)
    } else {
        Vector3::new(1.0, 0.0, 0.0)
    };
    let x = helper.cross(z).normalize();
    let y = z.cross(x);
    let local: Vec<[f32; 3]> = centre
        .iter()
        .map(|p| [p.dot(x), p.dot(y), p.dot(z)])
        .collect();
    // The box round all but the outermost hundredth either side on each
    // axis: one spark flown far off would otherwise stretch the grid and
    // coarsen it for the whole plume. Strays are clamped into its edge.
    let biggest = radius.iter().copied().fold(0.0f32, f32::max);
    let mut lo = [0.0f32; 3];
    let mut hi = [0.0f32; 3];
    let mut column: Vec<f32> = Vec::with_capacity(n);
    for a in 0..3 {
        column.clear();
        column.extend(local.iter().map(|c| c[a]));
        column.sort_unstable_by(f32::total_cmp);
        let cut = n / 100;
        lo[a] = column[cut] - biggest;
        hi[a] = column[n - 1 - cut] + biggest;
    }
    let local: Vec<[f32; 3]> = local
        .iter()
        .map(|c| std::array::from_fn(|a| c[a].clamp(lo[a], hi[a])))
        .collect();
    let beyond_light: Vec<bool> = centre.iter().map(|p| p.dot(z) > hi[2]).collect();
    // Cells no smaller than half the biggest particle (its sphere then
    // covers at most ~5 a side), and no more than CELLS a side.
    let longest = (0..3)
        .map(|a| hi[a] - lo[a])
        .fold(0.0f32, f32::max)
        .max(1e-3);
    let cell = (longest / CELLS as f32).max(biggest * 0.5).max(1e-4);
    let dims: [usize; 3] =
        std::array::from_fn(|a| (((hi[a] - lo[a]) / cell).ceil() as usize).clamp(1, CELLS * 4));
    let [nx, ny, nz] = dims;
    let at = |i: usize, j: usize, k: usize| i + nx * (j + ny * k);
    let mut rho = vec![0.0f32; nx * ny * nz];
    let volume = cell * cell * cell;
    let mut weights: Vec<(usize, f32)> = Vec::new();
    for ((c, &r), &s) in local.iter().zip(radius).zip(sigma) {
        if s <= 0.0 || r <= 0.0 {
            continue;
        }
        // What it holds: its extinction over its sphere.
        let mass = s * (4.0 / 3.0) * std::f32::consts::PI * r * r * r;
        let range = |a: usize| {
            let f = |v: f32| ((v - lo[a]) / cell - 0.5).floor().max(0.0) as usize;
            (f(c[a] - r), (f(c[a] + r) + 1).min(dims[a] - 1))
        };
        let ((i0, i1), (j0, j1), (k0, k1)) = (range(0), range(1), range(2));
        weights.clear();
        let mut total = 0.0;
        for k in k0..=k1 {
            for j in j0..=j1 {
                for i in i0..=i1 {
                    let p = [
                        lo[0] + (i as f32 + 0.5) * cell,
                        lo[1] + (j as f32 + 0.5) * cell,
                        lo[2] + (k as f32 + 0.5) * cell,
                    ];
                    let d2 = (p[0] - c[0]).powi(2) + (p[1] - c[1]).powi(2) + (p[2] - c[2]).powi(2);
                    let w = (1.0 - d2 / (r * r)).max(0.0);
                    if w > 0.0 {
                        weights.push((at(i, j, k), w));
                        total += w;
                    }
                }
            }
        }
        if total <= 0.0 {
            // Smaller than a cell: all of it in the one it is in.
            let idx = |a: usize| (((c[a] - lo[a]) / cell) as usize).min(dims[a] - 1);
            rho[at(idx(0), idx(1), idx(2))] += mass / volume;
        } else {
            for &(cell_index, w) in &weights {
                rho[cell_index] += mass * w / total / volume;
            }
        }
    }
    // Down each column from the light: the depth to each cell's centre.
    let mut tau = vec![0.0f32; nx * ny * nz];
    for j in 0..ny {
        for i in 0..nx {
            let mut acc = 0.0;
            for k in (0..nz).rev() {
                let d = rho[at(i, j, k)] * cell;
                tau[at(i, j, k)] = acc + 0.5 * d;
                acc += d;
            }
        }
    }
    // Read back where the light enters each particle — its sunward edge,
    // and half a cell on, out of its own cell — trilinearly: what is in the
    // way, not the particle itself, whose own body the shader accounts for
    // across its face.
    local
        .iter()
        .zip(radius)
        .zip(&beyond_light)
        .map(|((c, &r), &beyond_light)| {
            if beyond_light {
                return 0.0;
            }
            let mut idx = [0usize; 3];
            let mut f = [0.0f32; 3];
            let c = [c[0], c[1], c[2] + r + 0.5 * cell];
            // Past the last cells' centres toward the light the depth falls
            // to nothing at the grid's face: there is nothing beyond.
            let beyond = (c[2] - lo[2]) / cell - 0.5 - (nz - 1) as f32;
            let open = (1.0 - beyond * 2.0).clamp(0.0, 1.0);
            for a in 0..3 {
                let u = ((c[a] - lo[a]) / cell - 0.5).clamp(0.0, (dims[a] - 1) as f32);
                idx[a] = (u as usize).min(dims[a].saturating_sub(2));
                f[a] = (u - idx[a] as f32).clamp(0.0, 1.0);
                if dims[a] == 1 {
                    f[a] = 0.0;
                }
            }
            let mut v = 0.0;
            for dk in 0..2 {
                for dj in 0..2 {
                    for di in 0..2 {
                        let (i, j, k) = (
                            (idx[0] + di).min(nx - 1),
                            (idx[1] + dj).min(ny - 1),
                            (idx[2] + dk).min(nz - 1),
                        );
                        let w = (if di == 1 { f[0] } else { 1.0 - f[0] })
                            * (if dj == 1 { f[1] } else { 1.0 - f[1] })
                            * (if dk == 1 { f[2] } else { 1.0 - f[2] });
                        v += w * tau[at(i, j, k)];
                    }
                }
            }
            v * open
        })
        .collect()
}

/// `system`'s particles' optical depths toward `sun` (a unit vector toward
/// it) and toward the sky straight up, as drawn: each a sphere of its drawn
/// radius, its extinction `look.extinction` times its opacity. In the
/// system's particle order.
pub fn depths(system: &ParticleSystem, sun: Vector3) -> Vec<[f32; 2]> {
    let ps = &system.particles;
    let n = ps.len();
    let mut radius = Vec::with_capacity(n);
    let mut sigma = Vec::with_capacity(n);
    for i in 0..n {
        let t = system.life_fraction(i);
        radius.push(ps.size[i] * system.size_over_life.at(t) * 0.5);
        sigma.push(system.look.extinction * system.color_over_life.at(t)[3].clamp(0.0, 1.0));
    }
    let to_sun = optical_depths(&ps.position, &radius, &sigma, sun);
    let to_sky = optical_depths(&ps.position, &radius, &sigma, Vector3::new(0.0, 1.0, 0.0));
    to_sun
        .into_iter()
        .zip(to_sky)
        .map(|(a, b)| [a, b])
        .collect()
}

/// The two depths packed into one float for the shader: the sun's depth in
/// 1/128ths in the whole part (to 60), the sky's transmittance in the
/// fraction.
pub(crate) fn pack(d: [f32; 2]) -> f32 {
    (d[0].clamp(0.0, 60.0) * 128.0).round() + (-d[1].max(0.0)).exp().clamp(0.0, 0.999)
}

/// What [`pack`] makes of a particle with nothing in its way.
pub(crate) const CLEAR: f32 = 0.999;
