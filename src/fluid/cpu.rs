//! The same volume solver, on the CPU.
//!
//! Not a fallback — [`FluidSim3d`](super::FluidSim3d) runs anywhere wgpu
//! does, which is everywhere. This exists for two things the GPU version
//! cannot give you: a second implementation to check the first against, and
//! an answer to "how much of this is the GPU worth". Both want it to be the
//! *same* solver rather than a simpler one, so it is a transcription of
//! `volume_shader.rs` stage for stage — curl, confinement, divergence, the
//! same V-cycle, the same semi-Lagrangian advection — reading the same
//! [`FluidConfig3d`].
//!
//! # Shape
//!
//! Flat `Vec`s in voxel order, `x + n * (y + n * z)`, rather than the atlas
//! the GPU uses. The atlas exists because a render pass writes one 2D
//! attachment; nothing here has that problem, and the tiling would only cost
//! arithmetic on every neighbour lookup.
//!
//! Velocity, curl and dye are `[f32; 3]` per voxel rather than three separate
//! planes. Every one of them is read three components at a time — advection
//! wants the whole vector at a traced point — so interleaving them is one
//! cache line instead of three.
//!
//! # What it ignores
//!
//! Every field of [`FluidConfig3d`] that describes the *simulation* is read
//! here, and that is the point — a reference that quietly disagrees is worse
//! than none. What it ignores is everything about drawing and about the
//! GPU: `march_steps`, `absorption`, `emission`, `exposure`, `background`,
//! `transparent`, `bounds_min`, `bounds_max`, `show_fps`,
//! `temporal_samples`, `temporal_clip`, `empty_space_skip`,
//! `cache_bindings` and `compact_dye`. It does not render, and it has no
//! bind groups or texture formats to cache or pack.
//!
//! `seed` is unused because there is no `multiple_splats` here: the splats
//! a reference needs are the ones a test chose.

use super::volume::FluidConfig3d;
use super::PressureSolver;

/// What can be interpolated between: the fields are scalars or vectors and
/// the sampling is the same either way.
trait Lerp: Copy {
    const ZERO: Self;
    fn lerp(a: Self, b: Self, t: f32) -> Self;
}

impl Lerp for f32 {
    const ZERO: f32 = 0.0;
    fn lerp(a: f32, b: f32, t: f32) -> f32 {
        a + (b - a) * t
    }
}

impl Lerp for [f32; 3] {
    const ZERO: [f32; 3] = [0.0; 3];
    fn lerp(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
        [
            a[0] + (b[0] - a[0]) * t,
            a[1] + (b[1] - a[1]) * t,
            a[2] + (b[2] - a[2]) * t,
        ]
    }
}

#[inline]
fn idx(n: usize, x: usize, y: usize, z: usize) -> usize {
    x + n * (y + n * z)
}

/// A voxel, clamped at the wall — the same clamp-to-edge the GPU's sampler
/// does, and what the free-slip boundary is written against.
#[inline]
fn at<T: Copy>(f: &[T], n: usize, x: isize, y: isize, z: isize) -> T {
    let c = |v: isize| v.clamp(0, n as isize - 1) as usize;
    f[idx(n, c(x), c(y), c(z))]
}

/// Trilinear, at a position in the unit cube.
fn sample<T: Lerp>(f: &[T], n: usize, p: [f32; 3]) -> T {
    let g = |v: f32| (v * n as f32 - 0.5).clamp(0.0, n as f32 - 1.0);
    let (fx, fy, fz) = (g(p[0]), g(p[1]), g(p[2]));
    let (x0, y0, z0) = (fx.floor(), fy.floor(), fz.floor());
    let (tx, ty, tz) = (fx - x0, fy - y0, fz - z0);
    let (x0, y0, z0) = (x0 as isize, y0 as isize, z0 as isize);
    let mut face = [T::ZERO; 2];
    for (k, slot) in face.iter_mut().enumerate() {
        let z = z0 + k as isize;
        let a = T::lerp(at(f, n, x0, y0, z), at(f, n, x0 + 1, y0, z), tx);
        let b = T::lerp(at(f, n, x0, y0 + 1, z), at(f, n, x0 + 1, y0 + 1, z), tx);
        *slot = T::lerp(a, b, ty);
    }
    T::lerp(face[0], face[1], tz)
}

/// The centre of voxel `i` of an `n³` grid, in the unit cube.
#[inline]
fn centre(n: usize, x: usize, y: usize, z: usize) -> [f32; 3] {
    let inv = 1.0 / n as f32;
    [
        (x as f32 + 0.5) * inv,
        (y as f32 + 0.5) * inv,
        (z as f32 + 0.5) * inv,
    ]
}

/// Run `body` over every z-slice of an `n³` output, in parallel where the
/// crate was built with `parallel` and `threaded` is set.
///
/// A slice at a time, not a voxel at a time. Rayon's per-call cost is tens of
/// microseconds and a step has about fifty of these; splitting finer would
/// spend more in the scheduler than in the fluid.
fn for_slices<T: Send>(
    out: &mut [T],
    n: usize,
    threaded: bool,
    body: impl Fn(usize, &mut [T]) + Sync,
) {
    let plane = n * n;
    #[cfg(feature = "parallel")]
    if threaded {
        use rayon::prelude::*;
        out.par_chunks_mut(plane)
            .enumerate()
            .for_each(|(z, slab)| body(z, slab));
        return;
    }
    let _ = threaded;
    for (z, slab) in out.chunks_mut(plane).enumerate() {
        body(z, slab);
    }
}

/// A coarse level of the V-cycle.
struct Level {
    n: usize,
    pressure: Vec<f32>,
    scratch: Vec<f32>,
    rhs: Vec<f32>,
    residual: Vec<f32>,
}

/// A three-dimensional fluid on the CPU. See the module docs.
pub struct CpuFluid3d {
    config: FluidConfig3d,
    n: usize,
    dn: usize,
    vel: Vec<[f32; 3]>,
    vel_tmp: Vec<[f32; 3]>,
    curl: Vec<[f32; 3]>,
    dye: Vec<[f32; 3]>,
    dye_tmp: Vec<[f32; 3]>,
    pressure: Vec<f32>,
    pressure_tmp: Vec<f32>,
    divergence: Vec<f32>,
    residual: Vec<f32>,
    mg: Vec<Level>,
    /// Use every core. Off measures what one does.
    pub threaded: bool,
}

impl CpuFluid3d {
    pub fn new(config: FluidConfig3d) -> Self {
        let n = (config.resolution.max(4)) as usize;
        let dn = (config.dye_resolution.max(4)) as usize;
        let cells = n * n * n;
        let mut mg = Vec::new();
        let mut m = n;
        let coarsest = (config.multigrid_coarsest.max(4)) as usize;
        while m / 2 >= coarsest && mg.len() < 6 {
            m /= 2;
            let c = m * m * m;
            mg.push(Level {
                n: m,
                pressure: vec![0.0; c],
                scratch: vec![0.0; c],
                rhs: vec![0.0; c],
                residual: vec![0.0; c],
            });
        }
        Self {
            config,
            n,
            dn,
            vel: vec![[0.0; 3]; cells],
            vel_tmp: vec![[0.0; 3]; cells],
            curl: vec![[0.0; 3]; cells],
            dye: vec![[0.0; 3]; dn * dn * dn],
            dye_tmp: vec![[0.0; 3]; dn * dn * dn],
            pressure: vec![0.0; cells],
            pressure_tmp: vec![0.0; cells],
            divergence: vec![0.0; cells],
            residual: vec![0.0; cells],
            mg,
            threaded: true,
        }
    }

    /// Solver and dye grids, in voxels a side.
    pub fn grids(&self) -> (u32, u32) {
        (self.n as u32, self.dn as u32)
    }

    /// Bytes the fields occupy.
    pub fn memory_bytes(&self) -> u64 {
        let f = std::mem::size_of::<f32>() as u64;
        let cells = (self.n * self.n * self.n) as u64;
        let dye = (self.dn * self.dn * self.dn) as u64;
        let mg: u64 = self
            .mg
            .iter()
            .map(|l| 4 * (l.n * l.n * l.n) as u64 * f)
            .sum();
        (cells * 3 * 3 + dye * 3 * 2 + cells * 4) * f + mg
    }

    pub fn config(&self) -> &FluidConfig3d {
        &self.config
    }

    pub fn dye(&self) -> &[[f32; 3]] {
        &self.dye
    }

    /// A ball of dye and a push, at `at` in the unit cube — the same Gaussian
    /// the GPU splats, with the same radius convention.
    pub fn splat(&mut self, at: [f32; 3], velocity: [f32; 3], color: [f32; 3]) {
        let radius = self.config.splat_radius / 100.0;
        let blob = |p: [f32; 3]| {
            let d = [p[0] - at[0], p[1] - at[1], p[2] - at[2]];
            (-(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]) / radius).exp()
        };
        let (n, dn) = (self.n, self.dn);
        let threaded = self.threaded;
        for_slices(&mut self.vel, n, threaded, |z, slab| {
            for y in 0..n {
                for x in 0..n {
                    let w = blob(centre(n, x, y, z));
                    let v = &mut slab[x + n * y];
                    for k in 0..3 {
                        v[k] += w * velocity[k];
                    }
                }
            }
        });
        for_slices(&mut self.dye, dn, threaded, |z, slab| {
            for y in 0..dn {
                for x in 0..dn {
                    let w = blob(centre(dn, x, y, z));
                    let c = &mut slab[x + dn * y];
                    for k in 0..3 {
                        c[k] += w * color[k];
                    }
                }
            }
        });
    }

    /// A frame: [`step`](Self::step) by `dt` scaled by
    /// [`FluidConfig3d::time_scale`]. The GPU version splits these the same
    /// way, so the two stay comparable at a fixed step.
    pub fn update(&mut self, dt: f32) {
        let dt = dt * self.config.time_scale.max(0.0);
        if dt > 0.0 {
            self.step(dt);
        }
    }

    /// One step, in the order [`FluidSim3d::step`](super::FluidSim3d::step)
    /// takes it.
    pub fn step(&mut self, dt: f32) {
        let c = self.config;
        if c.buoyancy != 0.0 {
            self.buoyancy(dt);
        }
        if c.curl != 0.0 {
            self.curl_field();
            self.confine(dt);
        }
        self.divergence();
        let keep = c.pressure;
        for p in self.pressure.iter_mut() {
            *p *= keep;
        }
        match c.pressure_solver {
            PressureSolver::Jacobi => self.jacobi(c.pressure_iterations),
            PressureSolver::Multigrid => self.v_cycles(),
        }
        self.project(c.gradient_scale);
        self.advect_velocity(dt, c.velocity_dissipation);
        self.advect_dye(dt, c.density_dissipation);
    }

    fn buoyancy(&mut self, dt: f32) {
        let (n, dn, lift) = (self.n, self.dn, self.config.buoyancy);
        // Along the configured axis, as the GPU does. Hardcoding +Y here
        // would make the reference disagree with the thing it is the
        // reference for, in exactly the configuration someone reached for
        // the axis to get.
        let a = self.config.buoyancy_axis;
        let len = (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt();
        if lift == 0.0 || len < 1e-6 {
            return;
        }
        let axis = [a[0] / len, a[1] / len, a[2] / len];
        let dye = &self.dye;
        for_slices(&mut self.vel, n, self.threaded, |z, slab| {
            for y in 0..n {
                for x in 0..n {
                    let d = sample(dye, dn, centre(n, x, y, z));
                    let density = d[0].max(d[1]).max(d[2]);
                    let v = &mut slab[x + n * y];
                    for k in 0..3 {
                        v[k] += axis[k] * density * lift * dt;
                    }
                }
            }
        });
    }

    fn curl_field(&mut self) {
        let n = self.n;
        let vel = &self.vel;
        for_slices(&mut self.curl, n, self.threaded, |z, slab| {
            let z = z as isize;
            for y in 0..n {
                for x in 0..n {
                    let (xi, yi) = (x as isize, y as isize);
                    let l = at(vel, n, xi - 1, yi, z);
                    let r = at(vel, n, xi + 1, yi, z);
                    let d = at(vel, n, xi, yi - 1, z);
                    let u = at(vel, n, xi, yi + 1, z);
                    let b = at(vel, n, xi, yi, z - 1);
                    let f = at(vel, n, xi, yi, z + 1);
                    slab[x + n * y] = [
                        0.5 * ((u[2] - d[2]) - (f[1] - b[1])),
                        0.5 * ((f[0] - b[0]) - (r[2] - l[2])),
                        0.5 * ((r[1] - l[1]) - (u[0] - d[0])),
                    ];
                }
            }
        });
    }

    fn confine(&mut self, dt: f32) {
        let (n, eps) = (self.n, self.config.curl);
        let curl = &self.curl;
        for_slices(&mut self.vel, n, self.threaded, |z, slab| {
            let z = z as isize;
            for y in 0..n {
                for x in 0..n {
                    let (xi, yi) = (x as isize, y as isize);
                    let mag = |v: [f32; 3]| (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
                    let grad = [
                        0.5 * (mag(at(curl, n, xi + 1, yi, z)) - mag(at(curl, n, xi - 1, yi, z))),
                        0.5 * (mag(at(curl, n, xi, yi + 1, z)) - mag(at(curl, n, xi, yi - 1, z))),
                        0.5 * (mag(at(curl, n, xi, yi, z + 1)) - mag(at(curl, n, xi, yi, z - 1))),
                    ];
                    let len = mag(grad) + 0.0001;
                    let nrm = [grad[0] / len, grad[1] / len, grad[2] / len];
                    let w = curl[idx(n, x, y, z as usize)];
                    // f = e * (N x w), the cross product the plane's quarter
                    // turn was standing in for.
                    let force = [
                        nrm[1] * w[2] - nrm[2] * w[1],
                        nrm[2] * w[0] - nrm[0] * w[2],
                        nrm[0] * w[1] - nrm[1] * w[0],
                    ];
                    let v = &mut slab[x + n * y];
                    for k in 0..3 {
                        v[k] = (v[k] + force[k] * eps * dt).clamp(-1000.0, 1000.0);
                    }
                }
            }
        });
    }

    fn divergence(&mut self) {
        let n = self.n;
        let vel = &self.vel;
        for_slices(&mut self.divergence, n, self.threaded, |z, slab| {
            let zi = z as isize;
            let last = n as isize - 1;
            for y in 0..n {
                for x in 0..n {
                    let (xi, yi) = (x as isize, y as isize);
                    let c = vel[idx(n, x, y, z)];
                    // Free-slip walls: outside the box the normal component
                    // is mirrored, so the boundary neither leaks nor drags.
                    let l = if xi == 0 {
                        -c[0]
                    } else {
                        at(vel, n, xi - 1, yi, zi)[0]
                    };
                    let r = if xi == last {
                        -c[0]
                    } else {
                        at(vel, n, xi + 1, yi, zi)[0]
                    };
                    let d = if yi == 0 {
                        -c[1]
                    } else {
                        at(vel, n, xi, yi - 1, zi)[1]
                    };
                    let u = if yi == last {
                        -c[1]
                    } else {
                        at(vel, n, xi, yi + 1, zi)[1]
                    };
                    let b = if zi == 0 {
                        -c[2]
                    } else {
                        at(vel, n, xi, yi, zi - 1)[2]
                    };
                    let f = if zi == last {
                        -c[2]
                    } else {
                        at(vel, n, xi, yi, zi + 1)[2]
                    };
                    slab[x + n * y] = 0.5 * ((r - l) + (u - d) + (f - b));
                }
            }
        });
    }

    /// One Jacobi sweep of `dst = (sum of six neighbours - rhs) / 6`.
    fn sweep(src: &[f32], rhs: &[f32], dst: &mut [f32], n: usize, threaded: bool) {
        for_slices(dst, n, threaded, |z, slab| {
            let zi = z as isize;
            for y in 0..n {
                for x in 0..n {
                    let (xi, yi) = (x as isize, y as isize);
                    let sum = at(src, n, xi - 1, yi, zi)
                        + at(src, n, xi + 1, yi, zi)
                        + at(src, n, xi, yi - 1, zi)
                        + at(src, n, xi, yi + 1, zi)
                        + at(src, n, xi, yi, zi - 1)
                        + at(src, n, xi, yi, zi + 1);
                    slab[x + n * y] = (sum - rhs[idx(n, x, y, z)]) / 6.0;
                }
            }
        });
    }

    fn jacobi(&mut self, count: u32) {
        for _ in 0..count {
            Self::sweep(
                &self.pressure,
                &self.divergence,
                &mut self.pressure_tmp,
                self.n,
                self.threaded,
            );
            std::mem::swap(&mut self.pressure, &mut self.pressure_tmp);
        }
    }

    fn v_cycles(&mut self) {
        if self.mg.is_empty() {
            self.jacobi(self.config.pressure_iterations);
            return;
        }
        let nu = self.config.multigrid_smooth.max(1);
        let coarsest = self.mg.last().map(|l| l.n).unwrap_or(4) as u32;
        let coarse_sweeps = (2 * coarsest).clamp(nu * 2, self.config.pressure_iterations.max(8));
        for _ in 0..self.config.multigrid_cycles.max(1) {
            self.jacobi(nu);
            for level in 0..self.mg.len() {
                self.residual_into(level);
                self.restrict_into(level);
                self.mg[level].pressure.fill(0.0);
                let sweeps = if level + 1 == self.mg.len() {
                    coarse_sweeps
                } else {
                    nu
                };
                self.smooth(level, sweeps);
            }
            for level in (0..self.mg.len()).rev() {
                self.prolong(level);
                if level == 0 {
                    self.jacobi(nu);
                } else {
                    self.smooth(level - 1, nu);
                }
            }
        }
    }

    fn smooth(&mut self, level: usize, sweeps: u32) {
        let threaded = self.threaded;
        let l = &mut self.mg[level];
        for _ in 0..sweeps {
            Self::sweep(&l.pressure, &l.rhs, &mut l.scratch, l.n, threaded);
            std::mem::swap(&mut l.pressure, &mut l.scratch);
        }
    }

    /// `r = rhs - (sum - 6p)`, at the resolution of the level above.
    fn residual_into(&mut self, level: usize) {
        let threaded = self.threaded;
        // The target comes out first. A level's residual lives in the same
        // struct as the pressure it is computed from, so taking it after
        // borrowing the source would be two borrows of one `Level`.
        let mut out = if level == 0 {
            std::mem::take(&mut self.residual)
        } else {
            std::mem::take(&mut self.mg[level - 1].residual)
        };
        let (n, p, rhs) = if level == 0 {
            (self.n, &self.pressure, &self.divergence)
        } else {
            let l = &self.mg[level - 1];
            (l.n, &l.pressure, &l.rhs)
        };
        for_slices(&mut out, n, threaded, |z, slab| {
            let zi = z as isize;
            for y in 0..n {
                for x in 0..n {
                    let (xi, yi) = (x as isize, y as isize);
                    let sum = at(p, n, xi - 1, yi, zi)
                        + at(p, n, xi + 1, yi, zi)
                        + at(p, n, xi, yi - 1, zi)
                        + at(p, n, xi, yi + 1, zi)
                        + at(p, n, xi, yi, zi - 1)
                        + at(p, n, xi, yi, zi + 1);
                    let i = idx(n, x, y, z);
                    slab[x + n * y] = rhs[i] - (sum - 6.0 * p[i]);
                }
            }
        });
        if level == 0 {
            self.residual = out;
        } else {
            self.mg[level - 1].residual = out;
        }
    }

    /// The residual onto the half-size grid below, times four — the change of
    /// units, since the stencil carries no `1 / h²`.
    fn restrict_into(&mut self, level: usize) {
        let threaded = self.threaded;
        let coarse_n = self.mg[level].n;
        let mut out = std::mem::take(&mut self.mg[level].rhs);
        let (fine_n, fine) = if level == 0 {
            (self.n, &self.residual)
        } else {
            let l = &self.mg[level - 1];
            (l.n, &l.residual)
        };
        for_slices(&mut out, coarse_n, threaded, |z, slab| {
            for y in 0..coarse_n {
                for x in 0..coarse_n {
                    // The coarse voxel's centre is the corner its eight fine
                    // voxels share, so one trilinear fetch is their average.
                    let p = centre(coarse_n, x, y, z);
                    slab[x + coarse_n * y] = 4.0 * sample(fine, fine_n, p);
                }
            }
        });
        self.mg[level].rhs = out;
    }

    fn prolong(&mut self, level: usize) {
        let threaded = self.threaded;
        let coarse_n = self.mg[level].n;
        let coarse = std::mem::take(&mut self.mg[level].pressure);
        let fine_n = if level == 0 {
            self.n
        } else {
            self.mg[level - 1].n
        };
        let mut fine = if level == 0 {
            std::mem::take(&mut self.pressure)
        } else {
            std::mem::take(&mut self.mg[level - 1].pressure)
        };
        for_slices(&mut fine, fine_n, threaded, |z, slab| {
            for y in 0..fine_n {
                for x in 0..fine_n {
                    slab[x + fine_n * y] += sample(&coarse, coarse_n, centre(fine_n, x, y, z));
                }
            }
        });
        if level == 0 {
            self.pressure = fine;
        } else {
            self.mg[level - 1].pressure = fine;
        }
        self.mg[level].pressure = coarse;
    }

    fn project(&mut self, scale: f32) {
        let n = self.n;
        let p = &self.pressure;
        for_slices(&mut self.vel, n, self.threaded, |z, slab| {
            let zi = z as isize;
            for y in 0..n {
                for x in 0..n {
                    let (xi, yi) = (x as isize, y as isize);
                    let g = [
                        at(p, n, xi + 1, yi, zi) - at(p, n, xi - 1, yi, zi),
                        at(p, n, xi, yi + 1, zi) - at(p, n, xi, yi - 1, zi),
                        at(p, n, xi, yi, zi + 1) - at(p, n, xi, yi, zi - 1),
                    ];
                    let v = &mut slab[x + n * y];
                    for k in 0..3 {
                        v[k] -= scale * g[k];
                    }
                }
            }
        });
    }

    fn advect_velocity(&mut self, dt: f32, dissipation: f32) {
        let n = self.n;
        let vel = &self.vel;
        let decay = 1.0 + dissipation * dt;
        let step = dt / n as f32;
        for_slices(&mut self.vel_tmp, n, self.threaded, |z, slab| {
            for y in 0..n {
                for x in 0..n {
                    let p = centre(n, x, y, z);
                    let u = vel[idx(n, x, y, z)];
                    let back = [p[0] - u[0] * step, p[1] - u[1] * step, p[2] - u[2] * step];
                    let r = sample(vel, n, back);
                    slab[x + n * y] = [r[0] / decay, r[1] / decay, r[2] / decay];
                }
            }
        });
        std::mem::swap(&mut self.vel, &mut self.vel_tmp);
    }

    fn advect_dye(&mut self, dt: f32, dissipation: f32) {
        let (n, dn) = (self.n, self.dn);
        let (vel, dye) = (&self.vel, &self.dye);
        let decay = 1.0 + dissipation * dt;
        // The velocity is in *its own* voxels a second, so the step is scaled
        // by the solver grid and not by the dye's.
        let step = dt / n as f32;
        for_slices(&mut self.dye_tmp, dn, self.threaded, |z, slab| {
            for y in 0..dn {
                for x in 0..dn {
                    let p = centre(dn, x, y, z);
                    let u = sample(vel, n, p);
                    let back = [p[0] - u[0] * step, p[1] - u[1] * step, p[2] - u[2] * step];
                    let r = sample(dye, dn, back);
                    slab[x + dn * y] = [r[0] / decay, r[1] / decay, r[2] / decay];
                }
            }
        });
        std::mem::swap(&mut self.dye, &mut self.dye_tmp);
    }

    /// Mean |divergence| of the velocity field — what the projection is for,
    /// and the only honest way to compare two solvers.
    pub fn mean_divergence(&mut self) -> f32 {
        self.divergence();
        let total: f32 = self.divergence.iter().map(|d| d.abs()).sum();
        total / self.divergence.len() as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(n: u32) -> FluidConfig3d {
        FluidConfig3d {
            resolution: n,
            dye_resolution: n,
            curl: 0.0,
            buoyancy: 0.0,
            gradient_scale: 0.5,
            ..FluidConfig3d::default()
        }
    }

    /// A radial puff is pure divergence, and the projection takes it out.
    /// The same claim `projection_removes_divergence_in_three_dimensions`
    /// makes of the GPU, measured the same way.
    #[test]
    fn the_projection_removes_divergence() {
        let mut sim = CpuFluid3d::new(config(32));
        sim.splat([0.5, 0.5, 0.5], [300.0, 0.0, 0.0], [0.2, 0.2, 0.2]);
        let before = sim.mean_divergence();
        assert!(before > 1e-5, "the splat produced no divergence: {before}");
        // A short step, so advection — which runs after the projection and
        // puts divergence back — is out of the picture.
        sim.step(1.0 / 6000.0);
        let after = sim.mean_divergence() / before;
        eprintln!("  cpu v-cycle leaves {after:.3} of the divergence");
        assert!(after < 0.5, "the solve left {after:.3} of the divergence");
    }

    /// And the V-cycle beats sweeping here too, which is the check that the
    /// coarse grids are wired to the right levels: a restriction to the wrong
    /// grid still converges, just slowly.
    #[test]
    fn the_v_cycle_beats_the_sweeps() {
        let left = |solver, iterations| {
            let mut sim = CpuFluid3d::new(FluidConfig3d {
                pressure_solver: solver,
                pressure_iterations: iterations,
                ..config(32)
            });
            sim.splat([0.5, 0.5, 0.5], [300.0, 0.0, 0.0], [0.2, 0.2, 0.2]);
            let before = sim.mean_divergence();
            sim.step(1.0 / 6000.0);
            sim.mean_divergence() / before
        };
        let sweeps = left(PressureSolver::Jacobi, 20);
        let cycle = left(PressureSolver::Multigrid, 20);
        eprintln!("  cpu: 20 sweeps leave {sweeps:.3}, one V-cycle {cycle:.3}");
        assert!(
            cycle < sweeps * 0.6,
            "the V-cycle left {cycle:.3} against twenty sweeps' {sweeps:.3}"
        );
    }

    /// Buoyancy lifts the dye and not the rest: proportional to what is
    /// there, or the empty fluid rises too.
    #[test]
    fn buoyancy_lifts_the_dye() {
        let mut sim = CpuFluid3d::new(FluidConfig3d {
            buoyancy: 400.0,
            velocity_dissipation: 0.0,
            ..config(24)
        });
        sim.splat([0.5, 0.25, 0.5], [0.0; 3], [1.0, 1.0, 1.0]);
        let height = |sim: &CpuFluid3d| {
            let n = sim.grids().1 as usize;
            let (mut sum, mut mass) = (0.0f64, 0.0f64);
            for (i, d) in sim.dye().iter().enumerate() {
                let y = (i / n) % n;
                let w = d[0] as f64;
                sum += w * y as f64;
                mass += w;
            }
            sum / mass.max(1e-9)
        };
        let before = height(&sim);
        for _ in 0..40 {
            sim.step(1.0 / 60.0);
        }
        let after = height(&sim);
        eprintln!("  dye rose from slice {before:.1} to {after:.1}");
        assert!(
            after > before + 0.5,
            "the dye did not rise: {before} to {after}"
        );
    }

    /// Threads must not change the answer. Every stage writes one output
    /// voxel per input voxel and reads only the previous state, so slicing
    /// the output across cores is safe — this is what says it stayed that
    /// way.
    #[test]
    fn threading_changes_nothing() {
        let run = |threaded| {
            let mut sim = CpuFluid3d::new(FluidConfig3d {
                buoyancy: 40.0,
                curl: 20.0,
                ..config(24)
            });
            sim.threaded = threaded;
            sim.splat([0.5, 0.3, 0.5], [10.0, 50.0, -10.0], [1.0, 0.5, 0.25]);
            for _ in 0..6 {
                sim.step(1.0 / 60.0);
            }
            sim.dye().to_vec()
        };
        let one = run(false);
        let many = run(true);
        let worst = one
            .iter()
            .zip(&many)
            .flat_map(|(a, b)| (0..3).map(move |k| (a[k] - b[k]).abs()))
            .fold(0.0f32, f32::max);
        assert_eq!(worst, 0.0, "threading changed the dye by {worst}");
    }
}
