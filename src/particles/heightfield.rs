//! A layer of loose material — snow, sand — over whatever is beneath it, on a
//! grid seen from above.
//!
//! Each cell has a **base**, the top of what the layer rests on (the ground, a
//! roof, a crate: [`Heightfield::base_from_scene`] finds it by casting rays
//! down), and a **depth** of material on top. Particles land on the surface
//! (`base + depth`), hand it their volume ([`deposit`](Heightfield::deposit)),
//! and the layer **slumps** ([`relax`](Heightfield::relax)) wherever it is
//! steeper than the material's angle of repose: material moves downhill,
//! never more than the cell holds, so the volume is conserved — which is why a
//! poured pile ends up a cone at exactly that angle, and snow slides off the
//! edge of a roof.

use crate::core::{BufferAttribute, BufferGeometry, ObjectArena, ObjectId, Raycaster};
use crate::math::Vector3;

/// Material on a grid over a rectangle of the ground.
#[derive(Debug, Clone)]
pub struct Heightfield {
    /// The grid's corner (smallest x and z).
    pub origin: [f32; 2],
    /// Cell size, metres.
    pub cell: f32,
    /// Cells along x and z.
    pub nx: usize,
    pub nz: usize,
    /// What the layer rests on, per cell (row-major, x fastest).
    pub base: Vec<f32>,
    /// Material on top, metres, per cell.
    pub depth: Vec<f32>,
    /// Bumped on every change, for a mesh to know it is stale.
    pub version: u64,
    /// Cells changed since the last [`relax`](Heightfield::relax) settled
    /// (i0, i1, j0, j1, inclusive): the only place it need look.
    dirty: Option<[usize; 4]>,
}

impl Heightfield {
    /// A flat grid of `size` (x, z) metres centred on `center` (x, z), at
    /// `resolution` cells along the longer side, its base at `base_height`.
    pub fn new(center: [f32; 2], size: [f32; 2], resolution: usize, base_height: f32) -> Self {
        let cell = size[0].max(size[1]) / resolution.max(2) as f32;
        let nx = ((size[0] / cell).round() as usize).max(2);
        let nz = ((size[1] / cell).round() as usize).max(2);
        Self {
            origin: [
                center[0] - nx as f32 * cell * 0.5,
                center[1] - nz as f32 * cell * 0.5,
            ],
            cell,
            nx,
            nz,
            base: vec![base_height; nx * nz],
            depth: vec![0.0; nx * nz],
            version: 0,
            dirty: None,
        }
    }

    /// The centre of cell `(i, j)`.
    pub fn cell_center(&self, i: usize, j: usize) -> [f32; 2] {
        [
            self.origin[0] + (i as f32 + 0.5) * self.cell,
            self.origin[1] + (j as f32 + 0.5) * self.cell,
        ]
    }

    fn idx(&self, i: usize, j: usize) -> usize {
        i + self.nx * j
    }

    /// The base from a function of (x, z): a terrain, a slope.
    pub fn set_base(&mut self, f: impl Fn(f32, f32) -> f32) {
        for j in 0..self.nz {
            for i in 0..self.nx {
                let [x, z] = self.cell_center(i, j);
                let k = self.idx(i, j);
                self.base[k] = f(x, z);
            }
        }
        self.touch_all();
        self.version += 1;
    }

    /// The base from a scene: the highest surface under each cell, by a ray
    /// cast down from `from_height` against `objects` and their descendants.
    /// Cells with nothing below keep their base.
    pub fn base_from_scene(&mut self, arena: &ObjectArena, objects: &[ObjectId], from_height: f32) {
        let mut ray = Raycaster::new(
            Vector3::ZERO,
            Vector3::new(0.0, -1.0, 0.0),
            0.0,
            f32::INFINITY,
        );
        ray.first_hit_only = true;
        for j in 0..self.nz {
            for i in 0..self.nx {
                let [x, z] = self.cell_center(i, j);
                ray.set(
                    Vector3::new(x, from_height, z),
                    Vector3::new(0.0, -1.0, 0.0),
                );
                if let Some(hit) = ray.intersect_object_list(arena, objects, true).first() {
                    let k = self.idx(i, j);
                    self.base[k] = hit.point.y;
                }
            }
        }
        self.touch_all();
        self.version += 1;
    }

    /// Mark every cell changed, for the next [`relax`](Self::relax) to look
    /// at — after writing [`depth`](Self::depth) or [`base`](Self::base)
    /// directly. (Everything here that changes them marks what it changed.)
    pub fn touch_all(&mut self) {
        self.dirty = Some([0, self.nx - 1, 0, self.nz - 1]);
    }

    /// Mark cells `i0..=i1` × `j0..=j1` changed (clamped to the grid).
    fn touch(&mut self, i0: i64, i1: i64, j0: i64, j1: i64) {
        let c = |v: i64, n: usize| v.clamp(0, n as i64 - 1) as usize;
        let r = [
            c(i0, self.nx),
            c(i1, self.nx),
            c(j0, self.nz),
            c(j1, self.nz),
        ];
        self.dirty = Some(match self.dirty {
            None => r,
            Some(d) => [
                d[0].min(r[0]),
                d[1].max(r[1]),
                d[2].min(r[2]),
                d[3].max(r[3]),
            ],
        });
    }

    /// Fractional cell coordinates of world (x, z), if over the grid.
    fn grid_coords(&self, x: f32, z: f32) -> Option<(f32, f32)> {
        let u = (x - self.origin[0]) / self.cell - 0.5;
        let v = (z - self.origin[1]) / self.cell - 0.5;
        let inside =
            u >= -0.5 && v >= -0.5 && u <= self.nx as f32 - 0.5 && v <= self.nz as f32 - 0.5;
        inside.then_some((u, v))
    }

    /// The surface (base plus material) at cell `(i, j)`.
    pub fn surface(&self, i: usize, j: usize) -> f32 {
        let k = self.idx(i, j);
        self.base[k] + self.depth[k]
    }

    /// The surface's height at world (x, z): the cell's own, so a particle
    /// lands on a roof's snow and not on a blend of it with the ground beside
    /// the wall. `None` off the grid.
    pub fn height_at(&self, x: f32, z: f32) -> Option<f32> {
        let (u, v) = self.grid_coords(x, z)?;
        let i = (u.round().max(0.0) as usize).min(self.nx - 1);
        let j = (v.round().max(0.0) as usize).min(self.nz - 1);
        Some(self.surface(i, j))
    }

    /// The surface's normal at world (x, z), from its neighbours.
    pub fn normal_at(&self, x: f32, z: f32) -> Vector3 {
        let Some((u, v)) = self.grid_coords(x, z) else {
            return Vector3::new(0.0, 1.0, 0.0);
        };
        let i = (u.round().max(0.0) as usize).min(self.nx - 1);
        let j = (v.round().max(0.0) as usize).min(self.nz - 1);
        self.normal(i, j)
    }

    fn normal(&self, i: usize, j: usize) -> Vector3 {
        let h = |i: usize, j: usize| self.surface(i.min(self.nx - 1), j.min(self.nz - 1));
        let dx =
            (h(i + 1, j) - h(i.saturating_sub(1), j)) / (self.cell * (1 + (i > 0) as usize) as f32);
        let dz =
            (h(i, j + 1) - h(i, j.saturating_sub(1))) / (self.cell * (1 + (j > 0) as usize) as f32);
        Vector3::new(-dx, 1.0, -dz).normalize()
    }

    /// Add `volume` m³ at world (x, z), shared between the four nearest cells
    /// by distance. Off the grid it is lost.
    pub fn deposit(&mut self, x: f32, z: f32, volume: f32) {
        let Some((u, v)) = self.grid_coords(x, z) else {
            return;
        };
        let (i0, j0) = (u.floor(), v.floor());
        let (fu, fv) = (u - i0, v - j0);
        let area = self.cell * self.cell;
        for (di, dj, w) in [
            (0, 0, (1.0 - fu) * (1.0 - fv)),
            (1, 0, fu * (1.0 - fv)),
            (0, 1, (1.0 - fu) * fv),
            (1, 1, fu * fv),
        ] {
            let (i, j) = (i0 as i64 + di, j0 as i64 + dj);
            let (i, j) = (
                i.clamp(0, self.nx as i64 - 1) as usize,
                j.clamp(0, self.nz as i64 - 1) as usize,
            );
            let k = self.idx(i, j);
            self.depth[k] += volume * w / area;
        }
        self.touch(i0 as i64, i0 as i64 + 1, j0 as i64, j0 as i64 + 1);
        self.version += 1;
    }

    /// Add `volume` m³ spread over a disc of `radius` metres round world
    /// (x, z), weighted by a smooth bump: one drawn particle standing for
    /// many real ones landing round it — a flake in a time-lapse, which
    /// carries a centimetre of snow and would otherwise pock the surface.
    /// What falls off the grid is lost; the rest is shared out exactly.
    pub fn deposit_spread(&mut self, x: f32, z: f32, volume: f32, radius: f32) {
        let reach = (radius / self.cell).ceil() as i64;
        if reach <= 1 {
            return self.deposit(x, z, volume);
        }
        let Some((u, v)) = self.grid_coords(x, z) else {
            return;
        };
        let (ci, cj) = (u.round() as i64, v.round() as i64);
        let mut cells = Vec::with_capacity(((2 * reach + 1) * (2 * reach + 1)) as usize);
        let mut total = 0.0;
        for dj in -reach..=reach {
            for di in -reach..=reach {
                let (i, j) = (ci + di, cj + dj);
                if i < 0 || j < 0 || i >= self.nx as i64 || j >= self.nz as i64 {
                    continue;
                }
                let (du, dv) = (i as f32 - u, j as f32 - v);
                let r2 = (du * du + dv * dv) * self.cell * self.cell / (radius * radius);
                if r2 >= 1.0 {
                    continue;
                }
                let w = (1.0 - r2) * (1.0 - r2);
                cells.push((self.idx(i as usize, j as usize), w));
                total += w;
            }
        }
        if total <= 0.0 {
            return;
        }
        let per = volume / (total * self.cell * self.cell);
        for (k, w) in cells {
            self.depth[k] += per * w;
        }
        self.touch(ci - reach, ci + reach, cj - reach, cj + reach);
        self.version += 1;
    }

    /// Whether cell `(i, j)` is in the lee of a slope steeper than `shadow`
    /// (radians), looking upwind along `wind` — wrapping round the grid if
    /// `wrap`.
    fn shadowed(&self, i: usize, j: usize, wind: (i64, i64), shadow: f32, wrap: bool) -> bool {
        let h = self.surface(i, j);
        let tan = shadow.tan();
        let reach = (self.nx.max(self.nz) / 4).max(8);
        for d in 1..=reach {
            let Some((ii, jj)) = self.step_to(
                i as i64 - wind.0 * d as i64,
                j as i64 - wind.1 * d as i64,
                wrap,
            ) else {
                return false;
            };
            if self.surface(ii, jj) - h > tan * d as f32 * self.cell {
                return true;
            }
        }
        false
    }

    /// Cell `(i, j)`, wrapped round the grid or `None` off it.
    fn step_to(&self, i: i64, j: i64, wrap: bool) -> Option<(usize, usize)> {
        let (nx, nz) = (self.nx as i64, self.nz as i64);
        if wrap {
            Some((i.rem_euclid(nx) as usize, j.rem_euclid(nz) as usize))
        } else if i < 0 || j < 0 || i >= nx || j >= nz {
            None
        } else {
            Some((i as usize, j as usize))
        }
    }

    /// Werner's slab model of wind moving loose material (see
    /// [`SandBed::blow`](super::sand::SandBed::blow)): `slabs` slabs lifted at
    /// random from cells not in a lee, each carried `hop` cells downwind at a
    /// time until it settles — more readily on material than on bare base,
    /// always in a lee. With `wrap` the grid is periodic; without, what
    /// blows off the downwind edge is gone. Slumping is the caller's.
    pub(crate) fn saltate(
        &mut self,
        downwind: super::sand::Downwind,
        slabs: usize,
        s: &super::sand::Saltation,
        rng: &mut super::Rng,
        wrap: bool,
    ) {
        let wind = downwind.step();
        let (nx, nz) = (self.nx, self.nz);
        let mut changed = [i64::MAX, i64::MIN, i64::MAX, i64::MIN];
        let mut mark = |i: usize, j: usize| {
            changed = [
                changed[0].min(i as i64),
                changed[1].max(i as i64),
                changed[2].min(j as i64),
                changed[3].max(j as i64),
            ];
        };
        for _ in 0..slabs {
            let i = (rng.next() * nx as f32) as usize % nx;
            let j = (rng.next() * nz as f32) as usize % nz;
            let k = self.idx(i, j);
            // Nothing to lift, or sheltered from the wind.
            if self.depth[k] < s.slab || self.shadowed(i, j, wind, s.shadow, wrap) {
                continue;
            }
            self.depth[k] -= s.slab;
            mark(i, j);
            let (mut ci, mut cj) = (i as i64, j as i64);
            // Hop until it settles; a long run is cut off rather than looping
            // round an empty grid forever.
            for _ in 0..(nx.max(nz) * 4) {
                ci += wind.0 * s.hop as i64;
                cj += wind.1 * s.hop as i64;
                let Some((ii, jj)) = self.step_to(ci, cj, wrap) else {
                    break;
                };
                let kk = self.idx(ii, jj);
                let stays = self.shadowed(ii, jj, wind, s.shadow, wrap) || {
                    let p = if self.depth[kk] > 0.0 {
                        s.on_sand
                    } else {
                        s.on_bare
                    };
                    rng.next() < p
                };
                if stays {
                    self.depth[kk] += s.slab;
                    mark(ii, jj);
                    break;
                }
            }
        }
        if changed[0] <= changed[1] {
            self.touch(changed[0], changed[1], changed[2], changed[3]);
        }
        self.version += 1;
    }

    /// Total material, m³.
    pub fn volume(&self) -> f32 {
        self.depth.iter().sum::<f32>() * self.cell * self.cell
    }

    /// Let the material slump until nowhere is steeper than `angle` (radians)
    /// — or for `sweeps` passes, whichever comes first. Returns the volume
    /// moved in the last pass (0 when settled).
    ///
    /// Each pass looks at every cell's sixteen nearest neighbours at once (the
    /// eight round it and the eight a knight's move away — eight alone slump
    /// a pile into an octagonal pyramid): wherever the drop to one exceeds
    /// `tan(angle)` times the distance, a share of the excess moves downhill,
    /// split among all the neighbours it is too steep against and never more
    /// than the cell has. Computed from one pass's heights and applied
    /// together, so the order cells are visited in does not matter and a pile
    /// stays symmetric. Only cells changed since it last settled are looked
    /// at, and from then on only round the cells still moving: a pile being
    /// poured on a big grid costs what the pile's size does.
    pub fn relax(&mut self, angle: f32, sweeps: usize) -> f32 {
        self.relax_within(angle, sweeps, usize::MAX)
    }

    /// [`relax`](Self::relax), stopping once it has looked at `budget`
    /// cells: what is still moving carries on at the next call. For a layer
    /// relaxed every frame, where a dune field under a steady wind always
    /// has slip faces avalanching somewhere and a frame should cost a known
    /// amount — an avalanche spread over a few frames looks the same.
    pub fn relax_within(&mut self, angle: f32, sweeps: usize, budget: usize) -> f32 {
        let tan = angle.tan();
        const N: [(i64, i64); 16] = [
            (1, 0),
            (-1, 0),
            (0, 1),
            (0, -1),
            (1, 1),
            (1, -1),
            (-1, 1),
            (-1, -1),
            (2, 1),
            (2, -1),
            (-2, 1),
            (-2, -1),
            (1, 2),
            (1, -2),
            (-1, 2),
            (-1, -2),
        ];
        let dist: [f32; 16] = N.map(|(a, b)| ((a * a + b * b) as f32).sqrt() * self.cell);
        let Some(region) = self.dirty.take() else {
            return 0.0;
        };
        let (nx, nz) = (self.nx, self.nz);
        // The neighbours as offsets in the grid, for cells far enough from
        // its edge to need no bounds checks.
        let offset: [isize; 16] = N.map(|(a, b)| a as isize + b as isize * nx as isize);
        let reach: [f32; 16] = dist.map(|d| tan * d);
        // Steeper than the angle by more than a ten-thousandth of a cell, or
        // not at all: below that, a quarter of the excess added to metres of
        // sand is lost to rounding — counted as moving, but never moved, so
        // the field would never count as settled.
        let least = self.cell * 1e-4;
        let mut surf: Vec<f32> = self
            .base
            .iter()
            .zip(&self.depth)
            .map(|(b, d)| b + d)
            .collect();
        let mut flow = vec![0.0f32; nx * nz];
        // Which sweep last listed a cell, so each is listed once.
        let mut listed = vec![0u32; nx * nz];
        // What changed, and the two cells round it its neighbours reach;
        // from then on only the cells that moved and their neighbours — the
        // only ones a sweep can unsettle — so a dune field's slip faces cost
        // what they do rather than what the whole field does.
        let mut cells: Vec<u32> = Vec::new();
        for j in region[2].saturating_sub(2)..=(region[3] + 2).min(nz - 1) {
            for i in region[0].saturating_sub(2)..=(region[1] + 2).min(nx - 1) {
                cells.push((i + nx * j) as u32);
            }
        }
        let mut changed: Vec<u32> = Vec::new();
        let mut movers: Vec<u32> = Vec::new();
        let mut moved = 0.0;
        let mut looked = 0usize;
        for sweep in 0..sweeps.max(1) {
            if sweep > 0 && looked >= budget {
                break;
            }
            looked += cells.len();
            moved = 0.0;
            movers.clear();
            for &c in &cells {
                let k = c as usize;
                let d = self.depth[k];
                if d <= 0.0 {
                    continue;
                }
                let (i, j) = (k % nx, k / nx);
                let h = surf[k];
                let inside = i >= 2 && j >= 2 && i + 2 < nx && j + 2 < nz;
                let mut excess = [0.0f32; 16];
                let mut total = 0.0;
                let mut steepest = 0.0f32;
                for n in 0..16 {
                    let kk = if inside {
                        (k as isize + offset[n]) as usize
                    } else {
                        let (ii, jj) = (i as i64 + N[n].0, j as i64 + N[n].1);
                        if ii < 0 || jj < 0 || ii >= nx as i64 || jj >= nz as i64 {
                            continue;
                        }
                        ii as usize + nx * jj as usize
                    };
                    let drop = h - surf[kk] - reach[n];
                    if drop > least {
                        excess[n] = drop;
                        total += drop;
                        steepest = steepest.max(drop);
                    }
                }
                if total <= 0.0 {
                    continue;
                }
                // A quarter of the steepest excess per pass: enough to
                // settle in a few passes, little enough that neighbours
                // draining into each other do not overshoot.
                let out = (steepest * 0.25).min(d);
                if flow[k] == 0.0 {
                    changed.push(c);
                }
                flow[k] -= out;
                moved += out;
                movers.push(c);
                for n in 0..16 {
                    if excess[n] > 0.0 {
                        let kk =
                            ((i as i64 + N[n].0) as usize) + nx * ((j as i64 + N[n].1) as usize);
                        if flow[kk] == 0.0 {
                            changed.push(kk as u32);
                        }
                        flow[kk] += out * excess[n] / total;
                    }
                }
            }
            for &c in &changed {
                let k = c as usize;
                self.depth[k] = (self.depth[k] + flow[k]).max(0.0);
                surf[k] = self.base[k] + self.depth[k];
                flow[k] = 0.0;
            }
            changed.clear();
            if moved < 1e-7 || movers.is_empty() {
                moved = 0.0;
                break;
            }
            // Next: the cells that moved and their neighbours, in grid order
            // (read off the marks where that is cheaper than sorting).
            let stamp = sweep as u32 + 1;
            cells.clear();
            let mut r = [usize::MAX, 0, usize::MAX, 0];
            for &c in &movers {
                let k = c as usize;
                let (i, j) = (k % nx, k / nx);
                for (a, b) in std::iter::once((0, 0)).chain(N) {
                    let (ii, jj) = (i as i64 + a, j as i64 + b);
                    if ii < 0 || jj < 0 || ii >= nx as i64 || jj >= nz as i64 {
                        continue;
                    }
                    let kk = ii as usize + nx * jj as usize;
                    if listed[kk] != stamp {
                        listed[kk] = stamp;
                        cells.push(kk as u32);
                    }
                }
                r = [r[0].min(i), r[1].max(i), r[2].min(j), r[3].max(j)];
            }
            let (i0, i1) = (r[0].saturating_sub(2), (r[1] + 2).min(nx - 1));
            let (j0, j1) = (r[2].saturating_sub(2), (r[3] + 2).min(nz - 1));
            let area = (i1 - i0 + 1) * (j1 - j0 + 1);
            if cells.len() * (usize::BITS - cells.len().leading_zeros()) as usize > area {
                cells.clear();
                for j in j0..=j1 {
                    for i in i0..=i1 {
                        if listed[i + nx * j] == stamp {
                            cells.push((i + nx * j) as u32);
                        }
                    }
                }
            } else {
                cells.sort_unstable();
            }
        }
        // Still moving when the sweeps ran out: carry on there next time.
        if moved > 0.0 {
            let mut r = [usize::MAX, 0, usize::MAX, 0];
            for &c in &movers {
                let (i, j) = (c as usize % nx, c as usize / nx);
                r = [r[0].min(i), r[1].max(i), r[2].min(j), r[3].max(j)];
            }
            self.dirty = Some(r);
        }
        self.version += 1;
        moved * self.cell * self.cell
    }

    /// The steepest slope anywhere material lies, as an angle: what
    /// [`relax`](Self::relax) holds to its `angle`.
    pub fn steepest(&self, min_depth: f32) -> f32 {
        let mut worst = 0.0f32;
        for j in 0..self.nz {
            for i in 0..self.nx {
                if self.depth[self.idx(i, j)] <= min_depth {
                    continue;
                }
                let h = self.surface(i, j);
                for (a, b) in [(1i64, 0i64), (0, 1), (-1, 0), (0, -1)] {
                    let (ii, jj) = (i as i64 + a, j as i64 + b);
                    if ii < 0 || jj < 0 || ii >= self.nx as i64 || jj >= self.nz as i64 {
                        continue;
                    }
                    let kk = self.idx(ii as usize, jj as usize);
                    if self.depth[kk] <= min_depth {
                        continue;
                    }
                    let drop = (h - self.surface(ii as usize, jj as usize)) / self.cell;
                    worst = worst.max(drop.atan());
                }
            }
        }
        worst
    }

    /// A mesh of the material's surface: a vertex per cell centre, raised to
    /// the surface where there is material and sunk just under the base where
    /// there is none (hidden by whatever the base is — the ground, a roof).
    ///
    /// Triangles across a step in the base taller than a few cells (a roof's
    /// edge, a wall) are left out: the wall is there to be seen, not a sheet
    /// of snow draped from the eaves to the ground. In their place, material
    /// on the high side of a step gets a **skirt** — its own edge, from its
    /// surface down to what it rests on — so a roof's snow shows how deep it
    /// is rather than blending into the snow beyond.
    pub fn mesh(&self) -> BufferGeometry {
        self.mesh_with_coverage(|_| 1.0)
    }

    /// [`mesh`](Self::mesh), with a `color` attribute: white, its alpha
    /// `coverage(depth)` of each vertex's cell — for a layer that is
    /// see-through where thin.
    pub fn mesh_with_coverage(&self, coverage: impl Fn(f32) -> f32) -> BufferGeometry {
        let n = self.nx * self.nz;
        let mut p = Vec::with_capacity(n * 3);
        let mut nm = Vec::with_capacity(n * 3);
        let mut uv = Vec::with_capacity(n * 2);
        let mut color = Vec::with_capacity(n * 4);
        let sink = self.cell * 0.02;
        for j in 0..self.nz {
            for i in 0..self.nx {
                let [x, z] = self.cell_center(i, j);
                let k = self.idx(i, j);
                let y = if self.depth[k] > 1e-5 {
                    self.surface(i, j)
                } else {
                    self.base[k] - sink
                };
                p.extend([x, y, z]);
                let v = self.normal(i, j);
                nm.extend([v.x, v.y, v.z]);
                uv.extend([
                    i as f32 / (self.nx - 1) as f32,
                    j as f32 / (self.nz - 1) as f32,
                ]);
                color.extend([1.0, 1.0, 1.0, coverage(self.depth[k])]);
            }
        }
        let step = self.cell * 3.0;
        let mut ix = Vec::with_capacity((self.nx - 1) * (self.nz - 1) * 6);
        for j in 0..self.nz - 1 {
            for i in 0..self.nx - 1 {
                let a = self.idx(i, j);
                let b = self.idx(i + 1, j);
                let c = self.idx(i, j + 1);
                let d = self.idx(i + 1, j + 1);
                let bases = [self.base[a], self.base[b], self.base[c], self.base[d]];
                let lo = bases.iter().cloned().fold(f32::INFINITY, f32::min);
                let hi = bases.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                if hi - lo > step {
                    continue;
                }
                // Counter-clockwise from above.
                ix.extend([a as u32, c as u32, b as u32, b as u32, c as u32, d as u32]);
            }
        }
        // Skirts: along each pair of neighbouring cells on the high side of
        // a step, facing over it.
        let deep = |k: usize| self.depth[k] > 1e-5;
        let mut skirt = |a: (usize, usize), b: (usize, usize), out: (i64, i64)| {
            let (ka, kb) = (self.idx(a.0, a.1), self.idx(b.0, b.1));
            if !deep(ka) && !deep(kb) {
                return;
            }
            if (self.base[ka] - self.base[kb]).abs() > step {
                return;
            }
            let below = |c: (usize, usize)| {
                let (ii, jj) = (c.0 as i64 + out.0, c.1 as i64 + out.1);
                if ii < 0 || jj < 0 || ii >= self.nx as i64 || jj >= self.nz as i64 {
                    return false;
                }
                self.base[self.idx(ii as usize, jj as usize)] < self.base[self.idx(c.0, c.1)] - step
            };
            if !below(a) || !below(b) {
                return;
            }
            let first = (p.len() / 3) as u32;
            for (c, k) in [(a, ka), (b, kb)] {
                let [x, z] = self.cell_center(c.0, c.1);
                let top = if deep(k) {
                    self.surface(c.0, c.1)
                } else {
                    self.base[k]
                };
                for y in [top, self.base[k] - sink] {
                    p.extend([x, y, z]);
                    nm.extend([out.0 as f32, 0.0, out.1 as f32]);
                    uv.extend([0.0, 0.0]);
                    color.extend([1.0, 1.0, 1.0, coverage(self.depth[k])]);
                }
            }
            // a-top, a-bottom, b-top, b-bottom; wound to face `out`.
            let (at, ab, bt, bb) = (first, first + 1, first + 2, first + 3);
            let along = (b.0 as i64 - a.0 as i64, b.1 as i64 - a.1 as i64);
            // Facing `out` means (along × down) · out > 0 for this winding.
            let flip = along.0 * out.1 - along.1 * out.0 > 0;
            if flip {
                ix.extend([at, bt, ab, ab, bt, bb]);
            } else {
                ix.extend([at, ab, bt, ab, bb, bt]);
            }
        };
        for j in 0..self.nz {
            for i in 0..self.nx {
                if i + 1 < self.nx {
                    skirt((i, j), (i + 1, j), (0, 1));
                    skirt((i, j), (i + 1, j), (0, -1));
                }
                if j + 1 < self.nz {
                    skirt((i, j), (i, j + 1), (1, 0));
                    skirt((i, j), (i, j + 1), (-1, 0));
                }
            }
        }
        let mut g = BufferGeometry::new();
        g.set_attribute("position", BufferAttribute::new(p, 3));
        g.set_attribute("normal", BufferAttribute::new(nm, 3));
        g.set_attribute("uv", BufferAttribute::new(uv, 2));
        g.set_attribute("color", BufferAttribute::new(color, 4));
        g.set_index(ix);
        g
    }
}
